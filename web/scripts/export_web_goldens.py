#!/usr/bin/env python
"""Write super-resolution goldens for the browser engine from the PyTorch reference.

For every HR image and scale this crops the image to a multiple of ``8 * scale`` (the engine
runs LR sizes that are multiples of 8 unpadded), makes the LR image with MATLAB bicubic
(``gaussifier_sr.matlab_resize``, 8-bit) and runs ``gaussifier_sr.loop.upscale`` twice:

- ``ref``: the reference as released, ``load_model("cuda")`` with PyTorch's defaults (TF32
  convolutions, the head under bf16 autocast);
- ``ref_fp32``: the same loop in IEEE FP32 (TF32 off, no autocast), the browser's arithmetic.

Each case directory holds ``lr.png`` and ``hr.png`` (also as planar 8-bit ``lr_u8.bin`` and
``hr_u8.bin``), the LR image and both outputs as raw little-endian float32 ``[3, H, W]`` arrays
(``lr.bin``, ``ref.bin``, ``ref_fp32.bin``), and ``manifest.json`` with the counts of every
pass, the PSNR of every output against the HR image, the run-to-run spread of the reference
and its timing. ``index.json`` lists the cases.

    uv run python web/scripts/export_web_goldens.py --out web/fixtures/sr \\
        --image kodim03.png --image kodim08.png:416x288 --scale 2 4
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import gaussifier_sr.loop as loop
from gaussifier_sr.matlab_resize import imresize, matlab_downsample_uint8
from gaussifier_sr.metrics import psnr_y, quantize_8bit
from gaussifier_sr.model import base_weights_dir, load_model, sr_head_path

LR_MULTIPLE = 8


def sha256_of(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def psnr_rgb(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """PSNR (dB) over the three channels of the 8-bit images (the PNGs they would be saved as)."""
    d = quantize_8bit(pred).double() - quantize_8bit(ref).double()
    return 10.0 * math.log10(1.0 / float(d.square().mean()))


def scores(pred: torch.Tensor, hr: torch.Tensor, scale: int) -> dict[str, float]:
    return {"psnr_rgb": psnr_rgb(pred, hr), "psnr_y": psnr_y(pred, hr, scale)}


class Recorder:
    """Counts of every pass of ``gaussifier_sr.loop``: forward-map rate sums and placements."""

    def __init__(self, model) -> None:
        self.rate_sums: list[dict] = []
        self.placements: list[dict] = []
        self._model = model
        self._forward_map = model.forward_map
        self._voronoi = loop.production_voronoi

    def __enter__(self) -> Recorder:
        def forward_map(image: torch.Tensor) -> dict[str, torch.Tensor]:
            maps = self._forward_map(image)
            rate_sum = float(maps["rate"].float().sum())
            self.rate_sums.append({"size": list(image.shape[-2:]), "rate_sum": rate_sum})
            return maps

        def production_voronoi(density: torch.Tensor, n_points: int, *, seed: int):
            self.placements.append({"size": list(density.shape), "n": int(n_points), "seed": seed})
            return self._voronoi(density, n_points, seed=seed)

        self._model.forward_map = forward_map
        loop.production_voronoi = production_voronoi
        return self

    def __exit__(self, *exc: object) -> None:
        del self._model.forward_map
        loop.production_voronoi = self._voronoi

    @property
    def own_count(self) -> float:
        """own(LR): the first forward map of a call is the LR image's."""
        return self.rate_sums[0]["rate_sum"]

    def passes(self) -> list[dict]:
        """Per x2 pass: its placement and the rate sum of its input (after the LR's own map)."""
        return [
            {**place, "rate_sum": fm["rate_sum"], "adaptive_count": max(1, int(fm["rate_sum"]))}
            for place, fm in zip(self.placements, self.rate_sums[1:], strict=True)
        ]


@contextlib.contextmanager
def ieee_fp32():
    saved = torch.backends.cudnn.conv.fp32_precision
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.cudnn.conv.fp32_precision = saved


def repeated(fn, runs: int) -> tuple[list[torch.Tensor], float]:
    """``runs`` results of ``fn`` and the median wall time in ms of all but the first call.

    The placement's jump flood and the renderer's tile lists use atomics, so the reference
    varies slightly from run to run; the spread of the runs is the floor of any comparison.
    """
    outputs, times = [], []
    for _ in range(max(2, runs)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs.append(fn())
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return outputs, float(np.median(times[1:]))


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().float().cpu().contiguous().numpy().astype("<f4").tofile(path)


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """``(H, W, 3)`` uint8 -> ``(3, H, W)`` float in [0, 1] on the GPU."""
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).cuda()


def parse_image(spec: str) -> tuple[Path, tuple[int, int] | None]:
    """``path`` or ``path:WxH`` (the top-left crop of that size)."""
    path, _, crop = spec.partition(":")
    if not crop:
        return Path(path), None
    w, h = (int(v) for v in crop.lower().split("x"))
    return Path(path), (w, h)


def write_case(out: Path, hr_np: np.ndarray, scale: int, models: dict, runs: int) -> dict:
    """Run the references on the LR image of ``hr_np`` and write one case directory."""
    lr_np = matlab_downsample_uint8(hr_np, scale)
    out.mkdir(parents=True, exist_ok=True)
    for key, array in (("hr", hr_np), ("lr", lr_np)):
        Image.fromarray(array).save(out / f"{key}.png")
        # Planar 8-bit copies for the demo's resize test.
        np.ascontiguousarray(array.transpose(2, 0, 1)).tofile(out / f"{key}_u8.bin")
    hr, lr = to_tensor(hr_np), to_tensor(lr_np)
    write_f32(out / "lr.bin", lr)

    released, fp32 = models["released"], models["fp32"]
    with Recorder(released) as rec:
        ref = loop.upscale(released, lr, scale)
    spread, ref_ms = repeated(lambda: loop.upscale(released, lr, scale), runs)
    with ieee_fp32(), Recorder(fp32) as rec32:
        ref32 = loop.upscale(fp32, lr, scale)
    write_f32(out / "ref.bin", ref)
    write_f32(out / "ref_fp32.bin", ref32)
    bicubic = np.clip(np.round(imresize(lr_np / 255.0, scale) * 255.0), 0, 255).astype(np.uint8)
    return {
        "scale": scale,
        "lr": {"height": int(lr.shape[1]), "width": int(lr.shape[2])},
        "hr": {"height": int(hr.shape[1]), "width": int(hr.shape[2])},
        "own_count": {"ref": rec.own_count, "ref_fp32": rec32.own_count},
        "passes": {"ref": rec.passes(), "ref_fp32": rec32.passes()},
        "scores": {
            "ref": scores(ref, hr, scale),
            "ref_fp32": scores(ref32, hr, scale),
            "bicubic_matlab": scores(to_tensor(bicubic), hr, scale),
            "ref_fp32_vs_ref_psnr_rgb": psnr_rgb(ref32, ref),
            # Run-to-run spread of the reference itself.
            "ref_runs_psnr_rgb": [psnr_rgb(r, hr) for r in spread],
            "ref_runs_psnr_y": [psnr_y(r, hr, scale) for r in spread],
            "ref_run_vs_ref_psnr_rgb": [psnr_rgb(r, ref) for r in spread],
        },
        "timing_ms": {"ref_median": ref_ms, "runs": len(spread)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", action="append", required=True, help="path or path:WxH")
    parser.add_argument("--scale", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sr-head", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=5, help="repeat runs (spread and timing)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("goldens require CUDA")
    sr_head = args.sr_head or sr_head_path()
    models = {
        "released": load_model("cuda", sr_head=sr_head),
        "fp32": load_model("cuda", sr_head=sr_head, bf16=False),
    }
    weights = {"stage1": sha256_of(base_weights_dir() / "stage1.pt"), "head": sha256_of(sr_head)}

    names: list[str] = []
    for spec in args.image:
        path, crop = parse_image(spec)
        source = np.array(Image.open(path).convert("RGB"))
        if crop:
            source = source[: crop[1], : crop[0]]
        for scale in args.scale:
            m = LR_MULTIPLE * scale
            h, w = source.shape[0] // m * m, source.shape[1] // m * m
            name = f"{path.stem}{f'_{crop[0]}x{crop[1]}' if crop else ''}_x{scale}"
            case = write_case(
                args.out / name, np.ascontiguousarray(source[:h, :w]), scale, models, args.runs
            )
            manifest = {
                "name": name,
                "source": path.name,
                "crop": crop,
                **case,
                "weights_sha256": weights,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
            }
            (args.out / name / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
            names.append(name)
            s = manifest["scores"]
            print(
                f"{name}: N {[p['n'] for p in case['passes']['ref']]} "
                f"PSNR {s['ref']['psnr_rgb']:.3f} (FP32 {s['ref_fp32']['psnr_rgb']:.3f}) "
                f"{case['timing_ms']['ref_median']:.1f} ms",
                flush=True,
            )
    (args.out / "index.json").write_text(json.dumps({"cases": names}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
