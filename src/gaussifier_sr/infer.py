"""Super-resolve images with the released model.

    python -m gaussifier_sr.infer --input lr.png --scale 2 --output sr.png
    python -m gaussifier_sr.infer --input lr_dir/ --scale 4 --output sr_dir/ --backend fused

x2 is one pass; x4 is two x2 passes (the second keeps the LR image's point density).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def read_rgb(path: Path) -> np.ndarray:
    """``(H, W, 3)`` uint8 RGB."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def to_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """``(H, W, 3)`` uint8 -> ``(3, H, W)`` float in [0, 1] on ``device``."""
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255).to(device)


def save_image(image: torch.Tensor, path: Path) -> None:
    rgb = (image.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


class TorchUpscaler:
    """The PyTorch reference implementation (``gaussifier_sr.loop``)."""

    def __init__(self, device: torch.device, sr_head: Path | None = None) -> None:
        from gaussifier_sr.model import load_model

        self.model = load_model(device, sr_head=sr_head)

    def __call__(self, lr: torch.Tensor, scale: int) -> torch.Tensor:
        from gaussifier_sr.loop import upscale

        return upscale(self.model, lr, scale)


def make_upscaler(backend: str, device: torch.device, sr_head: Path | None = None):
    """``backend`` is ``torch`` (PyTorch reference) or ``fused`` (CUDA runtime)."""
    if backend == "torch":
        return TorchUpscaler(device, sr_head)
    if backend == "fused":
        from gaussifier_sr.fused import FusedSR

        return FusedSR.released()
    raise ValueError(f"unknown backend {backend!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path, required=True, help="image or folder")
    ap.add_argument("--output", type=Path, required=True, help="image or folder")
    ap.add_argument("--scale", type=int, default=2, choices=[2, 4, 8])
    ap.add_argument("--backend", default="torch", choices=["torch", "fused"])
    a = ap.parse_args()
    device = torch.device("cuda")
    upscaler = make_upscaler(a.backend, device)
    if a.input.is_dir():
        jobs = [
            (p, a.output / f"{p.stem}.png")
            for p in sorted(a.input.iterdir())
            if p.suffix.lower() in IMAGE_SUFFIXES
        ]
    else:
        jobs = [(a.input, a.output)]
    for src, dst in jobs:
        lr = to_tensor(read_rgb(src), device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sr = upscaler(lr, a.scale)
        torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0)
        save_image(sr, dst)
        h, w = lr.shape[-2:]
        print(f"{src.name}: {w}x{h} -> {sr.shape[-1]}x{sr.shape[-2]} in {ms:.1f} ms -> {dst}")


if __name__ == "__main__":
    main()
