"""Benchmark evaluation with the standard SR protocol.

    python -m gaussifier_sr.evaluate --hr-dir benchmarks/Set5/HR --scale 2 4 [--backend fused]

Each HR image is cropped to a multiple of the scale, downsampled with MATLAB ``imresize``
bicubic (a faithful port, ``gaussifier_sr.matlab_resize``) and quantized to 8 bits. PSNR and
SSIM use the BT.601 luma with ``scale`` border pixels removed. ``--lpips`` and ``--dists``
add the perceptual metrics (RGB, full image; ``--extra metrics``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gaussifier_sr.infer import IMAGE_SUFFIXES, make_upscaler, read_rgb, to_tensor
from gaussifier_sr.matlab_resize import imresize, matlab_downsample_uint8
from gaussifier_sr.metrics import psnr_y, ssim_y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hr-dir", type=Path, required=True)
    ap.add_argument("--scale", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--backend", default="torch", choices=["torch", "fused", "bicubic"])
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--dists", action="store_true")
    ap.add_argument("--json", type=Path, default=None, help="write per-image results")
    a = ap.parse_args()
    device = torch.device("cuda")
    upscaler = None if a.backend == "bicubic" else make_upscaler(a.backend, device)
    paths = sorted(p for p in a.hr_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    rows = []
    for s in a.scale:
        for path in paths:
            hr_np = read_rgb(path)
            h, w = hr_np.shape[0] - hr_np.shape[0] % s, hr_np.shape[1] - hr_np.shape[1] % s
            hr_np = np.ascontiguousarray(hr_np[:h, :w])
            lr_np = matlab_downsample_uint8(hr_np, s)
            hr = to_tensor(hr_np, device)
            if upscaler is None:
                up = np.clip(np.round(imresize(lr_np / 255.0, s) * 255.0), 0, 255)
                sr = to_tensor(up.astype(np.uint8), device)
            else:
                sr = upscaler(to_tensor(lr_np, device), s)
            row = {
                "image": path.stem,
                "scale": s,
                "psnr_y": psnr_y(sr, hr, s),
                "ssim_y": ssim_y(sr, hr, s),
            }
            if a.lpips:
                from gaussifier_sr.metrics import lpips_distance

                row["lpips"] = lpips_distance(sr, hr)
            if a.dists:
                from gaussifier_sr.metrics import dists_distance

                row["dists"] = dists_distance(sr, hr)
            rows.append(row)
        mine = [r for r in rows if r["scale"] == s]
        keys = [k for k in mine[0] if k not in {"image", "scale"}]
        means = "  ".join(f"{k} {np.mean([r[k] for r in mine]):.4f}" for k in keys)
        print(f"{a.hr_dir.name} x{s} ({len(mine)} images, {a.backend}): {means}", flush=True)
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
