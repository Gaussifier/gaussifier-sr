"""Gaussifier super-resolution in PyTorch (the reference for the fused CUDA runtime).

For an LR image ``y`` and scale 2:

1. The input is ``x = bicubic_up(y)``, the bicubic-upsampled LR image on the HR grid.
2. Stage 1 predicts density, log-covariance and colour maps on ``x``.
3. The point count is ``N = 4 * own(y)``, where ``own(y)`` is the model's own (adaptive)
   count on the LR image, so the HR grid gets the LR image's point density. Above the
   model's own count on ``x``, the log-covariance map is shrunk so the splatted mass is kept
   (Sigma *= own(x) / N).
4. Equal-mass Voronoi placement of N points; initial attributes are sampled from the maps.
5. K = 7 rendered states with 6 updates of the shared correction head. Each update sees the
   current render and the back-projection residual ``up(y - down(render))``, the render's
   disagreement with the observed LR image.
6. The last render is the output.

x4 (and x8) run the x2 step repeatedly (``cascade``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gaussifier_sr.base.model import (
    FirstStageModel,
    log_covariance_matrix,
    rasterize_point_count,
    sample_image_features,
    scale_rotation_from_log_covariance,
)
from gaussifier_sr.base.voronoi_samplers import production_voronoi
from gaussifier_sr.render import render

STATES = 7
POINTS_PER_PIXEL_MAX = 0.4
SEED = 1234


@dataclass(frozen=True)
class SRConfig:
    count_scale: float = 1.0  # multiplier on the point budget s^2 * own(LR)
    states: int = STATES  # rendered states K (the head runs K - 1 times)


def bicubic_up(lr: torch.Tensor, scale: float) -> torch.Tensor:
    """``(B, 3, h, w)`` -> ``(B, 3, s*h, s*w)`` bicubic (a = -0.75), clamped to [0, 1]."""
    out = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
    return out.clamp(0.0, 1.0)


@torch.no_grad()
def own_count(model: FirstStageModel, image: torch.Tensor) -> float:
    """The model's own point count for a ``(3, H, W)`` image: its predicted rate's integral."""
    return float(model.forward_map(image[None])["rate"].float().sum())


def _render(points, cov, rgb, weights, height, width) -> torch.Tensor:
    scale_xy, rotation, _ = scale_rotation_from_log_covariance(log_covariance_matrix(cov))
    colors = (rgb * weights[:, None]).clamp(0.0, 1.0)
    return render(points, scale_xy, rotation[:, 0], colors, height, width).clamp(0.0, 1.0)


@torch.no_grad()
def super_resolve(
    model: FirstStageModel,
    lr: torch.Tensor,
    *,
    budget: float | None = None,
    config: SRConfig = SRConfig(),
    seed: int = SEED,
) -> torch.Tensor:
    """One x2 pass: ``(3, h, w)`` LR in [0, 1] -> ``(3, 2h, 2w)``.

    ``budget`` replaces the point budget ``4 * own(lr)`` (the cascade passes its own).
    """
    lr_b = lr[None]
    image = bicubic_up(lr_b, 2)
    height, width = image.shape[-2:]
    maps = model.forward_map(image)
    if budget is None:
        budget = 4.0 * own_count(model, lr)
    cap = max(1, int(POINTS_PER_PIXEL_MAX * height * width))
    n = max(1, min(round(budget * config.count_scale), cap))
    n_own = max(1, int(float(maps["rate"].float().sum())))
    density = maps["density"].float()
    log_cov_map = maps["decoder_log_covariance"].float()
    rgb_map = maps["decoder_rgb"].float()
    if n > n_own:
        # Count-rescale: Sigma *= n_own / n keeps the splatted mass at the higher count.
        log_cov_map = log_cov_map.clone()
        log_cov_map[:, 0] -= math.log(n / n_own)
        log_cov_map[:, 2] -= math.log(n / n_own)
    points, weights = production_voronoi(density[0, 0], n, seed=seed)
    cov = sample_image_features(log_cov_map, points[None])[0]
    rgb = sample_image_features(rgb_map, points[None])[0].clamp(0.0, 1.0)
    xy_step = model.xy_step(height, width)
    for k in range(config.states):
        rendered = _render(points, cov, rgb, weights, height, width)
        if k == config.states - 1:
            return rendered
        down = F.interpolate(
            rendered[None], size=lr.shape[-2:], mode="bicubic", align_corners=False, antialias=True
        )
        residual = F.interpolate(
            lr_b - down, size=(height, width), mode="bicubic", align_corners=False
        )
        hard_count = rasterize_point_count(points, height, width)
        head_input = torch.cat(
            [image, density, log_cov_map, rgb_map, rendered[None], residual, hard_count], dim=1
        )
        delta = sample_image_features(model.correction_head(head_input), points[None])[0]
        cov = cov + delta[:, :3]
        rgb = (rgb + delta[:, 3:6]).clamp(0.0, 1.0)
        points = (points + delta[:, 6:8] * xy_step).clamp(0.0, 1.0)
    raise ValueError("config.states must be at least 1")


@torch.no_grad()
def cascade(
    model: FirstStageModel,
    lr: torch.Tensor,
    scale: int,
    *,
    config: SRConfig = SRConfig(),
    seed: int = SEED,
) -> torch.Tensor:
    """x4 (any power of two) as repeated x2 passes. Pass k uses the budget
    ``4^k * own(lr)`` of the original LR image, so every pass keeps its point density."""
    steps = round(math.log2(scale))
    if scale < 2 or 2**steps != scale:
        raise ValueError(f"scale {scale} is not a power of two")
    base = own_count(model, lr)
    out = lr
    for k in range(steps):
        out = super_resolve(
            model, out, budget=4 ** (k + 1) * base, config=config, seed=seed + 7919 * k
        )
    return out


def upscale(
    model: FirstStageModel,
    lr: torch.Tensor,
    scale: int = 2,
    *,
    config: SRConfig = SRConfig(),
    seed: int = SEED,
) -> torch.Tensor:
    """``(3, h, w)`` LR in [0, 1] -> ``(3, s*h, s*w)`` for s = 2, 4 or 8."""
    return cascade(model, lr, scale, config=config, seed=seed)
