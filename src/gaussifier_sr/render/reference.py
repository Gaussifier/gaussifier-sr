"""Pure PyTorch twin of the CUDA renderer: the CPU fallback and the test oracle.

It applies the CUDA forward's rules, so both paths agree up to float rounding:

* Each Gaussian gets a pixel-space centre, a conic (inverse covariance) and a
  binning radius ``ceil(3 sqrt(lambda_max))``, where ``lambda_max`` is the
  largest covariance eigenvalue computed with the discriminant floored at 0.1.
  A Gaussian with a singular covariance is dropped.
* Its bounding square of half-width ``radius`` is mapped to 16x16 tiles. A
  Gaussian touching more than 32 tiles is "large" and evaluated on the whole
  image. A smaller one is evaluated only on its tiles, which can clip its
  alpha >= 1/255 ellipse (3 to 3.33 sigma) at tile edges.
* If the small Gaussians touch more than ``max(8 N, 4096)`` tiles in total (the
  CUDA tile-list capacity), every Gaussian is evaluated on the whole image.
* A term is kept when ``sigma = 0.5 d^T conic d`` is finite and >= 0 and
  ``alpha = exp(-sigma) >= 1/255``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

TILE_SIZE = 16
LARGE_TILE_THRESHOLD = 32
ALPHA_MIN = 1.0 / 255.0
# Gaussian-pixel pairs evaluated at once; bounds the reference's memory use.
_PAIRS_PER_CHUNK = 1 << 22


def tile_list_capacity(num_points: int) -> int:
    """Length of the CUDA forward's tile list for ``num_points`` Gaussians."""
    return max(8 * num_points, 4096)


def check_inputs(
    means: Tensor,
    scales: Tensor,
    rotation: Tensor,
    colors: Tensor,
    height: int,
    width: int,
) -> None:
    """Raise ``ValueError`` unless the arguments describe N Gaussians and an image."""
    if means.dim() != 2 or means.shape[1] != 2:
        raise ValueError(f"means must be (N, 2), got {tuple(means.shape)}")
    n = means.shape[0]
    if scales.shape != (n, 2):
        raise ValueError(f"scales must be ({n}, 2), got {tuple(scales.shape)}")
    if rotation.shape not in ((n,), (n, 1)):
        raise ValueError(f"rotation must be ({n},) or ({n}, 1), got {tuple(rotation.shape)}")
    if colors.dim() != 2 or colors.shape[0] != n or colors.shape[1] < 1:
        raise ValueError(f"colors must be ({n}, C) with C >= 1, got {tuple(colors.shape)}")
    if int(height) < 1 or int(width) < 1:
        raise ValueError(f"height and width must be positive, got {height} x {width}")
    if len({t.device for t in (means, scales, rotation, colors)}) != 1:
        raise ValueError("means, scales, rotation and colors must be on one device")


class Projection(NamedTuple):
    center: Tensor  # (N, 2) centres in pixels
    conic: Tensor  # (N, 3) upper triangle (a, b, c) of the inverse covariance
    radius: Tensor  # (N,) binning radius in pixels
    valid: Tensor  # (N,) False where the covariance is singular


def project(means: Tensor, scales: Tensor, rotation: Tensor, height: int, width: int) -> Projection:
    """Pixel-space centres, conics and binning radii, as the CUDA kernels compute them."""
    center = torch.stack((means[:, 0] * width, means[:, 1] * height), dim=-1)
    theta = rotation.reshape(-1)
    cos_t, sin_t = theta.cos(), theta.sin()
    # M = R S with R = [[cos, sin], [-sin, cos]] and S = diag(scales); cov = M M^T.
    m00, m01 = cos_t * scales[:, 0], sin_t * scales[:, 1]
    m10, m11 = -sin_t * scales[:, 0], cos_t * scales[:, 1]
    cov_xx = m00 * m00 + m01 * m01
    cov_xy = m00 * m10 + m01 * m11
    cov_yy = m10 * m10 + m11 * m11
    det = cov_xx * cov_yy - cov_xy * cov_xy
    valid = det != 0
    # Zero conic for singular covariances, without dividing by zero.
    safe_det = torch.where(valid, det, torch.ones_like(det))
    inv_det = torch.where(valid, 1.0 / safe_det, torch.zeros_like(det))
    conic = torch.stack((cov_yy * inv_det, -cov_xy * inv_det, cov_xx * inv_det), dim=-1)
    with torch.no_grad():
        half_trace = 0.5 * (cov_xx + cov_yy)
        disc = torch.fmax(half_trace * half_trace - det, torch.full_like(det, 0.1)).sqrt()
        radius = (3.0 * torch.fmax(half_trace + disc, half_trace - disc).sqrt()).ceil()
    return Projection(center, conic, radius, valid)


class TileRules(NamedTuple):
    rect: Tensor  # (N, 4) int64 (x0, y0, x1, y1): pixels [x0, x1) x [y0, y1) evaluated
    large: Tensor  # (N,) bool, more than LARGE_TILE_THRESHOLD tiles
    overflow: bool  # the small Gaussians overflow the tile list


@torch.no_grad()
def tile_rules(projection: Projection, height: int, width: int) -> TileRules:
    """Pixel rectangle in which the CUDA forward evaluates each Gaussian."""
    center, radius, valid = projection.center, projection.radius, projection.valid
    r = radius / TILE_SIZE

    def tile_span(coord: Tensor, num_tiles: int) -> tuple[Tensor, Tensor]:
        # C++ (int) casts truncate toward zero; NaN becomes 0.
        c = coord / TILE_SIZE
        lo = torch.trunc(c - r).nan_to_num(0.0).clamp(0, num_tiles)
        hi = torch.trunc(c + r + 1).nan_to_num(0.0).clamp(0, num_tiles)
        return lo.long(), hi.long()

    tx0, tx1 = tile_span(center[:, 0], -(-width // TILE_SIZE))
    ty0, ty1 = tile_span(center[:, 1], -(-height // TILE_SIZE))
    hits = (tx1 - tx0) * (ty1 - ty0)
    binned = valid & (radius > 0)
    large = binned & (hits > LARGE_TILE_THRESHOLD)
    small = binned & ~large
    overflow = int(hits[small].sum()) > tile_list_capacity(center.shape[0])

    rect = torch.stack((tx0, ty0, tx1, ty1), dim=-1) * TILE_SIZE
    rect = torch.where(small[:, None], rect, torch.zeros_like(rect))
    whole = torch.tensor([0, 0, width, height], device=rect.device)
    everywhere = valid if overflow else large
    rect = torch.where(everywhere[:, None], whole, rect)
    return TileRules(rect, large, overflow)


def _compute_dtype(*tensors: Tensor) -> torch.dtype:
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype if dtype in (torch.float32, torch.float64) else torch.float32


def render_reference(
    means: Tensor,
    scales: Tensor,
    rotation: Tensor,
    colors: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """Pure PyTorch render with the CUDA path's rules (see the module docstring).

    Arguments are as for ``gaussifier_sr.render.render``. The image is computed
    in the promoted dtype of the inputs (float32 for half precision) on their
    device. Any channel count works.
    """
    check_inputs(means, scales, rotation, colors, height, width)
    height, width = int(height), int(width)
    dtype = _compute_dtype(means, scales, rotation, colors)
    means, scales, rotation, colors = (t.to(dtype) for t in (means, scales, rotation, colors))
    num_points, channels = colors.shape
    image = colors.new_zeros(channels, height, width)
    if num_points == 0:
        return image

    projection = project(means, scales, rotation, height, width)
    rules = tile_rules(projection, height, width)
    xs = torch.arange(width, device=means.device)
    ys = torch.arange(height, device=means.device)
    px, py = xs.to(dtype), ys.to(dtype)
    chunk = max(1, _PAIRS_PER_CHUNK // (height * width))
    for start in range(0, num_points, chunk):
        part = slice(start, start + chunk)
        cx = projection.center[part, 0, None, None]
        cy = projection.center[part, 1, None, None]
        a, b, c = projection.conic[part, :, None, None].unbind(1)
        dx = cx - px  # (n, 1, W)
        dy = cy - py[:, None]  # (n, H, 1)
        sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy  # (n, H, W)
        x0, y0, x1, y1 = rules.rect[part, :, None, None].unbind(1)
        inside = (xs >= x0) & (xs < x1) & (ys[:, None] >= y0) & (ys[:, None] < y1)
        keep = inside & (sigma >= 0) & torch.isfinite(sigma)
        alpha = torch.exp(-torch.where(keep, sigma, torch.zeros_like(sigma)))
        weight = torch.where(keep & (alpha >= ALPHA_MIN), alpha, torch.zeros_like(alpha))
        image = image + torch.einsum("nhw,nc->chw", weight, colors[part])
    return image
