"""Renderer for 2D Gaussians whose colours are summed.

``render(means, scales, rotation, colors, height, width)`` returns the
(C, H, W) image

    image[:, i, j] = sum_n colors[n] * alpha_n(j, i)
    alpha_n(p) = exp(-0.5 (c_n - p)^T cov_n^-1 (c_n - p))

with no normalisation and no transmittance, where

* ``c_n = (means[n, 0] * W, means[n, 1] * H)`` is the centre in pixels, and the
  pixel in row i, column j is evaluated at the point (j, i);
* ``cov_n = (R S)(R S)^T`` with ``S = diag(scales[n])`` (standard deviations in
  pixels) and ``R = [[cos t, sin t], [-sin t, cos t]]`` for ``t = rotation[n]``
  in radians, so ``scales[n, 0]`` is the standard deviation along
  ``[cos t, -sin t]``.

A term is dropped when alpha < 1/255, and the 16x16 tile-binning rules of the
CUDA rasterizer apply (see ``gaussifier_sr.render.reference``). The projection and
tile binning follow the gsplat rasterizer (Apache-2.0).

On CUDA tensors the native extension renders the image in float32 (at most 12
channels). It is compiled on first use (see ``gaussifier_sr.render._build``)
and is inference-only. Elsewhere, or if the extension cannot be built,
``render`` falls back to ``render_reference``, its pure PyTorch twin.

Example::

    with torch.inference_mode():
        image = render(means, scales, rotation, colors, height=1024, width=1024)
"""

from __future__ import annotations

from torch import Tensor

from ._cuda import cuda_backend_available, render_cuda
from .reference import check_inputs, render_reference

__all__ = ["render", "render_reference"]


def render(
    means: Tensor,
    scales: Tensor,
    rotation: Tensor,
    colors: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """Render N Gaussians to a (C, H, W) image.

    Args:
        means: (N, 2) centres ``[x, y]`` normalised to the image, ``[0, 1]`` on-image.
        scales: (N, 2) standard deviations in pixels along the two axes.
        rotation: (N,) or (N, 1) axis angle in radians.
        colors: (N, C) colours (any sign), summed weighted by alpha.
        height, width: output size in pixels.

    Returns:
        The image: float32 from the CUDA renderer, otherwise in the promoted
        dtype of the inputs.
    """
    check_inputs(means, scales, rotation, colors, height, width)
    if means.is_cuda and cuda_backend_available():
        return render_cuda(means, scales, rotation, colors, int(height), int(width))
    return render_reference(means, scales, rotation, colors, int(height), int(width))
