"""CUDA path: the tile-binned native renderer (inference only)."""

from __future__ import annotations

import warnings

import torch
from torch import Tensor

from ._build import load_extension

_BUILD_ERROR: Exception | None = None


def cuda_backend_available() -> bool:
    """Build and load the extension once; on failure warn once and return False."""
    global _BUILD_ERROR
    if _BUILD_ERROR is not None:
        return False
    try:
        load_extension()
    except Exception as exc:  # missing toolkit, compiler errors, ...
        _BUILD_ERROR = exc
        warnings.warn(
            f"CUDA renderer unavailable, falling back to the PyTorch reference: {exc}",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    return True


def render_cuda(
    means: Tensor,
    scales: Tensor,
    rotation: Tensor,
    colors: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """Native render of CUDA tensors to a float32 (C, H, W) image.

    Inference-only: raises ``RuntimeError`` for inputs that require grad while
    grad mode is enabled, and if the extension cannot be built.
    """
    inputs = (means, scales, rotation, colors)
    if torch.is_grad_enabled() and any(t.requires_grad for t in inputs):
        raise RuntimeError(
            "The CUDA renderer is inference-only, but an input requires grad. Call it "
            "under torch.no_grad() or torch.inference_mode(), or use "
            "gaussifier_sr.render.render_reference (pure PyTorch) instead."
        )
    means, scales, rotation, colors = (
        t.detach().to(torch.float32).contiguous()
        for t in (means, scales, rotation.reshape(-1), colors)
    )
    return load_extension().forward(means, scales, rotation, colors, int(height), int(width))
