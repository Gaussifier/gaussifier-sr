"""MATLAB ``imresize(img, 1/s, 'bicubic')`` port (antialiased Keys a=-0.5 kernel).

SR benchmark tables (Set5/Set14/B100/Urban100) are built from MATLAB-downsampled LR
images; PIL and torch bicubic differ by a few hundredths of a dB of input error, which
shows up as ~0.1-0.4 dB in reported PSNR. Follows the standard MATLAB algorithm
(contributions per output pixel, symmetric edge padding).
"""

from __future__ import annotations

import math

import numpy as np


def _cubic(x: np.ndarray) -> np.ndarray:
    absx = np.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + (
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    ) * ((absx > 1) & (absx <= 2))


def _contributions(in_length: int, out_length: int, scale: float):
    kernel_width = 4.0
    if scale < 1:
        kernel_width /= scale
    x = np.arange(1, out_length + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    p = math.ceil(kernel_width) + 2
    indices = left[:, None] + np.arange(p)[None, :]
    distance = u[:, None] - indices
    weights = scale * _cubic(distance * scale) if scale < 1 else _cubic(distance)
    weights = weights / weights.sum(axis=1, keepdims=True)
    keep = np.any(weights != 0, axis=0)
    weights = weights[:, keep]
    indices = indices[:, keep]
    # Symmetric (mirror) padding, 1-based -> 0-based.
    aux = np.concatenate([np.arange(in_length), np.arange(in_length - 1, -1, -1)])
    indices = aux[np.mod(indices - 1, aux.size).astype(np.int64)]
    return weights, indices


def _resize_axis(image: np.ndarray, out_length: int, scale: float, axis: int) -> np.ndarray:
    weights, indices = _contributions(image.shape[axis], out_length, scale)
    moved = np.moveaxis(image, axis, 0)
    gathered = moved[indices]  # (out, p, ...)
    w = weights.reshape(weights.shape + (1,) * (moved.ndim - 1))
    return np.moveaxis((gathered * w).sum(axis=1), 0, axis)


def imresize(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize float ``(H, W, C)`` by ``scale`` like MATLAB; returns float64, unclipped."""
    image = image.astype(np.float64)
    out_h = math.ceil(image.shape[0] * scale)
    out_w = math.ceil(image.shape[1] * scale)
    # MATLAB resizes the dimension with the smaller scale change first; equal here.
    out = _resize_axis(image, out_h, scale, axis=0)
    return _resize_axis(out, out_w, scale, axis=1)


def matlab_downsample_uint8(hr_uint8: np.ndarray, scale: int) -> np.ndarray:
    """``(H, W, 3)`` uint8 HR (H, W divisible by ``scale``) -> MATLAB bicubic LR uint8."""
    lr = imresize(hr_uint8.astype(np.float64) / 255.0, 1.0 / scale)
    return np.clip(np.round(lr * 255.0), 0, 255).astype(np.uint8)
