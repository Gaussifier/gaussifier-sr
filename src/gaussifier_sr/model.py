"""The base Gaussifier model with the super-resolution correction head.

The base model code is vendored in ``gaussifier_sr.base``. Its weights (``stage1.pt``,
``head.pt``) are downloaded once from the base model's v0.12.0 release. The SR model is the
same network with a fine-tuned correction head (``sr_head.pt``); stage 1 is unchanged.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import torch
from torch import nn

from gaussifier_sr.base.model import FirstStageModel, fold_frozen_gn, load_frozen_gn_release
from gaussifier_sr.render._build import cache_root

BASE_RELEASE = "https://github.com/Gaussifier/gaussifier/releases/download/v0.12.0"
BASE_FILES = ("stage1.pt", "head.pt")
SR_RELEASE = "https://github.com/Gaussifier/gaussifier-sr/releases/download/v1.0"
SR_HEAD = "sr_head.pt"


def cache_dir() -> Path:
    """``~/.cache/gaussifier_sr`` (``GAUSSIFIER_SR_CACHE_DIR`` or ``XDG_CACHE_HOME`` move it)."""
    path = cache_root()
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, dest: Path) -> Path:
    if not dest.is_file():
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return dest


def base_weights_dir() -> Path:
    """Folder with the base release's ``stage1.pt`` and ``head.pt`` (downloaded once).

    ``GAUSSIFIER_BASE_WEIGHTS`` points to an existing folder instead.
    """
    explicit = os.environ.get("GAUSSIFIER_BASE_WEIGHTS")
    if explicit:
        return Path(explicit)
    target = cache_dir() / "base_v0.12.0"
    target.mkdir(parents=True, exist_ok=True)
    for name in BASE_FILES:
        download(f"{BASE_RELEASE}/{name}", target / name)
    return target


def sr_head_path() -> Path:
    """The released SR head (downloaded once; ``GAUSSIFIER_SR_HEAD`` overrides)."""
    explicit = os.environ.get("GAUSSIFIER_SR_HEAD")
    if explicit:
        return Path(explicit)
    return download(f"{SR_RELEASE}/{SR_HEAD}", cache_dir() / SR_HEAD)


def bf16_head(head: nn.Module) -> nn.Module:
    """Run the correction head under bf16 autocast on channels-last tensors (fp32 output).

    The SR head was fine-tuned with this precision.
    """
    head.to(memory_format=torch.channels_last)
    inner = head.forward

    def forward(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = inner(x.contiguous(memory_format=torch.channels_last))
        return out.float().contiguous()

    head.forward = forward
    return head


def load_model(
    device: torch.device | str = "cuda",
    *,
    sr_head: str | Path | None = None,
    fold: bool = True,
    bf16: bool = True,
) -> FirstStageModel:
    """The frozen-GN base model with the SR correction head, ready for inference.

    ``sr_head`` defaults to the released head. ``fold`` merges every calibrated affine into
    its convolution, which gives the same function with fewer kernels.
    """
    model = load_frozen_gn_release(base_weights_dir())
    state = torch.load(sr_head or sr_head_path(), map_location="cpu", weights_only=True)
    model.correction_head.load_state_dict(state["correction_head"])
    if fold:
        model = fold_frozen_gn(model)
    model = model.to(device).eval().requires_grad_(False)
    if bf16 and torch.device(device).type == "cuda":
        bf16_head(model.correction_head)
    return model
