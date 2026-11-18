"""Just-in-time build of the CUDA extension with ``torch.utils.cpp_extension``.

The extension is compiled on first use into ``<cache>/build/<key>``, where
``<cache>`` is ``$GAUSSIFIER_SR_CACHE_DIR`` if set, else
``$XDG_CACHE_HOME/gaussifier_sr`` (``~/.cache/gaussifier_sr`` by default), and
``<key>`` hashes the sources, the compiler flags, the torch, CUDA and Python
versions and the target GPU architectures. ``TORCH_CUDA_ARCH_LIST`` selects the
architectures as for any torch extension (default: those of the visible GPUs).
Set ``GAUSSIFIER_SR_VERBOSE_BUILD=1`` to see the compiler commands.
"""

from __future__ import annotations

import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch

_CSRC = Path(__file__).resolve().parent / "csrc"
_SOURCES = ("ext.cpp", "forward.cu")
_HEADERS = ("common.cuh", "render.h")
_CFLAGS = ("-O3",)
_CUDA_CFLAGS = ("-O3", "--use_fast_math")


def cache_root() -> Path:
    """Package cache directory; builds go to its ``build`` subdirectory."""
    override = os.environ.get("GAUSSIFIER_SR_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "gaussifier_sr"


def _target_archs() -> str:
    archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if archs or not torch.cuda.is_available():
        return archs
    capabilities = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
    return ";".join(f"{major}.{minor}" for major, minor in sorted(capabilities))


def build_key() -> str:
    """Hash of everything that changes the compiled extension."""
    digest = hashlib.sha256()
    for name in (*_SOURCES, *_HEADERS):
        digest.update(name.encode())
        digest.update((_CSRC / name).read_bytes())
    for part in (
        *_CFLAGS,
        *_CUDA_CFLAGS,
        torch.__version__,
        str(torch.version.cuda),
        sys.implementation.cache_tag or "",
        _target_archs(),
    ):
        digest.update(b"\0" + part.encode())
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def load_extension() -> ModuleType:
    """Compile (first call only) and import the extension."""
    from torch.utils.cpp_extension import load

    key = build_key()
    build_dir = cache_root() / "build" / key
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=f"gaussifier_sr_render_{key}",
        sources=[str(_CSRC / name) for name in _SOURCES],
        extra_cflags=list(_CFLAGS),
        extra_cuda_cflags=list(_CUDA_CFLAGS),
        build_directory=str(build_dir),
        verbose=os.environ.get("GAUSSIFIER_SR_VERBOSE_BUILD") == "1",
    )
