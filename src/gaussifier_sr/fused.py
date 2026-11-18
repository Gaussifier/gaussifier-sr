"""Gaussifier super-resolution on the fused CUDA runtime (``cpp_inference/fused``).

``FusedSR.released()`` runs the released SR head, ``FusedSR.from_export(dir)`` any head
exported by ``tools/export_fused.py`` (the :func:`export` function below).
``sr_x2(lr)`` and ``sr(lr, 4)`` take a CUDA tensor ``(3, h, w)`` in [0, 1] and return a new
CUDA tensor ``(3, s*h, s*w)``; ``model(lr, scale)`` is ``sr``. They follow the PyTorch
reference ``gaussifier_sr.loop``:

1. the input is the bicubic x2 upsample of LR (``F.interpolate(..., scale_factor=2,
   mode="bicubic", align_corners=False).clamp(0, 1)``, in PyTorch on the GPU);
2. the point count is ``round(4 * own(LR) * count_scale)``, capped at 0.4 points per output
   pixel, where ``own(LR)`` is the sum of the model's rate on LR. A forward-map-only run of
   LR leaves that sum on the device and the x2 run reads it there (no host round trip). Above
   the model's own count of the upsampled input, the runtime shrinks the log-covariance map
   (count rescale);
3. K = 7 rendered states with six head updates; a back-projection head sees
   ``up(LR - down(render))`` (computed from LR, as in PyTorch) at every update;
4. the final render is the output.

``sr(lr, 4)`` cascades x2 steps: step k uses the budget ``4^(k+1) * own(LR)`` of the
original LR and seed ``seed + 7919 k``. Everything is enqueued on the current CUDA stream:
the calls return without waiting, the output is valid in stream order, and the inputs may be
freed or reused right away (the runtime copies them first). ``stats()`` waits and reports the
point counts of the last call.

Each output size keeps a prepared workspace (arena and CUDA graphs) in a small LRU cache, all
sharing one copy of the weights; a new size costs one allocation and one graph capture, a new
point count costs nothing. Heights and widths, LR included, must be at least 16.
"""

from __future__ import annotations

import contextlib
import ctypes as ct
import hashlib
import io
import json
import math
import os
import pickle
import shutil
import struct
import tempfile
import threading
import urllib.request
from collections import OrderedDict
from ctypes.util import find_library
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SEED = 1234
CASCADE_SEED_STRIDE = 7919  # cascade step k runs with seed + 7919 * k
STATES = 7
POINTS_PER_PIXEL_MAX = 0.4
MIN_SIZE = 16

# ----------------------------------------------------------------------------- library

_LIBRARY = "libgaussifier_fused.so.1"
_SIZE_MODES = {"pad": 0, "exact": 1}
_VP = ct.c_void_p
_LIBS: dict[str, ct.CDLL] = {}
_LIBS_LOCK = threading.Lock()
_SIGNATURES = {
    "gfused_error": ([], ct.c_char_p),
    "gfused_create": ([ct.c_char_p, ct.c_char_p, ct.c_int, ct.POINTER(_VP)], ct.c_int),
    "gfused_destroy": ([_VP], None),
    "gfused_clone": ([_VP, ct.POINTER(_VP)], ct.c_int),
    "gfused_set_max_points": ([_VP, ct.c_int], ct.c_int),
    "gfused_residual_mode": ([_VP], ct.c_int),
    "gfused_rate_sum_device": ([_VP, _VP, ct.c_int, ct.c_int, ct.c_int, _VP], ct.c_int),
    "gfused_read_rate_sum": ([_VP, _VP, ct.POINTER(ct.c_double)], ct.c_int),
    "gfused_run_device": (
        [
            _VP,
            _VP,
            _VP,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int64,
            ct.c_int,
            _VP,
            ct.c_double,
            ct.c_int,
            _VP,
            _VP,
        ],
        ct.c_int,
    ),
    "gfused_read_run_stats": (
        [_VP, _VP, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.POINTER(ct.c_double)],
        ct.c_int,
    ),
    "gfused_backprojection_residual": (
        [_VP, _VP, _VP, ct.c_int, ct.c_int, ct.c_int, ct.c_int, _VP, _VP],
        ct.c_int,
    ),
    "gfused_head_device": ([_VP, _VP, ct.c_int, ct.c_int, ct.c_int, ct.c_int, _VP, _VP], ct.c_int),
}


def library_path(explicit: str | os.PathLike | None = None) -> Path:
    """``explicit``, ``$GAUSSIFIER_SR_FUSED_LIBRARY``, or the CMake build in the repository."""
    if explicit is None:
        explicit = os.environ.get("GAUSSIFIER_SR_FUSED_LIBRARY")
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    for candidate in (
        root / "cpp_inference" / "build" / _LIBRARY,
        root / "cpp_inference" / "fused" / "build" / _LIBRARY,
    ):
        if candidate.is_file():
            return candidate
    found = find_library("gaussifier_fused")
    if found is None:
        raise FileNotFoundError(
            "libgaussifier_fused not found: build cpp_inference/fused (see its README) or set "
            "GAUSSIFIER_SR_FUSED_LIBRARY"
        )
    return Path(found)


def load_library(path: str | os.PathLike | None = None) -> ct.CDLL:
    """The runtime's C ABI (``cpp_inference/fused/python_api.h``), loaded once per path."""
    resolved = str(library_path(path))
    with _LIBS_LOCK:
        if resolved not in _LIBS:
            lib = ct.CDLL(resolved)
            for name, (argtypes, restype) in _SIGNATURES.items():
                fn = getattr(lib, name, None)
                if fn is None:
                    raise RuntimeError(f"{resolved} has no {name}: rebuild cpp_inference/fused")
                fn.argtypes, fn.restype = argtypes, restype
            _LIBS[resolved] = lib
        return _LIBS[resolved]


def _check(lib: ct.CDLL, result: int) -> int:
    if result < 0:
        raise RuntimeError(f"fused CUDA runtime: {lib.gfused_error().decode()}")
    return result


def _ptr(tensor: torch.Tensor | None) -> _VP | None:
    return None if tensor is None else _VP(tensor.data_ptr())


def _stream(device: torch.device) -> _VP:
    return _VP(torch.cuda.current_stream(device).cuda_stream)


# ----------------------------------------------------------------------------- export
#
# The runtime's weight blobs from a training checkpoint (tools/export_fused.py is the command
# line). Base release files come from $GAUSSIFIER_BASE_WEIGHTS or are downloaded once into the
# package cache; their SHA-256 is checked.

BASE_RELEASE = "https://github.com/Gaussifier/gaussifier/releases/download/v0.12.0"
BASE_SHA256 = {
    "fm_fp16.gffm": "ca999de9ae85cced4246929d5e94a3a41dd6012e7fa2a9b9f0df4f8852dbd676",
    "head_fp16.gfch": "a7e750bed52fc12a76e5af5ad98b8cee068552fde73b27afeaefe40802c286b3",
    "stage1.pt": "feeca419634a6b028bb3d7fdb6f4545ad271249b8cb023611903b29568be805f",
    "head.pt": "9c7c7edbe403b4d842bb450a573719f174fba50168fa3309293e4ba6d260394f",
}
RESIDUAL_MODES = {"image": 0, "backprojection": 1}
# The released SR head as a fused-runtime blob (tools/export_fused.py of the release sr_head.pt)
SR_RELEASE = "https://github.com/Gaussifier/gaussifier-sr/releases/download/v1.0"
SR_HEAD_BLOB = "sr_head_fp16.gfch"
SR_HEAD_BLOB_SHA256 = "221ea53b8de9452b8fdaa770d085af0136c2593a68e4b51cd10b7bb450c5392d"
GFCH_MAGIC = 0x48434647  # "GFCH"
GFFM_MAGIC = 0x4D464647  # "GFFM"
GN_EPS = 1e-5
HEAD_NETS = ("net", "aux_net")
HEAD_CONVS = [
    ("enc_blocks.0", 0),
    ("enc_blocks.0", 3),
    ("enc_blocks.1", 0),
    ("enc_blocks.1", 3),
    ("bottleneck", 0),
    ("bottleneck", 3),
    ("merge_blocks.0", 0),
    ("merge_blocks.0", 3),
    ("merge_blocks.1", 0),
    ("merge_blocks.1", 3),
]
HEAD_INPUTS = 17  # image 3, density, log_cov 3, rgb 3, rendered 3, residual 3, count
DIFF = slice(13, 16)
FOLDED_KEEP = [*range(13), 16]  # image-residual heads: 17 -> 14 inputs
FM_BLOCKS = [
    "stem",
    *(f"down_blocks.{i}.conv" for i in range(3)),
    *(f"up_blocks.{i}.conv" for i in range(3)),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_dir() -> Path:
    """``$GAUSSIFIER_BASE_WEIGHTS``, else the package cache's ``base_v0.12.0`` folder."""
    explicit = os.environ.get("GAUSSIFIER_BASE_WEIGHTS")
    if explicit:
        return Path(explicit)
    from gaussifier_sr.model import cache_dir

    return cache_dir() / "base_v0.12.0"


def base_file(name: str) -> Path:
    """A base release file, downloaded into the cache if missing; its SHA-256 is checked."""
    path = base_dir() / name
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(f"{BASE_RELEASE}/{name}", tmp)
        tmp.replace(path)
    if sha256(path) != BASE_SHA256[name]:
        raise ValueError(f"{path} is not the base release {name} (SHA-256 mismatch)")
    return path


def _fold(sd: dict, conv: str, gn: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Conv -> frozen GroupNorm as one conv: w * s, (b - mean) * s + beta (float64)."""
    w = sd[f"{conv}.weight"].double()
    s = sd[f"{gn}.weight"].double() / torch.sqrt(sd[f"{gn}.var"].double() + GN_EPS)
    b = sd.get(f"{conv}.bias", torch.zeros(w.shape[0])).double()
    b = (b - sd[f"{gn}.mean"].double()) * s + sd[f"{gn}.bias"].double()
    return w * s[:, None, None, None], b


def fold_diff(w: torch.Tensor) -> torch.Tensor:
    """Image-residual heads: ``[cout, 17, kh, kw] -> [cout, 14, kh, kw]``, with
    ``diff = image - rendered`` folded into the image (0..2) and rendered (10..12) weights."""
    w = w.clone()
    w[:, 0:3] += w[:, DIFF]
    w[:, 10:13] -= w[:, DIFF]
    return w[:, FOLDED_KEEP]


def head_convs(sd: dict, net: str, residual: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The 11 convs of one U-Net in blob order (GN folded; the out conv is a plain 1x1)."""
    convs = []
    for i, (block, idx) in enumerate(HEAD_CONVS):
        w, b = _fold(sd, f"{net}.{block}.{idx}", f"{net}.{block}.{idx + 1}")
        if i == 0:
            if w.shape[1] != HEAD_INPUTS:
                raise ValueError(f"{net}: first conv takes {w.shape[1]} channels, not 17")
            if residual == "image":
                w = fold_diff(w)
        convs.append((w, b))
    convs.append((sd[f"{net}.out.weight"].double(), sd[f"{net}.out.bias"].double()))
    return convs


def write_head(sd: dict, residual: str, path: str | os.PathLike) -> None:
    """GFCH v2 (image residual, the base release format) or v3 (back-projection), FP16.

    Layout (little endian): u32 magic, u32 version, u32 n_convs (22) [, u32 residual mode
    (v3)]; per conv: u32 ks, cin, cout, precision (0 = FP16), <pad to 16> FP16 weight
    [ks*ks][cin][cout], <pad to 16> FP32 bias [cout]. Main net then aux net, each enc0.a,
    enc0.b, enc1.a, enc1.b, mid.a, mid.b, dec0.a, dec0.b, dec1.a, dec1.b, out.
    """
    sd = {k: v.detach().cpu() for k, v in sd.items()}
    convs = [c for net in HEAD_NETS for c in head_convs(sd, net, residual)]
    if residual == "image":
        buf = bytearray(struct.pack("<III", GFCH_MAGIC, 2, len(convs)))
    else:
        buf = bytearray(struct.pack("<IIII", GFCH_MAGIC, 3, len(convs), RESIDUAL_MODES[residual]))
    for w, b in convs:
        cout, cin, kh, kw = w.shape
        buf += struct.pack("<IIII", kh, cin, cout, 0)
        buf += b"\0" * ((-len(buf)) % 16)
        taps = w.permute(2, 3, 1, 0).reshape(kh * kw, cin, cout)
        buf += taps.float().numpy().astype(np.float16).tobytes()
        buf += b"\0" * ((-len(buf)) % 16)
        buf += b.float().numpy().astype(np.float32).tobytes()
    Path(path).write_bytes(bytes(buf))


def head_state(ckpt: dict) -> dict[str, torch.Tensor]:
    """The head's state dict (``net.*`` / ``aux_net.*`` keys) from a checkpoint: its
    ``correction_head`` entry, a full model state dict, or a bare head state dict."""
    if isinstance(ckpt.get("correction_head"), dict):
        state = ckpt["correction_head"]
    elif any(k.startswith("correction_head.") for k in ckpt):
        state = {
            k.removeprefix("correction_head."): v
            for k, v in ckpt.items()
            if k.startswith("correction_head.")
        }
    else:
        state = ckpt
    if "net.net.enc_blocks.0.0.weight" in state:  # the base head.pt (training HeadWrapper)
        state = {k.removeprefix("net."): v for k, v in state.items()}
    for net in HEAD_NETS:
        if f"{net}.enc_blocks.0.0.weight" not in state:
            raise ValueError(f"no correction-head weights for {net} in the checkpoint")
    return {k: v for k, v in state.items() if torch.is_tensor(v)}


def _taps(w: torch.Tensor, cin_pad: int | None = None) -> np.ndarray:
    cout, cin, kh, kw = w.shape
    t = w.permute(2, 3, 1, 0).reshape(kh * kw, cin, cout)
    if cin_pad is not None and cin_pad > cin:
        t = torch.cat([t, t.new_zeros(kh * kw, cin_pad - cin, cout)], 1)
    return t.float().numpy().astype(np.float16)


def _fold_fm(sd: dict, conv: str, gn: str) -> tuple[torch.Tensor, torch.Tensor]:
    w = sd[f"{conv}.weight"].double()
    s = sd[f"{gn}.weight"].double() / torch.sqrt(sd[f"{gn}.var"].double() + GN_EPS)
    b = sd[f"{gn}.bias"].double() - sd[f"{gn}.mean"].double() * s
    if f"{conv}.bias" in sd:
        b = b + sd[f"{conv}.bias"].double() * s
    return w * s[:, None, None, None], b


def write_forward_map(sd: dict, path: str | os.PathLike) -> None:
    """GFFM v1 (FP16) of a frozen-GN stage-1 state dict (``backbone.*`` and the heads)."""
    sd = {k: v.detach().cpu() for k, v in sd.items() if not k.startswith("correction_head.")}
    arrays: list[np.ndarray] = []
    for i, name in enumerate(FM_BLOCKS):
        pad = 16 if i == 0 else None
        p = f"backbone.{name}"
        w1, b1 = _fold_fm(sd, f"{p}.conv1.block.0", f"{p}.conv1.block.1")
        w2, b2 = _fold_fm(sd, f"{p}.conv2.0", f"{p}.conv2.1")
        proj = sd[f"{p}.projection.weight"].double()
        arrays += [_taps(w1, pad), b1.float().numpy(), _taps(w2), b2.float().numpy()]
        arrays.append(_taps(proj, pad))
    head_w = torch.cat(
        [
            sd["density_head.head.weight"].flatten(1),
            sd["decoder_log_covariance_head.weight"].flatten(1),
            sd["decoder_rgb_head.weight"].flatten(1),
            torch.zeros(1, sd["density_head.head.weight"].shape[1]),
        ]
    )  # [8, 32]: density, log-cov 3, rgb 3, zero
    head_b = torch.cat(
        [
            sd["density_head.head.bias"],
            sd["decoder_log_covariance_head.bias"],
            sd["decoder_rgb_head.bias"],
            torch.zeros(1),
        ]
    )
    arrays += [head_w.t().contiguous().float().numpy().astype(np.float16)]
    arrays += [head_b.float().numpy()]
    buf = bytearray(struct.pack("<III", GFFM_MAGIC, 1, len(arrays)))
    for a in arrays:
        dtype = {np.dtype(np.float16): 0, np.dtype(np.float32): 1}[a.dtype]
        buf += struct.pack("<II", dtype, a.size)
        buf += b"\0" * ((-len(buf)) % 16)
        buf += np.ascontiguousarray(a).tobytes()
    Path(path).write_bytes(bytes(buf))


def load_checkpoint(path: str | os.PathLike) -> tuple[dict, str]:
    """The checkpoint and the SHA-256 of the bytes it was read from."""
    data = Path(path).read_bytes()
    try:
        ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Training checkpoints keep their options as plain Python objects; only load trusted
        # files.
        ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    return ckpt, hashlib.sha256(data).hexdigest()


def self_check() -> dict[str, bool]:
    """The base release weights through these code paths reproduce its FP16 blobs."""
    with tempfile.TemporaryDirectory() as tmp:
        head, fm = Path(tmp) / "head.gfch", Path(tmp) / "fm.gffm"
        release_head = torch.load(base_file("head.pt"), map_location="cpu", weights_only=True)
        write_head(head_state(release_head), "image", head)
        stage1 = torch.load(base_file("stage1.pt"), map_location="cpu", weights_only=True)
        write_forward_map(stage1, fm)
        result = {
            "head_matches_release": head.read_bytes() == base_file("head_fp16.gfch").read_bytes(),
            "fm_matches_release": fm.read_bytes() == base_file("fm_fp16.gffm").read_bytes(),
        }
    if not all(result.values()):
        raise RuntimeError(f"export self-check failed: {result}")
    return result


def export(
    checkpoint: str | os.PathLike,
    out: str | os.PathLike,
    *,
    residual: str | None = None,
    default_residual: str | None = "backprojection",
    fm: str | os.PathLike | None = None,
    check: bool = False,
) -> dict:
    """Write ``head.gfch``, ``fm.gffm`` and ``export.json`` for ``checkpoint`` into ``out``.

    ``residual`` overrides the checkpoint's ``args["residual"]``; ``default_residual`` is
    used when the checkpoint names none. ``fm``: the base release ``fm_fp16.gffm`` (default:
    the base weights folder, downloaded once). ``check``: run :func:`self_check` first.
    """
    checkpoint, out = Path(checkpoint), Path(out)
    ckpt, digest = load_checkpoint(checkpoint)
    args = ckpt.get("args") if isinstance(ckpt.get("args"), dict) else {}
    residual = residual or args.get("residual") or default_residual
    if residual not in RESIDUAL_MODES:
        raise ValueError(f"unknown residual {residual!r}: pass --residual image|backprojection")
    if (
        args.get("student_input", "upsampled") != "upsampled"
        or args.get("count_rule", "s2") != "s2"
    ):
        raise ValueError("the fused runtime implements student_input='upsampled', count_rule='s2'")
    head = head_state(ckpt)
    model = ckpt.get("model")
    checks = self_check() if check else {}
    out.mkdir(parents=True, exist_ok=True)
    head_path, fm_path = out / "head.gfch", out / "fm.gffm"
    write_head(head, residual, head_path)
    if model is None:
        source = Path(fm) if fm is not None else base_file("fm_fp16.gffm")
        if sha256(source) != BASE_SHA256["fm_fp16.gffm"]:
            raise ValueError(f"{source} is not the base release fm_fp16.gffm")
        shutil.copyfile(source, fm_path)
        fm_source = "base release fm_fp16.gffm (stage 1 frozen)"
    else:
        write_forward_map(model, fm_path)
        fm_source = "exported from the checkpoint's stage 1"
    meta = {
        "format": 1,
        "checkpoint": {"name": checkpoint.name, "sha256": digest, "step": ckpt.get("step")},
        "residual": residual,
        "residual_mode": RESIDUAL_MODES[residual],
        "student_input": "upsampled",
        "count_rule": "s2",
        "count_scale": float(args.get("count_scale", 1.0)),
        "cascade_count": args.get("cascade_count", "lr"),
        "points_per_pixel_max": float(args.get("points_per_pixel_max", POINTS_PER_PIXEL_MAX)),
        "states": STATES,
        "precision": "fp16",
        "head": {
            "path": head_path.name,
            "sha256": sha256(head_path),
            "format": "GFCH v2" if residual == "image" else "GFCH v3",
        },
        "fm": {"path": fm_path.name, "sha256": sha256(fm_path), "source": fm_source},
        "self_check": checks,
    }
    if meta["cascade_count"] != "lr" or not math.isclose(meta["points_per_pixel_max"], 0.4):
        raise ValueError("the fused runtime implements cascade_count='lr' at 0.4 points/pixel")
    (out / "export.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


# ----------------------------------------------------------------------------- runtime


def bicubic_x2(lr: torch.Tensor) -> torch.Tensor:
    """The x2 input: bicubic (a = -0.75, align_corners=False), clamped to [0, 1]."""
    up = F.interpolate(lr[None], scale_factor=2, mode="bicubic", align_corners=False)
    return up.clamp_(0.0, 1.0)[0].contiguous()


class FusedSR:
    """Gaussifier SR (bicubic input, s^2 count rule, K-state loop) on the fused CUDA runtime.

    ``exact_residual=False`` feeds a back-projection head ``input - up(down(render))``
    instead of ``up(LR - down(render))``: the two differ only where the bicubic upsample of
    LR leaves [0, 1] (the input is clamped). It exists to measure that difference.
    """

    def __init__(
        self,
        fm_path: str | os.PathLike,
        head_path: str | os.PathLike,
        *,
        count_scale: float = 1.0,
        states: int = STATES,
        seed: int = SEED,
        size_mode: str = "exact",
        exact_residual: bool = True,
        device: torch.device | str | int | None = None,
        library: str | os.PathLike | None = None,
        max_sessions: int = 8,
    ) -> None:
        if size_mode not in _SIZE_MODES:
            raise ValueError(f"size_mode must be one of {sorted(_SIZE_MODES)}")
        if not 1 <= states <= 16:
            raise ValueError("states must be in [1, 16]")
        if max_sessions < 3:
            raise ValueError("max_sessions must be at least 3 (LR count and two cascade steps)")
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", device)
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("the fused runtime needs a CUDA device")
        index = device.index if device.index is not None else torch.cuda.current_device()
        self.device = torch.device("cuda", index)
        self.fm_path, self.head_path = Path(fm_path), Path(head_path)
        for path in (self.fm_path, self.head_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.count_scale = float(count_scale)
        self.states = int(states)
        self.seed = int(seed)
        self.size_mode = size_mode
        self.exact_residual = bool(exact_residual)
        self.max_sessions = int(max_sessions)
        self._lib = load_library(library)
        self._lock = threading.RLock()
        self._sessions: OrderedDict[tuple, _VP] = OrderedDict()
        self._last: list[tuple[str, tuple, _VP]] = []
        self._base = _VP()
        _check(
            self._lib,
            self._lib.gfused_create(
                os.fsencode(self.fm_path),
                os.fsencode(self.head_path),
                self.device.index,
                ct.byref(self._base),
            ),
        )
        self.residual_mode = _check(self._lib, self._lib.gfused_residual_mode(self._base))

    @classmethod
    def from_export(cls, export_dir: str | os.PathLike, **kwargs) -> FusedSR:
        """Blobs and options written by :func:`export` (``tools/export_fused.py``)."""
        export_dir = Path(export_dir)
        meta = json.loads((export_dir / "export.json").read_text())
        options = {"count_scale": meta.get("count_scale", 1.0), "states": meta.get("states", 7)}
        options.update(kwargs)
        model = cls(export_dir / meta["fm"]["path"], export_dir / meta["head"]["path"], **options)
        if model.residual_mode != meta.get("residual_mode", model.residual_mode):
            model.close()
            raise ValueError("head.gfch residual mode differs from export.json")
        return model

    @classmethod
    def released(cls, **kwargs) -> FusedSR:
        """The released SR model: the base release's forward map (``fm_fp16.gffm``) and the
        SR release's head blob (``sr_head_fp16.gfch``), each downloaded once into the package
        cache and checked against its SHA-256. ``$GAUSSIFIER_BASE_WEIGHTS`` (a folder with
        ``fm_fp16.gffm``) and ``$GAUSSIFIER_SR_FUSED_HEAD`` (a head blob) use local files."""
        from gaussifier_sr.model import cache_dir

        fm = base_file("fm_fp16.gffm")
        head = Path(os.environ.get("GAUSSIFIER_SR_FUSED_HEAD") or cache_dir() / SR_HEAD_BLOB)
        if not head.is_file():
            tmp = head.with_suffix(head.suffix + ".part")
            urllib.request.urlretrieve(f"{SR_RELEASE}/{SR_HEAD_BLOB}", tmp)
            tmp.replace(head)
        if sha256(head) != SR_HEAD_BLOB_SHA256:
            raise ValueError(f"{head} is not the released {SR_HEAD_BLOB} (SHA-256 mismatch)")
        model = cls(fm, head, **kwargs)
        if model.residual_mode != RESIDUAL_MODES["backprojection"]:
            model.close()
            raise ValueError(f"{head} is not a back-projection head")
        return model

    # ------------------------------------------------------------------ sessions

    def close(self) -> None:
        """Release every workspace and the weights (waits for pending work)."""
        with self._lock:
            if self._sessions or self._base:
                torch.cuda.synchronize(self.device)
            while self._sessions:
                self._lib.gfused_destroy(self._sessions.popitem(last=False)[1])
            if self._base:
                self._lib.gfused_destroy(self._base)
                self._base = _VP()
            self._last = []

    def __enter__(self) -> FusedSR:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):  # interpreter shutdown
            self.close()

    def _session(self, key: tuple, pixels: int) -> _VP:
        session = self._sessions.get(key)
        if session is None:
            while len(self._sessions) >= self.max_sessions:
                torch.cuda.synchronize(self.device)  # its graphs may still be running
                self._lib.gfused_destroy(self._sessions.popitem(last=False)[1])
            session = _VP()
            _check(self._lib, self._lib.gfused_clone(self._base, ct.byref(session)))
            # The PyTorch loop caps the count at 0.4 points per pixel only (no 1e6 limit).
            limit = max(1_000_000, min(1 << 24, int(POINTS_PER_PIXEL_MAX * pixels)))
            _check(self._lib, self._lib.gfused_set_max_points(session, limit))
            self._sessions[key] = session
        self._sessions.move_to_end(key)
        return session

    def _image(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError("expected a torch tensor")
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"expected a (3, h, w) image, got {tuple(image.shape)}")
        if min(image.shape[-2:]) < MIN_SIZE:
            raise ValueError(f"image height and width must be at least {MIN_SIZE}")
        return image.to(device=self.device, dtype=torch.float32).contiguous()

    # ------------------------------------------------------------------ inference

    def _rate(self, image: torch.Tensor, stream: _VP) -> _VP:
        """Enqueue the forward map of ``image``; returns the session holding its rate sum."""
        h, w = image.shape[-2:]
        key = ("rate", h, w)
        session = self._session(key, h * w)
        mode = _SIZE_MODES[self.size_mode]
        _check(
            self._lib,
            self._lib.gfused_rate_sum_device(session, _ptr(image), h, w, mode, stream),
        )
        self._last.append(("rate", key, session))
        return session

    def _x2(
        self,
        lr: torch.Tensor,
        stream: _VP,
        *,
        seed: int,
        count: int | None,
        source: _VP | None,
        factor: float,
    ) -> torch.Tensor:
        """Enqueue one x2 step of ``lr`` (validated, on the device) -> ``(3, 2h, 2w)``."""
        height, width = 2 * lr.shape[-2], 2 * lr.shape[-1]
        image = bicubic_x2(lr)
        out = torch.empty(3, height, width, device=self.device)
        key = ("sr", height, width, seed)
        session = self._session(key, height * width)
        observation = lr if self.exact_residual else None
        _check(
            self._lib,
            self._lib.gfused_run_device(
                session,
                _ptr(image),
                _ptr(observation),
                height,
                width,
                self.states,
                seed,
                0 if count is None else int(count),
                source if count is None else None,
                factor,
                _SIZE_MODES[self.size_mode],
                stream,
                _ptr(out),
            ),
        )
        self._last.append(("sr", key, session))
        return out

    @torch.no_grad()
    def own_count(self, image: torch.Tensor) -> float:
        """The model's own point count of ``image``: the sum of its rate (waits for it)."""
        with self._lock:
            image = self._image(image)
            stream = _stream(self.device)
            self._last = []
            session = self._rate(image, stream)
            value = ct.c_double()
            _check(self._lib, self._lib.gfused_read_rate_sum(session, stream, ct.byref(value)))
            return value.value

    @torch.no_grad()
    def sr_x2(
        self, lr: torch.Tensor, *, count: int | None = None, seed: int | None = None
    ) -> torch.Tensor:
        """x2 super-resolution of a ``(3, h, w)`` image -> new ``(3, 2h, 2w)`` CUDA tensor.

        ``count`` pins the point count (validation); the count rescale still applies above
        the model's own count of the upsampled input.
        """
        with self._lock:
            lr = self._image(lr)
            stream = _stream(self.device)
            self._last = []
            source = self._rate(lr, stream) if count is None else None
            seed = self.seed if seed is None else int(seed)
            factor = 4.0 * self.count_scale
            return self._x2(lr, stream, seed=seed, count=count, source=source, factor=factor)

    @torch.no_grad()
    def sr(self, lr: torch.Tensor, scale: int = 4) -> torch.Tensor:
        """Power-of-two super-resolution by x2 steps; step k's point budget is ``4^(k+1)``
        times the model's own count of the original LR (``cascade_count="lr"``)."""
        steps = round(math.log2(scale)) if scale >= 2 else 0
        if scale < 2 or 2**steps != scale:
            raise ValueError(f"cascade scale {scale} is not a power of two")
        with self._lock:
            current = self._image(lr)
            stream = _stream(self.device)
            self._last = []
            source = self._rate(current, stream)
            for k in range(steps):
                current = self._x2(
                    current,
                    stream,
                    seed=self.seed + CASCADE_SEED_STRIDE * k,
                    count=None,
                    source=source,
                    factor=4.0 ** (k + 1) * self.count_scale,
                )
            return current

    def __call__(self, lr: torch.Tensor, scale: int = 2) -> torch.Tensor:
        return self.sr(lr, scale)

    def stats(self) -> dict:
        """Waits for the last call and returns its point counts.

        ``lr_rate_sum``: own(LR) before rounding; per x2 step, ``count`` (Gaussians used),
        ``own_count`` (the model's own count of that step's upsampled input, above which
        the count rescale applies) and ``rate_sum``.
        """
        with self._lock:
            stream = _stream(self.device)
            out: dict = {"steps": []}
            for kind, key, session in self._last:
                if self._sessions.get(key) is not session:
                    raise RuntimeError("the workspace of the last call was evicted")
                if kind == "rate":
                    value = ct.c_double()
                    _check(
                        self._lib,
                        self._lib.gfused_read_rate_sum(session, stream, ct.byref(value)),
                    )
                    out["lr_rate_sum"] = value.value
                    continue
                n, own, rate = ct.c_int(), ct.c_int(), ct.c_double()
                _check(
                    self._lib,
                    self._lib.gfused_read_run_stats(
                        session, stream, ct.byref(n), ct.byref(own), ct.byref(rate)
                    ),
                )
                out["steps"].append(
                    {
                        "size": key[1:3],
                        "count": n.value,
                        "own_count": own.value,
                        "rate_sum": rate.value,
                    }
                )
            return out
