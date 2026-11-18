"""Image-quality metrics for the SR benchmarks: PSNR (Y), SSIM (Y), LPIPS, DISTS and FID.

Inputs are float RGB in [0, 1]: one ``(3, H, W)`` image, or a ``(B, 3, H, W)`` batch / a
sequence of ``(3, H, W)`` tensors of any sizes, on any device. Every metric first clamps and
quantizes to 8 bits, ``(x.clamp(0, 1) * 255).round() / 255``, so it scores the PNG the
prediction would be saved as (the deep metrics take ``quantize=False`` to skip this). Same-size
images are batched on the GPU; images of different sizes run separately. The networks run in
IEEE fp32 (TF32 and autocast off, see ``exact_fp32``): PyTorch's TF32 default would move DISTS
and LPIPS by ~1e-5 and FID features by ~1e-3 relative, depending on the GPU generation.

* ``psnr_y``: BT.601 luma of the 8-bit images, ``shave`` border pixels removed (BasicSR
  ``calculate_psnr`` with ``test_y_channel=True``).
* ``ssim_y`` / ``ssim_y_batch``: the standard SR protocol (BasicSR ``calculate_ssim`` with
  ``test_y_channel=True``; MATLAB ``ssim_index`` in the RCAN ``Cal_Y_PSNRSSIM`` script).
  BT.601 luma of the 8-bit image, ``16 + 65.481 R + 128.553 G + 24.966 B`` for RGB in [0, 1]
  (in [16, 235], not rounded), ``shave`` border pixels removed, 11 x 11 Gaussian window with
  sigma 1.5, C1 = (0.01 * 255)^2, C2 = (0.03 * 255)^2, 'valid' filtering, mean of the SSIM map.
  Float64 on the GPU.
* ``lpips_distance`` / ``lpips_batch``: LPIPS v0.1 with AlexNet (the installed ``lpips``
  package; the variant most SR papers report), RGB scaled to [-1, 1].
* ``dists_distance`` / ``dists_batch``: the official DISTS code (``third_party/dists``), RGB in
  [0, 1], full image, no resize (pyiqa's default). ``resize_256=True`` is the variant of the
  official README: ``prepare_image`` (PIL bilinear resize of the short side to 256 when it is
  longer than 256).
* ``fid_features`` / ``frechet_distance``: pytorch-fid (``third_party/pytorch_fid``), pool3
  features (2048-d) of the FID Inception, input resized bilinearly to 299 x 299 by the network
  (``resize_input=True``, no antialiasing) and scaled to [-1, 1] (``normalize_input=True``),
  full images; the Frechet distance is pytorch-fid's ``calculate_frechet_distance``.

``shave`` (LPIPS, DISTS) crops border pixels first; it is 0 by default (pyiqa). GSASR crops
``scale`` pixels (at most 8) for PSNR, SSIM, LPIPS and DISTS alike.

Weights (downloaded on first use into ``~/.cache/gaussifier_sr/metrics`` or the torch hub
cache; set ``GAUSSIFIER_SR_METRIC_WEIGHTS`` to use another folder):

* DISTS alpha/beta: ``DISTS_pytorch/weights.pt`` of github.com/dingkeyan93/DISTS (commit
  1267d8cb; sha256 f5e65c96..., the file pyiqa ships as ``DISTS_weights-f5e65c96.pth``);
* VGG16 / AlexNet ImageNet weights: torchvision's checkpoints in the torch hub cache;
* FID Inception: ``pt_inception-2015-12-05-6726825d.pth`` of the pytorch-fid ``fid_weights``
  release (sha256 6726825d...).
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.util
import math
import os
import sys
import urllib.request
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch.nn.functional as F

THIRD_PARTY = Path(__file__).resolve().parent / "third_party"
WEIGHTS_DIR = Path(
    os.environ.get(
        "GAUSSIFIER_SR_METRIC_WEIGHTS", Path.home() / ".cache" / "gaussifier_sr" / "metrics"
    )
)
DISTS_WEIGHTS = WEIGHTS_DIR / "DISTS_weights-f5e65c96.pt"
FID_INCEPTION_WEIGHTS = WEIGHTS_DIR / "pt_inception-2015-12-05-6726825d.pth"
TORCHVISION_CHECKPOINTS = Path(torch.hub.get_dir()) / "checkpoints"
VGG16_WEIGHTS = TORCHVISION_CHECKPOINTS / "vgg16-397923af.pth"
ALEXNET_WEIGHTS = TORCHVISION_CHECKPOINTS / "alexnet-owt-7be5be79.pth"
# Official download locations, fetched on first use.
WEIGHT_URLS = {
    DISTS_WEIGHTS: "https://github.com/dingkeyan93/DISTS/raw/master/DISTS_pytorch/weights.pt",
    FID_INCEPTION_WEIGHTS: "https://github.com/mseitzer/pytorch-fid/releases/download/"
    "fid_weights/pt_inception-2015-12-05-6726825d.pth",
    VGG16_WEIGHTS: "https://download.pytorch.org/models/vgg16-397923af.pth",
    ALEXNET_WEIGHTS: "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
}

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_C1 = (0.01 * 255) ** 2
SSIM_C2 = (0.03 * 255) ** 2
FID_DIMS = 2048

Images = torch.Tensor | Sequence[torch.Tensor]


# --------------------------------------------------------------------------- shared helpers


@contextlib.contextmanager
def exact_fp32() -> Iterator[None]:
    """No autograd, no autocast and no TF32 (cuDNN convolutions, CUDA matmuls) in the block.

    PyTorch runs fp32 convolutions in TF32 by default on Ampere and newer GPUs (relative error
    ~1e-3 instead of ~1e-6). The flags are global and restored on exit.
    """
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            stack.enter_context(torch.autocast("cpu", enabled=False))
            if torch.cuda.is_available():
                stack.enter_context(torch.autocast("cuda", enabled=False))
            yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


def quantize_8bit(x: torch.Tensor) -> torch.Tensor:
    """``(x.clamp(0, 1) * 255).round() / 255`` in float32: the values of the saved 8-bit PNG.

    Bit-identical to ``torchvision.transforms.ToTensor()`` of that PNG.
    """
    return (x.float().clamp(0, 1) * 255).round() / 255


def _device(device: torch.device | str | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _as_list(images: Images, name: str) -> list[torch.Tensor]:
    """``(3, H, W)``, ``(B, 3, H, W)`` or a sequence of ``(3, H, W)`` -> list of ``(3, H, W)``."""
    if isinstance(images, torch.Tensor):
        if images.ndim == 3:
            images = images[None]
        if images.ndim != 4:
            raise ValueError(
                f"{name}: expected (3, H, W) or (B, 3, H, W), got {tuple(images.shape)}"
            )
        items = list(images.unbind(0))
    else:
        items = list(images)
    for x in items:
        if x.ndim != 3 or x.shape[0] != 3:
            raise ValueError(f"{name}: expected RGB images (3, H, W), got {tuple(x.shape)}")
    return items


def _pairs(preds: Images, refs: Images) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    preds, refs = _as_list(preds, "preds"), _as_list(refs, "refs")
    if len(preds) != len(refs):
        raise ValueError(f"{len(preds)} predictions but {len(refs)} references")
    for i, (p, r) in enumerate(zip(preds, refs, strict=True)):
        if p.shape != r.shape:
            raise ValueError(f"pair {i}: shapes differ, {tuple(p.shape)} vs {tuple(r.shape)}")
    return preds, refs


def _batches(
    shapes: Sequence[tuple[int, int]], batch_size: int, max_pixels: int
) -> Iterator[list[int]]:
    """Index lists of same-size images: at most ``batch_size`` images and ``max_pixels`` pixels
    per batch (one image at least)."""
    groups: dict[tuple[int, int], list[int]] = {}
    for i, shape in enumerate(shapes):
        groups.setdefault((int(shape[0]), int(shape[1])), []).append(i)
    for (h, w), indices in groups.items():
        step = max(1, min(batch_size, max_pixels // (h * w)))
        for start in range(0, len(indices), step):
            yield indices[start : start + step]


def _crop(x: torch.Tensor, shave: int) -> torch.Tensor:
    if shave < 0:
        raise ValueError(f"shave must be >= 0, got {shave}")
    h, w = x.shape[-2:]
    return x[..., shave : h - shave, shave : w - shave]


def _require(path: Path, what: str) -> Path:
    if not path.is_file() and path in WEIGHT_URLS:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(WEIGHT_URLS[path], tmp)
        tmp.replace(path)
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def _load_vendored(name: str, path: Path) -> ModuleType:
    """Import vendored third-party code from an explicit file path (never through sys.path)."""
    if name in sys.modules:
        return sys.modules[name]
    _require(path, "vendored module")
    search = [str(path.parent)] if path.name == "__init__.py" else None
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=search)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def vendored_dists() -> ModuleType:
    """The vendored official ``DISTS_pt`` module (``third_party/dists``)."""
    return _load_vendored("gaussifier_sr_third_party_dists", THIRD_PARTY / "dists" / "DISTS_pt.py")


def vendored_fid_score() -> ModuleType:
    """The vendored pytorch-fid ``fid_score`` module (``third_party/pytorch_fid``)."""
    package = "gaussifier_sr_third_party_pytorch_fid"
    _load_vendored(package, THIRD_PARTY / "pytorch_fid" / "__init__.py")
    return importlib.import_module(f"{package}.fid_score")


# --------------------------------------------------------------------------- SSIM (Y channel)


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """BT.601 luma in [16, 235], ``(..., 3, H, W) -> (..., H, W)`` in float64.

    MATLAB ``rgb2ycbcr`` / BasicSR ``bgr2ycbcr(y_only=True) * 255`` for RGB in [0, 1].
    """
    x = x.double()
    return 16 + 65.481 * x[..., 0, :, :] + 128.553 * x[..., 1, :, :] + 24.966 * x[..., 2, :, :]


def psnr_y(pred: torch.Tensor, ref: torch.Tensor, shave: int) -> float:
    """PSNR (dB) on the BT.601 luma of the 8-bit ``(3, H, W)`` images, ``shave`` px cropped."""
    d = rgb_to_y(quantize_8bit(pred.float())) - rgb_to_y(quantize_8bit(ref.float()))
    if shave:
        d = d[..., shave:-shave, shave:-shave]
    return 10.0 * math.log10(255.0**2 / float(d.double().square().mean()))


def _ssim_valid(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean SSIM map of ``(N, H, W)`` planes with 'valid' Gaussian filtering -> ``(N,)``."""
    coords = torch.arange(SSIM_WINDOW, dtype=x.dtype, device=x.device) - (SSIM_WINDOW - 1) / 2
    window = torch.exp(-(coords**2) / (2 * SSIM_SIGMA**2))
    window = window / window.sum()
    maps = torch.stack([x, y, x * x, y * y, x * y], dim=1)
    # Separable 11 x 11 window: along W, then along H, no padding.
    maps = F.conv2d(maps, window.view(1, 1, 1, -1).repeat(5, 1, 1, 1), groups=5)
    maps = F.conv2d(maps, window.view(1, 1, -1, 1).repeat(5, 1, 1, 1), groups=5)
    mu_x, mu_y, e_xx, e_yy, e_xy = maps.unbind(1)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    numerator = (2 * mu_xy + SSIM_C1) * (2 * (e_xy - mu_xy) + SSIM_C2)
    denominator = (mu_xx + mu_yy + SSIM_C1) * ((e_xx - mu_xx) + (e_yy - mu_yy) + SSIM_C2)
    return (numerator / denominator).mean(dim=(-2, -1))


def ssim_y_batch(
    preds: Images,
    refs: Images,
    shave: int,
    *,
    device: torch.device | str | None = None,
    batch_size: int = 16,
    max_pixels: int = 1 << 22,
) -> np.ndarray:
    """SSIM on the Y channel (standard SR protocol, see module docstring) for each image pair.

    ``preds`` / ``refs``: ``(B, 3, H, W)`` or sequences of ``(3, H, W)`` RGB in [0, 1], pairs of
    equal size (sizes may differ between pairs). Returns ``(N,)`` float64.
    """
    preds, refs = _pairs(preds, refs)
    device = _device(device)
    out = np.empty(len(preds), dtype=np.float64)
    shapes = [p.shape[-2:] for p in preds]
    for h, w in set(shapes):
        if min(h, w) - 2 * shave < SSIM_WINDOW:
            raise ValueError(f"{h}x{w} image with shave {shave} is smaller than the SSIM window")
    with exact_fp32():
        for idx in _batches(shapes, batch_size, max_pixels):
            planes = []
            for images in (preds, refs):
                batch = torch.stack([images[i] for i in idx]).to(device)
                levels = (batch.float().clamp(0, 1) * 255).round().double()  # 8-bit values
                planes.append(_crop(rgb_to_y(levels / 255), shave))
            out[idx] = _ssim_valid(*planes).cpu().numpy()
    return out


def ssim_y(
    pred: torch.Tensor, ref: torch.Tensor, shave: int, *, device: torch.device | str | None = None
) -> float:
    """SSIM on the Y channel of one ``(3, H, W)`` pair (see ``ssim_y_batch``)."""
    if pred.ndim != 3:
        raise ValueError(f"ssim_y takes one (3, H, W) image, got {tuple(pred.shape)}")
    return float(ssim_y_batch([pred], [ref], shave, device=device)[0])


# --------------------------------------------------------------------------- LPIPS


@functools.cache
def _lpips_model(device: torch.device) -> torch.nn.Module:
    import lpips

    _require(ALEXNET_WEIGHTS, "torchvision AlexNet ImageNet weights")
    # lpips 0.1.4 builds torchvision's alexnet(pretrained=True) (read from the cache above).
    with torch.random.fork_rng(devices=[]), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module=r"torchvision\.models\._utils"
        )
        model = lpips.LPIPS(net="alex", version="0.1", verbose=False)
    return model.eval().requires_grad_(False).to(device)


def lpips_batch(
    preds: Images,
    refs: Images,
    *,
    shave: int = 0,
    quantize: bool = True,
    device: torch.device | str | None = None,
    batch_size: int = 8,
    max_pixels: int = 1 << 22,
) -> np.ndarray:
    """LPIPS v0.1 (AlexNet) of each pair, RGB in [0, 1] scaled to [-1, 1]. ``(N,)`` float64."""
    preds, refs = _pairs(preds, refs)
    device = _device(device)
    model = _lpips_model(device)
    prepare = quantize_8bit if quantize else torch.Tensor.float
    out = np.empty(len(preds), dtype=np.float64)
    with exact_fp32():
        for idx in _batches([_crop(p, shave).shape[-2:] for p in preds], batch_size, max_pixels):
            x, y = (
                torch.stack([prepare(_crop(images[i], shave)) for i in idx]).to(device)
                for images in (preds, refs)
            )
            out[idx] = model(2 * x - 1, 2 * y - 1).reshape(-1).double().cpu().numpy()
    return out


def lpips_distance(pred: torch.Tensor, ref: torch.Tensor, **kwargs) -> float:
    """LPIPS of one ``(3, H, W)`` pair (see ``lpips_batch``)."""
    if pred.ndim != 3:
        raise ValueError(f"lpips_distance takes one (3, H, W) image, got {tuple(pred.shape)}")
    return float(lpips_batch([pred], [ref], **kwargs)[0])


# --------------------------------------------------------------------------- DISTS


@functools.cache
def _dists_model(device: torch.device) -> torch.nn.Module:
    _require(DISTS_WEIGHTS, "DISTS alpha/beta weights")
    _require(VGG16_WEIGHTS, "torchvision VGG16 ImageNet weights")
    with torch.random.fork_rng(devices=[]):  # construction draws random alpha/beta first
        model = vendored_dists().DISTS(weights_path=str(DISTS_WEIGHTS), vgg_path=str(VGG16_WEIGHTS))
    return model.eval().to(device)


def _dists_resize_256(x: torch.Tensor) -> torch.Tensor:
    """The official README preprocessing, ``prepare_image(PIL image, resize=True)``."""
    from PIL import Image

    array = (x.float().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return vendored_dists().prepare_image(Image.fromarray(array), resize=True)[0]


def dists_batch(
    preds: Images,
    refs: Images,
    *,
    resize_256: bool = False,
    shave: int = 0,
    quantize: bool = True,
    device: torch.device | str | None = None,
    batch_size: int = 8,
    max_pixels: int = 1 << 21,
) -> np.ndarray:
    """DISTS (official code and weights) of each pair, RGB in [0, 1]. ``(N,)`` float64.

    Full images by default; ``resize_256`` applies the official README resize (after ``shave``),
    which needs ``quantize`` (it goes through 8-bit PIL images).
    """
    preds, refs = _pairs(preds, refs)
    if resize_256 and not quantize:
        raise ValueError("resize_256 resizes 8-bit PIL images; it needs quantize=True")
    device = _device(device)
    model = _dists_model(device)
    prepare = quantize_8bit if quantize else torch.Tensor.float
    if resize_256:  # small CPU images, prepared up front
        preds = [_dists_resize_256(_crop(p, shave)) for p in preds]
        refs = [_dists_resize_256(_crop(r, shave)) for r in refs]
        prepare, shave = torch.Tensor.float, 0
    out = np.empty(len(preds), dtype=np.float64)
    with exact_fp32():
        for idx in _batches([_crop(p, shave).shape[-2:] for p in preds], batch_size, max_pixels):
            x, y = (
                torch.stack([prepare(_crop(images[i], shave)) for i in idx]).to(device)
                for images in (preds, refs)
            )
            out[idx] = model(x, y).reshape(-1).double().cpu().numpy()
    return out


def dists_distance(pred: torch.Tensor, ref: torch.Tensor, **kwargs) -> float:
    """DISTS of one ``(3, H, W)`` pair (see ``dists_batch``)."""
    if pred.ndim != 3:
        raise ValueError(f"dists_distance takes one (3, H, W) image, got {tuple(pred.shape)}")
    return float(dists_batch([pred], [ref], **kwargs)[0])


# --------------------------------------------------------------------------- FID


@functools.cache
def _fid_model(device: torch.device) -> torch.nn.Module:
    _require(FID_INCEPTION_WEIGHTS, "FID Inception weights")
    inception = vendored_fid_score().InceptionV3
    with torch.random.fork_rng(devices=[]):  # default init before the weights are loaded
        model = inception(
            [inception.BLOCK_INDEX_BY_DIM[FID_DIMS]],
            resize_input=True,
            normalize_input=True,
            weights_path=str(FID_INCEPTION_WEIGHTS),
        )
    return model.eval().to(device)


def fid_features(
    images: Images,
    *,
    quantize: bool = True,
    device: torch.device | str | None = None,
    batch_size: int = 50,
    max_pixels: int = 1 << 24,
) -> np.ndarray:
    """Pool3 features of the FID Inception for each image, ``(N, 2048)`` float64.

    pytorch-fid's ``get_activations``: full RGB images in [0, 1] (any sizes; the network
    resizes them bilinearly to 299 x 299), float32 activations stored as float64.

    Same-size images go through the network together (``batch_size``, pytorch-fid's default
    50), which changes their features by up to ~1e-4 relative: cuDNN picks other algorithms,
    and fp32 features are only that close to a float64 network anyway. ``batch_size=1`` makes
    every image's features independent of the others and bit-identical to pytorch-fid's
    ``get_activations(batch_size=1)``, the only setting pytorch-fid accepts for sets of
    mixed sizes.
    """
    images = _as_list(images, "images")
    device = _device(device)
    model = _fid_model(device)
    prepare = quantize_8bit if quantize else torch.Tensor.float
    out = np.empty((len(images), FID_DIMS), dtype=np.float64)
    with exact_fp32():
        for idx in _batches([x.shape[-2:] for x in images], batch_size, max_pixels):
            batch = torch.stack([prepare(images[i]) for i in idx]).to(device)
            out[idx] = model(batch)[0].squeeze(3).squeeze(2).cpu().numpy()
    return out


def fid_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance as pytorch-fid's ``calculate_activation_statistics``."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError(f"need (N >= 2, D) features, got {features.shape}")
    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def _blas_threads(threads: int | None) -> contextlib.AbstractContextManager:
    if threads is None:
        return contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # optional, a speed-up only
        return contextlib.nullcontext()
    return threadpool_limits(limits=threads, user_api="blas")


def frechet_distance(
    feats_a: np.ndarray, feats_b: np.ndarray, *, eps: float = 1e-6, blas_threads: int | None = 8
) -> float:
    """FID between two feature sets (``fid_features``): pytorch-fid's
    ``calculate_frechet_distance`` on their ``fid_statistics``, including its handling of a
    non-finite ``sqrtm`` (``eps`` added to the diagonals) and of imaginary parts (``ValueError``
    beyond 1e-3 on the diagonal, else dropped).

    With N images per set the 2048-d covariances have rank at most N - 1, so with ~100 images
    (Urban100, as in the SR papers that report FID on it) they are rank-deficient and the value
    carries a large small-sample bias: it is comparable only between methods evaluated with
    the same protocol on the same images, never with FID values from generative modelling.
    SciPy's "singular matrix" warning, expected in that regime, is silenced.

    The 2048 x 2048 ``scipy.linalg.sqrtm`` (Schur decomposition, CPU) dominates the cost.
    OpenBLAS gains nothing beyond ~4 threads there and slows down several times when its
    threads compete with other jobs, so it runs on ``blas_threads`` threads (``None``: no
    limit; needs threadpoolctl). The thread count moves the value by ~1e-9 relative.
    """
    mu_a, sigma_a = fid_statistics(feats_a)
    mu_b, sigma_b = fid_statistics(feats_b)
    from scipy.linalg import LinAlgWarning

    with warnings.catch_warnings(), _blas_threads(blas_threads):
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        value = vendored_fid_score().calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b, eps)
    return float(value)
