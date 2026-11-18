"""Fused CUDA runtime for super-resolution (cpp_inference/fused, gaussifier_sr.fused).

Skips without a CUDA device or a built ``libgaussifier_fused`` (``GAUSSIFIER_SR_FUSED_LIBRARY``
or ``cpp_inference/build``). Tests that need weights skip without them:

- the base release ``fm_fp16.gffm`` (and ``head_fp16.gfch`` / ``head.pt`` for the
  image-residual check) in ``$GAUSSIFIER_BASE_WEIGHTS`` or the cache of
  ``tools/export_fused.py``; nothing is downloaded here;
- an SR export (``tools/export_fused.py``) in ``$GAUSSIFIER_SR_FUSED_EXPORT``;
- for the PyTorch comparison, also the SR checkpoint in ``$GAUSSIFIER_SR_CHECKPOINT``.
"""

from __future__ import annotations

import ctypes as ct
import json
import math
import os
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
EXPORT = pytest.importorskip("gaussifier_sr.fused")  # blob writers (tools/export_fused.py)


@pytest.fixture(scope="module")
def fused():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    try:
        EXPORT.load_library()
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        pytest.skip(f"fused library unavailable: {exc}")
    return EXPORT


def _ptr(t):
    return ct.c_void_p(t.data_ptr())


def _stream():
    return ct.c_void_p(torch.cuda.current_stream().cuda_stream)


def _base(name: str) -> Path:
    path = EXPORT.base_dir() / name
    if not path.is_file():
        pytest.skip(f"base release {name} not found (set GAUSSIFIER_BASE_WEIGHTS)")
    return EXPORT.base_file(name)  # checks the SHA-256


# ----------------------------------------------------------------------------- blob format


def _synthetic_head(seed: int = 0) -> dict:
    """A random frozen-GN head (main U-Net 32/64/128, aux 24/48/96, 17 inputs, 8 outputs)."""
    g = torch.Generator().manual_seed(seed)
    state = {}
    for net, c in (("net", 32), ("aux_net", 24)):
        widths = {
            "enc_blocks.0": (17, c),
            "enc_blocks.1": (c, 2 * c),
            "bottleneck": (2 * c, 4 * c),
            "merge_blocks.0": (6 * c, 2 * c),
            "merge_blocks.1": (3 * c, c),
        }
        for block, (cin, cout) in widths.items():
            for idx, n_in in ((0, cin), (3, cout)):
                p = f"{net}.{block}"
                state[f"{p}.{idx}.weight"] = torch.randn(cout, n_in, 3, 3, generator=g) / math.sqrt(
                    9 * n_in
                )
                state[f"{p}.{idx}.bias"] = 0.1 * torch.randn(cout, generator=g)
                state[f"{p}.{idx + 1}.weight"] = 1 + 0.1 * torch.randn(cout, generator=g)
                state[f"{p}.{idx + 1}.bias"] = 0.1 * torch.randn(cout, generator=g)
                state[f"{p}.{idx + 1}.mean"] = 0.1 * torch.randn(cout, generator=g)
                state[f"{p}.{idx + 1}.var"] = 0.5 + torch.rand(cout, generator=g)
        state[f"{net}.out.weight"] = 0.1 * torch.randn(8, c, 1, 1, generator=g)
        state[f"{net}.out.bias"] = 0.1 * torch.randn(8, generator=g)
    return state


def test_head_blob_records_residual_mode(tmp_path):
    state = _synthetic_head()
    for residual, version, cin in (("backprojection", 3, 17), ("image", 2, 14)):
        path = tmp_path / f"{residual}.gfch"
        EXPORT.write_head(state, residual, path)
        data = path.read_bytes()
        magic, ver, n = struct.unpack_from("<III", data)
        assert (magic, ver, n) == (EXPORT.GFCH_MAGIC, version, 22)
        offset = 12
        if version == 3:
            assert struct.unpack_from("<I", data, 12)[0] == EXPORT.RESIDUAL_MODES[residual]
            offset = 16
        ks, first_cin, cout, precision = struct.unpack_from("<IIII", data, offset)
        assert (ks, first_cin, cout, precision) == (3, cin, 32, 0)


def test_image_fold_is_exact():
    """diff = image - rendered folded into the image / rendered weights."""
    w = torch.randn(4, 17, 3, 3, dtype=torch.float64)
    x = torch.randn(1, 17, 8, 8, dtype=torch.float64)
    x[:, 13:16] = x[:, 0:3] - x[:, 10:13]
    folded = EXPORT.fold_diff(w)
    ref = F.conv2d(x, w, padding=1)
    out = F.conv2d(x[:, EXPORT.FOLDED_KEEP], folded, padding=1)
    torch.testing.assert_close(out, ref)


# ----------------------------------------------------------------------------- residual


def _reference_residual(render, image=None, lr=None, lr_size=None):
    size = lr.shape[-2:] if lr is not None else lr_size
    down = F.interpolate(
        render[None], size=size, mode="bicubic", align_corners=False, antialias=True
    )
    if lr is not None:
        return F.interpolate(
            lr[None] - down, size=render.shape[-2:], mode="bicubic", align_corners=False
        )[0]
    return (
        image - F.interpolate(down, size=render.shape[-2:], mode="bicubic", align_corners=False)[0]
    )


@pytest.mark.parametrize(
    "size",
    [
        (64, 64, 32, 32),
        (74, 106, 37, 53),
        (34, 18, 17, 9),
        (60, 90, 20, 30),
        (40, 52, 10, 13),
        (512, 512, 256, 256),
    ],
)
def test_backprojection_residual_matches_torch(fused, size):
    """up(LR - down(render)) and image - up(down(render)) as PyTorch computes them."""
    H, W, h, w = size
    lib = fused.load_library()
    g = torch.Generator(device="cuda").manual_seed(H * W)
    render = torch.rand(3, H, W, device="cuda", generator=g)
    lr = torch.rand(3, h, w, device="cuda", generator=g)
    image = torch.rand(3, H, W, device="cuda", generator=g)
    out = torch.empty(3, H, W, device="cuda")
    assert (
        lib.gfused_backprojection_residual(
            _ptr(render), None, _ptr(lr), H, W, h, w, _stream(), _ptr(out)
        )
        == 0
    )
    torch.testing.assert_close(out, _reference_residual(render, lr=lr), rtol=0, atol=1e-6)
    assert (
        lib.gfused_backprojection_residual(
            _ptr(render), _ptr(image), None, H, W, h, w, _stream(), _ptr(out)
        )
        == 0
    )
    ref = _reference_residual(render, image=image, lr_size=(h, w))
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-6)


# ----------------------------------------------------------------------------- head


def _fold(sd, conv, gn):
    s = sd[f"{gn}.weight"] / torch.sqrt(sd[f"{gn}.var"] + 1e-5)
    return sd[f"{conv}.weight"] * s[:, None, None, None], (
        (sd[f"{conv}.bias"] - sd[f"{gn}.mean"]) * s + sd[f"{gn}.bias"]
    )


def _unet(sd, net, x):
    """The frozen-GN correction U-Net (depth 2) in float64."""

    def double_conv(x, block):
        for i in (0, 3):
            w, b = _fold(sd, f"{net}.{block}.{i}", f"{net}.{block}.{i + 1}")
            x = F.silu(F.conv2d(x, w, b, padding=1))
        return x

    s0 = double_conv(x, "enc_blocks.0")
    s1 = double_conv(F.max_pool2d(s0, 2), "enc_blocks.1")
    x = double_conv(F.max_pool2d(s1, 2), "bottleneck")
    for block, skip in (("merge_blocks.0", s1), ("merge_blocks.1", s0)):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = double_conv(torch.cat([x, skip], 1), block)
    return F.conv2d(x, sd[f"{net}.out.weight"], sd[f"{net}.out.bias"])


def _head_input(H, W, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    image = torch.rand(3, H, W, device="cuda", generator=g)
    density = 2 * torch.rand(1, H, W, device="cuda", generator=g)
    log_cov = torch.randn(3, H, W, device="cuda", generator=g)
    log_cov[0] -= 6
    log_cov[2] -= 6
    rgb = torch.rand(3, H, W, device="cuda", generator=g)
    rendered = (image + 0.05 * torch.randn(3, H, W, device="cuda", generator=g)).clamp(0, 1)
    residual = 0.05 * torch.randn(3, H, W, device="cuda", generator=g)
    count = (torch.rand(1, H, W, device="cuda", generator=g) < 0.3).float()
    return torch.cat([image, density, log_cov, rgb, rendered, residual, count]).contiguous()


def _session(fused, fm: Path, head: Path):
    lib = fused.load_library()
    session = ct.c_void_p()
    assert (
        lib.gfused_create(
            os.fsencode(fm), os.fsencode(head), torch.cuda.current_device(), ct.byref(session)
        )
        == 0
    ), lib.gfused_error().decode()
    return lib, session


def _check_head(lib, session, state, residual, sizes):
    sd = {k: v.double().cuda() for k, v in state.items()}
    for H, W in sizes:
        x = _head_input(H, W, seed=H + W)
        if residual == "image":
            x[13:16] = x[0:3] - x[10:13]
        ref = (_unet(sd, "net", x[None].double()) + _unet(sd, "aux_net", x[None].double()))[0]
        for schedule in (0, 1):  # small-image kernels, fused enc0 kernel
            for packed in (1, 0):  # packed head input (pipeline), FP32 gather
                out = torch.empty(8, H, W, device="cuda")
                assert (
                    lib.gfused_head_device(
                        session, _ptr(x), H, W, schedule, packed, _stream(), _ptr(out)
                    )
                    == 0
                ), lib.gfused_error()
                err = (out.double() - ref).abs()
                assert err.max() < 0.02 * ref.abs().max(), (H, W, schedule, packed)
                assert err.mean() < 2e-3 * ref.abs().mean() + 1e-4, (H, W, schedule, packed)


def test_backprojection_head_matches_torch(fused, tmp_path):
    """The 17-channel first conv (residual as a third input unit) in both head schedules."""
    state = _synthetic_head(seed=1)
    head = tmp_path / "head.gfch"
    EXPORT.write_head(state, "backprojection", head)
    lib, session = _session(fused, _base("fm_fp16.gffm"), head)
    try:
        assert lib.gfused_residual_mode(session) == 1
        _check_head(
            lib, session, state, "backprojection", [(64, 64), (96, 128), (66, 70), (50, 34)]
        )
        # the residual channels matter: zeroing them changes the reference
        sd = {k: v.double().cuda() for k, v in state.items()}
        x = _head_input(64, 64)
        x0 = x.clone()
        x0[13:16] = 0
        delta = _unet(sd, "net", x[None].double()) - _unet(sd, "net", x0[None].double())
        assert delta.abs().max() > 1e-3
    finally:
        lib.gfused_destroy(session)


def test_image_residual_head_unchanged(fused):
    """The base release head (image residual, folded GFCH v2) still runs as before."""
    release = torch.load(_base("head.pt"), map_location="cpu", weights_only=True)
    state = EXPORT.head_state(release)
    lib, session = _session(fused, _base("fm_fp16.gffm"), _base("head_fp16.gfch"))
    try:
        assert lib.gfused_residual_mode(session) == 0
        _check_head(lib, session, state, "image", [(64, 96), (66, 70)])
    finally:
        lib.gfused_destroy(session)


# ----------------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def sr_model(fused):
    export = os.environ.get("GAUSSIFIER_SR_FUSED_EXPORT")
    if not export or not (Path(export) / "export.json").is_file():
        pytest.skip("set GAUSSIFIER_SR_FUSED_EXPORT to a tools/export_fused.py output")
    with fused.FusedSR.from_export(export) as model:
        yield model


def _test_image(h, w, seed=0):
    """A smooth random image with edges (bicubic upsampling of noise plus a step)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    small = torch.rand(1, 3, max(2, h // 8), max(2, w // 8), device="cuda", generator=g)
    img = F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False)[0]
    img[:, :, : w // 3] += 0.3
    return img.clamp(0, 1).contiguous()


def test_sr_x2_and_cascade(sr_model):
    lr = _test_image(48, 64)
    out = sr_model.sr_x2(lr)
    stats = sr_model.stats()
    assert out.shape == (3, 96, 128) and out.is_cuda
    assert torch.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
    # s2 count rule: round(4 * own(LR) * count_scale), capped at 0.4 points per pixel
    want = round(4 * stats["lr_rate_sum"] * sr_model.count_scale)
    assert stats["steps"][0]["count"] == max(1, min(want, int(0.4 * 96 * 128)))
    again = sr_model.sr_x2(lr)
    assert (again - out).abs().mean() < 2e-3  # atomics: not bit-deterministic
    out4 = sr_model.sr(lr, 4)
    stats4 = sr_model.stats()
    assert out4.shape == (3, 192, 256)
    assert torch.isfinite(out4).all()
    for k, step in enumerate(stats4["steps"]):
        want = round(4 ** (k + 1) * stats4["lr_rate_sum"] * sr_model.count_scale)
        cap = int(0.4 * step["size"][0] * step["size"][1])
        assert step["count"] == max(1, min(want, cap))
    pinned = sr_model.sr_x2(lr, count=1234)
    assert sr_model.stats()["steps"][0]["count"] == 1234
    assert pinned.shape == out.shape


def test_sr_inputs_are_validated(sr_model):
    with pytest.raises(ValueError):
        sr_model.sr_x2(torch.rand(3, 8, 8, device="cuda"))
    with pytest.raises(ValueError):
        sr_model.sr(torch.rand(3, 32, 32, device="cuda"), 3)


def test_sr_matches_pytorch_loop(sr_model):
    """The fused x2 output against ``gaussifier_sr.loop`` with the same checkpoint."""
    checkpoint = os.environ.get("GAUSSIFIER_SR_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set GAUSSIFIER_SR_CHECKPOINT for the PyTorch comparison")
    try:
        from gaussifier_sr.loop import upscale
        from gaussifier_sr.model import load_model
    except ImportError as exc:
        pytest.skip(f"PyTorch loop unavailable: {exc}")
    meta = json.loads((Path(os.environ["GAUSSIFIER_SR_FUSED_EXPORT"]) / "export.json").read_text())
    if meta["residual"] != "backprojection":
        pytest.skip("the PyTorch loop uses the back-projection residual")
    model = load_model("cuda", sr_head=checkpoint)
    hr = _test_image(256, 256, seed=3)
    lr = F.interpolate(hr[None], scale_factor=0.5, mode="bicubic", antialias=True)[0].clamp(0, 1)
    with torch.no_grad():
        ref = upscale(model, lr, 2).clamp(0, 1)
    out = sr_model.sr_x2(lr)

    def psnr(x):
        return float(-10 * torch.log10(((x - hr) ** 2).mean()))

    assert abs(psnr(out) - psnr(ref)) < 0.1
    assert (out - ref).abs().mean() < 5e-3
