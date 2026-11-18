"""Tests for gaussifier_sr.render: the CUDA renderer against its PyTorch reference."""

from __future__ import annotations

import math

import pytest
import torch

from gaussifier_sr.render import render, render_reference
from gaussifier_sr.render.reference import ALPHA_MIN, project, tile_rules

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEVICES = ["cpu", pytest.param("cuda", marks=requires_cuda)]


def render_cuda(*args):
    from gaussifier_sr.render._cuda import render_cuda as native

    return native(*args)


def make_gaussians(n, sigma, *, aniso=1.0, spread=0.0, channels=3, seed=0):
    """Random float32 Gaussians; ``spread`` moves centres up to that far off-image."""
    gen = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 2, generator=gen) * (1.0 + 2.0 * spread) - spread
    scales = sigma * torch.exp(0.25 * torch.randn(n, 2, generator=gen))
    scales[:, 1] *= aniso
    rotation = (torch.rand(n, 1, generator=gen) - 0.5) * 6.0
    colors = 0.5 * torch.randn(n, channels, generator=gen)
    return means, scales, rotation, colors


def naive_render(means, scales, rotation, colors, height, width):
    """Every Gaussian on every pixel, with only the alpha cutoff (no tile rules)."""
    proj = project(means, scales, rotation, height, width)
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=means.dtype),
        torch.arange(width, dtype=means.dtype),
        indexing="ij",
    )
    dx = proj.center[:, 0, None, None] - xs
    dy = proj.center[:, 1, None, None] - ys
    a, b, c = proj.conic[:, :, None, None].unbind(1)
    sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
    alpha = torch.exp(-sigma)
    keep = proj.valid[:, None, None] & (sigma >= 0) & (alpha >= ALPHA_MIN)
    return torch.einsum("nhw,nc->chw", alpha * keep, colors)


def near_cutoff(means, scales, rotation, height, width, band=1e-5):
    """Pixels where some evaluated term has alpha within ``band`` (relative) of 1/255.

    Float32 and float64 can legitimately disagree on whether such a term passes
    the cutoff, which changes the pixel by about |colour| / 255.
    """
    proj = project(means, scales, rotation, height, width)
    rules = tile_rules(proj, height, width)
    ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    dx = proj.center[:, 0, None, None] - xs
    dy = proj.center[:, 1, None, None] - ys
    a, b, c = proj.conic[:, :, None, None].unbind(1)
    sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
    x0, y0, x1, y1 = rules.rect[:, :, None, None].unbind(1)
    inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    near = inside & ((sigma - math.log(255.0)).abs() < band * math.log(255.0))
    return near.any(0)


def assert_cuda_matches_reference(params, height, width):
    """CUDA float32 render vs the float64 reference, away from cutoff ties."""
    out = render_cuda(*[p.cuda() for p in params], height, width).cpu()
    ref = render_reference(*[p.double() for p in params], height, width)
    assert out.shape == ref.shape == (params[3].shape[1], height, width)
    assert out.dtype == torch.float32
    exact = ~near_cutoff(*[p.double() for p in params[:3]], height, width)
    assert exact.float().mean() > 0.99
    scale = max(1.0, ref.abs().max().item())
    torch.testing.assert_close(out.double()[:, exact], ref[:, exact], rtol=1e-4, atol=1e-5 * scale)
    return out


# (height, width, n, sigma, aniso, spread, channels, has_large, overflows)
CASES = {
    "sigma_0.5px": (40, 56, 60, 0.5, 1.0, 0.0, 3, False, False),
    "sigma_15px": (128, 144, 40, 15.0, 1.0, 0.0, 3, True, False),
    "anisotropic_rotated": (64, 64, 50, 3.0, 5.0, 0.0, 3, False, False),
    "partly_off_image": (50, 70, 80, 4.0, 1.0, 0.3, 1, False, False),
    "ragged_size_many_per_tile": (36, 44, 600, 6.0, 1.0, 0.0, 5, False, False),
    "tile_list_overflow": (128, 128, 300, 10.0, 1.0, 0.0, 3, True, True),
    "max_channels": (33, 47, 40, 2.0, 2.0, 0.1, 12, False, False),
}


@requires_cuda
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_cuda_matches_reference(case):
    height, width, n, sigma, aniso, spread, channels, has_large, overflows = case
    params = make_gaussians(n, sigma, aniso=aniso, spread=spread, channels=channels)
    rules = tile_rules(project(*params[:3], height, width), height, width)
    assert bool(rules.large.any()) == has_large
    assert rules.overflow == overflows
    out = assert_cuda_matches_reference(params, height, width)
    assert out.abs().max() > 0.1


@pytest.mark.parametrize("device", DEVICES)
def test_zero_and_one_gaussian(device):
    empty = (torch.zeros(0, 2), torch.zeros(0, 2), torch.zeros(0, 1), torch.zeros(0, 3))
    out = render(*[t.to(device) for t in empty], 20, 30)
    assert out.shape == (3, 20, 30) and not out.any()

    one = (
        torch.tensor([[19.0 / 45.0, 22.0 / 37.0]]),  # centred on pixel (row 22, column 19)
        torch.tensor([[3.5, 1.2]]),
        torch.tensor([[0.7]]),
        torch.tensor([[0.9, -0.3, 0.5]]),
    )
    out = render(*[t.to(device) for t in one], 37, 45).cpu()
    torch.testing.assert_close(out, render_reference(*one, 37, 45), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out[:, 22, 19], one[3][0], rtol=1e-5, atol=1e-6)
    if device == "cuda":
        assert_cuda_matches_reference(one, 37, 45)


@pytest.mark.parametrize("device", DEVICES)
def test_offscreen_gaussians_render_nothing(device):
    params = (
        torch.tensor([[5.0, 5.0], [-3.0, 0.5], [0.5, 1.6]]),
        torch.full((3, 2), 2.0),
        torch.zeros(3, 1),
        torch.ones(3, 3),
    )
    assert not render(*[t.to(device) for t in params], 20, 30).any()


@pytest.mark.parametrize("device", DEVICES)
def test_pixel_and_axis_conventions(device):
    """Pixel j sits at x * W == j, scales are pixel standard deviations, and
    scales[:, 0] runs along [cos t, -sin t]."""
    height, width = 24, 40
    means = torch.tensor([[10.0 / width, 12.0 / height]], device=device)
    scales = torch.tensor([[4.0, 1.0]], device=device)
    rotation = torch.tensor([[math.pi / 2]], device=device)  # first axis along -y
    colors = torch.tensor([[2.0]], device=device)
    out = render(means, scales, rotation, colors, height, width).cpu()
    expected = 2.0 * math.exp(-0.5)
    assert out[0, 12, 10].item() == pytest.approx(2.0, rel=1e-5)
    assert out[0, 16, 10].item() == pytest.approx(expected, rel=1e-5)  # 4 px along y
    assert out[0, 8, 10].item() == pytest.approx(expected, rel=1e-5)
    assert out[0, 12, 11].item() == pytest.approx(expected, rel=1e-5)  # 1 px along x
    assert out[0, 12, 14].item() == 0.0  # 4 px along x: alpha < 1/255


def _gaussian_column(n, sigma, center, size):
    means = torch.full((n, 2), center / size)
    scales = torch.full((n, 2), sigma)
    return means, scales, torch.zeros(n, 1), torch.full((n, 1), 1.0 / n)


@pytest.mark.parametrize("device", DEVICES)
def test_small_gaussian_is_clipped_to_its_tiles(device):
    # sigma 10 px: radius ceil(3 sqrt(100 + sqrt(0.1))) = 31. Centred at 48.5 px
    # the tile rectangle is [16, 80) on both axes while alpha >= 1/255 reaches
    # 33.3 px, so columns and rows 80-81 are clipped. 16 tiles: not "large".
    params = _gaussian_column(1, 10.0, 48.5, 96)
    rules = tile_rules(project(*params[:3], 96, 96), 96, 96)
    assert rules.rect.tolist() == [[16, 16, 80, 80]]
    assert not rules.large.any() and not rules.overflow
    out = render(*[p.to(device) for p in params], 96, 96).cpu()
    naive = naive_render(*params, 96, 96)
    assert naive[0, 48, 80] > 0 and naive[0, 48, 81] > 0 and naive[0, 80, 48] > 0
    assert out[0, :, 80:].abs().max() == 0 and out[0, 80:, :].abs().max() == 0
    torch.testing.assert_close(out[:, :80, :80], naive[:, :80, :80], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_large_gaussian_covers_the_whole_image(device):
    # sigma 20 px: radius 61 touches 8 x 8 = 64 > 32 tiles, so it is "large" and
    # only the alpha cutoff (66.6 px) limits it, beyond its tile square [16, 144).
    params = _gaussian_column(1, 20.0, 80.5, 160)
    rules = tile_rules(project(*params[:3], 160, 160), 160, 160)
    assert rules.large.all() and not rules.overflow
    out = render(*[p.to(device) for p in params], 160, 160).cpu()
    assert out[0, 80, 145] > 0 and out[0, 80, 15] > 0
    torch.testing.assert_close(out, naive_render(*params, 160, 160), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [256, 257])
def test_tile_list_overflow_threshold(device, n):
    # n copies of the clipped Gaussian above, 16 tiles each: 256 * 16 = 4096
    # entries fit the tile list (capacity max(8 n, 4096)), 257 * 16 overflow it
    # and every Gaussian is then evaluated on the whole image (no clipping).
    params = _gaussian_column(n, 10.0, 48.5, 96)
    rules = tile_rules(project(*params[:3], 96, 96), 96, 96)
    assert rules.overflow == (n == 257)
    out = render(*[p.to(device) for p in params], 96, 96).cpu()
    naive = naive_render(*params, 96, 96)
    if n == 257:
        torch.testing.assert_close(out, naive, rtol=1e-4, atol=1e-6)
    else:
        assert out[0, 48, 80] == 0 and naive[0, 48, 80] > 0


@requires_cuda
def test_cuda_render_is_inference_only():
    params = list(make_gaussians(20, 3.0, seed=4))
    params[2] = params[2].reshape(-1)  # (N,) rotation is accepted too
    cuda = [p.cuda().double() for p in params]
    out = render(*cuda, 32, 40)
    assert out.dtype == torch.float32 and out.grad_fn is None
    torch.testing.assert_close(out.cpu(), render(*params, 32, 40), rtol=1e-5, atol=1e-5)

    tracked = [p.clone().requires_grad_(True) for p in cuda]
    with pytest.raises(RuntimeError, match=r"inference-only.*render_reference"):
        render(*tracked, 32, 40)
    with torch.no_grad():
        torch.testing.assert_close(render(*tracked, 32, 40), out)

    with pytest.raises(RuntimeError, match="channels"):
        render_cuda(*[p.cuda() for p in make_gaussians(3, 2.0, channels=13)], 16, 16)
    with pytest.raises(ValueError, match="rotation"):
        render(*params[:2], torch.zeros(20, 2), params[3], 32, 40)
