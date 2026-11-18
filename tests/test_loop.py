"""SR loop checks (CUDA; the weights are downloaded on first use)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def model():
    from gaussifier_sr.model import load_model

    return load_model("cuda")


def _lr(height: int = 40, width: int = 56, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(3, height // 8, width // 8, generator=g)
    lr = torch.nn.functional.interpolate(base[None], size=(height, width), mode="bilinear")[0]
    return (lr + 0.05 * torch.rand(3, height, width, generator=g)).clamp(0, 1).cuda()


def test_folded_model_matches_unfolded() -> None:
    from gaussifier_sr.model import load_model

    plain = load_model("cuda", fold=False, bf16=False)
    folded = load_model("cuda", fold=True, bf16=False)
    x = _lr(64, 64)[None]
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False  # TF32 rounds the two weight layouts differently
    try:
        with torch.no_grad():
            a, b = plain.forward_map(x), folded.forward_map(x)
            for key in ("rate", "decoder_log_covariance", "decoder_rgb"):
                torch.testing.assert_close(a[key], b[key], rtol=1e-4, atol=1e-5)
            head_in = torch.rand(1, 17, 64, 64, device="cuda")
            torch.testing.assert_close(
                plain.correction_head(head_in),
                folded.correction_head(head_in),
                rtol=1e-4,
                atol=1e-5,
            )
    finally:
        torch.backends.cudnn.allow_tf32 = tf32


def test_x2_shape_and_range(model) -> None:
    from gaussifier_sr.loop import upscale

    out = upscale(model, _lr(), 2)
    assert out.shape == (3, 80, 112)
    assert torch.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_x4_cascade_shape(model) -> None:
    from gaussifier_sr.loop import upscale

    assert upscale(model, _lr(24, 32), 4).shape == (3, 96, 128)
