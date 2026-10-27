import torch

from vibebg_engine.composite import composite, linear_to_srgb, srgb_to_linear


def test_srgb_roundtrip():
    x = torch.linspace(0, 1, 1001)
    assert torch.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_opaque_and_transparent_pixels_pass_through():
    fg = torch.rand(3, 4, 4)
    bg = torch.rand(3, 4, 4)
    assert torch.allclose(composite(fg, torch.ones(1, 4, 4), bg), fg, atol=1e-5)
    assert torch.allclose(composite(fg, torch.zeros(1, 4, 4), bg), bg, atol=1e-5)


def test_half_coverage_blends_in_linear_light():
    white = torch.ones(3, 1, 1)
    black = torch.zeros(3, 1, 1)
    out = composite(white, torch.full((1, 1, 1), 0.5), black)
    # 50% linear light is ~0.735 in sRGB; a naive sRGB blend would give a too-dark 0.5.
    assert torch.allclose(out, torch.full((3, 1, 1), 0.7354), atol=1e-3)
