import torch

from vibebg_engine.imaging import luma, rgb_to_lab
from vibebg_engine.lighting.enhance import Enhancer
from vibebg_engine.lighting.match import Matcher
from vibebg_engine.lighting.smoothing import Ema
from vibebg_engine.lighting.wrap_shadow import drop_shadow, light_wrap, noise_sigma

H, W = 90, 160


def _subject_mask():
    a = torch.zeros(1, H, W)
    a[:, 20:, 60:100] = 1.0
    return a


def test_ema_converges_and_resets():
    ema = Ema(fps=30, tau=1.0)
    ema.update(torch.tensor(0.0))
    for _ in range(300):
        v = ema.update(torch.tensor(1.0))
    assert abs(float(v) - 1.0) < 1e-3
    ema.reset()
    assert float(ema.update(torch.tensor(5.0))) == 5.0


def test_enhance_brightens_underexposed_subject_and_reduces_cast():
    alpha = _subject_mask()
    # A dim, blue-tinted shot.
    frame = torch.ones(3, H, W) * torch.tensor([0.05, 0.06, 0.12]).view(3, 1, 1)
    enh = Enhancer(fps=30, strength=1.0)
    enh.update(frame, frame, alpha)
    out = enh.apply(frame)
    assert luma(out).mean() > luma(frame).mean() * 1.2
    ratio_before = frame[2].mean() / frame[0].mean()
    ratio_after = out[2].mean() / out[0].mean()
    assert ratio_after < ratio_before


def test_match_moves_subject_toward_background_color_partially():
    alpha = _subject_mask()
    fg = torch.ones(3, H, W) * torch.tensor([0.3, 0.3, 0.3]).view(3, 1, 1)
    bg = torch.ones(3, H, W) * torch.tensor([0.5, 0.25, 0.05]).view(3, 1, 1)  # warm room
    m = Matcher(fps=30, strength=1.0)
    m.update(fg, alpha, bg)
    out = m.apply(fg)
    before, after, room = (rgb_to_lab(x).mean(dim=(1, 2)) for x in (fg, out, bg))
    # b (yellow-blue) moves toward the room's, but stops short of it.
    assert before[2] < after[2] < room[2]
    assert after[2] - before[2] <= 10.0 + 1e-3


def test_match_with_zero_strength_is_identity():
    alpha = _subject_mask()
    fg = torch.rand(3, H, W) * 0.5 + 0.1
    m = Matcher(fps=30, strength=0.0)
    m.update(fg, alpha, torch.rand(3, H, W))
    assert torch.allclose(m.apply(fg), fg, atol=1e-3)


def test_drop_shadow_darkens_only_near_subject():
    alpha = _subject_mask()
    bg = torch.ones(3, H, W)
    out = drop_shadow(bg, alpha, amount=1.0)
    assert out[:, 60, 102].mean() < 0.9  # just right of the subject
    assert torch.allclose(out[:, 5, 5], torch.ones(3), atol=1e-3)  # far corner untouched


def test_light_wrap_only_touches_subject_edges():
    alpha = _subject_mask()
    fg = torch.zeros(3, H, W)
    bg = torch.ones(3, H, W)
    out = light_wrap(fg, alpha, bg, amount=1.0)
    assert out[:, 60, 61].mean() > 0.05  # inner edge picks up background light
    assert out[:, 60, 80].mean() < 1e-3  # center of the subject does not


def test_noise_estimate_matches_known_sigma():
    torch.manual_seed(0)
    img = torch.full((3, 256, 256), 0.5) + torch.randn(1, 256, 256) * 0.03
    est = noise_sigma(img, torch.ones(1, 256, 256))
    assert abs(float(est) - 0.03) < 0.004
