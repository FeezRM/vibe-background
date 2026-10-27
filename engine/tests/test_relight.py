import torch

from vibebg_engine.lighting.relight import (
    LightMap,
    _crop,
    apply_light_map,
    blend_maps,
    paste_map,
    subject_box,
)


def _alpha(x0, y0, x1, y1, h=100, w=200):
    a = torch.zeros(1, h, w)
    a[:, y0:y1, x0:x1] = 1.0
    return a


def test_subject_box_is_padded_fraction_of_frame():
    box = subject_box(_alpha(50, 20, 150, 100), pad=0.0)
    assert box == (0.25, 0.2, 0.75, 1.0)
    assert subject_box(torch.zeros(1, 10, 10)) is None


def test_crop_samples_the_box_region():
    img = torch.zeros(3, 100, 200)
    img[:, :, 100:] = 1.0  # right half white
    left = _crop(img, (0.0, 0.0, 0.5, 1.0), (32, 32))
    right = _crop(img, (0.5, 0.0, 1.0, 1.0), (32, 32))
    assert left.mean() < 0.05 and right.mean() > 0.95


def test_paste_map_covers_only_the_box():
    ratio = torch.full((3, 8, 8), 2.0)
    gain = paste_map(LightMap(ratio, (0.25, 0.2, 0.75, 1.0)), (0.25, 0.2, 0.75, 1.0), 100, 200)
    assert torch.allclose(gain[:, 60, 100], torch.full((3,), 2.0), atol=1e-4)  # inside
    assert torch.allclose(gain[:, 10, 10], torch.ones(3), atol=1e-4)  # outside


def test_light_map_follows_the_subject():
    # A map that brightens the left half of the subject. When the subject moves right, the
    # brightened area moves with it.
    ratio = torch.ones(3, 8, 8)
    ratio[:, :, :4] = 2.0
    light = LightMap(ratio, (0.0, 0.0, 0.5, 1.0))
    fg = torch.full((3, 100, 200), 0.5)
    out = apply_light_map(fg, _alpha(100, 0, 200, 100), light, strength=1.0)
    assert out[0, 50, 120] > 0.9  # left part of the moved subject is brightened
    assert abs(float(out[0, 50, 180]) - 0.5) < 1e-3
    assert abs(float(out[0, 50, 40]) - 0.5) < 1e-3  # where the subject used to be: untouched


def test_blend_maps_interpolates_ratio_and_box():
    a = LightMap(torch.ones(3, 4, 4), (0.0, 0.0, 0.5, 0.5))
    b = LightMap(torch.full((3, 4, 4), 3.0), (0.5, 0.5, 1.0, 1.0))
    mid = blend_maps(a, b, 0.5)
    assert torch.allclose(mid.ratio, torch.full((3, 4, 4), 2.0))
    assert mid.box == (0.25, 0.25, 0.75, 0.75)
