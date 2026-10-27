import numpy as np
import pytest
import torch

from vibebg_engine.placement import Placement
from vibebg_engine.scene import SceneCutDetector


def _box():
    a = torch.zeros(1, 100, 100)
    a[:, 50:, 40:60] = 1.0  # a "torso" touching the bottom edge
    return torch.rand(3, 100, 100), a


def test_identity_placement_is_a_no_op():
    fg, a = _box()
    out_fg, out_a = Placement().apply(fg, a)
    assert out_fg is fg and out_a is a


def test_scaling_keeps_subject_attached_to_bottom_edge():
    fg, a = _box()
    _, out = Placement(scale=0.5).apply(fg, a)
    rows = torch.nonzero(out[0] > 0.5)[:, 0]
    assert int(rows.max()) == 99  # still touches the bottom
    assert int(rows.min()) >= 74  # and is half as tall
    cols = torch.nonzero(out[0] > 0.5)[:, 1]
    assert 44 <= int(cols.min()) and int(cols.max()) <= 55  # shrunk toward the center


def test_shift_moves_subject():
    fg, a = _box()
    _, out = Placement(x=0.2).apply(fg, a)
    cols = torch.nonzero(out[0] > 0.5)[:, 1]
    assert 59 <= int(cols.min()) <= 61


def test_negative_scale_rejected():
    fg, a = _box()
    with pytest.raises(ValueError):
        Placement(scale=-1).apply(fg, a)


def test_scene_cut_detector():
    rng = np.random.default_rng(0)
    warm = np.zeros((90, 160, 3), np.uint8)
    warm[..., 0] = 200
    warm[..., 1] = 120
    cool = np.zeros_like(warm)
    cool[..., 2] = 220
    cool[..., 1] = 60
    det = SceneCutDetector()
    assert not det.is_cut(warm)
    noisy = np.clip(warm + rng.integers(-6, 6, warm.shape), 0, 255).astype(np.uint8)
    assert not det.is_cut(noisy)  # same shot, sensor noise
    assert det.is_cut(cool)  # new shot
    assert not det.is_cut(cool)
