import torch

from vibebg_engine.matting.refine import trimap


def test_trimap_marks_a_band_around_the_edge():
    alpha = torch.zeros(1, 40, 40)
    alpha[:, 10:30, 10:30] = 1.0
    tri = trimap(alpha, band=3)
    assert set(tri.unique().tolist()) == {0.0, 0.5, 1.0}
    assert tri[0, 20, 20] == 1.0  # deep inside the subject
    assert tri[0, 2, 2] == 0.0  # far background
    # Both sides of the boundary are unsure, out to the band width.
    assert tri[0, 20, 10] == 0.5 and tri[0, 20, 8] == 0.5 and tri[0, 20, 12] == 0.5
    assert tri[0, 20, 5] == 0.0 and tri[0, 20, 15] == 1.0


def test_trimap_treats_soft_alpha_as_unsure():
    alpha = torch.full((1, 30, 30), 0.5)  # e.g. motion blur or fine hair everywhere
    assert bool((trimap(alpha, band=2) == 0.5).all())
