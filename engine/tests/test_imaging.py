import torch

from vibebg_engine.imaging import (
    gaussian_blur,
    lab_to_rgb,
    normalized_blur,
    pool,
    rgb_to_lab,
    variable_blur,
)


def test_gaussian_blur_preserves_flat_images_and_shape():
    img = torch.full((3, 37, 53), 0.4)
    for sigma in (0.5, 2.0, 9.0, 30.0):  # covers both the direct and the downsampled path
        out = gaussian_blur(img, sigma)
        assert out.shape == img.shape
        assert torch.allclose(out, img, atol=1e-5)


def test_large_blur_stays_centered():
    # A bright dot must stay where it is when the blur takes the reduced-resolution path.
    img = torch.zeros(1, 101, 101)
    img[0, 50, 50] = 1.0
    out = gaussian_blur(img, 12.0)[0]
    peak = torch.nonzero(out == out.max())[0]
    assert abs(int(peak[0]) - 50) <= 1 and abs(int(peak[1]) - 50) <= 1


def test_pool_handles_non_divisible_sizes():
    assert pool(torch.ones(3, 10, 7), 4).shape == (3, 3, 2)


def test_normalized_blur_does_not_leak_masked_region():
    # A red "person" in the middle of a blue room. Blurring the room around the person must
    # not tint the room red.
    img = torch.zeros(3, 64, 64)
    img[2] = 1.0
    person = torch.zeros(1, 64, 64)
    person[:, 16:48, 24:40] = 1.0
    img[:, person[0] > 0] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1)
    out = normalized_blur(img, 1.0 - person, sigma=6.0)
    room = (person[0] == 0)
    assert out[0][room].max() < 1e-3
    assert torch.allclose(out[2][room], torch.ones_like(out[2][room]), atol=1e-3)


def test_variable_blur_endpoints():
    img = torch.rand(3, 40, 40)
    assert torch.allclose(variable_blur(img, torch.zeros(1, 40, 40), 8.0), img, atol=1e-6)
    assert torch.allclose(variable_blur(img, torch.ones(1, 40, 40), 8.0), gaussian_blur(img, 8.0),
                          atol=1e-5)


def test_lab_roundtrip_and_white_point():
    lin = torch.rand(3, 8, 8)
    assert torch.allclose(lab_to_rgb(rgb_to_lab(lin)), lin, atol=1e-4)
    white = rgb_to_lab(torch.ones(3, 1, 1))
    assert torch.allclose(white.flatten(), torch.tensor([100.0, 0.0, 0.0]), atol=0.05)
