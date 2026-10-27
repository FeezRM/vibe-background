"""Effects that tie the subject into the scene: light wrap, drop shadow and grain matching.
Images are 3xHxW linear, alpha is 1xHxW."""

import math

import torch
import torch.nn.functional as F

from vibebg_engine.imaging import gaussian_blur, luma, normalized_blur
from vibebg_engine.lighting.smoothing import Ema


def light_wrap(fg: torch.Tensor, alpha: torch.Tensor, bg: torch.Tensor, amount: float
               ) -> torch.Tensor:
    """Let background light spill over the subject's edges, as it would in a real shot."""
    if amount <= 0:
        return fg
    sigma = 0.012 * max(alpha.shape[-2:])
    inner_edge = alpha * (1.0 - gaussian_blur(alpha, sigma))
    spill = normalized_blur(bg, 1.0 - alpha, sigma)
    w = (inner_edge * 1.5 * amount).clamp(0, 0.8)
    return fg * (1 - w) + spill * w


def drop_shadow(bg: torch.Tensor, alpha: torch.Tensor, amount: float,
                offset: tuple[float, float] = (0.015, 0.01)) -> torch.Tensor:
    """Darken the background behind and beside the subject. offset is a fraction of the frame
    (positive = right and down, i.e. light from the upper left)."""
    if amount <= 0:
        return bg
    h, w = alpha.shape[-2:]
    dx, dy = round(offset[0] * w), round(offset[1] * h)
    shifted = torch.roll(alpha, shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        shifted[..., :dy, :] = 0
    if dx > 0:
        shifted[..., :, :dx] = 0
    shadow = gaussian_blur(shifted, 0.025 * max(h, w)) * (0.6 * amount)
    return bg * (1 - shadow)


_LAPLACE = torch.tensor([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])


def noise_sigma(img: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Immerkær's fast noise estimate (std of additive Gaussian noise), averaged with weights.
    img: 3xHxW in display space, weight: 1xHxW in [0, 1]."""
    y = luma(img)
    resp = F.conv2d(y[None], _LAPLACE.to(img.device).view(1, 1, 3, 3))[0].abs()
    w = weight[..., 1:-1, 1:-1]
    total = w.sum()
    return (resp * w).sum() / total.clamp_min(1.0) * math.sqrt(math.pi / 2) / 6 * (total >= 64)


_MAX_GRAIN = 0.02


class GrainMatcher:
    """Adds film-like grain to a clean background so it matches the camera noise on the
    subject. Works in display (sRGB) space, where camera noise lives."""

    def __init__(self, fps: float):
        self._sigma = Ema(fps, tau=2.0)

    def reset(self):
        self._sigma.reset()

    @torch.inference_mode()
    def apply(self, fg_srgb: torch.Tensor, alpha: torch.Tensor, bg_srgb: torch.Tensor
              ) -> torch.Tensor:
        s_fg = noise_sigma(fg_srgb, (alpha > 0.95).float())
        s_bg = noise_sigma(bg_srgb, (alpha < 0.05).float())
        # Texture on the subject reads as noise too, so cap what gets added.
        add = self._sigma.update(torch.sqrt((s_fg**2 - s_bg**2).clamp_min(0)).clamp(max=_MAX_GRAIN))
        grain = torch.randn(1, *bg_srgb.shape[-2:], device=bg_srgb.device)
        return (bg_srgb + grain * add).clamp(0, 1)
