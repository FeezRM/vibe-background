"""Enhance: auto white balance, exposure and contrast on the subject, independent of the
background. Operates in linear light."""

import torch

from vibebg_engine.composite import linear_to_srgb, srgb_to_linear
from vibebg_engine.imaging import luma
from vibebg_engine.lighting.smoothing import Ema, downsample

# Target for the subject's bright-but-not-highlight tones (80th percentile luma, sRGB).
# Using a high percentile keys exposure off skin and lit areas instead of dark clothing.
_TARGET_P80 = 0.62
_EXPOSURE_RANGE = (0.8, 1.8)
_WB_RANGE = (0.75, 1.33)


class Enhancer:
    def __init__(self, fps: float, strength: float):
        self.strength = strength
        self._gains = Ema(fps)

    def reset(self):
        self._gains.reset()

    @torch.inference_mode()
    def update(self, frame: torch.Tensor, fg: torch.Tensor, alpha: torch.Tensor):
        """frame, fg: 3xHxW linear. Measures the cast of the whole shot and the subject's
        exposure, and folds them into the smoothed per-channel gains."""
        small = downsample(frame)
        y = luma(small)[0]
        mid = (y > 0.02) & (y < 0.8)
        if mid.sum() < 16:
            mid = torch.ones_like(y, dtype=torch.bool)
        means = small[:, mid].mean(dim=1).clamp_min(1e-4)
        wb = (means.mean() / means) ** (0.6 * self.strength)
        wb = wb / (wb * torch.tensor([0.2126, 0.7152, 0.0722])).sum()
        wb = wb.clamp(*_WB_RANGE)

        fg_small, a_small = downsample(fg), downsample(alpha)[0]
        subject = a_small > 0.5
        exposure = torch.ones(())
        if subject.sum() >= 16:
            y_subj = linear_to_srgb(luma(fg_small * wb.view(3, 1, 1)))[0][subject]
            p80 = torch.quantile(y_subj, 0.8).clamp_min(1e-3)
            gain = srgb_to_linear(torch.tensor(_TARGET_P80)) / srgb_to_linear(p80)
            exposure = gain.clamp(*_EXPOSURE_RANGE) ** self.strength

        self._gains.update((wb * exposure).to(fg.device))

    def apply(self, fg: torch.Tensor) -> torch.Tensor:
        if self._gains.value is None:
            return fg
        out = fg * self._gains.value.view(3, 1, 1)
        # Gentle S-curve in display space for a little more punch.
        s = linear_to_srgb(out)
        curve = s * s * (3 - 2 * s)
        return srgb_to_linear(s + 0.15 * self.strength * (curve - s))


def apply_warmth(fg: torch.Tensor, warmth: float) -> torch.Tensor:
    """warmth in [-1, 1]: positive is warmer (more red, less blue)."""
    if warmth == 0:
        return fg
    gains = torch.tensor([1 + 0.12 * warmth, 1.0, 1 - 0.12 * warmth], device=fg.device)
    return fg * gains.view(3, 1, 1)
