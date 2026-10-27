"""Match: shift the subject's color toward the background's (Reinhard-style transfer in Lab),
so they look lit by the same room. Deliberately partial: a full transfer would make a person
as dark or as purple as the scene behind them."""

import torch

from vibebg_engine.imaging import lab_to_rgb, rgb_to_lab
from vibebg_engine.lighting.smoothing import Ema, downsample

_L_SHIFT, _AB_SHIFT = 0.35, 0.5  # fraction of the mean difference applied at strength 1
_L_SHIFT_MAX, _AB_SHIFT_MAX = 15.0, 10.0  # Lab units
_CONTRAST_RANGE = (0.85, 1.15)


def _lab_stats(lab: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    px = lab[:, mask]
    return torch.stack([px.mean(dim=1), px.std(dim=1)])  # 2 x 3: means, stds


class Matcher:
    def __init__(self, fps: float, strength: float):
        self.strength = strength
        self._subject = Ema(fps)
        self._background = Ema(fps)

    def reset(self):
        self._subject.reset()
        self._background.reset()

    @torch.inference_mode()
    def update(self, fg: torch.Tensor, alpha: torch.Tensor, bg: torch.Tensor):
        """fg, bg: 3xHxW linear."""
        a = downsample(alpha)[0]
        subject = a > 0.5
        if subject.sum() < 16:
            return
        self._subject.update(_lab_stats(rgb_to_lab(downsample(fg).clamp_min(0)), subject)
                             .to(fg.device))
        bg_lab = rgb_to_lab(downsample(bg).clamp_min(0))
        self._background.update(_lab_stats(bg_lab, torch.ones_like(a, dtype=torch.bool))
                                .to(fg.device))

    def apply(self, fg: torch.Tensor) -> torch.Tensor:
        if self._subject.value is None or self._background.value is None:
            return fg
        (mu_s, sd_s), (mu_b, sd_b) = self._subject.value, self._background.value
        st = self.strength
        diff = mu_b - mu_s
        shift = torch.stack([
            (_L_SHIFT * st * diff[0]).clamp(-_L_SHIFT_MAX, _L_SHIFT_MAX),
            (_AB_SHIFT * st * diff[1]).clamp(-_AB_SHIFT_MAX, _AB_SHIFT_MAX),
            (_AB_SHIFT * st * diff[2]).clamp(-_AB_SHIFT_MAX, _AB_SHIFT_MAX),
        ])
        ratio = torch.ones(3, device=fg.device)
        ratio[0] = ((sd_b[0] / sd_s[0].clamp_min(1e-3)) ** (0.3 * st)).clamp(*_CONTRAST_RANGE)

        lab = rgb_to_lab(fg.clamp_min(0))
        mu = mu_s.view(3, 1, 1)
        lab = (lab - mu) * ratio.view(3, 1, 1) + mu + shift.view(3, 1, 1)
        return lab_to_rgb(lab).clamp_min(0)
