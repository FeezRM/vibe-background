"""Temporal smoothing of per-frame measurements, so corrections drift instead of flicker."""

import math

import torch

from vibebg_engine.imaging import pool


class Ema:
    """Exponential moving average with a time constant in seconds. reset() makes the next
    value jump straight to the measurement (used at scene cuts)."""

    def __init__(self, fps: float, tau: float = 1.0):
        self._k = 1.0 - math.exp(-1.0 / (fps * tau))
        self.value: torch.Tensor | None = None

    def reset(self):
        self.value = None

    def update(self, x: torch.Tensor) -> torch.Tensor:
        self.value = x if self.value is None else self.value + self._k * (x - self.value)
        return self.value


def downsample(img: torch.Tensor, long_edge: int = 160) -> torch.Tensor:
    """Small CPU copy of a CxHxW tensor for statistics. Masked reductions on tiny tensors are
    much faster on the CPU than as a string of GPU kernel launches."""
    return pool(img, math.ceil(max(img.shape[-2:]) / long_edge)).float().cpu()
