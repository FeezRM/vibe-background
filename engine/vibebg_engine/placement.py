"""Moving and scaling the cut-out subject within the frame."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Placement:
    """Scale is about the bottom-center of the frame, so a torso cut off by the bottom of the
    original shot stays attached to the bottom edge. x and y shift the subject by a fraction of
    the frame width and height (positive = right and down)."""

    scale: float = 1.0
    x: float = 0.0
    y: float = 0.0

    @property
    def is_identity(self) -> bool:
        return self.scale == 1.0 and self.x == 0.0 and self.y == 0.0

    def apply(self, fg: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_identity:
            return fg, alpha
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        h, w = alpha.shape[-2:]
        # Normalized coordinates span [-1, 1]; the bottom-center anchor is (0, 1).
        s, ax, ay = self.scale, 0.0, 1.0
        dx, dy = 2 * self.x, 2 * self.y
        theta = torch.tensor([[1 / s, 0, ax - (ax + dx) / s],
                              [0, 1 / s, ay - (ay + dy) / s]], device=alpha.device)
        grid = F.affine_grid(theta[None], [1, 4, h, w], align_corners=False)
        both = torch.cat([fg, alpha])[None]
        out = F.grid_sample(both, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)[0]
        return out[:3], out[3:]
