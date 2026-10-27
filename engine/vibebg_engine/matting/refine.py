"""Max tier: RobustVideoMatting's matte, with hair and edges re-solved by ViTMatte (MIT code,
Apache-2.0 weights) inside a narrow band where the coarse matte is unsure.

RVM stays the base because it is temporally stable and keeps objects held in front of the
person (a microphone, a mug); ViTMatte only redraws the soft boundary.
"""

import torch
import torch.nn.functional as F

from vibebg_engine.matting.rvm import RVMMatter
from vibebg_engine.models.registry import ensure_hf_model

# Fraction of the frame's long edge: how far the unsure band reaches past the coarse edge.
_BAND = 0.008
# Long edge the refinement crop is resized to; ViTMatte cost grows with the square of it.
_REFINE_LONG_EDGE = 1024
_PAD = 16
# How much of the previous frame's refined alpha to keep where the image hasn't changed.
_TEMPORAL_KEEP = 0.8
_MOTION_SCALE = 0.04  # per-pixel color change at which smoothing fades out


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary erosion of a 1xHxW {0,1} float mask with a square of the given radius."""
    k = 2 * radius + 1
    return 1.0 - F.max_pool2d((1.0 - mask)[None], k, stride=1, padding=radius)[0]


def trimap(alpha: torch.Tensor, band: int) -> torch.Tensor:
    """1xHxW trimap from a soft alpha: 1 = surely subject, 0 = surely background, 0.5 = unsure.
    Anything not confidently opaque or transparent, widened by `band` pixels, is unsure."""
    sure_fg = _erode((alpha > 0.95).float(), band)
    sure_bg = _erode((alpha < 0.05).float(), band)
    return torch.where(sure_fg > 0, 1.0, torch.where(sure_bg > 0, 0.0, 0.5))


class EdgeRefiner:
    def __init__(self, device: torch.device):
        from transformers import VitMatteForImageMatting

        self.device = device
        self.dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
        path = ensure_hf_model("vitmatte_small_d646")
        self.model = VitMatteForImageMatting.from_pretrained(path).to(device, dtype=self.dtype)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, frame: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """frame: 3xHxW sRGB [0, 1]; alpha: 1xHxW coarse matte. Returns the refined 1xHxW matte."""
        h, w = alpha.shape[-2:]
        tri = trimap(alpha, max(2, round(_BAND * max(h, w))))
        unsure = tri[0] == 0.5
        if not bool(unsure.any()):
            return alpha

        rows = torch.nonzero(unsure.any(dim=1)).flatten()
        cols = torch.nonzero(unsure.any(dim=0)).flatten()
        y0, y1 = max(0, int(rows[0]) - _PAD), min(h, int(rows[-1]) + 1 + _PAD)
        x0, x1 = max(0, int(cols[0]) - _PAD), min(w, int(cols[-1]) + 1 + _PAD)
        img, tri_crop = frame[:, y0:y1, x0:x1], tri[:, y0:y1, x0:x1]

        ch, cw = y1 - y0, x1 - x0
        scale = min(1.0, _REFINE_LONG_EDGE / max(ch, cw))
        sh, sw = max(32, round(ch * scale)), max(32, round(cw * scale))
        img_s = F.interpolate(img[None], size=(sh, sw), mode="bilinear", align_corners=False)
        tri_s = F.interpolate(tri_crop[None], size=(sh, sw), mode="nearest")
        x = torch.cat([(img_s - 0.5) / 0.5, tri_s], dim=1)
        x = F.pad(x, (0, (-sw) % 32, 0, (-sh) % 32))

        out = self.model(pixel_values=x.to(self.dtype)).alphas[:, :, :sh, :sw].float()
        out = F.interpolate(out, size=(ch, cw), mode="bilinear", align_corners=False)[0]

        refined = tri.clone()
        refined[:, y0:y1, x0:x1] = torch.where(tri_crop == 0.5, out.clamp(0, 1), tri_crop)
        return refined


class MaxMatter:
    """Same interface as RVMMatter: call on frames in order, reset() at scene cuts."""

    def __init__(self, device: torch.device):
        self.rvm = RVMMatter("rvm_resnet50", device)
        self.refiner = EdgeRefiner(device)
        self.reset()

    def reset(self):
        self.rvm.reset()
        self._prev_frame: torch.Tensor | None = None
        self._prev_alpha: torch.Tensor | None = None

    @torch.inference_mode()
    def __call__(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fg, coarse = self.rvm(frame)
        alpha = self.refiner(frame, coarse)
        if self._prev_alpha is not None:
            # Per-frame refinement jitters on fine hair; where the picture is still, lean on the
            # previous frame's answer. Moving areas take the new one.
            change = (frame - self._prev_frame).abs().mean(dim=0, keepdim=True)
            keep = _TEMPORAL_KEEP * torch.exp(-change / _MOTION_SCALE)
            alpha = alpha * (1 - keep) + self._prev_alpha * keep
        self._prev_frame, self._prev_alpha = frame, alpha
        return fg, alpha
