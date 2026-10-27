"""RobustVideoMatting (Lin et al., GPL-3.0). Recurrent, so call it on frames in order."""

import torch

from vibebg_engine.models.registry import ensure_model

# Long edge the model's coarse pass runs at. RVM upsamples with a deep guided filter,
# so full-resolution detail comes back; larger values help hair at the cost of speed.
_COARSE_LONG_EDGE = {"rvm_mobilenetv3": 512, "rvm_resnet50": 768}
_WARMUP_PASSES = 8


class RVMMatter:
    def __init__(self, model_id: str, device: torch.device):
        self.device = device
        self.model = torch.jit.load(str(ensure_model(model_id)), map_location="cpu")
        self.model = self.model.eval().to(device)
        self._coarse = _COARSE_LONG_EDGE[model_id]
        self.reset()

    def reset(self):
        self._rec = [None] * 4

    @torch.inference_mode()
    def __call__(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """frame: 3xHxW sRGB in [0, 1] on device. Returns (foreground 3xHxW, alpha 1xHxW)."""
        h, w = frame.shape[-2:]
        ratio = min(self._coarse / max(h, w), 1.0)
        if self._rec[0] is None:
            # With no history the first frames are speckled; prime the recurrent state on frame 0.
            for _ in range(_WARMUP_PASSES):
                _, _, *self._rec = self.model(frame[None], *self._rec, ratio)
        fgr, pha, *self._rec = self.model(frame[None], *self._rec, ratio)
        return fgr[0].clamp(0, 1), pha[0].clamp(0, 1)
