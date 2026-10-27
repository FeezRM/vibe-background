"""Monocular depth for depth-of-field blur, via Depth Anything V2 Small (Apache-2.0)."""

import numpy as np
import torch
import torch.nn.functional as F

from vibebg_engine.models.registry import ensure_hf_model


@torch.inference_mode()
def estimate_depth(rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """HxWx3 uint8 RGB -> HxW float32 nearness in [0, 1] (1 = closest to the camera)."""
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    path = ensure_hf_model("depth_anything_v2_small")
    processor = AutoImageProcessor.from_pretrained(path)
    model = AutoModelForDepthEstimation.from_pretrained(path).to(device).eval()

    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt").to(device)
    pred = model(**inputs).predicted_depth  # relative inverse depth, larger = nearer
    pred = F.interpolate(pred[:, None], size=rgb.shape[:2], mode="bicubic",
                         align_corners=False)[0, 0]
    lo, hi = torch.quantile(pred.flatten().cpu(), torch.tensor([0.02, 0.98]))
    return ((pred.cpu() - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1).numpy().astype(np.float32)
