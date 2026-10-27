"""Alpha compositing in linear light. Tensors are float32 CHW in [0, 1] on any device."""

import torch


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def composite(fg: torch.Tensor, alpha: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    """Place a straight (un-premultiplied) sRGB foreground over an sRGB background.

    fg, bg: 3xHxW sRGB. alpha: 1xHxW coverage. Returns 3xHxW sRGB.
    """
    out = srgb_to_linear(fg) * alpha + srgb_to_linear(bg) * (1.0 - alpha)
    return linear_to_srgb(out)


def to_tensor(frame, device: torch.device) -> torch.Tensor:
    """HxWx3 uint8 numpy -> 3xHxW float32 [0, 1] on device."""
    return torch.from_numpy(frame).to(device).permute(2, 0, 1).float().div_(255.0)


def to_uint8(img: torch.Tensor):
    """3xHxW float [0, 1] -> HxWx3 uint8 numpy on CPU."""
    return img.clamp(0, 1).mul(255.0).round_().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
