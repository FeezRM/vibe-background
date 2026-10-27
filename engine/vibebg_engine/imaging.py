"""GPU image helpers: blurs and color-space conversions. Tensors are float CxHxW."""

import math

import torch
import torch.nn.functional as F


def _gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _separable_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    c = img.shape[0]
    k = _gaussian_kernel(sigma, img.device)
    r = (k.numel() - 1) // 2
    x = img[None]
    h, w = img.shape[-2:]
    mode = "reflect" if r < min(h, w) else "replicate"
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=mode), k.view(1, 1, 1, -1).expand(c, 1, 1, -1),
                 groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode=mode), k.view(1, 1, -1, 1).expand(c, 1, -1, 1),
                 groups=c)
    return x[0]


def pool(img: torch.Tensor, factor: int) -> torch.Tensor:
    """Box-downsample a CxHxW tensor by an integer factor, padding the edges to a multiple of
    it. (Area interpolation would do, but MPS only supports it for divisible sizes.)"""
    if factor <= 1:
        return img
    h, w = img.shape[-2:]
    ph, pw = (-h) % factor, (-w) % factor
    x = F.pad(img[None], (0, pw, 0, ph), mode="replicate") if ph or pw else img[None]
    return F.avg_pool2d(x, factor)[0]


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of a CxHxW tensor. Large sigmas run at reduced resolution, which is
    visually identical because the result has no high frequencies left."""
    if sigma <= 0:
        return img
    factor = max(1, int(sigma // 3))
    if factor == 1:
        return _separable_blur(img, sigma)
    h, w = img.shape[-2:]
    small = _separable_blur(pool(img, factor), sigma / factor)
    sh, sw = small.shape[-2:]
    # Upsample to the padded size so pixel centers line up, then drop the padding.
    up = F.interpolate(small[None], size=(sh * factor, sw * factor), mode="bilinear",
                       align_corners=False)[0]
    return up[..., :h, :w]


def normalized_blur(img: torch.Tensor, weight: torch.Tensor, sigma: float,
                    eps: float = 1e-4) -> torch.Tensor:
    """Blur only the pixels where weight is high, filling the rest from their surroundings.

    Used to blur a room without smearing the person who stands in front of it.
    """
    num = gaussian_blur(img * weight, sigma)
    den = gaussian_blur(weight, sigma)
    return num / den.clamp_min(eps)


def variable_blur(img: torch.Tensor, amount: torch.Tensor, max_sigma: float,
                  levels: int = 6) -> torch.Tensor:
    """Per-pixel blur: amount (1xHxW in [0, 1]) picks how much of max_sigma each pixel gets."""
    sigmas = [max_sigma * i / (levels - 1) for i in range(levels)]
    stack = torch.stack([gaussian_blur(img, s) for s in sigmas])  # L x C x H x W
    pos = amount.clamp(0, 1) * (levels - 1)
    lo = pos.floor().long().clamp(max=levels - 2)
    t = pos - lo
    lo_img = torch.gather(stack, 0, lo.expand_as(img)[None]).squeeze(0)
    hi_img = torch.gather(stack, 0, (lo + 1).expand_as(img)[None]).squeeze(0)
    return lo_img * (1 - t) + hi_img * t


# sRGB D65 <-> CIE Lab. Inputs to rgb_to_lab are *linear* RGB.
_RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])
_XYZ2RGB = torch.linalg.inv(_RGB2XYZ)
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])
_EPS = 216 / 24389
_KAPPA = 24389 / 27


def rgb_to_lab(lin: torch.Tensor) -> torch.Tensor:
    m = _RGB2XYZ.to(lin.device)
    xyz = torch.einsum("ij,jhw->ihw", m, lin) / _WHITE.to(lin.device).view(3, 1, 1)
    f = torch.where(xyz > _EPS, xyz.clamp_min(1e-12) ** (1 / 3), (_KAPPA * xyz + 16) / 116)
    return torch.stack([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    fy = (lab[0] + 16) / 116
    fx = fy + lab[1] / 500
    fz = fy - lab[2] / 200
    f = torch.stack([fx, fy, fz])
    xyz = torch.where(f**3 > _EPS, f**3, (116 * f - 16) / _KAPPA)
    xyz = xyz * _WHITE.to(lab.device).view(3, 1, 1)
    return torch.einsum("ij,jhw->ihw", _XYZ2RGB.to(lab.device), xyz)


def luma(img: torch.Tensor) -> torch.Tensor:
    """Rec. 709 luma of a 3xHxW tensor, as 1xHxW."""
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=img.device).view(3, 1, 1)
    return (img * w).sum(0, keepdim=True)
