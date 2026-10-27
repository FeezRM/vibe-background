"""AI Relight (experimental): IC-Light (Zhang et al., Apache-2.0 code, OpenRAIL-M weights)
re-renders the subject as if lit by the chosen background.

Diffusion output is never used directly. IC-Light runs on a few keyframes; from each we keep
only a smooth map of how the light changed (relit / original, heavily blurred). That map is
applied to every frame, so the face, its detail and the motion all stay the camera's own, and
nothing flickers from frame to frame. Maps are stored relative to the subject's bounding box,
so they follow the person as they move between keyframes.
"""

import threading
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vibebg_engine.composite import srgb_to_linear
from vibebg_engine.imaging import normalized_blur
from vibebg_engine.models.registry import ensure_hf_model

PROMPT = "a person, natural soft lighting, best quality"
NEGATIVE = "lowres, bad anatomy, bad hands, cropped, worst quality"
_WORK_LONG_EDGE = 640  # IC-Light is an SD 1.5 model; it works best around 512-768px
_STEPS = 20
_GUIDANCE = 7.0
_MAP_SIZE = 64  # the light map is low-frequency, so a small grid holds it
_RATIO_RANGE = (0.3, 3.0)


@dataclass
class LightMap:
    """How the light on the subject should change, over the subject's bounding box."""

    ratio: torch.Tensor  # 3 x _MAP_SIZE x _MAP_SIZE, multiplier in linear light
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of the frame


def subject_box(alpha: torch.Tensor, pad: float = 0.04) -> tuple[float, float, float, float] | None:
    """Bounding box of alpha > 0.5 as frame fractions, padded; None if there is no subject."""
    mask = alpha[0] > 0.5
    if not bool(mask.any()):
        return None
    h, w = mask.shape
    rows = torch.nonzero(mask.any(dim=1)).flatten()
    cols = torch.nonzero(mask.any(dim=0)).flatten()
    return (max(0.0, int(cols[0]) / w - pad), max(0.0, int(rows[0]) / h - pad),
            min(1.0, (int(cols[-1]) + 1) / w + pad), min(1.0, (int(rows[-1]) + 1) / h + pad))


def _crop(img: torch.Tensor, box, size: tuple[int, int]) -> torch.Tensor:
    """Resample the box region of a CxHxW image to size (h, w)."""
    x0, y0, x1, y1 = box
    theta = torch.tensor([[(x1 - x0), 0, x0 + x1 - 1], [0, (y1 - y0), y0 + y1 - 1]],
                         device=img.device, dtype=torch.float32)
    grid = F.affine_grid(theta[None], [1, img.shape[0], *size], align_corners=False)
    return F.grid_sample(img[None].float(), grid, mode="bilinear", padding_mode="border",
                         align_corners=False)[0]


def paste_map(light: LightMap, box, height: int, width: int) -> torch.Tensor:
    """Place a light map over `box` in a full frame; 1 (no change) everywhere else."""
    x0, y0, x1, y1 = box
    # For each output pixel, where it falls inside the box, in the map's [-1, 1] coordinates.
    kx, ky = 1 / max(x1 - x0, 1e-6), 1 / max(y1 - y0, 1e-6)
    theta = torch.tensor([[kx, 0, kx * (1 - x0 - x1)], [0, ky, ky * (1 - y0 - y1)]],
                         device=light.ratio.device)
    grid = F.affine_grid(theta[None], [1, 3, height, width], align_corners=False)
    out = F.grid_sample((light.ratio - 1)[None], grid, mode="bilinear", padding_mode="zeros",
                        align_corners=False)[0]
    return out + 1


def blend_maps(a: LightMap, b: LightMap, t: float) -> LightMap:
    """Linear blend from a (t=0) to b (t=1), including the box."""
    box = tuple(pa * (1 - t) + pb * t for pa, pb in zip(a.box, b.box, strict=True))
    return LightMap(ratio=a.ratio * (1 - t) + b.ratio * t, box=box)


_shared: dict[str, "ICLight"] = {}
_shared_lock = threading.Lock()


def get_iclight(device: torch.device) -> "ICLight":
    """One IC-Light per process: it is ~2.5 GB, and previews and exports share it."""
    with _shared_lock:
        if str(device) not in _shared:
            _shared[str(device)] = ICLight(device)
        return _shared[str(device)]


class ICLight:
    """IC-Light's background-conditioned model (fbc) on Realistic Vision 5.1."""

    def __init__(self, device: torch.device):
        from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
        from safetensors import safe_open
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = device
        self.dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
        base = ensure_hf_model("realistic_vision_v51")
        self.tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder",
                                                          torch_dtype=self.dtype)
        self.text_encoder.to(device).eval()
        # The VAE stays in float32: in float16 it overflows to black or NaN images.
        self.vae = AutoencoderKL.from_pretrained(base, subfolder="vae").to(device).eval()
        unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=self.dtype)

        # IC-Light widens the first conv to take the foreground and background latents as
        # 8 extra channels, and ships its weights as offsets from the base UNet. They are added
        # one tensor at a time so peak memory stays near the size of one model.
        with torch.no_grad():
            old = unet.conv_in
            conv = torch.nn.Conv2d(12, old.out_channels, old.kernel_size, old.stride, old.padding,
                                   dtype=self.dtype)
            conv.weight.zero_()
            conv.weight[:, :4] = old.weight
            conv.bias.copy_(old.bias)
            unet.conv_in = conv
            params = dict(unet.named_parameters())
            offsets = ensure_hf_model("iclight_fbc") / "iclight_sd15_fbc.safetensors"
            with safe_open(str(offsets), framework="pt") as f:
                for key in f.keys():  # noqa: SIM118 - safe_open is not a dict
                    params[key].add_(f.get_tensor(key).to(self.dtype))
        self.unet = unet.to(device).eval()
        self.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear", algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True, steps_offset=1)
        self._embeds = torch.cat([self._encode(NEGATIVE), self._encode(PROMPT)])
        self._lock = threading.Lock()  # the scheduler is stateful; one sampling run at a time

    def _encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, padding="max_length", max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(self.device)
        return self.text_encoder(ids)[0]

    def _latents(self, img: torch.Tensor) -> torch.Tensor:
        """3xHxW sRGB [0, 1] -> 4 x H/8 x W/8 scaled latent."""
        dist = self.vae.encode((img[None] * 2 - 1).float()).latent_dist
        return dist.mode()[0] * self.vae.config.scaling_factor

    @torch.inference_mode()
    def relight(self, subject: torch.Tensor, background: torch.Tensor, seed: int = 0
                ) -> torch.Tensor:
        """subject: the person on flat gray; background: what is behind them. Both 3xHxW sRGB
        [0, 1] with H and W multiples of 64. Returns the relit subject, same size."""
        with self._lock:
            return self._sample(subject, background, seed)

    def _sample(self, subject: torch.Tensor, background: torch.Tensor, seed: int) -> torch.Tensor:
        h, w = subject.shape[-2:]
        cond = torch.cat([self._latents(subject), self._latents(background)])[None]
        cond = cond.to(self.dtype)
        gen = torch.Generator("cpu").manual_seed(seed)
        latents = torch.randn((1, 4, h // 8, w // 8), generator=gen).to(self.device, self.dtype)
        self.scheduler.set_timesteps(_STEPS, device=self.device)
        latents = latents * self.scheduler.init_noise_sigma
        for t in self.scheduler.timesteps:
            x = self.scheduler.scale_model_input(torch.cat([latents] * 2), t)
            x = torch.cat([x, torch.cat([cond] * 2)], dim=1)
            noise = self.unet(x, t, encoder_hidden_states=self._embeds).sample
            uncond, text = noise.chunk(2)
            noise = uncond + _GUIDANCE * (text - uncond)
            latents = self.scheduler.step(noise, t, latents).prev_sample
        img = self.vae.decode(latents.float() / self.vae.config.scaling_factor).sample[0]
        return ((img + 1) / 2).clamp(0, 1)


def _work_size(box, height: int, width: int) -> tuple[int, int]:
    bw, bh = (box[2] - box[0]) * width, (box[3] - box[1]) * height
    scale = _WORK_LONG_EDGE / max(bw, bh)
    return max(64, round(bh * scale / 64) * 64), max(64, round(bw * scale / 64) * 64)


@torch.inference_mode()
def light_map(model: ICLight, fg: torch.Tensor, alpha: torch.Tensor, bg: torch.Tensor,
              seed: int = 0) -> LightMap | None:
    """Relight one keyframe and reduce the result to a smooth light map.

    fg, bg: 3xHxW sRGB as they will be composited (subject already placed); alpha: 1xHxW.
    """
    box = subject_box(alpha)
    if box is None:
        return None
    h, w = alpha.shape[-2:]
    size = _work_size(box, h, w)
    a = _crop(alpha, box, size).clamp(0, 1)
    subj = _crop(fg, box, size) * a + 0.5 * (1 - a)
    back = _crop(bg, box, size)
    relit = model.relight(subj, back, seed=seed)

    # Compare relit and original light only where the person is, blurred so that only broad
    # lighting survives; facial detail and texture never come from the diffusion output.
    sigma = 0.06 * max(size)
    weight = (a > 0.5).float()
    before = normalized_blur(srgb_to_linear(subj), weight, sigma)
    after = normalized_blur(srgb_to_linear(relit), weight, sigma)
    ratio = (after / before.clamp_min(1e-3)).clamp(*_RATIO_RANGE)
    ratio = F.interpolate(ratio[None], size=(_MAP_SIZE, _MAP_SIZE), mode="area")[0]
    return LightMap(ratio=ratio, box=box)


def apply_light_map(fg_linear: torch.Tensor, alpha: torch.Tensor, light: LightMap,
                    strength: float) -> torch.Tensor:
    """Multiply the map onto the subject (linear light), following the subject's current box."""
    box = subject_box(alpha) or light.box
    h, w = alpha.shape[-2:]
    gain = paste_map(light, box, h, w)
    return fg_linear * gain.clamp_min(0) ** strength

