"""Background layers, produced at the output frame size as 3xHxW sRGB tensors."""

import hashlib
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch

from vibebg_engine.background.depth import estimate_depth
from vibebg_engine.composite import to_tensor
from vibebg_engine.imaging import normalized_blur, variable_blur
from vibebg_engine.io.video import FrameReader, ffmpeg_bin, probe
from vibebg_engine.models.registry import models_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}

# Blur sliders (0..1) map to a Gaussian sigma that scales with the frame's long edge.
_MAX_SIGMA_FRACTION = 0.02
_DOF_BASE = 0.35


def blur_sigma(amount: float, width: int, height: int) -> float:
    return amount * _MAX_SIGMA_FRACTION * max(width, height)


def cover_fit(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale to fill width x height, cropping the overflow around the center."""
    h, w = img.shape[:2]
    scale = max(width / w, height / h)
    nw, nh = max(width, round(w * scale)), max(height, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    x, y = (nw - width) // 2, (nh - height) // 2
    return np.ascontiguousarray(resized[y:y + height, x:x + width])


def load_image(path: str | Path, width: int, height: int) -> np.ndarray:
    """Load an image file as HxWx3 uint8 RGB, cover-fit to the output size."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read background image {path}")
    return cover_fit(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), width, height)


class ImageBackground:
    """A still image, optionally with depth-of-field blur (far things blur more)."""

    def __init__(self, path: Path, width: int, height: int, device: torch.device,
                 dof: float = 0.0, depth_path: Path | None = None):
        rgb = load_image(path, width, height)
        img = to_tensor(rgb, device)
        if dof > 0:
            if depth_path is not None:
                depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
                depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
                near = depth.astype(np.float32) / 255.0
            else:
                near = estimate_depth(rgb, device)
            # Everything in the backdrop sits behind the subject, so even its nearest parts get
            # some blur; distance adds the rest.
            amount = torch.from_numpy(_DOF_BASE + (1.0 - _DOF_BASE) * (1.0 - near)).to(device)[None]
            img = variable_blur(img, amount, blur_sigma(dof, width, height))
        self._img = img

    def seek(self, t: float):
        """Stills look the same at every moment."""

    def next(self, frame: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return self._img


class VideoBackground:
    """A looping video, retimed to the output frame rate, optionally uniformly blurred."""

    def __init__(self, path: Path, width: int, height: int, fps: Fraction,
                 device: torch.device, blur: float = 0.0, crossfade: float = 0.5,
                 start: float = 0.0):
        src = make_seamless_loop(path, crossfade) if crossfade > 0 else Path(path)
        filters = ()
        if blur > 0:
            filters = (f"gblur=sigma={blur_sigma(blur, width, height):.2f}",)
        self._info = probe(src)
        self._opts = {"fps": fps, "size": (width, height), "filters": filters, "loop": True}
        self._device = device
        self._reader = None
        self.seek(start)

    def seek(self, t: float):
        """Continue from t seconds into the loop (wrapping around)."""
        if self._reader is not None:
            self._reader.close()
        self._reader = FrameReader(self._info, start=t % self._info.duration, **self._opts)
        self._frames = iter(self._reader)

    def next(self, frame: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return to_tensor(next(self._frames), self._device)

    def close(self):
        self._reader.close()


class BlurredOriginalBackground:
    """The person's own room, blurred, with the person cut out so they don't smear into it."""

    def __init__(self, width: int, height: int, blur: float):
        self._sigma = blur_sigma(max(blur, 0.05), width, height)

    def seek(self, t: float):
        """Built from each source frame, so there is nothing to reposition."""

    def next(self, frame: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # Squaring the weight keeps soft edges (hair, motion blur) out of the fill.
        weight = (1.0 - alpha) ** 2
        return normalized_blur(frame, weight, self._sigma)


def make_seamless_loop(path: str | Path, fade: float) -> Path:
    """Crossfade a clip's end into its start so it loops without a visible jump.

    The result is cached by source path, size and modification time.
    """
    path = Path(path)
    stat = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{fade}".encode())
    out = models_dir().parent / "cache" / "loops" / f"{key.hexdigest()}.mp4"
    if out.exists():
        return out

    duration = probe(path).duration
    if duration < 4 * fade:
        return path
    out.parent.mkdir(parents=True, exist_ok=True)
    end = duration - fade
    graph = (
        f"[0:v]split=3[a][b][c];"
        f"[a]trim=start={fade}:end={end},setpts=PTS-STARTPTS[body];"
        f"[b]trim=start={end},setpts=PTS-STARTPTS[tail];"
        f"[c]trim=end={fade},setpts=PTS-STARTPTS[head];"
        f"[tail][head]xfade=transition=fade:duration={fade}:offset=0[x];"
        f"[body][x]concat=n=2:v=1[out]"
    )
    tmp = out.with_suffix(".part.mp4")
    subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-y", "-i", str(path), "-filter_complex", graph,
         "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
         str(tmp)],
        check=True,
    )
    tmp.replace(out)
    return out


def is_video(path: str | Path) -> bool:
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTS:
        return True
    if ext in IMAGE_EXTS:
        return False
    raise ValueError(f"unsupported background file type {ext!r}")

