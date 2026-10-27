"""End-to-end render: decode, matte, relight, composite over a background, encode with audio."""

import itertools
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from vibebg_engine.background.source import (
    BlurredOriginalBackground,
    ImageBackground,
    VideoBackground,
    is_video,
)
from vibebg_engine.composite import linear_to_srgb, srgb_to_linear, to_tensor, to_uint8
from vibebg_engine.device import pick_device
from vibebg_engine.io.video import FrameReader, FrameWriter, VideoInfo, probe
from vibebg_engine.lighting.enhance import Enhancer, apply_warmth
from vibebg_engine.lighting.match import Matcher
from vibebg_engine.lighting.relight import (
    LightMap,
    apply_light_map,
    blend_maps,
    get_iclight,
    light_map,
)
from vibebg_engine.lighting.wrap_shadow import GrainMatcher, drop_shadow, light_wrap
from vibebg_engine.matting.refine import MaxMatter
from vibebg_engine.matting.rvm import RVMMatter
from vibebg_engine.placement import Placement
from vibebg_engine.scene import SceneCutDetector

TIERS = ("fast", "balanced", "max")
LIGHTING_MODES = ("off", "enhance", "match", "relight")
# Seconds between AI Relight keyframes; the light map is interpolated in between.
RELIGHT_INTERVAL = 2.0


class Cancelled(Exception):
    pass


@dataclass
class BackgroundSettings:
    """path: an image or video file; None means "blur my real background"."""

    path: Path | None = None
    dof: float = 0.0  # depth-of-field blur for still images, 0..1
    blur: float = 0.0  # uniform blur for videos and for the blurred original room, 0..1
    depth_path: Path | None = None  # precomputed depth map shipped with a preset
    seamless: bool = False  # video already loops cleanly; skip the crossfade


@dataclass
class LightingSettings:
    mode: str = "off"  # off | enhance | match | relight
    strength: float = 0.6
    wrap: float = 0.5  # match and relight
    shadow: float = 0.0  # match and relight
    warmth: float = 0.0  # -1..1, any mode
    denoise: bool = False


@dataclass
class RenderSettings:
    background: BackgroundSettings = field(default_factory=BackgroundSettings)
    tier: str = "fast"
    codec: str = "h264"
    placement: Placement = field(default_factory=Placement)
    lighting: LightingSettings = field(default_factory=LightingSettings)

    def validate(self):
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}; choose from {TIERS}")
        if self.lighting.mode not in LIGHTING_MODES:
            raise ValueError(f"unknown lighting mode {self.lighting.mode!r}")
        if self.background.path is None and not self.placement.is_identity:
            raise ValueError("the subject can't be moved when blurring the original background")


@dataclass
class Progress:
    frame: int
    total: int
    elapsed: float
    stage: str = "render"  # "relight" while AI Relight keyframes are being prepared

    @property
    def eta(self) -> float | None:
        if self.frame == 0:
            return None
        return self.elapsed / self.frame * (self.total - self.frame)


def decode_filters(settings: RenderSettings) -> tuple[str, ...]:
    return ("hqdn3d=2:1.5:3:3",) if settings.lighting.denoise else ()


class FrameProcessor:
    """Turns source frames into finished frames, one at a time and in order.

    `start` is where in the source the first processed frame sits (used by previews so a
    looping video background lines up with the moment being previewed).
    """

    def __init__(self, info: VideoInfo, settings: RenderSettings, device: torch.device,
                 start: float = 0.0):
        settings.validate()
        self.info = info
        self.device = device
        self.start = start
        self.matter = make_matter(settings.tier, device)
        self.background = _make_background(info, settings.background, device, start)
        self.cuts = SceneCutDetector()
        self.settings = settings
        self.frame_index = 0  # frames processed so far, i.e. the index of the next frame
        self.light_maps: list[tuple[int, LightMap]] = []
        self.configure_lighting(settings)

    def configure_lighting(self, settings: RenderSettings):
        """Apply new lighting and placement settings. These are cheap to change, unlike the
        tier and background, which need a new processor."""
        settings.validate()
        self.settings = settings
        fps = float(self.info.fps)
        light = settings.lighting
        self.enhancer = Enhancer(fps, light.strength) if light.mode != "off" else None
        self.matcher = Matcher(fps, light.strength) if light.mode == "match" else None
        self.grain = GrainMatcher(fps) if light.mode in ("match", "relight") else None

    @torch.inference_mode()
    def prepare_relight(self, keyframes: list[tuple[int, float, np.ndarray]],
                        on_done: Callable[[int], None] | None = None,
                        cancel: threading.Event | None = None):
        """Run IC-Light on keyframes given as (frame index, time in seconds, frame), in order.
        Must be called before process(); it leaves the matting state reset."""
        model = get_iclight(self.device)
        maps = []
        for n, (index, t, frame) in enumerate(keyframes):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            self.matter.reset()
            src = to_tensor(frame, self.device)
            fg, alpha = self.matter(src)
            self.background.seek(t)
            bg = self.background.next(src, alpha)
            fg, alpha = self.settings.placement.apply(fg, alpha)
            lm = light_map(model, fg, alpha, bg, seed=index)
            if lm is not None:
                maps.append((index, lm))
            if on_done:
                on_done(n + 1)
        self.matter.reset()
        self.background.seek(self.start)
        self.light_maps = maps

    def _light_map_at(self, index: int) -> LightMap | None:
        maps = self.light_maps
        if not maps:
            return None
        if index <= maps[0][0]:
            return maps[0][1]
        for (i0, m0), (i1, m1) in itertools.pairwise(maps):
            if index <= i1:
                return blend_maps(m0, m1, (index - i0) / (i1 - i0))
        return maps[-1][1]

    def reset(self):
        """Forget per-shot history, at a scene cut."""
        self.matter.reset()
        for part in (self.enhancer, self.matcher, self.grain):
            if part is not None:
                part.reset()

    @torch.inference_mode()
    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.cuts.is_cut(frame):
            self.reset()
        light, placement = self.settings.lighting, self.settings.placement

        src = to_tensor(frame, self.device)
        fg_srgb, alpha = self.matter(src)
        bg = srgb_to_linear(self.background.next(src, alpha))
        fg = srgb_to_linear(fg_srgb)

        if self.enhancer is not None:
            self.enhancer.update(srgb_to_linear(src), fg, alpha)
            fg = self.enhancer.apply(fg)
        if self.matcher is not None:
            self.matcher.update(fg, alpha, bg)
            fg = self.matcher.apply(fg)
        fg = apply_warmth(fg, light.warmth)

        fg, alpha = placement.apply(fg, alpha)
        if light.mode == "relight":
            lm = self._light_map_at(self.frame_index)
            if lm is not None:
                fg = apply_light_map(fg, alpha, lm, light.strength)
        self.frame_index += 1

        if light.mode in ("match", "relight"):
            bg = drop_shadow(bg, alpha, light.shadow)
            fg = light_wrap(fg, alpha, bg, light.wrap)
            bg = srgb_to_linear(self.grain.apply(linear_to_srgb(fg), alpha, linear_to_srgb(bg)))

        out = fg * alpha + bg * (1.0 - alpha)
        return to_uint8(linear_to_srgb(out))

    def close(self):
        close = getattr(self.background, "close", None)
        if close:
            close()


def make_matter(tier: str, device: torch.device) -> RVMMatter | MaxMatter:
    if tier == "max":
        return MaxMatter(device)
    return RVMMatter("rvm_mobilenetv3" if tier == "fast" else "rvm_resnet50", device)


def _make_background(info: VideoInfo, bgs: BackgroundSettings, device: torch.device,
                     start: float):
    w, h = info.width, info.height
    if bgs.path is None:
        return BlurredOriginalBackground(w, h, bgs.blur)
    if is_video(bgs.path):
        return VideoBackground(bgs.path, w, h, info.fps, device, blur=bgs.blur,
                               crossfade=0.0 if bgs.seamless else 0.5, start=start)
    return ImageBackground(bgs.path, w, h, device, dof=bgs.dof, depth_path=bgs.depth_path)


class _Failed:
    def __init__(self, error: BaseException):
        self.error = error


_END = object()


class _Prefetch:
    """Iterates `items` on a background thread, keeping up to `depth` items ready, so
    decoding overlaps with GPU work. Errors are re-raised in the consuming thread."""

    def __init__(self, items, depth: int = 4):
        self._q: queue.Queue = queue.Queue(depth)
        self._stop = threading.Event()
        threading.Thread(target=self._run, args=(items,), daemon=True).start()

    def _run(self, items):
        try:
            for item in items:
                if not self._put(item):
                    return
            self._put(_END)
        except BaseException as e:  # noqa: BLE001 - handed to the consumer
            self._put(_Failed(e))

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def __iter__(self):
        while (item := self._q.get()) is not _END:
            if isinstance(item, _Failed):
                raise item.error
            yield item

    def close(self):
        self._stop.set()


class _WriteBehind:
    """Feeds a FrameWriter from a background thread, so encoding overlaps with GPU work."""

    def __init__(self, writer: FrameWriter, depth: int = 4):
        self._writer = writer
        self._q: queue.Queue = queue.Queue(depth)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while (frame := self._q.get()) is not _END:
            if self._error is None:
                try:
                    self._writer.write(frame)
                except BaseException as e:  # noqa: BLE001 - re-raised by write()/close()
                    self._error = e

    def write(self, frame: np.ndarray):
        if self._error is not None:
            raise self._error
        self._q.put(frame)

    def close(self):
        self._q.put(_END)
        self._thread.join()
        if self._error is not None:
            raise self._error
        self._writer.close()

    def abort(self):
        self._error = self._error or Cancelled()
        self._q.put(_END)
        self._writer.abort()


def _relight_keyframes(info: VideoInfo, start: float, total: int,
                       filters: tuple[str, ...]) -> list[tuple[int, float, np.ndarray]]:
    """Frames to run IC-Light on: one every RELIGHT_INTERVAL seconds, plus the last frame."""
    step = max(1, round(RELIGHT_INTERVAL * float(info.fps)))
    indices = sorted({*range(0, total, step), total - 1})
    keyframes = []
    for i in indices:
        t = start + i / float(info.fps)
        frame = next(iter(FrameReader(info, start=t, max_frames=1, filters=filters)), None)
        if frame is not None:
            keyframes.append((i, t, frame))
    return keyframes


def _prepare_relight(processor: FrameProcessor, info: VideoInfo, start: float, total: int,
                     settings: RenderSettings, on_progress, cancel):
    keyframes = _relight_keyframes(info, start, total, decode_filters(settings))
    started = time.monotonic()

    def done(n: int):
        if on_progress:
            on_progress(Progress(n, len(keyframes), time.monotonic() - started, stage="relight"))

    processor.prepare_relight(keyframes, on_done=done, cancel=cancel)


def _encode(processor: FrameProcessor, info: VideoInfo, out: Path, settings: RenderSettings,
            start: float, max_frames: int | None, total: int, on_progress, cancel):
    reader = FrameReader(info, start=start, max_frames=max_frames,
                         filters=decode_filters(settings))
    frames = _Prefetch(reader)
    writer = _WriteBehind(FrameWriter(
        out, info.width, info.height, info.fps, audio_source=info, codec=settings.codec,
        audio_start=start, duration=None if max_frames is None else max_frames / info.fps))
    started = time.monotonic()
    try:
        for i, frame in enumerate(frames):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            writer.write(processor.process(frame))
            if on_progress:
                on_progress(Progress(i + 1, max(total, i + 1), time.monotonic() - started))
        writer.close()
    except BaseException:
        frames.close()
        reader.close()
        writer.abort()
        out.unlink(missing_ok=True)
        raise


def render(
    src: str | Path,
    out: str | Path,
    settings: RenderSettings,
    on_progress: Callable[[Progress], None] | None = None,
    cancel: threading.Event | None = None,
    start: float = 0.0,
    duration: float | None = None,
) -> Path:
    """Render src to out. start/duration render just a section (used for short previews)."""
    info = probe(src)
    processor = FrameProcessor(info, settings, pick_device(), start=start)

    out = Path(out)
    total = info.frame_count
    max_frames = None
    if duration is not None:
        max_frames = max(1, round(duration * info.fps))
        total = min(total, max_frames)
    try:
        if settings.lighting.mode == "relight":
            _prepare_relight(processor, info, start, total, settings, on_progress, cancel)
        _encode(processor, info, out, settings, start, max_frames, total, on_progress, cancel)
    finally:
        processor.close()
    return out
