"""ffprobe/ffmpeg wrappers: decode to RGB frames, encode RGB frames back with the source audio."""

import functools
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
# Audio codecs that can be stream-copied into MP4 without re-encoding.
MP4_AUDIO_COPY = {"aac", "mp3", "alac", "ac3", "eac3", "opus"}


def ffmpeg_bin(name: str = "ffmpeg") -> str:
    """Bundled binary if VIBEBG_FFMPEG_DIR is set, otherwise whatever is on PATH."""
    bundled = os.environ.get("VIBEBG_FFMPEG_DIR")
    if bundled:
        return str(Path(bundled) / name)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found on PATH and VIBEBG_FFMPEG_DIR is not set")
    return found


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int  # display width, after rotation
    height: int  # display height, after rotation
    fps: Fraction
    duration: float
    rotation: int
    color_space: str | None
    is_hdr: bool
    audio_codec: str | None

    @property
    def frame_count(self) -> int:
        return max(1, round(self.duration * self.fps))


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    out = subprocess.run(
        [ffmpeg_bin("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json",
         str(path)],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise ValueError(f"cannot read {path}: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError(f"{path} has no video stream")
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)

    rotation = 0
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(side["rotation"])
    if not rotation and "rotate" in video.get("tags", {}):
        rotation = int(video["tags"]["rotate"])

    w, h = int(video["width"]), int(video["height"])
    if rotation % 180:
        w, h = h, w

    fps = Fraction(video.get("r_frame_rate") or video["avg_frame_rate"])
    if fps <= 0:
        fps = Fraction(video["avg_frame_rate"])
    duration = float(video.get("duration") or data["format"]["duration"])

    return VideoInfo(
        path=path, width=w, height=h, fps=fps, duration=duration, rotation=rotation,
        color_space=video.get("color_space"),
        is_hdr=video.get("color_transfer") in HDR_TRANSFERS,
        audio_codec=audio["codec_name"] if audio else None,
    )


def _input_matrix(info: VideoInfo) -> str:
    if info.color_space in ("bt709", "bt470bg", "smpte170m", "bt2020nc"):
        return {"bt470bg": "bt601", "smpte170m": "bt601", "bt2020nc": "bt2020"}.get(
            info.color_space, info.color_space)
    # Untagged: HD and up is conventionally BT.709, SD is BT.601.
    return "bt709" if info.height >= 720 or info.width >= 1280 else "bt601"


def _to_sdr_filters(info: VideoInfo) -> list[str]:
    """Filters that bring the source to 8-bit-ready BT.709 SDR. HDR (HLG/PQ, e.g. iPhone
    footage) is tone-mapped, which needs an ffmpeg built with zimg (the zscale filter)."""
    if not info.is_hdr:
        return [f"scale=in_color_matrix={_input_matrix(info)}:in_range=auto"]
    if "zscale" not in _available_filters():
        raise RuntimeError(f"{info.path.name} is HDR video; tone-mapping it needs an ffmpeg "
                           "build with the zscale filter (libzimg)")
    return ["zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709",
            "tonemap=tonemap=hable:desat=0", "zscale=t=bt709:m=bt709:r=pc"]


class FrameReader:
    """Iterates a video as HxWx3 uint8 RGB frames at a constant frame rate, rotation applied.

    By default frames come out at the source's size and rate. `fps` retimes, `size` cover-fits
    (scale and center-crop) to (width, height), `filters` appends ffmpeg filters, and `loop`
    repeats the input forever.
    """

    def __init__(self, info: VideoInfo, start: float = 0.0, max_frames: int | None = None, *,
                 fps: Fraction | None = None, size: tuple[int, int] | None = None,
                 filters: tuple[str, ...] = (), loop: bool = False):
        self.info = info
        self.width, self.height = size or (info.width, info.height)
        self._frame_bytes = self.width * self.height * 3
        cmd = [ffmpeg_bin(), "-v", "error", "-nostdin"]
        if loop:
            cmd += ["-stream_loop", "-1"]
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", str(info.path), "-map", "0:v:0"]
        if max_frames is not None:
            cmd += ["-frames:v", str(max_frames)]
        chain = [f"fps={fps or info.fps}", *_to_sdr_filters(info)]
        if size:
            chain += [f"scale={self.width}:{self.height}:force_original_aspect_ratio=increase",
                      f"crop={self.width}:{self.height}"]
        chain += [*filters, "format=rgb24"]
        cmd += ["-vf", ",".join(chain), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._closed = False

    def __iter__(self):
        assert self._proc.stdout is not None and self._proc.stderr is not None
        while True:
            buf = bytearray(self._frame_bytes)
            if self._proc.stdout.readinto(buf) < self._frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)
        err = self._proc.stderr.read().decode(errors="replace")
        if self._proc.wait() != 0 and not self._closed:
            raise RuntimeError(f"ffmpeg decode of {self.info.path} failed: {err.strip()}")

    def close(self):
        self._closed = True
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()


@functools.cache
def _available_encoders() -> frozenset[str]:
    out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-encoders"], capture_output=True,
                         text=True, check=True).stdout
    return frozenset(line.split()[1] for line in out.splitlines()[10:] if len(line.split()) > 1)


@functools.cache
def _available_filters() -> frozenset[str]:
    out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-filters"], capture_output=True,
                         text=True, check=True).stdout
    return frozenset(line.split()[1] for line in out.splitlines() if len(line.split()) > 2)


def _video_codec_args(codec: str) -> list[str]:
    encoders = _available_encoders()
    if codec == "h264":
        if "h264_videotoolbox" in encoders:
            return ["-c:v", "h264_videotoolbox", "-q:v", "65", "-profile:v", "high"]
        if "h264_nvenc" in encoders:
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19"]
        return ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]
    if codec == "hevc":
        if "hevc_videotoolbox" in encoders:
            return ["-c:v", "hevc_videotoolbox", "-q:v", "60", "-tag:v", "hvc1"]
        if "hevc_nvenc" in encoders:
            return ["-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21",
                    "-tag:v", "hvc1"]
        return ["-c:v", "libx265", "-crf", "20", "-preset", "medium", "-tag:v", "hvc1"]
    raise ValueError(f"unsupported codec {codec!r}")


class FrameWriter:
    """Encodes HxWx3 uint8 RGB frames to MP4, muxing in the audio of `audio_source` if any."""

    def __init__(self, out_path: str | Path, width: int, height: int, fps: Fraction,
                 audio_source: VideoInfo | None = None, codec: str = "h264",
                 audio_start: float = 0.0, duration: float | None = None):
        self.width, self.height = width, height
        cmd = [ffmpeg_bin(), "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "pipe:0"]
        has_audio = audio_source is not None and audio_source.audio_codec is not None
        if has_audio:
            if audio_start > 0:
                cmd += ["-ss", f"{audio_start:.3f}"]
            cmd += ["-i", str(audio_source.path), "-map", "0:v:0", "-map", "1:a:0"]
            if audio_source.audio_codec in MP4_AUDIO_COPY:
                cmd += ["-c:a", "copy"]
            else:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
        if duration is not None:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += _video_codec_args(codec)
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-color_range", "tv", "-movflags", "+faststart", str(out_path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        assert frame.shape == (self.height, self.width, 3) and frame.dtype == np.uint8
        assert self._proc.stdin is not None
        self._proc.stdin.write(frame.tobytes())

    def close(self):
        assert self._proc.stdin is not None and self._proc.stderr is not None
        self._proc.stdin.close()
        err = self._proc.stderr.read().decode(errors="replace")
        if self._proc.wait() != 0:
            raise RuntimeError(f"ffmpeg encode failed: {err.strip()}")

    def abort(self):
        self._proc.kill()
        self._proc.wait()
