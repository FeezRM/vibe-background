import subprocess

import numpy as np
import pytest

from vibebg_engine.background.source import cover_fit, is_video, make_seamless_loop
from vibebg_engine.io.video import FrameReader, ffmpeg_bin, probe


def test_cover_fit_fills_target_and_crops_center():
    # Landscape 200x100 with a marker column in the center, fit into a portrait 50x100 frame.
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, 95:105] = 255
    out = cover_fit(img, 50, 100)
    assert out.shape == (100, 50, 3)
    assert out[:, 20:30].mean() > 200  # center marker survives the crop
    assert out[:, :10].mean() < 5


def test_cover_fit_upscales_small_images():
    out = cover_fit(np.full((10, 10, 3), 7, np.uint8), 64, 36)
    assert out.shape == (36, 64, 3)
    assert out.flags["C_CONTIGUOUS"]


def test_seamless_loop_wraps_without_a_jump(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEBG_MODELS_DIR", str(tmp_path / "models"))
    src = tmp_path / "bg.mp4"
    # A hue that drifts steadily, so the end of the clip looks nothing like its start.
    subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=red:size=64x64:rate=25:duration=4,hue=h=t*60",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    looped = make_seamless_loop(src, fade=0.5)
    info = probe(looped)
    assert abs(info.duration - 3.5) < 0.1

    frames = [f.astype(int) for f in FrameReader(info)]
    step = np.abs(frames[1] - frames[0]).mean()  # typical frame-to-frame change
    wrap = np.abs(frames[0] - frames[-1]).mean()  # change across the loop point
    assert wrap < 3 * step + 2
    # Cached: a second call reuses the file.
    assert make_seamless_loop(src, fade=0.5) == looped


def test_is_video():
    assert is_video("a.MP4") and not is_video("b.jpeg")
    with pytest.raises(ValueError):
        is_video("c.gif")
