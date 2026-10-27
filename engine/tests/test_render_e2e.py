"""Renders real clips through the models. Drop talking-head clips into tests/fixtures/ to run."""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from vibebg_engine.io.video import FrameReader, ffmpeg_bin, probe
from vibebg_engine.pipeline import BackgroundSettings, LightingSettings, RenderSettings, render

FIXTURES = Path(__file__).parent / "fixtures"
CLIPS = sorted(FIXTURES.glob("*.mp4"))

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not CLIPS, reason="no clips in tests/fixtures/"),
]


@pytest.fixture(scope="module")
def solid_bg(tmp_path_factory):
    path = tmp_path_factory.mktemp("bg") / "green.png"
    cv2.imwrite(str(path), np.full((720, 1280, 3), (40, 200, 40), np.uint8))  # BGR green
    return path


@pytest.mark.parametrize("tier", ["fast", "balanced", "max"])
@pytest.mark.parametrize("clip", CLIPS, ids=lambda p: p.stem)
def test_render_preserves_timing_and_audio(clip, tier, solid_bg, tmp_path):
    src = probe(clip)
    settings = RenderSettings(background=BackgroundSettings(path=solid_bg), tier=tier)
    out = render(clip, tmp_path / "out.mp4", settings)
    result = probe(out)

    assert (result.width, result.height) == (src.width, src.height)
    assert result.fps == src.fps
    assert result.audio_codec == src.audio_codec
    assert abs(result.duration - src.duration) <= 1.5 / float(src.fps)

    frames = list(FrameReader(result))
    assert len(frames) == len(list(FrameReader(src)))

    # The person is kept and the room is replaced: some pixels are the green background,
    # and some are not.
    mid = frames[len(frames) // 2].astype(int)
    is_bg = (np.abs(mid - [40, 200, 40]).sum(axis=2) < 30)
    assert 0.1 < is_bg.mean() < 0.95


@pytest.fixture(scope="module")
def video_bg(tmp_path_factory):
    path = tmp_path_factory.mktemp("bg") / "loop.mp4"
    subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=640x360:rate=25:duration=2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.mark.parametrize("background", ["video", "blur_original"])
def test_render_other_backgrounds_with_match_lighting(background, video_bg, tmp_path):
    clip = CLIPS[0]
    src = probe(clip)
    bgs = (BackgroundSettings(path=video_bg, blur=0.2) if background == "video"
           else BackgroundSettings(path=None, blur=0.5))
    settings = RenderSettings(background=bgs,
                              lighting=LightingSettings(mode="match", shadow=0.3, warmth=0.2))
    result = probe(render(clip, tmp_path / "out.mp4", settings))
    assert (result.width, result.height, result.fps) == (src.width, src.height, src.fps)
    assert abs(result.duration - src.duration) <= 1.5 / float(src.fps)
