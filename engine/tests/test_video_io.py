import subprocess
from fractions import Fraction

import numpy as np
import pytest

from vibebg_engine.io.video import FrameReader, FrameWriter, ffmpeg_bin, probe


def _make_clip(path, *, size="320x240", rate=30, seconds=1, audio="aac"):
    cmd = [ffmpeg_bin(), "-v", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", audio]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture
def clip(tmp_path):
    return _make_clip(tmp_path / "in.mp4")


def test_probe_reads_basic_metadata(clip):
    info = probe(clip)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == Fraction(30)
    assert info.frame_count == 30
    assert info.audio_codec == "aac"
    assert not info.is_hdr


def test_probe_swaps_dimensions_for_rotated_video(tmp_path):
    src = _make_clip(tmp_path / "plain.mp4", audio=None)
    rotated = tmp_path / "rotated.mp4"
    subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-display_rotation", "90", "-i", str(src),
                    "-c", "copy", str(rotated)], check=True)
    info = probe(rotated)
    assert info.rotation in (90, -90)
    assert (info.width, info.height) == (240, 320)
    frames = list(FrameReader(info))
    assert frames[0].shape == (320, 240, 3)


def test_probe_rejects_non_video(tmp_path):
    bogus = tmp_path / "x.mp4"
    bogus.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        probe(bogus)


def test_roundtrip_keeps_frames_fps_and_audio(clip, tmp_path):
    info = probe(clip)
    frames = list(FrameReader(info))
    assert len(frames) == 30
    assert frames[0].shape == (240, 320, 3) and frames[0].dtype == np.uint8

    out = tmp_path / "out.mp4"
    writer = FrameWriter(out, info.width, info.height, info.fps, audio_source=info)
    for f in frames:
        writer.write(f)
    writer.close()

    result = probe(out)
    assert result.fps == info.fps
    assert result.frame_count == 30
    assert result.audio_codec == "aac"
    # Colors survive the RGB -> yuv420p -> RGB trip (testsrc2 has flat color regions).
    back = next(iter(FrameReader(result)))
    assert np.abs(back.astype(int) - frames[0].astype(int)).mean() < 6


def test_writer_without_audio_source(tmp_path):
    out = tmp_path / "silent.mp4"
    writer = FrameWriter(out, 64, 64, Fraction(24))
    for _ in range(5):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.close()
    assert probe(out).audio_codec is None


def test_hdr_source_is_tone_mapped_or_rejected_clearly(tmp_path):
    from vibebg_engine.io.video import _available_filters

    src = tmp_path / "hlg.mp4"
    subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x240:rate=30:duration=0.5", "-c:v", "libx265",
                    "-pix_fmt", "yuv420p10le", "-x265-params",
                    "log-level=error:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
                    str(src)], check=True)
    info = probe(src)
    assert info.is_hdr
    if "zscale" in _available_filters():
        frames = list(FrameReader(info))
        assert len(frames) == 15 and frames[0].mean() > 10
    else:
        with pytest.raises(RuntimeError, match="zscale"):
            FrameReader(info)
