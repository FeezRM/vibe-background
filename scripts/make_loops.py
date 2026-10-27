"""Generate the built-in animated background loops.

Every motion is a whole number of cycles over the clip, so the last frame flows straight
into the first (the presets are marked seamless). The output is our own work, released with
the app.

    uv run --project engine python scripts/make_loops.py presets/backgrounds
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

SIZE = 1920  # square, so both landscape and portrait videos can cover-crop it
FPS = 24
SECONDS = 10
FRAMES = FPS * SECONDS
TAU = 2 * np.pi


def _grid(scale: int = 4):
    n = SIZE // scale
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    return x, y


def _finish(small: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Upscale a low-res float RGB frame and dither it so gradients don't band."""
    big = cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    big += rng.normal(0, 0.6 / 255, big.shape).astype(np.float32)
    return (np.clip(big, 0, 1) * 255).round().astype(np.uint8)


def bokeh(t: float, rng: np.random.Generator, orbs) -> np.ndarray:
    x, y = _grid()
    base = np.stack([0.10 + 0.10 * (1 - y), 0.06 + 0.05 * (1 - y), 0.05 + 0.02 * x], -1)
    img = base.copy()
    for cx, cy, r, color, ax, ay, k, phase in orbs:
        px = cx + ax * np.sin(TAU * k * t + phase)
        py = cy + ay * np.cos(TAU * k * t + phase)
        d = np.sqrt((x - px) ** 2 + (y - py) ** 2) / r
        disk = np.clip(1.15 - d, 0, 1) ** 1.5  # soft-edged disc, like an out-of-focus light
        img += disk[..., None] * color
    return img


def aurora(t: float, *_):
    x, y = _grid()
    a = np.sin(TAU * (x * 1.3 + t)) + np.sin(TAU * (y * 0.9 - 2 * t) + 1.7)
    b = np.sin(TAU * (x * 0.7 - y * 1.1 + t) + 0.4)
    teal = np.array([0.08, 0.36, 0.40], np.float32)
    violet = np.array([0.30, 0.14, 0.42], np.float32)
    night = np.array([0.03, 0.04, 0.09], np.float32)
    w1 = (0.5 + 0.25 * a)[..., None]
    w2 = (0.5 + 0.5 * b)[..., None]
    glow = (1 - y)[..., None] ** 1.4
    return night + glow * (w1 * teal + (1 - w1) * violet) * (0.55 + 0.45 * w2)


def studio(t: float, *_):
    x, y = _grid()
    # Seamless paper backdrop with a soft key light drifting slowly across it.
    base = np.array([0.42, 0.36, 0.31], np.float32) * (0.55 + 0.45 * (1 - y))[..., None]
    lx = 0.35 + 0.18 * np.sin(TAU * t)
    ly = 0.35 + 0.06 * np.sin(TAU * 2 * t)
    spot = np.exp(-((x - lx) ** 2 + (y - ly) ** 2) / 0.09)
    warm = np.array([0.36, 0.28, 0.18], np.float32)
    vignette = 1 - 0.45 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) * 2
    return (base + spot[..., None] * warm) * vignette[..., None]


def _orbs(rng: np.random.Generator):
    palette = [np.array(c, np.float32) for c in
               ([0.55, 0.36, 0.16], [0.50, 0.26, 0.12], [0.36, 0.30, 0.22], [0.48, 0.40, 0.30])]
    return [(rng.uniform(0, 1), rng.uniform(0.1, 0.9), rng.uniform(0.03, 0.09),
             palette[rng.integers(len(palette))] * rng.uniform(0.25, 0.6),
             rng.uniform(0.01, 0.05), rng.uniform(0.01, 0.04), int(rng.integers(1, 3)),
             rng.uniform(0, TAU)) for _ in range(38)]


LOOPS = {
    "warm-bokeh": ("Warm bokeh", bokeh, ["cozy", "evening"]),
    "aurora": ("Aurora", aurora, ["moody", "night"]),
    "studio-sweep": ("Studio sweep", studio, ["clean", "studio"]),
}


def render(folder: Path, name: str, fn, tags: list[str]):
    rng = np.random.default_rng(7)
    orbs = _orbs(rng)
    folder.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "pipe:0", "-c:v", "libx264",
         "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(folder / "bg.mp4")],
        stdin=subprocess.PIPE,
    )
    thumb = None
    for i in range(FRAMES):
        frame = _finish(fn(i / FRAMES, rng, orbs), rng)
        if i == 0:
            thumb = frame
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {name}")

    h = SIZE * 9 // 16
    crop = thumb[(SIZE - h) // 2:(SIZE + h) // 2]
    cv2.imwrite(str(folder / "thumb.jpg"),
                cv2.cvtColor(cv2.resize(crop, (480, 270), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_RGB2BGR))
    manifest = {"name": name, "type": "video", "file": "bg.mp4", "thumb": "thumb.jpg",
                "seamless": True, "tags": tags, "license": "GPL-3.0-or-later",
                "source": "Made for Vibe Background (scripts/make_loops.py)"}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--only", choices=sorted(LOOPS))
    args = parser.parse_args()
    for key, (name, fn, tags) in LOOPS.items():
        if args.only and key != args.only:
            continue
        print(f"rendering {key}…", file=sys.stderr)
        render(args.out / key, name, fn, tags)
    return 0


if __name__ == "__main__":
    sys.exit(main())
