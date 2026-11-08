"""Render the app icon: a person cut out in front of a warm, softly lit room.

Writes the PNGs, the Windows .ico and (on a Mac, with iconutil) the macOS .icns that
electron-builder and the app use. Our own work, released with the app.

    uv run --project engine python scripts/make_icon.py apps/desktop
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 1024
SS = 2  # supersampling factor
# Apple's icon grid: an 824 px rounded square centered on the 1024 px canvas.
BODY = 824
RADIUS = 185

BG_DARK = np.array([14, 14, 16]) / 255  # --bg in the app's CSS
ACCENT = np.array([242, 184, 128]) / 255  # --accent
EMBER = np.array([168, 86, 44]) / 255
DUSK = np.array([42, 29, 18]) / 255  # the warm glow behind the import screen
FIGURE = np.array([20, 18, 22]) / 255


def _coverage(sdf: np.ndarray, px: float) -> np.ndarray:
    """Antialiased fill for a signed distance field (negative inside)."""
    return np.clip(0.5 - sdf / px, 0, 1)


def _smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def _rounded_square(x, y, half: float, r: float) -> np.ndarray:
    qx, qy = np.abs(x) - half + r, np.abs(y) - half + r
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return outside + np.minimum(np.maximum(qx, qy), 0) - r


def _mix(a, b, t):
    return a * (1 - t[..., None]) + b * t[..., None]


def render() -> Image.Image:
    n = SIZE * SS
    px = 1.0 / SS  # one output pixel, in canvas units
    y, x = (np.mgrid[0:n, 0:n].astype(np.float32) + 0.5) / SS
    cx, cy = x - SIZE / 2, y - SIZE / 2

    body = _rounded_square(cx, cy, BODY / 2, RADIUS)

    # Room: a warm window glow up and to the left, falling off into dusk.
    light = np.array([340.0, 330.0])
    d = np.hypot(x - light[0], y - light[1]) / BODY
    glow = np.exp(-(d / 0.42) ** 2)
    room = _mix(_mix(BG_DARK, DUSK, np.clip(1.2 - d, 0, 1)), EMBER, glow * 0.85)
    room = _mix(room, ACCENT, np.exp(-(d / 0.18) ** 2) * 0.9)

    # Out-of-focus lights, for the depth-of-field look.
    for bx, by, br, a in ((715, 290, 66, 0.34), (800, 420, 40, 0.26), (235, 640, 44, 0.16),
                          (625, 200, 28, 0.3)):
        disc = _coverage(np.hypot(x - bx, y - by) - br, 10.0)
        room = _mix(room, ACCENT, disc * a)

    # The person: head and shoulders; the smooth union forms the neck.
    head = np.hypot(cx * 1.05, cy + 20) - 126
    sx, sy = cx / 340, (cy - 430) / 300
    shoulders = (np.hypot(sx, sy) - 1) * 300
    person = _smin(head, shoulders, 95)
    fig = _coverage(person, px)

    # Rim light on the edge facing the window: the app's light wrap.
    gy, gx = np.gradient(person)
    norm = np.hypot(gx, gy) + 1e-6
    to_light_x, to_light_y = light[0] - x, light[1] - y
    facing = np.clip((gx * to_light_x + gy * to_light_y) / (norm * np.hypot(to_light_x, to_light_y)),
                     0, 1)
    rim = (fig - _coverage(person + 6, px)) * facing ** 1.1  # a crisp 6 px lit edge
    spill = np.exp(-np.maximum(person, 0) / 14) * (1 - fig) * facing ** 1.5  # light wrap
    shade = np.clip(-person / 400, 0, 1)  # a touch darker toward the middle of the figure
    figure = FIGURE * (1 - 0.35 * shade[..., None])
    rgb = _mix(_mix(room, ACCENT, spill * 0.35), figure, fig)
    rgb = _mix(rgb, ACCENT, rim * 0.95)

    # A soft contact shadow under the tile, as on the macOS grid.
    shadow = np.clip(1 - np.maximum(_rounded_square(cx, cy - 12, BODY / 2, RADIUS), 0) / 28, 0, 1)
    alpha = np.maximum(_coverage(body, px), shadow ** 2 * 0.35)
    rgb = rgb * (_coverage(body, px) / np.maximum(alpha, 1e-6))[..., None]

    out = np.dstack([np.clip(rgb, 0, 1), alpha])
    img = Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")
    return img.resize((SIZE, SIZE), Image.LANCZOS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("app_dir", type=Path, help="apps/desktop")
    args = parser.parse_args(argv)
    build, resources = args.app_dir / "build", args.app_dir / "resources"

    icon = render()
    icon.save(build / "icon.png")
    icon.resize((512, 512), Image.LANCZOS).save(resources / "icon.png")

    # Windows shows icons edge to edge, so the .ico uses the tile without the macOS margin.
    margin = (SIZE - BODY) // 2
    tile = icon.crop((margin, margin, SIZE - margin, SIZE - margin))
    tile.save(build / "icon.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])

    if shutil.which("iconutil") is None:
        print("iconutil not found (it ships with macOS); icon.icns left unchanged",
              file=sys.stderr)
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            icon.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
            icon.resize((size * 2, size * 2), Image.LANCZOS).save(
                iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(build / "icon.icns")],
                       check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
