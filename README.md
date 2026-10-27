# Vibe Background

Drop in a video of yourself. Vibe Background cuts you out, puts you in a nicer room, and can
re-light you to match it. It runs entirely on your computer, and nothing is uploaded.

- **Cutout in three tiers:** Fast, Balanced, and Max. Max re-traces hair strand by strand.
- **Backgrounds:** built-in photo and animated-loop presets, your own image or video, or your
  real room blurred.
- **Depth of field:** the background blurs more with distance, like a real lens.
- **Lighting:**
  - **Enhance:** fixes exposure and color on you.
  - **Match:** pulls your color and light toward the background, with light wrap, shadow and
    grain matching.
  - **AI Relight (experimental):** uses IC-Light to re-light you for the scene.
- **Framing:** drag and resize yourself in the frame.
- **Export:** MP4 (H.264 or HEVC) at the original resolution and frame rate, with the original
  audio.

Supported on Apple Silicon Macs and Windows. On Windows, the GPU is used with NVIDIA cards;
other machines run on the CPU, which is slow beyond the Fast tier.

## Speed

Measured on an M2 Pro MacBook Pro, 1080p clip:

| Setting | Frames per second |
|---|---|
| Fast, lighting off | 21 |
| Fast, Match lighting | 10 |
| Balanced, Match lighting (1080×1920) | 6 |
| Max | 2.7 |

AI Relight adds a keyframe pass before export: one IC-Light run every 2 seconds of video.

## Development

Requirements: [uv](https://docs.astral.sh/uv/), Node 22+, and ffmpeg on your PATH.

```bash
cd engine && uv sync
```

```bash
cd apps/desktop && npm install
```

```bash
cd apps/desktop && npm run dev
```

The dev app starts the engine with `uv run`. Models download on first use to
`~/Library/Application Support/VibeBackground/models` on a Mac, or
`%APPDATA%\VibeBackground\models` on Windows.

### Engine CLI

```bash
cd engine && uv run vibebg render input.mp4 --bg ../presets/backgrounds/old-study/bg.jpg --dof 0.4 --light match -o out.mp4
```

Run `uv run vibebg render --help` for every option, including tiers, framing, loops, and
blur-my-room.

### Tests

```bash
cd engine && uv run pytest
```

The end-to-end tests render clips placed in `engine/tests/fixtures/`. That folder is
gitignored, so use any short talking-head MP4s.

### Presets

Each preset is a folder in `presets/backgrounds/` with a `manifest.json`, the media, a
thumbnail and an optional depth map. To rebuild them:

- **Photos:** `scripts/fetch_presets.py` downloads the CC0 photos from Wikimedia Commons and
  checks each license.
- **Loops:** `scripts/make_loops.py` renders the animated loops.

To add your own preset, use the **+ Your own** tile in the app.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE). Model and media licenses are listed in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
