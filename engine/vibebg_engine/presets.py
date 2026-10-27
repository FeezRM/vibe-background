"""Background presets: a folder holding the media, a thumbnail, an optional depth map and a
manifest.json with defaults for framing and lighting."""

import re
import shutil
import unicodedata
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from vibebg_engine.background.source import IMAGE_EXTS, VIDEO_EXTS, cover_fit, is_video
from vibebg_engine.io.video import FrameReader, probe

THUMB_SIZE = (480, 270)


class SubjectDefaults(BaseModel):
    scale: float = Field(1.0, gt=0)
    x: float = 0.0
    y: float = 0.0


class LightDefaults(BaseModel):
    shadow: float = Field(0.0, ge=0, le=1)
    warmth: float = Field(0.0, ge=-1, le=1)


class PresetManifest(BaseModel):
    name: str
    type: Literal["image", "video"]
    file: str
    thumb: str
    depth: str | None = None
    seamless: bool = False  # the video already loops cleanly; skip the crossfade
    tags: list[str] = []
    subject: SubjectDefaults = SubjectDefaults()
    light: LightDefaults = LightDefaults()
    license: str
    source: str  # human-readable credit, e.g. "Jane Doe on Wikimedia Commons"
    url: str | None = None  # where the media came from


class Preset(BaseModel):
    id: str
    dir: Path
    builtin: bool
    manifest: PresetManifest

    @property
    def file(self) -> Path:
        return self.dir / self.manifest.file

    @property
    def thumb(self) -> Path:
        return self.dir / self.manifest.thumb

    @property
    def depth(self) -> Path | None:
        return self.dir / self.manifest.depth if self.manifest.depth else None


def load_preset(folder: Path, builtin: bool) -> Preset:
    manifest = PresetManifest.model_validate_json((folder / "manifest.json").read_text())
    for name in (manifest.file, manifest.thumb, manifest.depth):
        if name is not None and not (folder / name).is_file():
            raise ValueError(f"preset {folder.name}: missing {name}")
    return Preset(id=folder.name, dir=folder, builtin=builtin, manifest=manifest)


def list_presets(builtin_dir: Path | None, user_dir: Path | None) -> list[Preset]:
    presets = []
    for root, builtin in ((builtin_dir, True), (user_dir, False)):
        if root is None or not root.is_dir():
            continue
        for folder in sorted(p for p in root.iterdir() if (p / "manifest.json").is_file()):
            presets.append(load_preset(folder, builtin))
    return presets


def _slug(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-") or "background"


def _write_rgb(path: Path, rgb: np.ndarray):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"cannot write {path}")


def create_preset(src: Path, root: Path, name: str, *, license_name: str, source: str,
                  depth: bool = True, device=None) -> Preset:
    """Copy an image or video into a new preset folder under root, with a thumbnail, a depth
    map for still images (unless depth=False) and a manifest with neutral defaults."""
    src = Path(src)
    video = is_video(src)
    root.mkdir(parents=True, exist_ok=True)
    slug, n = _slug(name), 1
    folder = root / slug
    while folder.exists():
        n += 1
        folder = root / f"{slug}-{n}"
    folder.mkdir()
    try:
        ext = src.suffix.lower()
        media = f"bg{ext}"
        shutil.copyfile(src, folder / media)
        if video:
            info = probe(src)
            still = next(iter(FrameReader(info, start=min(1.0, info.duration / 2), max_frames=1)))
        else:
            bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"cannot read image {src}")
            still = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        _write_rgb(folder / "thumb.jpg", cover_fit(still, *THUMB_SIZE))

        depth_name = None
        if depth and not video:
            from vibebg_engine.background.depth import estimate_depth
            from vibebg_engine.device import pick_device

            near = estimate_depth(still, device or pick_device())
            depth_name = "depth.png"
            cv2.imwrite(str(folder / depth_name), (near * 255).round().astype(np.uint8))

        manifest = PresetManifest(name=name, type="video" if video else "image", file=media,
                                  thumb="thumb.jpg", depth=depth_name, license=license_name,
                                  source=source)
        (folder / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        return load_preset(folder, builtin=False)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise


SUPPORTED_EXTS = IMAGE_EXTS | VIDEO_EXTS
