"""Local HTTP API used by the desktop app.

Binds to 127.0.0.1 on a free port and requires a random per-launch token on every request
(header `x-vibebg-token`, or `?token=` for <img>/<video> sources). On startup it prints one
line, `VIBEBG_READY {"port": ..., "token": ...}`, which the app reads to connect.
"""

import argparse
import json
import secrets
import shutil
import socket
import sys
import tempfile
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from vibebg_engine import __version__
from vibebg_engine.background.source import VideoBackground
from vibebg_engine.device import pick_device
from vibebg_engine.io.video import FrameReader, probe
from vibebg_engine.jobs import JobManager
from vibebg_engine.pipeline import (
    BackgroundSettings,
    FrameProcessor,
    LightingSettings,
    RenderSettings,
    decode_filters,
)
from vibebg_engine.placement import Placement
from vibebg_engine.presets import SUPPORTED_EXTS, Preset, create_preset, list_presets


class BackgroundIn(BaseModel):
    mode: Literal["preset", "file", "blur_original"]
    preset: str | None = None
    path: str | None = None
    dof: float = Field(0.0, ge=0, le=1)
    blur: float = Field(0.0, ge=0, le=1)


class PlacementIn(BaseModel):
    scale: float = Field(1.0, ge=0.2, le=3)
    x: float = Field(0.0, ge=-1, le=1)
    y: float = Field(0.0, ge=-1, le=1)


class LightingIn(BaseModel):
    mode: Literal["off", "enhance", "match", "relight"] = "off"
    strength: float = Field(0.6, ge=0, le=1)
    wrap: float = Field(0.5, ge=0, le=1)
    shadow: float = Field(0.0, ge=0, le=1)
    warmth: float = Field(0.0, ge=-1, le=1)
    denoise: bool = False


class SettingsIn(BaseModel):
    background: BackgroundIn
    tier: Literal["fast", "balanced", "max"] = "fast"
    codec: Literal["h264", "hevc"] = "h264"
    placement: PlacementIn = PlacementIn()
    lighting: LightingIn = LightingIn()


class PathIn(BaseModel):
    path: str


class FrameIn(BaseModel):
    path: str
    time: float = Field(0.0, ge=0)


class PreviewIn(FrameIn):
    settings: SettingsIn


class JobIn(BaseModel):
    path: str
    out: str | None = None  # None: a temporary preview clip
    settings: SettingsIn
    start: float = Field(0.0, ge=0)
    duration: float | None = Field(None, gt=0)


class PresetIn(BaseModel):
    path: str
    name: str = Field(min_length=1, max_length=80)


def _existing_file(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_file():
        raise HTTPException(404, f"file not found: {path}")
    return p


def _jpeg(rgb: np.ndarray, quality: int = 85) -> Response:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, "could not encode preview")
    return Response(buf.tobytes(), media_type="image/jpeg")


class PreviewSession:
    """Keeps the last preview's processor alive so dragging a slider doesn't reload models.

    A new processor is only built when the clip, tier or background changes.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._key = None
        self._processor: FrameProcessor | None = None
        self._relight_key = None  # what the processor's current light map was computed for

    def render(self, src: Path, t: float, settings: RenderSettings) -> np.ndarray:
        bgs = settings.background
        key = (str(src), settings.tier, bgs.path, bgs.dof, bgs.blur, bgs.depth_path, bgs.seamless,
               settings.lighting.denoise)
        with self._lock:
            info = probe(src)
            if key != self._key or self._processor is None:
                if self._processor is not None:
                    self._processor.close()
                self._processor = FrameProcessor(info, settings, pick_device(), start=t)
                self._key = key
                self._relight_key = None
            else:
                self._processor.configure_lighting(settings)
                if isinstance(self._processor.background, VideoBackground):
                    self._processor.background.seek(t)
            frame = _read_frame(info, t, decode_filters(settings))
            if settings.lighting.mode == "relight":
                # IC-Light takes many seconds; only rerun it when the frame or framing changes.
                p = settings.placement
                relight_key = (key, t, p.scale, p.x, p.y)
                if relight_key != self._relight_key:
                    self._processor.prepare_relight([(0, t, frame)])
                    self._relight_key = relight_key
            self._processor.reset()
            self._processor.frame_index = 0
            return self._processor.process(frame)


def _read_frame(info, t: float, filters: tuple[str, ...] = ()) -> np.ndarray:
    t = min(t, max(0.0, info.duration - 1.0 / float(info.fps)))
    frame = next(iter(FrameReader(info, start=t, max_frames=1, filters=filters)), None)
    if frame is None:
        raise HTTPException(422, f"no frame at {t:.2f}s")
    return frame


def create_app(token: str, builtin_presets: Path | None, user_presets: Path,
               on_ready: Callable[[], None] | None = None) -> FastAPI:
    preview_dir = Path(tempfile.mkdtemp(prefix="vibebg-preview-"))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if on_ready:
            on_ready()
        yield
        shutil.rmtree(preview_dir, ignore_errors=True)

    app = FastAPI(title="Vibe Background engine", version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    jobs = JobManager()
    previews = PreviewSession()
    servable: set[Path] = set()  # files the UI may stream through /media

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if request.method != "OPTIONS":
            given = request.headers.get("x-vibebg-token") or request.query_params.get("token")
            if not given or not secrets.compare_digest(given, token):
                return JSONResponse({"detail": "bad token"}, status_code=401)
        return await call_next(request)

    def presets() -> dict[str, Preset]:
        return {p.id: p for p in list_presets(builtin_presets, user_presets)}

    def to_settings(s: SettingsIn) -> RenderSettings:
        b = s.background
        if b.mode == "preset":
            preset = presets().get(b.preset or "")
            if preset is None:
                raise HTTPException(404, f"unknown preset {b.preset!r}")
            bg = BackgroundSettings(path=preset.file, dof=b.dof, blur=b.blur,
                                    depth_path=preset.depth, seamless=preset.manifest.seamless)
        elif b.mode == "file":
            bg = BackgroundSettings(path=_existing_file(b.path or ""), dof=b.dof, blur=b.blur)
        else:
            bg = BackgroundSettings(path=None, blur=b.blur)
        settings = RenderSettings(
            background=bg, tier=s.tier, codec=s.codec,
            placement=Placement(**s.placement.model_dump()),
            lighting=LightingSettings(**s.lighting.model_dump()),
        )
        try:
            settings.validate()
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return settings

    @app.get("/health")
    def health():
        return {"ok": True, "version": __version__, "device": str(pick_device())}

    @app.post("/probe")
    def probe_video(body: PathIn):
        src = _existing_file(body.path)
        try:
            info = probe(src)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"width": info.width, "height": info.height, "fps": float(info.fps),
                "duration": info.duration, "frames": info.frame_count,
                "has_audio": info.audio_codec is not None, "is_hdr": info.is_hdr}

    @app.post("/frame")
    def original_frame(body: FrameIn):
        return _jpeg(_read_frame(probe(_existing_file(body.path)), body.time))

    @app.post("/preview")
    def preview(body: PreviewIn):
        src = _existing_file(body.path)
        settings = to_settings(body.settings)
        return _jpeg(previews.render(src, body.time, settings))

    @app.post("/jobs")
    def submit_job(body: JobIn):
        src = _existing_file(body.path)
        settings = to_settings(body.settings)
        if body.out is None:
            out = preview_dir / f"preview-{secrets.token_hex(4)}.mp4"
        else:
            out = Path(body.out).expanduser()
            if out.suffix.lower() != ".mp4":
                raise HTTPException(422, "output must be an .mp4 file")
            if not out.parent.is_dir():
                raise HTTPException(422, f"folder does not exist: {out.parent}")
            if out.resolve() == src.resolve():
                raise HTTPException(422, "output would overwrite the input video")
        servable.add(out.resolve())
        return jobs.submit(src, out, settings, start=body.start, duration=body.duration).public()

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.public()

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = jobs.cancel(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.public()

    @app.get("/media")
    def media(path: str):
        p = Path(path).resolve()
        if p not in servable or not p.is_file():
            raise HTTPException(404, "not available")
        return FileResponse(p)

    @app.get("/presets")
    def get_presets():
        return [{"id": p.id, "builtin": p.builtin, **p.manifest.model_dump(
                    include={"name", "type", "tags", "subject", "light", "license", "source", "url"})}
                for p in presets().values()]

    @app.get("/presets/{preset_id}/thumb")
    def preset_thumb(preset_id: str):
        preset = presets().get(preset_id)
        if preset is None:
            raise HTTPException(404, "unknown preset")
        return FileResponse(preset.thumb)

    @app.post("/presets")
    def add_preset(body: PresetIn):
        src = _existing_file(body.path)
        if src.suffix.lower() not in SUPPORTED_EXTS:
            raise HTTPException(422, f"unsupported file type {src.suffix}")
        try:
            preset = create_preset(src, user_presets, body.name, license_name="user-provided",
                                   source=src.name)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"id": preset.id}

    @app.delete("/presets/{preset_id}")
    def delete_preset(preset_id: str):
        preset = presets().get(preset_id)
        if preset is None:
            raise HTTPException(404, "unknown preset")
        if preset.builtin:
            raise HTTPException(403, "built-in presets can't be deleted")
        shutil.rmtree(preset.dir)
        return {"ok": True}

    return app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(prog="vibebg-server")
    parser.add_argument("--presets", type=Path, help="built-in presets folder")
    parser.add_argument("--user-presets", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)

    token = secrets.token_urlsafe(24)
    port = args.port or _free_port()
    def announce():
        print("VIBEBG_READY " + json.dumps({"port": port, "token": token}), flush=True)

    app = create_app(token, args.presets, args.user_presets, on_ready=announce)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
