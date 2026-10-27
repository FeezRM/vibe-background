import json
import subprocess
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vibebg_engine.io.video import ffmpeg_bin
from vibebg_engine.server import create_app

TOKEN = "test-token"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("server")
    clip = root / "clip.mp4"
    subprocess.run([ffmpeg_bin(), "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(clip)],
                   check=True)
    image = root / "room.png"
    cv2.imwrite(str(image), np.full((360, 640, 3), (90, 140, 200), np.uint8))

    builtin = root / "builtin" / "plain-room"
    builtin.mkdir(parents=True)
    cv2.imwrite(str(builtin / "bg.png"), np.full((360, 640, 3), 120, np.uint8))
    cv2.imwrite(str(builtin / "thumb.jpg"), np.full((90, 160, 3), 120, np.uint8))
    (builtin / "manifest.json").write_text(json.dumps({
        "name": "Plain room", "type": "image", "file": "bg.png", "thumb": "thumb.jpg",
        "license": "CC0-1.0", "source": "test"}))

    app = create_app(TOKEN, root / "builtin", root / "user")
    client = TestClient(app, headers={"x-vibebg-token": TOKEN})
    return {"client": client, "clip": clip, "image": image, "root": root}


def _settings(**background):
    return {"background": background or {"mode": "preset", "preset": "plain-room"}}


def test_token_is_required(env):
    anon = TestClient(env["client"].app)
    assert anon.get("/health").status_code == 401
    assert anon.get("/health", headers={"x-vibebg-token": "wrong"}).status_code == 401
    assert anon.get(f"/health?token={TOKEN}").status_code == 200


def test_probe(env):
    r = env["client"].post("/probe", json={"path": str(env["clip"])})
    assert r.status_code == 200
    body = r.json()
    assert (body["width"], body["height"], body["fps"]) == (320, 180, 25.0)
    assert body["has_audio"] is True
    assert env["client"].post("/probe", json={"path": "/nope.mp4"}).status_code == 404


def test_original_frame_and_preview_are_jpegs_at_clip_size(env):
    c = env["client"]
    r = c.post("/frame", json={"path": str(env["clip"]), "time": 1.0})
    assert r.headers["content-type"] == "image/jpeg"
    for settings in (_settings(), _settings(mode="file", path=str(env["image"]), dof=0.3),
                     _settings(mode="blur_original", blur=0.4)):
        r = c.post("/preview", json={"path": str(env["clip"]), "time": 0.5, "settings": settings})
        assert r.status_code == 200, r.text
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (180, 320, 3)


def test_settings_are_validated(env):
    c = env["client"]
    bad = _settings(mode="blur_original")
    bad["placement"] = {"scale": 0.8}
    r = c.post("/preview", json={"path": str(env["clip"]), "settings": bad})
    assert r.status_code == 422
    r = c.post("/preview", json={"path": str(env["clip"]),
                                 "settings": _settings(mode="preset", preset="missing")})
    assert r.status_code == 404
    r = c.post("/jobs", json={"path": str(env["clip"]), "out": str(env["root"] / "x.mov"),
                              "settings": _settings()})
    assert r.status_code == 422


def _wait(client, job_id, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_export_job_and_media(env):
    c = env["client"]
    out = env["root"] / "export.mp4"
    job = c.post("/jobs", json={"path": str(env["clip"]), "out": str(out),
                                "settings": _settings()}).json()
    job = _wait(c, job["id"])
    assert job["status"] == "done", job["error"]
    assert job["frame"] == job["total"] == 50
    r = c.get("/media", params={"path": str(out)})
    assert r.status_code == 200 and len(r.content) == out.stat().st_size
    # Arbitrary files can't be read through /media.
    assert c.get("/media", params={"path": str(env["image"])}).status_code == 404


def test_preview_clip_job_renders_a_section(env):
    c = env["client"]
    job = c.post("/jobs", json={"path": str(env["clip"]), "settings": _settings(),
                                "start": 0.5, "duration": 1.0}).json()
    job = _wait(c, job["id"])
    assert job["status"] == "done", job["error"]
    assert job["total"] == 25


def test_cancel_job(env):
    c = env["client"]
    job = c.post("/jobs", json={"path": str(env["clip"]), "out": str(env["root"] / "c.mp4"),
                                "settings": _settings()}).json()
    c.post(f"/jobs/{job['id']}/cancel")
    job = _wait(c, job["id"])
    assert job["status"] in ("cancelled", "done")  # it may finish before the cancel lands
    if job["status"] == "cancelled":
        assert not (env["root"] / "c.mp4").exists()


def test_preset_lifecycle(env):
    c = env["client"]
    ids = [p["id"] for p in c.get("/presets").json()]
    assert ids == ["plain-room"]
    assert c.delete("/presets/plain-room").status_code == 403

    r = c.post("/presets", json={"path": str(env["image"]), "name": "My Desk!"})
    assert r.status_code == 200, r.text
    new_id = r.json()["id"]
    assert new_id == "my-desk"
    listed = {p["id"]: p for p in c.get("/presets").json()}
    assert listed[new_id]["type"] == "image" and listed[new_id]["builtin"] is False
    assert c.get(f"/presets/{new_id}/thumb").headers["content-type"] == "image/jpeg"
    assert (env["root"] / "user" / new_id / "depth.png").is_file()

    assert c.delete(f"/presets/{new_id}").status_code == 200
    assert [p["id"] for p in c.get("/presets").json()] == ["plain-room"]
