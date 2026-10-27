"""Pinned model weights: where they come from, what they hash to, and their license."""

import hashlib
import os
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_RVM = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    filename: str
    url: str
    sha256: str
    size: int
    license: str


MODELS: dict[str, ModelSpec] = {
    m.id: m
    for m in [
        ModelSpec(
            id="rvm_mobilenetv3",
            filename="rvm_mobilenetv3_fp32.torchscript",
            url=f"{_RVM}/rvm_mobilenetv3_fp32.torchscript",
            sha256="f01e0c9338b9a6a31b881ea6d4360d70c1e549701b3792e14c9ed88d6196c5a1",
            size=15_501_891,
            license="GPL-3.0",
        ),
        ModelSpec(
            id="rvm_resnet50",
            filename="rvm_resnet50_fp32.torchscript",
            url=f"{_RVM}/rvm_resnet50_fp32.torchscript",
            sha256="072adec3c75a1af773ec35d8b595612f8e3472b7a4b8a210684432b493376852",
            size=108_063_684,
            license="GPL-3.0",
        ),
    ]
}


@dataclass(frozen=True)
class HFModelSpec:
    id: str
    repo: str
    revision: str  # pinned commit
    license: str
    # Files to fetch. Repos often carry the same weights in several formats; take only what we load.
    files: tuple[str, ...] = ("*.json", "*.safetensors")


HF_MODELS: dict[str, HFModelSpec] = {
    m.id: m
    for m in [
        HFModelSpec(
            id="depth_anything_v2_small",
            repo="depth-anything/Depth-Anything-V2-Small-hf",
            revision="5426e4f0f36572d16453bbda7a8389317b1bef99",
            # Only the Small variant is Apache-2.0; Base/Large/Giant are non-commercial.
            license="Apache-2.0",
        ),
        HFModelSpec(
            id="realistic_vision_v51",
            repo="stablediffusionapi/realistic-vision-v51",
            revision="19e3643d7d963c156d01537188ec08f0b79a514a",
            license="CreativeML-OpenRAIL-M",
            files=("model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
                   "text_encoder/model.safetensors", "unet/config.json",
                   "unet/diffusion_pytorch_model.safetensors", "vae/config.json",
                   "vae/diffusion_pytorch_model.safetensors"),
        ),
        HFModelSpec(
            id="iclight_fbc",
            repo="lllyasviel/ic-light",
            revision="9cad1878695f546a7fb9eaca14e2a89131ba5ffe",
            license="CreativeML-OpenRAIL-M",
            files=("iclight_sd15_fbc.safetensors",),
        ),
        HFModelSpec(
            id="vitmatte_small_d646",
            repo="hustvl/vitmatte-small-distinctions-646",
            revision="6a0e75d7214b01f4d1163ede0f15b23afbbd480b",
            license="Apache-2.0",
        ),
    ]
}


def models_dir() -> Path:
    override = os.environ.get("VIBEBG_MODELS_DIR")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ["APPDATA"])
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "VibeBackground" / "models"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(
    model_id: str, on_progress: Callable[[int, int], None] | None = None
) -> Path:
    """Return the local path of a model, downloading and verifying it if missing."""
    spec = MODELS[model_id]
    path = models_dir() / spec.filename
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(spec.url) as resp, tmp.open("wb") as out:
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if on_progress:
                on_progress(done, spec.size)

    actual = _sha256(tmp)
    if actual != spec.sha256:
        tmp.unlink()
        raise RuntimeError(f"{model_id}: checksum mismatch (got {actual}, want {spec.sha256})")
    tmp.replace(path)
    return path


def ensure_hf_model(model_id: str) -> Path:
    """Return the local snapshot directory of a pinned Hugging Face model, downloading if needed."""
    from huggingface_hub import snapshot_download

    spec = HF_MODELS[model_id]
    return Path(snapshot_download(spec.repo, revision=spec.revision, cache_dir=models_dir() / "hf",
                                  allow_patterns=list(spec.files)))
