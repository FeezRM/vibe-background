import json

import cv2
import numpy as np
import pytest

from vibebg_engine.presets import _slug, create_preset, list_presets, load_preset


def test_slug_transliterates_and_falls_back():
    assert _slug("Cozy Café") == "cozy-cafe"
    assert _slug("  My Desk!! ") == "my-desk"
    assert _slug("夜") == "background"


def test_create_preset_from_image_without_depth(tmp_path):
    src = tmp_path / "room.png"
    cv2.imwrite(str(src), np.full((300, 500, 3), 90, np.uint8))
    a = create_preset(src, tmp_path / "presets", "Room", license_name="CC0-1.0", source="x",
                      depth=False)
    b = create_preset(src, tmp_path / "presets", "Room", license_name="CC0-1.0", source="x",
                      depth=False)
    assert (a.id, b.id) == ("room", "room-2")
    assert a.manifest.type == "image" and a.depth is None
    thumb = cv2.imread(str(a.thumb))
    assert thumb.shape == (270, 480, 3)
    assert [p.id for p in list_presets(None, tmp_path / "presets")] == ["room", "room-2"]


def test_manifest_must_reference_existing_files(tmp_path):
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({
        "name": "Broken", "type": "image", "file": "bg.jpg", "thumb": "thumb.jpg",
        "license": "CC0-1.0", "source": "test"}))
    with pytest.raises(ValueError, match="missing bg.jpg"):
        load_preset(folder, builtin=True)
