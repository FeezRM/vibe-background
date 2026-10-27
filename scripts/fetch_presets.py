"""Download the built-in photo backgrounds from Wikimedia Commons and turn them into presets.

Every file is checked to still be CC0 / public domain before it is used. Images are capped at
2560px on the long edge to keep the app small.

    uv run --project engine python scripts/fetch_presets.py presets/backgrounds
"""

import argparse
import json
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2

from vibebg_engine.presets import create_preset

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VibeBackgroundPresetFetcher/0.1 (open-source desktop app)"
MAX_EDGE = 2560
ALLOWED_LICENSES = {"CC0", "Public domain"}

# (Commons file title, preset name, tags)
PHOTOS = [
    ("Abbotsford_House_Study_Room_(248456373).jpeg", "Old study", ["warm", "classic"]),
    ("Harewood_House_The_Main_Library_(218492215).jpeg", "Grand library", ["classic", "books"]),
    ("Vimala's_Curryblossom_Cafe_-_interior.jpg", "Cozy café", ["warm", "cafe"]),
    ("Rooster_Oulu_20150713_01.jpg", "Bright café", ["bright", "cafe"]),
    ("Plectranthus_verticillatus_in_a_ceramic_pot.jpg", "Minimal plant", ["clean", "minimal"]),
    ("Huhtasuo_library_interior.jpg", "Reading corner", ["bright", "books"]),
    ("Kortepohja_library_interior.jpg", "Modern library", ["bright", "modern"]),
    ("Thiranagama_Sri_Lanka_Riff_Hotel_5-star_resort_Suite_bedroom.jpg", "Ocean view",
     ["bright", "calm"]),
    ("Ossian_Sweet_House_interior_-_bedroom.jpg", "Sunny bedroom", ["warm", "home"]),
]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _info(title: str) -> dict:
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "imageinfo", "titles": f"File:{title}",
        "iiprop": "extmetadata|url|size", "iiurlwidth": MAX_EDGE,
    })
    pages = json.loads(_get(f"{API}?{query}"))["query"]["pages"]
    page = next(iter(pages.values()))
    if "imageinfo" not in page:
        raise ValueError(f"{title}: not found on Commons")
    return page["imageinfo"][0]


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


def fetch(title: str, name: str, tags: list[str], out: Path, tmp: Path):
    info = _info(title)
    meta = info["extmetadata"]
    license_name = meta.get("LicenseShortName", {}).get("value", "")
    if license_name not in ALLOWED_LICENSES:
        raise ValueError(f"{title}: license is {license_name!r}, not CC0/public domain")
    author = _strip_html(meta.get("Artist", {}).get("value", "")) or "unknown"

    # thumburl is the scaled copy when the original is larger than MAX_EDGE wide.
    raw = tmp / Path(title).name
    raw.write_bytes(_get(info.get("thumburl") or info["url"]))
    img = cv2.imread(str(raw), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"{title}: download is not an image")
    h, w = img.shape[:2]
    scale = MAX_EDGE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    src = tmp / f"{Path(title).stem}.jpg"
    cv2.imwrite(str(src), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    preset = create_preset(src, out, name, license_name="CC0-1.0",
                           source=f"{author} on Wikimedia Commons")
    manifest_path = preset.dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tags"] = tags
    manifest["url"] = info["descriptionurl"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{preset.id}: {w}x{h} -> {img.shape[1]}x{img.shape[0]}, {license_name}, {author}",
          file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        for title, name, tags in PHOTOS:
            fetch(title, name, tags, args.out, Path(tmp))
            time.sleep(0.5)  # be polite to the Commons API
    return 0


if __name__ == "__main__":
    sys.exit(main())
