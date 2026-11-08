#!/usr/bin/env bash
# Installs Vibe Background on an Apple Silicon Mac. Run it from a clone of the repo, or:
#   curl -fsSL https://raw.githubusercontent.com/FeezRM/vibe-background/main/install.sh | bash
# Running it again updates an existing install.
set -euo pipefail

REPO_URL="https://github.com/FeezRM/vibe-background.git"
APP_NAME="Vibe Background"
BUNDLE_ID="app.vibebackground"
# Must match app.getPath('userData') in the packaged app (its productName).
RUNTIME="$HOME/Library/Application Support/$APP_NAME/engine"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31mError: %s\033[0m\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == Darwin ]] || die "this installer is for macOS. On Windows, use install.ps1."
[[ "$(uname -m)" == arm64 ]] || die "Vibe Background needs an Apple Silicon Mac."
xcode-select -p >/dev/null 2>&1 \
  || die "the Xcode command line tools are missing. Run 'xcode-select --install', then run this again."

ensure_brew() {
  if ! command -v brew >/dev/null 2>&1; then
    if [[ -x /opt/homebrew/bin/brew ]]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
    else
      say "Installing Homebrew (it will ask for your password)"
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" </dev/tty
      eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
  fi
}

node_major() { node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0; }

# --- Source -------------------------------------------------------------------------------------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
if [[ -f "$script_dir/engine/pyproject.toml" && -d "$script_dir/apps/desktop" ]]; then
  SRC="$script_dir"
else
  SRC="$(mktemp -d)/vibe-background"
  trap 'rm -rf "$(dirname "$SRC")"' EXIT
  say "Downloading Vibe Background"
  git clone --depth 1 "$REPO_URL" "$SRC"
fi

# --- Prerequisites ------------------------------------------------------------------------------
if (( $(node_major) < 22 )); then
  ensure_brew
  say "Installing Node.js"
  brew install node
  export PATH="$(brew --prefix)/bin:$PATH"
  (( $(node_major) >= 22 )) || die "Node.js 22 or newer is needed, found $(node --version 2>/dev/null)."
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  ensure_brew
  say "Installing ffmpeg"
  brew install ffmpeg
fi

if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- Engine -------------------------------------------------------------------------------------
say "Installing the video engine (about 1 GB the first time)"
mkdir -p "$RUNTIME"
UV_PROJECT_ENVIRONMENT="$RUNTIME/.venv" uv sync --directory "$SRC/engine" \
  --frozen --no-dev --no-editable --python-preference only-managed \
  --reinstall-package vibebg-engine  # uv won't notice source edits to a non-editable install

mkdir -p "$RUNTIME/ffmpeg"
ln -sf "$(command -v ffmpeg)" "$RUNTIME/ffmpeg/ffmpeg"
ln -sf "$(command -v ffprobe)" "$RUNTIME/ffmpeg/ffprobe"

# --- App ----------------------------------------------------------------------------------------
say "Building the app"
cd "$SRC/apps/desktop"
npm ci --no-audit --no-fund
npx electron-vite build
# Ad-hoc signed for this Mac only. Hardened runtime is off because its library validation
# rejects ad-hoc signed frameworks; it only matters for notarized builds.
npx electron-builder --mac --dir --publish never -c.mac.identity=- -c.mac.hardenedRuntime=false

built="$SRC/apps/desktop/dist/mac-arm64/$APP_NAME.app"
[[ -d "$built" ]] || die "the build did not produce $built"

dest_dir=/Applications
[[ -w "$dest_dir" ]] || { dest_dir="$HOME/Applications"; mkdir -p "$dest_dir"; }
dest="$dest_dir/$APP_NAME.app"

say "Installing to $dest"
if [[ -e "$dest" ]]; then
  existing_id="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$dest/Contents/Info.plist" 2>/dev/null || true)"
  [[ "$existing_id" == "$BUNDLE_ID" ]] \
    || die "$dest already exists and is a different app ($existing_id). Move it away and run this again."
  osascript -e "quit app id \"$BUNDLE_ID\"" >/dev/null 2>&1 || true
  rm -rf "$dest"
fi
ditto "$built" "$dest"

say "Done. Open $APP_NAME from $dest_dir."
echo "The cutout models (a few GB) download the first time you use each quality tier."
