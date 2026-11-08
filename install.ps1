# Installs Vibe Background on Windows. Run it from a clone of the repo, or in PowerShell:
#   irm https://raw.githubusercontent.com/FeezRM/vibe-background/main/install.ps1 | iex
# Running it again updates an existing install.
$ErrorActionPreference = 'Stop'

$RepoUrl = 'https://github.com/FeezRM/vibe-background.git'
$AppName = 'Vibe Background'
$ExeName = 'vibe-background'
# Must match app.getPath('userData') in the packaged app (its productName).
$Runtime = Join-Path $env:APPDATA "$AppName\engine"
$InstallDir = Join-Path $env:LOCALAPPDATA "Programs\$AppName"

function Say($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
# throw rather than exit: under `irm | iex`, exit would close the user's PowerShell window.
function Die($msg) { throw "Error: $msg" }

# Native commands don't throw on failure, so check each exit code.
function Run {
    param([Parameter(Mandatory)][string]$Exe, [Parameter(ValueFromRemainingArguments)][string[]]$Rest)
    & $Exe @Rest
    if ($LASTEXITCODE -ne 0) { Die "'$Exe $($Rest -join ' ')' failed (exit code $LASTEXITCODE)." }
}

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'User')
}

function Has($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function Winget-Install($id) {
    if (-not (Has 'winget')) {
        Die "winget is missing. Install 'App Installer' from the Microsoft Store, then run this again."
    }
    Say "Installing $id"
    Run winget install --id $id --exact --silent --accept-package-agreements --accept-source-agreements
    Refresh-Path
}

function Node-Major {
    if (-not (Has 'node')) { return 0 }
    [int](node -p 'process.versions.node.split(".")[0]')
}

# Resolves winget's shim links to the real executable so it can be copied.
function Resolve-Exe($cmd) {
    $item = Get-Item (Get-Command $cmd).Source
    if ($item.LinkType) { return ($item.Target | Select-Object -First 1) }
    $item.FullName
}

if (-not [Environment]::Is64BitOperatingSystem) { Die 'Vibe Background needs 64-bit Windows.' }

# --- Source -------------------------------------------------------------------------------------
$Temp = $null
if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot 'engine\pyproject.toml'))) {
    $Src = $PSScriptRoot
} else {
    if (-not (Has 'git')) { Winget-Install 'Git.Git' }
    $Temp = Join-Path ([IO.Path]::GetTempPath()) ("vibebg-" + [guid]::NewGuid())
    $Src = Join-Path $Temp 'vibe-background'
    Say 'Downloading Vibe Background'
    Run git clone --depth 1 $RepoUrl $Src
}

try {
    # --- Prerequisites --------------------------------------------------------------------------
    if ((Node-Major) -lt 22) {
        Winget-Install 'OpenJS.NodeJS.LTS'
        if ((Node-Major) -lt 22) { Die 'Node.js 22 or newer is needed. Restart PowerShell and run this again.' }
    }
    if (-not ((Has 'ffmpeg') -and (Has 'ffprobe'))) { Winget-Install 'Gyan.FFmpeg' }
    if (-not (Has 'uv')) { Winget-Install 'astral-sh.uv' }
    foreach ($cmd in 'ffmpeg', 'ffprobe', 'uv') {
        if (-not (Has $cmd)) { Die "$cmd was installed but isn't on PATH yet. Restart PowerShell and run this again." }
    }

    # --- Engine ---------------------------------------------------------------------------------
    Say 'Installing the video engine (a few GB the first time)'
    New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $Runtime '.venv'
    Run uv sync --directory (Join-Path $Src 'engine') --frozen --no-dev --no-editable --python-preference only-managed `
        --reinstall-package vibebg-engine  # uv won't notice source edits to a non-editable install
    Remove-Item Env:UV_PROJECT_ENVIRONMENT

    $ffmpegDir = Join-Path $Runtime 'ffmpeg'
    New-Item -ItemType Directory -Force -Path $ffmpegDir | Out-Null
    foreach ($cmd in 'ffmpeg', 'ffprobe') {
        Copy-Item (Resolve-Exe $cmd) (Join-Path $ffmpegDir "$cmd.exe") -Force
    }

    # --- App ------------------------------------------------------------------------------------
    Say 'Building the app'
    Push-Location (Join-Path $Src 'apps\desktop')
    try {
        Run npm ci --no-audit --no-fund
        Run npx electron-vite build
        Run npx electron-builder --win --dir --publish never
    } finally { Pop-Location }

    $built = Join-Path $Src 'apps\desktop\dist\win-unpacked'
    if (-not (Test-Path (Join-Path $built "$ExeName.exe"))) { Die "the build did not produce $built\$ExeName.exe" }

    Say "Installing to $InstallDir"
    Get-Process -Name $ExeName -ErrorAction SilentlyContinue | Stop-Process -Force
    if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir }
    New-Item -ItemType Directory -Force -Path (Split-Path $InstallDir) | Out-Null
    Copy-Item -Recurse $built $InstallDir

    $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut(
        (Join-Path ([Environment]::GetFolderPath('Programs')) "$AppName.lnk"))
    $shortcut.TargetPath = Join-Path $InstallDir "$ExeName.exe"
    $shortcut.WorkingDirectory = $InstallDir
    $shortcut.Save()

    Say "Done. Open $AppName from the Start menu."
    Write-Host 'The cutout models (a few GB) download the first time you use each quality tier.'
} finally {
    if ($Temp) { Remove-Item -Recurse -Force $Temp -ErrorAction SilentlyContinue }
}
