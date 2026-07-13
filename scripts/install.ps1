# LocalCode installer (Windows PowerShell)
# Usage:
#   irm https://raw.githubusercontent.com/view321/LocalCode/main/scripts/install.ps1 | iex
#   .\install.ps1
# Optional env:
#   $env:LOCALCODE_INSTALL_DIR, $env:LOCALCODE_BIN_DIR, $env:LOCALCODE_REPO_URL, $env:LOCALCODE_BRANCH

$ErrorActionPreference = "Stop"

$RepoUrl = if ($env:LOCALCODE_REPO_URL) { $env:LOCALCODE_REPO_URL } else { "https://github.com/view321/LocalCode.git" }
$Branch = if ($env:LOCALCODE_BRANCH) { $env:LOCALCODE_BRANCH } else { "main" }
$InstallDir = if ($env:LOCALCODE_INSTALL_DIR) { $env:LOCALCODE_INSTALL_DIR } else { Join-Path $env:USERPROFILE ".local\share\localcode" }
$BinDir = if ($env:LOCALCODE_BIN_DIR) { $env:LOCALCODE_BIN_DIR } else { Join-Path $env:USERPROFILE ".local\bin" }
$BinaryName = "localcode.exe"

function Write-Info([string]$Message) {
    Write-Host "==> " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Warn([string]$Message) {
    Write-Host "warn " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Assert-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $Name"
    }
}

function Ensure-Rust {
    if ((Get-Command cargo -ErrorAction SilentlyContinue) -and (Get-Command rustc -ErrorAction SilentlyContinue)) {
        $ver = (& rustc --version)
        Write-Info "Found Rust $ver"
        return
    }

    Write-Info "Rust not found — installing via rustup (default toolchain)"
    $rustup = Join-Path $env:TEMP "rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustup
    & $rustup -y --default-toolchain stable
    Remove-Item $rustup -ErrorAction SilentlyContinue

    $cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
    if (Test-Path $cargoBin) {
        $env:Path = "$cargoBin;$env:Path"
    }

    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw "cargo still not on PATH after rustup install. Open a new terminal and re-run."
    }

    Write-Info "Installed $((& rustc --version))"
}

function Clone-OrUpdate {
    Assert-Command git

    $parent = Split-Path -Parent $InstallDir
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    $gitDir = Join-Path $InstallDir ".git"
    if (Test-Path $gitDir) {
        Write-Info "Updating existing checkout at $InstallDir"
        Push-Location $InstallDir
        try {
            git fetch --depth 1 origin $Branch
            git checkout -B $Branch "origin/$Branch"
        }
        finally {
            Pop-Location
        }
    }
    else {
        # Never delete a non-empty directory we don't recognize — a mistyped
        # LOCALCODE_INSTALL_DIR must not wipe user data.
        if ((Test-Path $InstallDir) -and (Get-ChildItem -Force $InstallDir | Select-Object -First 1)) {
            throw "$InstallDir exists, is not empty, and is not a LocalCode checkout. Move it or set LOCALCODE_INSTALL_DIR elsewhere."
        }
        Write-Info "Cloning LocalCode into $InstallDir"
        if (Test-Path $InstallDir) {
            Remove-Item -Recurse -Force $InstallDir
        }
        git clone --depth 1 --branch $Branch $RepoUrl $InstallDir
    }
}

function Build-AndInstall {
    Assert-Command cargo

    if (-not (Test-Path $BinDir)) {
        New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    }

    Write-Info "Building localcode (release) — this may take a few minutes"
    Push-Location $InstallDir
    try {
        cargo build --release -p localcode-cli
    }
    finally {
        Pop-Location
    }

    $built = Join-Path $InstallDir "target\release\$BinaryName"
    if (-not (Test-Path $built)) {
        throw "Build succeeded but binary not found at $built"
    }

    $dest = Join-Path $BinDir $BinaryName
    Copy-Item -Path $built -Destination $dest -Force
    Write-Info "Installed $dest"
}

function Ensure-Path {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $userPath) { $userPath = "" }

    $parts = $userPath -split ";" | Where-Object { $_ -ne "" }
    if ($parts -contains $BinDir) {
        Write-Info "$BinDir is already on the user PATH"
        return
    }

    Write-Warn "$BinDir is not on your user PATH — adding it"
    $newPath = if ($userPath.TrimEnd(";")) { "$userPath;$BinDir" } else { $BinDir }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    $env:Path = "$BinDir;$env:Path"
    Write-Info "Added $BinDir to user PATH (new terminals will pick this up)"
}

function Verify-Install {
    $bin = Join-Path $BinDir $BinaryName
    if (-not (Test-Path $bin)) {
        throw "Install failed: $bin not found"
    }

    Write-Info "Done. Run: localcode"
    try {
        & $bin --help 2>$null | Select-Object -First 5
    }
    catch {
        # help may exit non-zero depending on clap config
    }
}

Write-Info "LocalCode installer"
Ensure-Rust
Clone-OrUpdate
Build-AndInstall
Ensure-Path
Verify-Install
