//! Backend installer.
//!
//! Resolves a platform-appropriate install plan per backend — including
//! prerequisites like Python — and runs it, streaming each output line back as
//! progress. Plans are **multi-step** so a Python-dependent backend can install
//! Python first.
//!
//! Honesty (see the architecture conventions): where an install can't run
//! non-interactively — e.g. a Linux system package manager needs `sudo`, which
//! would hang a stdin-null child waiting for a password — the plan is `Manual`
//! with copy-paste steps, never a fabricated success.

use crate::diagnose::RepairIntent;
use crate::BackendKind;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::paths::AppPaths;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc::UnboundedSender;

/// One step of an install plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallStep {
    /// A concrete command to spawn.
    Command {
        program: String,
        args: Vec<String>,
        /// Human-readable form shown in the confirm dialog / progress.
        display: String,
    },
    /// `<python> -m pip install <package>`. The interpreter is resolved at *run
    /// time* (not plan time) so a Python installed by an earlier step is found
    /// even though this process's PATH was captured before that install.
    PipInstall { package: String },
    /// A command that must run with elevated privileges. On Unix this runs as
    /// `sudo [-n|-S] <program> <args>`; it is **never emitted on Windows**
    /// (repairs there don't need elevation), so the non-elevated runner never
    /// sees one. `display` is the `sudo …` form shown for approval — the
    /// password (when one is needed) goes to stdin, never into `display`.
    Sudo {
        program: String,
        args: Vec<String>,
        display: String,
    },
    /// Download a prebuilt **PrismML** llama.cpp release (Bonsai / Q1_0 kernels)
    /// for this OS and extract it into `dest_dir`, then repoint the backend at
    /// the extracted `llama-server`. Same in-process HTTP + unarchive path as a
    /// generic release fetch (`.zip` on Windows, `.tar.gz` on Linux/macOS).
    FetchLlamaCpp { dest_dir: PathBuf, display: String },
    /// Clone + cmake-build the PrismML llama.cpp fork as described on the
    /// Bonsai model card (`git clone` → `cmake -B build [-DGGML_CUDA=ON]` →
    /// `cmake --build build -j`). Preferred when `git` and `cmake` are on PATH.
    BuildPrismLlamaCpp { dest_dir: PathBuf, display: String },
    /// Clone + build a colibrì engine from source (`git clone` → `cd c` →
    /// `setup.sh` / `make`), then repoint the backend at the built `coli`
    /// binary. `kind` picks the repo: upstream colibrì (GLM-5.2) or the Hy3
    /// fork.
    BuildColibri {
        kind: BackendKind,
        dest_dir: PathBuf,
        display: String,
    },
}

impl InstallStep {
    fn command(program: &str, args: &[&str]) -> Self {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let display = format!("{program} {}", args.join(" "));
        InstallStep::Command {
            program: program.to_string(),
            args,
            display,
        }
    }

    /// Like [`command`](Self::command) but for programs/args only known at
    /// resolve time (e.g. an absolute venv interpreter path).
    fn command_owned(program: String, args: Vec<String>) -> Self {
        let display = format!("{program} {}", args.join(" "));
        InstallStep::Command {
            program,
            args,
            display,
        }
    }

    /// An elevated command. `display` is the `sudo …` preview shown for approval.
    fn sudo(program: &str, args: &[&str]) -> Self {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let display = format!("sudo {program} {}", args.join(" "));
        InstallStep::Sudo {
            program: program.to_string(),
            args,
            display,
        }
    }

    /// Line shown in the confirm dialog / plan preview.
    pub fn display(&self) -> String {
        match self {
            InstallStep::Command { display, .. } => display.clone(),
            InstallStep::PipInstall { package } => format!("python -m pip install {package}"),
            InstallStep::Sudo { display, .. } => display.clone(),
            InstallStep::FetchLlamaCpp { display, .. } => display.clone(),
            InstallStep::BuildPrismLlamaCpp { display, .. } => display.clone(),
            InstallStep::BuildColibri { display, .. } => display.clone(),
        }
    }
}

/// How to install a backend on this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallPlan {
    /// Run these steps in order, each streamed.
    Automated {
        steps: Vec<InstallStep>,
        /// Full sequence, one step per line, for the confirm dialog.
        display: String,
    },
    /// Can't be automated safely here — show the user how to do it.
    Manual {
        summary: String,
        steps: Vec<String>,
        url: String,
    },
}

impl InstallPlan {
    fn automated(steps: Vec<InstallStep>) -> Self {
        let display = steps
            .iter()
            .map(InstallStep::display)
            .collect::<Vec<_>>()
            .join("\n");
        InstallPlan::Automated { steps, display }
    }

    /// True if any step needs elevation — the single source of truth for whether
    /// the UI must collect a sudo password before running the install. Mirrors
    /// [`RepairPlan::requires_sudo`].
    pub fn requires_sudo(&self) -> bool {
        matches!(
            self,
            InstallPlan::Automated { steps, .. }
                if steps.iter().any(|s| matches!(s, InstallStep::Sudo { .. }))
        )
    }
}

/// Resolve an install plan. Pure: OS, tool availability (`has`), and whether
/// Python is already present are injected, so it's fully unit-testable.
/// `managed_dir` is the app-managed install tree for the backend being planned
/// (llama.cpp fetch/build and colibrì source builds land there).
pub fn install_plan(
    kind: BackendKind,
    os: &str,
    has: &dyn Fn(&str) -> bool,
    has_python: bool,
    managed_dir: &Path,
) -> InstallPlan {
    match kind {
        BackendKind::Ollama => ollama_plan(os, has),
        BackendKind::LlamaCpp => llamacpp_plan(os, has, managed_dir),
        BackendKind::Vllm => pip_backend_plan("vllm", os, has, has_python),
        BackendKind::Sglang => pip_backend_plan("sglang[all]", os, has, has_python),
        BackendKind::Colibri | BackendKind::ColibriHy3 => {
            colibri_plan(kind, os, has, managed_dir)
        }
    }
}

/// The command that installs Python 3 non-interactively on this OS, if we have
/// a package manager for it. `None` = "can't auto-install Python here" (e.g.
/// Linux, where the system package manager needs sudo).
fn python_prereq(os: &str, has: &dyn Fn(&str) -> bool) -> Option<InstallStep> {
    match os {
        "windows" if has("winget") => Some(InstallStep::command(
            "winget",
            &[
                "install",
                "--id",
                "Python.Python.3.12",
                "-e",
                "--scope",
                "user",
                "--accept-source-agreements",
                "--accept-package-agreements",
                "--disable-interactivity",
            ],
        )),
        "macos" if has("brew") => Some(InstallStep::command("brew", &["install", "python"])),
        _ => None,
    }
}

fn ollama_plan(os: &str, has: &dyn Fn(&str) -> bool) -> InstallPlan {
    match os {
        // The official installer writes to /usr/local and registers a systemd
        // service, so it always needs root. We run the whole `curl … | sh` under
        // our own `sudo` so the script sees uid 0 and skips its *internal* sudo,
        // which would otherwise block on a tty the stdin-null child doesn't have.
        // Emitting a `Sudo` step (rather than forcing manual copy-paste) lets the
        // install run from the TUI: the runner uses `sudo -n` when elevation is
        // passwordless and `sudo -S` otherwise, with the password collected by
        // the masked prompt and fed to stdin — it never appears in `display`.
        "linux" if has("sudo") => {
            let script = "curl -fsSL https://ollama.com/install.sh | sh";
            InstallPlan::automated(vec![InstallStep::Sudo {
                program: "sh".into(),
                args: vec!["-c".into(), script.into()],
                display: format!("sudo sh -c '{script}'"),
            }])
        }
        // No `sudo` to elevate with (and we can't assume root): fall back to
        // honest copy-paste steps rather than a command that can't work here.
        "linux" => InstallPlan::Manual {
            summary: "Installing Ollama needs root, and `sudo` isn't available here.".into(),
            steps: vec![
                "Run this in a terminal with root privileges:".into(),
                "curl -fsSL https://ollama.com/install.sh | sh".into(),
                "Ollama starts a local service automatically — then re-detect.".into(),
            ],
            url: "https://ollama.com/download/linux".into(),
        },
        "macos" if has("brew") => {
            InstallPlan::automated(vec![InstallStep::command("brew", &["install", "ollama"])])
        }
        // `--scope user` installs into the user profile (Ollama ships a
        // per-user installer) so no UAC/admin-password prompt appears, and
        // `--disable-interactivity` stops winget blocking on any prompt — the
        // child runs with stdin null and could never answer one.
        "windows" if has("winget") => InstallPlan::automated(vec![InstallStep::command(
            "winget",
            &[
                "install",
                "--id",
                "Ollama.Ollama",
                "-e",
                "--scope",
                "user",
                "--accept-source-agreements",
                "--accept-package-agreements",
                "--disable-interactivity",
            ],
        )]),
        _ => InstallPlan::Manual {
            summary: "Download the Ollama installer for your platform, run it, then re-detect."
                .into(),
            steps: vec![
                "Download and run the installer".into(),
                "Ollama runs a local service automatically — then press [r] to re-detect".into(),
            ],
            url: "https://ollama.com/download".into(),
        },
    }
}

fn llamacpp_plan(os: &str, has: &dyn Fn(&str) -> bool, dest_dir: &Path) -> InstallPlan {
    // Bonsai (and other Prism 1-bit GGUFs) need the PrismML llama.cpp fork —
    // stock ggml-org / Homebrew builds lack the Q1_0_g128 hybrid-attention
    // kernels. Prefer the model-card source build when the toolchain is present;
    // otherwise fetch a Prism prebuilt release.
    //
    // Model card (https://huggingface.co/prism-ml/Bonsai-27B-gguf):
    //   git clone https://github.com/PrismML-Eng/llama.cpp
    //   cmake -B build -DGGML_CUDA=ON && cmake --build build -j   # CUDA
    //   cmake -B build && cmake --build build -j                 # Metal / CPU
    if has("git") && has("cmake") {
        let cuda_hint = if has("nvcc") {
            " with CUDA (GGML_CUDA=ON)"
        } else if os == "macos" {
            " with Metal (default on macOS)"
        } else {
            " (CPU)"
        };
        return InstallPlan::automated(vec![InstallStep::BuildPrismLlamaCpp {
            dest_dir: dest_dir.to_path_buf(),
            display: format!(
                "Build PrismML llama.cpp from source{cuda_hint} → {}",
                dest_dir.display()
            ),
        }]);
    }
    if matches!(os, "windows" | "linux" | "macos") {
        return InstallPlan::automated(vec![InstallStep::FetchLlamaCpp {
            dest_dir: dest_dir.to_path_buf(),
            display: format!(
                "Download prebuilt PrismML llama.cpp (Bonsai kernels) from github.com/PrismML-Eng/llama.cpp → {}",
                dest_dir.display()
            ),
        }]);
    }
    InstallPlan::Manual {
        summary: "Bonsai needs the PrismML llama.cpp fork (custom 1-bit kernels).".into(),
        steps: vec![
            "git clone https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp".into(),
            "cmake -B build -DGGML_CUDA=ON   # or omit -DGGML_CUDA on macOS/CPU".into(),
            "cmake --build build -j".into(),
            "Point backends.llamacpp.bin at build/bin/llama-server(.exe)".into(),
        ],
        url: "https://github.com/PrismML-Eng/llama.cpp".into(),
    }
}

/// Colibrì engines install by source build only — no releases, no packages.
/// Upstream (GLM-5.2) documents Linux, macOS, and native Windows via MinGW;
/// the Hy3 fork documents Linux/WSL. Where the toolchain (or the platform)
/// isn't there, the plan is honest copy-paste steps.
fn colibri_plan(
    kind: BackendKind,
    os: &str,
    has: &dyn Fn(&str) -> bool,
    dest_dir: &Path,
) -> InstallPlan {
    let (repo, engine) = match kind {
        BackendKind::ColibriHy3 => ("https://github.com/ErikTromp/colibri-hy3", "Hy3"),
        _ => ("https://github.com/JustVugg/colibri", "GLM-5.2"),
    };
    let has_cc = has("gcc") || has("cc") || has("clang");
    let build_step = |cuda: bool| {
        InstallStep::BuildColibri {
            kind,
            dest_dir: dest_dir.to_path_buf(),
            display: format!(
                "Build colibrì ({engine} engine) from {repo}{} → {}",
                if cuda { " with CUDA" } else { "" },
                dest_dir.display()
            ),
        }
    };

    match os {
        // Fork README: Linux (native ext4) / WSL. Build = clone + `cd c && setup.sh`.
        "linux" if has("git") && has_cc => {
            InstallPlan::automated(vec![build_step(has("nvcc"))])
        }
        // Upstream additionally documents macOS (Metal) and native Windows
        // (MinGW-w64: gcc + make). The fork doesn't — keep those manual for it.
        "macos" if kind == BackendKind::Colibri && has("git") && has_cc => {
            InstallPlan::automated(vec![build_step(false)])
        }
        "windows" if kind == BackendKind::Colibri && has("git") && has("gcc") && has("make") => {
            InstallPlan::automated(vec![build_step(false)])
        }
        "windows" if kind == BackendKind::Colibri => InstallPlan::Manual {
            summary: "Building colibrì on Windows needs git plus MinGW-w64 (gcc + make).".into(),
            steps: vec![
                "Install MinGW-w64 (winlibs.com or MSYS2) so `gcc` and `make` are on PATH".into(),
                format!("git clone {repo} && cd colibri/c"),
                "make glm.exe   # add ARCH=native for AVX-VNNI".into(),
                "Point backends.colibri.bin at the built binary — or use WSL2".into(),
            ],
            url: repo.into(),
        },
        "windows" | "macos" if kind == BackendKind::ColibriHy3 => InstallPlan::Manual {
            summary: "The colibrì Hy3 fork documents Linux / WSL only.".into(),
            steps: vec![
                "In WSL2 (or a Linux host):".into(),
                format!("git clone {repo} && cd colibri-hy3/c"),
                "./setup.sh   # checks gcc/OpenMP, builds, self-tests".into(),
                "make hy3 CUDA=1   # optional GPU build (needs nvcc)".into(),
                "Point backends.colibri_hy3.bin at the built `coli`".into(),
            ],
            url: repo.into(),
        },
        _ => InstallPlan::Manual {
            summary: format!(
                "Building colibrì ({engine}) needs git and a C toolchain (gcc/clang with OpenMP)."
            ),
            steps: vec![
                "Install git and gcc (with OpenMP support)".into(),
                format!("git clone {repo}"),
                "cd <clone>/c && ./setup.sh   # builds and self-tests".into(),
                format!(
                    "Point backends.{}.bin at the built `coli`",
                    if kind == BackendKind::ColibriHy3 { "colibri_hy3" } else { "colibri" }
                ),
            ],
            url: repo.into(),
        },
    }
}

/// vLLM / SGLang: install Python first if it's missing and auto-installable,
/// then pip-install the package (interpreter resolved at run time).
fn pip_backend_plan(
    package: &str,
    os: &str,
    has: &dyn Fn(&str) -> bool,
    has_python: bool,
) -> InstallPlan {
    let mut steps = Vec::new();
    if !has_python {
        match python_prereq(os, has) {
            Some(step) => steps.push(step),
            None => {
                return InstallPlan::Manual {
                    summary: "Python 3 is required and can't be auto-installed here.".into(),
                    steps: vec![
                        "Install Python 3.10+ (with pip) and put it on PATH".into(),
                        format!("python -m pip install {package}"),
                    ],
                    url: "https://www.python.org/downloads/".into(),
                };
            }
        }
    }
    steps.push(InstallStep::PipInstall {
        package: package.to_string(),
    });
    InstallPlan::automated(steps)
}

/// The app-managed directory a fetched llama.cpp release is extracted into.
pub fn llamacpp_managed_dir(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("backends").join("llamacpp")
}

/// The app-managed source/build tree for a colibrì engine. One tree per kind
/// ("colibri" / "colibri-hy3") so both engines can be installed side by side.
pub fn colibri_managed_dir(paths: &AppPaths, kind: BackendKind) -> PathBuf {
    paths.data_dir.join("backends").join(kind.as_str())
}

/// Locate a colibrì `coli` binary: explicit path → configured name on PATH →
/// managed build tree. The **upstream** kind additionally accepts a bare
/// `coli` found on PATH; the Hy3 fork does NOT — a PATH `coli` is almost
/// always the upstream engine, which cannot serve Hy3, so the fork resolves
/// managed-first and takes PATH only for an explicitly configured name.
pub fn resolve_colibri_bin(configured: &str, paths: &AppPaths, kind: BackendKind) -> Option<PathBuf> {
    let configured = configured.trim();
    if configured.is_empty() {
        return resolve_colibri_bin("coli", paths, kind);
    }
    let as_path = Path::new(configured);
    if as_path.is_file() {
        return Some(as_path.to_path_buf());
    }
    // A non-default configured name is the user pointing at a specific build.
    if configured != "coli" && configured != "coli.exe" {
        if let Ok(p) = which::which(configured) {
            return Some(p);
        }
    }

    let managed = colibri_managed_dir(paths, kind);
    // setup.sh produces `coli`; the Windows MinGW target is `glm.exe`, and the
    // fork's make target is `hy3` — accept the engine binary when no wrapper
    // was built (it speaks the same `serve` CLI).
    let candidates: &[&str] = match kind {
        BackendKind::ColibriHy3 => &["coli.exe", "coli", "hy3.exe", "hy3"],
        _ => &["coli.exe", "coli", "glm.exe", "glm"],
    };
    let managed_bin = candidates.iter().find_map(|n| find_file(&managed, n));

    if kind == BackendKind::Colibri {
        if let Ok(p) = which::which("coli") {
            return Some(p);
        }
    }
    managed_bin
}

/// Locate `llama-server`: explicit path → configured name on PATH →
/// `llama-server` on PATH → managed `backends/llamacpp` install tree.
///
/// Used by the assistant runtime, CLI `setup`, and install scripts so a
/// managed binary works even when it is not on PATH.
pub fn resolve_llamacpp_bin(configured: &str, paths: &AppPaths) -> Option<PathBuf> {
    let configured = configured.trim();
    if configured.is_empty() {
        return resolve_llamacpp_bin("llama-server", paths);
    }

    // Absolute or relative path the user already pointed at.
    let as_path = Path::new(configured);
    if as_path.is_file() {
        return Some(as_path.to_path_buf());
    }
    if let Ok(p) = which::which(configured) {
        return Some(p);
    }
    // Fall back to the canonical binary name when config has a stale path.
    if configured != "llama-server" && configured != "llama-server.exe" {
        if let Ok(p) = which::which("llama-server") {
            return Some(p);
        }
        if cfg!(windows) {
            if let Ok(p) = which::which("llama-server.exe") {
                return Some(p);
            }
        }
    }

    let managed = paths.llamacpp_dir();
    let primary = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    find_file(&managed, primary).or_else(|| find_file(&managed, "llama-server"))
}

/// Ensure a usable **PrismML** `llama-server` exists (Bonsai-compatible).
///
/// Reuses a managed install under [`llamacpp_managed_dir`] when a
/// [`.prism-ml`](PRISM_MARKER) marker is present; otherwise runs the platform
/// install plan (source build from PrismML-Eng/llama.cpp, or Prism prebuilt).
///
/// Stock `llama-server` on PATH alone is **not** sufficient for Bonsai — it
/// lacks the custom 1-bit kernels — so we do not short-circuit on PATH here.
///
/// Returns the absolute path to the binary. Callers should persist it as
/// `backends.llamacpp.bin` so later runs do not depend on PATH.
pub async fn ensure_llamacpp_installed(
    paths: &AppPaths,
    progress: UnboundedSender<String>,
) -> Result<PathBuf, LocalCodeError> {
    paths.ensure_dirs()?;

    if let Some(p) = resolve_prism_llamacpp_bin(paths) {
        let _ = progress.send(format!(
            "PrismML llama-server already available at {}",
            p.display()
        ));
        return Ok(p);
    }

    let plan = resolve_install_plan(BackendKind::LlamaCpp, paths);
    match &plan {
        InstallPlan::Automated { display, .. } => {
            let _ = progress.send(format!(
                "Installing PrismML llama.cpp (required for Bonsai)…\n{display}"
            ));
        }
        InstallPlan::Manual {
            summary,
            steps,
            url,
        } => {
            return Err(LocalCodeError::new(
                ErrorCode::BackendInstallFailed,
                summary.clone(),
            )
            .with_cause(steps.join("\n"))
            .with_hint(format!("See {url}"))
            .with_hint(
                "Build or download PrismML llama-server, then set backends.llamacpp.bin in config.toml",
            ));
        }
    }

    let repoint = run_install(&plan, None, progress.clone()).await?;
    if let Some(r) = repoint {
        return Ok(PathBuf::from(r.bin));
    }

    if let Some(p) = resolve_prism_llamacpp_bin(paths) {
        let _ = progress.send(format!("llama-server ready at {}", p.display()));
        return Ok(p);
    }

    // Last resort: any managed/PATH binary (legacy stock installs).
    if let Some(p) = resolve_llamacpp_bin("llama-server", paths) {
        let _ = progress.send(format!(
            "llama-server at {} (may lack Prism 1-bit kernels — Bonsai may fail)",
            p.display()
        ));
        return Ok(p);
    }

    Err(LocalCodeError::new(
        ErrorCode::BackendBinaryMissing,
        "llama-server not found after install",
    )
    .with_hint("Install llama.cpp from the Backends panel or set backends.llamacpp.bin")
    .with_hint("Bonsai requires https://github.com/PrismML-Eng/llama.cpp")
    .retryable(true))
}

/// Marker file written next to a managed PrismML llama.cpp install.
const PRISM_MARKER: &str = ".prism-ml";

/// Managed `llama-server` that was installed from the PrismML fork (build or
/// prebuilt). Returns `None` if only a stock / unmarked tree is present.
pub fn resolve_prism_llamacpp_bin(paths: &AppPaths) -> Option<PathBuf> {
    let managed = llamacpp_managed_dir(paths);
    if !managed.join(PRISM_MARKER).is_file() {
        return None;
    }
    let primary = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    find_file(&managed, primary).or_else(|| find_file(&managed, "llama-server"))
}

fn write_prism_marker(dest_dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
    std::fs::write(
        dest_dir.join(PRISM_MARKER),
        "repo=https://github.com/PrismML-Eng/llama.cpp\n\
         purpose=Bonsai / Q1_0_g128 hybrid-attention kernels\n\
         installed_by=localcode\n",
    )
    .map_err(|e| e.to_string())
}

/// Resolve an install plan against the real environment.
pub fn resolve_install_plan(kind: BackendKind, paths: &AppPaths) -> InstallPlan {
    let has = |b: &str| which::which(b).is_ok();
    let managed_dir = match kind {
        BackendKind::Colibri | BackendKind::ColibriHy3 => colibri_managed_dir(paths, kind),
        _ => llamacpp_managed_dir(paths),
    };
    // The Ollama-on-Linux plan now emits an elevated (`Sudo`) step instead of
    // being downgraded to manual copy-paste here: the runner elevates with
    // `sudo -n` when that works and `sudo -S` (password from the masked prompt)
    // otherwise, so the install runs from the TUI. See `ollama_plan`.
    install_plan(
        kind,
        std::env::consts::OS,
        &has,
        discover_python().is_some(),
        &managed_dir,
    )
}

/// Whether this process can gain root without an interactive password prompt:
/// either it is already root, or `sudo` is configured passwordless. Used to
/// decide whether a `sudo`-requiring installer can run non-interactively — and
/// by the repair flow to skip the password prompt entirely.
pub fn can_elevate_noninteractively() -> bool {
    use std::process::{Command, Stdio};
    if let Ok(o) = Command::new("id").arg("-u").output() {
        if o.status.success() && String::from_utf8_lossy(&o.stdout).trim() == "0" {
            return true;
        }
    }
    Command::new("sudo")
        .args(["-n", "true"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Find a Python interpreter, resilient to a just-installed one not yet on this
/// process's (already-captured) PATH: PATH → the `py` launcher → known per-user
/// install dirs.
pub fn discover_python() -> Option<PathBuf> {
    // Order matters on Windows: the bare `python3` on PATH is often the
    // Microsoft Store execution-alias stub (which opens the Store instead of
    // running), whereas a real python.org / winget install provides `python`
    // and the `py` launcher — so try those first. On Unix `python3` is
    // canonical and `python` may be Python 2 or absent.
    let order: &[&str] = if cfg!(windows) {
        &["python", "py", "python3"]
    } else {
        &["python3", "python"]
    };
    for name in order {
        if let Ok(p) = which::which(name) {
            return Some(p);
        }
    }
    // winget / python.org per-user installs land under %LOCALAPPDATA% — the
    // reliable finder right after a winget install, when PATH is still stale.
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let base = PathBuf::from(local).join("Programs").join("Python");
        if let Ok(entries) = std::fs::read_dir(&base) {
            let mut dirs: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect();
            dirs.sort();
            // Highest version dir first.
            for dir in dirs.into_iter().rev() {
                let exe = dir.join("python.exe");
                if exe.exists() {
                    return Some(exe);
                }
            }
        }
    }
    None
}

/// Run an automated install plan, forwarding each output line through
/// `progress`. `Manual` plans are handled by the caller (the UI shows steps),
/// so passing one here is an error. Elevated (`Sudo`) steps — e.g. the Ollama
/// Linux installer — run via `sudo -n` when elevation is passwordless, else
/// `sudo -S` with `sudo_password` fed to stdin (never forwarded to `progress`
/// or logged). Returns an optional [`Repoint`] when a step installed a managed
/// binary (llama.cpp fetch) the backend must be pointed at.
pub async fn run_install(
    plan: &InstallPlan,
    sudo_password: Option<&str>,
    progress: UnboundedSender<String>,
) -> Result<Option<Repoint>, LocalCodeError> {
    let InstallPlan::Automated { steps, .. } = plan else {
        return Err(LocalCodeError::new(
            ErrorCode::BackendInstallFailed,
            "No automated installer for this backend on this platform",
        )
        .with_hint("Follow the manual steps shown in the Backends panel"));
    };
    let cid = localcode_core::CorrelationId::new();
    let mut repoint = None;
    for step in steps {
        match step {
            InstallStep::FetchLlamaCpp { dest_dir, .. } => {
                let bin = fetch_llamacpp(dest_dir, &progress, cid).await?;
                repoint = Some(Repoint {
                    kind: BackendKind::LlamaCpp,
                    bin,
                });
            }
            InstallStep::BuildPrismLlamaCpp { dest_dir, .. } => {
                let bin = build_prism_llamacpp(dest_dir, &progress, cid).await?;
                repoint = Some(Repoint {
                    kind: BackendKind::LlamaCpp,
                    bin,
                });
            }
            InstallStep::BuildColibri { kind, dest_dir, .. } => {
                let bin = build_colibri(*kind, dest_dir, &progress, cid).await?;
                repoint = Some(Repoint { kind: *kind, bin });
            }
            InstallStep::Sudo {
                program,
                args,
                display,
            } => {
                run_sudo_step(program, args, display, sudo_password, &progress, cid).await?;
            }
            other => run_step(other, &progress, cid).await?,
        }
    }
    Ok(repoint)
}

async fn run_step(
    step: &InstallStep,
    progress: &UnboundedSender<String>,
    cid: localcode_core::CorrelationId,
) -> Result<(), LocalCodeError> {
    let (program, args, display): (String, Vec<String>, String) = match step {
        InstallStep::Command {
            program,
            args,
            display,
        } => (program.clone(), args.clone(), display.clone()),
        InstallStep::PipInstall { package } => {
            let py = discover_python().ok_or_else(|| {
                LocalCodeError::new(
                    ErrorCode::BackendInstallFailed,
                    "Python was installed but isn't on this session's PATH yet",
                )
                .with_correlation(cid)
                .with_cause("A newly-added PATH entry is only visible to newly-started programs")
                .with_hint("Restart LocalCode, then click Install again to finish the pip step")
                .retryable(true)
            })?;
            (
                py.display().to_string(),
                vec![
                    "-m".into(),
                    "pip".into(),
                    "install".into(),
                    package.clone(),
                ],
                format!("{} -m pip install {package}", py.display()),
            )
        }
        // Elevated steps only appear in repair plans and are executed by
        // `run_repair` (which has the password); they never reach this runner.
        InstallStep::Sudo { display, .. } => {
            return Err(LocalCodeError::new(
                ErrorCode::BackendInstallFailed,
                "Internal: an elevated step reached the non-elevated runner",
            )
            .with_correlation(cid)
            .with_cause(format!("step: {display}")));
        }
        // Handled by `run_install` directly (it needs to capture the repoint).
        InstallStep::FetchLlamaCpp { display, .. }
        | InstallStep::BuildPrismLlamaCpp { display, .. }
        | InstallStep::BuildColibri { display, .. } => {
            return Err(LocalCodeError::new(
                ErrorCode::BackendInstallFailed,
                "Internal: a managed build step reached the generic step runner",
            )
            .with_correlation(cid)
            .with_cause(format!("step: {display}")));
        }
    };

    let _ = progress.send(format!("$ {display}"));
    let mut child = tokio::process::Command::new(&program)
        .args(&args)
        .kill_on_drop(true)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| {
            LocalCodeError::new(
                ErrorCode::BackendInstallFailed,
                format!("Failed to start `{program}`: {e}"),
            )
            .with_correlation(cid)
            .with_cause(format!("`{program}` may not be installed or on PATH"))
            .with_hint(format!("Try running it manually: {display}"))
            .retryable(true)
        })?;

    // Drain BOTH pipes, forwarding each line as progress — an undrained pipe
    // would deadlock the child once its buffer fills.
    forward_lines(child.stdout.take(), progress.clone());
    forward_lines(child.stderr.take(), progress.clone());

    let status = child.wait().await.map_err(|e| {
        LocalCodeError::new(ErrorCode::BackendInstallFailed, e.to_string()).with_correlation(cid)
    })?;
    if !status.success() {
        return Err(LocalCodeError::new(
            ErrorCode::BackendInstallFailed,
            format!("Install step failed ({status})"),
        )
        .with_correlation(cid)
        .with_cause("The installer returned a non-zero exit code")
        .with_hint(format!("Run it manually to see the full output: {display}"))
        .retryable(true));
    }
    Ok(())
}

// ----------------------------------------------------------------------------
// PrismML llama.cpp (Bonsai kernels) — prebuilt fetch + source build
// ----------------------------------------------------------------------------

/// PrismML fork — required for Bonsai Q1_0 / DSpark GGUFs (model card).
const PRISM_LLAMACPP_GIT: &str = "https://github.com/PrismML-Eng/llama.cpp.git";
/// GitHub API for the latest PrismML llama.cpp release (prebuilt binaries).
const PRISM_LLAMACPP_RELEASES_API: &str =
    "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/latest";

/// Download and extract a prebuilt **PrismML** llama.cpp release into
/// `dest_dir`, returning the absolute path to `llama-server`. Prefer CUDA when
/// the host has NVIDIA tooling; otherwise a CPU/Metal asset.
async fn fetch_llamacpp(
    dest_dir: &Path,
    progress: &UnboundedSender<String>,
    cid: localcode_core::CorrelationId,
) -> Result<String, LocalCodeError> {
    let fail = |msg: String| {
        LocalCodeError::new(ErrorCode::BackendInstallFailed, msg)
            .with_correlation(cid)
            .with_hint(
                "Or build from source: git clone https://github.com/PrismML-Eng/llama.cpp \
                 && cmake -B build -DGGML_CUDA=ON && cmake --build build -j",
            )
            .with_hint("https://huggingface.co/prism-ml/Bonsai-27B-gguf")
            .retryable(true)
    };

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        // GitHub's API rejects requests without a User-Agent.
        .user_agent("localcode-installer")
        .build()
        .map_err(|e| fail(e.to_string()))?;

    let prefer_cuda = which::which("nvidia-smi").is_ok() || which::which("nvcc").is_ok();
    let _ = progress.send(
        "$ querying github.com/PrismML-Eng/llama.cpp for the latest Prism release".into(),
    );
    let resp = client
        .get(PRISM_LLAMACPP_RELEASES_API)
        .send()
        .await
        .map_err(|e| fail(format!("Couldn't reach the GitHub releases API: {e}")))?;
    if !resp.status().is_success() {
        return Err(fail(format!(
            "GitHub releases API returned {} (it may be rate-limiting)",
            resp.status()
        )));
    }
    let release: serde_json::Value = resp.json().await.map_err(|e| fail(e.to_string()))?;
    let assets: Vec<(String, String)> = release
        .get("assets")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|a| {
                    let name = a.get("name")?.as_str()?.to_string();
                    let url = a.get("browser_download_url")?.as_str()?.to_string();
                    Some((name, url))
                })
                .collect()
        })
        .unwrap_or_default();

    let (asset_name, asset_url) = pick_llamacpp_asset(
        &assets,
        std::env::consts::OS,
        std::env::consts::ARCH,
        prefer_cuda,
    )
    .ok_or_else(|| {
        fail(
            "The latest Prism release has no prebuilt binary for this OS/arch \
             (install git + cmake to build from source instead)"
                .into(),
        )
    })?;

    let _ = progress.send(format!("$ downloading {asset_name}"));
    let bytes = client
        .get(&asset_url)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| fail(format!("Download failed: {e}")))?
        .bytes()
        .await
        .map_err(|e| fail(format!("Download interrupted: {e}")))?
        .to_vec();

    let _ = progress.send(format!(
        "$ extracting {} ({} MiB)",
        asset_name,
        bytes.len() / 1_048_576
    ));
    let dest = dest_dir.to_path_buf();
    let name = asset_name.clone();
    let bin = tokio::task::spawn_blocking(move || extract_llamacpp_archive(&bytes, &dest, &name))
        .await
        .map_err(|e| fail(e.to_string()))?
        .map_err(fail)?;

    write_prism_marker(dest_dir).map_err(fail)?;
    let _ = progress.send(format!(
        "PrismML llama-server installed to {}",
        bin.display()
    ));
    Ok(bin.display().to_string())
}

/// Clone and cmake-build the PrismML llama.cpp fork (model-card quickstart).
///
/// ```text
/// git clone https://github.com/PrismML-Eng/llama.cpp
/// cmake -B build -DGGML_CUDA=ON   # when nvcc is present
/// cmake --build build -j
/// ```
async fn build_prism_llamacpp(
    dest_dir: &Path,
    progress: &UnboundedSender<String>,
    cid: localcode_core::CorrelationId,
) -> Result<String, LocalCodeError> {
    let fail = |msg: String| {
        LocalCodeError::new(ErrorCode::BackendInstallFailed, msg)
            .with_correlation(cid)
            .with_hint(
                "Model card build: https://huggingface.co/prism-ml/Bonsai-27B-gguf \
                 (git clone PrismML-Eng/llama.cpp, cmake -B build, cmake --build build -j)",
            )
            .with_hint("Need git, cmake, and a C/C++ toolchain (Visual Studio / Xcode / build-essential)")
            .retryable(true)
    };

    if which::which("git").is_err() {
        return Err(fail("`git` not found on PATH".into()));
    }
    if which::which("cmake").is_err() {
        return Err(fail("`cmake` not found on PATH".into()));
    }

    let src_dir = dest_dir.join("src");
    std::fs::create_dir_all(dest_dir).map_err(|e| fail(e.to_string()))?;

    // Clone or refresh the PrismML fork (shallow).
    if src_dir.join(".git").is_dir() {
        let _ = progress.send(format!(
            "$ git -C {} pull --ff-only (PrismML llama.cpp)",
            src_dir.display()
        ));
        let status = tokio::process::Command::new("git")
            .args(["-C"])
            .arg(&src_dir)
            .args(["pull", "--ff-only"])
            .kill_on_drop(true)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .status()
            .await
            .map_err(|e| fail(format!("git pull failed to start: {e}")))?;
        if !status.success() {
            let _ = progress.send(
                "git pull failed — continuing with existing tree (may be offline)".into(),
            );
        }
    } else {
        // Remove a partial non-git tree so clone can succeed.
        if src_dir.exists() {
            let _ = std::fs::remove_dir_all(&src_dir);
        }
        let _ = progress.send(format!("$ git clone --depth 1 {PRISM_LLAMACPP_GIT}"));
        let mut child = tokio::process::Command::new("git")
            .args(["clone", "--depth", "1", PRISM_LLAMACPP_GIT])
            .arg(&src_dir)
            .kill_on_drop(true)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| fail(format!("git clone failed to start: {e}")))?;
        forward_lines(child.stdout.take(), progress.clone());
        forward_lines(child.stderr.take(), progress.clone());
        let status = child
            .wait()
            .await
            .map_err(|e| fail(format!("git clone wait: {e}")))?;
        if !status.success() {
            return Err(fail(format!(
                "git clone of PrismML llama.cpp failed ({status})"
            )));
        }
    }

    let want_cuda = which::which("nvcc").is_ok();
    let jobs = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .to_string();

    // Configure (model card: cmake -B build [-DGGML_CUDA=ON]).
    let mut cmake_args: Vec<String> = vec![
        "-B".into(),
        "build".into(),
        "-DCMAKE_BUILD_TYPE=Release".into(),
        // Server binary is what LocalCode / Bonsai need.
        "-DLLAMA_BUILD_SERVER=ON".into(),
    ];
    if want_cuda {
        cmake_args.push("-DGGML_CUDA=ON".into());
        let _ = progress.send(
            "$ cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release (PrismML llama.cpp)".into(),
        );
    } else if cfg!(target_os = "macos") {
        let _ = progress.send(
            "$ cmake -B build -DCMAKE_BUILD_TYPE=Release (Metal default on macOS)".into(),
        );
    } else {
        let _ = progress.send(
            "$ cmake -B build -DCMAKE_BUILD_TYPE=Release (CPU; install CUDA toolkit + nvcc for GPU)"
                .into(),
        );
    }

    let mut cfg = tokio::process::Command::new("cmake");
    cfg.args(&cmake_args)
        .current_dir(&src_dir)
        .kill_on_drop(true)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cfg
        .spawn()
        .map_err(|e| fail(format!("cmake configure failed to start: {e}")))?;
    forward_lines(child.stdout.take(), progress.clone());
    forward_lines(child.stderr.take(), progress.clone());
    let status = child
        .wait()
        .await
        .map_err(|e| fail(format!("cmake configure wait: {e}")))?;
    if !status.success() {
        return Err(fail(format!(
            "cmake configure failed ({status}). On Windows install Visual Studio C++ tools; \
             for CUDA install the CUDA toolkit so `nvcc` is on PATH."
        )));
    }

    // Build (model card: cmake --build build -j).
    let mut build_args: Vec<String> = vec![
        "--build".into(),
        "build".into(),
        "-j".into(),
        jobs.clone(),
    ];
    // Multi-config generators (Visual Studio) need an explicit config.
    if cfg!(windows) {
        build_args.push("--config".into());
        build_args.push("Release".into());
    }
    let _ = progress.send(format!(
        "$ cmake --build build -j {jobs}{}",
        if cfg!(windows) {
            " --config Release"
        } else {
            ""
        }
    ));
    let mut child = tokio::process::Command::new("cmake")
        .args(&build_args)
        .current_dir(&src_dir)
        .kill_on_drop(true)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| fail(format!("cmake build failed to start: {e}")))?;
    forward_lines(child.stdout.take(), progress.clone());
    forward_lines(child.stderr.take(), progress.clone());
    let status = child
        .wait()
        .await
        .map_err(|e| fail(format!("cmake build wait: {e}")))?;
    if !status.success() {
        return Err(fail(format!(
            "cmake build failed ({status}). Check compiler / CUDA toolkit output above."
        )));
    }

    let bin_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    // Search the whole managed tree (build/bin, build/bin/Release, …).
    let bin = find_file(dest_dir, bin_name)
        .or_else(|| find_file(&src_dir, bin_name))
        .ok_or_else(|| {
            fail(format!(
                "{bin_name} was not produced under {}",
                dest_dir.display()
            ))
        })?;

    write_prism_marker(dest_dir).map_err(fail)?;
    let _ = progress.send(format!(
        "PrismML llama-server built at {} ({})",
        bin.display(),
        if want_cuda { "CUDA" } else { "CPU/Metal" }
    ));
    Ok(bin.display().to_string())
}

// ----------------------------------------------------------------------------
// Colibrì engines — source build (no binary releases exist)
// ----------------------------------------------------------------------------

/// Upstream colibrì (GLM-5.2 expert-streaming engine).
const COLIBRI_GIT: &str = "https://github.com/JustVugg/colibri.git";
/// Colibrì × Hy3 fork (Tencent Hy3; also serves GLM-5.2 containers).
const COLIBRI_HY3_GIT: &str = "https://github.com/ErikTromp/colibri-hy3.git";
/// Marker file written next to a managed colibrì build.
const COLIBRI_MARKER: &str = ".colibri";

fn write_colibri_marker(dest_dir: &Path, repo: &str) -> Result<(), String> {
    std::fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
    std::fs::write(
        dest_dir.join(COLIBRI_MARKER),
        format!("repo={repo}\ninstalled_by=localcode\n"),
    )
    .map_err(|e| e.to_string())
}

/// Spawn one build command with a working directory, streaming both pipes into
/// `progress`. Errors are strings; callers wrap them with backend context.
async fn run_build_cmd(
    program: &str,
    args: &[&str],
    cwd: &Path,
    progress: &UnboundedSender<String>,
) -> Result<(), String> {
    let _ = progress.send(format!("$ {program} {}", args.join(" ")));
    let mut child = tokio::process::Command::new(program)
        .args(args)
        .current_dir(cwd)
        .kill_on_drop(true)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("{program} failed to start: {e}"))?;
    forward_lines(child.stdout.take(), progress.clone());
    forward_lines(child.stderr.take(), progress.clone());
    let status = child
        .wait()
        .await
        .map_err(|e| format!("{program} wait: {e}"))?;
    if !status.success() {
        return Err(format!("{program} {} failed ({status})", args.join(" ")));
    }
    Ok(())
}

/// Clone and build a colibrì engine (README quickstart):
///
/// ```text
/// git clone <repo>
/// cd c && ./setup.sh          # checks gcc/OpenMP, builds, self-tests
/// make [hy3] CUDA=1           # optional GPU rebuild when nvcc is present
/// ```
///
/// On native Windows (upstream only) the documented path is MinGW-w64:
/// `make glm.exe` inside `c/`. Returns the built binary's absolute path.
async fn build_colibri(
    kind: BackendKind,
    dest_dir: &Path,
    progress: &UnboundedSender<String>,
    cid: localcode_core::CorrelationId,
) -> Result<String, LocalCodeError> {
    let (repo, cfg_key) = match kind {
        BackendKind::ColibriHy3 => (COLIBRI_HY3_GIT, "colibri_hy3"),
        _ => (COLIBRI_GIT, "colibri"),
    };
    let fail = move |msg: String| {
        LocalCodeError::new(ErrorCode::BackendInstallFailed, msg)
            .with_correlation(cid)
            .with_hint(format!("Manual build: git clone {repo} && cd c && ./setup.sh"))
            .with_hint(format!("Then set backends.{cfg_key}.bin to the built `coli`"))
            .retryable(true)
    };

    if which::which("git").is_err() {
        return Err(fail("`git` not found on PATH".into()));
    }

    let src_dir = dest_dir.join("src");
    std::fs::create_dir_all(dest_dir).map_err(|e| fail(e.to_string()))?;
    let src_str = src_dir.display().to_string();

    // Clone or refresh the engine repo (shallow), same rules as the PrismML
    // llama.cpp build: a failed pull keeps the existing tree (may be offline).
    if src_dir.join(".git").is_dir() {
        if let Err(e) = run_build_cmd(
            "git",
            &["-C", &src_str, "pull", "--ff-only"],
            dest_dir,
            progress,
        )
        .await
        {
            let _ = progress.send(format!("{e} — continuing with the existing tree"));
        }
    } else {
        // Remove a partial non-git tree so clone can succeed.
        if src_dir.exists() {
            let _ = std::fs::remove_dir_all(&src_dir);
        }
        run_build_cmd(
            "git",
            &["clone", "--depth", "1", repo, &src_str],
            dest_dir,
            progress,
        )
        .await
        .map_err(fail)?;
    }

    let c_dir = src_dir.join("c");
    if !c_dir.is_dir() {
        return Err(fail(format!(
            "unexpected repo layout: {} has no c/ directory",
            src_dir.display()
        )));
    }

    if cfg!(windows) {
        // Upstream's documented native-Windows path (MinGW-w64).
        let target = if kind == BackendKind::ColibriHy3 { "hy3.exe" } else { "glm.exe" };
        run_build_cmd("make", &[target], &c_dir, progress)
            .await
            .map_err(fail)?;
    } else {
        // Through `sh` so a lost exec bit on setup.sh can't break the build.
        run_build_cmd("sh", &["setup.sh"], &c_dir, progress)
            .await
            .map_err(fail)?;
        // Optional CUDA rebuild; a failure here keeps the working CPU build.
        if which::which("nvcc").is_ok() && which::which("make").is_ok() {
            let cuda_args: &[&str] = if kind == BackendKind::ColibriHy3 {
                &["hy3", "CUDA=1"]
            } else {
                &["CUDA=1"]
            };
            if let Err(e) = run_build_cmd("make", cuda_args, &c_dir, progress).await {
                let _ = progress.send(format!(
                    "CUDA build failed ({e}) — keeping the CPU build from setup.sh"
                ));
            }
        }
    }

    let candidates: &[&str] = match kind {
        BackendKind::ColibriHy3 => &["coli.exe", "coli", "hy3.exe", "hy3"],
        _ => &["coli.exe", "coli", "glm.exe", "glm"],
    };
    let bin = candidates
        .iter()
        .find_map(|n| find_file(dest_dir, n))
        .ok_or_else(|| {
            fail(format!(
                "no coli binary was produced under {}",
                dest_dir.display()
            ))
        })?;

    write_colibri_marker(dest_dir, repo).map_err(fail)?;
    let _ = progress.send(format!("colibrì ready at {}", bin.display()));
    Ok(bin.display().to_string())
}

/// Pick a Prism (or compatible) release asset for this OS/arch.
///
/// When `prefer_cuda` is true, CUDA builds are preferred on Windows/Linux;
/// otherwise CPU (or macOS Metal) archives win. HIP/Vulkan/OpenVINO are never
/// chosen automatically (driver matrix too wide). Pure for unit tests.
///
/// Archive format: Windows `.zip`, Linux/macOS `.tar.gz`.
fn pick_llamacpp_asset(
    assets: &[(String, String)],
    os: &str,
    arch: &str,
    prefer_cuda: bool,
) -> Option<(String, String)> {
    const SKIP_MARKERS: [&str; 8] = [
        "hip", "rocm", "vulkan", "sycl", "musa", "kompute", "openvino", "kleidiai",
    ];
    // cudart DLL packs are not the full binary archive.
    const SKIP_NAME: [&str; 2] = ["cudart", "xcframework"];

    let exts: &[&str] = if os == "windows" {
        &[".zip"]
    } else {
        &[".tar.gz", ".tgz"]
    };
    let is_archive = |n: &str| {
        let l = n.to_lowercase();
        exts.iter().any(|e| l.ends_with(e))
            && !SKIP_MARKERS.iter().any(|m| l.contains(m))
            && !SKIP_NAME.iter().any(|m| l.contains(m))
    };
    let is_cuda = |n: &str| {
        let l = n.to_lowercase();
        l.contains("cuda") || l.contains("cu12") || l.contains("cu11")
    };
    let all = |n: &str, pats: &[&str]| {
        let l = n.to_lowercase();
        pats.iter().all(|p| l.contains(p))
    };

    let os_prefs: Vec<Vec<&str>> = match (os, arch) {
        ("windows", "aarch64") => vec![vec!["win", "arm64"], vec!["bin-win"]],
        ("windows", _) => vec![vec!["win", "x64"], vec!["bin-win"], vec!["win"]],
        ("linux", "aarch64") => vec![
            vec!["ubuntu", "arm64"],
            vec!["linux", "arm64"],
            vec!["ubuntu"],
        ],
        ("linux", _) => vec![
            vec!["ubuntu", "x64"],
            vec!["linux", "x64"],
            vec!["ubuntu"],
            vec!["linux"],
        ],
        ("macos", "aarch64") => vec![vec!["macos", "arm64"], vec!["bin-macos"], vec!["macos"]],
        ("macos", _) => vec![vec!["macos", "x64"], vec!["bin-macos"], vec!["macos"]],
        _ => return None,
    };

    let candidates: Vec<&(String, String)> = assets
        .iter()
        .filter(|(n, _)| is_archive(n))
        .collect();

    // Pass 1: OS match + CUDA if preferred.
    // Pass 2: OS match + non-CUDA.
    for want_cuda in [prefer_cuda && os != "macos", false] {
        for pats in &os_prefs {
            if let Some((n, u)) = candidates.iter().find(|(n, _)| {
                all(n, pats) && is_cuda(n) == want_cuda
            }) {
                return Some((n.clone(), u.clone()));
            }
        }
    }
    // Pass 3: any OS-matching archive (CUDA or not).
    for pats in &os_prefs {
        if let Some((n, u)) = candidates.iter().find(|(n, _)| all(n, pats)) {
            return Some((n.clone(), u.clone()));
        }
    }
    None
}

/// Extract a downloaded llama.cpp release into `dest_dir`, dispatching on the
/// asset's archive format (Windows ships `.zip`, Linux/macOS ship `.tar.gz`),
/// and return the path to the extracted `llama-server` binary. Sync — run on a
/// blocking task.
fn extract_llamacpp_archive(
    bytes: &[u8],
    dest_dir: &Path,
    asset_name: &str,
) -> Result<PathBuf, String> {
    let lower = asset_name.to_lowercase();
    if lower.ends_with(".zip") {
        extract_llamacpp_zip(bytes, dest_dir)
    } else if lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
        extract_llamacpp_targz(bytes, dest_dir)
    } else {
        Err(format!("unsupported archive format: {asset_name}"))
    }
}

/// Extract a llama.cpp `.tar.gz` release into `dest_dir` and return the path to
/// the `llama-server` binary within it. `unpack_in` refuses entries that escape
/// `dest_dir` (absolute paths or `..`), and preserving permissions keeps the
/// binary and its co-located shared libraries executable. Sync — run on a
/// blocking task. `.tar.gz` is the Linux/macOS asset format, so the binary is
/// always the unix `llama-server` (no `.exe`).
fn extract_llamacpp_targz(bytes: &[u8], dest_dir: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
    let decoder = flate2::read::GzDecoder::new(std::io::Cursor::new(bytes));
    let mut archive = tar::Archive::new(decoder);
    archive.set_preserve_permissions(true);
    for entry in archive.entries().map_err(|e| format!("open archive: {e}"))? {
        let mut entry = entry.map_err(|e| e.to_string())?;
        // unpack_in sanitizes the path and skips anything that would traverse
        // outside dest_dir, returning Ok(false) for a skipped entry.
        entry.unpack_in(dest_dir).map_err(|e| e.to_string())?;
    }
    let found = find_file(dest_dir, "llama-server")
        .ok_or_else(|| "llama-server was not found inside the downloaded archive".to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&found, std::fs::Permissions::from_mode(0o755));
    }
    Ok(found)
}

/// Extract a llama.cpp release zip into `dest_dir` (zip-slip safe) and return the
/// path to the `llama-server` binary within it. Sync — run on a blocking task.
fn extract_llamacpp_zip(bytes: &[u8], dest_dir: &Path) -> Result<PathBuf, String> {
    use std::io::Read;
    std::fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes))
        .map_err(|e| format!("open archive: {e}"))?;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
        // Sanitize: no absolute paths, no `..` traversal (zip-slip).
        let name = entry.name().replace('\\', "/");
        if name.starts_with('/') || name.split('/').any(|c| c == "..") {
            continue;
        }
        let out = dest_dir.join(&name);
        if name.ends_with('/') {
            std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        std::fs::write(&out, &buf).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Some(mode) = entry.unix_mode() {
                let _ = std::fs::set_permissions(&out, std::fs::Permissions::from_mode(mode));
            }
        }
    }

    let bin_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let found = find_file(dest_dir, bin_name)
        .ok_or_else(|| format!("{bin_name} was not found inside the downloaded archive"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&found, std::fs::Permissions::from_mode(0o755));
    }
    Ok(found)
}

/// Depth-first search for a file named `name` (case-insensitive) under `dir`.
fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p
                .file_name()
                .map(|n| n.eq_ignore_ascii_case(name))
                .unwrap_or(false)
            {
                return Some(p);
            }
        }
    }
    None
}

// ----------------------------------------------------------------------------
// Repair plans (diagnosis → concrete fix)
// ----------------------------------------------------------------------------

/// After a repair's steps succeed, point a backend's `bin` at a new path and
/// persist — how the managed-venv repair actually takes effect.
#[derive(Debug, Clone)]
pub struct Repoint {
    pub kind: BackendKind,
    pub bin: String,
}

/// A concrete, resolved repair: ordered steps plus an optional post-step
/// repoint. Built from an abstract [`RepairIntent`] by [`resolve_repair`].
#[derive(Debug, Clone)]
pub struct RepairPlan {
    pub title: String,
    pub steps: Vec<InstallStep>,
    /// Extra note shown above the command preview (e.g. "re-downloads wheels").
    pub caveat: Option<String>,
    pub repoint: Option<Repoint>,
}

impl RepairPlan {
    /// True if any step needs elevation — the single source of truth for
    /// whether the UI must collect a sudo password.
    pub fn requires_sudo(&self) -> bool {
        self.steps
            .iter()
            .any(|s| matches!(s, InstallStep::Sudo { .. }))
    }

    /// One step per line for the confirm / preview banner.
    pub fn display(&self) -> String {
        self.steps
            .iter()
            .map(InstallStep::display)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Resolve an abstract [`RepairIntent`] into concrete commands for this machine.
/// Impure (reads OS, PATH, Python); the pure counterpart is `diagnose`. Mirrors
/// the `install_plan` / `resolve_install_plan` split.
pub fn resolve_repair(
    intent: &RepairIntent,
    kind: BackendKind,
    paths: &AppPaths,
) -> Result<RepairPlan, LocalCodeError> {
    let has = |b: &str| which::which(b).is_ok();
    resolve_repair_with(
        intent,
        kind,
        paths,
        std::env::consts::OS,
        &has,
        discover_python(),
    )
}

/// Testable core: OS, tool availability and the base interpreter are injected.
fn resolve_repair_with(
    intent: &RepairIntent,
    kind: BackendKind,
    paths: &AppPaths,
    os: &str,
    has: &dyn Fn(&str) -> bool,
    base_python: Option<PathBuf>,
) -> Result<RepairPlan, LocalCodeError> {
    match intent {
        RepairIntent::CleanVenvReinstall => clean_venv_reinstall(kind, paths, os, base_python),
        RepairIntent::StartOllamaService => start_ollama_service(os, has),
        RepairIntent::ReinstallFormula(formula) => reinstall_formula(formula, has),
    }
}

/// Build a fresh venv under the app data dir, install the backend into it, and
/// repoint the backend at that venv — the robust fix for a polluted global
/// Python (the op-registration bug).
fn clean_venv_reinstall(
    kind: BackendKind,
    paths: &AppPaths,
    os: &str,
    base_python: Option<PathBuf>,
) -> Result<RepairPlan, LocalCodeError> {
    let py = base_python.ok_or_else(|| {
        LocalCodeError::new(
            ErrorCode::BackendInstallFailed,
            "No Python interpreter found to build a clean environment",
        )
        .with_hint("Install Python 3.10+ and put it on PATH, then try Fix again")
    })?;
    let windows = os == "windows";
    let dir = paths.data_dir.join("venvs").join(kind.as_str());
    let (bin_sub, exe_ext) = if windows { ("Scripts", ".exe") } else { ("bin", "") };
    let venv_py = dir.join(bin_sub).join(format!("python{exe_ext}"));
    let venv_py_s = venv_py.display().to_string();

    let package = match kind {
        BackendKind::Sglang => "sglang[all]",
        _ => "vllm",
    };

    let steps = vec![
        InstallStep::command_owned(
            py.display().to_string(),
            vec!["-m".into(), "venv".into(), dir.display().to_string()],
        ),
        InstallStep::command_owned(
            venv_py_s.clone(),
            vec![
                "-m".into(),
                "pip".into(),
                "install".into(),
                "--upgrade".into(),
                "pip".into(),
            ],
        ),
        InstallStep::command_owned(
            venv_py_s.clone(),
            vec![
                "-m".into(),
                "pip".into(),
                "install".into(),
                package.into(),
            ],
        ),
    ];

    // vLLM is invoked as its own console script; SGLang as `<python> -m
    // sglang.launch_server`, so its bin IS the venv interpreter.
    let bin = match kind {
        BackendKind::Vllm => dir
            .join(bin_sub)
            .join(format!("vllm{exe_ext}"))
            .display()
            .to_string(),
        _ => venv_py_s,
    };

    Ok(RepairPlan {
        title: format!("Reinstall {} in a clean environment", kind.as_str()),
        steps,
        caveat: Some("Builds a fresh virtualenv — re-downloads wheels (several minutes).".into()),
        repoint: Some(Repoint { kind, bin }),
    })
}

fn start_ollama_service(
    os: &str,
    has: &dyn Fn(&str) -> bool,
) -> Result<RepairPlan, LocalCodeError> {
    let steps = match os {
        "linux" => vec![InstallStep::sudo("systemctl", &["start", "ollama"])],
        "macos" if has("brew") => {
            vec![InstallStep::command("brew", &["services", "start", "ollama"])]
        }
        _ => {
            return Err(LocalCodeError::new(
                ErrorCode::BackendInstallFailed,
                "Can't start the Ollama service automatically here",
            )
            .with_hint("Start the Ollama app, then press re-detect"));
        }
    };
    Ok(RepairPlan {
        title: "Start the Ollama service".into(),
        steps,
        caveat: None,
        repoint: None,
    })
}

fn reinstall_formula(
    formula: &str,
    has: &dyn Fn(&str) -> bool,
) -> Result<RepairPlan, LocalCodeError> {
    if !has("brew") {
        return Err(LocalCodeError::new(
            ErrorCode::BackendInstallFailed,
            "Homebrew isn't available to reinstall this formula",
        )
        .with_hint(format!("Reinstall {formula} with your system package manager")));
    }
    Ok(RepairPlan {
        title: format!("Reinstall {formula}"),
        steps: vec![InstallStep::command("brew", &["reinstall", formula])],
        caveat: None,
        repoint: None,
    })
}

/// Run a repair plan's steps in order, streaming output. Elevated (`Sudo`) steps
/// use `sudo -n` when passwordless elevation works, else `sudo -S` with
/// `sudo_password` fed to stdin — which is NEVER forwarded to `progress` or
/// logged.
pub async fn run_repair(
    plan: &RepairPlan,
    sudo_password: Option<&str>,
    progress: UnboundedSender<String>,
) -> Result<(), LocalCodeError> {
    let cid = localcode_core::CorrelationId::new();
    for step in &plan.steps {
        match step {
            InstallStep::Sudo {
                program,
                args,
                display,
            } => {
                run_sudo_step(program, args, display, sudo_password, &progress, cid).await?;
            }
            other => run_step(other, &progress, cid).await?,
        }
    }
    Ok(())
}

#[cfg(unix)]
async fn run_sudo_step(
    program: &str,
    args: &[String],
    display: &str,
    password: Option<&str>,
    progress: &UnboundedSender<String>,
    cid: localcode_core::CorrelationId,
) -> Result<(), LocalCodeError> {
    use tokio::io::AsyncWriteExt;
    let _ = progress.send(format!("$ {display}"));

    let mut cmd = tokio::process::Command::new("sudo");
    if password.is_some() {
        // -S reads the password from stdin; -p "" silences the prompt text.
        cmd.arg("-S").arg("-p").arg("");
    } else {
        cmd.arg("-n"); // passwordless sudo / already root
    }
    cmd.arg(program)
        .args(args)
        .kill_on_drop(true)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| {
        LocalCodeError::new(
            ErrorCode::BackendInstallFailed,
            format!("Failed to run sudo: {e}"),
        )
        .with_correlation(cid)
        .with_hint(format!("Run it manually: {display}"))
    })?;

    // Feed the password to stdin then close it (EOF) so sudo proceeds. The
    // password only ever travels this pipe — never `progress`, never tracing.
    if let Some(pw) = password {
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(pw.as_bytes()).await;
            let _ = stdin.write_all(b"\n").await;
            let _ = stdin.flush().await;
        }
    }

    forward_lines(child.stdout.take(), progress.clone());
    forward_lines(child.stderr.take(), progress.clone());

    let status = child.wait().await.map_err(|e| {
        LocalCodeError::new(ErrorCode::BackendInstallFailed, e.to_string()).with_correlation(cid)
    })?;
    if !status.success() {
        return Err(LocalCodeError::new(
            ErrorCode::BackendInstallFailed,
            format!("Elevated step failed ({status})"),
        )
        .with_correlation(cid)
        .with_cause("sudo returned non-zero (wrong password, or not permitted)")
        .with_hint(format!("Run it manually: {display}"))
        .retryable(true));
    }
    Ok(())
}

#[cfg(not(unix))]
async fn run_sudo_step(
    program: &str,
    args: &[String],
    display: &str,
    password: Option<&str>,
    _progress: &UnboundedSender<String>,
    cid: localcode_core::CorrelationId,
) -> Result<(), LocalCodeError> {
    // Repairs never emit Sudo steps on Windows; this arm exists only to keep
    // the runner total. Honest failure rather than a fabricated success.
    let _ = (program, args, password);
    Err(LocalCodeError::new(
        ErrorCode::BackendInstallFailed,
        "Elevated commands aren't supported on this platform",
    )
    .with_correlation(cid)
    .with_hint(format!("Run it manually: {display}")))
}

fn forward_lines<R>(reader: Option<R>, tx: UnboundedSender<String>)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    use tokio::io::{AsyncBufReadExt, BufReader};
    if let Some(r) = reader {
        tokio::spawn(async move {
            let mut lines = BufReader::new(r).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No tools available.
    fn no(_: &str) -> bool {
        false
    }

    fn automated(plan: InstallPlan) -> Vec<InstallStep> {
        match plan {
            InstallPlan::Automated { steps, .. } => steps,
            InstallPlan::Manual { .. } => panic!("expected an automated plan"),
        }
    }

    /// `install_plan` with a throwaway managed dir for llama.cpp fetch steps.
    fn ip(
        kind: BackendKind,
        os: &str,
        has: &dyn Fn(&str) -> bool,
        has_python: bool,
    ) -> InstallPlan {
        install_plan(kind, os, has, has_python, std::path::Path::new("/managed/llamacpp"))
    }

    #[test]
    fn ollama_linux_with_sudo_is_elevated_official_script() {
        // Ollama on Linux always needs root; with sudo present the plan runs the
        // official installer under an elevated step so it installs from the TUI.
        let has = |b: &str| b == "sudo";
        let plan = ip(BackendKind::Ollama, "linux", &has, false);
        assert!(plan.requires_sudo(), "an elevated step must be emitted");
        let steps = automated(plan);
        assert_eq!(steps.len(), 1);
        assert!(matches!(&steps[0], InstallStep::Sudo { .. }));
        let d = steps[0].display();
        assert!(d.starts_with("sudo "));
        assert!(d.contains("ollama.com/install.sh"));
    }

    #[test]
    fn ollama_linux_without_sudo_is_manual() {
        // No sudo to elevate with: honest copy-paste steps rather than a command
        // that can't run non-interactively here.
        let plan = ip(BackendKind::Ollama, "linux", &no, false);
        assert!(matches!(plan, InstallPlan::Manual { .. }));
        assert!(!plan.requires_sudo());
    }

    #[test]
    fn ollama_windows_winget() {
        let has = |b: &str| b == "winget";
        let steps = automated(ip(BackendKind::Ollama, "windows", &has, false));
        let d = steps[0].display();
        assert!(d.contains("winget"));
        assert!(d.contains("Ollama.Ollama"));
        // Non-interactive + user scope so no UAC/password prompt can block it.
        assert!(d.contains("--disable-interactivity"));
        assert!(d.contains("--scope user"));
    }

    #[test]
    fn ollama_windows_without_winget_is_manual() {
        let plan = ip(BackendKind::Ollama, "windows", &no, false);
        assert!(matches!(plan, InstallPlan::Manual { .. }));
    }

    #[test]
    fn ollama_macos_brew() {
        let has = |b: &str| b == "brew";
        let steps = automated(ip(BackendKind::Ollama, "macos", &has, false));
        assert!(steps[0].display().contains("brew install ollama"));
    }

    #[test]
    fn vllm_prepends_python_when_missing_on_windows() {
        let has = |b: &str| b == "winget";
        let steps = automated(ip(BackendKind::Vllm, "windows", &has, false));
        assert_eq!(steps.len(), 2);
        assert!(steps[0].display().to_lowercase().contains("python"));
        assert!(matches!(&steps[1], InstallStep::PipInstall { package } if package == "vllm"));
    }

    #[test]
    fn vllm_macos_brew_installs_python_prereq() {
        let has = |b: &str| b == "brew";
        let steps = automated(ip(BackendKind::Vllm, "macos", &has, false));
        assert_eq!(steps.len(), 2);
        assert!(steps[0].display().contains("brew install python"));
    }

    #[test]
    fn vllm_single_step_when_python_present() {
        let steps = automated(ip(BackendKind::Vllm, "windows", &no, true));
        assert_eq!(steps.len(), 1);
        assert!(matches!(&steps[0], InstallStep::PipInstall { .. }));
    }

    #[test]
    fn vllm_linux_without_python_is_manual() {
        // No package manager can install Python non-interactively (needs sudo).
        let plan = ip(BackendKind::Vllm, "linux", &no, false);
        assert!(matches!(plan, InstallPlan::Manual { .. }));
    }

    #[test]
    fn sglang_pip_package_has_all_extra() {
        let steps = automated(ip(BackendKind::Sglang, "linux", &no, true));
        assert!(matches!(&steps[0], InstallStep::PipInstall { package } if package == "sglang[all]"));
    }

    #[test]
    fn llamacpp_builds_prism_when_git_and_cmake_present() {
        // Model-card path: source-build PrismML-Eng/llama.cpp (not Homebrew stock).
        let has = |b: &str| b == "git" || b == "cmake";
        let steps = automated(ip(BackendKind::LlamaCpp, "macos", &has, false));
        assert!(matches!(&steps[0], InstallStep::BuildPrismLlamaCpp { .. }));
        assert!(steps[0].display().contains("PrismML"));
    }

    #[test]
    fn llamacpp_builds_with_cuda_hint_when_nvcc_present() {
        let has = |b: &str| matches!(b, "git" | "cmake" | "nvcc");
        let steps = automated(ip(BackendKind::LlamaCpp, "linux", &has, false));
        assert!(matches!(&steps[0], InstallStep::BuildPrismLlamaCpp { .. }));
        assert!(steps[0].display().contains("CUDA"));
    }

    #[test]
    fn llamacpp_windows_fetches_prism_prebuilt_without_toolchain() {
        // No git/cmake: fall back to Prism release download in-app.
        let steps = automated(ip(BackendKind::LlamaCpp, "windows", &no, false));
        assert!(matches!(&steps[0], InstallStep::FetchLlamaCpp { .. }));
        assert!(steps[0].display().contains("PrismML"));
    }

    #[test]
    fn llamacpp_linux_without_toolchain_fetches_prism_prebuilt() {
        let steps = automated(ip(BackendKind::LlamaCpp, "linux", &no, false));
        assert!(matches!(&steps[0], InstallStep::FetchLlamaCpp { .. }));
        assert!(steps[0].display().contains("PrismML"));
    }

    #[test]
    fn colibri_linux_with_toolchain_builds_from_source() {
        let has = |b: &str| matches!(b, "git" | "gcc");
        for kind in [BackendKind::Colibri, BackendKind::ColibriHy3] {
            let steps = automated(ip(kind, "linux", &has, false));
            assert!(
                matches!(&steps[0], InstallStep::BuildColibri { kind: k, .. } if *k == kind),
                "linux + git + gcc must source-build {kind:?}"
            );
        }
    }

    #[test]
    fn colibri_without_toolchain_is_manual() {
        // No binary releases exist; without git + a C compiler the only honest
        // plan is copy-paste steps pointing at the repo.
        let plan = ip(BackendKind::Colibri, "linux", &no, false);
        assert!(matches!(plan, InstallPlan::Manual { .. }));
    }

    #[test]
    fn colibri_windows_needs_mingw_else_manual() {
        // Upstream documents native Windows via MinGW-w64 (gcc + make).
        let mingw = |b: &str| matches!(b, "git" | "gcc" | "make");
        let steps = automated(ip(BackendKind::Colibri, "windows", &mingw, false));
        assert!(matches!(&steps[0], InstallStep::BuildColibri { .. }));

        let git_only = |b: &str| b == "git";
        let plan = ip(BackendKind::Colibri, "windows", &git_only, false);
        match plan {
            InstallPlan::Manual { steps, .. } => {
                assert!(steps.iter().any(|s| s.contains("MinGW")));
            }
            other => panic!("expected manual plan, got {other:?}"),
        }
    }

    #[test]
    fn colibri_hy3_is_manual_off_linux() {
        // The fork documents Linux/WSL only — never auto-build it on
        // Windows/macOS even with a full toolchain present.
        let all = |_: &str| true;
        for os in ["windows", "macos"] {
            let plan = ip(BackendKind::ColibriHy3, os, &all, false);
            match plan {
                InstallPlan::Manual { url, .. } => assert!(url.contains("colibri-hy3")),
                other => panic!("expected manual plan on {os}, got {other:?}"),
            }
        }
    }

    #[test]
    fn resolve_colibri_finds_managed_binary_per_kind() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_home(dir.path().to_path_buf());
        paths.ensure_dirs().unwrap();
        let bin_name = if cfg!(windows) { "coli.exe" } else { "coli" };
        for kind in [BackendKind::Colibri, BackendKind::ColibriHy3] {
            let managed = colibri_managed_dir(&paths, kind).join("src/c");
            std::fs::create_dir_all(&managed).unwrap();
            std::fs::write(managed.join(bin_name), b"x").unwrap();
        }
        // Each kind resolves its own tree — the fork never picks up upstream's.
        let up = resolve_colibri_bin("coli", &paths, BackendKind::Colibri).unwrap();
        let hy = resolve_colibri_bin("coli", &paths, BackendKind::ColibriHy3).unwrap();
        assert!(up.starts_with(colibri_managed_dir(&paths, BackendKind::Colibri)));
        assert!(hy.starts_with(colibri_managed_dir(&paths, BackendKind::ColibriHy3)));
    }

    #[test]
    fn resolve_colibri_prefers_explicit_file_path() {
        let dir = tempfile::tempdir().unwrap();
        let custom = dir.path().join("my-coli");
        std::fs::write(&custom, b"x").unwrap();
        let paths = AppPaths::from_home(dir.path().join("home"));
        paths.ensure_dirs().unwrap();
        let found =
            resolve_colibri_bin(custom.to_str().unwrap(), &paths, BackendKind::ColibriHy3).unwrap();
        assert_eq!(found, custom);
    }

    #[test]
    fn picks_cpu_windows_asset_when_cuda_not_preferred() {
        let assets = vec![
            ("llama-b100-bin-win-cuda-12.4-x64.zip".to_string(), "cuda-url".to_string()),
            ("llama-b100-bin-win-cpu-x64.zip".to_string(), "cpu-url".to_string()),
            ("llama-b100-bin-ubuntu-x64.zip".to_string(), "linux-url".to_string()),
        ];
        let (name, url) = pick_llamacpp_asset(&assets, "windows", "x86_64", false).unwrap();
        assert_eq!(url, "cpu-url");
        assert!(name.contains("cpu"));
    }

    #[test]
    fn picks_cuda_windows_asset_when_preferred() {
        let assets = vec![
            ("llama-prism-bin-win-cuda-12.4-x64.zip".to_string(), "cuda-url".to_string()),
            ("llama-bin-win-cpu-x64.zip".to_string(), "cpu-url".to_string()),
        ];
        let (_name, url) = pick_llamacpp_asset(&assets, "windows", "x86_64", true).unwrap();
        assert_eq!(url, "cuda-url");
    }

    #[test]
    fn picks_ubuntu_for_linux_and_nothing_for_unknown_os() {
        let assets = vec![
            ("llama-b100-bin-ubuntu-x64.tar.gz".to_string(), "linux-url".to_string()),
            ("llama-b100-bin-win-cpu-x64.zip".to_string(), "win-url".to_string()),
        ];
        assert_eq!(
            pick_llamacpp_asset(&assets, "linux", "x86_64", false)
                .unwrap()
                .1,
            "linux-url"
        );
        assert!(pick_llamacpp_asset(&assets, "freebsd", "x86_64", false).is_none());
    }

    #[test]
    fn linux_requires_targz_not_zip() {
        // Regression: llama.cpp ships Linux builds as .tar.gz, not .zip. A .zip
        // ubuntu asset must NOT be picked (that mismatch is why the in-app fetch
        // silently found nothing on Linux), while the .tar.gz is.
        let zip_only = vec![(
            "llama-b100-bin-ubuntu-x64.zip".to_string(),
            "zip".to_string(),
        )];
        assert!(pick_llamacpp_asset(&zip_only, "linux", "x86_64", false).is_none());

        let both = vec![
            ("llama-b100-bin-ubuntu-x64.zip".to_string(), "zip".to_string()),
            ("llama-b100-bin-ubuntu-x64.tar.gz".to_string(), "targz".to_string()),
        ];
        assert_eq!(
            pick_llamacpp_asset(&both, "linux", "x86_64", false)
                .unwrap()
                .1,
            "targz"
        );
    }

    #[test]
    fn excludes_openvino_vulkan_rocm_on_linux() {
        // Accelerator packs that need special drivers are never auto-picked;
        // plain CPU (or CUDA when preferred) wins.
        let assets = vec![
            ("llama-b100-bin-ubuntu-openvino-2026.2.1-x64.tar.gz".into(), "openvino".into()),
            ("llama-b100-bin-ubuntu-vulkan-x64.tar.gz".to_string(), "vulkan".into()),
            ("llama-b100-bin-ubuntu-rocm-7.2-x64.tar.gz".to_string(), "rocm".into()),
            ("llama-b100-bin-ubuntu-x64.tar.gz".to_string(), "cpu".to_string()),
        ];
        assert_eq!(
            pick_llamacpp_asset(&assets, "linux", "x86_64", false)
                .unwrap()
                .1,
            "cpu"
        );
    }

    #[test]
    fn picks_macos_targz() {
        let assets = vec![
            ("llama-b100-bin-macos-arm64.tar.gz".to_string(), "arm".to_string()),
            ("llama-b100-bin-macos-x64.tar.gz".to_string(), "x64".to_string()),
        ];
        assert_eq!(
            pick_llamacpp_asset(&assets, "macos", "aarch64", false)
                .unwrap()
                .1,
            "arm"
        );
        assert_eq!(
            pick_llamacpp_asset(&assets, "macos", "x86_64", false)
                .unwrap()
                .1,
            "x64"
        );
    }

    #[test]
    fn resolve_prism_requires_marker() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_home(dir.path().to_path_buf());
        paths.ensure_dirs().unwrap();
        let managed = paths.llamacpp_dir();
        std::fs::create_dir_all(managed.join("build/bin")).unwrap();
        let server = if cfg!(windows) {
            managed.join("build/bin/llama-server.exe")
        } else {
            managed.join("build/bin/llama-server")
        };
        std::fs::write(&server, b"x").unwrap();
        // Without marker → not treated as Prism.
        assert!(resolve_prism_llamacpp_bin(&paths).is_none());
        write_prism_marker(&managed).unwrap();
        assert_eq!(resolve_prism_llamacpp_bin(&paths).unwrap(), server);
    }

    #[test]
    fn extract_targz_finds_nested_server_binary() {
        use std::io::Write;
        // A real llama.cpp tarball nests the binary under build/bin; the DFS in
        // find_file must locate it. (Traversal safety is delegated to tar's
        // `unpack_in`, which refuses entries that escape the destination — and
        // the Builder here won't even let us forge a `..` entry to test it.)
        let mut tar_bytes = Vec::new();
        {
            let mut b = tar::Builder::new(&mut tar_bytes);
            let data = b"binary";
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o755);
            header.set_mtime(0);
            b.append_data(&mut header, "build/bin/llama-server", &data[..])
                .unwrap();
            b.finish().unwrap();
        }
        let mut gz = Vec::new();
        {
            let mut enc =
                flate2::write::GzEncoder::new(&mut gz, flate2::Compression::default());
            enc.write_all(&tar_bytes).unwrap();
            enc.finish().unwrap();
        }
        let dir = tempfile::tempdir().unwrap();
        let bin = extract_llamacpp_targz(&gz, dir.path()).unwrap();
        assert!(bin.exists());
        assert_eq!(bin.file_name().unwrap().to_string_lossy(), "llama-server");
    }

    #[test]
    fn archive_dispatch_rejects_unknown_format() {
        assert!(extract_llamacpp_archive(b"x", std::path::Path::new("/tmp/x"), "foo.rar").is_err());
    }

    #[test]
    fn refuses_vulkan_only_release_without_cpu_or_cuda() {
        // Vulkan/HIP packs alone are not auto-installed (driver matrix).
        let assets = vec![
            ("llama-b100-bin-win-vulkan-x64.zip".to_string(), "v".to_string()),
            ("llama-b100-bin-win-hip-radeon-x64.zip".to_string(), "h".to_string()),
        ];
        assert!(pick_llamacpp_asset(&assets, "windows", "x86_64", false).is_none());
        assert!(pick_llamacpp_asset(&assets, "windows", "x86_64", true).is_none());
    }

    #[test]
    fn cuda_only_release_picked_when_cuda_preferred() {
        let assets = vec![
            ("llama-prism-bin-win-cuda-12.4-x64.zip".to_string(), "c".to_string()),
        ];
        assert_eq!(
            pick_llamacpp_asset(&assets, "windows", "x86_64", true)
                .unwrap()
                .1,
            "c"
        );
        // Without prefer_cuda we still accept CUDA as last resort if it's the
        // only OS-matching archive.
        assert_eq!(
            pick_llamacpp_asset(&assets, "windows", "x86_64", false)
                .unwrap()
                .1,
            "c"
        );
    }

    #[test]
    fn resolve_finds_managed_binary_without_path() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_home(dir.path().to_path_buf());
        paths.ensure_dirs().unwrap();
        let managed = paths.llamacpp_dir();
        std::fs::create_dir_all(managed.join("build/bin")).unwrap();
        let server = if cfg!(windows) {
            managed.join("build/bin/llama-server.exe")
        } else {
            managed.join("build/bin/llama-server")
        };
        std::fs::write(&server, b"x").unwrap();
        let found = resolve_llamacpp_bin("llama-server", &paths).unwrap();
        assert_eq!(found, server);
    }

    #[test]
    fn resolve_prefers_explicit_file_path() {
        let dir = tempfile::tempdir().unwrap();
        let custom = dir.path().join("my-llama-server");
        std::fs::write(&custom, b"x").unwrap();
        let paths = AppPaths::from_home(dir.path().join("home"));
        paths.ensure_dirs().unwrap();
        let found = resolve_llamacpp_bin(custom.to_str().unwrap(), &paths).unwrap();
        assert_eq!(found, custom);
    }

    #[test]
    fn extract_finds_server_binary_and_is_zip_slip_safe() {
        use std::io::Write;
        let mut cur = std::io::Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut cur);
            let opts = zip::write::SimpleFileOptions::default();
            let server = if cfg!(windows) {
                "build/bin/llama-server.exe"
            } else {
                "build/bin/llama-server"
            };
            w.start_file(server, opts).unwrap();
            w.write_all(b"binary").unwrap();
            // A traversal entry that must be skipped, not written outside dest.
            w.start_file("../evil.txt", opts).unwrap();
            w.write_all(b"nope").unwrap();
            w.finish().unwrap();
        }
        let bytes = cur.into_inner();
        let dir = tempfile::tempdir().unwrap();
        let bin = extract_llamacpp_zip(&bytes, dir.path()).unwrap();
        assert!(bin.exists());
        assert!(bin
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("llama-server"));
        assert!(!dir.path().parent().unwrap().join("evil.txt").exists());
    }

    // ---- repair resolution (path manipulation only; no filesystem / no sudo) ----

    use std::path::Path;

    fn paths() -> AppPaths {
        // from_home does no IO; resolve_repair only joins path strings.
        AppPaths::from_home(PathBuf::from("/home/test/.localcode"))
    }

    #[test]
    fn venv_repair_builds_three_steps_and_repoints_vllm() {
        let plan = resolve_repair_with(
            &RepairIntent::CleanVenvReinstall,
            BackendKind::Vllm,
            &paths(),
            "linux",
            &no,
            Some(PathBuf::from("/usr/bin/python3")),
        )
        .unwrap();
        assert_eq!(plan.steps.len(), 3);
        assert!(plan.steps[0].display().contains("-m venv"));
        assert!(plan.steps[0].display().contains("venvs"));
        assert!(plan.steps[2].display().contains("install vllm"));
        assert!(!plan.requires_sudo());
        let rp = plan.repoint.expect("repoint");
        assert_eq!(rp.kind, BackendKind::Vllm);
        assert!(Path::new(&rp.bin).ends_with("vllm"));
        assert!(rp.bin.contains("bin"));
    }

    #[test]
    fn venv_repair_uses_sglang_all_and_repoints_python() {
        let plan = resolve_repair_with(
            &RepairIntent::CleanVenvReinstall,
            BackendKind::Sglang,
            &paths(),
            "linux",
            &no,
            Some(PathBuf::from("/usr/bin/python3")),
        )
        .unwrap();
        assert!(plan.steps[2].display().contains("sglang[all]"));
        // SGLang launches via `<python> -m sglang.launch_server`, so its bin is
        // the venv interpreter, not a console script.
        assert!(Path::new(&plan.repoint.unwrap().bin).ends_with("python"));
    }

    #[test]
    fn venv_repair_without_python_errors() {
        let err = resolve_repair_with(
            &RepairIntent::CleanVenvReinstall,
            BackendKind::Vllm,
            &paths(),
            "linux",
            &no,
            None,
        )
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::BackendInstallFailed);
    }

    #[test]
    fn ollama_service_repair_is_sudo_on_linux_and_leaks_no_password() {
        let plan = resolve_repair_with(
            &RepairIntent::StartOllamaService,
            BackendKind::Ollama,
            &paths(),
            "linux",
            &no,
            None,
        )
        .unwrap();
        assert!(plan.requires_sudo());
        assert!(plan.steps[0].display().starts_with("sudo systemctl start ollama"));
        // The approval preview must never contain the word "password".
        assert!(!plan.display().to_lowercase().contains("password"));
    }

    #[test]
    fn ollama_service_repair_manual_when_no_service_manager() {
        // Windows / macOS-without-brew: no automated start.
        assert!(resolve_repair_with(
            &RepairIntent::StartOllamaService,
            BackendKind::Ollama,
            &paths(),
            "windows",
            &no,
            None,
        )
        .is_err());
    }
}
