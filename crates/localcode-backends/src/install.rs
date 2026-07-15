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
    /// Download a prebuilt llama.cpp release for this OS and extract it into
    /// `dest_dir`, then repoint the backend at the extracted `llama-server`
    /// binary. Runs entirely in-process (HTTP + unarchive — `.zip` on Windows,
    /// `.tar.gz` on Linux/macOS) so llama.cpp installs from the TUI on platforms
    /// without a package manager.
    FetchLlamaCpp { dest_dir: PathBuf, display: String },
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
pub fn install_plan(
    kind: BackendKind,
    os: &str,
    has: &dyn Fn(&str) -> bool,
    has_python: bool,
    llamacpp_dir: &Path,
) -> InstallPlan {
    match kind {
        BackendKind::Ollama => ollama_plan(os, has),
        BackendKind::LlamaCpp => llamacpp_plan(os, has, llamacpp_dir),
        BackendKind::Vllm => pip_backend_plan("vllm", os, has, has_python),
        BackendKind::Sglang => pip_backend_plan("sglang[all]", os, has, has_python),
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
    // A package manager gives the cleanest, self-updating install when present.
    if matches!(os, "macos" | "linux") && has("brew") {
        return InstallPlan::automated(vec![InstallStep::command(
            "brew",
            &["install", "llama.cpp"],
        )]);
    }
    // Otherwise fetch a prebuilt release and extract it in-app (works on
    // Windows and package-manager-less Linux/macOS). CPU build: universally
    // compatible — GPU users can still use Ollama/vLLM.
    if matches!(os, "windows" | "linux" | "macos") {
        return InstallPlan::automated(vec![InstallStep::FetchLlamaCpp {
            dest_dir: dest_dir.to_path_buf(),
            display: format!(
                "Download prebuilt llama.cpp (CPU) from github.com/ggml-org/llama.cpp and install to {}",
                dest_dir.display()
            ),
        }]);
    }
    InstallPlan::Manual {
        summary: "Download a prebuilt llama.cpp server binary and point config at it.".into(),
        steps: vec![
            "Download the latest release build for your OS/GPU".into(),
            "Unzip it, then put llama-server(.exe) on PATH or set backends.llamacpp.bin".into(),
        ],
        url: "https://github.com/ggml-org/llama.cpp/releases".into(),
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

/// Ensure a usable `llama-server` exists: reuse PATH / managed install, or run
/// the platform install plan (Homebrew or in-app FetchLlamaCpp).
///
/// Returns the absolute path to the binary. Callers should persist it as
/// `backends.llamacpp.bin` so later runs do not depend on PATH.
pub async fn ensure_llamacpp_installed(
    paths: &AppPaths,
    progress: UnboundedSender<String>,
) -> Result<PathBuf, LocalCodeError> {
    paths.ensure_dirs()?;

    if let Some(p) = resolve_llamacpp_bin("llama-server", paths) {
        let _ = progress.send(format!(
            "llama-server already available at {}",
            p.display()
        ));
        return Ok(p);
    }

    let plan = resolve_install_plan(BackendKind::LlamaCpp, paths);
    match &plan {
        InstallPlan::Automated { display, .. } => {
            let _ = progress.send(format!("Installing llama.cpp…\n{display}"));
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
                "Install llama-server manually, then set backends.llamacpp.bin in config.toml",
            ));
        }
    }

    let repoint = run_install(&plan, None, progress.clone()).await?;
    if let Some(r) = repoint {
        return Ok(PathBuf::from(r.bin));
    }

    // Package-manager installs (e.g. brew) leave the binary on PATH without a repoint.
    if let Some(p) = resolve_llamacpp_bin("llama-server", paths) {
        let _ = progress.send(format!("llama-server ready at {}", p.display()));
        return Ok(p);
    }

    Err(LocalCodeError::new(
        ErrorCode::BackendBinaryMissing,
        "llama-server not found after install",
    )
    .with_hint("Install llama.cpp from the Backends panel or set backends.llamacpp.bin")
    .retryable(true))
}

/// Resolve an install plan against the real environment.
pub fn resolve_install_plan(kind: BackendKind, paths: &AppPaths) -> InstallPlan {
    let has = |b: &str| which::which(b).is_ok();
    let llamacpp_dir = llamacpp_managed_dir(paths);
    // The Ollama-on-Linux plan now emits an elevated (`Sudo`) step instead of
    // being downgraded to manual copy-paste here: the runner elevates with
    // `sudo -n` when that works and `sudo -S` (password from the masked prompt)
    // otherwise, so the install runs from the TUI. See `ollama_plan`.
    install_plan(
        kind,
        std::env::consts::OS,
        &has,
        discover_python().is_some(),
        &llamacpp_dir,
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
        InstallStep::FetchLlamaCpp { display, .. } => {
            return Err(LocalCodeError::new(
                ErrorCode::BackendInstallFailed,
                "Internal: a fetch step reached the generic step runner",
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
// llama.cpp prebuilt fetch (in-app download + extract, no package manager)
// ----------------------------------------------------------------------------

/// GitHub API for the latest llama.cpp release.
const LLAMACPP_RELEASES_API: &str =
    "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";

/// Download and extract a prebuilt llama.cpp release into `dest_dir`, returning
/// the absolute path to the extracted `llama-server` binary. Honest failure — a
/// clear, retryable error when no matching asset exists or extraction fails —
/// never a fabricated success.
async fn fetch_llamacpp(
    dest_dir: &Path,
    progress: &UnboundedSender<String>,
    cid: localcode_core::CorrelationId,
) -> Result<String, LocalCodeError> {
    let fail = |msg: String| {
        LocalCodeError::new(ErrorCode::BackendInstallFailed, msg)
            .with_correlation(cid)
            .with_hint("Or install llama.cpp manually from github.com/ggml-org/llama.cpp/releases")
            .retryable(true)
    };

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        // GitHub's API rejects requests without a User-Agent.
        .user_agent("localcode-installer")
        .build()
        .map_err(|e| fail(e.to_string()))?;

    let _ = progress.send("$ querying github.com/ggml-org/llama.cpp for the latest release".into());
    let resp = client
        .get(LLAMACPP_RELEASES_API)
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

    let (asset_name, asset_url) =
        pick_llamacpp_asset(&assets, std::env::consts::OS, std::env::consts::ARCH).ok_or_else(
            || fail("The latest release has no prebuilt CPU build for this OS/arch".into()),
        )?;

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

    let _ = progress.send(format!("$ extracting {} ({} MiB)", asset_name, bytes.len() / 1_048_576));
    let dest = dest_dir.to_path_buf();
    let name = asset_name.clone();
    let bin = tokio::task::spawn_blocking(move || extract_llamacpp_archive(&bytes, &dest, &name))
        .await
        .map_err(|e| fail(e.to_string()))?
        .map_err(fail)?;

    let _ = progress.send(format!("llama-server installed to {}", bin.display()));
    Ok(bin.display().to_string())
}

/// Pick the CPU release asset matching this OS/arch, preferring the most
/// specific name. Pure so the matching rules are unit-tested. Returns
/// `(name, download_url)`. GPU/accelerator builds (CUDA/HIP/Vulkan/OpenVINO/…)
/// are excluded — the CPU build runs everywhere; GPU users can use Ollama/vLLM
/// instead.
///
/// Archive format is OS-specific: llama.cpp ships Windows builds as `.zip` and
/// Linux/macOS builds as `.tar.gz`. Matching on the wrong extension is why the
/// in-app fetch silently found nothing on Linux — so the accepted extensions
/// are chosen per-OS here and honored by [`extract_llamacpp_archive`].
fn pick_llamacpp_asset(
    assets: &[(String, String)],
    os: &str,
    arch: &str,
) -> Option<(String, String)> {
    const GPU_MARKERS: [&str; 10] = [
        "cuda", "cu11", "cu12", "hip", "rocm", "vulkan", "sycl", "musa", "kompute", "openvino",
    ];
    let exts: &[&str] = if os == "windows" {
        &[".zip"]
    } else {
        &[".tar.gz", ".tgz"]
    };
    let cpu_archive = |n: &str| {
        let l = n.to_lowercase();
        exts.iter().any(|e| l.ends_with(e)) && !GPU_MARKERS.iter().any(|m| l.contains(m))
    };
    let all = |n: &str, pats: &[&str]| {
        let l = n.to_lowercase();
        pats.iter().all(|p| l.contains(p))
    };
    let prefs: Vec<Vec<&str>> = match (os, arch) {
        ("windows", "aarch64") => vec![vec!["win", "arm64"], vec!["bin-win"]],
        ("windows", _) => vec![vec!["win", "x64"], vec!["bin-win"]],
        ("linux", "aarch64") => vec![vec!["ubuntu", "arm64"], vec!["linux", "arm64"]],
        ("linux", _) => vec![vec!["ubuntu", "x64"], vec!["ubuntu"], vec!["linux", "x64"]],
        ("macos", "aarch64") => vec![vec!["macos", "arm64"], vec!["bin-macos"]],
        ("macos", _) => vec![vec!["macos", "x64"], vec!["bin-macos"]],
        _ => return None,
    };
    for pats in &prefs {
        if let Some((n, u)) = assets
            .iter()
            .find(|(n, _)| cpu_archive(n) && all(n, pats))
        {
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
    fn llamacpp_brew_when_available() {
        let has = |b: &str| b == "brew";
        let steps = automated(ip(BackendKind::LlamaCpp, "macos", &has, false));
        assert!(steps[0].display().contains("brew install llama.cpp"));
    }

    #[test]
    fn llamacpp_windows_fetches_prebuilt_in_app() {
        // No package manager on Windows: install by downloading a prebuilt
        // release in-app, so the user never has to leave the TUI.
        let steps = automated(ip(BackendKind::LlamaCpp, "windows", &no, false));
        assert!(matches!(&steps[0], InstallStep::FetchLlamaCpp { .. }));
        assert!(steps[0].display().contains("llama.cpp"));
    }

    #[test]
    fn llamacpp_linux_without_brew_fetches_prebuilt() {
        let steps = automated(ip(BackendKind::LlamaCpp, "linux", &no, false));
        assert!(matches!(&steps[0], InstallStep::FetchLlamaCpp { .. }));
    }

    #[test]
    fn picks_cpu_windows_asset_over_gpu_builds() {
        let assets = vec![
            ("llama-b100-bin-win-cuda-12.4-x64.zip".to_string(), "cuda-url".to_string()),
            ("llama-b100-bin-win-cpu-x64.zip".to_string(), "cpu-url".to_string()),
            ("llama-b100-bin-ubuntu-x64.zip".to_string(), "linux-url".to_string()),
        ];
        let (name, url) = pick_llamacpp_asset(&assets, "windows", "x86_64").unwrap();
        assert_eq!(url, "cpu-url");
        assert!(name.contains("cpu"));
    }

    #[test]
    fn picks_ubuntu_for_linux_and_nothing_for_unknown_os() {
        let assets = vec![
            ("llama-b100-bin-ubuntu-x64.tar.gz".to_string(), "linux-url".to_string()),
            ("llama-b100-bin-win-cpu-x64.zip".to_string(), "win-url".to_string()),
        ];
        assert_eq!(
            pick_llamacpp_asset(&assets, "linux", "x86_64").unwrap().1,
            "linux-url"
        );
        assert!(pick_llamacpp_asset(&assets, "freebsd", "x86_64").is_none());
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
        assert!(pick_llamacpp_asset(&zip_only, "linux", "x86_64").is_none());

        let both = vec![
            ("llama-b100-bin-ubuntu-x64.zip".to_string(), "zip".to_string()),
            ("llama-b100-bin-ubuntu-x64.tar.gz".to_string(), "targz".to_string()),
        ];
        assert_eq!(
            pick_llamacpp_asset(&both, "linux", "x86_64").unwrap().1,
            "targz"
        );
    }

    #[test]
    fn excludes_openvino_and_gpu_builds_on_linux() {
        // The real release lists accelerator builds (openvino, vulkan, rocm,
        // sycl) *before* the plain CPU tarball; the plain ubuntu-x64 build must
        // still win. openvino in particular is not a GPU marker by name, so it
        // was added explicitly.
        let assets = vec![
            ("llama-b100-bin-ubuntu-openvino-2026.2.1-x64.tar.gz".into(), "openvino".into()),
            ("llama-b100-bin-ubuntu-vulkan-x64.tar.gz".to_string(), "vulkan".into()),
            ("llama-b100-bin-ubuntu-rocm-7.2-x64.tar.gz".to_string(), "rocm".into()),
            ("llama-b100-bin-ubuntu-x64.tar.gz".to_string(), "cpu".to_string()),
        ];
        assert_eq!(
            pick_llamacpp_asset(&assets, "linux", "x86_64").unwrap().1,
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
            pick_llamacpp_asset(&assets, "macos", "aarch64").unwrap().1,
            "arm"
        );
        assert_eq!(
            pick_llamacpp_asset(&assets, "macos", "x86_64").unwrap().1,
            "x64"
        );
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
    fn refuses_to_pick_a_gpu_only_release() {
        // No CPU asset available → no silent GPU install; caller shows an error.
        let assets = vec![
            ("llama-b100-bin-win-cuda-x64.zip".to_string(), "c".to_string()),
            ("llama-b100-bin-win-vulkan-x64.zip".to_string(), "v".to_string()),
        ];
        assert!(pick_llamacpp_asset(&assets, "windows", "x86_64").is_none());
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
