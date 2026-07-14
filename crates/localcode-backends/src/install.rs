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
use std::path::PathBuf;
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

    /// A `sh -c "<script>"` step whose preview is the script itself, so a piped
    /// command reads honestly (`curl … | sh`) instead of `sh -c curl … | sh`.
    fn shell(script: &str) -> Self {
        InstallStep::Command {
            program: "sh".into(),
            args: vec!["-c".into(), script.into()],
            display: script.into(),
        }
    }

    /// Line shown in the confirm dialog / plan preview.
    pub fn display(&self) -> String {
        match self {
            InstallStep::Command { display, .. } => display.clone(),
            InstallStep::PipInstall { package } => format!("python -m pip install {package}"),
            InstallStep::Sudo { display, .. } => display.clone(),
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
}

/// Resolve an install plan. Pure: OS, tool availability (`has`), and whether
/// Python is already present are injected, so it's fully unit-testable.
pub fn install_plan(
    kind: BackendKind,
    os: &str,
    has: &dyn Fn(&str) -> bool,
    has_python: bool,
) -> InstallPlan {
    match kind {
        BackendKind::Ollama => ollama_plan(os, has),
        BackendKind::LlamaCpp => llamacpp_plan(os, has),
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
        "linux" => InstallPlan::automated(vec![InstallStep::shell(
            "curl -fsSL https://ollama.com/install.sh | sh",
        )]),
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

fn llamacpp_plan(os: &str, has: &dyn Fn(&str) -> bool) -> InstallPlan {
    if matches!(os, "macos" | "linux") && has("brew") {
        return InstallPlan::automated(vec![InstallStep::command(
            "brew",
            &["install", "llama.cpp"],
        )]);
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

/// Resolve an install plan against the real environment.
pub fn resolve_install_plan(kind: BackendKind) -> InstallPlan {
    let has = |b: &str| which::which(b).is_ok();
    let plan = install_plan(kind, std::env::consts::OS, &has, discover_python().is_some());
    // The official Ollama Linux installer runs `sudo` internally. Our install
    // steps spawn with stdin null, so if we can't elevate without a password
    // that `sudo` just fails ("a password is required") — a confusing failure.
    // Per the module's honesty rule, offer manual steps instead unless we can
    // elevate non-interactively (already root, or passwordless sudo).
    if kind == BackendKind::Ollama
        && std::env::consts::OS == "linux"
        && !can_elevate_noninteractively()
    {
        return InstallPlan::Manual {
            summary: "The Ollama installer needs root (it runs `sudo`), which can't be entered here."
                .into(),
            steps: vec![
                "Run this in a terminal where sudo can prompt for your password:".into(),
                "curl -fsSL https://ollama.com/install.sh | sh".into(),
                "Ollama starts a local service automatically — then re-detect.".into(),
            ],
            url: "https://ollama.com/download/linux".into(),
        };
    }
    plan
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
/// so passing one here is an error.
pub async fn run_install(
    plan: &InstallPlan,
    progress: UnboundedSender<String>,
) -> Result<(), LocalCodeError> {
    let InstallPlan::Automated { steps, .. } = plan else {
        return Err(LocalCodeError::new(
            ErrorCode::BackendInstallFailed,
            "No automated installer for this backend on this platform",
        )
        .with_hint("Follow the manual steps shown in the Backends panel"));
    };
    let cid = localcode_core::CorrelationId::new();
    for step in steps {
        run_step(step, &progress, cid).await?;
    }
    Ok(())
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

    #[test]
    fn ollama_linux_uses_official_script() {
        let steps = automated(install_plan(BackendKind::Ollama, "linux", &no, false));
        assert_eq!(steps.len(), 1);
        assert!(steps[0].display().contains("ollama.com/install.sh"));
    }

    #[test]
    fn ollama_windows_winget() {
        let has = |b: &str| b == "winget";
        let steps = automated(install_plan(BackendKind::Ollama, "windows", &has, false));
        let d = steps[0].display();
        assert!(d.contains("winget"));
        assert!(d.contains("Ollama.Ollama"));
        // Non-interactive + user scope so no UAC/password prompt can block it.
        assert!(d.contains("--disable-interactivity"));
        assert!(d.contains("--scope user"));
    }

    #[test]
    fn ollama_windows_without_winget_is_manual() {
        let plan = install_plan(BackendKind::Ollama, "windows", &no, false);
        assert!(matches!(plan, InstallPlan::Manual { .. }));
    }

    #[test]
    fn ollama_macos_brew() {
        let has = |b: &str| b == "brew";
        let steps = automated(install_plan(BackendKind::Ollama, "macos", &has, false));
        assert!(steps[0].display().contains("brew install ollama"));
    }

    #[test]
    fn vllm_prepends_python_when_missing_on_windows() {
        let has = |b: &str| b == "winget";
        let steps = automated(install_plan(BackendKind::Vllm, "windows", &has, false));
        assert_eq!(steps.len(), 2);
        assert!(steps[0].display().to_lowercase().contains("python"));
        assert!(matches!(&steps[1], InstallStep::PipInstall { package } if package == "vllm"));
    }

    #[test]
    fn vllm_macos_brew_installs_python_prereq() {
        let has = |b: &str| b == "brew";
        let steps = automated(install_plan(BackendKind::Vllm, "macos", &has, false));
        assert_eq!(steps.len(), 2);
        assert!(steps[0].display().contains("brew install python"));
    }

    #[test]
    fn vllm_single_step_when_python_present() {
        let steps = automated(install_plan(BackendKind::Vllm, "windows", &no, true));
        assert_eq!(steps.len(), 1);
        assert!(matches!(&steps[0], InstallStep::PipInstall { .. }));
    }

    #[test]
    fn vllm_linux_without_python_is_manual() {
        // No package manager can install Python non-interactively (needs sudo).
        let plan = install_plan(BackendKind::Vllm, "linux", &no, false);
        assert!(matches!(plan, InstallPlan::Manual { .. }));
    }

    #[test]
    fn sglang_pip_package_has_all_extra() {
        let steps = automated(install_plan(BackendKind::Sglang, "linux", &no, true));
        assert!(matches!(&steps[0], InstallStep::PipInstall { package } if package == "sglang[all]"));
    }

    #[test]
    fn llamacpp_brew_when_available() {
        let has = |b: &str| b == "brew";
        let steps = automated(install_plan(BackendKind::LlamaCpp, "macos", &has, false));
        assert!(steps[0].display().contains("brew install llama.cpp"));
    }

    #[test]
    fn llamacpp_windows_is_manual() {
        let plan = install_plan(BackendKind::LlamaCpp, "windows", &no, false);
        assert!(matches!(plan, InstallPlan::Manual { .. }));
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
