//! Thin wrapper over the `docker` CLI for benchmark containers.
//!
//! Same conventions as the OpenWebUI integration: shell out to `docker`
//! (found via PATH), manage containers by name, `docker rm -f` to clean up.
//! Every bench container carries the `localcode.bench` label so stale ones
//! from a crashed or cancelled run can be swept.

use localcode_core::error::{ErrorCode, LocalCodeError};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;

/// Label key stamped on every container the bench runner creates.
pub const BENCH_LABEL: &str = "localcode.bench";

/// Resource caps for task/grade containers. Generous for small tasks while
/// keeping a runaway build or fork bomb from taking the machine down.
const MEMORY_LIMIT: &str = "4g";
const CPU_LIMIT: &str = "2";
const PIDS_LIMIT: &str = "512";

/// True when the `docker` CLI is on PATH.
pub fn docker_available() -> bool {
    which::which("docker").is_ok()
}

pub fn docker_missing_err() -> LocalCodeError {
    LocalCodeError::new(
        ErrorCode::BackendBinaryMissing,
        "Docker is required to run benchmarks",
    )
    .with_cause("`docker` was not found on PATH")
    .with_hint("Install Docker Desktop (Windows/macOS) or Docker Engine (Linux)")
}

/// A bind-mount source docker accepts: canonical absolute path without the
/// Windows `\\?\` verbatim prefix (docker's CLI rejects verbatim paths).
pub fn mount_src(path: &Path) -> Result<String, LocalCodeError> {
    let canon = std::fs::canonicalize(path).map_err(|e| {
        LocalCodeError::new(ErrorCode::ConfigLoadFailed, e.to_string())
            .with_cause(format!("Cannot resolve mount path {}", path.display()))
    })?;
    let s = canon.display().to_string();
    Ok(s.strip_prefix(r"\\?\").unwrap_or(&s).to_string())
}

async fn docker(args: &[&str]) -> Result<std::process::Output, LocalCodeError> {
    tokio::process::Command::new("docker")
        .args(args)
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|e| {
            LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                .with_cause("Failed to invoke docker")
                .with_hint("Is the Docker daemon running?")
                .retryable(true)
        })
}

fn docker_failed(what: &str, out: &std::process::Output) -> LocalCodeError {
    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
    LocalCodeError::new(ErrorCode::BackendStartFailed, format!("docker {what} failed"))
        .with_cause(if stderr.is_empty() {
            "docker exited non-zero".into()
        } else {
            stderr
        })
        .with_hint("Is the Docker daemon running?")
        .retryable(true)
}

pub async fn image_present(image: &str) -> bool {
    docker(&["image", "inspect", image])
        .await
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `docker pull` — needs daemon-side network regardless of task network mode.
pub async fn pull_image(image: &str) -> Result<(), LocalCodeError> {
    let out = docker(&["pull", image]).await?;
    if !out.status.success() {
        return Err(docker_failed(&format!("pull {image}"), &out)
            .with_hint("Image pulls need network; pre-pull suites with `localcode bench pull`"));
    }
    Ok(())
}

/// Start the long-lived task container the agent's `bash` calls exec into.
/// The workspace is bind-mounted at `/workspace`, which is also the working
/// directory. The container idles on a long sleep and is removed with
/// [`remove_container`] when the task ends.
pub async fn start_task_container(
    name: &str,
    image: &str,
    workspace: &Path,
    network: &str,
    run_id: &str,
) -> Result<(), LocalCodeError> {
    let mount = format!(
        "type=bind,src={},dst={}",
        mount_src(workspace)?,
        crate::spec::CONTAINER_WORKSPACE
    );
    let label = format!("{BENCH_LABEL}={run_id}");
    let net = format!("--network={network}");
    let mem = format!("--memory={MEMORY_LIMIT}");
    let cpus = format!("--cpus={CPU_LIMIT}");
    let pids = format!("--pids-limit={PIDS_LIMIT}");
    let out = docker(&[
        "run",
        "-d",
        "--name",
        name,
        "--label",
        &label,
        &net,
        &mem,
        &cpus,
        &pids,
        "--mount",
        &mount,
        "-w",
        crate::spec::CONTAINER_WORKSPACE,
        image,
        "sh",
        "-c",
        // Portable idle (busybox `sleep` rejects `infinity`).
        "sleep 2147483647",
    ])
    .await?;
    if !out.status.success() {
        return Err(docker_failed(&format!("run {image}"), &out));
    }
    Ok(())
}

/// Stop and remove a container by name (best-effort).
pub async fn remove_container(name: &str) {
    let _ = docker(&["rm", "-f", name]).await;
}

/// Remove every container carrying the bench label — orphans left by a crash
/// or a cancelled run. Only one bench run exists at a time, so a sweep at run
/// start (and on cancel) is safe.
pub async fn cleanup_stale_containers() {
    let filter = format!("label={BENCH_LABEL}");
    let Ok(out) = docker(&["ps", "-aq", "--filter", &filter]).await else {
        return;
    };
    let ids: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    for id in ids {
        let _ = docker(&["rm", "-f", &id]).await;
    }
}

/// Outcome of one grader check command.
#[derive(Debug, Clone)]
pub struct CheckOutcome {
    pub exit_code: i32,
    pub output: String,
    pub timed_out: bool,
}

/// Run one grader check in a fresh container over `workspace` and capture the
/// combined output. The container is named, so a wall-clock timeout can
/// `docker rm -f` it (killing the CLI client alone would leave the container
/// running).
pub async fn run_check_container(
    image: &str,
    workspace: &Path,
    network: &str,
    cmd: &str,
    timeout: Duration,
    run_id: &str,
) -> Result<CheckOutcome, LocalCodeError> {
    let name = format!("lcb-grade-{}", &uuid::Uuid::new_v4().to_string()[..8]);
    let mount = format!(
        "type=bind,src={},dst={}",
        mount_src(workspace)?,
        crate::spec::CONTAINER_WORKSPACE
    );
    let label = format!("{BENCH_LABEL}={run_id}");
    let net = format!("--network={network}");
    let mem = format!("--memory={MEMORY_LIMIT}");
    let cpus = format!("--cpus={CPU_LIMIT}");
    let pids = format!("--pids-limit={PIDS_LIMIT}");

    let mut child = tokio::process::Command::new("docker")
        .args([
            "run",
            "--rm",
            "--name",
            &name,
            "--label",
            &label,
            &net,
            &mem,
            &cpus,
            &pids,
            "--mount",
            &mount,
            "-w",
            crate::spec::CONTAINER_WORKSPACE,
            image,
            // Non-login shell: `sh -l` sources /etc/profile, which RESETS
            // PATH and hides image-provided toolchains (rust:1-slim keeps
            // cargo in /usr/local/cargo/bin via an ENV that a login shell
            // clobbers).
            "sh",
            "-c",
            cmd,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| {
            LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                .with_cause("Failed to invoke docker")
                .retryable(true)
        })?;

    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let drain = async {
        let mut out_buf = Vec::new();
        let mut err_buf = Vec::new();
        if let Some(o) = &mut stdout {
            let _ = o.read_to_end(&mut out_buf).await;
        }
        if let Some(e) = &mut stderr {
            let _ = e.read_to_end(&mut err_buf).await;
        }
        (out_buf, err_buf)
    };

    let (status, bufs, timed_out) = tokio::select! {
        res = async {
            let bufs = drain.await;
            let status = child.wait().await;
            (status, bufs)
        } => (res.0.ok(), res.1, false),
        _ = tokio::time::sleep(timeout) => {
            // Kill the container (removes it too, --rm), then reap the client.
            remove_container(&name).await;
            let _ = child.kill().await;
            let _ = child.wait().await;
            (None, (Vec::new(), Vec::new()), true)
        }
    };

    let mut output = String::from_utf8_lossy(&bufs.0).to_string();
    let err = String::from_utf8_lossy(&bufs.1);
    if !err.is_empty() {
        output.push_str("\nSTDERR:\n");
        output.push_str(&err);
    }
    if timed_out {
        output.push_str(&format!(
            "\n[check timed out after {}s and its container was removed]",
            timeout.as_secs()
        ));
    }
    Ok(CheckOutcome {
        exit_code: status.and_then(|s| s.code()).unwrap_or(-1),
        output,
        timed_out,
    })
}
