//! OpenWebUI deployment — a browser chat UI (`/ui`) wired to the models the
//! user has deployed locally.
//!
//! OpenWebUI runs as a Docker container that talks to LocalCode's model servers
//! over their OpenAI-compatible endpoints. Because deployed servers now bind
//! `0.0.0.0` by default, the container reaches them via `host.docker.internal`
//! (built in on Docker Desktop; added explicitly on Linux). We run the container
//! detached and manage it by name, capturing `docker logs` for the `/dash` card
//! and probing `/health` to flip the card to Running.
//!
//! The container is wired to whatever models LocalCode is hosting *now*: the
//! endpoint list is passed via `OPENAI_API_BASE_URLS` with
//! `ENABLE_PERSISTENT_CONFIG=False` so the env wins on every (re)start, and the
//! TUI re-launches the container (same name/volume) whenever the runtime set
//! changes — so new deploys and the local assistant appear automatically.

use crate::ProcState;
use localcode_core::error::{ErrorCode, LocalCodeError};
use std::sync::{Arc, Mutex};
use tracing::info;

/// Fixed container name so re-running `/ui` replaces the previous instance
/// instead of colliding on the port.
pub const OPENWEBUI_CONTAINER: &str = "localcode-openwebui";
/// Default host port the UI is published on (`http://localhost:3000`).
pub const OPENWEBUI_DEFAULT_PORT: u16 = 3000;
/// Named volume so chats/settings persist across restarts.
const OPENWEBUI_VOLUME: &str = "localcode-openwebui-data";
const OPENWEBUI_IMAGE: &str = "ghcr.io/open-webui/open-webui:main";

/// A running (or starting) OpenWebUI container.
pub struct OpenWebUiHandle {
    container: String,
    /// Browser URL, e.g. `http://localhost:3000`.
    url: String,
    /// The exact `docker run …` command (click-to-copy on the dashboard).
    command: String,
    /// Model endpoints wired into the UI (host-rewritten, `/v1`).
    model_urls: Vec<String>,
    logs: Arc<Mutex<String>>,
    state: Arc<Mutex<ProcState>>,
}

impl OpenWebUiHandle {
    pub fn url(&self) -> &str {
        &self.url
    }
    pub fn container(&self) -> &str {
        &self.container
    }
    pub fn command(&self) -> &str {
        &self.command
    }
    pub fn model_urls(&self) -> &[String] {
        &self.model_urls
    }
    pub fn state(&self) -> ProcState {
        self.state.lock().map(|g| g.clone()).unwrap_or(ProcState::External)
    }

    /// Newest `max_lines` of captured container output (newest last).
    pub fn recent_logs(&self, max_lines: usize) -> Vec<String> {
        self.logs
            .lock()
            .map(|s| {
                let lines: Vec<&str> = s.lines().collect();
                let start = lines.len().saturating_sub(max_lines);
                lines[start..].iter().map(|l| l.to_string()).collect()
            })
            .unwrap_or_default()
    }

    /// Stop and remove the container (best-effort).
    pub async fn stop(&self) {
        let _ = tokio::process::Command::new("docker")
            .args(["rm", "-f", &self.container])
            .output()
            .await;
        if let Ok(mut g) = self.state.lock() {
            *g = ProcState::Exited { code: Some(0), ok: true };
        }
        info!(container = %self.container, "openwebui stopped");
    }
}

/// Deploy façade for OpenWebUI. Stateless — the running instance lives in the
/// returned [`OpenWebUiHandle`].
pub struct OpenWebUi;

impl OpenWebUi {
    /// True when the `docker` CLI is on PATH.
    pub fn docker_available() -> bool {
        which::which("docker").is_ok()
    }

    /// Rewrite a model server base URL so a container can reach the host, and
    /// ensure it ends with `/v1` (OpenAI-compatible root OpenWebUI expects).
    pub fn container_endpoint(model_url: &str) -> String {
        let mut u = model_url.trim().trim_end_matches('/').to_string();
        for hostish in ["0.0.0.0", "127.0.0.1", "localhost"] {
            // Match `//host:` and `//host/` and trailing `//host`.
            u = u
                .replace(&format!("//{hostish}:"), "//host.docker.internal:")
                .replace(&format!("//{hostish}/"), "//host.docker.internal/");
            if u.ends_with(&format!("//{hostish}")) {
                u = u.replace(&format!("//{hostish}"), "//host.docker.internal");
            }
        }
        if !u.ends_with("/v1") {
            u.push_str("/v1");
        }
        u
    }

    /// Build the `docker run …` argument list (everything after `docker`).
    pub fn build_run_args(port: u16, model_urls: &[String]) -> Vec<String> {
        let endpoints: Vec<String> = model_urls
            .iter()
            .map(|u| Self::container_endpoint(u))
            .collect();
        // OpenWebUI pairs each base URL with a key positionally; our local
        // servers don't check it, so a placeholder per endpoint is fine.
        let keys = vec!["localcode"; endpoints.len().max(1)].join(";");
        let base_urls = if endpoints.is_empty() {
            // No models yet — leave the UI usable; the user can add endpoints.
            "http://host.docker.internal:8080/v1".to_string()
        } else {
            endpoints.join(";")
        };

        let mut args: Vec<String> = vec![
            "run".into(),
            "-d".into(),
            "--rm".into(),
            "-p".into(),
            format!("{port}:8080"),
            "--name".into(),
            OPENWEBUI_CONTAINER.into(),
            "-v".into(),
            format!("{OPENWEBUI_VOLUME}:/app/backend/data"),
        ];
        // Docker Desktop resolves host.docker.internal natively; Linux needs it
        // mapped to the host gateway explicitly.
        if cfg!(target_os = "linux") {
            args.push("--add-host".into());
            args.push("host.docker.internal:host-gateway".into());
        }
        args.push("-e".into());
        args.push(format!("OPENAI_API_BASE_URLS={base_urls}"));
        args.push("-e".into());
        args.push(format!("OPENAI_API_KEYS={keys}"));
        // Make the env authoritative on every (re)start. OpenWebUI treats
        // OPENAI_API_BASE_URLS as a "PersistentConfig" value: by default it is
        // seeded into the DB on first boot and the DB then wins, so re-launching
        // with a different model set (a new deploy, the assistant coming online)
        // would keep showing the stale list. Disabling persistent config makes
        // the connections always reflect the models LocalCode is hosting now.
        args.push("-e".into());
        args.push("ENABLE_PERSISTENT_CONFIG=False".into());
        // Skip login for a local single-user tool.
        args.push("-e".into());
        args.push("WEBUI_AUTH=False".into());
        args.push(OPENWEBUI_IMAGE.into());
        args
    }

    /// Deploy OpenWebUI wired to `model_urls`. Returns once the container has
    /// been created (image pull + first boot continue in the background; the
    /// handle's state flips to Running when `/health` answers).
    pub async fn deploy(port: u16, model_urls: Vec<String>) -> Result<OpenWebUiHandle, LocalCodeError> {
        if !Self::docker_available() {
            return Err(LocalCodeError::new(
                ErrorCode::BackendBinaryMissing,
                "Docker is required to run OpenWebUI (/ui)",
            )
            .with_cause("`docker` was not found on PATH")
            .with_hint("Install Docker Desktop (Windows/macOS) or Docker Engine (Linux)")
            .with_hint("Alternatively run OpenWebUI yourself: pip install open-webui && open-webui serve"));
        }

        let args = Self::build_run_args(port, &model_urls);
        let command = crate::format_command("docker", &args);
        let url = format!("http://localhost:{port}");

        // Replace any previous instance so the name/port are free.
        let _ = tokio::process::Command::new("docker")
            .args(["rm", "-f", OPENWEBUI_CONTAINER])
            .output()
            .await;

        info!(%command, "deploying openwebui");
        let out = tokio::process::Command::new("docker")
            .args(&args)
            .output()
            .await
            .map_err(|e| {
                LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                    .with_cause("Failed to invoke docker")
                    .with_hint("Is the Docker daemon running?")
                    .retryable(true)
            })?;

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            return Err(LocalCodeError::new(
                ErrorCode::BackendStartFailed,
                "docker run failed for OpenWebUI",
            )
            .with_cause(if stderr.is_empty() { "docker exited non-zero".into() } else { stderr })
            .with_hint("Is the Docker daemon running and port free?")
            .retryable(true));
        }

        let logs = Arc::new(Mutex::new(String::new()));
        let state = Arc::new(Mutex::new(ProcState::Starting));
        spawn_logs_drain(OPENWEBUI_CONTAINER, logs.clone());
        spawn_health_probe(url.clone(), state.clone());

        Ok(OpenWebUiHandle {
            container: OPENWEBUI_CONTAINER.to_string(),
            url,
            command,
            model_urls,
            logs,
            state,
        })
    }

    /// Stop any running OpenWebUI container by name (used when no handle is held).
    pub async fn stop_by_name() {
        let _ = tokio::process::Command::new("docker")
            .args(["rm", "-f", OPENWEBUI_CONTAINER])
            .output()
            .await;
    }
}

/// Follow `docker logs` into a shared buffer for the `/dash` card.
fn spawn_logs_drain(container: &str, acc: Arc<Mutex<String>>) {
    let container = container.to_string();
    tokio::spawn(async move {
        use tokio::io::{AsyncBufReadExt, BufReader};
        let mut child = match tokio::process::Command::new("docker")
            .args(["logs", "-f", "--tail", "50", &container])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut readers: Vec<Box<dyn tokio::io::AsyncRead + Unpin + Send>> = Vec::new();
        if let Some(o) = child.stdout.take() {
            readers.push(Box::new(o));
        }
        if let Some(e) = child.stderr.take() {
            readers.push(Box::new(e));
        }
        for reader in readers {
            let acc = acc.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(reader).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if let Ok(mut g) = acc.lock() {
                        if g.len() < 32_000 {
                            g.push_str(&line);
                            g.push('\n');
                        }
                    }
                }
            });
        }
        let _ = child.wait().await;
    });
}

/// Probe `/health` until the container answers (or a ceiling), flipping state.
fn spawn_health_probe(url: String, state: Arc<Mutex<ProcState>>) {
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(2))
            .timeout(std::time::Duration::from_secs(4))
            .build()
            .unwrap_or_default();
        let health = format!("{}/health", url.trim_end_matches('/'));
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(600);
        loop {
            if tokio::time::Instant::now() > deadline {
                return;
            }
            if let Ok(r) = client.get(&health).send().await {
                if r.status().is_success() {
                    if let Ok(mut g) = state.lock() {
                        *g = ProcState::Running;
                    }
                    return;
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_rewrites_host_and_adds_v1() {
        assert_eq!(
            OpenWebUi::container_endpoint("http://0.0.0.0:8080"),
            "http://host.docker.internal:8080/v1"
        );
        assert_eq!(
            OpenWebUi::container_endpoint("http://127.0.0.1:8000/v1"),
            "http://host.docker.internal:8000/v1"
        );
        assert_eq!(
            OpenWebUi::container_endpoint("http://localhost:30000/"),
            "http://host.docker.internal:30000/v1"
        );
    }

    #[test]
    fn run_args_carry_port_name_and_endpoints() {
        let args = OpenWebUi::build_run_args(
            3000,
            &["http://0.0.0.0:8080/v1".into(), "http://0.0.0.0:8000/v1".into()],
        );
        assert!(args.contains(&"run".to_string()));
        assert!(args.iter().any(|a| a == "3000:8080"));
        assert!(args.iter().any(|a| a == OPENWEBUI_CONTAINER));
        let joined = args.iter().find(|a| a.starts_with("OPENAI_API_BASE_URLS=")).unwrap();
        assert!(joined.contains("host.docker.internal:8080/v1"));
        assert!(joined.contains("host.docker.internal:8000/v1"));
        assert!(joined.contains(';'));
        // Env must stay authoritative so a resync reflects the current models.
        assert!(args.iter().any(|a| a == "ENABLE_PERSISTENT_CONFIG=False"));
        assert!(args.last().unwrap().contains("open-webui"));
    }

    #[test]
    fn run_args_have_a_default_endpoint_when_no_models() {
        let args = OpenWebUi::build_run_args(3000, &[]);
        let joined = args.iter().find(|a| a.starts_with("OPENAI_API_BASE_URLS=")).unwrap();
        assert!(joined.contains("host.docker.internal"));
    }
}
