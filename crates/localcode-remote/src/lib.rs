//! Remote GPU servers over SSH.
//!
//! One-click flow ([`setup_server`]): connect over SSH (password or key),
//! probe the remote GPU and Ollama, install/start Ollama (with mirror
//! fallbacks for air-gapped networks), then forward the remote Ollama port back
//! to `localhost`. The returned [`RemoteSession`] carries an [`ActiveRuntime`]
//! whose `base_url` points at the tunnel, so the rest of LocalCode (deploy,
//! coding, bench) uses the remote GPU with no further changes.

mod provision;
mod ssh;

pub use provision::{ensure_ollama, probe, RemoteProbe};
pub use ssh::{AuthMethod, ExecOutput, SshClient, Tunnel};

use localcode_core::config::RemoteServer;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus, Severity};
use localcode_core::runtime::{ActiveRuntime, RuntimeKind, RuntimeStatus};
use localcode_gpu::GpuInventory;
use std::path::PathBuf;
use std::time::Duration;
use tracing::info;

/// A live remote GPU session. Keep it alive to keep the tunnel open; dropping
/// it closes the listener and (once in-flight requests drain) the SSH session.
pub struct RemoteSession {
    pub server_name: String,
    pub gpu: GpuInventory,
    pub runtime: ActiveRuntime,
    pub probe: RemoteProbe,
    // Field order matters for drop: tunnel first, then ssh.
    _tunnel: Tunnel,
    ssh: SshClient,
}

impl RemoteSession {
    pub fn base_url(&self) -> String {
        self.runtime.base_url.clone()
    }

    /// Run an ad-hoc command on the remote host (diagnostics).
    pub async fn exec(&self, cmd: &str) -> Result<ExecOutput, LocalCodeError> {
        self.ssh.exec(cmd).await
    }

    pub async fn close(self) {
        self.ssh.close().await;
    }
}

/// Resolve credentials from a server config: prefer an explicit key file, else
/// a stored password.
pub fn auth_from_server(server: &RemoteServer) -> Result<AuthMethod, LocalCodeError> {
    if let Some(key) = server.key_path.as_ref().filter(|k| !k.trim().is_empty()) {
        return Ok(AuthMethod::Key {
            path: PathBuf::from(key),
            passphrase: None,
        });
    }
    if !server.password.is_empty() {
        return Ok(AuthMethod::Password(server.password.clone()));
    }
    Err(LocalCodeError::new(
        ErrorCode::RemoteAuthFailed,
        format!("no credentials for {}", server.name),
    )
    .with_hint("Set a password (or key_path) for this server")
    .retryable(false))
}

/// The one-click setup. Progress is published to `events` as `Status` lines and
/// a final success/failure `Notification`.
pub async fn setup_server(
    server: &RemoteServer,
    events: &EventBus,
) -> Result<RemoteSession, LocalCodeError> {
    let status = |msg: String| {
        events.publish(AppEvent::Status {
            message: format!("Remote {}: {msg}", server.name),
        });
    };

    if server.backend != "ollama" {
        return Err(LocalCodeError::new(
            ErrorCode::RemoteProvisionFailed,
            format!("remote backend '{}' is not supported yet", server.backend),
        )
        .with_hint("Set backend = \"ollama\" for this server")
        .retryable(false));
    }

    status(format!("connecting to {}:{}", server.host, server.port));
    let auth = auth_from_server(server)?;
    let ssh = SshClient::connect(&server.host, server.port, &server.username, &auth).await?;
    info!(server = %server.name, "SSH connected");

    status("probing GPU & backend".into());
    let probe = probe(&ssh, server.remote_port).await?;

    if probe.gpu.devices.is_empty() {
        events.publish(AppEvent::Notification {
            severity: Severity::Warn,
            title: format!("{}: no GPU detected", server.name),
            body: "Remote deploys will run on CPU (slow) or fail.".into(),
            correlation_id: None,
        });
    } else {
        events.publish(AppEvent::Notification {
            severity: Severity::Info,
            title: format!("{}: GPU detected", server.name),
            body: probe.gpu.summary(),
            correlation_id: None,
        });
    }

    let mut log = |m: String| status(m);
    ensure_ollama(&ssh, server, probe.clone(), &mut log).await?;

    let local_port = server.effective_local_port();
    status(format!(
        "opening tunnel localhost:{local_port} → {}:{}",
        server.host, server.remote_port
    ));
    let tunnel = ssh
        .forward_local(local_port, "127.0.0.1", server.remote_port)
        .await?;

    // Verify the whole chain end-to-end from the local side.
    let base_url = server.tunnel_base_url();
    verify_tunnel(&base_url).await?;

    let mut runtime = ActiveRuntime::new(server.name.clone(), RuntimeKind::Ollama, base_url.clone());
    runtime.model_id = probe.models.first().cloned();
    runtime.status = RuntimeStatus::Healthy;

    events.publish(AppEvent::Notification {
        severity: Severity::Success,
        title: format!("{} connected", server.name),
        body: format!("Ollama on {} is tunneled to {base_url}", server.host),
        correlation_id: None,
    });

    Ok(RemoteSession {
        server_name: server.name.clone(),
        gpu: probe.gpu.clone(),
        runtime,
        probe,
        _tunnel: tunnel,
        ssh,
    })
}

/// GET `{base_url}/api/version` through the tunnel, retrying briefly while the
/// forward warms up.
async fn verify_tunnel(base_url: &str) -> Result<(), LocalCodeError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| LocalCodeError::new(ErrorCode::Internal, e.to_string()))?;
    let url = format!("{}/api/version", base_url.trim_end_matches('/'));
    let mut last = String::new();
    for _ in 0..10 {
        match client.get(&url).send().await {
            Ok(r) if r.status().is_success() => return Ok(()),
            Ok(r) => last = format!("status {}", r.status()),
            Err(e) => last = e.to_string(),
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Err(LocalCodeError::new(
        ErrorCode::RemoteTunnelFailed,
        "tunnel is up but the remote Ollama did not answer through it",
    )
    .with_cause(last)
    .with_hint("Check the remote_port and that Ollama is serving on the remote host")
    .retryable(true))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn srv() -> RemoteServer {
        RemoteServer {
            name: "gpu".into(),
            host: "10.0.0.5".into(),
            username: "user".into(),
            ..Default::default()
        }
    }

    #[test]
    fn auth_prefers_key_then_password() {
        let mut s = srv();
        s.password = "pw".into();
        assert!(matches!(auth_from_server(&s), Ok(AuthMethod::Password(_))));

        s.key_path = Some("/home/user/.ssh/id_ed25519".into());
        assert!(matches!(auth_from_server(&s), Ok(AuthMethod::Key { .. })));
    }

    #[test]
    fn auth_requires_some_credential() {
        let s = srv(); // no password, no key
        let err = auth_from_server(&s).unwrap_err();
        assert_eq!(err.code, ErrorCode::RemoteAuthFailed);
    }

    #[test]
    fn effective_local_port_defaults_to_remote() {
        let mut s = srv();
        s.remote_port = 11434;
        s.local_port = 0;
        assert_eq!(s.effective_local_port(), 11434);
        s.local_port = 21434;
        assert_eq!(s.effective_local_port(), 21434);
        assert_eq!(s.tunnel_base_url(), "http://127.0.0.1:21434");
    }
}
