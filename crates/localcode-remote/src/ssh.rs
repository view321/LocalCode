//! Thin async SSH client over `russh`: connect (password or key), run a
//! command, and forward a local TCP port to a remote address (the tunnel that
//! makes a remote backend look local).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use localcode_core::error::{ErrorCode, LocalCodeError};
use russh::client::{self, Handle};
use russh::keys::{load_secret_key, PrivateKeyWithHashAlg};
use russh::{ChannelMsg, Disconnect};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

/// How to authenticate to the remote host.
#[derive(Clone)]
pub enum AuthMethod {
    Password(String),
    Key {
        path: PathBuf,
        passphrase: Option<String>,
    },
}

// Redacting Debug: never print the password or passphrase in logs/panics.
impl std::fmt::Debug for AuthMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthMethod::Password(_) => write!(f, "Password(***)"),
            AuthMethod::Key { path, .. } => {
                write!(f, "Key {{ path: {}, passphrase: *** }}", path.display())
            }
        }
    }
}

/// russh event handler. We accept the server host key unconditionally
/// (trust-on-first-use): LAN GPU boxes rarely have a `known_hosts` entry and
/// the one-click flow has nowhere to prompt. Documented in the UI.
struct ClientHandler;

impl client::Handler for ClientHandler {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &russh::keys::ssh_key::PublicKey,
    ) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

/// Captured result of a remote command.
pub struct ExecOutput {
    pub stdout: String,
    pub stderr: String,
    pub code: i32,
}

impl ExecOutput {
    pub fn ok(&self) -> bool {
        self.code == 0
    }
    pub fn out(&self) -> &str {
        self.stdout.trim()
    }
    /// stderr (or, if empty, stdout) — for surfacing why a command failed.
    pub fn err_text(&self) -> String {
        let s = self.stderr.trim();
        if s.is_empty() {
            self.stdout.trim().to_string()
        } else {
            s.to_string()
        }
    }
}

/// A live local port-forward. Dropping it stops accepting new connections.
pub struct Tunnel {
    task: JoinHandle<()>,
    pub local_port: u16,
}

impl Drop for Tunnel {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// A connected, authenticated SSH session.
pub struct SshClient {
    handle: Arc<Handle<ClientHandler>>,
    pub host: String,
}

impl SshClient {
    /// Connect and authenticate. Times out so a wrong IP fails fast instead of
    /// hanging the one-click flow.
    pub async fn connect(
        host: &str,
        port: u16,
        user: &str,
        auth: &AuthMethod,
    ) -> Result<Self, LocalCodeError> {
        let config = Arc::new(client::Config {
            inactivity_timeout: None,
            keepalive_interval: Some(Duration::from_secs(20)),
            nodelay: true,
            ..Default::default()
        });

        let mut handle = tokio::time::timeout(
            Duration::from_secs(20),
            client::connect(config, (host, port), ClientHandler),
        )
        .await
        .map_err(|_| conn_err(host, "connection timed out after 20s"))?
        .map_err(|e| conn_err(host, e))?;

        let ok = match auth {
            AuthMethod::Password(pw) => handle
                .authenticate_password(user, pw.clone())
                .await
                .map_err(|e| auth_err(host, e))?
                .success(),
            AuthMethod::Key { path, passphrase } => {
                let key = load_secret_key(path, passphrase.as_deref()).map_err(|e| {
                    auth_err(host, format!("cannot load key {}: {e}", path.display()))
                })?;
                let hash = handle
                    .best_supported_rsa_hash()
                    .await
                    .map_err(|e| auth_err(host, e))?
                    .flatten();
                handle
                    .authenticate_publickey(user, PrivateKeyWithHashAlg::new(Arc::new(key), hash))
                    .await
                    .map_err(|e| auth_err(host, e))?
                    .success()
            }
        };

        if !ok {
            return Err(LocalCodeError::new(
                ErrorCode::RemoteAuthFailed,
                format!("SSH authentication failed for {user}@{host}"),
            )
            .with_hint("Check the username and password (or key_path)")
            .retryable(false));
        }

        Ok(Self {
            handle: Arc::new(handle),
            host: host.to_string(),
        })
    }

    /// Run a command to completion, capturing stdout, stderr, and the exit code.
    pub async fn exec(&self, cmd: &str) -> Result<ExecOutput, LocalCodeError> {
        let mut channel = self
            .handle
            .channel_open_session()
            .await
            .map_err(|e| cmd_err(cmd, e))?;
        channel.exec(true, cmd).await.map_err(|e| cmd_err(cmd, e))?;

        let mut stdout: Vec<u8> = Vec::new();
        let mut stderr: Vec<u8> = Vec::new();
        let mut code: Option<i32> = None;
        while let Some(msg) = channel.wait().await {
            match msg {
                ChannelMsg::Data { ref data } => stdout.extend_from_slice(data),
                ChannelMsg::ExtendedData { ref data, ext } => {
                    // ext == 1 is stderr in the SSH protocol.
                    if ext == 1 {
                        stderr.extend_from_slice(data);
                    } else {
                        stdout.extend_from_slice(data);
                    }
                }
                ChannelMsg::ExitStatus { exit_status } => code = Some(exit_status as i32),
                _ => {}
            }
        }

        Ok(ExecOutput {
            stdout: String::from_utf8_lossy(&stdout).to_string(),
            stderr: String::from_utf8_lossy(&stderr).to_string(),
            code: code.unwrap_or(-1),
        })
    }

    /// Run a command and fail if it exits non-zero.
    pub async fn exec_checked(&self, cmd: &str) -> Result<ExecOutput, LocalCodeError> {
        let out = self.exec(cmd).await?;
        if !out.ok() {
            return Err(LocalCodeError::new(
                ErrorCode::RemoteCommandFailed,
                format!("remote command failed (exit {}): {cmd}", out.code),
            )
            .with_cause(out.err_text())
            .retryable(true));
        }
        Ok(out)
    }

    /// Bind `127.0.0.1:local_port` locally and forward every connection to
    /// `remote_host:remote_port` through the SSH session. Returns a [`Tunnel`]
    /// whose drop stops the listener.
    pub async fn forward_local(
        &self,
        local_port: u16,
        remote_host: &str,
        remote_port: u16,
    ) -> Result<Tunnel, LocalCodeError> {
        let listener = TcpListener::bind(("127.0.0.1", local_port))
            .await
            .map_err(|e| {
                LocalCodeError::new(
                    ErrorCode::RemoteTunnelFailed,
                    format!("cannot bind local port {local_port}: {e}"),
                )
                .with_hint("Set a different local_port for this server")
            })?;
        let handle = self.handle.clone();
        let remote_host = remote_host.to_string();

        let task = tokio::spawn(async move {
            loop {
                let (socket, addr) = match listener.accept().await {
                    Ok(v) => v,
                    Err(e) => {
                        warn!(error = %e, "tunnel accept failed; stopping listener");
                        break;
                    }
                };
                let handle = handle.clone();
                let remote_host = remote_host.clone();
                tokio::spawn(async move {
                    let channel = match handle
                        .channel_open_direct_tcpip(
                            remote_host.clone(),
                            u32::from(remote_port),
                            "127.0.0.1".to_string(),
                            u32::from(addr.port()),
                        )
                        .await
                    {
                        Ok(c) => c,
                        Err(e) => {
                            warn!(error = %e, "tunnel channel open failed");
                            return;
                        }
                    };
                    if let Err(e) = pump(socket, channel).await {
                        debug!(error = %e, "tunnel connection ended");
                    }
                });
            }
        });

        Ok(Tunnel { task, local_port })
    }

    pub async fn close(&self) {
        let _ = self
            .handle
            .disconnect(Disconnect::ByApplication, "", "en")
            .await;
    }
}

/// Bidirectional byte pump between a local TCP stream and an SSH channel.
async fn pump(
    mut stream: TcpStream,
    mut channel: russh::Channel<russh::client::Msg>,
) -> std::io::Result<()> {
    let mut buf = vec![0u8; 65536];
    let mut stream_closed = false;
    loop {
        tokio::select! {
            r = stream.read(&mut buf), if !stream_closed => {
                match r {
                    Ok(0) => {
                        stream_closed = true;
                        let _ = channel.eof().await;
                    }
                    Ok(n) => {
                        if channel.data(&buf[..n]).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
            msg = channel.wait() => {
                match msg {
                    Some(ChannelMsg::Data { ref data }) => stream.write_all(data).await?,
                    Some(ChannelMsg::Eof) | None => break,
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

fn conn_err(host: &str, cause: impl std::fmt::Display) -> LocalCodeError {
    LocalCodeError::new(
        ErrorCode::RemoteConnectFailed,
        format!("cannot reach SSH host {host}"),
    )
    .with_cause(cause.to_string())
    .with_hint("Check the host/IP, port, and that the server is reachable (VPN up?)")
    .retryable(true)
}

fn auth_err(host: &str, cause: impl std::fmt::Display) -> LocalCodeError {
    LocalCodeError::new(
        ErrorCode::RemoteAuthFailed,
        format!("SSH authentication error for {host}"),
    )
    .with_cause(cause.to_string())
    .with_hint("Check username/password or the key file")
    .retryable(false)
}

fn cmd_err(cmd: &str, cause: impl std::fmt::Display) -> LocalCodeError {
    LocalCodeError::new(
        ErrorCode::RemoteCommandFailed,
        format!("failed to run remote command: {cmd}"),
    )
    .with_cause(cause.to_string())
    .retryable(true)
}
