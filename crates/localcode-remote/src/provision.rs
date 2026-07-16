//! Probe and provision a remote host: detect the GPU and Ollama, install
//! Ollama (mirror-aware) if missing, and make sure it is serving.

use localcode_core::config::RemoteServer;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_gpu::{parse_nvidia_smi_csv, GpuInventory, NVIDIA_SMI_QUERY_ARGS};
use tracing::debug;

use crate::ssh::SshClient;

/// What we learned about a remote host.
#[derive(Debug, Clone)]
pub struct RemoteProbe {
    pub uname: String,
    pub has_ollama: bool,
    pub ollama_running: bool,
    pub gpu: GpuInventory,
    /// Model tags already present on the remote Ollama (best-effort).
    pub models: Vec<String>,
}

/// The `nvidia-smi` command that yields the CSV `parse_nvidia_smi_csv` expects.
pub fn nvidia_smi_command() -> String {
    format!("nvidia-smi {}", NVIDIA_SMI_QUERY_ARGS.join(" "))
}

/// Command that installs Ollama from `url` (the official installer, or a
/// mirror of it). Piped to `sh`, matching the documented one-liner. Used when
/// the remote session is already root (or has passwordless sudo).
pub fn ollama_install_command(url: &str) -> String {
    format!("curl -fsSL {} | sh", shell_single_quote(url))
}

/// Like [`ollama_install_command`] but runs the installer under `sudo`, feeding
/// `password` on stdin (`-S`). The official installer calls `sudo` internally
/// for `/usr/local/bin` + systemd; over a non-interactive SSH exec channel that
/// inner `sudo` can't prompt, so a non-root remote otherwise fails asking for a
/// password. Fetching the script and running it as root (`sudo sh script`) makes
/// its internal `sudo` a no-op (`id -u == 0`), so nothing prompts.
pub fn ollama_install_command_sudo(url: &str, password: &str) -> String {
    format!(
        "curl -fsSL {} -o /tmp/lc-ollama-install.sh && printf '%s\\n' {} | \
         sudo -S -p '' sh /tmp/lc-ollama-install.sh; rm -f /tmp/lc-ollama-install.sh",
        shell_single_quote(url),
        shell_single_quote(password),
    )
}

/// Command that starts `ollama serve` detached, optionally pointing HF pulls at
/// a mirror via `HF_ENDPOINT` so an air-gapped box can still fetch `hf.co/*`
/// GGUF weights.
pub fn ollama_serve_command(hf_endpoint: &str, port: u16) -> String {
    let mut env = format!("OLLAMA_HOST=127.0.0.1:{port} ");
    if !hf_endpoint.trim().is_empty() {
        env.push_str(&format!("HF_ENDPOINT={} ", shell_single_quote(hf_endpoint.trim())));
    }
    // Detach so the serve outlives our exec channel.
    format!("sh -lc 'nohup env {env}ollama serve >/tmp/localcode-ollama.log 2>&1 & echo started'")
}

/// Command that evicts a single model from the remote Ollama's memory, freeing
/// the VRAM it held. `ollama serve` stays up (it's a shared service) — only the
/// model is unloaded.
pub fn ollama_stop_command(model: &str) -> String {
    format!("ollama stop {}", shell_single_quote(model))
}

/// Single-quote a string for POSIX `sh` (wrap in quotes, escape embedded ').
fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Probe the remote host. Never fails hard — missing tools show as false/empty.
pub async fn probe(ssh: &SshClient, remote_port: u16) -> Result<RemoteProbe, LocalCodeError> {
    let uname = ssh
        .exec("uname -sm")
        .await
        .map(|o| o.out().to_string())
        .unwrap_or_default();

    let has_ollama = ssh
        .exec("command -v ollama >/dev/null 2>&1 && echo yes || echo no")
        .await
        .map(|o| o.out() == "yes")
        .unwrap_or(false);

    let gpu = match ssh.exec(&nvidia_smi_command()).await {
        Ok(o) if o.ok() => parse_nvidia_smi_csv(&o.stdout, "nvidia-smi-remote"),
        _ => GpuInventory {
            devices: vec![],
            detection_method: "none".into(),
            warnings: vec!["No NVIDIA GPU detected on the remote host".into()],
        },
    };

    let ollama_running = ollama_health(ssh, remote_port).await;

    let models = if ollama_running {
        ssh.exec("ollama list")
            .await
            .map(|o| parse_ollama_list(&o.stdout))
            .unwrap_or_default()
    } else {
        vec![]
    };

    Ok(RemoteProbe {
        uname,
        has_ollama,
        ollama_running,
        gpu,
        models,
    })
}

/// True if the remote Ollama HTTP endpoint answers.
async fn ollama_health(ssh: &SshClient, port: u16) -> bool {
    let cmd = format!(
        "curl -fsS http://127.0.0.1:{port}/api/version >/dev/null 2>&1 && echo up || echo down"
    );
    ssh.exec(&cmd).await.map(|o| o.out() == "up").unwrap_or(false)
}

/// Parse `ollama list` table output into model tags (skip the header row).
fn parse_ollama_list(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .skip(1)
        .filter_map(|l| l.split_whitespace().next())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Parse `ollama ps` (models currently resident in memory) into their tags.
/// Same table shape as `ollama list` — a header row then one model per line with
/// the tag in the first column — so it shares the parser.
fn parse_ollama_ps(stdout: &str) -> Vec<String> {
    parse_ollama_list(stdout)
}

/// Ensure Ollama is installed and serving on the remote. `log` receives
/// human-readable progress lines. Installs from the first working mirror.
pub async fn ensure_ollama(
    ssh: &SshClient,
    server: &RemoteServer,
    mut probe_state: RemoteProbe,
    log: &mut (dyn FnMut(String) + Send),
) -> Result<(), LocalCodeError> {
    if !probe_state.has_ollama {
        log("Ollama not found — installing".into());
        install_ollama(ssh, server, log).await?;
        probe_state.has_ollama = true;
    } else {
        log("Ollama already installed".into());
    }

    if !probe_state.ollama_running {
        log("Starting ollama serve".into());
        let cmd = ollama_serve_command(&server.mirrors.hf_endpoint, server.remote_port);
        // Best-effort systemd first (installer sets up a service), else nohup.
        let _ = ssh.exec("systemctl start ollama 2>/dev/null || true").await;
        if !wait_for_ollama(ssh, server.remote_port, 3).await {
            ssh.exec_checked(&cmd).await?;
        }
        if !wait_for_ollama(ssh, server.remote_port, 30).await {
            return Err(LocalCodeError::new(
                ErrorCode::RemoteProvisionFailed,
                "Ollama did not become healthy on the remote host",
            )
            .with_cause("`ollama serve` started but /api/version never answered")
            .with_hint("Check /tmp/localcode-ollama.log on the server")
            .retryable(true));
        }
    }
    log("Ollama is serving".into());
    Ok(())
}

/// Best-effort: unload every model the remote Ollama currently holds in VRAM so
/// stopping/disconnecting actually frees the remote GPU. `ollama ps` lists only
/// models resident in memory, so an idle box is a no-op. This is the remote
/// analog of the local stop path — closing the tunnel alone leaves the model
/// loaded on the remote GPU. Errors are swallowed: a stop must never hang or
/// fail on a flaky remote.
pub async fn free_gpu_memory(ssh: &SshClient) {
    let loaded = match ssh.exec("ollama ps").await {
        Ok(o) if o.ok() => parse_ollama_ps(&o.stdout),
        _ => return,
    };
    for model in loaded {
        debug!(%model, "unloading remote Ollama model to free VRAM");
        let _ = ssh.exec(&ollama_stop_command(&model)).await;
    }
}

/// Try each configured installer mirror until one succeeds.
async fn install_ollama(
    ssh: &SshClient,
    server: &RemoteServer,
    log: &mut (dyn FnMut(String) + Send),
) -> Result<(), LocalCodeError> {
    let urls = if server.mirrors.ollama_install.is_empty() {
        vec!["https://ollama.com/install.sh".to_string()]
    } else {
        server.mirrors.ollama_install.clone()
    };
    // A root session (common for cloud GPU boxes) needs no sudo; a non-root one
    // with a stored password runs the installer under sudo so it never prompts.
    let is_root = ssh
        .exec("id -u")
        .await
        .map(|o| o.out().trim() == "0")
        .unwrap_or(false);
    let use_sudo = !is_root && !server.password.is_empty();

    let mut last_err: Option<LocalCodeError> = None;
    let n = urls.len();
    for (i, url) in urls.iter().enumerate() {
        log(format!("Installing Ollama from {} ({}/{})", url, i + 1, n));
        let cmd = if use_sudo {
            ollama_install_command_sudo(url, &server.password)
        } else {
            ollama_install_command(url)
        };
        match ssh.exec_checked(&cmd).await {
            Ok(_) => return Ok(()),
            Err(e) => {
                log(format!("Install source failed: {}", e.message));
                last_err = Some(e);
            }
        }
    }
    Err(last_err
        .unwrap_or_else(|| {
            LocalCodeError::new(ErrorCode::RemoteProvisionFailed, "no Ollama installer configured")
        })
        .with_hint("Add a reachable installer URL to the server's mirrors.ollama_install"))
}

/// Poll the remote Ollama health up to `secs` times (1s apart).
async fn wait_for_ollama(ssh: &SshClient, port: u16, secs: u32) -> bool {
    for _ in 0..secs {
        if ollama_health(ssh, port).await {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_command_quotes_url() {
        assert_eq!(
            ollama_install_command("https://ollama.com/install.sh"),
            "curl -fsSL 'https://ollama.com/install.sh' | sh"
        );
    }

    #[test]
    fn install_command_sudo_runs_script_as_root() {
        let c = ollama_install_command_sudo("https://ollama.com/install.sh", "pw");
        // Feeds the password to sudo on stdin and runs the fetched script as root.
        assert!(c.contains("sudo -S -p ''"));
        assert!(c.contains("sh /tmp/lc-ollama-install.sh"));
        assert!(c.contains("'https://ollama.com/install.sh'"));
        assert!(c.contains("'pw'"));
    }

    #[test]
    fn serve_command_sets_host_and_optional_hf_endpoint() {
        let no_mirror = ollama_serve_command("", 11434);
        assert!(no_mirror.contains("OLLAMA_HOST=127.0.0.1:11434"));
        assert!(!no_mirror.contains("HF_ENDPOINT"));

        let mirror = ollama_serve_command("https://hf-mirror.com", 11500);
        assert!(mirror.contains("OLLAMA_HOST=127.0.0.1:11500"));
        assert!(mirror.contains("HF_ENDPOINT='https://hf-mirror.com'"));
    }

    #[test]
    fn nvidia_smi_command_matches_query_args() {
        let c = nvidia_smi_command();
        assert!(c.starts_with("nvidia-smi "));
        assert!(c.contains("--query-gpu=index,name,memory.total,memory.free,driver_version"));
        assert!(c.contains("--format=csv,noheader,nounits"));
    }

    #[test]
    fn parse_ollama_list_skips_header() {
        let out = "NAME\tID\tSIZE\nqwen2.5-coder:7b\tabc\t4GB\nllama3:8b\tdef\t5GB\n";
        assert_eq!(parse_ollama_list(out), vec!["qwen2.5-coder:7b", "llama3:8b"]);
    }

    #[test]
    fn parse_ollama_ps_takes_the_name_column() {
        // `ollama ps` has extra columns (PROCESSOR "100% GPU", UNTIL); the model
        // tag must come from the first column, not a later token.
        let out = "NAME                ID      SIZE      PROCESSOR    UNTIL\n\
                   qwen2.5-coder:7b    abc123  6.0 GB    100% GPU     4 minutes from now\n";
        assert_eq!(parse_ollama_ps(out), vec!["qwen2.5-coder:7b"]);
    }

    #[test]
    fn parse_ollama_ps_empty_when_nothing_loaded() {
        // Header only (or nothing) → no models to unload.
        assert!(parse_ollama_ps("NAME    ID    SIZE    PROCESSOR    UNTIL\n").is_empty());
        assert!(parse_ollama_ps("").is_empty());
    }

    #[test]
    fn ollama_stop_command_quotes_the_model() {
        assert_eq!(
            ollama_stop_command("qwen2.5-coder:7b"),
            "ollama stop 'qwen2.5-coder:7b'"
        );
    }

    #[test]
    fn shell_quote_escapes_single_quotes() {
        assert_eq!(shell_single_quote("a'b"), "'a'\\''b'");
    }
}
