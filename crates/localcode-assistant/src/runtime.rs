//! Dedicated llama-server process for the local Bonsai assistant.

use crate::constants::{ASSISTANT_MODEL_ID, BONSAI_TEMPERATURE, BONSAI_TOP_K, BONSAI_TOP_P};
use crate::install::{model_installed, model_path, resolve_llama_bin};
use localcode_core::config::Config;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::paths::AppPaths;
use localcode_core::runtime::{ActiveRuntime, RuntimeKind, RuntimeStatus};
use std::sync::Arc;
use tokio::process::Child;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Long-lived handle for the assistant's `llama-server` child.
pub struct LocalAssistantRuntime {
    child: Arc<Mutex<Option<Child>>>,
    base_url: String,
    port: u16,
    model_id: String,
}

impl LocalAssistantRuntime {
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// OpenAI-compatible ActiveRuntime for the coding/assistant agent loop.
    pub fn as_active_runtime(&self) -> ActiveRuntime {
        let mut r = ActiveRuntime::new(
            "local-assistant-bonsai",
            RuntimeKind::LlamaCpp,
            self.base_url.clone(),
        );
        r.model_id = Some(self.model_id.clone());
        r.status = RuntimeStatus::Healthy;
        r
    }

    /// Probe `/v1/models` (or `/health`) — true when the server answers.
    pub async fn is_healthy(&self) -> bool {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(2))
            .timeout(std::time::Duration::from_secs(4))
            .build()
            .unwrap_or_default();
        let health = format!(
            "http://127.0.0.1:{}/health",
            self.port
        );
        if let Ok(resp) = client.get(&health).send().await {
            if resp.status().is_success() {
                return true;
            }
        }
        let models = format!("{}/models", self.base_url.trim_end_matches('/'));
        client
            .get(&models)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Stop the managed child (best-effort).
    pub async fn stop(&self) {
        let mut guard = self.child.lock().await;
        if let Some(mut child) = guard.take() {
            let _ = child.kill().await;
            info!("local assistant llama-server stopped");
        }
    }
}

/// Start (or reuse a healthy) assistant llama-server on `cfg.assistant.local_port`.
pub async fn ensure_running(
    cfg: &Config,
    paths: &AppPaths,
) -> Result<LocalAssistantRuntime, LocalCodeError> {
    if !model_installed(paths) {
        return Err(LocalCodeError::new(
            ErrorCode::BackendNotReady,
            "Local assistant model is not installed",
        )
        .with_hint("Accept the Bonsai install offer, or run /assistant install")
        .with_cause(format!("Missing {}", model_path(paths).display())));
    }

    let bin = resolve_llama_bin(cfg, paths).ok_or_else(|| {
        LocalCodeError::new(
            ErrorCode::BackendBinaryMissing,
            "llama-server not found for the local assistant",
        )
        .with_hint("Install llama.cpp from the Backends panel or accept the assistant install offer")
    })?;

    let port = cfg.assistant.local_port;
    let host = "127.0.0.1";
    let base_url = format!("http://{host}:{port}/v1");

    // Already healthy? Reuse without spawning another process.
    let probe = LocalAssistantRuntime {
        child: Arc::new(Mutex::new(None)),
        base_url: base_url.clone(),
        port,
        model_id: ASSISTANT_MODEL_ID.into(),
    };
    if probe.is_healthy().await {
        info!(port, "reusing healthy local assistant server");
        return Ok(probe);
    }

    if port_in_use(port) {
        return Err(LocalCodeError::new(
            ErrorCode::BackendPortInUse,
            format!("Assistant port {port} is in use but not healthy"),
        )
        .with_hint("Stop the other process or change assistant.local_port in config")
        .retryable(true));
    }

    let model = model_path(paths);
    let ctx = cfg.assistant.local_context.max(2048);
    let ngl = cfg.assistant.local_gpu_layers;

    info!(
        bin = %bin.display(),
        model = %model.display(),
        port,
        "starting local assistant llama-server"
    );

    let mut args: Vec<String> = vec![
        "-m".into(),
        model.display().to_string(),
        "--host".into(),
        host.into(),
        "--port".into(),
        port.to_string(),
        "-c".into(),
        ctx.to_string(),
        "--n-gpu-layers".into(),
        ngl.to_string(),
        // Generation defaults from the Bonsai card (server-side defaults).
        "--temp".into(),
        BONSAI_TEMPERATURE.to_string(),
        "--top-p".into(),
        BONSAI_TOP_P.to_string(),
        "--top-k".into(),
        BONSAI_TOP_K.to_string(),
    ];
    // Quiet chat UI; OpenAI API is enough.
    args.push("--jinja".into());

    let mut child = tokio::process::Command::new(&bin)
        .args(&args)
        .kill_on_drop(true)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| {
            LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                .with_cause("Failed to spawn llama-server for the assistant")
                .with_hint(
                    "Bonsai Q1_0 needs llama.cpp with Q1_0_g128 kernels \
                     (PrismML fork: https://github.com/PrismML-Eng/llama.cpp). \
                     Stock builds may reject the quant.",
                )
                .retryable(true)
        })?;

    drain_child_io("assistant-llama", &mut child);

    let runtime = LocalAssistantRuntime {
        child: Arc::new(Mutex::new(Some(child))),
        base_url: base_url.clone(),
        port,
        model_id: ASSISTANT_MODEL_ID.into(),
    };

    // Wait for health (model load can take a bit on CPU).
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(180);
    loop {
        if runtime.is_healthy().await {
            info!(port, "local assistant server ready");
            return Ok(runtime);
        }
        // Child died?
        {
            let mut guard = runtime.child.lock().await;
            if let Some(child) = guard.as_mut() {
                if let Ok(Some(status)) = child.try_wait() {
                    *guard = None;
                    return Err(LocalCodeError::new(
                        ErrorCode::BackendStartFailed,
                        format!("Assistant llama-server exited: {status}"),
                    )
                    .with_cause(
                        "The GGUF may need PrismML's llama.cpp fork (Q1_0_g128 hybrid-attention kernels)",
                    )
                    .with_hint("https://github.com/PrismML-Eng/llama.cpp")
                    .with_hint("https://huggingface.co/prism-ml/Bonsai-27B-gguf")
                    .retryable(true));
                }
            }
        }
        if tokio::time::Instant::now() > deadline {
            runtime.stop().await;
            return Err(LocalCodeError::new(
                ErrorCode::BackendHealthTimeout,
                "Assistant llama-server did not become healthy in time",
            )
            .with_hint("Check free RAM/VRAM; Bonsai needs ~5+ GB peak at short context")
            .retryable(true));
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(400)).await;
    }
}

fn port_in_use(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_err()
}

fn drain_child_io(tag: &str, child: &mut Child) {
    use tokio::io::{AsyncBufReadExt, BufReader};
    if let Some(stdout) = child.stdout.take() {
        let tag = tag.to_string();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(target: "assistant_io", backend = %tag, "{line}");
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
        let tag = tag.to_string();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(target: "assistant_io", backend = %tag, "{line}");
            }
        });
    }
}

/// Soft check used by the TUI without starting the server.
pub fn is_installed(cfg: &Config, paths: &AppPaths) -> bool {
    model_installed(paths) && resolve_llama_bin(cfg, paths).is_some()
}

/// Drop-in warning when stock llama.cpp may not load Q1_0.
pub fn quant_compatibility_note() -> &'static str {
    "Note: Bonsai Q1_0_g128 uses custom llama.cpp kernels from PrismML \
     (https://github.com/PrismML-Eng/llama.cpp). If load fails, build that fork \
     and set backends.llamacpp.bin to its llama-server."
}

impl Drop for LocalAssistantRuntime {
    fn drop(&mut self) {
        // Best-effort: kill_on_drop on the Child handles normal exit.
        // If Arc still has clones, the last drop wins.
        if let Ok(mut guard) = self.child.try_lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.start_kill();
                warn!("local assistant runtime dropped — killing llama-server");
            }
        }
    }
}
