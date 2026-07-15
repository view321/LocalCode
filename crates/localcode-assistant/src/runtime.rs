//! Dedicated llama-server process for the local Bonsai assistant.
//!
//! Launch shape (model card server example, Q4_1 quant):
//! ```text
//! ./build/bin/llama-server \
//!     -m Bonsai-27B-dspark-Q4_1.gguf \
//!     --host 127.0.0.1 --port … -ngl 99
//! ```

use crate::constants::{
    ASSISTANT_MODEL_ID, BONSAI_FILE, BONSAI_QUANT, BONSAI_TEMPERATURE, BONSAI_TOP_K, BONSAI_TOP_P,
};
use crate::install::{mark_ready, model_installed, model_path, resolve_llama_bin};
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
        r.quantization = Some(BONSAI_QUANT.into());
        r.status = RuntimeStatus::Healthy;
        r
    }

    /// Probe `/health` or `/v1/models` — true when the server answers.
    pub async fn is_healthy(&self) -> bool {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(2))
            .timeout(std::time::Duration::from_secs(4))
            .build()
            .unwrap_or_default();
        let health = format!("http://127.0.0.1:{}/health", self.port);
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
///
/// Command (model-card shape, Q4_1):
/// ```text
/// llama-server -m Bonsai-27B-dspark-Q4_1.gguf --host 127.0.0.1 --port … -ngl 99
/// ```
pub async fn ensure_running(
    cfg: &Config,
    paths: &AppPaths,
) -> Result<LocalAssistantRuntime, LocalCodeError> {
    let bin = resolve_llama_bin(cfg, paths).ok_or_else(|| {
        LocalCodeError::new(
            ErrorCode::BackendBinaryMissing,
            "PrismML llama-server not found for the local assistant",
        )
        .with_hint(
            "Accept the assistant install offer (builds PrismML-Eng/llama.cpp) \
             or run localcode setup",
        )
        .with_hint("https://github.com/PrismML-Eng/llama.cpp")
    })?;

    let gguf = model_path(paths);
    if !gguf.is_file() {
        return Err(LocalCodeError::new(
            ErrorCode::DeployDownloadFailed,
            format!("Bonsai GGUF not found at {}", gguf.display()),
        )
        .with_hint(format!(
            "Run /assistant install to download {BONSAI_FILE} (~1.8 GB)"
        ))
        .with_hint("https://huggingface.co/prism-ml/Bonsai-27B-gguf")
        .retryable(true));
    }

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
        mark_ready(paths);
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

    let ngl = cfg.assistant.local_gpu_layers;
    // Model card example uses -ngl 99; keep optional -c from config when set.
    let ctx = cfg.assistant.local_context.max(2048);

    info!(
        bin = %bin.display(),
        model = %gguf.display(),
        port,
        ngl,
        "starting local assistant llama-server (-m Q4_1)"
    );

    // Model card: llama-server -m Bonsai-….gguf --host … --port … -ngl 99
    let args: Vec<String> = vec![
        "-m".into(),
        gguf.display().to_string(),
        "--host".into(),
        host.into(),
        "--port".into(),
        port.to_string(),
        "-ngl".into(),
        ngl.to_string(),
        "-c".into(),
        ctx.to_string(),
        // Generation defaults from the Bonsai card (thinking mode).
        "--temp".into(),
        BONSAI_TEMPERATURE.to_string(),
        "--top-p".into(),
        BONSAI_TOP_P.to_string(),
        "--top-k".into(),
        BONSAI_TOP_K.to_string(),
        // OpenAI API + tool calling templates.
        "--jinja".into(),
    ];

    let mut cmd = tokio::process::Command::new(&bin);
    cmd.args(&args)
        .kill_on_drop(true)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    // HF token not required for -m of a local file, but keep for any runtime
    // fetches the binary may still attempt.
    if let Some(token) = cfg.hf_token() {
        cmd.env("HF_TOKEN", &token);
        cmd.env("HUGGING_FACE_HUB_TOKEN", &token);
    }

    let mut child = cmd.spawn().map_err(|e| {
        LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
            .with_cause("Failed to spawn llama-server for the assistant")
            .with_hint(format!(
                "Tried: {} -m {} --host {host} --port {port} -ngl {ngl}",
                bin.display(),
                gguf.display()
            ))
            .retryable(true)
    })?;

    let stderr_tail = drain_child_io_capture("assistant-llama", &mut child);

    let runtime = LocalAssistantRuntime {
        child: Arc::new(Mutex::new(Some(child))),
        base_url: base_url.clone(),
        port,
        model_id: ASSISTANT_MODEL_ID.into(),
    };

    // Wait for health (model load can take a while on CPU).
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(600);
    loop {
        if runtime.is_healthy().await {
            mark_ready(paths);
            info!(
                port,
                model = %gguf.display(),
                "local assistant server ready"
            );
            return Ok(runtime);
        }
        // Child died?
        {
            let mut guard = runtime.child.lock().await;
            if let Some(child) = guard.as_mut() {
                if let Ok(Some(status)) = child.try_wait() {
                    *guard = None;
                    let log = stderr_tail.lock().await.clone();
                    let log_snip = if log.is_empty() {
                        "(no stderr captured — check logs)".into()
                    } else {
                        // Keep last ~2 KB of server output for the error dialog.
                        let bytes = log.as_bytes();
                        let start = bytes.len().saturating_sub(2048);
                        String::from_utf8_lossy(&bytes[start..]).into_owned()
                    };
                    return Err(LocalCodeError::new(
                        ErrorCode::BackendStartFailed,
                        format!("Assistant llama-server exited: {status}"),
                    )
                    .with_cause(format!(
                        "Command was: llama-server -m {BONSAI_FILE} --host {host} --port {port} -ngl {ngl}"
                    ))
                    .with_cause(log_snip)
                    .with_hint("Requires PrismML llama.cpp (custom 1-bit / DSpark kernels)")
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
            .with_hint(format!(
                "Loading {BONSAI_FILE}; check free RAM/VRAM (several GB peak)"
            ))
            .retryable(true));
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }
}

fn port_in_use(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_err()
}

/// Drain stdout/stderr to tracing; also accumulate stderr for failure messages.
fn drain_child_io_capture(
    tag: &str,
    child: &mut Child,
) -> Arc<Mutex<String>> {
    use tokio::io::{AsyncBufReadExt, BufReader};
    let stderr_acc = Arc::new(Mutex::new(String::new()));
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
        let acc = stderr_acc.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::info!(target: "assistant_io", backend = %tag, "{line}");
                let mut g = acc.lock().await;
                if g.len() < 16_384 {
                    g.push_str(&line);
                    g.push('\n');
                }
            }
        });
    }
    stderr_acc
}

/// Soft check used by the TUI without starting the server.
/// Ready when a **PrismML** llama-server is present and the Q4_1 GGUF is on disk.
pub fn is_installed(_cfg: &Config, paths: &AppPaths) -> bool {
    localcode_backends::resolve_prism_llamacpp_bin(paths).is_some() && model_installed(paths)
}

/// Note about the -m launch path and custom runtime.
pub fn quant_compatibility_note() -> &'static str {
    "The assistant starts with: llama-server -m Bonsai-27B-dspark-Q4_1.gguf \
     --host 127.0.0.1 --port <local_port> -ngl 99 \
     using the PrismML llama.cpp fork. Stock llama.cpp cannot load this model."
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
