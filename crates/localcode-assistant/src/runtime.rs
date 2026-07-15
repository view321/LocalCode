//! Dedicated llama-server process for the local Bonsai assistant.
//!
//! Launch shape (model card server example):
//! ```text
//! ./build/bin/llama-server \
//!     -m Bonsai-27B-Q1_0.gguf \
//!     [--md Bonsai-27B-dspark-Q4_1.gguf] \
//!     --host 127.0.0.1 --port … -ngl 99
//! ```
//!
//! The Q4_1 file is a **DSpark drafter only** — never pass it as `-m` alone.

use crate::constants::{
    ASSISTANT_MODEL_ID, BONSAI_DRAFT_FILE, BONSAI_FILE, BONSAI_QUANT, BONSAI_TEMPERATURE,
    BONSAI_TOP_K, BONSAI_TOP_P,
};
use crate::install::{
    draft_path, mark_ready, model_installed, model_path, resolve_llama_bin,
};
use localcode_core::config::Config;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::paths::AppPaths;
use localcode_core::runtime::{ActiveRuntime, RuntimeKind, RuntimeStatus};
use std::path::{Path, PathBuf};
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
    if !model_installed(paths) {
        // Helpful message if only the old Q4_1 drafter was downloaded.
        let draft = draft_path(paths);
        let hint = if draft.is_file() {
            format!(
                "{BONSAI_DRAFT_FILE} is the DSpark *drafter*, not the language model. \
                 Download {BONSAI_FILE} (~3.8 GB) via /assistant install"
            )
        } else {
            format!("Run /assistant install to download {BONSAI_FILE} (~3.8 GB)")
        };
        return Err(LocalCodeError::new(
            ErrorCode::DeployDownloadFailed,
            format!("Bonsai language model not found at {}", gguf.display()),
        )
        .with_hint(hint)
        .with_hint("https://huggingface.co/prism-ml/Bonsai-27B-gguf")
        .retryable(true));
    }

    let draft = draft_path(paths);
    let use_draft = gguf_looks_present(&draft);

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
    let ctx = cfg.assistant.local_context.max(2048);

    // Model card: llama-server -m Bonsai-27B-Q1_0.gguf --host … --port … -ngl 99
    let mut args: Vec<String> = vec![
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
        "--temp".into(),
        BONSAI_TEMPERATURE.to_string(),
        "--top-p".into(),
        BONSAI_TOP_P.to_string(),
        "--top-k".into(),
        BONSAI_TOP_K.to_string(),
        "--jinja".into(),
    ];
    if use_draft {
        args.push("-md".into());
        args.push(draft.display().to_string());
    }

    let cmd_display = format!(
        "{} {}",
        bin.display(),
        args.iter()
            .map(|a| {
                if a.contains(' ') {
                    format!("\"{a}\"")
                } else {
                    a.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    );

    info!(
        bin = %bin.display(),
        model = %gguf.display(),
        draft = use_draft,
        port,
        ngl,
        "starting local assistant llama-server (-m Q1_0)"
    );
    // Persist the exact command so failures are diagnosable even if the UI
    // races the stderr drain.
    let _ = write_last_start_log(paths, &format!("spawn: {cmd_display}\n"));

    let mut cmd = tokio::process::Command::new(&bin);
    cmd.args(&args)
        .kill_on_drop(true)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    // Co-located shared libs (CUDA runtime, libggml-*.so) often sit next to the binary.
    if let Some(dir) = bin.parent() {
        cmd.current_dir(dir);
        prepend_lib_path(&mut cmd, dir);
    }

    if let Some(token) = cfg.hf_token() {
        cmd.env("HF_TOKEN", &token);
        cmd.env("HUGGING_FACE_HUB_TOKEN", &token);
    }

    let mut child = cmd.spawn().map_err(|e| {
        LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
            .with_cause("Failed to spawn llama-server for the assistant")
            .with_hint(format!("Tried: {cmd_display}"))
            .retryable(true)
    })?;

    let io_tail = drain_child_io_capture("assistant-llama", &mut child);

    let runtime = LocalAssistantRuntime {
        child: Arc::new(Mutex::new(Some(child))),
        base_url: base_url.clone(),
        port,
        model_id: ASSISTANT_MODEL_ID.into(),
    };

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
        {
            let mut guard = runtime.child.lock().await;
            if let Some(child) = guard.as_mut() {
                if let Ok(Some(status)) = child.try_wait() {
                    *guard = None;
                    // Wait for the async drain tasks to finish reading pipes.
                    let log = flush_io_tail(&io_tail).await;
                    let _ = write_last_start_log(
                        paths,
                        &format!("exit: {status}\ncommand: {cmd_display}\n\n{log}"),
                    );
                    let log_snip = if log.trim().is_empty() {
                        format!(
                            "(no stderr captured) See {} for the last launch attempt",
                            last_start_log_path(paths).display()
                        )
                    } else {
                        tail_chars(&log, 2500)
                    };
                    return Err(LocalCodeError::new(
                        ErrorCode::BackendStartFailed,
                        format!("Assistant llama-server exited: {status}"),
                    )
                    .with_cause(format!("Command: {cmd_display}"))
                    .with_cause(log_snip)
                    .with_hint(format!(
                        "Full output: {}",
                        last_start_log_path(paths).display()
                    ))
                    .with_hint(
                        "Needs PrismML llama.cpp + Bonsai-27B-Q1_0.gguf (not the Q4_1 drafter alone)",
                    )
                    .with_hint("https://huggingface.co/prism-ml/Bonsai-27B-gguf")
                    .retryable(true));
                }
            }
        }
        if tokio::time::Instant::now() > deadline {
            runtime.stop().await;
            let log = flush_io_tail(&io_tail).await;
            let _ = write_last_start_log(
                paths,
                &format!("timeout\ncommand: {cmd_display}\n\n{log}"),
            );
            return Err(LocalCodeError::new(
                ErrorCode::BackendHealthTimeout,
                "Assistant llama-server did not become healthy in time",
            )
            .with_hint(format!(
                "Loading {BONSAI_FILE}; check free RAM/VRAM (~5+ GB peak at short context)"
            ))
            .with_hint(format!(
                "Last output: {}",
                last_start_log_path(paths).display()
            ))
            .retryable(true));
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }
}

fn gguf_looks_present(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.len() > 1_000_000)
        .unwrap_or(false)
}

fn port_in_use(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_err()
}

fn last_start_log_path(paths: &AppPaths) -> PathBuf {
    paths.assistant_dir().join("last-llama-server.log")
}

fn write_last_start_log(paths: &AppPaths, body: &str) -> std::io::Result<()> {
    let dir = paths.assistant_dir();
    std::fs::create_dir_all(&dir)?;
    std::fs::write(last_start_log_path(paths), body)
}

fn prepend_lib_path(cmd: &mut tokio::process::Command, dir: &Path) {
    let dir_s = dir.display().to_string();
    #[cfg(unix)]
    {
        let key = if cfg!(target_os = "macos") {
            "DYLD_LIBRARY_PATH"
        } else {
            "LD_LIBRARY_PATH"
        };
        let mut val = dir_s.clone();
        if let Ok(old) = std::env::var(key) {
            if !old.is_empty() {
                val = format!("{val}:{old}");
            }
        }
        cmd.env(key, val);
    }
    #[cfg(windows)]
    {
        // Prepend bin dir so co-located CUDA/runtime DLLs resolve.
        let key = "PATH";
        let mut val = dir_s;
        if let Ok(old) = std::env::var(key) {
            if !old.is_empty() {
                val = format!("{val};{old}");
            }
        }
        cmd.env(key, val);
    }
}

/// Drain stdout/stderr to tracing + an in-memory buffer (and keep growing after exit).
fn drain_child_io_capture(tag: &str, child: &mut Child) -> Arc<Mutex<String>> {
    use tokio::io::{AsyncBufReadExt, BufReader};
    let acc = Arc::new(Mutex::new(String::new()));
    if let Some(stdout) = child.stdout.take() {
        let tag = tag.to_string();
        let acc = acc.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::info!(target: "assistant_io", backend = %tag, stream = "stdout", "{line}");
                let mut g = acc.lock().await;
                if g.len() < 48_000 {
                    g.push_str("[out] ");
                    g.push_str(&line);
                    g.push('\n');
                }
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
        let tag = tag.to_string();
        let acc = acc.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::info!(target: "assistant_io", backend = %tag, stream = "stderr", "{line}");
                let mut g = acc.lock().await;
                if g.len() < 48_000 {
                    g.push_str(&line);
                    g.push('\n');
                }
            }
        });
    }
    acc
}

/// After the child exits, wait briefly so drain tasks can finish reading pipes.
async fn flush_io_tail(acc: &Arc<Mutex<String>>) -> String {
    let mut last = 0usize;
    for _ in 0..40 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let len = acc.lock().await.len();
        if len > 0 && len == last {
            break;
        }
        last = len;
    }
    // One more beat for a final partial line.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    acc.lock().await.clone()
}

fn tail_chars(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        s.to_string()
    } else {
        chars[chars.len() - max..].iter().collect()
    }
}

/// Soft check used by the TUI without starting the server.
pub fn is_installed(_cfg: &Config, paths: &AppPaths) -> bool {
    localcode_backends::resolve_prism_llamacpp_bin(paths).is_some() && model_installed(paths)
}

/// Note about the launch path and custom runtime.
pub fn quant_compatibility_note() -> &'static str {
    "The assistant starts with: llama-server -m Bonsai-27B-Q1_0.gguf \
     [--md Bonsai-27B-dspark-Q4_1.gguf] --host 127.0.0.1 -ngl 99 \
     on the PrismML llama.cpp fork. Q4_1 alone is a DSpark drafter, not a full model."
}

impl Drop for LocalAssistantRuntime {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.child.try_lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.start_kill();
                warn!("local assistant runtime dropped — killing llama-server");
            }
        }
    }
}
