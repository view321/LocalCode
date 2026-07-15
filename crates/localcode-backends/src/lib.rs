//! Local inference backend adapters.

mod deploy;
mod diagnose;
mod install;
mod ollama;
mod llamacpp;
mod smoke;
mod vllm;
mod sglang;
mod registry;

pub use deploy::{DeployJob, DeployProgress, DeployRequest, DeployService};
pub use diagnose::{classify, diagnose, Confidence, Diagnosis, FailureClass, RepairIntent};
pub use install::{
    can_elevate_noninteractively, ensure_llamacpp_installed, llamacpp_managed_dir,
    resolve_install_plan, resolve_llamacpp_bin, resolve_prism_llamacpp_bin, resolve_repair,
    run_install, run_repair, InstallPlan, InstallStep, RepairPlan, Repoint,
};
pub use registry::BackendRegistry;
pub use smoke::{smoke_test, SmokeReport};
pub use ollama::OllamaBackend;
pub use llamacpp::LlamaCppBackend;
pub use vllm::VllmBackend;
pub use sglang::SglangBackend;

use async_trait::async_trait;
use localcode_core::error::LocalCodeError;
use localcode_core::events::EventBus;
use localcode_core::runtime::{ActiveRuntime, RuntimeKind};
use serde::{Deserialize, Serialize};

/// Default deploy context length. Also the sentinel the Ollama backend uses to
/// decide whether the user customized context (and so needs a derived model
/// with an overridden `num_ctx`); the TUI seeds `deploy_ctx` from this too.
pub const DEFAULT_DEPLOY_CTX: u32 = 8192;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Ollama,
    LlamaCpp,
    Vllm,
    Sglang,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::LlamaCpp => "llamacpp",
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ollama" => Some(Self::Ollama),
            "llamacpp" | "llama.cpp" | "llama_cpp" => Some(Self::LlamaCpp),
            "vllm" => Some(Self::Vllm),
            "sglang" => Some(Self::Sglang),
            _ => None,
        }
    }

    pub fn to_runtime_kind(self) -> RuntimeKind {
        match self {
            Self::Ollama => RuntimeKind::Ollama,
            Self::LlamaCpp => RuntimeKind::LlamaCpp,
            Self::Vllm => RuntimeKind::Vllm,
            Self::Sglang => RuntimeKind::Sglang,
        }
    }

    pub fn from_runtime_kind(kind: RuntimeKind) -> Option<Self> {
        match kind {
            RuntimeKind::Ollama => Some(Self::Ollama),
            RuntimeKind::LlamaCpp => Some(Self::LlamaCpp),
            RuntimeKind::Vllm => Some(Self::Vllm),
            RuntimeKind::Sglang => Some(Self::Sglang),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectReport {
    pub kind: BackendKind,
    pub installed: bool,
    pub version: Option<String>,
    pub base_url: Option<String>,
    pub binary_path: Option<String>,
    pub notes: Vec<String>,
    pub ready: bool,
}

/// Optional, backend-specific launch tuning chosen by the user at deploy time
/// (or by the local assistant after reading a model card). Fields the backend
/// does not understand are ignored.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeployTuning {
    /// Fraction (0.0–1.0) of VRAM the server may use. vLLM
    /// `--gpu-memory-utilization`, SGLang `--mem-fraction-static`.
    pub gpu_memory_fraction: Option<f32>,
    /// Number of GPUs to shard across. vLLM `--tensor-parallel-size`,
    /// SGLang `--tp-size`.
    pub tensor_parallel: Option<u32>,
    /// Layers to offload to the GPU. llama.cpp `--n-gpu-layers`, Ollama
    /// `num_gpu`. Negative (llama.cpp convention) offloads all layers.
    pub gpu_layers: Option<i32>,
    /// Extra CLI flags recommended by the model card (or set by the user).
    /// Appended after the standard args for llama.cpp / vLLM / SGLang.
    #[serde(default)]
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDeploySpec {
    pub job_id: String,
    pub model_id: String,
    pub quantization: Option<String>,
    pub weight_files: Vec<String>,
    pub download_urls: Vec<String>,
    pub local_path: Option<String>,
    pub port: Option<u16>,
    pub context_length: u32,
    pub force_oversize: bool,
    /// Per-backend launch tuning (VRAM fraction, tensor-parallel, GPU layers).
    #[serde(default)]
    pub tuning: DeployTuning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningEndpoint {
    pub runtime: ActiveRuntime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Health {
    pub healthy: bool,
    pub message: String,
}

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    fn name(&self) -> BackendKind;
    async fn detect(&self) -> DetectReport;
    async fn ensure_ready(&self) -> Result<(), LocalCodeError>;
    async fn list_models(&self) -> Result<Vec<String>, LocalCodeError>;
    async fn deploy(
        &self,
        spec: ModelDeploySpec,
        events: &EventBus,
    ) -> Result<RunningEndpoint, LocalCodeError>;
    async fn stop(&self, runtime_id: &str) -> Result<(), LocalCodeError>;
    async fn health(&self, base_url: &str) -> Result<Health, LocalCodeError>;
}

/// Build a reqwest client for health/detect probes: short timeouts so a dead
/// endpoint can't stall the UI or a deploy pipeline.
pub(crate) fn probe_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(3))
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default()
}

/// Drain a child's stdout/stderr into tracing.
///
/// Every spawned server MUST go through this (or Stdio::null): piped-but-
/// unread output deadlocks the child once the pipe buffer fills, and
/// inherited output writes straight over the raw-mode TUI.
pub(crate) fn spawn_io_drain(tag: String, child: &mut tokio::process::Child) {
    use tokio::io::{AsyncBufReadExt, BufReader};
    if let Some(stdout) = child.stdout.take() {
        let tag = tag.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(target: "backend_io", backend = %tag, "{line}");
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(target: "backend_io", backend = %tag, "{line}");
            }
        });
    }
}

/// Like [`spawn_io_drain`] but also retains the last `cap` combined output lines
/// in a ring buffer. A failing server — vLLM especially — prints *why* it died
/// (unsupported architecture, CUDA OOM, a too-large `--max-model-len`) to
/// stderr; the plain drain buries that at debug level, leaving a deploy with
/// only a bare exit status. Callers read this tail into the surfaced error.
pub(crate) fn spawn_io_capture(
    tag: String,
    child: &mut tokio::process::Child,
    cap: usize,
) -> std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>> {
    let buf = std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));
    capture_lines(child.stdout.take(), tag.clone(), buf.clone(), cap);
    capture_lines(child.stderr.take(), tag, buf.clone(), cap);
    buf
}

fn capture_lines<R>(
    reader: Option<R>,
    tag: String,
    buf: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    cap: usize,
) where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    use tokio::io::{AsyncBufReadExt, BufReader};
    let Some(reader) = reader else { return };
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::debug!(target: "backend_io", backend = %tag, "{line}");
            if let Ok(mut b) = buf.lock() {
                if b.len() >= cap {
                    b.pop_front();
                }
                b.push_back(line);
            }
        }
    });
}

/// Check whether a local TCP port is already bound.
pub(crate) fn port_in_use(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_err()
}
