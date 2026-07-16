//! Local inference backend adapters.

mod colibri;
mod deploy;
mod diagnose;
mod download;
mod install;
mod models_store;
mod monitor;
mod ollama;
mod openwebui;
mod llamacpp;
mod proc;
mod smoke;
mod vllm;
mod sglang;
mod registry;

pub use deploy::{
    preview_deploy_command, DeployJob, DeployProgress, DeployRequest, DeployService,
};
pub use download::{
    clear_state, model_dir, now_unix, read_state, run_download, run_worker, scan_active,
    spawn_detached_worker, write_progress_state, write_state, DownloadSpec, DownloadState,
    DownloadStatus, DOWNLOAD_STATE_FILE, STALE_AFTER_SECS,
};
pub use diagnose::{classify, diagnose, Confidence, Diagnosis, FailureClass, RepairIntent};
pub use models_store::{
    delete_downloaded, find_downloaded, human_size, is_downloaded, list_downloaded,
    sanitize_model_dir, DownloadedModel,
};
pub use openwebui::{OpenWebUi, OpenWebUiHandle, OPENWEBUI_CONTAINER, OPENWEBUI_DEFAULT_PORT};
pub use monitor::{
    capture_into_monitor, format_command, resolve_launch, split_command, spawn_exit_watch,
    DashSnapshot, ModelMonitor, ModelMonitors, ProcState, DASH_LOG_CAP,
};
pub use install::{
    can_elevate_noninteractively, colibri_managed_dir, ensure_llamacpp_installed,
    llamacpp_managed_dir, resolve_colibri_bin, resolve_install_plan, resolve_llamacpp_bin,
    resolve_prism_llamacpp_bin, resolve_repair, run_install, run_repair, InstallPlan, InstallStep,
    RepairPlan, Repoint,
};
pub use registry::BackendRegistry;
pub use smoke::{smoke_test, SmokeReport};
pub use colibri::ColibriBackend;
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
    /// Colibrì — GLM-5.2 expert-streaming engine (github.com/JustVugg/colibri).
    Colibri,
    /// Colibrì × Hy3 fork (github.com/ErikTromp/colibri-hy3) — Tencent Hy3
    /// (`model_type: hy_v3`); also serves GLM-5.2 containers.
    ColibriHy3,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::LlamaCpp => "llamacpp",
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
            Self::Colibri => "colibri",
            Self::ColibriHy3 => "colibri-hy3",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ollama" => Some(Self::Ollama),
            "llamacpp" | "llama.cpp" | "llama_cpp" => Some(Self::LlamaCpp),
            "vllm" => Some(Self::Vllm),
            "sglang" => Some(Self::Sglang),
            "colibri" | "coli" => Some(Self::Colibri),
            "colibri-hy3" | "colibri_hy3" | "colibrihy3" | "hy3" => Some(Self::ColibriHy3),
            _ => None,
        }
    }

    pub fn to_runtime_kind(self) -> RuntimeKind {
        match self {
            Self::Ollama => RuntimeKind::Ollama,
            Self::LlamaCpp => RuntimeKind::LlamaCpp,
            Self::Vllm => RuntimeKind::Vllm,
            Self::Sglang => RuntimeKind::Sglang,
            Self::Colibri => RuntimeKind::Colibri,
            Self::ColibriHy3 => RuntimeKind::ColibriHy3,
        }
    }

    pub fn from_runtime_kind(kind: RuntimeKind) -> Option<Self> {
        match kind {
            RuntimeKind::Ollama => Some(Self::Ollama),
            RuntimeKind::LlamaCpp => Some(Self::LlamaCpp),
            RuntimeKind::Vllm => Some(Self::Vllm),
            RuntimeKind::Sglang => Some(Self::Sglang),
            RuntimeKind::Colibri => Some(Self::Colibri),
            RuntimeKind::ColibriHy3 => Some(Self::ColibriHy3),
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
    /// Full launch command entered by the user (or assistant) that replaces the
    /// one this backend would build. When set, the backend shell-splits it into
    /// program + args and spawns exactly that. `None` = use the built command.
    #[serde(default)]
    pub command_override: Option<String>,
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

// The tail-retaining capture used by managed servers now lives in
// `monitor::capture_into_monitor`, which drains into a model's dashboard log
// ring instead of a throwaway buffer.

/// Check whether a local TCP port is already bound.
pub(crate) fn port_in_use(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_err()
}
