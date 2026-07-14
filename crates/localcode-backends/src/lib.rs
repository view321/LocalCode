//! Local inference backend adapters.

mod deploy;
mod install;
mod ollama;
mod llamacpp;
mod vllm;
mod sglang;
mod registry;

pub use deploy::{DeployJob, DeployProgress, DeployRequest, DeployService};
pub use install::{resolve_install_plan, run_install, InstallPlan, InstallStep};
pub use registry::BackendRegistry;
pub use ollama::OllamaBackend;
pub use llamacpp::LlamaCppBackend;
pub use vllm::VllmBackend;
pub use sglang::SglangBackend;

use async_trait::async_trait;
use localcode_core::error::LocalCodeError;
use localcode_core::events::EventBus;
use localcode_core::runtime::{ActiveRuntime, RuntimeKind};
use serde::{Deserialize, Serialize};

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

/// Check whether a local TCP port is already bound.
pub(crate) fn port_in_use(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_err()
}
