//! Local inference backend adapters.

mod deploy;
mod ollama;
mod llamacpp;
mod vllm;
mod sglang;
mod registry;

pub use deploy::{DeployJob, DeployProgress, DeployRequest, DeployService};
pub use registry::BackendRegistry;
pub use ollama::OllamaBackend;
pub use llamacpp::LlamaCppBackend;
pub use vllm::VllmBackend;
pub use sglang::SglangBackend;

use async_trait::async_trait;
use localcode_core::error::LocalCodeError;
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
    async fn deploy(&self, spec: ModelDeploySpec) -> Result<RunningEndpoint, LocalCodeError>;
    async fn stop(&self, runtime_id: &str) -> Result<(), LocalCodeError>;
    async fn health(&self, base_url: &str) -> Result<Health, LocalCodeError>;
}
