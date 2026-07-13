use crate::{
    BackendKind, DetectReport, InferenceBackend, LlamaCppBackend, OllamaBackend, SglangBackend,
    VllmBackend,
};
use localcode_core::config::Config;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::runtime::ActiveRuntime;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Registry of backends and active runtimes.
pub struct BackendRegistry {
    backends: HashMap<BackendKind, Arc<dyn InferenceBackend>>,
    runtimes: RwLock<Vec<ActiveRuntime>>,
}

impl BackendRegistry {
    pub fn from_config(cfg: &Config) -> Self {
        let mut backends: HashMap<BackendKind, Arc<dyn InferenceBackend>> = HashMap::new();
        backends.insert(
            BackendKind::Ollama,
            Arc::new(OllamaBackend::new(cfg.backends.ollama.clone())),
        );
        backends.insert(
            BackendKind::LlamaCpp,
            Arc::new(LlamaCppBackend::new(cfg.backends.llamacpp.clone())),
        );
        backends.insert(
            BackendKind::Vllm,
            Arc::new(VllmBackend::new(cfg.backends.vllm.clone())),
        );
        backends.insert(
            BackendKind::Sglang,
            Arc::new(SglangBackend::new(cfg.backends.sglang.clone())),
        );
        Self {
            backends,
            runtimes: RwLock::new(Vec::new()),
        }
    }

    pub fn get(&self, kind: BackendKind) -> Result<Arc<dyn InferenceBackend>, LocalCodeError> {
        self.backends.get(&kind).cloned().ok_or_else(|| {
            LocalCodeError::new(ErrorCode::BackendNotFound, format!("Unknown backend {kind:?}"))
        })
    }

    pub fn default_kind(&self, cfg: &Config) -> BackendKind {
        BackendKind::parse(&cfg.backends.default.kind).unwrap_or(BackendKind::Ollama)
    }

    pub async fn detect_all(&self) -> Vec<DetectReport> {
        let mut out = Vec::new();
        for kind in [
            BackendKind::Ollama,
            BackendKind::LlamaCpp,
            BackendKind::Vllm,
            BackendKind::Sglang,
        ] {
            if let Ok(b) = self.get(kind) {
                out.push(b.detect().await);
            }
        }
        out
    }

    pub async fn register_runtime(&self, runtime: ActiveRuntime) {
        self.runtimes.write().await.push(runtime);
    }

    pub async fn list_runtimes(&self) -> Vec<ActiveRuntime> {
        self.runtimes.read().await.clone()
    }

    pub async fn remove_runtime(&self, id: &str) {
        self.runtimes.write().await.retain(|r| r.id.to_string() != id);
    }
}
