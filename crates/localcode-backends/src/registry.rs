use crate::{
    BackendKind, ColibriBackend, DetectReport, InferenceBackend, LlamaCppBackend, ModelMonitors,
    OllamaBackend, SglangBackend, VllmBackend,
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
    /// Live per-model process monitors backing the `/dash` view. One store,
    /// shared (cloned) into every backend so each can register/refresh its own
    /// spawned processes.
    monitors: ModelMonitors,
}

impl BackendRegistry {
    pub fn from_config(cfg: &Config) -> Self {
        let monitors = ModelMonitors::new();
        let mut backends: HashMap<BackendKind, Arc<dyn InferenceBackend>> = HashMap::new();
        backends.insert(
            BackendKind::Ollama,
            Arc::new(OllamaBackend::new(cfg.backends.ollama.clone()).with_monitors(monitors.clone())),
        );
        backends.insert(
            BackendKind::LlamaCpp,
            Arc::new(
                LlamaCppBackend::new(cfg.backends.llamacpp.clone()).with_monitors(monitors.clone()),
            ),
        );
        backends.insert(
            BackendKind::Vllm,
            Arc::new(VllmBackend::new(cfg.backends.vllm.clone()).with_monitors(monitors.clone())),
        );
        backends.insert(
            BackendKind::Sglang,
            Arc::new(SglangBackend::new(cfg.backends.sglang.clone()).with_monitors(monitors.clone())),
        );
        backends.insert(
            BackendKind::Colibri,
            Arc::new(
                ColibriBackend::new(BackendKind::Colibri, cfg.backends.colibri.clone())
                    .with_monitors(monitors.clone()),
            ),
        );
        backends.insert(
            BackendKind::ColibriHy3,
            Arc::new(
                ColibriBackend::new(BackendKind::ColibriHy3, cfg.backends.colibri_hy3.clone())
                    .with_monitors(monitors.clone()),
            ),
        );
        Self {
            backends,
            runtimes: RwLock::new(Vec::new()),
            monitors,
        }
    }

    /// The shared model-monitor store (read by the `/dash` view, written to by
    /// the deploy pipeline to attach VRAM estimates).
    pub fn monitors(&self) -> ModelMonitors {
        self.monitors.clone()
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
            BackendKind::Colibri,
            BackendKind::ColibriHy3,
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
        // The dashboard card goes away with the runtime.
        self.monitors.remove(id);
    }

    /// Stop a runtime's backing process (when we own one) and deregister it.
    pub async fn stop_runtime(&self, id: &str) -> Result<(), LocalCodeError> {
        let runtime = self
            .runtimes
            .read()
            .await
            .iter()
            .find(|r| r.id.to_string() == id)
            .cloned();
        if let Some(rt) = runtime {
            if let Some(kind) = BackendKind::from_runtime_kind(rt.kind) {
                if let Ok(backend) = self.get(kind) {
                    backend.stop(id).await?;
                }
            }
        }
        self.remove_runtime(id).await;
        Ok(())
    }

    /// Stop every managed runtime and deregister it. Called on app shutdown so
    /// vLLM/llama.cpp/SGLang servers (and the VRAM they hold) don't outlive the
    /// TUI — the quit dialog promises exactly this. A single backend failing to
    /// stop must not abort the others, so per-runtime errors are swallowed.
    pub async fn stop_all(&self) {
        let ids: Vec<String> = self
            .runtimes
            .read()
            .await
            .iter()
            .map(|r| r.id.to_string())
            .collect();
        for id in ids {
            let _ = self.stop_runtime(&id).await;
        }
    }
}
