use crate::{
    BackendKind, DetectReport, Health, InferenceBackend, ModelDeploySpec, RunningEndpoint,
};
use async_trait::async_trait;
use localcode_core::config::OllamaConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::runtime::{ActiveRuntime, RuntimeStatus};
use tracing::{info, warn};

pub struct OllamaBackend {
    cfg: OllamaConfig,
    http: reqwest::Client,
}

impl OllamaBackend {
    pub fn new(cfg: OllamaConfig) -> Self {
        Self {
            cfg,
            http: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl InferenceBackend for OllamaBackend {
    fn name(&self) -> BackendKind {
        BackendKind::Ollama
    }

    async fn detect(&self) -> DetectReport {
        let binary = which::which("ollama").ok().map(|p| p.display().to_string());
        let mut notes = vec![];
        let mut version = None;
        let mut ready = false;

        if binary.is_none() {
            notes.push("ollama binary not found on PATH".into());
            notes.push("Install from https://ollama.com and restart the terminal".into());
        }

        match self.http.get(format!("{}/api/tags", self.cfg.base_url)).send().await {
            Ok(resp) if resp.status().is_success() => {
                ready = true;
                version = Some("reachable".into());
            }
            Ok(resp) => {
                notes.push(format!("Ollama HTTP status {}", resp.status()));
            }
            Err(e) => {
                notes.push(format!("Cannot reach {}: {e}", self.cfg.base_url));
                notes.push("Start the Ollama service (ollama serve)".into());
            }
        }

        DetectReport {
            kind: BackendKind::Ollama,
            installed: binary.is_some() || ready,
            version,
            base_url: Some(self.cfg.base_url.clone()),
            binary_path: binary,
            notes,
            ready,
        }
    }

    async fn ensure_ready(&self) -> Result<(), LocalCodeError> {
        let d = self.detect().await;
        if d.ready {
            return Ok(());
        }
        Err(LocalCodeError::new(
            ErrorCode::BackendNotReady,
            "Ollama is not ready",
        )
        .with_source("ollama", "ensure_ready")
        .with_causes_from_notes(&d.notes)
        .with_hint("Install Ollama and run: ollama serve")
        .with_hint("Verify base_url in config (default http://127.0.0.1:11434)")
        .retryable(true))
    }

    async fn list_models(&self) -> Result<Vec<String>, LocalCodeError> {
        self.ensure_ready().await?;
        let resp = self
            .http
            .get(format!("{}/api/tags", self.cfg.base_url))
            .send()
            .await
            .map_err(|e| {
                LocalCodeError::new(ErrorCode::BackendNotReady, e.to_string()).retryable(true)
            })?;
        #[derive(serde::Deserialize)]
        struct Tags {
            models: Vec<Model>,
        }
        #[derive(serde::Deserialize)]
        struct Model {
            name: String,
        }
        let tags: Tags = resp.json().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::BackendNotReady, e.to_string())
        })?;
        Ok(tags.models.into_iter().map(|m| m.name).collect())
    }

    async fn deploy(&self, spec: ModelDeploySpec) -> Result<RunningEndpoint, LocalCodeError> {
        let cid = localcode_core::CorrelationId::new();
        info!(%cid, model = %spec.model_id, "ollama deploy start");

        // Prefer ollama pull of a library name; for HF GGUF paths use model id as name.
        let model_name = ollama_model_name(&spec);
        let pull_body = serde_json::json!({ "name": model_name, "stream": false });

        // Try pull via API
        let pull_url = format!("{}/api/pull", self.cfg.base_url);
        match self.http.post(&pull_url).json(&pull_body).send().await {
            Ok(resp) if resp.status().is_success() => {
                info!(%cid, %model_name, "ollama pull ok");
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                // Fall back to create from local GGUF if provided
                if let Some(path) = &spec.local_path {
                    return self.create_from_gguf(&model_name, path, &spec, cid).await;
                }
                return Err(LocalCodeError::new(
                    ErrorCode::DeployDownloadFailed,
                    format!("Ollama pull failed ({status}): {body}"),
                )
                .with_correlation(cid)
                .with_cause("Model not available in Ollama library")
                .with_cause("Network or mirror issue during pull")
                .with_hint("Try a library model name (e.g. qwen2.5-coder:7b)")
                .with_hint("Or download GGUF and set local_path for Modelfile create")
                .retryable(true));
            }
            Err(e) => {
                // Offline / not ready — still try CLI if binary exists
                warn!(%cid, error = %e, "ollama API pull failed");
                if which::which("ollama").is_ok() {
                    let status = tokio::process::Command::new("ollama")
                        .args(["pull", &model_name])
                        .status()
                        .await
                        .map_err(|e| {
                            LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                                .with_correlation(cid)
                        })?;
                    if !status.success() {
                        return Err(LocalCodeError::new(
                            ErrorCode::DeployDownloadFailed,
                            format!("ollama pull {model_name} failed"),
                        )
                        .with_correlation(cid)
                        .retryable(true));
                    }
                } else {
                    return Err(LocalCodeError::new(
                        ErrorCode::BackendNotReady,
                        format!("Cannot reach Ollama: {e}"),
                    )
                    .with_correlation(cid)
                    .with_hint("Start ollama serve")
                    .retryable(true));
                }
            }
        }

        // OpenAI-compatible endpoint if available, else Ollama native
        let base = self.cfg.base_url.clone();
        let mut runtime = ActiveRuntime::new(
            format!("ollama:{model_name}"),
            BackendKind::Ollama.to_runtime_kind(),
            format!("{base}/v1"),
        );
        runtime.model_id = Some(spec.model_id.clone());
        runtime.quantization = spec.quantization.clone();
        runtime.status = RuntimeStatus::Healthy;
        runtime.correlation_id = cid.to_string();

        Ok(RunningEndpoint { runtime })
    }

    async fn stop(&self, _runtime_id: &str) -> Result<(), LocalCodeError> {
        // Ollama keeps models loaded; unloading is best-effort
        Ok(())
    }

    async fn health(&self, base_url: &str) -> Result<Health, LocalCodeError> {
        let url = if base_url.ends_with("/v1") {
            format!("{}/models", base_url)
        } else {
            format!("{}/api/tags", base_url.trim_end_matches('/'))
        };
        match self.http.get(&url).send().await {
            Ok(r) if r.status().is_success() => Ok(Health {
                healthy: true,
                message: "ok".into(),
            }),
            Ok(r) => Ok(Health {
                healthy: false,
                message: format!("status {}", r.status()),
            }),
            Err(e) => Ok(Health {
                healthy: false,
                message: e.to_string(),
            }),
        }
    }
}

impl OllamaBackend {
    async fn create_from_gguf(
        &self,
        name: &str,
        path: &str,
        spec: &ModelDeploySpec,
        cid: localcode_core::CorrelationId,
    ) -> Result<RunningEndpoint, LocalCodeError> {
        let modelfile = format!("FROM {path}\n");
        let body = serde_json::json!({ "name": name, "modelfile": modelfile });
        let resp = self
            .http
            .post(format!("{}/api/create", self.cfg.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                    .with_correlation(cid)
            })?;
        if !resp.status().is_success() {
            let t = resp.text().await.unwrap_or_default();
            return Err(LocalCodeError::new(
                ErrorCode::BackendStartFailed,
                format!("Ollama create failed: {t}"),
            )
            .with_correlation(cid)
            .with_hint("Ensure the GGUF path is absolute and readable by Ollama"));
        }
        let mut runtime = ActiveRuntime::new(
            format!("ollama:{name}"),
            BackendKind::Ollama.to_runtime_kind(),
            format!("{}/v1", self.cfg.base_url),
        );
        runtime.model_id = Some(spec.model_id.clone());
        runtime.quantization = spec.quantization.clone();
        runtime.status = RuntimeStatus::Healthy;
        runtime.correlation_id = cid.to_string();
        Ok(RunningEndpoint { runtime })
    }
}

fn ollama_model_name(spec: &ModelDeploySpec) -> String {
    // Map HF-style ids to ollama-friendly names when possible
    let id = spec.model_id.to_lowercase();
    if id.contains("qwen2.5-coder") && id.contains("7b") {
        return "qwen2.5-coder:7b".into();
    }
    if id.contains("codellama") {
        return "codellama".into();
    }
    // Default: use last path segment + quant tag
    let base = spec
        .model_id
        .split('/')
        .next_back()
        .unwrap_or(&spec.model_id)
        .to_lowercase()
        .replace(' ', "-");
    if let Some(q) = &spec.quantization {
        format!("{base}:{q}")
    } else {
        base
    }
}

trait NotesExt {
    fn with_causes_from_notes(self, notes: &[String]) -> Self;
}

impl NotesExt for LocalCodeError {
    fn with_causes_from_notes(mut self, notes: &[String]) -> Self {
        for n in notes {
            self.causes.push(n.clone());
        }
        self
    }
}
