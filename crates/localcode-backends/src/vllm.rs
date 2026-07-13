use crate::{
    BackendKind, DetectReport, Health, InferenceBackend, ModelDeploySpec, RunningEndpoint,
};
use async_trait::async_trait;
use localcode_core::config::VllmConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::runtime::{ActiveRuntime, RuntimeStatus};
use tracing::info;

pub struct VllmBackend {
    cfg: VllmConfig,
    http: reqwest::Client,
}

impl VllmBackend {
    pub fn new(cfg: VllmConfig) -> Self {
        Self {
            cfg,
            http: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl InferenceBackend for VllmBackend {
    fn name(&self) -> BackendKind {
        BackendKind::Vllm
    }

    async fn detect(&self) -> DetectReport {
        let binary = which::which(&self.cfg.bin)
            .or_else(|_| which::which("vllm"))
            .ok()
            .map(|p| p.display().to_string());
        let mut notes = vec![];
        if binary.is_none() {
            notes.push("vLLM not found on PATH (Linux preferred)".into());
            notes.push("pip install vllm  # or use Docker image".into());
            notes.push("Windows: best-effort only; prefer Linux host".into());
        }
        DetectReport {
            kind: BackendKind::Vllm,
            installed: binary.is_some(),
            version: None,
            base_url: Some(format!("http://{}:{}", self.cfg.host, self.cfg.port)),
            binary_path: binary.clone(),
            notes,
            ready: binary.is_some(),
        }
    }

    async fn ensure_ready(&self) -> Result<(), LocalCodeError> {
        if self.detect().await.ready {
            Ok(())
        } else {
            Err(LocalCodeError::new(
                ErrorCode::BackendBinaryMissing,
                "vLLM is not installed",
            )
            .with_hint("Install vLLM on Linux for best results")
            .with_hint("Documented Windows limitation — use WSL or remote host"))
        }
    }

    async fn list_models(&self) -> Result<Vec<String>, LocalCodeError> {
        Ok(vec![])
    }

    async fn deploy(&self, spec: ModelDeploySpec) -> Result<RunningEndpoint, LocalCodeError> {
        self.ensure_ready().await?;
        let cid = localcode_core::CorrelationId::new();
        let port = spec.port.unwrap_or(self.cfg.port);
        let model = spec.local_path.clone().unwrap_or_else(|| spec.model_id.clone());

        info!(%cid, %model, "starting vllm serve");
        let mut child = tokio::process::Command::new(&self.cfg.bin)
            .args([
                "serve",
                &model,
                "--host",
                &self.cfg.host,
                "--port",
                &port.to_string(),
            ])
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                    .with_correlation(cid)
                    .with_cause("Failed to spawn vllm")
                    .with_hint("Check CUDA version compatibility")
                    .retryable(true)
            })?;

        let base_url = format!("http://{}:{}/v1", self.cfg.host, port);
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(180);
        loop {
            if tokio::time::Instant::now() > deadline {
                let _ = child.kill().await;
                return Err(LocalCodeError::new(
                    ErrorCode::BackendHealthTimeout,
                    "vLLM did not become healthy in time",
                )
                .with_correlation(cid)
                .retryable(true));
            }
            if let Ok(h) = self.health(&base_url).await {
                if h.healthy {
                    break;
                }
            }
            if let Ok(Some(st)) = child.try_wait() {
                return Err(LocalCodeError::new(
                    ErrorCode::BackendStartFailed,
                    format!("vLLM exited: {st}"),
                )
                .with_correlation(cid));
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }

        // Keep child alive by leaking into background — process supervised by OS until stop
        // For v1 we don't track child in a registry here; DeployService does.
        std::mem::forget(child);

        let mut runtime = ActiveRuntime::new(
            format!("vllm:{}", spec.model_id),
            BackendKind::Vllm.to_runtime_kind(),
            base_url,
        );
        runtime.model_id = Some(spec.model_id);
        runtime.quantization = spec.quantization;
        runtime.status = RuntimeStatus::Healthy;
        runtime.correlation_id = cid.to_string();
        Ok(RunningEndpoint { runtime })
    }

    async fn stop(&self, _runtime_id: &str) -> Result<(), LocalCodeError> {
        Ok(())
    }

    async fn health(&self, base_url: &str) -> Result<Health, LocalCodeError> {
        let url = format!("{}/models", base_url.trim_end_matches('/'));
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
