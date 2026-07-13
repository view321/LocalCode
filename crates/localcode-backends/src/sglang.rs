use crate::{
    BackendKind, DetectReport, Health, InferenceBackend, ModelDeploySpec, RunningEndpoint,
};
use async_trait::async_trait;
use localcode_core::config::SglangConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::runtime::{ActiveRuntime, RuntimeStatus};
use tracing::info;

pub struct SglangBackend {
    cfg: SglangConfig,
    http: reqwest::Client,
}

impl SglangBackend {
    pub fn new(cfg: SglangConfig) -> Self {
        Self {
            cfg,
            http: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl InferenceBackend for SglangBackend {
    fn name(&self) -> BackendKind {
        BackendKind::Sglang
    }

    async fn detect(&self) -> DetectReport {
        let binary = which::which(&self.cfg.bin)
            .or_else(|_| which::which("python"))
            .ok()
            .map(|p| p.display().to_string());
        let mut notes = vec![];
        notes.push("SGLang typically launched via: python -m sglang.launch_server".into());
        notes.push("Linux preferred host for SGLang".into());
        DetectReport {
            kind: BackendKind::Sglang,
            installed: which::which("sglang").is_ok() || binary.is_some(),
            version: None,
            base_url: Some(format!("http://{}:{}", self.cfg.host, self.cfg.port)),
            binary_path: binary,
            notes,
            ready: which::which("sglang").is_ok(),
        }
    }

    async fn ensure_ready(&self) -> Result<(), LocalCodeError> {
        if self.detect().await.ready {
            Ok(())
        } else {
            Err(LocalCodeError::new(
                ErrorCode::BackendBinaryMissing,
                "SGLang not found",
            )
            .with_hint("pip install sglang")
            .with_hint("Linux preferred for SGLang"))
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

        info!(%cid, %model, "starting sglang");
        let mut child = tokio::process::Command::new("python")
            .args([
                "-m",
                "sglang.launch_server",
                "--model-path",
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
            })?;

        let base_url = format!("http://{}:{}/v1", self.cfg.host, port);
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(180);
        loop {
            if tokio::time::Instant::now() > deadline {
                let _ = child.kill().await;
                return Err(LocalCodeError::new(
                    ErrorCode::BackendHealthTimeout,
                    "SGLang health timeout",
                )
                .with_correlation(cid));
            }
            if let Ok(h) = self.health(&base_url).await {
                if h.healthy {
                    break;
                }
            }
            if let Ok(Some(st)) = child.try_wait() {
                return Err(LocalCodeError::new(
                    ErrorCode::BackendStartFailed,
                    format!("SGLang exited: {st}"),
                )
                .with_correlation(cid));
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
        std::mem::forget(child);

        let mut runtime = ActiveRuntime::new(
            format!("sglang:{}", spec.model_id),
            BackendKind::Sglang.to_runtime_kind(),
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
