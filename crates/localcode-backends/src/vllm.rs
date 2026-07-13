use crate::{
    port_in_use, probe_client, spawn_io_drain, BackendKind, DetectReport, Health,
    InferenceBackend, ModelDeploySpec, RunningEndpoint,
};
use async_trait::async_trait;
use localcode_core::config::VllmConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus};
use localcode_core::runtime::{ActiveRuntime, RuntimeStatus};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

pub struct VllmBackend {
    cfg: VllmConfig,
    http: reqwest::Client,
    /// Track children so stop() works and processes are reaped (previously
    /// they were mem::forget-ed: unstoppable and zombied on exit).
    children: Arc<Mutex<Vec<(String, tokio::process::Child)>>>,
}

impl VllmBackend {
    pub fn new(cfg: VllmConfig) -> Self {
        Self {
            cfg,
            http: probe_client(),
            children: Arc::new(Mutex::new(Vec::new())),
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

    async fn deploy(
        &self,
        spec: ModelDeploySpec,
        events: &EventBus,
    ) -> Result<RunningEndpoint, LocalCodeError> {
        self.ensure_ready().await?;
        let cid = localcode_core::CorrelationId::new();
        let port = spec.port.unwrap_or(self.cfg.port);
        let model = spec.local_path.clone().unwrap_or_else(|| spec.model_id.clone());

        if port_in_use(port) {
            return Err(LocalCodeError::new(
                ErrorCode::BackendPortInUse,
                format!("Port {port} is already in use"),
            )
            .with_correlation(cid)
            .with_hint("Stop the other process or pick a different port")
            .retryable(true));
        }

        info!(%cid, %model, "starting vllm serve");
        events.publish(AppEvent::DeployProgress {
            job_id: spec.job_id.clone(),
            percent: 50,
            message: "Starting vLLM (model download may take a while)".into(),
        });

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
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                    .with_correlation(cid)
                    .with_cause("Failed to spawn vllm")
                    .with_hint("Check CUDA version compatibility")
                    .retryable(true)
            })?;
        spawn_io_drain("vllm".into(), &mut child);

        let base_url = format!("http://{}:{}/v1", self.cfg.host, port);
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(180);
        let mut last_progress = tokio::time::Instant::now();
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
            if last_progress.elapsed() > tokio::time::Duration::from_secs(10) {
                last_progress = tokio::time::Instant::now();
                events.publish(AppEvent::DeployProgress {
                    job_id: spec.job_id.clone(),
                    percent: 60,
                    message: "Waiting for vLLM to become healthy…".into(),
                });
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }

        let mut runtime = ActiveRuntime::new(
            format!("vllm:{}", spec.model_id),
            BackendKind::Vllm.to_runtime_kind(),
            base_url,
        );
        runtime.model_id = Some(spec.model_id);
        runtime.quantization = spec.quantization;
        runtime.status = RuntimeStatus::Healthy;
        runtime.correlation_id = cid.to_string();

        self.children
            .lock()
            .await
            .push((runtime.id.to_string(), child));

        Ok(RunningEndpoint { runtime })
    }

    async fn stop(&self, runtime_id: &str) -> Result<(), LocalCodeError> {
        let mut kids = self.children.lock().await;
        if let Some(pos) = kids.iter().position(|(id, _)| id == runtime_id) {
            let (_, mut child) = kids.remove(pos);
            let _ = child.kill().await;
        }
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
