use crate::{
    port_in_use, probe_client, spawn_io_drain, BackendKind, DetectReport, Health,
    InferenceBackend, ModelDeploySpec, RunningEndpoint,
};
use async_trait::async_trait;
use localcode_core::config::LlamaCppConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus};
use localcode_core::runtime::{ActiveRuntime, RuntimeStatus};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

pub struct LlamaCppBackend {
    cfg: LlamaCppConfig,
    http: reqwest::Client,
    /// Track child processes by runtime id. kill_on_drop: managed runtimes
    /// stop when LocalCode exits (the quit dialog says so).
    children: Arc<Mutex<Vec<(String, tokio::process::Child)>>>,
}

impl LlamaCppBackend {
    pub fn new(cfg: LlamaCppConfig) -> Self {
        Self {
            cfg,
            http: probe_client(),
            children: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl InferenceBackend for LlamaCppBackend {
    fn name(&self) -> BackendKind {
        BackendKind::LlamaCpp
    }

    async fn detect(&self) -> DetectReport {
        let binary = which::which(&self.cfg.bin)
            .or_else(|_| which::which("llama-server"))
            .ok()
            .map(|p| p.display().to_string());
        let mut notes = vec![];
        if binary.is_none() {
            notes.push(format!("`{}` not found on PATH", self.cfg.bin));
            notes.push("Install llama.cpp server build and add to PATH".into());
            notes.push("https://github.com/ggerganov/llama.cpp".into());
        }
        DetectReport {
            kind: BackendKind::LlamaCpp,
            installed: binary.is_some(),
            version: None,
            base_url: Some(format!("http://{}:{}", self.cfg.host, self.cfg.port)),
            binary_path: binary.clone(),
            notes,
            ready: binary.is_some(),
        }
    }

    async fn ensure_ready(&self) -> Result<(), LocalCodeError> {
        let d = self.detect().await;
        if d.ready {
            return Ok(());
        }
        Err(LocalCodeError::new(
            ErrorCode::BackendBinaryMissing,
            format!("llama.cpp binary `{}` not found", self.cfg.bin),
        )
        .with_source("llamacpp", "ensure_ready")
        .with_cause("Binary not on PATH")
        .with_hint("Install llama-server and set backends.llamacpp.bin in config")
        .retryable(false))
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

        if port_in_use(port) {
            return Err(LocalCodeError::new(
                ErrorCode::BackendPortInUse,
                format!("Port {port} is already in use"),
            )
            .with_correlation(cid)
            .with_cause("Another process is bound to this port")
            .with_hint("Stop the other process or pick a different port in Deploy panel")
            .retryable(true));
        }

        let model_path = spec.local_path.clone().ok_or_else(|| {
            LocalCodeError::new(
                ErrorCode::DeployDownloadFailed,
                "llama.cpp deploy requires a local GGUF path",
            )
            .with_correlation(cid)
            .with_hint("Pick a quantization with GGUF files so LocalCode can download it")
            .with_cause("No local_path in deploy spec")
        })?;

        if !std::path::Path::new(&model_path).exists() {
            return Err(LocalCodeError::new(
                ErrorCode::DeployDownloadFailed,
                format!("Model file not found: {model_path}"),
            )
            .with_correlation(cid)
            .retryable(false));
        }

        let bin = which::which(&self.cfg.bin)
            .or_else(|_| which::which("llama-server"))
            .map_err(|_| {
                LocalCodeError::new(ErrorCode::BackendBinaryMissing, "llama-server not found")
                    .with_correlation(cid)
            })?;

        info!(%cid, %model_path, port, "starting llama-server");
        events.publish(AppEvent::DeployProgress {
            job_id: spec.job_id.clone(),
            percent: 50,
            message: "Starting llama-server".into(),
        });

        let mut child = tokio::process::Command::new(&bin)
            .args([
                "-m",
                &model_path,
                "--host",
                &self.cfg.host,
                "--port",
                &port.to_string(),
                "-c",
                &spec.context_length.to_string(),
            ])
            .kill_on_drop(true)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                    .with_correlation(cid)
                    .with_cause("Failed to spawn llama-server")
                    .with_hint("Check CUDA/driver compatibility if using GPU build")
                    .retryable(true)
            })?;
        // Drain output into tracing; llama-server logs every request and an
        // undrained pipe would eventually block the whole server.
        spawn_io_drain("llama-server".into(), &mut child);

        let base_url = format!("http://{}:{}", self.cfg.host, port);
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(60);
        let mut last_progress = tokio::time::Instant::now();
        loop {
            if tokio::time::Instant::now() > deadline {
                let _ = child.kill().await;
                return Err(LocalCodeError::new(
                    ErrorCode::BackendHealthTimeout,
                    "llama-server did not become healthy in time",
                )
                .with_correlation(cid)
                .with_cause("Model load slow or failed")
                .with_cause("CUDA/OOM during load")
                .with_hint("Check logs; try smaller quant; increase timeout")
                .retryable(true));
            }
            if let Ok(h) = self.health(&base_url).await {
                if h.healthy {
                    break;
                }
            }
            if let Ok(Some(status)) = child.try_wait() {
                return Err(LocalCodeError::new(
                    ErrorCode::BackendStartFailed,
                    format!("llama-server exited early: {status}"),
                )
                .with_correlation(cid)
                .with_cause("Invalid GGUF or missing shared libraries")
                .retryable(true));
            }
            if last_progress.elapsed() > tokio::time::Duration::from_secs(5) {
                last_progress = tokio::time::Instant::now();
                events.publish(AppEvent::DeployProgress {
                    job_id: spec.job_id.clone(),
                    percent: 60,
                    message: "Loading model (llama-server)…".into(),
                });
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        let mut runtime = ActiveRuntime::new(
            format!("llamacpp:{}", spec.model_id),
            BackendKind::LlamaCpp.to_runtime_kind(),
            format!("{base_url}/v1"),
        );
        runtime.model_id = Some(spec.model_id.clone());
        runtime.quantization = spec.quantization.clone();
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
        let url = if base_url.contains("/v1") {
            format!("{}/models", base_url.trim_end_matches('/'))
        } else {
            format!("{}/v1/models", base_url.trim_end_matches('/'))
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
