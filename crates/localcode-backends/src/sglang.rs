use crate::{
    port_in_use, probe_client, spawn_io_drain, BackendKind, DetectReport, Health,
    InferenceBackend, ModelDeploySpec, RunningEndpoint,
};
use async_trait::async_trait;
use localcode_core::config::SglangConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus};
use localcode_core::runtime::{ActiveRuntime, RuntimeStatus};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

pub struct SglangBackend {
    cfg: SglangConfig,
    http: reqwest::Client,
    children: Arc<Mutex<Vec<(String, tokio::process::Child)>>>,
}

impl SglangBackend {
    pub fn new(cfg: SglangConfig) -> Self {
        Self {
            cfg,
            http: probe_client(),
            children: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Resolve the interpreter to launch SGLang with. After a clean-venv repair
    /// `cfg.bin` is the venv's python (an absolute path); honor it so the repair
    /// actually takes effect. Otherwise fall back to `python3`/`python` on PATH.
    fn python_bin(&self) -> Option<std::path::PathBuf> {
        let configured = std::path::PathBuf::from(&self.cfg.bin);
        if configured.is_absolute() && configured.exists() {
            return Some(configured);
        }
        which::which("python3")
            .or_else(|_| which::which("python"))
            .ok()
    }

    /// SGLang launches as `python -m sglang.launch_server`, so readiness means
    /// "the python interpreter can import sglang" — not a `sglang` binary.
    async fn sglang_importable(&self) -> bool {
        let Some(py) = self.python_bin() else {
            return false;
        };
        let check = tokio::process::Command::new(py)
            .args(["-c", "import sglang"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        matches!(
            tokio::time::timeout(std::time::Duration::from_secs(10), check).await,
            Ok(Ok(status)) if status.success()
        )
    }
}

#[async_trait]
impl InferenceBackend for SglangBackend {
    fn name(&self) -> BackendKind {
        BackendKind::Sglang
    }

    async fn detect(&self) -> DetectReport {
        let python = self.python_bin().map(|p| p.display().to_string());
        let importable = self.sglang_importable().await;
        let mut notes = vec![
            "SGLang is launched via: python -m sglang.launch_server".into(),
            "Linux preferred host for SGLang".into(),
        ];
        if python.is_none() {
            notes.push("python not found on PATH".into());
        } else if !importable {
            notes.push("`import sglang` failed — pip install sglang".into());
        }
        DetectReport {
            kind: BackendKind::Sglang,
            installed: importable,
            version: None,
            base_url: Some(format!("http://{}:{}", self.cfg.host, self.cfg.port)),
            binary_path: python,
            notes,
            ready: importable,
        }
    }

    async fn ensure_ready(&self) -> Result<(), LocalCodeError> {
        if self.detect().await.ready {
            Ok(())
        } else {
            Err(LocalCodeError::new(
                ErrorCode::BackendBinaryMissing,
                "SGLang not found (python -c \"import sglang\" failed)",
            )
            .with_hint("pip install sglang")
            .with_hint("Linux preferred for SGLang"))
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

        let py = self.python_bin().ok_or_else(|| {
            LocalCodeError::new(ErrorCode::BackendBinaryMissing, "python not found")
                .with_correlation(cid)
        })?;

        info!(%cid, %model, "starting sglang");
        events.publish(AppEvent::DeployProgress {
            job_id: spec.job_id.clone(),
            percent: 50,
            message: "Starting SGLang".into(),
        });

        // Built as a Vec so context / tuning flags can be appended. SGLang
        // previously ignored context entirely (KV cache sized to the model's
        // native max); --context-length bounds it like the other backends.
        let mut args: Vec<String> = vec![
            "-m".into(),
            "sglang.launch_server".into(),
            "--model-path".into(),
            model.clone(),
            "--host".into(),
            self.cfg.host.clone(),
            "--port".into(),
            port.to_string(),
        ];
        if spec.context_length > 0 {
            args.push("--context-length".into());
            args.push(spec.context_length.to_string());
        }
        if let Some(frac) = spec.tuning.gpu_memory_fraction {
            args.push("--mem-fraction-static".into());
            args.push(format!("{frac}"));
        }
        if let Some(tp) = spec.tuning.tensor_parallel {
            args.push("--tp-size".into());
            args.push(tp.to_string());
        }
        for a in &spec.tuning.extra_args {
            if !a.is_empty() {
                args.push(a.clone());
            }
        }
        let mut child = tokio::process::Command::new(py)
            .args(&args)
            .kill_on_drop(true)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                    .with_correlation(cid)
            })?;
        spawn_io_drain("sglang".into(), &mut child);

        let base_url = format!("http://{}:{}/v1", self.cfg.host, port);
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(180);
        let mut last_progress = tokio::time::Instant::now();
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
            if last_progress.elapsed() > tokio::time::Duration::from_secs(10) {
                last_progress = tokio::time::Instant::now();
                events.publish(AppEvent::DeployProgress {
                    job_id: spec.job_id.clone(),
                    percent: 60,
                    message: "Waiting for SGLang to become healthy…".into(),
                });
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }

        let mut runtime = ActiveRuntime::new(
            format!("sglang:{}", spec.model_id),
            BackendKind::Sglang.to_runtime_kind(),
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
