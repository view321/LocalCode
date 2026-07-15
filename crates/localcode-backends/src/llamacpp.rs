use crate::{
    capture_into_monitor, port_in_use, probe_client, resolve_launch, resolve_llamacpp_bin,
    spawn_exit_watch, BackendKind, DeployTuning, DetectReport, Health, InferenceBackend,
    ModelDeploySpec, ModelMonitors, ProcState, RunningEndpoint,
};
use async_trait::async_trait;
use localcode_core::config::LlamaCppConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus};
use localcode_core::paths::AppPaths;
use localcode_core::runtime::{ActiveRuntime, RuntimeStatus};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;
use uuid::Uuid;

/// A managed child held so both `stop()` and the exit watcher can reach it. The
/// `Option` is emptied by whichever takes the child first.
type ChildHandle = Arc<Mutex<Option<tokio::process::Child>>>;

pub struct LlamaCppBackend {
    cfg: LlamaCppConfig,
    http: reqwest::Client,
    /// Track child processes by runtime id. kill_on_drop: managed runtimes
    /// stop when LocalCode exits (the quit dialog says so).
    children: Arc<Mutex<Vec<(String, ChildHandle)>>>,
    /// Shared dashboard monitors (`/dash`). Detached by default.
    monitors: ModelMonitors,
}

impl LlamaCppBackend {
    pub fn new(cfg: LlamaCppConfig) -> Self {
        Self {
            cfg,
            http: probe_client(),
            children: Arc::new(Mutex::new(Vec::new())),
            monitors: ModelMonitors::new(),
        }
    }

    /// Attach the shared `/dash` monitor store (called by the registry).
    pub fn with_monitors(mut self, monitors: ModelMonitors) -> Self {
        self.monitors = monitors;
        self
    }

    /// Configured path / PATH / managed install (same rules as the assistant).
    fn resolve_bin(&self) -> Option<std::path::PathBuf> {
        let paths = AppPaths::resolve().ok()?;
        resolve_llamacpp_bin(&self.cfg.bin, &paths)
    }
}

#[async_trait]
impl InferenceBackend for LlamaCppBackend {
    fn name(&self) -> BackendKind {
        BackendKind::LlamaCpp
    }

    async fn detect(&self) -> DetectReport {
        let binary = self.resolve_bin().map(|p| p.display().to_string());
        let mut notes = vec![];
        if binary.is_none() {
            notes.push(format!("`{}` not found on PATH or managed install", self.cfg.bin));
            notes.push("Run `localcode setup` or install from the Backends panel".into());
            notes.push("https://github.com/ggml-org/llama.cpp".into());
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
        .with_cause("Binary not on PATH and no managed install under data/backends/llamacpp")
        .with_hint("Run `localcode setup` or Install from the Backends panel")
        .retryable(true))
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

        let bin = self.resolve_bin().ok_or_else(|| {
            LocalCodeError::new(ErrorCode::BackendBinaryMissing, "llama-server not found")
                .with_correlation(cid)
                .with_hint("Run `localcode setup` to install llama-server automatically")
        })?;

        info!(%cid, %model_path, port, "starting llama-server");
        events.publish(AppEvent::DeployProgress {
            job_id: spec.job_id.clone(),
            percent: 50,
            message: "Starting llama-server".into(),
        });

        let built = build_args(&self.cfg.host, &model_path, port, spec.context_length, &spec.tuning);
        // Honor a full command override (deploy panel / assistant), else the
        // command we built. `command` is what the /dash card shows.
        let (program, args, command) = resolve_launch(
            &bin.display().to_string(),
            built,
            spec.command_override.as_deref(),
        );
        // Pre-generate the runtime id so its `/dash` monitor exists (and captures
        // startup logs) before the health loop.
        let runtime_id = Uuid::new_v4();
        let mut child = tokio::process::Command::new(&program)
            .args(&args)
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
        // Capture output into the dashboard monitor (and tracing); llama-server
        // logs every request and an undrained pipe would eventually block it.
        let monitor = self.monitors.register(
            runtime_id.to_string(),
            format!("llamacpp:{}", spec.model_id),
            BackendKind::LlamaCpp,
            Some(spec.model_id.clone()),
            command,
            ProcState::Starting,
        );
        capture_into_monitor("llama-server", &mut child, &monitor);

        let base_url = format!("http://{}:{}", self.cfg.host, port);
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(60);
        let mut last_progress = tokio::time::Instant::now();
        loop {
            if tokio::time::Instant::now() > deadline {
                let _ = child.kill().await;
                self.monitors.remove(&runtime_id.to_string());
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
                self.monitors.remove(&runtime_id.to_string());
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
        runtime.id = runtime_id;
        runtime.model_id = Some(spec.model_id.clone());
        runtime.quantization = spec.quantization.clone();
        // Report the served context window so the agent compacts before overflow.
        runtime.context_tokens = (spec.context_length > 0).then_some(spec.context_length);
        runtime.status = RuntimeStatus::Healthy;
        runtime.correlation_id = cid.to_string();

        // Healthy: flip the card to Running and hand the child to a shared handle
        // so both stop() and the exit watcher can reach it.
        monitor.set_state(ProcState::Running);
        let handle: ChildHandle = Arc::new(Mutex::new(Some(child)));
        spawn_exit_watch(handle.clone(), monitor);
        self.children
            .lock()
            .await
            .push((runtime.id.to_string(), handle));

        Ok(RunningEndpoint { runtime })
    }

    async fn stop(&self, runtime_id: &str) -> Result<(), LocalCodeError> {
        let mut kids = self.children.lock().await;
        if let Some(pos) = kids.iter().position(|(id, _)| id == runtime_id) {
            let (_, handle) = kids.remove(pos);
            // Bind the child out first so the MutexGuard is released before the
            // block ends (it must not outlive the owned `handle`).
            let child = handle.lock().await.take();
            if let Some(mut child) = child {
                let _ = child.kill().await;
            }
        }
        self.monitors.remove(runtime_id);
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

/// Build the `llama-server …` argument list (without the binary).
fn build_args(
    host: &str,
    model_path: &str,
    port: u16,
    context_length: u32,
    tuning: &DeployTuning,
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "-m".into(),
        model_path.to_string(),
        "--host".into(),
        host.to_string(),
        "--port".into(),
        port.to_string(),
        "-c".into(),
        context_length.to_string(),
    ];
    if let Some(ngl) = tuning.gpu_layers {
        args.push("--n-gpu-layers".into());
        args.push(ngl.to_string());
    }
    // Model-card / assistant recommended flags (already validated as tokens).
    for a in &tuning.extra_args {
        if !a.is_empty() {
            args.push(a.clone());
        }
    }
    args
}

/// The `(program, args)` a llama.cpp deploy would spawn — used to seed the
/// editable deploy-command field. `model_path` is the local GGUF path (or the
/// model id as a placeholder before download).
pub(crate) fn plan_command(
    cfg: &LlamaCppConfig,
    model_path: &str,
    port: u16,
    context_length: u32,
    tuning: &DeployTuning,
) -> (String, Vec<String>) {
    let bin = AppPaths::resolve()
        .ok()
        .and_then(|p| resolve_llamacpp_bin(&cfg.bin, &p))
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| cfg.bin.clone());
    (bin, build_args(&cfg.host, model_path, port, context_length, tuning))
}
