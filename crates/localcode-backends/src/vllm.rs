use crate::{
    capture_into_monitor, port_in_use, probe_client, resolve_launch, spawn_exit_watch, BackendKind,
    DeployTuning, DetectReport, Health, InferenceBackend, ModelDeploySpec, ModelMonitors, ProcState,
    RunningEndpoint,
};
use async_trait::async_trait;
use localcode_core::config::VllmConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus};
use localcode_core::runtime::{ActiveRuntime, RuntimeStatus};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;
use uuid::Uuid;

/// A managed child held so both `stop()` and the exit watcher can reach it.
type ChildHandle = Arc<Mutex<Option<tokio::process::Child>>>;

pub struct VllmBackend {
    cfg: VllmConfig,
    http: reqwest::Client,
    /// Track children so stop() works and processes are reaped (previously
    /// they were mem::forget-ed: unstoppable and zombied on exit).
    children: Arc<Mutex<Vec<(String, ChildHandle)>>>,
    /// Shared dashboard monitors (`/dash`). Detached by default.
    monitors: ModelMonitors,
}

impl VllmBackend {
    pub fn new(cfg: VllmConfig) -> Self {
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

        let built = serve_args(&model, &self.cfg.host, port, spec.context_length, &spec.tuning);
        // Honor a full command override (from the deploy panel or assistant),
        // else spawn the command we built. `command` is what the /dash card shows.
        let (program, args, command) =
            resolve_launch(&self.cfg.bin, built, spec.command_override.as_deref());
        // Pre-generate the runtime id so its `/dash` monitor captures startup
        // logs before the (potentially very long) health loop.
        let runtime_id = Uuid::new_v4();
        let mut cmd = tokio::process::Command::new(&program);
        cmd.args(&args)
            .kill_on_drop(true)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        // Own process group so `stop()` can signal vLLM's EngineCore/worker
        // children (which hold the VRAM), not just the launcher.
        crate::proc::spawn_in_own_group(&mut cmd);
        let mut child = cmd.spawn().map_err(|e| {
            LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                .with_correlation(cid)
                .with_cause("Failed to spawn vllm")
                .with_hint("Check CUDA version compatibility")
                .retryable(true)
        })?;
        // Capture vLLM's own output into the dashboard monitor so a failure below
        // can explain itself instead of surfacing a bare exit code, and so the
        // `/dash` card shows the newest lines live.
        let monitor = self.monitors.register(
            runtime_id.to_string(),
            format!("vllm:{}", spec.model_id),
            BackendKind::Vllm,
            Some(spec.model_id.clone()),
            command,
            ProcState::Starting,
        );
        capture_into_monitor("vllm", &mut child, &monitor);
        let logs = monitor.logs_handle();
        let tail = |n: usize| -> String {
            logs.lock()
                .ok()
                .map(|b| {
                    let start = b.len().saturating_sub(n);
                    b.iter().skip(start).cloned().collect::<Vec<_>>().join("\n")
                })
                .unwrap_or_default()
        };

        let base_url = format!("http://{}:{}/v1", self.cfg.host, port);
        // vLLM downloads the model *during* `serve` (unlike llama.cpp, which we
        // pre-download), then loads weights and compiles CUDA graphs — a cold
        // start on a large model routinely exceeds a few minutes. A hard crash
        // still fails fast via `try_wait` below, so a generous ceiling only
        // affects a server that is genuinely still making progress.
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(1800);
        let mut last_progress = tokio::time::Instant::now();
        loop {
            if tokio::time::Instant::now() > deadline {
                crate::proc::kill_tree(&mut child).await;
                self.monitors.remove(&runtime_id.to_string());
                let mut err = LocalCodeError::new(
                    ErrorCode::BackendHealthTimeout,
                    "vLLM did not become healthy in time",
                )
                .with_correlation(cid)
                .with_hint("First-run model downloads can be large — re-running resumes them")
                .with_hint("If the GPU is out of memory, lower the context with /context")
                .retryable(true);
                let t = tail(6);
                if !t.trim().is_empty() {
                    err = err.with_cause(format!("last vLLM output:\n{t}"));
                }
                return Err(err);
            }
            if let Ok(h) = self.health(&base_url).await {
                if h.healthy {
                    break;
                }
            }
            if let Ok(Some(st)) = child.try_wait() {
                self.monitors.remove(&runtime_id.to_string());
                let t = tail(12);
                let mut err = LocalCodeError::new(
                    ErrorCode::BackendStartFailed,
                    format!("vLLM exited: {st}"),
                )
                .with_correlation(cid)
                .with_cause("vLLM stopped before serving the model");
                // Hints depend on *why* it died — the output tells us. Getting
                // this wrong is how a torch/vLLM install bug gets misread as a
                // model-format problem.
                for h in failure_hints(&t) {
                    err = err.with_hint(h);
                }
                if !t.trim().is_empty() {
                    err = err.with_cause(format!("vLLM output:\n{t}"));
                }
                return Err(err);
            }
            if last_progress.elapsed() > tokio::time::Duration::from_secs(10) {
                last_progress = tokio::time::Instant::now();
                // Show vLLM's own latest line so a long download/load reads as
                // progress rather than a hang.
                let latest = tail(1);
                let message = if latest.trim().is_empty() {
                    "Waiting for vLLM to become healthy…".to_string()
                } else {
                    format!("vLLM: {}", latest.trim())
                };
                events.publish(AppEvent::DeployProgress {
                    job_id: spec.job_id.clone(),
                    percent: 60,
                    message,
                });
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }

        let mut runtime = ActiveRuntime::new(
            format!("vllm:{}", spec.model_id),
            BackendKind::Vllm.to_runtime_kind(),
            base_url,
        );
        runtime.id = runtime_id;
        runtime.model_id = Some(spec.model_id);
        runtime.quantization = spec.quantization;
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
                // Kill the whole process group — vLLM's worker/EngineCore
                // children hold the VRAM, so killing only the launcher leaks it.
                crate::proc::kill_tree(&mut child).await;
            }
        }
        self.monitors.remove(runtime_id);
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

/// Build the `vllm serve …` argument list.
///
/// `--max-model-len` is the load-bearing part: without it vLLM sizes the KV
/// cache for the model's full native context (often 128k) and OOMs at startup
/// on a single consumer GPU. We bound it to the context the user picked — the
/// same value llama.cpp receives via `-c`. A zero context (unset) is skipped so
/// vLLM keeps its own default. The optional tuning flags (VRAM fraction,
/// tensor-parallel) are only emitted when the user set them.
fn serve_args(
    model: &str,
    host: &str,
    port: u16,
    context_length: u32,
    tuning: &DeployTuning,
) -> Vec<String> {
    let mut args = vec![
        "serve".into(),
        model.to_string(),
        "--host".into(),
        host.to_string(),
        "--port".into(),
        port.to_string(),
    ];
    if context_length > 0 {
        args.push("--max-model-len".into());
        args.push(context_length.to_string());
    }
    if let Some(frac) = tuning.gpu_memory_fraction {
        args.push("--gpu-memory-utilization".into());
        args.push(format!("{frac}"));
    }
    if let Some(tp) = tuning.tensor_parallel {
        args.push("--tensor-parallel-size".into());
        args.push(tp.to_string());
    }
    for a in &tuning.extra_args {
        if !a.is_empty() {
            args.push(a.clone());
        }
    }
    args
}

/// The `(program, args)` a vLLM deploy would spawn — used to seed the editable
/// deploy-command field so "no edit" reproduces the built command exactly.
pub(crate) fn plan_command(
    cfg: &VllmConfig,
    model: &str,
    port: u16,
    context_length: u32,
    tuning: &DeployTuning,
) -> (String, Vec<String>) {
    (
        cfg.bin.clone(),
        serve_args(model, &cfg.host, port, context_length, tuning),
    )
}

/// Map vLLM's dying output to targeted hints via the shared diagnosis engine.
///
/// The duplicate custom-op registration — `vllm::_fused_mul_mat_gguf`
/// "registered ... multiple times" — is a broken or version-mismatched install,
/// **not** a model-format problem: it fires while vLLM imports its quantization
/// ops at startup, which happens for any model (safetensors included). Reporting
/// it as "GGUF isn't supported" sends the user chasing the wrong thing. The
/// classification now lives in `diagnose::classify` so it is shared with the
/// smoke test and the Fix flow; only the vLLM-specific fallback stays here.
fn failure_hints(output: &str) -> Vec<&'static str> {
    let hints = crate::diagnose::classify(output).hints();
    if hints.is_empty() {
        // Nothing recognized — fall back to the format / VRAM guesses.
        vec![
            "GGUF-only repos aren't served by vLLM — pick a full (safetensors) model",
            "A too-large context or low VRAM can OOM at startup — try /context",
        ]
    } else {
        hints
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_crash_is_not_blamed_on_format() {
        let out = "RuntimeError: Tried to register an operator (vllm::_fused_mul_mat_gguf(...)) \
                   with the same name and overload name multiple times.";
        let hints = failure_hints(out);
        assert!(hints.iter().any(|h| h.contains("version-mismatched")));
        assert!(
            !hints.iter().any(|h| h.contains("GGUF-only")),
            "must not misreport an install bug as a format problem"
        );
    }

    #[test]
    fn generic_exit_keeps_format_and_vram_hints() {
        let hints = failure_hints("ValueError: model architecture not supported");
        assert!(hints.iter().any(|h| h.contains("GGUF-only")));
        assert!(hints.iter().any(|h| h.contains("/context")));
    }

    #[test]
    fn serve_args_bound_kv_cache_to_context() {
        let args = serve_args("org/model", "127.0.0.1", 8000, 8192, &DeployTuning::default());
        assert_eq!(&args[..2], &["serve", "org/model"]);
        // The KV-cache bound that keeps vLLM from OOMing on a 128k default.
        let i = args.iter().position(|a| a == "--max-model-len").expect("flag present");
        assert_eq!(args[i + 1], "8192");
    }

    #[test]
    fn serve_args_omit_context_when_zero() {
        let args = serve_args("org/model", "127.0.0.1", 8000, 0, &DeployTuning::default());
        assert!(!args.iter().any(|a| a == "--max-model-len"));
    }

    #[test]
    fn serve_args_emit_tuning_flags_when_set() {
        let tuning = DeployTuning {
            gpu_memory_fraction: Some(0.85),
            tensor_parallel: Some(2),
            gpu_layers: None,
            extra_args: vec!["--enforce-eager".into()],
        };
        let args = serve_args("org/model", "127.0.0.1", 8000, 8192, &tuning);
        let i = args
            .iter()
            .position(|a| a == "--gpu-memory-utilization")
            .expect("frac flag present");
        assert_eq!(args[i + 1], "0.85");
        let j = args
            .iter()
            .position(|a| a == "--tensor-parallel-size")
            .expect("tp flag present");
        assert_eq!(args[j + 1], "2");
        assert!(args.iter().any(|a| a == "--enforce-eager"));
    }

    #[test]
    fn serve_args_omit_tuning_flags_when_unset() {
        let args = serve_args("org/model", "127.0.0.1", 8000, 8192, &DeployTuning::default());
        assert!(!args.iter().any(|a| a == "--gpu-memory-utilization"));
        assert!(!args.iter().any(|a| a == "--tensor-parallel-size"));
    }
}
