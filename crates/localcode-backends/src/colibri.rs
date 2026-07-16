//! Colibrì expert-streaming engines.
//!
//! One adapter serves two [`BackendKind`]s:
//!
//! * [`BackendKind::Colibri`] — upstream colibrì (github.com/JustVugg/colibri),
//!   the GLM-5.2 engine. Model path can go on the command line (`--model`).
//! * [`BackendKind::ColibriHy3`] — the Hy3 fork
//!   (github.com/ErikTromp/colibri-hy3). Serves Tencent Hy3 (`model_type:
//!   hy_v3`) and still loads GLM-5.2 containers. Its README passes the model
//!   dir only via the `COLI_MODEL` env var, so that is the portable channel.
//!
//! Both expose an OpenAI-compatible API (`/v1/models`, `/v1/chat/completions`)
//! from `coli serve`. Unlike vLLM/SGLang, colibrì cannot pull weights itself:
//! `COLI_MODEL` must point at a **local directory** holding a pre-converted
//! int4 container (`out-*.safetensors` shards + config/tokenizer), e.g. a
//! downloaded `…-colibri-int4` Hugging Face repo.

use crate::{
    capture_into_monitor, colibri_build_has_cuda, port_in_use, probe_client, resolve_colibri_bin,
    resolve_launch, spawn_exit_watch, BackendKind, DeployTuning, DetectReport, Health,
    InferenceBackend, ModelDeploySpec, ModelMonitors, ProcState, RunningEndpoint,
};
use async_trait::async_trait;
use localcode_gpu::GpuInventory;
use localcode_core::config::ColibriConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus};
use localcode_core::paths::AppPaths;
use localcode_core::runtime::{ActiveRuntime, RuntimeStatus};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;
use uuid::Uuid;

/// A managed child held so both `stop()` and the exit watcher can reach it.
type ChildHandle = Arc<Mutex<Option<tokio::process::Child>>>;

/// Dense weights load from disk before the port opens; on a slow disk that is
/// minutes, not the seconds llama-server needs — but bounded, unlike vLLM's
/// CUDA-graph capture.
const HEALTH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(300);

pub struct ColibriBackend {
    /// Which engine this instance is (`Colibri` or `ColibriHy3`).
    kind: BackendKind,
    cfg: ColibriConfig,
    http: reqwest::Client,
    children: Arc<Mutex<Vec<(String, ChildHandle)>>>,
    /// Shared dashboard monitors (`/dash`). Detached by default.
    monitors: ModelMonitors,
}

impl ColibriBackend {
    /// `kind` must be `Colibri` or `ColibriHy3`.
    pub fn new(kind: BackendKind, cfg: ColibriConfig) -> Self {
        debug_assert!(matches!(kind, BackendKind::Colibri | BackendKind::ColibriHy3));
        Self {
            kind,
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

    fn resolve_bin(&self) -> Option<PathBuf> {
        let paths = AppPaths::resolve().ok()?;
        resolve_colibri_bin(&self.cfg.bin, &paths, self.kind)
    }

    fn repo_url(&self) -> &'static str {
        match self.kind {
            BackendKind::ColibriHy3 => "https://github.com/ErikTromp/colibri-hy3",
            _ => "https://github.com/JustVugg/colibri",
        }
    }

    /// The model id `coli serve` registers on `/v1/models` when we can't read
    /// it back (documented defaults).
    fn default_served_id(&self) -> &'static str {
        match self.kind {
            BackendKind::ColibriHy3 => "hy3-colibri",
            _ => "glm-5.2-colibri",
        }
    }

    /// Read the actual served model id from `/v1/models`, falling back to the
    /// documented default. Engine versions differ on `--model-id` support, so
    /// asking the server is the only reliable way to get the id chat requests
    /// must carry.
    async fn served_model_id(&self, base_url: &str, api_key: &str) -> String {
        let url = format!("{}/models", base_url.trim_end_matches('/'));
        let resp = self.http.get(&url).bearer_auth(api_key).send().await;
        if let Ok(r) = resp {
            if let Ok(v) = r.json::<serde_json::Value>().await {
                if let Some(id) = v
                    .get("data")
                    .and_then(|d| d.as_array())
                    .and_then(|a| a.first())
                    .and_then(|m| m.get("id"))
                    .and_then(|s| s.as_str())
                {
                    return id.to_string();
                }
            }
        }
        self.default_served_id().to_string()
    }

    /// Resolve the local container directory `COLI_MODEL` will point at.
    /// `local_path` may be a file (the deploy pipeline hands back the first
    /// downloaded shard) or the directory itself.
    fn model_dir(&self, spec: &ModelDeploySpec) -> Result<PathBuf, LocalCodeError> {
        let raw = spec.local_path.clone().ok_or_else(|| {
            LocalCodeError::new(
                ErrorCode::DeployDownloadFailed,
                format!("{} needs the model on local disk", self.kind.as_str()),
            )
            .with_cause("colibrì streams experts from a local int4 container; it cannot pull from Hugging Face itself")
            .with_hint(format!(
                "Download a pre-converted container, e.g.: hf download {} --local-dir <dir>",
                spec.model_id
            ))
        })?;
        let p = PathBuf::from(&raw);
        let dir = if p.is_file() {
            p.parent().map(Path::to_path_buf).unwrap_or(p)
        } else {
            p
        };
        if !dir.is_dir() {
            return Err(LocalCodeError::new(
                ErrorCode::DeployDownloadFailed,
                format!("Model directory not found: {}", dir.display()),
            )
            .retryable(false));
        }
        // Fail fast on a shard-only download: the engine needs the whole repo
        // (config.json identifies the architecture, tokenizer files, .qs scales).
        if !dir.join("config.json").is_file() {
            return Err(LocalCodeError::new(
                ErrorCode::DeployUnsupportedFormat,
                format!("{} has no config.json — not a complete colibrì container", dir.display()),
            )
            .with_cause("A colibrì int4 container is the full repo: out-*.safetensors shards plus config/tokenizer files")
            .with_hint(format!(
                "Fetch the whole repo: hf download {} --local-dir {}",
                spec.model_id,
                dir.display()
            )));
        }
        Ok(dir)
    }
}

#[async_trait]
impl InferenceBackend for ColibriBackend {
    fn name(&self) -> BackendKind {
        self.kind
    }

    async fn detect(&self) -> DetectReport {
        let binary = self.resolve_bin().map(|p| p.display().to_string());
        let mut notes = vec![];
        match self.kind {
            BackendKind::ColibriHy3 => {
                notes.push("Serves Tencent Hy3 (and GLM-5.2) colibrì int4 containers".into())
            }
            _ => notes.push("Serves GLM-5.2 colibrì int4 containers".into()),
        }
        if binary.is_none() {
            notes.push(format!("`{}` not found on PATH or managed install", self.cfg.bin));
            notes.push("Install from the Backends panel (builds from source)".into());
            notes.push(self.repo_url().into());
        }
        let installed = binary.is_some();
        DetectReport {
            kind: self.kind,
            installed,
            version: None,
            base_url: Some(format!("http://{}:{}", self.cfg.host, self.cfg.port)),
            binary_path: binary,
            notes,
            ready: installed,
        }
    }

    async fn ensure_ready(&self) -> Result<(), LocalCodeError> {
        if self.detect().await.ready {
            return Ok(());
        }
        Err(LocalCodeError::new(
            ErrorCode::BackendBinaryMissing,
            format!("{} binary `{}` not found", self.kind.as_str(), self.cfg.bin),
        )
        .with_source(self.kind.as_str(), "ensure_ready")
        .with_cause(format!(
            "Binary not on PATH and no managed install under data/backends/{}",
            self.kind.as_str()
        ))
        .with_hint("Install from the Backends panel (git clone + setup.sh)")
        .with_hint(self.repo_url())
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
            .with_hint("Stop the other process or pick a different port")
            .retryable(true));
        }

        let model_dir = self.model_dir(&spec)?;
        let bin = self.resolve_bin().ok_or_else(|| {
            LocalCodeError::new(ErrorCode::BackendBinaryMissing, "coli binary not found")
                .with_correlation(cid)
                .with_hint("Install from the Backends panel")
        })?;

        // `coli serve` honors COLI_API_KEY; generate one per deploy and carry it
        // on the runtime so agent clients authenticate. On engine builds without
        // auth the extra header is simply ignored.
        let api_key = format!("coli-{}", Uuid::new_v4().simple());

        info!(%cid, model_dir = %model_dir.display(), port, engine = self.kind.as_str(), "starting coli serve");
        events.publish(AppEvent::DeployProgress {
            job_id: spec.job_id.clone(),
            percent: 50,
            message: format!("Starting {} (coli serve)", self.kind.as_str()),
        });

        let built = build_args(self.kind, &self.cfg.host, port, &model_dir, &spec.tuning);
        let (program, args, mut command) = resolve_launch(
            &bin.display().to_string(),
            built,
            spec.command_override.as_deref(),
        );

        // Use a GPU when one is present AND this build can actually use CUDA.
        // Forcing COLI_CUDA on a CPU-only engine makes `coli serve` exit at
        // startup — the native-Windows build has no CUDA target, and a Unix
        // CPU-only build would crash the same way — so gate on the install's
        // CUDA marker. A bare-PATH binary (no marker) stays CPU-safe; the user
        // can still pin `--gpu` in extra_args to opt in. discover() degrades to
        // an empty inventory (CPU only) when there's no nvidia-smi.
        let user_pinned_gpu = args.iter().any(|a| a == "--gpu");
        let cuda_capable = AppPaths::resolve()
            .ok()
            .is_some_and(|p| colibri_build_has_cuda(&p, self.kind));
        let inventory = localcode_gpu::discover().unwrap_or_else(|_| GpuInventory {
            devices: vec![],
            detection_method: "none".into(),
            warnings: vec![],
        });
        let gpu_env = if cuda_capable {
            gpu_env_for(&inventory, user_pinned_gpu)
        } else {
            Vec::new()
        };
        if !gpu_env.is_empty() {
            info!(%cid, engine = self.kind.as_str(), "colibrì: GPU detected — enabling CUDA ({gpu_env:?})");
        } else if !inventory.devices.is_empty() && !cuda_capable && !user_pinned_gpu {
            info!(
                %cid, engine = self.kind.as_str(),
                "colibrì: GPU present but this build is CPU-only — reinstall on Linux/WSL with nvcc for a CUDA build"
            );
        }

        // The dash card's command is click-to-copy; surface the env the launch
        // actually uses so the copied line reproduces it. COLI_MODEL rides here
        // when it can't go on the command line (Hy3 fork, or a user override);
        // COLI_API_KEY is a secret and deliberately omitted.
        let mut env_prefix = String::new();
        for (k, v) in &gpu_env {
            env_prefix.push_str(&format!("{k}={v} "));
        }
        if !args.iter().any(|a| a == "--model") {
            env_prefix.push_str(&format!("COLI_MODEL={} ", model_dir.display()));
        }
        if !env_prefix.is_empty() {
            command = format!("{env_prefix}{command}");
        }

        let runtime_id = Uuid::new_v4();
        let mut cmd = tokio::process::Command::new(&program);
        cmd.args(&args)
            .env("COLI_MODEL", &model_dir)
            .env("COLI_API_KEY", &api_key)
            .kill_on_drop(true)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        for (k, v) in &gpu_env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().map_err(|e| {
            LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                .with_correlation(cid)
                .with_cause("Failed to spawn coli serve")
                .retryable(true)
        })?;
        let monitor = self.monitors.register(
            runtime_id.to_string(),
            format!("{}:{}", self.kind.as_str(), spec.model_id),
            self.kind,
            Some(spec.model_id.clone()),
            command,
            ProcState::Starting,
        );
        capture_into_monitor(self.kind.as_str(), &mut child, &monitor);

        let base_url = format!("http://{}:{}", self.cfg.host, port);
        let deadline = tokio::time::Instant::now() + HEALTH_DEADLINE;
        let mut last_progress = tokio::time::Instant::now();
        loop {
            if tokio::time::Instant::now() > deadline {
                let _ = child.kill().await;
                self.monitors.remove(&runtime_id.to_string());
                return Err(LocalCodeError::new(
                    ErrorCode::BackendHealthTimeout,
                    "coli serve did not become healthy in time",
                )
                .with_correlation(cid)
                .with_cause("Loading the dense weights from disk can take minutes; a hang past that usually means a bad container")
                .with_hint("Check the /dash logs; verify the container downloaded completely")
                .retryable(true));
            }
            if let Ok(h) = self.health(&base_url).await {
                if h.healthy {
                    break;
                }
            }
            if let Ok(Some(status)) = child.try_wait() {
                // Attach coli's own output so diagnosis names the REAL cause
                // (missing libgomp, SIGILL/ISA, CUDA OOM, a rejected flag…)
                // instead of the old blanket "needs AVX2" guess.
                let tail = colibri_log_tail(&monitor.logs_handle(), 12);
                self.monitors.remove(&runtime_id.to_string());
                let mut err = LocalCodeError::new(
                    ErrorCode::BackendStartFailed,
                    format!("coli serve exited early: {status}"),
                )
                .with_correlation(cid);
                if tail.trim().is_empty() {
                    // No output captured — fall back to a hint that blames AVX2
                    // only when this CPU actually lacks it (coli runs here too).
                    err = err.with_cause(colibri_early_exit_hint());
                } else {
                    err = err.with_cause(tail.clone());
                    if looks_like_arg_error(&tail) {
                        err = err.with_hint(
                            "This coli build may reject the --auto-tier flag LocalCode adds; \
                             pin an explicit tier (extra_args: --ram <GB> [--vram <GB>]) to skip it",
                        );
                    }
                }
                return Err(err.retryable(true));
            }
            if last_progress.elapsed() > tokio::time::Duration::from_secs(10) {
                last_progress = tokio::time::Instant::now();
                events.publish(AppEvent::DeployProgress {
                    job_id: spec.job_id.clone(),
                    percent: 60,
                    message: "Loading dense weights (colibrì streams experts on demand)…".into(),
                });
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }

        let v1 = format!("{base_url}/v1");
        let served_id = self.served_model_id(&v1, &api_key).await;

        let mut runtime = ActiveRuntime::new(
            format!("{}:{}", self.kind.as_str(), spec.model_id),
            self.kind.to_runtime_kind(),
            v1,
        );
        runtime.id = runtime_id;
        // Chat requests must carry the engine's registered id (e.g.
        // "glm-5.2-colibri"), not the HF repo id.
        runtime.model_id = Some(served_id);
        runtime.quantization = spec.quantization;
        runtime.api_key = Some(api_key);
        // We pass no context flag (coli sizes its own KV slots), so don't claim
        // a window we didn't set — the agent falls back to its char budget.
        runtime.context_tokens = None;
        runtime.status = RuntimeStatus::Healthy;
        runtime.correlation_id = cid.to_string();

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
                // Single C process — no worker tree to signal.
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
            // 401/403 = the server is up and enforcing COLI_API_KEY; that IS
            // healthy for a probe that doesn't carry the key.
            Ok(r) if r.status().is_success()
                || r.status() == reqwest::StatusCode::UNAUTHORIZED
                || r.status() == reqwest::StatusCode::FORBIDDEN =>
            {
                Ok(Health {
                    healthy: true,
                    message: "ok".into(),
                })
            }
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

/// Build the `coli serve …` argument list (without the binary). The model dir
/// goes on the command line only for the upstream engine (documented `--model`
/// flag); the Hy3 fork takes it via `COLI_MODEL`, which `deploy` always sets.
fn build_args(
    kind: BackendKind,
    host: &str,
    port: u16,
    model_dir: &Path,
    tuning: &DeployTuning,
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "serve".into(),
        "--host".into(),
        host.to_string(),
        "--port".into(),
        port.to_string(),
    ];
    if kind == BackendKind::Colibri {
        args.push("--model".into());
        args.push(model_dir.display().to_string());
    }
    // colibrì's tiering knobs (--ram/--vram/--gpu) ride through extra_args —
    // the generic tuning fields (gpu fraction / tensor parallel / gpu layers)
    // have no coli equivalent and are deliberately not mapped.
    let mut has_manual_tier = false;
    for a in &tuning.extra_args {
        if a.is_empty() {
            continue;
        }
        // An explicit budget (or a hand-supplied --auto-tier) opts out of the
        // default planner below.
        if matches!(a.as_str(), "--auto-tier" | "--ram" | "--vram") {
            has_manual_tier = true;
        }
        args.push(a.clone());
    }
    // Default to colibrì's automatic tier planner: it sizes the RAM / VRAM /
    // disk hierarchy to this machine, and (with COLI_CUDA/COLI_GPU set by
    // `deploy` when a GPU is present) folds the GPU tier into that plan. Skip it
    // only when the caller pinned an explicit budget.
    if !has_manual_tier {
        args.push("--auto-tier".into());
    }
    args
}

/// The environment that switches colibrì onto a GPU: enable the CUDA backend
/// and select the device with the most VRAM. Empty when the caller already
/// pinned `--gpu` (manual control) or no GPU is present — `--auto-tier` then
/// plans a CPU/RAM hierarchy. The VRAM budget is left to `--auto-tier`, so no
/// `CUDA_EXPERT_GB` is forced here.
///
/// `COLI_CUDA`/`COLI_GPU` are the portable channel across both engines:
/// upstream colibrì reads them directly, and the Hy3 fork treats `COLI_GPU` as
/// its `--gpu` flag — so this turns the GPU on without a fork-specific serve
/// flag that the other engine might reject.
fn gpu_env_for(inv: &GpuInventory, user_pinned_gpu: bool) -> Vec<(&'static str, String)> {
    if user_pinned_gpu {
        return vec![];
    }
    match inv.devices.iter().max_by_key(|d| d.total_vram_bytes) {
        Some(dev) => vec![
            ("COLI_CUDA", "1".to_string()),
            ("COLI_GPU", dev.index.to_string()),
        ],
        None => vec![],
    }
}

/// The newest `n` lines the monitor captured from `coli`'s stdout/stderr, joined
/// for an error cause. Empty when nothing was captured (an immediate exec
/// failure). Feeds the diagnosis engine so a real failure isn't mislabeled.
fn colibri_log_tail(
    logs: &std::sync::Mutex<std::collections::VecDeque<String>>,
    n: usize,
) -> String {
    logs.lock()
        .map(|b| {
            let start = b.len().saturating_sub(n);
            b.iter().skip(start).map(String::as_str).collect::<Vec<_>>().join("\n")
        })
        .unwrap_or_default()
}

/// True when `coli`'s output looks like a CLI argument/usage error — the shape
/// of a build rejecting a flag LocalCode added (e.g. `--auto-tier`).
fn looks_like_arg_error(s: &str) -> bool {
    let l = s.to_lowercase();
    [
        "unknown option",
        "unrecognized",
        "invalid option",
        "unexpected argument",
        "usage:",
        "--auto-tier",
    ]
    .iter()
    .any(|m| l.contains(m))
}

/// colibrì's AVX2 requirement is x86-only, and `coli` runs on this same CPU — so
/// we can check the host directly instead of guessing. Always false off x86
/// (Apple Silicon uses Metal/NEON, not AVX2).
fn host_missing_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        !std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Fallback cause when `coli` exited with no captured output: name AVX2 only
/// when this CPU truly lacks it, so we stop crying wolf on machines that have it.
fn colibri_early_exit_hint() -> String {
    if host_missing_avx2() {
        "This x86 CPU lacks AVX2, which colibrì requires; also verify the container downloaded \
         completely and RAM is sufficient"
            .to_string()
    } else {
        "Likely an incomplete container or not enough RAM — this CPU has AVX2, so that isn't it"
            .to_string()
    }
}

/// The `(program, args)` a colibrì deploy would spawn — used to seed the
/// editable deploy-command field. `model` is the local container dir when
/// known, else the HF model id as a placeholder.
pub(crate) fn plan_command(
    cfg: &ColibriConfig,
    kind: BackendKind,
    model: &str,
    port: u16,
    tuning: &DeployTuning,
) -> (String, Vec<String>) {
    let bin = AppPaths::resolve()
        .ok()
        .and_then(|p| resolve_colibri_bin(&cfg.bin, &p, kind))
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| cfg.bin.clone());
    (bin, build_args(kind, &cfg.host, port, Path::new(model), tuning))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_args_carry_model_flag() {
        let args = build_args(
            BackendKind::Colibri,
            "127.0.0.1",
            8091,
            Path::new("/nvme/glm52_i4"),
            &DeployTuning::default(),
        );
        assert_eq!(args[0], "serve");
        let model_pos = args.iter().position(|a| a == "--model").expect("--model");
        assert_eq!(args[model_pos + 1], "/nvme/glm52_i4");
    }

    #[test]
    fn hy3_args_omit_model_flag_env_carries_it() {
        // The fork documents COLI_MODEL only; an unknown --model flag could be
        // rejected by its arg parser, so it must not appear.
        let args = build_args(
            BackendKind::ColibriHy3,
            "0.0.0.0",
            8092,
            Path::new("/data/hy3_i4"),
            &DeployTuning::default(),
        );
        assert!(!args.iter().any(|a| a == "--model"));
        assert!(args.iter().any(|a| a == "8092"));
    }

    #[test]
    fn extra_args_pass_through_for_tiering_knobs() {
        let tuning = DeployTuning {
            extra_args: vec!["--ram".into(), "12".into(), "--gpu".into(), "0".into()],
            ..Default::default()
        };
        let args = build_args(
            BackendKind::ColibriHy3,
            "0.0.0.0",
            8092,
            Path::new("/data/hy3_i4"),
            &tuning,
        );
        let ram_pos = args.iter().position(|a| a == "--ram").expect("--ram");
        assert_eq!(args[ram_pos + 1], "12");
        // An explicit --ram budget opts out of the default auto-tier planner.
        assert!(!args.iter().any(|a| a == "--auto-tier"));
    }

    #[test]
    fn default_deploy_enables_auto_tier() {
        for kind in [BackendKind::Colibri, BackendKind::ColibriHy3] {
            let args = build_args(kind, "127.0.0.1", 8091, Path::new("/m"), &DeployTuning::default());
            assert!(
                args.iter().any(|a| a == "--auto-tier"),
                "{kind:?} should default to --auto-tier"
            );
        }
    }

    #[test]
    fn explicit_vram_or_auto_tier_is_not_doubled() {
        // A hand-supplied --auto-tier isn't appended a second time.
        let tuning = DeployTuning {
            extra_args: vec!["--auto-tier".into()],
            ..Default::default()
        };
        let args = build_args(BackendKind::Colibri, "127.0.0.1", 8091, Path::new("/m"), &tuning);
        assert_eq!(args.iter().filter(|a| *a == "--auto-tier").count(), 1);

        // An explicit --vram budget opts out of the planner entirely.
        let tuning = DeployTuning {
            extra_args: vec!["--vram".into(), "14".into()],
            ..Default::default()
        };
        let args = build_args(BackendKind::ColibriHy3, "0.0.0.0", 8092, Path::new("/m"), &tuning);
        assert!(!args.iter().any(|a| a == "--auto-tier"));
    }

    fn gpu(index: u32, total_gib: u64) -> localcode_gpu::GpuDevice {
        localcode_gpu::GpuDevice {
            index,
            name: format!("dev{index}"),
            total_vram_bytes: total_gib << 30,
            free_vram_bytes: total_gib << 30,
            driver_version: None,
            backend_affinity: vec![],
            temperature_c: None,
            utilization_pct: None,
            power_draw_w: None,
        }
    }

    fn inventory(devices: Vec<localcode_gpu::GpuDevice>) -> GpuInventory {
        GpuInventory { devices, detection_method: "test".into(), warnings: vec![] }
    }

    #[test]
    fn gpu_env_enables_cuda_on_largest_device() {
        // Two GPUs: pick the one with the most VRAM (index 1, 24 GiB).
        let env = gpu_env_for(&inventory(vec![gpu(0, 8), gpu(1, 24)]), false);
        assert!(env.iter().any(|(k, v)| *k == "COLI_CUDA" && v == "1"));
        assert!(
            env.iter().any(|(k, v)| *k == "COLI_GPU" && v == "1"),
            "should target the 24 GiB device (index 1)"
        );
    }

    #[test]
    fn gpu_env_empty_without_gpu() {
        assert!(gpu_env_for(&inventory(vec![]), false).is_empty());
    }

    #[test]
    fn gpu_env_empty_when_user_pinned_gpu() {
        // A user-supplied --gpu means manual control; don't inject the env.
        assert!(gpu_env_for(&inventory(vec![gpu(0, 24)]), true).is_empty());
    }

    #[test]
    fn log_tail_joins_the_newest_lines() {
        let logs = std::sync::Mutex::new(std::collections::VecDeque::from(vec![
            "line1".to_string(),
            "line2".to_string(),
            "line3".to_string(),
        ]));
        assert_eq!(colibri_log_tail(&logs, 2), "line2\nline3");
        assert_eq!(colibri_log_tail(&logs, 10), "line1\nline2\nline3");
        let empty: std::sync::Mutex<std::collections::VecDeque<String>> = std::sync::Mutex::new(std::collections::VecDeque::new());
        assert_eq!(colibri_log_tail(&empty, 5), "");
    }

    #[test]
    fn arg_error_detection() {
        assert!(looks_like_arg_error("coli: unknown option '--auto-tier'"));
        assert!(looks_like_arg_error("Usage: coli serve [OPTIONS]"));
        assert!(looks_like_arg_error("error: unexpected argument --auto-tier"));
        // A genuine runtime failure must NOT be read as an arg error (it would
        // suppress the real diagnosis).
        assert!(!looks_like_arg_error("error while loading shared libraries: libgomp.so.1"));
        assert!(!looks_like_arg_error("CUDA error: out of memory"));
    }

    #[test]
    fn early_exit_hint_is_avx2_honest() {
        // Never empty, and it only asserts an AVX2 deficiency when the host
        // truly lacks it (this test box, running coli-less, reflects the host).
        let hint = colibri_early_exit_hint();
        assert!(!hint.is_empty());
        assert_eq!(hint.contains("lacks AVX2"), host_missing_avx2());
    }
}
