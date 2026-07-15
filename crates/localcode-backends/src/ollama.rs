use crate::{
    probe_client, spawn_io_drain, BackendKind, DetectReport, Health, InferenceBackend,
    ModelDeploySpec, ModelMonitors, ProcState, RunningEndpoint,
};
use async_trait::async_trait;
use futures::StreamExt;
use localcode_core::config::OllamaConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus};
use localcode_core::runtime::{ActiveRuntime, RuntimeStatus};
use tracing::{info, warn};

pub struct OllamaBackend {
    cfg: OllamaConfig,
    /// Short-timeout client for tags/health probes.
    http: reqwest::Client,
    /// Connect-timeout-only client for long-running pulls.
    pull_http: reqwest::Client,
    /// Shared dashboard monitors (`/dash`). Detached by default.
    monitors: ModelMonitors,
}

impl OllamaBackend {
    pub fn new(cfg: OllamaConfig) -> Self {
        let pull_http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();
        Self {
            cfg,
            http: probe_client(),
            pull_http,
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

    async fn deploy(
        &self,
        spec: ModelDeploySpec,
        events: &EventBus,
    ) -> Result<RunningEndpoint, LocalCodeError> {
        let cid = localcode_core::CorrelationId::new();
        info!(%cid, model = %spec.model_id, "ollama deploy start");

        let model_name = ollama_model_name(&spec);

        match self.pull_streaming(&model_name, &spec.job_id, events).await {
            Ok(()) => {
                info!(%cid, %model_name, "ollama pull ok");
            }
            Err(pull_err) => {
                // Fall back to create-from-GGUF when a local file is at hand.
                if let Some(path) = &spec.local_path {
                    warn!(%cid, error = %pull_err.message, "pull failed; trying local GGUF create");
                    return self.create_from_gguf(&model_name, path, &spec, cid).await;
                }
                // Or the CLI, which may have credentials/proxy config the API lacks.
                if which::which("ollama").is_ok() {
                    warn!(%cid, error = %pull_err.message, "API pull failed; trying ollama CLI");
                    self.pull_via_cli(&model_name, cid).await?;
                } else {
                    return Err(pull_err.with_correlation(cid));
                }
            }
        }

        // Ollama has no launch-time flags (it's a shared server we don't spawn),
        // so a requested context / GPU-offload is honored by baking it into a
        // derived model. `effective` is that derived model (or `model_name`
        // unchanged if creation was skipped or best-effort-failed).
        let effective = self.apply_tuning(&model_name, &spec, cid).await;

        let base = self.cfg.base_url.clone();
        // Display the model the user actually chose; serve the (possibly
        // tuning-derived) model the OpenAI-compatible endpoint should call.
        let mut runtime = ActiveRuntime::new(
            format!("ollama:{model_name}"),
            BackendKind::Ollama.to_runtime_kind(),
            format!("{base}/v1"),
        );
        runtime.model_id = Some(effective.clone());
        runtime.quantization = spec.quantization.clone();
        // Report the served context window so the agent compacts before overflow.
        runtime.context_tokens = (spec.context_length > 0).then_some(spec.context_length);
        runtime.status = RuntimeStatus::Healthy;
        runtime.correlation_id = cid.to_string();

        // Register a `/dash` card. Ollama runs in its own shared `ollama serve`
        // process we don't own, so the card is External (no exit code / captured
        // logs) and the command is the equivalent CLI invocation.
        let monitor = self.monitors.register(
            runtime.id.to_string(),
            format!("ollama:{model_name}"),
            BackendKind::Ollama,
            Some(effective.clone()),
            format!("ollama run {effective}"),
            ProcState::External,
        );
        monitor.push_log(format!("served by `ollama serve` at {base} (logs via ollama)"));

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
    /// Streaming `/api/pull`: parses the NDJSON progress feed and forwards it
    /// as DeployProgress events (mapped into the 40..=85 band), instead of
    /// blocking silently for the whole multi-GB download.
    async fn pull_streaming(
        &self,
        model_name: &str,
        job_id: &str,
        events: &EventBus,
    ) -> Result<(), LocalCodeError> {
        let pull_url = format!("{}/api/pull", self.cfg.base_url);
        let body = serde_json::json!({ "name": model_name, "stream": true });
        let resp = self
            .pull_http
            .post(&pull_url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                LocalCodeError::new(ErrorCode::BackendNotReady, format!("Cannot reach Ollama: {e}"))
                    .with_hint("Start ollama serve")
                    .retryable(true)
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(pull_error(model_name, &format!("({status}): {body}")));
        }

        let mut stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut last_emit = std::time::Instant::now();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| pull_error(model_name, &e.to_string()))?;
            buf.extend_from_slice(&chunk);
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let Ok(v) = serde_json::from_slice::<serde_json::Value>(&line) else {
                    continue;
                };
                if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                    return Err(pull_error(model_name, err));
                }
                let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("");
                let total = v.get("total").and_then(|t| t.as_u64()).unwrap_or(0);
                let completed = v.get("completed").and_then(|c| c.as_u64()).unwrap_or(0);
                if last_emit.elapsed() > std::time::Duration::from_millis(500) {
                    last_emit = std::time::Instant::now();
                    let percent = if total > 0 {
                        40 + ((completed as f64 / total as f64) * 45.0) as u8
                    } else {
                        45
                    };
                    events.publish(AppEvent::DeployProgress {
                        job_id: job_id.to_string(),
                        percent: percent.min(85),
                        message: format!("Ollama pull: {status}"),
                    });
                }
            }
        }
        Ok(())
    }

    async fn pull_via_cli(
        &self,
        model_name: &str,
        cid: localcode_core::CorrelationId,
    ) -> Result<(), LocalCodeError> {
        // Piped + drained: inherited stdio would draw over the TUI.
        let mut child = tokio::process::Command::new("ollama")
            .args(["pull", model_name])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                    .with_correlation(cid)
            })?;
        spawn_io_drain("ollama-pull".into(), &mut child);
        let status = child.wait().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::BackendStartFailed, e.to_string())
                .with_correlation(cid)
        })?;
        if !status.success() {
            return Err(pull_error(model_name, "CLI pull failed").with_correlation(cid));
        }
        Ok(())
    }

    /// Bake a customized context / GPU-offload into a derived Ollama model and
    /// return its name. Ollama exposes these only as model PARAMETERs (there is
    /// no serve flag), so `/api/create` with `FROM <base>` — which references the
    /// existing blobs, no re-download — is the way to honor them. Best-effort:
    /// on any failure we log and return the base model unchanged, so tuning can
    /// never convert a working deploy into a broken one.
    async fn apply_tuning(
        &self,
        base: &str,
        spec: &ModelDeploySpec,
        cid: localcode_core::CorrelationId,
    ) -> String {
        // Ollama's built-in default context is small (~2048) and the deploy
        // panel always shows a concrete context, so honor whatever it shows —
        // otherwise the served context would silently disagree with the UI (and
        // with what vLLM/llama.cpp/SGLang do with the same value). Only skip when
        // context is explicitly unset (0) and there is no GPU-layer override, in
        // which case there's nothing to bake in.
        if spec.context_length == 0 && spec.tuning.gpu_layers.is_none() {
            return base.to_string();
        }
        let mut modelfile = format!("FROM {base}\n");
        if spec.context_length > 0 {
            modelfile.push_str(&format!("PARAMETER num_ctx {}\n", spec.context_length));
        }
        if let Some(n) = spec.tuning.gpu_layers {
            modelfile.push_str(&format!("PARAMETER num_gpu {n}\n"));
        }
        let derived = derived_model_name(base, spec.context_length);
        let body =
            serde_json::json!({ "name": derived, "modelfile": modelfile, "stream": false });
        match self
            .pull_http
            .post(format!("{}/api/create", self.cfg.base_url))
            .json(&body)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => {
                info!(%cid, %derived, "ollama tuned model created (num_ctx/num_gpu)");
                derived
            }
            Ok(r) => {
                warn!(%cid, status = %r.status(), "ollama tuning create failed; using base model");
                base.to_string()
            }
            Err(e) => {
                warn!(%cid, error = %e, "ollama tuning create errored; using base model");
                base.to_string()
            }
        }
    }

    async fn create_from_gguf(
        &self,
        name: &str,
        path: &str,
        spec: &ModelDeploySpec,
        cid: localcode_core::CorrelationId,
    ) -> Result<RunningEndpoint, LocalCodeError> {
        // Bake context / GPU-offload straight into the Modelfile here (this path
        // already creates a model), so a local-GGUF deploy honors tuning too.
        let mut modelfile = format!("FROM {path}\n");
        if spec.context_length > 0 {
            modelfile.push_str(&format!("PARAMETER num_ctx {}\n", spec.context_length));
        }
        if let Some(n) = spec.tuning.gpu_layers {
            modelfile.push_str(&format!("PARAMETER num_gpu {n}\n"));
        }
        let body = serde_json::json!({ "name": name, "modelfile": modelfile });
        let resp = self
            .pull_http
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
        runtime.model_id = Some(name.to_string());
        runtime.quantization = spec.quantization.clone();
        // Report the served context window so the agent compacts before overflow.
        runtime.context_tokens = (spec.context_length > 0).then_some(spec.context_length);
        runtime.status = RuntimeStatus::Healthy;
        runtime.correlation_id = cid.to_string();
        Ok(RunningEndpoint { runtime })
    }
}

fn pull_error(model_name: &str, detail: &str) -> LocalCodeError {
    LocalCodeError::new(
        ErrorCode::DeployDownloadFailed,
        format!("Ollama pull {model_name} failed: {detail}"),
    )
    .with_cause("Model not available under that name, or a network issue")
    .with_hint("HF GGUF repos pull as hf.co/{org}/{repo}:{QUANT}")
    .with_hint("Or use a library name (e.g. qwen2.5-coder:7b)")
    .retryable(true)
}

/// Map a deploy spec to an Ollama model name.
///
/// HF repo ids (contain '/') use Ollama's native Hugging Face integration:
/// `hf.co/{org}/{repo}[:{quant}]` — works for any public GGUF repo instead of
/// guessing library names that mostly don't exist.
fn ollama_model_name(spec: &ModelDeploySpec) -> String {
    let id = spec.model_id.trim();
    let lower = id.to_lowercase();

    // Well-known library shortcuts first (smaller + faster than HF pulls).
    if lower.contains("qwen2.5-coder") && lower.contains("7b") {
        return "qwen2.5-coder:7b".into();
    }

    if id.contains('/') {
        return match &spec.quantization {
            Some(q) if !q.is_empty() && q.to_uppercase() != "UNKNOWN_GGUF" => {
                format!("hf.co/{id}:{q}")
            }
            _ => format!("hf.co/{id}"),
        };
    }

    // Already an Ollama library name ("codellama", "qwen2.5-coder:7b", …).
    lower.replace(' ', "-")
}

/// A valid Ollama name for a tuning-derived model. Ollama names allow only
/// `[a-zA-Z0-9._-]` (plus one `:tag`), so the base's org/quant separators are
/// flattened; the `ctx{n}` tag keeps distinct contexts as distinct models.
fn derived_model_name(base: &str, ctx: u32) -> String {
    let stem = base.rsplit('/').next().unwrap_or(base);
    let safe: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let safe = safe.trim_matches('-');
    let safe = if safe.is_empty() { "model" } else { safe };
    format!("localcode-{safe}:ctx{ctx}")
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

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(model_id: &str, quant: Option<&str>) -> ModelDeploySpec {
        ModelDeploySpec {
            job_id: "j".into(),
            model_id: model_id.into(),
            quantization: quant.map(|q| q.to_string()),
            weight_files: vec![],
            download_urls: vec![],
            local_path: None,
            port: None,
            context_length: 8192,
            force_oversize: false,
            tuning: crate::DeployTuning::default(),
            command_override: None,
        }
    }

    #[test]
    fn hf_repos_use_hf_co_pull() {
        assert_eq!(
            ollama_model_name(&spec("TheBloke/CodeLlama-7B-GGUF", Some("Q4_K_M"))),
            "hf.co/TheBloke/CodeLlama-7B-GGUF:Q4_K_M"
        );
        assert_eq!(
            ollama_model_name(&spec("org/Repo-GGUF", None)),
            "hf.co/org/Repo-GGUF"
        );
    }

    #[test]
    fn library_names_pass_through() {
        assert_eq!(
            ollama_model_name(&spec("qwen2.5-coder:7b", None)),
            "qwen2.5-coder:7b"
        );
        assert_eq!(ollama_model_name(&spec("codellama", None)), "codellama");
    }

    #[test]
    fn derived_name_is_ollama_safe() {
        // org/quant separators flattened, context encoded as the tag.
        assert_eq!(
            derived_model_name("hf.co/TheBloke/CodeLlama-7B-GGUF:Q4_K_M", 16384),
            "localcode-CodeLlama-7B-GGUF-Q4_K_M:ctx16384"
        );
        assert_eq!(
            derived_model_name("qwen2.5-coder:7b", 4096),
            "localcode-qwen2.5-coder-7b:ctx4096"
        );
    }
}
