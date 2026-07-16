//! One-click deploy pipeline with progress events.

use crate::{BackendKind, BackendRegistry, ModelDeploySpec};
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus, Severity};
use localcode_core::ids::CorrelationId;
use localcode_gpu::{predict_fit, FitRequest, GpuInventory};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeployRequest {
    pub model_id: String,
    pub quantization: Option<String>,
    pub weight_bytes: u64,
    pub weight_files: Vec<String>,
    pub download_urls: Vec<String>,
    pub local_path: Option<String>,
    pub backend: BackendKind,
    pub port: Option<u16>,
    pub context_length: u32,
    /// Per-backend launch tuning (VRAM fraction, tensor-parallel, GPU layers,
    /// plus optional extra CLI flags from the model card / assistant).
    #[serde(default)]
    pub tuning: crate::DeployTuning,
    /// Full launch command entered by the user (or assistant) that replaces the
    /// backend-built one. `None` = build the command from the fields above.
    #[serde(default)]
    pub command_override: Option<String>,
    /// User acknowledged oversize warning.
    pub continue_despite_oversize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeployProgress {
    pub job_id: String,
    pub percent: u8,
    pub message: String,
    pub correlation_id: String,
}

#[derive(Debug, Clone)]
pub struct DeployJob {
    pub id: String,
    pub correlation_id: CorrelationId,
}

pub struct DeployService {
    registry: Arc<BackendRegistry>,
    events: EventBus,
    gpu: GpuInventory,
    /// Where downloaded weights live (usually AppPaths::models_cache).
    models_dir: PathBuf,
    /// Bearer token for the model registry (HF_TOKEN) — used only for weight
    /// downloads from the configured registry endpoints.
    registry_token: Option<String>,
    /// Ordered HF web roots (primary first, mirrors, then huggingface.co). A
    /// download URL on any of these hosts is retried against the others when it
    /// fails — the key to deploying from an isolated network.
    mirror_hosts: Vec<String>,
}

impl DeployService {
    pub fn new(
        registry: Arc<BackendRegistry>,
        events: EventBus,
        gpu: GpuInventory,
        models_dir: PathBuf,
        registry_token: Option<String>,
        mirror_hosts: Vec<String>,
    ) -> Self {
        Self {
            registry,
            events,
            gpu,
            models_dir,
            registry_token,
            mirror_hosts,
        }
    }

    pub fn fit_check(&self, req: &DeployRequest) -> localcode_gpu::FitPrediction {
        predict_fit(
            &self.gpu,
            &FitRequest {
                weight_bytes: req.weight_bytes,
                param_count: None,
                quant_label: req.quantization.clone(),
                context_length: req.context_length,
                backend: req.backend.as_str().into(),
            },
        )
    }

    /// Run deploy. Never hard-blocks on VRAM; requires continue flag if oversize.
    ///
    /// Error contract: the oversize soft-gate is returned as
    /// `DeployOversizedWarning` WITHOUT publishing events (the caller shows a
    /// Continue dialog); every other failure publishes exactly one
    /// `DeployFailed` event and also returns the error.
    pub async fn deploy(&self, req: DeployRequest) -> Result<DeployJob, LocalCodeError> {
        let job_id = Uuid::new_v4().to_string();
        let cid = CorrelationId::new();
        info!(%cid, job_id = %job_id, model = %req.model_id, "deploy start");

        let fit = self.fit_check(&req);
        if let Some(warning) = &fit.warning {
            if !req.continue_despite_oversize {
                // Soft gate: structured warning; caller shows Continue.
                return Err(LocalCodeError::new(
                    ErrorCode::DeployOversizedWarning,
                    warning.clone(),
                )
                .with_correlation(cid)
                .with_cause("Predicted VRAM usage exceeds free or total memory")
                .with_hint("Click Continue to deploy anyway (may spill to RAM/CPU or fail)")
                .with_hint("Or pick a smaller quantization")
                .with_details(serde_json::to_value(&fit).unwrap_or_default())
                .retryable(true));
            }
            self.events.publish(AppEvent::Notification {
                severity: Severity::Warn,
                title: "VRAM oversize — continuing".into(),
                body: warning.clone(),
                correlation_id: Some(cid.to_string()),
            });
        }

        match self.deploy_inner(&req, &job_id, &cid, fit.estimated_vram_bytes).await {
            Ok(job) => Ok(job),
            Err(e) => {
                self.events.publish(AppEvent::DeployFailed {
                    job_id: job_id.clone(),
                    error: e.clone(),
                });
                Err(e)
            }
        }
    }

    async fn deploy_inner(
        &self,
        req: &DeployRequest,
        job_id: &str,
        cid: &CorrelationId,
        est_vram_bytes: u64,
    ) -> Result<DeployJob, LocalCodeError> {
        self.emit_progress(job_id, cid, 5, "Preflight checks");

        // Fail fast on a format the backend can't load, before we spawn a server
        // and hand the user a Python traceback. vLLM/SGLang serve HF checkpoints
        // (safetensors/AWQ/GPTQ); a GGUF repo sends vLLM down its experimental
        // GGUF path, which on many builds crashes at import.
        if is_gguf_on_transformers_backend(req) {
            return Err(LocalCodeError::new(
                ErrorCode::DeployUnsupportedFormat,
                format!("{} can't serve GGUF weights", req.backend.as_str()),
            )
            .with_correlation(*cid)
            .with_cause("GGUF is a llama.cpp format; vLLM/SGLang load safetensors/AWQ/GPTQ checkpoints")
            .with_hint("Deploy this GGUF model with the llama.cpp or Ollama backend (/backend)")
            .with_hint("For vLLM, pick a full-precision or AWQ/GPTQ (safetensors) model"));
        }
        // Same fail-fast for colibrì containers: they are engine-specific int4
        // shards no other backend can parse, and the colibrì engines in turn
        // can't load GGUF or plain transformers checkpoints.
        if let Some(err) = colibri_format_conflict(req) {
            return Err(err.with_correlation(*cid));
        }

        self.emit_progress(job_id, cid, 10, "Ensuring backend ready");
        let backend = self.registry.get(req.backend)?;
        backend.ensure_ready().await?;

        // llama.cpp consumes a local GGUF and colibrì a local int4 container;
        // download weights if we only have registry URLs. Ollama pulls by
        // itself; vLLM/SGLang resolve HF ids through their own hub caches.
        let mut local_path = req.local_path.clone();
        if matches!(
            req.backend,
            BackendKind::LlamaCpp | BackendKind::Colibri | BackendKind::ColibriHy3
        ) && local_path.is_none()
        {
            local_path = Some(self.download_weights(req, job_id, cid).await?);
        }

        self.emit_progress(job_id, cid, 40, "Deploying model");
        let spec = ModelDeploySpec {
            job_id: job_id.to_string(),
            model_id: req.model_id.clone(),
            quantization: req.quantization.clone(),
            weight_files: req.weight_files.clone(),
            download_urls: req.download_urls.clone(),
            local_path,
            port: req.port,
            context_length: req.context_length,
            force_oversize: req.continue_despite_oversize,
            tuning: req.tuning.clone(),
            command_override: req.command_override.clone(),
        };

        let endpoint = backend.deploy(spec, &self.events).await?;

        self.emit_progress(job_id, cid, 90, "Registering runtime");
        // Attach the VRAM estimate to the dashboard monitor the backend just
        // registered (keyed by the same runtime id).
        self.registry
            .monitors()
            .set_vram(&endpoint.runtime.id.to_string(), est_vram_bytes);
        self.registry
            .register_runtime(endpoint.runtime.clone())
            .await;
        self.emit_progress(job_id, cid, 100, "Deploy complete");
        self.events.publish(AppEvent::DeployFinished {
            job_id: job_id.to_string(),
            runtime: endpoint.runtime.clone(),
        });
        self.events.publish(AppEvent::Notification {
            severity: Severity::Success,
            title: "Deploy complete".into(),
            body: format!("{} is ready for Coding", endpoint.runtime.name),
            correlation_id: Some(cid.to_string()),
        });
        Ok(DeployJob {
            id: job_id.to_string(),
            correlation_id: *cid,
        })
    }

    /// Download quant weight files into the models cache, emitting progress.
    /// Returns the path to the primary weight file (first shard).
    ///
    /// Delegates to the shared resumable [`crate::download::run_download`] engine
    /// so a dropped connection resumes from the partial `.part` (HTTP `Range`)
    /// rather than restarting — the same behaviour the detached background worker
    /// gives, kept identical by sharing one engine.
    async fn download_weights(
        &self,
        req: &DeployRequest,
        job_id: &str,
        cid: &CorrelationId,
    ) -> Result<String, LocalCodeError> {
        if req.download_urls.is_empty() {
            let err = LocalCodeError::new(
                ErrorCode::DeployDownloadFailed,
                "No downloadable weight files for this quantization",
            )
            .with_correlation(*cid);
            let err = if matches!(req.backend, BackendKind::Colibri | BackendKind::ColibriHy3) {
                err.with_cause("colibrì needs the full pre-converted container on local disk")
                    .with_hint(format!(
                        "Fetch it manually: hf download {} --local-dir <dir>, then deploy again",
                        req.model_id
                    ))
            } else {
                err.with_cause("Model detail has no resolvable GGUF files")
                    .with_hint("Pick a quantization with GGUF files, or use Ollama")
                    .with_hint("Or download manually and pass a local path")
            };
            return Err(err);
        }

        let spec = crate::download::DownloadSpec {
            model_id: req.model_id.clone(),
            quantization: req.quantization.clone(),
            backend: req.backend,
            dir: self.models_dir.join(sanitize_dir(&req.model_id)),
            files: req.weight_files.clone(),
            download_urls: req.download_urls.clone(),
            total_bytes: req.weight_bytes,
        };

        let total = req.weight_bytes;
        let events = self.events.clone();
        let job = job_id.to_string();
        let primary = crate::download::run_download(
            &spec,
            self.registry_token.as_deref(),
            &self.mirror_hosts,
            move |downloaded, msg| {
                events.publish(AppEvent::DeployProgress {
                    job_id: job.clone(),
                    percent: download_percent(downloaded, total),
                    message: msg.to_string(),
                });
            },
        )
        .await
        .map_err(|e| e.with_correlation(*cid))?;

        Ok(primary.display().to_string())
    }

    fn emit_progress(&self, job_id: &str, cid: &CorrelationId, percent: u8, message: &str) {
        info!(%cid, percent, message);
        self.events.publish(AppEvent::DeployProgress {
            job_id: job_id.to_string(),
            percent,
            message: message.to_string(),
        });
    }
}

/// Map download progress into the 15..=38 band of the overall deploy.
fn download_percent(fetched: u64, total: u64) -> u8 {
    if total == 0 {
        25
    } else {
        let frac = (fetched as f64 / total as f64).clamp(0.0, 1.0);
        15 + (frac * 23.0) as u8
    }
}

/// True when a GGUF model is being deployed to a backend that loads HF
/// transformers checkpoints rather than GGUF. Detected from the weight
/// filenames (definitive) or a GGUF-style quant label (fallback when a model id
/// is deployed without file metadata).
fn is_gguf_on_transformers_backend(req: &DeployRequest) -> bool {
    if !matches!(req.backend, BackendKind::Vllm | BackendKind::Sglang) {
        return false;
    }
    let file_is_gguf = req
        .weight_files
        .iter()
        .any(|f| f.to_lowercase().ends_with(".gguf"));
    let label_is_gguf = req.quantization.as_deref().is_some_and(|q| {
        let u = q.to_uppercase();
        // GGUF K-quants (Q4_K_M, Q8_0, …) and I-quants (IQ4_XS, …); AWQ/GPTQ/
        // FP16/INT8 (vLLM-servable safetensors labels) don't match.
        u.contains("GGUF") || u.starts_with("IQ") || u.starts_with('Q')
    });
    file_is_gguf || label_is_gguf
}

/// True when this deploy's weights look like a colibrì int4 container:
/// `out-*.safetensors` shards / `.qs` per-row scale files (definitive), or —
/// when file metadata is missing — a quant label or model id that names
/// colibrì together with an engine family marker (int4/GLM/Hy3/Hunyuan).
/// The extra marker keeps an unrelated model that merely has "colibri" in its
/// name from being blocked on vLLM.
fn looks_like_colibri_container(req: &DeployRequest) -> bool {
    let file_hit = req.weight_files.iter().any(|f| {
        let l = f.to_lowercase();
        let name = l.rsplit(['/', '\\']).next().unwrap_or(&l);
        l.ends_with(".qs") || (name.starts_with("out-") && name.ends_with(".safetensors"))
    });
    let marked = |s: &str| {
        let l = s.to_lowercase();
        l.contains("colibri")
            && (l.contains("int4") || l.contains("glm") || l.contains("hy3") || l.contains("hunyuan"))
    };
    file_hit
        || req.quantization.as_deref().is_some_and(marked)
        || marked(&req.model_id)
}

/// Fail-fast check for colibrì-format mismatches in either direction. Returns
/// the ready-made error (minus correlation) when the deploy can't work.
fn colibri_format_conflict(req: &DeployRequest) -> Option<LocalCodeError> {
    let on_colibri = matches!(req.backend, BackendKind::Colibri | BackendKind::ColibriHy3);
    let looks_colibri = looks_like_colibri_container(req);

    if !on_colibri && looks_colibri {
        return Some(
            LocalCodeError::new(
                ErrorCode::DeployUnsupportedFormat,
                format!("{} can't serve a colibrì int4 container", req.backend.as_str()),
            )
            .with_cause("colibrì containers (out-*.safetensors shards + .qs scales) are engine-specific")
            .with_hint("Deploy GLM-5.2 containers with the colibri backend, Hy3 with colibri-hy3"),
        );
    }
    if on_colibri && !looks_colibri {
        // Block only when the metadata affirmatively says another format; a
        // deploy with no file metadata may still be a valid local container
        // (the backend validates config.json before spawning).
        let is_gguf = req
            .weight_files
            .iter()
            .any(|f| f.to_lowercase().ends_with(".gguf"))
            || req.quantization.as_deref().is_some_and(|q| {
                let u = q.to_uppercase();
                u.contains("GGUF") || u.starts_with("IQ") || u.starts_with('Q')
            });
        let is_plain_checkpoint = req.weight_files.iter().any(|f| {
            let l = f.to_lowercase();
            l.ends_with(".safetensors") || l.ends_with(".bin") || l.ends_with(".pt")
        });
        if is_gguf || is_plain_checkpoint {
            return Some(
                LocalCodeError::new(
                    ErrorCode::DeployUnsupportedFormat,
                    format!(
                        "{} serves only pre-converted colibrì int4 containers",
                        req.backend.as_str()
                    ),
                )
                .with_cause("GGUF belongs on llama.cpp/Ollama; safetensors checkpoints on vLLM/SGLang")
                .with_hint("Pick a pre-converted repo (e.g. mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp, UnderstandLing/Hy3-colibri-int4)")
                .with_hint("Or convert once with `coli convert` and deploy the output directory"),
            );
        }
    }
    None
}

fn sanitize_dir(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .take(120)
        .collect()
}

/// Preview the exact launch command a deploy would run, so the deploy panel (and
/// the assistant) can show it as an editable field whose "no edit" value
/// reproduces the built command byte-for-byte. Empty for Ollama, which has no
/// spawned process to override. `model` is the local weight path when known,
/// else the HF model id.
pub fn preview_deploy_command(
    cfg: &localcode_core::config::BackendsConfig,
    backend: BackendKind,
    model: &str,
    port: Option<u16>,
    context_length: u32,
    tuning: &crate::DeployTuning,
) -> String {
    let (program, args) = match backend {
        BackendKind::Vllm => {
            let p = port.unwrap_or(cfg.vllm.port);
            crate::vllm::plan_command(&cfg.vllm, model, p, context_length, tuning)
        }
        BackendKind::LlamaCpp => {
            let p = port.unwrap_or(cfg.llamacpp.port);
            crate::llamacpp::plan_command(&cfg.llamacpp, model, p, context_length, tuning)
        }
        BackendKind::Sglang => {
            let p = port.unwrap_or(cfg.sglang.port);
            crate::sglang::plan_command(&cfg.sglang, model, p, context_length, tuning)
        }
        // No context flag: coli sizes its own KV slots.
        BackendKind::Colibri => {
            let p = port.unwrap_or(cfg.colibri.port);
            crate::colibri::plan_command(&cfg.colibri, BackendKind::Colibri, model, p, tuning)
        }
        BackendKind::ColibriHy3 => {
            let p = port.unwrap_or(cfg.colibri_hy3.port);
            crate::colibri::plan_command(&cfg.colibri_hy3, BackendKind::ColibriHy3, model, p, tuning)
        }
        BackendKind::Ollama => return String::new(),
    };
    crate::format_command(&program, &args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_percent_bands() {
        assert_eq!(download_percent(0, 0), 25);
        assert_eq!(download_percent(0, 100), 15);
        assert_eq!(download_percent(100, 100), 38);
    }

    fn req(backend: BackendKind, files: &[&str], quant: Option<&str>) -> DeployRequest {
        DeployRequest {
            model_id: "org/model".into(),
            quantization: quant.map(Into::into),
            weight_bytes: 0,
            weight_files: files.iter().map(|s| s.to_string()).collect(),
            download_urls: vec![],
            local_path: None,
            backend,
            port: None,
            context_length: 8192,
            tuning: crate::DeployTuning::default(),
            command_override: None,
            continue_despite_oversize: false,
        }
    }

    #[test]
    fn gguf_rejected_on_transformers_backends() {
        assert!(is_gguf_on_transformers_backend(&req(
            BackendKind::Vllm,
            &["model-Q4_K_M.gguf"],
            Some("Q4_K_M"),
        )));
        // Detected from the file even without a label.
        assert!(is_gguf_on_transformers_backend(&req(
            BackendKind::Sglang,
            &["a.gguf"],
            None,
        )));
    }

    #[test]
    fn safetensors_allowed_on_vllm() {
        assert!(!is_gguf_on_transformers_backend(&req(
            BackendKind::Vllm,
            &["model.safetensors"],
            Some("FP16"),
        )));
        assert!(!is_gguf_on_transformers_backend(&req(
            BackendKind::Vllm,
            &["model.safetensors"],
            Some("AWQ"),
        )));
    }

    #[test]
    fn gguf_allowed_on_llamacpp() {
        // The guard is transformers-backends only; llama.cpp wants GGUF.
        assert!(!is_gguf_on_transformers_backend(&req(
            BackendKind::LlamaCpp,
            &["a.gguf"],
            Some("Q4_K_M"),
        )));
    }

    #[test]
    fn colibri_container_blocked_off_colibri_backends() {
        // Shard names are definitive, whatever the label says.
        let r = req(BackendKind::Vllm, &["out-00001.safetensors", "out-mtp-0.safetensors"], None);
        assert!(colibri_format_conflict(&r).is_some());
        // Without files, the id must carry colibri + a family marker.
        let mut r = req(BackendKind::LlamaCpp, &[], None);
        r.model_id = "mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp".into();
        assert!(colibri_format_conflict(&r).is_some());
        // A model that merely mentions colibri is NOT blocked (no marker).
        let mut r = req(BackendKind::Vllm, &["model.safetensors"], Some("FP16"));
        r.model_id = "someone/colibri-7b-instruct".into();
        assert!(colibri_format_conflict(&r).is_none());
    }

    #[test]
    fn foreign_formats_blocked_on_colibri() {
        assert!(colibri_format_conflict(&req(
            BackendKind::Colibri,
            &["m-Q4_K_M.gguf"],
            Some("Q4_K_M"),
        ))
        .is_some());
        assert!(colibri_format_conflict(&req(
            BackendKind::ColibriHy3,
            &["model-00001-of-00002.safetensors"],
            Some("BF16"),
        ))
        .is_some());
    }

    #[test]
    fn colibri_container_and_bare_local_deploys_allowed_on_colibri() {
        let mut r = req(
            BackendKind::ColibriHy3,
            &["out-00001.safetensors", "scales.qs"],
            None,
        );
        r.model_id = "UnderstandLing/Hy3-colibri-int4".into();
        assert!(colibri_format_conflict(&r).is_none());
        // No metadata at all (local dir deploy): let the backend validate.
        assert!(colibri_format_conflict(&req(BackendKind::Colibri, &[], None)).is_none());
    }
}
