//! One-click deploy pipeline with progress events.

use crate::{BackendKind, BackendRegistry, ModelDeploySpec};
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus, Severity};
use localcode_core::ids::CorrelationId;
use localcode_gpu::{predict_fit, FitRequest, GpuInventory};
use serde::{Deserialize, Serialize};
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
}

impl DeployService {
    pub fn new(registry: Arc<BackendRegistry>, events: EventBus, gpu: GpuInventory) -> Self {
        Self {
            registry,
            events,
            gpu,
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
    pub async fn deploy(&self, req: DeployRequest) -> Result<DeployJob, LocalCodeError> {
        let job_id = Uuid::new_v4().to_string();
        let cid = CorrelationId::new();
        info!(%cid, job_id = %job_id, model = %req.model_id, "deploy start");

        let fit = self.fit_check(&req);
        if let Some(warning) = &fit.warning {
            if !req.continue_despite_oversize {
                // Soft gate: return structured warning (caller shows Continue)
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

        self.emit_progress(&job_id, &cid, 5, "Preflight checks");

        // Disk space warn (best-effort)
        if let Some(path) = req.local_path.as_ref() {
            if !std::path::Path::new(path).exists() && req.weight_bytes > 0 {
                self.emit_progress(&job_id, &cid, 10, "Weights will be pulled by backend");
            }
        }

        self.emit_progress(&job_id, &cid, 20, "Ensuring backend ready");
        let backend = self.registry.get(req.backend)?;
        if let Err(e) = backend.ensure_ready().await {
            self.events.publish(AppEvent::DeployFailed {
                job_id: job_id.clone(),
                error: e.clone(),
            });
            return Err(e);
        }

        self.emit_progress(&job_id, &cid, 40, "Deploying model");
        let spec = ModelDeploySpec {
            model_id: req.model_id.clone(),
            quantization: req.quantization.clone(),
            weight_files: req.weight_files.clone(),
            download_urls: req.download_urls.clone(),
            local_path: req.local_path.clone(),
            port: req.port,
            context_length: req.context_length,
            force_oversize: req.continue_despite_oversize,
        };

        match backend.deploy(spec).await {
            Ok(endpoint) => {
                self.emit_progress(&job_id, &cid, 90, "Registering runtime");
                self.registry
                    .register_runtime(endpoint.runtime.clone())
                    .await;
                self.emit_progress(&job_id, &cid, 100, "Deploy complete");
                self.events.publish(AppEvent::DeployFinished {
                    job_id: job_id.clone(),
                    runtime: endpoint.runtime.clone(),
                });
                self.events.publish(AppEvent::Notification {
                    severity: Severity::Success,
                    title: "Deploy complete".into(),
                    body: format!("{} is ready for Coding", endpoint.runtime.name),
                    correlation_id: Some(cid.to_string()),
                });
                Ok(DeployJob {
                    id: job_id,
                    correlation_id: cid,
                })
            }
            Err(e) => {
                self.events.publish(AppEvent::DeployFailed {
                    job_id: job_id.clone(),
                    error: e.clone(),
                });
                self.events.publish(AppEvent::ErrorRaised { error: e.clone() });
                Err(e)
            }
        }
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
