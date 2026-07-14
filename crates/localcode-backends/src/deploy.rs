//! One-click deploy pipeline with progress events.

use crate::{BackendKind, BackendRegistry, ModelDeploySpec};
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus, Severity};
use localcode_core::ids::CorrelationId;
use localcode_gpu::{predict_fit, FitRequest, GpuInventory};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
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

        match self.deploy_inner(&req, &job_id, &cid).await {
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
    ) -> Result<DeployJob, LocalCodeError> {
        self.emit_progress(job_id, cid, 5, "Preflight checks");

        self.emit_progress(job_id, cid, 10, "Ensuring backend ready");
        let backend = self.registry.get(req.backend)?;
        backend.ensure_ready().await?;

        // llama.cpp consumes a local GGUF; download weights if we only have
        // registry URLs. Ollama pulls by itself; vLLM/SGLang resolve HF ids
        // through their own hub caches.
        let mut local_path = req.local_path.clone();
        if req.backend == BackendKind::LlamaCpp && local_path.is_none() {
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
        };

        let endpoint = backend.deploy(spec, &self.events).await?;

        self.emit_progress(job_id, cid, 90, "Registering runtime");
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
    async fn download_weights(
        &self,
        req: &DeployRequest,
        job_id: &str,
        cid: &CorrelationId,
    ) -> Result<String, LocalCodeError> {
        if req.download_urls.is_empty() {
            return Err(LocalCodeError::new(
                ErrorCode::DeployDownloadFailed,
                "No downloadable weight files for this quantization",
            )
            .with_correlation(*cid)
            .with_cause("Model detail has no resolvable GGUF files")
            .with_hint("Pick a quantization with GGUF files, or use Ollama")
            .with_hint("Or download manually and pass a local path"));
        }

        let dir = self.models_dir.join(sanitize_dir(&req.model_id));
        std::fs::create_dir_all(&dir).map_err(LocalCodeError::from)?;

        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .build()
            .unwrap_or_default();

        let total = req.weight_bytes;
        let mut fetched: u64 = 0;
        let mut paths: Vec<PathBuf> = Vec::new();

        for (i, url) in req.download_urls.iter().enumerate() {
            let filename = req
                .weight_files
                .get(i)
                .map(|f| f.as_str())
                .unwrap_or_else(|| url.rsplit('/').next().unwrap_or("weights.bin"));
            // Only the file name — never let a path component escape the dir.
            let filename = std::path::Path::new(filename)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "weights.bin".into());
            let dest = dir.join(&filename);

            if dest.exists()
                && std::fs::metadata(&dest).map(|m| m.len() > 0).unwrap_or(false)
            {
                self.emit_progress(
                    job_id,
                    cid,
                    download_percent(fetched, total),
                    &format!("Cached: {filename}"),
                );
                fetched += std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
                paths.push(dest);
                continue;
            }

            // Try the primary URL then every mirror host in turn: an isolated
            // network can reach a mirror even when huggingface.co times out.
            let candidates = expand_mirror_candidates(url, &self.mirror_hosts);
            let part = dir.join(format!("{filename}.part"));
            let bytes = self
                .download_one(&client, &candidates, &filename, &part, job_id, cid, fetched, total)
                .await?;
            fetched += bytes;
            tokio::fs::rename(&part, &dest).await.map_err(|e| {
                LocalCodeError::new(ErrorCode::IoError, e.to_string()).with_correlation(*cid)
            })?;
            paths.push(dest);
        }

        paths.sort();
        let primary = paths
            .iter()
            .find(|p| {
                p.extension()
                    .map(|e| e.eq_ignore_ascii_case("gguf"))
                    .unwrap_or(false)
            })
            .or_else(|| paths.first())
            .ok_or_else(|| {
                LocalCodeError::new(ErrorCode::DeployDownloadFailed, "No files downloaded")
                    .with_correlation(*cid)
            })?;
        Ok(primary.display().to_string())
    }

    /// Download one file, trying each candidate URL (primary then mirrors) into
    /// `part`. Returns the byte count on success. A transport error or non-2xx
    /// status falls through to the next mirror; only a local write error (disk)
    /// is fatal. Emits progress against the overall `total`.
    #[allow(clippy::too_many_arguments)]
    async fn download_one(
        &self,
        client: &reqwest::Client,
        candidates: &[String],
        filename: &str,
        part: &std::path::Path,
        job_id: &str,
        cid: &CorrelationId,
        fetched_base: u64,
        total: u64,
    ) -> Result<u64, LocalCodeError> {
        use futures::StreamExt;
        let mut last_err: Option<LocalCodeError> = None;
        let n = candidates.len();
        for (attempt, url) in candidates.iter().enumerate() {
            let via = if attempt == 0 {
                format!("Downloading {filename}")
            } else {
                format!("Downloading {filename} (mirror {}/{})", attempt + 1, n)
            };
            self.emit_progress(job_id, cid, download_percent(fetched_base, total), &via);

            let mut request = client.get(url);
            if let Some(token) = &self.registry_token {
                request = request.bearer_auth(token);
            }
            let resp = match request.send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(
                        LocalCodeError::new(ErrorCode::DeployDownloadFailed, e.to_string())
                            .with_correlation(*cid)
                            .with_cause(format!("Network error downloading {filename}"))
                            .with_hint("Check connectivity or configure registry.mirrors")
                            .retryable(true),
                    );
                    continue;
                }
            };
            if !resp.status().is_success() {
                last_err = Some(
                    LocalCodeError::new(
                        ErrorCode::DeployDownloadFailed,
                        format!("Download failed ({}): {filename}", resp.status()),
                    )
                    .with_correlation(*cid)
                    .with_hint("Gated model? Set HF_TOKEN")
                    .retryable(true),
                );
                continue;
            }

            let mut file = tokio::fs::File::create(part).await.map_err(|e| {
                LocalCodeError::new(ErrorCode::IoError, e.to_string()).with_correlation(*cid)
            })?;
            let mut stream = resp.bytes_stream();
            let mut file_bytes: u64 = 0;
            let mut last_emit = std::time::Instant::now();
            let mut stream_err: Option<LocalCodeError> = None;
            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        stream_err = Some(
                            LocalCodeError::new(ErrorCode::DeployDownloadFailed, e.to_string())
                                .with_correlation(*cid)
                                .retryable(true),
                        );
                        break;
                    }
                };
                file_bytes += chunk.len() as u64;
                file.write_all(&chunk).await.map_err(|e| {
                    LocalCodeError::new(ErrorCode::IoError, e.to_string())
                        .with_correlation(*cid)
                        .with_hint("Check free disk space")
                })?;
                if last_emit.elapsed() > std::time::Duration::from_millis(750) {
                    last_emit = std::time::Instant::now();
                    self.emit_progress(
                        job_id,
                        cid,
                        download_percent(fetched_base + file_bytes, total),
                        &format!("{via} — {:.1} GiB", (fetched_base + file_bytes) as f64 / GIB),
                    );
                }
            }
            file.flush().await.ok();
            drop(file);
            if let Some(e) = stream_err {
                // Partial download: discard and try the next mirror.
                let _ = tokio::fs::remove_file(part).await;
                last_err = Some(e);
                continue;
            }
            return Ok(file_bytes);
        }
        Err(last_err.unwrap_or_else(|| {
            LocalCodeError::new(ErrorCode::DeployDownloadFailed, format!("No source for {filename}"))
                .with_correlation(*cid)
                .retryable(true)
        }))
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

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Map download progress into the 15..=38 band of the overall deploy.
fn download_percent(fetched: u64, total: u64) -> u8 {
    if total == 0 {
        25
    } else {
        let frac = (fetched as f64 / total as f64).clamp(0.0, 1.0);
        15 + (frac * 23.0) as u8
    }
}

/// Expand a resolved download URL into the same path on every mirror host
/// (primary first, in `hosts` order). If the URL isn't on a known host it is
/// returned unchanged as the sole candidate. Mirrors the logic of
/// `localcode_hf::UrlBuilder::mirror_candidates` without a crate dependency.
fn expand_mirror_candidates(url: &str, hosts: &[String]) -> Vec<String> {
    let trimmed: Vec<String> = hosts
        .iter()
        .map(|h| h.trim_end_matches('/').to_string())
        .collect();
    for h in &trimmed {
        if let Some(rest) = url.strip_prefix(h.as_str()) {
            let mut out: Vec<String> = trimmed.iter().map(|host| format!("{host}{rest}")).collect();
            out.dedup();
            return out;
        }
    }
    vec![url.to_string()]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_candidates_swap_and_order() {
        let hosts = vec![
            "https://huggingface.co".to_string(),
            "https://hf-mirror.com".to_string(),
        ];
        let got = expand_mirror_candidates(
            "https://huggingface.co/org/model/resolve/main/m.gguf",
            &hosts,
        );
        assert_eq!(
            got,
            vec![
                "https://huggingface.co/org/model/resolve/main/m.gguf".to_string(),
                "https://hf-mirror.com/org/model/resolve/main/m.gguf".to_string(),
            ]
        );
    }

    #[test]
    fn mirror_candidates_unknown_host_passthrough() {
        let hosts = vec!["https://huggingface.co".to_string()];
        let got = expand_mirror_candidates("https://other.example/a/b.gguf", &hosts);
        assert_eq!(got, vec!["https://other.example/a/b.gguf".to_string()]);
    }

    #[test]
    fn download_percent_bands() {
        assert_eq!(download_percent(0, 0), 25);
        assert_eq!(download_percent(0, 100), 15);
        assert_eq!(download_percent(100, 100), 38);
    }
}
