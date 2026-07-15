//! Install the local Bonsai assistant: managed llama.cpp + GGUF weights.
//!
//! Reuses the backends install plan for llama.cpp and downloads
//! `prism-ml/Bonsai-27B-gguf` `Bonsai-27B-Q1_0.gguf` (~3.8 GB) into the
//! assistant data directory. Progress lines are streamed for the TUI.

use crate::constants::{ASSISTANT_DISPLAY_NAME, BONSAI_BYTES, BONSAI_FILE, BONSAI_REPO};
use localcode_backends::{
    resolve_install_plan, run_install, InstallPlan, Repoint,
};
use localcode_core::config::Config;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::paths::AppPaths;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc::UnboundedSender;
use tracing::info;

/// Where the Bonsai GGUF lives under the assistant data dir.
pub fn model_path(paths: &AppPaths) -> PathBuf {
    paths.assistant_dir().join(BONSAI_FILE)
}

/// True when the GGUF is present and non-empty (partial downloads use `.part`).
pub fn model_installed(paths: &AppPaths) -> bool {
    let p = model_path(paths);
    std::fs::metadata(&p)
        .map(|m| m.is_file() && m.len() > 1_000_000)
        .unwrap_or(false)
}

/// Resolve `llama-server`: config path → PATH → managed backends/llamacpp dir.
pub fn resolve_llama_bin(cfg: &Config, paths: &AppPaths) -> Option<PathBuf> {
    let configured = &cfg.backends.llamacpp.bin;
    if let Ok(p) = which::which(configured) {
        return Some(p);
    }
    if configured != "llama-server" {
        if let Ok(p) = which::which("llama-server") {
            return Some(p);
        }
    }
    // Managed install from FetchLlamaCpp.
    let managed = paths.llamacpp_dir();
    find_file(&managed, if cfg!(windows) { "llama-server.exe" } else { "llama-server" })
        .or_else(|| find_file(&managed, "llama-server"))
}

fn find_file(root: &Path, name: &str) -> Option<PathBuf> {
    if !root.exists() {
        return None;
    }
    let direct = root.join(name);
    if direct.is_file() {
        return Some(direct);
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|n| n.to_str()) == Some(name) {
                return Some(p);
            }
        }
    }
    None
}

/// What still needs to happen before the local assistant can start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallNeed {
    /// Both llama.cpp and the GGUF are present.
    Ready,
    /// Need llama.cpp only (model already on disk).
    LlamaCppOnly,
    /// Need GGUF only (llama-server already on PATH / managed).
    ModelOnly,
    /// Need both.
    Both,
}

pub fn install_need(cfg: &Config, paths: &AppPaths) -> InstallNeed {
    let llama = resolve_llama_bin(cfg, paths).is_some();
    let model = model_installed(paths);
    match (llama, model) {
        (true, true) => InstallNeed::Ready,
        (false, true) => InstallNeed::LlamaCppOnly,
        (true, false) => InstallNeed::ModelOnly,
        (false, false) => InstallNeed::Both,
    }
}

/// Human-readable install offer body for the confirm banner.
pub fn install_offer_body(need: &InstallNeed) -> String {
    let size_gb = BONSAI_BYTES as f64 / 1_073_741_824.0;
    match need {
        InstallNeed::Ready => format!(
            "{ASSISTANT_DISPLAY_NAME} is already installed and ready."
        ),
        InstallNeed::LlamaCppOnly => format!(
            "Install llama.cpp (auto) so LocalCode can run the local {ASSISTANT_DISPLAY_NAME} \
             assistant. The ~{size_gb:.1} GB model is already on disk."
        ),
        InstallNeed::ModelOnly => format!(
            "Download {ASSISTANT_DISPLAY_NAME} (~{size_gb:.1} GB, {BONSAI_REPO}/{BONSAI_FILE}) \
             for the local repair assistant. llama.cpp is already available."
        ),
        InstallNeed::Both => format!(
            "Install the local {ASSISTANT_DISPLAY_NAME} assistant?\n\n\
             • Auto-install llama.cpp (managed binary)\n\
             • Download {BONSAI_REPO} ({BONSAI_FILE}, ~{size_gb:.1} GB)\n\n\
             The assistant helps fix LocalCode errors, reads model cards for deploy flags, \
             and can use shell + Hugging Face tools. You can decline and use a hosted provider later."
        ),
    }
}

/// Full install: llama.cpp (if needed) + Bonsai GGUF download.
/// Returns optional [`Repoint`] when a managed llama-server was fetched.
pub async fn install_local_assistant(
    cfg: &Config,
    paths: &AppPaths,
    progress: UnboundedSender<String>,
) -> Result<Option<Repoint>, LocalCodeError> {
    paths.ensure_dirs()?;
    let need = install_need(cfg, paths);
    let mut repoint = None;

    if matches!(need, InstallNeed::LlamaCppOnly | InstallNeed::Both) {
        let _ = progress.send("Installing llama.cpp…".into());
        let plan = resolve_install_plan(localcode_backends::BackendKind::LlamaCpp, paths);
        match &plan {
            InstallPlan::Automated { display, .. } => {
                let _ = progress.send(format!("$ {display}"));
            }
            InstallPlan::Manual { summary, steps, url } => {
                return Err(LocalCodeError::new(
                    ErrorCode::BackendInstallFailed,
                    summary.clone(),
                )
                .with_cause(steps.join("\n"))
                .with_hint(format!("See {url}"))
                .with_hint("Install llama-server manually, then re-run the assistant install"));
            }
        }
        repoint = run_install(&plan, None, progress.clone()).await?;
        if let Some(r) = &repoint {
            let _ = progress.send(format!("llama-server at {}", r.bin));
        }
    }

    if matches!(need, InstallNeed::ModelOnly | InstallNeed::Both)
        || !model_installed(paths)
    {
        download_bonsai(cfg, paths, &progress).await?;
    }

    if !model_installed(paths) {
        return Err(LocalCodeError::new(
            ErrorCode::DeployDownloadFailed,
            "Bonsai GGUF is missing after install",
        )
        .with_hint(format!("Expected {}", model_path(paths).display()))
        .retryable(true));
    }

    let _ = progress.send(format!(
        "{ASSISTANT_DISPLAY_NAME} ready at {}",
        model_path(paths).display()
    ));
    info!(path = %model_path(paths).display(), "local assistant model installed");
    Ok(repoint)
}

async fn download_bonsai(
    cfg: &Config,
    paths: &AppPaths,
    progress: &UnboundedSender<String>,
) -> Result<(), LocalCodeError> {
    let dest = model_path(paths);
    if model_installed(paths) {
        let _ = progress.send(format!("Cached: {}", dest.display()));
        return Ok(());
    }

    let dir = paths.assistant_dir();
    std::fs::create_dir_all(&dir).map_err(LocalCodeError::from)?;

    let hosts = cfg.hf_mirror_hosts();
    let mut urls: Vec<String> = hosts
        .iter()
        .map(|h| format!("{h}/{BONSAI_REPO}/resolve/main/{BONSAI_FILE}"))
        .collect();
    // Also try the HF hub CDN-style path used by some mirrors.
    urls.push(format!(
        "https://huggingface.co/{BONSAI_REPO}/resolve/main/{BONSAI_FILE}"
    ));
    // Dedupe
    let mut seen = std::collections::HashSet::new();
    urls.retain(|u| seen.insert(u.clone()));

    let _ = progress.send(format!(
        "Downloading {BONSAI_FILE} (~{:.1} GB)…",
        BONSAI_BYTES as f64 / 1_073_741_824.0
    ));

    let client = reqwest::Client::builder()
        .user_agent(format!("LocalCode/{}", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(20))
        // Large download — no total timeout; cancellation is the caller's job.
        .build()
        .map_err(|e| {
            LocalCodeError::new(ErrorCode::Internal, e.to_string())
                .with_source("assistant", "download_client")
        })?;

    let part = dir.join(format!("{BONSAI_FILE}.part"));
    let mut last_err: Option<LocalCodeError> = None;
    let token = cfg.hf_token();

    for (i, url) in urls.iter().enumerate() {
        let label = if i == 0 {
            format!("GET {BONSAI_FILE}")
        } else {
            format!("GET {BONSAI_FILE} (mirror {})", i + 1)
        };
        let _ = progress.send(label);

        let mut req = client.get(url);
        if let Some(t) = &token {
            req = req.bearer_auth(t);
        }
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                last_err = Some(
                    LocalCodeError::new(ErrorCode::DeployDownloadFailed, e.to_string())
                        .with_cause(format!("Network error: {url}"))
                        .retryable(true),
                );
                continue;
            }
        };
        if !resp.status().is_success() {
            last_err = Some(
                LocalCodeError::new(
                    ErrorCode::DeployDownloadFailed,
                    format!("Download failed ({}): {BONSAI_FILE}", resp.status()),
                )
                .with_cause(url.clone())
                .with_hint("Set HF_TOKEN if the repo is gated")
                .retryable(true),
            );
            continue;
        }

        match stream_to_file(resp, &part, progress).await {
            Ok(bytes) => {
                tokio::fs::rename(&part, &dest).await.map_err(|e| {
                    LocalCodeError::new(ErrorCode::IoError, e.to_string())
                })?;
                let _ = progress.send(format!(
                    "Downloaded {BONSAI_FILE} ({:.1} GB)",
                    bytes as f64 / 1_073_741_824.0
                ));
                return Ok(());
            }
            Err(e) => {
                let _ = tokio::fs::remove_file(&part).await;
                last_err = Some(e);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| {
        LocalCodeError::new(ErrorCode::DeployDownloadFailed, "No download URLs for Bonsai")
            .retryable(true)
    }))
}

async fn stream_to_file(
    resp: reqwest::Response,
    part: &Path,
    progress: &UnboundedSender<String>,
) -> Result<u64, LocalCodeError> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    let total = resp.content_length().unwrap_or(BONSAI_BYTES);
    let mut file = tokio::fs::File::create(part).await.map_err(|e| {
        LocalCodeError::new(ErrorCode::IoError, e.to_string())
            .with_cause(format!("Cannot create {}", part.display()))
    })?;

    let mut stream = resp.bytes_stream();
    let mut written: u64 = 0;
    let mut last_pct: u8 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            LocalCodeError::new(ErrorCode::DeployDownloadFailed, e.to_string())
                .with_cause("Download stream interrupted")
                .retryable(true)
        })?;
        file.write_all(&chunk).await.map_err(|e| {
            LocalCodeError::new(ErrorCode::IoError, e.to_string())
                .with_cause("Disk write failed — is there enough free space?")
        })?;
        written += chunk.len() as u64;
        let pct = if total > 0 {
            ((written * 100) / total).min(100) as u8
        } else {
            0
        };
        if pct >= last_pct.saturating_add(5) || pct == 100 {
            last_pct = pct;
            let _ = progress.send(format!(
                "Downloading {BONSAI_FILE}: {pct}% ({:.1}/{:.1} GB)",
                written as f64 / 1_073_741_824.0,
                total as f64 / 1_073_741_824.0
            ));
        }
    }
    file.flush().await.map_err(|e| {
        LocalCodeError::new(ErrorCode::IoError, e.to_string())
    })?;
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use localcode_core::config::Config;
    use tempfile::tempdir;

    #[test]
    fn need_both_on_empty_home() {
        let dir = tempdir().unwrap();
        let paths = AppPaths::from_home(dir.path().to_path_buf());
        paths.ensure_dirs().unwrap();
        let cfg = Config::default();
        assert_eq!(install_need(&cfg, &paths), InstallNeed::Both);
        assert!(!model_installed(&paths));
    }

    #[test]
    fn offer_body_mentions_size() {
        let body = install_offer_body(&InstallNeed::Both);
        assert!(body.contains("Bonsai"));
        assert!(body.contains("3.5") || body.contains("3.8") || body.contains("GB"));
    }
}
