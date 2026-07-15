//! Install the local Bonsai assistant: PrismML llama.cpp + language GGUF.
//!
//! Launch (model card):
//! ```text
//! llama-server -m Bonsai-27B-Q1_0.gguf [--md Bonsai-27B-dspark-Q4_1.gguf] \
//!     --host 127.0.0.1 --port … -ngl 99
//! ```
//! Q4_1 is the optional DSpark **drafter** only — never the sole `-m` weights.

use crate::constants::{
    ASSISTANT_DISPLAY_NAME, BONSAI_BYTES, BONSAI_DRAFT_BYTES, BONSAI_DRAFT_FILE, BONSAI_FILE,
    BONSAI_HF_REF, BONSAI_REPO,
};
use localcode_backends::{resolve_install_plan, InstallPlan, Repoint};
use localcode_core::config::Config;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::paths::AppPaths;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc::UnboundedSender;
use tracing::info;

/// Marker written after a successful download + healthy start.
pub fn ready_marker_path(paths: &AppPaths) -> PathBuf {
    paths.assistant_dir().join(".bonsai-ready")
}

/// Record that the local assistant model is available on disk.
pub fn mark_ready(paths: &AppPaths) {
    let dir = paths.assistant_dir();
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(
        ready_marker_path(paths),
        format!("{BONSAI_FILE}\n{BONSAI_HF_REF}\n"),
    );
}

/// True when the **language model** Q1_0 GGUF is present and looks complete.
pub fn model_installed(paths: &AppPaths) -> bool {
    gguf_looks_complete(&model_path(paths), BONSAI_BYTES)
}

fn gguf_looks_complete(path: &Path, expected: u64) -> bool {
    match std::fs::metadata(path) {
        Ok(m) => m.is_file() && m.len() >= expected * 9 / 10,
        Err(_) => false,
    }
}

/// On-disk path for the language-model Q1_0 GGUF.
pub fn model_path(paths: &AppPaths) -> PathBuf {
    paths.assistant_dir().join(BONSAI_FILE)
}

/// On-disk path for the optional DSpark Q4_1 drafter.
pub fn draft_path(paths: &AppPaths) -> PathBuf {
    paths.assistant_dir().join(BONSAI_DRAFT_FILE)
}

/// Resolve `llama-server` for Bonsai: prefer a managed **PrismML** build.
pub fn resolve_llama_bin(cfg: &Config, paths: &AppPaths) -> Option<PathBuf> {
    if let Some(p) = localcode_backends::resolve_prism_llamacpp_bin(paths) {
        return Some(p);
    }
    localcode_backends::resolve_llamacpp_bin(&cfg.backends.llamacpp.bin, paths)
}

/// What still needs to happen before the local assistant can start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallNeed {
    Ready,
    LlamaCppOnly,
    ModelOnly,
    Both,
}

pub fn install_need(_cfg: &Config, paths: &AppPaths) -> InstallNeed {
    let llama = localcode_backends::resolve_prism_llamacpp_bin(paths).is_some();
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
        InstallNeed::Ready => {
            format!("{ASSISTANT_DISPLAY_NAME} is already installed and ready.")
        }
        InstallNeed::LlamaCppOnly => format!(
            "Build/install PrismML llama.cpp (auto) so LocalCode can run the local \
             {ASSISTANT_DISPLAY_NAME} assistant. Stock llama.cpp cannot load this model — \
             the Bonsai card requires https://github.com/PrismML-Eng/llama.cpp. \
             The Q1_0 language GGUF is already on disk."
        ),
        InstallNeed::ModelOnly => format!(
            "Download {ASSISTANT_DISPLAY_NAME} language model `{BONSAI_FILE}` (~{size_gb:.1} GB), then:\n\
             `llama-server -m {BONSAI_FILE} -ngl 99`\n\
             Q1_0 only — the Q4_1 DSpark drafter is not loaded."
        ),
        InstallNeed::Both => format!(
            "Install the local {ASSISTANT_DISPLAY_NAME} assistant?\n\n\
             • Auto-build PrismML llama.cpp (git clone + cmake, as on the model card)\n\
             • Download language model `{BONSAI_FILE}` (~{size_gb:.1} GB), Q1_0 only\n\
             • Launch: `llama-server -m {BONSAI_FILE} --host 127.0.0.1 -ngl 99`\n\n\
             This becomes your default conversation model."
        ),
    }
}

/// Full install: PrismML llama.cpp + Q1_0 GGUF (+ optional Q4_1 draft) + smoke start.
pub async fn install_local_assistant(
    cfg: &Config,
    paths: &AppPaths,
    progress: UnboundedSender<String>,
) -> Result<Option<Repoint>, LocalCodeError> {
    paths.ensure_dirs()?;
    let need = install_need(cfg, paths);
    let mut repoint = None;

    if matches!(need, InstallNeed::LlamaCppOnly | InstallNeed::Both)
        || localcode_backends::resolve_prism_llamacpp_bin(paths).is_none()
    {
        let _ = progress.send(
            "Installing PrismML llama.cpp (Bonsai kernels; model-card build)…".into(),
        );
        match localcode_backends::ensure_llamacpp_installed(paths, progress.clone()).await {
            Ok(bin) => {
                let bin_s = bin.display().to_string();
                let _ = progress.send(format!("llama-server at {bin_s}"));
                repoint = Some(Repoint {
                    kind: localcode_backends::BackendKind::LlamaCpp,
                    bin: bin_s,
                });
            }
            Err(e) => {
                let plan = resolve_install_plan(localcode_backends::BackendKind::LlamaCpp, paths);
                if let InstallPlan::Manual {
                    summary,
                    steps,
                    url,
                } = &plan
                {
                    return Err(LocalCodeError::new(
                        ErrorCode::BackendInstallFailed,
                        summary.clone(),
                    )
                    .with_cause(steps.join("\n"))
                    .with_hint(format!("See {url}"))
                    .with_hint(
                        "Build PrismML llama.cpp: git clone https://github.com/PrismML-Eng/llama.cpp \
                         && cmake -B build -DGGML_CUDA=ON && cmake --build build -j",
                    ));
                }
                return Err(e.with_hint(
                    "https://huggingface.co/prism-ml/Bonsai-27B-gguf — custom llama.cpp required",
                ));
            }
        }
    }

    let mut cfg = cfg.clone();
    if let Some(r) = &repoint {
        cfg.backends.llamacpp.bin = r.bin.clone();
    }
    if resolve_llama_bin(&cfg, paths).is_none() {
        return Err(LocalCodeError::new(
            ErrorCode::BackendBinaryMissing,
            "llama-server not found after PrismML install",
        )
        .with_hint(
            "Install git + cmake, or download a Prism release from \
             github.com/PrismML-Eng/llama.cpp/releases",
        ));
    }

    // Required language model.
    if !model_installed(paths) {
        // Detect mistaken Q4_1-only install from earlier builds.
        let draft_only = gguf_looks_complete(&draft_path(paths), BONSAI_DRAFT_BYTES)
            && !model_installed(paths);
        if draft_only {
            let _ = progress.send(format!(
                "Found {BONSAI_DRAFT_FILE} (DSpark drafter only). \
                 Downloading language model {BONSAI_FILE} — Q4_1 alone cannot run."
            ));
        }
        let _ = progress.send(format!(
            "Downloading {BONSAI_FILE} (~{:.1} GB) from Hugging Face…",
            BONSAI_BYTES as f64 / 1_073_741_824.0
        ));
        download_gguf(
            paths,
            BONSAI_FILE,
            BONSAI_BYTES,
            cfg.hf_token().as_deref(),
            progress.clone(),
        )
        .await?;
    } else {
        let _ = progress.send(format!(
            "Language model already present ({})",
            model_path(paths).display()
        ));
    }

    // Q1-only: the Q4_1 DSpark drafter is deliberately not downloaded or loaded,
    // so no Q4 weights ever reach memory. (The `-md` speculative path is off.)

    let _ = progress.send(format!(
        "Smoke-start: llama-server -m {BONSAI_FILE} -ngl {} …",
        cfg.assistant.local_gpu_layers
    ));
    let rt = crate::runtime::ensure_running(&cfg, paths).await.map_err(|e| {
        e.with_hint(format!(
            "Expected: llama-server -m {} --host 127.0.0.1 --port {} -ngl {}",
            model_path(paths).display(),
            cfg.assistant.local_port,
            cfg.assistant.local_gpu_layers
        ))
        .with_hint(format!(
            "If stderr is empty, check {}",
            paths.assistant_dir().join("last-llama-server.log").display()
        ))
    })?;
    rt.stop().await;
    mark_ready(paths);
    let _ = progress.send(format!(
        "{ASSISTANT_DISPLAY_NAME} ready — `llama-server -m {BONSAI_FILE} -ngl 99`"
    ));

    if !model_installed(paths) {
        return Err(LocalCodeError::new(
            ErrorCode::DeployDownloadFailed,
            "Bonsai Q1_0 language GGUF missing after install",
        )
        .with_hint(format!("Expected file: {}", model_path(paths).display()))
        .retryable(true));
    }

    info!(file = BONSAI_FILE, "local assistant model installed (Q1_0 -m)");
    Ok(repoint)
}

/// Download a named GGUF from the Bonsai repo into the assistant data dir.
async fn download_gguf(
    paths: &AppPaths,
    filename: &str,
    expected_bytes: u64,
    hf_token: Option<&str>,
    progress: UnboundedSender<String>,
) -> Result<PathBuf, LocalCodeError> {
    use futures::StreamExt;

    let dest = paths.assistant_dir().join(filename);
    if gguf_looks_complete(&dest, expected_bytes) {
        return Ok(dest);
    }
    if dest.exists() {
        let _ = std::fs::remove_file(&dest);
    }
    let dir = paths.assistant_dir();
    std::fs::create_dir_all(&dir).map_err(LocalCodeError::from)?;
    let part = dir.join(format!("{filename}.part"));

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(12 * 3600))
        .user_agent(format!("LocalCode/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| {
            LocalCodeError::new(ErrorCode::Internal, e.to_string()).with_source("assistant", "http")
        })?;

    let urls = [
        format!("https://huggingface.co/{BONSAI_REPO}/resolve/main/{filename}"),
        format!("https://hf-mirror.com/{BONSAI_REPO}/resolve/main/{filename}"),
    ];

    let mut last_err: Option<LocalCodeError> = None;
    for (i, url) in urls.iter().enumerate() {
        let _ = progress.send(format!(
            "$ GET {url}{}",
            if i > 0 { " (mirror)" } else { "" }
        ));
        let mut req = client.get(url);
        if let Some(token) = hf_token {
            req = req.bearer_auth(token);
        }
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                last_err = Some(
                    LocalCodeError::new(ErrorCode::DeployDownloadFailed, e.to_string())
                        .with_hint("Check network / set HF_TOKEN if gated")
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
                .with_hint("Gated model? Set HF_TOKEN")
                .retryable(true),
            );
            continue;
        }

        let mut file = tokio::fs::File::create(&part).await.map_err(|e| {
            LocalCodeError::new(ErrorCode::IoError, e.to_string())
                .with_hint(format!("Cannot write {}", part.display()))
        })?;
        let mut stream = resp.bytes_stream();
        let mut written: u64 = 0;
        let mut last_log = std::time::Instant::now();
        let mut stream_err: Option<LocalCodeError> = None;
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    stream_err = Some(
                        LocalCodeError::new(ErrorCode::DeployDownloadFailed, e.to_string())
                            .retryable(true),
                    );
                    break;
                }
            };
            written += chunk.len() as u64;
            file.write_all(&chunk).await.map_err(|e| {
                LocalCodeError::new(ErrorCode::IoError, e.to_string())
                    .with_hint("Check free disk space")
            })?;
            if last_log.elapsed() > std::time::Duration::from_secs(2) {
                last_log = std::time::Instant::now();
                let pct = if expected_bytes > 0 {
                    (written * 100 / expected_bytes).min(99)
                } else {
                    0
                };
                let _ = progress.send(format!(
                    "Downloading {filename}: {pct}% ({:.2} / {:.1} GiB)",
                    written as f64 / 1_073_741_824.0,
                    expected_bytes as f64 / 1_073_741_824.0
                ));
            }
        }
        file.flush().await.ok();
        drop(file);
        if let Some(e) = stream_err {
            let _ = tokio::fs::remove_file(&part).await;
            last_err = Some(e);
            continue;
        }
        tokio::fs::rename(&part, &dest).await.map_err(|e| {
            LocalCodeError::new(ErrorCode::IoError, e.to_string())
                .with_hint("Failed to finalize GGUF download")
        })?;
        if !gguf_looks_complete(&dest, expected_bytes) {
            let got = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
            let _ = std::fs::remove_file(&dest);
            return Err(LocalCodeError::new(
                ErrorCode::DeployDownloadFailed,
                format!(
                    "Downloaded {filename} looks incomplete ({got} bytes, expected ~{expected_bytes})"
                ),
            )
            .retryable(true));
        }
        let _ = progress.send(format!("Downloaded {filename} → {}", dest.display()));
        return Ok(dest);
    }

    Err(last_err.unwrap_or_else(|| {
        LocalCodeError::new(
            ErrorCode::DeployDownloadFailed,
            format!("Could not download {filename}"),
        )
        .with_hint(format!(
            "Manual: hf download {BONSAI_REPO} {filename} --local-dir {}",
            paths.assistant_dir().display()
        ))
        .retryable(true)
    }))
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
    fn offer_body_mentions_q1_0_and_dash_m() {
        let body = install_offer_body(&InstallNeed::Both);
        assert!(body.contains("Bonsai"));
        assert!(body.contains("Q1_0") || body.contains(BONSAI_FILE));
        assert!(body.contains("-m"));
        assert!(body.contains("GB"));
    }

    #[test]
    fn model_installed_requires_q1_0_not_draft_only() {
        let dir = tempdir().unwrap();
        let paths = AppPaths::from_home(dir.path().to_path_buf());
        paths.ensure_dirs().unwrap();
        // Q4_1 draft alone must NOT count as installed language model.
        std::fs::write(draft_path(&paths), vec![0u8; 2_000_000]).unwrap();
        assert!(!model_installed(&paths));
        // Full-size language model.
        let mut big = vec![0u8; (BONSAI_BYTES * 9 / 10) as usize];
        big[0] = 1;
        std::fs::write(model_path(&paths), &big).unwrap();
        assert!(model_installed(&paths));
    }

    #[test]
    fn model_path_is_q1_0_file() {
        let dir = tempdir().unwrap();
        let paths = AppPaths::from_home(dir.path().to_path_buf());
        let p = model_path(&paths);
        assert!(
            p.ends_with(BONSAI_FILE) || p.to_string_lossy().ends_with(BONSAI_FILE),
            "{p:?}"
        );
    }
}
