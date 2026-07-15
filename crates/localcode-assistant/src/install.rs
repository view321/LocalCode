//! Install the local Bonsai assistant: PrismML llama.cpp + first-run `-hf` pull.
//!
//! The model is served with:
//!   `llama-server -hf prism-ml/Bonsai-27B-gguf:Q4_1`
//! so llama.cpp downloads the GGUF into its cache on first start. Runtime must
//! be the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp)
//! (model card: `git clone` + `cmake -B build [-DGGML_CUDA=ON]` + build). We keep
//! a small readiness marker under the assistant data dir so the TUI knows when
//! a previous pull succeeded without re-scanning the llama cache.

use crate::constants::{ASSISTANT_DISPLAY_NAME, BONSAI_BYTES, BONSAI_HF_REF, BONSAI_REPO};
use localcode_backends::{resolve_install_plan, InstallPlan, Repoint};
use localcode_core::config::Config;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::paths::AppPaths;
use std::path::PathBuf;
use tokio::sync::mpsc::UnboundedSender;
use tracing::info;

/// Marker written after a successful `ensure_running` (model pulled + healthy).
pub fn ready_marker_path(paths: &AppPaths) -> PathBuf {
    paths.assistant_dir().join(".bonsai-hf-ready")
}

/// Record that the local assistant model is available (downloaded via -hf).
pub fn mark_ready(paths: &AppPaths) {
    let dir = paths.assistant_dir();
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(
        ready_marker_path(paths),
        format!("{BONSAI_HF_REF}\n"),
    );
}

/// True when a previous successful -hf start left a readiness marker, or when
/// a GGUF from a prior manual install still sits in the assistant dir.
pub fn model_installed(paths: &AppPaths) -> bool {
    if ready_marker_path(paths).is_file() {
        return true;
    }
    // Backward-compat: older installs dropped a GGUF into the assistant dir.
    let legacy = [
        paths.assistant_dir().join("Bonsai-27B-dspark-Q4_1.gguf"),
        paths.assistant_dir().join("Bonsai-27B-Q1_0.gguf"),
    ];
    legacy.iter().any(|p| {
        std::fs::metadata(p)
            .map(|m| m.is_file() && m.len() > 1_000_000)
            .unwrap_or(false)
    })
}

/// Where a manually placed GGUF would live (legacy / diagnostics only).
pub fn model_path(paths: &AppPaths) -> PathBuf {
    paths
        .assistant_dir()
        .join("Bonsai-27B-dspark-Q4_1.gguf")
}

/// Resolve `llama-server` for Bonsai: prefer a managed **PrismML** build
/// (custom 1-bit kernels), then config path / PATH / any managed install.
pub fn resolve_llama_bin(cfg: &Config, paths: &AppPaths) -> Option<PathBuf> {
    if let Some(p) = localcode_backends::resolve_prism_llamacpp_bin(paths) {
        return Some(p);
    }
    localcode_backends::resolve_llamacpp_bin(&cfg.backends.llamacpp.bin, paths)
}

/// What still needs to happen before the local assistant can start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallNeed {
    /// llama-server is present; model will be pulled via -hf on first start
    /// (or is already marked ready).
    Ready,
    /// Need llama.cpp only.
    LlamaCppOnly,
    /// Need first -hf model pull (llama already available).
    ModelOnly,
    /// Need both.
    Both,
}

pub fn install_need(_cfg: &Config, paths: &AppPaths) -> InstallNeed {
    // Bonsai requires the PrismML fork — a stock llama-server on PATH is not enough.
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
             The model pack is already marked ready."
        ),
        InstallNeed::ModelOnly => format!(
            "Pull {ASSISTANT_DISPLAY_NAME} via llama-server -hf {BONSAI_HF_REF} \
             (~{size_gb:.1} GB on first launch). PrismML llama.cpp is already available."
        ),
        InstallNeed::Both => format!(
            "Install the local {ASSISTANT_DISPLAY_NAME} assistant?\n\n\
             • Auto-build PrismML llama.cpp (git clone + cmake, as on the model card)\n\
               — or download a Prism prebuilt if git/cmake are missing\n\
             • First start: `llama-server -hf {BONSAI_HF_REF}` (~{size_gb:.1} GB)\n\n\
             This becomes your default conversation model. It can search Hugging Face, \
             read model cards, help deploy models, and fix LocalCode issues. \
             You can decline and use a hosted provider later."
        ),
    }
}

/// Full install: llama.cpp (if needed) + first -hf pull (start server until healthy).
/// Returns optional [`Repoint`] when a managed llama-server was fetched.
pub async fn install_local_assistant(
    cfg: &Config,
    paths: &AppPaths,
    progress: UnboundedSender<String>,
) -> Result<Option<Repoint>, LocalCodeError> {
    paths.ensure_dirs()?;
    let need = install_need(cfg, paths);
    let mut repoint = None;

    // Bonsai always needs the PrismML fork — even if a stock llama-server is
    // already on PATH. ensure_llamacpp_installed builds/fetches Prism when the
    // managed tree has no .prism-ml marker.
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
                // Fall back to the raw plan error path for Manual installs.
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

    // Ensure llama-server is resolvable after install.
    let mut cfg = cfg.clone();
    if let Some(r) = &repoint {
        cfg.backends.llamacpp.bin = r.bin.clone();
    }
    if resolve_llama_bin(&cfg, paths).is_none() {
        return Err(LocalCodeError::new(
            ErrorCode::BackendBinaryMissing,
            "llama-server not found after PrismML install",
        )
        .with_hint("Install git + cmake, or download a Prism release from github.com/PrismML-Eng/llama.cpp/releases"));
    }

    if !model_installed(paths) {
        let _ = progress.send(format!(
            "Starting llama-server -hf {BONSAI_HF_REF} (downloads ~{:.1} GB on first run)…",
            BONSAI_BYTES as f64 / 1_073_741_824.0
        ));
        let _ = progress.send(format!(
            "Repo: https://huggingface.co/{BONSAI_REPO}"
        ));
        // Pull + load: ensure_running marks ready on health. Stop afterward so
        // the TUI warm-start owns the long-lived child (avoids a leaked process).
        let rt = crate::runtime::ensure_running(&cfg, paths).await.map_err(|e| {
            e.with_hint(format!(
                "First-run command: llama-server -hf {BONSAI_HF_REF}"
            ))
        })?;
        rt.stop().await;
        let _ = progress.send(format!(
            "{ASSISTANT_DISPLAY_NAME} ready via -hf {BONSAI_HF_REF}"
        ));
    } else {
        let _ = progress.send(format!(
            "{ASSISTANT_DISPLAY_NAME} already marked ready ({BONSAI_HF_REF})"
        ));
    }

    if !model_installed(paths) {
        return Err(LocalCodeError::new(
            ErrorCode::DeployDownloadFailed,
            "Bonsai model was not marked ready after install",
        )
        .with_hint(format!("Expected readiness after: llama-server -hf {BONSAI_HF_REF}"))
        .retryable(true));
    }

    info!(hf = BONSAI_HF_REF, "local assistant model installed via -hf");
    Ok(repoint)
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
    fn offer_body_mentions_hf_ref() {
        let body = install_offer_body(&InstallNeed::Both);
        assert!(body.contains("Bonsai"));
        assert!(body.contains("Q4_1") || body.contains("-hf"));
        assert!(body.contains("GB"));
    }

    #[test]
    fn mark_ready_sets_installed() {
        let dir = tempdir().unwrap();
        let paths = AppPaths::from_home(dir.path().to_path_buf());
        paths.ensure_dirs().unwrap();
        assert!(!model_installed(&paths));
        mark_ready(&paths);
        assert!(model_installed(&paths));
    }
}
