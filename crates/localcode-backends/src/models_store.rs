//! On-disk model store: enumerate, inspect, and delete the model weights the
//! app downloaded into the models cache.
//!
//! LocalCode downloads GGUF weights for llama.cpp deploys into
//! `{models_cache}/{sanitized_model_id}/`. This module scans that tree so the
//! UI can show a "downloaded" tag in `/models`, list downloaded models on the
//! `/dash` dashboard, and free disk space by deleting a model's directory.
//!
//! [`sanitize_model_dir`] is lossy (both `org/model` and `org-model` map to
//! `org_model`), so at download time the deploy pipeline drops a
//! [`MODEL_ID_MARKER`] file holding the exact id; [`list_downloaded`] reads it
//! and falls back to the directory name when it is absent (older downloads).
//! Lookups by id ([`find_downloaded`]) go the other, lossless direction:
//! sanitize the requested id and probe that directory.

use localcode_core::error::{ErrorCode, LocalCodeError};
use std::path::{Path, PathBuf};

/// Marker file written into a model's cache directory holding its exact HF id,
/// so the lossy directory-name sanitization can be reversed for display.
pub const MODEL_ID_MARKER: &str = ".localcode_model_id";

/// Weight-file extensions that mark a directory as a real (non-empty) download.
const WEIGHT_EXTS: &[&str] = &["gguf", "safetensors", "bin", "pt", "pth"];

/// One model present on disk in the models cache.
#[derive(Debug, Clone)]
pub struct DownloadedModel {
    /// Exact HF id when known (from the marker file), else the directory name.
    pub model_id: String,
    /// Absolute path to the model's cache directory.
    pub dir: PathBuf,
    /// Total bytes of weight files in the directory.
    pub total_bytes: u64,
    /// Distinct quantization labels inferred from filenames (best-effort).
    pub quants: Vec<String>,
    /// Weight file names present (sorted).
    pub files: Vec<String>,
}

/// Map an HF model id to its cache subdirectory name. Mirrors the sanitizer the
/// deploy pipeline uses (`deploy::sanitize_dir`) so lookups are deterministic.
pub fn sanitize_model_dir(model_id: &str) -> String {
    model_id
        .chars()
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

fn is_weight_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| WEIGHT_EXTS.iter().any(|w| e.eq_ignore_ascii_case(w)))
        .unwrap_or(false)
}

/// Best-effort quantization label from a weight filename, e.g.
/// `model-Q4_K_M.gguf` → `Q4_K_M`, `x-IQ4_XS.gguf` → `IQ4_XS`,
/// `model.fp16.safetensors` → `FP16`. Returns `None` when nothing matches.
fn quant_label_from_filename(name: &str) -> Option<String> {
    let stem = name.rsplit_once('.').map(|(s, _)| s).unwrap_or(name);
    let upper: Vec<char> = stem.to_uppercase().chars().collect();
    let n = upper.len();
    // GGUF-style K/I-quants: a run starting at `Q<digit>` or `IQ`.
    let mut i = 0;
    while i < n {
        let c = upper[i];
        let starts = (c == 'Q' && i + 1 < n && upper[i + 1].is_ascii_digit())
            || (c == 'I' && i + 1 < n && upper[i + 1] == 'Q');
        if starts {
            let mut j = i;
            while j < n && (upper[j].is_ascii_alphanumeric() || upper[j] == '_') {
                j += 1;
            }
            // Trim a trailing underscore left by a shard separator.
            let mut end = j;
            while end > i && upper[end - 1] == '_' {
                end -= 1;
            }
            let token: String = upper[i..end].iter().collect();
            if token.len() >= 2 && token.chars().any(|c| c.is_ascii_digit()) {
                return Some(token);
            }
        }
        i += 1;
    }
    // Non-GGUF quant families / precisions.
    let flat: String = upper.iter().collect();
    for fam in ["FP16", "BF16", "F16", "F32", "AWQ", "GPTQ", "INT8", "INT4"] {
        if flat.contains(fam) {
            return Some((*fam).to_string());
        }
    }
    None
}

fn read_marker(dir: &Path) -> Option<String> {
    let s = std::fs::read_to_string(dir.join(MODEL_ID_MARKER)).ok()?;
    let s = s.trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// Write the model-id marker into a downloaded model's directory (best-effort;
/// called by the deploy pipeline right after creating the directory).
pub fn write_marker(dir: &Path, model_id: &str) {
    let _ = std::fs::write(dir.join(MODEL_ID_MARKER), model_id);
}

/// Scan one model directory, returning a [`DownloadedModel`] when it holds at
/// least one non-empty weight file. `.part` files (partial downloads) and empty
/// files are ignored so a half-finished download does not read as present.
fn scan_dir(dir: &Path) -> Option<DownloadedModel> {
    let mut total_bytes = 0u64;
    let mut files = Vec::new();
    let mut quants = Vec::new();
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if !path.is_file() || !is_weight_file(&path) {
            continue;
        }
        let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
        if len == 0 {
            continue;
        }
        total_bytes += len;
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            files.push(name.to_string());
            if let Some(q) = quant_label_from_filename(name) {
                if !quants.contains(&q) {
                    quants.push(q);
                }
            }
        }
    }
    if files.is_empty() {
        return None;
    }
    files.sort();
    let model_id = read_marker(dir).unwrap_or_else(|| {
        dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string()
    });
    Some(DownloadedModel {
        model_id,
        dir: dir.to_path_buf(),
        total_bytes,
        quants,
        files,
    })
}

/// Every model with weights on disk under `models_dir`, sorted by id.
pub fn list_downloaded(models_dir: &Path) -> Vec<DownloadedModel> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(models_dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(m) = scan_dir(&path) {
                out.push(m);
            }
        }
    }
    out.sort_by_key(|m| m.model_id.to_lowercase());
    out
}

/// Look up one model by id. Deterministic: sanitizes the id to a directory name
/// and scans it, so it works even without a marker file.
pub fn find_downloaded(models_dir: &Path, model_id: &str) -> Option<DownloadedModel> {
    let dir = models_dir.join(sanitize_model_dir(model_id));
    scan_dir(&dir).map(|mut m| {
        m.model_id = model_id.to_string();
        m
    })
}

/// True when weights for `model_id` are present on disk.
pub fn is_downloaded(models_dir: &Path, model_id: &str) -> bool {
    find_downloaded(models_dir, model_id).is_some()
}

/// Delete a downloaded model's directory, returning the number of bytes freed.
/// Refuses to touch anything outside `models_dir`.
pub fn delete_downloaded(models_dir: &Path, model_id: &str) -> Result<u64, LocalCodeError> {
    // Prefer the deterministic sanitized path; if the id only round-trips via a
    // marker file (older download with a differently-shaped dir name), fall back
    // to a full scan to find the matching directory.
    let direct = models_dir.join(sanitize_model_dir(model_id));
    let dir = if direct.is_dir() {
        direct
    } else {
        list_downloaded(models_dir)
            .into_iter()
            .find(|m| m.model_id == model_id)
            .map(|m| m.dir)
            .ok_or_else(|| {
                LocalCodeError::new(
                    ErrorCode::IoError,
                    format!("No downloaded model '{model_id}' in the cache"),
                )
            })?
    };

    // Safety: only ever remove a subdirectory of the models cache.
    let root = models_dir
        .canonicalize()
        .unwrap_or_else(|_| models_dir.to_path_buf());
    let target = dir.canonicalize().unwrap_or_else(|_| dir.clone());
    if target == root || !target.starts_with(&root) {
        return Err(LocalCodeError::new(
            ErrorCode::IoError,
            "Refusing to delete a path outside the models cache",
        )
        .with_cause(format!("target: {}", target.display())));
    }

    let freed = scan_dir(&dir).map(|m| m.total_bytes).unwrap_or(0);
    std::fs::remove_dir_all(&dir).map_err(|e| {
        LocalCodeError::from(e)
            .with_source("models_store", "delete_downloaded")
            .with_hint(format!("Delete manually: {}", dir.display()))
    })?;
    Ok(freed)
}

/// Human-readable size like `4.2 GB` / `812 MB` for UI and tool output.
pub fn human_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * KB;
    const GB: f64 = 1024.0 * MB;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.0} MB", b / MB)
    } else if b >= KB {
        format!("{:.0} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write_file(dir: &Path, name: &str, bytes: usize) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join(name), vec![0u8; bytes]).unwrap();
    }

    #[test]
    fn sanitize_matches_deploy_convention() {
        assert_eq!(sanitize_model_dir("org/model"), "org_model");
        assert_eq!(sanitize_model_dir("Qwen/Qwen2.5-Coder-7B"), "Qwen_Qwen2.5-Coder-7B");
    }

    #[test]
    fn quant_label_extraction() {
        assert_eq!(quant_label_from_filename("model-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_label_from_filename("x-IQ4_XS.gguf").as_deref(), Some("IQ4_XS"));
        assert_eq!(
            quant_label_from_filename("m-Q6_K-00001-of-00002.gguf").as_deref(),
            Some("Q6_K")
        );
        assert_eq!(quant_label_from_filename("model.fp16.safetensors").as_deref(), Some("FP16"));
        assert_eq!(quant_label_from_filename("readme.gguf"), None);
    }

    #[test]
    fn list_find_and_delete() {
        let root = tempdir().unwrap();
        let models = root.path();
        let d = models.join(sanitize_model_dir("org/coder-7b"));
        write_file(&d, "coder-Q4_K_M.gguf", 2048);
        write_file(&d, "coder-Q4_K_M.gguf.part", 10); // ignored (not a weight ext)
        write_marker(&d, "org/coder-7b");

        // Empty / partial-only dir is not "downloaded".
        let empty = models.join("empty");
        write_file(&empty, "weights.gguf", 0);

        let all = list_downloaded(models);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].model_id, "org/coder-7b");
        assert_eq!(all[0].quants, vec!["Q4_K_M".to_string()]);
        assert_eq!(all[0].total_bytes, 2048);

        assert!(is_downloaded(models, "org/coder-7b"));
        assert!(!is_downloaded(models, "org/other"));
        let found = find_downloaded(models, "org/coder-7b").unwrap();
        assert_eq!(found.model_id, "org/coder-7b");

        let freed = delete_downloaded(models, "org/coder-7b").unwrap();
        assert_eq!(freed, 2048);
        assert!(!is_downloaded(models, "org/coder-7b"));
    }

    #[test]
    fn human_size_formats() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(2 * 1024 * 1024), "2 MB");
        assert_eq!(human_size(3 * 1024 * 1024 * 1024), "3.0 GB");
    }
}
