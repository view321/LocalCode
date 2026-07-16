//! Backend smoke tests.
//!
//! Verify an installed backend can actually *run*, not merely that a binary
//! exists — catching a broken install (the vLLM op-registration crash, a missing
//! `libcudart`, a dead Ollama service) before the user hits it mid-deploy. Each
//! probe is bounded by a timeout so a slow import can never hang the doctor or
//! the UI, and every failure is fed through [`diagnose`] so the surface shows a
//! root cause plus (when available) an automatic fix.

use crate::diagnose::{diagnose, Diagnosis};
use crate::install::discover_python;
use crate::{probe_client, BackendKind, DetectReport};
use localcode_core::config::Config;
use localcode_core::error::{ErrorCode, LocalCodeError};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Ceiling for any single probe. `import vllm` legitimately takes 10–20s cold;
/// anything past this is treated as a (diagnosable) failure, never a hang.
const PROBE_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmokeReport {
    pub kind: BackendKind,
    pub ok: bool,
    /// What the probe did, e.g. `python -c "import vllm"`.
    pub checked: String,
    /// Captured output tail (empty on success).
    pub output: String,
    /// Root-cause diagnosis when the probe failed and something was recognized.
    pub diagnosis: Option<Diagnosis>,
}

impl SmokeReport {
    fn ok(kind: BackendKind, checked: impl Into<String>) -> Self {
        Self {
            kind,
            ok: true,
            checked: checked.into(),
            output: String::new(),
            diagnosis: None,
        }
    }

    /// Build a failing report and diagnose it from a synthetic error carrying
    /// the captured output (the same shape the deploy path produces).
    fn fail(kind: BackendKind, checked: impl Into<String>, output: String) -> Self {
        let err = LocalCodeError::new(ErrorCode::BackendStartFailed, "smoke test failed")
            .with_cause(output.clone());
        let diagnosis = diagnose(kind, &err);
        Self {
            kind,
            ok: false,
            checked: checked.into(),
            output,
            diagnosis,
        }
    }
}

/// Smoke-test one backend given its detect report and config. Bounded by a
/// per-probe timeout; callers run these concurrently over installed backends.
pub async fn smoke_test(report: &DetectReport, cfg: &Config) -> SmokeReport {
    match report.kind {
        BackendKind::Vllm => python_import_probe(BackendKind::Vllm, "vllm", vllm_python(cfg)).await,
        BackendKind::Sglang => {
            python_import_probe(BackendKind::Sglang, "sglang", sglang_python(cfg)).await
        }
        BackendKind::Ollama => ollama_probe(cfg).await,
        BackendKind::LlamaCpp => llamacpp_probe(report, cfg).await,
        BackendKind::Colibri => {
            colibri_probe(report, &cfg.backends.colibri.bin, BackendKind::Colibri).await
        }
        BackendKind::ColibriHy3 => {
            colibri_probe(report, &cfg.backends.colibri_hy3.bin, BackendKind::ColibriHy3).await
        }
    }
}

/// `<python> -c "import <module>"` with a timeout, capturing stderr. This is what
/// catches the op-registration / torch-mismatch crash before a deploy.
async fn python_import_probe(
    kind: BackendKind,
    module: &str,
    python: Option<PathBuf>,
) -> SmokeReport {
    let checked = format!("python -c \"import {module}\"");
    let Some(py) = python else {
        return SmokeReport::fail(
            kind,
            checked,
            format!("No Python interpreter found to import {module}"),
        );
    };
    let run = tokio::process::Command::new(&py)
        .args(["-c", &format!("import {module}")])
        .kill_on_drop(true)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();
    match tokio::time::timeout(PROBE_TIMEOUT, run).await {
        Ok(Ok(out)) if out.status.success() => SmokeReport::ok(kind, checked),
        Ok(Ok(out)) => {
            SmokeReport::fail(kind, checked, tail_of(&String::from_utf8_lossy(&out.stderr), 40))
        }
        Ok(Err(e)) => SmokeReport::fail(kind, checked, e.to_string()),
        Err(_) => SmokeReport::fail(
            kind,
            format!("{checked} (timed out)"),
            format!("import {module} did not finish within {}s", PROBE_TIMEOUT.as_secs()),
        ),
    }
}

async fn ollama_probe(cfg: &Config) -> SmokeReport {
    let url = format!(
        "{}/api/tags",
        cfg.backends.ollama.base_url.trim_end_matches('/')
    );
    let checked = format!("GET {url}");
    let client = probe_client();
    match tokio::time::timeout(PROBE_TIMEOUT, client.get(&url).send()).await {
        Ok(Ok(r)) if r.status().is_success() => SmokeReport::ok(BackendKind::Ollama, checked),
        Ok(Ok(r)) => SmokeReport::fail(BackendKind::Ollama, checked, format!("HTTP {}", r.status())),
        Ok(Err(e)) => {
            SmokeReport::fail(BackendKind::Ollama, checked, format!("connection refused: {e}"))
        }
        Err(_) => SmokeReport::fail(BackendKind::Ollama, checked, "request timed out".into()),
    }
}

async fn llamacpp_probe(report: &DetectReport, cfg: &Config) -> SmokeReport {
    let bin = report
        .binary_path
        .clone()
        .unwrap_or_else(|| cfg.backends.llamacpp.bin.clone());
    let checked = format!("{bin} --version");
    let run = tokio::process::Command::new(&bin)
        .arg("--version")
        .kill_on_drop(true)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();
    match tokio::time::timeout(PROBE_TIMEOUT, run).await {
        Ok(Ok(out)) if out.status.success() => SmokeReport::ok(BackendKind::LlamaCpp, checked),
        Ok(Ok(out)) => {
            // Missing shared libs surface here; combine both streams to catch it.
            let mut combined = String::from_utf8_lossy(&out.stdout).into_owned();
            combined.push_str(&String::from_utf8_lossy(&out.stderr));
            SmokeReport::fail(BackendKind::LlamaCpp, checked, tail_of(&combined, 40))
        }
        Ok(Err(e)) => SmokeReport::fail(BackendKind::LlamaCpp, checked, e.to_string()),
        Err(_) => SmokeReport::fail(BackendKind::LlamaCpp, checked, "process timed out".into()),
    }
}

/// `coli --help` proves the binary runs at all — catching a missing libgomp
/// (OpenMP) or an AVX build on a non-AVX CPU (SIGILL) without loading a model.
async fn colibri_probe(report: &DetectReport, cfg_bin: &str, kind: BackendKind) -> SmokeReport {
    let bin = report
        .binary_path
        .clone()
        .unwrap_or_else(|| cfg_bin.to_string());
    let checked = format!("{bin} --help");
    let run = tokio::process::Command::new(&bin)
        .arg("--help")
        .kill_on_drop(true)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();
    match tokio::time::timeout(PROBE_TIMEOUT, run).await {
        Ok(Ok(out)) if out.status.success() => SmokeReport::ok(kind, checked),
        Ok(Ok(out)) => {
            // Loader errors and SIGILL notes land on either stream.
            let mut combined = String::from_utf8_lossy(&out.stdout).into_owned();
            combined.push_str(&String::from_utf8_lossy(&out.stderr));
            SmokeReport::fail(kind, checked, tail_of(&combined, 40))
        }
        Ok(Err(e)) => SmokeReport::fail(kind, checked, e.to_string()),
        Err(_) => SmokeReport::fail(kind, checked, "process timed out".into()),
    }
}

/// The interpreter that owns the configured vLLM. After a venv repoint,
/// `cfg.vllm.bin` is `<venv>/bin/vllm`, whose sibling `python` is the venv's — so
/// we test the same environment that will actually serve.
fn vllm_python(cfg: &Config) -> Option<PathBuf> {
    sibling_python(&cfg.backends.vllm.bin).or_else(discover_python)
}

/// SGLang's configured bin, after a venv repoint, *is* the venv python.
fn sglang_python(cfg: &Config) -> Option<PathBuf> {
    let bin = PathBuf::from(&cfg.backends.sglang.bin);
    if is_python_path(&bin) {
        return Some(bin);
    }
    sibling_python(&cfg.backends.sglang.bin).or_else(discover_python)
}

/// If `bin` is an existing absolute path, return the `python[.exe]` next to it
/// (a venv layout). `None` otherwise (e.g. the bare `"vllm"` default).
fn sibling_python(bin: &str) -> Option<PathBuf> {
    let p = Path::new(bin);
    if !p.is_absolute() || !p.exists() {
        return None;
    }
    let py = p
        .parent()?
        .join(if cfg!(windows) { "python.exe" } else { "python" });
    py.exists().then_some(py)
}

fn is_python_path(p: &Path) -> bool {
    p.is_absolute()
        && p.exists()
        && p.file_stem()
            .map(|s| s.to_string_lossy().to_lowercase().contains("python"))
            .unwrap_or(false)
}

/// Keep the last `n` lines — a broken import can print a long traceback.
fn tail_of(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_probe_carries_diagnosis_from_output() {
        // A registration-bug stderr must produce a High-confidence diagnosis
        // with the clean-venv repair, exactly as the deploy path would.
        let out = "RuntimeError: Tried to register an operator \
                   (vllm::_fused_mul_mat_gguf) ... registered multiple times."
            .to_string();
        let r = SmokeReport::fail(BackendKind::Vllm, "python -c import vllm", out);
        assert!(!r.ok);
        let dg = r.diagnosis.expect("diagnosis");
        assert_eq!(dg.repair, Some(crate::RepairIntent::CleanVenvReinstall));
    }

    #[test]
    fn tail_keeps_last_lines() {
        let s = (0..100).map(|i| i.to_string()).collect::<Vec<_>>().join("\n");
        let t = tail_of(&s, 3);
        assert_eq!(t, "97\n98\n99");
    }

    #[test]
    fn bare_bin_has_no_sibling_python() {
        assert!(sibling_python("vllm").is_none());
    }
}
