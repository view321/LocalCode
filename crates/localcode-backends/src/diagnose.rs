//! Backend failure diagnosis.
//!
//! Maps a failed backend operation (its captured output + error code) to a
//! structured root cause and an optional automated repair. Classification is
//! **pure** and string-based, so it is fully unit-testable against real
//! tracebacks; the concrete repair — which needs paths, the OS and tool
//! availability — is built from the abstract [`RepairIntent`] by
//! [`crate::install::resolve_repair`] (the same pure-planner / env-resolver
//! split as `install_plan` vs `resolve_install_plan`).
//!
//! This generalizes what used to be vLLM-only `failure_hints`. Getting the
//! diagnosis wrong is exactly how a torch/vLLM install bug gets misread as a
//! model-format problem — the regression [`classify`] is tested against.

use crate::BackendKind;
use localcode_core::error::{ErrorCode, LocalCodeError};
use serde::{Deserialize, Serialize};

/// A recognized failure mode, independent of which backend hit it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureClass {
    /// vLLM/SGLang duplicate custom-op registration — a broken or
    /// version-mismatched torch/vLLM install, **not** a model-format problem.
    VllmOpRegistration,
    /// torch and the backend were built against incompatible versions (ABI).
    TorchVersionMismatch,
    /// CUDA ran out of memory during model load.
    CudaOom,
    /// A Python module isn't importable (carries the module name).
    ModuleNotFound(String),
    /// GPU driver too old / no kernel image for this GPU architecture.
    DriverMismatch,
    /// The server process wasn't reachable / refused the connection.
    ConnectionRefused,
    /// A GGUF / model file couldn't be parsed or loaded.
    InvalidGguf,
    /// A native shared library couldn't be loaded (carries the lib name).
    MissingSharedLib(String),
    /// The requested port was already bound.
    PortInUse,
    /// Nothing specific matched.
    Generic,
}

/// How sure we are of a diagnosis. Surfaced so the UI can hedge low-confidence
/// guesses instead of presenting them as fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    High,
    Medium,
    Low,
}

/// An abstract repair. Resolved into concrete commands by
/// [`crate::install::resolve_repair`], which knows the OS and app paths.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairIntent {
    /// Build a clean managed venv, install the backend there, repoint at it.
    /// The package is derived from the backend kind by the resolver.
    CleanVenvReinstall,
    /// Start the local Ollama service (may require sudo on Linux).
    StartOllamaService,
    /// Reinstall a Homebrew formula (e.g. a llama.cpp build with fresh libs).
    ReinstallFormula(String),
}

/// A structured root-cause diagnosis with an optional automated fix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnosis {
    pub backend: BackendKind,
    /// One-line root cause, shown as the headline.
    pub summary: String,
    /// Why this happened / what it means.
    pub explanation: String,
    pub confidence: Confidence,
    /// The automated fix, if one is safe and applicable. `None` ⇒ no Fix button.
    pub repair: Option<RepairIntent>,
    /// Steps the user can take themselves — always shown alongside the fix.
    pub manual_steps: Vec<String>,
}

/// True when any needle appears in the already-lowercased haystack.
fn any(low: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| low.contains(n))
}

/// Classify raw backend output into a [`FailureClass`]. Ordered most-specific
/// first; the first matching group wins. Pure and case-insensitive.
pub fn classify(text: &str) -> FailureClass {
    let low = text.to_lowercase();

    // The motivating bug: fires while vLLM imports its quantization ops at
    // startup (any model, safetensors included), so it must beat every
    // model-format guess below.
    if any(
        &low,
        &[
            "register an operator",
            "registered multiple times",
            "direct_register_custom_op",
            "_fused_mul_mat_gguf",
        ],
    ) {
        return FailureClass::VllmOpRegistration;
    }
    // A native lib that won't load — check before the generic import errors so
    // a missing libcudart isn't misread as a Python packaging problem.
    if any(
        &low,
        &[
            "error while loading shared libraries",
            "cannot open shared object",
            "dll load failed",
        ],
    ) {
        return FailureClass::MissingSharedLib(extract_shared_lib(&low));
    }
    if any(&low, &["no module named", "modulenotfounderror"]) {
        return FailureClass::ModuleNotFound(extract_module(text));
    }
    // ABI/version skew between torch and the backend's compiled extension.
    if any(
        &low,
        &[
            "undefined symbol",
            "abi mismatch",
            "compiled with a different version",
            "was compiled against",
        ],
    ) {
        return FailureClass::TorchVersionMismatch;
    }
    // Driver/toolkit before OOM — both mention CUDA, but a driver mismatch is
    // the more actionable (and more specific) cause.
    if any(
        &low,
        &[
            "no kernel image is available",
            "cuda driver version is insufficient",
            "forward compatibility was attempted",
            "no cuda-capable device is detected",
        ],
    ) {
        return FailureClass::DriverMismatch;
    }
    if any(
        &low,
        &["out of memory", "outofmemoryerror", "cuda out of memory"],
    ) {
        return FailureClass::CudaOom;
    }
    if any(
        &low,
        &[
            "invalid gguf",
            "gguf_init",
            "unknown model architecture",
            "wrong number of tensors",
        ],
    ) {
        return FailureClass::InvalidGguf;
    }
    // Windows: "only one usage of each socket address"; Unix: "address already
    // in use"; both mean the port is taken.
    if any(
        &low,
        &[
            "address already in use",
            "eaddrinuse",
            "only one usage of each socket",
        ],
    ) {
        return FailureClass::PortInUse;
    }
    if any(
        &low,
        &[
            "connection refused",
            "actively refused",
            "cannot reach",
            "failed to connect",
            "connection reset",
        ],
    ) {
        return FailureClass::ConnectionRefused;
    }
    FailureClass::Generic
}

/// Pull the module name out of a `No module named 'x'` / `ModuleNotFoundError:
/// No module named "x"` line. Falls back to empty when it can't be isolated.
fn extract_module(text: &str) -> String {
    let lower = text.to_lowercase();
    let Some(pos) = lower.find("no module named") else {
        return String::new();
    };
    let rest = &text[pos + "no module named".len()..];
    rest.split(['\'', '"'])
        .nth(1)
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Pull a `lib*.so` / `*.dll` token out of a loader error. Best-effort.
fn extract_shared_lib(low: &str) -> String {
    low.split(|c: char| c.is_whitespace() || c == ':' || c == '(' || c == ')' || c == '\'')
        .find(|tok| tok.contains(".so") || tok.ends_with(".dll"))
        .unwrap_or("")
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '.' && c != '_' && c != '-')
        .to_string()
}

impl FailureClass {
    /// vLLM-oriented one-line hints for the deploy error surface. Recognized
    /// install bugs get targeted advice; everything else returns empty so the
    /// caller can add its own generic hints.
    pub fn hints(&self) -> Vec<&'static str> {
        match self {
            FailureClass::VllmOpRegistration | FailureClass::TorchVersionMismatch => vec![
                "vLLM crashed importing its CUDA ops — a broken or version-mismatched install, not your model",
                "Check versions: python -c \"import torch, vllm; print(torch.__version__, vllm.__version__)\"",
                "Reinstall vLLM in a clean venv so it pulls a matching torch, or run the official vLLM Docker image",
            ],
            _ => vec![],
        }
    }
}

/// Diagnose a failed backend operation. Reads the captured output from the
/// error's `causes` (that's where the backend adapters push the process tail —
/// see `vllm.rs` `deploy`) plus the message, and keys off the error code for
/// text-less failures. Returns `None` when nothing actionable is recognized, so
/// the UI shows only the error's own hints and no misleading Fix button.
pub fn diagnose(kind: BackendKind, err: &LocalCodeError) -> Option<Diagnosis> {
    let mut text = err.message.clone();
    for c in &err.causes {
        text.push('\n');
        text.push_str(c);
    }
    let mut class = classify(&text);

    // Some failures carry no useful output (a bare connection error, a port
    // bind) — recover the class from the stable error code.
    if class == FailureClass::Generic {
        match err.code {
            ErrorCode::BackendPortInUse => class = FailureClass::PortInUse,
            ErrorCode::BackendNotReady if kind == BackendKind::Ollama => {
                class = FailureClass::ConnectionRefused
            }
            _ => {}
        }
    }

    build_diagnosis(kind, class)
}

/// Map a (backend, class) pair to a Diagnosis. Returns `None` when the class
/// isn't meaningful for that backend or offers nothing beyond the raw error.
fn build_diagnosis(kind: BackendKind, class: FailureClass) -> Option<Diagnosis> {
    use BackendKind::*;
    let d = |summary: &str,
             explanation: &str,
             confidence: Confidence,
             repair: Option<RepairIntent>,
             manual: &[&str]| {
        Some(Diagnosis {
            backend: kind,
            summary: summary.to_string(),
            explanation: explanation.to_string(),
            confidence,
            repair,
            manual_steps: manual.iter().map(|s| s.to_string()).collect(),
        })
    };

    match (kind, class) {
        // ---- vLLM / SGLang (pip/venv backends) ----
        (Vllm | Sglang, FailureClass::VllmOpRegistration)
        | (Vllm | Sglang, FailureClass::TorchVersionMismatch) => d(
            &format!("Broken or version-mismatched {} install", kind.as_str()),
            "The server crashed while importing its CUDA/quantization ops — torch and \
             the backend are mismatched, or a duplicate/partial install is on the path. \
             This is not a problem with your model.",
            Confidence::High,
            Some(RepairIntent::CleanVenvReinstall),
            &[
                "Check versions: python -c \"import torch, vllm; print(torch.__version__, vllm.__version__)\"",
                "Or run the official vLLM Docker image instead of a host install",
            ],
        ),
        (Vllm | Sglang, FailureClass::ModuleNotFound(m)) => d(
            &format!("{} isn't installed in the active Python", kind.as_str()),
            &format!("Importing `{m}` failed — the backend package (or one of its \
                      dependencies) is missing from the interpreter LocalCode is using."),
            Confidence::High,
            Some(RepairIntent::CleanVenvReinstall),
            &["Install into a clean venv so dependencies resolve together"],
        ),
        (Vllm | Sglang, FailureClass::CudaOom) => d(
            "GPU ran out of memory at startup",
            "The model plus its KV cache didn't fit in VRAM while loading. This is a \
             sizing choice, not a broken install.",
            Confidence::High,
            None,
            &[
                "Lower the context with /context (bounds the KV cache)",
                "Pick a smaller quantization, or a smaller model",
            ],
        ),
        (Vllm | Sglang, FailureClass::DriverMismatch) => d(
            "GPU driver / CUDA toolkit mismatch",
            "The GPU driver is older than the CUDA build the backend was compiled \
             against, or no compatible device was found.",
            Confidence::Medium,
            None,
            &[
                "Check the driver with nvidia-smi",
                "Update the NVIDIA driver to match the backend's CUDA version",
            ],
        ),

        // ---- Ollama ----
        (Ollama, FailureClass::ConnectionRefused) => d(
            "Ollama service isn't running",
            "The Ollama binary is present but its local API didn't answer — the \
             background service needs to be started.",
            Confidence::High,
            Some(RepairIntent::StartOllamaService),
            &[
                "Run: ollama serve",
                "Verify base_url (default http://127.0.0.1:11434)",
            ],
        ),

        // ---- llama.cpp ----
        (LlamaCpp, FailureClass::MissingSharedLib(lib)) => d(
            &if lib.is_empty() {
                "A required shared library is missing".to_string()
            } else {
                format!("Missing shared library: {lib}")
            },
            "llama-server couldn't load a native dependency (often the CUDA runtime). \
             A matching build or the runtime library needs to be present.",
            Confidence::Medium,
            Some(RepairIntent::ReinstallFormula("llama.cpp".into())),
            &["Install the CUDA runtime, or use a CPU build of llama.cpp"],
        ),
        (LlamaCpp, FailureClass::InvalidGguf) => d(
            "Model file couldn't be loaded",
            "The GGUF failed to parse — it may be truncated, an unsupported \
             architecture, or a partial download.",
            Confidence::Medium,
            None,
            &[
                "Re-download the model (a partial file won't load)",
                "Try a different quantization",
            ],
        ),

        // ---- Any backend: a taken port is self-explanatory but worth naming. ----
        (_, FailureClass::PortInUse) => d(
            "Port already in use",
            "Another process is bound to the port this backend wants.",
            Confidence::High,
            None,
            &["Stop the other process, or pick a different port in Deploy"],
        ),

        // Nothing actionable — let the raw error's hints speak.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact shape of the screenshot traceback this whole feature exists
    /// for. Must classify as an install bug and never as a format problem.
    const REGISTRATION_TRACE: &str = "RuntimeError: Tried to register an operator \
        (vllm::_fused_mul_mat_gguf(Tensor x, Tensor qweight, SymInt qweight_type) -> Tensor) \
        with the same name and overload name multiple times. Duplicate registration: \
        registered at /home/user/.local/lib/python3.12/site-packages/vllm/utils/torch_utils.py:928";

    #[test]
    fn op_registration_classifies_and_is_not_format() {
        assert_eq!(classify(REGISTRATION_TRACE), FailureClass::VllmOpRegistration);
        let hints = FailureClass::VllmOpRegistration.hints();
        assert!(hints.iter().any(|h| h.contains("version-mismatched")));
        assert!(
            !hints.iter().any(|h| h.contains("GGUF-only")),
            "an install bug must never be reported as a format problem"
        );
    }

    #[test]
    fn op_registration_diagnosis_offers_clean_venv() {
        let err = LocalCodeError::new(ErrorCode::BackendStartFailed, "vLLM exited: exit status: 1")
            .with_cause(format!("vLLM output:\n{REGISTRATION_TRACE}"));
        let dg = diagnose(BackendKind::Vllm, &err).expect("a diagnosis");
        assert_eq!(dg.confidence, Confidence::High);
        assert_eq!(dg.repair, Some(RepairIntent::CleanVenvReinstall));
        assert!(dg.summary.to_lowercase().contains("mismatch"));
    }

    #[test]
    fn diagnose_reads_output_from_causes() {
        // The backend pushes the process tail into causes, not the message.
        let err = LocalCodeError::new(ErrorCode::BackendStartFailed, "vLLM exited: exit status: 1")
            .with_cause("vLLM stopped before serving the model")
            .with_cause(format!("vLLM output:\n{REGISTRATION_TRACE}"));
        assert!(diagnose(BackendKind::Vllm, &err).is_some());
    }

    #[test]
    fn generic_vllm_exit_keeps_no_repair() {
        let err = LocalCodeError::new(
            ErrorCode::BackendStartFailed,
            "ValueError: model architecture not supported",
        );
        // classify → Generic; no false Fix button.
        assert_eq!(classify(&err.message), FailureClass::Generic);
        assert!(diagnose(BackendKind::Vllm, &err).is_none());
    }

    #[test]
    fn module_not_found_captures_name_and_offers_repair() {
        assert_eq!(
            classify("ModuleNotFoundError: No module named 'vllm'"),
            FailureClass::ModuleNotFound("vllm".into())
        );
        let err = LocalCodeError::new(ErrorCode::BackendStartFailed, "boot failed")
            .with_cause("No module named 'sglang'");
        let dg = diagnose(BackendKind::Sglang, &err).unwrap();
        assert_eq!(dg.repair, Some(RepairIntent::CleanVenvReinstall));
    }

    #[test]
    fn cuda_oom_diagnosed_but_not_auto_fixable() {
        let err = LocalCodeError::new(ErrorCode::BackendStartFailed, "boot failed")
            .with_cause("torch.cuda.OutOfMemoryError: CUDA out of memory");
        let dg = diagnose(BackendKind::Vllm, &err).unwrap();
        assert_eq!(dg.confidence, Confidence::High);
        assert!(dg.repair.is_none(), "OOM is a user sizing choice, not a repair");
    }

    #[test]
    fn ollama_unreachable_offers_service_start() {
        // No output text — driven purely by the error code + backend.
        let err = LocalCodeError::new(ErrorCode::BackendNotReady, "Ollama is not ready");
        let dg = diagnose(BackendKind::Ollama, &err).unwrap();
        assert_eq!(dg.repair, Some(RepairIntent::StartOllamaService));
    }

    #[test]
    fn port_in_use_from_code_without_text() {
        let err = LocalCodeError::new(ErrorCode::BackendPortInUse, "Port 8000 is already in use");
        let dg = diagnose(BackendKind::Vllm, &err).unwrap();
        assert!(dg.repair.is_none());
        assert!(dg.summary.to_lowercase().contains("port"));
    }

    #[test]
    fn missing_shared_lib_extracts_name() {
        let c = classify("llama-server: error while loading shared libraries: libcudart.so.12: cannot open shared object file");
        match c {
            FailureClass::MissingSharedLib(lib) => assert!(lib.contains("libcudart.so")),
            other => panic!("expected MissingSharedLib, got {other:?}"),
        }
    }

    #[test]
    fn unknown_backend_class_pairs_return_none() {
        // A GGUF parse error isn't meaningful for vLLM (guarded earlier).
        assert!(build_diagnosis(BackendKind::Ollama, FailureClass::InvalidGguf).is_none());
    }
}
