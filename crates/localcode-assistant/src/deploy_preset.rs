//! Recommend a full deploy preset (backend + context + flags) from a model's
//! metadata and card — the "assistant reads the card and sets adequate defaults"
//! path.
//!
//! [`extract_deploy_hints`](crate::deploy_hints::extract_deploy_hints) already
//! pulls backend flags out of a card, but it does so for a *fixed* backend and
//! never touches the backend choice or the context length. This module adds the
//! two decisions that make a browsed model deploy-ready without hand-tuning:
//!
//! 1. **The correct backend for the weight format.** GGUF is a llama.cpp format;
//!    vLLM/SGLang load safetensors/AWQ/GPTQ checkpoints. Picking the backend
//!    from the format is what stops a GGUF model from being sent to vLLM (which
//!    would crash at load — see `deploy::is_gguf_on_transformers_backend`).
//! 2. **A fitting context length.** Read the model's native context from the
//!    card / model name so the default isn't stuck at 8k for a 128k model.
//!
//! Everything here is pure and unit-tested; the caller (TUI) applies the result
//! and the user can still override every field.

use crate::deploy_hints::extract_deploy_hints;
use localcode_backends::{BackendKind, DeployTuning, DEFAULT_DEPLOY_CTX};
use regex::Regex;

/// Upper bound for any auto-selected context. Keeps a "1M context" card from
/// seeding an absurd default; the user can still type a larger value by hand.
const MAX_PRESET_CTX: u32 = 131_072;

/// llama.cpp `--n-gpu-layers` value that offloads every layer to the GPU. The
/// GGUF cards in the wild use `99`, and every llama.cpp build treats
/// "layers ≥ model layers" as "all", so it's the safe portable sentinel.
const LLAMACPP_OFFLOAD_ALL: i32 = 99;

/// Which family of runtimes can serve a model, inferred from its weight format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// GGUF — served by llama.cpp / Ollama.
    Gguf,
    /// Hugging Face transformers checkpoint (safetensors/bin), including AWQ and
    /// GPTQ — served by vLLM / SGLang.
    Transformers,
    /// Couldn't tell from the label or filenames.
    Unknown,
}

/// Classify a quantization from its label and/or weight filenames. Filenames are
/// definitive; the label is the fallback for an id deployed without file
/// metadata. The GGUF label test mirrors `deploy::is_gguf_on_transformers_backend`
/// so browsing and the deploy-time guard agree on what "GGUF" means.
pub fn classify_weight_format(quant_label: Option<&str>, weight_files: &[String]) -> WeightFormat {
    if weight_files
        .iter()
        .any(|f| f.to_lowercase().ends_with(".gguf"))
    {
        return WeightFormat::Gguf;
    }
    if weight_files.iter().any(|f| {
        let l = f.to_lowercase();
        l.ends_with(".safetensors")
            || l.ends_with(".bin")
            || l.ends_with(".pt")
            || l.ends_with(".pth")
    }) {
        return WeightFormat::Transformers;
    }
    match quant_label {
        Some(q) => {
            let u = q.to_uppercase();
            // GGUF K-quants (Q4_K_M, Q8_0, …) and I-quants (IQ4_XS, …). AWQ/GPTQ/
            // EXL2/FP16/INT8 are transformers-servable labels and don't start
            // with Q/IQ.
            if u.contains("GGUF") || u.starts_with("IQ") || u.starts_with('Q') {
                WeightFormat::Gguf
            } else if u.contains("AWQ")
                || u.contains("GPTQ")
                || u.contains("EXL2")
                || u.contains("SAFETENSOR")
                || u.contains("FP16")
                || u.contains("BF16")
                || u.contains("FP32")
                || u.contains("F16")
                || u.contains("INT8")
                || u.contains("INT4")
            {
                WeightFormat::Transformers
            } else {
                WeightFormat::Unknown
            }
        }
        None => WeightFormat::Unknown,
    }
}

/// Backends able to serve a format, most-preferred first.
fn compatible_backends(fmt: WeightFormat) -> &'static [BackendKind] {
    match fmt {
        WeightFormat::Gguf => &[BackendKind::LlamaCpp, BackendKind::Ollama],
        WeightFormat::Transformers => &[BackendKind::Vllm, BackendKind::Sglang],
        WeightFormat::Unknown => &[],
    }
}

/// True when `backend` can actually load a model of this format. Used to decide
/// whether an already-selected backend needs correcting (e.g. GGUF on vLLM).
pub fn backend_supports(backend: BackendKind, fmt: WeightFormat) -> bool {
    match fmt {
        WeightFormat::Unknown => true, // don't fight the user when we can't tell
        _ => compatible_backends(fmt).contains(&backend),
    }
}

/// Pick the backend best suited to a weight format, preferring one that's
/// already installed. Returns the canonical backend even when nothing compatible
/// is installed, so the panel shows the right choice and the user gets the usual
/// install prompt; returns `fallback` only when the format is unknown.
pub fn recommend_backend(
    fmt: WeightFormat,
    installed: &[BackendKind],
    fallback: BackendKind,
) -> BackendKind {
    let compat = compatible_backends(fmt);
    match compat.first() {
        None => fallback,
        Some(&canonical) => compat
            .iter()
            .find(|b| installed.contains(b))
            .copied()
            .unwrap_or(canonical),
    }
}

/// Everything the preset needs, borrowed from the browsed model's detail.
pub struct PresetInput<'a> {
    pub model_id: &'a str,
    pub selected_quant: Option<&'a str>,
    pub weight_files: &'a [String],
    pub tags: &'a [String],
    pub card_markdown: Option<&'a str>,
    /// Backends detected as installed (registry detect reports). Recommendation
    /// prefers these but never restricts itself to them.
    pub installed_backends: &'a [BackendKind],
    /// The user's configured default backend, used when the format is unknown.
    pub configured_default: BackendKind,
    /// Whether a GPU was detected — gates the llama.cpp "offload all layers"
    /// default (pointless on a CPU-only host).
    pub has_gpu: bool,
}

/// A full set of adequate deploy defaults derived from a model card.
#[derive(Debug, Clone)]
pub struct DeployPreset {
    /// The backend the model should deploy on.
    pub backend: BackendKind,
    /// The model's native/fitting context, when we could determine one. `None`
    /// means "keep the current default" — the caller does not lower context.
    pub desired_context: Option<u32>,
    /// Per-backend launch tuning (GPU fraction / tensor-parallel / GPU layers /
    /// extra flags) for `backend`.
    pub tuning: DeployTuning,
    /// Human-readable notes explaining the choices (shown in the panel/status).
    pub notes: Vec<String>,
}

/// Derive adequate deploy defaults for a browsed model: the right backend for
/// its weight format, a fitting context, and the card's recommended flags.
pub fn recommend_deploy_preset(input: &PresetInput) -> DeployPreset {
    let fmt = classify_weight_format(input.selected_quant, input.weight_files);
    let backend = recommend_backend(fmt, input.installed_backends, input.configured_default);
    preset_for_backend(backend, input)
}

/// Build the preset (context + flags + notes) for an already-chosen `backend`,
/// skipping the backend recommendation. Used when the user has explicitly picked
/// a backend but we still want the card's flags and a fitting context for it.
pub fn preset_for_backend(backend: BackendKind, input: &PresetInput) -> DeployPreset {
    let fmt = classify_weight_format(input.selected_quant, input.weight_files);

    let mut notes = Vec::new();
    match fmt {
        WeightFormat::Gguf => {
            notes.push(format!("GGUF weights → {} backend", backend.as_str()));
        }
        WeightFormat::Transformers => {
            notes.push(format!(
                "safetensors checkpoint → {} backend",
                backend.as_str()
            ));
        }
        WeightFormat::Unknown => {}
    }
    if fmt != WeightFormat::Unknown && !input.installed_backends.contains(&backend) {
        notes.push(format!("{} not detected — install via /backends", backend.as_str()));
    }

    // Card flags for the *chosen* backend (overwrite=true: the preset owns these
    // fields on a fresh model; the TUI decides whether to keep user edits).
    let hints = input
        .card_markdown
        .map(|c| extract_deploy_hints(c, backend))
        .unwrap_or_default();
    let mut tuning = DeployTuning::default();
    hints.apply_to_tuning(&mut tuning, true);
    notes.extend(hints.notes.iter().cloned());

    // GGUF on a GPU: offload all layers unless the card already said otherwise —
    // without --n-gpu-layers llama.cpp keeps every layer on the CPU.
    if backend == BackendKind::LlamaCpp && input.has_gpu && tuning.gpu_layers.is_none() {
        tuning.gpu_layers = Some(LLAMACPP_OFFLOAD_ALL);
        notes.push(format!("llama.cpp: offload all layers (-ngl {LLAMACPP_OFFLOAD_ALL})"));
    }

    // Context: card recommendation (already in hints) → native context stated in
    // the card prose or the model name → leave the default alone. Only the upper
    // bound is clamped: an explicit small max (e.g. a 4k model) must be respected,
    // not raised past the model's real limit.
    let desired_context = hints
        .context_length
        .or_else(|| parse_native_context(input.card_markdown, input.model_id, input.tags))
        .map(|c| c.clamp(512, MAX_PRESET_CTX));
    if let Some(c) = desired_context {
        if !notes.iter().any(|n| n.contains("context")) {
            notes.push(format!("context {c} (model native)"));
        }
    }

    DeployPreset {
        backend,
        desired_context,
        tuning,
        notes,
    }
}

/// Best-effort read of a model's native context window from its card prose and
/// name/tags, for when the card's example commands don't spell out
/// `--max-model-len`. Returns a value in a sane range, else `None`.
pub fn parse_native_context(card: Option<&str>, model_id: &str, tags: &[String]) -> Option<u32> {
    let mut best: Option<u32> = None;
    let mut consider = |v: u32| {
        if (4_096..=1_048_576).contains(&v) {
            best = Some(best.map_or(v, |b| b.max(v)));
        }
    };

    if let Some(card) = card {
        // config.json-style: "max_position_embeddings": 32768
        if let Some(n) = capture_u32(card, r#"(?i)max_position_embeddings["'\s:=]{1,6}(\d{3,7})"#) {
            consider(n);
        }
        // Prose: "context length of 32768", "128K context window".
        if let Some(n) = capture_u32(
            card,
            r"(?i)context(?:\s+(?:length|window|size))?[^\d]{0,16}(\d[\d,]{2,})",
        ) {
            consider(n);
        }
        for k in capture_all_kmb(card, r"(?i)(\d{1,4})\s*([km])\s*(?:context|tokens?|ctx)") {
            consider(k);
        }
    }

    // Model name / tags: "…-128k", "…-1M-…". Only allow these to *raise* the
    // default (>= 8k) so a stray token can't shrink context.
    let mut names: Vec<&str> = vec![model_id];
    names.extend(tags.iter().map(|s| s.as_str()));
    for name in names {
        for k in capture_all_kmb(name, r"(?i)[-_.](\d{1,4})\s*([km])\b") {
            if k >= DEFAULT_DEPLOY_CTX {
                consider(k);
            }
        }
    }

    best
}

/// Capture every `<num><k|m>` occurrence and expand to a token count.
fn capture_all_kmb(text: &str, pat: &str) -> Vec<u32> {
    let Ok(re) = Regex::new(pat) else {
        return vec![];
    };
    re.captures_iter(text)
        .filter_map(|c| {
            let n: u64 = c.get(1)?.as_str().parse().ok()?;
            let mult: u64 = match c.get(2)?.as_str().to_lowercase().as_str() {
                "k" => 1_024,
                "m" => 1_024 * 1_024,
                _ => return None,
            };
            u32::try_from(n * mult).ok()
        })
        .collect()
}

fn capture_u32(text: &str, pat: &str) -> Option<u32> {
    let re = Regex::new(pat).ok()?;
    let c = re.captures(text)?;
    // Values in prose may carry thousands separators ("32,768").
    c.get(1)?.as_str().replace(',', "").parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn gguf_classified_from_filename() {
        assert_eq!(
            classify_weight_format(None, &s(&["model-Q4_K_M.gguf"])),
            WeightFormat::Gguf
        );
    }

    #[test]
    fn safetensors_classified_from_filename() {
        assert_eq!(
            classify_weight_format(None, &s(&["model-00001-of-2.safetensors"])),
            WeightFormat::Transformers
        );
    }

    #[test]
    fn gguf_classified_from_label_without_files() {
        assert_eq!(classify_weight_format(Some("Q4_K_M"), &[]), WeightFormat::Gguf);
        assert_eq!(classify_weight_format(Some("IQ4_XS"), &[]), WeightFormat::Gguf);
    }

    #[test]
    fn awq_gptq_are_transformers() {
        assert_eq!(classify_weight_format(Some("AWQ"), &[]), WeightFormat::Transformers);
        assert_eq!(classify_weight_format(Some("GPTQ"), &[]), WeightFormat::Transformers);
        // GPTQ contains a Q but must not be mistaken for GGUF.
        assert_ne!(classify_weight_format(Some("GPTQ"), &[]), WeightFormat::Gguf);
    }

    #[test]
    fn recommend_prefers_installed_compatible_backend() {
        // GGUF: llama.cpp is canonical, but if only Ollama is installed, use it.
        assert_eq!(
            recommend_backend(WeightFormat::Gguf, &[BackendKind::Ollama], BackendKind::Vllm),
            BackendKind::Ollama
        );
        // Nothing installed → canonical llama.cpp (not the vLLM fallback).
        assert_eq!(
            recommend_backend(WeightFormat::Gguf, &[], BackendKind::Vllm),
            BackendKind::LlamaCpp
        );
    }

    #[test]
    fn recommend_transformers_goes_to_vllm_not_llamacpp() {
        assert_eq!(
            recommend_backend(
                WeightFormat::Transformers,
                &[BackendKind::LlamaCpp, BackendKind::Vllm],
                BackendKind::Ollama
            ),
            BackendKind::Vllm
        );
    }

    #[test]
    fn unknown_format_keeps_configured_default() {
        assert_eq!(
            recommend_backend(WeightFormat::Unknown, &[BackendKind::Vllm], BackendKind::Ollama),
            BackendKind::Ollama
        );
    }

    #[test]
    fn backend_supports_matches_format() {
        assert!(backend_supports(BackendKind::LlamaCpp, WeightFormat::Gguf));
        assert!(!backend_supports(BackendKind::Vllm, WeightFormat::Gguf));
        assert!(backend_supports(BackendKind::Vllm, WeightFormat::Transformers));
        // Unknown never triggers a correction.
        assert!(backend_supports(BackendKind::Vllm, WeightFormat::Unknown));
    }

    #[test]
    fn preset_picks_llamacpp_for_gguf_and_offloads_layers() {
        let files = s(&["m-Q4_K_M.gguf"]);
        let input = PresetInput {
            model_id: "org/cool-gguf",
            selected_quant: Some("Q4_K_M"),
            weight_files: &files,
            tags: &[],
            card_markdown: None,
            installed_backends: &[BackendKind::LlamaCpp],
            configured_default: BackendKind::Vllm,
            has_gpu: true,
        };
        let p = recommend_deploy_preset(&input);
        assert_eq!(p.backend, BackendKind::LlamaCpp);
        assert_eq!(p.tuning.gpu_layers, Some(LLAMACPP_OFFLOAD_ALL));
    }

    #[test]
    fn preset_no_gpu_layers_without_gpu() {
        let files = s(&["m-Q4_K_M.gguf"]);
        let input = PresetInput {
            model_id: "org/cool-gguf",
            selected_quant: Some("Q4_K_M"),
            weight_files: &files,
            tags: &[],
            card_markdown: None,
            installed_backends: &[],
            configured_default: BackendKind::Vllm,
            has_gpu: false,
        };
        let p = recommend_deploy_preset(&input);
        assert_eq!(p.backend, BackendKind::LlamaCpp);
        assert_eq!(p.tuning.gpu_layers, None);
    }

    #[test]
    fn preset_reads_context_and_flags_from_card() {
        let card = r#"
# Model
```bash
vllm serve org/m --max-model-len 32768 --gpu-memory-utilization 0.9 --trust-remote-code
```
"#;
        let files = s(&["model.safetensors"]);
        let input = PresetInput {
            model_id: "org/m",
            selected_quant: Some("FP16"),
            weight_files: &files,
            tags: &[],
            card_markdown: Some(card),
            installed_backends: &[BackendKind::Vllm],
            configured_default: BackendKind::Vllm,
            has_gpu: true,
        };
        let p = recommend_deploy_preset(&input);
        assert_eq!(p.backend, BackendKind::Vllm);
        assert_eq!(p.desired_context, Some(32768));
        assert_eq!(p.tuning.gpu_memory_fraction, Some(0.9));
        assert!(p.tuning.extra_args.iter().any(|a| a == "--trust-remote-code"));
    }

    #[test]
    fn native_context_from_name() {
        assert_eq!(
            parse_native_context(None, "Qwen/Qwen2.5-Coder-7B-Instruct-128k", &[]),
            Some(131_072)
        );
    }

    #[test]
    fn native_context_from_prose() {
        let card = "This model supports a context length of 32768 tokens.";
        assert_eq!(parse_native_context(Some(card), "org/m", &[]), Some(32_768));
    }

    #[test]
    fn native_context_kb_suffix_in_prose() {
        let card = "Trained with 128K context window for long documents.";
        assert_eq!(parse_native_context(Some(card), "org/m", &[]), Some(131_072));
    }

    #[test]
    fn native_context_ignores_param_count() {
        // "7B" must not be read as a context; no k/m context token here.
        assert_eq!(parse_native_context(None, "org/Model-7B-Instruct", &[]), None);
    }

    #[test]
    fn preset_respects_small_explicit_context() {
        // A card that pins a 4k max must not be raised to the 8k default.
        let card = "```bash\nvllm serve org/m --max-model-len 4096\n```";
        let input = PresetInput {
            model_id: "org/m",
            selected_quant: Some("FP16"),
            weight_files: &s(&["model.safetensors"]),
            tags: &[],
            card_markdown: Some(card),
            installed_backends: &[BackendKind::Vllm],
            configured_default: BackendKind::Vllm,
            has_gpu: true,
        };
        let p = recommend_deploy_preset(&input);
        assert_eq!(p.desired_context, Some(4096));
    }

    #[test]
    fn preset_context_clamped_to_ceiling() {
        // A 1M-context name is clamped to the preset ceiling.
        let input = PresetInput {
            model_id: "org/Model-1M",
            selected_quant: Some("FP16"),
            weight_files: &s(&["model.safetensors"]),
            tags: &[],
            card_markdown: None,
            installed_backends: &[BackendKind::Vllm],
            configured_default: BackendKind::Vllm,
            has_gpu: true,
        };
        let p = recommend_deploy_preset(&input);
        assert_eq!(p.desired_context, Some(MAX_PRESET_CTX));
    }
}
