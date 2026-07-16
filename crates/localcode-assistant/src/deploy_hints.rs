//! Parse Hugging Face model cards for recommended inference/server flags.
//!
//! When the local assistant (or the deploy path) reads a README, this module
//! extracts concrete CLI tokens that map onto LocalCode's deploy knobs
//! (`DeployTuning`) plus free-form `extra_args` for the backend launcher.

use localcode_backends::{BackendKind, DeployTuning};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Structured deploy recommendations derived from a model card.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DeployHints {
    pub gpu_memory_fraction: Option<f32>,
    pub tensor_parallel: Option<u32>,
    pub gpu_layers: Option<i32>,
    pub context_length: Option<u32>,
    /// Free-form extra CLI tokens (e.g. `--enforce-eager`, `--trust-remote-code`).
    pub extra_args: Vec<String>,
    /// Human-readable notes (why a flag was chosen).
    pub notes: Vec<String>,
}

impl DeployHints {
    /// Merge into a `DeployTuning`, keeping existing user-set fields unless
    /// `overwrite` is true. Extra args are always appended (deduped).
    pub fn apply_to_tuning(&self, tuning: &mut DeployTuning, overwrite: bool) {
        if overwrite || tuning.gpu_memory_fraction.is_none() {
            if let Some(v) = self.gpu_memory_fraction {
                tuning.gpu_memory_fraction = Some(v);
            }
        }
        if overwrite || tuning.tensor_parallel.is_none() {
            if let Some(v) = self.tensor_parallel {
                tuning.tensor_parallel = Some(v);
            }
        }
        if overwrite || tuning.gpu_layers.is_none() {
            if let Some(v) = self.gpu_layers {
                tuning.gpu_layers = Some(v);
            }
        }
        for a in &self.extra_args {
            if !tuning.extra_args.iter().any(|e| e == a) {
                tuning.extra_args.push(a.clone());
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.gpu_memory_fraction.is_none()
            && self.tensor_parallel.is_none()
            && self.gpu_layers.is_none()
            && self.context_length.is_none()
            && self.extra_args.is_empty()
    }
}

/// Extract deploy-relevant flags from model-card markdown (and optional fenced
/// shell blocks). Pure and fully unit-tested.
pub fn extract_deploy_hints(card_markdown: &str, backend: BackendKind) -> DeployHints {
    let mut hints = DeployHints::default();
    if card_markdown.trim().is_empty() {
        return hints;
    }

    // Prefer fenced code blocks (usage examples) over free-form prose.
    let code = collect_code_blocks(card_markdown);
    let haystack = if code.is_empty() {
        card_markdown.to_string()
    } else {
        code.join("\n")
    };

    parse_flags(&haystack, backend, &mut hints);

    // Backend-specific prose cues when no explicit flags were found.
    let low = card_markdown.to_lowercase();
    if backend == BackendKind::Vllm || backend == BackendKind::Sglang {
        if hints.extra_args.iter().all(|a| a != "--trust-remote-code")
            && (low.contains("trust_remote_code") || low.contains("trust-remote-code"))
        {
            hints.extra_args.push("--trust-remote-code".into());
            hints
                .notes
                .push("Model card mentions trust_remote_code".into());
        }
        if hints.extra_args.iter().all(|a| a != "--enforce-eager")
            && (low.contains("enforce_eager") || low.contains("enforce-eager") || low.contains("enforce eager"))
        {
            hints.extra_args.push("--enforce-eager".into());
            hints.notes.push("Model card recommends enforce-eager".into());
        }
    }

    if backend == BackendKind::LlamaCpp
        && hints.gpu_layers.is_none()
        && (low.contains("-ngl 99") || low.contains("--n-gpu-layers 99") || low.contains("n-gpu-layers"))
    {
        // Common "offload all" recommendation on GGUF cards.
        if let Some(n) = capture_i32(&haystack, r"(?:-ngl|--n-gpu-layers)\s+(-?\d+)") {
            hints.gpu_layers = Some(n);
            hints.notes.push(format!("llama.cpp GPU layers from card: {n}"));
        } else if low.contains("-ngl 99") {
            hints.gpu_layers = Some(99);
            hints.notes.push("llama.cpp -ngl 99 from card example".into());
        }
    }

    hints
}

fn collect_code_blocks(md: &str) -> Vec<String> {
    let re = code_fence_re();
    re.captures_iter(md)
        .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
        .collect()
}

fn parse_flags(text: &str, backend: BackendKind, hints: &mut DeployHints) {
    // --gpu-memory-utilization 0.9  /  --gpu-memory-utilization=0.9
    if let Some(v) = capture_f32(
        text,
        r"--gpu-memory-utilization(?:=|\s+)(0?\.\d+|1(?:\.0+)?)",
    ) {
        if (0.0..=1.0).contains(&v) {
            hints.gpu_memory_fraction = Some(v);
            hints
                .notes
                .push(format!("--gpu-memory-utilization {v} from model card"));
        }
    }
    // SGLang: --mem-fraction-static
    if hints.gpu_memory_fraction.is_none() {
        if let Some(v) = capture_f32(text, r"--mem-fraction-static(?:=|\s+)(0?\.\d+|1(?:\.0+)?)") {
            if (0.0..=1.0).contains(&v) {
                hints.gpu_memory_fraction = Some(v);
                hints
                    .notes
                    .push(format!("--mem-fraction-static {v} from model card"));
            }
        }
    }

    // --tensor-parallel-size / -tp / --tp-size
    if let Some(n) = capture_u32(
        text,
        r"(?:--tensor-parallel-size|--tp-size|-tp)(?:=|\s+)(\d+)",
    ) {
        if n >= 1 && n <= 64 {
            hints.tensor_parallel = Some(n);
            hints
                .notes
                .push(format!("tensor parallel size {n} from model card"));
        }
    }

    // --n-gpu-layers / -ngl
    if let Some(n) = capture_i32(text, r"(?:--n-gpu-layers|-ngl)(?:=|\s+)(-?\d+)") {
        hints.gpu_layers = Some(n);
        hints
            .notes
            .push(format!("--n-gpu-layers {n} from model card"));
    }

    // --max-model-len / --context-length / -c (llama.cpp)
    if let Some(n) = capture_u32(
        text,
        r"(?:--max-model-len|--context-length|-c)(?:=|\s+)(\d+)",
    ) {
        if (512..=1_048_576).contains(&n) {
            hints.context_length = Some(n);
            hints
                .notes
                .push(format!("context length {n} from model card"));
        }
    }

    // Standalone boolean-ish flags commonly required by cards.
    let known = match backend {
        BackendKind::Vllm => &[
            "--enforce-eager",
            "--trust-remote-code",
            "--enable-chunked-prefill",
            "--disable-log-stats",
            "--dtype",
        ][..],
        BackendKind::Sglang => &["--trust-remote-code", "--disable-radix-cache", "--enable-torch-compile"][..],
        BackendKind::LlamaCpp => &["--flash-attn", "--mlock", "--no-mmap", "--cont-batching"][..],
        // coli serve's memory-tier knobs, e.g. a card's
        // `./coli serve --ram 12 --gpu 0 --vram 14` — numeric values are
        // captured into extra_args and passed straight through by the backend.
        BackendKind::Colibri | BackendKind::ColibriHy3 => {
            &["--ram", "--vram", "--gpu", "--kv-slots", "--max-queue"][..]
        }
        BackendKind::Ollama => &[][..],
    };

    for flag in known {
        // Match the flag as a token; if it takes a value, capture the next token.
        let pat = format!(r"(?m)(?:^|\s)({flag})(?:\s+([^\s\\]+))?");
        if let Ok(re) = Regex::new(&pat) {
            if let Some(c) = re.captures(text) {
                let f = c.get(1).map(|m| m.as_str()).unwrap_or(flag);
                if !hints.extra_args.iter().any(|a| a == f) {
                    hints.extra_args.push(f.to_string());
                }
                if let Some(val) = c.get(2).map(|m| m.as_str()) {
                    // Skip values that look like shell redirection or another flag.
                    if !val.starts_with('-') && !val.contains('>') && !val.contains('|') {
                        // Flags that always take a value
                        if *flag == "--dtype" || val.parse::<f64>().is_ok() || val.contains('/') {
                            if !hints.extra_args.iter().any(|a| a == val) {
                                hints.extra_args.push(val.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
}

fn capture_f32(text: &str, pat: &str) -> Option<f32> {
    let re = Regex::new(pat).ok()?;
    let c = re.captures(text)?;
    c.get(1)?.as_str().parse().ok()
}

fn capture_u32(text: &str, pat: &str) -> Option<u32> {
    let re = Regex::new(pat).ok()?;
    let c = re.captures(text)?;
    c.get(1)?.as_str().parse().ok()
}

fn capture_i32(text: &str, pat: &str) -> Option<i32> {
    let re = Regex::new(pat).ok()?;
    let c = re.captures(text)?;
    c.get(1)?.as_str().parse().ok()
}

fn code_fence_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?s)```[^\n]*\n(.*?)```").expect("code fence regex")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_vllm_flags_from_code_block() {
        let card = r#"
# Cool Model

```bash
vllm serve org/cool \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --enforce-eager \
  --trust-remote-code
```
"#;
        let h = extract_deploy_hints(card, BackendKind::Vllm);
        assert_eq!(h.gpu_memory_fraction, Some(0.9));
        assert_eq!(h.tensor_parallel, Some(2));
        assert_eq!(h.context_length, Some(32768));
        assert!(h.extra_args.iter().any(|a| a == "--enforce-eager"));
        assert!(h.extra_args.iter().any(|a| a == "--trust-remote-code"));
    }

    #[test]
    fn extracts_llamacpp_ngl() {
        let card = r#"
```bash
./llama-cli -m model.gguf -ngl 99 -c 8192 --temp 0.7
```
"#;
        let h = extract_deploy_hints(card, BackendKind::LlamaCpp);
        assert_eq!(h.gpu_layers, Some(99));
        assert_eq!(h.context_length, Some(8192));
    }

    #[test]
    fn apply_to_tuning_preserves_user_values() {
        let hints = DeployHints {
            gpu_memory_fraction: Some(0.85),
            tensor_parallel: Some(4),
            gpu_layers: Some(40),
            context_length: Some(16384),
            extra_args: vec!["--enforce-eager".into()],
            notes: vec![],
        };
        let mut tuning = DeployTuning {
            gpu_memory_fraction: Some(0.5),
            tensor_parallel: None,
            gpu_layers: None,
            extra_args: vec![],
        };
        hints.apply_to_tuning(&mut tuning, false);
        assert_eq!(tuning.gpu_memory_fraction, Some(0.5)); // preserved
        assert_eq!(tuning.tensor_parallel, Some(4));
        assert_eq!(tuning.gpu_layers, Some(40));
        assert_eq!(tuning.extra_args, vec!["--enforce-eager".to_string()]);
    }

    #[test]
    fn empty_card_yields_empty_hints() {
        assert!(extract_deploy_hints("", BackendKind::Vllm).is_empty());
    }
}
