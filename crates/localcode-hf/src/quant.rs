//! Quantization discovery from HF sibling filenames.

use crate::ModelFile;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantFile {
    pub filename: String,
    pub size: Option<u64>,
    pub quant_label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantGroup {
    pub label: String,
    pub files: Vec<QuantFile>,
    pub total_size: u64,
    pub known: bool,
}

fn quant_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)(IQ[1-4]_[A-Z0-9_]+|Q[2-8]_[A-Z0-9_]+|Q[2-8]_0|Q[2-8]_1|f16|fp16|bf16|fp32|awq|gptq|exl2|int4|int8)",
        )
        .expect("quant regex")
    })
}

/// Parse quant label from a weight filename. Returns None if unknown.
pub fn parse_quant_from_filename(name: &str) -> Option<String> {
    let lower = name.to_lowercase();
    // Prefer explicit GGUF-style tokens
    if let Some(c) = quant_re().captures(name) {
        return Some(c[1].to_uppercase().replace("F16", "FP16"));
    }
    if lower.contains("awq") {
        return Some("AWQ".into());
    }
    if lower.contains("gptq") {
        return Some("GPTQ".into());
    }
    if lower.contains("exl2") {
        return Some("EXL2".into());
    }
    if lower.ends_with(".safetensors") || lower.ends_with(".gguf") {
        return None;
    }
    None
}

pub fn discover_quants(siblings: &[ModelFile]) -> Vec<QuantGroup> {
    let weight_exts = [".gguf", ".safetensors", ".bin", ".pt", ".pth", ".gemma"];
    let mut groups: BTreeMap<String, QuantGroup> = BTreeMap::new();

    for f in siblings {
        let lower = f.rfilename.to_lowercase();
        if !weight_exts.iter().any(|e| lower.ends_with(e)) {
            continue;
        }
        // Skip non-weight noise
        if lower.contains("optimizer") || lower.contains("training_args") {
            continue;
        }

        let (label, known) = match parse_quant_from_filename(&f.rfilename) {
            Some(q) => (q, true),
            None => {
                if lower.ends_with(".gguf") {
                    ("UNKNOWN_GGUF".into(), false)
                } else if lower.contains("safetensor") {
                    ("SAFETENSORS".into(), false)
                } else {
                    ("UNKNOWN".into(), false)
                }
            }
        };

        let entry = groups.entry(label.clone()).or_insert_with(|| QuantGroup {
            label: label.clone(),
            files: vec![],
            total_size: 0,
            known,
        });
        let size = f.size.unwrap_or(0);
        entry.total_size += size;
        entry.files.push(QuantFile {
            filename: f.rfilename.clone(),
            size: f.size,
            quant_label: label,
        });
    }

    let mut out: Vec<_> = groups.into_values().collect();
    out.sort_by(|a, b| a.label.cmp(&b.label));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gguf_quants() {
        assert_eq!(
            parse_quant_from_filename("model-Q4_K_M.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            parse_quant_from_filename("foo.IQ4_XS.gguf").as_deref(),
            Some("IQ4_XS")
        );
        assert_eq!(
            parse_quant_from_filename("model.awq.safetensors").as_deref(),
            Some("AWQ")
        );
    }

    #[test]
    fn group_shards() {
        let files = vec![
            ModelFile {
                rfilename: "a-Q4_K_M-00001-of-00002.gguf".into(),
                size: Some(1000),
            },
            ModelFile {
                rfilename: "a-Q4_K_M-00002-of-00002.gguf".into(),
                size: Some(500),
            },
        ];
        let g = discover_quants(&files);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].total_size, 1500);
        assert_eq!(g[0].files.len(), 2);
    }
}
