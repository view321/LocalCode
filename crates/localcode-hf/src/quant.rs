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

/// A colibrì int4 container ships as one directory the engine loads whole:
/// `out-*.safetensors` shards, `.qs` per-row scales, plus `config.json` and the
/// tokenizer files. Detect it from that shard/scale naming (colibrì-specific —
/// upstream HF checkpoints don't use an `out-` prefix or `.qs` sidecars).
fn looks_like_colibri_container(siblings: &[ModelFile]) -> bool {
    siblings.iter().any(|f| {
        let l = f.rfilename.to_lowercase();
        let name = l.rsplit(['/', '\\']).next().unwrap_or(&l);
        l.ends_with(".qs") || (name.starts_with("out-") && name.ends_with(".safetensors"))
    })
}

pub fn discover_quants(siblings: &[ModelFile]) -> Vec<QuantGroup> {
    // A colibrì container is a single indivisible int4 group: the engine needs
    // config.json (architecture), the tokenizer, and the `.qs` scales alongside
    // the shards, and it can't pull them from HF itself. Expose the *whole*
    // sibling list as one group so every download/deploy path fetches the
    // complete container — a shard-only download is missing config.json and
    // `coli serve` refuses to start.
    if looks_like_colibri_container(siblings) {
        let files: Vec<QuantFile> = siblings
            .iter()
            .map(|f| QuantFile {
                filename: f.rfilename.clone(),
                size: f.size,
                quant_label: "INT4".into(),
            })
            .collect();
        let total_size = siblings.iter().filter_map(|f| f.size).sum();
        return vec![QuantGroup {
            label: "INT4".into(),
            files,
            total_size,
            known: true,
        }];
    }

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
    fn colibri_container_collapses_to_one_group_with_all_files() {
        // A colibrì repo must download whole: config.json + tokenizer + .qs
        // scales alongside the out-*.safetensors shards, not just the shards.
        let files = vec![
            ModelFile { rfilename: "config.json".into(), size: Some(2_000) },
            ModelFile { rfilename: "tokenizer.json".into(), size: Some(5_000) },
            ModelFile { rfilename: "out-00001-of-00002.safetensors".into(), size: Some(1_000_000) },
            ModelFile { rfilename: "out-00002-of-00002.safetensors".into(), size: Some(1_000_000) },
            ModelFile { rfilename: "out-mtp-0.safetensors".into(), size: Some(300) },
            ModelFile { rfilename: "layer0.qs".into(), size: Some(100) },
            ModelFile { rfilename: ".gitattributes".into(), size: Some(50) },
        ];
        let g = discover_quants(&files);
        assert_eq!(g.len(), 1, "colibrì repo is one indivisible container group");
        let names: Vec<&str> = g[0].files.iter().map(|f| f.filename.as_str()).collect();
        assert!(names.contains(&"config.json"), "config.json must be in the download set");
        assert!(names.contains(&"tokenizer.json"), "tokenizer must be in the download set");
        assert!(names.contains(&"layer0.qs"), ".qs scales must be in the download set");
        assert!(names.contains(&"out-00001-of-00002.safetensors"));
        assert_eq!(g[0].total_size, 2_007_450, "size sums the whole container");
        assert_eq!(g[0].label, "INT4");
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
