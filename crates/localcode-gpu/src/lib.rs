//! GPU discovery and VRAM fit estimation.

use localcode_core::error::{ErrorCode, LocalCodeError};
use serde::{Deserialize, Serialize};
use std::process::Command;
use tracing::{debug, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub index: u32,
    pub name: String,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
    pub driver_version: Option<String>,
    pub backend_affinity: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInventory {
    pub devices: Vec<GpuDevice>,
    pub detection_method: String,
    pub warnings: Vec<String>,
}

impl GpuInventory {
    pub fn total_vram(&self) -> u64 {
        self.devices.iter().map(|d| d.total_vram_bytes).sum()
    }

    pub fn free_vram(&self) -> u64 {
        self.devices.iter().map(|d| d.free_vram_bytes).sum()
    }

    pub fn summary(&self) -> String {
        if self.devices.is_empty() {
            return "No GPU detected".into();
        }
        self.devices
            .iter()
            .map(|d| {
                format!(
                    "{} {:.1}/{:.1} GiB free",
                    d.name,
                    d.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    d.total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                )
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

/// Discover GPUs via nvidia-smi when available; graceful empty inventory otherwise.
pub fn discover() -> Result<GpuInventory, LocalCodeError> {
    if which::which("nvidia-smi").is_ok() {
        match discover_nvidia_smi() {
            Ok(inv) if !inv.devices.is_empty() => return Ok(inv),
            Ok(_) => {
                warn!("nvidia-smi returned no devices");
            }
            Err(e) => {
                warn!(error = %e, "nvidia-smi detection failed");
            }
        }
    }

    // CPU-only fallback — never hard-fail the app
    Ok(GpuInventory {
        devices: vec![],
        detection_method: "none".into(),
        warnings: vec![
            "No GPU detected. Local deploys may run on CPU (slow) or fail.".into(),
            "Install NVIDIA drivers and ensure nvidia-smi is on PATH for CUDA GPUs.".into(),
        ],
    })
}

fn discover_nvidia_smi() -> Result<GpuInventory, LocalCodeError> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| {
            LocalCodeError::new(ErrorCode::GpuDetectFailed, e.to_string())
                .with_cause("Failed to execute nvidia-smi")
                .with_hint("Install NVIDIA drivers or add nvidia-smi to PATH")
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(LocalCodeError::new(
            ErrorCode::GpuDetectFailed,
            format!("nvidia-smi exited with error: {stderr}"),
        )
        .with_hint("Check GPU drivers"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut devices = Vec::new();
    for line in stdout.lines() {
        let parts: Vec<_> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            continue;
        }
        let index: u32 = parts[0].parse().unwrap_or(0);
        let name = parts[1].to_string();
        let total_mib: u64 = parts[2].parse().unwrap_or(0);
        let free_mib: u64 = parts[3].parse().unwrap_or(0);
        let driver = parts.get(4).map(|s| s.to_string());
        devices.push(GpuDevice {
            index,
            name,
            total_vram_bytes: total_mib * 1024 * 1024,
            free_vram_bytes: free_mib * 1024 * 1024,
            driver_version: driver,
            backend_affinity: vec!["cuda".into()],
        });
    }

    debug!(count = devices.len(), "discovered GPUs via nvidia-smi");
    Ok(GpuInventory {
        devices,
        detection_method: "nvidia-smi".into(),
        warnings: vec![],
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitConfidence {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitPrediction {
    pub estimated_vram_bytes: u64,
    pub free_vram_bytes: u64,
    pub total_vram_bytes: u64,
    pub fits_free: bool,
    pub fits_total: bool,
    pub confidence: FitConfidence,
    pub assumptions: Vec<String>,
    pub warning: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FitRequest {
    pub weight_bytes: u64,
    pub param_count: Option<u64>,
    pub quant_label: Option<String>,
    pub context_length: u32,
    pub backend: String,
}

/// Heuristic VRAM fit model (v1). Never used to hard-block deploys.
pub fn predict_fit(inventory: &GpuInventory, req: &FitRequest) -> FitPrediction {
    let free = inventory.free_vram();
    let total = inventory.total_vram();

    let dtype_factor = quant_dtype_factor(req.quant_label.as_deref());
    let weight = if req.weight_bytes > 0 {
        (req.weight_bytes as f64 * dtype_factor) as u64
    } else if let Some(params) = req.param_count {
        let bytes_per = quant_bytes_per_param(req.quant_label.as_deref());
        (params as f64 * bytes_per) as u64
    } else {
        0
    };

    // Rough KV cache: 2 * layers_est * ctx * hidden_est * 2 bytes; use simplified linear model
    let hidden_est = 4096.0_f64;
    let layers_est = 32.0_f64;
    let kv = (2.0 * layers_est * req.context_length as f64 * hidden_est * 2.0) as u64;

    let overhead = backend_overhead_bytes(&req.backend);
    let estimated = weight + kv + overhead;

    let fits_free = free == 0 || estimated <= free;
    let fits_total = total == 0 || estimated <= total;

    let mut assumptions = vec![
        format!("ctx={}", req.context_length),
        format!("backend={}", req.backend),
        "kv_dtype=fp16".into(),
        format!("dtype_factor={dtype_factor:.2}"),
    ];
    if let Some(q) = &req.quant_label {
        assumptions.push(format!("quant={q}"));
    }

    let confidence = if req.weight_bytes > 0 {
        FitConfidence::High
    } else if req.param_count.is_some() {
        FitConfidence::Medium
    } else {
        FitConfidence::Low
    };

    let warning = if total == 0 {
        Some("No GPU VRAM detected; estimate is informational only.".into())
    } else if !fits_total {
        Some(format!(
            "Model may exceed total VRAM ({:.1} GiB need vs {:.1} GiB total); deploy may fail or spill to system RAM.",
            estimated as f64 / GIB,
            total as f64 / GIB
        ))
    } else if !fits_free {
        Some(format!(
            "Model may exceed free VRAM ({:.1} GiB need vs {:.1} GiB free); deploy may spill to RAM/CPU or fail.",
            estimated as f64 / GIB,
            free as f64 / GIB
        ))
    } else {
        None
    };

    FitPrediction {
        estimated_vram_bytes: estimated,
        free_vram_bytes: free,
        total_vram_bytes: total,
        fits_free: if free == 0 { true } else { fits_free },
        fits_total: if total == 0 { true } else { fits_total },
        confidence,
        assumptions,
        warning,
    }
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn quant_dtype_factor(quant: Option<&str>) -> f64 {
    // weight_bytes already includes file size; factor ~1.0–1.15 for runtime unpacking
    match quant.map(|q| q.to_uppercase()) {
        Some(q) if q.contains("Q2") || q.contains("IQ2") => 1.05,
        Some(q) if q.contains("Q3") || q.contains("IQ3") => 1.08,
        Some(q) if q.contains("Q4") || q.contains("IQ4") => 1.10,
        Some(q) if q.contains("Q5") => 1.12,
        Some(q) if q.contains("Q6") || q.contains("Q8") => 1.15,
        Some(q) if q.contains("AWQ") || q.contains("GPTQ") => 1.12,
        Some(q) if q.contains("FP16") || q.contains("BF16") => 1.05,
        _ => 1.10,
    }
}

fn quant_bytes_per_param(quant: Option<&str>) -> f64 {
    match quant.map(|q| q.to_uppercase()) {
        Some(q) if q.contains("Q2") || q.contains("IQ2") => 0.3,
        Some(q) if q.contains("Q3") || q.contains("IQ3") => 0.4,
        Some(q) if q.contains("Q4") || q.contains("IQ4") => 0.55,
        Some(q) if q.contains("Q5") => 0.7,
        Some(q) if q.contains("Q6") => 0.8,
        Some(q) if q.contains("Q8") => 1.0,
        Some(q) if q.contains("AWQ") || q.contains("GPTQ") => 0.5,
        Some(q) if q.contains("FP16") || q.contains("BF16") => 2.0,
        Some(q) if q.contains("FP32") => 4.0,
        _ => 0.55,
    }
}

fn backend_overhead_bytes(backend: &str) -> u64 {
    match backend.to_lowercase().as_str() {
        "ollama" => 512 * 1024 * 1024,
        "llamacpp" | "llama.cpp" => 256 * 1024 * 1024,
        "vllm" => 1024 * 1024 * 1024,
        "sglang" => 1024 * 1024 * 1024,
        _ => 512 * 1024 * 1024,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversize_warns_not_block_semantics() {
        let inv = GpuInventory {
            devices: vec![GpuDevice {
                index: 0,
                name: "Test GPU".into(),
                total_vram_bytes: 8 * 1024 * 1024 * 1024,
                free_vram_bytes: 4 * 1024 * 1024 * 1024,
                driver_version: None,
                backend_affinity: vec!["cuda".into()],
            }],
            detection_method: "test".into(),
            warnings: vec![],
        };
        let pred = predict_fit(
            &inv,
            &FitRequest {
                weight_bytes: 7 * 1024 * 1024 * 1024,
                param_count: None,
                quant_label: Some("Q4_K_M".into()),
                context_length: 8192,
                backend: "llamacpp".into(),
            },
        );
        assert!(!pred.fits_free);
        assert!(pred.warning.is_some());
        // Policy: caller must still allow deploy
    }

    #[test]
    fn empty_gpu_is_ok() {
        let inv = GpuInventory {
            devices: vec![],
            detection_method: "none".into(),
            warnings: vec![],
        };
        let pred = predict_fit(
            &inv,
            &FitRequest {
                weight_bytes: 1_000_000,
                param_count: None,
                quant_label: None,
                context_length: 4096,
                backend: "ollama".into(),
            },
        );
        assert!(pred.fits_free);
    }
}
