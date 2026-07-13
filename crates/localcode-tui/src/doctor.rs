//! Doctor diagnostics for CLI and Setup.

use localcode_backends::BackendRegistry;
use localcode_core::config::Config;
use localcode_core::paths::AppPaths;
use localcode_gpu::discover;
use serde_json::{json, Value};
use std::sync::Arc;

pub async fn run_doctor(paths: &AppPaths, cfg: &Config) -> Value {
    let gpu = discover().unwrap_or_else(|e| localcode_gpu::GpuInventory {
        devices: vec![],
        detection_method: "error".into(),
        warnings: vec![e.message],
    });

    let registry = Arc::new(BackendRegistry::from_config(cfg));
    let backends = registry.detect_all().await;

    let hf_ok = check_url(&cfg.registry.api_endpoint).await;
    let api_ok = check_url(&format!(
        "{}/v1/health",
        cfg.api.base_url.trim_end_matches('/')
    ))
    .await;

    let disk = disk_space(&paths.data_dir);

    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "paths": {
            "config": paths.config_file().display().to_string(),
            "data": paths.data_dir.display().to_string(),
            "logs": paths.log_dir.display().to_string(),
            "cache": paths.cache_dir.display().to_string(),
        },
        "gpu": gpu,
        "backends": backends,
        "huggingface": {
            "endpoint": cfg.registry.endpoint,
            "api_endpoint": cfg.registry.api_endpoint,
            "reachable": hf_ok,
            "token_set": cfg.hf_token().is_some(),
        },
        "api": {
            "base_url": cfg.api.base_url,
            "reachable": api_ok,
        },
        "assistant": {
            "provider": cfg.assistant.provider,
            "configured": cfg.assistant_api_key().is_some() || cfg.assistant.provider == "self_hosted",
        },
        "disk": disk,
        "keyring": { "preferred": true, "note": "OS keyring preferred; env vars used as fallback" },
    })
}

async fn check_url(url: &str) -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build();
    let Ok(client) = client else {
        return false;
    };
    client.get(url).send().await.map(|r| r.status().is_success()).unwrap_or(false)
}

fn disk_space(path: &std::path::Path) -> Value {
    // Portable: just report path exists
    json!({
        "path": path.display().to_string(),
        "exists": path.exists(),
        "note": "Detailed free-space probe is platform-specific; ensure adequate disk for models",
    })
}
