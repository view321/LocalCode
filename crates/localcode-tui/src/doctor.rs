//! Doctor diagnostics for CLI and Setup.

use localcode_backends::{smoke_test, BackendRegistry};
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

    // Smoke-test only *installed* backends (nothing to run otherwise). Each
    // probe is timeout-bounded internally; running them concurrently keeps the
    // added latency ≈ the slowest single probe rather than their sum.
    let smoke = futures::future::join_all(
        backends
            .iter()
            .filter(|r| r.installed)
            .map(|r| smoke_test(r, cfg)),
    )
    .await;

    // Use the same env-resolved endpoints the actual clients use.
    let (hf_endpoint, hf_api_endpoint) = cfg.hf_endpoints();
    let api_base = cfg.api_base_url();
    let hf_ok = check_url(&hf_api_endpoint).await;
    let api_ok = check_url(&format!("{}/v1/health", api_base.trim_end_matches('/'))).await;

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
        "smoke": smoke,
        "huggingface": {
            "endpoint": hf_endpoint,
            "api_endpoint": hf_api_endpoint,
            "reachable": hf_ok,
            "token_set": cfg.hf_token().is_some(),
            // The doctor JSON is fed verbatim to the assistant; without this
            // note a bare `token_set: false` reads as a blocking failure and
            // the model starts demanding a token for public models.
            "token_note": "HF token only matters for gated models; public models download and deploy without one",
        },
        "api": {
            "base_url": api_base,
            "reachable": api_ok,
        },
        "assistant": {
            "provider": cfg.assistant.provider,
            "local_preference": format!("{:?}", cfg.assistant.local_preference),
            "local_ready": localcode_assistant::is_installed(cfg, paths),
            "configured": cfg.assistant_api_key().is_some()
                || matches!(
                    cfg.assistant.provider.as_str(),
                    "self_hosted" | "openai_compatible" | "custom" | "local"
                )
                || localcode_assistant::is_installed(cfg, paths),
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
