//! LocalCode CLI entrypoint.

use clap::{Parser, Subcommand};
use localcode_backends::{BackendKind, BackendRegistry, DeployRequest, DeployService};
use localcode_core::config::Config;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::EventBus;
use localcode_core::paths::AppPaths;
use localcode_core::runtime::{ActiveRuntime, RuntimeKind};
use localcode_gpu::discover;
use localcode_hf::HfClient;
use localcode_tui::{run_doctor, run_tui};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "localcode", version, about = "LocalCode — local-first coding agent TUI")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Launch the terminal UI (default)
    Tui,
    /// Run environment diagnostics
    Doctor {
        /// Emit JSON
        #[arg(long)]
        json: bool,
    },
    /// Model registry commands
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
    /// Deploy a model to a local backend
    Deploy {
        /// Hugging Face model id
        model: String,
        /// Quantization label
        #[arg(long)]
        quant: Option<String>,
        /// Backend: ollama | llamacpp | vllm | sglang
        #[arg(long, default_value = "ollama")]
        backend: String,
        /// Continue even if VRAM fit predicts oversize
        #[arg(long)]
        force: bool,
        /// Local GGUF path (llama.cpp)
        #[arg(long)]
        path: Option<String>,
    },
    /// Run a benchmark suite
    Bench {
        #[command(subcommand)]
        cmd: BenchCmd,
    },
    /// Headless coding agent
    Agent {
        #[command(subcommand)]
        cmd: AgentCmd,
    },
    /// Check for a new version and install it (git pull + rebuild + swap)
    Update {
        /// Only check and report; do not install
        #[arg(long)]
        check: bool,
    },
}

#[derive(Subcommand, Debug)]
enum ModelsCmd {
    Search {
        query: String,
        #[arg(long, default_value_t = 20)]
        limit: u32,
    },
    Popular {
        #[arg(long, default_value_t = 20)]
        limit: u32,
    },
    Info {
        model_id: String,
    },
}

#[derive(Subcommand, Debug)]
enum BenchCmd {
    Run {
        /// Suite path or "sample"
        #[arg(default_value = "sample")]
        suite: String,
        #[arg(long)]
        endpoint: Option<String>,
        #[arg(long)]
        model: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
enum AgentCmd {
    Run {
        prompt: String,
        #[arg(long)]
        workspace: Option<PathBuf>,
        #[arg(long)]
        endpoint: Option<String>,
        #[arg(long)]
        model: Option<String>,
    },
}

#[tokio::main]
async fn main() {
    if let Err(e) = real_main().await {
        eprintln!("error {}: {}", e.code, e.message);
        for c in &e.causes {
            eprintln!("  cause: {c}");
        }
        for h in &e.hints {
            eprintln!("  hint: {h}");
        }
        eprintln!("  correlation_id: {}", e.correlation_id);
        std::process::exit(1);
    }
}

async fn real_main() -> Result<(), LocalCodeError> {
    let cli = Cli::parse();
    // Remove `.old`/`.new` binaries a previous self-update left behind.
    localcode_upgrade::cleanup_stale_artifacts();
    let paths = AppPaths::resolve()?;
    paths.ensure_dirs()?;
    let config = Config::load(&paths)?;

    let command = cli.command.unwrap_or(Commands::Tui);
    // In TUI mode logs go to file only — a stderr layer would write straight
    // over the raw-mode alternate screen and corrupt the display.
    let is_tui = matches!(command, Commands::Tui);
    let _log_guard = localcode_log::init(&paths, &config.logging, !is_tui)?;

    match command {
        Commands::Tui => {
            info!("launching TUI");
            run_tui(paths, config).await
        }
        Commands::Doctor { json } => {
            let report = run_doctor(&paths, &config).await;
            if json {
                println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
            } else {
                print_doctor_human(&report);
            }
            Ok(())
        }
        Commands::Models { cmd } => {
            let hf = HfClient::new(
                &config.registry,
                config.hf_token(),
                paths.models_cache.clone(),
            )?;
            match cmd {
                ModelsCmd::Search { query, limit } => {
                    let models = hf.search(&query, true, limit, "downloads").await?;
                    for m in models {
                        println!(
                            "{}\t↓{}\t♥{}",
                            m.id,
                            m.downloads.unwrap_or(0),
                            m.likes.unwrap_or(0)
                        );
                    }
                }
                ModelsCmd::Popular { limit } => {
                    let models = hf.popular_coding(limit).await?;
                    for m in models {
                        println!("{}", m.id);
                    }
                }
                ModelsCmd::Info { model_id } => {
                    let d = hf.model_info(&model_id).await?;
                    println!("id: {}", d.summary.id);
                    println!("license: {:?}", d.license);
                    println!("params: {:?}", d.parameter_size);
                    println!("quants:");
                    for q in d.quants {
                        println!(
                            "  {}  {} bytes  {} files",
                            q.label,
                            q.total_size,
                            q.files.len()
                        );
                    }
                }
            }
            Ok(())
        }
        Commands::Deploy {
            model,
            quant,
            backend,
            force,
            path,
        } => {
            let kind = BackendKind::parse(&backend).ok_or_else(|| {
                LocalCodeError::new(ErrorCode::BackendNotFound, format!("Unknown backend {backend}"))
            })?;
            let gpu = discover().unwrap_or_else(|_| localcode_gpu::GpuInventory {
                devices: vec![],
                detection_method: "none".into(),
                warnings: vec![],
            });
            let registry = Arc::new(BackendRegistry::from_config(&config));
            let events = EventBus::new();
            let svc = DeployService::new(
                registry,
                events.clone(),
                gpu,
                paths.models_cache.clone(),
                config.hf_token(),
                config.hf_mirror_hosts(),
            );

            // Enrich HF ids with quant metadata so fit prediction has real
            // sizes and llama.cpp deploys can download weights.
            let mut quantization = quant;
            let mut weight_bytes = 0u64;
            let mut weight_files: Vec<String> = vec![];
            let mut download_urls: Vec<String> = vec![];
            if model.contains('/') && path.is_none() {
                if let Ok(hf) = HfClient::new(
                    &config.registry,
                    config.hf_token(),
                    paths.models_cache.clone(),
                ) {
                    if let Ok(detail) = hf.model_info(&model).await {
                        let group = match &quantization {
                            Some(q) => detail
                                .quants
                                .iter()
                                .find(|g| g.label.eq_ignore_ascii_case(q)),
                            None => detail.quants.first(),
                        };
                        if let Some(g) = group {
                            quantization = Some(g.label.clone());
                            weight_bytes = g.total_size;
                            weight_files =
                                g.files.iter().map(|f| f.filename.clone()).collect();
                            download_urls = weight_files
                                .iter()
                                .map(|f| hf.download_url(&model, f))
                                .collect();
                        }
                    }
                }
            }

            // Print deploy progress as it happens.
            events.subscribe(|ev| {
                if let localcode_core::events::AppEvent::DeployProgress {
                    percent, message, ..
                } = ev
                {
                    eprintln!("[{percent:>3}%] {message}");
                }
            });

            let req = DeployRequest {
                model_id: model,
                quantization,
                weight_bytes,
                weight_files,
                download_urls,
                local_path: path,
                backend: kind,
                port: None,
                context_length: 8192,
                continue_despite_oversize: force,
            };
            let job = svc.deploy(req).await?;
            println!("deploy job {} correlation {}", job.id, job.correlation_id);
            Ok(())
        }
        Commands::Bench { cmd } => match cmd {
            BenchCmd::Run {
                suite,
                endpoint,
                model,
            } => {
                let suite_obj = if suite == "sample" {
                    localcode_bench::sample_coding_suite()
                } else {
                    localcode_bench::load_suite(std::path::Path::new(&suite))?
                };
                let endpoint = endpoint.unwrap_or_else(|| config.backends.ollama.base_url.clone());
                let model = model.unwrap_or_else(|| "default".into());
                let events = EventBus::new();
                let runner = localcode_bench::BenchRunner::new(events);
                let subject = localcode_bench::Subject {
                    hf_model_id: model.clone(),
                    quantization: "unknown".into(),
                    weight_source: "cli".into(),
                    backend: "cli".into(),
                    backend_version: env!("CARGO_PKG_VERSION").into(),
                    precision_notes: String::new(),
                    hardware: serde_json::json!({}),
                };
                let result = runner
                    .run(&suite_obj, subject, &endpoint, None, &model)
                    .await?;
                println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
                Ok(())
            }
        },
        Commands::Agent { cmd } => match cmd {
            AgentCmd::Run {
                prompt,
                workspace,
                endpoint,
                model,
            } => {
                let workspace = workspace.unwrap_or_else(|| {
                    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
                });
                // Local-first default: talk to the local Ollama OpenAI
                // endpoint unless the user passes --endpoint explicitly.
                let endpoint = endpoint
                    .unwrap_or_else(|| format!("{}/v1", config.backends.ollama.base_url));
                let mut runtime = ActiveRuntime::new(
                    "cli",
                    RuntimeKind::OpenAiCompatible,
                    endpoint,
                );
                runtime.model_id = model;
                let out = localcode_agent::run_headless(
                    &prompt,
                    workspace,
                    &runtime,
                    config.agent.clone(),
                    None,
                )
                .await?;
                println!("{out}");
                Ok(())
            }
        },
        Commands::Update { check } => run_update(&config, check).await,
    }
}

async fn run_update(config: &Config, check_only: bool) -> Result<(), LocalCodeError> {
    let checker = localcode_upgrade::UpdateChecker::new(
        &config.updates.repo_url,
        &config.updates.branch,
    )?;
    println!(
        "localcode v{} — checking {} ({})",
        localcode_upgrade::CURRENT_VERSION,
        config.updates.repo_url,
        config.updates.branch
    );
    match checker.check().await? {
        None => {
            println!("Already up to date.");
            return Ok(());
        }
        Some(info) => {
            println!("Update available: v{} → v{}", info.current, info.latest);
            if check_only {
                println!("Run `localcode update` to install.");
                return Ok(());
            }
        }
    }

    let updater = localcode_upgrade::SelfUpdater::resolve(
        config.updates.install_dir.as_deref(),
        &config.updates.repo_url,
        &config.updates.branch,
        &config.updates.mirrors,
    )?;
    println!("Updating from checkout at {}", updater.install_dir.display());

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let printer = tokio::spawn(async move {
        while let Some(line) = rx.recv().await {
            eprintln!("==> {line}");
        }
    });
    let report = updater.run(tx).await?;
    let _ = printer.await;
    println!(
        "Installed v{} at {} — restart any running LocalCode to use it.",
        report.version,
        report.binary_path.display()
    );
    Ok(())
}

fn print_doctor_human(report: &serde_json::Value) {
    println!("LocalCode doctor v{}", env!("CARGO_PKG_VERSION"));
    if let Some(paths) = report.get("paths").and_then(|p| p.as_object()) {
        println!("\nPaths:");
        for (k, v) in paths {
            println!("  {k}: {}", v.as_str().unwrap_or_default());
        }
    }
    if let Some(gpu) = report.get("gpu") {
        println!("\nGPU ({}):", gpu["detection_method"].as_str().unwrap_or("?"));
        if let Some(devices) = gpu["devices"].as_array() {
            if devices.is_empty() {
                println!("  none detected");
            }
            for d in devices {
                println!(
                    "  {} — {:.1}/{:.1} GiB free",
                    d["name"].as_str().unwrap_or("?"),
                    d["free_vram_bytes"].as_u64().unwrap_or(0) as f64 / 1e9,
                    d["total_vram_bytes"].as_u64().unwrap_or(0) as f64 / 1e9,
                );
            }
        }
    }
    if let Some(backends) = report.get("backends").and_then(|b| b.as_array()) {
        println!("\nBackends:");
        for b in backends {
            let ready = b["ready"].as_bool().unwrap_or(false);
            let mark = if ready { "✓" } else { "✗" };
            println!(
                "  {mark} {} {}",
                b["kind"].as_str().unwrap_or("?"),
                b["notes"]
                    .as_array()
                    .and_then(|n| n.first())
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
            );
        }
    }
    if let Some(smoke) = report.get("smoke").and_then(|s| s.as_array()) {
        if !smoke.is_empty() {
            println!("\nSmoke tests:");
            for s in smoke {
                let ok = s["ok"].as_bool().unwrap_or(false);
                let mark = if ok { "✓" } else { "✗" };
                println!(
                    "  {mark} {} — {}",
                    s["kind"].as_str().unwrap_or("?"),
                    s["checked"].as_str().unwrap_or(""),
                );
                if let Some(dg) = s.get("diagnosis").filter(|d| !d.is_null()) {
                    println!("      → {}", dg["summary"].as_str().unwrap_or(""));
                    if dg.get("repair").is_some_and(|r| !r.is_null()) {
                        println!("      → fix available: open LocalCode and click Fix (or /backends)");
                    }
                }
            }
        }
    }
    if let Some(hf) = report.get("huggingface") {
        println!(
            "\nHugging Face: reachable={} token_set={}",
            hf["reachable"].as_bool().unwrap_or(false),
            hf["token_set"].as_bool().unwrap_or(false)
        );
    }
    if let Some(api) = report.get("api") {
        println!(
            "API: {} reachable={}",
            api["base_url"].as_str().unwrap_or("?"),
            api["reachable"].as_bool().unwrap_or(false)
        );
    }
    if let Some(a) = report.get("assistant") {
        println!(
            "Assistant: provider={} configured={}",
            a["provider"].as_str().unwrap_or("?"),
            a["configured"].as_bool().unwrap_or(false)
        );
    }
    println!("\n(Re-run with --json for the full report.)");
}
