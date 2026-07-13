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
    let paths = AppPaths::resolve()?;
    paths.ensure_dirs()?;
    let config = Config::load(&paths)?;
    let _log_guard = localcode_log::init(&paths, &config.logging)?;

    match cli.command.unwrap_or(Commands::Tui) {
        Commands::Tui => {
            info!("launching TUI");
            run_tui(paths, config).await
        }
        Commands::Doctor { json } => {
            let report = run_doctor(&paths, &config).await;
            if json {
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
            } else {
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
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
            let svc = DeployService::new(registry, events, gpu);
            let req = DeployRequest {
                model_id: model,
                quantization: quant,
                weight_bytes: 0,
                weight_files: vec![],
                download_urls: vec![],
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
                println!("{}", serde_json::to_string_pretty(&result).unwrap());
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
                let endpoint =
                    endpoint.unwrap_or_else(|| config.assistant.base_url.clone());
                let mut runtime = ActiveRuntime::new(
                    "cli",
                    RuntimeKind::OpenAiCompatible,
                    endpoint,
                );
                runtime.model_id = model.or_else(|| {
                    if config.assistant.model.is_empty() {
                        None
                    } else {
                        Some(config.assistant.model.clone())
                    }
                });
                runtime.api_key = config.assistant_api_key();
                let out = localcode_agent::run_headless(
                    &prompt,
                    workspace,
                    &runtime,
                    config.agent.clone(),
                    runtime.api_key.as_deref(),
                )
                .await?;
                println!("{out}");
                Ok(())
            }
        },
    }
}
