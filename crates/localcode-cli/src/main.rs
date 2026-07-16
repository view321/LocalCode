//! LocalCode CLI entrypoint.

use clap::{Parser, Subcommand, ValueEnum};
use localcode_agent::{AgentEvent, AgentSession, CodingAgent};
use localcode_backends::{BackendKind, BackendRegistry, DeployRequest, DeployService};
use localcode_core::config::{ApprovalMode, Config};
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::EventBus;
use localcode_core::paths::AppPaths;
use localcode_core::runtime::{ActiveRuntime, RuntimeKind, RuntimeStatus};
use localcode_gpu::discover;
use localcode_hf::HfClient;
use localcode_tui::{run_doctor, run_tui};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
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
    /// Coding-agent commands (headless, non-interactive)
    Agent {
        #[command(subcommand)]
        cmd: AgentCmd,
    },
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
    /// Download a model's weights in the background (resumable).
    ///
    /// Used by the TUI as a detached worker that survives app exit, but also
    /// runnable directly. Writes progress to a state file in the model's cache
    /// directory; a re-run resumes a partial download from where it stopped.
    Download {
        /// Hugging Face model id
        model: String,
        /// Quantization label (defaults to the model's first quant)
        #[arg(long)]
        quant: Option<String>,
        /// Backend the weights are for: llamacpp | colibri | colibri-hy3
        #[arg(long, default_value = "llamacpp")]
        backend: String,
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
        /// Max context length (vLLM --max-model-len, llama.cpp -c, SGLang
        /// --context-length, Ollama num_ctx)
        #[arg(long)]
        context: Option<u32>,
        /// Server port to bind (default: backend's configured port)
        #[arg(long)]
        port: Option<u16>,
        /// Fraction of VRAM to use, 0.0–1.0 (vLLM --gpu-memory-utilization,
        /// SGLang --mem-fraction-static)
        #[arg(long)]
        gpu_memory_fraction: Option<f32>,
        /// GPUs to shard across (vLLM --tensor-parallel-size, SGLang --tp-size)
        #[arg(long)]
        tensor_parallel: Option<u32>,
        /// Layers to offload to GPU (llama.cpp --n-gpu-layers, Ollama num_gpu)
        #[arg(long)]
        gpu_layers: Option<i32>,
    },
    /// Run a benchmark suite
    Bench {
        #[command(subcommand)]
        cmd: BenchCmd,
    },
    /// Check for a new version and install it (git pull + rebuild + swap)
    Update {
        /// Only check and report; do not install
        #[arg(long)]
        check: bool,
    },
    /// Install runtime dependencies (llama-server) and write config paths
    Setup {
        /// Skip installing llama-server even if missing
        #[arg(long)]
        skip_llama: bool,
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
    /// Run the coding agent for one turn: a prompt in, the answer out.
    ///
    /// Streams the model's reply to stdout and tool activity to stderr, so
    /// `localcode agent run "…" > out.txt` captures just the answer. Exits
    /// non-zero if the turn fails.
    #[command(visible_alias = "exec")]
    Run {
        /// Prompt for the agent. Omit (or pass "-") to read it from stdin.
        prompt: Option<String>,
        /// Workspace root the agent operates in
        /// (default: agent.workspace_root, else the current directory).
        #[arg(long, short = 'C')]
        workspace: Option<String>,
        /// OpenAI-compatible base URL to run against — e.g. a running vLLM,
        /// Ollama, or llama.cpp server (http://host:port/v1). Skips runtime
        /// auto-resolution (local assistant / cloud fallback).
        #[arg(long)]
        base_url: Option<String>,
        /// Model id to request. Required with --base-url when the server does
        /// not have a single default model (Ollama, vLLM multi-model, …).
        #[arg(long)]
        model: Option<String>,
        /// Bearer API key for --base-url (defaults to the assistant provider key).
        #[arg(long)]
        api_key: Option<String>,
        /// Approval ceiling for tool calls. With no interactive prompt available,
        /// any call this mode would ask about is declined instead. Pass
        /// `--approvals always` for fully unattended runs.
        #[arg(long, visible_alias = "approval", value_enum)]
        approvals: Option<ApprovalArg>,
        /// Auto-approve every tool call (same as --approvals always). Lets the
        /// agent write files and run shell commands unattended — use with care.
        #[arg(long)]
        yes: bool,
        /// Emit newline-delimited JSON events (deltas, tool calls, final) on stdout.
        #[arg(long)]
        json: bool,
        /// Print only the final answer — suppress streamed tokens and tool activity.
        #[arg(long, short = 'q')]
        quiet: bool,
    },
}

/// Approval ceiling for a headless run. Mirrors [`ApprovalMode`]; with no
/// interactive approver present, calls the mode would prompt for are declined.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum ApprovalArg {
    /// Never ask — run every tool call, destructive shell included.
    Always,
    /// Decline only destructive shell commands; reads, writes, and safe shell run.
    Auto,
    /// Decline every workspace mutation (writes, edits, shell); reads run.
    Edits,
    /// Decline every tool call — the agent answers from the prompt alone.
    Ask,
}

impl From<ApprovalArg> for ApprovalMode {
    fn from(a: ApprovalArg) -> Self {
        match a {
            ApprovalArg::Always => ApprovalMode::AlwaysApprove,
            ApprovalArg::Auto => ApprovalMode::Auto,
            ApprovalArg::Edits => ApprovalMode::ApproveEdits,
            ApprovalArg::Ask => ApprovalMode::AskPermission,
        }
    }
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
    // The TUI and headless `agent run` keep the terminal for their own output —
    // a stderr log layer would corrupt the TUI's alternate screen and pollute
    // the headless answer/JSON stream, so both log to file only.
    let clean_output = matches!(command, Commands::Tui | Commands::Agent { .. });
    let _log_guard = localcode_log::init(&paths, &config.logging, !clean_output)?;

    match command {
        Commands::Tui => {
            info!("launching TUI");
            run_tui(paths, config).await
        }
        Commands::Agent { cmd } => match cmd {
            AgentCmd::Run {
                prompt,
                workspace,
                base_url,
                model,
                api_key,
                approvals,
                yes,
                json,
                quiet,
            } => {
                let args = RunArgs {
                    prompt,
                    workspace,
                    base_url,
                    model,
                    api_key,
                    approval: approvals,
                    yes,
                    json,
                    quiet,
                };
                run_headless(paths, config, args).await
            }
        },
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
        Commands::Download {
            model,
            quant,
            backend,
        } => run_download_worker(paths, config, model, quant, backend).await,
        Commands::Deploy {
            model,
            quant,
            backend,
            force,
            path,
            context,
            port,
            gpu_memory_fraction,
            tensor_parallel,
            gpu_layers,
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
                port,
                context_length: context.unwrap_or(localcode_backends::DEFAULT_DEPLOY_CTX),
                tuning: localcode_backends::DeployTuning {
                    gpu_memory_fraction,
                    tensor_parallel,
                    gpu_layers,
                    extra_args: Vec::new(),
                },
                command_override: None,
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
        Commands::Update { check } => run_update(&config, check).await,
        Commands::Setup { skip_llama } => run_setup(paths, config, skip_llama).await,
    }
}

/// Parsed flags for [`run_headless`].
struct RunArgs {
    prompt: Option<String>,
    workspace: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
    approval: Option<ApprovalArg>,
    yes: bool,
    json: bool,
    quiet: bool,
}

/// Run the coding agent for a single turn without the TUI.
///
/// Resolves a runtime (explicit `--base-url` → local Bonsai assistant → cloud
/// fallback), runs one turn, streams tokens to stdout and tool activity to
/// stderr (or newline-delimited JSON to stdout with `--json`), and returns the
/// turn's error as the process exit status.
async fn run_headless(
    paths: AppPaths,
    config: Config,
    args: RunArgs,
) -> Result<(), LocalCodeError> {
    // Prompt: positional argument, or stdin when omitted / given as "-".
    let prompt = match args.prompt.as_deref() {
        Some(p) if p != "-" => p.to_string(),
        _ => std::io::read_to_string(std::io::stdin()).map_err(|e| {
            LocalCodeError::new(ErrorCode::Internal, format!("Failed to read prompt from stdin: {e}"))
        })?,
    };
    let prompt = prompt.trim().to_string();
    if prompt.is_empty() {
        return Err(LocalCodeError::new(
            ErrorCode::AgentToolFailed,
            "No prompt provided",
        )
        .with_hint("Pass a prompt argument (localcode agent run \"…\") or pipe one on stdin"));
    }

    // Workspace: flag → config → current directory (matches the TUI).
    let workspace = args
        .workspace
        .clone()
        .map(PathBuf::from)
        .or_else(|| config.agent.workspace_root.clone().map(PathBuf::from))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    // Approval mode: --yes wins, else the explicit --approvals, else inherit config.
    let mut agent_cfg = config.agent.clone();
    let mode = if args.yes {
        Some(ApprovalMode::AlwaysApprove)
    } else {
        args.approval.map(ApprovalMode::from)
    };
    if let Some(m) = mode {
        agent_cfg.approval_mode = m;
        // Keep the legacy "confirm_destructive_tools=false" override from
        // silently forcing AlwaysApprove past an explicit choice here.
        agent_cfg.confirm_destructive_tools = m != ApprovalMode::AlwaysApprove;
    }

    let runtime = resolve_headless_runtime(&config, &paths, &args).await?;

    // No model-management tools in headless (they need the TUI's live registry);
    // wire HF so the model-catalogue tools still work.
    let mut agent = CodingAgent::new(agent_cfg);
    if let Ok(hf) = HfClient::new(&config.registry, config.hf_token(), paths.models_cache.clone()) {
        agent = agent.with_hf(Arc::new(hf));
    }

    let mut session = AgentSession::new(workspace);

    // Live events flow through their own channel to a printer task so tokens
    // stream as they arrive; the task exits when the turn drops the sender.
    let (ev_tx, mut ev_rx) = mpsc::unbounded_channel::<AgentEvent>();
    let json = args.json;
    let quiet = args.quiet;
    let printer = tokio::spawn(async move {
        while let Some(ev) = ev_rx.recv().await {
            print_event(&ev, json, quiet);
        }
    });

    let api_key = runtime.api_key.clone();
    let result = agent
        .run_turn(
            &mut session,
            &prompt,
            &runtime,
            api_key.as_deref(),
            None, // no interactive approver: gated calls are declined
            Some(&ev_tx),
        )
        .await;
    drop(ev_tx);
    let _ = printer.await;

    match result {
        Ok(final_text) => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({ "type": "final", "text": final_text })
                );
            } else if quiet {
                println!("{final_text}");
            } else {
                // Tokens already streamed to stdout; end with a newline.
                println!();
            }
            Ok(())
        }
        Err(e) => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "type": "error",
                        "code": e.code.to_string(),
                        "message": e.message,
                    })
                );
            }
            Err(e)
        }
    }
}

/// Pick the runtime a headless run talks to. Priority: an explicit
/// `--base-url` endpoint, then the installed local Bonsai assistant, then the
/// configured cloud assistant provider.
async fn resolve_headless_runtime(
    config: &Config,
    paths: &AppPaths,
    args: &RunArgs,
) -> Result<ActiveRuntime, LocalCodeError> {
    // 1. Explicit OpenAI-compatible endpoint.
    if let Some(base) = &args.base_url {
        let mut r =
            ActiveRuntime::new("headless-endpoint", RuntimeKind::OpenAiCompatible, base.clone());
        r.model_id = args.model.clone();
        r.api_key = args.api_key.clone().or_else(|| config.assistant_api_key());
        r.status = RuntimeStatus::Healthy;
        return Ok(r);
    }

    // 2. Local Bonsai assistant, when its weights are installed. `ensure_running`
    //    reuses a healthy server or spawns one; on failure we fall through.
    if localcode_assistant::model_installed(paths) {
        match localcode_assistant::ensure_running(config, paths).await {
            Ok(rt) => {
                let mut r = rt.as_active_runtime();
                if let Some(m) = &args.model {
                    r.model_id = Some(m.clone());
                }
                return Ok(r);
            }
            Err(e) => {
                tracing::warn!(error = %e.message, "local assistant unavailable; trying cloud fallback");
            }
        }
    }

    // 3. Cloud assistant provider (needs an API key).
    if let Some(key) = config.assistant_api_key() {
        let mut r = ActiveRuntime::new(
            "assistant-provider",
            RuntimeKind::OpenAiCompatible,
            config.assistant.base_url.clone(),
        );
        r.model_id = Some(args.model.clone().unwrap_or_else(|| {
            if config.assistant.model.is_empty() {
                "openai/gpt-4o-mini".into()
            } else {
                config.assistant.model.clone()
            }
        }));
        r.api_key = Some(key);
        r.status = RuntimeStatus::Healthy;
        return Ok(r);
    }

    Err(
        LocalCodeError::new(ErrorCode::BackendNotReady, "No runtime available for a headless run")
            .with_hint("Pass --base-url <endpoint> to target a running vLLM/Ollama/llama.cpp server")
            .with_hint("or install the local assistant (localcode setup, then /assistant install in the TUI)")
            .with_hint("or set the assistant provider API key for a cloud fallback"),
    )
}

/// Render one [`AgentEvent`] for a headless run. In `json` mode every event is
/// a JSON line on stdout; otherwise model text streams to stdout and tool
/// activity goes to stderr (so a redirected stdout captures only the answer).
/// `quiet` suppresses everything but the final answer (printed by the caller).
fn print_event(ev: &AgentEvent, json: bool, quiet: bool) {
    use std::io::Write;
    if json {
        let v = match ev {
            AgentEvent::Delta(t) => serde_json::json!({ "type": "delta", "text": t }),
            AgentEvent::ThinkingDelta(t) => serde_json::json!({ "type": "thinking", "text": t }),
            AgentEvent::MessageComplete => serde_json::json!({ "type": "message_complete" }),
            AgentEvent::ToolStarted { name, args_preview } => {
                serde_json::json!({ "type": "tool_started", "name": name, "args": args_preview })
            }
            AgentEvent::ToolFinished { name, ok, summary, .. } => serde_json::json!({
                "type": "tool_finished", "name": name, "ok": ok, "summary": summary,
            }),
        };
        println!("{v}");
        return;
    }
    if quiet {
        return;
    }
    match ev {
        AgentEvent::Delta(t) => {
            let mut out = std::io::stdout();
            let _ = write!(out, "{t}");
            let _ = out.flush();
        }
        // Reasoning and message boundaries are noise on a plain stream.
        AgentEvent::ThinkingDelta(_) | AgentEvent::MessageComplete => {}
        AgentEvent::ToolStarted { name, args_preview } => {
            eprintln!("  ▶ {name} {args_preview}");
        }
        AgentEvent::ToolFinished { name, ok, summary, .. } => {
            let mark = if *ok { "✓" } else { "✗" };
            eprintln!("  {mark} {name} — {summary}");
        }
    }
}

/// Headless post-install setup: ensure `llama-server` exists and persist its path.
async fn run_setup(
    paths: AppPaths,
    mut config: Config,
    skip_llama: bool,
) -> Result<(), LocalCodeError> {
    if skip_llama {
        println!("Skipping llama-server install (--skip-llama).");
        return Ok(());
    }

    println!("Ensuring llama-server is installed…");
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let printer = tokio::spawn(async move {
        while let Some(line) = rx.recv().await {
            for part in line.lines() {
                eprintln!("==> {part}");
            }
        }
    });

    let bin = localcode_backends::ensure_llamacpp_installed(&paths, tx).await?;
    let _ = printer.await;

    let bin_str = bin.display().to_string();
    if config.backends.llamacpp.bin != bin_str {
        config.backends.llamacpp.bin = bin_str.clone();
        config.save(&paths)?;
        println!("Wrote backends.llamacpp.bin = {bin_str}");
    } else {
        println!("backends.llamacpp.bin already points at {bin_str}");
    }
    println!("llama-server ready: {bin_str}");
    Ok(())
}

/// Background download worker (`localcode download`). Resolves the weight list —
/// reusing a task the TUI already wrote to avoid a fragile HF round-trip, else
/// querying HF — then drives the resumable [`run_worker`], which publishes
/// progress to the model's state file. Prints the primary weight path on success.
async fn run_download_worker(
    paths: AppPaths,
    config: Config,
    model: String,
    quant: Option<String>,
    backend: String,
) -> Result<(), LocalCodeError> {
    let kind = BackendKind::parse(&backend).ok_or_else(|| {
        LocalCodeError::new(ErrorCode::BackendNotFound, format!("Unknown backend {backend}"))
    })?;
    let dir = localcode_backends::model_dir(&paths.models_cache, &model);

    // Prefer a task already resolved into the state file (by the TUI or a prior
    // run) so an unstable link isn't needed just to learn the file list.
    let (quantization, files, urls, total_bytes) = match localcode_backends::read_state(&dir) {
        Some(st) if !st.files.is_empty() && !st.download_urls.is_empty() => {
            (st.quantization, st.files, st.download_urls, st.total_bytes)
        }
        _ => resolve_download_task(&config, &paths, &model, quant.as_deref()).await?,
    };

    let spec = localcode_backends::DownloadSpec {
        model_id: model.clone(),
        quantization,
        backend: kind,
        dir,
        files,
        download_urls: urls,
        total_bytes,
    };
    eprintln!("Downloading {model} → {}", spec.dir.display());
    let primary =
        localcode_backends::run_worker(&spec, config.hf_token().as_deref(), &config.hf_mirror_hosts())
            .await?;
    println!("{}", primary.display());
    Ok(())
}

/// Resolve a model+quant to its weight files, download URLs, and total size via
/// the HF registry (mirrors applied). Mirrors the enrichment the `deploy`
/// command does.
async fn resolve_download_task(
    config: &Config,
    paths: &AppPaths,
    model: &str,
    quant: Option<&str>,
) -> Result<(Option<String>, Vec<String>, Vec<String>, u64), LocalCodeError> {
    let hf = HfClient::new(&config.registry, config.hf_token(), paths.models_cache.clone())?;
    let detail = hf.model_info(model).await?;
    let group = match quant {
        Some(q) => detail.quants.iter().find(|g| g.label.eq_ignore_ascii_case(q)),
        None => detail.quants.first(),
    }
    .ok_or_else(|| {
        LocalCodeError::new(
            ErrorCode::HfModelNotFound,
            format!("No downloadable quant for {model}"),
        )
        .with_hint("List quants with: localcode models info <id>")
    })?;
    let files: Vec<String> = group.files.iter().map(|f| f.filename.clone()).collect();
    let urls: Vec<String> = files.iter().map(|f| hf.download_url(model, f)).collect();
    Ok((Some(group.label.clone()), files, urls, group.total_size))
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
