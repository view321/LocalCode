//! Application state and event loop.

use crate::doctor::run_doctor;
use crate::ui;
use crate::widgets::{ModalKind, ModalState};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use localcode_agent::CodingAgent;
use localcode_api_client::ApiClient;
use localcode_assistant::{Assistant, AssistantRequest};
use localcode_backends::{
    BackendKind, BackendRegistry, DeployRequest, DeployService, DetectReport,
};
use localcode_bench::{sample_coding_suite, BenchRunner, Subject};
use localcode_core::config::Config;
use localcode_core::error::LocalCodeError;
use localcode_core::events::{AppEvent, EventBus, Severity};
use localcode_core::paths::AppPaths;
use localcode_core::runtime::ActiveRuntime;
use localcode_core::theme::{Theme, ThemeMode};
use localcode_gpu::{discover, FitPrediction, GpuInventory};
use localcode_hf::{HfClient, ModelDetail, ModelSummary};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Dashboard,
    Models,
    Benchmarks,
    Coding,
    Setup,
    Notifications,
    Settings,
}

impl Tab {
    pub fn all() -> [Tab; 7] {
        [
            Tab::Dashboard,
            Tab::Models,
            Tab::Benchmarks,
            Tab::Coding,
            Tab::Setup,
            Tab::Notifications,
            Tab::Settings,
        ]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Tab::Dashboard => "dashboard",
            Tab::Models => "models",
            Tab::Benchmarks => "benchmarks",
            Tab::Coding => "coding",
            Tab::Setup => "setup",
            Tab::Notifications => "notifications",
            Tab::Settings => "settings",
        }
    }

    pub fn from_index(i: usize) -> Tab {
        *Self::all().get(i).unwrap_or(&Tab::Dashboard)
    }
}

#[derive(Debug, Clone)]
pub struct NotificationItem {
    pub severity: Severity,
    pub title: String,
    pub body: String,
}

pub struct App {
    pub paths: AppPaths,
    pub config: Config,
    pub theme: Theme,
    pub tab: Tab,
    pub rail_focus: bool,
    pub rail_index: usize,
    pub rail_hover: bool,
    pub status_line: String,
    pub status_is_error: bool,
    pub last_error: Option<LocalCodeError>,
    pub modal: Option<ModalState>,
    pub palette_open: bool,
    pub palette_query: String,
    pub palette_selected: usize,
    pub assistant_open: bool,
    pub assistant_reply: Option<String>,
    pub assistant_configured: bool,
    pub api_healthy: bool,
    pub gpu: GpuInventory,
    pub runtimes: Vec<ActiveRuntime>,
    pub notifications: Vec<NotificationItem>,
    pub backend_reports: Vec<DetectReport>,
    pub doctor_summary: Option<String>,
    // Models
    pub model_query: String,
    pub model_search_focus: bool,
    pub models: Vec<ModelSummary>,
    pub model_selected: usize,
    pub model_detail: Option<ModelDetail>,
    pub selected_quant: Option<String>,
    pub deploy_backend: BackendKind,
    pub deploy_ctx: u32,
    pub deploy_port: Option<u16>,
    pub deploy_cloud: bool,
    pub deploy_progress: u8,
    pub last_fit: Option<FitPrediction>,
    pub pending_oversize_deploy: Option<DeployRequest>,
    // Coding
    pub coding_input: String,
    pub coding_input_focus: bool,
    pub coding_transcript: Vec<String>,
    pub subagents_enabled: bool,
    pub skill_count: usize,
    pub mcp_status_summary: String,
    // Bench
    pub last_bench_summary: String,
    pub last_bench_result: Option<localcode_bench::RunResult>,
    // Services
    events: EventBus,
    registry: Arc<BackendRegistry>,
    hf: Option<HfClient>,
    should_quit: bool,
}

impl App {
    pub async fn new(paths: AppPaths, config: Config) -> Self {
        let events = EventBus::new();
        let gpu = discover().unwrap_or_else(|_| GpuInventory {
            devices: vec![],
            detection_method: "none".into(),
            warnings: vec!["GPU discovery failed".into()],
        });
        let registry = Arc::new(BackendRegistry::from_config(&config));
        let backend_reports = registry.detect_all().await;
        let hf = HfClient::new(
            &config.registry,
            config.hf_token(),
            paths.models_cache.clone(),
        )
        .ok();

        let assistant = Assistant::new(config.assistant.clone());
        let assistant_configured = assistant.is_configured(config.assistant_api_key().as_deref());

        let mut api_healthy = false;
        if let Ok(api) = ApiClient::new(&config.api, None) {
            api_healthy = api.health().await.is_ok();
        }

        let skill_count = CodingAgent::new(config.agent.clone()).skills.list().len();

        let mut app = Self {
            theme: Theme::new(config.ui.theme),
            paths,
            tab: Tab::Dashboard,
            rail_focus: false,
            rail_index: 0,
            rail_hover: false,
            status_line: "Ready — one-click local models, no account required".into(),
            status_is_error: false,
            last_error: None,
            modal: None,
            palette_open: false,
            palette_query: String::new(),
            palette_selected: 0,
            assistant_open: false,
            assistant_reply: None,
            assistant_configured,
            api_healthy,
            gpu,
            runtimes: vec![],
            notifications: vec![],
            backend_reports,
            doctor_summary: None,
            model_query: String::new(),
            model_search_focus: false,
            models: vec![],
            model_selected: 0,
            model_detail: None,
            selected_quant: None,
            deploy_backend: BackendKind::parse(&config.backends.default.kind)
                .unwrap_or(BackendKind::Ollama),
            deploy_ctx: 8192,
            deploy_port: None,
            deploy_cloud: false,
            deploy_progress: 0,
            last_fit: None,
            pending_oversize_deploy: None,
            coding_input: String::new(),
            coding_input_focus: false,
            coding_transcript: vec![
                "system: Welcome to LocalCode Coding. Select a runtime and type a message.".into(),
            ],
            subagents_enabled: config.agent.subagents_enabled,
            skill_count,
            mcp_status_summary: "not connected".into(),
            last_bench_summary: "No runs yet.".into(),
            last_bench_result: None,
            events,
            registry,
            hf,
            should_quit: false,
            config,
        };

        // Seed cached popular models if available
        if let Some(hf) = &app.hf {
            if let Some(cached) = hf.cache().get_search("code") {
                app.models = cached;
            }
        }

        app
    }

    pub fn active_runtime_name(&self) -> Option<String> {
        self.runtimes.first().map(|r| r.name.clone())
    }

    pub fn pane_ratios(&self, view: &str, default: &[f32]) -> Vec<f32> {
        self.config
            .panes
            .views
            .get(view)
            .cloned()
            .unwrap_or_else(|| default.to_vec())
    }

    pub fn palette_items(&self) -> Vec<String> {
        let all = vec![
            "Go: dashboard".into(),
            "Go: models".into(),
            "Go: benchmarks".into(),
            "Go: coding".into(),
            "Go: setup".into(),
            "Go: notifications".into(),
            "Go: settings".into(),
            "Deploy last model".into(),
            "Toggle subagents".into(),
            "Open logs".into(),
            "Run doctor".into(),
            "Ask assistant".into(),
            "Quit".into(),
        ];
        if self.palette_query.is_empty() {
            return all;
        }
        let q = self.palette_query.to_lowercase();
        all.into_iter()
            .filter(|i| i.to_lowercase().contains(&q))
            .collect()
    }

    fn raise_error(&mut self, err: LocalCodeError) {
        self.status_line = format!("{}: {}", err.code, err.message);
        self.status_is_error = true;
        self.last_error = Some(err.clone());
        self.notifications.push(NotificationItem {
            severity: Severity::Error,
            title: err.code.as_str().into(),
            body: err.message.clone(),
        });
        self.modal = Some(ModalState::error(err));
    }

    fn set_status(&mut self, msg: impl Into<String>, is_error: bool) {
        self.status_line = msg.into();
        self.status_is_error = is_error;
    }

    async fn process_events(&mut self) {
        for ev in self.events.drain() {
            match ev {
                AppEvent::Notification {
                    severity,
                    title,
                    body,
                    ..
                } => {
                    self.notifications.push(NotificationItem {
                        severity,
                        title: title.clone(),
                        body: body.clone(),
                    });
                    self.set_status(format!("{title}: {body}"), severity == Severity::Error);
                }
                AppEvent::DeployProgress {
                    percent, message, ..
                } => {
                    self.deploy_progress = percent;
                    self.set_status(format!("Deploy {percent}% — {message}"), false);
                }
                AppEvent::DeployFinished { runtime, .. } => {
                    self.deploy_progress = 100;
                    self.runtimes.push(runtime);
                    self.set_status("Deploy finished", false);
                }
                AppEvent::DeployFailed { error, .. } => {
                    self.deploy_progress = 0;
                    self.raise_error(error);
                }
                AppEvent::ErrorRaised { error } => {
                    self.raise_error(error);
                }
                AppEvent::RuntimeUpdated { runtime } => {
                    if let Some(r) = self.runtimes.iter_mut().find(|r| r.id == runtime.id) {
                        *r = runtime;
                    } else {
                        self.runtimes.push(runtime);
                    }
                }
                AppEvent::BenchProgress {
                    completed,
                    total,
                    message,
                    ..
                } => {
                    self.set_status(format!("Bench {completed}/{total}: {message}"), false);
                }
                AppEvent::Status { message } => {
                    self.set_status(message, false);
                }
            }
        }
        // Refresh runtimes from registry
        self.runtimes = self.registry.list_runtimes().await;
    }

    async fn search_models(&mut self) {
        let Some(hf) = self.hf.clone() else {
            self.set_status("HF client unavailable", true);
            return;
        };
        let q = if self.model_query.is_empty() {
            "code".to_string()
        } else {
            self.model_query.clone()
        };
        self.set_status(format!("Searching HF for '{q}'…"), false);
        match hf.search(&q, true, 30, "downloads").await {
            Ok(models) => {
                self.models = models;
                self.model_selected = 0;
                self.set_status(format!("Found {} models", self.models.len()), false);
            }
            Err(e) => self.raise_error(e),
        }
    }

    async fn load_model_detail(&mut self) {
        let Some(id) = self.models.get(self.model_selected).map(|m| m.id.clone()) else {
            return;
        };
        let Some(hf) = self.hf.clone() else {
            return;
        };
        self.set_status(format!("Loading {id}…"), false);
        match hf.model_info(&id).await {
            Ok(detail) => {
                if let Some(q) = detail.quants.first() {
                    self.selected_quant = Some(q.label.clone());
                }
                // Fit estimate
                let weight = detail
                    .quants
                    .iter()
                    .find(|q| Some(q.label.as_str()) == self.selected_quant.as_deref())
                    .map(|q| q.total_size)
                    .unwrap_or(0);
                let deploy_svc =
                    DeployService::new(self.registry.clone(), self.events.clone(), self.gpu.clone());
                let fit = deploy_svc.fit_check(&DeployRequest {
                    model_id: id,
                    quantization: self.selected_quant.clone(),
                    weight_bytes: weight,
                    weight_files: vec![],
                    download_urls: vec![],
                    local_path: None,
                    backend: self.deploy_backend,
                    port: self.deploy_port,
                    context_length: self.deploy_ctx,
                    continue_despite_oversize: false,
                });
                self.last_fit = Some(fit);
                self.model_detail = Some(detail);
                self.set_status("Model detail loaded", false);
            }
            Err(e) => self.raise_error(e),
        }
    }

    async fn start_deploy(&mut self, continue_oversize: bool) {
        let Some(detail) = &self.model_detail else {
            self.set_status("Select a model detail first (Enter)", true);
            return;
        };
        let quant = self.selected_quant.clone();
        let weight = detail
            .quants
            .iter()
            .find(|q| Some(q.label.as_str()) == quant.as_deref())
            .map(|q| q.total_size)
            .unwrap_or(0);
        let weight_files: Vec<String> = detail
            .quants
            .iter()
            .find(|q| Some(q.label.as_str()) == quant.as_deref())
            .map(|q| q.files.iter().map(|f| f.filename.clone()).collect())
            .unwrap_or_default();

        let download_urls: Vec<String> = if let Some(hf) = &self.hf {
            weight_files
                .iter()
                .map(|f| hf.download_url(&detail.summary.id, f))
                .collect()
        } else {
            vec![]
        };

        let req = DeployRequest {
            model_id: detail.summary.id.clone(),
            quantization: quant,
            weight_bytes: weight,
            weight_files,
            download_urls,
            local_path: None,
            backend: self.deploy_backend,
            port: self.deploy_port,
            context_length: self.deploy_ctx,
            continue_despite_oversize: continue_oversize,
        };

        if self.deploy_cloud {
            self.set_status("Cloud deploy: open Setup to add keys; using local path for now", true);
            // Cloud path would go through CloudOrchestrator
        }

        let svc = DeployService::new(self.registry.clone(), self.events.clone(), self.gpu.clone());
        match svc.deploy(req.clone()).await {
            Ok(_) => {
                self.pending_oversize_deploy = None;
            }
            Err(e)
                if e.code == localcode_core::error::ErrorCode::DeployOversizedWarning
                    && !continue_oversize =>
            {
                self.pending_oversize_deploy = Some(req);
                self.modal = Some(ModalState::warning(
                    "VRAM may be insufficient",
                    format!(
                        "{}\n\nPossible causes:\n• Model larger than free VRAM\n• Other processes using GPU\n\nYou can Continue anyway (never hard-blocked).",
                        e.message
                    ),
                ));
                self.last_error = Some(e);
            }
            Err(e) => self.raise_error(e),
        }
    }

    async fn run_coding_turn(&mut self) {
        let input = self.coding_input.trim().to_string();
        if input.is_empty() {
            return;
        }
        self.coding_input.clear();
        self.coding_transcript.push(format!("you: {input}"));

        let runtime = match self.runtimes.first().cloned() {
            Some(r) => r,
            None => {
                // Fall back to assistant provider as runtime
                let mut r = ActiveRuntime::new(
                    "assistant-provider",
                    localcode_core::runtime::RuntimeKind::OpenAiCompatible,
                    self.config.assistant.base_url.clone(),
                );
                r.model_id = if self.config.assistant.model.is_empty() {
                    Some("openai/gpt-4o-mini".into())
                } else {
                    Some(self.config.assistant.model.clone())
                };
                r.api_key = self.config.assistant_api_key();
                r
            }
        };

        let workspace = self
            .config
            .agent
            .workspace_root
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        let mut agent_cfg = self.config.agent.clone();
        agent_cfg.subagents_enabled = self.subagents_enabled;
        let agent = CodingAgent::new(agent_cfg);
        let mut session =
            localcode_agent::AgentSession::new(workspace, self.subagents_enabled);

        match agent
            .run_turn(
                &mut session,
                &input,
                &runtime,
                runtime.api_key.as_deref(),
            )
            .await
        {
            Ok(reply) => {
                self.coding_transcript.push(format!("agent: {reply}"));
                self.set_status("Agent replied", false);
            }
            Err(e) => {
                self.coding_transcript
                    .push(format!("error: {}: {}", e.code, e.message));
                self.raise_error(e);
            }
        }
    }

    async fn run_bench(&mut self) {
        let runtime = match self.runtimes.first() {
            Some(r) => r.clone(),
            None => {
                self.set_status("Deploy a runtime first", true);
                return;
            }
        };
        let suite = sample_coding_suite();
        let subject = Subject {
            hf_model_id: runtime.model_id.clone().unwrap_or_else(|| "unknown".into()),
            quantization: runtime.quantization.clone().unwrap_or_else(|| "unknown".into()),
            weight_source: "local".into(),
            backend: format!("{:?}", runtime.kind),
            backend_version: "unknown".into(),
            precision_notes: String::new(),
            hardware: serde_json::to_value(&self.gpu).unwrap_or_default(),
        };
        let runner = BenchRunner::new(self.events.clone());
        self.set_status("Running benchmark…", false);
        match runner
            .run(
                &suite,
                subject,
                &runtime.base_url,
                runtime.api_key.as_deref(),
                runtime.model_id.as_deref().unwrap_or("default"),
            )
            .await
        {
            Ok(result) => {
                self.last_bench_summary = format!(
                    "score={:.2} pass_rate={:.0}% p50={}ms",
                    result.metrics.score,
                    result.metrics.pass_rate * 100.0,
                    result.metrics.latency_p50_ms
                );
                self.last_bench_result = Some(result);
                self.set_status("Benchmark complete", false);
            }
            Err(e) => self.raise_error(e),
        }
    }

    async fn ask_assistant(&mut self) {
        let assistant = Assistant::new(self.config.assistant.clone());
        let error_context = self.last_error.as_ref().map(|e| e.assistant_context());
        let logs = localcode_log::read_recent_logs(
            &self.paths.log_dir,
            80,
            self.last_error
                .as_ref()
                .map(|e| e.correlation_id.to_string())
                .as_deref(),
            self.config.logging.redact_secrets,
        )
        .ok();
        let doctor = run_doctor(&self.paths, &self.config).await;
        let req = AssistantRequest {
            user_message: self
                .last_error
                .as_ref()
                .map(|e| format!("Help me fix: {}", e.message))
                .unwrap_or_else(|| "Help me diagnose LocalCode setup.".into()),
            error_context,
            doctor_report: Some(doctor),
            recent_logs: logs,
            config_snapshot_redacted: Some(serde_json::json!({
                "backends": self.config.backends.default.kind,
                "registry": self.config.registry.endpoint,
                "api": self.config.api.base_url,
            })),
        };
        match assistant
            .ask(req, self.config.assistant_api_key().as_deref())
            .await
        {
            Ok(reply) => {
                self.assistant_reply = Some(reply.message);
                self.assistant_open = true;
                self.set_status("Assistant replied", false);
            }
            Err(e) => self.raise_error(e),
        }
    }

    async fn handle_key(&mut self, key: crossterm::event::KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }

        // Global
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('k') {
            self.palette_open = !self.palette_open;
            self.palette_query.clear();
            self.palette_selected = 0;
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.should_quit = true;
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('s') {
            if let Err(e) = self.config.save(&self.paths) {
                self.raise_error(e);
            } else {
                self.set_status("Config saved", false);
            }
            return;
        }

        if self.palette_open {
            self.handle_palette_key(key).await;
            return;
        }

        if self.assistant_open {
            if matches!(key.code, KeyCode::Esc | KeyCode::Char('q')) {
                self.assistant_open = false;
            }
            return;
        }

        if let Some(modal) = self.modal.clone() {
            self.handle_modal_key(key, &modal).await;
            return;
        }

        // Search focus on models
        if self.model_search_focus && self.tab == Tab::Models {
            match key.code {
                KeyCode::Esc => self.model_search_focus = false,
                KeyCode::Enter => {
                    self.model_search_focus = false;
                    self.search_models().await;
                }
                KeyCode::Backspace => {
                    self.model_query.pop();
                }
                KeyCode::Char(c) => self.model_query.push(c),
                _ => {}
            }
            return;
        }

        if self.coding_input_focus && self.tab == Tab::Coding {
            match key.code {
                KeyCode::Esc => self.coding_input_focus = false,
                KeyCode::Enter => {
                    self.run_coding_turn().await;
                }
                KeyCode::Backspace => {
                    self.coding_input.pop();
                }
                KeyCode::Char(c) => self.coding_input.push(c),
                _ => {}
            }
            return;
        }

        // Tab switching
        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('1') => self.set_tab(Tab::Dashboard),
            KeyCode::Char('2') => self.set_tab(Tab::Models),
            KeyCode::Char('3') => self.set_tab(Tab::Benchmarks),
            KeyCode::Char('4') => self.set_tab(Tab::Coding),
            KeyCode::Char('5') => self.set_tab(Tab::Setup),
            KeyCode::Char('6') => self.set_tab(Tab::Notifications),
            KeyCode::Char('7') => self.set_tab(Tab::Settings),
            KeyCode::Tab => {
                let idx = (self.rail_index + 1) % 7;
                self.rail_index = idx;
                self.rail_focus = true;
                self.set_tab(Tab::from_index(idx));
            }
            KeyCode::BackTab => {
                let idx = if self.rail_index == 0 {
                    6
                } else {
                    self.rail_index - 1
                };
                self.rail_index = idx;
                self.rail_focus = true;
                self.set_tab(Tab::from_index(idx));
            }
            KeyCode::Char('j') if self.rail_focus => {
                self.rail_index = (self.rail_index + 1) % 7;
                self.set_tab(Tab::from_index(self.rail_index));
            }
            KeyCode::Char('k') if self.rail_focus => {
                self.rail_index = if self.rail_index == 0 {
                    6
                } else {
                    self.rail_index - 1
                };
                self.set_tab(Tab::from_index(self.rail_index));
            }
            KeyCode::Char('a') => {
                let _ = self.ask_assistant().await;
            }
            KeyCode::Char('l') => {
                self.set_status(
                    format!("Logs: {}", self.paths.log_dir.display()),
                    false,
                );
            }
            other => self.handle_tab_key(other).await,
        }
    }

    fn set_tab(&mut self, tab: Tab) {
        self.tab = tab;
        self.rail_index = Tab::all().iter().position(|t| *t == tab).unwrap_or(0);
        self.rail_focus = true;
    }

    async fn handle_tab_key(&mut self, code: KeyCode) {
        match self.tab {
            Tab::Models => match code {
                KeyCode::Char('/') => {
                    self.model_search_focus = true;
                    self.rail_focus = false;
                }
                KeyCode::Char('p') => {
                    self.model_query = "code".into();
                    self.search_models().await;
                }
                KeyCode::Char('t') => {
                    if let Some(hf) = &self.hf {
                        match hf.trending_coding(30).await {
                            Ok(m) => {
                                self.models = m;
                                self.set_status("Trending coding models", false);
                            }
                            Err(e) => self.raise_error(e),
                        }
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if !self.models.is_empty() {
                        self.model_selected =
                            (self.model_selected + 1).min(self.models.len() - 1);
                    }
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.model_selected = self.model_selected.saturating_sub(1);
                }
                KeyCode::Enter => self.load_model_detail().await,
                KeyCode::Char('d') => self.start_deploy(false).await,
                KeyCode::Char('b') => {
                    self.deploy_backend = match self.deploy_backend {
                        BackendKind::Ollama => BackendKind::LlamaCpp,
                        BackendKind::LlamaCpp => BackendKind::Vllm,
                        BackendKind::Vllm => BackendKind::Sglang,
                        BackendKind::Sglang => BackendKind::Ollama,
                    };
                }
                KeyCode::Char('c') => self.deploy_cloud = !self.deploy_cloud,
                KeyCode::Char('[') => self.adjust_pane("models", -0.05),
                KeyCode::Char(']') => self.adjust_pane("models", 0.05),
                _ => {}
            },
            Tab::Coding => match code {
                KeyCode::Char('i') | KeyCode::Enter => {
                    self.coding_input_focus = true;
                    self.rail_focus = false;
                }
                KeyCode::Char('s') => {
                    self.subagents_enabled = !self.subagents_enabled;
                    self.set_status(
                        format!(
                            "Subagents {}",
                            if self.subagents_enabled { "ON" } else { "OFF" }
                        ),
                        false,
                    );
                }
                _ => {}
            },
            Tab::Benchmarks => match code {
                KeyCode::Char('r') => self.run_bench().await,
                KeyCode::Char('p') => {
                    self.set_status(
                        "Publish requires sign-in (Setup → Account)",
                        true,
                    );
                }
                _ => {}
            },
            Tab::Setup => {
                if code == KeyCode::Char('d') {
                    let report = run_doctor(&self.paths, &self.config).await;
                    self.doctor_summary = Some(
                        serde_json::to_string_pretty(&report).unwrap_or_default(),
                    );
                    self.set_status("Doctor complete", false);
                }
            }
            Tab::Settings => {
                if code == KeyCode::Char('t') {
                    self.config.ui.theme = match self.config.ui.theme {
                        ThemeMode::Dark => ThemeMode::Light,
                        ThemeMode::Light => ThemeMode::HighContrast,
                        ThemeMode::HighContrast => ThemeMode::Dark,
                    };
                    self.theme = Theme::new(self.config.ui.theme);
                }
            }
            _ => {}
        }
    }

    fn adjust_pane(&mut self, view: &str, delta: f32) {
        let mut ratios = self.pane_ratios(view, &[0.3, 0.4, 0.3]);
        if ratios.len() >= 2 {
            ratios[0] = (ratios[0] + delta).clamp(0.15, 0.6);
            ratios[1] = (ratios[1] - delta).clamp(0.15, 0.6);
            // normalize
            let sum: f32 = ratios.iter().sum();
            for r in &mut ratios {
                *r /= sum;
            }
            self.config.panes.views.insert(view.into(), ratios);
        }
    }

    async fn handle_modal_key(&mut self, key: crossterm::event::KeyEvent, modal: &ModalState) {
        let n = modal.buttons().len();
        match key.code {
            KeyCode::Left | KeyCode::Char('h') => {
                if let Some(m) = &mut self.modal {
                    m.selected = m.selected.saturating_sub(1);
                }
            }
            KeyCode::Right | KeyCode::Char('l') | KeyCode::Tab => {
                if let Some(m) = &mut self.modal {
                    m.selected = (m.selected + 1).min(n.saturating_sub(1));
                }
            }
            KeyCode::Esc => {
                self.modal = None;
            }
            KeyCode::Enter => {
                let selected = modal.selected;
                let kind = modal.kind.clone();
                self.modal = None;
                match kind {
                    ModalKind::Error { error } => match selected {
                        0 if error.retryable => {
                            self.set_status("Retry from the view that failed", false);
                        }
                        1 => {
                            self.set_status(
                                format!("Logs: {}", self.paths.log_dir.display()),
                                false,
                            );
                        }
                        2 => self.ask_assistant().await,
                        _ => {}
                    },
                    ModalKind::Warning { .. } => {
                        if selected == 0 {
                            // Continue oversize deploy
                            if self.pending_oversize_deploy.is_some() {
                                self.start_deploy(true).await;
                            }
                        }
                    }
                    ModalKind::Confirm { .. } | ModalKind::Payment { .. } => {
                        if selected == 0 {
                            self.set_status("Confirmed", false);
                        }
                    }
                    ModalKind::Info { .. } => {}
                }
            }
            _ => {}
        }
    }

    async fn handle_palette_key(&mut self, key: crossterm::event::KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.palette_open = false;
            }
            KeyCode::Up => {
                self.palette_selected = self.palette_selected.saturating_sub(1);
            }
            KeyCode::Down => {
                let n = self.palette_items().len();
                if n > 0 {
                    self.palette_selected = (self.palette_selected + 1).min(n - 1);
                }
            }
            KeyCode::Backspace => {
                self.palette_query.pop();
            }
            KeyCode::Enter => {
                let items = self.palette_items();
                if let Some(item) = items.get(self.palette_selected).cloned() {
                    self.palette_open = false;
                    self.run_palette_action(&item).await;
                }
            }
            KeyCode::Char(c) => {
                self.palette_query.push(c);
                self.palette_selected = 0;
            }
            _ => {}
        }
    }

    async fn run_palette_action(&mut self, item: &str) {
        match item {
            "Go: dashboard" => self.set_tab(Tab::Dashboard),
            "Go: models" => self.set_tab(Tab::Models),
            "Go: benchmarks" => self.set_tab(Tab::Benchmarks),
            "Go: coding" => self.set_tab(Tab::Coding),
            "Go: setup" => self.set_tab(Tab::Setup),
            "Go: notifications" => self.set_tab(Tab::Notifications),
            "Go: settings" => self.set_tab(Tab::Settings),
            "Deploy last model" => self.start_deploy(false).await,
            "Toggle subagents" => {
                self.subagents_enabled = !self.subagents_enabled;
            }
            "Open logs" => {
                self.set_status(format!("Logs: {}", self.paths.log_dir.display()), false);
            }
            "Run doctor" => {
                let report = run_doctor(&self.paths, &self.config).await;
                self.doctor_summary =
                    Some(serde_json::to_string_pretty(&report).unwrap_or_default());
                self.set_tab(Tab::Setup);
            }
            "Ask assistant" => self.ask_assistant().await,
            "Quit" => self.should_quit = true,
            _ => {}
        }
    }

    fn handle_mouse(&mut self, mouse: crossterm::event::MouseEvent, term_width: u16) {
        // Right rail is last 14 columns
        let rail_x = term_width.saturating_sub(14);
        self.rail_hover = mouse.column >= rail_x;

        if matches!(mouse.kind, MouseEventKind::Down(event::MouseButton::Left))
            && mouse.column >= rail_x
        {
            // Approximate row → tab (title takes 3, then list)
            let row = mouse.row.saturating_sub(1);
            if (row as usize) < 7 {
                self.rail_index = row as usize;
                self.set_tab(Tab::from_index(self.rail_index));
            }
        }
    }
}

pub async fn run_tui(paths: AppPaths, config: Config) -> Result<(), LocalCodeError> {
    info!("starting TUI");
    enable_raw_mode().map_err(|e| LocalCodeError::from(e))?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)
        .map_err(|e| LocalCodeError::from(e))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(|e| LocalCodeError::from(e))?;

    let mut app = App::new(paths, config).await;
    let result = run_loop(&mut terminal, &mut app).await;

    disable_raw_mode().ok();
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )
    .ok();
    terminal.show_cursor().ok();
    result
}

async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<(), LocalCodeError> {
    loop {
        app.process_events().await;
        terminal
            .draw(|f| ui::draw(f, app))
            .map_err(|e| LocalCodeError::from(e))?;

        if app.should_quit {
            break;
        }

        if event::poll(Duration::from_millis(100)).map_err(|e| LocalCodeError::from(e))? {
            match event::read().map_err(|e| LocalCodeError::from(e))? {
                Event::Key(key) => app.handle_key(key).await,
                Event::Mouse(m) => {
                    let w = terminal.size().map(|s| s.width).unwrap_or(80);
                    app.handle_mouse(m, w);
                }
                Event::Resize(_, _) => {
                    // Layout recomputes on next draw
                }
                _ => {}
            }
        }
    }
    // Persist pane ratios etc.
    let _ = app.config.save(&app.paths);
    Ok(())
}
