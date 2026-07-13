//! Application state and event loop.
//!
//! Architecture: the event loop never awaits long-running work. Every slow
//! operation (HF search, deploys, agent turns, benchmarks, assistant calls)
//! runs in a spawned task and reports back through an mpsc channel of
//! [`BgMsg`]. The loop keeps drawing at ~10fps, so progress is live, input
//! stays responsive, and Esc can cancel by aborting the task.

use crate::doctor::run_doctor;
use crate::ui;
use crate::widgets::{ConfirmAction, ModalKind, ModalState};
use async_trait::async_trait;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use localcode_agent::{AgentEvent, AgentSession, CodingAgent, ToolApprover};
use localcode_api_client::ApiClient;
use localcode_assistant::{Assistant, AssistantRequest};
use localcode_backends::{
    BackendKind, BackendRegistry, DeployRequest, DeployService, DetectReport,
};
use localcode_bench::{sample_coding_suite, BenchRunner, Subject};
use localcode_core::config::Config;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus, Severity};
use localcode_core::paths::AppPaths;
use localcode_core::runtime::ActiveRuntime;
use localcode_core::theme::{Theme, ThemeMode};
use localcode_gpu::{discover, predict_fit, FitPrediction, FitRequest, GpuInventory};
use localcode_hf::{HfClient, ModelDetail, ModelSummary};
use localcode_upgrade::{SelfUpdater, UpdateChecker, UpdateInfo, UpdateReport};
use ratatui::backend::CrosstermBackend;
use ratatui::widgets::ListState;
use ratatui::Terminal;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex};
use tokio::task::JoinHandle;
use tracing::info;

const MAX_NOTIFICATIONS: usize = 200;
const MAX_INPUT_HISTORY: usize = 100;

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

    pub fn index(self) -> usize {
        Self::all().iter().position(|t| *t == self).unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct NotificationItem {
    pub severity: Severity,
    pub title: String,
    pub body: String,
    pub at: chrono::DateTime<chrono::Local>,
}

/// Who said a transcript entry, and how to style it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    You,
    Agent,
    Tool,
    System,
    Error,
}

#[derive(Debug, Clone)]
pub struct TranscriptEntry {
    pub kind: EntryKind,
    pub text: String,
    /// Still receiving streamed text (Agent) or still running (Tool).
    pub live: bool,
}

impl TranscriptEntry {
    pub fn new(kind: EntryKind, text: impl Into<String>) -> Self {
        Self {
            kind,
            text: text.into(),
            live: false,
        }
    }
}

/// Which pane owns navigation keys on the Models tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelsPane {
    List,
    Card,
}

/// Memoized rendered model card — markdown is parsed once per model/theme,
/// not on every frame of the ~10fps draw loop.
pub struct CardCache {
    pub model_id: String,
    pub mode: ThemeMode,
    pub lines: Vec<ratatui::text::Line<'static>>,
}

/// Kinds of foreground background-work (one at a time; Esc cancels).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusyKind {
    Search,
    Detail,
    Coding,
    Bench,
    Assistant,
    Doctor,
    Update,
}

pub struct Busy {
    pub kind: BusyKind,
    pub label: String,
    pub started: Instant,
    handle: JoinHandle<()>,
}

/// The last failed operation, so the error modal's Retry button can actually
/// retry instead of telling the user to do it themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryAction {
    SearchModels,
    LoadDetail,
    Deploy,
    RunBench,
    AskAssistant,
    CheckUpdate,
    InstallUpdate,
}

pub enum DeployOutcome {
    Done,
    Oversize(Box<(DeployRequest, LocalCodeError)>),
}

/// Results and requests flowing from spawned tasks back to the UI loop.
pub enum BgMsg {
    SearchDone(Result<Vec<ModelSummary>, LocalCodeError>),
    DetailDone(Result<ModelDetail, LocalCodeError>),
    DeployEnded(DeployOutcome),
    /// Live progress from a running coding turn (streamed tokens, tools).
    CodingEvent(AgentEvent),
    CodingDone(Result<String, LocalCodeError>),
    ToolConfirm {
        description: String,
        respond: oneshot::Sender<bool>,
    },
    BenchDone(Result<localcode_bench::RunResult, LocalCodeError>),
    AssistantDone(Result<String, LocalCodeError>),
    DoctorDone(String),
    DetectDone(Vec<DetectReport>),
    ApiHealth(bool),
    RuntimeStopped {
        result: Result<(), LocalCodeError>,
    },
    UpdateCheckDone {
        result: Result<Option<UpdateInfo>, LocalCodeError>,
        /// Manual checks report "up to date" and errors; startup checks stay quiet.
        manual: bool,
    },
    UpdateProgress(String),
    UpdateDone(Result<UpdateReport, LocalCodeError>),
}

/// Bridges the agent's destructive-command approval to a TUI modal.
struct ChannelApprover {
    tx: mpsc::UnboundedSender<BgMsg>,
}

#[async_trait]
impl ToolApprover for ChannelApprover {
    async fn approve(&self, description: &str) -> bool {
        let (respond, rx) = oneshot::channel();
        if self
            .tx
            .send(BgMsg::ToolConfirm {
                description: description.to_string(),
                respond,
            })
            .is_err()
        {
            return false;
        }
        rx.await.unwrap_or(false)
    }
}

pub struct App {
    pub paths: AppPaths,
    pub config: Config,
    pub theme: Theme,
    pub tab: Tab,
    /// Tab-strip hit ranges `(x_start, x_end_exclusive, tab)` and row, filled
    /// during draw so mouse handling matches what is actually on screen.
    pub tab_hit: Vec<(u16, u16, Tab)>,
    pub tab_strip_row: u16,
    pub tab_hover: Option<Tab>,
    pub status_line: String,
    pub status_is_error: bool,
    pub last_error: Option<LocalCodeError>,
    pub last_failed_action: Option<RetryAction>,
    pub modal: Option<ModalState>,
    pending_tool_confirm: Option<oneshot::Sender<bool>>,
    pub palette_open: bool,
    pub palette_query: String,
    pub palette_selected: usize,
    pub assistant_open: bool,
    pub assistant_reply: Option<String>,
    pub assistant_scroll: u16,
    pub assistant_configured: bool,
    /// None while the startup probe is still running.
    pub api_healthy: Option<bool>,
    pub gpu: GpuInventory,
    pub runtimes: Vec<ActiveRuntime>,
    pub runtime_selected: usize,
    pub runtime_list_state: ListState,
    pub notifications: Vec<NotificationItem>,
    pub notif_selected: usize,
    pub notif_list_state: ListState,
    pub backend_reports: Vec<DetectReport>,
    pub detecting: bool,
    pub doctor_summary: Option<String>,
    pub setup_scroll: u16,
    // Models
    pub model_query: String,
    pub model_search_focus: bool,
    pub models: Vec<ModelSummary>,
    pub model_selected: usize,
    pub model_list_state: ListState,
    pub model_detail: Option<ModelDetail>,
    /// Which Models pane owns j/k/PgUp/PgDn (Left/Right switches).
    pub models_focus: ModelsPane,
    pub card_cache: Option<CardCache>,
    pub card_scroll: usize,
    /// Updated during draw so scrolling knows the bounds.
    pub card_view_height: u16,
    pub card_total_lines: usize,
    pub selected_quant: Option<String>,
    pub deploy_backend: BackendKind,
    pub deploy_ctx: u32,
    pub deploy_port: Option<u16>,
    pub deploy_progress: u8,
    pub last_fit: Option<FitPrediction>,
    pub pending_oversize_deploy: Option<DeployRequest>,
    // Coding
    pub coding_input: String,
    pub coding_cursor: usize, // char index into coding_input
    pub coding_input_focus: bool,
    pub coding_history: Vec<String>,
    coding_hist_idx: Option<usize>,
    pub coding_transcript: Vec<TranscriptEntry>,
    pub coding_scroll: usize,
    pub coding_follow: bool,
    /// Updated during draw so PgUp/PgDn know the scroll bounds.
    pub coding_view_height: u16,
    pub coding_total_lines: usize,
    pub skill_count: usize,
    pub mcp_status_summary: String,
    // Bench
    pub last_bench_summary: String,
    pub last_bench_result: Option<localcode_bench::RunResult>,
    // Services
    events: EventBus,
    registry: Arc<BackendRegistry>,
    hf: Option<HfClient>,
    session: Arc<AsyncMutex<AgentSession>>,
    bg_tx: mpsc::UnboundedSender<BgMsg>,
    bg_rx: mpsc::UnboundedReceiver<BgMsg>,
    pub busy: Option<Busy>,
    pub deploy_busy: Option<Busy>,
    // Updates
    pub update_available: Option<UpdateInfo>,
    pub update_busy: Option<Busy>,
    pub update_progress_line: String,
    /// Version installed this session; restart required to run it.
    pub update_installed: Option<String>,
    should_quit: bool,
}

impl App {
    pub fn new(paths: AppPaths, config: Config) -> Self {
        let events = EventBus::new();
        let gpu = discover().unwrap_or_else(|_| GpuInventory {
            devices: vec![],
            detection_method: "none".into(),
            warnings: vec!["GPU discovery failed".into()],
        });
        let registry = Arc::new(BackendRegistry::from_config(&config));
        let hf = HfClient::new(
            &config.registry,
            config.hf_token(),
            paths.models_cache.clone(),
        )
        .ok();

        let assistant = Assistant::new(config.assistant.clone());
        let assistant_configured = assistant.is_configured(config.assistant_api_key().as_deref());

        let agent_probe = CodingAgent::new(config.agent.clone());
        let skill_count = agent_probe.skills.list().len();
        let mcp_count = agent_probe.mcp.configured_count();
        let mcp_status_summary = if mcp_count == 0 {
            "none configured".into()
        } else {
            format!("{mcp_count} configured (not connected)")
        };

        let workspace = config
            .agent
            .workspace_root
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let session = Arc::new(AsyncMutex::new(AgentSession::new(workspace, false)));

        let (bg_tx, bg_rx) = mpsc::unbounded_channel();

        let mut app = Self {
            theme: Theme::new(config.ui.theme),
            paths,
            tab: Tab::Dashboard,
            tab_hit: vec![],
            tab_strip_row: 0,
            tab_hover: None,
            status_line: "Ready — one-click local models, no account required".into(),
            status_is_error: false,
            last_error: None,
            last_failed_action: None,
            modal: None,
            pending_tool_confirm: None,
            palette_open: false,
            palette_query: String::new(),
            palette_selected: 0,
            assistant_open: false,
            assistant_reply: None,
            assistant_scroll: 0,
            assistant_configured,
            api_healthy: None,
            gpu,
            runtimes: vec![],
            runtime_selected: 0,
            runtime_list_state: ListState::default(),
            notifications: vec![],
            notif_selected: 0,
            notif_list_state: ListState::default(),
            backend_reports: vec![],
            detecting: false,
            doctor_summary: None,
            setup_scroll: 0,
            model_query: String::new(),
            model_search_focus: false,
            models: vec![],
            model_selected: 0,
            model_list_state: ListState::default(),
            model_detail: None,
            models_focus: ModelsPane::List,
            card_cache: None,
            card_scroll: 0,
            card_view_height: 0,
            card_total_lines: 0,
            selected_quant: None,
            deploy_backend: BackendKind::parse(&config.backends.default.kind)
                .unwrap_or(BackendKind::Ollama),
            deploy_ctx: 8192,
            deploy_port: None,
            deploy_progress: 0,
            last_fit: None,
            pending_oversize_deploy: None,
            coding_input: String::new(),
            coding_cursor: 0,
            coding_input_focus: false,
            coding_history: vec![],
            coding_hist_idx: None,
            coding_transcript: vec![TranscriptEntry::new(
                EntryKind::System,
                "Welcome to LocalCode Coding. Deploy a runtime ([2] Models), then type a message.",
            )],
            coding_scroll: 0,
            coding_follow: true,
            coding_view_height: 0,
            coding_total_lines: 0,
            skill_count,
            mcp_status_summary,
            last_bench_summary: "No runs yet.".into(),
            last_bench_result: None,
            events,
            registry,
            hf,
            session,
            bg_tx,
            bg_rx,
            busy: None,
            deploy_busy: None,
            update_available: None,
            update_busy: None,
            update_progress_line: String::new(),
            update_installed: None,
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

    pub fn active_runtime(&self) -> Option<&ActiveRuntime> {
        self.runtimes
            .get(self.runtime_selected)
            .or_else(|| self.runtimes.first())
    }

    pub fn active_runtime_name(&self) -> Option<String> {
        self.active_runtime().map(|r| r.name.clone())
    }

    fn workspace_path(&self) -> PathBuf {
        self.config
            .agent
            .workspace_root
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }

    /// Pane ratios for a view, validated: exactly `default.len()` finite,
    /// positive entries — anything else (e.g. a hand-edited config) falls back
    /// to the default instead of panicking the draw loop.
    pub fn pane_ratios(&self, view: &str, default: &[f32]) -> Vec<f32> {
        let ratios = self
            .config
            .panes
            .views
            .get(view)
            .cloned()
            .unwrap_or_else(|| default.to_vec());
        let valid = ratios.len() == default.len()
            && ratios.iter().all(|r| r.is_finite() && *r > 0.0);
        if valid {
            ratios
        } else {
            default.to_vec()
        }
    }

    // ------------------------------------------------------------------
    // Busy management
    // ------------------------------------------------------------------

    pub fn fg_busy(&self) -> bool {
        self.busy.is_some()
    }

    fn begin_busy(&mut self, kind: BusyKind, label: impl Into<String>, handle: JoinHandle<()>) {
        self.busy = Some(Busy {
            kind,
            label: label.into(),
            started: Instant::now(),
            handle,
        });
    }

    fn finish_busy(&mut self) {
        self.busy = None;
    }

    /// Esc: cancel the foreground task, else a running deploy.
    fn cancel_current(&mut self) {
        // A pending tool approval belongs to the running turn — deny it first.
        if matches!(
            self.modal,
            Some(ModalState {
                kind: ModalKind::Confirm {
                    action: ConfirmAction::ToolApproval,
                    ..
                },
                ..
            })
        ) {
            self.respond_tool_confirm(false);
            self.modal = None;
        }
        if let Some(b) = self.busy.take() {
            b.handle.abort();
            if b.kind == BusyKind::Coding {
                // Keep whatever streamed before the cancel; just close it out.
                self.finalize_transcript_live();
                self.coding_transcript
                    .push(TranscriptEntry::new(EntryKind::System, "turn cancelled"));
                self.coding_follow = true;
            }
            self.set_status(format!("Cancelled: {}", b.label), false);
        } else if let Some(b) = self.deploy_busy.take() {
            b.handle.abort();
            self.deploy_progress = 0;
            self.set_status("Deploy cancelled", false);
        } else if let Some(b) = self.update_busy.take() {
            // Safe: the binary swap happens only in the final instant of the
            // update task; aborting mid-fetch/mid-build changes nothing.
            b.handle.abort();
            self.update_progress_line.clear();
            self.set_status("Update cancelled — installed binary untouched", false);
        }
    }

    /// Close any live transcript entries (stream ended, cancelled, or errored).
    fn finalize_transcript_live(&mut self) {
        self.coding_transcript
            .retain(|e| !(e.live && e.kind == EntryKind::Agent && e.text.trim().is_empty()));
        for e in &mut self.coding_transcript {
            e.live = false;
        }
    }

    /// Apply one live agent event to the transcript.
    fn apply_agent_event(&mut self, ev: AgentEvent) {
        match ev {
            AgentEvent::Delta(text) => {
                match self.coding_transcript.last_mut() {
                    Some(e) if e.kind == EntryKind::Agent && e.live => e.text.push_str(&text),
                    _ => self.coding_transcript.push(TranscriptEntry {
                        kind: EntryKind::Agent,
                        text,
                        live: true,
                    }),
                }
                self.coding_follow = true;
            }
            AgentEvent::MessageComplete => {
                if let Some(e) = self.coding_transcript.last_mut() {
                    if e.kind == EntryKind::Agent && e.live {
                        if e.text.trim().is_empty() {
                            self.coding_transcript.pop();
                        } else {
                            e.live = false;
                        }
                    }
                }
            }
            AgentEvent::ToolStarted { name, args_preview } => {
                self.coding_transcript.push(TranscriptEntry {
                    kind: EntryKind::Tool,
                    text: format!("⚙ {name} {args_preview}"),
                    live: true,
                });
                self.coding_follow = true;
            }
            AgentEvent::ToolFinished { name, ok, summary } => {
                let mark = if ok { "✓" } else { "✗" };
                let text = format!("{mark} {name} — {summary}");
                match self
                    .coding_transcript
                    .iter_mut()
                    .rev()
                    .find(|e| e.kind == EntryKind::Tool && e.live)
                {
                    Some(e) => {
                        e.text = text;
                        e.live = false;
                    }
                    None => self
                        .coding_transcript
                        .push(TranscriptEntry::new(EntryKind::Tool, text)),
                }
                self.coding_follow = true;
            }
        }
    }

    fn respond_tool_confirm(&mut self, approved: bool) {
        if let Some(tx) = self.pending_tool_confirm.take() {
            let _ = tx.send(approved);
        }
    }

    // ------------------------------------------------------------------
    // Error surface: status + notification (modal only on demand via `e`)
    // ------------------------------------------------------------------

    fn raise_error(&mut self, err: LocalCodeError) {
        self.status_line = format!("{}: {} — [e] details", err.code, err.message);
        self.status_is_error = true;
        self.push_notification(Severity::Error, err.code.as_str(), &err.message);
        self.last_error = Some(err);
    }

    fn push_notification(&mut self, severity: Severity, title: &str, body: &str) {
        self.notifications.push(NotificationItem {
            severity,
            title: title.to_string(),
            body: body.to_string(),
            at: chrono::Local::now(),
        });
        if self.notifications.len() > MAX_NOTIFICATIONS {
            let excess = self.notifications.len() - MAX_NOTIFICATIONS;
            self.notifications.drain(0..excess);
        }
    }

    fn set_status(&mut self, msg: impl Into<String>, is_error: bool) {
        self.status_line = msg.into();
        self.status_is_error = is_error;
    }

    fn open_error_modal(&mut self) {
        if let Some(err) = self.last_error.clone() {
            self.modal = Some(ModalState::error(err));
        } else {
            self.set_status("No recent error", false);
        }
    }

    fn retry_last(&mut self) {
        match self.last_failed_action {
            Some(RetryAction::SearchModels) => self.start_search(),
            Some(RetryAction::LoadDetail) => self.start_load_detail(),
            Some(RetryAction::Deploy) => self.start_deploy(false),
            Some(RetryAction::RunBench) => self.start_bench(),
            Some(RetryAction::AskAssistant) => self.start_assistant(),
            Some(RetryAction::CheckUpdate) => self.start_update_check(true),
            Some(RetryAction::InstallUpdate) => self.start_install_update(),
            None => self.set_status("Nothing to retry", false),
        }
    }

    // ------------------------------------------------------------------
    // Background task starters (all non-blocking)
    // ------------------------------------------------------------------

    pub fn start_detect(&mut self) {
        self.detecting = true;
        let registry = self.registry.clone();
        let tx = self.bg_tx.clone();
        tokio::spawn(async move {
            let reports = registry.detect_all().await;
            let _ = tx.send(BgMsg::DetectDone(reports));
        });
        let api_cfg = self.config.api.clone();
        let tx = self.bg_tx.clone();
        tokio::spawn(async move {
            let healthy = match ApiClient::new(&api_cfg, None) {
                Ok(api) => api.health().await.is_ok(),
                Err(_) => false,
            };
            let _ = tx.send(BgMsg::ApiHealth(healthy));
        });
    }

    fn start_search(&mut self) {
        if self.fg_busy() {
            self.set_status("Busy — Esc to cancel first", false);
            return;
        }
        let Some(hf) = self.hf.clone() else {
            self.set_status("HF client unavailable", true);
            return;
        };
        let q = if self.model_query.is_empty() {
            "code".to_string()
        } else {
            self.model_query.clone()
        };
        let tx = self.bg_tx.clone();
        let label = format!("Searching HF for '{q}'");
        let handle = tokio::spawn(async move {
            let result = hf.search(&q, true, 30, "downloads").await;
            let _ = tx.send(BgMsg::SearchDone(result));
        });
        self.begin_busy(BusyKind::Search, label, handle);
    }

    fn start_trending(&mut self) {
        if self.fg_busy() {
            self.set_status("Busy — Esc to cancel first", false);
            return;
        }
        let Some(hf) = self.hf.clone() else {
            self.set_status("HF client unavailable", true);
            return;
        };
        let tx = self.bg_tx.clone();
        let handle = tokio::spawn(async move {
            let result = hf.trending_coding(30).await;
            let _ = tx.send(BgMsg::SearchDone(result));
        });
        self.begin_busy(BusyKind::Search, "Loading trending coding models", handle);
    }

    fn start_load_detail(&mut self) {
        if self.fg_busy() {
            self.set_status("Busy — Esc to cancel first", false);
            return;
        }
        let Some(id) = self.models.get(self.model_selected).map(|m| m.id.clone()) else {
            return;
        };
        let Some(hf) = self.hf.clone() else {
            return;
        };
        let tx = self.bg_tx.clone();
        let label = format!("Loading {id}");
        let handle = tokio::spawn(async move {
            let result = hf.model_info(&id).await;
            let _ = tx.send(BgMsg::DetailDone(result));
        });
        self.begin_busy(BusyKind::Detail, label, handle);
    }

    /// Recompute the VRAM fit estimate from current selections. Cheap & sync.
    fn refresh_fit(&mut self) {
        let Some(detail) = &self.model_detail else {
            self.last_fit = None;
            return;
        };
        let weight = detail
            .quants
            .iter()
            .find(|q| Some(q.label.as_str()) == self.selected_quant.as_deref())
            .map(|q| q.total_size)
            .unwrap_or(0);
        self.last_fit = Some(predict_fit(
            &self.gpu,
            &FitRequest {
                weight_bytes: weight,
                param_count: None,
                quant_label: self.selected_quant.clone(),
                context_length: self.deploy_ctx,
                backend: self.deploy_backend.as_str().into(),
            },
        ));
    }

    fn cycle_quant(&mut self, delta: i32) {
        let Some(detail) = &self.model_detail else {
            self.set_status("Open a model detail first (Enter)", false);
            return;
        };
        if detail.quants.is_empty() {
            return;
        }
        let cur = detail
            .quants
            .iter()
            .position(|q| Some(q.label.as_str()) == self.selected_quant.as_deref())
            .unwrap_or(0);
        let n = detail.quants.len() as i32;
        let next = ((cur as i32 + delta) % n + n) % n;
        self.selected_quant = Some(detail.quants[next as usize].label.clone());
        self.refresh_fit();
    }

    fn adjust_ctx(&mut self, up: bool) {
        self.deploy_ctx = if up {
            (self.deploy_ctx.saturating_mul(2)).min(131_072)
        } else {
            (self.deploy_ctx / 2).max(2_048)
        };
        self.refresh_fit();
    }

    fn build_deploy_request(&mut self, continue_oversize: bool) -> Option<DeployRequest> {
        let Some(detail) = &self.model_detail else {
            self.set_status("Select a model detail first (Enter)", true);
            return None;
        };
        let quant = self.selected_quant.clone();
        let group = detail
            .quants
            .iter()
            .find(|q| Some(q.label.as_str()) == quant.as_deref());
        let weight = group.map(|q| q.total_size).unwrap_or(0);
        let weight_files: Vec<String> = group
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

        Some(DeployRequest {
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
        })
    }

    fn start_deploy(&mut self, continue_oversize: bool) {
        let Some(req) = self.build_deploy_request(continue_oversize) else {
            return;
        };
        self.spawn_deploy(req);
    }

    fn spawn_deploy(&mut self, req: DeployRequest) {
        if self.deploy_busy.is_some() {
            self.set_status("A deploy is already running (Esc to cancel it)", false);
            return;
        }
        let svc = DeployService::new(
            self.registry.clone(),
            self.events.clone(),
            self.gpu.clone(),
            self.paths.models_cache.clone(),
            self.config.hf_token(),
        );
        let tx = self.bg_tx.clone();
        let label = format!("Deploying {}", req.model_id);
        self.deploy_progress = 0;
        let handle = tokio::spawn(async move {
            let continue_oversize = req.continue_despite_oversize;
            let outcome = match svc.deploy(req.clone()).await {
                Ok(_) => DeployOutcome::Done,
                Err(e)
                    if e.code == ErrorCode::DeployOversizedWarning && !continue_oversize =>
                {
                    DeployOutcome::Oversize(Box::new((req, e)))
                }
                // Other failures already published a DeployFailed event.
                Err(_) => DeployOutcome::Done,
            };
            let _ = tx.send(BgMsg::DeployEnded(outcome));
        });
        self.deploy_busy = Some(Busy {
            kind: BusyKind::Detail, // unused for deploy; label is what shows
            label,
            started: Instant::now(),
            handle,
        });
    }

    fn start_bench(&mut self) {
        if self.fg_busy() {
            self.set_status("Busy — Esc to cancel first", false);
            return;
        }
        let Some(runtime) = self.active_runtime().cloned() else {
            self.set_status("Deploy a runtime first", true);
            return;
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
        let tx = self.bg_tx.clone();
        let handle = tokio::spawn(async move {
            let result = runner
                .run(
                    &suite,
                    subject,
                    &runtime.base_url,
                    runtime.api_key.as_deref(),
                    runtime.model_id.as_deref().unwrap_or("default"),
                )
                .await;
            let _ = tx.send(BgMsg::BenchDone(result));
        });
        self.begin_busy(BusyKind::Bench, "Running benchmark", handle);
    }

    fn start_assistant(&mut self) {
        if !self.assistant_configured {
            self.set_status(
                "Assistant not configured — set OPENROUTER_API_KEY (or a self-hosted URL), see [5] Setup",
                true,
            );
            return;
        }
        if self.fg_busy() {
            self.set_status("Busy — Esc to cancel first", false);
            return;
        }
        let assistant = Assistant::new(self.config.assistant.clone());
        let api_key = self.config.assistant_api_key();
        let error_context = self.last_error.as_ref().map(|e| e.assistant_context());
        let user_message = self
            .last_error
            .as_ref()
            .map(|e| format!("Help me fix: {}", e.message))
            .unwrap_or_else(|| "Help me diagnose LocalCode setup.".into());
        let correlation = self
            .last_error
            .as_ref()
            .map(|e| e.correlation_id.to_string());
        let redact = self.config.logging.redact_secrets;
        let paths = self.paths.clone();
        let config = self.config.clone();
        let config_snapshot = serde_json::json!({
            "backends": config.backends.default.kind,
            "registry": config.registry.endpoint,
            "api": config.api.base_url,
        });
        let tx = self.bg_tx.clone();
        let handle = tokio::spawn(async move {
            let logs = localcode_log::read_recent_logs(
                &paths.log_dir,
                80,
                correlation.as_deref(),
                redact,
            )
            .ok();
            let doctor = run_doctor(&paths, &config).await;
            let req = AssistantRequest {
                user_message,
                error_context,
                doctor_report: Some(doctor),
                recent_logs: logs,
                config_snapshot_redacted: Some(config_snapshot),
            };
            let result = assistant
                .ask(req, api_key.as_deref())
                .await
                .map(|r| r.message);
            let _ = tx.send(BgMsg::AssistantDone(result));
        });
        self.begin_busy(BusyKind::Assistant, "Asking assistant", handle);
    }

    fn start_doctor(&mut self) {
        if self.fg_busy() {
            self.set_status("Busy — Esc to cancel first", false);
            return;
        }
        let paths = self.paths.clone();
        let config = self.config.clone();
        let tx = self.bg_tx.clone();
        let handle = tokio::spawn(async move {
            let report = run_doctor(&paths, &config).await;
            let _ = tx.send(BgMsg::DoctorDone(
                serde_json::to_string_pretty(&report).unwrap_or_default(),
            ));
        });
        self.begin_busy(BusyKind::Doctor, "Running doctor", handle);
    }

    fn start_coding_turn(&mut self) {
        let input = self.coding_input.trim().to_string();
        if input.is_empty() {
            return;
        }
        if self.fg_busy() {
            self.set_status("Agent is busy — Esc to cancel", false);
            return;
        }

        let runtime = match self.active_runtime().cloned() {
            Some(r) => r,
            None => {
                // Local-first: never silently route the workspace to a cloud
                // provider. Users opt in via agent.allow_cloud_fallback.
                if self.config.agent.allow_cloud_fallback && self.assistant_configured {
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
                    self.coding_transcript.push(TranscriptEntry::new(
                        EntryKind::System,
                        "no local runtime — using the cloud assistant provider (agent.allow_cloud_fallback=true)",
                    ));
                    r
                } else {
                    self.coding_transcript.push(TranscriptEntry::new(
                        EntryKind::System,
                        "no runtime. Deploy one from [2] Models — or set agent.allow_cloud_fallback=true to use the cloud assistant provider.",
                    ));
                    self.coding_follow = true;
                    self.set_status("No runtime — deploy a model first ([2] Models)", true);
                    return;
                }
            }
        };

        self.coding_history.push(input.clone());
        if self.coding_history.len() > MAX_INPUT_HISTORY {
            self.coding_history.remove(0);
        }
        self.coding_hist_idx = None;
        self.coding_input.clear();
        self.coding_cursor = 0;
        self.coding_transcript
            .push(TranscriptEntry::new(EntryKind::You, input.clone()));
        self.coding_follow = true;

        // Live events (streamed tokens, tool activity) flow through their own
        // channel; the forwarder exits when the turn task drops the sender.
        let (ev_tx, mut ev_rx) = mpsc::unbounded_channel::<AgentEvent>();
        let fwd_tx = self.bg_tx.clone();
        tokio::spawn(async move {
            while let Some(ev) = ev_rx.recv().await {
                if fwd_tx.send(BgMsg::CodingEvent(ev)).is_err() {
                    break;
                }
            }
        });

        let agent_cfg = self.config.agent.clone();
        let session = self.session.clone();
        let tx = self.bg_tx.clone();
        let approver_tx = self.bg_tx.clone();
        let handle = tokio::spawn(async move {
            let agent = CodingAgent::new(agent_cfg);
            let approver = ChannelApprover { tx: approver_tx };
            let mut session = session.lock().await;
            let api_key = runtime.api_key.clone();
            let result = agent
                .run_turn(
                    &mut session,
                    &input,
                    &runtime,
                    api_key.as_deref(),
                    Some(&approver),
                    Some(&ev_tx),
                )
                .await;
            drop(session);
            let _ = tx.send(BgMsg::CodingDone(result));
        });
        self.begin_busy(BusyKind::Coding, "Agent working", handle);
    }

    fn new_coding_session(&mut self) {
        if self.fg_busy() {
            self.set_status("Agent is busy — Esc to cancel first", false);
            return;
        }
        self.session = Arc::new(AsyncMutex::new(AgentSession::new(
            self.workspace_path(),
            false,
        )));
        self.coding_transcript = vec![TranscriptEntry::new(EntryKind::System, "new session started")];
        self.coding_scroll = 0;
        self.coding_follow = true;
        self.set_status("New coding session", false);
    }

    fn start_stop_runtime(&mut self) {
        let Some(rt) = self.runtimes.get(self.runtime_selected) else {
            self.set_status("No runtime selected", false);
            return;
        };
        let id = rt.id.to_string();
        let name = rt.name.clone();
        let registry = self.registry.clone();
        let tx = self.bg_tx.clone();
        self.set_status(format!("Stopping {name}…"), false);
        tokio::spawn(async move {
            let result = registry.stop_runtime(&id).await;
            let _ = tx.send(BgMsg::RuntimeStopped { result });
        });
    }

    // ------------------------------------------------------------------
    // Updates
    // ------------------------------------------------------------------

    /// Background version check. Startup checks (`manual=false`) stay quiet
    /// unless an update exists; manual checks always report the outcome.
    pub fn start_update_check(&mut self, manual: bool) {
        let checker = match UpdateChecker::new(
            &self.config.updates.repo_url,
            &self.config.updates.branch,
        ) {
            Ok(c) => c,
            Err(e) => {
                if manual {
                    self.raise_error(e);
                }
                return;
            }
        };
        if manual {
            self.set_status("Checking for updates…", false);
        }
        let tx = self.bg_tx.clone();
        tokio::spawn(async move {
            let result = checker.check().await;
            let _ = tx.send(BgMsg::UpdateCheckDone { result, manual });
        });
    }

    fn open_update_modal(&mut self) {
        if let Some(v) = &self.update_installed {
            self.set_status(format!("v{v} already installed — restart LocalCode to apply"), false);
            return;
        }
        if self.update_busy.is_some() {
            self.set_status("Update already running (Esc cancels)", false);
            return;
        }
        let Some(info) = &self.update_available else {
            self.set_status("No update available — Ctrl+K → 'Check for updates'", false);
            return;
        };
        self.modal = Some(ModalState::confirm(
            format!("Install update v{}?", info.latest),
            format!(
                "Current: v{}\nLatest:  v{}\n\nLocalCode fetches {} ({}), rebuilds, and swaps the binary.\n\
                 The build runs in the background and can take a few minutes;\n\
                 Esc cancels safely and you can keep working meanwhile.\n\
                 Restart LocalCode afterwards to run the new version.",
                info.current, info.latest, self.config.updates.repo_url, self.config.updates.branch
            ),
            ConfirmAction::InstallUpdate,
        ));
    }

    fn start_install_update(&mut self) {
        if self.update_busy.is_some() {
            self.set_status("Update already running (Esc cancels)", false);
            return;
        }
        let updater = match SelfUpdater::resolve(
            self.config.updates.install_dir.as_deref(),
            &self.config.updates.repo_url,
            &self.config.updates.branch,
        ) {
            Ok(u) => u,
            Err(e) => {
                self.last_failed_action = Some(RetryAction::InstallUpdate);
                self.raise_error(e);
                return;
            }
        };

        let (ptx, mut prx) = mpsc::unbounded_channel::<String>();
        let fwd = self.bg_tx.clone();
        tokio::spawn(async move {
            while let Some(line) = prx.recv().await {
                if fwd.send(BgMsg::UpdateProgress(line)).is_err() {
                    break;
                }
            }
        });

        let tx = self.bg_tx.clone();
        let handle = tokio::spawn(async move {
            let result = updater.run(ptx).await;
            let _ = tx.send(BgMsg::UpdateDone(result));
        });
        self.update_progress_line = "starting…".into();
        self.update_busy = Some(Busy {
            kind: BusyKind::Update,
            label: "Updating LocalCode".into(),
            started: Instant::now(),
            handle,
        });
    }

    // ------------------------------------------------------------------
    // Message processing
    // ------------------------------------------------------------------

    pub fn process_bg(&mut self) {
        while let Ok(msg) = self.bg_rx.try_recv() {
            match msg {
                BgMsg::SearchDone(result) => {
                    self.finish_busy();
                    match result {
                        Ok(models) => {
                            self.models = models;
                            self.model_selected = 0;
                            self.set_status(format!("Found {} models", self.models.len()), false);
                        }
                        Err(e) => {
                            self.last_failed_action = Some(RetryAction::SearchModels);
                            self.raise_error(e);
                        }
                    }
                }
                BgMsg::DetailDone(result) => {
                    self.finish_busy();
                    match result {
                        Ok(detail) => {
                            self.selected_quant =
                                detail.quants.first().map(|q| q.label.clone());
                            self.model_detail = Some(detail);
                            self.card_scroll = 0;
                            self.refresh_fit();
                            self.set_status(
                                "Model loaded — → focus card, [,/.] quant, [d] deploy",
                                false,
                            );
                        }
                        Err(e) => {
                            self.last_failed_action = Some(RetryAction::LoadDetail);
                            self.raise_error(e);
                        }
                    }
                }
                BgMsg::DeployEnded(outcome) => {
                    self.deploy_busy = None;
                    if let DeployOutcome::Oversize(boxed) = outcome {
                        let (req, error) = *boxed;
                        self.deploy_progress = 0;
                        self.pending_oversize_deploy = Some(req);
                        self.last_error = Some(error.clone());
                        self.modal = Some(ModalState::warning(
                            "VRAM may be insufficient",
                            format!(
                                "{}\n\nPossible causes:\n• Model larger than free VRAM\n• Other processes using GPU\n\nYou can Continue anyway (never hard-blocked).",
                                error.message
                            ),
                        ));
                    }
                }
                BgMsg::CodingEvent(ev) => self.apply_agent_event(ev),
                BgMsg::CodingDone(result) => {
                    self.finish_busy();
                    match result {
                        Ok(reply) => {
                            self.finalize_transcript_live();
                            // Streaming already put the reply in the transcript;
                            // only fall back to the returned text when the turn
                            // produced no visible agent output.
                            let has_agent_text = self
                                .coding_transcript
                                .iter()
                                .rev()
                                .take_while(|e| e.kind != EntryKind::You)
                                .any(|e| e.kind == EntryKind::Agent && !e.text.trim().is_empty());
                            if !has_agent_text {
                                self.coding_transcript
                                    .push(TranscriptEntry::new(EntryKind::Agent, reply));
                            }
                            self.coding_follow = true;
                            self.set_status("Agent replied", false);
                        }
                        Err(e) => {
                            self.finalize_transcript_live();
                            self.coding_transcript.push(TranscriptEntry::new(
                                EntryKind::Error,
                                format!("{}: {}", e.code, e.message),
                            ));
                            self.coding_follow = true;
                            self.raise_error(e);
                        }
                    }
                }
                BgMsg::ToolConfirm {
                    description,
                    respond,
                } => {
                    self.respond_tool_confirm(false); // supersede any stale one
                    self.pending_tool_confirm = Some(respond);
                    self.modal = Some(ModalState::confirm(
                        "Agent wants to run a risky command",
                        format!("{description}\n\nConfirm to run it in the workspace, Cancel to refuse."),
                        ConfirmAction::ToolApproval,
                    ));
                }
                BgMsg::BenchDone(result) => {
                    self.finish_busy();
                    match result {
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
                        Err(e) => {
                            self.last_failed_action = Some(RetryAction::RunBench);
                            self.raise_error(e);
                        }
                    }
                }
                BgMsg::AssistantDone(result) => {
                    self.finish_busy();
                    match result {
                        Ok(message) => {
                            self.assistant_reply = Some(message);
                            self.assistant_scroll = 0;
                            self.assistant_open = true;
                            self.set_status("Assistant replied", false);
                        }
                        Err(e) => {
                            self.last_failed_action = Some(RetryAction::AskAssistant);
                            self.raise_error(e);
                        }
                    }
                }
                BgMsg::DoctorDone(report) => {
                    self.finish_busy();
                    self.doctor_summary = Some(report);
                    self.setup_scroll = 0;
                    self.set_status("Doctor complete — see Setup", false);
                }
                BgMsg::DetectDone(reports) => {
                    self.backend_reports = reports;
                    self.detecting = false;
                }
                BgMsg::ApiHealth(healthy) => {
                    self.api_healthy = Some(healthy);
                }
                BgMsg::RuntimeStopped { result } => match result {
                    Ok(()) => self.set_status("Runtime stopped", false),
                    Err(e) => self.raise_error(e),
                },
                BgMsg::UpdateCheckDone { result, manual } => match result {
                    Ok(Some(info)) => {
                        self.push_notification(
                            Severity::Info,
                            "Update available",
                            &format!(
                                "v{} → v{} — press u to install, or run `localcode update`",
                                info.current, info.latest
                            ),
                        );
                        self.set_status(
                            format!(
                                "Update available: v{} → v{} — press u to install",
                                info.current, info.latest
                            ),
                            false,
                        );
                        self.update_available = Some(info);
                    }
                    Ok(None) => {
                        if manual {
                            self.set_status(
                                format!("Up to date (v{})", localcode_upgrade::CURRENT_VERSION),
                                false,
                            );
                        }
                    }
                    Err(e) => {
                        if manual {
                            self.last_failed_action = Some(RetryAction::CheckUpdate);
                            self.raise_error(e);
                        } else {
                            // Startup checks fail quietly — offline is normal.
                            tracing::debug!(error = %e, "startup update check failed");
                        }
                    }
                },
                BgMsg::UpdateProgress(line) => {
                    self.update_progress_line = line;
                }
                BgMsg::UpdateDone(result) => {
                    self.update_busy = None;
                    self.update_progress_line.clear();
                    match result {
                        Ok(report) => {
                            self.update_available = None;
                            self.update_installed = Some(report.version.clone());
                            self.push_notification(
                                Severity::Success,
                                "Update installed",
                                &format!(
                                    "v{} — restart LocalCode to apply",
                                    report.version
                                ),
                            );
                            self.modal = Some(ModalState::info(
                                "Update installed",
                                format!(
                                    "LocalCode v{} is installed at\n{}\n\nRestart LocalCode to start using it.",
                                    report.version,
                                    report.binary_path.display()
                                ),
                            ));
                        }
                        Err(e) => {
                            self.last_failed_action = Some(RetryAction::InstallUpdate);
                            self.raise_error(e);
                        }
                    }
                }
            }
        }
    }

    pub async fn process_events(&mut self) {
        for ev in self.events.drain() {
            match ev {
                AppEvent::Notification {
                    severity,
                    title,
                    body,
                    ..
                } => {
                    self.push_notification(severity, &title, &body);
                    self.set_status(format!("{title}: {body}"), severity == Severity::Error);
                }
                AppEvent::DeployProgress {
                    percent, message, ..
                } => {
                    self.deploy_progress = percent;
                    self.set_status(format!("Deploy {percent}% — {message}"), false);
                }
                AppEvent::DeployFinished { .. } => {
                    self.deploy_progress = 100;
                    self.set_status("Deploy finished", false);
                }
                AppEvent::DeployFailed { error, .. } => {
                    self.deploy_progress = 0;
                    self.last_failed_action = Some(RetryAction::Deploy);
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
        if self.runtime_selected >= self.runtimes.len() {
            self.runtime_selected = self.runtimes.len().saturating_sub(1);
        }
    }

    // ------------------------------------------------------------------
    // Palette
    // ------------------------------------------------------------------

    pub fn palette_items(&self) -> Vec<String> {
        let mut all = vec![
            "Go: dashboard".to_string(),
            "Go: models".into(),
            "Go: benchmarks".into(),
            "Go: coding".into(),
            "Go: setup".into(),
            "Go: notifications".into(),
            "Go: settings".into(),
            "Deploy selected model".into(),
            "New coding session".into(),
            "Run doctor".into(),
            "Refresh backend detection".into(),
            "Check for updates".into(),
            "Clear notifications".into(),
            "Open logs".into(),
            "Ask assistant".into(),
            "Help".into(),
            "Quit".into(),
        ];
        if self.update_available.is_some() && self.update_busy.is_none() {
            all.insert(0, "Install update".into());
        }
        if self.palette_query.is_empty() {
            return all;
        }
        let q = self.palette_query.to_lowercase();
        all.into_iter()
            .filter(|i| i.to_lowercase().contains(&q))
            .collect()
    }

    fn run_palette_action(&mut self, item: &str) {
        match item {
            "Go: dashboard" => self.set_tab(Tab::Dashboard),
            "Go: models" => self.set_tab(Tab::Models),
            "Go: benchmarks" => self.set_tab(Tab::Benchmarks),
            "Go: coding" => self.set_tab(Tab::Coding),
            "Go: setup" => self.set_tab(Tab::Setup),
            "Go: notifications" => self.set_tab(Tab::Notifications),
            "Go: settings" => self.set_tab(Tab::Settings),
            "Deploy selected model" => self.start_deploy(false),
            "New coding session" => self.new_coding_session(),
            "Run doctor" => {
                self.start_doctor();
                self.set_tab(Tab::Setup);
            }
            "Refresh backend detection" => self.start_detect(),
            "Check for updates" => self.start_update_check(true),
            "Install update" => self.open_update_modal(),
            "Clear notifications" => {
                self.notifications.clear();
                self.notif_selected = 0;
            }
            "Open logs" => {
                self.set_status(format!("Logs: {}", self.paths.log_dir.display()), false);
            }
            "Ask assistant" => self.start_assistant(),
            "Help" => self.open_help(),
            "Quit" => self.request_quit(),
            _ => {}
        }
    }

    // ------------------------------------------------------------------
    // Help / quit
    // ------------------------------------------------------------------

    fn open_help(&mut self) {
        let tab_help = match self.tab {
            Tab::Models => {
                "Models:\n  / search   p popular   t trending\n  ←/→ focus list or card   j/k move / scroll\n  Enter open detail   PgUp/PgDn·g/G scroll card\n  ,/. pick quant   +/- context size\n  b cycle backend   d deploy\n  [ ] resize list/card   { } resize card/deploy"
            }
            Tab::Coding => {
                "Coding:\n  i or Enter focus composer   Esc unfocus\n  ↑/↓ input history (while typing)\n  PgUp/PgDn scroll transcript   End follow\n  Ctrl+↑/↓ (or +/-) composer height\n  n new session"
            }
            Tab::Benchmarks => "Benchmarks:\n  r run sample suite   p publish (sign-in)",
            Tab::Dashboard => "Dashboard:\n  j/k select runtime   x stop runtime\n  [ ] resize columns",
            Tab::Setup => "Setup:\n  d run doctor   r refresh detection\n  PgUp/PgDn scroll",
            Tab::Notifications => "Notifications:\n  j/k select   c clear all",
            Tab::Settings => "Settings:\n  t cycle theme   Ctrl+S save config",
        };
        let body = format!(
            "Global:\n  1-7 switch tab   Tab/Shift+Tab cycle tabs (click them too)\n  Ctrl+K palette   ? this help\n  a ask assistant   u install update   e last error details\n  l show log path   Esc cancel running task\n  q quit   Ctrl+C force quit   Ctrl+S save config\n\n{tab_help}"
        );
        self.modal = Some(ModalState::info("Keyboard shortcuts", body));
    }

    fn request_quit(&mut self) {
        if self.runtimes.is_empty() {
            self.should_quit = true;
        } else {
            self.modal = Some(ModalState::confirm(
                "Quit LocalCode?",
                format!(
                    "{} managed runtime(s) are running and will be stopped when LocalCode exits.",
                    self.runtimes.len()
                ),
                ConfirmAction::Quit,
            ));
        }
    }

    // ------------------------------------------------------------------
    // Key handling
    // ------------------------------------------------------------------

    fn handle_key(&mut self, key: crossterm::event::KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }

        // Global chords
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') => {
                    self.should_quit = true;
                    return;
                }
                KeyCode::Char('k') => {
                    self.palette_open = !self.palette_open;
                    self.palette_query.clear();
                    self.palette_selected = 0;
                    return;
                }
                KeyCode::Char('s') => {
                    if let Err(e) = self.config.save(&self.paths) {
                        self.raise_error(e);
                    } else {
                        self.set_status("Config saved", false);
                    }
                    return;
                }
                // Composer height (works while typing too).
                KeyCode::Up if self.tab == Tab::Coding => {
                    self.adjust_composer_rows(1);
                    return;
                }
                KeyCode::Down if self.tab == Tab::Coding => {
                    self.adjust_composer_rows(-1);
                    return;
                }
                _ => {}
            }
        }

        if self.palette_open {
            self.handle_palette_key(key);
            return;
        }

        if let Some(modal) = self.modal.clone() {
            self.handle_modal_key(key, &modal);
            return;
        }

        if self.assistant_open {
            match key.code {
                KeyCode::Esc | KeyCode::Char('q') => self.assistant_open = false,
                KeyCode::Up => self.assistant_scroll = self.assistant_scroll.saturating_sub(1),
                KeyCode::Down => self.assistant_scroll = self.assistant_scroll.saturating_add(1),
                KeyCode::PageUp => {
                    self.assistant_scroll = self.assistant_scroll.saturating_sub(10)
                }
                KeyCode::PageDown => {
                    self.assistant_scroll = self.assistant_scroll.saturating_add(10)
                }
                _ => {}
            }
            return;
        }

        // Search focus on models
        if self.model_search_focus && self.tab == Tab::Models {
            match key.code {
                KeyCode::Esc => self.model_search_focus = false,
                KeyCode::Enter => {
                    self.model_search_focus = false;
                    self.start_search();
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
            self.handle_composer_key(key);
            return;
        }

        // Tab switching & global keys
        match key.code {
            KeyCode::Char('q') => self.request_quit(),
            KeyCode::Char('?') => self.open_help(),
            KeyCode::Char('e') => self.open_error_modal(),
            KeyCode::Char('1') => self.set_tab(Tab::Dashboard),
            KeyCode::Char('2') => self.set_tab(Tab::Models),
            KeyCode::Char('3') => self.set_tab(Tab::Benchmarks),
            KeyCode::Char('4') => self.set_tab(Tab::Coding),
            KeyCode::Char('5') => self.set_tab(Tab::Setup),
            KeyCode::Char('6') => self.set_tab(Tab::Notifications),
            KeyCode::Char('7') => self.set_tab(Tab::Settings),
            KeyCode::Tab => {
                let idx = (self.tab.index() + 1) % 7;
                self.set_tab(Tab::from_index(idx));
            }
            KeyCode::BackTab => {
                let idx = (self.tab.index() + 6) % 7;
                self.set_tab(Tab::from_index(idx));
            }
            KeyCode::Char('a') => self.start_assistant(),
            KeyCode::Char('u') => self.open_update_modal(),
            KeyCode::Char('l') => {
                self.set_status(format!("Logs: {}", self.paths.log_dir.display()), false);
            }
            KeyCode::Esc => self.cancel_current(),
            other => self.handle_tab_key(other),
        }
    }

    fn adjust_composer_rows(&mut self, delta: i32) {
        let rows = i32::from(self.config.ui.composer_rows.clamp(1, 10));
        self.config.ui.composer_rows = (rows + delta).clamp(1, 10) as u16;
        self.set_status(
            format!(
                "Composer height: {} (Ctrl+↑/↓ adjust, saved on quit)",
                self.config.ui.composer_rows
            ),
            false,
        );
    }

    fn handle_composer_key(&mut self, key: crossterm::event::KeyEvent) {
        match key.code {
            KeyCode::Esc => self.coding_input_focus = false,
            KeyCode::Enter => self.start_coding_turn(),
            KeyCode::Left => self.coding_cursor = self.coding_cursor.saturating_sub(1),
            KeyCode::Right => {
                self.coding_cursor =
                    (self.coding_cursor + 1).min(self.coding_input.chars().count());
            }
            KeyCode::Home => self.coding_cursor = 0,
            KeyCode::End => self.coding_cursor = self.coding_input.chars().count(),
            KeyCode::Backspace => {
                if self.coding_cursor > 0 {
                    let idx = char_to_byte(&self.coding_input, self.coding_cursor - 1);
                    self.coding_input.remove(idx);
                    self.coding_cursor -= 1;
                }
            }
            KeyCode::Delete => {
                if self.coding_cursor < self.coding_input.chars().count() {
                    let idx = char_to_byte(&self.coding_input, self.coding_cursor);
                    self.coding_input.remove(idx);
                }
            }
            KeyCode::Up => {
                if self.coding_history.is_empty() {
                    return;
                }
                let idx = match self.coding_hist_idx {
                    None => self.coding_history.len() - 1,
                    Some(0) => 0,
                    Some(i) => i - 1,
                };
                self.coding_hist_idx = Some(idx);
                self.coding_input = self.coding_history[idx].clone();
                self.coding_cursor = self.coding_input.chars().count();
            }
            KeyCode::Down => match self.coding_hist_idx {
                Some(i) if i + 1 < self.coding_history.len() => {
                    self.coding_hist_idx = Some(i + 1);
                    self.coding_input = self.coding_history[i + 1].clone();
                    self.coding_cursor = self.coding_input.chars().count();
                }
                Some(_) => {
                    self.coding_hist_idx = None;
                    self.coding_input.clear();
                    self.coding_cursor = 0;
                }
                None => {}
            },
            KeyCode::Char(c) => {
                let idx = char_to_byte(&self.coding_input, self.coding_cursor);
                self.coding_input.insert(idx, c);
                self.coding_cursor += 1;
                self.coding_hist_idx = None;
            }
            _ => {}
        }
    }

    fn set_tab(&mut self, tab: Tab) {
        self.tab = tab;
    }

    fn handle_tab_key(&mut self, code: KeyCode) {
        match self.tab {
            Tab::Models => match code {
                KeyCode::Char('/') => {
                    self.model_search_focus = true;
                }
                KeyCode::Char('p') => {
                    self.model_query = "code".into();
                    self.start_search();
                }
                KeyCode::Char('t') => self.start_trending(),
                KeyCode::Left => self.models_focus = ModelsPane::List,
                KeyCode::Right => {
                    if self.model_detail.is_some() {
                        self.models_focus = ModelsPane::Card;
                    } else {
                        self.set_status("Open a model first (Enter) to focus the card", false);
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => match self.models_focus {
                    ModelsPane::List => {
                        if !self.models.is_empty() {
                            self.model_selected =
                                (self.model_selected + 1).min(self.models.len() - 1);
                        }
                    }
                    ModelsPane::Card => self.scroll_card(1),
                },
                KeyCode::Up | KeyCode::Char('k') => match self.models_focus {
                    ModelsPane::List => {
                        self.model_selected = self.model_selected.saturating_sub(1)
                    }
                    ModelsPane::Card => self.scroll_card(-1),
                },
                KeyCode::PageDown => self.scroll_card(self.card_view_height.max(1) as i64),
                KeyCode::PageUp => self.scroll_card(-(self.card_view_height.max(1) as i64)),
                KeyCode::Char('g') => self.card_scroll = 0,
                KeyCode::Char('G') => self.scroll_card(i64::MAX / 2),
                KeyCode::Enter => {
                    if self.models_focus == ModelsPane::List {
                        self.start_load_detail();
                    }
                }
                KeyCode::Char('d') => self.start_deploy(false),
                KeyCode::Char('b') => {
                    self.deploy_backend = match self.deploy_backend {
                        BackendKind::Ollama => BackendKind::LlamaCpp,
                        BackendKind::LlamaCpp => BackendKind::Vllm,
                        BackendKind::Vllm => BackendKind::Sglang,
                        BackendKind::Sglang => BackendKind::Ollama,
                    };
                    self.refresh_fit();
                }
                KeyCode::Char(',') => self.cycle_quant(-1),
                KeyCode::Char('.') => self.cycle_quant(1),
                KeyCode::Char('+') | KeyCode::Char('=') => self.adjust_ctx(true),
                KeyCode::Char('-') => self.adjust_ctx(false),
                KeyCode::Char('[') => self.adjust_pane("models", 0, -0.03),
                KeyCode::Char(']') => self.adjust_pane("models", 0, 0.03),
                KeyCode::Char('{') => self.adjust_pane("models", 1, -0.03),
                KeyCode::Char('}') => self.adjust_pane("models", 1, 0.03),
                _ => {}
            },
            Tab::Coding => match code {
                KeyCode::Char('i') | KeyCode::Enter => {
                    self.coding_input_focus = true;
                }
                KeyCode::Char('n') => self.new_coding_session(),
                KeyCode::Char('+') | KeyCode::Char('=') => self.adjust_composer_rows(1),
                KeyCode::Char('-') => self.adjust_composer_rows(-1),
                KeyCode::PageUp => {
                    let max = self
                        .coding_total_lines
                        .saturating_sub(self.coding_view_height as usize);
                    let cur = if self.coding_follow { max } else { self.coding_scroll };
                    self.coding_scroll =
                        cur.saturating_sub(self.coding_view_height.max(1) as usize);
                    self.coding_follow = false;
                }
                KeyCode::PageDown => {
                    let max = self
                        .coding_total_lines
                        .saturating_sub(self.coding_view_height as usize);
                    self.coding_scroll = (self.coding_scroll
                        + self.coding_view_height.max(1) as usize)
                        .min(max);
                    if self.coding_scroll >= max {
                        self.coding_follow = true;
                    }
                }
                KeyCode::End => self.coding_follow = true,
                _ => {}
            },
            Tab::Benchmarks => match code {
                KeyCode::Char('r') => self.start_bench(),
                KeyCode::Char('p') => {
                    self.set_status("Publish requires sign-in (Setup → Account)", true);
                }
                _ => {}
            },
            Tab::Dashboard => match code {
                KeyCode::Down | KeyCode::Char('j') => {
                    if !self.runtimes.is_empty() {
                        self.runtime_selected =
                            (self.runtime_selected + 1).min(self.runtimes.len() - 1);
                    }
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.runtime_selected = self.runtime_selected.saturating_sub(1);
                }
                KeyCode::Char('x') => self.start_stop_runtime(),
                KeyCode::Char('[') => self.adjust_pane("dashboard", 0, -0.03),
                KeyCode::Char(']') => self.adjust_pane("dashboard", 0, 0.03),
                _ => {}
            },
            Tab::Setup => match code {
                KeyCode::Char('d') => self.start_doctor(),
                KeyCode::Char('r') => {
                    self.start_detect();
                    self.set_status("Re-detecting backends…", false);
                }
                KeyCode::PageUp => self.setup_scroll = self.setup_scroll.saturating_sub(10),
                KeyCode::PageDown => self.setup_scroll = self.setup_scroll.saturating_add(10),
                _ => {}
            },
            Tab::Notifications => match code {
                KeyCode::Down | KeyCode::Char('j') => {
                    if !self.notifications.is_empty() {
                        self.notif_selected =
                            (self.notif_selected + 1).min(self.notifications.len() - 1);
                    }
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.notif_selected = self.notif_selected.saturating_sub(1);
                }
                KeyCode::Char('c') => {
                    self.notifications.clear();
                    self.notif_selected = 0;
                    self.set_status("Notifications cleared", false);
                }
                _ => {}
            },
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
        }
    }

    /// Shift the boundary between panes `idx` and `idx+1` of a view by
    /// `delta`. Ratios are persisted with the config (saved on quit/Ctrl+S).
    fn adjust_pane(&mut self, view: &str, idx: usize, delta: f32) {
        let defaults = Self::pane_defaults(view);
        let mut ratios = self.pane_ratios(view, defaults);
        if idx + 1 >= ratios.len() {
            return;
        }
        let moved = delta
            .min(ratios[idx + 1] - 0.15) // don't shrink the neighbor below min
            .max(0.15 - ratios[idx]); // nor this pane
        ratios[idx] += moved;
        ratios[idx + 1] -= moved;
        let sum: f32 = ratios.iter().sum();
        for r in &mut ratios {
            *r /= sum;
        }
        self.config.panes.views.insert(view.into(), ratios);
        self.set_status(
            format!("Pane resized ({view}) — [/] and {{/}} adjust, saved on quit"),
            false,
        );
    }

    /// Default split per view; ui.rs uses the same values when drawing.
    pub fn pane_defaults(view: &str) -> &'static [f32] {
        match view {
            "dashboard" => &[0.5, 0.5],
            _ => &[0.28, 0.44, 0.28],
        }
    }

    fn scroll_card(&mut self, delta: i64) {
        let max = self
            .card_total_lines
            .saturating_sub(self.card_view_height as usize) as i64;
        self.card_scroll = (self.card_scroll as i64 + delta).clamp(0, max.max(0)) as usize;
    }

    fn handle_modal_key(&mut self, key: crossterm::event::KeyEvent, modal: &ModalState) {
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
                match &modal.kind {
                    ModalKind::Confirm {
                        action: ConfirmAction::ToolApproval,
                        ..
                    } => self.respond_tool_confirm(false),
                    ModalKind::Warning { .. } => {
                        self.pending_oversize_deploy = None;
                    }
                    _ => {}
                }
                self.modal = None;
            }
            KeyCode::Enter => {
                let label = modal.selected_button();
                let kind = modal.kind.clone();
                self.modal = None;
                match (&kind, label) {
                    (ModalKind::Error { .. }, "Retry") => self.retry_last(),
                    (ModalKind::Error { .. }, "Open logs") => {
                        self.set_status(
                            format!("Logs: {}", self.paths.log_dir.display()),
                            false,
                        );
                    }
                    (ModalKind::Error { .. }, "Ask assistant") => self.start_assistant(),
                    (ModalKind::Warning { .. }, "Continue") => {
                        if let Some(mut req) = self.pending_oversize_deploy.take() {
                            req.continue_despite_oversize = true;
                            self.spawn_deploy(req);
                        }
                    }
                    (ModalKind::Warning { .. }, "Cancel") => {
                        self.pending_oversize_deploy = None;
                        self.set_status("Deploy cancelled", false);
                    }
                    (
                        ModalKind::Confirm {
                            action: ConfirmAction::Quit,
                            ..
                        },
                        "Confirm",
                    ) => self.should_quit = true,
                    (
                        ModalKind::Confirm {
                            action: ConfirmAction::InstallUpdate,
                            ..
                        },
                        "Confirm",
                    ) => self.start_install_update(),
                    (
                        ModalKind::Confirm {
                            action: ConfirmAction::ToolApproval,
                            ..
                        },
                        label,
                    ) => self.respond_tool_confirm(label == "Confirm"),
                    (ModalKind::Payment { .. }, "Confirm pay") => {
                        self.set_status("Confirmed", false);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn handle_palette_key(&mut self, key: crossterm::event::KeyEvent) {
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
                    self.run_palette_action(&item);
                }
            }
            KeyCode::Char(c) => {
                self.palette_query.push(c);
                self.palette_selected = 0;
            }
            _ => {}
        }
    }

    fn handle_mouse(&mut self, mouse: crossterm::event::MouseEvent) {
        // Tab strip hit ranges are recorded during draw, so clicks always
        // match what is actually on screen.
        self.tab_hover = if mouse.row == self.tab_strip_row {
            self.tab_hit
                .iter()
                .find(|(x0, x1, _)| mouse.column >= *x0 && mouse.column < *x1)
                .map(|(_, _, t)| *t)
        } else {
            None
        };

        match mouse.kind {
            MouseEventKind::Down(event::MouseButton::Left) => {
                if let Some(t) = self.tab_hover {
                    self.set_tab(t);
                }
            }
            MouseEventKind::ScrollUp => self.wheel_scroll(-3),
            MouseEventKind::ScrollDown => self.wheel_scroll(3),
            _ => {}
        }
    }

    /// Route the mouse wheel to the current view's main scrollable.
    fn wheel_scroll(&mut self, delta: i64) {
        match self.tab {
            Tab::Coding => {
                let max = self
                    .coding_total_lines
                    .saturating_sub(self.coding_view_height as usize);
                let cur = if self.coding_follow {
                    max
                } else {
                    self.coding_scroll
                };
                self.coding_scroll = (cur as i64 + delta).clamp(0, max as i64) as usize;
                self.coding_follow = self.coding_scroll >= max;
            }
            Tab::Models => self.scroll_card(delta),
            Tab::Setup => {
                self.setup_scroll = (i64::from(self.setup_scroll) + delta).max(0) as u16;
            }
            Tab::Notifications if !self.notifications.is_empty() => {
                let max = self.notifications.len() as i64 - 1;
                self.notif_selected =
                    (self.notif_selected as i64 + delta.signum()).clamp(0, max) as usize;
            }
            _ => {}
        }
    }
}

/// Byte index of the `char_idx`-th char (== s.len() when at the end).
fn char_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}

pub async fn run_tui(paths: AppPaths, config: Config) -> Result<(), LocalCodeError> {
    info!("starting TUI");

    // Restore the terminal on panic — otherwise a panic leaves the user's
    // shell in raw mode + alternate screen with mouse capture on.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
        default_hook(panic_info);
    }));

    let mouse_enabled = config.ui.mouse;
    enable_raw_mode().map_err(LocalCodeError::from)?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).map_err(LocalCodeError::from)?;
    if mouse_enabled {
        execute!(io::stdout(), EnableMouseCapture).map_err(LocalCodeError::from)?;
    }
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(LocalCodeError::from)?;

    let mut app = App::new(paths, config);
    app.start_detect();
    if app.config.updates.check_on_startup {
        app.start_update_check(false);
    }
    let result = run_loop(&mut terminal, &mut app).await;

    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    if mouse_enabled {
        execute!(terminal.backend_mut(), DisableMouseCapture).ok();
    }
    terminal.show_cursor().ok();
    result
}

async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<(), LocalCodeError> {
    loop {
        app.process_events().await;
        app.process_bg();
        terminal
            .draw(|f| ui::draw(f, app))
            .map_err(LocalCodeError::from)?;

        if app.should_quit {
            break;
        }

        if event::poll(Duration::from_millis(100)).map_err(LocalCodeError::from)? {
            match event::read().map_err(LocalCodeError::from)? {
                Event::Key(key) => app.handle_key(key),
                Event::Mouse(m) => app.handle_mouse(m),
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
