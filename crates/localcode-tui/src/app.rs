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
    resolve_install_plan, run_install, BackendKind, BackendRegistry, DeployRequest, DeployService,
    DetectReport, InstallPlan,
};
use localcode_bench::{sample_coding_suite, BenchRunner, Subject};
use localcode_core::config::Config;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus, Severity};
use localcode_core::paths::AppPaths;
use localcode_core::runtime::ActiveRuntime;
use localcode_core::theme::{Theme, ThemeMode};
use localcode_gpu::{discover, predict_fit, FitPrediction, FitRequest, GpuInventory};
use localcode_remote::{setup_server, RemoteSession};
use localcode_hf::{HfClient, ModelDetail, ModelSummary};
use localcode_upgrade::{SelfUpdater, UpdateChecker, UpdateInfo, UpdateReport};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
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

/// A popup panel raised over the home transcript. `None` (no panel) is the
/// home surface: the coding transcript plus the omnibar. Panels replace the
/// old tabs — they are opened with slash commands and dismissed with Esc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    Models,
    Runtimes,
    Benchmarks,
    Setup,
    Notifications,
    Settings,
    Remote,
}

impl Panel {
    pub fn title(self) -> &'static str {
        match self {
            Panel::Models => "Models",
            Panel::Runtimes => "Runtimes",
            Panel::Benchmarks => "Benchmarks",
            Panel::Setup => "Setup & Doctor",
            Panel::Notifications => "Notifications",
            Panel::Settings => "Settings",
            Panel::Remote => "Remote GPU servers",
        }
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

/// A clickable/scrollable region recorded during draw so the mouse handler
/// acts on exactly what is on screen (mirrors the `tab_hit` pattern). Lists
/// record a single region over their inner rect; the row index is derived in
/// the handler from `ListState::offset()` so scrolling stays correct.
#[derive(Debug, Clone, Copy)]
pub struct ClickRegion {
    pub rect: Rect,
    pub target: ClickTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClickTarget {
    RuntimeList,
    ModelList,
    ModelCard,
    DeployPane,
    DeployButton,
    QuantList,
    BackendCycle,
    Transcript,
    Composer,
    NotificationList,
    SetupBody,
    SetupManageBackends,
    ModalButton(usize),
    PaletteItem(usize),
    // Backends overlay
    BackendMgrItem,
    BackendMgrField,
    BackendMgrInstall,
    BackendMgrSave,
    BackendMgrRedetect,
    // Remote panel
    RemoteList,
    RemoteField,
    RemoteConnect,
    RemoteSave,
    RemoteDisconnect,
    RemoteDelete,
    RemoteNew,
    // Panel close (the ✕ on a popup)
    PanelClose,
}

/// A draggable vertical seam between two horizontal panes, recorded during
/// draw. `area` is the full span of the split so drags map a column back to a
/// fraction; `idx` is the boundary between panes `idx` and `idx + 1`.
#[derive(Debug, Clone, Copy)]
pub struct ResizeBorder {
    pub x: u16,
    pub y0: u16,
    pub y1: u16,
    pub view: &'static str,
    pub idx: usize,
    pub area: Rect,
}

/// An in-progress border drag.
#[derive(Debug, Clone, Copy)]
pub struct DragState {
    pub view: &'static str,
    pub idx: usize,
    pub area: Rect,
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
    Install,
    Remote,
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
    InstallBackend(BackendKind),
    ConnectRemote(usize),
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
    /// A backend install: streamed output lines, then the terminal result.
    InstallProgress(String),
    InstallDone {
        kind: BackendKind,
        result: Result<(), LocalCodeError>,
    },
    /// A remote SSH GPU server connection finished (setup + tunnel).
    RemoteConnected {
        server_idx: usize,
        result: Result<Box<RemoteSession>, LocalCodeError>,
    },
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
    /// The popup panel raised over the home transcript, or `None` for home.
    pub panel: Option<Panel>,
    /// Clickable regions and draggable pane borders for the current frame,
    /// refilled every draw so the mouse handler never recomputes layout.
    pub click_regions: Vec<ClickRegion>,
    pub resize_borders: Vec<ResizeBorder>,
    pub dragging: Option<DragState>,
    pub status_line: String,
    pub status_is_error: bool,
    pub last_error: Option<LocalCodeError>,
    pub last_failed_action: Option<RetryAction>,
    pub modal: Option<ModalState>,
    pending_tool_confirm: Option<oneshot::Sender<bool>>,
    /// Highlighted index in the slash-command menu (shown when the omnibar
    /// text starts with '/').
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
    // Backends manager overlay
    pub backends_open: bool,
    pub backend_sel: usize,
    pub backend_field: usize,
    pub backend_field_edit: String,
    pub backend_editing: bool,
    /// Cached install-plan preview for the selected backend (recomputed on
    /// selection change, not per-frame).
    pub backend_plan_preview: String,
    pub install_busy: Option<Busy>,
    pub install_progress_line: String,
    pub installing_kind: Option<BackendKind>,
    // Remote SSH GPU servers
    pub remote_sessions: Vec<RemoteSession>,
    pub remote_selected: usize,
    pub remote_field: usize,
    pub remote_field_edit: String,
    pub remote_editing: bool,
    pub remote_connecting: Option<usize>,
    /// Ollama base_url saved before it was repointed at a remote tunnel, so it
    /// can be restored on disconnect.
    pub pre_remote_ollama_url: Option<String>,
    // Models
    pub model_query: String,
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
    // Omnibar (prompt / search / command bar) — always focused.
    pub coding_input: String,
    pub coding_cursor: usize, // char index into coding_input
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
            panel: None,
            click_regions: vec![],
            resize_borders: vec![],
            dragging: None,
            status_line: "Type to chat with the agent · press / for commands".into(),
            status_is_error: false,
            last_error: None,
            last_failed_action: None,
            modal: None,
            pending_tool_confirm: None,
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
            backends_open: false,
            backend_sel: 0,
            backend_field: 0,
            backend_field_edit: String::new(),
            backend_editing: false,
            backend_plan_preview: String::new(),
            install_busy: None,
            install_progress_line: String::new(),
            installing_kind: None,
            remote_sessions: vec![],
            remote_selected: 0,
            remote_field: 0,
            remote_field_edit: String::new(),
            remote_editing: false,
            remote_connecting: None,
            pre_remote_ollama_url: None,
            model_query: String::new(),
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
            coding_history: vec![],
            coding_hist_idx: None,
            coding_transcript: vec![TranscriptEntry::new(
                EntryKind::System,
                "Welcome to LocalCode. Type a message to chat with the agent, or press / for commands (/models, /remote, /backends, /help). Deploy a model with /models, or connect a remote GPU with /remote.",
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

    /// All runtimes the agent can use: locally-managed (from the registry) plus
    /// remote runtimes tunneled over SSH.
    pub fn all_runtimes(&self) -> Vec<&ActiveRuntime> {
        self.runtimes
            .iter()
            .chain(self.remote_sessions.iter().map(|s| &s.runtime))
            .collect()
    }

    pub fn active_runtime(&self) -> Option<ActiveRuntime> {
        let all = self.all_runtimes();
        all.get(self.runtime_selected)
            .or_else(|| all.first())
            .map(|r| (*r).clone())
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
        let valid =
            ratios.len() == default.len() && ratios.iter().all(|r| r.is_finite() && *r > 0.0);
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
            if b.kind == BusyKind::Remote {
                self.remote_connecting = None;
                // Undo the speculative Ollama repoint.
                if let Some(url) = self.pre_remote_ollama_url.take() {
                    self.config.backends.ollama.base_url = url;
                }
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
        } else if self.install_busy.is_some() {
            self.cancel_install();
        }
    }

    /// Abort a running backend install (kill_on_drop stops the child when the
    /// aborted task's Command is dropped). Returns whether one was running.
    fn cancel_install(&mut self) -> bool {
        if let Some(b) = self.install_busy.take() {
            b.handle.abort();
            self.installing_kind = None;
            self.install_progress_line.clear();
            self.set_status("Install cancelled", false);
            true
        } else {
            false
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
            Some(RetryAction::InstallBackend(kind)) => self.start_install(kind),
            Some(RetryAction::ConnectRemote(idx)) => self.connect_remote(idx),
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
            self.config.hf_mirror_hosts(),
        );
        let tx = self.bg_tx.clone();
        let label = format!("Deploying {}", req.model_id);
        self.deploy_progress = 0;
        let handle = tokio::spawn(async move {
            let continue_oversize = req.continue_despite_oversize;
            let outcome = match svc.deploy(req.clone()).await {
                Ok(_) => DeployOutcome::Done,
                Err(e) if e.code == ErrorCode::DeployOversizedWarning && !continue_oversize => {
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
        let Some(runtime) = self.active_runtime() else {
            self.set_status("Deploy a runtime first", true);
            return;
        };
        let suite = sample_coding_suite();
        let subject = Subject {
            hf_model_id: runtime.model_id.clone().unwrap_or_else(|| "unknown".into()),
            quantization: runtime
                .quantization
                .clone()
                .unwrap_or_else(|| "unknown".into()),
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
                "Assistant not configured — set OPENROUTER_API_KEY (or a self-hosted URL), see the Setup tab",
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
            let logs =
                localcode_log::read_recent_logs(&paths.log_dir, 80, correlation.as_deref(), redact)
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

        let runtime = match self.active_runtime() {
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
                        "no runtime. Deploy one from the Models tab — or set agent.allow_cloud_fallback=true to use the cloud assistant provider.",
                    ));
                    self.coding_follow = true;
                    self.set_status("No runtime — deploy a model first (Models tab)", true);
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
        self.coding_transcript = vec![TranscriptEntry::new(
            EntryKind::System,
            "new session started",
        )];
        self.coding_scroll = 0;
        self.coding_follow = true;
        self.set_status("New coding session", false);
    }

    fn start_stop_runtime(&mut self) {
        // Selection indexes registry runtimes first, then remote sessions.
        let reg_len = self.runtimes.len();
        if self.runtime_selected >= reg_len {
            let remote_idx = self.runtime_selected - reg_len;
            if let Some(name) = self.remote_sessions.get(remote_idx).map(|s| s.server_name.clone()) {
                match self.config.remote.servers.iter().position(|s| s.name == name) {
                    Some(cfg_idx) => self.disconnect_remote(cfg_idx),
                    None => {
                        self.remote_sessions.remove(remote_idx);
                        self.set_status(format!("Disconnected from {name}"), false);
                    }
                }
            }
            return;
        }
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
    // Backends manager (configure + install)
    // ------------------------------------------------------------------

    fn open_backend_manager(&mut self) {
        self.backends_open = true;
        self.assistant_open = false;
        self.backend_sel = 0;
        self.backend_field = 0;
        self.backend_editing = false;
        self.load_field_edit();
        self.refresh_plan_preview();
        if self.backend_reports.is_empty() && !self.detecting {
            self.start_detect();
        }
        self.set_status(
            "Backends — Tab switch · ↑/↓ field · Enter edit · i install · s save · r re-detect · Esc close",
            false,
        );
    }

    pub(crate) fn backend_sel_kind(&self) -> BackendKind {
        BACKEND_ORDER[self.backend_sel.min(BACKEND_ORDER.len() - 1)]
    }

    /// Recompute the cached install-plan preview for the selected backend.
    fn refresh_plan_preview(&mut self) {
        let kind = self.backend_sel_kind();
        self.backend_plan_preview = match resolve_install_plan(kind) {
            InstallPlan::Automated { display, .. } => display,
            InstallPlan::Manual { url, .. } => format!("manual — {url}"),
        };
    }

    /// Editable config field labels for a backend.
    pub(crate) fn backend_field_labels(kind: BackendKind) -> &'static [&'static str] {
        match kind {
            BackendKind::Ollama => &["base_url"],
            _ => &["bin", "host", "port"],
        }
    }

    fn backend_field_count(&self) -> usize {
        Self::backend_field_labels(self.backend_sel_kind()).len()
    }

    /// Current value of a backend's config field, read from `config.backends`.
    pub(crate) fn backend_field_value(&self, kind: BackendKind, idx: usize) -> String {
        let b = &self.config.backends;
        match (kind, idx) {
            (BackendKind::Ollama, 0) => b.ollama.base_url.clone(),
            (BackendKind::LlamaCpp, 0) => b.llamacpp.bin.clone(),
            (BackendKind::LlamaCpp, 1) => b.llamacpp.host.clone(),
            (BackendKind::LlamaCpp, 2) => b.llamacpp.port.to_string(),
            (BackendKind::Vllm, 0) => b.vllm.bin.clone(),
            (BackendKind::Vllm, 1) => b.vllm.host.clone(),
            (BackendKind::Vllm, 2) => b.vllm.port.to_string(),
            (BackendKind::Sglang, 0) => b.sglang.bin.clone(),
            (BackendKind::Sglang, 1) => b.sglang.host.clone(),
            (BackendKind::Sglang, 2) => b.sglang.port.to_string(),
            _ => String::new(),
        }
    }

    /// Write a field back into `config.backends`. Ports that don't parse are
    /// ignored (the last valid value stays).
    fn set_backend_field_value(&mut self, kind: BackendKind, idx: usize, val: &str) {
        let v = val.trim().to_string();
        let b = &mut self.config.backends;
        match (kind, idx) {
            (BackendKind::Ollama, 0) => b.ollama.base_url = v,
            (BackendKind::LlamaCpp, 0) => b.llamacpp.bin = v,
            (BackendKind::LlamaCpp, 1) => b.llamacpp.host = v,
            (BackendKind::LlamaCpp, 2) => {
                if let Ok(p) = v.parse() {
                    b.llamacpp.port = p;
                }
            }
            (BackendKind::Vllm, 0) => b.vllm.bin = v,
            (BackendKind::Vllm, 1) => b.vllm.host = v,
            (BackendKind::Vllm, 2) => {
                if let Ok(p) = v.parse() {
                    b.vllm.port = p;
                }
            }
            (BackendKind::Sglang, 0) => b.sglang.bin = v,
            (BackendKind::Sglang, 1) => b.sglang.host = v,
            (BackendKind::Sglang, 2) => {
                if let Ok(p) = v.parse() {
                    b.sglang.port = p;
                }
            }
            _ => {}
        }
    }

    /// Load the selected field's current value into the edit buffer.
    fn load_field_edit(&mut self) {
        let k = self.backend_sel_kind();
        self.backend_field_edit = self.backend_field_value(k, self.backend_field);
    }

    /// Commit the edit buffer back into config for the selected field.
    fn commit_field_edit(&mut self) {
        let k = self.backend_sel_kind();
        let v = self.backend_field_edit.clone();
        self.set_backend_field_value(k, self.backend_field, &v);
    }

    fn save_backend_config(&mut self) {
        self.commit_field_edit();
        if let Err(e) = self.config.save(&self.paths) {
            self.raise_error(e);
            return;
        }
        // Rebuilding the registry drops backend Arcs; for a backend with a
        // managed child that would kill it (kill_on_drop). Only rebuild when no
        // runtime is active — otherwise persist and defer to a restart.
        if self.runtimes.is_empty() {
            self.registry = Arc::new(BackendRegistry::from_config(&self.config));
            self.start_detect();
            self.set_status("Backend config saved & applied", false);
        } else {
            self.set_status(
                "Backend config saved — restart LocalCode (or stop runtimes) to apply",
                false,
            );
        }
    }

    /// Show the install plan: automated plans go through a confirm dialog that
    /// prints the exact commands; unautomatable ones show honest manual steps.
    fn start_install(&mut self, kind: BackendKind) {
        if self.install_busy.is_some() {
            self.set_status("An install is already running (Esc to cancel)", false);
            return;
        }
        match resolve_install_plan(kind) {
            InstallPlan::Automated { display, .. } => {
                let body = format!(
                    "LocalCode will run:\n\n{display}\n\n{}Esc cancels — nothing runs until you confirm.",
                    install_caveat(kind),
                );
                self.modal = Some(ModalState::confirm(
                    format!("Install {}?", kind.as_str()),
                    body,
                    ConfirmAction::InstallBackend(kind),
                ));
            }
            InstallPlan::Manual {
                summary,
                steps,
                url,
            } => {
                let mut body = format!("{summary}\n\n");
                for s in &steps {
                    body.push_str(&format!("• {s}\n"));
                }
                body.push_str(&format!("\n{url}"));
                self.modal = Some(ModalState::info(
                    format!("Install {} — manual steps", kind.as_str()),
                    body,
                ));
            }
        }
    }

    /// Spawn the install (called after the confirm dialog). Streams output lines
    /// through `InstallProgress`; Esc aborts (kill_on_drop stops the child).
    fn spawn_install(&mut self, kind: BackendKind) {
        if self.install_busy.is_some() {
            self.set_status("An install is already running (Esc to cancel)", false);
            return;
        }
        let (ptx, mut prx) = mpsc::unbounded_channel::<String>();
        let fwd = self.bg_tx.clone();
        tokio::spawn(async move {
            while let Some(line) = prx.recv().await {
                if fwd.send(BgMsg::InstallProgress(line)).is_err() {
                    break;
                }
            }
        });
        let tx = self.bg_tx.clone();
        let handle = tokio::spawn(async move {
            let plan = resolve_install_plan(kind);
            let result = run_install(&plan, ptx).await;
            let _ = tx.send(BgMsg::InstallDone { kind, result });
        });
        self.installing_kind = Some(kind);
        self.install_progress_line = "starting…".into();
        self.install_busy = Some(Busy {
            kind: BusyKind::Install,
            label: format!("Installing {}", kind.as_str()),
            started: Instant::now(),
            handle,
        });
    }

    fn handle_backends_key(&mut self, key: crossterm::event::KeyEvent) {
        if self.backend_editing {
            match key.code {
                KeyCode::Esc | KeyCode::Enter => {
                    self.commit_field_edit();
                    self.backend_editing = false;
                }
                KeyCode::Backspace => {
                    self.backend_field_edit.pop();
                }
                KeyCode::Char(c) => self.backend_field_edit.push(c),
                _ => {}
            }
            return;
        }
        match key.code {
            // Esc cancels a running install first (the status bar promises so);
            // only closes the overlay when nothing is installing.
            KeyCode::Esc => {
                if !self.cancel_install() {
                    self.backends_open = false;
                }
            }
            KeyCode::Tab => {
                self.backend_sel = (self.backend_sel + 1) % BACKEND_ORDER.len();
                self.backend_field = 0;
                self.load_field_edit();
                self.refresh_plan_preview();
            }
            KeyCode::BackTab => {
                self.backend_sel = (self.backend_sel + BACKEND_ORDER.len() - 1) % BACKEND_ORDER.len();
                self.backend_field = 0;
                self.load_field_edit();
                self.refresh_plan_preview();
            }
            KeyCode::Up | KeyCode::Char('k') => {
                let n = self.backend_field_count();
                if n > 0 {
                    self.backend_field = (self.backend_field + n - 1) % n;
                    self.load_field_edit();
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let n = self.backend_field_count();
                if n > 0 {
                    self.backend_field = (self.backend_field + 1) % n;
                    self.load_field_edit();
                }
            }
            KeyCode::Enter | KeyCode::Char('e') => {
                self.load_field_edit();
                self.backend_editing = true;
            }
            KeyCode::Char('i') => {
                let k = self.backend_sel_kind();
                self.start_install(k);
            }
            KeyCode::Char('s') => self.save_backend_config(),
            KeyCode::Char('r') => {
                self.start_detect();
                self.set_status("Re-detecting backends…", false);
            }
            _ => {}
        }
    }

    // ------------------------------------------------------------------
    // Remote SSH GPU servers
    // ------------------------------------------------------------------

    /// Connect the first server marked `auto_connect` on startup.
    pub fn autoconnect_remotes(&mut self) {
        if let Some(idx) = self
            .config
            .remote
            .servers
            .iter()
            .position(|s| s.auto_connect)
        {
            self.set_status(
                format!("Auto-connecting to {}…", self.config.remote.servers[idx].name),
                false,
            );
            self.connect_remote(idx);
        }
    }

    fn open_remote_panel(&mut self) {
        self.set_panel(Some(Panel::Remote));
        self.assistant_open = false;
        if self.remote_selected >= self.config.remote.servers.len() {
            self.remote_selected = 0;
        }
        self.remote_field = 0;
        self.remote_editing = false;
        self.set_status(
            "Remote: click a field to edit · Connect to link · or /remote add <name> <host> <user> <password>",
            false,
        );
    }

    /// Append a fresh server template and select it for editing.
    fn new_remote_server(&mut self) {
        let s = localcode_core::config::RemoteServer {
            name: format!("server-{}", self.config.remote.servers.len() + 1),
            ..Default::default()
        };
        self.config.remote.servers.push(s);
        self.remote_selected = self.config.remote.servers.len() - 1;
        self.remote_field = 0;
        self.remote_editing = false;
        self.set_status("New server — fill in host/username/password, then Save & Connect", false);
    }

    fn delete_remote(&mut self, idx: usize) {
        if idx < self.config.remote.servers.len() {
            let name = self.config.remote.servers[idx].name.clone();
            self.disconnect_remote(idx);
            self.config.remote.servers.remove(idx);
            self.remote_selected = self.remote_selected.min(
                self.config.remote.servers.len().saturating_sub(1),
            );
            let _ = self.config.save(&self.paths);
            self.set_status(format!("Removed server '{name}'"), false);
        }
    }

    fn save_remote_config(&mut self) {
        self.commit_remote_field();
        match self.config.save(&self.paths) {
            Ok(()) => self.set_status("Remote servers saved", false),
            Err(e) => self.raise_error(e),
        }
    }

    /// `/remote add <name> <host> <user> <password> [port] [local_port]`
    fn remote_quick_add(&mut self, args: &[String]) {
        if args.len() < 4 {
            self.set_status(
                "Usage: /remote add <name> <host> <username> <password> [port] [local_port]",
                true,
            );
            self.open_remote_panel();
            return;
        }
        let mut s = localcode_core::config::RemoteServer {
            name: args[0].clone(),
            host: args[1].clone(),
            username: args[2].clone(),
            password: args[3].clone(),
            ..Default::default()
        };
        if let Some(p) = args.get(4).and_then(|v| v.parse().ok()) {
            s.port = p;
        }
        if let Some(lp) = args.get(5).and_then(|v| v.parse().ok()) {
            s.local_port = lp;
        }
        self.config.remote.servers.push(s);
        self.remote_selected = self.config.remote.servers.len() - 1;
        let _ = self.config.save(&self.paths);
        self.open_remote_panel();
        self.set_status(
            format!("Added '{}' — press Connect (or /connect)", args[0]),
            false,
        );
    }

    pub(crate) fn remote_field_value(&self, idx: usize) -> String {
        let Some(s) = self.config.remote.servers.get(self.remote_selected) else {
            return String::new();
        };
        match idx {
            0 => s.name.clone(),
            1 => s.host.clone(),
            2 => s.port.to_string(),
            3 => s.username.clone(),
            4 => s.password.clone(),
            5 => {
                if s.local_port == 0 {
                    String::new()
                } else {
                    s.local_port.to_string()
                }
            }
            _ => String::new(),
        }
    }

    fn set_remote_field_value(&mut self, idx: usize, val: &str) {
        let v = val.trim().to_string();
        let Some(s) = self.config.remote.servers.get_mut(self.remote_selected) else {
            return;
        };
        match idx {
            0 => s.name = v,
            1 => s.host = v,
            2 => {
                if let Ok(p) = v.parse() {
                    s.port = p;
                }
            }
            3 => s.username = v,
            4 => s.password = v,
            5 => s.local_port = v.parse().unwrap_or(0),
            _ => {}
        }
    }

    fn load_remote_field_edit(&mut self) {
        self.remote_field_edit = self.remote_field_value(self.remote_field);
    }

    fn commit_remote_field(&mut self) {
        if self.remote_editing {
            let v = self.remote_field_edit.clone();
            self.set_remote_field_value(self.remote_field, &v);
        }
    }

    fn handle_remote_field_key(&mut self, key: crossterm::event::KeyEvent) {
        match key.code {
            KeyCode::Esc | KeyCode::Enter => {
                self.commit_remote_field();
                self.remote_editing = false;
            }
            KeyCode::Backspace => {
                self.remote_field_edit.pop();
            }
            KeyCode::Char(c) => self.remote_field_edit.push(c),
            _ => {}
        }
    }

    fn connect_remote(&mut self, idx: usize) {
        if self.remote_connecting.is_some() || self.fg_busy() {
            self.set_status("Busy — Esc to cancel first", false);
            return;
        }
        self.commit_remote_field();
        let Some(server) = self.config.remote.servers.get(idx).cloned() else {
            self.set_status("No server selected — add one first", true);
            return;
        };
        if server.host.trim().is_empty() || server.username.trim().is_empty() {
            self.set_status("Set host and username first", true);
            return;
        }
        // Point the Ollama backend at the tunnel so /deploy pulls on the remote,
        // remembering the old URL to restore on disconnect.
        if self.pre_remote_ollama_url.is_none() {
            self.pre_remote_ollama_url = Some(self.config.backends.ollama.base_url.clone());
        }
        self.config.backends.ollama.base_url = server.tunnel_base_url();

        let events = self.events.clone();
        let tx = self.bg_tx.clone();
        let server_for_task = server.clone();
        self.remote_connecting = Some(idx);
        self.last_failed_action = Some(RetryAction::ConnectRemote(idx));
        let handle = tokio::spawn(async move {
            let result = setup_server(&server_for_task, &events)
                .await
                .map(Box::new);
            let _ = tx.send(BgMsg::RemoteConnected {
                server_idx: idx,
                result,
            });
        });
        self.begin_busy(
            BusyKind::Remote,
            format!("Connecting to {} ({})", server.name, server.host),
            handle,
        );
    }

    fn disconnect_remote(&mut self, idx: usize) {
        let Some(name) = self
            .config
            .remote
            .servers
            .get(idx)
            .map(|s| s.name.clone())
        else {
            return;
        };
        let before = self.remote_sessions.len();
        self.remote_sessions.retain(|s| s.server_name != name);
        if self.remote_sessions.len() == before {
            return; // wasn't connected
        }
        // Restore the local Ollama URL if no remote sessions remain.
        if self.remote_sessions.is_empty() {
            if let Some(url) = self.pre_remote_ollama_url.take() {
                self.config.backends.ollama.base_url = url;
            }
            self.gpu = discover().unwrap_or_else(|_| GpuInventory {
                devices: vec![],
                detection_method: "none".into(),
                warnings: vec![],
            });
        }
        self.runtime_selected = 0;
        self.set_status(format!("Disconnected from {name}"), false);
    }

    // ------------------------------------------------------------------
    // Updates
    // ------------------------------------------------------------------

    /// Background version check. Startup checks (`manual=false`) stay quiet
    /// unless an update exists; manual checks always report the outcome.
    pub fn start_update_check(&mut self, manual: bool) {
        let checker =
            match UpdateChecker::new(&self.config.updates.repo_url, &self.config.updates.branch) {
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
            self.set_status(
                format!("v{v} already installed — restart LocalCode to apply"),
                false,
            );
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
            &self.config.updates.mirrors,
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
                            self.selected_quant = detail.quants.first().map(|q| q.label.clone());
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
                                &format!("v{} — restart LocalCode to apply", report.version),
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
                BgMsg::InstallProgress(line) => {
                    self.install_progress_line = line;
                }
                BgMsg::InstallDone { kind, result } => {
                    self.install_busy = None;
                    self.installing_kind = None;
                    self.install_progress_line.clear();
                    match result {
                        Ok(()) => {
                            self.push_notification(
                                Severity::Success,
                                "Backend installed",
                                &format!("{} installed — re-detecting", kind.as_str()),
                            );
                            self.set_status(
                                format!("{} installed — re-detecting…", kind.as_str()),
                                false,
                            );
                            // Freshly-installed binary is picked up by detect()
                            // (which::which is live); no registry rebuild needed.
                            self.start_detect();
                        }
                        Err(e) => {
                            self.last_failed_action = Some(RetryAction::InstallBackend(kind));
                            self.raise_error(e);
                        }
                    }
                }
                BgMsg::RemoteConnected { server_idx, result } => {
                    self.finish_busy();
                    self.remote_connecting = None;
                    match result {
                        Ok(session) => {
                            let session = *session;
                            let name = session.server_name.clone();
                            // Show the remote GPU in the top bar.
                            self.gpu = session.gpu.clone();
                            // Apply the repointed Ollama URL (set in connect_remote)
                            // when it's safe — no locally-managed runtimes to kill.
                            if self.runtimes.is_empty() {
                                self.registry = Arc::new(BackendRegistry::from_config(&self.config));
                            }
                            // Replace any prior session for the same server.
                            self.remote_sessions.retain(|s| s.server_name != name);
                            self.remote_sessions.push(session);
                            self.runtime_selected = self.all_runtimes().len().saturating_sub(1);
                            self.set_status(
                                format!("{name} connected — coding uses the remote GPU"),
                                false,
                            );
                        }
                        Err(e) => {
                            // Roll back the speculative Ollama repoint on failure.
                            if let Some(url) = self.pre_remote_ollama_url.take() {
                                self.config.backends.ollama.base_url = url;
                            }
                            let _ = server_idx;
                            self.last_failed_action = Some(RetryAction::ConnectRemote(server_idx));
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
        // Refresh runtimes from registry (remote runtimes live in
        // `remote_sessions`; selection indexes the combined list).
        self.runtimes = self.registry.list_runtimes().await;
        let total = self.all_runtimes().len();
        if self.runtime_selected >= total {
            self.runtime_selected = total.saturating_sub(1);
        }
    }

    // ------------------------------------------------------------------
    // Slash command menu (the popup above the omnibar when input starts "/")
    // ------------------------------------------------------------------

    /// The command catalog: `(name, args-hint, description)`. `name` includes
    /// the leading slash. Order is the display order in the menu.
    pub fn slash_catalog() -> &'static [(&'static str, &'static str, &'static str)] {
        &[
            ("/models", "[query]", "Search & deploy HuggingFace models"),
            ("/remote", "", "Connect a GPU server over SSH"),
            ("/backends", "", "Install & configure inference backends"),
            ("/runtimes", "", "Active runtimes & system overview"),
            ("/deploy", "", "Deploy the selected model"),
            ("/bench", "", "Benchmark the active runtime"),
            ("/setup", "", "Setup checklist & doctor"),
            ("/doctor", "", "Run environment diagnostics"),
            ("/settings", "", "Settings"),
            ("/theme", "", "Cycle the color theme"),
            ("/alerts", "", "Notifications"),
            ("/new", "", "Start a new coding session"),
            ("/assistant", "", "Ask the in-app assistant"),
            ("/update", "", "Install the available update"),
            ("/logs", "", "Show the log directory path"),
            ("/help", "", "Keyboard & mouse help"),
            ("/quit", "", "Quit LocalCode"),
        ]
    }

    /// Menu items filtered by whatever follows the leading slash in the omnibar.
    pub fn palette_items(&self) -> Vec<String> {
        let q = self
            .coding_input
            .trim_start_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_lowercase();
        Self::slash_catalog()
            .iter()
            .filter(|(name, _, desc)| {
                q.is_empty()
                    || name[1..].to_lowercase().contains(&q)
                    || desc.to_lowercase().contains(&q)
            })
            .map(|(name, args, desc)| {
                if args.is_empty() {
                    format!("{name}  —  {desc}")
                } else {
                    format!("{name} {args}  —  {desc}")
                }
            })
            .collect()
    }

    /// Is the omnibar currently a slash-command entry?
    pub fn slash_active(&self) -> bool {
        self.coding_input.starts_with('/')
    }

    /// Run the currently-highlighted menu command, keeping any args the user
    /// typed after the command word.
    fn run_selected_slash(&mut self) {
        let items = self.palette_items();
        let Some(item) = items.get(self.palette_selected).cloned() else {
            // No match — treat the raw text as a command line anyway.
            let raw = self.coding_input.clone();
            self.clear_input();
            self.dispatch_slash(&raw);
            return;
        };
        // Extract "/name" from the display string.
        let name = item.split_whitespace().next().unwrap_or("").to_string();
        // Preserve any args the user typed: "/models qwen" -> keep "qwen".
        let typed = self.coding_input.clone();
        let args = typed
            .split_whitespace()
            .skip(1)
            .collect::<Vec<_>>()
            .join(" ");
        let line = if args.is_empty() {
            name
        } else {
            format!("{name} {args}")
        };
        self.clear_input();
        self.dispatch_slash(&line);
    }

    /// Parse and run a slash command line (leading '/'), with space-separated
    /// arguments.
    fn dispatch_slash(&mut self, line: &str) {
        let line = line.trim();
        let mut parts = line.trim_start_matches('/').split_whitespace();
        let cmd = parts.next().unwrap_or("").to_lowercase();
        let args: Vec<String> = parts.map(|s| s.to_string()).collect();
        let rest = args.join(" ");
        match cmd.as_str() {
            "models" | "search" => {
                self.set_panel(Some(Panel::Models));
                if !rest.is_empty() {
                    self.model_query = rest;
                    self.start_search();
                } else if self.models.is_empty() {
                    self.start_search();
                }
            }
            "popular" => {
                self.set_panel(Some(Panel::Models));
                self.model_query = "code".into();
                self.start_search();
            }
            "trending" => {
                self.set_panel(Some(Panel::Models));
                self.start_trending();
            }
            "remote" => {
                if args.first().map(|s| s.as_str()) == Some("add") {
                    self.remote_quick_add(&args[1..]);
                } else {
                    self.open_remote_panel();
                }
            }
            "connect" => {
                self.open_remote_panel();
                self.connect_remote(self.remote_selected);
            }
            "disconnect" => self.disconnect_remote(self.remote_selected),
            "backends" => self.open_backend_manager(),
            "runtimes" | "dashboard" | "home" => self.set_panel(Some(Panel::Runtimes)),
            "deploy" => self.start_deploy(false),
            "bench" | "benchmark" => self.set_panel(Some(Panel::Benchmarks)),
            "setup" => self.set_panel(Some(Panel::Setup)),
            "doctor" => {
                self.set_panel(Some(Panel::Setup));
                self.start_doctor();
            }
            "settings" => self.set_panel(Some(Panel::Settings)),
            "theme" => self.cycle_theme(),
            "alerts" | "notifications" => self.set_panel(Some(Panel::Notifications)),
            "new" => self.new_coding_session(),
            "assistant" | "ask" => self.start_assistant(),
            "update" => self.open_update_modal(),
            "redetect" | "detect" => {
                self.start_detect();
                self.set_status("Re-detecting backends…", false);
            }
            "stop" => self.start_stop_runtime(),
            "error" => self.open_error_modal(),
            "context" | "ctx" => match args.first().map(|s| s.as_str()) {
                Some("down") => self.adjust_ctx(false),
                Some("up") | None => self.adjust_ctx(true),
                Some(n) => {
                    if let Ok(v) = n.parse::<u32>() {
                        self.deploy_ctx = v.clamp(512, 131_072);
                        self.refresh_fit();
                    } else {
                        self.set_status("Usage: /context <number|up|down>", true);
                    }
                }
            },
            "quant" => {
                if args.first().map(|s| s.as_str()) == Some("prev") {
                    self.cycle_quant(-1);
                } else {
                    self.cycle_quant(1);
                }
            }
            "backend" => {
                if let Some(name) = args.first() {
                    if let Some(k) = BackendKind::parse(name) {
                        self.deploy_backend = k;
                        self.refresh_fit();
                    } else {
                        self.set_status(format!("Unknown backend: {name}"), true);
                    }
                } else {
                    self.cycle_deploy_backend();
                }
            }
            "logs" => self.set_status(format!("Logs: {}", self.paths.log_dir.display()), false),
            "help" => self.open_help(),
            "quit" | "exit" | "q" => self.request_quit(),
            "" => {}
            other => self.set_status(format!("Unknown command: /{other} — press / to list"), true),
        }
    }

    fn cycle_theme(&mut self) {
        self.config.ui.theme = match self.config.ui.theme {
            ThemeMode::Dark => ThemeMode::Light,
            ThemeMode::Light => ThemeMode::HighContrast,
            ThemeMode::HighContrast => ThemeMode::Dark,
        };
        self.theme = Theme::new(self.config.ui.theme);
        self.set_status(format!("Theme: {:?}", self.config.ui.theme), false);
    }

    // ------------------------------------------------------------------
    // Help / quit
    // ------------------------------------------------------------------

    fn open_help(&mut self) {
        let body = "\
The bottom bar is always active — just type.\n\
  • Type a message + Enter  → chat with the agent\n\
  • Type /  → open the command menu (↑/↓ pick, Enter run, Esc close)\n\
  • Ctrl+K  → open the command menu\n\
  • Esc  → close a panel, or cancel the running task at home\n\
  • Ctrl+↑/↓  → grow/shrink the bar    Ctrl+S save    Ctrl+C quit\n\
\n\
Commands (type / to see them all):\n\
  /models [q]   search & deploy HuggingFace models\n\
  /remote       connect a GPU server over SSH (one-click)\n\
  /backends     install & configure inference backends\n\
  /runtimes     active runtimes & system overview\n\
  /deploy       deploy the selected model\n\
  /bench /setup /doctor /settings /theme /alerts /new /assistant /update /logs /quit\n\
\n\
Panels open as popups above the bar. Inside a panel: ↑/↓ navigate,\n\
Enter acts, click rows/buttons with the mouse, Esc closes.\n\
Models: click a model to open it, click the card to focus, scroll to read,\n\
click Backend/Quant/Deploy in the right pane.\n\
Remote: click a field to edit, then Connect — or /remote add <name> <host> <user> <password>."
            .to_string();
        self.modal = Some(ModalState::info("Help", body));
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
                    // Open the slash-command menu by seeding a leading '/'.
                    if !self.coding_input.starts_with('/') {
                        self.coding_input.insert(0, '/');
                        self.coding_cursor = self.coding_input.chars().count();
                    }
                    self.palette_selected = 0;
                    return;
                }
                KeyCode::Char('s') => {
                    // Keep the in-progress backend field edit in the saved file.
                    if self.backends_open {
                        self.commit_field_edit();
                    }
                    if let Err(e) = self.config.save(&self.paths) {
                        self.raise_error(e);
                    } else {
                        self.set_status("Config saved", false);
                    }
                    return;
                }
                // Omnibar height.
                KeyCode::Up => {
                    self.adjust_composer_rows(1);
                    return;
                }
                KeyCode::Down => {
                    self.adjust_composer_rows(-1);
                    return;
                }
                _ => {}
            }
        }

        // Overlays that fully capture input, in stacking order.
        if let Some(modal) = self.modal.clone() {
            self.handle_modal_key(key, &modal);
            return;
        }
        if self.assistant_open {
            match key.code {
                KeyCode::Esc => self.assistant_open = false,
                KeyCode::Up => self.assistant_scroll = self.assistant_scroll.saturating_sub(1),
                KeyCode::Down => self.assistant_scroll = self.assistant_scroll.saturating_add(1),
                KeyCode::PageUp => self.assistant_scroll = self.assistant_scroll.saturating_sub(10),
                KeyCode::PageDown => {
                    self.assistant_scroll = self.assistant_scroll.saturating_add(10)
                }
                _ => {}
            }
            return;
        }
        if self.backends_open {
            self.handle_backends_key(key);
            return;
        }

        // The slash-command menu (omnibar text starts with '/') owns navigation
        // and Enter; other keys still edit the text.
        if self.slash_active() {
            match key.code {
                KeyCode::Esc => self.clear_input(),
                KeyCode::Up => self.palette_selected = self.palette_selected.saturating_sub(1),
                KeyCode::Down => {
                    let n = self.palette_items().len();
                    if n > 0 {
                        self.palette_selected = (self.palette_selected + 1).min(n - 1);
                    }
                }
                KeyCode::Enter | KeyCode::Tab => self.run_selected_slash(),
                _ => self.omnibar_edit(key),
            }
            return;
        }

        // A panel owns navigation keys; typing still flows to the omnibar so the
        // prompt bar works in every mode.
        if let Some(panel) = self.panel {
            if self.remote_editing && panel == Panel::Remote {
                self.handle_remote_field_key(key);
                return;
            }
            match key.code {
                KeyCode::Esc => self.set_panel(None),
                KeyCode::Up
                | KeyCode::Down
                | KeyCode::Left
                | KeyCode::Right
                | KeyCode::PageUp
                | KeyCode::PageDown
                | KeyCode::Tab
                | KeyCode::BackTab => self.panel_nav(panel, key.code),
                KeyCode::Enter => self.panel_enter(panel),
                _ => self.omnibar_edit(key),
            }
            return;
        }

        // Home: the omnibar.
        match key.code {
            KeyCode::Esc => self.cancel_current(),
            KeyCode::Enter => self.start_coding_turn(),
            KeyCode::Up => self.history_prev(),
            KeyCode::Down => self.history_next(),
            _ => self.omnibar_edit(key),
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

    // ------------------------------------------------------------------
    // Omnibar (the persistent prompt/search/command bar)
    // ------------------------------------------------------------------

    /// Text-editing keys for the omnibar (no Enter/Up/Down — those are
    /// context-dependent and handled by the caller).
    fn omnibar_edit(&mut self, key: crossterm::event::KeyEvent) {
        match key.code {
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
                self.palette_selected = 0;
            }
            KeyCode::Delete => {
                if self.coding_cursor < self.coding_input.chars().count() {
                    let idx = char_to_byte(&self.coding_input, self.coding_cursor);
                    self.coding_input.remove(idx);
                }
            }
            KeyCode::Char(c) => {
                let idx = char_to_byte(&self.coding_input, self.coding_cursor);
                self.coding_input.insert(idx, c);
                self.coding_cursor += 1;
                self.coding_hist_idx = None;
                self.palette_selected = 0;
            }
            _ => {}
        }
    }

    fn history_prev(&mut self) {
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

    fn history_next(&mut self) {
        match self.coding_hist_idx {
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
        }
    }

    fn clear_input(&mut self) {
        self.coding_input.clear();
        self.coding_cursor = 0;
        self.coding_hist_idx = None;
        self.palette_selected = 0;
    }

    fn set_panel(&mut self, panel: Option<Panel>) {
        self.panel = panel;
        self.remote_editing = false;
    }

    /// Navigation keys within the active panel (arrows, page keys, Tab).
    fn panel_nav(&mut self, panel: Panel, code: KeyCode) {
        match panel {
            Panel::Models => match code {
                KeyCode::Left => self.models_focus = ModelsPane::List,
                KeyCode::Right => {
                    if self.model_detail.is_some() {
                        self.models_focus = ModelsPane::Card;
                    }
                }
                KeyCode::Down => match self.models_focus {
                    ModelsPane::List => {
                        if !self.models.is_empty() {
                            self.model_selected =
                                (self.model_selected + 1).min(self.models.len() - 1);
                        }
                    }
                    ModelsPane::Card => self.scroll_card(1),
                },
                KeyCode::Up => match self.models_focus {
                    ModelsPane::List => self.model_selected = self.model_selected.saturating_sub(1),
                    ModelsPane::Card => self.scroll_card(-1),
                },
                KeyCode::PageDown => self.scroll_card(self.card_view_height.max(1) as i64),
                KeyCode::PageUp => self.scroll_card(-(self.card_view_height.max(1) as i64)),
                _ => {}
            },
            Panel::Runtimes => match code {
                KeyCode::Down => {
                    let n = self.all_runtimes().len();
                    if n > 0 {
                        self.runtime_selected = (self.runtime_selected + 1).min(n - 1);
                    }
                }
                KeyCode::Up => self.runtime_selected = self.runtime_selected.saturating_sub(1),
                _ => {}
            },
            Panel::Notifications => match code {
                KeyCode::Down => {
                    if !self.notifications.is_empty() {
                        self.notif_selected =
                            (self.notif_selected + 1).min(self.notifications.len() - 1);
                    }
                }
                KeyCode::Up => self.notif_selected = self.notif_selected.saturating_sub(1),
                _ => {}
            },
            Panel::Setup => match code {
                KeyCode::PageUp => self.setup_scroll = self.setup_scroll.saturating_sub(10),
                KeyCode::PageDown => self.setup_scroll = self.setup_scroll.saturating_add(10),
                _ => {}
            },
            Panel::Remote => match code {
                KeyCode::Down => {
                    let n = self.config.remote.servers.len();
                    if n > 0 {
                        self.remote_selected = (self.remote_selected + 1).min(n - 1);
                        self.remote_field = 0;
                    }
                }
                KeyCode::Up => {
                    self.remote_selected = self.remote_selected.saturating_sub(1);
                    self.remote_field = 0;
                }
                KeyCode::Tab => {
                    let n = REMOTE_FIELDS.len();
                    self.remote_field = (self.remote_field + 1) % n;
                    self.load_remote_field_edit();
                    self.remote_editing = true;
                }
                _ => {}
            },
            Panel::Benchmarks | Panel::Settings => {}
        }
    }

    /// Enter within the active panel (the primary action).
    fn panel_enter(&mut self, panel: Panel) {
        match panel {
            Panel::Models => {
                let q = self.coding_input.trim().to_string();
                if !q.is_empty() {
                    self.model_query = q;
                    self.clear_input();
                    self.start_search();
                } else {
                    self.start_load_detail();
                }
            }
            Panel::Remote => self.connect_remote(self.remote_selected),
            // Other panels: Enter sends any typed text to the agent so the bar
            // still prompts from anywhere.
            _ => {
                if !self.coding_input.trim().is_empty() {
                    self.start_coding_turn();
                }
            }
        }
    }

    /// Shift the boundary between panes `idx` and `idx+1` of a view by
    /// `delta`. Ratios are persisted with the config (saved on quit/Ctrl+S).
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
            KeyCode::Enter => self.activate_modal_button(),
            _ => {}
        }
    }

    /// Run the currently-selected modal button's action. Shared by the Enter
    /// key and by clicking a button (which sets `selected` first).
    fn activate_modal_button(&mut self) {
        let Some(modal) = self.modal.clone() else {
            return;
        };
        let label = modal.selected_button();
        let kind = modal.kind.clone();
        self.modal = None;
        match (&kind, label) {
            (ModalKind::Error { .. }, "Retry") => self.retry_last(),
            (ModalKind::Error { .. }, "Open logs") => {
                self.set_status(format!("Logs: {}", self.paths.log_dir.display()), false);
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
                    action: ConfirmAction::InstallBackend(kind),
                    ..
                },
                "Confirm",
            ) => self.spawn_install(*kind),
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

    fn handle_mouse(&mut self, mouse: crossterm::event::MouseEvent) {
        use event::MouseButton::Left;
        let (col, row) = (mouse.column, mouse.row);

        // While an overlay owns the screen, the view underneath is inert —
        // border drags are suppressed (matches the key path, which routes to
        // the overlay and returns).
        let overlay = self.modal.is_some()
            || self.slash_active()
            || self.assistant_open
            || self.backends_open
            || self.panel.is_some();

        match mouse.kind {
            MouseEventKind::Down(Left) => {
                if !overlay {
                    // A draggable pane seam, then a normal click.
                    if let Some(b) = self.hit_resize_border(col, row) {
                        self.dragging = Some(DragState {
                            view: b.view,
                            idx: b.idx,
                            area: b.area,
                        });
                        self.set_pane_boundary(b.view, b.idx, col, b.area);
                        self.set_status("Resizing pane — drag the border, saved on quit", false);
                        return;
                    }
                }
                self.handle_left_click(col, row);
            }
            MouseEventKind::Drag(Left) => {
                if let Some(d) = self.dragging {
                    self.set_pane_boundary(d.view, d.idx, col, d.area);
                }
            }
            MouseEventKind::Up(Left) => self.dragging = None,
            MouseEventKind::ScrollUp => self.wheel_scroll_at(-3, col, row),
            MouseEventKind::ScrollDown => self.wheel_scroll_at(3, col, row),
            _ => {}
        }
    }

    /// The draggable pane seam under `(col, row)`, if any. A ±1 column
    /// tolerance makes the 1-cell border easy to grab.
    fn hit_resize_border(&self, col: u16, row: u16) -> Option<ResizeBorder> {
        self.resize_borders
            .iter()
            .copied()
            .find(|b| row >= b.y0 && row < b.y1 && b.x.abs_diff(col) <= 1)
    }

    /// The topmost click region under `(col, row)`, if any. Regions recorded
    /// later in the frame (overlays) win, hence the reverse scan.
    fn region_at(&self, col: u16, row: u16) -> Option<ClickRegion> {
        self.click_regions.iter().rev().copied().find(|r| {
            col >= r.rect.x
                && col < r.rect.x.saturating_add(r.rect.width)
                && row >= r.rect.y
                && row < r.rect.y.saturating_add(r.rect.height)
        })
    }

    /// Move the seam between panes `idx` and `idx + 1` of `view` so it lands
    /// under column `col`, trading width only between those two panes (others
    /// stay fixed), with a 0.15 floor each side. Persisted like `[`/`]`.
    fn set_pane_boundary(&mut self, view: &'static str, idx: usize, col: u16, area: Rect) {
        let mut ratios = self.pane_ratios(view, Self::pane_defaults(view));
        if idx + 1 >= ratios.len() || area.width == 0 {
            return;
        }
        let left_frac: f32 = ratios[..idx].iter().sum();
        let pair = ratios[idx] + ratios[idx + 1];
        let raw = col.saturating_sub(area.x) as f32 / area.width as f32;
        let mut a = (raw - left_frac).clamp(0.15, pair - 0.15);
        if !a.is_finite() {
            a = pair * 0.5; // pair < 0.30: degenerate, split evenly
        }
        ratios[idx] = a;
        ratios[idx + 1] = pair - a;
        let sum: f32 = ratios.iter().sum();
        if sum > 0.0 {
            for r in &mut ratios {
                *r /= sum;
            }
        }
        self.config.panes.views.insert(view.into(), ratios);
    }

    /// Dispatch a left click to whatever region it landed on. While a modal or
    /// palette is open, only that overlay's controls are actionable.
    fn handle_left_click(&mut self, col: u16, row: u16) {
        // The assistant dock covers the view and is dismissed with Esc/q; it
        // has no click controls, so swallow clicks rather than passing them
        // through to whatever is drawn underneath.
        if self.assistant_open {
            return;
        }
        let Some(region) = self.region_at(col, row) else {
            return;
        };

        if self.modal.is_some() {
            if let ClickTarget::ModalButton(i) = region.target {
                if let Some(m) = &mut self.modal {
                    m.selected = i;
                }
                self.activate_modal_button();
            }
            return;
        }
        if self.slash_active() {
            if let ClickTarget::PaletteItem(i) = region.target {
                self.palette_selected = i;
                self.run_selected_slash();
            }
            return;
        }
        if self.backends_open {
            let rel_row = row.saturating_sub(region.rect.y) as usize;
            match region.target {
                ClickTarget::BackendMgrItem => {
                    if rel_row < BACKEND_ORDER.len() {
                        self.commit_field_edit();
                        self.backend_sel = rel_row;
                        self.backend_field = 0;
                        self.backend_editing = false;
                        self.load_field_edit();
                        self.refresh_plan_preview();
                    }
                }
                ClickTarget::BackendMgrField => {
                    if rel_row < self.backend_field_count() {
                        self.commit_field_edit();
                        self.backend_field = rel_row;
                        self.load_field_edit();
                        self.backend_editing = true;
                    }
                }
                ClickTarget::BackendMgrInstall => {
                    let k = self.backend_sel_kind();
                    self.start_install(k);
                }
                ClickTarget::BackendMgrSave => self.save_backend_config(),
                ClickTarget::BackendMgrRedetect => {
                    self.start_detect();
                    self.set_status("Re-detecting backends…", false);
                }
                _ => {}
            }
            return;
        }
        if self.panel == Some(Panel::Remote) {
            let rel_row = row.saturating_sub(region.rect.y) as usize;
            match region.target {
                ClickTarget::RemoteList => {
                    if rel_row < self.config.remote.servers.len() {
                        self.commit_remote_field();
                        self.remote_selected = rel_row;
                        self.remote_field = 0;
                        self.remote_editing = false;
                    }
                }
                ClickTarget::RemoteField => {
                    if rel_row < REMOTE_FIELDS.len() {
                        self.commit_remote_field();
                        self.remote_field = rel_row;
                        self.load_remote_field_edit();
                        self.remote_editing = true;
                    }
                }
                ClickTarget::RemoteConnect => self.connect_remote(self.remote_selected),
                ClickTarget::RemoteSave => self.save_remote_config(),
                ClickTarget::RemoteDisconnect => self.disconnect_remote(self.remote_selected),
                ClickTarget::RemoteDelete => self.delete_remote(self.remote_selected),
                ClickTarget::RemoteNew => self.new_remote_server(),
                ClickTarget::PanelClose => self.set_panel(None),
                _ => {}
            }
            return;
        }

        // Row index within a list = first visible row + click offset.
        let rel_row = row.saturating_sub(region.rect.y) as usize;
        match region.target {
            ClickTarget::PanelClose => self.set_panel(None),
            ClickTarget::RuntimeList => {
                let n = self.all_runtimes().len();
                if n > 0 {
                    let idx = self.runtime_list_state.offset() + rel_row;
                    if idx < n {
                        self.runtime_selected = idx;
                    }
                }
            }
            ClickTarget::ModelList => {
                self.models_focus = ModelsPane::List;
                if !self.models.is_empty() {
                    let idx = self.model_list_state.offset() + rel_row;
                    if idx < self.models.len() {
                        self.model_selected = idx;
                        self.start_load_detail(); // single-click opens the card
                    }
                }
            }
            ClickTarget::ModelCard => {
                if self.model_detail.is_some() {
                    self.models_focus = ModelsPane::Card;
                }
            }
            ClickTarget::DeployButton => self.start_deploy(false),
            ClickTarget::QuantList => self.cycle_quant(1),
            ClickTarget::BackendCycle => self.cycle_deploy_backend(),
            ClickTarget::SetupManageBackends => self.open_backend_manager(),
            ClickTarget::NotificationList => {
                if !self.notifications.is_empty() {
                    let idx = self.notif_list_state.offset() + rel_row;
                    if idx < self.notifications.len() {
                        self.notif_selected = idx;
                    }
                }
            }
            // Panels/regions with no per-click action (their keys still work).
            ClickTarget::DeployPane
            | ClickTarget::SetupBody
            | ClickTarget::Transcript
            | ClickTarget::Composer => {}
            ClickTarget::ModalButton(_) | ClickTarget::PaletteItem(_) => {}
            ClickTarget::BackendMgrItem
            | ClickTarget::BackendMgrField
            | ClickTarget::BackendMgrInstall
            | ClickTarget::BackendMgrSave
            | ClickTarget::BackendMgrRedetect => {}
            ClickTarget::RemoteList
            | ClickTarget::RemoteField
            | ClickTarget::RemoteConnect
            | ClickTarget::RemoteSave
            | ClickTarget::RemoteDisconnect
            | ClickTarget::RemoteDelete
            | ClickTarget::RemoteNew => {}
        }
    }

    fn cycle_deploy_backend(&mut self) {
        self.deploy_backend = match self.deploy_backend {
            BackendKind::Ollama => BackendKind::LlamaCpp,
            BackendKind::LlamaCpp => BackendKind::Vllm,
            BackendKind::Vllm => BackendKind::Sglang,
            BackendKind::Sglang => BackendKind::Ollama,
        };
        self.refresh_fit();
    }

    /// Route the wheel to whatever scrollable is under the cursor, falling back
    /// to the active tab's main scrollable.
    fn wheel_scroll_at(&mut self, delta: i64, col: u16, row: u16) {
        match self.region_at(col, row).map(|r| r.target) {
            Some(ClickTarget::ModelList) => self.scroll_list_models(delta),
            Some(ClickTarget::ModelCard) | Some(ClickTarget::DeployPane) => self.scroll_card(delta),
            Some(ClickTarget::Transcript) => self.scroll_transcript(delta),
            Some(ClickTarget::SetupBody) => {
                self.setup_scroll = (i64::from(self.setup_scroll) + delta).max(0) as u16;
            }
            Some(ClickTarget::NotificationList) => self.scroll_notifs(delta),
            Some(ClickTarget::RuntimeList) => self.scroll_runtimes(delta),
            _ => self.wheel_scroll(delta),
        }
    }

    /// Fallback wheel routing by active panel (cursor not over a known region).
    fn wheel_scroll(&mut self, delta: i64) {
        match self.panel {
            None => self.scroll_transcript(delta),
            Some(Panel::Models) => self.scroll_card(delta),
            Some(Panel::Setup) => {
                self.setup_scroll = (i64::from(self.setup_scroll) + delta).max(0) as u16;
            }
            Some(Panel::Notifications) => self.scroll_notifs(delta),
            _ => {}
        }
    }

    fn scroll_transcript(&mut self, delta: i64) {
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

    fn scroll_notifs(&mut self, delta: i64) {
        if self.notifications.is_empty() {
            return;
        }
        let max = self.notifications.len() as i64 - 1;
        self.notif_selected = (self.notif_selected as i64 + delta.signum()).clamp(0, max) as usize;
    }

    fn scroll_list_models(&mut self, delta: i64) {
        if self.models.is_empty() {
            return;
        }
        let max = self.models.len() as i64 - 1;
        self.model_selected = (self.model_selected as i64 + delta.signum()).clamp(0, max) as usize;
    }

    fn scroll_runtimes(&mut self, delta: i64) {
        let n = self.all_runtimes().len();
        if n == 0 {
            return;
        }
        let max = n as i64 - 1;
        self.runtime_selected =
            (self.runtime_selected as i64 + delta.signum()).clamp(0, max) as usize;
    }
}

/// Fixed display order of backends in the manager overlay.
pub(crate) const BACKEND_ORDER: [BackendKind; 4] = [
    BackendKind::Ollama,
    BackendKind::LlamaCpp,
    BackendKind::Vllm,
    BackendKind::Sglang,
];

/// Editable fields of a remote server, in display order.
pub(crate) const REMOTE_FIELDS: [&str; 6] =
    ["name", "host", "port", "username", "password", "local_port"];

/// Extra context shown above the install confirm command, per backend.
fn install_caveat(kind: BackendKind) -> String {
    match kind {
        BackendKind::Vllm | BackendKind::Sglang => {
            "Note: vLLM and SGLang are Linux-preferred; on Windows this is best-effort.\n\n".into()
        }
        _ => String::new(),
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
    app.autoconnect_remotes();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn test_app() -> App {
        let dir = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_home(dir.path().to_path_buf());
        paths.ensure_dirs().unwrap();
        App::new(paths, Config::default())
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    #[test]
    fn backend_manager_open_navigate_edit_and_install_modal() {
        let mut app = test_app();
        // Skip the detect spawn (this sync test has no tokio runtime).
        app.detecting = true;

        app.open_backend_manager();
        assert!(app.backends_open);
        assert_eq!(app.backend_sel_kind(), BackendKind::Ollama);

        // Tab cycles backends and resets the field selection.
        app.handle_backends_key(key(KeyCode::Tab));
        assert_eq!(app.backend_sel_kind(), BackendKind::LlamaCpp);
        assert_eq!(app.backend_field, 0);
        app.handle_backends_key(key(KeyCode::BackTab));
        assert_eq!(app.backend_sel_kind(), BackendKind::Ollama);

        // Edit base_url: 'e' focuses the field, Enter commits to config.
        app.handle_backends_key(key(KeyCode::Char('e')));
        assert!(app.backend_editing);
        app.backend_field_edit = "http://host:1234".into();
        app.handle_backends_key(key(KeyCode::Enter));
        assert!(!app.backend_editing);
        assert_eq!(app.config.backends.ollama.base_url, "http://host:1234");

        // Install shows a modal (confirm or manual) but never spawns here.
        app.start_install(BackendKind::Ollama);
        assert!(app.modal.is_some());
        assert!(app.install_busy.is_none());

        // With no install running, Esc closes the overlay.
        app.modal = None;
        app.handle_backends_key(key(KeyCode::Esc));
        assert!(!app.backends_open);
    }

    #[test]
    fn llamacpp_has_three_editable_fields() {
        assert_eq!(App::backend_field_labels(BackendKind::LlamaCpp).len(), 3);
        assert_eq!(App::backend_field_labels(BackendKind::Ollama), &["base_url"]);
    }

    fn typ(app: &mut App, s: &str) {
        for c in s.chars() {
            app.handle_key(key(KeyCode::Char(c)));
        }
    }

    #[test]
    fn typing_slash_opens_command_menu_and_filters() {
        let mut app = test_app();
        assert!(!app.slash_active());
        typ(&mut app, "/rem");
        assert!(app.slash_active());
        let items = app.palette_items();
        assert!(items.iter().any(|i| i.starts_with("/remote")));
        assert!(!items.iter().any(|i| i.starts_with("/models")));
    }

    #[test]
    fn slash_command_opens_panel_then_esc_closes() {
        let mut app = test_app();
        typ(&mut app, "/settings");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.panel, Some(Panel::Settings));
        assert!(app.coding_input.is_empty(), "omnibar cleared after running command");
        // Esc closes the panel and returns home.
        app.handle_key(key(KeyCode::Esc));
        assert_eq!(app.panel, None);
    }

    #[test]
    fn theme_command_cycles_theme() {
        let mut app = test_app();
        let before = app.config.ui.theme;
        typ(&mut app, "/theme");
        app.handle_key(key(KeyCode::Enter));
        assert_ne!(app.config.ui.theme, before);
    }

    #[test]
    fn plain_text_is_not_a_command() {
        let mut app = test_app();
        typ(&mut app, "hello world");
        assert!(!app.slash_active());
        assert_eq!(app.coding_input, "hello world");
    }

    #[test]
    fn remote_quick_add_appends_server_and_opens_panel() {
        let mut app = test_app();
        typ(&mut app, "/remote add gpu 10.0.0.9 root secret");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.panel, Some(Panel::Remote));
        assert_eq!(app.config.remote.servers.len(), 1);
        let s = &app.config.remote.servers[0];
        assert_eq!(s.name, "gpu");
        assert_eq!(s.host, "10.0.0.9");
        assert_eq!(s.username, "root");
        assert_eq!(s.password, "secret");
    }

    #[test]
    fn remote_field_edit_writes_config() {
        let mut app = test_app();
        app.new_remote_server();
        app.open_remote_panel();
        // Click-to-edit the host field (index 1), type, commit.
        app.remote_field = 1;
        app.load_remote_field_edit();
        app.remote_editing = true;
        app.remote_field_edit = "192.168.1.50".into();
        app.handle_remote_field_key(key(KeyCode::Enter));
        assert!(!app.remote_editing);
        assert_eq!(app.config.remote.servers[0].host, "192.168.1.50");
    }

    /// Render every surface once to a headless backend — catches layout panics
    /// (out-of-bounds rects, bad splits) across the redesign.
    #[test]
    fn all_surfaces_render_without_panic() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = test_app();
        app.detecting = false;
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();

        // Home (transcript + omnibar).
        terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();

        // Each panel.
        for panel in [
            Panel::Models,
            Panel::Runtimes,
            Panel::Benchmarks,
            Panel::Setup,
            Panel::Notifications,
            Panel::Settings,
            Panel::Remote,
        ] {
            app.panel = Some(panel);
            terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
        }
        app.panel = None;

        // Slash menu open.
        typ(&mut app, "/mod");
        terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
        app.clear_input();

        // Backends overlay + a remote server present.
        app.backends_open = true;
        app.new_remote_server();
        app.panel = Some(Panel::Remote);
        terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();

        // A tiny terminal must not panic either.
        let mut tiny = Terminal::new(TestBackend::new(30, 8)).unwrap();
        tiny.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
    }
}
