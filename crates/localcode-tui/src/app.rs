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
use localcode_agent::{AgentEvent, AgentSession, CodingAgent, Skill, SkillLoader, ToolApprover, ToolRegistry};
use localcode_api_client::ApiClient;
use localcode_assistant::{Assistant, AssistantRequest};
use localcode_backends::{
    can_elevate_noninteractively, diagnose, resolve_install_plan, resolve_repair, run_install,
    run_repair, smoke_test, BackendKind, BackendRegistry, DeployRequest, DeployService, DetectReport,
    Diagnosis, InstallPlan, RepairPlan, Repoint, SmokeReport,
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
use zeroize::Zeroizing;

const MAX_INPUT_HISTORY: usize = 100;

/// Which view fills the working area. `Chat` is home (the transcript). The rest
/// are switched to by slash commands; a leading '/' in the omnibar transiently
/// shows the command list over whichever mode is active (see `slash_active`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Chat,
    Models,
    Runtimes,
    Remote,
    Backends,
    Bench,
    Setup,
    Settings,
}

impl Mode {
    /// Short square-tag label shown in the omnibar for non-chat modes.
    pub fn tag(self) -> &'static str {
        match self {
            Mode::Chat => "chat",
            Mode::Models => "models",
            Mode::Runtimes => "runtimes",
            Mode::Remote => "remote",
            Mode::Backends => "backends",
            Mode::Bench => "bench",
            Mode::Setup => "setup",
            Mode::Settings => "settings",
        }
    }

    /// Omnibar placeholder text for this mode (chat / search / read-only views).
    pub fn placeholder(self) -> &'static str {
        match self {
            Mode::Chat => "message the agent…    / for commands",
            Mode::Models => "search models — type to filter, Enter to run",
            Mode::Settings => "↑↓ move · Enter toggle/edit · or type to chat",
            _ => "type to chat, or / for commands",
        }
    }
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
    // Status bar
    Theme(ThemeMode),
    UpdateBadge,
    // Command list (shown while the omnibar starts with '/')
    CommandItem(usize),
    // Inline banner buttons (were modal buttons)
    ModalButton(usize),
    // Chat
    Transcript,
    // Models (two-pane)
    ModelList,
    QuantList,
    BackendCycle,
    DeployButton,
    // Runtimes
    RuntimeList,
    // Backends — index into BACKEND_ORDER
    BackendInstall(usize),
    BackendSmoke(usize),
    BackendFix(usize),
    BackendRedetect,
    // Remote (two-pane)
    RemoteList,
    RemoteField,
    RemoteConnect,
    RemoteSave,
    RemoteDisconnect,
    RemoteDelete,
    RemoteNew,
    // Setup — index into the checklist
    SetupStep(usize),
    SetupDoctor,
    // Bench
    BenchRun,
    // Settings — a row's action (toggle / edit / theme / reload / …)
    Setting(SettingAction),
}

/// An editable free-text Settings field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingField {
    SystemPrompt,
    SkillsDir,
    McpPath,
    Workspace,
}

/// What activating a Settings row does. `Copy` so it can ride inside
/// [`ClickTarget`]; index-carrying variants point into stable catalogs
/// (`ToolRegistry::catalog()` / `App::skills`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingAction {
    ThemeToggle,
    ToggleMouse,
    ToggleStream,
    ToggleConfirmShell,
    ToggleCloudFallback,
    ToggleAgentsMd,
    ToggleCheckUpdates,
    ToggleTool(usize),
    ToggleSkill(usize),
    ReloadSkills,
    Edit(SettingField),
}

/// How a Settings row renders. `Toggle` carries the current on/off state.
#[derive(Debug, Clone)]
pub enum SettingsRowKind {
    Header,
    Toggle(bool),
    /// Editable free text (click to edit inline).
    Text,
    /// A one-shot action word, e.g. `[ reload ]`.
    Action(&'static str),
    /// Read-only informational row.
    Info,
}

/// One row of the Settings view. Built by [`App::settings_rows`] and consumed by
/// both the renderer and the keyboard/mouse handlers, so they never drift.
#[derive(Debug, Clone)]
pub struct SettingsRow {
    pub label: String,
    pub value: String,
    pub kind: SettingsRowKind,
    pub action: Option<SettingAction>,
}

impl SettingsRow {
    fn header(label: &str) -> Self {
        Self {
            label: label.to_string(),
            value: String::new(),
            kind: SettingsRowKind::Header,
            action: None,
        }
    }

    fn info(label: &str, value: impl Into<String>) -> Self {
        Self {
            label: label.to_string(),
            value: value.into(),
            kind: SettingsRowKind::Info,
            action: None,
        }
    }
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
    ApplyRepair,
}

/// What a collected sudo password authorizes — an elevated backend install
/// (Ollama on Linux) or a diagnosed repair. Both carry `InstallStep`s, so the
/// same masked prompt and `sudo -S` runner serve both.
enum SudoJob {
    Install { kind: BackendKind, plan: InstallPlan },
    Repair(RepairPlan),
}

impl SudoJob {
    /// The `sudo …` command being authorized, shown next to the masked prompt so
    /// the user sees exactly what their password runs. Falls back to a human
    /// label if the plan somehow carries no elevated step.
    fn sudo_display(&self) -> String {
        let steps: &[localcode_backends::InstallStep] = match self {
            SudoJob::Install {
                plan: InstallPlan::Automated { steps, .. },
                ..
            } => steps,
            SudoJob::Repair(plan) => &plan.steps,
            SudoJob::Install { .. } => &[],
        };
        steps
            .iter()
            .find_map(|s| match s {
                localcode_backends::InstallStep::Sudo { display, .. } => Some(display.clone()),
                _ => None,
            })
            .unwrap_or_else(|| match self {
                SudoJob::Install { kind, .. } => format!("install {}", kind.as_str()),
                SudoJob::Repair(plan) => plan.title.clone(),
            })
    }
}

/// An elevated action blocked on a sudo password. The password lives ONLY here
/// (a `Zeroizing<String>`, wiped on drop) until the user submits it; it is never
/// stored on `App` afterward, never crosses the `BgMsg` channel, never logged.
struct PendingSudo {
    job: SudoJob,
    buf: Zeroizing<String>,
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
    /// A backend install: streamed output lines, then the terminal result. On
    /// success `repoint` (Some for a fetched llama.cpp binary) points the
    /// backend at the freshly-installed managed binary.
    InstallProgress(String),
    InstallDone {
        kind: BackendKind,
        result: Result<Option<Repoint>, LocalCodeError>,
    },
    /// A diagnosed repair: streamed output lines, then the terminal result. On
    /// success the `repoint` (if any) points the backend at the repaired install.
    RepairProgress(String),
    RepairDone {
        result: Result<(), LocalCodeError>,
        repoint: Option<Repoint>,
    },
    /// A single backend's smoke-test finished (Backends view / doctor probe).
    SmokeDone(SmokeReport),
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
    /// Whether the app is currently capturing the mouse. "Select mode" (F2 /
    /// `/select`) flips this off so the terminal's own drag-to-select/copy
    /// works; the run loop applies the change to the real terminal.
    pub mouse_capture: bool,
    /// Which view fills the working area (Chat is home).
    pub mode: Mode,
    /// Clickable regions for the current frame, refilled every draw so the
    /// mouse handler never recomputes layout.
    pub click_regions: Vec<ClickRegion>,
    pub status_line: String,
    pub status_is_error: bool,
    pub last_error: Option<LocalCodeError>,
    pub last_failed_action: Option<RetryAction>,
    /// Diagnosis of the last error (drives the Fix button + inline explanation).
    pub last_diagnosis: Option<Diagnosis>,
    /// Which backend the last error came from — set by callers right before
    /// `raise_error`, consumed there, so the Fix button knows what to repair.
    pub error_backend: Option<BackendKind>,
    /// The resolved repair for the last error, if one applies on this machine.
    last_repair: Option<RepairPlan>,
    /// The inline banner (confirm / warning / error / info), rendered at the top
    /// of the working area. Replaces the old centered modal.
    pub modal: Option<ModalState>,
    pending_tool_confirm: Option<oneshot::Sender<bool>>,
    /// Highlighted index in the command list (shown when the omnibar text
    /// starts with '/').
    pub palette_selected: usize,
    pub assistant_configured: bool,
    /// None while the startup probe is still running.
    pub api_healthy: Option<bool>,
    pub gpu: GpuInventory,
    pub runtimes: Vec<ActiveRuntime>,
    pub runtime_selected: usize,
    pub runtime_list_state: ListState,
    pub backend_reports: Vec<DetectReport>,
    pub detecting: bool,
    /// Keyboard selection in the Backends view (index into `BACKEND_ORDER`).
    pub backend_sel: usize,
    pub doctor_summary: Option<String>,
    pub setup_scroll: u16,
    pub install_busy: Option<Busy>,
    pub install_progress_line: String,
    pub installing_kind: Option<BackendKind>,
    /// A diagnosed repair in progress (mirrors `install_busy`).
    pub repair_busy: Option<Busy>,
    pub repair_progress_line: String,
    /// Collecting a sudo password for a repair that needs elevation.
    pending_sudo: Option<PendingSudo>,
    /// Latest per-backend smoke results, shown in the Backends view.
    pub smoke_reports: Vec<SmokeReport>,
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
    /// Skills discovered from the skills dir, for the Settings view.
    pub skills: Vec<Skill>,
    /// `name → target` for each configured MCP server, for the Settings view.
    pub mcp_servers: Vec<String>,
    // Settings view: selection over actionable rows, scroll, and inline editor.
    pub settings_sel: usize,
    pub settings_scroll: u16,
    pub settings_view_height: u16,
    settings_editing: Option<SettingField>,
    settings_field_edit: String,
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
        let skills = agent_probe.skills.list().to_vec();
        let mcp_servers = agent_probe.mcp.server_summaries();

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
            mouse_capture: config.ui.mouse,
            paths,
            mode: Mode::Chat,
            click_regions: vec![],
            // Empty at rest: the omnibar placeholder already invites input, and
            // the status bar now appends this after the model/ctx cluster.
            status_line: String::new(),
            status_is_error: false,
            last_error: None,
            last_failed_action: None,
            last_diagnosis: None,
            error_backend: None,
            last_repair: None,
            modal: None,
            pending_tool_confirm: None,
            palette_selected: 0,
            assistant_configured,
            api_healthy: None,
            gpu,
            runtimes: vec![],
            runtime_selected: 0,
            runtime_list_state: ListState::default(),
            backend_reports: vec![],
            detecting: false,
            backend_sel: 0,
            doctor_summary: None,
            setup_scroll: 0,
            install_busy: None,
            install_progress_line: String::new(),
            installing_kind: None,
            repair_busy: None,
            repair_progress_line: String::new(),
            pending_sudo: None,
            smoke_reports: vec![],
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
            skills,
            mcp_servers,
            settings_sel: 0,
            settings_scroll: 0,
            settings_view_height: 0,
            settings_editing: None,
            settings_field_edit: String::new(),
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
        } else if let Some(b) = self.repair_busy.take() {
            // kill_on_drop stops the child when the aborted task's Command drops.
            b.handle.abort();
            self.repair_progress_line.clear();
            self.set_status("Fix cancelled", false);
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
                    text: format!("{name}  {args_preview}"),
                    live: true,
                });
                self.coding_follow = true;
            }
            AgentEvent::ToolFinished { name, ok, summary } => {
                let text = if ok {
                    format!("{name}  {summary}")
                } else {
                    format!("{name}  failed — {summary}")
                };
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
    // Error surface: status line + an inline error banner.
    // ------------------------------------------------------------------

    fn raise_error(&mut self, mut err: LocalCodeError) {
        self.status_line = format!("{}: {}", err.code, err.message);
        self.status_is_error = true;
        // Diagnose using the backend the error came from (a caller sets
        // `error_backend` right before raising; we consume it so it can't leak
        // to an unrelated error).
        let backend = self.error_backend.take();
        let diagnosis = backend.and_then(|b| diagnose(b, &err));
        // Resolve the repair now so the Fix button appears only when the fix is
        // actually applicable on this machine (right OS, Python present, …).
        self.last_repair = match (&diagnosis, backend) {
            (Some(d), Some(b)) => d
                .repair
                .as_ref()
                .and_then(|intent| resolve_repair(intent, b, &self.paths).ok()),
            _ => None,
        };
        // Attach the structured diagnosis so "Ask assistant" sees it too.
        if let Some(d) = &diagnosis {
            if let Ok(v) = serde_json::to_value(d) {
                err = err.with_details(serde_json::json!({ "diagnosis": v }));
            }
        }
        let has_repair = self.last_repair.is_some();
        self.last_diagnosis = diagnosis;
        self.last_error = Some(err.clone());
        self.modal = Some(ModalState::error(err, has_repair));
    }

    fn set_status(&mut self, msg: impl Into<String>, is_error: bool) {
        self.status_line = msg.into();
        self.status_is_error = is_error;
    }

    fn open_error_modal(&mut self) {
        if let Some(err) = self.last_error.clone() {
            self.modal = Some(ModalState::error(err, self.last_repair.is_some()));
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
            Some(RetryAction::ApplyRepair) => self.start_repair(),
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
    // Backend install
    // ------------------------------------------------------------------

    /// Show the install plan: automated plans go through a confirm dialog that
    /// prints the exact commands; unautomatable ones show honest manual steps.
    fn start_install(&mut self, kind: BackendKind) {
        if self.install_busy.is_some() {
            self.set_status("An install is already running (Esc to cancel)", false);
            return;
        }
        let plan = resolve_install_plan(kind, &self.paths);
        match &plan {
            InstallPlan::Automated { display, .. } => {
                let sudo_note = if plan.requires_sudo() {
                    "Includes a command that needs sudo — you'll be asked to approve your password.\n\n"
                } else {
                    ""
                };
                let body = format!(
                    "LocalCode will run:\n\n{display}\n\n{}{sudo_note}Esc cancels — nothing runs until you confirm.",
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
                for s in steps {
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

    /// After the install is confirmed: run it straight away, unless an elevated
    /// step needs a password we can't get non-interactively — then collect it
    /// with the masked prompt first. Mirrors [`Self::confirm_repair`].
    fn confirm_install(&mut self, kind: BackendKind) {
        let plan = resolve_install_plan(kind, &self.paths);
        if plan.requires_sudo() && !can_elevate_noninteractively() {
            self.set_status(
                "Enter your sudo password to install — Enter to confirm, Esc to cancel",
                false,
            );
            self.pending_sudo = Some(PendingSudo {
                job: SudoJob::Install { kind, plan },
                buf: Zeroizing::new(String::new()),
            });
        } else {
            self.spawn_install(kind, plan, None);
        }
    }

    /// Spawn the resolved install plan. Streams output lines through
    /// `InstallProgress`; Esc aborts (kill_on_drop stops the child). The sudo
    /// password (when present) is moved into the task, written to `sudo -S`, and
    /// zeroized when the task ends — it never touches the `BgMsg` channel.
    fn spawn_install(
        &mut self,
        kind: BackendKind,
        plan: InstallPlan,
        password: Option<Zeroizing<String>>,
    ) {
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
            let pw = password.as_ref().map(|z| z.as_str());
            let result = run_install(&plan, pw, ptx).await;
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

    // ------------------------------------------------------------------
    // Backend repair (diagnosis → fix)
    // ------------------------------------------------------------------

    /// Show the resolved repair for approval: the exact commands (incl. any
    /// `sudo`) are previewed before anything runs — mirrors `start_install`.
    fn start_repair(&mut self) {
        if self.repair_busy.is_some() {
            self.set_status("A fix is already running (Esc to cancel)", false);
            return;
        }
        let Some(plan) = self.last_repair.clone() else {
            self.set_status("No automatic fix is available for this error", false);
            return;
        };
        let caveat = plan
            .caveat
            .clone()
            .map(|c| format!("{c}\n\n"))
            .unwrap_or_default();
        let sudo_note = if plan.requires_sudo() {
            "Includes a command that needs sudo — you'll be asked to approve your password.\n\n"
        } else {
            ""
        };
        let body = format!(
            "LocalCode will run:\n\n{}\n\n{caveat}{sudo_note}Esc cancels — nothing runs until you confirm.",
            plan.display(),
        );
        self.modal = Some(ModalState::confirm(
            plan.title.clone(),
            body,
            ConfirmAction::ApplyRepair,
        ));
    }

    /// After the repair is confirmed: run it straight away, unless a sudo step
    /// needs a password we can't get non-interactively — then prompt for it.
    fn confirm_repair(&mut self) {
        let Some(plan) = self.last_repair.clone() else {
            return;
        };
        if plan.requires_sudo() && !can_elevate_noninteractively() {
            self.set_status(
                "Enter your sudo password to run the fix — Enter to confirm, Esc to cancel",
                false,
            );
            self.pending_sudo = Some(PendingSudo {
                job: SudoJob::Repair(plan),
                buf: Zeroizing::new(String::new()),
            });
        } else {
            self.spawn_repair(plan, None);
        }
    }

    /// Spawn the repair. Streams output through `RepairProgress`; the password
    /// (when present) is moved into the task, written to `sudo -S`, and zeroized
    /// when the task ends — it never touches the `BgMsg` channel.
    fn spawn_repair(&mut self, plan: RepairPlan, password: Option<Zeroizing<String>>) {
        if self.repair_busy.is_some() {
            self.set_status("A fix is already running (Esc to cancel)", false);
            return;
        }
        let (ptx, mut prx) = mpsc::unbounded_channel::<String>();
        let fwd = self.bg_tx.clone();
        tokio::spawn(async move {
            while let Some(line) = prx.recv().await {
                if fwd.send(BgMsg::RepairProgress(line)).is_err() {
                    break;
                }
            }
        });
        let tx = self.bg_tx.clone();
        let repoint = plan.repoint.clone();
        let handle = tokio::spawn(async move {
            let pw = password.as_ref().map(|z| z.as_str());
            let result = run_repair(&plan, pw, ptx).await;
            let _ = tx.send(BgMsg::RepairDone { result, repoint });
        });
        self.repair_progress_line = "starting…".into();
        self.repair_busy = Some(Busy {
            kind: BusyKind::Install,
            label: "Applying fix".into(),
            started: Instant::now(),
            handle,
        });
    }

    /// Point a backend's configured binary at the repaired install and persist,
    /// then rebuild the registry so the new path is used immediately.
    fn apply_repoint(&mut self, rp: Repoint) {
        match rp.kind {
            BackendKind::Vllm => self.config.backends.vllm.bin = rp.bin,
            BackendKind::Sglang => self.config.backends.sglang.bin = rp.bin,
            BackendKind::LlamaCpp => self.config.backends.llamacpp.bin = rp.bin,
            // Ollama repairs restart a service; there's no binary to repoint.
            BackendKind::Ollama => {}
        }
        self.registry = Arc::new(BackendRegistry::from_config(&self.config));
        if let Err(e) = self.config.save(&self.paths) {
            self.set_status(
                format!("Fix applied, but saving config failed: {}", e.message),
                true,
            );
        }
    }

    /// Key handling while the masked sudo prompt is open. Modeled on the remote
    /// field editor; Enter runs the fix, Esc cancels (buffer zeroized on drop).
    fn handle_sudo_key(&mut self, key: crossterm::event::KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                if let Some(ps) = self.pending_sudo.take() {
                    let PendingSudo { job, buf } = ps;
                    match job {
                        SudoJob::Install { kind, plan } => {
                            self.spawn_install(kind, plan, Some(buf))
                        }
                        SudoJob::Repair(plan) => self.spawn_repair(plan, Some(buf)),
                    }
                }
            }
            KeyCode::Esc => {
                self.pending_sudo = None;
                self.set_status("Cancelled", false);
            }
            KeyCode::Backspace => {
                if let Some(ps) = &mut self.pending_sudo {
                    ps.buf.pop();
                }
            }
            KeyCode::Char(c) => {
                if let Some(ps) = &mut self.pending_sudo {
                    ps.buf.push(c);
                }
            }
            _ => {}
        }
    }

    /// Render state for the masked sudo prompt: `(chars entered, command being
    /// authorized)`. `None` when no prompt is open. The password itself is never
    /// exposed — only its length, for the `•` mask.
    pub fn sudo_prompt(&self) -> Option<(usize, String)> {
        self.pending_sudo
            .as_ref()
            .map(|ps| (ps.buf.chars().count(), ps.job.sudo_display()))
    }

    /// Smoke-test one backend from the Backends view, storing the result.
    fn start_smoke(&mut self, kind: BackendKind) {
        let Some(report) = self.backend_reports.iter().find(|r| r.kind == kind).cloned() else {
            self.set_status("Detecting first — try again in a moment", false);
            self.start_detect();
            return;
        };
        let cfg = self.config.clone();
        let tx = self.bg_tx.clone();
        self.set_status(format!("Smoke-testing {}…", kind.as_str()), false);
        tokio::spawn(async move {
            let sr = smoke_test(&report, &cfg).await;
            let _ = tx.send(BgMsg::SmokeDone(sr));
        });
    }

    /// Start the fix for a backend from its stored smoke diagnosis (Backends
    /// view). Falls back to running a smoke test if we don't have one yet.
    fn fix_backend(&mut self, kind: BackendKind) {
        let Some(intent) = self
            .smoke_reports
            .iter()
            .find(|r| r.kind == kind)
            .and_then(|r| r.diagnosis.as_ref())
            .and_then(|d| d.repair.clone())
        else {
            self.set_status(format!("Smoke-test {} first to diagnose it", kind.as_str()), false);
            self.start_smoke(kind);
            return;
        };
        match resolve_repair(&intent, kind, &self.paths) {
            Ok(plan) => {
                self.last_repair = Some(plan);
                self.start_repair();
            }
            Err(e) => {
                self.error_backend = Some(kind);
                self.raise_error(e);
            }
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
        self.set_mode(Mode::Remote);
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
                            self.modal = Some(ModalState::info("Assistant", message));
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
                        self.set_status(
                            format!(
                                "Update available: v{} → v{} — /update to install",
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
                        Ok(repoint) => {
                            // A fetched llama.cpp binary lives in a managed dir
                            // (not on PATH): repoint config at it and rebuild the
                            // registry so detect finds it.
                            if let Some(rp) = repoint {
                                self.apply_repoint(rp);
                            }
                            self.set_status(
                                format!("{} installed — re-detecting…", kind.as_str()),
                                false,
                            );
                            // Freshly-installed binary is picked up by detect()
                            // (which::which is live); no registry rebuild needed.
                            self.start_detect();
                        }
                        Err(e) => {
                            self.error_backend = Some(kind);
                            self.last_failed_action = Some(RetryAction::InstallBackend(kind));
                            self.raise_error(e);
                        }
                    }
                }
                BgMsg::RepairProgress(line) => {
                    self.repair_progress_line = line.clone();
                    self.set_status(format!("Fixing: {line}"), false);
                }
                BgMsg::RepairDone { result, repoint } => {
                    self.repair_busy = None;
                    self.repair_progress_line.clear();
                    match result {
                        Ok(()) => {
                            if let Some(rp) = repoint {
                                self.apply_repoint(rp);
                            }
                            // Clear the resolved fix and re-check health.
                            self.last_repair = None;
                            self.set_status("Fix applied — re-checking…", false);
                            self.start_detect();
                        }
                        Err(e) => {
                            self.last_failed_action = Some(RetryAction::ApplyRepair);
                            self.raise_error(e);
                        }
                    }
                }
                BgMsg::SmokeDone(report) => {
                    let kind = report.kind;
                    let ok = report.ok;
                    let summary = report.diagnosis.as_ref().map(|d| d.summary.clone());
                    self.smoke_reports.retain(|r| r.kind != kind);
                    self.smoke_reports.push(report);
                    if ok {
                        self.set_status(format!("{} smoke test passed", kind.as_str()), false);
                    } else {
                        self.set_status(
                            summary
                                .map(|s| format!("{}: {s}", kind.as_str()))
                                .unwrap_or_else(|| format!("{} smoke test failed", kind.as_str())),
                            true,
                        );
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
                    // The Fix button needs to know which backend failed.
                    self.error_backend = Some(self.deploy_backend);
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
            ("/models", "[q]", "search & deploy HuggingFace models"),
            ("/runtimes", "", "active runtimes & system overview"),
            ("/remote", "", "connect a GPU server over SSH"),
            ("/backends", "", "install & configure inference backends"),
            ("/bench", "", "run the sample benchmark suite"),
            ("/setup", "", "first-run setup & doctor"),
            ("/settings", "", "preferences & config file"),
            ("/theme", "", "cycle dark / neon / pink"),
            ("/select", "", "release mouse to select & copy text (F2)"),
            ("/chat", "", "back to the conversation"),
            ("/new", "", "start a new conversation"),
            ("/deploy", "", "deploy the selected model"),
            ("/doctor", "", "run environment diagnostics"),
            ("/assistant", "", "ask the in-app assistant"),
            ("/update", "", "install the available update"),
            ("/logs", "", "show the log directory path"),
            ("/help", "", "keyboard & mouse help"),
            ("/quit", "", "exit"),
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
                self.set_mode(Mode::Models);
                if !rest.is_empty() {
                    self.model_query = rest;
                    self.start_search();
                } else if self.models.is_empty() {
                    self.start_search();
                }
            }
            "popular" => {
                self.set_mode(Mode::Models);
                self.model_query = "code".into();
                self.start_search();
            }
            "trending" => {
                self.set_mode(Mode::Models);
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
            "backends" => {
                self.set_mode(Mode::Backends);
                if self.backend_reports.is_empty() && !self.detecting {
                    self.start_detect();
                }
            }
            "runtimes" | "dashboard" | "home" => self.set_mode(Mode::Runtimes),
            "deploy" => {
                self.set_mode(Mode::Models);
                self.start_deploy(false);
            }
            "bench" | "benchmark" => self.set_mode(Mode::Bench),
            "setup" => self.set_mode(Mode::Setup),
            "doctor" => {
                self.set_mode(Mode::Setup);
                self.start_doctor();
            }
            "settings" => self.set_mode(Mode::Settings),
            "theme" => self.toggle_theme(),
            "select" => self.toggle_select_mode(),
            "chat" => self.set_mode(Mode::Chat),
            "new" => {
                self.set_mode(Mode::Chat);
                self.new_coding_session();
            }
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
            "logs" => self.open_logs(),
            "help" => self.open_help(),
            "quit" | "exit" | "q" => self.request_quit(),
            "" => {}
            other => self.set_status(format!("Unknown command: /{other} — press / to list"), true),
        }
    }

    /// Set the theme (used by the status-bar switcher and Settings).
    pub(crate) fn set_theme(&mut self, mode: ThemeMode) {
        self.config.ui.theme = mode;
        self.theme = Theme::new(mode);
        self.set_status(format!("Theme: {}", mode.label()), false);
    }

    /// `/theme` cycles through the shipped themes (dark → neon → pink).
    fn toggle_theme(&mut self) {
        self.set_theme(self.config.ui.theme.next());
    }

    /// Toggle "select mode": release the mouse so the terminal's native
    /// drag-to-select/copy works, or re-grab it for click interaction. Bound to
    /// F2 and `/select`. Transient (not saved) — the run loop applies the change
    /// to the real terminal; the persistent default lives in Settings → mouse.
    pub(crate) fn toggle_select_mode(&mut self) {
        self.mouse_capture = !self.mouse_capture;
        if self.mouse_capture {
            self.set_status("Mouse mode — click to interact", false);
        } else {
            self.set_status(
                "Select mode — drag to select & copy · F2 or /select to resume mouse",
                false,
            );
        }
    }

    // ------------------------------------------------------------------
    // Settings (interactive: toggles, inline text edits, skill/tool/MCP config)
    // ------------------------------------------------------------------

    /// The Settings view as an ordered list of rows. Single source of truth for
    /// the renderer and the keyboard/mouse handlers.
    pub fn settings_rows(&self) -> Vec<SettingsRow> {
        let toggle = |label: &str, on: bool, action: SettingAction| SettingsRow {
            label: label.to_string(),
            value: if on { "on".into() } else { "off".into() },
            kind: SettingsRowKind::Toggle(on),
            action: Some(action),
        };
        let text = |label: &str, value: String, field: SettingField| SettingsRow {
            label: label.to_string(),
            value,
            kind: SettingsRowKind::Text,
            action: Some(SettingAction::Edit(field)),
        };
        let mut r = Vec::new();

        // General.
        r.push(SettingsRow::header("general"));
        r.push(SettingsRow {
            label: "theme".into(),
            value: self.config.ui.theme.label().to_string(),
            kind: SettingsRowKind::Action("cycle"),
            action: Some(SettingAction::ThemeToggle),
        });
        r.push(SettingsRow {
            label: "mouse".into(),
            value: if self.config.ui.mouse {
                "on — click to interact".into()
            } else {
                "off — drag to select & copy text".into()
            },
            kind: SettingsRowKind::Toggle(self.config.ui.mouse),
            action: Some(SettingAction::ToggleMouse),
        });
        r.push(toggle("token streaming", self.config.agent.stream, SettingAction::ToggleStream));
        r.push(toggle(
            "confirm shell",
            self.config.agent.confirm_destructive_tools,
            SettingAction::ToggleConfirmShell,
        ));
        r.push(toggle(
            "cloud fallback",
            self.config.agent.allow_cloud_fallback,
            SettingAction::ToggleCloudFallback,
        ));
        r.push(toggle(
            "check updates",
            self.config.updates.check_on_startup,
            SettingAction::ToggleCheckUpdates,
        ));

        // Agent: system prompt, AGENTS.md, workspace.
        r.push(SettingsRow::header("agent — prompt & project context"));
        let sp = self.config.agent.system_prompt.as_deref().unwrap_or("");
        let sp_val = if sp.trim().is_empty() {
            "(built-in default) — click to customize".to_string()
        } else {
            compact_ws(sp)
        };
        r.push(text("system prompt", sp_val, SettingField::SystemPrompt));
        let agents_here = self.workspace_path().join("AGENTS.md").exists();
        r.push(SettingsRow {
            label: "use AGENTS.md".into(),
            value: format!(
                "{} · {}",
                if self.config.agent.use_agents_md { "on" } else { "off" },
                if agents_here { "found in workspace" } else { "none in workspace" },
            ),
            kind: SettingsRowKind::Toggle(self.config.agent.use_agents_md),
            action: Some(SettingAction::ToggleAgentsMd),
        });
        let ws = self
            .config
            .agent
            .workspace_root
            .clone()
            .unwrap_or_else(|| self.workspace_path().display().to_string());
        r.push(text("workspace", ws, SettingField::Workspace));

        // Tools.
        r.push(SettingsRow::header("tools — what the agent may call"));
        for (i, (name, desc)) in ToolRegistry::catalog().iter().enumerate() {
            let enabled = !self.config.agent.disabled_tools.iter().any(|t| t == name);
            r.push(SettingsRow {
                label: (*name).to_string(),
                value: (*desc).to_string(),
                kind: SettingsRowKind::Toggle(enabled),
                action: Some(SettingAction::ToggleTool(i)),
            });
        }

        // Skills.
        r.push(SettingsRow::header("skills"));
        let sd = self
            .config
            .agent
            .skills_dir
            .clone()
            .unwrap_or_else(|| ".localcode/skills".into());
        r.push(text("skills dir", sd, SettingField::SkillsDir));
        r.push(SettingsRow {
            label: "reload skills".into(),
            value: format!("{} discovered", self.skills.len()),
            kind: SettingsRowKind::Action("reload"),
            action: Some(SettingAction::ReloadSkills),
        });
        for (i, sk) in self.skills.iter().enumerate() {
            let enabled = !self.config.agent.disabled_skills.iter().any(|d| d == &sk.name);
            r.push(SettingsRow {
                label: sk.name.clone(),
                value: compact_ws(&sk.description),
                kind: SettingsRowKind::Toggle(enabled),
                action: Some(SettingAction::ToggleSkill(i)),
            });
        }

        // MCP servers.
        r.push(SettingsRow::header("mcp servers"));
        let mp = self
            .config
            .agent
            .mcp_config
            .clone()
            .unwrap_or_else(|| ".localcode/mcp.json".into());
        r.push(text("mcp config", mp, SettingField::McpPath));
        if self.mcp_servers.is_empty() {
            r.push(SettingsRow::info("", "no servers — add them to the mcp.json above"));
        } else {
            for s in &self.mcp_servers {
                r.push(SettingsRow::info("server", s.clone()));
            }
        }

        // Config file / env.
        r.push(SettingsRow::header("config"));
        r.push(SettingsRow::info(
            "config file",
            self.paths.config_file().display().to_string(),
        ));
        r.push(SettingsRow::info(
            "",
            "toggles & edits save immediately · Ctrl+S also saves",
        ));
        r.push(SettingsRow::info(
            "env overrides",
            "LOCALCODE_API_URL · LOCALCODE_HF_ENDPOINT · HF_TOKEN · OPENROUTER_API_KEY",
        ));
        r
    }

    /// `(flat row index, action)` for every actionable Settings row, in order.
    fn settings_actionable(&self) -> Vec<(usize, SettingAction)> {
        self.settings_rows()
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.action.map(|a| (i, a)))
            .collect()
    }

    /// Is `field` the one currently being edited inline?
    pub fn settings_editing_field(&self) -> Option<SettingField> {
        self.settings_editing
    }

    /// The inline edit buffer (rendered with a caret while editing).
    pub fn settings_edit_buffer(&self) -> &str {
        &self.settings_field_edit
    }

    fn settings_move(&mut self, delta: i32) {
        let n = self.settings_actionable().len();
        if n == 0 {
            return;
        }
        let cur = self.settings_sel.min(n - 1) as i32;
        self.settings_sel = (cur + delta).clamp(0, n as i32 - 1) as usize;
        self.settings_ensure_visible();
    }

    /// Scroll so the selected actionable row stays on screen.
    fn settings_ensure_visible(&mut self) {
        let Some((flat, _)) = self.settings_actionable().get(self.settings_sel).copied() else {
            return;
        };
        let vh = self.settings_view_height.max(1) as usize;
        let scroll = self.settings_scroll as usize;
        if flat < scroll {
            self.settings_scroll = flat as u16;
        } else if flat >= scroll + vh {
            self.settings_scroll = (flat + 1 - vh) as u16;
        }
    }

    fn activate_selected_setting(&mut self) {
        if let Some((_, action)) = self.settings_actionable().get(self.settings_sel).copied() {
            self.activate_setting(action);
        }
    }

    /// Run a Settings row's action. Toggles/reloads persist immediately; text
    /// edits open the inline editor and persist on commit.
    fn activate_setting(&mut self, action: SettingAction) {
        // Move the keyboard selection to the activated row for visual feedback.
        if let Some(idx) = self
            .settings_actionable()
            .iter()
            .position(|(_, a)| *a == action)
        {
            self.settings_sel = idx;
        }
        match action {
            SettingAction::ThemeToggle => self.toggle_theme(),
            SettingAction::ToggleMouse => {
                self.config.ui.mouse = !self.config.ui.mouse;
                self.mouse_capture = self.config.ui.mouse;
            }
            SettingAction::ToggleStream => {
                self.config.agent.stream = !self.config.agent.stream;
            }
            SettingAction::ToggleConfirmShell => {
                self.config.agent.confirm_destructive_tools =
                    !self.config.agent.confirm_destructive_tools;
            }
            SettingAction::ToggleCloudFallback => {
                self.config.agent.allow_cloud_fallback = !self.config.agent.allow_cloud_fallback;
            }
            SettingAction::ToggleAgentsMd => {
                self.config.agent.use_agents_md = !self.config.agent.use_agents_md;
            }
            SettingAction::ToggleCheckUpdates => {
                self.config.updates.check_on_startup = !self.config.updates.check_on_startup;
            }
            SettingAction::ToggleTool(i) => self.toggle_tool(i),
            SettingAction::ToggleSkill(i) => self.toggle_skill(i),
            SettingAction::ReloadSkills => {
                self.reload_skills_list();
                self.set_status(format!("Reloaded skills — {} found", self.skills.len()), false);
            }
            SettingAction::Edit(field) => {
                self.begin_settings_edit(field);
                return; // editor persists on commit, not now
            }
        }
        if let Err(e) = self.config.save(&self.paths) {
            self.raise_error(e);
        }
    }

    fn toggle_tool(&mut self, i: usize) {
        let Some((name, _)) = ToolRegistry::catalog().get(i) else {
            return;
        };
        let name = name.to_string();
        if let Some(pos) = self.config.agent.disabled_tools.iter().position(|t| *t == name) {
            self.config.agent.disabled_tools.remove(pos);
            self.set_status(format!("Enabled tool {name}"), false);
        } else {
            self.config.agent.disabled_tools.push(name.clone());
            self.set_status(format!("Disabled tool {name}"), false);
        }
    }

    fn toggle_skill(&mut self, i: usize) {
        let Some(name) = self.skills.get(i).map(|s| s.name.clone()) else {
            return;
        };
        if let Some(pos) = self.config.agent.disabled_skills.iter().position(|s| *s == name) {
            self.config.agent.disabled_skills.remove(pos);
            self.set_status(format!("Enabled skill {name}"), false);
        } else {
            self.config.agent.disabled_skills.push(name.clone());
            self.set_status(format!("Disabled skill {name}"), false);
        }
        self.refresh_agent_summaries();
    }

    fn begin_settings_edit(&mut self, field: SettingField) {
        self.settings_field_edit = match field {
            SettingField::SystemPrompt => {
                self.config.agent.system_prompt.clone().unwrap_or_default()
            }
            SettingField::SkillsDir => self.config.agent.skills_dir.clone().unwrap_or_default(),
            SettingField::McpPath => self.config.agent.mcp_config.clone().unwrap_or_default(),
            SettingField::Workspace => self.config.agent.workspace_root.clone().unwrap_or_default(),
        };
        self.settings_editing = Some(field);
        self.set_status("Editing — type a value, ↵ save, Esc cancel", false);
    }

    /// Key handling while a Settings text field is being edited inline.
    fn handle_settings_field_key(&mut self, key: crossterm::event::KeyEvent) {
        match key.code {
            KeyCode::Enter => self.commit_settings_field(),
            KeyCode::Esc => {
                self.settings_editing = None;
                self.set_status("Edit cancelled", false);
            }
            KeyCode::Backspace => {
                self.settings_field_edit.pop();
            }
            KeyCode::Char(c) => self.settings_field_edit.push(c),
            _ => {}
        }
    }

    fn commit_settings_field(&mut self) {
        let Some(field) = self.settings_editing.take() else {
            return;
        };
        let raw = self.settings_field_edit.trim().to_string();
        let opt = if raw.is_empty() { None } else { Some(raw) };
        match field {
            SettingField::SystemPrompt => self.config.agent.system_prompt = opt,
            SettingField::SkillsDir => {
                self.config.agent.skills_dir = opt;
                self.reload_skills_list();
            }
            SettingField::McpPath => self.config.agent.mcp_config = opt,
            SettingField::Workspace => self.config.agent.workspace_root = opt,
        }
        self.refresh_agent_summaries();
        match self.config.save(&self.paths) {
            Ok(()) => self.set_status("Saved", false),
            Err(e) => self.raise_error(e),
        }
    }

    /// Rebuild the discovered-skills list from the configured skills dir.
    fn reload_skills_list(&mut self) {
        let dir = self
            .config
            .agent
            .skills_dir
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".localcode/skills"));
        self.skills = SkillLoader::new(dir).list().to_vec();
        self.refresh_agent_summaries();
    }

    /// Recompute the cached skill/MCP summaries from current config (after an
    /// edit that changes skills dir, mcp path, or disabled lists).
    fn refresh_agent_summaries(&mut self) {
        let probe = CodingAgent::new(self.config.agent.clone());
        self.skills = probe.skills.list().to_vec();
        self.mcp_servers = probe.mcp.server_summaries();
        let mcp_count = probe.mcp.configured_count();
        self.mcp_status_summary = if mcp_count == 0 {
            "none configured".into()
        } else {
            format!("{mcp_count} configured (not connected)")
        };
        self.skill_count = self
            .skills
            .iter()
            .filter(|s| !self.config.agent.disabled_skills.iter().any(|d| d == &s.name))
            .count();
    }

    // ------------------------------------------------------------------
    // Help / quit
    // ------------------------------------------------------------------

    /// `/logs` — show where logs live plus a readable tail, inline (the
    /// redesign renders info as a banner, not a popup). Structured JSON lines
    /// are compacted to `HH:MM:SS LEVEL message` so the banner is skimmable.
    fn open_logs(&mut self) {
        fn compact(line: &str) -> String {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                return line.to_string();
            };
            let time = v
                .get("timestamp")
                .and_then(|t| t.as_str())
                .and_then(|ts| ts.split('T').nth(1))
                .map(|s| s.split('.').next().unwrap_or(s))
                .unwrap_or("");
            let level = v.get("level").and_then(|l| l.as_str()).unwrap_or("");
            match v
                .get("fields")
                .and_then(|f| f.get("message"))
                .and_then(|m| m.as_str())
            {
                Some(m) => format!("{time} {level:<5} {m}").trim().to_string(),
                None => line.to_string(),
            }
        }

        let dir = self.paths.log_dir.display().to_string();
        let redact = self.config.logging.redact_secrets;
        let tail = localcode_log::read_recent_logs(&self.paths.log_dir, 8, None, redact)
            .ok()
            .map(|s| s.lines().map(compact).collect::<Vec<_>>().join("\n"))
            .filter(|s| !s.trim().is_empty());
        let body = match tail {
            Some(t) => format!("{dir}\n\n{t}"),
            None => format!("{dir}\n\n(no log entries yet)"),
        };
        self.modal = Some(ModalState::info("Logs", body));
    }

    fn open_help(&mut self) {
        let body = "\
The omnibar at the bottom is always active — just type.\n\
  type a message + Enter   chat with the agent (from any mode)\n\
  type /                   the working area becomes the command list\n\
  Enter                    run command / search / submit / mode action\n\
  Esc                      cancel a running task, else return to chat\n\
  ↑/↓                      history in chat, selection in list modes\n\
  Ctrl+S save    Ctrl+K command entry    Ctrl+C quit\n\
\n\
Commands (type / to see them all):\n\
  /models [q]   search & deploy HuggingFace models\n\
  /runtimes     active runtimes & system overview\n\
  /remote       connect a GPU server over SSH (one-click)\n\
  /backends     install inference backends\n\
  /bench /setup /settings /theme /chat /new /deploy /update /logs /quit\n\
\n\
Everything renders inline in the working area — no popups. Rows, fields\n\
and buttons are clickable. Models: click a model to open it, click a quant\n\
to select, click deploy. Remote: click a field to edit, then connect."
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

        // Global: toggle "select mode". Releasing the mouse lets the terminal
        // select/copy text; works in every mode, even over a modal or prompt.
        if key.code == KeyCode::F(2) {
            self.toggle_select_mode();
            return;
        }

        // Global chords.
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') => {
                    self.should_quit = true;
                    return;
                }
                KeyCode::Char('k') => {
                    // Ctrl+K is aliased to the command entry: prefill a '/'.
                    if !self.coding_input.starts_with('/') {
                        self.coding_input.insert(0, '/');
                        self.coding_cursor = self.coding_input.chars().count();
                    }
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
                _ => {}
            }
        }

        // The masked sudo prompt captures all input while it is open (it has no
        // modal — it renders in the omnibar), so check it before everything else.
        if self.pending_sudo.is_some() {
            self.handle_sudo_key(key);
            return;
        }

        // The inline banner captures all input while it is open.
        if let Some(modal) = self.modal.clone() {
            self.handle_modal_key(key, &modal);
            return;
        }

        // A leading '/' turns the working area into the command list; it owns
        // navigation and Enter, but other keys still edit the omnibar text.
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

        // An in-place remote-form field edit captures input.
        if self.mode == Mode::Remote && self.remote_editing {
            self.handle_remote_field_key(key);
            return;
        }

        // An in-place settings field edit captures input.
        if self.mode == Mode::Settings && self.settings_editing.is_some() {
            self.handle_settings_field_key(key);
            return;
        }

        // Esc: cancel a running task; else leave a non-chat mode / clear input.
        if key.code == KeyCode::Esc {
            if self.fg_busy()
                || self.deploy_busy.is_some()
                || self.install_busy.is_some()
                || self.repair_busy.is_some()
                || self.update_busy.is_some()
            {
                self.cancel_current();
            } else if self.mode != Mode::Chat {
                self.set_mode(Mode::Chat);
                self.clear_input();
            } else {
                self.clear_input();
            }
            return;
        }

        // Enter submits; Shift+Enter / Alt+Enter insert a newline so the omnibar
        // is a real multi-line composer.
        if key.code == KeyCode::Enter {
            if key.modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) {
                self.omnibar_insert_newline();
            } else {
                self.omnibar_submit();
            }
            return;
        }

        // In a multi-line composer, Up/Down move the caret between its lines
        // before falling back to history / list navigation.
        if self.coding_input.contains('\n') && matches!(key.code, KeyCode::Up | KeyCode::Down) {
            let delta = if key.code == KeyCode::Up { -1 } else { 1 };
            self.omnibar_cursor_vertical(delta);
            return;
        }

        // Up/Down are history in chat, selection in list modes; other keys edit
        // the omnibar so the prompt/search bar works from anywhere.
        match key.code {
            KeyCode::Up if self.mode == Mode::Chat => self.history_prev(),
            KeyCode::Down if self.mode == Mode::Chat => self.history_next(),
            KeyCode::Up
            | KeyCode::Down
            | KeyCode::PageUp
            | KeyCode::PageDown
            | KeyCode::Tab
            | KeyCode::BackTab => self.mode_nav(key.code),
            _ => self.omnibar_edit(key),
        }
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
            KeyCode::Home => self.coding_cursor = self.line_start_index(),
            KeyCode::End => self.coding_cursor = self.line_end_index(),
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

    /// Insert a newline at the caret (Shift/Alt+Enter) — makes the omnibar a
    /// multi-line composer.
    fn omnibar_insert_newline(&mut self) {
        let idx = char_to_byte(&self.coding_input, self.coding_cursor);
        self.coding_input.insert(idx, '\n');
        self.coding_cursor += 1;
        self.coding_hist_idx = None;
    }

    /// `(line, column)` of the caret within the multi-line composer.
    pub fn omnibar_cursor_line_col(&self) -> (usize, usize) {
        let cur = self.coding_cursor.min(self.coding_input.chars().count());
        let mut line = 0usize;
        let mut col = 0usize;
        for c in self.coding_input.chars().take(cur) {
            if c == '\n' {
                line += 1;
                col = 0;
            } else {
                col += 1;
            }
        }
        (line, col)
    }

    /// Char index of the given `(line, column)` within the composer, clamping
    /// the column to that line's length.
    fn line_col_to_index(&self, line: usize, col: usize) -> usize {
        let lines: Vec<&str> = self.coding_input.split('\n').collect();
        let line = line.min(lines.len().saturating_sub(1));
        let mut idx = 0usize;
        for l in &lines[..line] {
            idx += l.chars().count() + 1; // +1 for the '\n'
        }
        idx + col.min(lines[line].chars().count())
    }

    fn line_start_index(&self) -> usize {
        let (line, _) = self.omnibar_cursor_line_col();
        self.line_col_to_index(line, 0)
    }

    fn line_end_index(&self) -> usize {
        let (line, _) = self.omnibar_cursor_line_col();
        self.line_col_to_index(line, usize::MAX)
    }

    /// Move the caret one composer line up (`-1`) or down (`+1`), keeping the
    /// column where possible.
    fn omnibar_cursor_vertical(&mut self, delta: i32) {
        let (line, col) = self.omnibar_cursor_line_col();
        let n = self.coding_input.split('\n').count();
        let target = (line as i32 + delta).clamp(0, n.saturating_sub(1) as i32) as usize;
        self.coding_cursor = self.line_col_to_index(target, col);
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

    fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        self.remote_editing = false;
        self.settings_editing = None;
    }

    /// Selection / scroll keys for the active mode (arrows, page keys, Tab).
    fn mode_nav(&mut self, code: KeyCode) {
        match self.mode {
            Mode::Chat => match code {
                KeyCode::PageDown => self.scroll_transcript(self.coding_view_height.max(1) as i64),
                KeyCode::PageUp => self.scroll_transcript(-(self.coding_view_height.max(1) as i64)),
                _ => {}
            },
            Mode::Models => match code {
                KeyCode::Down => {
                    if !self.models.is_empty() {
                        self.model_selected = (self.model_selected + 1).min(self.models.len() - 1);
                    }
                }
                KeyCode::Up => self.model_selected = self.model_selected.saturating_sub(1),
                KeyCode::PageDown => self.scroll_card(self.card_view_height.max(1) as i64),
                KeyCode::PageUp => self.scroll_card(-(self.card_view_height.max(1) as i64)),
                _ => {}
            },
            Mode::Runtimes => match code {
                KeyCode::Down => {
                    let n = self.all_runtimes().len();
                    if n > 0 {
                        self.runtime_selected = (self.runtime_selected + 1).min(n - 1);
                    }
                }
                KeyCode::Up => self.runtime_selected = self.runtime_selected.saturating_sub(1),
                _ => {}
            },
            Mode::Backends => match code {
                KeyCode::Down => self.backend_sel = (self.backend_sel + 1) % BACKEND_ORDER.len(),
                KeyCode::Up => {
                    self.backend_sel =
                        (self.backend_sel + BACKEND_ORDER.len() - 1) % BACKEND_ORDER.len();
                }
                _ => {}
            },
            Mode::Setup => match code {
                KeyCode::PageUp => self.setup_scroll = self.setup_scroll.saturating_sub(6),
                KeyCode::PageDown => self.setup_scroll = self.setup_scroll.saturating_add(6),
                _ => {}
            },
            Mode::Remote => match code {
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
            Mode::Settings => match code {
                KeyCode::Down | KeyCode::Tab => self.settings_move(1),
                KeyCode::Up | KeyCode::BackTab => self.settings_move(-1),
                KeyCode::PageDown => {
                    self.settings_scroll = self.settings_scroll.saturating_add(6);
                }
                KeyCode::PageUp => {
                    self.settings_scroll = self.settings_scroll.saturating_sub(6);
                }
                _ => {}
            },
            Mode::Bench => {}
        }
    }

    /// Enter in the omnibar: run a model search, submit a chat prompt (from any
    /// mode), or trigger the current mode's primary action when input is empty.
    fn omnibar_submit(&mut self) {
        let input = self.coding_input.trim().to_string();
        if self.mode == Mode::Models && !input.is_empty() {
            self.model_query = input;
            self.clear_input();
            self.start_search();
            return;
        }
        if !input.is_empty() {
            // Typing chats with the agent from anywhere.
            if self.mode != Mode::Chat {
                self.set_mode(Mode::Chat);
            }
            self.start_coding_turn();
            return;
        }
        match self.mode {
            Mode::Models => self.start_load_detail(),
            Mode::Remote => self.connect_remote(self.remote_selected),
            Mode::Bench => self.start_bench(),
            Mode::Setup => self.start_doctor(),
            Mode::Backends => {
                let kind = BACKEND_ORDER[self.backend_sel.min(BACKEND_ORDER.len() - 1)];
                self.start_install(kind);
            }
            Mode::Settings => self.activate_selected_setting(),
            _ => {}
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
            (ModalKind::Error { .. }, "Fix") => self.start_repair(),
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
            ) => self.confirm_install(*kind),
            (
                ModalKind::Confirm {
                    action: ConfirmAction::ApplyRepair,
                    ..
                },
                "Confirm",
            ) => self.confirm_repair(),
            (
                ModalKind::Confirm {
                    action: ConfirmAction::ToolApproval,
                    ..
                },
                label,
            ) => self.respond_tool_confirm(label == "Confirm"),
            _ => {}
        }
    }

    fn handle_mouse(&mut self, mouse: crossterm::event::MouseEvent) {
        use event::MouseButton::Left;
        let (col, row) = (mouse.column, mouse.row);
        match mouse.kind {
            MouseEventKind::Down(Left) => self.handle_left_click(col, row),
            MouseEventKind::ScrollUp => self.wheel_scroll_at(-3, col, row),
            MouseEventKind::ScrollDown => self.wheel_scroll_at(3, col, row),
            _ => {}
        }
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

    /// Dispatch a left click to whatever region it landed on. While the banner
    /// or command list is open, only that overlay's controls are actionable.
    fn handle_left_click(&mut self, col: u16, row: u16) {
        let Some(region) = self.region_at(col, row) else {
            return;
        };

        // Inline banner buttons win while the banner is open.
        if self.modal.is_some() {
            if let ClickTarget::ModalButton(i) = region.target {
                if let Some(m) = &mut self.modal {
                    m.selected = i;
                }
                self.activate_modal_button();
            }
            return;
        }
        // Command list.
        if self.slash_active() {
            if let ClickTarget::CommandItem(i) = region.target {
                self.palette_selected = i;
                self.run_selected_slash();
            }
            return;
        }

        // Row index within a list = first visible row + click offset.
        let rel_row = row.saturating_sub(region.rect.y) as usize;
        match region.target {
            // Status bar.
            ClickTarget::Theme(m) => self.set_theme(m),
            ClickTarget::UpdateBadge => {
                if self.update_available.is_some() {
                    self.open_update_modal();
                }
            }
            ClickTarget::Transcript => {}
            // Models.
            ClickTarget::ModelList => {
                if !self.models.is_empty() {
                    // Each model row is two screen lines.
                    let idx = self.model_list_state.offset() + rel_row / 2;
                    if idx < self.models.len() {
                        self.model_selected = idx;
                        self.start_load_detail();
                    }
                }
            }
            ClickTarget::QuantList => {
                if let Some(d) = &self.model_detail {
                    if rel_row < d.quants.len() {
                        self.selected_quant = Some(d.quants[rel_row].label.clone());
                        self.refresh_fit();
                    }
                }
            }
            ClickTarget::BackendCycle => self.cycle_deploy_backend(),
            ClickTarget::DeployButton => self.start_deploy(false),
            // Runtimes.
            ClickTarget::RuntimeList => {
                let n = self.all_runtimes().len();
                if n > 0 {
                    let idx = self.runtime_list_state.offset() + rel_row;
                    if idx < n {
                        self.runtime_selected = idx;
                    }
                }
            }
            // Backends.
            ClickTarget::BackendInstall(i) => {
                self.backend_sel = i.min(BACKEND_ORDER.len() - 1);
                self.start_install(BACKEND_ORDER[self.backend_sel]);
            }
            ClickTarget::BackendSmoke(i) => {
                self.backend_sel = i.min(BACKEND_ORDER.len() - 1);
                self.start_smoke(BACKEND_ORDER[self.backend_sel]);
            }
            ClickTarget::BackendFix(i) => {
                self.backend_sel = i.min(BACKEND_ORDER.len() - 1);
                self.fix_backend(BACKEND_ORDER[self.backend_sel]);
            }
            ClickTarget::BackendRedetect => {
                self.start_detect();
                self.set_status("Re-detecting backends…", false);
            }
            // Remote.
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
            // Setup.
            ClickTarget::SetupStep(i) => self.setup_step_action(i),
            ClickTarget::SetupDoctor => self.start_doctor(),
            // Bench.
            ClickTarget::BenchRun => self.start_bench(),
            // Settings — toggle / edit / theme / reload.
            ClickTarget::Setting(action) => self.activate_setting(action),
            // Handled above.
            ClickTarget::ModalButton(_) | ClickTarget::CommandItem(_) => {}
        }
    }

    /// The action word on a Setup checklist step (recheck / manage / open / …).
    fn setup_step_action(&mut self, i: usize) {
        match i {
            0 => {
                self.gpu = discover().unwrap_or_else(|_| GpuInventory {
                    devices: vec![],
                    detection_method: "none".into(),
                    warnings: vec![],
                });
                self.set_status("GPU re-detected", false);
            }
            1 => {
                self.set_mode(Mode::Backends);
                if self.backend_reports.is_empty() && !self.detecting {
                    self.start_detect();
                }
            }
            2 => self.set_mode(Mode::Models),
            3 => self.open_remote_panel(),
            _ => self.set_mode(Mode::Settings),
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
    /// to the active mode's main scrollable.
    fn wheel_scroll_at(&mut self, delta: i64, col: u16, row: u16) {
        match self.region_at(col, row).map(|r| r.target) {
            Some(ClickTarget::ModelList) => self.scroll_list_models(delta),
            Some(ClickTarget::QuantList) => self.scroll_card(delta),
            Some(ClickTarget::Transcript) => self.scroll_transcript(delta),
            Some(ClickTarget::RuntimeList) => self.scroll_runtimes(delta),
            _ => self.wheel_scroll(delta),
        }
    }

    /// Fallback wheel routing by active mode (cursor not over a known region).
    fn wheel_scroll(&mut self, delta: i64) {
        match self.mode {
            Mode::Chat => self.scroll_transcript(delta),
            Mode::Models => self.scroll_card(delta),
            Mode::Setup => {
                self.setup_scroll = (i64::from(self.setup_scroll) + delta).max(0) as u16;
            }
            Mode::Settings => {
                self.settings_scroll = (i64::from(self.settings_scroll) + delta).max(0) as u16;
            }
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

/// Collapse all runs of whitespace (incl. newlines) to single spaces, so a
/// multi-line value renders on one Settings row.
fn compact_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
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
    // Always release the mouse — select mode may have toggled capture at
    // runtime, so the startup flag no longer reflects the live state.
    execute!(terminal.backend_mut(), DisableMouseCapture).ok();
    terminal.show_cursor().ok();
    result
}

async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<(), LocalCodeError> {
    let mut mouse_on = app.mouse_capture;
    loop {
        app.process_events().await;
        app.process_bg();

        // Apply a pending mouse-capture change. "Select mode" (F2 / `/select`)
        // releases the mouse so the terminal can drag-select/copy text; the
        // Settings toggle sets the persistent default.
        if app.mouse_capture != mouse_on {
            let _ = if app.mouse_capture {
                execute!(terminal.backend_mut(), EnableMouseCapture)
            } else {
                execute!(terminal.backend_mut(), DisableMouseCapture)
            };
            mouse_on = app.mouse_capture;
        }

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
    fn backends_command_switches_mode_and_install_shows_banner() {
        let mut app = test_app();
        app.detecting = true; // skip the detect spawn (no tokio runtime here)
        typ(&mut app, "/backends");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.mode, Mode::Backends);
        // Install shows an inline banner (confirm or manual), never spawns here.
        app.start_install(BackendKind::Ollama);
        assert!(app.modal.is_some());
        assert!(app.install_busy.is_none());
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
    fn slash_command_switches_mode_then_esc_returns_to_chat() {
        let mut app = test_app();
        typ(&mut app, "/settings");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.mode, Mode::Settings);
        assert!(app.coding_input.is_empty(), "omnibar cleared after running command");
        // Esc returns to chat.
        app.handle_key(key(KeyCode::Esc));
        assert_eq!(app.mode, Mode::Chat);
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
    fn theme_cycle_visits_neon_variants() {
        let mut app = test_app();
        app.set_theme(ThemeMode::Dark);
        app.toggle_theme();
        assert_eq!(app.config.ui.theme, ThemeMode::Neon);
        app.toggle_theme();
        assert_eq!(app.config.ui.theme, ThemeMode::NeonPink);
        app.toggle_theme();
        assert_eq!(app.config.ui.theme, ThemeMode::Dark);
    }

    #[test]
    fn selected_row_paints_a_highlight_bar() {
        use localcode_core::theme::ThemeToken;
        use ratatui::style::Color;
        use ratatui::{backend::TestBackend, Terminal};
        let mut app = test_app();
        app.set_theme(ThemeMode::Neon);
        app.mode = Mode::Backends;
        app.backend_sel = 1;
        let mut terminal = Terminal::new(TestBackend::new(84, 20)).unwrap();
        terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
        let buf = terminal.backend().buffer().clone();
        let sel_bg = {
            let (r, g, b) = app.theme.token_rgb(ThemeToken::SelBg);
            Color::Rgb(r, g, b)
        };
        let wide = |y: u16| (0..84).filter(|&x| buf[(x, y)].bg == sel_bg).count() > 20;
        // Exactly one row (the selected backend) shows a wide selection bar —
        // not zero (the old bold-only selection) and not the whole screen.
        let highlighted = (0..20).filter(|&y| wide(y)).count();
        assert_eq!(highlighted, 1, "exactly the selected row is highlighted");
    }

    #[test]
    fn f2_toggles_select_mode_releasing_and_regrabbing_the_mouse() {
        let mut app = test_app();
        assert!(app.mouse_capture, "the mouse is captured by default");
        app.handle_key(key(KeyCode::F(2)));
        assert!(!app.mouse_capture, "F2 releases the mouse so text is selectable");
        app.handle_key(key(KeyCode::F(2)));
        assert!(app.mouse_capture, "F2 again re-grabs the mouse");
    }

    #[test]
    fn settings_mouse_toggle_persists_and_syncs_live_state() {
        let mut app = test_app();
        let before = app.config.ui.mouse;
        app.activate_setting(SettingAction::ToggleMouse);
        assert_eq!(app.config.ui.mouse, !before, "the persistent preference flips");
        assert_eq!(app.mouse_capture, app.config.ui.mouse, "live capture tracks the setting");
    }

    #[test]
    fn plain_text_is_not_a_command() {
        let mut app = test_app();
        typ(&mut app, "hello world");
        assert!(!app.slash_active());
        assert_eq!(app.coding_input, "hello world");
    }

    #[test]
    fn remote_quick_add_appends_server_and_opens_remote() {
        let mut app = test_app();
        typ(&mut app, "/remote add gpu 10.0.0.9 root secret");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.mode, Mode::Remote);
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

    /// Render every mode once to a headless backend — catches layout panics
    /// (out-of-bounds rects, bad splits) across the redesign.
    #[test]
    fn all_surfaces_render_without_panic() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = test_app();
        app.detecting = false;
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();

        for mode in [
            Mode::Chat,
            Mode::Models,
            Mode::Runtimes,
            Mode::Remote,
            Mode::Backends,
            Mode::Bench,
            Mode::Setup,
            Mode::Settings,
        ] {
            app.mode = mode;
            terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
        }
        app.mode = Mode::Chat;

        // Command list open.
        typ(&mut app, "/mod");
        terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
        app.clear_input();

        // A remote server present + an inline banner over the working area.
        app.new_remote_server();
        app.mode = Mode::Remote;
        app.modal = Some(crate::widgets::ModalState::info("Note", "body"));
        terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();

        // A tiny terminal must not panic either.
        let mut tiny = Terminal::new(TestBackend::new(30, 8)).unwrap();
        tiny.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
    }

    /// Full frame flattened to text, for asserting what actually reaches screen.
    fn render_to_string(app: &mut App, w: u16, h: u16) -> String {
        use ratatui::{backend::TestBackend, Terminal};
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        terminal.draw(|f| crate::ui::draw(f, app)).unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut s = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                s.push_str(buf[(x, y)].symbol());
            }
            s.push('\n');
        }
        s
    }

    #[test]
    fn status_line_message_is_rendered() {
        // Regression: the "Complete redesign" orphaned `status_line` — set_status
        // wrote text that nothing drew, so `/logs`, errors and deploy progress
        // showed nothing. The status bar must render it.
        let mut app = test_app();
        app.set_status("hello-status-xyz", false);
        let screen = render_to_string(&mut app, 120, 40);
        assert!(
            screen.contains("hello-status-xyz"),
            "status_line must be drawn in the status bar"
        );
    }

    #[test]
    fn logs_command_opens_inline_banner_with_path() {
        let mut app = test_app();
        typ(&mut app, "/logs");
        app.handle_key(key(KeyCode::Enter));
        let modal = app
            .modal
            .as_ref()
            .expect("/logs must open an inline banner");
        match &modal.kind {
            crate::widgets::ModalKind::Info { title, body } => {
                assert_eq!(title, "Logs");
                assert!(
                    body.contains(&app.paths.log_dir.display().to_string()),
                    "banner shows the log directory path"
                );
            }
            _ => panic!("expected an Info banner"),
        }
        assert!(render_to_string(&mut app, 120, 40).contains("Logs"));
    }

    #[test]
    fn raise_error_diagnoses_vllm_registration_bug() {
        let mut app = test_app();
        // A caller sets the backend context right before raising.
        app.error_backend = Some(BackendKind::Vllm);
        let err = LocalCodeError::new(ErrorCode::BackendStartFailed, "vLLM exited: exit status: 1")
            .with_cause(
                "vLLM output:\nRuntimeError: Tried to register an operator \
                 (vllm::_fused_mul_mat_gguf) with the same name ... registered multiple times.",
            );
        app.raise_error(err);
        assert!(app.last_diagnosis.is_some(), "the install bug must be diagnosed");
        assert!(app.error_backend.is_none(), "backend context is consumed once");
    }

    #[test]
    fn generic_backend_error_offers_no_fix() {
        let mut app = test_app();
        app.error_backend = Some(BackendKind::Vllm);
        app.raise_error(LocalCodeError::new(ErrorCode::Internal, "unexpected"));
        assert!(app.last_diagnosis.is_none());
        assert!(app.last_repair.is_none());
        let buttons = app.modal.as_ref().unwrap().buttons();
        assert!(!buttons.contains(&"Fix"), "no Fix button without a repair");
    }

    #[test]
    fn fix_button_leads_when_repair_available() {
        let mut app = test_app();
        // Inject a resolved repair (as if a diagnosis produced one on this OS).
        app.last_repair = Some(RepairPlan {
            title: "Reinstall vllm in a clean environment".into(),
            steps: vec![],
            caveat: None,
            repoint: None,
        });
        app.last_error = Some(LocalCodeError::new(ErrorCode::BackendStartFailed, "x"));
        app.open_error_modal();
        let buttons = app.modal.as_ref().unwrap().buttons();
        assert_eq!(buttons.first(), Some(&"Fix"), "Fix leads as the recommended action");
    }

    #[test]
    fn sudo_prompt_masks_input_shows_command_and_cancels() {
        let mut app = test_app();
        app.pending_sudo = Some(PendingSudo {
            job: SudoJob::Repair(RepairPlan {
                title: "Start the Ollama service".into(),
                steps: vec![localcode_backends::InstallStep::Sudo {
                    program: "systemctl".into(),
                    args: vec!["start".into(), "ollama".into()],
                    display: "sudo systemctl start ollama".into(),
                }],
                caveat: None,
                repoint: None,
            }),
            buf: Zeroizing::new(String::new()),
        });
        // Typing accumulates; only the length is ever exposed (masked).
        app.handle_sudo_key(key(KeyCode::Char('h')));
        app.handle_sudo_key(key(KeyCode::Char('i')));
        let (len, cmd) = app.sudo_prompt().expect("prompt open");
        assert_eq!(len, 2);
        assert!(cmd.contains("sudo systemctl start ollama"));
        // The rendered omnibar masks the password and shows the command.
        let screen = render_to_string(&mut app, 120, 40);
        assert!(screen.contains("sudo password"));
        assert!(screen.contains("••"), "password is masked, not echoed");
        // Esc cancels and drops the (zeroized) buffer.
        app.handle_sudo_key(key(KeyCode::Esc));
        assert!(app.sudo_prompt().is_none());
    }

    #[test]
    fn sudo_prompt_covers_elevated_install() {
        // The masked prompt is shared with installs: an Ollama-on-Linux install
        // carries a Sudo step, and the prompt must show that exact command so the
        // user knows what their password authorizes.
        let mut app = test_app();
        let script = "curl -fsSL https://ollama.com/install.sh | sh";
        let plan = InstallPlan::Automated {
            steps: vec![localcode_backends::InstallStep::Sudo {
                program: "sh".into(),
                args: vec!["-c".into(), script.into()],
                display: format!("sudo sh -c '{script}'"),
            }],
            display: format!("sudo sh -c '{script}'"),
        };
        app.pending_sudo = Some(PendingSudo {
            job: SudoJob::Install {
                kind: BackendKind::Ollama,
                plan,
            },
            buf: Zeroizing::new(String::new()),
        });
        let (len, cmd) = app.sudo_prompt().expect("prompt open");
        assert_eq!(len, 0);
        assert!(cmd.starts_with("sudo "));
        assert!(cmd.contains("ollama.com/install.sh"));
        let screen = render_to_string(&mut app, 120, 40);
        assert!(screen.contains("sudo password"));
    }

    #[test]
    fn shift_enter_makes_a_multiline_composer_and_hint_bar_is_last() {
        let mut app = test_app();
        typ(&mut app, "first");
        // Shift+Enter inserts a newline instead of submitting.
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
        typ(&mut app, "second");
        assert_eq!(app.coding_input, "first\nsecond");

        let screen = render_to_string(&mut app, 100, 30);
        assert!(screen.contains("first") && screen.contains("second"));
        // A hint bar sits below the omnibar (so the ❯ input row is not last) and
        // advertises Shift+Enter for a newline.
        let rows: Vec<&str> = screen.lines().collect();
        let last = rows.iter().rev().find(|r| !r.trim().is_empty()).unwrap();
        assert!(last.contains("newline") || last.contains("commands"), "last: {last:?}");
        assert!(!last.contains('❯'), "the input row must not be the terminal's last line");
    }

    #[test]
    fn settings_toggle_tool_and_edit_system_prompt_persist() {
        let mut app = test_app();
        typ(&mut app, "/settings");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.mode, Mode::Settings);

        // Toggle the shell.exec tool off via its row action.
        let toggle = app
            .settings_rows()
            .into_iter()
            .find_map(|r| match r.action {
                Some(SettingAction::ToggleTool(i)) if r.label == "shell.exec" => {
                    Some(SettingAction::ToggleTool(i))
                }
                _ => None,
            })
            .expect("a shell.exec tool row");
        app.activate_setting(toggle);
        assert!(app.config.agent.disabled_tools.iter().any(|t| t == "shell.exec"));

        // Edit the system prompt inline: begin, type, commit.
        app.activate_setting(SettingAction::Edit(SettingField::SystemPrompt));
        assert!(app.settings_editing_field().is_some());
        typ(&mut app, "Be terse.");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.config.agent.system_prompt.as_deref(), Some("Be terse."));
        assert!(app.settings_editing_field().is_none());
    }

    #[test]
    fn model_card_renders_full_wrapped_body_with_controls() {
        let mut app = test_app();
        app.mode = Mode::Models;
        let long = "This model is a code-specialised transformer. ".repeat(20);
        app.model_detail = Some(ModelDetail {
            summary: ModelSummary {
                id: "acme/coder-7b".into(),
                author: None,
                pipeline_tag: Some("text-generation".into()),
                tags: vec![],
                likes: Some(10),
                downloads: Some(1234),
                last_modified: None,
                private: None,
                gated: None,
            },
            siblings: vec![],
            card_data: None,
            sha: None,
            card_markdown: Some(format!("# Coder 7B\n\n{long}\n\n## Usage\nUse it well.")),
            license: Some("apache-2.0".into()),
            parameter_size: Some("7B".into()),
            quants: vec![],
        });
        let screen = render_to_string(&mut app, 100, 40);
        // The full card renders under a "model card" header (wrapped, not a
        // single runaway line), with the controls block still above it.
        assert!(screen.contains("model card"));
        assert!(screen.contains("Coder 7B"));
        assert!(screen.contains("code-specialised transformer"));
        assert!(screen.contains("acme/coder-7b"));
        // The deploy button now renders as a pseudographic pill, not `[ deploy ]`.
        assert!(screen.contains("▐ deploy ▌"));
    }

    #[test]
    fn settings_expose_skills_agents_md_mcp_and_prompt() {
        let mut app = test_app();
        app.mode = Mode::Settings;
        let rows = app.settings_rows();
        let labels: Vec<&str> = rows.iter().map(|r| r.label.as_str()).collect();
        for expected in ["system prompt", "use AGENTS.md", "skills dir", "mcp config"] {
            assert!(labels.contains(&expected), "missing settings row: {expected}");
        }
        // The bundled sample skill shows up as an enable/disable toggle.
        assert!(rows
            .iter()
            .any(|r| matches!(r.kind, SettingsRowKind::Toggle(_)) && r.label.contains("localcode-doctor")));
        // Every built-in tool is listed as a toggle.
        assert!(rows
            .iter()
            .any(|r| matches!(r.kind, SettingsRowKind::Toggle(_)) && r.label == "fs.read"));
    }
}
