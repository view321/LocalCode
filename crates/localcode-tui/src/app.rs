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
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags, MouseEventKind,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use localcode_agent::{
    list_sessions, sessions_root, AgentEvent, AgentSession, CodingAgent, SessionMeta,
    SessionStore, Skill, ToolApprover, ToolRegistry,
};
use localcode_api_client::ApiClient;
use localcode_assistant::{
    backend_supports, classify_weight_format_for, ensure_running, install_local_assistant,
    install_need, install_offer_body, is_installed, preset_for_backend, quant_compatibility_note,
    recommend_backend_for, should_offer_install, startup_greeting, Assistant, AssistantContext,
    DeployToolArgs, LocalAssistantRuntime, ModelActions, PresetInput, ASSISTANT_DISPLAY_NAME,
    BONSAI_FILE, InstallNeed,
};
use localcode_backends::{
    can_elevate_noninteractively, clear_state, delete_downloaded, diagnose,
    ensure_llamacpp_installed, find_downloaded, human_size, list_downloaded, model_dir, now_unix,
    preview_deploy_command, resolve_install_plan, resolve_llamacpp_bin, resolve_repair, run_install,
    run_repair, sanitize_model_dir, scan_active, smoke_test, spawn_detached_worker, write_state,
    BackendKind, BackendRegistry, DashSnapshot, DeployRequest, DeployService, DeployTuning,
    DetectReport, Diagnosis, DownloadState, DownloadStatus, DownloadedModel, InstallPlan, OpenWebUi,
    OpenWebUiHandle, ProcState, RepairPlan, Repoint, SmokeReport, DASH_LOG_CAP, DEFAULT_DEPLOY_CTX,
    OPENWEBUI_DEFAULT_PORT,
};
use localcode_core::config::LocalAssistantPreference;
use localcode_bench::{sample_coding_suite, BenchRunner, Subject};
use localcode_core::config::{ApprovalMode, Config};
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus, Severity};
use localcode_core::paths::AppPaths;
use localcode_core::runtime::{ActiveRuntime, RuntimeStatus};
use localcode_core::theme::{Theme, ThemeMode};
use localcode_gpu::{discover, predict_fit, FitPrediction, FitRequest, GpuInventory};
use localcode_remote::{setup_server, RemoteSession};
use localcode_hf::{discover_quants, HfClient, ModelDetail, ModelFile, ModelSummary, QuantGroup};
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
use unicode_width::UnicodeWidthChar;
use zeroize::Zeroizing;

const MAX_INPUT_HISTORY: usize = 100;

/// How long the results-list selection must sit still before the highlighted
/// model's detail card is loaded automatically. Short enough to feel live,
/// long enough that arrow-key/wheel scrolling doesn't fire a request per row.
const DETAIL_AUTOLOAD_DEBOUNCE: Duration = Duration::from_millis(250);

/// How the assistant's auto-preset should treat the current backend selection
/// when it re-derives deploy defaults from a model card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PresetMode {
    /// A new model was opened — pick the backend that fits its weight format.
    Fresh,
    /// The quantization changed — keep the backend unless it can no longer serve
    /// the new format (e.g. switched from a safetensors quant to a GGUF one).
    QuantChanged,
    /// The user picked the backend explicitly — never override it, just refresh
    /// the card flags and a fitting context for that backend.
    BackendPinned,
}

/// The largest context ≤ `desired` whose predicted VRAM fits total memory,
/// halving down but never below the deploy default. Returns `desired` unchanged
/// when no GPU is detected (the fit estimate is informational only there). Shared
/// by the deploy panel and the agent-driven deploy so both land a fitting value.
fn fit_context_to_vram(
    gpu: &GpuInventory,
    backend: BackendKind,
    quant: Option<&str>,
    desired: u32,
    weight_bytes: u64,
) -> u32 {
    let mut ctx = desired.clamp(512, 1_048_576);
    if gpu.total_vram() == 0 {
        return ctx;
    }
    // Halve until the estimate fits, but never below a usable floor — and never
    // above the model's own desired max (so a small-context model isn't raised).
    let floor = desired.min(DEFAULT_DEPLOY_CTX);
    while ctx > floor {
        let fit = predict_fit(
            gpu,
            &FitRequest {
                weight_bytes,
                param_count: None,
                quant_label: quant.map(|s| s.to_string()),
                context_length: ctx,
                backend: backend.as_str().into(),
            },
        );
        if fit.fits_total {
            break;
        }
        ctx /= 2;
    }
    ctx.max(floor)
}

/// Owned snapshot of the browsed model's fields, gathered from `self` so the
/// deploy-preset computation can borrow them while `self` is mutated.
struct PresetInputsOwned {
    model_id: String,
    quant: Option<String>,
    weight_files: Vec<String>,
    weight_bytes: u64,
    tags: Vec<String>,
    card: Option<String>,
    installed: Vec<BackendKind>,
    has_gpu: bool,
}

/// Which view fills the working area. `Chat` is home (the transcript). The rest
/// are switched to by slash commands; a leading '/' in the omnibar transiently
/// shows the command list over whichever mode is active (see `slash_active`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Chat,
    Models,
    Runtimes,
    Dash,
    Sessions,
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
            Mode::Dash => "dash",
            Mode::Sessions => "sessions",
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
            Mode::Chat => "message the agent…    @ attaches a file · / for commands",
            Mode::Models => "search models — type to filter, Enter to run",
            Mode::Dash => "↑↓ pick a model for the next request · Enter to chat with it",
            Mode::Sessions => "↑↓ pick a past chat · Enter to resume it",
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
    /// Model chain-of-thought / reasoning (streamed; collapsible when done).
    Thinking,
    Tool,
    System,
    Error,
}

#[derive(Debug, Clone)]
pub struct TranscriptEntry {
    pub kind: EntryKind,
    pub text: String,
    /// Still receiving streamed text (Agent/Thinking) or still running (Tool).
    pub live: bool,
    /// Expandable body: full tool args+output, or (unused for plain agent text).
    pub detail: Option<String>,
    /// Whether [`Self::detail`] (or full thinking body) is shown.
    pub expanded: bool,
}

impl TranscriptEntry {
    pub fn new(kind: EntryKind, text: impl Into<String>) -> Self {
        Self {
            kind,
            text: text.into(),
            live: false,
            detail: None,
            expanded: false,
        }
    }

    /// A finished model response the user can click to select + auto-copy.
    /// Restricted to the agent's answer text (not thinking/tool/system rows) and
    /// only once it has content, so an empty streaming placeholder isn't copied.
    pub fn is_model_response(&self) -> bool {
        self.kind == EntryKind::Agent && !self.text.trim().is_empty()
    }

    /// Entries the user can click to toggle verbose / collapsed body.
    pub fn can_toggle(&self) -> bool {
        match self.kind {
            EntryKind::Tool => self
                .detail
                .as_ref()
                .is_some_and(|d| !d.trim().is_empty()),
            EntryKind::Thinking => !self.live && !self.text.trim().is_empty(),
            _ => false,
        }
    }

    pub fn toggle_expanded(&mut self) {
        if self.can_toggle() {
            self.expanded = !self.expanded;
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
    /// The `approvals <tag>` cluster — click to cycle the agent approval mode.
    ApprovalCycle,
    UpdateBadge,
    /// Click the status dashboard body to pin/unpin the expanded log view.
    StatusToggle,
    // Command list (shown while the omnibar starts with '/')
    CommandItem(usize),
    // '@' file-picker rows (shown while the caret is in an '@' token)
    AtItem(usize),
    // Inline banner buttons (were modal buttons)
    ModalButton(usize),
    // Chat
    Transcript,
    /// A collapsible transcript row (thinking or tool). Index into `coding_transcript`.
    TranscriptEntry(usize),
    // Models (two-pane)
    ModelList,
    QuantList,
    BackendCycle,
    /// Apply the model card's recommended backend / context / server flags to
    /// the deploy params in one click (the manual auto-preset trigger).
    DeployAutoConfig,
    DeployButton,
    DeployCancel,
    /// Resume the current model's interrupted background download.
    DownloadResume,
    /// Edit an inline deploy parameter (context, port, per-backend tuning).
    DeployField(DeployField),
    // Runtimes
    RuntimeList,
    // Dash (multi-model manager) — index into all_runtimes()
    /// Select this model as the active one for the next request.
    DashCard(usize),
    /// Copy the launch command to the clipboard.
    DashCopyCmd(usize),
    /// Stop this model.
    DashStop(usize),
    /// Make this model the active one (explicit button).
    DashUse(usize),
    /// Copy the captured error of a model that exited non-zero.
    DashCopyErr(usize),
    /// Deploy this downloaded (not-running) model.
    DashDeploy(usize),
    /// Delete this downloaded model's weights from disk.
    DashDelete(usize),
    /// Toggle this model's favourite state.
    DashFavorite(usize),
    /// Start a new model (opens the Models search/deploy view).
    DashStartNew,
    // Sessions (past chats) — one region over the visible rows; the row index
    // is derived from the click offset plus `sessions_scroll`.
    SessionList,
    /// Start a fresh chat (same as /new) from the sessions header button.
    SessionsNew,
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
    Workspace,
}

/// A tunable deploy parameter, edited inline in the Models view before deploy.
/// Which fields are shown depends on the selected backend (see
/// [`App::deploy_fields`]); each maps to a concrete launch flag per backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployField {
    /// Max context / KV-cache length. vLLM `--max-model-len`, llama.cpp `-c`,
    /// SGLang `--context-length`, Ollama `num_ctx`.
    Context,
    /// Server port to bind (blank = backend default).
    Port,
    /// Fraction of VRAM the server may use. vLLM `--gpu-memory-utilization`,
    /// SGLang `--mem-fraction-static`.
    GpuMemFraction,
    /// GPUs to shard the model across. vLLM `--tensor-parallel-size`,
    /// SGLang `--tp-size`.
    TensorParallel,
    /// Layers to offload to GPU. llama.cpp `--n-gpu-layers`, Ollama `num_gpu`.
    GpuLayers,
    /// Full launch command, editable directly. Blank = build it from the fields
    /// above; anything else is spawned verbatim (shell-split by the backend).
    Command,
}

impl DeployField {
    /// Left-column label shown in the deploy panel.
    pub fn label(self) -> &'static str {
        match self {
            DeployField::Context => "context",
            DeployField::Port => "port",
            DeployField::GpuMemFraction => "gpu mem",
            DeployField::TensorParallel => "tensor par",
            DeployField::GpuLayers => "gpu layers",
            DeployField::Command => "command",
        }
    }
}

/// Width of the label column in the deploy panel. Field values render (and
/// clicks map to caret positions) starting at this offset, so the renderer
/// and the click handler must agree on it.
pub const DEPLOY_LABEL_W: usize = 11;

/// What activating a Settings row does. `Copy` so it can ride inside
/// [`ClickTarget`]; index-carrying variants point into stable catalogs
/// (`ToolRegistry::catalog()` / `App::skills`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingAction {
    ThemeToggle,
    ToggleMouse,
    ToggleStream,
    ApprovalCycle,
    ToggleCloudFallback,
    ToggleAgentsMd,
    ToggleCheckUpdates,
    ToggleShellSandbox,
    ToggleSessions,
    ToggleAutoCompact,
    /// Accept / install the bundled local Bonsai assistant.
    AcceptLocalAssistant,
    TogglePreferLocal,
    ToggleAutoHandleErrors,
    /// Toggle whether models preset their deploy params automatically as they
    /// load (off by default; the deploy panel's auto-config button is manual).
    ToggleAutoDeployHints,
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

/// One model's row in the `/dash` multi-model view. Built by [`App::dash_cards`]
/// in `all_runtimes()` order so a card's index also selects that runtime. Merges
/// the live process monitor (command, logs, exit) with per-model metrics
/// (VRAM estimate, tokens/s, context load).
#[derive(Debug, Clone)]
pub struct DashCard {
    pub name: String,
    /// Backend label ("llamacpp", "vllm", "remote", …).
    pub backend_label: String,
    /// Human status word ("running", "starting", "exited 1", "healthy", …).
    pub status_label: String,
    pub status_is_error: bool,
    /// Status glyph: ● running · ◐ starting · ○ external/idle · ✗ failed.
    pub glyph: &'static str,
    /// The exact launch command (click-to-copy). Empty when unknown (remote).
    pub command: String,
    /// Newest backend log lines (oldest first, at most a couple shown).
    pub logs: Vec<String>,
    /// Estimated VRAM the model occupies, when known.
    pub vram_bytes: Option<u64>,
    /// Decode rate — only the active model is streaming, so `None` for the rest.
    pub tok_per_sec: Option<f64>,
    /// Context tokens used / window — only meaningful for the active model
    /// (the coding session is singular), so `None` when idle.
    pub ctx_used: Option<u32>,
    pub ctx_max: u32,
    /// Full captured error for a non-zero exit, for the "copy error" button.
    pub error_text: Option<String>,
    /// The model selected for the next request.
    pub is_active: bool,
    /// Whether a Stop button applies (false for a reused/external server we
    /// still surface but manage elsewhere).
    pub can_stop: bool,
    /// This running card's index into `all_runtimes()` (drives use/stop, which
    /// address runtimes positionally). `None` for downloaded-only / OpenWebUI.
    pub runtime_index: Option<usize>,
    /// The HF model id, when this card corresponds to a model (drives the
    /// favourite / delete / deploy buttons). `None` for the OpenWebUI card.
    pub model_id: Option<String>,
    /// Whether the model is starred (favourites render first).
    pub is_favorite: bool,
    /// Size of the model's weights on disk, when downloaded.
    pub disk_bytes: Option<u64>,
    /// This model has weights on disk but isn't running — offer Deploy.
    pub can_deploy: bool,
    /// Weights on disk and not currently running — offer Delete.
    pub can_delete: bool,
    /// A curated model we surface for discovery that isn't downloaded yet —
    /// render a "download & run" button instead of Deploy/Delete (see
    /// [`FEATURED_MODELS`]). `command` holds a human descriptor, not a shell line.
    pub is_featured: bool,
    /// The OpenWebUI infrastructure card (rendered specially, no fav/delete).
    pub is_openwebui: bool,
    /// A browser/endpoint URL to surface (OpenWebUI page, or runtime base_url).
    pub url: Option<String>,
}

impl DashCard {
    /// A neutral card with only the common fields set; builders fill the rest.
    fn base(name: String, backend_label: String) -> Self {
        DashCard {
            name,
            backend_label,
            status_label: String::new(),
            status_is_error: false,
            glyph: "○",
            command: String::new(),
            logs: Vec::new(),
            vram_bytes: None,
            tok_per_sec: None,
            ctx_used: None,
            ctx_max: 1,
            error_text: None,
            is_active: false,
            can_stop: false,
            runtime_index: None,
            model_id: None,
            is_favorite: false,
            disk_bytes: None,
            can_deploy: false,
            can_delete: false,
            is_featured: false,
            is_openwebui: false,
            url: None,
        }
    }
}

/// A pre-converted model we surface in `/dash` so it can be downloaded and run
/// without hunting Hugging Face for the right repo. Currently the two known-good
/// colibrì int4 containers (the engine cannot pull weights itself, so a curated
/// pointer to a complete container is the difference between one-click and a
/// support ticket).
struct FeaturedModel {
    /// Hugging Face repo id (what the deploy flow downloads).
    id: &'static str,
    /// Human engine/backend label shown on the card.
    engine: &'static str,
    /// One-line descriptor: approximate on-disk size + what it is.
    descriptor: &'static str,
}

/// The curated download-and-run entries appended to `/dash`. Deduped against
/// running + downloaded models, so a featured card disappears once its weights
/// land on disk.
const FEATURED_MODELS: &[FeaturedModel] = &[
    FeaturedModel {
        id: "mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp",
        engine: "colibrì · GLM-5.2",
        descriptor: "~370 GB · GLM-5.2 744B MoE (upstream colibrì engine)",
    },
    FeaturedModel {
        id: "UnderstandLing/Hy3-colibri-int4",
        engine: "colibrì-hy3 · Hy3",
        descriptor: "~142 GB · Tencent Hy3 295B (colibrì-hy3 fork)",
    },
];

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

/// A coding turn that is still running for a session the user has switched away
/// from. Its task keeps running (detached) and persists to that session's store
/// on its own `Arc` clones, so switching sessions never loses progress. We
/// retain the transcript snapshot taken at switch time so returning to the
/// session restores its view, and the handle so we can abort it on quit.
struct BgChat {
    handle: JoinHandle<()>,
    started: Instant,
    /// The session + store `Arc`s the detached task shares. Held so returning to
    /// the session re-attaches the very same in-memory state the turn is mutating.
    session: Arc<AsyncMutex<AgentSession>>,
    session_store: Arc<AsyncMutex<Option<SessionStore>>>,
    /// Transcript as it looked when the user switched away (includes the
    /// in-flight user message). The final reply is appended when the turn ends.
    transcript: Vec<TranscriptEntry>,
    ctx_used: u32,
    /// True once the turn has completed (result persisted); kept so a returning
    /// user still sees the "finished" transcript without a disk reload race.
    done: bool,
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
    InstallLocalAssistant,
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
    /// A model detail card finished loading. `id` is the model that was asked
    /// for — a result whose id no longer matches the latest request is stale
    /// (the user scrolled on) and is dropped. `auto` marks a scroll-triggered
    /// load, whose failures are reported quietly instead of raising the error
    /// modal.
    DetailDone {
        id: String,
        auto: bool,
        result: Result<ModelDetail, LocalCodeError>,
    },
    DeployEnded(DeployOutcome),
    /// Live progress from a running coding turn (streamed tokens, tools). Tagged
    /// with the session it belongs to so background turns don't corrupt whichever
    /// session is in the foreground.
    CodingEvent {
        session_id: String,
        ev: AgentEvent,
    },
    CodingDone {
        session_id: String,
        result: Result<String, LocalCodeError>,
    },
    ToolConfirm {
        description: String,
        respond: oneshot::Sender<bool>,
    },
    BenchDone(Result<localcode_bench::RunResult, LocalCodeError>),
    AssistantDone(Result<String, LocalCodeError>),
    /// Background auto-repair: assistant finished diagnosing an error.
    AssistantAutoDone(Result<String, LocalCodeError>),
    /// Local Bonsai assistant install progress / completion.
    AssistantInstallProgress(String),
    AssistantInstallDone(Result<Option<Repoint>, LocalCodeError>),
    /// Background warm-start of the local assistant server finished.
    AssistantRuntimeReady(Result<LocalAssistantRuntime, LocalCodeError>),
    DoctorDone(String),
    DetectDone(Vec<DetectReport>),
    ApiHealth(bool),
    /// Fresh GPU/VRAM inventory from the periodic background poll — keeps the
    /// status-bar VRAM meter live instead of frozen at the startup snapshot.
    GpuRefreshed(GpuInventory),
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
    /// Silent startup ensure of llama-server (install scripts / first launch).
    LlamaSetupDone(Result<std::path::PathBuf, LocalCodeError>),
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
    /// Deployed/downloaded model set changed (deploy/stop/delete via a tool):
    /// refresh the runtime list and the on-disk downloaded models.
    ModelsChanged,
    /// An OpenWebUI deploy (`/ui` or the assistant tool) finished.
    OpenWebUiReady(Result<Box<OpenWebUiHandle>, LocalCodeError>),
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

/// Backend-management capability handed to the assistant so it can deploy, stop,
/// delete, and list models on the user's behalf (feature: "assistant deploys the
/// models themselves"). Holds cheap clones/`Arc`s so it can move into the
/// assistant's background task and reach the same live registry the UI uses — a
/// model the assistant deploys therefore appears on `/dash` exactly like a
/// manual deploy. UI-visible changes are signalled back via [`BgMsg`].
struct ModelOps {
    registry: Arc<BackendRegistry>,
    events: EventBus,
    gpu: GpuInventory,
    paths: AppPaths,
    config: Config,
    tx: mpsc::UnboundedSender<BgMsg>,
}

#[async_trait]
impl ModelActions for ModelOps {
    async fn deploy(&self, args: DeployToolArgs) -> Result<String, LocalCodeError> {
        let explicit_backend = args.backend.as_deref().and_then(BackendKind::parse);
        let model_id = args.model_id.clone();

        // Enrich with quant + card metadata (weights / urls / size / tags /
        // README) exactly like the deploy panel, so fit prediction, llama.cpp
        // downloads, and the auto-preset below all work.
        let mut quantization = args.quant.clone();
        let mut weight_bytes = 0u64;
        let mut weight_files: Vec<String> = vec![];
        let mut download_urls: Vec<String> = vec![];
        let mut tags: Vec<String> = vec![];
        let mut card: Option<String> = None;
        if model_id.contains('/') {
            if let Ok(hf) = HfClient::new(
                &self.config.registry,
                self.config.hf_token(),
                self.paths.models_cache.clone(),
            ) {
                if let Ok(detail) = hf.model_info(&model_id).await {
                    let group = match &quantization {
                        Some(q) => detail.quants.iter().find(|g| g.label.eq_ignore_ascii_case(q)),
                        None => detail.quants.first(),
                    };
                    if let Some(g) = group {
                        quantization = Some(g.label.clone());
                        weight_bytes = g.total_size;
                        weight_files = g.files.iter().map(|f| f.filename.clone()).collect();
                        download_urls =
                            weight_files.iter().map(|f| hf.download_url(&model_id, f)).collect();
                    }
                    tags = detail.summary.tags.clone();
                    card = detail.card_markdown.clone();
                }
            }
        }

        // Auto-preset (unless the caller was explicit): pick the backend that
        // matches the weight format so a GGUF model never lands on vLLM (and a
        // colibrì container always lands on its engine), and fill a fitting
        // context + the card's flags when the caller didn't supply them.
        let fmt = classify_weight_format_for(&model_id, quantization.as_deref(), &weight_files);
        let auto_backend = explicit_backend.is_none();
        let backend = explicit_backend.unwrap_or_else(|| {
            recommend_backend_for(
                fmt,
                &model_id,
                &tags,
                card.as_deref(),
                &[],
                self.registry.default_kind(&self.config),
            )
        });
        let preset = preset_for_backend(
            backend,
            &PresetInput {
                model_id: &model_id,
                selected_quant: quantization.as_deref(),
                weight_files: &weight_files,
                tags: &tags,
                card_markdown: card.as_deref(),
                installed_backends: &[],
                configured_default: backend,
                has_gpu: self.gpu.total_vram() > 0,
            },
        );
        let context_length = args.context.unwrap_or_else(|| {
            preset
                .desired_context
                .map(|d| {
                    fit_context_to_vram(&self.gpu, backend, quantization.as_deref(), d, weight_bytes)
                })
                .unwrap_or(DEFAULT_DEPLOY_CTX)
        });

        let req = DeployRequest {
            model_id: model_id.clone(),
            quantization,
            weight_bytes,
            weight_files,
            download_urls,
            local_path: None,
            backend,
            port: args.port,
            context_length,
            tuning: preset.tuning,
            command_override: args.command.clone(),
            // Tool-driven deploy: proceed past the VRAM soft-gate (it may spill
            // to RAM/CPU or fail loudly rather than silently blocking).
            continue_despite_oversize: true,
        };
        let svc = DeployService::new(
            self.registry.clone(),
            self.events.clone(),
            self.gpu.clone(),
            self.paths.models_cache.clone(),
            self.config.hf_token(),
            self.config.hf_mirror_hosts(),
        );
        svc.deploy(req).await?;
        let _ = self.tx.send(BgMsg::ModelsChanged);
        let picked = if auto_backend { " (auto-selected for the weight format)" } else { "" };
        Ok(format!(
            "Deployed {model_id} on {}{picked} with context {context_length}. \
             It is on /dash and selectable for chat.",
            backend.as_str()
        ))
    }

    async fn stop(&self, target: &str) -> Result<String, LocalCodeError> {
        let t = target.trim();
        if t.eq_ignore_ascii_case("ui")
            || t.eq_ignore_ascii_case("openwebui")
            || t.contains("open-webui")
        {
            OpenWebUi::stop_by_name().await;
            let _ = self.tx.send(BgMsg::ModelsChanged);
            return Ok("Stopped OpenWebUI.".into());
        }
        let runtimes = self.registry.list_runtimes().await;
        let found = runtimes.iter().find(|r| {
            r.id.to_string() == t
                || r.name == t
                || r.name.contains(t)
                || r.model_id.as_deref() == Some(t)
        });
        let Some(rt) = found else {
            return Err(LocalCodeError::new(
                ErrorCode::BackendNotFound,
                format!("No running runtime matching '{t}'"),
            )
            .with_hint("Use list_deployments to see names/ids"));
        };
        let id = rt.id.to_string();
        let name = rt.name.clone();
        self.registry.stop_runtime(&id).await?;
        let _ = self.tx.send(BgMsg::ModelsChanged);
        Ok(format!("Stopped {name} and freed its VRAM."))
    }

    async fn list_deployments(&self) -> Result<String, LocalCodeError> {
        let runtimes = self.registry.list_runtimes().await;
        if runtimes.is_empty() {
            return Ok("No models are currently running. Use deploy_model to start one.".into());
        }
        let lines: Vec<String> = runtimes
            .iter()
            .map(|r| {
                format!(
                    "- {} ({:?}) status={:?} url={}",
                    r.name, r.kind, r.status, r.base_url
                )
            })
            .collect();
        Ok(lines.join("\n"))
    }

    async fn list_downloaded(&self) -> Result<String, LocalCodeError> {
        let models = list_downloaded(&self.paths.models_cache);
        if models.is_empty() {
            return Ok("No models are downloaded on disk yet.".into());
        }
        let lines: Vec<String> = models
            .iter()
            .map(|m| {
                let quants = if m.quants.is_empty() {
                    String::new()
                } else {
                    format!(" [{}]", m.quants.join(", "))
                };
                format!("- {} — {}{}", m.model_id, human_size(m.total_bytes), quants)
            })
            .collect();
        Ok(lines.join("\n"))
    }

    async fn delete_model(&self, model_id: &str) -> Result<String, LocalCodeError> {
        // Refuse to delete weights that a running runtime is serving.
        let runtimes = self.registry.list_runtimes().await;
        if runtimes.iter().any(|r| r.model_id.as_deref() == Some(model_id)) {
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("{model_id} is currently running — stop it before deleting"),
            )
            .with_hint("Use stop_model first"));
        }
        let freed = delete_downloaded(&self.paths.models_cache, model_id)?;
        let _ = self.tx.send(BgMsg::ModelsChanged);
        Ok(format!("Deleted {model_id} — freed {}.", human_size(freed)))
    }

    async fn deploy_ui(&self) -> Result<String, LocalCodeError> {
        let urls: Vec<String> = self
            .registry
            .list_runtimes()
            .await
            .iter()
            .map(|r| r.base_url.clone())
            .collect();
        let handle = OpenWebUi::deploy(OPENWEBUI_DEFAULT_PORT, urls).await?;
        let url = handle.url().to_string();
        let _ = self.tx.send(BgMsg::OpenWebUiReady(Ok(Box::new(handle))));
        Ok(format!(
            "OpenWebUI is starting at {url} (ready in ~30s), wired to your deployed models."
        ))
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
    /// Last known mouse position (col, row) — drives hover affordances like
    /// the theme-switcher name reveal. `None` until the mouse first moves.
    pub hover: Option<(u16, u16)>,
    /// Previous-frame status dashboard rect (including borders). Used so hover
    /// expand can grow/shrink without a layout chicken-egg.
    pub status_bar_rect: Rect,
    /// When true the status dashboard stays expanded (10 lines) until clicked
    /// again; when false it expands only while the mouse hovers it.
    pub status_pinned: bool,
    /// Compacted recent log lines for the status dashboard tail (newest last).
    pub status_logs: Vec<String>,
    /// Live decode rate (approx. tokens/s from streamed chars÷4). `None` when
    /// no recent stream sample is available.
    pub tokens_per_sec: Option<f64>,
    /// Streaming window start for the tok/s estimate.
    tok_rate_started: Option<Instant>,
    /// Characters received in the current tok/s window.
    tok_rate_chars: usize,
    /// Highlighted index in the '@' file picker.
    pub at_selected: usize,
    /// The '@' token the picker was dismissed for (Esc). Typing changes the
    /// token, which reopens the picker.
    at_dismissed: Option<String>,
    /// Lazily-walked workspace file list (relative, '/'-separated) backing the
    /// '@' picker. Invalidated on new session / workspace change.
    workspace_files: Option<Vec<String>>,
    pub assistant_configured: bool,
    /// None while the startup probe is still running.
    pub api_healthy: Option<bool>,
    pub gpu: GpuInventory,
    /// Approximate tokens currently in the coding session (chars÷4). Refreshed
    /// from the live session every GPU poll so the status-bar context meter
    /// moves as the conversation grows.
    pub ctx_used_tokens: u32,
    pub runtimes: Vec<ActiveRuntime>,
    pub runtime_selected: usize,
    pub runtime_list_state: ListState,
    /// First visible card index in the `/dash` multi-model view.
    pub dash_scroll: usize,
    /// Highlighted card in `/dash`, indexing the full [`App::dash_cards`] list
    /// (running + downloaded + featured), not just `all_runtimes()`. Arrows and
    /// the wheel move this; the draw auto-scrolls to keep it visible.
    pub dash_selected: usize,
    /// Cards that fit on screen in `/dash`, stored during draw so PageUp/Pages
    /// down can move by a page.
    pub dash_view_cards: usize,
    // Sessions view (/sessions): past chats for this workspace, newest first.
    pub sessions: Vec<SessionMeta>,
    pub session_selected: usize,
    /// First visible row; the draw clamps it and keeps the selection on screen.
    pub sessions_scroll: usize,
    /// Id of the live `session`, so the list can mark the current chat.
    pub current_session_id: String,
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
    /// Model id of the in-flight detail request. `DetailDone` results carrying
    /// any other id are stale (a newer request replaced them) and are dropped.
    detail_request: Option<String>,
    /// Deadline of the debounced follow-the-selection detail load. Scrolling
    /// the results list re-arms it; when it expires the highlighted model's
    /// card is loaded so the right pane switches automatically.
    detail_autoload_at: Option<Instant>,
    pub card_cache: Option<CardCache>,
    pub card_scroll: usize,
    /// Updated during draw so scrolling knows the bounds.
    pub card_view_height: u16,
    pub card_total_lines: usize,
    pub selected_quant: Option<String>,
    pub deploy_backend: BackendKind,
    pub deploy_ctx: u32,
    /// Set once the user edits the context by hand, so the assistant's
    /// auto-preset stops overriding it on the next model/quant/backend change.
    deploy_ctx_user_set: bool,
    pub deploy_port: Option<u16>,
    /// Per-backend deploy tuning (VRAM fraction, tensor-parallel, GPU layers).
    /// `None` means "let the backend choose its default".
    pub deploy_gpu_frac: Option<f32>,
    pub deploy_tensor_parallel: Option<u32>,
    pub deploy_gpu_layers: Option<i32>,
    /// Extra CLI flags from model-card / assistant (merged into DeployTuning).
    pub deploy_extra_args: Vec<String>,
    /// Notes from the last model-card parse (shown in status when applied).
    deploy_hints_notes: Vec<String>,
    /// Which deploy parameter is being edited inline (Models view), if any.
    deploy_editing: Option<DeployField>,
    deploy_field_edit: String,
    /// Caret position (char index) within `deploy_field_edit`. Lets the command
    /// field — the only value long enough to overflow the panel — be edited in
    /// the middle, not just at the tail.
    deploy_field_cursor: usize,
    /// First visible char index while editing: the horizontal scroll offset that
    /// keeps the caret on-screen when the value is wider than the panel. Adjusted
    /// by the renderer, which alone knows the available width.
    deploy_field_scroll: usize,
    /// Full launch-command override from the deploy panel's command field.
    /// `None` = build the command from the fields; `Some` = spawn this verbatim.
    deploy_command_override: Option<String>,
    pub deploy_progress: u8,
    /// Models with weights on disk (cached; refreshed on enter/deploy/delete).
    /// Backs the `/models` "downloaded" tag and the `/dash` downloaded section.
    pub downloaded_models: Vec<DownloadedModel>,
    /// In-flight background downloads, read from each model's on-disk state file
    /// (the source of truth, so tracking survives app restarts). Refreshed on a
    /// timer by [`App::poll_downloads`]; drives the deploy panel's progress bar
    /// and the resume-on-relaunch scan.
    pub active_downloads: Vec<DownloadState>,
    /// Throttle for the background-download poll (rescan + auto-resume).
    last_download_poll: Instant,
    /// Running OpenWebUI container (from `/ui` or the assistant tool), if any.
    pub openwebui: Option<OpenWebUiHandle>,
    /// A `/ui` deploy is in flight (Docker pull/run).
    pub openwebui_busy: bool,
    /// Model id awaiting a delete confirmation (paired with `ConfirmAction::DeleteModel`).
    pending_delete_model: Option<String>,
    /// Coding turns running for sessions the user has switched away from, keyed
    /// by session id. Backs "switch /sessions without losing progress".
    bg_chats: std::collections::HashMap<String, BgChat>,
    /// Local Bonsai weights + llama-server are present.
    pub local_assistant_ready: bool,
    /// Live handle for the local Bonsai assistant (`llama-server -m … Q4_1`).
    /// Kept so the default conversation can use it without `/assistant`.
    local_assistant: Option<std::sync::Arc<LocalAssistantRuntime>>,
    /// Install of the local assistant in progress.
    pub assistant_install_busy: Option<Busy>,
    pub assistant_install_progress: String,
    pub last_fit: Option<FitPrediction>,
    pub pending_oversize_deploy: Option<DeployRequest>,
    // Omnibar (prompt / search / command bar) — always focused.
    pub coding_input: String,
    pub coding_cursor: usize, // char index into coding_input
    pub coding_history: Vec<String>,
    coding_hist_idx: Option<usize>,
    pub coding_transcript: Vec<TranscriptEntry>,
    /// Index into `coding_transcript` of the model response the user clicked to
    /// select. Selecting a response auto-copies it to the clipboard; the row is
    /// painted with the selection highlight. Cleared on `/new` and session load.
    pub coding_selected: Option<usize>,
    pub coding_scroll: usize,
    pub coding_follow: bool,
    /// Updated during draw so PgUp/PgDn know the scroll bounds.
    pub coding_view_height: u16,
    pub coding_total_lines: usize,
    pub skill_count: usize,
    /// Skills discovered from the skills dir, for the Settings view.
    pub skills: Vec<Skill>,
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
    /// Append-only JSONL store when sessions_enabled; None if persistence off.
    session_store: Arc<AsyncMutex<Option<SessionStore>>>,
    /// Shared HTTP client for coding turns, so the connection pool survives the
    /// per-turn agent rebuild.
    coding_http: reqwest::Client,
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
        let local_assistant_ready = is_installed(&config, &paths);
        let assistant_configured = assistant.is_configured(
            config.assistant_api_key().as_deref(),
            Some(&paths),
            Some(&config),
        ) || local_assistant_ready;

        let agent_probe = CodingAgent::new(config.agent.clone());
        let skill_count = agent_probe.skills.list().len();
        let skills = agent_probe.skills.list().to_vec();

        let workspace = config
            .agent
            .workspace_root
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let (session, session_store, resume_note) =
            bootstrap_session(&config.agent, &paths, workspace);
        // Fresh Arc, no other holder yet — try_lock cannot fail here.
        let current_session_id = session
            .try_lock()
            .map(|s| s.id.clone())
            .unwrap_or_default();

        let (bg_tx, bg_rx) = mpsc::unbounded_channel();

        let coding_http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        // Enumerate models already on disk once at startup (refreshed later on
        // deploy/delete and when entering /models or /dash).
        let downloaded_models = list_downloaded(&paths.models_cache);
        // Background downloads left running (or interrupted) by a prior session.
        let active_downloads = scan_active(&paths.models_cache);

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
            hover: None,
            status_bar_rect: Rect::default(),
            status_pinned: false,
            status_logs: Vec::new(),
            tokens_per_sec: None,
            tok_rate_started: None,
            tok_rate_chars: 0,
            at_selected: 0,
            at_dismissed: None,
            workspace_files: None,
            assistant_configured,
            api_healthy: None,
            gpu,
            ctx_used_tokens: 0,
            runtimes: vec![],
            runtime_selected: 0,
            runtime_list_state: ListState::default(),
            dash_scroll: 0,
            dash_selected: 0,
            dash_view_cards: 1,
            sessions: vec![],
            session_selected: 0,
            sessions_scroll: 0,
            current_session_id,
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
            detail_request: None,
            detail_autoload_at: None,
            card_cache: None,
            card_scroll: 0,
            card_view_height: 0,
            card_total_lines: 0,
            selected_quant: None,
            deploy_backend: BackendKind::parse(&config.backends.default.kind)
                .unwrap_or(BackendKind::Ollama),
            deploy_ctx: DEFAULT_DEPLOY_CTX,
            deploy_ctx_user_set: false,
            deploy_port: None,
            deploy_gpu_frac: None,
            deploy_tensor_parallel: None,
            deploy_gpu_layers: None,
            deploy_extra_args: Vec::new(),
            deploy_hints_notes: Vec::new(),
            deploy_editing: None,
            deploy_field_edit: String::new(),
            deploy_field_cursor: 0,
            deploy_field_scroll: 0,
            deploy_command_override: None,
            deploy_progress: 0,
            downloaded_models,
            active_downloads,
            last_download_poll: Instant::now(),
            openwebui: None,
            openwebui_busy: false,
            pending_delete_model: None,
            bg_chats: std::collections::HashMap::new(),
            local_assistant_ready,
            local_assistant: None,
            assistant_install_busy: None,
            assistant_install_progress: String::new(),
            last_fit: None,
            pending_oversize_deploy: None,
            coding_input: String::new(),
            coding_cursor: 0,
            coding_history: vec![],
            coding_hist_idx: None,
            coding_transcript: {
                let mut t = vec![TranscriptEntry::new(
                    EntryKind::System,
                    "Welcome to LocalCode. Type a message to chat with the agent, or press / for commands (/models, /remote, /backends, /help). Deploy a model with /models, or connect a remote GPU with /remote.",
                )];
                if config.assistant.greet_on_startup {
                    t.push(TranscriptEntry::new(
                        EntryKind::System,
                        startup_greeting(local_assistant_ready),
                    ));
                }
                t
            },
            coding_selected: None,
            coding_scroll: 0,
            coding_follow: true,
            coding_view_height: 0,
            coding_total_lines: 0,
            skill_count,
            skills,
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
            session_store,
            coding_http,
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
        if let Some(note) = resume_note {
            app.coding_transcript
                .push(TranscriptEntry::new(EntryKind::System, note));
            // Rebuild visible history from the loaded session messages.
            if let Ok(sess) = app.session.try_lock() {
                app.coding_transcript.extend(transcript_from_session(&sess));
            }
        }

        // Seed cached popular models if available
        if let Some(hf) = &app.hf {
            if let Some(cached) = hf.cache().get_search("code") {
                app.models = cached;
            }
        }

        // If llama-server is missing (skipped install script, cargo install, …),
        // fetch the managed prebuilt in the background so deploys / assistant work.
        if resolve_llamacpp_bin(&app.config.backends.llamacpp.bin, &app.paths).is_none() {
            app.ensure_llama_server_background();
        }

        // One-time offer to accept/install the local Bonsai assistant (user can decline).
        // Re-offer any time with `/assistant`, `/assistant accept`, or Settings.
        if should_offer_install(&app.config) {
            app.offer_local_assistant_accept();
        } else if app.local_assistant_ready && app.config.assistant.prefer_local {
            // Already accepted in a previous session. Don't warm-start here: the
            // assistant loads lazily on the first message (see start_coding_turn),
            // so opening the app never spins up llama-server on its own.
            app.coding_transcript.push(TranscriptEntry::new(
                EntryKind::System,
                format!(
                    "{ASSISTANT_DISPLAY_NAME} is your default chat — it starts on your first \
                     message. Ask it to find, deploy, or stop models, or to fix LocalCode. A \
                     model you deploy takes over chat once selected on /dash."
                ),
            ));
        }

        app
    }

    /// Background `ensure_llamacpp_installed` when the binary is not yet available.
    fn ensure_llama_server_background(&mut self) {
        // Unit tests construct `App` without a Tokio runtime; skip the spawn
        // there so hermetic tests don't panic on `tokio::spawn`.
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        let paths = self.paths.clone();
        let tx = self.bg_tx.clone();
        self.set_status("Installing llama-server (one-time setup)…", false);
        tokio::spawn(async move {
            let (ptx, mut prx) = tokio::sync::mpsc::unbounded_channel::<String>();
            let fwd = tx.clone();
            tokio::spawn(async move {
                while let Some(line) = prx.recv().await {
                    let _ = fwd.send(BgMsg::InstallProgress(line));
                }
            });
            let result = ensure_llamacpp_installed(&paths, ptx).await;
            let _ = tx.send(BgMsg::LlamaSetupDone(result));
        });
    }

    /// Kick off `llama-server -m … Q1_0` in the background when the assistant is installed.
    fn warm_start_local_assistant(&mut self) {
        if self.local_assistant.is_some() {
            return;
        }
        let paths = self.paths.clone();
        let config = self.config.clone();
        let tx = self.bg_tx.clone();
        tokio::spawn(async move {
            let result = ensure_running(&config, &paths).await;
            let _ = tx.send(BgMsg::AssistantRuntimeReady(result));
        });
        self.set_status(
            format!("Starting local {ASSISTANT_DISPLAY_NAME} (-m {BONSAI_FILE} -ngl 99)…"),
            false,
        );
    }

    /// Snapshot of the local assistant as an ActiveRuntime, when healthy.
    fn local_assistant_runtime_snapshot(&self) -> Option<ActiveRuntime> {
        self.local_assistant
            .as_ref()
            .map(|rt| rt.as_active_runtime())
    }

    /// All runtimes the agent can use: local assistant (default), registry
    /// deploys, and remote runtimes tunneled over SSH.
    pub fn all_runtimes(&self) -> Vec<ActiveRuntime> {
        let mut out = Vec::new();
        // Prefer the local Bonsai assistant first so it is the default conversation.
        if let Some(rt) = self.local_assistant_runtime_snapshot() {
            out.push(rt);
        }
        out.extend(self.runtimes.iter().cloned());
        out.extend(self.remote_sessions.iter().map(|s| s.runtime.clone()));
        out
    }

    pub fn active_runtime(&self) -> Option<ActiveRuntime> {
        let all = self.all_runtimes();
        all.get(self.runtime_selected)
            .cloned()
            .or_else(|| all.into_iter().next())
    }

    pub fn active_runtime_name(&self) -> Option<String> {
        self.active_runtime().map(|r| r.name.clone())
    }

    // ------------------------------------------------------------------
    // /dash — multi-model manager
    // ------------------------------------------------------------------

    /// One card per model, in `all_runtimes()` order so a card's index also
    /// indexes the runtime it selects. Sources, in order:
    /// the local Bonsai assistant (command + logs from its runtime handle),
    /// registry deploys (merged with their live process [`DashSnapshot`]), and
    /// remote SSH runtimes (managed on the remote box, so command/logs there).
    pub fn dash_cards(&self) -> Vec<DashCard> {
        let runtimes = self.all_runtimes();
        // Match `active_runtime()`: the selected index when valid, else the first
        // (so the "★ next request" marker never disagrees with what actually runs).
        let active = if self.runtime_selected < runtimes.len() {
            self.runtime_selected
        } else {
            0
        };
        let assistant_offset = usize::from(self.local_assistant.is_some());
        let reg_len = self.runtimes.len();

        // Index the live process monitors by runtime id for the registry cards.
        let snaps: std::collections::HashMap<String, DashSnapshot> = self
            .registry
            .monitors()
            .snapshot(DASH_LOG_CAP)
            .into_iter()
            .map(|s| (s.runtime_id.clone(), s))
            .collect();

        // 1) Running-runtime cards. Each carries its all_runtimes() index so the
        // use/stop actions still address runtimes positionally after we reorder.
        let mut running = Vec::with_capacity(runtimes.len());
        for (i, rt) in runtimes.iter().enumerate() {
            let is_active = i == active;
            // Per-model live metrics: only the active model streams / holds the
            // singular coding session, so tok/s and ctx are attributed to it.
            let tok_per_sec = if is_active { self.tokens_per_sec } else { None };
            // Each card shows its own model's context window when known (the
            // assistant's 128k, a deploy's `-c`), else the deploy-form value.
            let ctx_max = rt.context_tokens.unwrap_or(self.deploy_ctx).max(1);
            let ctx_used = if is_active {
                Some(self.ctx_used_tokens)
            } else {
                None
            };

            let mut card = if assistant_offset == 1 && i == 0 {
                self.dash_card_assistant(rt, is_active, tok_per_sec, ctx_used, ctx_max)
            } else if i >= assistant_offset && i < assistant_offset + reg_len {
                self.dash_card_registry(rt, &snaps, is_active, tok_per_sec, ctx_used, ctx_max)
            } else {
                self.dash_card_remote(rt, is_active, tok_per_sec, ctx_used, ctx_max)
            };
            card.runtime_index = Some(i);
            card.model_id = rt.model_id.clone();
            card.url = Some(rt.base_url.clone());
            if let Some(mid) = &rt.model_id {
                card.is_favorite = self.config.is_favorite(mid);
                card.disk_bytes = self
                    .downloaded_models
                    .iter()
                    .find(|m| sanitize_model_dir(&m.model_id) == sanitize_model_dir(mid))
                    .map(|m| m.total_bytes);
            }
            running.push(card);
        }

        // 2) Downloaded-only cards: models with weights on disk that aren't
        // currently running. These show size + Deploy/Delete buttons.
        let running_ids: std::collections::HashSet<String> = runtimes
            .iter()
            .filter_map(|r| r.model_id.as_ref().map(|m| sanitize_model_dir(m)))
            .collect();
        for m in &self.downloaded_models {
            if running_ids.contains(&sanitize_model_dir(&m.model_id)) {
                continue;
            }
            let mut card = DashCard::base(m.model_id.clone(), "on disk".into());
            card.glyph = "▪";
            card.status_label = "downloaded".into();
            card.model_id = Some(m.model_id.clone());
            card.is_favorite = self.config.is_favorite(&m.model_id);
            card.disk_bytes = Some(m.total_bytes);
            card.can_deploy = true;
            card.can_delete = true;
            card.command = if m.quants.is_empty() {
                human_size(m.total_bytes)
            } else {
                format!("{} · {}", m.quants.join(", "), human_size(m.total_bytes))
            };
            card.logs = vec![format!("weights on disk: {}", m.dir.display())];
            running.push(card);
        }

        // 2b) Featured download-and-run cards: curated containers not yet on disk
        // and not already running. These offer a single "download & run" button
        // that routes into the normal deploy flow (which fetches then serves).
        for fm in FEATURED_MODELS {
            let sid = sanitize_model_dir(fm.id);
            let already = running_ids.contains(&sid)
                || self
                    .downloaded_models
                    .iter()
                    .any(|m| sanitize_model_dir(&m.model_id) == sid);
            if already {
                continue;
            }
            let mut card = DashCard::base(fm.id.to_string(), fm.engine.to_string());
            card.glyph = "✦";
            card.status_label = "available".into();
            card.model_id = Some(fm.id.to_string());
            card.is_favorite = self.config.is_favorite(fm.id);
            card.can_deploy = true;
            card.is_featured = true;
            card.command = fm.descriptor.to_string();
            card.logs = vec!["not downloaded — click download & run to fetch it".into()];
            running.push(card);
        }

        // 3) Order: favourites first, then running before downloaded, featured
        // (not-yet-downloaded) last, then name.
        running.sort_by(|a, b| {
            b.is_favorite
                .cmp(&a.is_favorite)
                .then(b.runtime_index.is_some().cmp(&a.runtime_index.is_some()))
                .then(a.is_featured.cmp(&b.is_featured))
                .then(a.name.to_lowercase().cmp(&b.name.to_lowercase()))
        });

        // 4) The OpenWebUI card (infrastructure) pins to the top when running.
        let mut cards = Vec::with_capacity(running.len() + 1);
        if let Some(ui) = &self.openwebui {
            cards.push(self.dash_card_openwebui(ui));
        }
        cards.extend(running);
        cards
    }

    fn dash_card_openwebui(&self, ui: &OpenWebUiHandle) -> DashCard {
        let (glyph, status, is_err) = match ui.state() {
            ProcState::Running => ("●", "running".to_string(), false),
            ProcState::Starting => ("◐", "starting".to_string(), false),
            ProcState::Exited { ok: false, .. } => ("✗", "exited".to_string(), true),
            _ => ("○", "stopped".to_string(), false),
        };
        let mut card = DashCard::base("OpenWebUI".into(), "openwebui".into());
        card.glyph = glyph;
        card.status_label = status;
        card.status_is_error = is_err;
        card.command = ui.command().to_string();
        card.logs = ui.recent_logs(3);
        card.can_stop = true;
        card.is_openwebui = true;
        card.url = Some(ui.url().to_string());
        card
    }

    fn dash_card_assistant(
        &self,
        rt: &ActiveRuntime,
        is_active: bool,
        tok_per_sec: Option<f64>,
        ctx_used: Option<u32>,
        ctx_max: u32,
    ) -> DashCard {
        let (command, logs) = match &self.local_assistant {
            Some(a) => (a.command().to_string(), a.recent_logs(4)),
            None => (String::new(), Vec::new()),
        };
        DashCard {
            name: rt.name.clone(),
            backend_label: "llamacpp".into(),
            status_label: "running (default)".into(),
            status_is_error: false,
            glyph: "●",
            command,
            logs,
            vram_bytes: None,
            tok_per_sec,
            ctx_used,
            ctx_max,
            error_text: None,
            is_active,
            can_stop: true,
            runtime_index: None,
            model_id: None,
            is_favorite: false,
            disk_bytes: None,
            can_deploy: false,
            can_delete: false,
            is_featured: false,
            is_openwebui: false,
            url: None,
        }
    }

    fn dash_card_registry(
        &self,
        rt: &ActiveRuntime,
        snaps: &std::collections::HashMap<String, DashSnapshot>,
        is_active: bool,
        tok_per_sec: Option<f64>,
        ctx_used: Option<u32>,
        ctx_max: u32,
    ) -> DashCard {
        let snap = snaps.get(&rt.id.to_string());
        let (glyph, status_label, status_is_error) = match snap.map(|s| &s.state) {
            Some(ProcState::Running) => ("●", "running".to_string(), false),
            Some(ProcState::Starting) => ("◐", "starting".to_string(), false),
            Some(ProcState::External) => ("○", "external".to_string(), false),
            Some(ProcState::Exited { code, ok: false }) => (
                "✗",
                match code {
                    Some(c) => format!("exited (code {c})"),
                    None => "exited (killed)".to_string(),
                },
                true,
            ),
            Some(ProcState::Exited { ok: true, .. }) => ("○", "exited".to_string(), false),
            None => ("○", crate::ui::status_word(rt.status).to_string(), false),
        };
        DashCard {
            name: rt.name.clone(),
            backend_label: BackendKind::from_runtime_kind(rt.kind)
                .map(|b| b.as_str().to_string())
                .unwrap_or_else(|| "backend".into()),
            status_label,
            status_is_error,
            glyph,
            command: snap.map(|s| s.command.clone()).unwrap_or_default(),
            logs: snap.map(|s| s.log_tail.clone()).unwrap_or_default(),
            vram_bytes: snap.and_then(|s| s.est_vram_bytes),
            tok_per_sec,
            ctx_used,
            ctx_max,
            error_text: snap.and_then(|s| s.error_text()),
            is_active,
            can_stop: true,
            runtime_index: None,
            model_id: None,
            is_favorite: false,
            disk_bytes: None,
            can_deploy: false,
            can_delete: false,
            is_featured: false,
            is_openwebui: false,
            url: None,
        }
    }

    fn dash_card_remote(
        &self,
        rt: &ActiveRuntime,
        is_active: bool,
        tok_per_sec: Option<f64>,
        ctx_used: Option<u32>,
        ctx_max: u32,
    ) -> DashCard {
        DashCard {
            name: rt.name.clone(),
            backend_label: "remote".into(),
            status_label: crate::ui::status_word(rt.status).to_string(),
            status_is_error: rt.status == RuntimeStatus::Unhealthy,
            glyph: "◈",
            command: format!("ssh tunnel → {}", rt.base_url),
            logs: vec![format!("managed on the remote GPU box · {}", rt.base_url)],
            vram_bytes: None,
            tok_per_sec,
            ctx_used,
            ctx_max,
            error_text: None,
            is_active,
            can_stop: true,
            runtime_index: None,
            model_id: None,
            is_favorite: false,
            disk_bytes: None,
            can_deploy: false,
            can_delete: false,
            is_featured: false,
            is_openwebui: false,
            url: None,
        }
    }

    /// Copy `text` to the system clipboard via the OSC 52 terminal escape (works
    /// over SSH and inside tmux with `set-clipboard on`, no native dependency),
    /// and confirm in the status line. Terminals that don't support OSC 52
    /// ignore it — the user can still F2 into select mode to copy manually.
    fn copy_to_clipboard(&mut self, text: &str, what: &str) {
        if text.trim().is_empty() {
            self.set_status(format!("Nothing to copy for {what}"), false);
            return;
        }
        osc52_copy(text);
        self.set_status(format!("Copied {what} to clipboard"), false);
    }

    /// Stop the runtime/OpenWebUI backing dash card `card_idx` (Stop button).
    fn dash_stop(&mut self, card_idx: usize) {
        let cards = self.dash_cards();
        let Some(card) = cards.get(card_idx) else {
            return;
        };
        if card.is_openwebui {
            self.stop_openwebui();
            return;
        }
        if let Some(ri) = card.runtime_index {
            self.runtime_selected = ri;
            self.start_stop_runtime();
        }
    }

    /// Clicking dash card `card_idx`: move the highlight there, then act — a
    /// running card becomes the active runtime; a downloaded-only or featured
    /// card offers to deploy/download it; the OpenWebUI card copies its URL.
    fn dash_use(&mut self, card_idx: usize) {
        let cards = self.dash_cards();
        let Some(card) = cards.get(card_idx) else {
            return;
        };
        self.dash_selected = card_idx;
        if let Some(ri) = card.runtime_index {
            if let Some(name) = self.all_runtimes().get(ri).map(|r| r.name.clone()) {
                self.runtime_selected = ri;
                self.set_status(format!("Next request uses {name}"), false);
            }
        } else if card.is_openwebui {
            if let Some(url) = card.url.clone() {
                self.copy_to_clipboard(&url, "OpenWebUI URL");
            }
        } else if card.can_deploy {
            self.dash_deploy(card_idx);
        }
    }

    /// Move the `/dash` highlight by `delta` cards over the whole list (running +
    /// downloaded + featured) and keep `runtime_selected` in step when the new
    /// selection is a running model, so "next request" and the live metrics
    /// track the highlight. This is the single mover used by arrows and the wheel.
    fn dash_move(&mut self, delta: i64) {
        let n = self.dash_cards().len();
        if n == 0 {
            return;
        }
        let cur = self.dash_selected.min(n - 1) as i64;
        self.dash_selected = (cur + delta).clamp(0, n as i64 - 1) as usize;
        self.sync_runtime_from_dash();
    }

    /// If the highlighted `/dash` card is a running runtime, adopt it as the
    /// active runtime (leaves it unchanged when browsing a downloaded/featured or
    /// OpenWebUI card, so scrolling past them doesn't swap the active model).
    fn sync_runtime_from_dash(&mut self) {
        let ri = self
            .dash_cards()
            .get(self.dash_selected)
            .and_then(|c| c.runtime_index);
        if let Some(ri) = ri {
            if ri < self.all_runtimes().len() {
                self.runtime_selected = ri;
            }
        }
    }

    /// Enter / empty-omnibar on the highlighted `/dash` card: chat with a running
    /// model, download-and-run a downloaded/featured one, or copy the OpenWebUI
    /// URL. Mirrors [`App::dash_use`] but drops into Chat for a running model.
    fn dash_activate(&mut self, card_idx: usize) {
        let cards = self.dash_cards();
        let Some(card) = cards.get(card_idx) else {
            return;
        };
        if let Some(ri) = card.runtime_index {
            if let Some(name) = self.all_runtimes().get(ri).map(|r| r.name.clone()) {
                self.runtime_selected = ri;
                self.set_status(format!("Next request uses {name}"), false);
            }
            self.set_mode(Mode::Chat);
        } else if card.is_openwebui {
            if let Some(url) = card.url.clone() {
                self.copy_to_clipboard(&url, "OpenWebUI URL");
            }
        } else if card.can_deploy {
            self.dash_deploy(card_idx);
        }
    }

    /// Deploy the model backing dash card `card_idx` (Deploy / "download & run"
    /// button). Opens the Models view seeded with that model so the user confirms
    /// the backend/quant, matching how a normal deploy is reviewed. For a
    /// featured card the weights aren't on disk yet, so Deploy there downloads
    /// first — the confirm screen shows the size before the fetch commits.
    fn dash_deploy(&mut self, card_idx: usize) {
        let cards = self.dash_cards();
        let Some(card) = cards.get(card_idx) else {
            return;
        };
        let Some(model_id) = card.model_id.clone() else {
            return;
        };
        let featured = card.is_featured;
        self.set_mode(Mode::Models);
        self.model_query = model_id.clone();
        self.start_load_detail_for(&model_id);
        let msg = if featured {
            format!("Loading {model_id} — review the size/backend, then Deploy to download & run")
        } else {
            format!("Loading {model_id} — pick a quant/backend, then Deploy")
        };
        self.set_status(msg, false);
    }

    /// Toggle the favourite state of the model backing dash card `card_idx`.
    fn dash_favorite(&mut self, card_idx: usize) {
        let cards = self.dash_cards();
        if let Some(model_id) = cards.get(card_idx).and_then(|c| c.model_id.clone()) {
            self.toggle_favorite_model(&model_id);
        }
    }

    /// Ask to delete the downloaded weights backing dash card `card_idx`.
    fn dash_delete(&mut self, card_idx: usize) {
        let cards = self.dash_cards();
        let Some(card) = cards.get(card_idx) else {
            return;
        };
        let Some(model_id) = card.model_id.clone() else {
            return;
        };
        if !card.can_delete {
            self.set_status("Stop the model before deleting its weights", true);
            return;
        }
        let size = card
            .disk_bytes
            .map(human_size)
            .unwrap_or_else(|| "its".into());
        self.pending_delete_model = Some(model_id.clone());
        self.modal = Some(ModalState::confirm(
            format!("Delete {model_id}?"),
            format!(
                "Remove {size} of downloaded weights from disk. This frees space; the \
                 model must be re-downloaded before it can be deployed again."
            ),
            ConfirmAction::DeleteModel,
        ));
    }

    /// Perform the delete confirmed by the modal.
    fn confirm_delete_model(&mut self) {
        let Some(model_id) = self.pending_delete_model.take() else {
            return;
        };
        match delete_downloaded(&self.paths.models_cache, &model_id) {
            Ok(freed) => {
                self.refresh_downloaded();
                self.set_status(format!("Deleted {model_id} — freed {}", human_size(freed)), false);
            }
            Err(e) => self.raise_error(e),
        }
    }

    /// Stop and forget the running OpenWebUI container.
    fn stop_openwebui(&mut self) {
        if let Some(ui) = self.openwebui.take() {
            self.set_status("Stopping OpenWebUI…", false);
            tokio::spawn(async move {
                ui.stop().await;
            });
        } else {
            // Nothing tracked, but a stale container may exist — clean up by name.
            tokio::spawn(async move {
                OpenWebUi::stop_by_name().await;
            });
        }
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

    /// Esc: cancel the foreground task (search/coding/remote), or a running
    /// install/repair/update. A deploy is intentionally excluded — it is
    /// cancelled only via its on-screen Cancel button (see `cancel_deploy`).
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
            if b.kind == BusyKind::Detail {
                // No DetailDone will arrive for the aborted request; forget it
                // so the next request for the same id isn't skipped as a dup.
                self.detail_request = None;
            }
            if b.kind == BusyKind::Coding {
                // Keep whatever streamed before the cancel; just close it out.
                self.finalize_transcript_live();
                self.coding_transcript
                    .push(TranscriptEntry::new(EntryKind::System, "turn cancelled"));
                self.coding_follow = true;
                // Persist whatever the aborted turn produced. Aborting drops the
                // turn task's session lock, so this sync can acquire it; without
                // it, a cancelled turn is lost from the session file (the only
                // sync call lived at the end of the turn task, past the abort).
                let session = self.session.clone();
                let store = self.session_store.clone();
                tokio::spawn(async move {
                    let session = session.lock().await;
                    if let Some(store) = store.lock().await.as_mut() {
                        if let Err(e) = store.sync(&session) {
                            tracing::warn!(error = %e, "failed to persist cancelled coding turn");
                        }
                    }
                });
            }
            if b.kind == BusyKind::Remote {
                self.remote_connecting = None;
                // Undo the speculative Ollama repoint.
                if let Some(url) = self.pre_remote_ollama_url.take() {
                    self.config.backends.ollama.base_url = url;
                }
            }
            self.set_status(format!("Cancelled: {}", b.label), false);
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
        self.coding_transcript.retain(|e| {
            !(e.live
                && matches!(e.kind, EntryKind::Agent | EntryKind::Thinking)
                && e.text.trim().is_empty())
        });
        for e in &mut self.coding_transcript {
            e.live = false;
        }
    }

    /// Mark currently-live Agent/Thinking rows as finished (before a new kind starts).
    fn seal_live_stream_entries(&mut self) {
        for e in &mut self.coding_transcript {
            if e.live && matches!(e.kind, EntryKind::Agent | EntryKind::Thinking) {
                e.live = false;
            }
        }
        // Drop empty sealed stream rows.
        self.coding_transcript.retain(|e| {
            !(matches!(e.kind, EntryKind::Agent | EntryKind::Thinking)
                && !e.live
                && e.text.trim().is_empty())
        });
    }

    /// Apply one live agent event to the transcript.
    fn apply_agent_event(&mut self, ev: AgentEvent) {
        match ev {
            AgentEvent::ThinkingDelta(text) => {
                self.note_stream_chars(&text);
                // Seal prior agent text if any; thinking usually precedes the answer.
                if let Some(e) = self.coding_transcript.last_mut() {
                    if e.kind == EntryKind::Agent && e.live {
                        e.live = false;
                    }
                }
                match self.coding_transcript.last_mut() {
                    Some(e) if e.kind == EntryKind::Thinking && e.live => e.text.push_str(&text),
                    _ => self.coding_transcript.push(TranscriptEntry {
                        kind: EntryKind::Thinking,
                        text,
                        live: true,
                        detail: None,
                        // Stream expanded so users see reasoning live.
                        expanded: true,
                    }),
                }
                self.coding_follow = true;
            }
            AgentEvent::Delta(text) => {
                self.note_stream_chars(&text);
                if let Some(e) = self.coding_transcript.last_mut() {
                    if e.kind == EntryKind::Thinking && e.live {
                        e.live = false;
                    }
                }
                match self.coding_transcript.last_mut() {
                    Some(e) if e.kind == EntryKind::Agent && e.live => e.text.push_str(&text),
                    _ => self.coding_transcript.push(TranscriptEntry {
                        kind: EntryKind::Agent,
                        text,
                        live: true,
                        detail: None,
                        expanded: false,
                    }),
                }
                self.coding_follow = true;
            }
            AgentEvent::MessageComplete => {
                // Seal live thinking and agent rows for this round.
                for e in &mut self.coding_transcript {
                    if e.live && matches!(e.kind, EntryKind::Agent | EntryKind::Thinking) {
                        e.live = false;
                    }
                }
                self.coding_transcript.retain(|e| {
                    !(matches!(e.kind, EntryKind::Agent | EntryKind::Thinking)
                        && e.text.trim().is_empty()
                        && e.detail.is_none())
                });
            }
            AgentEvent::ToolStarted { name, args_preview } => {
                self.seal_live_stream_entries();
                let preview = if args_preview.is_empty() {
                    String::new()
                } else {
                    format!("  {args_preview}")
                };
                self.coding_transcript.push(TranscriptEntry {
                    kind: EntryKind::Tool,
                    text: format!("⋯ {name}{preview}"),
                    live: true,
                    detail: None,
                    expanded: false,
                });
                self.coding_follow = true;
            }
            AgentEvent::ToolFinished {
                name,
                ok,
                summary,
                args,
                output,
            } => {
                let mark = if ok { "✓" } else { "✗" };
                let text = format!("{mark} {name}  {summary}");
                let detail = format_tool_detail(&args, &output);
                let expandable = !detail.trim().is_empty();
                let text = if expandable {
                    format!("{text}  ▸")
                } else {
                    text
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
                        e.detail = if expandable { Some(detail) } else { None };
                        e.expanded = false;
                    }
                    None => {
                        let mut entry = TranscriptEntry::new(EntryKind::Tool, text);
                        if expandable {
                            entry.detail = Some(detail);
                        }
                        self.coding_transcript.push(entry);
                    }
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

        // When the local assistant is ready, kick off a background diagnosis
        // first so install/backend errors get agent help without an extra click.
        if self.config.assistant.auto_handle_errors
            && self.local_assistant_ready
            && !self.fg_busy()
            && self.assistant_install_busy.is_none()
        {
            self.start_assistant_auto();
        }
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
            Some(RetryAction::InstallLocalAssistant) => self.start_install_local_assistant(),
            None => self.set_status("Nothing to retry", false),
        }
    }

    // ------------------------------------------------------------------
    // Background task starters (all non-blocking)
    // ------------------------------------------------------------------

    /// Poll GPU/VRAM/temp in the background and stream it back to the UI.
    /// `discover()` shells out to `nvidia-smi` (blocking, ~100-300ms), so it runs
    /// on the blocking pool — never on the render thread — and the result arrives
    /// as a `BgMsg`. Interval is 1s so the status dashboard meters stay live.
    pub fn start_gpu_refresh(&mut self) {
        let tx = self.bg_tx.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(1));
            // Skip the immediate first tick: `new()` already seeded `app.gpu`.
            tick.tick().await;
            loop {
                tick.tick().await;
                match tokio::task::spawn_blocking(discover).await {
                    Ok(Ok(inv)) => {
                        // The receiver is dropped once the app exits; stop polling.
                        if tx.send(BgMsg::GpuRefreshed(inv)).is_err() {
                            break;
                        }
                    }
                    // A failed probe (no nvidia-smi, driver hiccup) just means we
                    // keep the last good inventory and try again next tick.
                    // Still wake the UI so context usage can refresh on the same
                    // cadence even when nvidia-smi is missing.
                    Ok(Err(_)) | Err(_) => {
                        if tx
                            .send(BgMsg::GpuRefreshed(GpuInventory {
                                devices: vec![],
                                detection_method: "probe-failed".into(),
                                warnings: vec![],
                            }))
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
        });
    }

    /// Recompute `ctx_used_tokens` from the live agent session when the lock is
    /// free; otherwise fall back to an estimate from the transcript. Called on
    /// the 1s GPU tick and after coding turns so the status meter stays honest.
    pub fn refresh_ctx_usage(&mut self) {
        if let Ok(session) = self.session.try_lock() {
            self.ctx_used_tokens = estimate_session_tokens(&session);
            return;
        }
        // Session locked mid-turn — approximate from the transcript the UI already has.
        let chars: usize = self
            .coding_transcript
            .iter()
            .map(|e| e.text.chars().count())
            .sum();
        self.ctx_used_tokens = (chars / 4) as u32;
    }

    /// Whether the status dashboard should render expanded (10 content lines).
    /// Pinned by click, or temporarily while the mouse hovers the bar.
    pub fn status_expanded(&self) -> bool {
        self.status_pinned || self.hover_over_status()
    }

    /// True when the last known mouse position falls inside the previous-frame
    /// status dashboard rect.
    pub fn hover_over_status(&self) -> bool {
        let Some((c, r)) = self.hover else {
            return false;
        };
        let a = self.status_bar_rect;
        if a.width == 0 || a.height == 0 {
            return false;
        }
        c >= a.x
            && c < a.x.saturating_add(a.width)
            && r >= a.y
            && r < a.y.saturating_add(a.height)
    }

    /// Toggle pinned expansion of the status dashboard (click to hold open).
    pub fn toggle_status_pin(&mut self) {
        self.status_pinned = !self.status_pinned;
    }

    /// Refresh the compact log tail shown under the status metrics.
    pub fn refresh_status_logs(&mut self) {
        let redact = self.config.logging.redact_secrets;
        let lines = localcode_log::read_recent_logs(&self.paths.log_dir, 8, None, redact)
            .ok()
            .map(|s| {
                s.lines()
                    .map(compact_log_line)
                    .filter(|l| !l.trim().is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        self.status_logs = lines;
    }

    /// Feed streamed text into the live tokens/s estimate (chars÷4 / elapsed).
    fn note_stream_chars(&mut self, text: &str) {
        let n = text.chars().count();
        if n == 0 {
            return;
        }
        let now = Instant::now();
        if self.tok_rate_started.is_none() {
            self.tok_rate_started = Some(now);
            self.tok_rate_chars = 0;
        }
        self.tok_rate_chars = self.tok_rate_chars.saturating_add(n);
        if let Some(start) = self.tok_rate_started {
            let elapsed = now.duration_since(start).as_secs_f64();
            if elapsed >= 0.2 {
                let tokens = self.tok_rate_chars as f64 / 4.0;
                self.tokens_per_sec = Some(tokens / elapsed);
            }
            // Slide the window so a long idle stretch mid-turn doesn't freeze the rate.
            if elapsed >= 2.0 {
                self.tok_rate_started = Some(now);
                self.tok_rate_chars = 0;
            }
        }
    }

    /// Clear the streaming window at end-of-turn; keep the last rate on screen.
    fn end_stream_rate_window(&mut self) {
        self.tok_rate_started = None;
        self.tok_rate_chars = 0;
    }

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
        let Some(id) = self.models.get(self.model_selected).map(|m| m.id.clone()) else {
            return;
        };
        self.start_load_detail_for(&id);
    }

    /// Load the HF detail card for an explicit model id (used by `/dash`'s Deploy
    /// button, which acts on a downloaded model that isn't in the search list).
    fn start_load_detail_for(&mut self, id: &str) {
        self.request_detail(id, false);
    }

    /// Spawn a detail load for `id`. An in-flight detail load is aborted and
    /// replaced (its result would be stale); any other foreground work keeps
    /// its claim — explicit requests report "busy", auto (scroll-triggered)
    /// ones silently retry after the next debounce window.
    fn request_detail(&mut self, id: &str, auto: bool) {
        if self.busy.as_ref().is_some_and(|b| b.kind == BusyKind::Detail) {
            if self.detail_request.as_deref() == Some(id) {
                return; // already loading exactly this model
            }
            if let Some(b) = self.busy.take() {
                b.handle.abort();
            }
            self.detail_request = None;
        } else if self.fg_busy() {
            if auto {
                self.arm_detail_autoload();
            } else {
                self.set_status("Busy — Esc to cancel first", false);
            }
            return;
        }
        let hf = self.hf.clone();
        let models_dir = self.paths.models_cache.clone();
        let id = id.to_string();
        let tx = self.bg_tx.clone();
        let label = format!("Loading {id}");
        self.detail_request = Some(id.clone());
        let handle = tokio::spawn(async move {
            let result = fetch_detail_or_offline(hf, &models_dir, &id).await;
            let _ = tx.send(BgMsg::DetailDone { id, auto, result });
        });
        self.begin_busy(BusyKind::Detail, label, handle);
    }

    /// (Re)arm the debounced follow-the-selection detail load. Called on every
    /// selection move in `/models`; the load fires from the event loop once the
    /// highlight has sat still for [`DETAIL_AUTOLOAD_DEBOUNCE`].
    fn arm_detail_autoload(&mut self) {
        if self.mode == Mode::Models && !self.models.is_empty() {
            self.detail_autoload_at = Some(Instant::now() + DETAIL_AUTOLOAD_DEBOUNCE);
        }
    }

    /// Fire the debounced auto-load when its deadline has passed: loading the
    /// highlighted model's card so scrolling the results list switches the
    /// right-hand pane without an explicit Enter. Runs every event-loop tick.
    fn maybe_autoload_detail(&mut self) {
        let Some(at) = self.detail_autoload_at else {
            return;
        };
        if Instant::now() < at {
            return;
        }
        self.detail_autoload_at = None;
        if self.mode != Mode::Models {
            return;
        }
        let Some(id) = self.models.get(self.model_selected).map(|m| m.id.clone()) else {
            return;
        };
        if self.model_detail.as_ref().is_some_and(|d| d.summary.id == id)
            || self.detail_request.as_deref() == Some(id.as_str())
        {
            return;
        }
        self.request_detail(&id, true);
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
        // A different quant can be a different format (e.g. a repo carrying both
        // GGUF and safetensors): re-derive the preset, correcting the backend if
        // the current one can no longer serve it.
        self.apply_deploy_preset(PresetMode::QuantChanged);
        self.refresh_fit();
    }

    fn adjust_ctx(&mut self, up: bool) {
        // A manual context change opts out of the assistant's context preset.
        self.deploy_ctx_user_set = true;
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
        // Weights already on disk serve the deploy directly — llama.cpp
        // launches from the local GGUF and Ollama falls back to
        // create-from-GGUF when its pull fails — so a downloaded model
        // deploys without any network. vLLM/SGLang take a local path
        // verbatim as the model argument, which only suits directories, so
        // they keep resolving the HF id themselves.
        let local_path = if matches!(
            self.deploy_backend,
            BackendKind::LlamaCpp | BackendKind::Ollama
        ) {
            self.downloaded_quant_path(&detail.summary.id, &weight_files)
        } else if matches!(
            self.deploy_backend,
            BackendKind::Colibri | BackendKind::ColibriHy3
        ) {
            // colibrì consumes the container DIRECTORY (COLI_MODEL); pin the
            // model dir when every listed shard is already on disk.
            self.downloaded_quant_path(&detail.summary.id, &weight_files)
                .and_then(|p| {
                    std::path::Path::new(&p)
                        .parent()
                        .map(|d| d.display().to_string())
                })
        } else {
            None
        };

        Some(DeployRequest {
            model_id: detail.summary.id.clone(),
            quantization: quant,
            weight_bytes: weight,
            weight_files,
            download_urls,
            local_path,
            backend: self.deploy_backend,
            port: self.deploy_port,
            context_length: self.deploy_ctx,
            tuning: DeployTuning {
                gpu_memory_fraction: self.deploy_gpu_frac,
                tensor_parallel: self.deploy_tensor_parallel,
                gpu_layers: self.deploy_gpu_layers,
                // Filled by the auto-config button (or the opt-in auto path).
                extra_args: self.deploy_extra_args.clone(),
            },
            command_override: self.deploy_command_override.clone(),
            continue_despite_oversize: continue_oversize,
        })
    }

    /// Notes from the last preset, explaining why the backend / flags / context
    /// were chosen. Empty until the auto-config button (or the opt-in auto path)
    /// runs; drives the panel's "auto" tag and reason line.
    pub fn deploy_preset_notes(&self) -> &[String] {
        &self.deploy_hints_notes
    }

    /// Deploy parameters shown (and editable) for the current backend, in
    /// display order. Only the fields a backend actually honors are listed, so
    /// the panel never offers a knob that would be silently ignored. Ollama is a
    /// shared server we don't launch, so it exposes only the model-level knobs it
    /// supports via a derived model (context, GPU layers) — not host/port or the
    /// launch-only vLLM/SGLang flags.
    pub fn deploy_fields(&self) -> Vec<DeployField> {
        match self.deploy_backend {
            BackendKind::Vllm | BackendKind::Sglang => vec![
                DeployField::Context,
                DeployField::Port,
                DeployField::GpuMemFraction,
                DeployField::TensorParallel,
                DeployField::Command,
            ],
            BackendKind::LlamaCpp => vec![
                DeployField::Context,
                DeployField::Port,
                DeployField::GpuLayers,
                DeployField::Command,
            ],
            // coli sizes its own KV slots (no context flag) and its memory
            // tiers (--ram/--vram/--gpu) ride the editable command line.
            BackendKind::Colibri | BackendKind::ColibriHy3 => {
                vec![DeployField::Port, DeployField::Command]
            }
            // Ollama is a shared server we don't spawn — no launch command to edit.
            BackendKind::Ollama => vec![DeployField::Context, DeployField::GpuLayers],
        }
    }

    /// The launch command a deploy would run right now, given the current
    /// selection (model, backend, params). Seeds the editable command field so
    /// "no edit" reproduces the built command. Empty for Ollama.
    pub fn deploy_command_seed(&self) -> String {
        let model_id = self.current_model_id().unwrap_or_default();
        let model_str = match self.deploy_backend {
            // Prefer a downloaded local GGUF path when we already have one.
            BackendKind::LlamaCpp => self
                .downloaded_models
                .iter()
                .find(|m| sanitize_model_dir(&m.model_id) == sanitize_model_dir(&model_id))
                .and_then(|m| {
                    m.files
                        .iter()
                        .find(|f| f.ends_with(".gguf"))
                        .map(|f| m.dir.join(f).display().to_string())
                })
                .unwrap_or_else(|| model_id.clone()),
            // colibrì takes the downloaded container directory (COLI_MODEL).
            BackendKind::Colibri | BackendKind::ColibriHy3 => self
                .downloaded_models
                .iter()
                .find(|m| sanitize_model_dir(&m.model_id) == sanitize_model_dir(&model_id))
                .map(|m| m.dir.display().to_string())
                .unwrap_or_else(|| model_id.clone()),
            _ => model_id.clone(),
        };
        preview_deploy_command(
            &self.config.backends,
            self.deploy_backend,
            &model_str,
            self.deploy_port,
            self.deploy_ctx,
            &DeployTuning {
                gpu_memory_fraction: self.deploy_gpu_frac,
                tensor_parallel: self.deploy_tensor_parallel,
                gpu_layers: self.deploy_gpu_layers,
                extra_args: self.deploy_extra_args.clone(),
            },
        )
    }

    /// Which deploy field is being edited inline (for the renderer).
    pub fn deploy_editing_field(&self) -> Option<DeployField> {
        self.deploy_editing
    }

    /// The in-progress deploy-field edit buffer (for the renderer).
    pub fn deploy_field_edit_buf(&self) -> &str {
        &self.deploy_field_edit
    }

    /// Caret position (char index) within the deploy-field edit buffer.
    pub fn deploy_field_cursor(&self) -> usize {
        self.deploy_field_cursor
    }

    /// Current horizontal scroll offset (first visible char) while editing.
    pub fn deploy_field_scroll(&self) -> usize {
        self.deploy_field_scroll
    }

    /// Store the scroll offset the renderer computed from the live width so the
    /// caret stays on-screen next frame.
    pub fn set_deploy_field_scroll(&mut self, v: usize) {
        self.deploy_field_scroll = v;
    }

    /// Current value of a deploy field as display text (`auto` when unset).
    pub fn deploy_field_display(&self, field: DeployField) -> String {
        match field {
            DeployField::Context => self.deploy_ctx.to_string(),
            DeployField::Port => self
                .deploy_port
                .map(|p| p.to_string())
                .unwrap_or_else(|| "auto".into()),
            DeployField::GpuMemFraction => self
                .deploy_gpu_frac
                .map(|f| format!("{f:.2}"))
                .unwrap_or_else(|| "auto".into()),
            DeployField::TensorParallel => self
                .deploy_tensor_parallel
                .map(|n| n.to_string())
                .unwrap_or_else(|| "auto".into()),
            DeployField::GpuLayers => self
                .deploy_gpu_layers
                .map(|n| n.to_string())
                .unwrap_or_else(|| "auto".into()),
            DeployField::Command => self
                .deploy_command_override
                .clone()
                .unwrap_or_else(|| self.deploy_command_seed()),
        }
    }

    /// Enter inline edit for a deploy parameter, seeding the buffer with the
    /// current value (blank when unset so the user types fresh).
    fn begin_deploy_edit(&mut self, field: DeployField) {
        self.deploy_field_edit = match field {
            DeployField::Context => self.deploy_ctx.to_string(),
            DeployField::Port => self.deploy_port.map(|p| p.to_string()).unwrap_or_default(),
            DeployField::GpuMemFraction => {
                self.deploy_gpu_frac.map(|f| format!("{f}")).unwrap_or_default()
            }
            DeployField::TensorParallel => self
                .deploy_tensor_parallel
                .map(|n| n.to_string())
                .unwrap_or_default(),
            DeployField::GpuLayers => {
                self.deploy_gpu_layers.map(|n| n.to_string()).unwrap_or_default()
            }
            // Seed with the current override, else the built command, so the
            // user edits a full, working command line rather than a blank field.
            DeployField::Command => self
                .deploy_command_override
                .clone()
                .unwrap_or_else(|| self.deploy_command_seed()),
        };
        // Caret at the end of the seeded value; the renderer re-derives the
        // scroll window so the caret is visible even on a long command.
        self.deploy_field_cursor = self.deploy_field_edit.chars().count();
        self.deploy_field_scroll = 0;
        self.deploy_editing = Some(field);
        self.set_status(
            format!("Editing {} — ↵ save, Esc cancel (blank = default)", field.label()),
            false,
        );
    }

    /// A click on a deploy field row. Starts editing the field, or — if this
    /// field is already being edited — only moves the caret to the clicked
    /// character: re-seeding on every click would silently drop the edit in
    /// progress. `rel_col` is measured from the row's left edge; the value
    /// area begins after the [`DEPLOY_LABEL_W`] label column.
    fn click_deploy_field(&mut self, field: DeployField, rel_col: usize) {
        let editing = self.deploy_editing == Some(field);
        // While editing, the drawn value window starts at the renderer-kept
        // horizontal scroll; a freshly opened edit always shows the head.
        let scroll = if editing { self.deploy_field_scroll } else { 0 };
        if !editing {
            self.begin_deploy_edit(field);
        }
        // A click on the label column leaves the caret where the edit put it.
        if let Some(col) = rel_col.checked_sub(DEPLOY_LABEL_W) {
            self.deploy_field_cursor = caret_from_col(&self.deploy_field_edit, scroll, col);
        }
    }

    /// Key handling while a deploy parameter is being edited inline. Supports
    /// caret movement and mid-string insert/delete so a long launch command (the
    /// only field that overflows the panel) can be edited anywhere, not just at
    /// the tail.
    fn handle_deploy_field_key(&mut self, key: crossterm::event::KeyEvent) {
        let len = self.deploy_field_edit.chars().count();
        match key.code {
            KeyCode::Enter => self.commit_deploy_field(),
            KeyCode::Esc => {
                self.deploy_editing = None;
                self.set_status("Edit cancelled", false);
            }
            KeyCode::Backspace => {
                if self.deploy_field_cursor > 0 {
                    let at = self.deploy_field_cursor - 1;
                    let idx = char_to_byte(&self.deploy_field_edit, at);
                    self.deploy_field_edit.remove(idx);
                    self.deploy_field_cursor = at;
                }
            }
            KeyCode::Delete => {
                if self.deploy_field_cursor < len {
                    let idx = char_to_byte(&self.deploy_field_edit, self.deploy_field_cursor);
                    self.deploy_field_edit.remove(idx);
                }
            }
            KeyCode::Left => {
                self.deploy_field_cursor = self.deploy_field_cursor.saturating_sub(1);
            }
            KeyCode::Right => {
                self.deploy_field_cursor = (self.deploy_field_cursor + 1).min(len);
            }
            KeyCode::Home => self.deploy_field_cursor = 0,
            KeyCode::End => self.deploy_field_cursor = len,
            KeyCode::Char(c) => {
                let idx = char_to_byte(&self.deploy_field_edit, self.deploy_field_cursor);
                self.deploy_field_edit.insert(idx, c);
                self.deploy_field_cursor += 1;
            }
            _ => {}
        }
    }

    /// Parse and store the edited deploy parameter. A blank value resets the
    /// field to its default (`auto`); an unparseable value is rejected with a
    /// message and leaves the field unchanged.
    fn commit_deploy_field(&mut self) {
        let Some(field) = self.deploy_editing.take() else {
            return;
        };
        let raw = self.deploy_field_edit.trim().to_string();
        let blank = raw.is_empty();
        let mut ok = true;
        match field {
            DeployField::Context => {
                // A successful hand-edit (explicit value or blank "back to
                // default") opts out of the assistant's context preset.
                if blank {
                    self.deploy_ctx = DEFAULT_DEPLOY_CTX;
                    self.deploy_ctx_user_set = true;
                } else if let Ok(v) = raw.parse::<u32>() {
                    self.deploy_ctx = v.clamp(512, 1_048_576);
                    self.deploy_ctx_user_set = true;
                } else {
                    ok = false;
                }
            }
            DeployField::Port => {
                if blank {
                    self.deploy_port = None;
                } else if let Ok(v) = raw.parse::<u16>() {
                    self.deploy_port = Some(v);
                } else {
                    ok = false;
                }
            }
            DeployField::GpuMemFraction => {
                if blank {
                    self.deploy_gpu_frac = None;
                } else if let Ok(v) = raw.parse::<f32>() {
                    if (0.0..=1.0).contains(&v) {
                        self.deploy_gpu_frac = Some(v);
                    } else {
                        ok = false;
                    }
                } else {
                    ok = false;
                }
            }
            DeployField::TensorParallel => {
                if blank {
                    self.deploy_tensor_parallel = None;
                } else if let Ok(v) = raw.parse::<u32>() {
                    self.deploy_tensor_parallel = (v >= 1).then_some(v);
                } else {
                    ok = false;
                }
            }
            DeployField::GpuLayers => {
                if blank {
                    self.deploy_gpu_layers = None;
                } else if let Ok(v) = raw.parse::<i32>() {
                    self.deploy_gpu_layers = Some(v);
                } else {
                    ok = false;
                }
            }
            DeployField::Command => {
                // Blank, or identical to the freshly-built command, means "use
                // the built command" (no override) so later field edits still
                // take effect. Anything else is spawned verbatim.
                if blank || raw == self.deploy_command_seed() {
                    self.deploy_command_override = None;
                } else {
                    self.deploy_command_override = Some(raw.clone());
                }
            }
        }
        if ok {
            // The command field doesn't affect the VRAM fit estimate.
            if field != DeployField::Command {
                self.refresh_fit();
            }
            let msg = match field {
                DeployField::Command if self.deploy_command_override.is_some() => {
                    "custom deploy command set".to_string()
                }
                DeployField::Command => "using the built deploy command".to_string(),
                _ => format!("{} set", field.label()),
            };
            self.set_status(msg, false);
        } else {
            self.set_status(format!("Invalid {} value", field.label()), true);
        }
    }

    /// Cancel an in-progress deploy. Triggered by the on-screen Cancel button
    /// (deploy is deliberately *not* cancellable with Esc). Aborting the task
    /// drops its `Command`, and `kill_on_drop` stops any child already spawned.
    fn cancel_deploy(&mut self) {
        if let Some(b) = self.deploy_busy.take() {
            b.handle.abort();
            self.deploy_progress = 0;
            self.set_status("Deploy cancelled", false);
        }
    }

    fn start_deploy(&mut self, continue_oversize: bool) {
        // Refresh the card flags for the chosen backend right before deploy, in
        // case the user changed backend since the model loaded. Backend stays
        // pinned to the user's selection here (no auto-switch at deploy time).
        self.apply_deploy_preset(PresetMode::BackendPinned);
        let Some(req) = self.build_deploy_request(continue_oversize) else {
            return;
        };
        // A locally-served backend whose weights aren't on disk yet gets a
        // background, resumable, exit-surviving download instead of a blocking
        // in-deploy fetch. vLLM/SGLang/Ollama resolve their own weights, and an
        // already-downloaded model has a `local_path`, so those deploy directly.
        let needs_weights = req.local_path.is_none()
            && !req.download_urls.is_empty()
            && matches!(
                req.backend,
                BackendKind::LlamaCpp | BackendKind::Colibri | BackendKind::ColibriHy3
            );
        if needs_weights {
            if self.download_for(&req.model_id).is_some() {
                self.set_status(
                    "Already downloading this model in the background — deploy when it finishes",
                    false,
                );
                return;
            }
            self.start_background_download(&req);
            return;
        }
        self.spawn_deploy(req);
    }

    /// Auto-preset deploy parameters as the model / quant / backend changes.
    /// Automatic presets are opt-in (`assistant.auto_deploy_hints`, off by
    /// default); the manual "auto-config" button applies them on demand via
    /// [`App::auto_configure_deploy`]. See [`App::run_deploy_preset`] for what
    /// gets set.
    fn apply_deploy_preset(&mut self, mode: PresetMode) {
        if !self.config.assistant.auto_deploy_hints {
            return;
        }
        self.run_deploy_preset(mode);
    }

    /// The deploy panel's "auto-config" button: derive the backend that matches
    /// the weight format (so a GGUF model never lands on vLLM), a fitting
    /// context, and the card's recommended flags — all in one click, regardless
    /// of the automatic opt-in. This is the only preset trigger that runs by
    /// default, so the assistant's picks never apply until the user asks.
    fn auto_configure_deploy(&mut self) {
        if self.model_detail.is_none() {
            self.set_status(
                "Load a model first (Enter) to auto-configure its deploy params",
                true,
            );
            return;
        }
        self.run_deploy_preset(PresetMode::Fresh);
        if self.deploy_hints_notes.is_empty() {
            self.set_status("Auto-config: params already match the model card", false);
        }
    }

    /// Core preset logic shared by the automatic path and the manual button: the
    /// assistant picks the backend that matches the weight format, a fitting
    /// context, and the card's recommended flags. Every field stays
    /// user-editable and an explicit user edit (tracked per field) is never
    /// overwritten.
    fn run_deploy_preset(&mut self, mode: PresetMode) {
        let Some(owned) = self.gather_preset_inputs() else {
            return;
        };
        let input = PresetInput {
            model_id: &owned.model_id,
            selected_quant: owned.quant.as_deref(),
            weight_files: &owned.weight_files,
            tags: &owned.tags,
            card_markdown: owned.card.as_deref(),
            installed_backends: &owned.installed,
            configured_default: self.deploy_backend,
            has_gpu: owned.has_gpu,
        };

        // Decide the effective backend for this format under the given mode.
        let fmt =
            classify_weight_format_for(&owned.model_id, input.selected_quant, input.weight_files);
        let backend = match mode {
            PresetMode::Fresh => recommend_backend_for(
                fmt,
                &owned.model_id,
                &owned.tags,
                owned.card.as_deref(),
                &owned.installed,
                self.deploy_backend,
            ),
            PresetMode::QuantChanged if !backend_supports(self.deploy_backend, fmt) => {
                recommend_backend_for(
                    fmt,
                    &owned.model_id,
                    &owned.tags,
                    owned.card.as_deref(),
                    &owned.installed,
                    self.deploy_backend,
                )
            }
            PresetMode::QuantChanged | PresetMode::BackendPinned => self.deploy_backend,
        };
        let preset = preset_for_backend(backend, &input);

        // Adopt the backend if the preset changed it; its command override was
        // built for the previous backend and is no longer valid.
        if self.deploy_backend != preset.backend {
            self.deploy_backend = preset.backend;
            self.deploy_command_override = None;
        }

        // Numeric knobs: fill only what the user hasn't set. Extra CLI flags are
        // card-owned for the chosen backend, so replace them wholesale (the old
        // set may belong to a different backend).
        if self.deploy_gpu_frac.is_none() {
            self.deploy_gpu_frac = preset.tuning.gpu_memory_fraction;
        }
        if self.deploy_tensor_parallel.is_none() {
            self.deploy_tensor_parallel = preset.tuning.tensor_parallel;
        }
        if self.deploy_gpu_layers.is_none() {
            self.deploy_gpu_layers = preset.tuning.gpu_layers;
        }
        self.deploy_extra_args = preset.tuning.extra_args.clone();

        // Context: honor an explicit user value; else use the model-native one,
        // reduced until the VRAM estimate fits.
        if !self.deploy_ctx_user_set {
            if let Some(desired) = preset.desired_context {
                self.deploy_ctx = self.fit_context(desired, owned.weight_bytes);
            }
        }

        self.deploy_hints_notes = preset.notes.clone();
        self.refresh_fit();
        if !self.deploy_hints_notes.is_empty() {
            self.set_status(
                format!("Assistant preset: {}", self.deploy_hints_notes.join("; ")),
                false,
            );
        }
    }

    /// Collect the owned inputs the pure preset needs, decoupled from `self` so
    /// the caller can mutate deploy state afterwards without borrow conflicts.
    fn gather_preset_inputs(&self) -> Option<PresetInputsOwned> {
        let detail = self.model_detail.as_ref()?;
        let group = detail
            .quants
            .iter()
            .find(|q| Some(q.label.as_str()) == self.selected_quant.as_deref());
        Some(PresetInputsOwned {
            model_id: detail.summary.id.clone(),
            quant: self.selected_quant.clone(),
            weight_files: group
                .map(|q| q.files.iter().map(|f| f.filename.clone()).collect())
                .unwrap_or_default(),
            weight_bytes: group.map(|q| q.total_size).unwrap_or(0),
            tags: detail.summary.tags.clone(),
            card: detail.card_markdown.clone(),
            installed: self
                .backend_reports
                .iter()
                .filter(|r| r.installed)
                .map(|r| r.kind)
                .collect(),
            has_gpu: self.gpu.total_vram() > 0,
        })
    }

    /// The largest context ≤ `desired` whose predicted VRAM fits total memory,
    /// for the current backend / quant selection.
    fn fit_context(&self, desired: u32, weight_bytes: u64) -> u32 {
        fit_context_to_vram(
            &self.gpu,
            self.deploy_backend,
            self.selected_quant.as_deref(),
            desired,
            weight_bytes,
        )
    }

    fn spawn_deploy(&mut self, req: DeployRequest) {
        if self.deploy_busy.is_some() {
            self.set_status("A deploy is already running (click Cancel to stop it)", false);
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
        // Prefer the local accept/install flow when Bonsai is not ready.
        if !self.local_assistant_ready
            || self.config.assistant.local_preference != LocalAssistantPreference::Accepted
        {
            self.offer_local_assistant_accept();
            return;
        }
        if !self.assistant_configured {
            self.set_status(
                "Assistant not configured — accept the local Bonsai install (Settings or /assistant) \
                 or set OPENROUTER_API_KEY",
                true,
            );
            return;
        }
        if self.fg_busy() {
            self.set_status("Busy — Esc to cancel first", false);
            return;
        }
        self.spawn_assistant_turn(false);
    }

    /// Prompt the user to accept (and install) the local Bonsai assistant.
    /// Used by `/assistant`, first-run offer, and Settings → Accept.
    fn offer_local_assistant_accept(&mut self) {
        if self.assistant_install_busy.is_some() {
            self.set_status("Assistant install already running", false);
            return;
        }
        let need = install_need(&self.config, &self.paths);
        if need == InstallNeed::Ready {
            // Weights + Prism binary already present — mark accepted and warm-start.
            self.config.assistant.local_preference = LocalAssistantPreference::Accepted;
            let _ = self.config.save(&self.paths);
            self.local_assistant_ready = is_installed(&self.config, &self.paths);
            self.assistant_configured = true;
            self.set_status(
                format!("{ASSISTANT_DISPLAY_NAME} accepted — starting local server…"),
                false,
            );
            self.warm_start_local_assistant();
            return;
        }
        self.modal = Some(ModalState::confirm(
            format!("Accept local {ASSISTANT_DISPLAY_NAME} assistant?"),
            format!(
                "{}\n\n{}",
                install_offer_body(&need),
                quant_compatibility_note()
            ),
            ConfirmAction::InstallLocalAssistant,
        ));
    }

    /// Background auto-handle after an error (no modal until the reply arrives).
    fn start_assistant_auto(&mut self) {
        if self.fg_busy() || !self.local_assistant_ready {
            return;
        }
        self.spawn_assistant_turn(true);
    }

    /// Re-scan the models cache for downloaded weights (cheap; sync).
    fn refresh_downloaded(&mut self) {
        self.downloaded_models = list_downloaded(&self.paths.models_cache);
        self.refresh_active_downloads();
    }

    /// Rescan the on-disk download state files. Completed downloads are consumed
    /// (state file cleared, on-disk list refreshed, one notification) so they
    /// stop showing as "in progress" and the model reads as downloaded.
    fn refresh_active_downloads(&mut self) {
        let mut active = Vec::new();
        let mut completed = Vec::new();
        for st in scan_active(&self.paths.models_cache) {
            if st.status == DownloadStatus::Completed {
                clear_state(&model_dir(&self.paths.models_cache, &st.model_id));
                completed.push(st.model_id);
            } else {
                active.push(st);
            }
        }
        self.active_downloads = active;
        if !completed.is_empty() {
            self.downloaded_models = list_downloaded(&self.paths.models_cache);
            self.set_status(
                format!("✓ Download complete: {} — ready to deploy", completed.join(", ")),
                false,
            );
        }
    }

    /// The active background download for a model id, if any.
    pub fn download_for(&self, model_id: &str) -> Option<&DownloadState> {
        self.active_downloads.iter().find(|d| d.model_id == model_id)
    }

    /// Periodic tick (called from the event loop, throttled): rescan progress and
    /// auto-resume any download whose worker died (stale). A live detached worker
    /// keeps its state fresh and is left running — this only revives crashes,
    /// reboots, and app exits that killed the worker.
    fn poll_downloads(&mut self) {
        if self.last_download_poll.elapsed() < Duration::from_millis(1000) {
            return;
        }
        self.last_download_poll = Instant::now();
        self.refresh_active_downloads();
        self.resume_stale_downloads();
    }

    /// Spawn (or re-spawn) a detached worker for a download whose worker is no
    /// longer running. Failed downloads are left for an explicit user resume so a
    /// permanently-failing fetch (gated model, bad url) can't spin-loop.
    fn resume_stale_downloads(&mut self) {
        let now = now_unix();
        let stale: Vec<DownloadState> = self
            .active_downloads
            .iter()
            .filter(|d| d.status == DownloadStatus::Downloading && d.is_stale(now))
            .cloned()
            .collect();
        let mut resumed = 0;
        for st in stale {
            if self.spawn_download_worker(&st) {
                resumed += 1;
            }
        }
        if resumed > 0 {
            self.set_status(
                format!("Resuming {resumed} interrupted download(s) in the background"),
                false,
            );
            self.refresh_active_downloads();
        }
    }

    /// Explicit user resume of a stalled or failed download (from the panel's
    /// resume button).
    fn resume_download(&mut self, model_id: &str) {
        let Some(st) = self.download_for(model_id).cloned() else {
            return;
        };
        if self.spawn_download_worker(&st) {
            self.set_status(format!("Resuming download of {model_id}"), false);
            self.refresh_active_downloads();
        }
    }

    /// Launch the detached `localcode download` worker for `st` and touch its
    /// state file so the next poll doesn't see it as stale before the worker's
    /// first heartbeat (which would double-spawn). Returns whether it launched.
    fn spawn_download_worker(&mut self, st: &DownloadState) -> bool {
        let exe = match std::env::current_exe() {
            Ok(e) => e,
            Err(e) => {
                self.raise_error(
                    LocalCodeError::new(ErrorCode::Internal, e.to_string())
                        .with_cause("Could not locate the localcode binary to run the download worker"),
                );
                return false;
            }
        };
        match spawn_detached_worker(&exe, &st.model_id, st.quantization.as_deref(), st.backend) {
            Ok(pid) => {
                let mut fresh = st.clone();
                fresh.pid = pid;
                fresh.status = DownloadStatus::Downloading;
                fresh.message = "Resuming…".into();
                fresh.updated_unix = now_unix();
                write_state(&model_dir(&self.paths.models_cache, &st.model_id), &fresh);
                true
            }
            Err(e) => {
                self.raise_error(e);
                false
            }
        }
    }

    /// Start a background, resumable, exit-surviving download for a deploy whose
    /// weights aren't on disk yet: write the initial state file, spawn the
    /// detached worker, and let the deploy panel track it. Falls back to a
    /// foreground deploy (still resumable) if the worker can't be launched.
    fn start_background_download(&mut self, req: &DeployRequest) {
        let dir = model_dir(&self.paths.models_cache, &req.model_id);
        if std::fs::create_dir_all(&dir).is_err() {
            self.spawn_deploy(req.clone());
            return;
        }
        let mut state = DownloadState {
            model_id: req.model_id.clone(),
            quantization: req.quantization.clone(),
            backend: req.backend,
            files: req.weight_files.clone(),
            download_urls: req.download_urls.clone(),
            total_bytes: req.weight_bytes,
            downloaded_bytes: 0,
            status: DownloadStatus::Downloading,
            message: "Queued".into(),
            primary_file: None,
            pid: 0,
            updated_unix: now_unix(),
        };
        write_state(&dir, &state);

        let exe = std::env::current_exe().ok();
        let spawned = exe.and_then(|exe| {
            spawn_detached_worker(&exe, &req.model_id, req.quantization.as_deref(), req.backend).ok()
        });
        match spawned {
            Some(pid) => {
                state.pid = pid;
                state.message = "Downloading in the background".into();
                state.updated_unix = now_unix();
                write_state(&dir, &state);
                self.set_status(
                    format!(
                        "Downloading {} in the background — safe to quit; it resumes on its own. Deploy once it finishes.",
                        req.model_id
                    ),
                    false,
                );
                self.refresh_active_downloads();
            }
            None => {
                // No detached worker (e.g. current_exe failed) — don't strand the
                // user; do the download inline via the normal deploy pipeline.
                clear_state(&dir);
                self.set_status(
                    "Background worker unavailable — downloading in the foreground",
                    false,
                );
                self.spawn_deploy(req.clone());
            }
        }
    }

    /// True when a model's weights are on disk (drives the /models "downloaded"
    /// tag). Matches by the deterministic sanitized cache-directory name.
    pub fn is_downloaded(&self, model_id: &str) -> bool {
        self.downloaded_entry(model_id).is_some()
    }

    /// The downloaded-weights entry for a model, matched by exact id or by the
    /// deterministic sanitized cache-directory name.
    fn downloaded_entry(&self, model_id: &str) -> Option<&DownloadedModel> {
        let dir = sanitize_model_dir(model_id);
        self.downloaded_models
            .iter()
            .find(|m| m.model_id == model_id || sanitize_model_dir(&m.model_id) == dir)
    }

    /// True when every file of a quant group is already on disk for `model_id`.
    /// Drives the per-quant "downloaded" tag in the model detail view. Compares
    /// bare file names because the downloader flattens repo subdirectories.
    pub fn quant_downloaded(&self, model_id: &str, group: &QuantGroup) -> bool {
        let Some(m) = self.downloaded_entry(model_id) else {
            return false;
        };
        !group.files.is_empty()
            && group.files.iter().all(|f| {
                let name = std::path::Path::new(&f.filename)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(f.filename.as_str());
                m.files.iter().any(|have| have == name)
            })
    }

    /// Absolute path to the primary on-disk weight file when every file of the
    /// selected quant is already downloaded (first GGUF in sorted order, else
    /// first file — mirroring the deploy downloader's pick). `None` when any
    /// file is missing, so a partial download still goes through the normal
    /// download path.
    fn downloaded_quant_path(&self, model_id: &str, weight_files: &[String]) -> Option<String> {
        if weight_files.is_empty() {
            return None;
        }
        let m = self.downloaded_entry(model_id)?;
        let mut names: Vec<&str> = Vec::with_capacity(weight_files.len());
        for f in weight_files {
            let name = std::path::Path::new(f)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(f.as_str());
            let have = m.files.iter().find(|have| have.as_str() == name)?;
            names.push(have.as_str());
        }
        names.sort_unstable();
        let primary = names
            .iter()
            .find(|n| n.to_lowercase().ends_with(".gguf"))
            .or_else(|| names.first())?;
        Some(m.dir.join(primary).display().to_string())
    }

    /// The model currently in focus in `/models`: the loaded detail, else the
    /// highlighted search result. Used by `/favorite` with no argument.
    fn current_model_id(&self) -> Option<String> {
        if let Some(d) = &self.model_detail {
            return Some(d.summary.id.clone());
        }
        self.models.get(self.model_selected).map(|m| m.id.clone())
    }

    /// Star / unstar a model and persist the change.
    fn toggle_favorite_model(&mut self, id: &str) {
        let now_fav = self.config.toggle_favorite(id);
        if let Err(e) = self.config.save(&self.paths) {
            self.raise_error(e);
            return;
        }
        self.set_status(
            if now_fav {
                format!("★ Favourited {id} — shows first on /dash")
            } else {
                format!("☆ Unfavourited {id}")
            },
            false,
        );
    }

    /// Launch OpenWebUI (`/ui`) in the background, wired to the deployed models.
    fn start_openwebui(&mut self) {
        if self.openwebui_busy {
            self.set_status("OpenWebUI is already starting…", false);
            return;
        }
        if !OpenWebUi::docker_available() {
            self.raise_error(
                LocalCodeError::new(
                    ErrorCode::BackendBinaryMissing,
                    "Docker is required to run OpenWebUI (/ui)",
                )
                .with_hint("Install Docker Desktop (Windows/macOS) or Docker Engine (Linux)")
                .with_hint("Or run it yourself: pip install open-webui && open-webui serve"),
            );
            return;
        }
        let urls = self.openwebui_desired_urls();
        let n = urls.len();
        self.spawn_openwebui_deploy(urls);
        self.set_status(
            format!("Deploying OpenWebUI (Docker) wired to {n} model endpoint(s)…"),
            false,
        );
    }

    /// The OpenAI-compatible endpoints OpenWebUI should be wired to: every model
    /// LocalCode is hosting right now (local assistant + deploys + remote).
    fn openwebui_desired_urls(&self) -> Vec<String> {
        self.all_runtimes()
            .iter()
            .map(|r| r.base_url.clone())
            .collect()
    }

    /// Spawn the (re)deploy of the OpenWebUI container wired to `urls`. The
    /// container has a fixed name, so this replaces any previous instance.
    fn spawn_openwebui_deploy(&mut self, urls: Vec<String>) {
        self.openwebui_busy = true;
        let tx = self.bg_tx.clone();
        tokio::spawn(async move {
            let result = OpenWebUi::deploy(OPENWEBUI_DEFAULT_PORT, urls)
                .await
                .map(Box::new);
            let _ = tx.send(BgMsg::OpenWebUiReady(result));
        });
    }

    /// Keep a running OpenWebUI wired to the current model set. When the runtimes
    /// change — a deploy finishes, a model is stopped, the local assistant comes
    /// online — re-launch the container so its model list matches what LocalCode
    /// is hosting. No-op when nothing is running, a (re)deploy is already in
    /// flight, or the endpoint set is unchanged.
    fn maybe_resync_openwebui(&mut self) {
        if self.openwebui_busy {
            return;
        }
        // Owned copy so the immutable borrow of `self.openwebui` is released
        // before the mutable (re)deploy call below.
        let prev = match self.openwebui.as_ref() {
            Some(h) => h.model_urls().to_vec(),
            None => return,
        };
        let desired = self.openwebui_desired_urls();
        let norm = |mut v: Vec<String>| {
            v.sort();
            v.dedup();
            v
        };
        if norm(desired.clone()) == norm(prev) {
            return;
        }
        let n = desired.len();
        self.spawn_openwebui_deploy(desired);
        self.set_status(
            format!("Syncing OpenWebUI with {n} model endpoint(s)…"),
            false,
        );
    }

    /// Build the backend-management capability handed to the assistant. Wired to
    /// the live registry so an assistant deploy shows up on `/dash` immediately.
    fn model_ops(&self) -> Arc<dyn ModelActions> {
        Arc::new(ModelOps {
            registry: self.registry.clone(),
            events: self.events.clone(),
            gpu: self.gpu.clone(),
            paths: self.paths.clone(),
            config: self.config.clone(),
            tx: self.bg_tx.clone(),
        })
    }

    fn spawn_assistant_turn(&mut self, auto: bool) {
        let assistant = Assistant::new(self.config.assistant.clone());
        let api_key = self.config.assistant_api_key();
        let error_context = self.last_error.as_ref().map(|e| e.assistant_context());
        let user_message = self
            .last_error
            .as_ref()
            .map(|e| {
                if auto {
                    format!(
                        "An error just occurred in LocalCode. Diagnose and fix it if possible \
                         (use tools: shell, doctor.snapshot, fs). Error: {}",
                        e.message
                    )
                } else {
                    format!("Help me fix: {}", e.message)
                }
            })
            .unwrap_or_else(|| {
                if auto {
                    "Help me diagnose LocalCode setup.".into()
                } else {
                    "How can you help me with LocalCode right now?".into()
                }
            });
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
            "assistant_local": self.local_assistant_ready,
        });
        // Coding chat excerpt so the assistant has access to recent conversation.
        let chat_excerpt = {
            let lines: Vec<String> = self
                .coding_transcript
                .iter()
                .rev()
                .take(24)
                .map(|e| format!("{:?}: {}", e.kind, e.text.chars().take(400).collect::<String>()))
                .collect();
            let mut lines = lines;
            lines.reverse();
            if lines.is_empty() {
                None
            } else {
                Some(lines.join("\n"))
            }
        };
        let deploy_model_id = self.model_detail.as_ref().map(|d| d.summary.id.clone());
        let deploy_backend = Some(self.deploy_backend.as_str().to_string());
        let model_card_markdown = self
            .model_detail
            .as_ref()
            .and_then(|d| d.card_markdown.clone());
        let workspace = self.workspace_path();
        let hf_arc = self.hf.clone().map(Arc::new);
        // Interactive turns can deploy/stop/delete models on the user's behalf;
        // silent auto-repair only advises (no unattended model management).
        let model_ops: Option<Arc<dyn ModelActions>> =
            if auto { None } else { Some(self.model_ops()) };
        let tx = self.bg_tx.clone();
        let handle = tokio::spawn(async move {
            let logs =
                localcode_log::read_recent_logs(&paths.log_dir, 80, correlation.as_deref(), redact)
                    .ok();
            let doctor = run_doctor(&paths, &config).await;
            let ctx = AssistantContext {
                user_message: user_message.clone(),
                error_context,
                doctor_report: Some(doctor),
                recent_logs: logs,
                config_snapshot_redacted: Some(config_snapshot),
                chat_excerpt,
                deploy_model_id,
                deploy_backend,
                model_card_markdown,
            };
            // Interactive /assistant gets a tool approver; background auto-repair
            // refuses gated tools (no silent destructive shell).
            let approver = if auto {
                None
            } else {
                Some(ChannelApprover { tx: tx.clone() })
            };
            let result = assistant
                .ask_with_context(
                    &config,
                    &paths,
                    ctx,
                    api_key.as_deref(),
                    hf_arc,
                    model_ops,
                    workspace,
                    None,
                    approver
                        .as_ref()
                        .map(|a| a as &dyn localcode_agent::ToolApprover),
                )
                .await
                .map(|r| r.message);
            if auto {
                let _ = tx.send(BgMsg::AssistantAutoDone(result));
            } else {
                let _ = tx.send(BgMsg::AssistantDone(result));
            }
        });
        let label = if auto {
            format!("{ASSISTANT_DISPLAY_NAME} diagnosing error…")
        } else {
            "Asking assistant".into()
        };
        self.begin_busy(BusyKind::Assistant, label, handle);
    }

    /// Persist accept/decline and kick off the llama.cpp + GGUF install.
    fn start_install_local_assistant(&mut self) {
        self.config.assistant.local_preference = LocalAssistantPreference::Accepted;
        let _ = self.config.save(&self.paths);
        if self.assistant_install_busy.is_some() {
            self.set_status("Assistant install already running", false);
            return;
        }
        let paths = self.paths.clone();
        let config = self.config.clone();
        let tx = self.bg_tx.clone();
        let (p_tx, mut p_rx) = mpsc::unbounded_channel::<String>();
        let progress_tx = tx.clone();
        tokio::spawn(async move {
            while let Some(line) = p_rx.recv().await {
                let _ = progress_tx.send(BgMsg::AssistantInstallProgress(line));
            }
        });
        let handle = tokio::spawn(async move {
            let result = install_local_assistant(&config, &paths, p_tx).await;
            let _ = tx.send(BgMsg::AssistantInstallDone(result));
        });
        self.assistant_install_busy = Some(Busy {
            kind: BusyKind::Install,
            label: format!("Installing {ASSISTANT_DISPLAY_NAME}"),
            started: Instant::now(),
            handle,
        });
        self.assistant_install_progress = "Starting…".into();
        self.set_status(format!("Installing local {ASSISTANT_DISPLAY_NAME} assistant…"), false);
    }

    fn decline_local_assistant(&mut self) {
        self.config.assistant.local_preference = LocalAssistantPreference::Declined;
        let _ = self.config.save(&self.paths);
        self.set_status(
            "Local assistant declined — accept later with /assistant or Settings → Accept local assistant",
            false,
        );
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

        // Prefer an already-selected/deployed runtime; otherwise the local Bonsai
        // assistant is the default conversation model (no /assistant needed).
        let existing = self.active_runtime();
        let use_local_assistant = existing.is_none()
            && self.local_assistant_ready
            && self.config.assistant.prefer_local;
        let use_cloud_fallback = existing.is_none()
            && !use_local_assistant
            && self.config.agent.allow_cloud_fallback
            && self.assistant_configured;

        if existing.is_none() && !use_local_assistant && !use_cloud_fallback {
            let hint = if self.config.assistant.local_preference
                == LocalAssistantPreference::Declined
            {
                "no runtime. Deploy a model from the Models tab, or run /assistant install for the local Bonsai assistant."
            } else if !self.local_assistant_ready {
                "no runtime. Accept the local Bonsai install offer, deploy a model from Models, or set agent.allow_cloud_fallback=true."
            } else {
                "no runtime. Starting the local assistant — try again in a moment, or deploy a model from Models."
            };
            self.coding_transcript
                .push(TranscriptEntry::new(EntryKind::System, hint));
            self.coding_follow = true;
            if self.local_assistant_ready {
                self.warm_start_local_assistant();
                self.set_status("Starting local assistant…", false);
            } else {
                self.set_status("No runtime — install the assistant or deploy a model", true);
            }
            return;
        }

        self.coding_history.push(input.clone());
        if self.coding_history.len() > MAX_INPUT_HISTORY {
            self.coding_history.remove(0);
        }
        self.coding_hist_idx = None;
        self.coding_input.clear();
        self.coding_cursor = 0;
        self.coding_transcript
            .push(TranscriptEntry::new(EntryKind::You, input.clone()));
        // '@path' references: the transcript shows what was typed; the agent
        // receives the prompt with each referenced file's contents appended.
        let (prompt, attached) = self.expand_at_context(&input);
        if !attached.is_empty() {
            self.coding_transcript.push(TranscriptEntry::new(
                EntryKind::System,
                format!("attached: {}", attached.join(", ")),
            ));
        }
        self.coding_follow = true;

        // The session this turn belongs to. Tagging events with it lets the user
        // switch to another chat mid-turn without the background turn's tokens
        // landing in the foreground transcript.
        let sid = self.current_session_id.clone();

        // Live events (streamed tokens, tool activity) flow through their own
        // channel; the forwarder exits when the turn task drops the sender.
        let (ev_tx, mut ev_rx) = mpsc::unbounded_channel::<AgentEvent>();
        let fwd_tx = self.bg_tx.clone();
        let fwd_sid = sid.clone();
        tokio::spawn(async move {
            while let Some(ev) = ev_rx.recv().await {
                if fwd_tx
                    .send(BgMsg::CodingEvent {
                        session_id: fwd_sid.clone(),
                        ev,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });

        let agent_cfg = self.config.agent.clone();
        let session = self.session.clone();
        let session_store = self.session_store.clone();
        let tx = self.bg_tx.clone();
        let approver_tx = self.bg_tx.clone();
        let coding_http = self.coding_http.clone();
        let hf_arc = self.hf.clone().map(std::sync::Arc::new);
        // Chat can manage models (deploy/stop/list/delete/ui) no matter which
        // runtime serves it — this is what makes the default chat the
        // assistant chat. Mutating calls still gate through the approver.
        let model_ops = self.model_ops();
        let paths = self.paths.clone();
        let config = self.config.clone();
        let cloud_base = self.config.assistant.base_url.clone();
        let cloud_model = self.config.assistant.model.clone();
        let cloud_key = self.config.assistant_api_key();
        let done_sid = sid.clone();
        let handle = tokio::spawn(async move {
            // Resolve runtime inside the task so we can ensure_running the local assistant.
            let runtime = if let Some(r) = existing {
                r
            } else if use_local_assistant {
                match ensure_running(&config, &paths).await {
                    Ok(rt) => {
                        // Publish handle back to the UI for subsequent turns.
                        let snap = rt.as_active_runtime();
                        let _ = tx.send(BgMsg::AssistantRuntimeReady(Ok(rt)));
                        snap
                    }
                    Err(e) => {
                        let _ = tx.send(BgMsg::CodingDone {
                            session_id: done_sid.clone(),
                            result: Err(e),
                        });
                        return;
                    }
                }
            } else {
                // Hosted fallback (opt-in).
                let mut r = ActiveRuntime::new(
                    "assistant-provider",
                    localcode_core::runtime::RuntimeKind::OpenAiCompatible,
                    cloud_base,
                );
                r.model_id = if cloud_model.is_empty() {
                    Some("openai/gpt-4o-mini".into())
                } else {
                    Some(cloud_model)
                };
                r.api_key = cloud_key.clone();
                r
            };

            let mut agent = CodingAgent::new(agent_cfg)
                .with_http_client(coding_http)
                .with_model_ops(model_ops);
            if let Some(hf) = hf_arc {
                agent = agent.with_hf(hf);
            }
            let approver = ChannelApprover { tx: approver_tx };
            let mut session = session.lock().await;
            let api_key = runtime.api_key.clone();
            let result = agent
                .run_turn(
                    &mut session,
                    &prompt,
                    &runtime,
                    api_key.as_deref(),
                    Some(&approver),
                    Some(&ev_tx),
                )
                .await;
            if let Some(store) = session_store.lock().await.as_mut() {
                if let Err(e) = store.sync(&session) {
                    tracing::warn!(error = %e, "failed to persist coding session");
                }
            }
            drop(session);
            let _ = tx.send(BgMsg::CodingDone {
                session_id: done_sid.clone(),
                result,
            });
        });
        self.begin_busy(BusyKind::Coding, "Agent working", handle);
    }

    fn new_coding_session(&mut self) {
        // A non-coding foreground task still blocks; a coding turn is detached to
        // the background so starting a new chat never loses the running one.
        if self.fg_busy() && !self.busy.as_ref().is_some_and(|b| b.kind == BusyKind::Coding) {
            self.set_status("Busy — Esc to cancel first", false);
            return;
        }
        self.detach_current_chat_if_running();
        let workspace = self.workspace_path();
        let session = AgentSession::new(workspace);
        let store = if self.config.agent.sessions_enabled {
            let root = sessions_root(&self.config.agent, &self.paths);
            match SessionStore::create(&root, &session) {
                Ok(s) => Some(s),
                Err(e) => {
                    self.set_status(format!("Session file not created: {}", e.message), true);
                    None
                }
            }
        } else {
            None
        };
        self.current_session_id = session.id.clone();
        self.session = Arc::new(AsyncMutex::new(session));
        self.session_store = Arc::new(AsyncMutex::new(store));
        // Re-walk the workspace next time '@' is used — files likely changed.
        self.workspace_files = None;
        self.coding_transcript = vec![TranscriptEntry::new(
            EntryKind::System,
            "new session started",
        )];
        self.coding_selected = None;
        self.coding_scroll = 0;
        self.coding_follow = true;
        self.ctx_used_tokens = 0;
        self.set_status("New coding session", false);
    }

    // ------------------------------------------------------------------
    // Sessions view (/sessions) — list past chats, resume one
    // ------------------------------------------------------------------

    /// If a coding turn is running for the current session, detach it to the
    /// background so the user can switch away without cancelling it. The task is
    /// already running on its own `Arc` clones and persists to disk on its own —
    /// we just stop treating it as the foreground turn and keep its handle +
    /// transcript snapshot so returning restores the live view. No-op unless a
    /// coding turn is in flight.
    fn detach_current_chat_if_running(&mut self) {
        let is_coding = self
            .busy
            .as_ref()
            .map(|b| b.kind == BusyKind::Coding)
            .unwrap_or(false);
        if !is_coding {
            return;
        }
        let Some(busy) = self.busy.take() else {
            return;
        };
        self.bg_chats.insert(
            self.current_session_id.clone(),
            BgChat {
                handle: busy.handle,
                started: busy.started,
                session: self.session.clone(),
                session_store: self.session_store.clone(),
                transcript: self.coding_transcript.clone(),
                ctx_used: self.ctx_used_tokens,
                done: false,
            },
        );
    }

    /// A background turn finished. Its task already persisted the result to the
    /// session's store; append the reply/error to the retained snapshot so
    /// returning shows it, and notify the user.
    fn finish_bg_chat(&mut self, sid: &str, result: &Result<String, LocalCodeError>) {
        if let Some(chat) = self.bg_chats.get_mut(sid) {
            chat.done = true;
            match result {
                Ok(reply) if !reply.trim().is_empty() => chat
                    .transcript
                    .push(TranscriptEntry::new(EntryKind::Agent, reply.clone())),
                Ok(_) => {}
                Err(e) => chat.transcript.push(TranscriptEntry::new(
                    EntryKind::Error,
                    format!("{}: {}", e.code, e.message),
                )),
            }
        }
        let ok = result.is_ok();
        self.set_status(
            if ok {
                "A background chat finished — open it in /sessions".to_string()
            } else {
                "A background chat failed — open it in /sessions".to_string()
            },
            !ok,
        );
    }

    /// Open the sessions view with a fresh listing for this workspace.
    fn open_sessions(&mut self) {
        if !self.config.agent.sessions_enabled {
            self.set_status(
                "Session persistence is off — enable agent.sessions_enabled to keep past chats",
                true,
            );
            return;
        }
        let root = sessions_root(&self.config.agent, &self.paths);
        self.sessions = list_sessions(&root, &self.workspace_path());
        // Land on the newest chat that isn't the current one, so a bare Enter
        // switches somewhere instead of re-opening the chat you came from.
        self.session_selected = self
            .sessions
            .iter()
            .position(|m| m.id != self.current_session_id)
            .unwrap_or(0);
        self.sessions_scroll = 0;
        self.set_mode(Mode::Sessions);
    }

    /// Switch the live chat to the picked past session. A coding turn running for
    /// the current session is detached to the background (it keeps running and
    /// persisting) rather than blocking the switch, so the user never loses
    /// progress. If the target session has a turn running in the background, its
    /// live state is re-attached instead of a disk reload.
    fn resume_session(&mut self, idx: usize) {
        // A foreground *non-coding* task (search/bench/…) still blocks — only a
        // coding turn is safe to background.
        if self.fg_busy() && !self.busy.as_ref().is_some_and(|b| b.kind == BusyKind::Coding) {
            self.set_status("Busy — Esc to cancel first", false);
            return;
        }
        let Some(meta) = self.sessions.get(idx) else {
            return;
        };
        if meta.id == self.current_session_id {
            self.set_mode(Mode::Chat);
            self.set_status("Already in this chat", false);
            return;
        }
        let target_id = meta.id.clone();
        let meta_path = meta.path.clone();

        // Park the current in-flight turn (if any) so switching doesn't kill it.
        self.detach_current_chat_if_running();

        // If the target's turn is still running in the background, re-attach its
        // live state (same Arcs) instead of loading a stale copy from disk.
        if let Some(chat) = self.bg_chats.get(&target_id) {
            if !chat.done {
                let chat = self.bg_chats.remove(&target_id).unwrap();
                self.current_session_id = target_id;
                self.session = chat.session;
                self.session_store = chat.session_store;
                self.coding_transcript = chat.transcript;
                self.coding_selected = None;
                self.ctx_used_tokens = chat.ctx_used;
                self.workspace_files = None;
                self.coding_scroll = 0;
                self.coding_follow = true;
                self.busy = Some(Busy {
                    kind: BusyKind::Coding,
                    label: "Agent working".into(),
                    started: chat.started,
                    handle: chat.handle,
                });
                self.set_mode(Mode::Chat);
                self.set_status("Resumed a chat that's still working…", false);
                return;
            }
            // Completed in the background — drop the parked entry and load the
            // full (tool-inclusive) history from disk below.
            self.bg_chats.remove(&target_id);
        }

        let loaded = match SessionStore::load(&meta_path) {
            Ok(l) => l,
            Err(e) => {
                self.set_status(format!("Could not open session: {}", e.message), true);
                return;
            }
        };
        // Newest mtime wins the startup auto-resume; make that this session.
        loaded.store.touch();
        let title = display_title(&loaded.session.title).to_string();
        let note = if loaded.warnings.is_empty() {
            format!("Resumed session “{title}”")
        } else {
            format!(
                "Resumed session “{title}” ({} repair note(s))",
                loaded.warnings.len()
            )
        };
        let mut transcript = vec![TranscriptEntry::new(EntryKind::System, note)];
        transcript.extend(transcript_from_session(&loaded.session));
        self.ctx_used_tokens = estimate_session_tokens(&loaded.session);
        self.current_session_id = loaded.session.id.clone();
        self.session = Arc::new(AsyncMutex::new(loaded.session));
        self.session_store = Arc::new(AsyncMutex::new(Some(loaded.store)));
        self.workspace_files = None;
        self.coding_transcript = transcript;
        self.coding_selected = None;
        self.coding_scroll = 0;
        self.coding_follow = true;
        self.set_mode(Mode::Chat);
        self.set_status(format!("Resumed “{title}”"), false);
    }

    /// Wheel over the sessions list moves the selection (rows are one line
    /// each; the draw keeps the selection visible).
    fn scroll_sessions(&mut self, delta: i64) {
        let n = self.sessions.len();
        if n == 0 {
            return;
        }
        let cur = self.session_selected.min(n - 1) as i64;
        self.session_selected = (cur + delta.signum()).clamp(0, n as i64 - 1) as usize;
    }

    fn start_stop_runtime(&mut self) {
        // Layout of all_runtimes(): [local assistant?] + registry + remotes.
        let assistant_offset = usize::from(self.local_assistant.is_some());
        let reg_len = self.runtimes.len();
        let idx = self.runtime_selected;

        if assistant_offset == 1 && idx == 0 {
            // Stop the managed local Bonsai assistant.
            if let Some(rt) = self.local_assistant.take() {
                let name = ASSISTANT_DISPLAY_NAME;
                self.set_status(format!("Stopping {name}…"), false);
                tokio::spawn(async move {
                    rt.stop().await;
                });
                self.set_status(format!("{ASSISTANT_DISPLAY_NAME} stopped"), false);
            }
            return;
        }

        let reg_idx = idx.saturating_sub(assistant_offset);
        if reg_idx < reg_len {
            let Some(rt) = self.runtimes.get(reg_idx) else {
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
            return;
        }

        let remote_idx = reg_idx - reg_len;
        if let Some(name) = self
            .remote_sessions
            .get(remote_idx)
            .map(|s| s.server_name.clone())
        {
            match self
                .config
                .remote
                .servers
                .iter()
                .position(|s| s.name == name)
            {
                Some(cfg_idx) => self.disconnect_remote(cfg_idx),
                None => {
                    Self::spawn_remote_close(self.remote_sessions.remove(remote_idx));
                    self.set_status(
                        format!("Disconnected from {name} — freeing remote GPU"),
                        false,
                    );
                }
            }
        }
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
            BackendKind::Colibri => self.config.backends.colibri.bin = rp.bin,
            BackendKind::ColibriHy3 => self.config.backends.colibri_hy3.bin = rp.bin,
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
        // Take the session OUT and close it on a task: freeing the remote GPU
        // (unloading its resident models) is async, and Drop can't await — a
        // plain `retain`/drop would leave the model loaded on the remote box.
        let Some(pos) = self
            .remote_sessions
            .iter()
            .position(|s| s.server_name == name)
        else {
            return; // wasn't connected
        };
        Self::spawn_remote_close(self.remote_sessions.remove(pos));
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
        self.set_status(format!("Disconnected from {name} — freeing remote GPU"), false);
    }

    /// Move a remote session onto a background task that unloads its resident
    /// models (freeing the remote GPU) and then closes the SSH session + tunnel.
    /// Drop alone can't do this — the unload is async.
    fn spawn_remote_close(session: RemoteSession) {
        tokio::spawn(async move {
            session.close().await;
        });
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
        // The event loop redraws at least every 100ms, so this doubles as the
        // debounce tick for the follow-the-selection detail load.
        self.maybe_autoload_detail();
        while let Ok(msg) = self.bg_rx.try_recv() {
            match msg {
                BgMsg::SearchDone(result) => {
                    self.finish_busy();
                    match result {
                        Ok(models) => {
                            self.models = models;
                            self.model_selected = 0;
                            self.set_status(format!("Found {} models", self.models.len()), false);
                            // Bring up the first result's card without an Enter.
                            self.arm_detail_autoload();
                        }
                        Err(e) => {
                            self.last_failed_action = Some(RetryAction::SearchModels);
                            self.raise_error(e);
                        }
                    }
                }
                BgMsg::DetailDone { id, auto, result } => {
                    // Stale: a newer request replaced this one while it was in
                    // flight (the user scrolled on). Its busy marker — and the
                    // eventual DetailDone — belong to the newer request.
                    if self.detail_request.as_deref() != Some(id.as_str()) {
                        continue;
                    }
                    self.detail_request = None;
                    self.finish_busy();
                    match result {
                        Ok(detail) => {
                            self.selected_quant = detail.quants.first().map(|q| q.label.clone());
                            self.model_detail = Some(detail);
                            self.card_scroll = 0;
                            // Fresh model: clear all previous card-derived params so
                            // the auto-preset re-derives cleanly (nothing bleeds in
                            // from the last model).
                            self.deploy_extra_args.clear();
                            self.deploy_hints_notes.clear();
                            self.deploy_command_override = None;
                            self.deploy_gpu_frac = None;
                            self.deploy_tensor_parallel = None;
                            self.deploy_gpu_layers = None;
                            self.deploy_ctx = DEFAULT_DEPLOY_CTX;
                            self.deploy_ctx_user_set = false;
                            self.refresh_fit();
                            self.set_status(
                                "Model loaded — → focus card, [,/.] quant, [d] deploy",
                                false,
                            );
                            // Assistant reads the card and presets the backend,
                            // context, and flags (overriding the status above with a
                            // summary when it changes anything).
                            self.apply_deploy_preset(PresetMode::Fresh);
                        }
                        Err(e) if auto => {
                            // Scroll-triggered load: report quietly instead of
                            // raising the error modal over the list.
                            self.set_status(format!("Couldn't load {id}: {}", e.message), true);
                        }
                        Err(e) => {
                            self.last_failed_action = Some(RetryAction::LoadDetail);
                            self.raise_error(e);
                        }
                    }
                }
                BgMsg::DeployEnded(outcome) => {
                    self.deploy_busy = None;
                    // A finished deploy may have just downloaded weights —
                    // rescan so "downloaded" tags and /dash reflect them.
                    self.refresh_downloaded();
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
                BgMsg::CodingEvent { session_id, ev } => {
                    // Only the foreground session's tokens render; a backgrounded
                    // turn's events are dropped (its result persists to disk and
                    // its snapshot is restored on return).
                    if session_id == self.current_session_id {
                        self.apply_agent_event(ev);
                        // Keep the context meter moving while the turn streams.
                        self.refresh_ctx_usage();
                    }
                }
                BgMsg::CodingDone { session_id, result } => {
                    // A turn for a backgrounded session finished — record it there
                    // (it already persisted) without touching the foreground.
                    if session_id != self.current_session_id {
                        self.finish_bg_chat(&session_id, &result);
                        continue;
                    }
                    // Turn stats for the status line: duration from the busy
                    // marker (grabbed before finish_busy clears it) + tool
                    // calls made since the user's message.
                    let took = self
                        .busy
                        .as_ref()
                        .filter(|b| b.kind == BusyKind::Coding)
                        .map(|b| b.started.elapsed());
                    self.finish_busy();
                    self.end_stream_rate_window();
                    self.refresh_ctx_usage();
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
                            let tools = self
                                .coding_transcript
                                .iter()
                                .rev()
                                .take_while(|e| e.kind != EntryKind::You)
                                .filter(|e| e.kind == EntryKind::Tool)
                                .count();
                            let mut status = String::from("Agent replied");
                            if let Some(d) = took {
                                status.push_str(&format!(" · {:.1}s", d.as_secs_f64()));
                            }
                            if tools > 0 {
                                status.push_str(&format!(
                                    " · {tools} tool{}",
                                    if tools == 1 { "" } else { "s" }
                                ));
                            }
                            self.set_status(status, false);
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
                    let mode = self.config.agent.approval();
                    self.modal = Some(ModalState::confirm(
                        "Agent asks for approval",
                        format!(
                            "{description}\n\nConfirm to run it in the workspace, Cancel to refuse.\n(approvals: {} — Shift+Tab or /mode to change)",
                            mode.label()
                        ),
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
                            self.coding_transcript.push(TranscriptEntry::new(
                                EntryKind::System,
                                format!("[{ASSISTANT_DISPLAY_NAME}] {message}"),
                            ));
                            self.modal = Some(ModalState::info(
                                format!("{ASSISTANT_DISPLAY_NAME} assistant"),
                                message,
                            ));
                            self.set_status("Assistant replied", false);
                        }
                        Err(e) => {
                            self.last_failed_action = Some(RetryAction::AskAssistant);
                            // Don't re-enter auto-handle loop on assistant failure.
                            let prev = self.config.assistant.auto_handle_errors;
                            self.config.assistant.auto_handle_errors = false;
                            self.raise_error(e);
                            self.config.assistant.auto_handle_errors = prev;
                        }
                    }
                }
                BgMsg::AssistantAutoDone(result) => {
                    self.finish_busy();
                    match result {
                        Ok(message) => {
                            self.coding_transcript.push(TranscriptEntry::new(
                                EntryKind::System,
                                format!("[{ASSISTANT_DISPLAY_NAME} auto] {message}"),
                            ));
                            self.set_status(
                                format!("{ASSISTANT_DISPLAY_NAME} finished diagnosing — see chat"),
                                false,
                            );
                            // Replace error modal with assistant findings when still open.
                            if matches!(
                                self.modal.as_ref().map(|m| &m.kind),
                                Some(ModalKind::Error { .. })
                            ) {
                                self.modal = Some(ModalState::info(
                                    format!("{ASSISTANT_DISPLAY_NAME} diagnosis"),
                                    message,
                                ));
                            }
                        }
                        Err(e) => {
                            // Quiet failure on auto path — the error modal is already shown.
                            self.set_status(
                                format!("Assistant auto-diagnose failed: {}", e.message),
                                true,
                            );
                        }
                    }
                }
                BgMsg::AssistantInstallProgress(line) => {
                    self.assistant_install_progress = line.clone();
                    self.set_status(line, false);
                }
                BgMsg::AssistantInstallDone(result) => {
                    if let Some(b) = self.assistant_install_busy.take() {
                        // Task finished; drop handle.
                        drop(b);
                    }
                    match result {
                        Ok(repoint) => {
                            if let Some(r) = repoint {
                                self.config.backends.llamacpp.bin = r.bin;
                                let _ = self.config.save(&self.paths);
                            }
                            self.local_assistant_ready =
                                is_installed(&self.config, &self.paths);
                            self.assistant_configured = true;
                            self.config.assistant.local_preference =
                                LocalAssistantPreference::Accepted;
                            let _ = self.config.save(&self.paths);
                            self.coding_transcript.push(TranscriptEntry::new(
                                EntryKind::System,
                                startup_greeting(self.local_assistant_ready),
                            ));
                            self.set_status(
                                format!("{ASSISTANT_DISPLAY_NAME} installed and ready"),
                                false,
                            );
                            self.modal = Some(ModalState::info(
                                format!("{ASSISTANT_DISPLAY_NAME} ready"),
                                format!(
                                    "Local assistant is installed and is your default conversation model.\n\n\
                                     Launch: llama-server -m {BONSAI_FILE} -ngl 99\n\n{}\n\n\
                                     Just type in chat — no /assistant needed. It can search Hugging Face, \
                                     read model cards, help deploy models, and fix LocalCode issues.",
                                    quant_compatibility_note()
                                ),
                            ));
                            // Attach the already-running (or warm-start) server as the default runtime.
                            self.warm_start_local_assistant();
                        }
                        Err(e) => {
                            self.last_failed_action =
                                Some(RetryAction::InstallLocalAssistant);
                            let prev = self.config.assistant.auto_handle_errors;
                            self.config.assistant.auto_handle_errors = false;
                            self.raise_error(e);
                            self.config.assistant.auto_handle_errors = prev;
                        }
                    }
                }
                BgMsg::AssistantRuntimeReady(result) => match result {
                    Ok(rt) => {
                        // Keep the first handle that owns the child. Later
                        // ensure_running probes reuse a healthy server with
                        // child=None — replacing would Drop the owner and kill
                        // llama-server mid-conversation.
                        if self.local_assistant.is_none() {
                            self.local_assistant = Some(std::sync::Arc::new(rt));
                        }
                        self.local_assistant_ready = true;
                        self.assistant_configured = true;
                        // Keep selection on the assistant (index 0) when nothing else was selected.
                        if self.runtimes.is_empty() && self.remote_sessions.is_empty() {
                            self.runtime_selected = 0;
                        }
                        self.set_status(
                            format!(
                                "{ASSISTANT_DISPLAY_NAME} ready — default chat uses -m {BONSAI_FILE} -ngl 99"
                            ),
                            false,
                        );
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "local assistant warm-start failed");
                        self.set_status(
                            format!(
                                "Local assistant start failed: {} — try /assistant install",
                                e.message
                            ),
                            true,
                        );
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
                BgMsg::GpuRefreshed(inv) => {
                    // Ignore an empty inventory from a transient probe failure so
                    // a live GPU doesn't blink to "no GPU" mid-session; a real
                    // GPU-less machine was already empty from startup discovery.
                    if !inv.devices.is_empty() {
                        self.gpu = inv;
                    }
                    // Same 1s cadence refreshes the context meter even when the
                    // GPU probe is a no-op (CPU-only host / failed smi).
                    self.refresh_ctx_usage();
                    self.refresh_status_logs();
                }
                BgMsg::RuntimeStopped { result } => match result {
                    Ok(()) => self.set_status("Runtime stopped", false),
                    Err(e) => self.raise_error(e),
                },
                BgMsg::ModelsChanged => {
                    // A tool deployed/stopped/deleted a model. Runtimes refresh on
                    // the next tick; refresh the on-disk set now for /models & /dash.
                    self.refresh_downloaded();
                    self.set_status("Models updated", false);
                }
                BgMsg::OpenWebUiReady(result) => {
                    self.openwebui_busy = false;
                    match result {
                        Ok(handle) => {
                            let url = handle.url().to_string();
                            self.openwebui = Some(*handle);
                            self.set_status(
                                format!("OpenWebUI starting at {url} — open it in ~30s"),
                                false,
                            );
                        }
                        Err(e) => self.raise_error(e),
                    }
                }
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
                BgMsg::LlamaSetupDone(result) => {
                    self.install_progress_line.clear();
                    match result {
                        Ok(bin) => {
                            let bin_str = bin.display().to_string();
                            if self.config.backends.llamacpp.bin != bin_str {
                                self.apply_repoint(Repoint {
                                    kind: BackendKind::LlamaCpp,
                                    bin: bin_str.clone(),
                                });
                            }
                            self.set_status(
                                format!("llama-server ready at {bin_str}"),
                                false,
                            );
                            self.start_detect();
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "startup llama-server setup failed");
                            self.set_status(
                                format!(
                                    "llama-server setup failed: {} — run `localcode setup`",
                                    e.message
                                ),
                                true,
                            );
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
        // Keep a running OpenWebUI wired to the current model set (fires only when
        // the endpoint set actually changed).
        self.maybe_resync_openwebui();
    }

    // ------------------------------------------------------------------
    // Slash command menu (the popup above the omnibar when input starts "/")
    // ------------------------------------------------------------------

    /// The command catalog: `(name, args-hint, description)`. `name` includes
    /// the leading slash. Order is the display order in the menu.
    pub fn slash_catalog() -> &'static [(&'static str, &'static str, &'static str)] {
        &[
            ("/models", "[q]", "search & deploy HuggingFace models"),
            ("/dash", "", "manage running models — logs, stop/start, switch model"),
            ("/runtimes", "", "active runtimes & system overview"),
            ("/remote", "", "connect a GPU server over SSH"),
            ("/backends", "", "install & configure inference backends"),
            ("/bench", "", "run the sample benchmark suite"),
            ("/setup", "", "first-run setup & doctor"),
            ("/settings", "", "preferences & config file"),
            ("/mode", "[always|auto|edits|ask]", "how much the agent asks before running tools"),
            ("/theme", "", "cycle ember / dark / neon / pink / sage"),
            ("/select", "", "release mouse to select & copy text (F2)"),
            ("/chat", "", "back to the conversation"),
            ("/new", "", "start a new conversation"),
            ("/sessions", "", "switch to a past chat (resume)"),
            ("/deploy", "", "deploy the selected model"),
            ("/ui", "", "launch the OpenWebUI browser chat (Docker)"),
            ("/favorite", "[id]", "star/unstar a model (favourites show first on /dash)"),
            ("/doctor", "", "run environment diagnostics"),
            ("/assistant", "[install|accept]", "accept/install local Bonsai (default chat when ready)"),
            ("/update", "", "install the available update"),
            ("/logs", "", "show the log directory path"),
            ("/help", "", "keyboard & mouse help"),
            ("/quit", "", "exit"),
        ]
    }

    /// Menu items filtered by whatever follows the leading slash in the
    /// omnibar, best match first: an exact command name beats a prefix match
    /// beats a substring beats a description hit. Enter runs the first item,
    /// so typing a full command name always runs that command (e.g. `/mode`
    /// must not run `/models`).
    pub fn palette_items(&self) -> Vec<String> {
        let q = self
            .coding_input
            .trim_start_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_lowercase();
        let mut scored: Vec<(u8, String)> = Self::slash_catalog()
            .iter()
            .filter_map(|(name, args, desc)| {
                let n = name[1..].to_lowercase();
                let rank = if q.is_empty() || n == q {
                    0
                } else if n.starts_with(&q) {
                    1
                } else if n.contains(&q) {
                    2
                } else if desc.to_lowercase().contains(&q) {
                    3
                } else {
                    return None;
                };
                let display = if args.is_empty() {
                    format!("{name}  —  {desc}")
                } else {
                    format!("{name} {args}  —  {desc}")
                };
                Some((rank, display))
            })
            .collect();
        // Stable sort: equal ranks keep catalog order.
        scored.sort_by_key(|(rank, _)| *rank);
        scored.into_iter().map(|(_, display)| display).collect()
    }

    /// Is the omnibar currently a slash-command entry?
    pub fn slash_active(&self) -> bool {
        self.coding_input.starts_with('/')
    }

    // ------------------------------------------------------------------
    // '@' context picker (attach workspace files to an agent message)
    // ------------------------------------------------------------------

    /// The '@' token the caret is currently inside, as `(start_char_index,
    /// query_after_the_at)`. A token is the whitespace-delimited word ending at
    /// the caret; it counts only when that word starts with '@'.
    pub fn at_token(&self) -> Option<(usize, String)> {
        if self.slash_active() {
            return None;
        }
        let chars: Vec<char> = self.coding_input.chars().collect();
        let cur = self.coding_cursor.min(chars.len());
        let mut start = cur;
        while start > 0 && !chars[start - 1].is_whitespace() {
            start -= 1;
        }
        let word: String = chars[start..cur].iter().collect();
        word.strip_prefix('@').map(|rest| (start, rest.to_string()))
    }

    /// Is the '@' file picker open? True while the caret sits in an '@' token
    /// that has matches and wasn't just dismissed with Esc.
    pub fn at_picker_active(&mut self) -> bool {
        let Some((_, q)) = self.at_token() else {
            return false;
        };
        if self.at_dismissed.as_deref() == Some(q.as_str()) {
            return false;
        }
        !self.at_matches().is_empty()
    }

    /// Workspace files matching the current '@' query — file-name prefix hits
    /// first, then path substring hits. Empty query lists everything.
    pub fn at_matches(&mut self) -> Vec<String> {
        let q = match self.at_token() {
            Some((_, q)) => q.to_lowercase(),
            None => return Vec::new(),
        };
        let mut scored: Vec<(u8, &String)> = self
            .workspace_file_list()
            .iter()
            .filter_map(|p| {
                let pl = p.to_lowercase();
                let name = pl.rsplit('/').next().unwrap_or(&pl);
                let rank = if q.is_empty() || name.starts_with(&q) {
                    0
                } else if pl.contains(&q) {
                    1
                } else {
                    return None;
                };
                Some((rank, p))
            })
            .collect();
        scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
        scored.into_iter().map(|(_, p)| p.clone()).collect()
    }

    /// Replace the '@' token at the caret with the selected file path (plus a
    /// trailing space) so typing can continue naturally.
    fn at_complete(&mut self) {
        let Some((start, _)) = self.at_token() else {
            return;
        };
        let items = self.at_matches();
        if items.is_empty() {
            return;
        }
        let path = items[self.at_selected.min(items.len() - 1)].clone();
        let chars: Vec<char> = self.coding_input.chars().collect();
        let cur = self.coding_cursor.min(chars.len());
        let mut out: String = chars[..start].iter().collect();
        let insert = format!("@{path} ");
        out.push_str(&insert);
        out.extend(chars[cur..].iter());
        self.coding_input = out;
        self.coding_cursor = start + insert.chars().count();
        self.at_selected = 0;
        self.at_dismissed = None;
    }

    /// The workspace file list backing the '@' picker, walked once and cached.
    /// Skips VCS/build/dependency directories and caps the walk so a giant
    /// workspace can't stall the UI thread.
    fn workspace_file_list(&mut self) -> &[String] {
        if self.workspace_files.is_none() {
            const SKIP_DIRS: [&str; 10] = [
                ".git", ".hg", ".svn", "target", "node_modules", "dist", "build",
                "__pycache__", ".venv", "venv",
            ];
            const MAX_FILES: usize = 2000;
            let root = self.workspace_path();
            let mut out: Vec<String> = Vec::new();
            let mut stack = vec![root.clone()];
            while let Some(dir) = stack.pop() {
                if out.len() >= MAX_FILES {
                    break;
                }
                let Ok(rd) = std::fs::read_dir(&dir) else { continue };
                for entry in rd.flatten() {
                    let path = entry.path();
                    let name = entry.file_name().to_string_lossy().to_string();
                    if path.is_dir() {
                        if !name.starts_with('.') && !SKIP_DIRS.contains(&name.as_str()) {
                            stack.push(path);
                        }
                    } else if out.len() < MAX_FILES {
                        if let Ok(rel) = path.strip_prefix(&root) {
                            out.push(rel.to_string_lossy().replace('\\', "/"));
                        }
                    }
                }
            }
            out.sort();
            self.workspace_files = Some(out);
        }
        self.workspace_files.as_deref().unwrap_or(&[])
    }

    /// Expand `@path` references in a prompt: the agent receives the original
    /// text plus each referenced file's contents in a fenced block. Returns the
    /// expanded prompt and the list of attached paths (for transcript feedback).
    fn expand_at_context(&mut self, input: &str) -> (String, Vec<String>) {
        // Keep attachments within the model's history budget. `trimmed_tail`
        // never splits a single user message, so an attachment larger than the
        // budget would guarantee context overflow on the small local models
        // this targets. Reserve ~3/4 of the budget for attachments, leaving room
        // for the prompt, system, and tool schemas.
        let budget = self.config.agent.max_history_chars;
        let total_chars = if budget == 0 {
            96_000
        } else {
            (budget.saturating_mul(3) / 4).min(96_000)
        };
        let per_file_chars = 24_000.min(total_chars);
        let root = self.workspace_path();
        let mut attached: Vec<String> = Vec::new();
        let mut blocks = String::new();
        for word in input.split_whitespace() {
            let Some(frag) = word.strip_prefix('@').filter(|f| !f.is_empty()) else {
                continue;
            };
            if attached.iter().any(|a| a == frag) {
                continue;
            }
            let path = root.join(frag);
            let Ok(mut content) = std::fs::read_to_string(&path) else {
                continue;
            };
            if content.len() > per_file_chars {
                let mut cut = per_file_chars;
                while !content.is_char_boundary(cut) {
                    cut -= 1;
                }
                content.truncate(cut);
                content.push_str("\n… (truncated)");
            }
            if blocks.len() + content.len() > total_chars {
                break;
            }
            blocks.push_str(&format!("\n--- {frag} ---\n{content}\n"));
            attached.push(frag.to_string());
        }
        if attached.is_empty() {
            (input.to_string(), attached)
        } else {
            (
                format!("{input}\n\nAttached files (referenced with @):\n{blocks}"),
                attached,
            )
        }
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
            "runtimes" | "home" => self.set_mode(Mode::Runtimes),
            "dash" | "dashboard" | "models-running" => self.set_mode(Mode::Dash),
            "deploy" => {
                self.set_mode(Mode::Models);
                self.start_deploy(false);
            }
            "ui" | "openwebui" | "webui" => self.start_openwebui(),
            "favorite" | "favourite" | "fav" | "star" => {
                let id = if rest.is_empty() {
                    self.current_model_id()
                } else {
                    Some(rest.clone())
                };
                match id {
                    Some(id) => self.toggle_favorite_model(&id),
                    None => self.set_status(
                        "Usage: /favorite <model-id> (or select a model in /models first)",
                        true,
                    ),
                }
            }
            "bench" | "benchmark" => self.set_mode(Mode::Bench),
            "setup" => self.set_mode(Mode::Setup),
            "doctor" => {
                self.set_mode(Mode::Setup);
                self.start_doctor();
            }
            "settings" => self.set_mode(Mode::Settings),
            "mode" | "approvals" | "approve" => match args.first() {
                None => self.cycle_approval_mode(),
                Some(name) => match ApprovalMode::parse(&rest) {
                    Some(m) => self.set_approval_mode(m),
                    None => self.set_status(
                        format!("Unknown approval mode: {name} — use always, auto, edits or ask"),
                        true,
                    ),
                },
            },
            "theme" => self.toggle_theme(),
            "select" => self.toggle_select_mode(),
            "chat" => self.set_mode(Mode::Chat),
            "new" => {
                self.set_mode(Mode::Chat);
                self.new_coding_session();
            }
            "sessions" | "resume" | "chats" => self.open_sessions(),
            "assistant" | "ask" => match args.first().map(|s| s.as_str()) {
                // Force install after accept (skips re-prompt only if already accepted).
                Some("install") => {
                    if self.config.assistant.local_preference
                        != LocalAssistantPreference::Accepted
                    {
                        self.offer_local_assistant_accept();
                    } else {
                        self.start_install_local_assistant();
                    }
                }
                // Explicit accept / re-offer.
                Some("accept") => self.offer_local_assistant_accept(),
                // Default: prompt to accept when not ready; otherwise run a help turn.
                _ => {
                    if !self.local_assistant_ready
                        || self.config.assistant.local_preference
                            != LocalAssistantPreference::Accepted
                    {
                        self.offer_local_assistant_accept();
                    } else {
                        self.start_assistant();
                    }
                }
            },
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
                        self.deploy_ctx_user_set = true;
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
                        self.deploy_command_override = None;
                        // Re-derive the card flags / context for the pinned backend.
                        self.apply_deploy_preset(PresetMode::BackendPinned);
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

    /// `/theme` cycles through the shipped themes (dark → neon → pink → sage).
    fn toggle_theme(&mut self) {
        self.set_theme(self.config.ui.theme.next());
    }

    /// Set the agent approval mode (status-bar click, Shift+Tab, `/mode`,
    /// Settings) and persist it. Also neutralizes the legacy
    /// `confirm_destructive_tools` off-switch so an explicit choice always
    /// means what it says (see `AgentConfig::approval`).
    pub(crate) fn set_approval_mode(&mut self, mode: ApprovalMode) {
        self.config.agent.approval_mode = mode;
        self.config.agent.confirm_destructive_tools = true;
        self.set_status(
            format!("Approvals: {} — {}", mode.label(), mode.describe()),
            false,
        );
        if let Err(e) = self.config.save(&self.paths) {
            self.raise_error(e);
        }
    }

    /// Shift+Tab / `/mode` with no argument: next mode in the cycle.
    pub(crate) fn cycle_approval_mode(&mut self) {
        self.set_approval_mode(self.config.agent.approval().next());
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
        let approval = self.config.agent.approval();
        r.push(SettingsRow {
            label: "approvals".into(),
            value: format!("{} — {}", approval.label(), approval.describe()),
            kind: SettingsRowKind::Action("cycle"),
            action: Some(SettingAction::ApprovalCycle),
        });
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
        r.push(toggle(
            "shell sandbox",
            self.config.agent.shell_sandbox,
            SettingAction::ToggleShellSandbox,
        ));
        r.push(toggle(
            "session persistence",
            self.config.agent.sessions_enabled,
            SettingAction::ToggleSessions,
        ));
        r.push(toggle(
            "auto-compact",
            self.config.agent.auto_compact,
            SettingAction::ToggleAutoCompact,
        ));

        // Local Bonsai assistant (accept / prefer / auto-diagnose).
        r.push(SettingsRow::header("assistant — local Bonsai"));
        let pref = self.config.assistant.local_preference;
        let accept_value = match (pref, self.local_assistant_ready) {
            (LocalAssistantPreference::Accepted, true) => {
                "accepted · ready — default chat".to_string()
            }
            (LocalAssistantPreference::Accepted, false) => {
                "accepted · not installed — Enter to install".to_string()
            }
            (LocalAssistantPreference::Declined, _) => {
                "declined — Enter to accept & install".to_string()
            }
            (LocalAssistantPreference::NotPrompted, _) => {
                "not accepted — Enter to accept & install".to_string()
            }
        };
        r.push(SettingsRow {
            label: "accept local assistant".into(),
            value: accept_value,
            kind: SettingsRowKind::Action(match pref {
                LocalAssistantPreference::Accepted if self.local_assistant_ready => "ready",
                LocalAssistantPreference::Accepted => "install",
                _ => "accept",
            }),
            action: Some(SettingAction::AcceptLocalAssistant),
        });
        r.push(toggle(
            "prefer local",
            self.config.assistant.prefer_local,
            SettingAction::TogglePreferLocal,
        ));
        r.push(toggle(
            "auto-handle errors",
            self.config.assistant.auto_handle_errors,
            SettingAction::ToggleAutoHandleErrors,
        ));
        r.push(toggle(
            "auto-preset deploy params",
            self.config.assistant.auto_deploy_hints,
            SettingAction::ToggleAutoDeployHints,
        ));
        r.push(SettingsRow::info(
            "",
            "off = use the deploy panel's auto-config button on demand",
        ));
        r.push(SettingsRow::info(
            "",
            "also: /assistant or /assistant accept · install with /assistant install",
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
            .unwrap_or_else(|| self.paths.data_dir.join("skills").display().to_string());
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
            SettingAction::ApprovalCycle => {
                self.cycle_approval_mode();
                return; // cycle_approval_mode already saved
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
            SettingAction::ToggleShellSandbox => {
                self.config.agent.shell_sandbox = !self.config.agent.shell_sandbox;
            }
            SettingAction::ToggleSessions => {
                self.config.agent.sessions_enabled = !self.config.agent.sessions_enabled;
            }
            SettingAction::ToggleAutoCompact => {
                self.config.agent.auto_compact = !self.config.agent.auto_compact;
            }
            SettingAction::AcceptLocalAssistant => {
                // Opens confirm / starts install; do not double-save here.
                self.offer_local_assistant_accept();
                return;
            }
            SettingAction::TogglePreferLocal => {
                self.config.assistant.prefer_local = !self.config.assistant.prefer_local;
                // A routing preference, not a "start now" switch: the local
                // assistant still loads lazily on the next message.
                self.set_status(
                    if self.config.assistant.prefer_local {
                        "Prefer local Bonsai when ready — starts on your next message"
                    } else {
                        "Local Bonsai not preferred for default chat"
                    },
                    false,
                );
            }
            SettingAction::ToggleAutoHandleErrors => {
                self.config.assistant.auto_handle_errors =
                    !self.config.assistant.auto_handle_errors;
            }
            SettingAction::ToggleAutoDeployHints => {
                self.config.assistant.auto_deploy_hints =
                    !self.config.assistant.auto_deploy_hints;
                self.set_status(
                    if self.config.assistant.auto_deploy_hints {
                        "Auto-preset on — models configure their deploy params as they load"
                    } else {
                        "Auto-preset off — use the deploy panel's auto-config button"
                    },
                    false,
                );
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
            SettingField::Workspace => {
                self.config.agent.workspace_root = opt;
                // The '@' picker's file cache is rooted at the workspace.
                self.workspace_files = None;
            }
        }
        self.refresh_agent_summaries();
        match self.config.save(&self.paths) {
            Ok(()) => self.set_status("Saved", false),
            Err(e) => self.raise_error(e),
        }
    }

    /// Rebuild the discovered-skills list from the configured skills dir.
    fn reload_skills_list(&mut self) {
        let probe = CodingAgent::new(self.config.agent.clone());
        self.skills = probe.skills.list().to_vec();
        self.refresh_agent_summaries();
    }

    /// Recompute the cached skill summary from current config.
    fn refresh_agent_summaries(&mut self) {
        let probe = CodingAgent::new(self.config.agent.clone());
        self.skills = probe.skills.list().to_vec();
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
        self.refresh_status_logs();
        let dir = self.paths.log_dir.display().to_string();
        let body = if self.status_logs.is_empty() {
            format!("{dir}\n\n(no log entries yet)")
        } else {
            format!("{dir}\n\n{}", self.status_logs.join("\n"))
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
  Shift+Tab                cycle agent approvals (always/auto/edits/ask)\n\
  Ctrl+S save    Ctrl+K command entry    Ctrl+C quit\n\
\n\
Commands (type / to see them all):\n\
  /models [q]   search & deploy HuggingFace models\n\
  /dash         manage running models — logs, stop/start, switch model\n\
  /runtimes     active runtimes & system overview\n\
  /remote       connect a GPU server over SSH (one-click)\n\
  /backends     install inference backends\n\
  /mode [m]     how much the agent asks before running tools\n\
  /bench /setup /settings /theme /chat /new /deploy /update /logs /quit\n\
\n\
Everything renders inline in the working area — no popups. Rows, fields\n\
and buttons are clickable. Models: click a model to open it, click a quant\n\
to select, click deploy. Dash: run several models at once, click a card (or\n\
↑/↓) to pick which one the next request uses, copy its launch command, read\n\
its backend logs, or stop it. Remote: click a field to edit, then connect."
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

    /// Stop everything we spawned before the process exits. Runs on every quit
    /// path — the quit dialog, `q`, and Ctrl+C — so model servers don't linger
    /// as orphaned processes holding VRAM (the exact bug users hit on Ctrl+C).
    /// A deploy still in flight owns its child directly, so aborting the task
    /// drops the `Command` and `kill_on_drop` reaps it; already-registered
    /// runtimes are killed through the backend `stop` path. Ollama is a shared
    /// service (its stop is a deliberate no-op) and remote SSH runtimes live on
    /// their own host, so both are correctly left untouched here.
    pub async fn shutdown(&mut self) {
        if let Some(b) = self.deploy_busy.take() {
            b.handle.abort();
        }
        // Detached background coding turns end with the process; abort them so
        // they don't keep hitting a runtime we're about to stop.
        for (_, chat) in self.bg_chats.drain() {
            chat.handle.abort();
        }
        self.registry.stop_all().await;
        self.runtimes.clear();
    }

    /// Whether a coding turn is currently running for session `id` — either in
    /// the foreground or detached to the background. Drives the `/sessions`
    /// "working" marker.
    pub fn session_is_working(&self, id: &str) -> bool {
        if id == self.current_session_id
            && self.busy.as_ref().is_some_and(|b| b.kind == BusyKind::Coding)
        {
            return true;
        }
        self.bg_chats.get(id).is_some_and(|c| !c.done)
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

        // An in-place deploy-parameter edit captures input.
        if self.mode == Mode::Models && self.deploy_editing.is_some() {
            self.handle_deploy_field_key(key);
            return;
        }

        // The '@' file picker owns navigation and completion while open; every
        // other key falls through and keeps editing the omnibar text.
        if self.at_picker_active() {
            match key.code {
                KeyCode::Esc => {
                    self.at_dismissed = self.at_token().map(|(_, q)| q);
                    return;
                }
                KeyCode::Up => {
                    self.at_selected = self.at_selected.saturating_sub(1);
                    return;
                }
                KeyCode::Down => {
                    let n = self.at_matches().len();
                    if n > 0 {
                        self.at_selected = (self.at_selected + 1).min(n - 1);
                    }
                    return;
                }
                KeyCode::Tab => {
                    self.at_complete();
                    return;
                }
                KeyCode::Enter if key.modifiers.is_empty() => {
                    self.at_complete();
                    return;
                }
                _ => {}
            }
        }

        // Esc: cancel a running task; else leave a non-chat mode / clear input.
        // A running deploy is deliberately NOT in this set — it is cancelled only
        // via its Cancel button, so here Esc just leaves Models / clears input.
        if key.code == KeyCode::Esc {
            if self.fg_busy()
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

        // Enter submits; Shift+Enter / Alt+Enter / Ctrl+Enter insert a newline
        // so the omnibar is a real multi-line composer. Ctrl+J is the fallback
        // for terminals that don't report modified Enter at all.
        if key.code == KeyCode::Enter {
            if key
                .modifiers
                .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT | KeyModifiers::CONTROL)
            {
                self.omnibar_insert_newline();
            } else {
                self.omnibar_submit();
            }
            return;
        }
        if key.code == KeyCode::Char('j') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.omnibar_insert_newline();
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
                self.at_selected = 0;
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
                self.at_selected = 0;
            }
            _ => {}
        }
    }

    /// Bracketed paste: insert the pasted text where input currently goes.
    /// Newlines are preserved in the omnibar (this is what makes multi-line
    /// paste work — without it each pasted line-break would submit the turn)
    /// and flattened for the single-line field editors.
    pub fn handle_paste(&mut self, pasted: &str) {
        let text = pasted.replace("\r\n", "\n").replace('\r', "\n");
        if let Some(ps) = &mut self.pending_sudo {
            for c in text.chars().filter(|c| *c != '\n') {
                ps.buf.push(c);
            }
            return;
        }
        if self.modal.is_some() {
            return;
        }
        let flat = || text.replace('\n', " ");
        if self.mode == Mode::Remote && self.remote_editing {
            self.remote_field_edit.push_str(&flat());
            return;
        }
        if self.mode == Mode::Settings && self.settings_editing.is_some() {
            self.settings_field_edit.push_str(&flat());
            return;
        }
        if self.mode == Mode::Models && self.deploy_editing.is_some() {
            let text = flat();
            let idx = char_to_byte(&self.deploy_field_edit, self.deploy_field_cursor);
            self.deploy_field_edit.insert_str(idx, &text);
            self.deploy_field_cursor += text.chars().count();
            return;
        }
        let idx = char_to_byte(&self.coding_input, self.coding_cursor);
        self.coding_input.insert_str(idx, &text);
        self.coding_cursor += text.chars().count();
        self.coding_hist_idx = None;
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
        self.at_selected = 0;
        self.at_dismissed = None;
    }

    fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        self.remote_editing = false;
        self.settings_editing = None;
        // Leaving Models must also close any open deploy-parameter editor, so it
        // can't reappear mid-edit with a stale buffer on return.
        self.deploy_editing = None;
        // The downloaded-model tag (/models) and the /dash downloaded section
        // both read the on-disk set — refresh it on entry so it reflects reality.
        if matches!(mode, Mode::Models | Mode::Dash) {
            self.refresh_downloaded();
        }
        // Open /dash focused on the active model's card (cards are reordered, so
        // the active runtime isn't necessarily card 0).
        if mode == Mode::Dash {
            self.dash_selected = self
                .dash_cards()
                .iter()
                .position(|c| c.runtime_index == Some(self.runtime_selected))
                .unwrap_or(0);
        }
    }

    /// Selection / scroll keys for the active mode (arrows, page keys, Tab).
    fn mode_nav(&mut self, code: KeyCode) {
        match self.mode {
            Mode::Chat => match code {
                KeyCode::PageDown => self.scroll_transcript(self.coding_view_height.max(1) as i64),
                KeyCode::PageUp => self.scroll_transcript(-(self.coding_view_height.max(1) as i64)),
                // Shift+Tab cycles the agent approval mode (like /mode).
                KeyCode::BackTab => self.cycle_approval_mode(),
                _ => {}
            },
            Mode::Models => match code {
                KeyCode::Down => {
                    if !self.models.is_empty() {
                        self.model_selected = (self.model_selected + 1).min(self.models.len() - 1);
                        self.arm_detail_autoload();
                    }
                }
                KeyCode::Up => {
                    self.model_selected = self.model_selected.saturating_sub(1);
                    self.arm_detail_autoload();
                }
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
            // The `/dash` highlight moves over the WHOLE card list (running +
            // downloaded + featured), not just running runtimes — otherwise the
            // cards below the running set are unreachable. Landing on a running
            // card also makes it the active model for the next request; the draw
            // auto-scrolls to keep the highlight visible.
            Mode::Dash => match code {
                KeyCode::Down => self.dash_move(1),
                KeyCode::Up => self.dash_move(-1),
                KeyCode::Tab => {
                    let n = self.dash_cards().len();
                    if n > 0 {
                        self.dash_selected = (self.dash_selected + 1) % n;
                        self.sync_runtime_from_dash();
                    }
                }
                KeyCode::PageDown => self.dash_move(self.dash_view_cards.max(1) as i64),
                KeyCode::PageUp => self.dash_move(-(self.dash_view_cards.max(1) as i64)),
                _ => {}
            },
            // Sessions selection follows the same follow-the-highlight model
            // as Dash: the draw scrolls to keep it visible.
            Mode::Sessions => match code {
                KeyCode::Down => {
                    let n = self.sessions.len();
                    if n > 0 {
                        self.session_selected = (self.session_selected + 1).min(n - 1);
                    }
                }
                KeyCode::Up => self.session_selected = self.session_selected.saturating_sub(1),
                KeyCode::PageDown => {
                    let n = self.sessions.len();
                    if n > 0 {
                        self.session_selected = (self.session_selected + 10).min(n - 1);
                    }
                }
                KeyCode::PageUp => {
                    self.session_selected = self.session_selected.saturating_sub(10);
                }
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
            // Empty Enter in the dashboard: act on the highlighted card — chat
            // with a running model, download-and-run a downloaded/featured one,
            // or copy the OpenWebUI URL.
            Mode::Dash => self.dash_activate(self.dash_selected),
            Mode::Sessions => self.resume_session(self.session_selected),
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
                    ModalKind::Confirm {
                        action: ConfirmAction::InstallLocalAssistant,
                        ..
                    } => self.decline_local_assistant(),
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
                    action: ConfirmAction::InstallLocalAssistant,
                    ..
                },
                "Confirm",
            ) => self.start_install_local_assistant(),
            (
                ModalKind::Confirm {
                    action: ConfirmAction::InstallLocalAssistant,
                    ..
                },
                "Cancel",
            ) => self.decline_local_assistant(),
            (
                ModalKind::Confirm {
                    action: ConfirmAction::ToolApproval,
                    ..
                },
                label,
            ) => self.respond_tool_confirm(label == "Confirm"),
            (
                ModalKind::Confirm {
                    action: ConfirmAction::DeleteModel,
                    ..
                },
                "Confirm",
            ) => self.confirm_delete_model(),
            _ => {}
        }
    }

    fn handle_mouse(&mut self, mouse: crossterm::event::MouseEvent) {
        use event::MouseButton::Left;
        let (col, row) = (mouse.column, mouse.row);
        self.hover = Some((col, row));
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
        // '@' file picker: clicking a row attaches that file.
        if let ClickTarget::AtItem(i) = region.target {
            self.at_selected = i;
            self.at_complete();
            return;
        }

        // Row index within a list = first visible row + click offset.
        let rel_row = row.saturating_sub(region.rect.y) as usize;
        match region.target {
            // Status bar.
            ClickTarget::Theme(m) => self.set_theme(m),
            ClickTarget::ApprovalCycle => self.cycle_approval_mode(),
            ClickTarget::UpdateBadge => {
                if self.update_available.is_some() {
                    self.open_update_modal();
                }
            }
            ClickTarget::StatusToggle => self.toggle_status_pin(),
            ClickTarget::Transcript => {}
            ClickTarget::TranscriptEntry(i) => {
                // A model response: select it and auto-copy to the clipboard.
                if self
                    .coding_transcript
                    .get(i)
                    .is_some_and(|e| e.is_model_response())
                {
                    self.coding_selected = Some(i);
                    let text = self.coding_transcript[i].text.clone();
                    self.copy_to_clipboard(&text, "model response");
                } else if let Some(e) = self.coding_transcript.get_mut(i) {
                    if e.can_toggle() {
                        e.expanded = !e.expanded;
                        // Expanding often needs a re-scroll so the detail is visible.
                        if e.expanded {
                            self.coding_follow = true;
                        }
                    }
                }
            }
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
                let picked = self
                    .model_detail
                    .as_ref()
                    .and_then(|d| d.quants.get(rel_row))
                    .map(|q| q.label.clone());
                if let Some(label) = picked {
                    self.selected_quant = Some(label);
                    self.apply_deploy_preset(PresetMode::QuantChanged);
                    self.refresh_fit();
                }
            }
            ClickTarget::BackendCycle => self.cycle_deploy_backend(),
            ClickTarget::DeployAutoConfig => self.auto_configure_deploy(),
            ClickTarget::DeployButton => self.start_deploy(false),
            ClickTarget::DeployCancel => self.cancel_deploy(),
            ClickTarget::DownloadResume => {
                if let Some(id) = self.current_model_id() {
                    self.resume_download(&id);
                }
            }
            ClickTarget::DeployField(field) => {
                let rel_col = col.saturating_sub(region.rect.x) as usize;
                self.click_deploy_field(field, rel_col);
            }
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
            // Dash (multi-model manager). Button regions are registered after the
            // card region, so the reverse hit-test lets them win over a card click.
            ClickTarget::DashCard(i) => self.dash_use(i),
            ClickTarget::DashUse(i) => self.dash_use(i),
            ClickTarget::DashCopyCmd(i) => {
                if let Some(card) = self.dash_cards().get(i) {
                    let cmd = card.command.clone();
                    self.copy_to_clipboard(&cmd, "command");
                }
            }
            ClickTarget::DashStop(i) => self.dash_stop(i),
            ClickTarget::DashCopyErr(i) => {
                if let Some(err) = self.dash_cards().get(i).and_then(|c| c.error_text.clone()) {
                    self.copy_to_clipboard(&err, "error");
                }
            }
            ClickTarget::DashDeploy(i) => self.dash_deploy(i),
            ClickTarget::DashDelete(i) => self.dash_delete(i),
            ClickTarget::DashFavorite(i) => self.dash_favorite(i),
            // Sessions: clicking a row resumes it (same as select + Enter).
            ClickTarget::SessionList => {
                let idx = self.sessions_scroll + rel_row;
                if idx < self.sessions.len() {
                    self.session_selected = idx;
                    self.resume_session(idx);
                }
            }
            ClickTarget::SessionsNew => {
                self.new_coding_session();
                self.set_mode(Mode::Chat);
            }
            ClickTarget::DashStartNew => {
                self.set_mode(Mode::Models);
                if self.models.is_empty() && !self.fg_busy() {
                    self.start_search();
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
            ClickTarget::ModalButton(_) | ClickTarget::CommandItem(_) | ClickTarget::AtItem(_) => {}
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
        // Switching backends changes which params are shown; drop any in-progress
        // edit so the editor can't get stuck on a now-hidden field.
        self.deploy_editing = None;
        // A command override built for the old backend is meaningless for the
        // new one — clear it so the field reseeds from the new backend's command.
        self.deploy_command_override = None;
        self.deploy_backend = match self.deploy_backend {
            BackendKind::Ollama => BackendKind::LlamaCpp,
            BackendKind::LlamaCpp => BackendKind::Vllm,
            BackendKind::Vllm => BackendKind::Sglang,
            BackendKind::Sglang => BackendKind::Colibri,
            BackendKind::Colibri => BackendKind::ColibriHy3,
            BackendKind::ColibriHy3 => BackendKind::Ollama,
        };
        // Re-derive the card flags / context for the newly-pinned backend.
        self.apply_deploy_preset(PresetMode::BackendPinned);
        self.refresh_fit();
    }

    /// Route the wheel to whatever scrollable is under the cursor, falling back
    /// to the active mode's main scrollable. While the command palette or the
    /// '@' file picker is open, the wheel moves their selection — wherever the
    /// cursor is — so the list is scrollable the moment it appears.
    fn wheel_scroll_at(&mut self, delta: i64, col: u16, row: u16) {
        if self.slash_active() {
            let n = self.palette_items().len();
            if n > 0 {
                let cur = self.palette_selected.min(n - 1) as i64;
                self.palette_selected = (cur + delta.signum()).clamp(0, n as i64 - 1) as usize;
            }
            return;
        }
        if self.at_picker_active() {
            let n = self.at_matches().len();
            if n > 0 {
                let cur = self.at_selected.min(n - 1) as i64;
                self.at_selected = (cur + delta.signum()).clamp(0, n as i64 - 1) as usize;
            }
            return;
        }
        match self.region_at(col, row).map(|r| r.target) {
            Some(ClickTarget::ModelList) => self.scroll_list_models(delta),
            Some(ClickTarget::QuantList) => self.scroll_card(delta),
            Some(ClickTarget::Transcript) | Some(ClickTarget::TranscriptEntry(_)) => {
                self.scroll_transcript(delta)
            }
            Some(ClickTarget::RuntimeList) => self.scroll_runtimes(delta),
            Some(ClickTarget::DashCard(_))
            | Some(ClickTarget::DashCopyCmd(_))
            | Some(ClickTarget::DashStop(_))
            | Some(ClickTarget::DashUse(_))
            | Some(ClickTarget::DashCopyErr(_))
            | Some(ClickTarget::DashDeploy(_))
            | Some(ClickTarget::DashDelete(_))
            | Some(ClickTarget::DashFavorite(_))
            | Some(ClickTarget::DashStartNew) => self.scroll_dash(delta),
            Some(ClickTarget::SessionList) => self.scroll_sessions(delta),
            _ => self.wheel_scroll(delta),
        }
    }

    /// Fallback wheel routing by active mode (cursor not over a known region).
    fn wheel_scroll(&mut self, delta: i64) {
        match self.mode {
            Mode::Chat => self.scroll_transcript(delta),
            Mode::Models => self.scroll_card(delta),
            Mode::Dash => self.scroll_dash(delta),
            Mode::Sessions => self.scroll_sessions(delta),
            Mode::Setup => {
                self.setup_scroll = (i64::from(self.setup_scroll) + delta).max(0) as u16;
            }
            Mode::Settings => {
                self.settings_scroll = (i64::from(self.settings_scroll) + delta).max(0) as u16;
            }
            _ => {}
        }
    }

    /// Wheel over `/dash`: move the highlight one card (the draw follows it).
    /// Matches the wheel behaviour of every other list view (sessions, models,
    /// runtimes) rather than a decoupled scroll offset — that decoupling was the
    /// bug where downloaded models below the running set couldn't be reached.
    fn scroll_dash(&mut self, delta: i64) {
        self.dash_move(delta.signum());
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
        // Wheel-scrolling the list follows with the detail card on the right.
        self.arm_detail_autoload();
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

/// Load a model's detail card from HF, falling back to one synthesized from
/// the weights already on disk. The fallback is what lets `/dash`'s Deploy
/// button work with no network: the downloaded files still yield quant groups
/// and sizes, which is everything the deploy panel needs.
async fn fetch_detail_or_offline(
    hf: Option<HfClient>,
    models_dir: &std::path::Path,
    id: &str,
) -> Result<ModelDetail, LocalCodeError> {
    let err = match hf {
        Some(hf) => match hf.model_info(id).await {
            Ok(detail) => return Ok(detail),
            Err(e) => e,
        },
        None => LocalCodeError::new(ErrorCode::HfUnreachable, "HF client unavailable"),
    };
    offline_model_detail(models_dir, id).ok_or(err)
}

/// Synthesize a [`ModelDetail`] for a model whose weights are in the models
/// cache: siblings are the on-disk files (with their real sizes, so VRAM fit
/// prediction still works) and quant groups are rediscovered from the
/// filenames. Registry-only data (likes, license, card) is absent.
fn offline_model_detail(models_dir: &std::path::Path, id: &str) -> Option<ModelDetail> {
    let m = find_downloaded(models_dir, id)?;
    let siblings: Vec<ModelFile> = m
        .files
        .iter()
        .map(|f| ModelFile {
            rfilename: f.clone(),
            size: std::fs::metadata(m.dir.join(f)).ok().map(|md| md.len()),
        })
        .collect();
    let quants = discover_quants(&siblings);
    Some(ModelDetail {
        summary: ModelSummary {
            id: id.to_string(),
            author: None,
            pipeline_tag: None,
            tags: Vec::new(),
            likes: None,
            downloads: None,
            last_modified: None,
            private: None,
            gated: None,
        },
        siblings,
        card_data: None,
        sha: None,
        card_markdown: Some(format!(
            "**Offline** — Hugging Face is unreachable, showing the weights \
             downloaded to `{}`. Deploying uses the local files; no network is \
             needed.",
            m.dir.display()
        )),
        license: None,
        parameter_size: None,
        quants,
    })
}

/// Copy `text` to the terminal's clipboard using the OSC 52 escape sequence.
/// Dependency-free and works over SSH / inside tmux (with `set-clipboard on`);
/// terminals without OSC 52 support silently ignore it. Written straight to
/// stdout between frames, so it doesn't interleave with a ratatui draw.
fn osc52_copy(text: &str) {
    use base64::Engine;
    use std::io::Write;
    let encoded = base64::engine::general_purpose::STANDARD.encode(text.as_bytes());
    let seq = format!("\x1b]52;c;{encoded}\x07");
    let mut out = io::stdout();
    let _ = out.write_all(seq.as_bytes());
    let _ = out.flush();
}

/// Fixed display order of backends in the manager overlay.
pub(crate) const BACKEND_ORDER: [BackendKind; 6] = [
    BackendKind::Ollama,
    BackendKind::LlamaCpp,
    BackendKind::Vllm,
    BackendKind::Sglang,
    BackendKind::Colibri,
    BackendKind::ColibriHy3,
];

/// Load the latest session for this workspace when persistence is on, or
/// create a fresh one (and its store file).
fn bootstrap_session(
    agent_cfg: &localcode_core::config::AgentConfig,
    paths: &localcode_core::paths::AppPaths,
    workspace: PathBuf,
) -> (
    Arc<AsyncMutex<AgentSession>>,
    Arc<AsyncMutex<Option<SessionStore>>>,
    Option<String>,
) {
    if !agent_cfg.sessions_enabled {
        let session = AgentSession::new(workspace);
        return (
            Arc::new(AsyncMutex::new(session)),
            Arc::new(AsyncMutex::new(None)),
            None,
        );
    }
    let root = sessions_root(agent_cfg, paths);
    if let Some(meta) = list_sessions(&root, &workspace).into_iter().next() {
        match SessionStore::load(&meta.path) {
            Ok(loaded) => {
                let note = if loaded.warnings.is_empty() {
                    format!("Resumed session “{}”", loaded.session.title)
                } else {
                    format!(
                        "Resumed session “{}” ({} repair note(s))",
                        loaded.session.title,
                        loaded.warnings.len()
                    )
                };
                return (
                    Arc::new(AsyncMutex::new(loaded.session)),
                    Arc::new(AsyncMutex::new(Some(loaded.store))),
                    Some(note),
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "could not load latest session; starting fresh");
            }
        }
    }
    let session = AgentSession::new(workspace);
    let store = SessionStore::create(&root, &session).ok();
    (
        Arc::new(AsyncMutex::new(session)),
        Arc::new(AsyncMutex::new(store)),
        None,
    )
}

/// Listing/banner label for a session title; blank titles read as "untitled".
pub(crate) fn display_title(title: &str) -> &str {
    let t = title.trim();
    if t.is_empty() {
        "untitled"
    } else {
        t
    }
}

/// Visible transcript rows for a session loaded from disk: one entry per
/// user/assistant message, thinking collapsed, tool results as expandable
/// `✓ name` rows — the same shapes a live turn leaves behind.
fn transcript_from_session(session: &AgentSession) -> Vec<TranscriptEntry> {
    let mut out = Vec::new();
    for m in &session.messages {
        match m.role.as_str() {
            "user" => out.push(TranscriptEntry::new(EntryKind::You, m.content.clone())),
            "assistant" => {
                if let Some(t) = m.thinking.as_ref().filter(|s| !s.is_empty()) {
                    out.push(TranscriptEntry {
                        kind: EntryKind::Thinking,
                        text: t.clone(),
                        live: false,
                        detail: None,
                        expanded: false,
                    });
                }
                if !m.content.trim().is_empty() {
                    out.push(TranscriptEntry::new(EntryKind::Agent, m.content.clone()));
                }
            }
            "tool" => {
                let name = m.name.as_deref().unwrap_or("tool");
                out.push(TranscriptEntry {
                    kind: EntryKind::Tool,
                    text: format!("✓ {name}"),
                    live: false,
                    detail: Some(m.content.clone()),
                    expanded: false,
                });
            }
            _ => {}
        }
    }
    out
}

/// Rough token estimate for the status-bar context meter: ~4 chars per token
/// (latin code-heavy text). Not a tokenizer — just a live fill level.
fn estimate_session_tokens(session: &AgentSession) -> u32 {
    let chars: usize = session
        .messages
        .iter()
        .map(|m| {
            m.content.chars().count()
                + m.tool_calls
                    .as_ref()
                    .map(|v| v.to_string().chars().count())
                    .unwrap_or(0)
                + m.thinking.as_ref().map(|t| t.chars().count()).unwrap_or(0)
        })
        .sum();
    (chars / 4) as u32
}

/// Compact a structured JSON log line to `HH:MM:SS LEVEL message` for the
/// status dashboard and `/logs` banner. Non-JSON lines pass through as-is.
fn compact_log_line(line: &str) -> String {
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

/// Editable fields of a remote server, in display order.
pub(crate) const REMOTE_FIELDS: [&str; 6] =
    ["name", "host", "port", "username", "password", "local_port"];

/// Extra context shown above the install confirm command, per backend.
/// Build the expandable tool body (args + full output) for the transcript.
fn format_tool_detail(args: &str, output: &str) -> String {
    let mut parts = Vec::new();
    if !args.trim().is_empty() {
        parts.push(format!("args\n{args}"));
    }
    let n = output.chars().count();
    if n == 0 {
        parts.push("output\n(empty)".into());
    } else {
        parts.push(format!("output ({n} chars)\n{output}"));
    }
    parts.join("\n\n")
}

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

/// Char index in `buf` for a click `col` cells into the visible value window,
/// which starts `scroll` chars into the buffer (mirrors how the deploy panel
/// draws the field). Walks display widths so a double-width glyph maps both
/// of its cells to the same caret; a click past the end appends.
fn caret_from_col(buf: &str, scroll: usize, col: usize) -> usize {
    let mut w = 0usize;
    let mut idx = scroll.min(buf.chars().count());
    for c in buf.chars().skip(idx) {
        let cw = c.width().unwrap_or(0);
        if w + cw > col {
            break;
        }
        w += cw;
        idx += 1;
    }
    idx
}

pub async fn run_tui(paths: AppPaths, config: Config) -> Result<(), LocalCodeError> {
    info!("starting TUI");

    // Restore the terminal on panic — otherwise a panic leaves the user's
    // shell in raw mode + alternate screen with mouse capture on.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            LeaveAlternateScreen,
            DisableMouseCapture,
            DisableBracketedPaste
        );
        default_hook(panic_info);
    }));

    let mouse_enabled = config.ui.mouse;
    enable_raw_mode().map_err(LocalCodeError::from)?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste).map_err(LocalCodeError::from)?;
    if mouse_enabled {
        execute!(io::stdout(), EnableMouseCapture).map_err(LocalCodeError::from)?;
    }
    // Where the terminal supports the kitty keyboard protocol (a no-op query on
    // Windows), opt in to disambiguated escape codes so Shift+Enter/Alt+Enter
    // are reported and the multi-line composer works over SSH too.
    let keyboard_enhanced = crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false);
    if keyboard_enhanced {
        let _ = execute!(
            io::stdout(),
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        );
    }
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(LocalCodeError::from)?;

    let mut app = App::new(paths, config);
    app.start_detect();
    app.start_gpu_refresh();
    app.refresh_status_logs();
    if app.config.updates.check_on_startup {
        app.start_update_check(false);
    }
    app.autoconnect_remotes();
    // Revive any background downloads whose worker didn't survive the last exit
    // (crash/reboot). Workers that are still running keep their state fresh and
    // are simply re-attached (rendered) rather than restarted.
    app.resume_stale_downloads();
    let result = run_loop(&mut terminal, &mut app).await;

    // Kill managed model servers on EVERY exit path — including an early return
    // from a terminal I/O error inside run_loop — while the tokio runtime is
    // still alive to reap them. Idempotent, so the normal-quit call is harmless.
    app.shutdown().await;

    if keyboard_enhanced {
        execute!(terminal.backend_mut(), PopKeyboardEnhancementFlags).ok();
    }
    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableBracketedPaste).ok();
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
        app.poll_downloads();

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
                Event::Paste(text) => app.handle_paste(&text),
                Event::Resize(_, _) => {
                    // Layout recomputes on next draw
                }
                _ => {}
            }
        }
    }
    // Persist pane ratios etc. Managed model servers are killed by the caller
    // (`run_tui`) so cleanup also runs if this loop returns early on an error.
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
        // Don't open the first-run assistant install modal (it captures keys
        // and confirming it would tokio::spawn without a runtime in unit tests).
        let mut cfg = Config::default();
        cfg.assistant.local_preference = LocalAssistantPreference::Declined;
        // Keep unit tests hermetic: no session files from prior runs, no skills I/O.
        cfg.agent.sessions_enabled = false;
        App::new(paths, cfg)
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

    // ---- /dash multi-model manager ----------------------------------------

    /// A deployed runtime plus a matching live process monitor, both keyed by the
    /// same id (mirrors what a real backend deploy registers).
    fn add_running_model(
        app: &mut App,
        name: &str,
        model: &str,
        command: &str,
        state: ProcState,
    ) -> uuid::Uuid {
        let mut rt = ActiveRuntime::new(
            name,
            localcode_core::runtime::RuntimeKind::Vllm,
            "http://127.0.0.1:8000/v1",
        );
        rt.model_id = Some(model.into());
        rt.status = RuntimeStatus::Healthy;
        let id = rt.id;
        app.runtimes.push(rt);
        app.registry.monitors().register(
            id.to_string(),
            name,
            BackendKind::Vllm,
            Some(model.into()),
            command,
            state,
        );
        id
    }

    /// `/dash` cards for the user's own running/downloaded models, dropping the
    /// always-present featured "download & run" entries so count assertions stay
    /// about the user's models.
    fn own_cards(app: &App) -> Vec<DashCard> {
        app.dash_cards()
            .into_iter()
            .filter(|c| !c.is_featured)
            .collect()
    }

    #[test]
    fn dash_command_switches_mode() {
        let mut app = test_app();
        typ(&mut app, "/dash");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.mode, Mode::Dash);
        assert!(app.coding_input.is_empty());
    }

    #[test]
    fn dash_cards_reflect_registered_monitor() {
        let mut app = test_app();
        let id = add_running_model(
            &mut app,
            "vllm:test/model",
            "test/model",
            "vllm serve test/model --max-model-len 8192",
            ProcState::Running,
        );
        app.registry
            .monitors()
            .get(&id.to_string())
            .unwrap()
            .push_log("Uvicorn running on http://127.0.0.1:8000");

        let cards = own_cards(&app);
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].name, "vllm:test/model");
        assert_eq!(cards[0].backend_label, "vllm");
        assert!(cards[0].command.contains("vllm serve"));
        assert_eq!(
            cards[0].logs.last().map(String::as_str),
            Some("Uvicorn running on http://127.0.0.1:8000")
        );
        // Sole runtime → it's the active model for the next request.
        assert!(cards[0].is_active);
        assert!(cards[0].error_text.is_none());
    }

    #[test]
    fn dash_card_surfaces_nonzero_exit_error() {
        let mut app = test_app();
        let id = add_running_model(&mut app, "vllm:x", "x", "vllm serve x", ProcState::Running);
        let mon = app.registry.monitors().get(&id.to_string()).unwrap();
        mon.push_log("CUDA error: out of memory");
        mon.set_state(ProcState::Exited { code: Some(1), ok: false });

        let cards = own_cards(&app);
        assert_eq!(cards.len(), 1);
        assert!(cards[0].status_is_error);
        let err = cards[0].error_text.as_ref().expect("failed model has error text");
        assert!(err.contains("out of memory"), "{err}");
        assert!(err.contains("code 1"), "{err}");
    }

    #[test]
    fn dash_use_selects_active_model_for_next_request() {
        let mut app = test_app();
        add_running_model(&mut app, "m0", "a", "cmd0", ProcState::Running);
        add_running_model(&mut app, "m1", "b", "cmd1", ProcState::Running);

        app.dash_use(1);
        assert_eq!(app.runtime_selected, 1);
        assert_eq!(app.active_runtime().unwrap().name, "m1");
        // Exactly the selected card is marked active.
        let cards = app.dash_cards();
        assert!(!cards[0].is_active);
        assert!(cards[1].is_active);
    }

    #[test]
    fn dash_stop_removes_the_model_card() {
        let mut app = test_app();
        let id = add_running_model(&mut app, "m0", "a", "cmd0", ProcState::Running);
        assert_eq!(own_cards(&app).len(), 1);
        // Stopping drops it from the registry runtimes and its monitor. (No child
        // process exists in the test, so the backend stop is a no-op; we simulate
        // the deregistration the stop path performs.)
        app.runtimes.retain(|r| r.id != id);
        app.registry.monitors().remove(&id.to_string());
        assert!(own_cards(&app).is_empty());
    }

    #[test]
    fn dash_shows_downloaded_models_with_favorites_first() {
        let mut app = test_app();
        app.downloaded_models = vec![
            DownloadedModel {
                model_id: "org/alpha".into(),
                dir: std::path::PathBuf::from("/models/alpha"),
                total_bytes: 2048,
                quants: vec!["Q4_K_M".into()],
                files: vec!["a.gguf".into()],
            },
            DownloadedModel {
                model_id: "org/zeta".into(),
                dir: std::path::PathBuf::from("/models/zeta"),
                total_bytes: 4096,
                quants: vec!["Q8_0".into()],
                files: vec!["z.gguf".into()],
            },
        ];
        app.config.add_favorite("org/zeta");

        let cards = own_cards(&app);
        assert_eq!(cards.len(), 2);
        // The favourite sorts first despite "zeta" > "alpha" alphabetically.
        assert_eq!(cards[0].model_id.as_deref(), Some("org/zeta"));
        assert!(cards[0].is_favorite);
        assert!(cards[0].can_deploy && cards[0].can_delete);
        assert!(cards[0].runtime_index.is_none(), "downloaded-only, not running");
        assert_eq!(cards[0].disk_bytes, Some(4096));
        assert_eq!(cards[1].model_id.as_deref(), Some("org/alpha"));
        assert!(!cards[1].is_favorite);
    }

    #[test]
    fn dash_deploy_delete_favorite_target_the_right_model() {
        let mut app = test_app();
        app.downloaded_models = vec![DownloadedModel {
            model_id: "org/alpha".into(),
            dir: std::path::PathBuf::from("/models/alpha"),
            total_bytes: 2048,
            quants: vec!["Q4_K_M".into()],
            files: vec!["a.gguf".into()],
        }];
        // Favouriting the downloaded card toggles config for that model id.
        assert!(!app.config.is_favorite("org/alpha"));
        app.dash_favorite(0);
        assert!(app.config.is_favorite("org/alpha"));
        // Deleting opens a confirm modal naming the model (no accidental delete).
        app.dash_delete(0);
        assert_eq!(app.pending_delete_model.as_deref(), Some("org/alpha"));
        assert!(app.modal.is_some());
    }

    #[test]
    fn dash_lists_featured_colibri_download_and_run_cards() {
        // With nothing running or downloaded, /dash still surfaces the curated
        // colibrì containers so they can be fetched straight from the dashboard.
        let app = test_app();
        let cards = app.dash_cards();
        let featured: Vec<_> = cards.iter().filter(|c| c.is_featured).collect();
        assert_eq!(featured.len(), FEATURED_MODELS.len());
        for f in &featured {
            assert!(f.can_deploy, "featured card offers download & run");
            assert!(!f.can_delete, "nothing on disk to delete yet");
            assert!(f.disk_bytes.is_none());
            assert!(f.model_id.is_some());
        }
        assert!(cards.iter().any(|c| c.model_id.as_deref()
            == Some("mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp")));
    }

    #[test]
    fn dash_featured_card_hidden_once_downloaded() {
        let mut app = test_app();
        let id = FEATURED_MODELS[0].id;
        app.downloaded_models = vec![DownloadedModel {
            model_id: id.into(),
            dir: std::path::PathBuf::from("/models/glm"),
            total_bytes: 1024,
            quants: vec![],
            files: vec![],
        }];
        let cards = app.dash_cards();
        // The downloaded copy replaces the featured pointer — exactly one card
        // for that id, and it's a real (deletable) downloaded card now.
        let matching: Vec<_> = cards
            .iter()
            .filter(|c| c.model_id.as_deref() == Some(id))
            .collect();
        assert_eq!(matching.len(), 1);
        assert!(!matching[0].is_featured);
        assert!(matching[0].can_delete);
    }

    #[test]
    fn dash_selection_reaches_cards_below_the_running_set() {
        // Regression: the highlight used to be bound to `runtime_selected` (only
        // running runtimes), so downloaded/featured cards below the running set
        // were unreachable and the view snapped back to the running card. The
        // highlight now indexes the whole card list.
        let mut app = test_app();
        add_running_model(&mut app, "m0", "run/model", "cmd0", ProcState::Running);
        app.set_mode(Mode::Dash);
        let total = app.dash_cards().len();
        assert!(total > 1, "running model + featured cards");
        // Drive the highlight to the bottom of the list.
        for _ in 0..total {
            app.dash_move(1);
        }
        assert_eq!(app.dash_selected, total - 1);
        assert!(
            app.dash_cards()[app.dash_selected].is_featured,
            "last card is a featured one, previously unreachable"
        );
        // Browsing a non-running card must not swap the active model.
        assert_eq!(app.runtime_selected, 0);
        assert_eq!(app.active_runtime().unwrap().name, "m0");
    }

    #[test]
    fn dash_move_syncs_active_runtime_on_running_cards() {
        let mut app = test_app();
        add_running_model(&mut app, "m0", "a", "cmd0", ProcState::Running);
        add_running_model(&mut app, "m1", "b", "cmd1", ProcState::Running);
        app.set_mode(Mode::Dash); // opens on the active card (runtime 0 → card 0)
        assert_eq!(app.dash_selected, 0);
        app.dash_move(1);
        // Landing on the second running card adopts it for the next request.
        assert_eq!(app.dash_selected, 1);
        assert_eq!(app.runtime_selected, 1);
        assert_eq!(app.active_runtime().unwrap().name, "m1");
    }

    // ---- /sessions past-chat switcher --------------------------------------

    fn chat_msg(role: &str, content: &str) -> localcode_agent::ChatMessage {
        localcode_agent::ChatMessage {
            role: role.into(),
            content: content.into(),
            tool_call_id: None,
            name: None,
            tool_calls: None,
            thinking: None,
        }
    }

    /// Write a finished one-turn session file under `root` for `ws`, aged by
    /// `age_secs` so listing order (newest mtime first) is deterministic.
    fn seed_session(
        root: &std::path::Path,
        ws: &std::path::Path,
        title: &str,
        user_text: &str,
        age_secs: u64,
    ) -> String {
        let mut s = AgentSession::new(ws.to_path_buf());
        s.messages.push(chat_msg("user", user_text));
        s.messages.push(chat_msg("assistant", "done"));
        s.title = title.into();
        let mut store = SessionStore::create(root, &s).unwrap();
        store.sync(&s).unwrap();
        let past = std::time::SystemTime::now() - Duration::from_secs(age_secs);
        std::fs::OpenOptions::new()
            .append(true)
            .open(store.path())
            .unwrap()
            .set_modified(past)
            .unwrap();
        s.id
    }

    #[test]
    fn sessions_command_reports_when_persistence_is_off() {
        let mut app = test_app(); // sessions_enabled = false
        typ(&mut app, "/sessions");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.mode, Mode::Chat, "stays in chat when nothing is saved");
        assert!(app.status_is_error);
        assert!(app.status_line.contains("sessions_enabled"), "{}", app.status_line);
    }

    #[test]
    fn sessions_lists_past_chats_and_enter_switches_to_one() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let ws = dir.path().join("ws");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(&ws).unwrap();
        let paths = AppPaths::from_home(home);
        paths.ensure_dirs().unwrap();

        let mut cfg = Config::default();
        cfg.assistant.local_preference = LocalAssistantPreference::Declined;
        cfg.agent.workspace_root = Some(ws.display().to_string());
        // Seed two chats before the app starts; both aged so the resume touch
        // below lands strictly newer even on coarse-mtime filesystems.
        let root = sessions_root(&cfg.agent, &paths);
        let old_id = seed_session(&root, &ws, "old chat", "old question", 600);
        let new_id = seed_session(&root, &ws, "new chat", "new question", 300);

        let mut app = App::new(paths, cfg);
        assert_eq!(app.current_session_id, new_id, "startup resumes the newest chat");

        typ(&mut app, "/sessions");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.mode, Mode::Sessions);
        assert_eq!(app.sessions.len(), 2);
        assert_eq!(
            app.sessions[app.session_selected].id, old_id,
            "preselects the newest chat that isn't the current one"
        );

        // Enter resumes the highlighted chat: state, transcript and marker move.
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.mode, Mode::Chat);
        assert_eq!(app.current_session_id, old_id);
        assert!(
            app.coding_transcript
                .iter()
                .any(|e| e.kind == EntryKind::You && e.text == "old question"),
            "resumed history is visible in the transcript"
        );
        assert_eq!(
            app.session.try_lock().unwrap().messages.len(),
            2,
            "agent session now holds the resumed history"
        );

        // The resume touched the file: reopening lists it first, and the
        // preselection points back at the chat we came from.
        typ(&mut app, "/sessions");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.sessions[0].id, old_id, "resumed chat is now newest");
        assert_eq!(app.sessions[app.session_selected].id, new_id);

        // Resuming the row of the current chat is a no-op back to chat.
        let cur_row = app
            .sessions
            .iter()
            .position(|m| m.id == app.current_session_id)
            .unwrap();
        app.resume_session(cur_row);
        assert_eq!(app.mode, Mode::Chat);
        assert_eq!(app.current_session_id, old_id);
    }

    #[test]
    fn sessions_view_renders_rows_with_current_marker() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let ws = dir.path().join("ws");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(&ws).unwrap();
        let paths = AppPaths::from_home(home);
        paths.ensure_dirs().unwrap();
        let mut cfg = Config::default();
        cfg.assistant.local_preference = LocalAssistantPreference::Declined;
        cfg.agent.workspace_root = Some(ws.display().to_string());
        let root = sessions_root(&cfg.agent, &paths);
        seed_session(&root, &ws, "old chat", "old question", 600);
        seed_session(&root, &ws, "new chat", "new question", 300);

        let mut app = App::new(paths, cfg);
        app.open_sessions();
        let s = render_to_string(&mut app, 100, 24);
        assert!(s.contains("past chats (2)"), "{s}");
        assert!(s.contains("new chat"), "{s}");
        assert!(s.contains("old chat"), "{s}");
        assert!(s.contains("· current"), "the live chat's row is marked: {s}");
        assert!(s.contains("+ new chat"), "{s}");
        // Rows and the header button register click regions.
        assert!(app
            .click_regions
            .iter()
            .any(|r| r.target == ClickTarget::SessionList));
        assert!(app
            .click_regions
            .iter()
            .any(|r| r.target == ClickTarget::SessionsNew));
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
    fn theme_cycle_starts_at_ember_and_visits_all() {
        let mut app = test_app();
        assert_eq!(app.config.ui.theme, ThemeMode::Ember, "ember is the default");
        for expected in [
            ThemeMode::Dark,
            ThemeMode::Neon,
            ThemeMode::NeonPink,
            ThemeMode::Sage,
            ThemeMode::Ember,
        ] {
            app.toggle_theme();
            assert_eq!(app.config.ui.theme, expected);
        }
    }

    #[test]
    fn approval_mode_cycles_via_slash_backtab_and_statusbar_click() {
        let mut app = test_app();
        assert_eq!(app.config.agent.approval(), ApprovalMode::Auto);

        // /mode with an argument sets the mode directly.
        typ(&mut app, "/mode ask");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.config.agent.approval(), ApprovalMode::AskPermission);

        // /mode with no argument cycles (ask → wraps to always).
        typ(&mut app, "/mode");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.config.agent.approval(), ApprovalMode::AlwaysApprove);

        // Shift+Tab cycles from chat.
        assert_eq!(app.mode, Mode::Chat);
        app.handle_key(key(KeyCode::BackTab));
        assert_eq!(app.config.agent.approval(), ApprovalMode::Auto);

        // An explicit choice must persist — and survive a config reload.
        let loaded = Config::load(&app.paths).unwrap();
        assert_eq!(loaded.agent.approval(), ApprovalMode::Auto);

        // The status bar registers a clickable `approvals` cluster; clicking
        // it cycles too.
        use ratatui::{backend::TestBackend, Terminal};
        let mut terminal = Terminal::new(TestBackend::new(100, 20)).unwrap();
        terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
        let region = app
            .click_regions
            .iter()
            .find(|r| r.target == ClickTarget::ApprovalCycle)
            .copied()
            .expect("status bar exposes the approvals cluster");
        app.handle_left_click(region.rect.x, region.rect.y);
        assert_eq!(app.config.agent.approval(), ApprovalMode::ApproveEdits);
    }

    #[test]
    fn all_switcher_themes_render_and_sage_paints_its_background() {
        use ratatui::style::Color;
        use ratatui::{backend::TestBackend, Terminal};
        let mut app = test_app();
        for m in ThemeMode::SWITCHER {
            app.set_theme(m);
            let mut terminal = Terminal::new(TestBackend::new(90, 24)).unwrap();
            terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
            if m == ThemeMode::Sage {
                let buf = terminal.backend().buffer().clone();
                let (r, g, b) = app.theme.token_rgb(localcode_core::theme::ThemeToken::Bg);
                assert_eq!(buf[(0, 0)].bg, Color::Rgb(r, g, b), "sage bg fills the frame");
            }
        }
    }

    #[test]
    fn exact_command_name_outranks_prefix_collisions() {
        let mut app = test_app();
        // "/mode" collides with "/models" on substring; the exact name must
        // rank first so Enter runs it.
        typ(&mut app, "/mode");
        let items = app.palette_items();
        assert!(items[0].starts_with("/mode "), "exact match first: {items:?}");
        // A pure prefix query keeps catalog order among equals.
        app.clear_input();
        typ(&mut app, "/mo");
        let items = app.palette_items();
        assert!(items[0].starts_with("/models"), "{items:?}");
    }

    #[test]
    fn unknown_approval_mode_is_reported_not_applied() {
        let mut app = test_app();
        typ(&mut app, "/mode bogus");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.config.agent.approval(), ApprovalMode::Auto);
        assert!(app.status_is_error);
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
    fn thinking_streams_and_tools_toggle_verbose_detail() {
        let mut app = test_app();
        app.apply_agent_event(AgentEvent::ThinkingDelta("plan A".into()));
        app.apply_agent_event(AgentEvent::ThinkingDelta(" then B".into()));
        app.apply_agent_event(AgentEvent::Delta("answer".into()));
        app.apply_agent_event(AgentEvent::MessageComplete);
        app.apply_agent_event(AgentEvent::ToolStarted {
            name: "read".into(),
            args_preview: "path=a.txt".into(),
        });
        app.apply_agent_event(AgentEvent::ToolFinished {
            name: "read".into(),
            ok: true,
            summary: "hello  · 5 chars".into(),
            args: "path: a.txt".into(),
            output: "hello".into(),
        });

        let thinking = app
            .coding_transcript
            .iter()
            .find(|e| e.kind == EntryKind::Thinking)
            .expect("thinking entry");
        assert_eq!(thinking.text, "plan A then B");
        assert!(!thinking.live);
        assert!(thinking.expanded);
        assert!(thinking.can_toggle());

        let tool = app
            .coding_transcript
            .iter()
            .find(|e| e.kind == EntryKind::Tool)
            .expect("tool entry");
        assert!(tool.text.contains("read"));
        assert!(tool.text.contains('▸'));
        assert!(!tool.expanded);
        assert!(tool.detail.as_ref().unwrap().contains("hello"));
        assert!(tool.can_toggle());

        // Toggle tool verbose body on.
        let tool_idx = app
            .coding_transcript
            .iter()
            .position(|e| e.kind == EntryKind::Tool)
            .unwrap();
        app.coding_transcript[tool_idx].toggle_expanded();
        assert!(app.coding_transcript[tool_idx].expanded);

        // Toggle thinking collapsed.
        let think_idx = app
            .coding_transcript
            .iter()
            .position(|e| e.kind == EntryKind::Thinking)
            .unwrap();
        app.coding_transcript[think_idx].toggle_expanded();
        assert!(!app.coding_transcript[think_idx].expanded);

        let screen = render_to_string(&mut app, 100, 30);
        assert!(
            screen.contains("thinking") && screen.contains("chars"),
            "collapsed thinking summary should render: {screen}"
        );
        assert!(
            screen.contains("read") && screen.contains("output"),
            "expanded tool detail should render: {screen}"
        );
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
            Mode::Sessions,
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
    fn modified_enter_makes_a_multiline_composer_inside_the_bordered_omnibar() {
        let mut app = test_app();
        typ(&mut app, "first");
        // Shift+Enter inserts a newline instead of submitting; Ctrl+Enter and
        // Ctrl+J are the fallbacks for terminals that don't report Shift+Enter.
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
        typ(&mut app, "second");
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL));
        typ(&mut app, "third");
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL));
        typ(&mut app, "fourth");
        assert_eq!(app.coding_input, "first\nsecond\nthird\nfourth");

        let screen = render_to_string(&mut app, 100, 30);
        assert!(screen.contains("second") && screen.contains("fourth"));
        // The omnibar is a full pseudographic box: its bottom border — not the
        // input row — is the last thing on screen.
        let rows: Vec<&str> = screen.lines().collect();
        let last = rows.iter().rev().find(|r| !r.trim().is_empty()).unwrap();
        assert!(last.contains('╰') && last.contains('╯'), "bordered omnibar bottom: {last:?}");
        assert!(!last.contains('❯'), "the input row must not be the terminal's last line");
    }

    #[test]
    fn paste_inserts_multiline_text_without_submitting() {
        let mut app = test_app();
        typ(&mut app, "start ");
        app.handle_paste("line1\r\nline2");
        assert_eq!(app.coding_input, "start line1\nline2");
        assert!(
            app.coding_transcript.iter().all(|e| e.kind != EntryKind::You),
            "a pasted newline must not submit the turn"
        );
    }

    #[test]
    fn empty_omnibar_shows_a_visible_caret() {
        use ratatui::style::Modifier;
        use ratatui::{backend::TestBackend, Terminal};
        let mut app = test_app();
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
        let buf = terminal.backend().buffer().clone();
        let reversed = (0..buf.area.height)
            .flat_map(|y| (0..buf.area.width).map(move |x| (x, y)))
            .filter(|&(x, y)| buf[(x, y)].modifier.contains(Modifier::REVERSED))
            .count();
        assert!(reversed >= 1, "a caret block is drawn even with no input");
    }

    #[test]
    fn settings_toggle_tool_and_edit_system_prompt_persist() {
        let mut app = test_app();
        typ(&mut app, "/settings");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.mode, Mode::Settings);

        // Toggle the bash tool off via its row action.
        let toggle = app
            .settings_rows()
            .into_iter()
            .find_map(|r| match r.action {
                Some(SettingAction::ToggleTool(i)) if r.label == "bash" => {
                    Some(SettingAction::ToggleTool(i))
                }
                _ => None,
            })
            .expect("a bash tool row");
        app.activate_setting(toggle);
        assert!(app.config.agent.disabled_tools.iter().any(|t| t == "bash"));

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
    fn settings_expose_skills_agents_md_and_prompt() {
        let mut app = test_app();
        app.mode = Mode::Settings;
        let rows = app.settings_rows();
        let labels: Vec<&str> = rows.iter().map(|r| r.label.as_str()).collect();
        for expected in [
            "system prompt",
            "use AGENTS.md",
            "skills dir",
            "shell sandbox",
            "accept local assistant",
            "prefer local",
            "auto-handle errors",
        ] {
            assert!(labels.contains(&expected), "missing settings row: {expected}");
        }
        // The bundled sample skill shows up as an enable/disable toggle.
        assert!(rows
            .iter()
            .any(|r| matches!(r.kind, SettingsRowKind::Toggle(_)) && r.label.contains("localcode-doctor")));
        // Every built-in tool is listed as a toggle.
        assert!(rows
            .iter()
            .any(|r| matches!(r.kind, SettingsRowKind::Toggle(_)) && r.label == "read"));
    }

    #[test]
    fn deploy_fields_are_backend_aware() {
        let mut app = test_app();
        // Launched servers expose their launch flags; Ollama (shared server we
        // don't spawn) exposes only its model-level knobs — no host/port.
        app.deploy_backend = BackendKind::Vllm;
        assert_eq!(
            app.deploy_fields(),
            vec![
                DeployField::Context,
                DeployField::Port,
                DeployField::GpuMemFraction,
                DeployField::TensorParallel,
                DeployField::Command,
            ]
        );
        app.deploy_backend = BackendKind::LlamaCpp;
        assert_eq!(
            app.deploy_fields(),
            vec![
                DeployField::Context,
                DeployField::Port,
                DeployField::GpuLayers,
                DeployField::Command,
            ]
        );
        // Ollama is a shared server we don't spawn — no launch command to edit.
        app.deploy_backend = BackendKind::Ollama;
        assert_eq!(app.deploy_fields(), vec![DeployField::Context, DeployField::GpuLayers]);
        assert!(!app.deploy_fields().contains(&DeployField::Port));
        assert!(!app.deploy_fields().contains(&DeployField::Command));
    }

    #[test]
    fn editing_context_field_captures_keys_and_commits() {
        let mut app = test_app();
        app.mode = Mode::Models;
        app.begin_deploy_edit(DeployField::Context);
        assert!(app.deploy_editing.is_some());
        // While editing in Models mode, keys route to the field buffer (not the
        // always-focused omnibar). Clear the seeded old value, then type a new one.
        app.deploy_field_edit.clear();
        for c in "16384".chars() {
            app.handle_key(key(KeyCode::Char(c)));
        }
        assert_eq!(app.deploy_field_edit, "16384");
        app.handle_key(key(KeyCode::Enter));
        assert!(app.deploy_editing.is_none());
        assert_eq!(app.deploy_ctx, 16384);
        // Typing into the field must not leak into the omnibar / model search.
        assert!(app.coding_input.is_empty());
    }

    #[test]
    fn command_field_edits_the_tail_and_sets_override() {
        // The reported bug: the launch command couldn't be edited. It's the only
        // field long enough to overflow the panel, so a caret past the right edge
        // made edits look dead. Editing it must still update the override.
        let mut app = test_app();
        app.mode = Mode::Models;
        app.deploy_backend = BackendKind::LlamaCpp;
        app.deploy_command_override = Some("llama-server --port 8080".into());
        app.begin_deploy_edit(DeployField::Command);
        // Caret seeds at the end of the command.
        assert_eq!(app.deploy_field_cursor(), "llama-server --port 8080".chars().count());
        // Retype the trailing port 8080 -> 9090.
        for _ in 0..4 {
            app.handle_key(key(KeyCode::Backspace));
        }
        for c in "9090".chars() {
            app.handle_key(key(KeyCode::Char(c)));
        }
        assert_eq!(app.deploy_field_edit_buf(), "llama-server --port 9090");
        app.handle_key(key(KeyCode::Enter));
        assert!(app.deploy_editing.is_none());
        assert_eq!(
            app.deploy_command_override.as_deref(),
            Some("llama-server --port 9090")
        );
    }

    #[test]
    fn clicking_the_command_row_edits_at_the_clicked_char() {
        let mut app = test_app();
        app.mode = Mode::Models;
        app.deploy_backend = BackendKind::LlamaCpp;
        app.deploy_command_override = Some("llama-server --port 8080".into());
        // First click opens the edit with the caret under the clicked column
        // (13 chars into the value = the first '-' of "--port").
        app.click_deploy_field(DeployField::Command, DEPLOY_LABEL_W + 13);
        assert_eq!(app.deploy_editing, Some(DeployField::Command));
        assert_eq!(app.deploy_field_cursor(), 13);
        app.handle_key(key(KeyCode::Char('X')));
        assert_eq!(app.deploy_field_edit_buf(), "llama-server X--port 8080");
        // A click on the label column keeps caret and buffer as they are.
        app.click_deploy_field(DeployField::Command, 3);
        assert_eq!(app.deploy_field_cursor(), 14);
        // A click mid-edit moves the caret without re-seeding the buffer —
        // the typed-but-uncommitted 'X' must survive.
        app.click_deploy_field(DeployField::Command, DEPLOY_LABEL_W);
        assert_eq!(app.deploy_field_edit_buf(), "llama-server X--port 8080");
        assert_eq!(app.deploy_field_cursor(), 0);
        // Past the end of the value: caret parks after the last char.
        app.click_deploy_field(DeployField::Command, DEPLOY_LABEL_W + 500);
        assert_eq!(app.deploy_field_cursor(), 25);
        // With the value window scrolled, the click maps through the offset.
        app.deploy_field_scroll = 10;
        app.click_deploy_field(DeployField::Command, DEPLOY_LABEL_W + 2);
        assert_eq!(app.deploy_field_cursor(), 12);
    }

    #[test]
    fn editing_the_command_draws_a_caret_cell() {
        use ratatui::style::Modifier;
        use ratatui::{backend::TestBackend, Terminal};
        let mut app = test_app();
        app.mode = Mode::Models;
        app.deploy_backend = BackendKind::LlamaCpp;
        app.model_detail = Some(ModelDetail {
            summary: ModelSummary {
                id: "acme/coder-7b".into(),
                author: None,
                pipeline_tag: None,
                tags: vec![],
                likes: None,
                downloads: None,
                last_modified: None,
                private: None,
                gated: None,
            },
            siblings: vec![],
            card_data: None,
            sha: None,
            card_markdown: None,
            license: None,
            parameter_size: None,
            quants: vec![],
        });
        app.deploy_command_override = Some("llama-server --port 8080".into());
        // Click 13 value-cells in: the caret must land on the first '-' of
        // "--port" and be drawn as a REVERSED cell right there.
        app.click_deploy_field(DeployField::Command, DEPLOY_LABEL_W + 13);
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
        let buf = terminal.backend().buffer().clone();
        let row_text = |y: u16| -> String {
            (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect()
        };
        let y = (0..buf.area.height)
            .find(|&y| row_text(y).contains("llama-server"))
            .expect("the command row is on screen while editing");
        let carets: Vec<u16> = (0..buf.area.width)
            .filter(|&x| buf[(x, y)].modifier.contains(Modifier::REVERSED))
            .collect();
        assert_eq!(carets.len(), 1, "exactly one caret cell in the command row");
        assert_eq!(
            buf[(carets[0], y)].symbol(),
            "-",
            "caret sits under the clicked char"
        );
    }

    #[test]
    fn caret_from_col_maps_cells_to_chars() {
        assert_eq!(caret_from_col("abc", 0, 0), 0);
        assert_eq!(caret_from_col("abc", 0, 2), 2);
        // Past the end → append position; a scrolled window offsets the walk.
        assert_eq!(caret_from_col("abc", 0, 99), 3);
        assert_eq!(caret_from_col("abcdef", 2, 1), 3);
        // Both cells of a double-width glyph select the glyph itself.
        assert_eq!(caret_from_col("日本", 0, 0), 0);
        assert_eq!(caret_from_col("日本", 0, 1), 0);
        assert_eq!(caret_from_col("日本", 0, 2), 1);
        // Scroll clamped to the buffer, degenerate input safe.
        assert_eq!(caret_from_col("ab", 99, 0), 2);
        assert_eq!(caret_from_col("", 0, 5), 0);
    }

    fn sample_download(model_id: &str, status: DownloadStatus) -> DownloadState {
        DownloadState {
            model_id: model_id.into(),
            quantization: Some("Q4_K_M".into()),
            backend: BackendKind::LlamaCpp,
            files: vec!["m.gguf".into()],
            download_urls: vec!["https://h/m.gguf".into()],
            total_bytes: 100,
            downloaded_bytes: 40,
            status,
            message: "…".into(),
            primary_file: None,
            pid: 0,
            updated_unix: now_unix(),
        }
    }

    #[test]
    fn deploy_panel_renders_background_download() {
        let mut app = test_app();
        app.mode = Mode::Models;
        app.deploy_backend = BackendKind::LlamaCpp;
        app.model_detail = Some(detail_with_quant("org/x", "Q4_K_M", "x-Q4_K_M.gguf"));
        app.selected_quant = Some("Q4_K_M".into());

        // A live download shows progress and the "safe to quit" note.
        app.active_downloads = vec![sample_download("org/x", DownloadStatus::Downloading)];
        let s = render_to_string(&mut app, 120, 40);
        assert!(s.contains("downloading"), "expected progress line, got:\n{s}");

        // A failed download surfaces the warning and a resume button.
        app.active_downloads = vec![sample_download("org/x", DownloadStatus::Failed)];
        let s = render_to_string(&mut app, 120, 40);
        assert!(s.contains("download failed"), "{s}");
        assert!(s.contains("resume"), "expected a resume button, got:\n{s}");
    }

    #[test]
    fn active_download_tracked_and_completed_one_consumed() {
        let mut app = test_app();
        let cache = app.paths.models_cache.clone();
        // One in-progress and one completed download, each with a state file.
        let dir_a = model_dir(&cache, "org/a");
        std::fs::create_dir_all(&dir_a).unwrap();
        write_state(&dir_a, &sample_download("org/a", DownloadStatus::Downloading));
        let dir_b = model_dir(&cache, "org/b");
        std::fs::create_dir_all(&dir_b).unwrap();
        write_state(&dir_b, &sample_download("org/b", DownloadStatus::Completed));

        app.refresh_active_downloads();

        // The in-progress one is tracked and its state file preserved.
        assert!(app.download_for("org/a").is_some());
        assert!(localcode_backends::read_state(&dir_a).is_some());
        // The completed one is consumed: dropped from the active list and its
        // state file cleared so it stops showing as "downloading".
        assert!(app.download_for("org/b").is_none());
        assert!(localcode_backends::read_state(&dir_b).is_none());
    }

    #[test]
    fn command_field_caret_moves_and_edits_mid_string() {
        let mut app = test_app();
        app.mode = Mode::Models;
        app.deploy_backend = BackendKind::LlamaCpp;
        app.deploy_command_override = Some("abd".into());
        app.begin_deploy_edit(DeployField::Command);
        // ← between 'b' and 'd', insert 'c' -> "abcd" (mid-string edit).
        app.handle_key(key(KeyCode::Left));
        app.handle_key(key(KeyCode::Char('c')));
        assert_eq!(app.deploy_field_edit_buf(), "abcd");
        assert_eq!(app.deploy_field_cursor(), 3);
        // Home + Delete drops the leading char.
        app.handle_key(key(KeyCode::Home));
        app.handle_key(key(KeyCode::Delete));
        assert_eq!(app.deploy_field_edit_buf(), "bcd");
        assert_eq!(app.deploy_field_cursor(), 0);
        // Backspace at column 0 is a no-op (doesn't underflow).
        app.handle_key(key(KeyCode::Backspace));
        assert_eq!(app.deploy_field_edit_buf(), "bcd");
        assert_eq!(app.deploy_field_cursor(), 0);
    }

    #[test]
    fn blank_commit_resets_to_default_and_bad_value_is_rejected() {
        let mut app = test_app();
        app.deploy_ctx = 4096;
        app.begin_deploy_edit(DeployField::Context);
        app.deploy_field_edit.clear();
        app.commit_deploy_field();
        assert_eq!(app.deploy_ctx, DEFAULT_DEPLOY_CTX);

        // Out-of-range GPU fraction is rejected and leaves the field unset.
        app.begin_deploy_edit(DeployField::GpuMemFraction);
        app.deploy_field_edit = "2.0".to_string();
        app.commit_deploy_field();
        assert!(app.deploy_gpu_frac.is_none());
        assert!(app.status_is_error);

        // A valid fraction is stored.
        app.begin_deploy_edit(DeployField::GpuMemFraction);
        app.deploy_field_edit = "0.85".to_string();
        app.commit_deploy_field();
        assert_eq!(app.deploy_gpu_frac, Some(0.85));
    }

    #[test]
    fn esc_in_models_does_not_cancel_deploy_but_leaves_mode() {
        let mut app = test_app();
        app.mode = Mode::Models;
        // No deploy running and not editing: Esc leaves Models for Chat rather
        // than being swallowed (deploy is cancelled only via its button).
        app.handle_key(key(KeyCode::Esc));
        assert_eq!(app.mode, Mode::Chat);
    }

    /// A test app whose agent workspace is a temp dir seeded with `files`.
    fn workspace_app(files: &[&str]) -> (App, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        for f in files {
            let p = dir.path().join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, format!("content of {f}")).unwrap();
        }
        let mut app = test_app();
        app.config.agent.workspace_root = Some(dir.path().to_string_lossy().to_string());
        (app, dir)
    }

    #[test]
    fn at_opens_file_picker_and_enter_completes_the_path() {
        let (mut app, _dir) = workspace_app(&["src/main.rs", "src/lib.rs", "README.md"]);
        typ(&mut app, "explain @ma");
        assert!(app.at_picker_active());
        let items = app.at_matches();
        assert!(items.iter().any(|p| p == "src/main.rs"), "{items:?}");
        // Enter completes the token in place instead of submitting the turn.
        app.at_selected = items.iter().position(|p| p == "src/main.rs").unwrap();
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.coding_input, "explain @src/main.rs ");
        assert!(!app.at_picker_active(), "picker closes after completion");
        assert!(
            app.coding_transcript.iter().all(|e| e.kind != EntryKind::You),
            "completing must not submit"
        );
    }

    #[test]
    fn at_esc_dismisses_picker_but_typing_reopens_it() {
        let (mut app, _dir) = workspace_app(&["alpha.rs", "beta.rs"]);
        typ(&mut app, "@");
        assert!(app.at_picker_active());
        app.handle_key(key(KeyCode::Esc));
        assert!(!app.at_picker_active(), "Esc closes the picker");
        assert_eq!(app.coding_input, "@", "Esc keeps the typed text");
        typ(&mut app, "a");
        assert!(app.at_picker_active(), "typing reopens it");
    }

    #[test]
    fn at_references_attach_file_contents_to_the_prompt() {
        let (mut app, _dir) = workspace_app(&["src/main.rs"]);
        let (prompt, attached) = app.expand_at_context("explain @src/main.rs please");
        assert_eq!(attached, vec!["src/main.rs".to_string()]);
        assert!(prompt.starts_with("explain @src/main.rs please"));
        assert!(prompt.contains("content of src/main.rs"));
        // A reference that resolves to nothing leaves the prompt untouched.
        let (p2, a2) = app.expand_at_context("see @nope.txt");
        assert_eq!(p2, "see @nope.txt");
        assert!(a2.is_empty());
    }

    #[test]
    fn wheel_moves_the_command_palette_selection() {
        let mut app = test_app();
        typ(&mut app, "/");
        assert_eq!(app.palette_selected, 0);
        app.wheel_scroll_at(3, 0, 0);
        app.wheel_scroll_at(3, 0, 0);
        assert_eq!(app.palette_selected, 2, "wheel-down walks the palette");
        app.wheel_scroll_at(-3, 0, 0);
        assert_eq!(app.palette_selected, 1, "wheel-up walks back");
    }

    #[test]
    fn command_palette_docks_directly_above_the_omnibar() {
        let mut app = test_app();
        typ(&mut app, "/models");
        let screen = render_to_string(&mut app, 100, 30);
        let rows: Vec<&str> = screen.lines().collect();
        // The row showing the command's catalog description is the palette row
        // (the transcript's welcome text also mentions "/models").
        let cmd_row = rows
            .iter()
            .position(|r| r.contains("search & deploy HuggingFace models"))
            .expect("palette row on screen");
        let prompt_row = rows.iter().position(|r| r.contains('❯')).expect("input row");
        assert!(
            cmd_row < prompt_row && prompt_row - cmd_row <= 3,
            "palette docks above the omnibar (palette row {cmd_row}, input row {prompt_row})"
        );
    }

    #[test]
    fn hovering_a_theme_dot_reveals_its_name() {
        let mut app = test_app();
        let screen = render_to_string(&mut app, 120, 40);
        assert!(!screen.contains("neon"), "theme names stay hidden at rest");
        let region = app
            .click_regions
            .iter()
            .find(|r| r.target == ClickTarget::Theme(ThemeMode::Neon))
            .copied()
            .expect("a swatch dot per theme");
        app.hover = Some((region.rect.x, region.rect.y));
        let screen = render_to_string(&mut app, 120, 40);
        assert!(screen.contains("neon"), "hovering a dot reveals its name");
    }

    #[test]
    fn status_dashboard_expands_on_hover_and_pins_on_click() {
        use ratatui::{backend::TestBackend, Terminal};

        let mut app = test_app();
        app.status_logs = (1..=8).map(|i| format!("log-line-{i}")).collect();
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();

        // Collapsed: 3 content rows → outer height 5 (borders included).
        assert_eq!(app.status_bar_rect.height, 5, "collapsed dashboard is 3 content lines");
        assert!(!app.status_expanded());
        let collapsed_h = app.status_bar_rect.height;

        // Hover over the bar expands it.
        app.hover = Some((app.status_bar_rect.x + 2, app.status_bar_rect.y + 1));
        assert!(app.status_expanded());
        terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
        assert_eq!(
            app.status_bar_rect.height, 12,
            "expanded dashboard is 10 content lines"
        );
        let screen = render_to_string(&mut app, 120, 40);
        assert!(screen.contains("log-line-8"), "expanded view shows latest logs");
        assert!(screen.contains("log-line-1"), "expanded view shows older log lines");

        // Click pins so it stays open after the mouse leaves.
        let toggle = app
            .click_regions
            .iter()
            .find(|r| r.target == ClickTarget::StatusToggle)
            .copied()
            .expect("status bar registers StatusToggle");
        app.handle_left_click(toggle.rect.x + 1, toggle.rect.y + 1);
        assert!(app.status_pinned);
        app.hover = None;
        assert!(app.status_expanded(), "pinned stays expanded without hover");
        terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
        assert_eq!(app.status_bar_rect.height, 12);

        // Second click collapses.
        app.handle_left_click(toggle.rect.x + 1, toggle.rect.y + 1);
        assert!(!app.status_pinned);
        assert!(!app.status_expanded());
        terminal.draw(|f| crate::ui::draw(f, &mut app)).unwrap();
        assert_eq!(app.status_bar_rect.height, collapsed_h);

        // Metrics lines include tok/s and energy when present.
        app.tokens_per_sec = Some(48.0);
        app.gpu.devices = vec![localcode_gpu::GpuDevice {
            index: 0,
            name: "Test".into(),
            total_vram_bytes: 24 * 1024 * 1024 * 1024,
            free_vram_bytes: 10 * 1024 * 1024 * 1024,
            driver_version: None,
            backend_affinity: vec!["cuda".into()],
            temperature_c: Some(60),
            utilization_pct: Some(20),
            power_draw_w: Some(145.0),
        }];
        let screen = render_to_string(&mut app, 120, 40);
        assert!(screen.contains("tok/s"), "dashboard shows tokens per second");
        assert!(screen.contains("48"), "dashboard shows tok/s value");
        assert!(screen.contains("energy"), "dashboard shows GPU energy");
        assert!(screen.contains("145W"), "dashboard shows power draw");
    }

    /// While the agent is working but has not streamed yet, the status-dashboard
    /// spinner + log tail move next to the user message (3 lines), then clear
    /// once thinking/agent text arrives.
    #[test]
    fn wait_state_shows_logs_and_spinner_next_to_user_message() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _enter = rt.enter();

        let mut app = test_app();
        app.coding_transcript.push(TranscriptEntry::new(
            EntryKind::You,
            "fix the flaky suite",
        ));
        app.status_logs = vec![
            "wait-log-alpha".into(),
            "wait-log-beta".into(),
            "wait-log-gamma".into(),
        ];
        let handle = tokio::spawn(async {
            std::future::pending::<()>().await;
        });
        app.busy = Some(Busy {
            kind: BusyKind::Coding,
            label: "Agent working".into(),
            started: Instant::now(),
            handle,
        });

        let screen = render_to_string(&mut app, 120, 40);
        assert!(
            screen.contains("fix the flaky suite"),
            "user prompt is visible"
        );
        assert!(
            screen.contains("wait-log-alpha")
                && screen.contains("wait-log-beta")
                && screen.contains("wait-log-gamma"),
            "three backend log lines sit under the user message while waiting:\n{screen}"
        );
        // Braille spinner frames (any one is enough — tick index varies).
        let has_spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            .iter()
            .any(|c| screen.contains(*c));
        assert!(has_spinner, "spinner rides next to the user message while waiting");

        // Collapsed status is metrics-only while logs are redirected (height 4
        // = 2 content + top/bottom border), not the usual 5.
        assert_eq!(
            app.status_bar_rect.height, 4,
            "collapsed status drops the log row while wait logs live in chat"
        );

        // Agent starts responding → wait-under-prompt logs clear; spinner
        // rides the live thinking line; status dashboard regains its log row
        // (so a single latest log may reappear there — that is expected).
        app.coding_transcript.push(TranscriptEntry {
            kind: EntryKind::Thinking,
            text: "considering options".into(),
            live: true,
            detail: None,
            expanded: true,
        });
        let screen = render_to_string(&mut app, 120, 40);
        assert!(
            screen.contains("considering options"),
            "agent thinking is shown"
        );
        // The three-line wait block under the user prompt is gone: older logs
        // (alpha/beta) only lived there, while gamma may reappear as the
        // status dashboard's single latest-log line.
        assert!(
            !screen.contains("wait-log-alpha") && !screen.contains("wait-log-beta"),
            "multi-line wait log tail under the user message is gone:\n{screen}"
        );
        let has_spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            .iter()
            .any(|c| screen.contains(*c));
        assert!(has_spinner, "spinner moves onto the live thinking line");
        assert_eq!(
            app.status_bar_rect.height, 5,
            "status log row returns after wait ends"
        );

        if let Some(b) = app.busy.take() {
            b.handle.abort();
        }
    }

    fn summary(id: &str) -> ModelSummary {
        ModelSummary {
            id: id.into(),
            author: None,
            pipeline_tag: None,
            tags: vec![],
            likes: None,
            downloads: None,
            last_modified: None,
            private: None,
            gated: None,
        }
    }

    fn detail_with_quant(id: &str, label: &str, filename: &str) -> ModelDetail {
        ModelDetail {
            summary: summary(id),
            siblings: vec![],
            card_data: None,
            sha: None,
            card_markdown: None,
            license: None,
            parameter_size: None,
            quants: vec![localcode_hf::QuantGroup {
                label: label.into(),
                files: vec![localcode_hf::QuantFile {
                    filename: filename.into(),
                    size: Some(2048),
                    quant_label: label.into(),
                }],
                total_size: 2048,
                known: true,
            }],
        }
    }

    /// A fresh GGUF model must auto-select llama.cpp even when the panel was on
    /// vLLM — this is the "GGUF never lands on vLLM" guarantee at browse time.
    #[test]
    fn fresh_gguf_model_auto_selects_llamacpp() {
        let mut app = test_app();
        app.deploy_backend = BackendKind::Vllm;
        app.model_detail = Some(detail_with_quant("org/g", "Q4_K_M", "g-Q4_K_M.gguf"));
        app.selected_quant = Some("Q4_K_M".into());
        app.run_deploy_preset(PresetMode::Fresh);
        assert_eq!(app.deploy_backend, BackendKind::LlamaCpp);
    }

    /// A fresh safetensors model routes to vLLM, not a GGUF backend.
    #[test]
    fn fresh_safetensors_model_auto_selects_vllm() {
        let mut app = test_app();
        app.deploy_backend = BackendKind::Ollama;
        app.model_detail = Some(detail_with_quant("org/s", "FP16", "model.safetensors"));
        app.selected_quant = Some("FP16".into());
        app.run_deploy_preset(PresetMode::Fresh);
        assert_eq!(app.deploy_backend, BackendKind::Vllm);
    }

    /// Switching to a GGUF quant while pinned to vLLM corrects the backend, since
    /// vLLM cannot serve GGUF.
    #[test]
    fn quant_change_to_gguf_corrects_incompatible_backend() {
        let mut app = test_app();
        app.deploy_backend = BackendKind::Vllm;
        app.model_detail = Some(detail_with_quant("org/g", "Q4_K_M", "g-Q4_K_M.gguf"));
        app.selected_quant = Some("Q4_K_M".into());
        app.run_deploy_preset(PresetMode::QuantChanged);
        assert_eq!(app.deploy_backend, BackendKind::LlamaCpp);
    }

    /// A backend the user pinned is never auto-switched, even if it can't serve
    /// the format — the deploy-time guard surfaces the error instead.
    #[test]
    fn backend_pinned_mode_never_switches() {
        let mut app = test_app();
        app.deploy_backend = BackendKind::Vllm;
        app.model_detail = Some(detail_with_quant("org/g", "Q4_K_M", "g-Q4_K_M.gguf"));
        app.selected_quant = Some("Q4_K_M".into());
        app.run_deploy_preset(PresetMode::BackendPinned);
        assert_eq!(app.deploy_backend, BackendKind::Vllm);
    }

    /// A hand-edited context is not overwritten by the assistant preset.
    #[test]
    fn user_set_context_survives_preset() {
        let mut app = test_app();
        app.model_detail = Some(detail_with_quant("org/g", "Q4_K_M", "g-Q4_K_M.gguf"));
        app.selected_quant = Some("Q4_K_M".into());
        app.deploy_ctx = 5000;
        app.deploy_ctx_user_set = true;
        app.run_deploy_preset(PresetMode::Fresh);
        assert_eq!(app.deploy_ctx, 5000);
    }

    /// With the automatic path disabled (the default), browsing/quant changes
    /// leave the backend exactly as configured — nothing presets on its own.
    #[test]
    fn preset_disabled_leaves_backend_untouched() {
        let mut app = test_app();
        app.config.assistant.auto_deploy_hints = false;
        app.deploy_backend = BackendKind::Vllm;
        app.model_detail = Some(detail_with_quant("org/g", "Q4_K_M", "g-Q4_K_M.gguf"));
        app.selected_quant = Some("Q4_K_M".into());
        app.apply_deploy_preset(PresetMode::Fresh);
        assert_eq!(app.deploy_backend, BackendKind::Vllm);
    }

    /// The deploy panel's auto-config button applies the preset on demand even
    /// when the automatic path is off (the default) — it's the manual trigger.
    #[test]
    fn auto_config_button_presets_when_auto_disabled() {
        let mut app = test_app();
        assert!(
            !app.config.assistant.auto_deploy_hints,
            "auto-preset must be off by default"
        );
        app.deploy_backend = BackendKind::Vllm;
        app.model_detail = Some(detail_with_quant("org/g", "Q4_K_M", "g-Q4_K_M.gguf"));
        app.selected_quant = Some("Q4_K_M".into());
        app.auto_configure_deploy();
        assert_eq!(app.deploy_backend, BackendKind::LlamaCpp);
    }

    /// Scrolling the results list arms the debounced auto-load; once the
    /// deadline passes with the highlighted model already loaded, it disarms
    /// without spawning anything.
    #[test]
    fn scrolling_models_arms_detail_autoload() {
        let mut app = test_app();
        app.mode = Mode::Models;
        app.models = vec![summary("org/a"), summary("org/b")];

        assert!(app.detail_autoload_at.is_none());
        app.scroll_list_models(1);
        assert_eq!(app.model_selected, 1);
        assert!(
            app.detail_autoload_at.is_some(),
            "wheel scroll arms the follow-the-selection load"
        );

        // Deadline reached but the selected model is already on screen: the
        // timer disarms without a request (no busy, no in-flight id).
        app.model_detail = Some(detail_with_quant("org/b", "Q4_K_M", "b-Q4_K_M.gguf"));
        app.detail_autoload_at = Some(Instant::now());
        app.maybe_autoload_detail();
        assert!(app.detail_autoload_at.is_none());
        assert!(app.busy.is_none());
        assert!(app.detail_request.is_none());
    }

    /// A DetailDone for anything but the latest requested id is stale (the
    /// user scrolled on) and must not replace the pane or clear the in-flight
    /// marker; the matching id applies normally.
    #[test]
    fn stale_detail_results_are_dropped() {
        let mut app = test_app();
        app.detail_request = Some("org/current".into());

        app.bg_tx
            .send(BgMsg::DetailDone {
                id: "org/old".into(),
                auto: true,
                result: Ok(detail_with_quant("org/old", "Q4_K_M", "old.gguf")),
            })
            .unwrap();
        app.process_bg();
        assert!(app.model_detail.is_none(), "stale result must be dropped");
        assert_eq!(app.detail_request.as_deref(), Some("org/current"));

        app.bg_tx
            .send(BgMsg::DetailDone {
                id: "org/current".into(),
                auto: true,
                result: Ok(detail_with_quant("org/current", "Q4_K_M", "cur-Q4_K_M.gguf")),
            })
            .unwrap();
        app.process_bg();
        assert_eq!(
            app.model_detail.as_ref().map(|d| d.summary.id.as_str()),
            Some("org/current")
        );
        assert!(app.detail_request.is_none());
        assert_eq!(app.selected_quant.as_deref(), Some("Q4_K_M"));
    }

    /// With HF unreachable (no client), the detail card is synthesized from
    /// the weights on disk: same quant labels and real file sizes, so the
    /// deploy panel keeps working offline.
    #[test]
    fn offline_detail_synthesized_from_downloaded_weights() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let root = tempfile::tempdir().unwrap();
        let models = root.path();
        let dir = models.join(sanitize_model_dir("org/coder-7b"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("coder-Q4_K_M.gguf"), vec![0u8; 2048]).unwrap();

        let detail = rt
            .block_on(fetch_detail_or_offline(None, models, "org/coder-7b"))
            .expect("offline synthesis from downloaded weights");
        assert_eq!(detail.summary.id, "org/coder-7b");
        assert_eq!(detail.quants.len(), 1);
        assert_eq!(detail.quants[0].label, "Q4_K_M");
        assert_eq!(detail.quants[0].total_size, 2048);
        assert!(detail.card_markdown.unwrap_or_default().contains("Offline"));

        // Nothing on disk for an unknown model: the original error surfaces.
        assert!(rt
            .block_on(fetch_detail_or_offline(None, models, "org/absent"))
            .is_err());
    }

    /// A quant group is tagged downloaded only when every one of its files is
    /// on disk; repo subdirectories are matched by bare file name.
    #[test]
    fn quant_downloaded_requires_every_file() {
        let mut app = test_app();
        app.downloaded_models = vec![DownloadedModel {
            model_id: "org/coder-7b".into(),
            dir: PathBuf::from("/models/org_coder-7b"),
            total_bytes: 2048,
            quants: vec!["Q4_K_M".into()],
            files: vec!["coder-Q4_K_M.gguf".into()],
        }];

        let complete = detail_with_quant("org/coder-7b", "Q4_K_M", "sub/coder-Q4_K_M.gguf");
        assert!(app.quant_downloaded("org/coder-7b", &complete.quants[0]));

        let mut sharded = detail_with_quant("org/coder-7b", "Q6_K", "coder-Q6_K-00001-of-00002.gguf");
        sharded.quants[0].files.push(localcode_hf::QuantFile {
            filename: "coder-Q6_K-00002-of-00002.gguf".into(),
            size: Some(2048),
            quant_label: "Q6_K".into(),
        });
        assert!(
            !app.quant_downloaded("org/coder-7b", &sharded.quants[0]),
            "missing shard means not downloaded"
        );
        assert!(!app.quant_downloaded("org/other", &complete.quants[0]));
    }

    /// When the selected quant's weights are on disk, the deploy request pins
    /// `local_path` for GGUF-capable backends so no network is touched; vLLM
    /// keeps resolving the HF id itself.
    #[test]
    fn deploy_request_pins_local_path_for_downloaded_quant() {
        let mut app = test_app();
        let dir = app.paths.models_cache.join(sanitize_model_dir("org/coder-7b"));
        app.model_detail = Some(detail_with_quant("org/coder-7b", "Q4_K_M", "coder-Q4_K_M.gguf"));
        app.selected_quant = Some("Q4_K_M".into());
        app.downloaded_models = vec![DownloadedModel {
            model_id: "org/coder-7b".into(),
            dir: dir.clone(),
            total_bytes: 2048,
            quants: vec!["Q4_K_M".into()],
            files: vec!["coder-Q4_K_M.gguf".into()],
        }];

        app.deploy_backend = BackendKind::LlamaCpp;
        let req = app.build_deploy_request(false).expect("deploy request");
        assert_eq!(
            req.local_path.as_deref(),
            Some(dir.join("coder-Q4_K_M.gguf").display().to_string().as_str()),
            "downloaded quant deploys from the local file"
        );

        app.deploy_backend = BackendKind::Vllm;
        let req = app.build_deploy_request(false).expect("deploy request");
        assert!(req.local_path.is_none(), "vLLM resolves the HF id itself");

        // A quant that is not on disk keeps the download path.
        app.deploy_backend = BackendKind::LlamaCpp;
        app.downloaded_models.clear();
        let req = app.build_deploy_request(false).expect("deploy request");
        assert!(req.local_path.is_none());
    }
}
