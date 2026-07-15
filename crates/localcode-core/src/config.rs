use crate::error::{ErrorCode, LocalCodeError};
use crate::paths::AppPaths;
use crate::theme::ThemeMode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub type ConfigError = LocalCodeError;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub ui: UiConfig,
    #[serde(default)]
    pub registry: RegistryConfig,
    #[serde(default)]
    pub backends: BackendsConfig,
    #[serde(default)]
    pub assistant: AssistantConfig,
    #[serde(default)]
    pub agent: AgentConfig,
    #[serde(default)]
    pub cloud: CloudConfig,
    #[serde(default)]
    pub api: ApiConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub panes: PaneRatios,
    #[serde(default)]
    pub updates: UpdatesConfig,
    #[serde(default)]
    pub remote: RemoteConfig,
    #[serde(default)]
    pub models: ModelsConfig,
}

/// Read an env var, treating empty values as unset.
fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    #[serde(default)]
    pub theme: ThemeMode,
    #[serde(default = "default_true")]
    pub mouse: bool,
    #[serde(default = "default_true")]
    pub right_rail_hover_brightens: bool,
    /// Height (text rows) of the Coding composer. Clamped to 1..=10 at use.
    #[serde(default = "default_composer_rows")]
    pub composer_rows: u16,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: ThemeMode::default(),
            mouse: true,
            right_rail_hover_brightens: true,
            composer_rows: default_composer_rows(),
        }
    }
}

/// Where updates come from and whether to look for them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdatesConfig {
    /// Check for a new version in the background on TUI startup.
    #[serde(default = "default_true")]
    pub check_on_startup: bool,
    /// Git repository updates are pulled from (same source as the installer).
    #[serde(default = "default_repo_url")]
    pub repo_url: String,
    /// Branch the installer tracks.
    #[serde(default = "default_repo_branch")]
    pub branch: String,
    /// Source checkout used for self-update. Defaults to the installer's
    /// location; `LOCALCODE_INSTALL_DIR` env overrides at use time.
    #[serde(default)]
    pub install_dir: Option<String>,
    /// Alternate git remote URLs for self-update, tried in order after
    /// `repo_url` fails. Useful when github.com is unreachable from the machine
    /// running the update (mirror the repo to an internal git host).
    #[serde(default)]
    pub mirrors: Vec<String>,
}

impl Default for UpdatesConfig {
    fn default() -> Self {
        Self {
            check_on_startup: true,
            repo_url: default_repo_url(),
            branch: default_repo_branch(),
            install_dir: None,
            mirrors: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    #[serde(default = "default_hf_provider")]
    pub provider: String,
    #[serde(default = "default_hf_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_hf_api")]
    pub api_endpoint: String,
    #[serde(default = "default_hf_token_env")]
    pub token_env: String,
    /// Extra HuggingFace mirror web roots (e.g. `https://hf-mirror.com`), tried
    /// in order after `endpoint` and before the canonical `huggingface.co`
    /// fallback. Each api base is derived as `{root}/api`. Essential when the
    /// deploy target sits in an isolated network that can't reach HF directly.
    #[serde(default)]
    pub mirrors: Vec<String>,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            provider: default_hf_provider(),
            endpoint: default_hf_endpoint(),
            api_endpoint: default_hf_api(),
            token_env: default_hf_token_env(),
            mirrors: Vec::new(),
        }
    }
}

/// Model-catalogue preferences (favourites shown first in `/dash` and `/models`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelsConfig {
    /// Hugging Face model ids the user starred. Order is insertion order; the
    /// UI renders these above other models on the dashboard.
    #[serde(default)]
    pub favorites: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackendsConfig {
    #[serde(default)]
    pub default: DefaultBackend,
    #[serde(default)]
    pub ollama: OllamaConfig,
    #[serde(default)]
    pub llamacpp: LlamaCppConfig,
    #[serde(default)]
    pub vllm: VllmConfig,
    #[serde(default)]
    pub sglang: SglangConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultBackend {
    #[serde(default = "default_backend_kind")]
    pub kind: String,
}

impl Default for DefaultBackend {
    fn default() -> Self {
        Self {
            kind: default_backend_kind(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    #[serde(default = "default_ollama_url")]
    pub base_url: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: default_ollama_url(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppConfig {
    #[serde(default = "default_llama_bin")]
    pub bin: String,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_llama_port")]
    pub port: u16,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            bin: default_llama_bin(),
            host: default_host(),
            port: default_llama_port(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmConfig {
    #[serde(default = "default_vllm_bin")]
    pub bin: String,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_vllm_port")]
    pub port: u16,
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            bin: default_vllm_bin(),
            host: default_host(),
            port: default_vllm_port(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SglangConfig {
    #[serde(default = "default_sglang_bin")]
    pub bin: String,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_sglang_port")]
    pub port: u16,
}

impl Default for SglangConfig {
    fn default() -> Self {
        Self {
            bin: default_sglang_bin(),
            host: default_host(),
            port: default_sglang_port(),
        }
    }
}

/// Whether the user has accepted, declined, or not yet been asked about the
/// bundled local Bonsai assistant (`llama-server -m Bonsai-27B-Q1_0.gguf -ngl 99`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LocalAssistantPreference {
    /// First launch: show the install offer.
    #[default]
    NotPrompted,
    /// User declined; do not auto-offer again (can re-enable in Settings).
    Declined,
    /// User accepted; llama.cpp + model should be installed / kept ready.
    Accepted,
}

/// In-app / default-conversation assistant. Prefer the local Bonsai model
/// (`llama-server -m Bonsai-27B-Q1_0.gguf -ngl 99`) when installed; fall back
/// to a hosted OpenAI-compatible provider when configured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantConfig {
    /// Hosted provider id when not using the local Bonsai runtime:
    /// `openrouter` | `openai_compatible` | `self_hosted` | `custom` | `local`.
    #[serde(default = "default_assistant_provider")]
    pub provider: String,
    #[serde(default = "default_openrouter_url")]
    pub base_url: String,
    #[serde(default)]
    pub model: String,
    /// Env var name holding the API key (never store key in config).
    #[serde(default = "default_assistant_key_env")]
    pub api_key_env: String,
    /// Install decision for the bundled local Bonsai assistant.
    #[serde(default)]
    pub local_preference: LocalAssistantPreference,
    /// When true (default), use the local Bonsai runtime whenever it is ready,
    /// even if a hosted provider is also configured.
    #[serde(default = "default_true")]
    pub prefer_local: bool,
    /// Dedicated `llama-server` port for the assistant (avoids clashing with
    /// user deploys on the default 8080).
    #[serde(default = "default_assistant_port")]
    pub local_port: u16,
    /// Context length for the local assistant server.
    #[serde(default = "default_assistant_ctx")]
    pub local_context: u32,
    /// GPU layers for the local assistant (`-1` / large = offload all).
    #[serde(default = "default_assistant_ngl")]
    pub local_gpu_layers: i32,
    /// On structured errors, invoke the local assistant first (when ready)
    /// before relying only on the Fix/Retry UI.
    #[serde(default = "default_true")]
    pub auto_handle_errors: bool,
    /// Before deploy, read the HF model card and apply recommended server flags.
    #[serde(default = "default_true")]
    pub auto_deploy_hints: bool,
    /// Offer help when the user enters the app (transcript system line).
    #[serde(default = "default_true")]
    pub greet_on_startup: bool,
}

impl Default for AssistantConfig {
    fn default() -> Self {
        Self {
            provider: default_assistant_provider(),
            base_url: default_openrouter_url(),
            model: String::new(),
            api_key_env: default_assistant_key_env(),
            local_preference: LocalAssistantPreference::default(),
            prefer_local: true,
            local_port: default_assistant_port(),
            local_context: default_assistant_ctx(),
            local_gpu_layers: default_assistant_ngl(),
            auto_handle_errors: true,
            auto_deploy_hints: true,
            greet_on_startup: true,
        }
    }
}

/// How much the agent asks before running tools. Ordered from most permissive
/// to most careful; the TUI cycles through them in this order (Shift+Tab,
/// `/mode`, or the status-bar indicator).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalMode {
    /// Never ask — every tool call runs, destructive shell commands included.
    #[serde(alias = "always", alias = "yolo")]
    AlwaysApprove,
    /// Ask only for destructive shell commands (rm -rf, git reset --hard, …).
    #[default]
    Auto,
    /// Ask before anything that changes the workspace: file writes, patches,
    /// and every shell command. Reads and searches run freely.
    #[serde(alias = "edits")]
    ApproveEdits,
    /// Ask before every tool call, reads included.
    #[serde(alias = "ask")]
    AskPermission,
}

impl ApprovalMode {
    /// Cycle order (most permissive → most careful).
    pub const ALL: [ApprovalMode; 4] = [
        ApprovalMode::AlwaysApprove,
        ApprovalMode::Auto,
        ApprovalMode::ApproveEdits,
        ApprovalMode::AskPermission,
    ];

    /// Full name as the user asked for it ("always approve", "auto", …).
    pub fn label(self) -> &'static str {
        match self {
            ApprovalMode::AlwaysApprove => "always approve",
            ApprovalMode::Auto => "auto",
            ApprovalMode::ApproveEdits => "approve edits",
            ApprovalMode::AskPermission => "ask permission",
        }
    }

    /// Short tag for the status bar.
    pub fn tag(self) -> &'static str {
        match self {
            ApprovalMode::AlwaysApprove => "always",
            ApprovalMode::Auto => "auto",
            ApprovalMode::ApproveEdits => "edits",
            ApprovalMode::AskPermission => "ask",
        }
    }

    /// One-line explanation shown in Settings and the status line.
    pub fn describe(self) -> &'static str {
        match self {
            ApprovalMode::AlwaysApprove => "everything runs without asking",
            ApprovalMode::Auto => "asks only for destructive shell commands",
            ApprovalMode::ApproveEdits => "asks before file edits & shell commands",
            ApprovalMode::AskPermission => "asks before every tool call",
        }
    }

    /// The next mode in the cycle.
    pub fn next(self) -> Self {
        let all = Self::ALL;
        let i = all.iter().position(|m| *m == self).unwrap_or(1);
        all[(i + 1) % all.len()]
    }

    /// Parse a user-typed mode name (`/mode edits`). Accepts the tag, the full
    /// label (with spaces, dashes or underscores), and common synonyms.
    pub fn parse(s: &str) -> Option<Self> {
        let k: String = s
            .trim()
            .to_lowercase()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect();
        match k.as_str() {
            "alwaysapprove" | "always" | "yolo" | "never" | "none" => {
                Some(ApprovalMode::AlwaysApprove)
            }
            "auto" | "default" => Some(ApprovalMode::Auto),
            "approveedits" | "edits" | "edit" => Some(ApprovalMode::ApproveEdits),
            "askpermission" | "ask" | "askpermissions" | "all" => {
                Some(ApprovalMode::AskPermission)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Directory of skill folders (`*/SKILL.md`). Default: `<data>/skills`.
    #[serde(default)]
    pub skills_dir: Option<String>,
    /// How much the agent asks before running tools. See [`ApprovalMode`];
    /// read it through [`AgentConfig::approval`], which honors the legacy
    /// `confirm_destructive_tools` off-switch from older configs.
    #[serde(default)]
    pub approval_mode: ApprovalMode,
    /// Legacy switch superseded by `approval_mode`. Kept so old configs load:
    /// `false` (with `approval_mode` unset) still means "never ask".
    #[serde(default = "default_true")]
    pub confirm_destructive_tools: bool,
    /// Allow the Coding tab to fall back to the (cloud) assistant provider
    /// when no local runtime is deployed. Off by default: local-first.
    #[serde(default)]
    pub allow_cloud_fallback: bool,
    #[serde(default)]
    pub workspace_root: Option<String>,
    #[serde(default = "default_max_tools")]
    pub max_tool_rounds: u32,
    /// Stream model output token-by-token (SSE). Turn off for OpenAI-compatible
    /// servers that reject `stream: true` together with tools.
    #[serde(default = "default_true")]
    pub stream: bool,
    /// Custom system prompt. When set (non-empty), it replaces LocalCode's
    /// built-in agent preamble; workspace path, project AGENTS.md and available
    /// skills are still appended. Empty/None uses the built-in preamble.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Read a project `AGENTS.md` (from the workspace root) into the system
    /// prompt so the agent follows repo-specific instructions. On by default.
    #[serde(default = "default_true")]
    pub use_agents_md: bool,
    /// Tool names (e.g. `bash`) the agent is NOT allowed to use. Disabled
    /// tools are hidden from the model and refused if the model asks for them.
    #[serde(default)]
    pub disabled_tools: Vec<String>,
    /// Skill names to hide from the agent's system prompt / skill tool.
    #[serde(default)]
    pub disabled_skills: Vec<String>,
    /// Char budget for the message history sent to the model per request.
    /// Local models often run with small context windows; when a session grows
    /// past this, the oldest turns are compacted (or trimmed). The stored
    /// session keeps everything. 0 disables trimming/compaction budgets.
    #[serde(default = "default_history_chars")]
    pub max_history_chars: usize,
    /// Persist coding sessions to disk (JSONL under the data dir) so they can
    /// be resumed after a restart. On by default; the full history is always
    /// kept in the file even when the request view is trimmed or compacted.
    #[serde(default = "default_true")]
    pub sessions_enabled: bool,
    /// Override directory for session files. Default: `<data>/sessions`.
    #[serde(default)]
    pub sessions_dir: Option<String>,
    /// When history exceeds `max_history_chars`, summarize the older turns
    /// with the active model instead of silently dropping them. Falls back to
    /// plain trimming if summarization fails.
    #[serde(default = "default_true")]
    pub auto_compact: bool,
    /// How many chars of recent history stay verbatim after a compaction.
    #[serde(default = "default_compact_keep_chars")]
    pub compact_keep_recent_chars: usize,
    /// Confine `bash` to the workspace (cwd + path checks). On by default.
    #[serde(default = "default_true")]
    pub shell_sandbox: bool,
    /// Wall-clock cap for a single `bash` tool command, in seconds. Builds and
    /// test suites routinely need more than a couple of minutes; 0 uses the
    /// built-in default.
    #[serde(default = "default_bash_timeout_secs")]
    pub bash_timeout_secs: u64,
}

impl AgentConfig {
    /// Effective approval mode. Old configs that set
    /// `confirm_destructive_tools = false` (and never chose a mode) keep their
    /// "never ask" behavior.
    pub fn approval(&self) -> ApprovalMode {
        if self.approval_mode == ApprovalMode::Auto && !self.confirm_destructive_tools {
            ApprovalMode::AlwaysApprove
        } else {
            self.approval_mode
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            skills_dir: None,
            approval_mode: ApprovalMode::default(),
            confirm_destructive_tools: true,
            allow_cloud_fallback: false,
            workspace_root: None,
            max_tool_rounds: default_max_tools(),
            stream: true,
            system_prompt: None,
            use_agents_md: true,
            disabled_tools: Vec::new(),
            disabled_skills: Vec::new(),
            max_history_chars: default_history_chars(),
            sessions_enabled: true,
            sessions_dir: None,
            auto_compact: true,
            compact_keep_recent_chars: default_compact_keep_chars(),
            shell_sandbox: true,
            bash_timeout_secs: default_bash_timeout_secs(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CloudConfig {
    #[serde(default)]
    pub runpod: CloudProviderConfig,
    #[serde(default)]
    pub vast: CloudProviderConfig,
    #[serde(default)]
    pub akash: AkashConfig,
    #[serde(default = "default_spend_threshold")]
    pub spend_confirm_threshold_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CloudProviderConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub api_key_env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AkashConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_true")]
    pub managed_account: bool,
    #[serde(default)]
    pub api_key_env: String,
}

impl Default for AkashConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            managed_account: true,
            api_key_env: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    #[serde(default = "default_api_url")]
    pub base_url: String,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            base_url: default_api_url(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default = "default_true")]
    pub redact_secrets: bool,
    #[serde(default = "default_max_log_files")]
    pub max_files: u32,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            redact_secrets: true,
            max_files: default_max_log_files(),
        }
    }
}

/// Persisted pane ratios per view name (0.0–1.0 splits).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PaneRatios {
    #[serde(default)]
    pub views: HashMap<String, Vec<f32>>,
}

/// Remote GPU servers reachable over SSH. LocalCode connects, ensures a
/// backend (Ollama) runs there, and forwards its port back to localhost so the
/// agent codes against the remote GPU as if it were local.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RemoteConfig {
    #[serde(default)]
    pub servers: Vec<RemoteServer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteServer {
    /// Friendly label shown in the UI and used as the runtime name.
    pub name: String,
    pub host: String,
    #[serde(default = "default_ssh_port")]
    pub port: u16,
    pub username: String,
    /// SSH password, stored in plaintext so a saved server reconnects in one
    /// click (the AmneziaVPN desktop model). SECURITY: this file is on your
    /// local disk in the clear. Prefer `key_path` for key-based auth, or leave
    /// this empty and type the password per session in the TUI.
    #[serde(default)]
    pub password: String,
    /// Path to a private key file (OpenSSH format). Alternative to `password`.
    #[serde(default)]
    pub key_path: Option<String>,
    /// Backend to run remotely. Only "ollama" is wired end-to-end today.
    #[serde(default = "default_remote_backend")]
    pub backend: String,
    /// Port the remote backend listens on (Ollama = 11434).
    #[serde(default = "default_remote_backend_port")]
    pub remote_port: u16,
    /// Local port the SSH tunnel binds. 0 means "same as remote_port".
    #[serde(default)]
    pub local_port: u16,
    /// Attempt to connect automatically on TUI startup.
    #[serde(default)]
    pub auto_connect: bool,
    /// Provisioning mirror fallbacks for the remote (used when the server can't
    /// reach the public internet directly).
    #[serde(default)]
    pub mirrors: RemoteMirrors,
}

impl RemoteServer {
    /// Local port the tunnel should bind — `local_port` or, if unset, the
    /// remote port so the mapping is 1:1 by default.
    pub fn effective_local_port(&self) -> u16 {
        if self.local_port == 0 {
            self.remote_port
        } else {
            self.local_port
        }
    }

    /// The base URL the local agent uses once the tunnel is up.
    pub fn tunnel_base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.effective_local_port())
    }
}

impl Default for RemoteServer {
    fn default() -> Self {
        Self {
            name: "gpu-server".into(),
            host: String::new(),
            port: default_ssh_port(),
            username: String::new(),
            password: String::new(),
            key_path: None,
            backend: default_remote_backend(),
            remote_port: default_remote_backend_port(),
            local_port: 0,
            auto_connect: false,
            mirrors: RemoteMirrors::default(),
        }
    }
}

/// Mirror fallbacks used while provisioning a remote server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteMirrors {
    /// URLs hosting the Ollama install script, tried in order. Default is the
    /// official installer; add an internal mirror for air-gapped LANs.
    #[serde(default = "default_ollama_install_urls")]
    pub ollama_install: Vec<String>,
    /// HuggingFace endpoint the remote Ollama should pull GGUF weights from
    /// (sets `HF_ENDPOINT` for the remote pull). Empty = default huggingface.co.
    #[serde(default)]
    pub hf_endpoint: String,
}

impl Default for RemoteMirrors {
    fn default() -> Self {
        Self {
            ollama_install: default_ollama_install_urls(),
            hf_endpoint: String::new(),
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_hf_provider() -> String {
    "huggingface".into()
}
fn default_hf_endpoint() -> String {
    "https://huggingface.co".into()
}
fn default_hf_api() -> String {
    "https://huggingface.co/api".into()
}
fn default_hf_token_env() -> String {
    "HF_TOKEN".into()
}
fn default_backend_kind() -> String {
    "ollama".into()
}
fn default_ollama_url() -> String {
    "http://127.0.0.1:11434".into()
}
fn default_llama_bin() -> String {
    "llama-server".into()
}
fn default_host() -> String {
    // Bind deployed model servers on all interfaces by default so they are
    // reachable from the LAN and from containers (e.g. the OpenWebUI Docker
    // image reaching the host via host.docker.internal). The in-app assistant
    // server is separate and always binds 127.0.0.1.
    "0.0.0.0".into()
}
fn default_llama_port() -> u16 {
    8080
}
fn default_vllm_bin() -> String {
    "vllm".into()
}
fn default_vllm_port() -> u16 {
    8000
}
fn default_sglang_bin() -> String {
    "sglang".into()
}
fn default_sglang_port() -> u16 {
    30000
}
fn default_assistant_provider() -> String {
    // Prefer local Bonsai when installed; hosted providers remain available.
    "local".into()
}
fn default_openrouter_url() -> String {
    "https://openrouter.ai/api/v1".into()
}
fn default_assistant_key_env() -> String {
    "OPENROUTER_API_KEY".into()
}
fn default_assistant_port() -> u16 {
    18080
}
fn default_assistant_ctx() -> u32 {
    8192
}
fn default_assistant_ngl() -> i32 {
    // Offload as many layers as fit; llama.cpp clamps to model size.
    99
}
fn default_max_tools() -> u32 {
    32
}
fn default_history_chars() -> usize {
    // ≈12k tokens — comfortable for the 16k–32k contexts local coding models
    // typically run with, while still leaving room for tool schemas + output.
    48_000
}
fn default_bash_timeout_secs() -> u64 {
    // 5 minutes: long enough for a cold `cargo build`/test run without letting a
    // genuinely hung command tie up the agent forever.
    300
}

fn default_compact_keep_chars() -> usize {
    // Recent tail kept verbatim through a compaction — enough for the current
    // task's tool exchanges without re-triggering compaction immediately.
    20_000
}
fn default_composer_rows() -> u16 {
    3
}
fn default_repo_url() -> String {
    "https://github.com/view321/LocalCode.git".into()
}
fn default_repo_branch() -> String {
    "main".into()
}
fn default_spend_threshold() -> f64 {
    1.0
}
fn default_ssh_port() -> u16 {
    22
}
fn default_remote_backend() -> String {
    "ollama".into()
}
fn default_remote_backend_port() -> u16 {
    11434
}
fn default_ollama_install_urls() -> Vec<String> {
    vec!["https://ollama.com/install.sh".into()]
}
// NOTE: defaults must be pure — env overrides are resolved at *use* time
// (api_base_url(), log_level(), hf endpoints), never baked into a saved config.
fn default_api_url() -> String {
    "https://api.localcode.example".into()
}
fn default_log_level() -> String {
    "info".into()
}
fn default_max_log_files() -> u32 {
    20
}

impl Config {
    pub fn load(paths: &AppPaths) -> Result<Self, LocalCodeError> {
        paths.ensure_dirs()?;
        let path = paths.config_file();
        if !path.exists() {
            let cfg = Self::default();
            cfg.save(paths)?;
            return Ok(cfg);
        }
        let raw = fs::read_to_string(&path).map_err(|e| {
            LocalCodeError::new(ErrorCode::ConfigLoadFailed, e.to_string())
                .with_hint(format!("Check {}", path.display()))
        })?;
        toml::from_str(&raw).map_err(|e| {
            LocalCodeError::new(ErrorCode::ConfigParseFailed, e.to_string())
                .with_cause("Invalid TOML in config.toml")
                .with_hint("Fix syntax or delete config to regenerate defaults")
        })
    }

    pub fn save(&self, paths: &AppPaths) -> Result<(), LocalCodeError> {
        paths.ensure_dirs()?;
        let path = paths.config_file();
        self.save_to(&path)
    }

    pub fn save_to(&self, path: &Path) -> Result<(), LocalCodeError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let raw = toml::to_string_pretty(self).map_err(|e| {
            LocalCodeError::new(ErrorCode::ConfigSaveFailed, e.to_string())
        })?;
        fs::write(path, raw).map_err(|e| {
            LocalCodeError::new(ErrorCode::ConfigSaveFailed, e.to_string())
                .with_hint(format!("Ensure writable: {}", path.display()))
        })
    }

    pub fn hf_token(&self) -> Option<String> {
        env_nonempty(&self.registry.token_env)
    }

    /// True when `model_id` is starred as a favourite.
    pub fn is_favorite(&self, model_id: &str) -> bool {
        self.models.favorites.iter().any(|m| m == model_id)
    }

    /// Add a favourite (no-op if already present). Returns true if it was added.
    pub fn add_favorite(&mut self, model_id: &str) -> bool {
        if self.is_favorite(model_id) || model_id.is_empty() {
            return false;
        }
        self.models.favorites.push(model_id.to_string());
        true
    }

    /// Remove a favourite. Returns true if it was present and removed.
    pub fn remove_favorite(&mut self, model_id: &str) -> bool {
        let before = self.models.favorites.len();
        self.models.favorites.retain(|m| m != model_id);
        self.models.favorites.len() != before
    }

    /// Toggle a favourite. Returns the new state (true = now a favourite).
    pub fn toggle_favorite(&mut self, model_id: &str) -> bool {
        if self.is_favorite(model_id) {
            self.remove_favorite(model_id);
            false
        } else {
            self.add_favorite(model_id)
        }
    }

    pub fn assistant_api_key(&self) -> Option<String> {
        env_nonempty(&self.assistant.api_key_env)
    }

    /// API base URL with `LOCALCODE_API_URL` env override (never persisted).
    pub fn api_base_url(&self) -> String {
        env_nonempty("LOCALCODE_API_URL").unwrap_or_else(|| self.api.base_url.clone())
    }

    /// Log level with `LOCALCODE_LOG_LEVEL` env override (never persisted).
    pub fn log_level(&self) -> String {
        env_nonempty("LOCALCODE_LOG_LEVEL").unwrap_or_else(|| self.logging.level.clone())
    }

    /// HF endpoints with `LOCALCODE_HF_ENDPOINT` env override (never persisted).
    /// Returns (web endpoint, api endpoint).
    pub fn hf_endpoints(&self) -> (String, String) {
        if let Some(ep) = env_nonempty("LOCALCODE_HF_ENDPOINT") {
            let ep = ep.trim_end_matches('/').to_string();
            let api = format!("{ep}/api");
            (ep, api)
        } else {
            (
                self.registry.endpoint.clone(),
                self.registry.api_endpoint.clone(),
            )
        }
    }

    /// Ordered, de-duplicated HuggingFace web roots to try for weight
    /// downloads: the primary endpoint (honoring the env override) first, then
    /// configured mirrors, then canonical `huggingface.co` as a last resort.
    pub fn hf_mirror_hosts(&self) -> Vec<String> {
        let (endpoint, _api) = self.hf_endpoints();
        let mut hosts = vec![endpoint];
        hosts.extend(self.registry.mirrors.iter().cloned());
        hosts.push("https://huggingface.co".to_string());
        let mut seen = std::collections::HashSet::new();
        hosts
            .into_iter()
            .map(|h| h.trim_end_matches('/').to_string())
            .filter(|h| !h.is_empty() && seen.insert(h.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::AppPaths;
    use tempfile::tempdir;

    #[test]
    fn roundtrip_config() {
        let dir = tempdir().unwrap();
        let paths = AppPaths::from_home(dir.path().to_path_buf());
        paths.ensure_dirs().unwrap();
        let cfg = Config::default();
        cfg.save(&paths).unwrap();
        let loaded = Config::load(&paths).unwrap();
        assert_eq!(loaded.ui.theme, ThemeMode::default(), "default theme round-trips");
        assert_eq!(loaded.backends.default.kind, "ollama");
        assert_eq!(
            loaded.assistant.local_preference,
            LocalAssistantPreference::NotPrompted
        );
        assert!(loaded.assistant.prefer_local);
        assert_eq!(loaded.assistant.local_port, 18080);
    }

    #[test]
    fn approval_mode_parses_and_honors_legacy_switch() {
        // Old config: no approval_mode key, confirms turned off → never ask.
        let cfg: AgentConfig =
            toml::from_str("confirm_destructive_tools = false").unwrap();
        assert_eq!(cfg.approval(), ApprovalMode::AlwaysApprove);

        // Old config default: confirms on → Auto.
        let cfg: AgentConfig = toml::from_str("").unwrap();
        assert_eq!(cfg.approval(), ApprovalMode::Auto);

        // An explicit mode wins over the legacy switch.
        let cfg: AgentConfig = toml::from_str(
            "approval_mode = \"ask_permission\"\nconfirm_destructive_tools = false",
        )
        .unwrap();
        assert_eq!(cfg.approval(), ApprovalMode::AskPermission);

        // Serde aliases keep short names loading.
        let cfg: AgentConfig = toml::from_str("approval_mode = \"edits\"").unwrap();
        assert_eq!(cfg.approval(), ApprovalMode::ApproveEdits);

        // User-typed names (the /mode argument).
        assert_eq!(ApprovalMode::parse("always approve"), Some(ApprovalMode::AlwaysApprove));
        assert_eq!(ApprovalMode::parse("Approve-Edits"), Some(ApprovalMode::ApproveEdits));
        assert_eq!(ApprovalMode::parse("ask"), Some(ApprovalMode::AskPermission));
        assert_eq!(ApprovalMode::parse("auto"), Some(ApprovalMode::Auto));
        assert_eq!(ApprovalMode::parse("bogus"), None);

        // The cycle visits every mode and wraps.
        let mut m = ApprovalMode::Auto;
        let mut seen = vec![m];
        for _ in 0..3 {
            m = m.next();
            seen.push(m);
        }
        assert_eq!(m.next(), ApprovalMode::Auto);
        seen.sort_by_key(|m| m.tag());
        seen.dedup();
        assert_eq!(seen.len(), 4, "cycle must visit all modes");
    }

    #[test]
    fn roundtrip_remote_and_mirrors() {
        let dir = tempdir().unwrap();
        let paths = AppPaths::from_home(dir.path().to_path_buf());
        paths.ensure_dirs().unwrap();
        let mut cfg = Config::default();
        cfg.registry.endpoint = "https://hf-mirror.com".into();
        cfg.registry.mirrors = vec!["https://hf-mirror-2.com".into()];
        cfg.updates.mirrors = vec!["https://git.internal.lan/lc.git".into()];
        cfg.remote.servers.push(RemoteServer {
            name: "gpu".into(),
            host: "10.0.0.5".into(),
            username: "root".into(),
            password: "pw".into(),
            local_port: 21434,
            mirrors: RemoteMirrors {
                ollama_install: vec!["https://internal.lan/install.sh".into()],
                hf_endpoint: "https://hf-mirror.com".into(),
            },
            ..Default::default()
        });
        cfg.save(&paths).unwrap();

        let loaded = Config::load(&paths).unwrap();
        assert_eq!(loaded.registry.mirrors, vec!["https://hf-mirror-2.com"]);
        assert_eq!(loaded.remote.servers.len(), 1);
        let s = &loaded.remote.servers[0];
        assert_eq!(s.host, "10.0.0.5");
        assert_eq!(s.effective_local_port(), 21434);
        assert_eq!(s.tunnel_base_url(), "http://127.0.0.1:21434");
        assert_eq!(s.mirrors.hf_endpoint, "https://hf-mirror.com");

        // Mirror host list is ordered (endpoint first) with HF appended last.
        let hosts = loaded.hf_mirror_hosts();
        assert_eq!(hosts[0], "https://hf-mirror.com");
        assert_eq!(hosts[1], "https://hf-mirror-2.com");
        assert_eq!(hosts.last().unwrap(), "https://huggingface.co");
    }

    #[test]
    fn default_host_binds_all_interfaces() {
        // Deployed servers must be reachable from the LAN / containers by default.
        assert_eq!(Config::default().backends.vllm.host, "0.0.0.0");
        assert_eq!(Config::default().backends.llamacpp.host, "0.0.0.0");
        assert_eq!(Config::default().backends.sglang.host, "0.0.0.0");
    }

    #[test]
    fn favorites_toggle_and_persist() {
        let dir = tempdir().unwrap();
        let paths = AppPaths::from_home(dir.path().to_path_buf());
        paths.ensure_dirs().unwrap();
        let mut cfg = Config::default();
        assert!(!cfg.is_favorite("org/model"));
        assert!(cfg.toggle_favorite("org/model"));
        assert!(cfg.is_favorite("org/model"));
        // Idempotent add, real remove.
        assert!(!cfg.add_favorite("org/model"));
        assert!(!cfg.toggle_favorite("org/model"));
        assert!(!cfg.is_favorite("org/model"));

        cfg.add_favorite("a/one");
        cfg.add_favorite("b/two");
        cfg.save(&paths).unwrap();
        let loaded = Config::load(&paths).unwrap();
        assert_eq!(loaded.models.favorites, vec!["a/one", "b/two"]);
    }
}
