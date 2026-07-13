use crate::error::{ErrorCode, LocalCodeError};
use crate::paths::AppPaths;
use crate::theme::ThemeMode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub type ConfigError = LocalCodeError;

#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ui: UiConfig::default(),
            registry: RegistryConfig::default(),
            backends: BackendsConfig::default(),
            assistant: AssistantConfig::default(),
            agent: AgentConfig::default(),
            cloud: CloudConfig::default(),
            api: ApiConfig::default(),
            logging: LoggingConfig::default(),
            panes: PaneRatios::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    #[serde(default)]
    pub theme: ThemeMode,
    #[serde(default = "default_true")]
    pub mouse: bool,
    #[serde(default = "default_true")]
    pub right_rail_hover_brightens: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: ThemeMode::Dark,
            mouse: true,
            right_rail_hover_brightens: true,
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
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            provider: default_hf_provider(),
            endpoint: default_hf_endpoint(),
            api_endpoint: default_hf_api(),
            token_env: default_hf_token_env(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for BackendsConfig {
    fn default() -> Self {
        Self {
            default: DefaultBackend::default(),
            ollama: OllamaConfig::default(),
            llamacpp: LlamaCppConfig::default(),
            vllm: VllmConfig::default(),
            sglang: SglangConfig::default(),
        }
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantConfig {
    #[serde(default = "default_assistant_provider")]
    pub provider: String,
    #[serde(default = "default_openrouter_url")]
    pub base_url: String,
    #[serde(default)]
    pub model: String,
    /// Env var name holding the API key (never store key in config).
    #[serde(default = "default_assistant_key_env")]
    pub api_key_env: String,
}

impl Default for AssistantConfig {
    fn default() -> Self {
        Self {
            provider: default_assistant_provider(),
            base_url: default_openrouter_url(),
            model: String::new(),
            api_key_env: default_assistant_key_env(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    #[serde(default = "default_true")]
    pub subagents_enabled: bool,
    #[serde(default)]
    pub skills_dir: Option<String>,
    #[serde(default)]
    pub mcp_config: Option<String>,
    #[serde(default = "default_true")]
    pub confirm_destructive_tools: bool,
    #[serde(default)]
    pub workspace_root: Option<String>,
    #[serde(default = "default_max_tools")]
    pub max_tool_rounds: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            subagents_enabled: true,
            skills_dir: None,
            mcp_config: None,
            confirm_destructive_tools: true,
            workspace_root: None,
            max_tool_rounds: default_max_tools(),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudProviderConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub api_key_env: String,
}

impl Default for CloudProviderConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key_env: String::new(),
        }
    }
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
    "127.0.0.1".into()
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
    "openrouter".into()
}
fn default_openrouter_url() -> String {
    "https://openrouter.ai/api/v1".into()
}
fn default_assistant_key_env() -> String {
    "OPENROUTER_API_KEY".into()
}
fn default_max_tools() -> u32 {
    32
}
fn default_spend_threshold() -> f64 {
    1.0
}
fn default_api_url() -> String {
    std::env::var("LOCALCODE_API_URL")
        .unwrap_or_else(|_| "https://api.localcode.example".into())
}
fn default_log_level() -> String {
    std::env::var("LOCALCODE_LOG_LEVEL").unwrap_or_else(|_| "info".into())
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
        std::env::var(&self.registry.token_env).ok().filter(|s| !s.is_empty())
    }

    pub fn assistant_api_key(&self) -> Option<String> {
        std::env::var(&self.assistant.api_key_env)
            .ok()
            .filter(|s| !s.is_empty())
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
        assert_eq!(loaded.ui.theme, ThemeMode::Dark);
        assert_eq!(loaded.backends.default.kind, "ollama");
    }
}
