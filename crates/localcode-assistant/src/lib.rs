//! In-app repair assistant: local Bonsai 27B (llama.cpp `-m` Q1_0) + hosted fallback.
//!
//! ## Local assistant
//! - Model: [`constants::BONSAI_FILE`] (`Q1_0`) via `llama-server -m … -ngl 99`
//! - Optional draft: [`constants::BONSAI_DRAFT_FILE`] (`Q4_1`) via `-md`
//! - Backend: managed PrismML `llama-server` (auto-built)
//! - Tools: shell, filesystem, git, Hugging Face model cards/search, doctor snapshot
//! - Onboarding: user is prompted once; decline is remembered
//! - Default conversation: TUI uses this runtime when no other model is deployed
//!
//! ## Hosted fallback
//! OpenRouter / OpenAI-compatible when local is unavailable and a key is set.

mod agent;
mod constants;
mod deploy_hints;
mod deploy_preset;
mod install;
mod runtime;

pub use agent::{
    assistant_workspace, ensure_workspace, hints_from_card, AssistantAgent, AssistantContext,
    AssistantReply, AssistantRequest, DeployToolArgs, ModelActions, ProposedFix,
};
pub use constants::{
    ASSISTANT_DISPLAY_NAME, ASSISTANT_MODEL_ID, ASSISTANT_SYSTEM_PROMPT, BONSAI_BYTES,
    BONSAI_DRAFT_BYTES, BONSAI_DRAFT_FILE, BONSAI_DRAFT_QUANT, BONSAI_FILE, BONSAI_HF_REF,
    BONSAI_QUANT, BONSAI_REPO,
};
pub use deploy_hints::{extract_deploy_hints, DeployHints};
pub use deploy_preset::{
    backend_supports, classify_weight_format, parse_native_context, preset_for_backend,
    recommend_backend, recommend_deploy_preset, DeployPreset, PresetInput, WeightFormat,
};
pub use install::{
    draft_path, install_local_assistant, install_need, install_offer_body, mark_ready,
    model_installed, model_path, resolve_llama_bin, InstallNeed,
};
pub use runtime::{
    ensure_running, is_installed, quant_compatibility_note, LocalAssistantRuntime,
};

use localcode_core::config::{AssistantConfig, Config, LocalAssistantPreference};
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::paths::AppPaths;
use localcode_core::runtime::ActiveRuntime;
use localcode_hf::HfClient;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

/// High-level façade used by the TUI and CLI.
pub struct Assistant {
    cfg: AssistantConfig,
    http: reqwest::Client,
}

impl Assistant {
    pub fn new(cfg: AssistantConfig) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_default();
        Self { cfg, http }
    }

    /// Hosted providers need an API key; self-hosted / custom need a base_url.
    /// Local Bonsai is handled separately via [`is_installed`].
    pub fn is_hosted_configured(&self, api_key: Option<&str>) -> bool {
        let has_key = api_key.map(|k| !k.is_empty()).unwrap_or(false);
        match self.cfg.provider.as_str() {
            "local" => false,
            "self_hosted" | "openai_compatible" | "custom" => !self.cfg.base_url.is_empty(),
            _ => has_key,
        }
    }

    /// True when *some* assistant path can answer (local installed or hosted).
    pub fn is_configured(
        &self,
        api_key: Option<&str>,
        paths: Option<&AppPaths>,
        full: Option<&Config>,
    ) -> bool {
        if let (Some(paths), Some(full)) = (paths, full) {
            if self.cfg.prefer_local && is_installed(full, paths) {
                return true;
            }
        }
        self.is_hosted_configured(api_key)
    }

    /// Legacy helper used by older call sites that only know the API key.
    pub fn is_configured_legacy(&self, api_key: Option<&str>) -> bool {
        self.is_hosted_configured(api_key)
            || self.cfg.provider == "local"
            || self.cfg.local_preference == LocalAssistantPreference::Accepted
    }

    /// Simple non-tool chat against a hosted endpoint (legacy OpenRouter path).
    pub async fn ask_hosted(
        &self,
        req: AssistantRequest,
        api_key: Option<&str>,
    ) -> Result<AssistantReply, LocalCodeError> {
        if !self.is_hosted_configured(api_key) {
            return Err(LocalCodeError::new(
                ErrorCode::BackendNotReady,
                "Hosted assistant is not configured",
            )
            .with_cause("No API key / base URL for assistant provider")
            .with_hint("Install the local Bonsai assistant, or set OPENROUTER_API_KEY")
            .with_hint("Open Setup → Assistant"));
        }

        info!(provider = %self.cfg.provider, "assistant hosted ask");

        let system = ASSISTANT_SYSTEM_PROMPT;
        let mut user = req.user_message.clone();
        if let Some(ctx) = &req.error_context {
            user.push_str("\n\n## Error context\n");
            user.push_str(&serde_json::to_string_pretty(ctx).unwrap_or_default());
        }
        if let Some(doc) = &req.doctor_report {
            user.push_str("\n\n## Doctor report\n");
            user.push_str(&serde_json::to_string_pretty(doc).unwrap_or_default());
        }
        if let Some(logs) = &req.recent_logs {
            user.push_str("\n\n## Recent logs\n```\n");
            user.push_str(logs);
            user.push_str("\n```\n");
        }
        if let Some(cfg) = &req.config_snapshot_redacted {
            user.push_str("\n\n## Config (redacted)\n");
            user.push_str(&serde_json::to_string_pretty(cfg).unwrap_or_default());
        }

        let base = self.cfg.base_url.trim_end_matches('/');
        let url = if base.ends_with("/v1") {
            format!("{base}/chat/completions")
        } else {
            format!("{base}/v1/chat/completions")
        };

        let model = if self.cfg.model.is_empty() {
            "openai/gpt-4o-mini".to_string()
        } else {
            self.cfg.model.clone()
        };

        let body = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.2,
        });

        let mut http_req = self.http.post(&url).json(&body);
        if let Some(k) = api_key {
            http_req = http_req.bearer_auth(k);
        }

        let resp = http_req.send().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::BackendNotReady, e.to_string())
                .with_cause("Assistant provider unreachable")
                .with_hint("Check base_url and API key")
                .retryable(true)
        })?;

        if !resp.status().is_success() {
            let t = resp.text().await.unwrap_or_default();
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("Assistant provider error: {t}"),
            )
            .retryable(true));
        }

        let v: serde_json::Value = resp.json().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::Internal, e.to_string())
        })?;
        let message = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("No response")
            .to_string();

        Ok(AssistantReply {
            message: message.clone(),
            proposed_fixes: extract_proposed_fixes_legacy(&message),
            deploy_hints: None,
        })
    }

    /// Full tool-using ask: prefer local Bonsai, else hosted.
    pub async fn ask(
        &self,
        req: AssistantRequest,
        api_key: Option<&str>,
    ) -> Result<AssistantReply, LocalCodeError> {
        // Hosted-only path when no paths/runtime were provided (CLI simplicity).
        self.ask_hosted(req, api_key).await
    }

    /// Tool-using ask with full LocalCode context (preferred by the TUI).
    #[allow(clippy::too_many_arguments)]
    pub async fn ask_with_context(
        &self,
        full_cfg: &Config,
        paths: &AppPaths,
        ctx: AssistantContext,
        api_key: Option<&str>,
        hf: Option<Arc<HfClient>>,
        model_ops: Option<Arc<dyn ModelActions>>,
        workspace: PathBuf,
        local_runtime: Option<&LocalAssistantRuntime>,
        approver: Option<&dyn localcode_agent::ToolApprover>,
    ) -> Result<AssistantReply, LocalCodeError> {
        let agent = AssistantAgent::new(full_cfg, workspace, hf, model_ops, &ctx);
        let user_text = if ctx.user_message.is_empty() {
            "Help me with LocalCode.".into()
        } else {
            ctx.user_message.clone()
        };

        // Prefer local when ready.
        if full_cfg.assistant.prefer_local {
            if let Some(rt) = local_runtime {
                if rt.is_healthy().await {
                    info!("assistant ask via local Bonsai");
                    return agent
                        .run(&rt.as_active_runtime(), None, &user_text, approver)
                        .await;
                }
            }
            if is_installed(full_cfg, paths) {
                match ensure_running(full_cfg, paths).await {
                    Ok(rt) => {
                        info!("assistant ask via freshly started local Bonsai");
                        return agent
                            .run(&rt.as_active_runtime(), None, &user_text, approver)
                            .await;
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "local assistant start failed; trying hosted");
                        if !self.is_hosted_configured(api_key) {
                            return Err(e);
                        }
                    }
                }
            }
        }

        // Hosted with tools if OpenAI-compatible endpoint is configured.
        if self.is_hosted_configured(api_key) {
            let mut runtime = ActiveRuntime::new(
                "assistant-hosted",
                localcode_core::runtime::RuntimeKind::OpenAiCompatible,
                full_cfg.assistant.base_url.clone(),
            );
            runtime.model_id = if full_cfg.assistant.model.is_empty() {
                Some("openai/gpt-4o-mini".into())
            } else {
                Some(full_cfg.assistant.model.clone())
            };
            runtime.api_key = api_key.map(|s| s.to_string());
            info!("assistant ask via hosted provider with tools");
            return agent.run(&runtime, api_key, &user_text, approver).await;
        }

        Err(LocalCodeError::new(
            ErrorCode::BackendNotReady,
            "Assistant is not available",
        )
        .with_cause("Local Bonsai not installed/running and no hosted provider configured")
        .with_hint("Accept the local assistant install offer, or set OPENROUTER_API_KEY"))
    }
}

fn extract_proposed_fixes_legacy(message: &str) -> Vec<ProposedFix> {
    message
        .lines()
        .filter_map(|l| {
            l.trim().strip_prefix("FIX:").map(|rest| ProposedFix {
                title: rest.trim().chars().take(60).collect(),
                description: rest.trim().to_string(),
                risk: "low".into(),
                auto_applyable: false,
                action: serde_json::json!({ "type": "manual" }),
            })
        })
        .collect()
}

/// Whether the TUI should show the one-time install offer on startup.
pub fn should_offer_install(cfg: &Config) -> bool {
    cfg.assistant.local_preference == LocalAssistantPreference::NotPrompted
}

/// Greeting line when the user enters the app and the local assistant is ready.
pub fn startup_greeting(ready: bool) -> String {
    if ready {
        format!(
            "Local assistant ({ASSISTANT_DISPLAY_NAME}) is ready and is the default conversation \
             model. I can search Hugging Face, read model cards, help deploy models, fix LocalCode \
             issues, and code in your workspace — just type a message (no /assistant needed)."
        )
    } else {
        format!(
            "Tip: install the local {ASSISTANT_DISPLAY_NAME} assistant \
             (`llama-server -m {BONSAI_FILE} -ngl 99`, ~3.8 GB Q1_0) for offline repair, HF catalogue \
             access, and a default local chat. Accept with /assistant or Settings → Accept local assistant."
        )
    }
}
