//! In-app repair assistant (OpenRouter / OpenAI-compatible / self-hosted).

use localcode_core::config::AssistantConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantRequest {
    pub user_message: String,
    pub error_context: Option<serde_json::Value>,
    pub doctor_report: Option<serde_json::Value>,
    pub recent_logs: Option<String>,
    pub config_snapshot_redacted: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantReply {
    pub message: String,
    pub proposed_fixes: Vec<ProposedFix>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedFix {
    pub title: String,
    pub description: String,
    pub risk: String,
    pub auto_applyable: bool,
    pub action: serde_json::Value,
}

pub struct Assistant {
    cfg: AssistantConfig,
    http: reqwest::Client,
}

impl Assistant {
    pub fn new(cfg: AssistantConfig) -> Self {
        Self {
            cfg,
            http: reqwest::Client::new(),
        }
    }

    pub fn is_configured(&self, api_key: Option<&str>) -> bool {
        match self.cfg.provider.as_str() {
            "self_hosted" => !self.cfg.base_url.is_empty(),
            _ => api_key.map(|k| !k.is_empty()).unwrap_or(false) || !self.cfg.base_url.is_empty(),
        }
    }

    pub async fn ask(
        &self,
        req: AssistantRequest,
        api_key: Option<&str>,
    ) -> Result<AssistantReply, LocalCodeError> {
        if !self.is_configured(api_key) {
            return Err(LocalCodeError::new(
                ErrorCode::BackendNotReady,
                "Assistant is not configured",
            )
            .with_cause("No API key / base URL for assistant provider")
            .with_hint("Open Setup → Assistant and set OpenRouter or OpenAI-compatible endpoint")
            .with_hint("Or use a self-hosted model already deployed"));
        }

        info!(provider = %self.cfg.provider, "assistant ask");

        let system = r#"You are the LocalCode in-app assistant. Help users fix LocalCode itself:
config, backends, deploys, GPU, cloud keys, payments. Be concrete: list causes and next steps.
Never initiate crypto spend. Propose config edits as structured suggestions.
If logs/errors are provided, ground your answer in them."#;

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

        let proposed = extract_proposed_fixes(&message);

        Ok(AssistantReply {
            message,
            proposed_fixes: proposed,
        })
    }
}

fn extract_proposed_fixes(message: &str) -> Vec<ProposedFix> {
    // Heuristic: lines starting with "FIX:" become proposed fixes
    message
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            if let Some(rest) = l.strip_prefix("FIX:") {
                Some(ProposedFix {
                    title: rest.trim().chars().take(60).collect(),
                    description: rest.trim().to_string(),
                    risk: "low".into(),
                    auto_applyable: false,
                    action: serde_json::json!({ "type": "manual" }),
                })
            } else {
                None
            }
        })
        .collect()
}
