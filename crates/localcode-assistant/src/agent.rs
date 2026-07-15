//! Tool-using local assistant loop (repair + deploy help).
//!
//! Reuses the coding agent's OpenAI-compatible chat + tool protocol against
//! the dedicated Bonsai llama-server (or a hosted fallback), with extra tools
//! for Hugging Face model cards and a frozen context pack (errors, doctor,
//! chats).

use crate::constants::{ASSISTANT_SYSTEM_PROMPT, BONSAI_TEMPERATURE};
use crate::deploy_hints::{extract_deploy_hints, DeployHints};
use localcode_agent::{ToolApprover, ToolCall, ToolRegistry, ToolResult};
use localcode_backends::BackendKind;
use localcode_core::config::{ApprovalMode, Config};
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::runtime::ActiveRuntime;
use localcode_hf::HfClient;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

/// Pack of context attached when the assistant is invoked from an error or
/// the user opens `/assistant`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssistantContext {
    pub user_message: String,
    pub error_context: Option<Value>,
    pub doctor_report: Option<Value>,
    pub recent_logs: Option<String>,
    pub config_snapshot_redacted: Option<Value>,
    /// Recent coding transcript lines (role: text), for "access to chats".
    pub chat_excerpt: Option<String>,
    /// When helping with deploy: model id + backend + current card markdown.
    pub deploy_model_id: Option<String>,
    pub deploy_backend: Option<String>,
    pub model_card_markdown: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantRequest {
    pub user_message: String,
    pub error_context: Option<Value>,
    pub doctor_report: Option<Value>,
    pub recent_logs: Option<String>,
    pub config_snapshot_redacted: Option<Value>,
}

impl From<AssistantRequest> for AssistantContext {
    fn from(r: AssistantRequest) -> Self {
        Self {
            user_message: r.user_message,
            error_context: r.error_context,
            doctor_report: r.doctor_report,
            recent_logs: r.recent_logs,
            config_snapshot_redacted: r.config_snapshot_redacted,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedFix {
    pub title: String,
    pub description: String,
    pub risk: String,
    pub auto_applyable: bool,
    pub action: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantReply {
    pub message: String,
    pub proposed_fixes: Vec<ProposedFix>,
    /// When the turn was about deploy, structured flags parsed from the card.
    pub deploy_hints: Option<DeployHints>,
}

/// Runs the in-app assistant against a local or hosted OpenAI-compatible runtime.
pub struct AssistantAgent {
    http: reqwest::Client,
    tools: ToolRegistry,
    /// Workspace for shell/fs tools (usually the user's coding workspace).
    workspace: PathBuf,
    approval: ApprovalMode,
    max_rounds: u32,
    hf: Option<Arc<HfClient>>,
    /// Frozen context pack available to tools and the system prompt.
    context_blob: String,
}

impl AssistantAgent {
    pub fn new(
        cfg: &Config,
        workspace: PathBuf,
        hf: Option<Arc<HfClient>>,
        context: &AssistantContext,
    ) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        // Assistant is allowed shell + fs; keep approval mode from agent config
        // so destructive commands still gate when the user wants that.
        let tools = ToolRegistry::new(
            cfg.agent.disabled_tools.clone(),
            cfg.agent.shell_sandbox,
        );
        let context_blob = format_context_pack(context);

        Self {
            http,
            tools,
            workspace,
            approval: cfg.agent.approval(),
            max_rounds: cfg.agent.max_tool_rounds.max(4),
            hf,
            context_blob,
        }
    }

    /// One user turn with tools against `runtime`.
    pub async fn run(
        &self,
        runtime: &ActiveRuntime,
        api_key: Option<&str>,
        user_text: &str,
        approver: Option<&dyn ToolApprover>,
    ) -> Result<AssistantReply, LocalCodeError> {
        let system = format!(
            "{ASSISTANT_SYSTEM_PROMPT}\n\n## Current context pack\n{}",
            self.context_blob
        );

        let mut messages: Vec<Value> = vec![
            json!({"role": "system", "content": system}),
            json!({"role": "user", "content": user_text}),
        ];

        let mut final_text = String::new();
        let mut last_sig: Option<String> = None;
        let mut repeats = 0u32;

        for round in 0..self.max_rounds {
            info!(round, "assistant tool round");
            let response = self
                .chat_completion(runtime, api_key, &messages)
                .await?;

            if let Some(tool_calls) = response.tool_calls {
                messages.push(json!({
                    "role": "assistant",
                    "content": response.content.unwrap_or_default(),
                    "tool_calls": response.raw_tool_calls,
                }));

                for tc in tool_calls {
                    let sig = format!("{}|{}", tc.name, tc.arguments);
                    if last_sig.as_deref() == Some(sig.as_str()) {
                        repeats += 1;
                    } else {
                        last_sig = Some(sig);
                        repeats = 0;
                    }

                    let content = if repeats >= 2 {
                        format!(
                            "ERROR: identical call to {} repeated — change approach",
                            tc.name
                        )
                    } else {
                        match self.execute_tool(&tc, approver).await {
                            Ok(r) => r.output,
                            Err(e) => format!("ERROR {}: {}", e.code, e.message),
                        }
                    };

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": content,
                    }));
                }
                continue;
            }

            final_text = response.content.unwrap_or_default();
            break;
        }

        if final_text.is_empty() {
            final_text = "(assistant finished tool rounds without a final message)".into();
        }

        let proposed = extract_proposed_fixes(&final_text);
        Ok(AssistantReply {
            message: final_text,
            proposed_fixes: proposed,
            deploy_hints: None,
        })
    }

    async fn execute_tool(
        &self,
        call: &ToolCall,
        approver: Option<&dyn ToolApprover>,
    ) -> Result<ToolResult, LocalCodeError> {
        // Extra tools not in the coding registry (no approval gate).
        match call.name.as_str() {
            "hf.model_card" => return self.tool_hf_model_card(call).await,
            "hf.search" => return self.tool_hf_search(call).await,
            "doctor.snapshot" => {
                return Ok(ToolResult {
                    output: self.context_blob.clone(),
                    risk: localcode_agent::ToolRisk::Low,
                });
            }
            _ => {}
        }

        // Single approval path via ToolRegistry (no double-gate).
        // Background auto-repair passes `approver = None`: gated calls are
        // refused by the registry instead of silently elevated.
        self.tools
            .execute(call, &self.workspace, self.approval, approver, None)
            .await
    }

    async fn tool_hf_model_card(&self, call: &ToolCall) -> Result<ToolResult, LocalCodeError> {
        let Some(hf) = &self.hf else {
            return Err(LocalCodeError::new(
                ErrorCode::HfUnreachable,
                "Hugging Face client is not available",
            )
            .with_hint("Check registry endpoint / network"));
        };
        let model_id = call
            .arguments
            .get("model_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                LocalCodeError::new(ErrorCode::AgentToolFailed, "hf.model_card requires model_id")
            })?;
        let detail = hf.model_info(model_id).await?;
        let card = detail
            .card_markdown
            .unwrap_or_else(|| "(no README on this model)".into());
        // Cap size for context.
        let card = truncate(&card, 24_000);
        let backend = call
            .arguments
            .get("backend")
            .and_then(|v| v.as_str())
            .and_then(BackendKind::parse)
            .unwrap_or(BackendKind::Vllm);
        let hints = extract_deploy_hints(&card, backend);
        let mut out = format!("# Model card: {model_id}\n\n{card}");
        if !hints.is_empty() {
            out.push_str("\n\n## Parsed deploy hints\n");
            out.push_str(&serde_json::to_string_pretty(&hints).unwrap_or_default());
        }
        Ok(ToolResult {
            output: out,
            risk: localcode_agent::ToolRisk::Low,
        })
    }

    async fn tool_hf_search(&self, call: &ToolCall) -> Result<ToolResult, LocalCodeError> {
        let Some(hf) = &self.hf else {
            return Err(LocalCodeError::new(
                ErrorCode::HfUnreachable,
                "Hugging Face client is not available",
            ));
        };
        let query = call
            .arguments
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("coding");
        let results = hf.search(query, true, 8, "downloads").await?;
        let lines: Vec<String> = results
            .iter()
            .map(|m| {
                format!(
                    "- {}  downloads={:?}  tags={}",
                    m.id,
                    m.downloads,
                    m.tags.iter().take(5).cloned().collect::<Vec<_>>().join(",")
                )
            })
            .collect();
        Ok(ToolResult {
            output: if lines.is_empty() {
                "No models found".into()
            } else {
                lines.join("\n")
            },
            risk: localcode_agent::ToolRisk::Low,
        })
    }

    fn tools_schema(&self) -> Value {
        let mut tools = self.tools.openai_tools_schema();
        // Append HF / doctor tools.
        if let Some(arr) = tools.as_array_mut() {
            arr.push(tool_schema(
                "hf.model_card",
                "Fetch a Hugging Face model card (README) and parse deploy flags. Pass backend (vllm|llamacpp|sglang|ollama) for better flag extraction.",
                json!({
                    "type": "object",
                    "properties": {
                        "model_id": { "type": "string" },
                        "backend": { "type": "string" }
                    },
                    "required": ["model_id"]
                }),
            ));
            arr.push(tool_schema(
                "hf.search",
                "Search Hugging Face models by free-text query",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"]
                }),
            ));
            arr.push(tool_schema(
                "doctor.snapshot",
                "Return the frozen error / doctor / config / chat context pack for this turn",
                json!({ "type": "object", "properties": {} }),
            ));
        }
        tools
    }

    async fn chat_completion(
        &self,
        runtime: &ActiveRuntime,
        api_key: Option<&str>,
        messages: &[Value],
    ) -> Result<LlmResponse, LocalCodeError> {
        let base = runtime.base_url.trim_end_matches('/');
        let url = if base.ends_with("/v1") {
            format!("{base}/chat/completions")
        } else {
            format!("{base}/v1/chat/completions")
        };

        let model = runtime
            .model_id
            .clone()
            .unwrap_or_else(|| "default".into());

        let body = json!({
            "model": model,
            "messages": messages,
            "tools": self.tools_schema(),
            "temperature": BONSAI_TEMPERATURE,
        });

        let mut req = self.http.post(&url).json(&body);
        let key = api_key.or(runtime.api_key.as_deref());
        if let Some(k) = key {
            req = req.bearer_auth(k);
        }

        let resp = req.send().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::BackendNotReady, e.to_string())
                .with_cause("Assistant runtime unreachable")
                .with_hint("Is the local Bonsai server running, or is the hosted key valid?")
                .retryable(true)
        })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(LocalCodeError::new(
                ErrorCode::BackendNotReady,
                format!("Assistant runtime returned {status}: {t}"),
            )
            .retryable(true));
        }

        let v: Value = resp.json().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::Internal, e.to_string())
        })?;

        parse_completion(&v)
    }
}

struct LlmResponse {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    raw_tool_calls: Option<Value>,
}

fn parse_completion(v: &Value) -> Result<LlmResponse, LocalCodeError> {
    let choice = v
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| {
            LocalCodeError::new(ErrorCode::Internal, "Assistant response missing choices")
        })?;

    let message = choice.get("message").cloned().unwrap_or(json!({}));
    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .map(|s| s.to_string());

    let raw_tool_calls = message.get("tool_calls").cloned();
    let tool_calls = raw_tool_calls.as_ref().and_then(|tcs| {
        let arr = tcs.as_array()?;
        let mut out = Vec::new();
        for tc in arr {
            let id = tc
                .get("id")
                .and_then(|i| i.as_str())
                .unwrap_or("")
                .to_string();
            let id = if id.is_empty() {
                Uuid::new_v4().to_string()
            } else {
                id
            };
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            let args_str = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            let arguments: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
            if !name.is_empty() {
                out.push(ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    });

    Ok(LlmResponse {
        content,
        tool_calls,
        raw_tool_calls,
    })
}

fn tool_schema(name: &str, description: &str, parameters: Value) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    })
}

fn format_context_pack(ctx: &AssistantContext) -> String {
    let mut parts = Vec::new();
    if let Some(e) = &ctx.error_context {
        parts.push(format!(
            "### Error\n{}",
            serde_json::to_string_pretty(e).unwrap_or_default()
        ));
    }
    if let Some(d) = &ctx.doctor_report {
        parts.push(format!(
            "### Doctor\n{}",
            serde_json::to_string_pretty(d).unwrap_or_default()
        ));
    }
    if let Some(logs) = &ctx.recent_logs {
        parts.push(format!("### Recent logs\n```\n{}\n```", truncate(logs, 8_000)));
    }
    if let Some(cfg) = &ctx.config_snapshot_redacted {
        parts.push(format!(
            "### Config (redacted)\n{}",
            serde_json::to_string_pretty(cfg).unwrap_or_default()
        ));
    }
    if let Some(chat) = &ctx.chat_excerpt {
        parts.push(format!(
            "### Recent coding chat\n{}",
            truncate(chat, 6_000)
        ));
    }
    if let Some(id) = &ctx.deploy_model_id {
        parts.push(format!(
            "### Deploy target\nmodel_id={id} backend={}",
            ctx.deploy_backend.as_deref().unwrap_or("?")
        ));
    }
    if let Some(card) = &ctx.model_card_markdown {
        parts.push(format!(
            "### Model card (excerpt)\n{}",
            truncate(card, 12_000)
        ));
    }
    if parts.is_empty() {
        "(no error/doctor/chat context attached)".into()
    } else {
        parts.join("\n\n")
    }
}

fn extract_proposed_fixes(message: &str) -> Vec<ProposedFix> {
    message
        .lines()
        .filter_map(|l| {
            l.trim().strip_prefix("FIX:").map(|rest| ProposedFix {
                title: rest.trim().chars().take(60).collect(),
                description: rest.trim().to_string(),
                risk: "low".into(),
                auto_applyable: false,
                action: json!({ "type": "manual" }),
            })
        })
        .collect()
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push('…');
        out
    }
}

/// Pure helper: read deploy hints from a card for a backend (no LLM).
pub fn hints_from_card(card: &str, backend: BackendKind) -> DeployHints {
    extract_deploy_hints(card, backend)
}

/// Workspace used by the assistant tools — prefers config, else cwd.
pub fn assistant_workspace(cfg: &Config) -> PathBuf {
    cfg.agent
        .workspace_root
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

/// Ensure path exists for tool use.
pub fn ensure_workspace(path: &Path) -> Result<(), LocalCodeError> {
    if path.exists() {
        Ok(())
    } else {
        Err(LocalCodeError::new(
            ErrorCode::AgentWorkspaceMissing,
            format!("Workspace does not exist: {}", path.display()),
        )
        .with_hint("Set agent.workspace_root in Settings"))
    }
}
