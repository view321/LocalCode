//! Coding agent with tools, skills, and sessions.

mod session_store;
mod skills;
mod tools;

pub use session_store::{
    list_sessions, sessions_root, LoadedSession, SessionMeta, SessionStore,
};
pub use skills::{Skill, SkillLoader};
pub use tools::{
    approval_request, ToolApprover, ToolCall, ToolRegistry, ToolResult, ToolRisk,
};

use futures::StreamExt;
use localcode_core::config::AgentConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::paths::AppPaths;
use localcode_core::runtime::ActiveRuntime;
use localcode_hf::HfClient;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info};
use uuid::Uuid;

/// Live progress from a running turn, for UIs that render output as it
/// happens. Sends are best-effort: a dropped receiver never fails the turn.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Streamed assistant text (a token when streaming, or a whole message
    /// when the runtime doesn't stream).
    Delta(String),
    /// Streamed model reasoning / chain-of-thought (OpenAI-style
    /// `reasoning_content`, or `<think>` tags inside content).
    ThinkingDelta(String),
    /// The assistant message of the current round is complete; later deltas
    /// belong to a new message.
    MessageComplete,
    ToolStarted {
        name: String,
        args_preview: String,
    },
    ToolFinished {
        name: String,
        ok: bool,
        /// One-line summary for the compact transcript row.
        summary: String,
        /// Pretty-printed tool arguments (for the expandable detail pane).
        args: String,
        /// Full tool output returned to the model (for the expandable detail pane).
        output: String,
    },
}

fn emit(events: Option<&mpsc::UnboundedSender<AgentEvent>>, ev: AgentEvent) {
    if let Some(tx) = events {
        let _ = tx.send(ev);
    }
}

/// First line of `s`, truncated to `max` chars (char-safe).
fn preview(s: &str, max: usize) -> String {
    let first = s.lines().next().unwrap_or("");
    let mut out: String = first.chars().take(max).collect();
    if first.chars().count() > max || s.lines().nth(1).is_some() {
        out.push('…');
    }
    out
}

/// Compact one-line view of tool args (`key=value key2=value2 …`).
fn args_preview(args: &Value, max: usize) -> String {
    let raw = match args {
        Value::Object(map) if !map.is_empty() => map
            .iter()
            .map(|(k, v)| {
                let val = match v {
                    Value::String(s) => preview(s, 40),
                    other => other.to_string(),
                };
                format!("{k}={val}")
            })
            .collect::<Vec<_>>()
            .join(" "),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    preview(&raw, max)
}

/// Multi-line pretty args for the expandable tool detail pane.
fn format_args_detail(args: &Value) -> String {
    match args {
        Value::Object(map) if !map.is_empty() => map
            .iter()
            .map(|(k, v)| match v {
                Value::String(s) => {
                    if s.contains('\n') || s.chars().count() > 80 {
                        format!("{k}:\n{s}")
                    } else {
                        format!("{k}: {s}")
                    }
                }
                other => format!("{k}: {other}"),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Compact tool-result summary with size so users know they can expand.
fn tool_output_summary(output: &str) -> String {
    let n = output.chars().count();
    let head = preview(output, 72);
    if n == 0 {
        "(empty)".into()
    } else if head.is_empty() {
        format!("{n} chars")
    } else if n > 72 || output.lines().nth(1).is_some() {
        format!("{head}  · {n} chars")
    } else {
        head
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Raw OpenAI-style `tool_calls` array on assistant messages. Preserved
    /// verbatim so the follow-up request replays a valid tool-call exchange
    /// (strict servers reject `role:"tool"` without a preceding assistant
    /// message that carries matching `tool_calls` ids).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    /// Model reasoning / chain-of-thought for this assistant message (stored
    /// for the UI and compaction; not re-sent as a separate protocol field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

impl ChatMessage {
    pub(crate) fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
            tool_call_id: None,
            name: None,
            tool_calls: None,
            thinking: None,
        }
    }

    pub(crate) fn assistant(
        content: impl Into<String>,
        tool_calls: Option<Value>,
        thinking: Option<String>,
    ) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
            tool_call_id: None,
            name: None,
            tool_calls,
            thinking,
        }
    }

    pub(crate) fn tool(content: impl Into<String>, call_id: String, name: String) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
            tool_call_id: Some(call_id),
            name: Some(name),
            tool_calls: None,
            thinking: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSession {
    pub id: String,
    pub title: String,
    pub messages: Vec<ChatMessage>,
    pub workspace_root: PathBuf,
    pub runtime_id: Option<String>,
    /// Compacted-view marker: when set, requests replace
    /// `messages[..first_kept_index]` with the summary. The full history stays
    /// in `messages` and in the session file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction: Option<SessionCompaction>,
}

/// See [`AgentSession::compaction`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCompaction {
    pub first_kept_index: usize,
    pub summary: String,
    pub chars_before: usize,
}

impl AgentSession {
    pub fn new(workspace: PathBuf) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            title: "New session".into(),
            messages: vec![],
            workspace_root: workspace,
            runtime_id: None,
            compaction: None,
        }
    }
}

/// Truncate `messages` at the first protocol violation: an assistant message
/// carrying `tool_calls` must be followed by one `role:"tool"` reply per call
/// id, and a tool reply must follow such an assistant message. Heals
/// histories cut short by an aborted turn (the only way LocalCode produces
/// them); truncation is the one repair that is always protocol-valid.
/// Returns how many messages were dropped.
pub(crate) fn sanitize_history(messages: &mut Vec<ChatMessage>) -> usize {
    let mut valid = messages.len();
    let mut i = 0;
    while i < messages.len() {
        let m = &messages[i];
        if m.role == "assistant" {
            if let Some(calls) = m.tool_calls.as_ref().and_then(|v| v.as_array()) {
                let mut awaiting: std::collections::HashSet<&str> = calls
                    .iter()
                    .filter_map(|c| c.get("id").and_then(Value::as_str))
                    .collect();
                let mut j = i + 1;
                while j < messages.len() && messages[j].role == "tool" {
                    if let Some(id) = &messages[j].tool_call_id {
                        awaiting.remove(id.as_str());
                    }
                    j += 1;
                }
                if !awaiting.is_empty() {
                    valid = i;
                    break;
                }
                i = j;
                continue;
            }
        }
        if m.role == "tool" {
            // A tool reply not consumed above has no matching assistant call.
            valid = i;
            break;
        }
        i += 1;
    }
    let dropped = messages.len() - valid;
    messages.truncate(valid);
    dropped
}

pub struct CodingAgent {
    pub config: AgentConfig,
    pub tools: ToolRegistry,
    pub skills: SkillLoader,
    http: reqwest::Client,
    /// When set, `hf.model_card` / `hf.search` tools are available so the default
    /// conversation can read the Hugging Face catalogue and model descriptions.
    hf: Option<Arc<HfClient>>,
}

impl CodingAgent {
    pub fn new(config: AgentConfig) -> Self {
        let skills_dir = resolve_skills_dir(&config);
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        Self {
            tools: ToolRegistry::new(config.disabled_tools.clone(), config.shell_sandbox),
            skills: SkillLoader::new(skills_dir),
            config,
            http,
            hf: None,
        }
    }

    /// Attach a Hugging Face client so the agent can search models and read cards.
    pub fn with_hf(mut self, hf: Arc<HfClient>) -> Self {
        self.hf = Some(hf);
        self
    }

    pub fn set_hf(&mut self, hf: Option<Arc<HfClient>>) {
        self.hf = hf;
    }

    pub fn reload_skills(&mut self) {
        self.skills.reload();
    }

    /// Run one user turn: LLM + optional tool loop. `events` receives live
    /// progress (streamed tokens, tool activity) when provided.
    pub async fn run_turn(
        &self,
        session: &mut AgentSession,
        user_text: &str,
        runtime: &ActiveRuntime,
        api_key: Option<&str>,
        approver: Option<&dyn ToolApprover>,
        events: Option<&mpsc::UnboundedSender<AgentEvent>>,
    ) -> Result<String, LocalCodeError> {
        if !session.workspace_root.exists() {
            return Err(LocalCodeError::new(
                ErrorCode::AgentWorkspaceMissing,
                format!(
                    "Workspace does not exist: {}",
                    session.workspace_root.display()
                ),
            )
            .with_hint("Set agent.workspace_root in Settings")
            .with_cause("No workspace root"));
        }

        let healed = sanitize_history(&mut session.messages);
        if healed > 0 {
            debug!(healed, "dropped incomplete trailing tool exchange from history");
        }

        session.messages.push(ChatMessage::user(user_text));

        // Compact older history before this turn if over budget.
        self.maybe_auto_compact(session, runtime, api_key).await;

        let system = self.build_system_prompt(session);
        let mut final_text = String::new();
        // Loop breaker: track recent signatures; block exact repeats and
        // short alternating loops so local models cannot thrash forever.
        let mut recent_sigs: Vec<String> = Vec::new();
        let mut blocked_sigs: std::collections::HashSet<String> = std::collections::HashSet::new();

        for round in 0..self.config.max_tool_rounds {
            info!(round, "agent tool round");
            let response = self
                .chat_completion(runtime, api_key, &system, session, events)
                .await?;

            if !response.streamed {
                if let Some(t) = response.thinking.as_deref().filter(|c| !c.is_empty()) {
                    emit(events, AgentEvent::ThinkingDelta(t.to_string()));
                }
                if let Some(c) = response.content.as_deref().filter(|c| !c.is_empty()) {
                    emit(events, AgentEvent::Delta(c.to_string()));
                }
            }
            emit(events, AgentEvent::MessageComplete);

            if let Some(tool_calls) = response.tool_calls {
                session.messages.push(ChatMessage::assistant(
                    response.content.unwrap_or_default(),
                    response.raw_tool_calls,
                    response.thinking,
                ));

                for tc in tool_calls {
                    let args_full = format_args_detail(&tc.arguments);
                    emit(
                        events,
                        AgentEvent::ToolStarted {
                            name: tc.name.clone(),
                            args_preview: args_preview(&tc.arguments, 60),
                        },
                    );

                    let sig = format!("{}|{}", tc.name, tc.arguments);
                    let block = should_block_tool_sig(&sig, &recent_sigs, &blocked_sigs);
                    if block {
                        blocked_sigs.insert(sig.clone());
                    }
                    recent_sigs.push(sig.clone());
                    if recent_sigs.len() > 12 {
                        recent_sigs.remove(0);
                    }

                    let skill_body = if tc.name == "skill" {
                        self.skill_body_for_call(&tc)
                    } else {
                        None
                    };

                    let result = if block {
                        Err(LocalCodeError::new(
                            ErrorCode::AgentToolFailed,
                            format!(
                                "Tool call loop detected for {} — not executed",
                                tc.name
                            ),
                        )
                        .with_hint(
                            "Change the arguments or the approach, or give your final answer",
                        ))
                    } else {
                        self.execute_tool_call(
                            &tc,
                            &session.workspace_root,
                            approver,
                            skill_body,
                        )
                        .await
                    };

                    let content = match &result {
                        Ok(r) => r.output.clone(),
                        Err(e) => format!("ERROR {}: {}", e.code, e.message),
                    };
                    emit(
                        events,
                        AgentEvent::ToolFinished {
                            name: tc.name.clone(),
                            ok: result.is_ok(),
                            summary: tool_output_summary(&content),
                            args: args_full,
                            output: content.clone(),
                        },
                    );

                    session
                        .messages
                        .push(ChatMessage::tool(content, tc.id, tc.name));
                }
                continue;
            }

            final_text = response.content.unwrap_or_default();
            session.messages.push(ChatMessage::assistant(
                final_text.clone(),
                None,
                response.thinking,
            ));
            break;
        }

        if final_text.is_empty() {
            final_text =
                "(agent completed tool rounds without a final message)".into();
            // Persist the synthetic stop so history and UI agree (issue 9).
            let last_is_tool = session
                .messages
                .last()
                .map(|m| m.role == "tool")
                .unwrap_or(false);
            let last_is_assistant_with_tools = session
                .messages
                .last()
                .map(|m| m.role == "assistant" && m.tool_calls.is_some())
                .unwrap_or(false);
            if last_is_tool || last_is_assistant_with_tools || session.messages.last().map(|m| m.role == "user").unwrap_or(false) {
                // Only append if we didn't already push a final assistant text message.
                let needs = match session.messages.last() {
                    Some(m) if m.role == "assistant" && m.tool_calls.is_none() => false,
                    _ => true,
                };
                if needs {
                    session
                        .messages
                        .push(ChatMessage::assistant(final_text.clone(), None, None));
                }
            }
        }

        if session.title == "New session" {
            session.title = user_text.chars().take(40).collect();
        }

        Ok(final_text)
    }

    fn skill_body_for_call(&self, tc: &ToolCall) -> Option<String> {
        let name = tc.arguments.get("name")?.as_str()?;
        if self.config.disabled_skills.iter().any(|d| d == name) {
            return None;
        }
        let skill = self.skills.get(name)?;
        Some(format!(
            "# Skill: {}\n\n{}\n\n{}",
            skill.name, skill.description, skill.body
        ))
    }

    /// Summarize older turns when over budget; fall back to leaving compaction
    /// unset (trimmed_tail still applies).
    async fn maybe_auto_compact(
        &self,
        session: &mut AgentSession,
        runtime: &ActiveRuntime,
        api_key: Option<&str>,
    ) {
        if !self.config.auto_compact || self.config.max_history_chars == 0 {
            return;
        }
        let total: usize = session.messages.iter().map(approx_chars).sum();
        if total <= self.config.max_history_chars {
            return;
        }
        let keep = self.config.compact_keep_recent_chars.max(1);
        let first_kept = first_kept_index(&session.messages, keep);
        if first_kept == 0 {
            return;
        }
        // Don't re-compact the same cut point.
        if session
            .compaction
            .as_ref()
            .is_some_and(|c| c.first_kept_index == first_kept)
        {
            return;
        }
        let to_sum = &session.messages[..first_kept];
        let summary = match self.summarize_history(runtime, api_key, to_sum).await {
            Ok(s) if !s.trim().is_empty() => s,
            _ => fallback_summary(to_sum),
        };
        info!(
            first_kept,
            chars_before = total,
            "auto-compacted session history"
        );
        session.compaction = Some(SessionCompaction {
            first_kept_index: first_kept,
            summary,
            chars_before: total,
        });
    }

    async fn summarize_history(
        &self,
        runtime: &ActiveRuntime,
        api_key: Option<&str>,
        messages: &[ChatMessage],
    ) -> Result<String, LocalCodeError> {
        let mut transcript = String::new();
        for m in messages.iter().take(80) {
            let role = &m.role;
            let content = preview(&m.content, 400);
            transcript.push_str(&format!("{role}: {content}\n"));
            if let Some(t) = &m.thinking {
                transcript.push_str(&format!("  (thinking: {})\n", preview(t, 200)));
            }
        }
        let system = "Summarize the earlier coding-agent conversation for continuity. \
Keep goals, decisions, file paths touched, and unfinished work. Be concise (under 600 words). \
No tools — reply with the summary only.";
        let body_msgs = vec![
            json!({"role": "system", "content": system}),
            json!({
                "role": "user",
                "content": format!("Conversation to summarize:\n\n{transcript}")
            }),
        ];
        let text = self
            .simple_completion(runtime, api_key, body_msgs, false)
            .await?;
        Ok(text)
    }

    /// Non-tool chat completion used for compaction summaries.
    async fn simple_completion(
        &self,
        runtime: &ActiveRuntime,
        api_key: Option<&str>,
        messages: Vec<Value>,
        stream: bool,
    ) -> Result<String, LocalCodeError> {
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
        let mut body = json!({
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        });
        if stream {
            body["stream"] = json!(true);
        }
        let mut req = self.http.post(&url).json(&body);
        let key = api_key.or(runtime.api_key.as_deref());
        if let Some(k) = key {
            req = req.bearer_auth(k);
        }
        let resp = req.send().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::BackendNotReady, e.to_string())
                .with_cause("Summarization request failed")
        })?;
        if !resp.status().is_success() {
            let status = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(LocalCodeError::new(
                ErrorCode::BackendNotReady,
                format!("Summarization failed {status}: {t}"),
            ));
        }
        let v: Value = resp
            .json()
            .await
            .map_err(|e| LocalCodeError::new(ErrorCode::Internal, e.to_string()))?;
        let parsed = parse_full_response(&v)?;
        Ok(parsed.content.unwrap_or_default())
    }

    /// Run a single tool call: HF catalogue tools first, then the workspace registry.
    async fn execute_tool_call(
        &self,
        call: &ToolCall,
        workspace: &std::path::Path,
        approver: Option<&dyn ToolApprover>,
        skill_body: Option<String>,
    ) -> Result<ToolResult, LocalCodeError> {
        match call.name.as_str() {
            "hf.model_card" => return self.tool_hf_model_card(call).await,
            "hf.search" => return self.tool_hf_search(call).await,
            _ => {}
        }
        self.tools
            .execute(
                call,
                workspace,
                self.config.approval(),
                approver,
                skill_body,
            )
            .await
    }

    async fn tool_hf_model_card(&self, call: &ToolCall) -> Result<ToolResult, LocalCodeError> {
        let Some(hf) = &self.hf else {
            return Err(LocalCodeError::new(
                ErrorCode::HfUnreachable,
                "Hugging Face client is not available",
            )
            .with_hint("Check registry endpoint / network in Settings"));
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
        let card = if card.chars().count() > 24_000 {
            let mut out: String = card.chars().take(24_000).collect();
            out.push('…');
            out
        } else {
            card
        };
        Ok(ToolResult {
            output: format!("# Model card: {model_id}\n\n{card}"),
            risk: ToolRisk::Low,
        })
    }

    async fn tool_hf_search(&self, call: &ToolCall) -> Result<ToolResult, LocalCodeError> {
        let Some(hf) = &self.hf else {
            return Err(LocalCodeError::new(
                ErrorCode::HfUnreachable,
                "Hugging Face client is not available",
            )
            .with_hint("Check registry endpoint / network in Settings"));
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
            risk: ToolRisk::Low,
        })
    }

    fn build_system_prompt(&self, session: &AgentSession) -> String {
        let mut parts: Vec<String> = Vec::new();

        match self.config.system_prompt.as_deref().map(str::trim) {
            Some(custom) if !custom.is_empty() => parts.push(custom.to_string()),
            _ => {
                parts.push(
                    "You are LocalCode's default assistant — a local coding agent with full \
access to the workspace and the Hugging Face model catalogue. You help users write code, \
fix LocalCode itself (config, backends, deploys, GPU, logs), discover models, and launch them."
                        .into(),
                );
                parts.push(
                    "Tools: read, write, bash, ls, grep, skill, hf.model_card, hf.search. \
Work in a loop: gather context (read/ls/grep/hf.*), then act (write/bash). Prefer write for \
file edits over shell redirects."
                        .into(),
                );
                parts.push(
                    "For Hugging Face models: use hf.search to find candidates, then \
hf.model_card to read descriptions and recommended server flags before suggesting a deploy. \
Users deploy from the Models tab (/models) or with your guidance."
                        .into(),
                );
                parts.push(
                    "Read only what you need (read takes start_line/max_lines). Verify changes \
when possible with bash (build/test)."
                        .into(),
                );
                parts.push(
                    "Some tool calls may need user approval; if refused, don't retry — adjust or ask. \
Never repeat an identical failing call or alternate between two failing calls."
                        .into(),
                );
                parts.push("Be concise. When tools fail, explain causes and next steps.".into());
            }
        }

        parts.push(format!("Workspace: {}", session.workspace_root.display()));
        if self.config.shell_sandbox {
            parts.push("Shell sandbox is on: bash runs in the workspace only.".into());
        }

        let enabled_skills: Vec<&Skill> = self
            .skills
            .list()
            .iter()
            .filter(|s| {
                s.enabled && !self.config.disabled_skills.iter().any(|d| d == &s.name)
            })
            .collect();
        if !enabled_skills.is_empty() {
            parts.push("Skills (call tool skill with name=… to load full instructions):".into());
            for skill in enabled_skills {
                parts.push(format!("- {} — {}", skill.name, skill.description));
            }
        }

        if self.config.use_agents_md {
            if let Some(agents) = read_agents_md(&session.workspace_root) {
                parts.push(String::new());
                parts.push(
                    "Project instructions from AGENTS.md (follow these for this repo):".into(),
                );
                parts.push(agents);
            }
        }

        parts.join("\n")
    }

    async fn chat_completion(
        &self,
        runtime: &ActiveRuntime,
        api_key: Option<&str>,
        system: &str,
        session: &AgentSession,
        events: Option<&mpsc::UnboundedSender<AgentEvent>>,
    ) -> Result<LlmResponse, LocalCodeError> {
        let base = runtime.base_url.trim_end_matches('/');
        let url = if base.ends_with("/v1") {
            format!("{base}/chat/completions")
        } else {
            format!("{base}/v1/chat/completions")
        };

        let (messages, system_content) =
            build_request_view(session, system, self.config.max_history_chars);

        let mut api_messages = vec![json!({
            "role": "system",
            "content": system_content,
        })];
        for m in messages {
            let mut msg = json!({
                "role": m.role,
                "content": m.content,
            });
            if let Some(tcs) = &m.tool_calls {
                msg["tool_calls"] = tcs.clone();
            }
            if let Some(id) = &m.tool_call_id {
                msg["tool_call_id"] = json!(id);
            }
            if m.role == "tool" {
                if let Some(name) = &m.name {
                    msg["name"] = json!(name);
                }
            }
            api_messages.push(msg);
        }

        let model = runtime
            .model_id
            .clone()
            .unwrap_or_else(|| "default".into());

        let stream = self.config.stream;
        let mut tools = self.tools.openai_tools_schema();
        // Hide HF tools from the model when no client is wired (avoids dead-end calls).
        if self.hf.is_none() {
            if let Some(arr) = tools.as_array_mut() {
                arr.retain(|t| {
                    !matches!(
                        t.get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str()),
                        Some("hf.model_card" | "hf.search")
                    )
                });
            }
        }
        let mut body = json!({
            "model": model,
            "messages": api_messages,
            "tools": tools,
            "temperature": 0.2,
        });
        if stream {
            body["stream"] = json!(true);
        }

        let mut req = self.http.post(&url).json(&body);
        let key = api_key.or(runtime.api_key.as_deref());
        if let Some(k) = key {
            req = req.bearer_auth(k);
        }

        let resp = req.send().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::BackendNotReady, e.to_string())
                .with_cause("Coding runtime unreachable")
                .with_hint("Deploy a model or configure assistant provider")
                .retryable(true)
        })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let t = resp.text().await.unwrap_or_default();
            let mut err = LocalCodeError::new(
                ErrorCode::BackendNotReady,
                format!("Runtime returned {status}: {t}"),
            )
            .with_hint("Check runtime logs")
            .retryable(true);
            if stream {
                err = err.with_hint(
                    "If your runtime rejects streaming with tools, set agent.stream=false in config",
                );
            }
            return Err(err);
        }

        // Servers that ignore `stream: true` reply with one JSON body
        // (content-type application/json instead of text/event-stream);
        // fall back to full-response parsing rather than failing the turn.
        let is_event_stream = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.contains("text/event-stream"));

        let mut response = if stream && is_event_stream {
            self.read_sse_response(resp, events).await?
        } else {
            let v: Value = resp
                .json()
                .await
                .map_err(|e| LocalCodeError::new(ErrorCode::Internal, e.to_string()))?;
            parse_full_response(&v)?
        };

        // Fallback for models without native tool-call support: lines of the
        // form `tool:read {"path": "..."}` (all matching lines).
        if response.tool_calls.is_none() {
            if let Some(ref c) = response.content {
                let tcs = parse_pseudo_tools(c);
                if !tcs.is_empty() {
                    let raw: Vec<Value> = tcs
                        .iter()
                        .map(|tc| {
                            json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments.to_string(),
                                }
                            })
                        })
                        .collect();
                    response.raw_tool_calls = Some(Value::Array(raw));
                    response.tool_calls = Some(tcs);
                }
            }
        }

        Ok(response)
    }

    /// Consume an OpenAI-style SSE stream, emitting `Delta` / `ThinkingDelta`
    /// events as tokens arrive and accumulating tool-call fragments into a
    /// full response.
    async fn read_sse_response(
        &self,
        resp: reqwest::Response,
        events: Option<&mpsc::UnboundedSender<AgentEvent>>,
    ) -> Result<LlmResponse, LocalCodeError> {
        let mut byte_stream = resp.bytes_stream();
        // Buffer bytes, not text: a multi-byte char can split across chunks,
        // so UTF-8 decoding happens per complete line only.
        let mut buf: Vec<u8> = Vec::new();
        let mut acc = SseAccumulator::default();
        let mut done = false;

        while let Some(chunk) = byte_stream.next().await {
            let bytes = chunk.map_err(|e| {
                LocalCodeError::new(ErrorCode::BackendNotReady, e.to_string())
                    .with_cause("Stream from the runtime broke mid-response")
                    .retryable(true)
            })?;
            for payload in drain_sse_lines(&mut buf, &bytes) {
                if payload == "[DONE]" {
                    done = true;
                    break;
                }
                for piece in acc.feed(&payload)? {
                    match piece {
                        StreamPiece::Content(t) => emit(events, AgentEvent::Delta(t)),
                        StreamPiece::Thinking(t) => emit(events, AgentEvent::ThinkingDelta(t)),
                    }
                }
            }
            if done {
                break;
            }
        }

        let response = acc.finish();
        if response.content.is_none()
            && response.tool_calls.is_none()
            && response.thinking.is_none()
        {
            return Err(LocalCodeError::new(
                ErrorCode::Internal,
                "Runtime stream ended without content or tool calls",
            )
            .with_cause("Malformed or empty SSE response from the model server")
            .with_hint("Check runtime logs")
            .retryable(true));
        }
        Ok(response)
    }
}

/// One streamed token bucket: final answer text vs model reasoning.
#[derive(Debug, Clone, PartialEq, Eq)]
enum StreamPiece {
    Content(String),
    Thinking(String),
}

/// Parse a non-streaming chat completion body.
fn parse_full_response(v: &Value) -> Result<LlmResponse, LocalCodeError> {
    let choices = v
        .get("choices")
        .and_then(|c| c.as_array())
        .filter(|c| !c.is_empty())
        .ok_or_else(|| {
            LocalCodeError::new(ErrorCode::Internal, "Runtime response has no choices")
                .with_cause("Malformed or error response from the model server")
                .with_hint("Check runtime logs")
        })?;
    let choice = &choices[0]["message"];
    let mut content = choice["content"].as_str().map(|s| s.to_string());
    // Structured reasoning fields (OpenAI o-series, DeepSeek, OpenRouter, …).
    let mut thinking = choice
        .get("reasoning_content")
        .or_else(|| choice.get("reasoning"))
        .or_else(|| choice.get("thinking"))
        .and_then(|t| t.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    // Local models often put CoT inside `<think>…</think>` in content.
    if let Some(c) = content.take() {
        let (tag_think, rest) = split_think_tags(&c);
        if let Some(t) = tag_think {
            thinking = Some(match thinking {
                Some(mut existing) => {
                    existing.push_str(&t);
                    existing
                }
                None => t,
            });
        }
        content = if rest.is_empty() { None } else { Some(rest) };
    }

    let mut tool_calls = None;
    let mut raw_tool_calls = None;
    if let Some(arr) = choice["tool_calls"].as_array() {
        let mut tcs = Vec::new();
        for tc in arr {
            let id = tc["id"].as_str().unwrap_or("call").to_string();
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            let args = tc["function"]["arguments"]
                .as_str()
                .unwrap_or("{}")
                .to_string();
            tcs.push(ToolCall {
                id,
                name,
                arguments: serde_json::from_str(&args).unwrap_or(json!({})),
            });
        }
        if !tcs.is_empty() {
            tool_calls = Some(tcs);
            raw_tool_calls = Some(Value::Array(arr.clone()));
        }
    }

    Ok(LlmResponse {
        content,
        thinking,
        tool_calls,
        raw_tool_calls,
        streamed: false,
    })
}

/// Pull `<think>…</think>` (and `<thinking>…`) blocks out of assistant text.
/// Returns `(thinking, remaining_content)`.
fn split_think_tags(s: &str) -> (Option<String>, String) {
    let lower = s.to_ascii_lowercase();
    let open_tags = ["<think>", "<thinking>"];
    let close_tags = ["</think>", "</thinking>"];

    let mut thinking = String::new();
    let mut content = String::new();
    let mut i = 0;
    let bytes = s.as_bytes();
    let lower_bytes = lower.as_bytes();

    while i < bytes.len() {
        // Find the next open tag at or after i.
        let mut open_at: Option<(usize, usize, usize)> = None; // (pos, open_len, close_idx)
        for (ti, tag) in open_tags.iter().enumerate() {
            if let Some(rel) = find_bytes(&lower_bytes[i..], tag.as_bytes()) {
                let pos = i + rel;
                let candidate = (pos, tag.len(), ti);
                open_at = Some(match open_at {
                    Some(prev) if prev.0 <= pos => prev,
                    _ => candidate,
                });
            }
        }
        let Some((pos, open_len, ti)) = open_at else {
            content.push_str(&s[i..]);
            break;
        };
        content.push_str(&s[i..pos]);
        let body_start = pos + open_len;
        let close = close_tags[ti].as_bytes();
        if let Some(rel) = find_bytes(&lower_bytes[body_start..], close) {
            let body_end = body_start + rel;
            if !thinking.is_empty() {
                thinking.push('\n');
            }
            thinking.push_str(&s[body_start..body_end]);
            i = body_end + close.len();
        } else {
            // Unclosed tag: rest is thinking.
            if !thinking.is_empty() {
                thinking.push('\n');
            }
            thinking.push_str(&s[body_start..]);
            i = bytes.len();
        }
    }

    let thinking = if thinking.is_empty() {
        None
    } else {
        Some(thinking)
    };
    (thinking, content)
}

fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Append `chunk` and drain complete lines, returning the payload of each
/// `data:` line. SSE payloads are JSON (no raw newlines), so complete lines
/// are complete UTF-8.
fn drain_sse_lines(buf: &mut Vec<u8>, chunk: &[u8]) -> Vec<String> {
    buf.extend_from_slice(chunk);
    let mut out = Vec::new();
    while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
        let line_bytes: Vec<u8> = buf.drain(..=pos).collect();
        let line = String::from_utf8_lossy(&line_bytes);
        let line = line.trim();
        if let Some(payload) = line.strip_prefix("data:") {
            let payload = payload.trim();
            if !payload.is_empty() {
                out.push(payload.to_string());
            }
        }
    }
    out
}

/// Accumulates OpenAI streaming deltas (content, reasoning, and indexed
/// tool-call fragments) into a complete [`LlmResponse`].
#[derive(Default)]
struct SseAccumulator {
    content: String,
    thinking: String,
    tool_calls: Vec<PartialToolCall>,
    /// Streaming splitter for models that emit CoT as `<think>` inside content.
    think_filter: ThinkTagFilter,
}

#[derive(Default, Clone)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Incremental splitter for `<think>…</think>` / `<thinking>…</thinking>` tags
/// that arrive mid-stream inside `delta.content`.
#[derive(Default)]
struct ThinkTagFilter {
    in_think: bool,
    /// Partial match for an open/close tag split across chunks.
    hold: String,
}

impl ThinkTagFilter {
    fn push(&mut self, text: &str) -> Vec<StreamPiece> {
        let mut out = Vec::new();
        // Work on a combined buffer so tags split across chunks still match.
        self.hold.push_str(text);
        loop {
            if self.hold.is_empty() {
                break;
            }
            let lower = self.hold.to_ascii_lowercase();
            if self.in_think {
                let close = if let Some(i) = lower.find("</think>") {
                    Some((i, 8))
                } else if let Some(i) = lower.find("</thinking>") {
                    Some((i, 11))
                } else {
                    None
                };
                if let Some((i, n)) = close {
                    let body = self.hold[..i].to_string();
                    if !body.is_empty() {
                        out.push(StreamPiece::Thinking(body));
                    }
                    self.hold = self.hold[i + n..].to_string();
                    self.in_think = false;
                    continue;
                }
                // Keep a short tail that might be a partial close tag.
                let keep = partial_tag_suffix(&lower, &["</think>", "</thinking>"]);
                let emit_len = self.hold.len().saturating_sub(keep);
                if emit_len > 0 {
                    out.push(StreamPiece::Thinking(self.hold[..emit_len].to_string()));
                    self.hold = self.hold[emit_len..].to_string();
                }
                break;
            } else {
                let open = if let Some(i) = lower.find("<think>") {
                    Some((i, 7))
                } else if let Some(i) = lower.find("<thinking>") {
                    Some((i, 10))
                } else {
                    None
                };
                if let Some((i, n)) = open {
                    let body = self.hold[..i].to_string();
                    if !body.is_empty() {
                        out.push(StreamPiece::Content(body));
                    }
                    self.hold = self.hold[i + n..].to_string();
                    self.in_think = true;
                    continue;
                }
                let keep = partial_tag_suffix(&lower, &["<think>", "<thinking>"]);
                let emit_len = self.hold.len().saturating_sub(keep);
                if emit_len > 0 {
                    out.push(StreamPiece::Content(self.hold[..emit_len].to_string()));
                    self.hold = self.hold[emit_len..].to_string();
                }
                break;
            }
        }
        out
    }

    /// Flush any held partial tag text at end of stream (treat as content/thinking).
    fn finish(&mut self) -> Option<StreamPiece> {
        if self.hold.is_empty() {
            return None;
        }
        let rest = std::mem::take(&mut self.hold);
        Some(if self.in_think {
            StreamPiece::Thinking(rest)
        } else {
            StreamPiece::Content(rest)
        })
    }
}

/// Bytes at the end of `lower` that could still grow into one of `tags`.
fn partial_tag_suffix(lower: &str, tags: &[&str]) -> usize {
    let max = tags.iter().map(|t| t.len()).max().unwrap_or(0);
    let n = lower.len().min(max.saturating_sub(1));
    for keep in (1..=n).rev() {
        let suffix = &lower[lower.len() - keep..];
        if tags.iter().any(|t| t.starts_with(suffix)) {
            return keep;
        }
    }
    0
}

impl SseAccumulator {
    /// Feed one `data:` payload. Returns zero or more content/thinking pieces.
    /// Unparsable payloads are skipped (some servers interleave keep-alives);
    /// an explicit `error` object fails the turn.
    fn feed(&mut self, payload: &str) -> Result<Vec<StreamPiece>, LocalCodeError> {
        let v: Value = match serde_json::from_str(payload) {
            Ok(v) => v,
            Err(e) => {
                debug!(error = %e, payload, "skipping unparsable SSE payload");
                return Ok(Vec::new());
            }
        };
        if let Some(err) = v.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("runtime reported an error mid-stream");
            return Err(LocalCodeError::new(
                ErrorCode::BackendNotReady,
                format!("Runtime stream error: {msg}"),
            )
            .with_hint("Check runtime logs")
            .retryable(true));
        }

        let delta = &v["choices"][0]["delta"];
        if let Some(arr) = delta["tool_calls"].as_array() {
            for tc in arr {
                let idx = tc["index"].as_u64().unwrap_or(self.tool_calls.len() as u64) as usize;
                if self.tool_calls.len() <= idx {
                    self.tool_calls.resize(idx + 1, PartialToolCall::default());
                }
                let slot = &mut self.tool_calls[idx];
                if let Some(id) = tc["id"].as_str() {
                    slot.id.push_str(id);
                }
                if let Some(name) = tc["function"]["name"].as_str() {
                    slot.name.push_str(name);
                }
                if let Some(args) = tc["function"]["arguments"].as_str() {
                    slot.arguments.push_str(args);
                }
            }
        }

        let mut pieces = Vec::new();

        // Structured reasoning channels (preferred over tag scraping).
        for key in ["reasoning_content", "reasoning", "thinking"] {
            if let Some(text) = delta.get(key).and_then(|t| t.as_str()) {
                if !text.is_empty() {
                    self.thinking.push_str(text);
                    pieces.push(StreamPiece::Thinking(text.to_string()));
                }
            }
        }

        if let Some(text) = delta["content"].as_str() {
            if !text.is_empty() {
                for piece in self.think_filter.push(text) {
                    match &piece {
                        StreamPiece::Content(t) => self.content.push_str(t),
                        StreamPiece::Thinking(t) => self.thinking.push_str(t),
                    }
                    pieces.push(piece);
                }
            }
        }
        Ok(pieces)
    }

    fn finish(mut self) -> LlmResponse {
        if let Some(piece) = self.think_filter.finish() {
            match piece {
                StreamPiece::Content(t) => self.content.push_str(&t),
                StreamPiece::Thinking(t) => self.thinking.push_str(&t),
            }
        }

        let content = if self.content.is_empty() {
            None
        } else {
            Some(self.content)
        };
        let thinking = if self.thinking.is_empty() {
            None
        } else {
            Some(self.thinking)
        };

        let mut tool_calls = None;
        let mut raw_tool_calls = None;
        let complete: Vec<PartialToolCall> = self
            .tool_calls
            .into_iter()
            .filter(|t| !t.name.is_empty())
            .collect();
        if !complete.is_empty() {
            let raw: Vec<Value> = complete
                .iter()
                .map(|t| {
                    json!({
                        "id": if t.id.is_empty() { "call" } else { t.id.as_str() },
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "arguments": if t.arguments.is_empty() { "{}" } else { t.arguments.as_str() },
                        }
                    })
                })
                .collect();
            let calls: Vec<ToolCall> = complete
                .iter()
                .map(|t| ToolCall {
                    id: if t.id.is_empty() {
                        "call".to_string()
                    } else {
                        t.id.clone()
                    },
                    name: t.name.clone(),
                    arguments: serde_json::from_str(&t.arguments).unwrap_or(json!({})),
                })
                .collect();
            tool_calls = Some(calls);
            raw_tool_calls = Some(Value::Array(raw));
        }

        LlmResponse {
            content,
            thinking,
            tool_calls,
            raw_tool_calls,
            streamed: true,
        }
    }
}

struct LlmResponse {
    content: Option<String>,
    thinking: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    raw_tool_calls: Option<Value>,
    /// True when content already went out as `Delta` / `ThinkingDelta` events.
    streamed: bool,
}

/// Approximate size of a message as sent to the model.
fn approx_chars(m: &ChatMessage) -> usize {
    m.content.chars().count()
        + m.tool_calls
            .as_ref()
            .map(|v| v.to_string().chars().count())
            .unwrap_or(0)
        + m.thinking.as_ref().map(|t| t.chars().count()).unwrap_or(0)
}

/// Build the message list for a request: optional compaction summary + kept
/// tail, then char-budget trim at user boundaries.
fn build_request_view(
    session: &AgentSession,
    system: &str,
    budget: usize,
) -> (Vec<ChatMessage>, String) {
    let mut system_content = system.to_string();
    let mut view: Vec<ChatMessage> = Vec::new();

    if let Some(c) = &session.compaction {
        let idx = c.first_kept_index.min(session.messages.len());
        view.push(ChatMessage::user(format!(
            "[Summary of earlier conversation]\n{}",
            c.summary
        )));
        view.extend(session.messages[idx..].iter().cloned());
        system_content
            .push_str("\n\n(Earlier turns were replaced by the summary message above.)");
    } else {
        view.extend(session.messages.iter().cloned());
    }

    let full_len = view.len();
    let start = {
        let trimmed = trimmed_tail(&view, budget);
        full_len.saturating_sub(trimmed.len())
    };
    if start > 0 {
        debug!(kept = full_len - start, of = full_len, "history trimmed for context budget");
        system_content.push_str(
            "\n\n(Older turns were dropped from this request to fit the model's context window.)",
        );
        view.drain(..start);
    }
    (view, system_content)
}

/// Newest suffix of `messages` that fits `budget` chars, cut only at a user message.
fn trimmed_tail(messages: &[ChatMessage], budget: usize) -> &[ChatMessage] {
    if budget == 0 || messages.is_empty() {
        return messages;
    }
    let mut total = 0usize;
    let mut start: Option<usize> = None;
    for (i, m) in messages.iter().enumerate().rev() {
        total += approx_chars(m);
        let within = total <= budget;
        if m.role == "user" && (within || start.is_none()) {
            start = Some(i);
        }
        if !within && start.is_some() {
            break;
        }
    }
    &messages[start.unwrap_or(0)..]
}

/// Index of the first message to keep verbatim so the recent tail is ~`keep` chars.
fn first_kept_index(messages: &[ChatMessage], keep: usize) -> usize {
    if messages.is_empty() {
        return 0;
    }
    let mut total = 0usize;
    let mut start = 0usize;
    for (i, m) in messages.iter().enumerate().rev() {
        total += approx_chars(m);
        if m.role == "user" {
            start = i;
        }
        if total >= keep && m.role == "user" {
            break;
        }
    }
    start
}

fn fallback_summary(messages: &[ChatMessage]) -> String {
    let mut lines = vec!["Earlier conversation (auto-summary fallback):".to_string()];
    for m in messages.iter().filter(|m| m.role == "user" || m.role == "assistant").take(20) {
        lines.push(format!("{}: {}", m.role, preview(&m.content, 160)));
    }
    lines.join("\n")
}

/// Block exact 3-in-a-row repeats and 2-signature alternating thrash.
fn should_block_tool_sig(
    sig: &str,
    recent: &[String],
    blocked: &std::collections::HashSet<String>,
) -> bool {
    if blocked.contains(sig) {
        return true;
    }
    // Three identical in a row (including this one).
    let n = recent.len();
    if n >= 2 && recent[n - 1] == sig && recent[n - 2] == sig {
        return true;
    }
    // Alternating A B A B A with this as the next A or B.
    if n >= 4 {
        let a = &recent[n - 2];
        let b = &recent[n - 1];
        if a != b
            && recent[n - 4] == *a
            && recent[n - 3] == *b
            && ((sig == a.as_str() && b.as_str() != sig) || (sig == b.as_str() && a.as_str() != sig))
        {
            // Pattern A B A B + next is A or B again
            if sig == a.as_str() || sig == b.as_str() {
                return true;
            }
        }
    }
    // Count occurrences of this sig in recent window.
    let count = recent.iter().filter(|s| s.as_str() == sig).count();
    count >= 3
}

fn resolve_skills_dir(config: &AgentConfig) -> PathBuf {
    if let Some(p) = config.skills_dir.as_deref() {
        return expand_user_path(p);
    }
    AppPaths::resolve()
        .map(|p| p.data_dir.join("skills"))
        .unwrap_or_else(|_| PathBuf::from(".localcode/skills"))
}

fn expand_user_path(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
            return PathBuf::from(home).join(rest);
        }
    }
    if p == "~" {
        if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(p)
}

/// Read a project `AGENTS.md` from the workspace root (or `.localcode/AGENTS.md`).
fn read_agents_md(workspace: &std::path::Path) -> Option<String> {
    const MAX: usize = 8_000;
    for path in [
        workspace.join("AGENTS.md"),
        workspace.join(".localcode").join("AGENTS.md"),
    ] {
        let Ok(body) = std::fs::read_to_string(&path) else {
            continue;
        };
        let body = body.trim();
        if body.is_empty() {
            continue;
        }
        if body.chars().count() > MAX {
            let mut s: String = body.chars().take(MAX).collect();
            s.push_str("\n… (AGENTS.md truncated)");
            return Some(s);
        }
        return Some(body.to_string());
    }
    None
}

/// Parse all pseudo tool-call lines: `tool:name {...}` at line start.
fn parse_pseudo_tools(text: &str) -> Vec<ToolCall> {
    let Ok(re) = regex::Regex::new(r"(?m)^\s*tool:([\w.]+)\s+(\{.*\})\s*$") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for caps in re.captures_iter(text) {
        let Ok(arguments) = serde_json::from_str::<Value>(&caps[2]) else {
            continue;
        };
        out.push(ToolCall {
            id: Uuid::new_v4().to_string(),
            name: caps[1].to_string(),
            arguments,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agents_md_is_read_into_system_prompt() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "Always run cargo fmt before commits.")
            .unwrap();
        let agent = CodingAgent::new(AgentConfig::default());
        let session = AgentSession::new(dir.path().to_path_buf());
        let prompt = agent.build_system_prompt(&session);
        assert!(prompt.contains("Always run cargo fmt"), "{prompt}");
        assert!(prompt.contains("AGENTS.md"));

        // Off switch is honored.
        let agent = CodingAgent::new(AgentConfig {
            use_agents_md: false,
            ..AgentConfig::default()
        });
        let prompt = agent.build_system_prompt(&session);
        assert!(!prompt.contains("Always run cargo fmt"));
    }

    #[test]
    fn custom_system_prompt_replaces_preamble_but_keeps_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let agent = CodingAgent::new(AgentConfig {
            system_prompt: Some("You are Grover, a terse assistant.".into()),
            ..AgentConfig::default()
        });
        let session = AgentSession::new(dir.path().to_path_buf());
        let prompt = agent.build_system_prompt(&session);
        assert!(prompt.contains("Grover"));
        assert!(!prompt.contains("LocalCode coding agent"));
        assert!(prompt.contains("Workspace:"));
    }

    #[test]
    fn disabled_tools_are_hidden_from_schema() {
        let agent = CodingAgent::new(AgentConfig {
            disabled_tools: vec!["bash".into()],
            ..AgentConfig::default()
        });
        let schema = agent.tools.openai_tools_schema();
        let names: Vec<String> = schema
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t["function"]["name"].as_str().map(str::to_string))
            .collect();
        assert!(!names.iter().any(|n| n == "bash"), "{names:?}");
        assert!(names.iter().any(|n| n == "read"));
    }

    #[test]
    fn pseudo_tools_parse_all_line_start() {
        let multi = "tool:read {\"path\": \"a.txt\"}\ntool:ls {\"path\": \".\"}\n";
        let tcs = parse_pseudo_tools(multi);
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0].name, "read");
        assert_eq!(tcs[1].name, "ls");
        assert!(parse_pseudo_tools("  tool:ls {}").len() == 1);
        assert!(
            parse_pseudo_tools("you could run tool:read {\"path\": \"a\"} to do this").is_empty()
        );
        assert!(parse_pseudo_tools("tool:read {not json}").is_empty());
    }

    #[test]
    fn history_trimming_cuts_at_user_turns() {
        let user = |s: &str| ChatMessage::user(s);
        let asst = |s: &str, tc: Option<Value>| ChatMessage::assistant(s, tc, None);
        let tool = |s: &str| ChatMessage::tool(s, "c1".into(), "read".into());

        // Two turns; the second fits the budget on its own.
        let msgs = vec![
            user("first question"),                       // 14
            asst("", Some(json!([{"id": "c1"}]))),        // tool_calls json
            tool(&"x".repeat(200)),                       // 200
            asst("first answer", None),
            user("second question"),
            asst("second answer", None),
        ];
        let t = trimmed_tail(&msgs, 60);
        assert_eq!(t.len(), 2, "keeps only the last user turn");
        assert_eq!(t[0].role, "user");
        assert_eq!(t[0].content, "second question");

        // A budget big enough for everything keeps everything.
        assert_eq!(trimmed_tail(&msgs, 100_000).len(), msgs.len());
        // Zero disables trimming.
        assert_eq!(trimmed_tail(&msgs, 0).len(), msgs.len());

        // Even when the last turn alone exceeds the budget, it is kept whole —
        // and starts at its user message, not mid-exchange.
        let big = vec![
            user("old"),
            asst("old answer", None),
            user("new"),
            asst("", Some(json!([{"id": "c9"}]))),
            tool(&"y".repeat(5_000)),
            asst("done", None),
        ];
        let t = trimmed_tail(&big, 100);
        assert_eq!(t[0].content, "new");
        assert_eq!(t.len(), 4);
    }

    #[test]
    fn sanitize_history_truncates_at_first_violation() {
        let user = |s: &str| ChatMessage::user(s);
        let asst = |s: &str, tc: Option<Value>| ChatMessage::assistant(s, tc, None);
        let tool = |s: &str, id: &str| ChatMessage::tool(s, id.into(), "read".into());
        let calls = |ids: &[&str]| {
            let arr: Vec<Value> = ids.iter().map(|id| json!({"id": id})).collect();
            Some(json!(arr))
        };

        // A complete history is untouched.
        let mut ok = vec![
            user("q"),
            asst("", calls(&["c1", "c2"])),
            tool("r1", "c1"),
            tool("r2", "c2"),
            asst("answer", None),
        ];
        assert_eq!(sanitize_history(&mut ok), 0);
        assert_eq!(ok.len(), 5);

        // Aborted turn: calls with no replies drop from the assistant on.
        let mut aborted = vec![user("q"), asst("", calls(&["c1"]))];
        assert_eq!(sanitize_history(&mut aborted), 1);
        assert_eq!(aborted.len(), 1);

        // Partial replies count as a violation too.
        let mut partial = vec![
            user("q"),
            asst("", calls(&["c1", "c2"])),
            tool("r1", "c1"),
            user("later"),
        ];
        assert_eq!(sanitize_history(&mut partial), 3);
        assert_eq!(partial.len(), 1);

        // A stray tool reply with no preceding call is a violation.
        let mut stray = vec![user("q"), tool("r", "c9"), asst("a", None)];
        assert_eq!(sanitize_history(&mut stray), 2);
        assert_eq!(stray.len(), 1);
    }

    #[test]
    fn tool_messages_serialize_protocol_fields() {
        let m = ChatMessage::tool("ok", "call_1".into(), "read".into());
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["tool_call_id"], "call_1");
        let a = ChatMessage::assistant("", Some(json!([{"id": "call_1"}])), None);
        let v = serde_json::to_value(&a).unwrap();
        assert_eq!(v["tool_calls"][0]["id"], "call_1");
    }

    #[test]
    fn sse_lines_survive_chunk_splits() {
        let mut buf = Vec::new();
        // "é" (2 bytes) split across chunks inside one data line.
        let full = "data: {\"choices\":[{\"delta\":{\"content\":\"café\"}}]}\n".as_bytes();
        let (a, b) = full.split_at(full.len() - 4); // split inside the é
        assert!(drain_sse_lines(&mut buf, a).is_empty());
        let lines = drain_sse_lines(&mut buf, b);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("café"));

        // CRLF + keep-alive comments + [DONE].
        let mut buf = Vec::new();
        let lines = drain_sse_lines(&mut buf, b": ping\r\ndata: [DONE]\r\n".as_ref());
        assert_eq!(lines, vec!["[DONE]".to_string()]);
    }

    #[test]
    fn sse_accumulates_content_deltas() {
        let mut acc = SseAccumulator::default();
        let d1 = acc
            .feed(r#"{"choices":[{"delta":{"content":"Hel"}}]}"#)
            .unwrap();
        assert_eq!(d1, vec![StreamPiece::Content("Hel".into())]);
        let d2 = acc
            .feed(r#"{"choices":[{"delta":{"content":"lo"}}]}"#)
            .unwrap();
        assert_eq!(d2, vec![StreamPiece::Content("lo".into())]);
        // Role-only / finish chunks produce no delta.
        assert!(acc
            .feed(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#)
            .unwrap()
            .is_empty());
        let resp = acc.finish();
        assert_eq!(resp.content.as_deref(), Some("Hello"));
        assert!(resp.thinking.is_none());
        assert!(resp.tool_calls.is_none());
        assert!(resp.streamed);
    }

    #[test]
    fn sse_accumulates_reasoning_content() {
        let mut acc = SseAccumulator::default();
        let d = acc
            .feed(r#"{"choices":[{"delta":{"reasoning_content":"hmm"}}]}"#)
            .unwrap();
        assert_eq!(d, vec![StreamPiece::Thinking("hmm".into())]);
        let d = acc
            .feed(r#"{"choices":[{"delta":{"content":"ok"}}]}"#)
            .unwrap();
        assert_eq!(d, vec![StreamPiece::Content("ok".into())]);
        let resp = acc.finish();
        assert_eq!(resp.thinking.as_deref(), Some("hmm"));
        assert_eq!(resp.content.as_deref(), Some("ok"));
    }

    #[test]
    fn sse_splits_think_tags_in_content() {
        let mut acc = SseAccumulator::default();
        // Tag split across chunks.
        for payload in [
            r#"{"choices":[{"delta":{"content":"<thi"}}]}"#,
            r#"{"choices":[{"delta":{"content":"nk>plan"}}]}"#,
            r#"{"choices":[{"delta":{"content":"</think>done"}}]}"#,
        ] {
            acc.feed(payload).unwrap();
        }
        let resp = acc.finish();
        assert_eq!(resp.thinking.as_deref(), Some("plan"));
        assert_eq!(resp.content.as_deref(), Some("done"));
    }

    #[test]
    fn split_think_tags_extracts_blocks() {
        let (t, c) = split_think_tags("<think>reason</think>\nanswer");
        assert_eq!(t.as_deref(), Some("reason"));
        assert_eq!(c, "\nanswer");
        let (t, c) = split_think_tags("no tags here");
        assert!(t.is_none());
        assert_eq!(c, "no tags here");
    }

    #[test]
    fn sse_accumulates_tool_call_fragments() {
        let mut acc = SseAccumulator::default();
        for payload in [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_9","function":{"name":"fs.read","arguments":""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.txt\"}"}}]}}]}"#,
        ] {
            acc.feed(payload).unwrap();
        }
        let resp = acc.finish();
        let calls = resp.tool_calls.expect("tool calls accumulated");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_9");
        assert_eq!(calls[0].name, "fs.read");
        assert_eq!(calls[0].arguments["path"], "a.txt");
        // Raw array must replay as valid OpenAI tool_calls JSON.
        let raw = resp.raw_tool_calls.unwrap();
        assert_eq!(raw[0]["function"]["name"], "fs.read");
        assert_eq!(
            raw[0]["function"]["arguments"].as_str().unwrap(),
            "{\"path\":\"a.txt\"}"
        );
    }

    #[test]
    fn sse_error_payload_fails_turn() {
        let mut acc = SseAccumulator::default();
        let err = acc
            .feed(r#"{"error":{"message":"model exploded"}}"#)
            .expect_err("error payload must fail");
        assert!(err.message.contains("model exploded"));
        // Garbage payloads are skipped, not fatal.
        assert!(acc.feed("not json").unwrap().is_empty());
    }
}
