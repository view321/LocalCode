//! Coding agent with tools, skills, and MCP configuration.

mod mcp;
mod skills;
mod tools;

pub use mcp::{McpConfig, McpManager, McpServerStatus};
pub use skills::{Skill, SkillLoader};
pub use tools::{ToolApprover, ToolCall, ToolRegistry, ToolResult};

use futures::StreamExt;
use localcode_core::config::AgentConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::runtime::ActiveRuntime;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
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
        summary: String,
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
}

impl ChatMessage {
    fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
            tool_call_id: None,
            name: None,
            tool_calls: None,
        }
    }

    fn assistant(content: impl Into<String>, tool_calls: Option<Value>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
            tool_call_id: None,
            name: None,
            tool_calls,
        }
    }

    fn tool(content: impl Into<String>, call_id: String, name: String) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
            tool_call_id: Some(call_id),
            name: Some(name),
            tool_calls: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSession {
    pub id: String,
    pub title: String,
    pub messages: Vec<ChatMessage>,
    pub workspace_root: PathBuf,
    pub subagents_enabled: bool,
    pub runtime_id: Option<String>,
}

impl AgentSession {
    pub fn new(workspace: PathBuf, subagents_enabled: bool) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            title: "New session".into(),
            messages: vec![],
            workspace_root: workspace,
            subagents_enabled,
            runtime_id: None,
        }
    }
}

pub struct CodingAgent {
    pub config: AgentConfig,
    pub tools: ToolRegistry,
    pub skills: SkillLoader,
    pub mcp: McpManager,
    http: reqwest::Client,
}

impl CodingAgent {
    pub fn new(config: AgentConfig) -> Self {
        let skills_dir = config
            .skills_dir
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".localcode/skills"));
        let mcp_path = config
            .mcp_config
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".localcode/mcp.json"));

        // Connect timeout only: local generations can legitimately take
        // minutes, so the total duration is unbounded and cancellation is the
        // caller's job (the TUI aborts the task on Esc).
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        Self {
            tools: ToolRegistry::default_tools(),
            skills: SkillLoader::new(skills_dir),
            mcp: McpManager::new(mcp_path),
            config,
            http,
        }
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

        session.messages.push(ChatMessage::user(user_text));

        let system = self.build_system_prompt(session);
        let mut final_text = String::new();

        for round in 0..self.config.max_tool_rounds {
            info!(round, "agent tool round");
            let response = self
                .chat_completion(runtime, api_key, &system, &session.messages, events)
                .await?;

            // When the runtime didn't stream, surface the whole message as one
            // delta so UIs render intermediate rounds the same way.
            if !response.streamed {
                if let Some(c) = response.content.as_deref().filter(|c| !c.is_empty()) {
                    emit(events, AgentEvent::Delta(c.to_string()));
                }
            }
            emit(events, AgentEvent::MessageComplete);

            if let Some(tool_calls) = response.tool_calls {
                // One assistant message carrying the full tool_calls array,
                // then one tool message per call — the shape the OpenAI
                // protocol requires.
                session.messages.push(ChatMessage::assistant(
                    response.content.unwrap_or_default(),
                    response.raw_tool_calls,
                ));

                for tc in tool_calls {
                    emit(
                        events,
                        AgentEvent::ToolStarted {
                            name: tc.name.clone(),
                            args_preview: preview(&tc.arguments.to_string(), 60),
                        },
                    );
                    let result = self
                        .tools
                        .execute(
                            &tc,
                            &session.workspace_root,
                            self.config.confirm_destructive_tools,
                            approver,
                        )
                        .await;

                    let content = match &result {
                        Ok(r) => r.output.clone(),
                        Err(e) => format!("ERROR {}: {}", e.code, e.message),
                    };
                    emit(
                        events,
                        AgentEvent::ToolFinished {
                            name: tc.name.clone(),
                            ok: result.is_ok(),
                            summary: preview(&content, 80),
                        },
                    );

                    session
                        .messages
                        .push(ChatMessage::tool(content, tc.id, tc.name));
                }
                continue;
            }

            final_text = response.content.unwrap_or_default();
            session
                .messages
                .push(ChatMessage::assistant(final_text.clone(), None));
            break;
        }

        if final_text.is_empty() {
            final_text = "(agent completed tool rounds without a final message)".into();
        }

        if session.title == "New session" {
            session.title = user_text.chars().take(40).collect();
        }

        Ok(final_text)
    }

    fn build_system_prompt(&self, session: &AgentSession) -> String {
        let mut parts = vec![
            "You are LocalCode coding agent. Work inside the user workspace.".into(),
            format!("Workspace: {}", session.workspace_root.display()),
            "Use tools to read/search/edit files. fs.apply_patch does exact string replacement."
                .into(),
            "Be concise. When tools fail, explain causes and next steps.".into(),
            "Subagents are not available in this build; do the work yourself.".into(),
        ];
        for skill in self.skills.list() {
            if skill.enabled {
                parts.push(format!(
                    "Skill available: {} — {}",
                    skill.name, skill.description
                ));
            }
        }
        parts.join("\n")
    }

    async fn chat_completion(
        &self,
        runtime: &ActiveRuntime,
        api_key: Option<&str>,
        system: &str,
        messages: &[ChatMessage],
        events: Option<&mpsc::UnboundedSender<AgentEvent>>,
    ) -> Result<LlmResponse, LocalCodeError> {
        let base = runtime.base_url.trim_end_matches('/');
        let url = if base.ends_with("/v1") {
            format!("{base}/chat/completions")
        } else {
            format!("{base}/v1/chat/completions")
        };

        let mut api_messages = vec![json!({
            "role": "system",
            "content": system,
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
        let mut body = json!({
            "model": model,
            "messages": api_messages,
            "tools": self.tools.openai_tools_schema(),
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

        // Fallback for models without native tool-call support: a line of the
        // form `tool:fs.read {"path": "..."}`.
        if response.tool_calls.is_none() {
            if let Some(ref c) = response.content {
                if let Some(tc) = parse_pseudo_tool(c) {
                    response.raw_tool_calls = Some(json!([{
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments.to_string(),
                        }
                    }]));
                    response.tool_calls = Some(vec![tc]);
                }
            }
        }

        Ok(response)
    }

    /// Consume an OpenAI-style SSE stream, emitting `Delta` events as content
    /// arrives and accumulating tool-call fragments into a full response.
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
                if let Some(delta) = acc.feed(&payload)? {
                    emit(events, AgentEvent::Delta(delta));
                }
            }
            if done {
                break;
            }
        }

        let response = acc.finish();
        if response.content.is_none() && response.tool_calls.is_none() {
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
    let content = choice["content"].as_str().map(|s| s.to_string());

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
        tool_calls,
        raw_tool_calls,
        streamed: false,
    })
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

/// Accumulates OpenAI streaming deltas (content text and indexed tool-call
/// fragments) into a complete [`LlmResponse`].
#[derive(Default)]
struct SseAccumulator {
    content: String,
    tool_calls: Vec<PartialToolCall>,
}

#[derive(Default, Clone)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl SseAccumulator {
    /// Feed one `data:` payload. Returns the content delta to surface, if any.
    /// Unparsable payloads are skipped (some servers interleave keep-alives);
    /// an explicit `error` object fails the turn.
    fn feed(&mut self, payload: &str) -> Result<Option<String>, LocalCodeError> {
        let v: Value = match serde_json::from_str(payload) {
            Ok(v) => v,
            Err(e) => {
                debug!(error = %e, payload, "skipping unparsable SSE payload");
                return Ok(None);
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
        if let Some(text) = delta["content"].as_str() {
            if !text.is_empty() {
                self.content.push_str(text);
                return Ok(Some(text.to_string()));
            }
        }
        Ok(None)
    }

    fn finish(self) -> LlmResponse {
        let content = if self.content.is_empty() {
            None
        } else {
            Some(self.content)
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
            tool_calls,
            raw_tool_calls,
            streamed: true,
        }
    }
}

struct LlmResponse {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    raw_tool_calls: Option<Value>,
    /// True when content already went out as `Delta` events.
    streamed: bool,
}

/// Parse a pseudo tool call. Anchored to a line start so a model merely
/// *mentioning* `tool:foo {}` mid-sentence doesn't trigger execution.
fn parse_pseudo_tool(text: &str) -> Option<ToolCall> {
    let re = regex::Regex::new(r"(?m)^\s*tool:([\w.]+)\s+(\{.*\})\s*$").ok()?;
    let caps = re.captures(text)?;
    let arguments: Value = serde_json::from_str(&caps[2]).ok()?;
    Some(ToolCall {
        id: Uuid::new_v4().to_string(),
        name: caps[1].to_string(),
        arguments,
    })
}

/// Headless agent run for CLI.
pub async fn run_headless(
    prompt: &str,
    workspace: PathBuf,
    runtime: &ActiveRuntime,
    config: AgentConfig,
    api_key: Option<&str>,
) -> Result<String, LocalCodeError> {
    let agent = CodingAgent::new(config);
    let mut session = AgentSession::new(workspace, agent.config.subagents_enabled);
    agent
        .run_turn(&mut session, prompt, runtime, api_key, None, None)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pseudo_tool_requires_line_start() {
        assert!(parse_pseudo_tool("tool:fs.read {\"path\": \"a.txt\"}").is_some());
        assert!(parse_pseudo_tool("  tool:fs.list {}").is_some());
        assert!(
            parse_pseudo_tool("you could run tool:fs.read {\"path\": \"a\"} to do this").is_none()
        );
        // Invalid JSON must not silently execute with empty args.
        assert!(parse_pseudo_tool("tool:fs.read {not json}").is_none());
    }

    #[test]
    fn tool_messages_serialize_protocol_fields() {
        let m = ChatMessage::tool("ok", "call_1".into(), "fs.read".into());
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["tool_call_id"], "call_1");
        let a = ChatMessage::assistant("", Some(json!([{"id": "call_1"}])));
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
        assert_eq!(d1.as_deref(), Some("Hel"));
        let d2 = acc
            .feed(r#"{"choices":[{"delta":{"content":"lo"}}]}"#)
            .unwrap();
        assert_eq!(d2.as_deref(), Some("lo"));
        // Role-only / finish chunks produce no delta.
        assert!(acc
            .feed(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#)
            .unwrap()
            .is_none());
        let resp = acc.finish();
        assert_eq!(resp.content.as_deref(), Some("Hello"));
        assert!(resp.tool_calls.is_none());
        assert!(resp.streamed);
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
        assert!(acc.feed("not json").unwrap().is_none());
    }
}
