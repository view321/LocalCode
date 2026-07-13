//! Coding agent with tools, skills, MCP, and subagents.

mod mcp;
mod skills;
mod tools;

pub use mcp::{McpConfig, McpManager, McpServerStatus};
pub use skills::{Skill, SkillLoader};
pub use tools::{ToolCall, ToolResult, ToolRegistry};

use localcode_core::config::AgentConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::runtime::ActiveRuntime;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::info;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
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

        Self {
            tools: ToolRegistry::default_tools(),
            skills: SkillLoader::new(skills_dir),
            mcp: McpManager::new(mcp_path),
            config,
            http: reqwest::Client::new(),
        }
    }

    pub fn reload_skills(&mut self) {
        self.skills.reload();
    }

    /// Run one user turn: LLM + optional tool loop.
    pub async fn run_turn(
        &self,
        session: &mut AgentSession,
        user_text: &str,
        runtime: &ActiveRuntime,
        api_key: Option<&str>,
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

        session.messages.push(ChatMessage {
            role: "user".into(),
            content: user_text.into(),
            tool_call_id: None,
            name: None,
        });

        let system = self.build_system_prompt(session);
        let mut final_text = String::new();
        let subagents = session.subagents_enabled;

        for round in 0..self.config.max_tool_rounds {
            info!(round, "agent tool round");
            let response = self
                .chat_completion(runtime, api_key, &system, &session.messages, subagents)
                .await?;

            if let Some(tool_calls) = response.tool_calls {
                for tc in tool_calls {
                    if tc.name == "subagent.spawn" && !session.subagents_enabled {
                        session.messages.push(ChatMessage {
                            role: "tool".into(),
                            content: "Subagents disabled. Enable in Settings or Coding toggle."
                                .into(),
                            tool_call_id: Some(tc.id.clone()),
                            name: Some(tc.name.clone()),
                        });
                        continue;
                    }

                    let result = self
                        .tools
                        .execute(
                            &tc,
                            &session.workspace_root,
                            self.config.confirm_destructive_tools,
                        )
                        .await;

                    let content = match &result {
                        Ok(r) => r.output.clone(),
                        Err(e) => format!("ERROR {}: {}", e.code, e.message),
                    };

                    session.messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: format!("[tool_call {}]", tc.name),
                        tool_call_id: None,
                        name: None,
                    });
                    session.messages.push(ChatMessage {
                        role: "tool".into(),
                        content,
                        tool_call_id: Some(tc.id),
                        name: Some(tc.name),
                    });
                }
                continue;
            }

            final_text = response.content.unwrap_or_default();
            session.messages.push(ChatMessage {
                role: "assistant".into(),
                content: final_text.clone(),
                tool_call_id: None,
                name: None,
            });
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
            "Use tools to read/search/edit files. Prefer apply_patch for edits.".into(),
            "Be concise. When tools fail, explain causes and next steps.".into(),
        ];
        if session.subagents_enabled {
            parts.push("Subagents are enabled; you may use subagent.spawn for explore/plan.".into());
        } else {
            parts.push("Subagents are DISABLED; do not call subagent.spawn.".into());
        }
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
        subagents: bool,
    ) -> Result<LlmResponse, LocalCodeError> {
        let base = runtime.base_url.trim_end_matches('/');
        let url = if base.ends_with("/v1") {
            format!("{base}/chat/completions")
        } else {
            format!("{base}/v1/chat/completions")
        };

        let mut api_messages = vec![serde_json::json!({
            "role": "system",
            "content": system,
        })];
        for m in messages {
            api_messages.push(serde_json::json!({
                "role": m.role,
                "content": m.content,
            }));
        }

        let model = runtime
            .model_id
            .clone()
            .unwrap_or_else(|| "default".into());

        let tools_schema = self.tools.openai_tools_schema(subagents);

        let body = serde_json::json!({
            "model": model,
            "messages": api_messages,
            "tools": tools_schema,
            "temperature": 0.2,
        });

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
            let t = resp.text().await.unwrap_or_default();
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("LLM error: {t}"),
            )
            .with_hint("Check runtime logs")
            .retryable(true));
        }

        let v: serde_json::Value = resp.json().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::Internal, e.to_string())
        })?;

        let choice = &v["choices"][0]["message"];
        let content = choice["content"].as_str().map(|s| s.to_string());

        let mut tool_calls = None;
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
                    arguments: serde_json::from_str(&args).unwrap_or(serde_json::json!({})),
                });
            }
            if !tcs.is_empty() {
                tool_calls = Some(tcs);
            }
        }

        if tool_calls.is_none() {
            if let Some(ref c) = content {
                if let Some(tc) = parse_pseudo_tool(c) {
                    tool_calls = Some(vec![tc]);
                }
            }
        }

        Ok(LlmResponse {
            content,
            tool_calls,
        })
    }
}

struct LlmResponse {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

fn parse_pseudo_tool(text: &str) -> Option<ToolCall> {
    let re = regex::Regex::new(r"tool:([\w.]+)\s+(\{.*\})").ok()?;
    let caps = re.captures(text)?;
    Some(ToolCall {
        id: Uuid::new_v4().to_string(),
        name: caps[1].to_string(),
        arguments: serde_json::from_str(&caps[2]).unwrap_or(serde_json::json!({})),
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
        .run_turn(&mut session, prompt, runtime, api_key)
        .await
}
