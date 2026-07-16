//! Model-management capability shared by the coding agent and the assistant.
//!
//! [`ModelActions`] is implemented by the TUI (wired to its live registry +
//! deploy pipeline) and handed to whichever agent serves the conversation, so
//! "deploy X" works the same in the default chat and in `/assistant`. The tool
//! schemas, approval gating, and dispatch live here so both agents offer the
//! exact same tool surface.

use crate::tools::{ToolApprover, ToolCall, ToolResult, ToolRisk};
use async_trait::async_trait;
use localcode_core::config::ApprovalMode;
use localcode_core::error::{ErrorCode, LocalCodeError};
use serde_json::{json, Value};

/// Parameters for the `deploy_model` tool.
#[derive(Debug, Clone, Default)]
pub struct DeployToolArgs {
    pub model_id: String,
    pub backend: Option<String>,
    pub quant: Option<String>,
    pub port: Option<u16>,
    pub context: Option<u32>,
    /// Full launch command override (replaces the built one).
    pub command: Option<String>,
}

/// Backend-management actions an agent can perform on the user's behalf.
///
/// Implemented by the TUI and wired to its live `BackendRegistry` + deploy
/// pipeline, so a model the agent deploys/stops/deletes appears in `/dash`
/// and `/models` exactly like a manual action. Each method returns a short,
/// human-readable summary that becomes the tool's output to the model.
#[async_trait]
pub trait ModelActions: Send + Sync {
    async fn deploy(&self, args: DeployToolArgs) -> Result<String, LocalCodeError>;
    async fn stop(&self, id_or_name: &str) -> Result<String, LocalCodeError>;
    async fn list_deployments(&self) -> Result<String, LocalCodeError>;
    async fn list_downloaded(&self) -> Result<String, LocalCodeError>;
    async fn delete_model(&self, model_id: &str) -> Result<String, LocalCodeError>;
    async fn deploy_ui(&self) -> Result<String, LocalCodeError>;
}

/// Names of the model-management tools, in the order they are offered.
pub const MODEL_TOOL_NAMES: &[&str] = &[
    "deploy_model",
    "stop_model",
    "list_deployments",
    "list_downloaded_models",
    "delete_model",
    "deploy_ui",
];

pub fn is_model_tool(name: &str) -> bool {
    MODEL_TOOL_NAMES.contains(&name)
}

/// OpenAI-style function schemas for the model-management tools.
pub fn model_tools_schema() -> Vec<Value> {
    vec![
        tool_schema(
            "deploy_model",
            "Deploy a Hugging Face model to a local backend and wait until it is serving. \
             backend is ollama|llamacpp|vllm|sglang (default: the app's configured backend). \
             Public models need no HF token. Optionally set quant, port, context, or a full \
             command override.",
            json!({
                "type": "object",
                "properties": {
                    "model_id": { "type": "string", "description": "HF model id, e.g. Qwen/Qwen2.5-Coder-7B-Instruct" },
                    "backend": { "type": "string" },
                    "quant": { "type": "string", "description": "quantization label, e.g. Q4_K_M" },
                    "port": { "type": "integer" },
                    "context": { "type": "integer", "description": "max context length" },
                    "command": { "type": "string", "description": "full launch command that replaces the built one" }
                },
                "required": ["model_id"]
            }),
        ),
        tool_schema(
            "stop_model",
            "Stop a running model runtime and free its VRAM. Pass its name or id \
             (see list_deployments).",
            json!({
                "type": "object",
                "properties": { "model": { "type": "string" } },
                "required": ["model"]
            }),
        ),
        tool_schema(
            "list_deployments",
            "List currently running model runtimes (name, backend, status, url).",
            json!({ "type": "object", "properties": {} }),
        ),
        tool_schema(
            "list_downloaded_models",
            "List models whose weights are already downloaded on disk, with sizes.",
            json!({ "type": "object", "properties": {} }),
        ),
        tool_schema(
            "delete_model",
            "Delete a downloaded model's weights from disk to free space (irreversible).",
            json!({
                "type": "object",
                "properties": { "model_id": { "type": "string" } },
                "required": ["model_id"]
            }),
        ),
        tool_schema(
            "deploy_ui",
            "Launch the OpenWebUI browser chat (Docker) wired to the deployed models.",
            json!({ "type": "object", "properties": {} }),
        ),
    ]
}

/// Gate + run one model-management call against `ops`.
///
/// Destructive tools (stop/delete) gate in every mode but AlwaysApprove;
/// deploys gate only under ApproveEdits / AskPermission; lists never gate.
/// When a gated call has no approver (headless), it is refused rather than
/// silently run.
pub async fn execute_model_tool(
    ops: &dyn ModelActions,
    call: &ToolCall,
    approval: ApprovalMode,
    approver: Option<&dyn ToolApprover>,
) -> Result<ToolResult, LocalCodeError> {
    match call.name.as_str() {
        "deploy_model" => {
            let model_id = str_arg(call, "model_id").ok_or_else(|| {
                LocalCodeError::new(ErrorCode::AgentToolFailed, "deploy_model requires model_id")
            })?;
            let args = DeployToolArgs {
                model_id: model_id.clone(),
                backend: str_arg(call, "backend"),
                quant: str_arg(call, "quant"),
                port: call.arguments.get("port").and_then(|v| v.as_u64()).map(|n| n as u16),
                context: call.arguments.get("context").and_then(|v| v.as_u64()).map(|n| n as u32),
                command: str_arg(call, "command"),
            };
            gate(
                approval,
                approver,
                false,
                &format!(
                    "Deploy model {model_id} on {}",
                    args.backend.as_deref().unwrap_or("the default backend")
                ),
            )
            .await?;
            let out = ops.deploy(args).await?;
            Ok(ToolResult { output: out, risk: ToolRisk::Medium })
        }
        "stop_model" => {
            let target = str_arg(call, "model").or_else(|| str_arg(call, "id")).ok_or_else(|| {
                LocalCodeError::new(
                    ErrorCode::AgentToolFailed,
                    "stop_model requires model (name or id)",
                )
            })?;
            gate(approval, approver, true, &format!("Stop runtime {target}")).await?;
            let out = ops.stop(&target).await?;
            Ok(ToolResult { output: out, risk: ToolRisk::High })
        }
        "list_deployments" => Ok(ToolResult {
            output: ops.list_deployments().await?,
            risk: ToolRisk::Low,
        }),
        "list_downloaded_models" => Ok(ToolResult {
            output: ops.list_downloaded().await?,
            risk: ToolRisk::Low,
        }),
        "delete_model" => {
            let model_id = str_arg(call, "model_id").ok_or_else(|| {
                LocalCodeError::new(ErrorCode::AgentToolFailed, "delete_model requires model_id")
            })?;
            gate(
                approval,
                approver,
                true,
                &format!("Delete downloaded weights for {model_id} from disk"),
            )
            .await?;
            let out = ops.delete_model(&model_id).await?;
            Ok(ToolResult { output: out, risk: ToolRisk::High })
        }
        "deploy_ui" => {
            gate(approval, approver, false, "Deploy the OpenWebUI browser chat (Docker)").await?;
            let out = ops.deploy_ui().await?;
            Ok(ToolResult { output: out, risk: ToolRisk::Medium })
        }
        other => Err(LocalCodeError::new(
            ErrorCode::AgentToolFailed,
            format!("'{other}' is not a model-management tool"),
        )),
    }
}

async fn gate(
    approval: ApprovalMode,
    approver: Option<&dyn ToolApprover>,
    destructive: bool,
    desc: &str,
) -> Result<(), LocalCodeError> {
    let needs = match approval {
        ApprovalMode::AlwaysApprove => false,
        ApprovalMode::Auto => destructive,
        ApprovalMode::ApproveEdits | ApprovalMode::AskPermission => true,
    };
    if !needs {
        return Ok(());
    }
    let approved = match approver {
        Some(a) => a.approve(desc).await,
        None => false,
    };
    if approved {
        Ok(())
    } else {
        Err(LocalCodeError::new(
            ErrorCode::Cancelled,
            format!("User declined: {desc}"),
        ))
    }
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

/// Read a non-empty string argument from a tool call.
fn str_arg(call: &ToolCall, key: &str) -> Option<String> {
    call.arguments
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct NoopOps;

    #[async_trait]
    impl ModelActions for NoopOps {
        async fn deploy(&self, args: DeployToolArgs) -> Result<String, LocalCodeError> {
            Ok(format!("deployed {}", args.model_id))
        }
        async fn stop(&self, id: &str) -> Result<String, LocalCodeError> {
            Ok(format!("stopped {id}"))
        }
        async fn list_deployments(&self) -> Result<String, LocalCodeError> {
            Ok("none".into())
        }
        async fn list_downloaded(&self) -> Result<String, LocalCodeError> {
            Ok("none".into())
        }
        async fn delete_model(&self, id: &str) -> Result<String, LocalCodeError> {
            Ok(format!("deleted {id}"))
        }
        async fn deploy_ui(&self) -> Result<String, LocalCodeError> {
            Ok("ui".into())
        }
    }

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: "t1".into(),
            name: name.into(),
            arguments: args,
        }
    }

    #[tokio::test]
    async fn deploy_runs_ungated_under_auto() {
        let out = execute_model_tool(
            &NoopOps,
            &call("deploy_model", json!({"model_id": "org/m"})),
            ApprovalMode::Auto,
            None,
        )
        .await
        .unwrap();
        assert_eq!(out.output, "deployed org/m");
    }

    #[tokio::test]
    async fn destructive_tools_refuse_without_approver() {
        let err = execute_model_tool(
            &NoopOps,
            &call("stop_model", json!({"model": "m"})),
            ApprovalMode::Auto,
            None,
        )
        .await
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::Cancelled);

        let err = execute_model_tool(
            &NoopOps,
            &call("delete_model", json!({"model_id": "org/m"})),
            ApprovalMode::Auto,
            None,
        )
        .await
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::Cancelled);
    }

    #[tokio::test]
    async fn lists_never_gate() {
        let out = execute_model_tool(
            &NoopOps,
            &call("list_deployments", json!({})),
            ApprovalMode::AskPermission,
            None,
        )
        .await
        .unwrap();
        assert_eq!(out.output, "none");
    }

    #[test]
    fn schema_covers_all_names() {
        let schemas = model_tools_schema();
        assert_eq!(schemas.len(), MODEL_TOOL_NAMES.len());
        for (s, name) in schemas.iter().zip(MODEL_TOOL_NAMES) {
            assert_eq!(
                s.pointer("/function/name").and_then(|v| v.as_str()),
                Some(*name)
            );
            assert!(is_model_tool(name));
        }
    }
}
