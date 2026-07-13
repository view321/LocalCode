//! Coding agent tools.

use localcode_core::error::{ErrorCode, LocalCodeError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;
use tracing::info;
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub output: String,
    pub risk: ToolRisk,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolRisk {
    Low,
    Medium,
    High,
}

pub struct ToolRegistry;

impl ToolRegistry {
    pub fn default_tools() -> Self {
        Self
    }

    pub fn openai_tools_schema(&self, subagents: bool) -> Value {
        let mut tools = vec![
            tool_schema("fs.read", "Read a file", json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            })),
            tool_schema("fs.list", "List directory", json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            })),
            tool_schema("fs.search", "Search files for pattern", json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "path": { "type": "string" }
                },
                "required": ["pattern"]
            })),
            tool_schema("fs.write", "Write full file contents", json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            })),
            tool_schema("fs.apply_patch", "Apply a unified diff patch", json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" }
                },
                "required": ["path", "old_string", "new_string"]
            })),
            tool_schema("shell.exec", "Run shell command in workspace", json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            })),
            tool_schema("git.status", "Git status", json!({"type":"object","properties":{}})),
            tool_schema("git.diff", "Git diff", json!({"type":"object","properties":{}})),
        ];
        if subagents {
            tools.push(tool_schema(
                "subagent.spawn",
                "Delegate a task to a subagent",
                json!({
                    "type": "object",
                    "properties": {
                        "task": { "type": "string" },
                        "role": { "type": "string" }
                    },
                    "required": ["task"]
                }),
            ));
        }
        Value::Array(tools)
    }

    pub async fn execute(
        &self,
        call: &ToolCall,
        workspace: &Path,
        confirm_destructive: bool,
    ) -> Result<ToolResult, LocalCodeError> {
        info!(tool = %call.name, "tool execute");
        match call.name.as_str() {
            "fs.read" => {
                let path = arg_path(call, workspace, "path")?;
                let content = std::fs::read_to_string(&path).map_err(|e| {
                    LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string())
                        .with_cause(format!("Cannot read {}", path.display()))
                })?;
                Ok(ToolResult {
                    output: content,
                    risk: ToolRisk::Low,
                })
            }
            "fs.list" => {
                let path = arg_path(call, workspace, "path").unwrap_or_else(|_| workspace.to_path_buf());
                let mut entries = vec![];
                for e in std::fs::read_dir(&path).map_err(|e| {
                    LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string())
                })? {
                    let e = e.map_err(|e| {
                        LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string())
                    })?;
                    let meta = e.file_type().ok();
                    let kind = if meta.map(|m| m.is_dir()).unwrap_or(false) {
                        "dir"
                    } else {
                        "file"
                    };
                    entries.push(format!("{kind} {}", e.file_name().to_string_lossy()));
                }
                entries.sort();
                Ok(ToolResult {
                    output: entries.join("\n"),
                    risk: ToolRisk::Low,
                })
            }
            "fs.search" => {
                let pattern = call
                    .arguments
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        LocalCodeError::new(ErrorCode::AgentToolFailed, "pattern required")
                    })?;
                let root = call
                    .arguments
                    .get("path")
                    .and_then(|v| v.as_str())
                    .map(|p| resolve_path(workspace, p))
                    .transpose()?
                    .unwrap_or_else(|| workspace.to_path_buf());
                let re = regex::Regex::new(pattern).map_err(|e| {
                    LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string())
                })?;
                let mut hits = vec![];
                for entry in WalkDir::new(&root).into_iter().filter_map(|e| e.ok()) {
                    if !entry.file_type().is_file() {
                        continue;
                    }
                    if hits.len() >= 50 {
                        break;
                    }
                    if let Ok(text) = std::fs::read_to_string(entry.path()) {
                        for (i, line) in text.lines().enumerate() {
                            if re.is_match(line) {
                                hits.push(format!(
                                    "{}:{}:{}",
                                    entry.path().display(),
                                    i + 1,
                                    line
                                ));
                                if hits.len() >= 50 {
                                    break;
                                }
                            }
                        }
                    }
                }
                Ok(ToolResult {
                    output: hits.join("\n"),
                    risk: ToolRisk::Low,
                })
            }
            "fs.write" => {
                if confirm_destructive {
                    // In headless/auto path we still apply; TUI layer may pre-confirm
                }
                let path = arg_path(call, workspace, "path")?;
                let content = call
                    .arguments
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&path, content)?;
                Ok(ToolResult {
                    output: format!("wrote {} bytes to {}", content.len(), path.display()),
                    risk: ToolRisk::Medium,
                })
            }
            "fs.apply_patch" => {
                let path = arg_path(call, workspace, "path")?;
                let old = call
                    .arguments
                    .get("old_string")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let new = call
                    .arguments
                    .get("new_string")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let content = std::fs::read_to_string(&path).map_err(|e| {
                    LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string())
                })?;
                if !content.contains(old) {
                    return Err(LocalCodeError::new(
                        ErrorCode::AgentToolFailed,
                        "old_string not found in file",
                    )
                    .with_cause("Patch context mismatch")
                    .with_hint("Re-read the file and retry with exact context"));
                }
                let updated = content.replacen(old, new, 1);
                std::fs::write(&path, updated)?;
                Ok(ToolResult {
                    output: format!("patched {}", path.display()),
                    risk: ToolRisk::Medium,
                })
            }
            "shell.exec" => {
                let command = call
                    .arguments
                    .get("command")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        LocalCodeError::new(ErrorCode::AgentToolFailed, "command required")
                    })?;
                if confirm_destructive && is_destructive(command) {
                    return Err(LocalCodeError::new(
                        ErrorCode::AgentToolFailed,
                        format!("Destructive command requires confirmation: {command}"),
                    )
                    .with_hint("Disable confirm_destructive_tools or confirm in UI")
                    .with_cause("Safety policy blocked shell.exec"));
                }
                let output = if cfg!(windows) {
                    Command::new("cmd")
                        .args(["/C", command])
                        .current_dir(workspace)
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .output()
                        .await
                } else {
                    Command::new("sh")
                        .args(["-c", command])
                        .current_dir(workspace)
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .output()
                        .await
                }
                .map_err(|e| LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string()))?;

                let mut text = String::from_utf8_lossy(&output.stdout).to_string();
                let err = String::from_utf8_lossy(&output.stderr);
                if !err.is_empty() {
                    text.push_str("\nSTDERR:\n");
                    text.push_str(&err);
                }
                text.push_str(&format!("\nexit: {}", output.status.code().unwrap_or(-1)));
                Ok(ToolResult {
                    output: text,
                    risk: ToolRisk::High,
                })
            }
            "git.status" => shell_git(workspace, &["status", "--short"]).await,
            "git.diff" => shell_git(workspace, &["diff"]).await,
            "subagent.spawn" => {
                let task = call
                    .arguments
                    .get("task")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let role = call
                    .arguments
                    .get("role")
                    .and_then(|v| v.as_str())
                    .unwrap_or("explore");
                // Simplified: return structured note (full isolation later)
                Ok(ToolResult {
                    output: format!(
                        "[subagent:{role}] Completed analysis of task: {task}\n(Subagent returned summary placeholder — wire full isolation in hardening phase.)"
                    ),
                    risk: ToolRisk::Medium,
                })
            }
            other => Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("Unknown tool: {other}"),
            )),
        }
    }
}

fn tool_schema(name: &str, description: &str, parameters: Value) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    })
}

fn arg_path(call: &ToolCall, workspace: &Path, key: &str) -> Result<PathBuf, LocalCodeError> {
    let p = call
        .arguments
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| LocalCodeError::new(ErrorCode::AgentToolFailed, format!("{key} required")))?;
    resolve_path(workspace, p)
}

fn resolve_path(workspace: &Path, p: &str) -> Result<PathBuf, LocalCodeError> {
    use std::path::Component;
    let path = PathBuf::from(p);

    // Reject absolute paths outside workspace by requiring relative or under workspace.
    if path.is_absolute() {
        let ws = std::fs::canonicalize(workspace).unwrap_or_else(|_| workspace.to_path_buf());
        let full = std::fs::canonicalize(&path).unwrap_or(path.clone());
        let ws_s = ws.to_string_lossy().to_lowercase();
        let full_s = full.to_string_lossy().to_lowercase();
        if !full_s.starts_with(ws_s.trim_end_matches(['\\', '/'])) {
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                "Path escapes workspace root",
            )
            .with_cause("Safety confinement")
            .with_hint("Use paths under the workspace only"));
        }
        return Ok(full);
    }

    // Normalize relative path: reject `..` that would leave the root
    let mut depth: i32 = 0;
    for c in path.components() {
        match c {
            Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return Err(LocalCodeError::new(
                        ErrorCode::AgentToolFailed,
                        "Path escapes workspace root",
                    )
                    .with_cause("Safety confinement")
                    .with_hint("Use paths under the workspace only"));
                }
            }
            Component::Normal(_) => depth += 1,
            Component::RootDir | Component::Prefix(_) => {
                return Err(LocalCodeError::new(
                    ErrorCode::AgentToolFailed,
                    "Invalid path",
                ));
            }
            Component::CurDir => {}
        }
    }

    Ok(workspace.join(path))
}

fn is_destructive(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();
    lower.contains("rm -rf")
        || lower.contains("del /f")
        || lower.contains("format ")
        || lower.contains("mkfs")
        || lower.contains("shutdown")
        || lower.contains("rd /s")
}

async fn shell_git(workspace: &Path, args: &[&str]) -> Result<ToolResult, LocalCodeError> {
    let output = Command::new("git")
        .args(args)
        .current_dir(workspace)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| {
            LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string())
                .with_cause("git not available")
        })?;
    Ok(ToolResult {
        output: format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        risk: ToolRisk::Low,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn read_write_patch() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        let write = ToolCall {
            id: "1".into(),
            name: "fs.write".into(),
            arguments: json!({"path": "a.txt", "content": "hello world"}),
        };
        reg.execute(&write, dir.path(), false).await.unwrap();
        let patch = ToolCall {
            id: "2".into(),
            name: "fs.apply_patch".into(),
            arguments: json!({
                "path": "a.txt",
                "old_string": "hello",
                "new_string": "hi"
            }),
        };
        reg.execute(&patch, dir.path(), false).await.unwrap();
        let read = ToolCall {
            id: "3".into(),
            name: "fs.read".into(),
            arguments: json!({"path": "a.txt"}),
        };
        let r = reg.execute(&read, dir.path(), false).await.unwrap();
        assert_eq!(r.output, "hi world");
    }
}
