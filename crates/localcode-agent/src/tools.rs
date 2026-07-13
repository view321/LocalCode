//! Coding agent tools.

use async_trait::async_trait;
use localcode_core::error::{ErrorCode, LocalCodeError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tracing::info;
use walkdir::WalkDir;

/// Max characters of file content returned to the model. Local models often
/// run with small contexts (8k), so tool output must be bounded.
const MAX_READ_CHARS: usize = 24_000;
/// Max characters of shell/git output returned to the model.
const MAX_EXEC_CHARS: usize = 16_000;
/// Max wall-clock time for a shell command.
const EXEC_TIMEOUT_SECS: u64 = 120;
/// Directories that fs.search never descends into.
const SEARCH_IGNORED_DIRS: [&str; 8] = [
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "__pycache__",
    "vendor",
];
/// fs.search skips files larger than this.
const SEARCH_MAX_FILE_BYTES: u64 = 1_000_000;

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

/// Interactive approval hook for risky tool calls. The TUI implements this by
/// showing a confirmation modal; headless runs have no approver, so risky
/// commands are refused instead of silently executed.
#[async_trait]
pub trait ToolApprover: Send + Sync {
    async fn approve(&self, description: &str) -> bool;
}

pub struct ToolRegistry;

impl ToolRegistry {
    pub fn default_tools() -> Self {
        Self
    }

    pub fn openai_tools_schema(&self) -> Value {
        let tools = vec![
            tool_schema("fs.read", "Read a file (output truncated for large files)", json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            })),
            tool_schema("fs.list", "List directory", json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            })),
            tool_schema("fs.search", "Search workspace files for a regex pattern", json!({
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
            tool_schema(
                "fs.apply_patch",
                "Edit a file by exact string replacement: old_string must appear verbatim in the file and is replaced once by new_string. Not a unified diff.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "old_string": { "type": "string" },
                        "new_string": { "type": "string" }
                    },
                    "required": ["path", "old_string", "new_string"]
                }),
            ),
            tool_schema("shell.exec", "Run shell command in workspace", json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            })),
            tool_schema("git.status", "Git status", json!({"type":"object","properties":{}})),
            tool_schema("git.diff", "Git diff", json!({"type":"object","properties":{}})),
        ];
        Value::Array(tools)
    }

    pub async fn execute(
        &self,
        call: &ToolCall,
        workspace: &Path,
        confirm_destructive: bool,
        approver: Option<&dyn ToolApprover>,
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
                    output: truncate_output(&content, MAX_READ_CHARS),
                    risk: ToolRisk::Low,
                })
            }
            "fs.list" => {
                let path =
                    arg_path(call, workspace, "path").unwrap_or_else(|_| workspace.to_path_buf());
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
                let walker = WalkDir::new(&root).into_iter().filter_entry(|e| {
                    !(e.file_type().is_dir()
                        && e.file_name()
                            .to_str()
                            .map(|n| SEARCH_IGNORED_DIRS.contains(&n))
                            .unwrap_or(false))
                });
                'outer: for entry in walker.filter_map(|e| e.ok()) {
                    if !entry.file_type().is_file() {
                        continue;
                    }
                    if entry
                        .metadata()
                        .map(|m| m.len() > SEARCH_MAX_FILE_BYTES)
                        .unwrap_or(true)
                    {
                        continue;
                    }
                    if let Ok(text) = std::fs::read_to_string(entry.path()) {
                        for (i, line) in text.lines().enumerate() {
                            if re.is_match(line) {
                                hits.push(format!(
                                    "{}:{}:{}",
                                    entry.path().display(),
                                    i + 1,
                                    line.trim_end()
                                ));
                                if hits.len() >= 50 {
                                    hits.push("…(more matches truncated at 50)".into());
                                    break 'outer;
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
                    let approved = match approver {
                        Some(a) => {
                            a.approve(&format!("Run shell command:\n\n  {command}")).await
                        }
                        None => false,
                    };
                    if !approved {
                        return Err(LocalCodeError::new(
                            ErrorCode::Cancelled,
                            format!("Destructive command not approved: {command}"),
                        )
                        .with_cause("Safety policy: destructive shell commands need approval")
                        .with_hint(
                            "The user declined (or no interactive approval is available)",
                        ));
                    }
                }
                run_shell(command, workspace).await
            }
            "git.status" => shell_git(workspace, &["status", "--short"]).await,
            "git.diff" => shell_git(workspace, &["diff"]).await,
            "subagent.spawn" => Err(LocalCodeError::new(
                ErrorCode::NotImplemented,
                "Subagents are not available in this build",
            )
            .with_hint("Do the task yourself with the available tools")),
            other => Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("Unknown tool: {other}"),
            )),
        }
    }
}

/// Truncate tool output so it cannot blow a small local-model context.
fn truncate_output(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let head: String = s.chars().take(max_chars).collect();
    format!(
        "{head}\n…[output truncated: {} of {} chars shown]",
        max_chars,
        s.chars().count()
    )
}

async fn run_shell(command: &str, workspace: &Path) -> Result<ToolResult, LocalCodeError> {
    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/C", command]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", command]);
        c
    };
    let mut child = cmd
        .current_dir(workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string()))?;

    // Read pipes concurrently so a chatty command can't fill the pipe buffer
    // and deadlock against our wait().
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let out_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(ref mut s) = stdout {
            let _ = s.read_to_end(&mut buf).await;
        }
        buf
    });
    let err_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(ref mut s) = stderr {
            let _ = s.read_to_end(&mut buf).await;
        }
        buf
    });

    let status = match tokio::time::timeout(
        std::time::Duration::from_secs(EXEC_TIMEOUT_SECS),
        child.wait(),
    )
    .await
    {
        Ok(res) => res.map_err(|e| {
            LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string())
        })?,
        Err(_) => {
            let _ = child.kill().await;
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("Command timed out after {EXEC_TIMEOUT_SECS}s and was killed: {command}"),
            )
            .with_hint("Run long-lived processes outside the agent"));
        }
    };

    let stdout = out_task.await.unwrap_or_default();
    let stderr = err_task.await.unwrap_or_default();
    let mut text = String::from_utf8_lossy(&stdout).to_string();
    let err = String::from_utf8_lossy(&stderr);
    if !err.is_empty() {
        text.push_str("\nSTDERR:\n");
        text.push_str(&err);
    }
    text.push_str(&format!("\nexit: {}", status.code().unwrap_or(-1)));
    Ok(ToolResult {
        output: truncate_output(&text, MAX_EXEC_CHARS),
        risk: ToolRisk::High,
    })
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

fn escape_err() -> LocalCodeError {
    LocalCodeError::new(ErrorCode::AgentToolFailed, "Path escapes workspace root")
        .with_cause("Safety confinement")
        .with_hint("Use paths under the workspace only")
}

/// Resolve a tool-supplied path and confine it to the workspace.
///
/// Strategy: lexically normalize (rejecting `..` past the root), then
/// canonicalize the longest existing ancestor so symlinks and Windows `\\?\`
/// forms are handled, and finally compare with `Path::starts_with`
/// (component-wise — immune to the `/ws` vs `/ws-evil` prefix trap).
fn resolve_path(workspace: &Path, p: &str) -> Result<PathBuf, LocalCodeError> {
    let ws = std::fs::canonicalize(workspace)
        .map_err(|e| LocalCodeError::new(ErrorCode::AgentWorkspaceMissing, e.to_string()))?;

    let raw = PathBuf::from(p);
    let base = if raw.is_absolute() {
        raw
    } else {
        ws.join(raw)
    };

    // Lexical normalization: resolve `.` and `..` without touching the fs.
    let mut normal = PathBuf::new();
    for c in base.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !normal.pop() {
                    return Err(escape_err());
                }
            }
            other => normal.push(other.as_os_str()),
        }
    }

    // Canonicalize the longest existing ancestor, then re-append the rest so
    // not-yet-existing files (fs.write targets) still resolve consistently.
    let resolved = canonicalize_lenient(&normal);

    if resolved.starts_with(&ws) {
        Ok(resolved)
    } else {
        Err(escape_err())
    }
}

fn canonicalize_lenient(path: &Path) -> PathBuf {
    if let Ok(c) = std::fs::canonicalize(path) {
        return c;
    }
    let mut missing: Vec<std::ffi::OsString> = Vec::new();
    let mut cur = path.to_path_buf();
    while let Some(parent) = cur.parent() {
        if let Some(name) = cur.file_name() {
            missing.push(name.to_os_string());
        } else {
            break;
        }
        if let Ok(c) = std::fs::canonicalize(parent) {
            let mut out = c;
            for part in missing.iter().rev() {
                out.push(part);
            }
            return out;
        }
        cur = parent.to_path_buf();
    }
    path.to_path_buf()
}

/// Best-effort blocklist for obviously destructive shell commands. This is a
/// safety net that routes matches through interactive approval — it is not,
/// and cannot be, a complete sandbox.
fn is_destructive(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();
    let patterns = [
        r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+",   // rm -rf / -fr / -r -f
        r"\brm\s+--(recursive|force)",
        r"\bdel\s+/[fsq]",
        r"\brd\s+/s",
        r"\brmdir\s+/s",
        r"remove-item\b.*(-recurse|-force)",
        r"\bformat\s",
        r"\bmkfs",
        r"\bdd\s+[^|]*\bof=",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bgit\s+clean\b.*-[a-z]*f",
        r"\bgit\s+reset\s+--hard",
        r"\bgit\s+push\b.*(--force|-f)\b",
        r":\(\)\s*\{.*\};\s*:",              // fork bomb
        r"\bchmod\s+-r\b",
        r"\bchown\s+-r\b",
    ];
    patterns.iter().any(|p| {
        regex::Regex::new(p)
            .map(|re| re.is_match(&lower))
            .unwrap_or(false)
    })
}

async fn shell_git(workspace: &Path, args: &[&str]) -> Result<ToolResult, LocalCodeError> {
    let output = Command::new("git")
        .args(args)
        .current_dir(workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| {
            LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string())
                .with_cause("git not available")
        })?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(ToolResult {
        output: truncate_output(&text, MAX_EXEC_CHARS),
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
        reg.execute(&write, dir.path(), false, None).await.unwrap();
        let patch = ToolCall {
            id: "2".into(),
            name: "fs.apply_patch".into(),
            arguments: json!({
                "path": "a.txt",
                "old_string": "hello",
                "new_string": "hi"
            }),
        };
        reg.execute(&patch, dir.path(), false, None).await.unwrap();
        let read = ToolCall {
            id: "3".into(),
            name: "fs.read".into(),
            arguments: json!({"path": "a.txt"}),
        };
        let r = reg.execute(&read, dir.path(), false, None).await.unwrap();
        assert_eq!(r.output, "hi world");
    }

    #[test]
    fn sibling_dir_is_rejected() {
        let dir = tempdir().unwrap();
        let ws = dir.path().join("ws");
        let evil = dir.path().join("ws-evil");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&evil).unwrap();
        std::fs::write(evil.join("x.txt"), "secret").unwrap();

        // Absolute sibling path sharing the workspace prefix must be rejected.
        let attack = evil.join("x.txt").display().to_string();
        assert!(resolve_path(&ws, &attack).is_err());

        // Relative traversal out of the workspace must be rejected.
        assert!(resolve_path(&ws, "../ws-evil/x.txt").is_err());

        // Legit relative path (including to a not-yet-existing file) works.
        let ok = resolve_path(&ws, "sub/new.txt").unwrap();
        assert!(ok.starts_with(std::fs::canonicalize(&ws).unwrap()));
    }

    #[test]
    fn destructive_variants_detected() {
        for cmd in [
            "rm -rf /",
            "rm -fr .",
            "rm -r -f target",
            "rm --recursive --force x",
            "del /f /s /q C:\\x",
            "rd /s /q C:\\x",
            "Remove-Item -Recurse -Force x",
            "git clean -fdx",
            "git reset --hard HEAD~5",
            "git push --force origin main",
            "dd if=/dev/zero of=/dev/sda",
        ] {
            assert!(is_destructive(cmd), "should flag: {cmd}");
        }
        for cmd in ["cargo test", "rm", "git status", "ls -la", "git push origin main"] {
            assert!(!is_destructive(cmd), "should not flag: {cmd}");
        }
    }

    #[tokio::test]
    async fn shell_output_is_truncated_and_bounded() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        let call = ToolCall {
            id: "1".into(),
            name: "shell.exec".into(),
            arguments: json!({ "command": "echo hello" }),
        };
        let r = reg.execute(&call, dir.path(), false, None).await.unwrap();
        assert!(r.output.contains("hello"));
        assert!(r.output.contains("exit: 0"));
    }

    #[tokio::test]
    async fn destructive_without_approver_is_refused() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        let call = ToolCall {
            id: "1".into(),
            name: "shell.exec".into(),
            arguments: json!({ "command": "rm -rf ." }),
        };
        let err = reg
            .execute(&call, dir.path(), true, None)
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::Cancelled);
    }
}
