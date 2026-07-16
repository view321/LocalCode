//! Coding agent tools (Pi-style minimal set: read, write, bash, ls, grep + skill).

use async_trait::async_trait;
use localcode_core::config::ApprovalMode;
use localcode_core::error::{ErrorCode, LocalCodeError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tracing::info;
use walkdir::WalkDir;

/// Max characters of file content returned to the model.
const MAX_READ_CHARS: usize = 24_000;
/// Max characters of shell/grep output returned to the model.
const MAX_EXEC_CHARS: usize = 16_000;
/// Default wall-clock cap for a shell command when the config doesn't set one.
const DEFAULT_EXEC_TIMEOUT_SECS: u64 = 120;
/// After the child exits, how long to keep draining its pipes before giving up
/// (a leaked background grandchild can hold them open forever).
const DRAIN_GRACE_SECS: u64 = 5;
/// Longest single line `grep` returns before it is clipped (keeps a match in a
/// minified/lockfile line from flooding the model context).
const MAX_GREP_LINE_CHARS: usize = 400;
/// Max directory entries returned by `ls`.
const MAX_LIST_ENTRIES: usize = 200;
/// Directories that `grep` never descends into.
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
/// `grep` skips files larger than this.
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

/// Interactive approval hook for gated tool calls.
#[async_trait]
pub trait ToolApprover: Send + Sync {
    async fn approve(&self, description: &str) -> bool;
}

/// Whether `call` needs interactive approval under `mode`.
pub fn approval_request(call: &ToolCall, mode: ApprovalMode) -> Option<String> {
    let arg = |k: &str| call.arguments.get(k).and_then(|v| v.as_str()).unwrap_or("?");
    let describe = || match call.name.as_str() {
        "bash" => {
            let cmd = arg("command");
            let flag = if is_destructive(cmd) {
                " (flagged destructive)"
            } else {
                ""
            };
            format!("Run shell command{flag}:\n\n  {cmd}")
        }
        "write" => {
            // Note: workspace isn't available here, so we can't reliably tell
            // "new file" from "overwrite" — say the honest thing and let the
            // content preview show exactly what would be written.
            let content = call
                .arguments
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let n = content.chars().count();
            format!(
                "Write file (replaces any existing content): {} ({n} chars)\n\n{}",
                arg("path"),
                content_preview(content)
            )
        }
        "edit" => {
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
            format!(
                "Edit file: {}\n\n- remove:\n{}\n\n+ insert:\n{}",
                arg("path"),
                content_preview(old),
                content_preview(new)
            )
        }
        "read" => format!("Read file: {}", arg("path")),
        "ls" => format!("List directory: {}", arg("path")),
        "grep" => format!("Search files for: {}", arg("pattern")),
        "skill" => format!("Load skill: {}", arg("name")),
        other => format!("Run tool: {other}"),
    };
    let gated = match mode {
        ApprovalMode::AlwaysApprove => false,
        ApprovalMode::Auto => {
            call.name == "bash"
                && call
                    .arguments
                    .get("command")
                    .and_then(|v| v.as_str())
                    .map(is_destructive)
                    .unwrap_or(false)
        }
        ApprovalMode::ApproveEdits => {
            matches!(call.name.as_str(), "write" | "edit" | "bash")
        }
        ApprovalMode::AskPermission => true,
    };
    gated.then(describe)
}

pub struct ToolRegistry {
    disabled: std::collections::HashSet<String>,
    /// When true, shell stays confined to the workspace (cwd + path checks).
    shell_sandbox: bool,
    /// Wall-clock cap for a single `bash` command.
    exec_timeout: Duration,
}

impl ToolRegistry {
    pub fn default_tools() -> Self {
        Self {
            disabled: std::collections::HashSet::new(),
            shell_sandbox: true,
            exec_timeout: Duration::from_secs(DEFAULT_EXEC_TIMEOUT_SECS),
        }
    }

    pub fn new(
        disabled: impl IntoIterator<Item = String>,
        shell_sandbox: bool,
        exec_timeout_secs: u64,
    ) -> Self {
        let secs = if exec_timeout_secs == 0 {
            DEFAULT_EXEC_TIMEOUT_SECS
        } else {
            exec_timeout_secs
        };
        Self {
            disabled: disabled.into_iter().collect(),
            shell_sandbox,
            exec_timeout: Duration::from_secs(secs),
        }
    }

    /// Built-in tool catalog: `(name, one-line description)`.
    pub fn catalog() -> &'static [(&'static str, &'static str)] {
        &[
            ("read", "Read a file"),
            ("write", "Write full file contents"),
            ("edit", "Replace an exact string in a file"),
            ("bash", "Run a shell command in the workspace"),
            ("ls", "List a directory"),
            ("grep", "Search files by regex"),
            ("skill", "Load a skill's full instructions"),
            ("hf.model_card", "Fetch a Hugging Face model card (README)"),
            ("hf.search", "Search the Hugging Face model catalogue"),
            ("deploy_model", "Deploy a Hugging Face model to a local backend"),
            ("stop_model", "Stop a running model runtime"),
            ("list_deployments", "List running model runtimes"),
            ("list_downloaded_models", "List models downloaded on disk"),
            ("delete_model", "Delete a downloaded model's weights"),
            ("deploy_ui", "Launch the OpenWebUI browser chat"),
        ]
    }

    pub fn is_enabled(&self, name: &str) -> bool {
        !self.disabled.contains(name)
    }

    pub fn openai_tools_schema(&self) -> Value {
        let tools = vec![
            tool_schema(
                "read",
                "Read a file (output truncated for large files). For big files pass start_line (1-based) and max_lines to read a slice.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "start_line": { "type": "integer", "description": "1-based first line to read" },
                        "max_lines": { "type": "integer", "description": "max lines to return" }
                    },
                    "required": ["path"]
                }),
            ),
            tool_schema(
                "write",
                "Write full file contents (creates parent directories). Best for new files or full rewrites; to change part of an existing file prefer `edit`, which cannot silently drop unseen content.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    },
                    "required": ["path", "content"]
                }),
            ),
            tool_schema(
                "edit",
                "Replace an exact substring in an existing file. `old_string` must occur exactly once (include surrounding lines to disambiguate) unless replace_all is true. Prefer this over `write` for edits to large files so untouched content is never lost.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "old_string": { "type": "string", "description": "Exact text to replace (must match verbatim, including indentation)" },
                        "new_string": { "type": "string", "description": "Replacement text" },
                        "replace_all": { "type": "boolean", "description": "Replace every occurrence instead of requiring a unique match" }
                    },
                    "required": ["path", "old_string", "new_string"]
                }),
            ),
            tool_schema(
                "bash",
                "Run a shell command in the workspace. Output is truncated. Prefer read/write/ls/grep for file work.",
                json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" },
                        "working_directory": {
                            "type": "string",
                            "description": "Optional subpath under the workspace"
                        }
                    },
                    "required": ["command"]
                }),
            ),
            tool_schema(
                "ls",
                "List a directory (capped; dirs first).",
                json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
            ),
            tool_schema(
                "grep",
                "Search workspace files for a regex pattern",
                json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string" },
                        "path": { "type": "string" }
                    },
                    "required": ["pattern"]
                }),
            ),
            tool_schema(
                "skill",
                "Load the full body of a named skill (use the skill catalog in the system prompt).",
                json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    },
                    "required": ["name"]
                }),
            ),
            tool_schema(
                "hf.model_card",
                "Fetch a Hugging Face model card (README) by repo id (e.g. org/model). Use before recommending deploy flags or describing a model.",
                json!({
                    "type": "object",
                    "properties": {
                        "model_id": { "type": "string", "description": "Hugging Face repo id, e.g. Qwen/Qwen2.5-Coder-7B-Instruct" }
                    },
                    "required": ["model_id"]
                }),
            ),
            tool_schema(
                "hf.search",
                "Search the Hugging Face model catalogue by free-text query. Biased toward coding / text-generation models by default; pass coding_only=false for the full catalogue (embeddings, vision, TTS, …). Returns top matches with download counts and tags.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Search query, e.g. coding, gguf, 7B instruct" },
                        "coding_only": { "type": "boolean", "description": "Restrict to coding/text-generation models (default true)" }
                    },
                    "required": ["query"]
                }),
            ),
        ];
        let tools: Vec<Value> = tools
            .into_iter()
            .filter(|t| {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .map(|n| self.is_enabled(n))
                    .unwrap_or(true)
            })
            .collect();
        Value::Array(tools)
    }

    pub async fn execute(
        &self,
        call: &ToolCall,
        workspace: &Path,
        approval: ApprovalMode,
        approver: Option<&dyn ToolApprover>,
        skill_body: Option<String>,
    ) -> Result<ToolResult, LocalCodeError> {
        info!(tool = %call.name, "tool execute");
        if !self.is_enabled(&call.name) {
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("Tool '{}' is disabled in LocalCode settings", call.name),
            )
            .with_hint("Enable it in Settings → tools if you want the agent to use it"));
        }
        if let Some(request) = approval_request(call, approval) {
            let approved = match approver {
                Some(a) => a.approve(&request).await,
                None => false,
            };
            if !approved {
                return Err(LocalCodeError::new(
                    ErrorCode::Cancelled,
                    format!("Tool call not approved: {}", call.name),
                )
                .with_cause(format!(
                    "Approval mode '{}' requires confirmation for this call",
                    approval.label()
                ))
                .with_hint("The user declined (or no interactive approval is available)"));
            }
        }
        match call.name.as_str() {
            "read" => self.tool_read(call, workspace),
            "write" => self.tool_write(call, workspace),
            "edit" => self.tool_edit(call, workspace),
            "bash" => self.tool_bash(call, workspace).await,
            "ls" => self.tool_ls(call, workspace),
            "grep" => self.tool_grep(call, workspace),
            "skill" => {
                let name = call
                    .arguments
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                match skill_body {
                    Some(body) => Ok(ToolResult {
                        output: body,
                        risk: ToolRisk::Low,
                    }),
                    None => Err(LocalCodeError::new(
                        ErrorCode::AgentToolFailed,
                        format!("Unknown or disabled skill: {name}"),
                    )
                    .with_hint("Check the skill catalog in the system prompt")),
                }
            }
            other => Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("Unknown tool: {other}"),
            )),
        }
    }

    fn tool_read(&self, call: &ToolCall, workspace: &Path) -> Result<ToolResult, LocalCodeError> {
        let path = arg_path(call, workspace, "path")?;
        let meta = std::fs::metadata(&path).map_err(|e| {
            LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string())
                .with_cause(format!("Cannot stat {}", path.display()))
        })?;
        if !meta.is_file() {
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("Not a file: {}", path.display()),
            ));
        }
        if meta.len() > 8 * 1024 * 1024 {
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!(
                    "File too large to read ({} bytes): {}",
                    meta.len(),
                    path.display()
                ),
            )
            .with_hint("Use start_line/max_lines on a smaller file, or grep for a pattern"));
        }
        let bytes = std::fs::read(&path).map_err(|e| {
            LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string())
                .with_cause(format!("Cannot read {}", path.display()))
        })?;
        if is_binary(&bytes) {
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!(
                    "Binary file ({} bytes) — not shown as text: {}",
                    bytes.len(),
                    path.display()
                ),
            )
            .with_hint("Use bash for binary inspection if needed"));
        }
        let content = String::from_utf8_lossy(&bytes).into_owned();
        let start = call
            .arguments
            .get("start_line")
            .and_then(|v| v.as_u64())
            .map(|n| (n.max(1) - 1) as usize);
        let max = call.arguments.get("max_lines").and_then(|v| v.as_u64());
        let output = match (start, max) {
            (None, None) => content,
            (start, max) => {
                let start = start.unwrap_or(0);
                // Split keeping the line terminators (`split_inclusive` retains
                // the trailing "\n", and any "\r" before it), so a windowed read
                // is a byte-for-byte slice of the file. Rebuilding with
                // `lines().join("\n")` instead would strip "\r" from CRLF files,
                // and a follow-up exact-match `edit` copied from this window
                // would then never match on Windows.
                let all: Vec<&str> = content.split_inclusive('\n').collect();
                let total = all.len();
                if start >= total {
                    return Err(LocalCodeError::new(
                        ErrorCode::AgentToolFailed,
                        format!(
                            "start_line {} is past the end of the file ({total} lines)",
                            start + 1
                        ),
                    )
                    .with_hint("Use a start_line within the file, or read without a window"));
                }
                let take = match max {
                    Some(0) => {
                        return Err(LocalCodeError::new(
                            ErrorCode::AgentToolFailed,
                            "max_lines must be at least 1",
                        ));
                    }
                    Some(n) => n as usize,
                    None => usize::MAX,
                };
                let end = start.saturating_add(take).min(total);
                let mut slice = all[start..end].concat();
                // Internal line endings are preserved (so an `edit` copied from
                // this window matches CRLF files); only a single trailing
                // terminator is trimmed so the window doesn't show a blank last
                // line.
                if slice.ends_with('\n') {
                    slice.pop();
                    if slice.ends_with('\r') {
                        slice.pop();
                    }
                }
                format!("[lines {}-{} of {total}]\n{slice}", start + 1, end)
            }
        };
        Ok(ToolResult {
            output: truncate_output(&output, MAX_READ_CHARS),
            risk: ToolRisk::Low,
        })
    }

    fn tool_write(&self, call: &ToolCall, workspace: &Path) -> Result<ToolResult, LocalCodeError> {
        let path = arg_path(call, workspace, "path")?;
        // `content` is required and must be a string. Defaulting a missing or
        // non-string value to "" would silently truncate an existing file to
        // zero bytes and report success — a dropped `content` field (or one
        // sent as an object/number) must fail like `edit`'s missing args do.
        let content = call
            .arguments
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                LocalCodeError::new(
                    ErrorCode::AgentToolFailed,
                    "write requires content (a string)",
                )
                .with_hint("Pass the full file contents in the `content` field")
            })?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, content)?;
        Ok(ToolResult {
            output: format!("wrote {} bytes to {}", content.len(), path.display()),
            risk: ToolRisk::Medium,
        })
    }

    /// Exact-string replacement in an existing file. Unlike `write`, this can
    /// only ever touch the matched region, so it never drops content the model
    /// hasn't seen (e.g. the tail of a file too large to read in full).
    fn tool_edit(&self, call: &ToolCall, workspace: &Path) -> Result<ToolResult, LocalCodeError> {
        let path = arg_path(call, workspace, "path")?;
        let old = call
            .arguments
            .get("old_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                LocalCodeError::new(ErrorCode::AgentToolFailed, "edit requires old_string")
            })?;
        let new = call
            .arguments
            .get("new_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                LocalCodeError::new(ErrorCode::AgentToolFailed, "edit requires new_string")
            })?;
        let replace_all = call
            .arguments
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if old == new {
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                "edit old_string and new_string are identical — nothing to do",
            ));
        }
        if old.is_empty() {
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                "edit old_string is empty — use write to create a file",
            ));
        }
        let meta = std::fs::metadata(&path).map_err(|e| {
            LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string())
                .with_cause(format!("Cannot stat {}", path.display()))
        })?;
        if !meta.is_file() {
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("Not a file: {}", path.display()),
            ));
        }
        let bytes = std::fs::read(&path).map_err(|e| {
            LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string())
                .with_cause(format!("Cannot read {}", path.display()))
        })?;
        if is_binary(&bytes) {
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("Binary file — refusing to edit as text: {}", path.display()),
            ));
        }
        let content = String::from_utf8_lossy(&bytes).into_owned();
        let count = content.matches(old).count();
        if count == 0 {
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                "edit old_string not found in the file",
            )
            .with_hint("Read the file and copy the exact text, including whitespace"));
        }
        if count > 1 && !replace_all {
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("edit old_string matches {count} places — not unique"),
            )
            .with_hint("Add surrounding lines to make it unique, or pass replace_all=true"));
        }
        let updated = if replace_all {
            content.replace(old, new)
        } else {
            content.replacen(old, new, 1)
        };
        std::fs::write(&path, &updated).map_err(|e| {
            LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string())
                .with_cause(format!("Cannot write {}", path.display()))
        })?;
        let replaced = if replace_all { count } else { 1 };
        Ok(ToolResult {
            output: format!(
                "edited {} ({replaced} replacement{})",
                path.display(),
                if replaced == 1 { "" } else { "s" }
            ),
            risk: ToolRisk::Medium,
        })
    }

    fn tool_ls(&self, call: &ToolCall, workspace: &Path) -> Result<ToolResult, LocalCodeError> {
        // A missing path lists the workspace root, but a *supplied* path that is
        // invalid or escapes the workspace must surface an error rather than
        // silently listing the root (the model would trust a wrong result).
        let path = match call.arguments.get("path").and_then(|v| v.as_str()) {
            Some(p) if !p.is_empty() && p != "." => resolve_path(workspace, p)?,
            _ => workspace.to_path_buf(),
        };
        let mut entries = vec![];
        for e in std::fs::read_dir(&path)
            .map_err(|e| LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string()))?
        {
            let e = e
                .map_err(|e| LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string()))?;
            let meta = e.file_type().ok();
            let kind = if meta.map(|m| m.is_dir()).unwrap_or(false) {
                "dir"
            } else {
                "file"
            };
            entries.push(format!("{kind} {}", e.file_name().to_string_lossy()));
        }
        entries.sort();
        let total = entries.len();
        let truncated = total > MAX_LIST_ENTRIES;
        entries.truncate(MAX_LIST_ENTRIES);
        let mut out = entries.join("\n");
        if truncated {
            out.push_str(&format!(
                "\n…({} more entries truncated; list a subdirectory)",
                total - MAX_LIST_ENTRIES
            ));
        }
        Ok(ToolResult {
            output: out,
            risk: ToolRisk::Low,
        })
    }

    fn tool_grep(&self, call: &ToolCall, workspace: &Path) -> Result<ToolResult, LocalCodeError> {
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
        let re = regex::Regex::new(pattern)
            .map_err(|e| LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string()))?;
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
            let Ok(bytes) = std::fs::read(entry.path()) else {
                continue;
            };
            if is_binary(&bytes) {
                continue;
            }
            let text = String::from_utf8_lossy(&bytes);
            for (i, line) in text.lines().enumerate() {
                if re.is_match(line) {
                    hits.push(format!(
                        "{}:{}:{}",
                        entry.path().display(),
                        i + 1,
                        clip_line(line.trim_end(), MAX_GREP_LINE_CHARS)
                    ));
                    if hits.len() >= 50 {
                        hits.push("…(more matches truncated at 50)".into());
                        break 'outer;
                    }
                }
            }
        }
        Ok(ToolResult {
            // Cap total size too: 50 matched lines from minified/lockfiles can
            // still blow a small local-model context even after per-line clipping.
            output: truncate_output(&hits.join("\n"), MAX_EXEC_CHARS),
            risk: ToolRisk::Low,
        })
    }

    async fn tool_bash(
        &self,
        call: &ToolCall,
        workspace: &Path,
    ) -> Result<ToolResult, LocalCodeError> {
        let command = call
            .arguments
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                LocalCodeError::new(ErrorCode::AgentToolFailed, "command required")
            })?;
        let cwd = match call
            .arguments
            .get("working_directory")
            .and_then(|v| v.as_str())
        {
            Some(p) if !p.is_empty() => resolve_path(workspace, p)?,
            _ => std::fs::canonicalize(workspace).map_err(|e| {
                LocalCodeError::new(ErrorCode::AgentWorkspaceMissing, e.to_string())
            })?,
        };
        if !cwd.is_dir() {
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("working_directory is not a directory: {}", cwd.display()),
            ));
        }
        if self.shell_sandbox {
            sandbox_check_command(command, workspace)?;
        }
        run_shell(command, &cwd, workspace, self.shell_sandbox, self.exec_timeout).await
    }
}

fn is_binary(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    // NUL in the first 8 KiB ⇒ binary. Also reject if a large fraction is non-text.
    let sample = &bytes[..bytes.len().min(8192)];
    if sample.contains(&0) {
        return true;
    }
    let non_text = sample
        .iter()
        .filter(|&&b| b < 0x09 || (b > 0x0d && b < 0x20) || b == 0x7f)
        .count();
    non_text * 100 / sample.len() > 10
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

/// Clip a single line to `max` chars (char-safe), marking the elision.
fn clip_line(line: &str, max: usize) -> String {
    if line.chars().count() <= max {
        return line.to_string();
    }
    let head: String = line.chars().take(max).collect();
    format!("{head}… [line clipped]")
}

/// A short, bounded preview of proposed file content for an approval prompt, so
/// the user sees *what* is being written, not just a byte count.
fn content_preview(s: &str) -> String {
    const MAX_LINES: usize = 20;
    const MAX_CHARS: usize = 800;
    let mut out: String = s.chars().take(MAX_CHARS).collect();
    let char_trunc = s.chars().count() > MAX_CHARS;
    let line_trunc = out.lines().count() > MAX_LINES;
    if line_trunc {
        out = out.lines().take(MAX_LINES).collect::<Vec<_>>().join("\n");
    }
    if char_trunc || line_trunc {
        out.push_str("\n… (preview truncated)");
    }
    out
}

/// Read a child pipe into a shared buffer in chunks. Chunked (rather than
/// `read_to_end`) so that if a leaked background process holds the pipe open,
/// we can still recover whatever was captured before giving up on the drain.
fn spawn_pipe_reader<R>(reader: Option<R>) -> (Arc<Mutex<Vec<u8>>>, tokio::task::JoinHandle<()>)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let sink = buf.clone();
    let handle = tokio::spawn(async move {
        if let Some(mut r) = reader {
            let mut chunk = [0u8; 8192];
            loop {
                match r.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if let Ok(mut b) = sink.lock() {
                            b.extend_from_slice(&chunk[..n]);
                        }
                    }
                }
            }
        }
    });
    (buf, handle)
}

async fn run_shell(
    command: &str,
    cwd: &Path,
    workspace: &Path,
    sandbox: bool,
    exec_timeout: Duration,
) -> Result<ToolResult, LocalCodeError> {
    let mut cmd = if cfg!(windows) {
        // PowerShell handles quoting and pipelines more reliably than cmd /C.
        // Confinement is the cwd + the pre-flight sandbox check; Set-Location
        // just anchors the session at the workspace.
        let ws = powershell_single_quote(&cwd.display().to_string());
        let body = format!("Set-Location -LiteralPath {ws}; {command}");
        let mut c = Command::new("powershell");
        c.args(["-NoProfile", "-NonInteractive", "-Command", &body]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", command]);
        c.env("PWD", cwd);
        c
    };
    cmd.current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if sandbox {
        // Hint for tools that respect it; real confinement is cwd + checks.
        cmd.env("LOCALCODE_WORKSPACE", workspace);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| LocalCodeError::new(ErrorCode::AgentToolFailed, e.to_string()))?;

    let (out_buf, mut out_task) = spawn_pipe_reader(child.stdout.take());
    let (err_buf, mut err_task) = spawn_pipe_reader(child.stderr.take());

    let (status, timed_out) = match tokio::time::timeout(exec_timeout, child.wait()).await {
        Ok(res) => (res.ok(), false),
        Err(_) => {
            let _ = child.kill().await;
            // Reap the killed child so it doesn't linger as a zombie.
            (child.wait().await.ok(), true)
        }
    };
    let status_code = status.and_then(|s| s.code()).unwrap_or(-1);

    // Drain the readers, but cap it: the child may have exited while a leaked
    // grandchild still holds the pipes open (e.g. `foo &`), which would
    // otherwise hang the turn forever. Whatever was captured so far survives.
    let drained = tokio::time::timeout(Duration::from_secs(DRAIN_GRACE_SECS), async {
        let _ = (&mut out_task).await;
        let _ = (&mut err_task).await;
    })
    .await
    .is_ok();
    out_task.abort();
    err_task.abort();

    let stdout = out_buf.lock().map(|b| b.clone()).unwrap_or_default();
    let stderr = err_buf.lock().map(|b| b.clone()).unwrap_or_default();
    let mut text = String::from_utf8_lossy(&stdout).to_string();
    let err = String::from_utf8_lossy(&stderr);
    if !err.is_empty() {
        text.push_str("\nSTDERR:\n");
        text.push_str(&err);
    }
    if timed_out {
        text.push_str(&format!(
            "\n[command timed out after {}s and was killed; output may be partial]",
            exec_timeout.as_secs()
        ));
    } else if !drained {
        text.push_str(
            "\n[a background process kept output open; captured what was available]",
        );
    }
    text.push_str(&format!("\nexit: {status_code}"));
    Ok(ToolResult {
        output: truncate_output(&text, MAX_EXEC_CHARS),
        risk: ToolRisk::High,
    })
}

/// Quote a string for PowerShell single-quoted literals (double embedded `'`).
fn powershell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
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
        .ok_or_else(|| {
            LocalCodeError::new(ErrorCode::AgentToolFailed, format!("{key} required"))
        })?;
    resolve_path(workspace, p)
}

fn escape_err() -> LocalCodeError {
    LocalCodeError::new(ErrorCode::AgentToolFailed, "Path escapes workspace root")
        .with_cause("Safety confinement")
        .with_hint("Use paths under the workspace only")
}

/// Resolve a tool-supplied path and confine it to the workspace.
fn resolve_path(workspace: &Path, p: &str) -> Result<PathBuf, LocalCodeError> {
    let ws = std::fs::canonicalize(workspace)
        .map_err(|e| LocalCodeError::new(ErrorCode::AgentWorkspaceMissing, e.to_string()))?;

    let raw = PathBuf::from(p);
    let base = if raw.is_absolute() {
        raw
    } else {
        ws.join(raw)
    };

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

/// Normalize a path-ish string for substring matching: drop the Windows `\\?\`
/// verbatim prefix `std::fs::canonicalize` adds, unify slashes, and lowercase
/// on Windows (its paths are case-insensitive). Without this the workspace
/// allow-check could never match — `canonicalize` returns `\\?\C:\proj` while
/// the model writes `C:\proj`, so every in-workspace absolute path was blocked.
fn norm_path_str(s: &str) -> String {
    let s = s.strip_prefix(r"\\?\").unwrap_or(s);
    let s = s.replace('\\', "/");
    if cfg!(windows) {
        s.to_lowercase()
    } else {
        s
    }
}

/// Best-effort: reject obvious workspace escapes in sandboxed shell commands.
/// Text-based and deliberately conservative — real confinement is the child's
/// cwd plus the per-path checks; this just stops the model from casually
/// reaching outside via absolute paths, `..`, or `~`.
fn sandbox_check_command(cmd: &str, workspace: &Path) -> Result<(), LocalCodeError> {
    let ws = std::fs::canonicalize(workspace)
        .map_err(|e| LocalCodeError::new(ErrorCode::AgentWorkspaceMissing, e.to_string()))?;
    let ws_s = norm_path_str(&ws.display().to_string());
    let cmd_n = norm_path_str(cmd);
    // A flagged path is allowed only if the (normalized) command still points
    // inside the workspace path string.
    let outside = |flagged: bool| flagged && !cmd_n.contains(&ws_s);

    let block = |what: &str, hint: &'static str| -> Result<(), LocalCodeError> {
        Err(LocalCodeError::new(
            ErrorCode::AgentToolFailed,
            format!("Shell sandbox: {what}"),
        )
        .with_cause("Safety confinement")
        .with_hint(hint))
    };

    let matches = |pattern: &str| {
        regex::Regex::new(pattern)
            .map(|re| re.is_match(cmd))
            .unwrap_or(false)
    };

    // Parent-directory traversal climbs out of the workspace wherever cwd is.
    // (`a..b` git ranges aren't matched — `..` must follow a path separator.)
    if matches(r#"(?:^|[\s"'`=(;|&/\\])\.\.(?:[/\\]|\s|$)"#) {
        return block(
            "parent-directory (`..`) traversal is blocked",
            "Use relative paths under the workspace, or disable agent.shell_sandbox in config",
        );
    }

    // Home-directory references resolve outside the workspace.
    if matches(
        r#"(?i)(?:^|[\s"'`=(;|&])~[/\\]|\$home\b|\$\{home\}|\$env:userprofile|\$env:home|%userprofile%|%homepath%"#,
    ) {
        return block(
            "home-directory references (~ / $HOME / %USERPROFILE%) are blocked",
            "Use paths under the workspace instead",
        );
    }

    // Absolute paths: Unix system roots, any Windows drive, or a UNC share.
    if outside(matches(
        r#"(?xi)
        (?:^|[\s"'`=(;|&])
        (
            /(?:bin|boot|dev|etc|home|lib|mnt|opt|proc|root|run|sbin|sys|tmp|usr|var)(?:/|\s|$)|
            [a-z]:[/\\]|
            \\\\[a-z0-9._-]+\\
        )
        "#,
    )) {
        return block(
            "command references a path outside the workspace",
            "Use relative paths under the workspace, or disable agent.shell_sandbox in config",
        );
    }

    // `cd`/`pushd`/`Set-Location` to an absolute, parent, or UNC location.
    if outside(matches(
        r#"(?i)\b(?:cd|chdir|pushd|set-location|push-location)\s+["'`]?(/|[a-z]:[/\\]|~|\.\.|\\\\)"#,
    )) {
        return block(
            "changing directory outside the workspace is blocked",
            "Use working_directory for a workspace subfolder instead",
        );
    }

    Ok(())
}

/// Best-effort blocklist for obviously destructive shell commands.
fn is_destructive(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();
    let patterns = [
        r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+",
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
        r":\(\)\s*\{.*\};\s*:",
        r"\bchmod\s+-r\b",
        r"\bchown\s+-r\b",
        // Pipe-to-shell installers and remote code execution.
        r"\|\s*(ba)?sh\b",
        r"\|\s*iex\b",
        r"invoke-expression",
        r"invoke-webrequest.*\|\s*",
        r"curl\b.*\|\s*(ba)?sh",
        r"wget\b.*\|\s*(ba)?sh",
        r"irm\b.*\|\s*iex",
        r">\s*/dev/sd",
        r"of=/dev/sd",
        r"\bmkfs\.",
        r"\bdiskpart\b",
        r"\bformat-volume\b",
        r"\btruncate\b.*-s\s*0",
        // `: > file` / `:>file` truncation, anchored so it doesn't fire on an
        // innocent `foo:>bar` substring.
        r"(?:^|[\s;&|(])\s*:\s*>",
    ];
    patterns.iter().any(|p| {
        regex::Regex::new(p)
            .map(|re| re.is_match(&lower))
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn read_write_roundtrip() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        let write = ToolCall {
            id: "1".into(),
            name: "write".into(),
            arguments: json!({"path": "a.txt", "content": "hello world"}),
        };
        reg.execute(&write, dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap();
        let read = ToolCall {
            id: "2".into(),
            name: "read".into(),
            arguments: json!({"path": "a.txt"}),
        };
        let r = reg
            .execute(&read, dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap();
        assert_eq!(r.output, "hello world");
    }

    #[tokio::test]
    async fn read_supports_line_ranges() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        std::fs::write(dir.path().join("n.txt"), "one\ntwo\nthree\nfour").unwrap();
        let call = ToolCall {
            id: "1".into(),
            name: "read".into(),
            arguments: json!({"path": "n.txt", "start_line": 2, "max_lines": 2}),
        };
        let r = reg
            .execute(&call, dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap();
        assert_eq!(r.output, "[lines 2-3 of 4]\ntwo\nthree");
    }

    #[tokio::test]
    async fn write_without_content_is_rejected_and_leaves_file_intact() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        std::fs::write(dir.path().join("keep.txt"), "important").unwrap();
        // A dropped `content` field must fail, not truncate the file to 0 bytes.
        let call = ToolCall {
            id: "1".into(),
            name: "write".into(),
            arguments: json!({"path": "keep.txt"}),
        };
        let err = reg
            .execute(&call, dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap_err();
        assert!(err.message.contains("content"), "{}", err.message);
        assert_eq!(std::fs::read_to_string(dir.path().join("keep.txt")).unwrap(), "important");
        // A non-string content is likewise rejected.
        let call = ToolCall {
            id: "2".into(),
            name: "write".into(),
            arguments: json!({"path": "keep.txt", "content": 42}),
        };
        assert!(reg
            .execute(&call, dir.path(), ApprovalMode::Auto, None, None)
            .await
            .is_err());
        assert_eq!(std::fs::read_to_string(dir.path().join("keep.txt")).unwrap(), "important");
    }

    #[tokio::test]
    async fn windowed_read_preserves_crlf_for_exact_edits() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        std::fs::write(dir.path().join("crlf.txt"), "one\r\ntwo\r\nthree\r\nfour").unwrap();
        let read = ToolCall {
            id: "1".into(),
            name: "read".into(),
            arguments: json!({"path": "crlf.txt", "start_line": 2, "max_lines": 2}),
        };
        let r = reg
            .execute(&read, dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap();
        // Internal CRLF is preserved (not normalized to LF).
        assert_eq!(r.output, "[lines 2-3 of 4]\ntwo\r\nthree");
        // The exact two-line window then matches for an edit on the CRLF file.
        let edit = ToolCall {
            id: "2".into(),
            name: "edit".into(),
            arguments: json!({"path": "crlf.txt", "old_string": "two\r\nthree", "new_string": "2\r\n3"}),
        };
        reg.execute(&edit, dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("crlf.txt")).unwrap(),
            "one\r\n2\r\n3\r\nfour"
        );
    }

    #[tokio::test]
    async fn windowed_read_rejects_out_of_range() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        std::fs::write(dir.path().join("n.txt"), "a\nb\nc\nd").unwrap();
        let past = ToolCall {
            id: "1".into(),
            name: "read".into(),
            arguments: json!({"path": "n.txt", "start_line": 500}),
        };
        let err = reg
            .execute(&past, dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap_err();
        assert!(err.message.contains("past the end"), "{}", err.message);
        let zero = ToolCall {
            id: "2".into(),
            name: "read".into(),
            arguments: json!({"path": "n.txt", "start_line": 1, "max_lines": 0}),
        };
        assert!(reg
            .execute(&zero, dir.path(), ApprovalMode::Auto, None, None)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn read_rejects_binary() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        std::fs::write(dir.path().join("b.bin"), [0u8, 1, 2, 3, 0, 5]).unwrap();
        let call = ToolCall {
            id: "1".into(),
            name: "read".into(),
            arguments: json!({"path": "b.bin"}),
        };
        let err = reg
            .execute(&call, dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap_err();
        assert!(err.message.to_lowercase().contains("binary"), "{}", err.message);
    }

    #[tokio::test]
    async fn ls_caps_entries() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        for i in 0..210 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), "x").unwrap();
        }
        let call = ToolCall {
            id: "1".into(),
            name: "ls".into(),
            arguments: json!({"path": "."}),
        };
        let r = reg
            .execute(&call, dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap();
        assert!(r.output.contains("truncated"), "{}", r.output);
        assert!(r.output.lines().count() <= MAX_LIST_ENTRIES + 2);
    }

    #[test]
    fn sibling_dir_is_rejected() {
        let dir = tempdir().unwrap();
        let ws = dir.path().join("ws");
        let evil = dir.path().join("ws-evil");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&evil).unwrap();
        std::fs::write(evil.join("x.txt"), "secret").unwrap();

        let attack = evil.join("x.txt").display().to_string();
        assert!(resolve_path(&ws, &attack).is_err());
        assert!(resolve_path(&ws, "../ws-evil/x.txt").is_err());

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
            "curl http://x | bash",
            "wget http://x | sh",
            "irm http://x | iex",
            "echo hi | iex",
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
            name: "bash".into(),
            arguments: json!({ "command": "echo hello" }),
        };
        let r = reg
            .execute(&call, dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap();
        assert!(r.output.contains("hello"));
        assert!(r.output.contains("exit: 0"));
    }

    #[tokio::test]
    async fn destructive_without_approver_is_refused() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        let call = ToolCall {
            id: "1".into(),
            name: "bash".into(),
            arguments: json!({ "command": "rm -rf ." }),
        };
        let err = reg
            .execute(&call, dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::Cancelled);
    }

    #[tokio::test]
    async fn sandbox_blocks_cd_outside() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        let call = ToolCall {
            id: "1".into(),
            name: "bash".into(),
            arguments: json!({ "command": "cd /tmp && ls" }),
        };
        let err = reg
            .execute(&call, dir.path(), ApprovalMode::AlwaysApprove, None, None)
            .await
            .unwrap_err();
        assert!(
            err.message.to_lowercase().contains("sandbox")
                || err.message.to_lowercase().contains("workspace"),
            "{}",
            err.message
        );
    }

    #[test]
    fn approval_matrix() {
        let call = |name: &str, args: Value| ToolCall {
            id: "t".into(),
            name: name.into(),
            arguments: args,
        };
        let read = call("read", json!({"path": "a.txt"}));
        let search = call("grep", json!({"pattern": "x"}));
        let write = call("write", json!({"path": "a.txt", "content": "hi"}));
        let sh = call("bash", json!({"command": "cargo test"}));
        let sh_destr = call("bash", json!({"command": "rm -rf target"}));

        use ApprovalMode::*;
        for c in [&read, &search, &write, &sh, &sh_destr] {
            assert!(approval_request(c, AlwaysApprove).is_none());
        }
        for c in [&read, &search, &write, &sh] {
            assert!(approval_request(c, Auto).is_none());
        }
        let req = approval_request(&sh_destr, Auto).expect("destructive asks");
        assert!(req.contains("rm -rf target") && req.contains("destructive"), "{req}");
        for c in [&read, &search] {
            assert!(approval_request(c, ApproveEdits).is_none());
        }
        for c in [&write, &sh, &sh_destr] {
            assert!(approval_request(c, ApproveEdits).is_some());
        }
        for c in [&read, &search, &write, &sh, &sh_destr] {
            assert!(approval_request(c, AskPermission).is_some());
        }
    }

    #[tokio::test]
    async fn approve_edits_gates_writes() {
        struct Yes;
        #[async_trait]
        impl ToolApprover for Yes {
            async fn approve(&self, _d: &str) -> bool {
                true
            }
        }
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        let write = ToolCall {
            id: "1".into(),
            name: "write".into(),
            arguments: json!({"path": "a.txt", "content": "hi"}),
        };
        let err = reg
            .execute(&write, dir.path(), ApprovalMode::ApproveEdits, None, None)
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::Cancelled);
        assert!(!dir.path().join("a.txt").exists());

        reg.execute(
            &write,
            dir.path(),
            ApprovalMode::ApproveEdits,
            Some(&Yes),
            None,
        )
        .await
        .unwrap();
        assert!(dir.path().join("a.txt").exists());

        let read = ToolCall {
            id: "2".into(),
            name: "read".into(),
            arguments: json!({"path": "a.txt"}),
        };
        reg.execute(&read, dir.path(), ApprovalMode::ApproveEdits, None, None)
            .await
            .unwrap();
    }

    fn edit_call(path: &str, old: &str, new: &str, all: bool) -> ToolCall {
        ToolCall {
            id: "e".into(),
            name: "edit".into(),
            arguments: json!({
                "path": path, "old_string": old, "new_string": new, "replace_all": all
            }),
        }
    }

    #[tokio::test]
    async fn edit_replaces_unique_string() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        std::fs::write(dir.path().join("f.txt"), "alpha beta gamma").unwrap();
        reg.execute(
            &edit_call("f.txt", "beta", "DELTA", false),
            dir.path(),
            ApprovalMode::Auto,
            None,
            None,
        )
        .await
        .unwrap();
        let got = std::fs::read_to_string(dir.path().join("f.txt")).unwrap();
        assert_eq!(got, "alpha DELTA gamma");
    }

    #[tokio::test]
    async fn edit_rejects_ambiguous_and_missing() {
        let dir = tempdir().unwrap();
        let reg = ToolRegistry::default_tools();
        std::fs::write(dir.path().join("f.txt"), "x x x").unwrap();
        // Not unique without replace_all.
        let err = reg
            .execute(&edit_call("f.txt", "x", "y", false), dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap_err();
        assert!(err.message.contains("not unique"), "{}", err.message);
        // replace_all touches every occurrence and leaves the rest intact.
        reg.execute(&edit_call("f.txt", "x", "y", true), dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(dir.path().join("f.txt")).unwrap(), "y y y");
        // Missing text errors rather than silently doing nothing.
        let err = reg
            .execute(&edit_call("f.txt", "zzz", "q", false), dir.path(), ApprovalMode::Auto, None, None)
            .await
            .unwrap_err();
        assert!(err.message.contains("not found"), "{}", err.message);
    }

    #[test]
    fn edit_is_gated_like_write() {
        let e = edit_call("a.txt", "a", "b", false);
        // Auto: only destructive bash gates, edits run freely.
        assert!(approval_request(&e, ApprovalMode::Auto).is_none());
        // ApproveEdits + AskPermission both gate an edit.
        assert!(approval_request(&e, ApprovalMode::ApproveEdits).is_some());
        assert!(approval_request(&e, ApprovalMode::AskPermission).is_some());
        assert!(approval_request(&e, ApprovalMode::AlwaysApprove).is_none());
    }

    #[test]
    fn sandbox_allows_absolute_path_inside_workspace() {
        // The bug: canonicalize() returns a `\\?\` verbatim path on Windows, so
        // the allow-check never matched and in-workspace absolute paths were
        // wrongly blocked. An absolute path under the workspace must pass.
        let dir = tempdir().unwrap();
        let ws = dir.path().join("proj");
        std::fs::create_dir_all(&ws).unwrap();
        let inside = ws.join("src").display().to_string();
        let cmd = format!("cat {inside}/main.rs");
        assert!(
            sandbox_check_command(&cmd, &ws).is_ok(),
            "in-workspace absolute path should be allowed: {cmd}"
        );
    }

    #[test]
    fn sandbox_blocks_traversal_home_and_outside() {
        let dir = tempdir().unwrap();
        let ws = dir.path().join("proj");
        std::fs::create_dir_all(&ws).unwrap();
        assert!(sandbox_check_command("cat ../../etc/passwd", &ws).is_err());
        assert!(sandbox_check_command("cat ~/secrets", &ws).is_err());
        assert!(sandbox_check_command("type $env:USERPROFILE\\x", &ws).is_err());
        // A git revision range (`a..b`) is not path traversal.
        assert!(sandbox_check_command("git log main..HEAD", &ws).is_ok());
        // A plain relative command is fine.
        assert!(sandbox_check_command("cargo build", &ws).is_ok());
    }

    #[tokio::test]
    async fn ls_rejects_escaping_path() {
        let dir = tempdir().unwrap();
        let ws = dir.path().join("proj");
        std::fs::create_dir_all(&ws).unwrap();
        let reg = ToolRegistry::default_tools();
        let call = ToolCall {
            id: "1".into(),
            name: "ls".into(),
            arguments: json!({"path": "../"}),
        };
        // Previously this silently listed the workspace root; now it errors.
        assert!(reg
            .execute(&call, &ws, ApprovalMode::Auto, None, None)
            .await
            .is_err());
    }
}
