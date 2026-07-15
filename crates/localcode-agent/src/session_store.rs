//! Append-only JSONL session store with an in-file branching tree.
//!
//! One file per session, grouped per workspace under the sessions root. The
//! first line is a [`SessionHeader`]; every later line is a [`SessionEntry`]
//! whose `id`/`parent_id` form a tree, pi-style: rewinding just points the
//! active leaf at an older entry and later appends branch from there — the
//! abandoned suffix stays in the file, recoverable, without a new file.
//!
//! The store never rewrites lines. Repair happens on load: unreadable or
//! orphaned lines are skipped with warnings, and histories cut short by an
//! aborted turn are truncated to the last protocol-valid message.

use crate::{sanitize_history, AgentSession, ChatMessage, SessionCompaction};
use chrono::{DateTime, Local, Utc};
use localcode_core::config::AgentConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::paths::AppPaths;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use tracing::warn;
use uuid::Uuid;

const FORMAT_VERSION: u32 = 1;

/// First line of every session file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHeader {
    #[serde(rename = "type")]
    kind: String, // always "session"
    pub version: u32,
    pub id: String,
    pub cwd: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
}

/// One line of the session tree. Serialized compact (one JSON object per
/// line); the `type` tag is always the first field, which the cheap
/// [`list_sessions`] scan relies on for files this module wrote.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEntry {
    Message(MessageEntry),
    Title(TitleEntry),
    Compaction(CompactionEntry),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEntry {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub ts: DateTime<Utc>,
    pub message: ChatMessage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TitleEntry {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub ts: DateTime<Utc>,
    pub title: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionEntry {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub ts: DateTime<Utc>,
    pub summary: String,
    /// Entry id of the first message kept verbatim after the compaction.
    pub first_kept_id: String,
    pub chars_before: usize,
}

impl SessionEntry {
    fn id(&self) -> &str {
        match self {
            Self::Message(e) => &e.id,
            Self::Title(e) => &e.id,
            Self::Compaction(e) => &e.id,
        }
    }

    fn parent_id(&self) -> Option<&str> {
        match self {
            Self::Message(e) => e.parent_id.as_deref(),
            Self::Title(e) => e.parent_id.as_deref(),
            Self::Compaction(e) => e.parent_id.as_deref(),
        }
    }
}

/// A session loaded from disk, ready to continue.
pub struct LoadedSession {
    pub store: SessionStore,
    pub session: AgentSession,
    /// Human-readable notes about anything repaired or skipped during load.
    pub warnings: Vec<String>,
}

/// Listing info for the resume picker — cheap to compute, newest first.
#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub path: PathBuf,
    pub id: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: SystemTime,
    /// Message lines in the file, across all branches.
    pub message_count: usize,
}

/// Where session files live: the config override, else `<data>/sessions`.
pub fn sessions_root(cfg: &AgentConfig, paths: &AppPaths) -> PathBuf {
    cfg.sessions_dir
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| paths.sessions_dir.clone())
}

pub struct SessionStore {
    path: PathBuf,
    header: SessionHeader,
    entries: Vec<SessionEntry>,
    by_id: HashMap<String, usize>,
    /// Active leaf entry id; `None` means the header (empty session).
    leaf: Option<String>,
    /// Entry ids of the active path's messages, 1:1 with the persisted
    /// prefix of `AgentSession::messages`.
    message_ids: Vec<String>,
    /// Fingerprints matching `message_ids`, so [`SessionStore::sync`] can
    /// detect when the caller rewrote history (sanitize, rewind) and branch
    /// instead of mis-appending.
    message_fps: Vec<u64>,
    last_title: String,
    /// Message count when the title was last persisted (0 = header). A title
    /// written past a truncation point sits on an abandoned branch and must
    /// be re-anchored.
    last_title_at: usize,
    /// `first_kept_index` of the compaction marker already persisted for the
    /// current path; `None` forces the next sync to (re-)write the marker.
    synced_compaction: Option<usize>,
    /// Message count when the marker was last persisted, like `last_title_at`.
    synced_compaction_at: usize,
}

impl SessionStore {
    /// Start a new session file (header only). Call [`SessionStore::sync`]
    /// afterwards to persist any messages the session already holds.
    pub fn create(root: &Path, session: &AgentSession) -> Result<Self, LocalCodeError> {
        let dir = root.join(escape_workspace_dir(&session.workspace_root));
        fs::create_dir_all(&dir).map_err(|e| save_err(e, &dir))?;
        let short: String = session.id.chars().take(8).collect();
        let name = format!("{}_{}.jsonl", Local::now().format("%Y%m%d-%H%M%S"), short);
        let path = dir.join(name);
        let header = SessionHeader {
            kind: "session".into(),
            version: FORMAT_VERSION,
            id: session.id.clone(),
            cwd: session.workspace_root.display().to_string(),
            title: session.title.clone(),
            created_at: Utc::now(),
        };
        let line = to_line(&header)?;
        fs::write(&path, line).map_err(|e| save_err(e, &path))?;
        Ok(Self {
            path,
            last_title: header.title.clone(),
            last_title_at: 0,
            header,
            entries: Vec::new(),
            by_id: HashMap::new(),
            leaf: None,
            message_ids: Vec::new(),
            message_fps: Vec::new(),
            synced_compaction: None,
            synced_compaction_at: 0,
        })
    }

    /// Append everything that changed in `session` since the last sync: a
    /// branch point if history was rewritten, new messages, a moved
    /// compaction marker, and a title change. Returns the number of entries
    /// appended. Never rewrites existing lines.
    pub fn sync(&mut self, session: &AgentSession) -> Result<usize, LocalCodeError> {
        // Detect history rewrites (sanitize, rewind): keep the longest common
        // prefix and branch after it. The abandoned entries stay in the file.
        let fps: Vec<u64> = session.messages.iter().map(fingerprint).collect();
        let common = self
            .message_fps
            .iter()
            .zip(&fps)
            .take_while(|(a, b)| a == b)
            .count();
        if common < self.message_ids.len() {
            self.message_ids.truncate(common);
            self.message_fps.truncate(common);
            self.leaf = self.message_ids.last().cloned();
            // Title/compaction entries written past the branch point sit on
            // the abandoned branch; force those to re-anchor on this one.
            if self.last_title_at > common {
                self.last_title = String::new();
            }
            if self.synced_compaction_at > common {
                self.synced_compaction = None;
            }
        }

        let mut appended: Vec<SessionEntry> = Vec::new();
        for (m, fp) in session.messages[common..].iter().zip(&fps[common..]) {
            let entry = MessageEntry {
                id: Uuid::new_v4().to_string(),
                parent_id: self.leaf.clone(),
                ts: Utc::now(),
                message: m.clone(),
            };
            self.leaf = Some(entry.id.clone());
            self.message_ids.push(entry.id.clone());
            self.message_fps.push(*fp);
            appended.push(SessionEntry::Message(entry));
        }

        if let Some(c) = &session.compaction {
            if self.synced_compaction != Some(c.first_kept_index)
                && c.first_kept_index < self.message_ids.len()
            {
                let entry = CompactionEntry {
                    id: Uuid::new_v4().to_string(),
                    parent_id: self.leaf.clone(),
                    ts: Utc::now(),
                    summary: c.summary.clone(),
                    first_kept_id: self.message_ids[c.first_kept_index].clone(),
                    chars_before: c.chars_before,
                };
                self.leaf = Some(entry.id.clone());
                self.synced_compaction = Some(c.first_kept_index);
                self.synced_compaction_at = self.message_ids.len();
                appended.push(SessionEntry::Compaction(entry));
            }
        }

        if session.title != self.last_title {
            let entry = TitleEntry {
                id: Uuid::new_v4().to_string(),
                parent_id: self.leaf.clone(),
                ts: Utc::now(),
                title: session.title.clone(),
            };
            self.leaf = Some(entry.id.clone());
            self.last_title = session.title.clone();
            self.last_title_at = self.message_ids.len();
            appended.push(SessionEntry::Title(entry));
        }

        if appended.is_empty() {
            return Ok(0);
        }

        let mut lines = String::new();
        for entry in &appended {
            lines.push_str(&to_line(entry)?);
        }
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&self.path)
            .map_err(|e| save_err(e, &self.path))?;
        file.write_all(lines.as_bytes())
            .map_err(|e| save_err(e, &self.path))?;
        file.flush().map_err(|e| save_err(e, &self.path))?;

        let n = appended.len();
        for entry in appended {
            self.by_id.insert(entry.id().to_string(), self.entries.len());
            self.entries.push(entry);
        }
        Ok(n)
    }

    /// Load a session file, repairing what it can. The active path is the
    /// chain from the last good entry back to the root.
    pub fn load(path: &Path) -> Result<LoadedSession, LocalCodeError> {
        let content = fs::read_to_string(path).map_err(|e| {
            LocalCodeError::new(
                ErrorCode::AgentSessionLoadFailed,
                format!("Could not read session file: {e}"),
            )
            .with_cause(path.display().to_string())
        })?;
        let mut lines = content.lines();
        let header: SessionHeader = lines
            .next()
            .filter(|l| !l.trim().is_empty())
            .and_then(|l| serde_json::from_str(l).ok())
            .ok_or_else(|| {
                LocalCodeError::new(
                    ErrorCode::AgentSessionLoadFailed,
                    "Session file has no valid header line",
                )
                .with_cause(path.display().to_string())
            })?;
        if header.kind != "session" {
            return Err(LocalCodeError::new(
                ErrorCode::AgentSessionLoadFailed,
                "Not a LocalCode session file",
            )
            .with_cause(path.display().to_string()));
        }
        if header.version > FORMAT_VERSION {
            return Err(LocalCodeError::new(
                ErrorCode::AgentSessionLoadFailed,
                format!(
                    "Session file version {} is newer than this build supports ({FORMAT_VERSION})",
                    header.version
                ),
            )
            .with_hint("Update LocalCode (/update) to open this session"));
        }

        let mut warnings: Vec<String> = Vec::new();
        let mut entries: Vec<SessionEntry> = Vec::new();
        let mut by_id: HashMap<String, usize> = HashMap::new();
        for (n, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let entry: SessionEntry = match serde_json::from_str(line) {
                Ok(e) => e,
                Err(e) => {
                    warn!(line = n + 2, error = %e, "skipping unreadable session entry");
                    warnings.push(format!("skipped an unreadable entry (line {})", n + 2));
                    continue;
                }
            };
            if let Some(pid) = entry.parent_id() {
                if !by_id.contains_key(pid) {
                    warnings.push(format!("skipped an orphaned entry (line {})", n + 2));
                    continue;
                }
            }
            by_id.insert(entry.id().to_string(), entries.len());
            entries.push(entry);
        }

        // Active path: last good entry back to the root. Parents always exist
        // (orphans were skipped) and precede their children (fresh uuids), so
        // this terminates.
        let mut leaf = entries.last().map(|e| e.id().to_string());
        let mut chain: Vec<usize> = Vec::new();
        let mut cur = leaf.clone();
        while let Some(id) = cur {
            let Some(&idx) = by_id.get(&id) else { break };
            chain.push(idx);
            cur = entries[idx].parent_id().map(str::to_string);
        }
        chain.reverse();

        let mut messages: Vec<ChatMessage> = Vec::new();
        let mut message_ids: Vec<String> = Vec::new();
        let mut title = header.title.clone();
        let mut title_at = 0usize;
        let mut raw_compaction: Option<&CompactionEntry> = None;
        let mut compaction_at = 0usize;
        for &idx in &chain {
            match &entries[idx] {
                SessionEntry::Message(m) => {
                    messages.push(m.message.clone());
                    message_ids.push(m.id.clone());
                }
                SessionEntry::Title(t) => {
                    title = t.title.clone();
                    title_at = messages.len();
                }
                SessionEntry::Compaction(c) => {
                    raw_compaction = Some(c);
                    compaction_at = messages.len();
                }
            }
        }
        let mut compaction = raw_compaction.and_then(|c| {
            match message_ids.iter().position(|id| *id == c.first_kept_id) {
                Some(i) => Some(SessionCompaction {
                    first_kept_index: i,
                    summary: c.summary.clone(),
                    chars_before: c.chars_before,
                }),
                None => {
                    warnings.push("dropped a compaction marker for a missing message".into());
                    None
                }
            }
        });

        let mut last_title = title.clone();
        let mut synced_compaction = compaction.as_ref().map(|c| c.first_kept_index);
        let dropped = sanitize_history(&mut messages);
        if dropped > 0 {
            message_ids.truncate(messages.len());
            leaf = message_ids.last().cloned();
            warnings.push(format!(
                "dropped {dropped} trailing message(s) left by an interrupted turn"
            ));
            // Title/marker entries past the cut sit on the abandoned suffix;
            // re-anchor those on the repaired branch at the next sync.
            if title_at > messages.len() {
                last_title = String::new();
            }
            if compaction_at > messages.len() {
                synced_compaction = None;
            }
            if compaction
                .as_ref()
                .is_some_and(|c| c.first_kept_index >= messages.len())
            {
                compaction = None;
                synced_compaction = None;
            }
        }
        let message_fps = messages.iter().map(fingerprint).collect();
        let messages_len = messages.len();

        let session = AgentSession {
            id: header.id.clone(),
            title,
            messages,
            workspace_root: PathBuf::from(&header.cwd),
            subagents_enabled: false,
            runtime_id: None,
            compaction,
        };
        let store = Self {
            path: path.to_path_buf(),
            header,
            entries,
            by_id,
            leaf,
            message_ids,
            message_fps,
            last_title,
            last_title_at: title_at.min(messages_len),
            synced_compaction,
            synced_compaction_at: compaction_at.min(messages_len),
        };
        Ok(LoadedSession {
            store,
            session,
            warnings,
        })
    }

    pub fn session_id(&self) -> &str {
        &self.header.id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Sessions for `workspace` under `root`, newest first by file mtime. Reads
/// at most the 50 newest files (title and count need a scan).
pub fn list_sessions(root: &Path, workspace: &Path) -> Vec<SessionMeta> {
    let dir = root.join(escape_workspace_dir(workspace));
    let Ok(read) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut files: Vec<(PathBuf, SystemTime)> = read
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "jsonl"))
        .filter_map(|e| {
            let mtime = e.metadata().ok()?.modified().ok()?;
            Some((e.path(), mtime))
        })
        .collect();
    files.sort_by_key(|(_, mtime)| std::cmp::Reverse(*mtime));
    files.truncate(50);
    files
        .into_iter()
        .filter_map(|(path, mtime)| scan_meta(&path, mtime))
        .collect()
}

/// Cheap single-pass scan for the listing: header + last title + message
/// count. Relies on the `type` tag being the first field, which holds for
/// every line this module writes.
fn scan_meta(path: &Path, updated_at: SystemTime) -> Option<SessionMeta> {
    let content = fs::read_to_string(path).ok()?;
    let mut lines = content.lines();
    let header: SessionHeader = serde_json::from_str(lines.next()?).ok()?;
    let mut title = header.title;
    let mut message_count = 0usize;
    for line in lines {
        if line.starts_with("{\"type\":\"message\"") {
            message_count += 1;
        } else if line.starts_with("{\"type\":\"title\"") {
            if let Ok(t) = serde_json::from_str::<TitleEntry>(line) {
                title = t.title;
            }
        }
    }
    Some(SessionMeta {
        path: path.to_path_buf(),
        id: header.id,
        title,
        created_at: header.created_at,
        updated_at,
        message_count,
    })
}

fn to_line<T: Serialize>(value: &T) -> Result<String, LocalCodeError> {
    let mut line = serde_json::to_string(value).map_err(|e| {
        LocalCodeError::new(
            ErrorCode::AgentSessionSaveFailed,
            format!("Could not serialize session entry: {e}"),
        )
    })?;
    line.push('\n');
    Ok(line)
}

fn save_err(e: std::io::Error, path: &Path) -> LocalCodeError {
    LocalCodeError::new(
        ErrorCode::AgentSessionSaveFailed,
        format!("Could not write session file: {e}"),
    )
    .with_cause(path.display().to_string())
}

/// Directory name for a workspace: a readable prefix plus a hash of the
/// normalized full path. Identity lives in the hash — the prefix is only for
/// humans browsing the sessions dir.
fn escape_workspace_dir(workspace: &Path) -> String {
    let canon = workspace
        .canonicalize()
        .unwrap_or_else(|_| workspace.to_path_buf());
    let mut s = canon.to_string_lossy().into_owned();
    // Windows canonicalize yields a \\?\ verbatim path; drop it for readability.
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        s = rest.to_string();
    }
    if cfg!(windows) {
        s = s.to_lowercase();
    }
    let hash = fnv1a(s.as_bytes());
    let pretty: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .take(80)
        .collect();
    format!("{pretty}-{hash:016x}")
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Position-stable fingerprint of a message, for divergence detection in
/// [`SessionStore::sync`]. A collision only means a rewrite goes unnoticed at
/// that position — never data loss in the file.
fn fingerprint(m: &ChatMessage) -> u64 {
    let mut buf = String::with_capacity(m.content.len() + 32);
    buf.push_str(&m.role);
    buf.push('\u{1}');
    buf.push_str(&m.content);
    buf.push('\u{1}');
    if let Some(id) = &m.tool_call_id {
        buf.push_str(id);
    }
    buf.push('\u{1}');
    if let Some(name) = &m.name {
        buf.push_str(name);
    }
    buf.push('\u{1}');
    if let Some(tc) = &m.tool_calls {
        buf.push_str(&tc.to_string());
    }
    fnv1a(buf.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn user(s: &str) -> ChatMessage {
        ChatMessage::user(s)
    }
    fn asst(s: &str) -> ChatMessage {
        ChatMessage::assistant(s, None)
    }
    fn asst_calls(ids: &[&str]) -> ChatMessage {
        let calls: Vec<_> = ids
            .iter()
            .map(|id| json!({"id": id, "type": "function", "function": {"name": "fs.read", "arguments": "{}"}}))
            .collect();
        ChatMessage::assistant("", Some(json!(calls)))
    }
    fn tool(s: &str, id: &str) -> ChatMessage {
        ChatMessage::tool(s, id.into(), "fs.read".into())
    }

    fn new_session(dir: &Path) -> AgentSession {
        AgentSession::new(dir.to_path_buf(), false)
    }

    #[test]
    fn round_trip_two_turns() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("sessions");
        let ws = dir.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();

        let mut session = new_session(&ws);
        session.messages.push(user("first question"));
        session.messages.push(asst("first answer"));
        let mut store = SessionStore::create(&root, &session).unwrap();
        assert_eq!(store.sync(&session).unwrap(), 2);

        session.messages.push(user("second question"));
        session.messages.push(asst("second answer"));
        session.title = "first question".into();
        assert_eq!(store.sync(&session).unwrap(), 3); // 2 messages + title
        assert_eq!(store.sync(&session).unwrap(), 0); // idempotent

        let loaded = SessionStore::load(store.path()).unwrap();
        assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);
        assert_eq!(loaded.session.id, session.id);
        assert_eq!(loaded.session.title, "first question");
        assert_eq!(loaded.session.messages.len(), 4);
        assert_eq!(loaded.session.messages[3].content, "second answer");
        assert_eq!(loaded.session.workspace_root, ws);
    }

    #[test]
    fn title_entry_written_once_per_change() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("sessions");
        let mut session = new_session(dir.path());
        session.messages.push(user("hi"));
        let mut store = SessionStore::create(&root, &session).unwrap();
        store.sync(&session).unwrap();

        session.title = "hi".into();
        assert_eq!(store.sync(&session).unwrap(), 1);
        assert_eq!(store.sync(&session).unwrap(), 0);
        let text = std::fs::read_to_string(store.path()).unwrap();
        assert_eq!(text.matches("{\"type\":\"title\"").count(), 1);
    }

    #[test]
    fn corrupted_tail_is_skipped() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("sessions");
        let mut session = new_session(dir.path());
        session.messages.push(user("hello"));
        session.messages.push(asst("world"));
        let mut store = SessionStore::create(&root, &session).unwrap();
        store.sync(&session).unwrap();

        let mut text = std::fs::read_to_string(store.path()).unwrap();
        text.push_str("{ this is not json\n");
        std::fs::write(store.path(), text).unwrap();

        let loaded = SessionStore::load(store.path()).unwrap();
        assert_eq!(loaded.session.messages.len(), 2);
        assert!(loaded.warnings.iter().any(|w| w.contains("unreadable")));
    }

    #[test]
    fn orphaned_parent_is_skipped() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("sessions");
        let mut session = new_session(dir.path());
        session.messages.push(user("hello"));
        let mut store = SessionStore::create(&root, &session).unwrap();
        store.sync(&session).unwrap();

        let orphan = SessionEntry::Message(MessageEntry {
            id: "orphan".into(),
            parent_id: Some("no-such-entry".into()),
            ts: Utc::now(),
            message: user("ghost"),
        });
        let mut text = std::fs::read_to_string(store.path()).unwrap();
        text.push_str(&to_line(&orphan).unwrap());
        std::fs::write(store.path(), text).unwrap();

        let loaded = SessionStore::load(store.path()).unwrap();
        assert_eq!(loaded.session.messages.len(), 1);
        assert!(loaded.warnings.iter().any(|w| w.contains("orphaned")));
    }

    #[test]
    fn dangling_tool_calls_repaired_on_load() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("sessions");

        // No replies at all.
        let mut session = new_session(dir.path());
        session.messages.push(user("do a thing"));
        session.messages.push(asst_calls(&["call_1"]));
        let mut store = SessionStore::create(&root, &session).unwrap();
        store.sync(&session).unwrap();
        let loaded = SessionStore::load(store.path()).unwrap();
        assert_eq!(loaded.session.messages.len(), 1);
        assert!(loaded.warnings.iter().any(|w| w.contains("interrupted")));

        // Partial replies: two calls, one reply.
        let mut session = new_session(dir.path());
        session.messages.push(user("do two things"));
        session.messages.push(asst_calls(&["call_1", "call_2"]));
        session.messages.push(tool("ok", "call_1"));
        let mut store = SessionStore::create(&root, &session).unwrap();
        store.sync(&session).unwrap();
        let loaded = SessionStore::load(store.path()).unwrap();
        assert_eq!(loaded.session.messages.len(), 1);

        // A loaded-and-repaired session keeps working: the next sync branches.
        let mut session = loaded.session;
        let mut store = loaded.store;
        session.messages.push(user("try again"));
        session.messages.push(asst("done"));
        assert_eq!(store.sync(&session).unwrap(), 2);
        let reloaded = SessionStore::load(store.path()).unwrap();
        assert_eq!(reloaded.session.messages.len(), 3);
        assert_eq!(reloaded.session.messages[2].content, "done");
    }

    #[test]
    fn sync_truncation_creates_branch() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("sessions");
        let mut session = new_session(dir.path());
        for i in 0..3 {
            session.messages.push(user(&format!("q{i}")));
            session.messages.push(asst(&format!("a{i}")));
        }
        let mut store = SessionStore::create(&root, &session).unwrap();
        assert_eq!(store.sync(&session).unwrap(), 6);

        // Rewind to after turn 2, then take a different path.
        session.messages.truncate(4);
        session.messages.push(user("q2-variant"));
        session.messages.push(asst("a2-variant"));
        assert_eq!(store.sync(&session).unwrap(), 2);

        let text = std::fs::read_to_string(store.path()).unwrap();
        assert_eq!(text.matches("{\"type\":\"message\"").count(), 8);

        let loaded = SessionStore::load(store.path()).unwrap();
        assert_eq!(loaded.session.messages.len(), 6);
        assert_eq!(loaded.session.messages[4].content, "q2-variant");
        assert_eq!(loaded.session.messages[5].content, "a2-variant");
    }

    #[test]
    fn workspace_escape_is_windows_safe_and_stable() {
        let dir = tempdir().unwrap();
        let a = escape_workspace_dir(dir.path());
        let b = escape_workspace_dir(dir.path());
        assert_eq!(a, b);
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')));

        let other = dir.path().join("elsewhere");
        assert_ne!(escape_workspace_dir(&other), a);
    }

    #[test]
    fn list_sessions_newest_first_with_last_title() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("sessions");
        let ws = dir.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();

        let mut old = new_session(&ws);
        old.messages.push(user("old question"));
        old.title = "old question".into();
        let mut old_store = SessionStore::create(&root, &old).unwrap();
        old_store.sync(&old).unwrap();

        let mut new = new_session(&ws);
        new.messages.push(user("new question"));
        new.messages.push(asst("new answer"));
        new.title = "new question".into();
        let mut new_store = SessionStore::create(&root, &new).unwrap();
        new_store.sync(&new).unwrap();

        // Force distinct mtimes regardless of filesystem granularity.
        let past = SystemTime::now() - std::time::Duration::from_secs(120);
        fs::OpenOptions::new()
            .append(true)
            .open(old_store.path())
            .unwrap()
            .set_modified(past)
            .unwrap();

        let listed = list_sessions(&root, &ws);
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].id, new.id);
        assert_eq!(listed[0].title, "new question");
        assert_eq!(listed[0].message_count, 2);
        assert_eq!(listed[1].id, old.id);
        assert_eq!(listed[1].title, "old question");

        // Unknown workspace → empty, not an error.
        assert!(list_sessions(&root, &ws.join("nope")).is_empty());
    }
}
