//! Live per-model process monitors backing the `/dash` multi-model view.
//!
//! Every backend that spawns a server process registers a [`ModelMonitor`] here
//! keyed by the runtime id it hands back. The monitor holds three things the
//! dashboard needs but that never fit on [`ActiveRuntime`](localcode_core::runtime::ActiveRuntime):
//!
//! * the exact **launch command** (click-to-copy), so a deploy is reproducible
//!   outside the app;
//! * a bounded **log ring** fed by the child's stdout/stderr drain, so the card
//!   can show the newest backend output live; and
//! * the process **state** — including a real non-zero **exit code** captured by
//!   a watcher task, so a server that dies after a healthy start surfaces its
//!   error instead of silently vanishing from the runtime list.
//!
//! The store is a cheap `Arc` clone shared between [`BackendRegistry`] (which
//! hands a clone to every backend at construction) and the TUI (which snapshots
//! it every frame). Ollama, which we don't spawn, registers a monitor with no
//! child so its card still shows the served model and command.

use crate::BackendKind;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Max log lines retained per model. A failing server prints its traceback once;
/// keeping a couple hundred lines lets the card show the tail without unbounded
/// growth.
pub const DASH_LOG_CAP: usize = 200;

/// Lifecycle of a monitored model process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcState {
    /// Spawned, not yet confirmed healthy.
    Starting,
    /// Health check passed; serving.
    Running,
    /// Process exited. `code` is `None` when killed by a signal (Unix) or the
    /// code was unavailable. `ok` is true only for an explicit success status.
    Exited { code: Option<i32>, ok: bool },
    /// We don't own the process (Ollama's shared server, a remote box) — state
    /// is whatever the runtime's health says elsewhere.
    External,
}

impl ProcState {
    /// A non-zero / signal exit — the case the dashboard shows an error for.
    pub fn is_failure(&self) -> bool {
        matches!(self, ProcState::Exited { ok: false, .. })
    }
}

/// One monitored model. Cloneable field handles (`logs`, `state`) are shared with
/// the drain and watcher tasks; the plain fields are set once at registration.
pub struct ModelMonitor {
    pub runtime_id: String,
    pub name: String,
    pub backend: BackendKind,
    pub model_id: Option<String>,
    /// The exact command the process was launched with (for click-to-copy).
    pub command: String,
    /// Estimated VRAM the model occupies, filled in by the deploy pipeline once
    /// the fit is known. `None` for Ollama / externally-managed models.
    est_vram_bytes: Mutex<Option<u64>>,
    logs: Arc<Mutex<VecDeque<String>>>,
    state: Arc<Mutex<ProcState>>,
}

impl ModelMonitor {
    pub fn logs_handle(&self) -> Arc<Mutex<VecDeque<String>>> {
        self.logs.clone()
    }

    pub fn state_handle(&self) -> Arc<Mutex<ProcState>> {
        self.state.clone()
    }

    /// Append a line to the log ring, trimming to [`DASH_LOG_CAP`].
    pub fn push_log(&self, line: impl Into<String>) {
        push_line(&self.logs, line.into());
    }

    pub fn set_state(&self, s: ProcState) {
        if let Ok(mut g) = self.state.lock() {
            *g = s;
        }
    }

    pub fn set_vram(&self, bytes: u64) {
        if let Ok(mut g) = self.est_vram_bytes.lock() {
            *g = Some(bytes);
        }
    }

    fn snapshot(&self, log_lines: usize) -> DashSnapshot {
        let state = self
            .state
            .lock()
            .map(|g| g.clone())
            .unwrap_or(ProcState::External);
        let log_tail = self
            .logs
            .lock()
            .map(|b| {
                let start = b.len().saturating_sub(log_lines);
                b.iter().skip(start).cloned().collect()
            })
            .unwrap_or_default();
        DashSnapshot {
            runtime_id: self.runtime_id.clone(),
            name: self.name.clone(),
            backend: self.backend,
            model_id: self.model_id.clone(),
            command: self.command.clone(),
            est_vram_bytes: self.est_vram_bytes.lock().ok().and_then(|g| *g),
            state,
            log_tail,
        }
    }
}

/// A plain, lock-free copy of a monitor for the render thread.
#[derive(Debug, Clone)]
pub struct DashSnapshot {
    pub runtime_id: String,
    pub name: String,
    pub backend: BackendKind,
    pub model_id: Option<String>,
    pub command: String,
    pub est_vram_bytes: Option<u64>,
    pub state: ProcState,
    /// Newest log lines last.
    pub log_tail: Vec<String>,
}

impl DashSnapshot {
    /// The captured error tail for a failed process — the newest lines, joined,
    /// for the "copy error" button. Empty when the process didn't fail or wrote
    /// nothing.
    pub fn error_text(&self) -> Option<String> {
        if !self.state.is_failure() {
            return None;
        }
        let body = self.log_tail.join("\n");
        let head = match &self.state {
            ProcState::Exited { code: Some(c), .. } => format!("{} exited with code {c}\n", self.name),
            _ => format!("{} exited abnormally\n", self.name),
        };
        Some(format!("{head}{body}").trim().to_string())
    }
}

/// Shared registry of live model monitors. Cheap to clone (`Arc` inside).
#[derive(Clone, Default)]
pub struct ModelMonitors {
    inner: Arc<Mutex<Vec<Arc<ModelMonitor>>>>,
}

impl ModelMonitors {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or replace) a monitor for `runtime_id`, returning the shared
    /// handle so the caller can wire log capture / an exit watcher to it.
    pub fn register(
        &self,
        runtime_id: impl Into<String>,
        name: impl Into<String>,
        backend: BackendKind,
        model_id: Option<String>,
        command: impl Into<String>,
        initial: ProcState,
    ) -> Arc<ModelMonitor> {
        let runtime_id = runtime_id.into();
        let monitor = Arc::new(ModelMonitor {
            runtime_id: runtime_id.clone(),
            name: name.into(),
            backend,
            model_id,
            command: command.into(),
            est_vram_bytes: Mutex::new(None),
            logs: Arc::new(Mutex::new(VecDeque::new())),
            state: Arc::new(Mutex::new(initial)),
        });
        if let Ok(mut v) = self.inner.lock() {
            v.retain(|m| m.runtime_id != runtime_id);
            v.push(monitor.clone());
        }
        monitor
    }

    /// The monitor for a runtime id, if registered.
    pub fn get(&self, runtime_id: &str) -> Option<Arc<ModelMonitor>> {
        self.inner
            .lock()
            .ok()?
            .iter()
            .find(|m| m.runtime_id == runtime_id)
            .cloned()
    }

    /// Attach a stored estimated-VRAM figure to a monitor (called by the deploy
    /// pipeline once the fit is known).
    pub fn set_vram(&self, runtime_id: &str, bytes: u64) {
        if let Some(m) = self.get(runtime_id) {
            m.set_vram(bytes);
        }
    }

    /// Drop a monitor (on stop). A no-op if it was never registered.
    pub fn remove(&self, runtime_id: &str) {
        if let Ok(mut v) = self.inner.lock() {
            v.retain(|m| m.runtime_id != runtime_id);
        }
    }

    /// Lock-free snapshot of every monitor for rendering, newest `log_lines`
    /// each. Order is registration order (stable across frames).
    pub fn snapshot(&self, log_lines: usize) -> Vec<DashSnapshot> {
        self.inner
            .lock()
            .map(|v| v.iter().map(|m| m.snapshot(log_lines)).collect())
            .unwrap_or_default()
    }
}

/// Render a spawn as a copy-pasteable shell command: the program followed by
/// its args, with any arg containing whitespace double-quoted. Used for the
/// click-to-copy command line on each dashboard card.
pub fn format_command(program: &str, args: &[String]) -> String {
    let mut out = quote_arg(program);
    for a in args {
        out.push(' ');
        out.push_str(&quote_arg(a));
    }
    out
}

fn quote_arg(a: &str) -> String {
    if a.is_empty() || a.chars().any(|c| c.is_whitespace()) {
        format!("\"{a}\"")
    } else {
        a.to_string()
    }
}

/// Split a shell-ish command line into program + args, the inverse of
/// [`format_command`]. Honors single and double quotes and backslash escapes so
/// a user-edited launch command (which may contain quoted paths) round-trips.
/// Returns `None` when the line is empty or has no program token.
pub fn split_command(line: &str) -> Option<(String, Vec<String>)> {
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut has_token = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' if !in_single => {
                // Escape the next char (common on Unix paths / spaces).
                if let Some(next) = chars.next() {
                    cur.push(next);
                    has_token = true;
                }
            }
            '\'' if !in_double => {
                in_single = !in_single;
                has_token = true;
            }
            '"' if !in_single => {
                in_double = !in_double;
                has_token = true;
            }
            c if c.is_whitespace() && !in_single && !in_double => {
                if has_token {
                    tokens.push(std::mem::take(&mut cur));
                    has_token = false;
                }
            }
            c => {
                cur.push(c);
                has_token = true;
            }
        }
    }
    if has_token {
        tokens.push(cur);
    }
    if tokens.is_empty() {
        return None;
    }
    let program = tokens.remove(0);
    Some((program, tokens))
}

/// Resolve what a backend should actually spawn: honor a user/assistant command
/// override when present and parseable, otherwise the backend-built command.
/// Returns `(program, args, display_command)` — `display_command` is what the
/// `/dash` card shows and is byte-identical to what runs.
pub fn resolve_launch(
    default_bin: &str,
    default_args: Vec<String>,
    override_cmd: Option<&str>,
) -> (String, Vec<String>, String) {
    if let Some(line) = override_cmd {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            if let Some((prog, args)) = split_command(trimmed) {
                return (prog, args, trimmed.to_string());
            }
        }
    }
    let display = format_command(default_bin, &default_args);
    (default_bin.to_string(), default_args, display)
}

fn push_line(buf: &Arc<Mutex<VecDeque<String>>>, line: String) {
    if let Ok(mut b) = buf.lock() {
        if b.len() >= DASH_LOG_CAP {
            b.pop_front();
        }
        b.push_back(line);
    }
}

/// Drain a child's stdout+stderr into a monitor's log ring (and tracing). Like
/// [`crate::spawn_io_drain`] but writes into the shared, dashboard-visible ring
/// buffer instead of only tracing. Call once, right after spawn, before the
/// child is moved into a shared handle.
pub fn capture_into_monitor(tag: &str, child: &mut tokio::process::Child, monitor: &Arc<ModelMonitor>) {
    capture_stream(child.stdout.take(), tag.to_string(), monitor.logs_handle());
    capture_stream(child.stderr.take(), tag.to_string(), monitor.logs_handle());
}

fn capture_stream<R>(reader: Option<R>, tag: String, buf: Arc<Mutex<VecDeque<String>>>)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    use tokio::io::{AsyncBufReadExt, BufReader};
    let Some(reader) = reader else { return };
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::debug!(target: "backend_io", backend = %tag, "{line}");
            push_line(&buf, line);
        }
    });
}

/// Spawn a watcher that polls the shared child for exit and records the real
/// exit code on the monitor. Runs until the process exits or `stop()` takes the
/// child (in which case the watcher sees `None` and ends quietly).
///
/// Polling (rather than owning + `wait()`) keeps `stop()`'s `child.kill()` able
/// to take the child out from under it — mirroring how the local assistant
/// holds its child in an `Arc<Mutex<Option<Child>>>`.
pub fn spawn_exit_watch(
    child: Arc<tokio::sync::Mutex<Option<tokio::process::Child>>>,
    monitor: Arc<ModelMonitor>,
) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
            let mut guard = child.lock().await;
            match guard.as_mut() {
                Some(c) => match c.try_wait() {
                    Ok(Some(status)) => {
                        monitor.set_state(ProcState::Exited {
                            code: status.code(),
                            ok: status.success(),
                        });
                        *guard = None;
                        if !status.success() {
                            monitor.push_log(format!("[localcode] process exited: {status}"));
                        }
                        return;
                    }
                    Ok(None) => {}
                    Err(_) => return,
                },
                // stop() already took and killed the child.
                None => return,
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_get_remove_roundtrip() {
        let m = ModelMonitors::new();
        let mon = m.register(
            "rt-1",
            "vllm:foo",
            BackendKind::Vllm,
            Some("org/foo".into()),
            "vllm serve org/foo",
            ProcState::Starting,
        );
        mon.push_log("loading weights");
        mon.set_state(ProcState::Running);
        m.set_vram("rt-1", 8 * 1024 * 1024 * 1024);

        let snap = m.snapshot(10);
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].name, "vllm:foo");
        assert_eq!(snap[0].state, ProcState::Running);
        assert_eq!(snap[0].est_vram_bytes, Some(8 * 1024 * 1024 * 1024));
        assert_eq!(snap[0].log_tail, vec!["loading weights".to_string()]);

        m.remove("rt-1");
        assert!(m.snapshot(10).is_empty());
    }

    #[test]
    fn register_replaces_same_id() {
        let m = ModelMonitors::new();
        m.register("rt", "a", BackendKind::LlamaCpp, None, "cmd-a", ProcState::Running);
        m.register("rt", "b", BackendKind::LlamaCpp, None, "cmd-b", ProcState::Running);
        let snap = m.snapshot(1);
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].name, "b");
    }

    #[test]
    fn failure_state_yields_error_text() {
        let m = ModelMonitors::new();
        let mon = m.register("rt", "vllm:x", BackendKind::Vllm, None, "vllm serve x", ProcState::Running);
        mon.push_log("CUDA error: out of memory");
        mon.set_state(ProcState::Exited { code: Some(1), ok: false });
        let snap = &m.snapshot(10)[0];
        assert!(snap.state.is_failure());
        let err = snap.error_text().expect("failure has error text");
        assert!(err.contains("code 1"), "{err}");
        assert!(err.contains("out of memory"), "{err}");
    }

    #[test]
    fn healthy_snapshot_has_no_error_text() {
        let m = ModelMonitors::new();
        let mon = m.register("rt", "x", BackendKind::LlamaCpp, None, "cmd", ProcState::Running);
        mon.push_log("all good");
        assert!(m.snapshot(5)[0].error_text().is_none());
    }

    #[test]
    fn split_command_round_trips_format() {
        let args = vec![
            "serve".to_string(),
            "org/model".to_string(),
            "--host".to_string(),
            "0.0.0.0".to_string(),
        ];
        let line = format_command("vllm", &args);
        let (prog, got) = split_command(&line).unwrap();
        assert_eq!(prog, "vllm");
        assert_eq!(got, args);
    }

    #[test]
    fn split_command_honors_quotes() {
        let (prog, args) =
            split_command("llama-server -m \"C:/my models/x.gguf\" --port 8080").unwrap();
        assert_eq!(prog, "llama-server");
        assert_eq!(args[1], "C:/my models/x.gguf");
        assert_eq!(args.last().unwrap(), "8080");
    }

    #[test]
    fn resolve_launch_prefers_override() {
        let (prog, args, display) =
            resolve_launch("vllm", vec!["serve".into()], Some("  custom serve x  "));
        assert_eq!(prog, "custom");
        assert_eq!(args, vec!["serve".to_string(), "x".to_string()]);
        assert_eq!(display, "custom serve x");

        // Blank / whitespace override falls back to the built command.
        let (prog, _args, display) = resolve_launch("vllm", vec!["serve".into()], Some("   "));
        assert_eq!(prog, "vllm");
        assert_eq!(display, "vllm serve");
        let (prog, _args, _display) = resolve_launch("vllm", vec!["serve".into()], None);
        assert_eq!(prog, "vllm");
    }

    #[test]
    fn log_ring_is_bounded() {
        let m = ModelMonitors::new();
        let mon = m.register("rt", "x", BackendKind::LlamaCpp, None, "cmd", ProcState::Running);
        for i in 0..(DASH_LOG_CAP + 50) {
            mon.push_log(format!("line {i}"));
        }
        let snap = &m.snapshot(DASH_LOG_CAP + 100)[0];
        assert_eq!(snap.log_tail.len(), DASH_LOG_CAP);
        // Oldest lines were dropped; newest retained.
        assert_eq!(snap.log_tail.last().unwrap(), &format!("line {}", DASH_LOG_CAP + 49));
    }
}
