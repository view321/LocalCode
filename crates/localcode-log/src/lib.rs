//! Structured logging with secret redaction for LocalCode.

use localcode_core::config::LoggingConfig;
use localcode_core::paths::AppPaths;
use regex::Regex;
use std::path::PathBuf;
use std::sync::OnceLock;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

static REDACT_PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();

fn redact_patterns() -> &'static Vec<Regex> {
    REDACT_PATTERNS.get_or_init(|| {
        vec![
            Regex::new(r"(?i)(api[_-]?key|token|authorization|bearer|secret|password)\s*[:=]\s*\S+")
                .expect("regex"),
            Regex::new(r"(?i)Bearer\s+[A-Za-z0-9\-._~+/]+=*").expect("regex"),
            Regex::new(r"hf_[A-Za-z0-9]{20,}").expect("regex"),
            Regex::new(r"sk-[A-Za-z0-9]{20,}").expect("regex"),
        ]
    })
}

/// Redact known secret patterns from a string.
pub fn redact(input: &str) -> String {
    let mut out = input.to_string();
    for re in redact_patterns() {
        out = re
            .replace_all(&out, |caps: &regex::Captures| {
                if let Some(m) = caps.get(0) {
                    let s = m.as_str();
                    if let Some(idx) = s.find(|c: char| c == ':' || c == '=') {
                        format!("{}***REDACTED***", &s[..=idx])
                    } else if s.to_lowercase().starts_with("bearer ") {
                        "Bearer ***REDACTED***".into()
                    } else {
                        "***REDACTED***".into()
                    }
                } else {
                    "***REDACTED***".into()
                }
            })
            .into_owned();
    }
    out
}

pub struct LogGuard {
    _guard: tracing_appender::non_blocking::WorkerGuard,
    pub log_dir: PathBuf,
}

/// Initialize tracing to stderr + rotating file under log_dir.
pub fn init(paths: &AppPaths, cfg: &LoggingConfig) -> Result<LogGuard, localcode_core::LocalCodeError> {
    paths.ensure_dirs()?;
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&cfg.level));

    let file_appender = RollingFileAppender::new(Rotation::DAILY, &paths.log_dir, "localcode.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let file_layer = fmt::layer()
        .with_writer(non_blocking)
        .with_ansi(false)
        .json();

    let stderr_layer = fmt::layer().with_writer(std::io::stderr).with_ansi(true);

    // Ignore double-init in tests
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .with(stderr_layer)
        .try_init();

    Ok(LogGuard {
        _guard: guard,
        log_dir: paths.log_dir.clone(),
    })
}

/// Read recent log lines, optionally filtered by correlation id. Redacts if requested.
pub fn read_recent_logs(
    log_dir: &std::path::Path,
    max_lines: usize,
    correlation_id: Option<&str>,
    do_redact: bool,
) -> std::io::Result<String> {
    use std::fs;
    let mut files: Vec<_> = fs::read_dir(log_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("localcode.log")
        })
        .collect();
    files.sort_by_key(|e| e.file_name());
    let mut lines: Vec<String> = Vec::new();
    for entry in files.iter().rev() {
        if let Ok(content) = fs::read_to_string(entry.path()) {
            for line in content.lines().rev() {
                if let Some(cid) = correlation_id {
                    if !line.contains(cid) {
                        continue;
                    }
                }
                let line = if do_redact {
                    redact(line)
                } else {
                    line.to_string()
                };
                lines.push(line);
                if lines.len() >= max_lines {
                    break;
                }
            }
        }
        if lines.len() >= max_lines {
            break;
        }
    }
    lines.reverse();
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_bearer_and_keys() {
        let s = "Authorization: Bearer sk-abc123xyz78901234567890 and api_key=secretvalue";
        let r = redact(s);
        assert!(r.contains("REDACTED"));
        assert!(!r.contains("sk-abc123"));
    }
}
