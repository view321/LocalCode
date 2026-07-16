//! Results and live events for agentic benchmark runs.
//!
//! Every task produces one [`TaskReport`] line in `results.jsonl`; the run
//! ends with a `summary.json`. Results record (model × harness): the harness
//! version rides along so numbers from different LocalCode builds are never
//! compared as equals.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use localcode_core::error::{ErrorCode, LocalCodeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Every grader check passed.
    Passed,
    /// Graded, but at least one check failed (partial credit in `score`).
    Failed,
    /// Infrastructure or agent-phase error — grading didn't complete.
    Error,
}

/// One grader check outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckReport {
    pub name: String,
    pub passed: bool,
    pub weight: f64,
    pub exit_code: i32,
    /// Last ~2k chars of combined output, for diagnosis without re-running.
    pub output_tail: String,
}

/// What the agent phase cost and how it behaved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    pub wall_ms: u64,
    /// Model⇄tool rounds used (assistant messages seen).
    pub rounds: u32,
    pub tool_calls: u32,
    pub tool_errors: u32,
    /// The agent phase hit the task wall-clock cap (grading still ran).
    pub timed_out: bool,
    /// Agent-phase error, if the turn failed outright (grading still ran).
    pub error: Option<String>,
    /// Tail of the agent's final message.
    pub final_message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskReport {
    pub task_id: String,
    pub title: String,
    pub domain: String,
    pub tier: u8,
    pub weight: f64,
    pub status: TaskStatus,
    /// Weight-fraction of passing checks in [0, 1].
    pub score: f64,
    pub checks: Vec<CheckReport>,
    /// Absent in oracle (verify) runs.
    pub agent: Option<AgentStats>,
    /// Explanation for `status == Error`, or oracle-verify failure reasons.
    pub error: Option<String>,
    pub started_at: String,
    pub finished_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub suite_id: String,
    pub suite_version: String,
    /// True for `bench verify` (oracle solutions instead of a model).
    pub oracle: bool,
    pub model: String,
    pub runtime_name: String,
    pub tasks_total: usize,
    pub tasks_passed: usize,
    pub tasks_failed: usize,
    pub tasks_errored: usize,
    /// Task-weighted mean of task scores in [0, 1].
    pub score: f64,
    /// Fraction of tasks with every check passing.
    pub strict_pass_rate: f64,
    pub started_at: String,
    pub finished_at: String,
    pub results_dir: String,
    pub harness_version: String,
}

/// Aggregate task reports into the run-level numbers.
pub fn summarize_tasks(reports: &[TaskReport]) -> (f64, f64, usize, usize, usize) {
    let total_w: f64 = reports.iter().map(|r| r.weight).sum::<f64>().max(1e-9);
    let score = reports.iter().map(|r| r.weight * r.score).sum::<f64>() / total_w;
    let passed = reports
        .iter()
        .filter(|r| r.status == TaskStatus::Passed)
        .count();
    let failed = reports
        .iter()
        .filter(|r| r.status == TaskStatus::Failed)
        .count();
    let errored = reports
        .iter()
        .filter(|r| r.status == TaskStatus::Error)
        .count();
    let strict = if reports.is_empty() {
        0.0
    } else {
        passed as f64 / reports.len() as f64
    };
    (score, strict, passed, failed, errored)
}

/// Phases a task moves through, for live progress displays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskPhase {
    PullImage,
    StartContainer,
    Agent,
    Grade,
}

impl std::fmt::Display for TaskPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TaskPhase::PullImage => "pulling image",
            TaskPhase::StartContainer => "starting container",
            TaskPhase::Agent => "agent working",
            TaskPhase::Grade => "grading",
        })
    }
}

/// Live progress from a running suite (TUI cards, CLI lines, `--json`).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunEvent {
    RunStarted {
        run_id: String,
        suite_id: String,
        total: usize,
    },
    TaskStarted {
        index: usize,
        total: usize,
        task_id: String,
        title: String,
    },
    TaskPhase {
        task_id: String,
        phase: TaskPhase,
    },
    /// A compact one-liner of agent activity (tool call started/finished).
    AgentActivity {
        task_id: String,
        line: String,
    },
    TaskFinished {
        task_id: String,
        report: TaskReport,
    },
    RunFinished {
        summary: RunSummary,
    },
}

/// The directory a run writes into.
#[derive(Debug, Clone)]
pub struct RunDir {
    pub root: PathBuf,
}

impl RunDir {
    pub fn create(runs_root: &Path, name: &str) -> Result<Self, LocalCodeError> {
        let root = runs_root.join(name);
        std::fs::create_dir_all(&root).map_err(|e| {
            LocalCodeError::new(ErrorCode::ConfigSaveFailed, e.to_string())
                .with_cause(format!("Cannot create run directory {}", root.display()))
        })?;
        Ok(Self { root })
    }

    pub fn task_dir(&self, task_id: &str) -> PathBuf {
        self.root.join("tasks").join(task_id)
    }

    /// Append one task report to `results.jsonl` (best-effort durable: each
    /// line lands on disk as the run progresses, so a crash keeps prior tasks).
    pub fn append_task(&self, report: &TaskReport) -> Result<(), LocalCodeError> {
        use std::io::Write;
        let path = self.root.join("results.jsonl");
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| LocalCodeError::new(ErrorCode::ConfigSaveFailed, e.to_string()))?;
        let line = serde_json::to_string(report)
            .map_err(|e| LocalCodeError::new(ErrorCode::Internal, e.to_string()))?;
        writeln!(f, "{line}")
            .map_err(|e| LocalCodeError::new(ErrorCode::ConfigSaveFailed, e.to_string()))?;
        Ok(())
    }

    pub fn write_summary(&self, summary: &RunSummary) -> Result<(), LocalCodeError> {
        let path = self.root.join("summary.json");
        let body = serde_json::to_string_pretty(summary)
            .map_err(|e| LocalCodeError::new(ErrorCode::Internal, e.to_string()))?;
        std::fs::write(&path, body)
            .map_err(|e| LocalCodeError::new(ErrorCode::ConfigSaveFailed, e.to_string()))?;
        Ok(())
    }
}

/// Keep the last `max` chars of `s` (char-safe), marking elision.
pub fn tail(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    let skipped: String = s.chars().skip(n - max).collect();
    format!("…{skipped}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(id: &str, status: TaskStatus, score: f64, weight: f64) -> TaskReport {
        TaskReport {
            task_id: id.into(),
            title: id.into(),
            domain: "t".into(),
            tier: 1,
            weight,
            status,
            score,
            checks: vec![],
            agent: None,
            error: None,
            started_at: "".into(),
            finished_at: "".into(),
        }
    }

    #[test]
    fn summary_weights_task_scores() {
        let reports = vec![
            report("a", TaskStatus::Passed, 1.0, 1.0),
            report("b", TaskStatus::Failed, 0.5, 1.0),
            report("c", TaskStatus::Error, 0.0, 2.0),
        ];
        let (score, strict, passed, failed, errored) = summarize_tasks(&reports);
        // (1*1 + 0.5*1 + 0*2) / 4
        assert!((score - 0.375).abs() < 1e-9, "{score}");
        assert!((strict - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!((passed, failed, errored), (1, 1, 1));
    }

    #[test]
    fn results_jsonl_appends_lines() {
        let dir = tempfile::tempdir().unwrap();
        let run = RunDir::create(dir.path(), "r1").unwrap();
        run.append_task(&report("a", TaskStatus::Passed, 1.0, 1.0)).unwrap();
        run.append_task(&report("b", TaskStatus::Failed, 0.0, 1.0)).unwrap();
        let body = std::fs::read_to_string(run.root.join("results.jsonl")).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        let first: TaskReport = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first.task_id, "a");
    }

    #[test]
    fn tail_keeps_suffix() {
        assert_eq!(tail("hello", 10), "hello");
        assert_eq!(tail("hello world", 5), "…world");
        // Char-safe on multibyte.
        assert_eq!(tail("日本語です", 2), "…です");
    }
}
