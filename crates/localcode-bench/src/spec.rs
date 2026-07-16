//! On-disk schema for agentic benchmark suites.
//!
//! A suite is a directory:
//!
//! ```text
//! <suite>/
//!   suite.toml            id, title, version, description
//!   tasks/<task-id>/
//!     task.toml           prompt, image, budgets, grader checks
//!     workspace/          the files the agent starts from
//!     hidden/             grader-only overlay (tests) — copied over the
//!                         agent's tree at grade time, so tampering with a
//!                         test file never survives into grading
//!     solution/           reference-solution overlay for `bench verify`
//! ```
//!
//! Grading runs each check command in a fresh container; a check passes iff
//! it exits 0. A task's score is the weight-fraction of passing checks, and
//! the task passes iff every check passes.

use localcode_core::error::{ErrorCode, LocalCodeError};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const SUITE_MANIFEST: &str = "suite.toml";
pub const TASK_MANIFEST: &str = "task.toml";
pub const TASKS_DIR: &str = "tasks";
pub const WORKSPACE_DIR: &str = "workspace";
pub const HIDDEN_DIR: &str = "hidden";
pub const SOLUTION_DIR: &str = "solution";

/// Where the workspace is mounted inside every task/grade container.
pub const CONTAINER_WORKSPACE: &str = "/workspace";

fn default_weight() -> f64 {
    1.0
}
fn default_network() -> String {
    "none".into()
}
fn default_task_timeout() -> u64 {
    // Agent phase wall-clock. Local models are slow; a tier-1 task at
    // ~30 tok/s regularly needs several minutes of generation.
    600
}
fn default_grade_timeout() -> u64 {
    180
}

/// `suite.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteMeta {
    pub id: String,
    pub title: String,
    pub version: String,
    #[serde(default)]
    pub description: String,
}

/// One grader check: a shell command run in a fresh container over the graded
/// tree; passes iff it exits 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckSpec {
    pub name: String,
    pub cmd: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraderSpec {
    /// Wall-clock cap per check command.
    #[serde(default = "default_grade_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub checks: Vec<CheckSpec>,
}

impl Default for GraderSpec {
    fn default() -> Self {
        Self {
            timeout_secs: default_grade_timeout(),
            checks: Vec::new(),
        }
    }
}

/// `task.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    /// Defaults to the task directory name.
    #[serde(default)]
    pub id: String,
    pub title: String,
    /// The user message handed to the agent.
    pub prompt: String,
    /// Freeform grouping tag (python / web / rust / …).
    #[serde(default)]
    pub domain: String,
    /// Difficulty rung (0 = function-level … 4 = greenfield e2e).
    #[serde(default)]
    pub tier: u8,
    /// Docker image for both the agent container and the grade containers.
    pub image: String,
    /// Container network: "none" (default) or "bridge".
    #[serde(default = "default_network")]
    pub network: String,
    /// Wall-clock cap for the whole agent phase. Grading still runs when it
    /// fires — partial work can earn partial credit.
    #[serde(default = "default_task_timeout")]
    pub timeout_secs: u64,
    /// Cap on model⇄tool rounds for this task (default 24).
    #[serde(default)]
    pub agent_max_rounds: Option<u32>,
    /// Wall-clock cap per `bash` call inside the container (default 120s).
    #[serde(default)]
    pub exec_timeout_secs: Option<u64>,
    /// Weight of this task in the suite score.
    #[serde(default = "default_weight")]
    pub weight: f64,
    #[serde(default)]
    pub grader: GraderSpec,
}

/// A loaded task: parsed manifest + its directory.
#[derive(Debug, Clone)]
pub struct BenchTask {
    pub spec: TaskSpec,
    pub dir: PathBuf,
}

impl BenchTask {
    pub fn workspace_dir(&self) -> PathBuf {
        self.dir.join(WORKSPACE_DIR)
    }
    pub fn hidden_dir(&self) -> PathBuf {
        self.dir.join(HIDDEN_DIR)
    }
    pub fn solution_dir(&self) -> PathBuf {
        self.dir.join(SOLUTION_DIR)
    }
    pub fn has_solution(&self) -> bool {
        self.solution_dir().is_dir()
    }
}

/// A loaded suite: metadata + tasks, in directory order.
#[derive(Debug, Clone)]
pub struct BenchSuite {
    pub meta: SuiteMeta,
    pub root: PathBuf,
    pub tasks: Vec<BenchTask>,
}

impl BenchSuite {
    /// Distinct Docker images the suite needs, in first-use order.
    pub fn images(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for t in &self.tasks {
            if !out.contains(&t.spec.image) {
                out.push(t.spec.image.clone());
            }
        }
        out
    }
}

/// Cheap listing entry for pickers (TUI suite list, `bench list`).
#[derive(Debug, Clone, Serialize)]
pub struct SuiteInfo {
    pub id: String,
    pub title: String,
    pub version: String,
    pub description: String,
    pub path: PathBuf,
    pub task_count: usize,
    pub images: Vec<String>,
}

fn parse_err(path: &Path, e: impl std::fmt::Display) -> LocalCodeError {
    LocalCodeError::new(ErrorCode::ConfigParseFailed, e.to_string())
        .with_cause(format!("Invalid manifest: {}", path.display()))
}

/// Load and validate a suite directory.
pub fn load_bench_suite(root: &Path) -> Result<BenchSuite, LocalCodeError> {
    let manifest = root.join(SUITE_MANIFEST);
    let raw = std::fs::read_to_string(&manifest).map_err(|e| {
        LocalCodeError::new(ErrorCode::ConfigLoadFailed, e.to_string())
            .with_cause(format!("Cannot read {}", manifest.display()))
            .with_hint("A suite directory must contain suite.toml (see `localcode bench list`)")
    })?;
    let meta: SuiteMeta = toml::from_str(&raw).map_err(|e| parse_err(&manifest, e))?;

    let tasks_root = root.join(TASKS_DIR);
    let mut dirs: Vec<PathBuf> = match std::fs::read_dir(&tasks_root) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
        Err(e) => {
            return Err(LocalCodeError::new(ErrorCode::ConfigLoadFailed, e.to_string())
                .with_cause(format!("Cannot read {}", tasks_root.display()))
                .with_hint("A suite needs a tasks/ directory with one folder per task"))
        }
    };
    dirs.sort();

    let mut tasks = Vec::new();
    for dir in dirs {
        let manifest = dir.join(TASK_MANIFEST);
        if !manifest.is_file() {
            continue; // stray folder — not a task
        }
        let raw = std::fs::read_to_string(&manifest)
            .map_err(|e| parse_err(&manifest, e))?;
        let mut spec: TaskSpec = toml::from_str(&raw).map_err(|e| parse_err(&manifest, e))?;
        if spec.id.is_empty() {
            spec.id = dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
        }
        validate_task(&spec, &dir)?;
        tasks.push(BenchTask { spec, dir });
    }

    if tasks.is_empty() {
        return Err(LocalCodeError::new(
            ErrorCode::ConfigLoadFailed,
            format!("Suite '{}' has no tasks", meta.id),
        )
        .with_cause(format!("No tasks/*/task.toml under {}", root.display())));
    }

    Ok(BenchSuite {
        meta,
        root: root.to_path_buf(),
        tasks,
    })
}

fn validate_task(spec: &TaskSpec, dir: &Path) -> Result<(), LocalCodeError> {
    let fail = |msg: String| {
        Err(LocalCodeError::new(ErrorCode::ConfigParseFailed, msg)
            .with_cause(format!("Task at {}", dir.display())))
    };
    if spec.image.trim().is_empty() {
        return fail(format!("Task '{}' has no image", spec.id));
    }
    if spec.prompt.trim().is_empty() {
        return fail(format!("Task '{}' has an empty prompt", spec.id));
    }
    if !matches!(spec.network.as_str(), "none" | "bridge") {
        return fail(format!(
            "Task '{}': network must be \"none\" or \"bridge\", got {:?}",
            spec.id, spec.network
        ));
    }
    if spec.grader.checks.is_empty() {
        return fail(format!("Task '{}' has no grader checks", spec.id));
    }
    if spec.weight <= 0.0 || spec.grader.checks.iter().any(|c| c.weight <= 0.0) {
        return fail(format!("Task '{}': weights must be > 0", spec.id));
    }
    if !dir.join(WORKSPACE_DIR).is_dir() {
        return fail(format!("Task '{}' has no workspace/ directory", spec.id));
    }
    Ok(())
}

/// List the suites under `suites_root`. Unreadable entries are skipped with a
/// warning rather than failing the listing.
pub fn list_suites(suites_root: &Path) -> Vec<SuiteInfo> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(suites_root) else {
        return out;
    };
    let mut dirs: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join(SUITE_MANIFEST).is_file())
        .collect();
    dirs.sort();
    for dir in dirs {
        match load_bench_suite(&dir) {
            Ok(s) => out.push(SuiteInfo {
                id: s.meta.id.clone(),
                title: s.meta.title.clone(),
                version: s.meta.version.clone(),
                description: s.meta.description.clone(),
                path: dir,
                task_count: s.tasks.len(),
                images: s.images(),
            }),
            Err(e) => {
                tracing::warn!(dir = %dir.display(), error = %e.message, "skipping broken suite");
            }
        }
    }
    out
}

/// Resolve `name_or_path` to a suite: a directory path containing suite.toml,
/// or the name of a suite installed under `suites_root`.
pub fn find_suite(suites_root: &Path, name_or_path: &str) -> Result<BenchSuite, LocalCodeError> {
    let as_path = Path::new(name_or_path);
    if as_path.join(SUITE_MANIFEST).is_file() {
        return load_bench_suite(as_path);
    }
    let installed = suites_root.join(name_or_path);
    if installed.join(SUITE_MANIFEST).is_file() {
        return load_bench_suite(&installed);
    }
    Err(LocalCodeError::new(
        ErrorCode::ConfigLoadFailed,
        format!("No benchmark suite named '{name_or_path}'"),
    )
    .with_hint("List installed suites with: localcode bench list")
    .with_hint("Install one with: localcode bench pull <dir-or-tar.gz-url>"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, content: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn minimal_suite(root: &Path) {
        write(
            &root.join("suite.toml"),
            "id = \"s1\"\ntitle = \"S1\"\nversion = \"0.1.0\"\n",
        );
        write(
            &root.join("tasks/alpha/task.toml"),
            r#"
title = "Alpha"
prompt = "do the thing"
image = "python:3.12-slim"
[[grader.checks]]
name = "tests"
cmd = "python3 -m unittest -v t"
"#,
        );
        write(&root.join("tasks/alpha/workspace/keep.py"), "x = 1\n");
    }

    #[test]
    fn loads_minimal_suite_with_defaults() {
        let dir = tempfile::tempdir().unwrap();
        minimal_suite(dir.path());
        let s = load_bench_suite(dir.path()).unwrap();
        assert_eq!(s.meta.id, "s1");
        assert_eq!(s.tasks.len(), 1);
        let t = &s.tasks[0].spec;
        assert_eq!(t.id, "alpha", "id defaults to the directory name");
        assert_eq!(t.network, "none");
        assert_eq!(t.timeout_secs, 600);
        assert_eq!(t.weight, 1.0);
        assert_eq!(t.grader.timeout_secs, 180);
        assert_eq!(t.grader.checks[0].weight, 1.0);
        assert_eq!(s.images(), vec!["python:3.12-slim".to_string()]);
    }

    #[test]
    fn rejects_task_without_checks_or_workspace() {
        let dir = tempfile::tempdir().unwrap();
        minimal_suite(dir.path());
        // No checks.
        write(
            &dir.path().join("tasks/beta/task.toml"),
            "title = \"B\"\nprompt = \"p\"\nimage = \"i\"\n",
        );
        write(&dir.path().join("tasks/beta/workspace/x"), "x");
        let err = load_bench_suite(dir.path()).unwrap_err();
        assert!(err.message.contains("no grader checks"), "{}", err.message);
    }

    #[test]
    fn rejects_unknown_network() {
        let dir = tempfile::tempdir().unwrap();
        write(
            &dir.path().join("suite.toml"),
            "id = \"s\"\ntitle = \"S\"\nversion = \"1\"\n",
        );
        write(
            &dir.path().join("tasks/a/task.toml"),
            r#"
title = "A"
prompt = "p"
image = "i"
network = "host"
[[grader.checks]]
name = "t"
cmd = "true"
"#,
        );
        write(&dir.path().join("tasks/a/workspace/x"), "x");
        let err = load_bench_suite(dir.path()).unwrap_err();
        assert!(err.message.contains("network"), "{}", err.message);
    }

    #[test]
    fn find_suite_by_name_and_path() {
        let root = tempfile::tempdir().unwrap();
        let suites = root.path().join("suites");
        minimal_suite(&suites.join("s1"));
        assert_eq!(find_suite(&suites, "s1").unwrap().meta.id, "s1");
        let by_path = find_suite(&suites, suites.join("s1").to_str().unwrap()).unwrap();
        assert_eq!(by_path.meta.id, "s1");
        assert!(find_suite(&suites, "nope").is_err());
    }

    #[test]
    fn list_skips_broken_suites() {
        let root = tempfile::tempdir().unwrap();
        minimal_suite(&root.path().join("good"));
        write(&root.path().join("bad/suite.toml"), "not toml at all [[[");
        let infos = list_suites(root.path());
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].id, "s1");
        assert_eq!(infos[0].task_count, 1);
    }
}
