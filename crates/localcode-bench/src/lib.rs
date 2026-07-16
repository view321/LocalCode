//! Benchmarking for LocalCode: sandboxed, agentic, local-only.
//!
//! The agentic benchmark ([`runner`]) runs the LocalCode coding agent against
//! suites of Docker-sandboxed tasks and grades the resulting workspace with
//! hidden tests in fresh containers. Suites live on disk (see [`spec`]) and
//! ship separately from the app (`localcode bench pull`); a built-in `smoke`
//! suite is embedded for a zero-setup pipeline check. Everything is
//! local-only: no external judges, no result uploads.
//!
//! The legacy prompt-level QA bench (single completions graded by regex)
//! remains in [`qa`] for compatibility.

mod builtin;
mod docker;
mod fsops;
mod pull;
mod qa;
mod report;
mod runner;
mod spec;

pub use builtin::{ensure_smoke_suite, SMOKE_SUITE_ID};
pub use docker::{cleanup_stale_containers, docker_available, docker_missing_err};
pub use pull::{install_suite_from_dir, install_suite_from_url, pull_suite_images};
pub use qa::*;
pub use report::{
    AgentStats, CheckReport, RunEvent, RunSummary, TaskPhase, TaskReport, TaskStatus,
};
pub use runner::{run_suite, RunOptions};
pub use spec::{
    find_suite, list_suites, load_bench_suite, BenchSuite, BenchTask, CheckSpec, GraderSpec,
    SuiteInfo, SuiteMeta, TaskSpec,
};

use std::path::{Path, PathBuf};

/// Where installed suites live: `<data>/bench/suites/<suite-id>/`.
pub fn suites_root(data_dir: &Path) -> PathBuf {
    data_dir.join("bench").join("suites")
}

/// Where run results live: `<data>/bench/runs/<stamp>-<suite>-<model>/`.
pub fn runs_root(data_dir: &Path) -> PathBuf {
    data_dir.join("bench").join("runs")
}

/// Materialize the built-in smoke suite and list everything installed.
pub fn prepare_suites(data_dir: &Path) -> Vec<SuiteInfo> {
    let root = suites_root(data_dir);
    if let Err(e) = std::fs::create_dir_all(&root) {
        tracing::warn!(error = %e, "cannot create suites dir");
        return Vec::new();
    }
    if let Err(e) = ensure_smoke_suite(&root) {
        tracing::warn!(error = %e.message, "cannot materialize smoke suite");
    }
    list_suites(&root)
}
