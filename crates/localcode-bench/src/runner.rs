//! Agentic benchmark runner: one sandboxed Docker container per task, the
//! LocalCode coding agent driving it in-process, hidden-test grading in fresh
//! containers, and an oracle mode that validates tasks without a model.
//!
//! Per task (agent mode):
//! 1. copy the task's `workspace/` into the run directory
//! 2. start the task container (workspace bind-mounted at /workspace)
//! 3. run one agent turn against the chosen runtime — `bash` executes inside
//!    the container ([`ContainerShell`]), file tools work the host copy and
//!    stay confined to it
//! 4. remove the container, copy the workspace to a `graded/` tree, overlay
//!    `hidden/` (stomping any agent edits to grader files), and run each
//!    check in a fresh container
//!
//! Oracle mode (`bench verify`) grades `workspace + solution + hidden`
//! (must pass) and `workspace + hidden` (must fail) instead — proving a task
//! is solvable and actually discriminates before any model burns time on it.

use chrono::Utc;
use localcode_agent::{AgentEvent, AgentSession, CodingAgent, ContainerShell};
use localcode_core::config::{AgentConfig, ApprovalMode};
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::runtime::ActiveRuntime;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::info;

use crate::docker;
use crate::fsops::copy_tree;
use crate::report::{
    summarize_tasks, tail, AgentStats, CheckReport, RunDir, RunEvent, RunSummary, TaskPhase,
    TaskReport, TaskStatus,
};
use crate::spec::{BenchSuite, BenchTask, CONTAINER_WORKSPACE};

/// Model⇄tool rounds a task may use unless its manifest says otherwise.
const DEFAULT_MAX_ROUNDS: u32 = 24;
/// Per-`bash`-call wall clock unless the manifest says otherwise.
const DEFAULT_EXEC_TIMEOUT_SECS: u64 = 120;
/// Kept output tail per grader check.
const CHECK_OUTPUT_TAIL: usize = 2_000;

const BENCH_SYSTEM_PROMPT: &str = "You are LocalCode's coding agent completing an evaluated \
benchmark task. bash runs inside a Linux task container whose working directory is the \
workspace; the file tools (read/write/edit/ls/grep) address the same files by \
workspace-relative path. The available toolchain depends on the task's image (python3, node, \
cargo, gcc, …). There is no network unless the task says otherwise. Work autonomously: read \
the task, inspect the workspace, implement the change, and verify it yourself (compile or run \
the code) before finishing. Hidden tests grade the final state of the workspace files — only \
the files matter, not your final message. When done, reply with a short summary of what you \
changed.";

/// Everything a suite run needs.
pub struct RunOptions {
    pub suite: BenchSuite,
    /// Where run directories are created (see [`crate::runs_root`]).
    pub runs_root: PathBuf,
    /// Endpoint + model the agent talks to. Ignored (may be None) in oracle mode.
    pub runtime: Option<ActiveRuntime>,
    /// The user's agent config; bench-critical fields are overridden per task
    /// (approvals, sandbox, budgets, system prompt) while tuning like
    /// `stream` and history budgets is kept.
    pub base_agent_config: AgentConfig,
    /// Verify tasks with their reference solutions instead of running a model.
    pub oracle: bool,
    /// Only run tasks whose id contains this string.
    pub task_filter: Option<String>,
}

/// Run a whole suite, emitting [`RunEvent`]s as it goes. The returned summary
/// is also written to `summary.json` in the run directory. Task-level
/// problems (agent errors, infra failures) become `Error` reports; only
/// run-level problems (no docker, no tasks, unwritable run dir) return `Err`.
pub async fn run_suite(
    opts: RunOptions,
    events: mpsc::UnboundedSender<RunEvent>,
) -> Result<RunSummary, LocalCodeError> {
    if !docker::docker_available() {
        return Err(docker::docker_missing_err());
    }

    let tasks: Vec<&BenchTask> = opts
        .suite
        .tasks
        .iter()
        .filter(|t| match &opts.task_filter {
            Some(f) => t.spec.id.contains(f.as_str()),
            None => true,
        })
        .collect();
    if tasks.is_empty() {
        return Err(LocalCodeError::new(
            ErrorCode::ConfigLoadFailed,
            match &opts.task_filter {
                Some(f) => format!("No tasks in '{}' match '{f}'", opts.suite.meta.id),
                None => format!("Suite '{}' has no tasks", opts.suite.meta.id),
            },
        ));
    }

    let runtime = if opts.oracle {
        None
    } else {
        match &opts.runtime {
            Some(r) => Some(r.clone()),
            None => {
                return Err(LocalCodeError::new(
                    ErrorCode::BackendNotReady,
                    "A benchmark run needs a model runtime",
                )
                .with_hint("Deploy a model first, or pass --base-url to `localcode bench run`"))
            }
        }
    };

    let model = match (&runtime, opts.oracle) {
        (_, true) => "oracle".to_string(),
        (Some(r), _) => r.model_id.clone().unwrap_or_else(|| r.name.clone()),
        (None, false) => unreachable!("checked above"),
    };
    let runtime_name = runtime.as_ref().map(|r| r.name.clone()).unwrap_or_default();

    let run_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
    let started = Utc::now();
    let dir_name = format!(
        "{}-{}-{}",
        started.format("%Y%m%d-%H%M%S"),
        slug(&opts.suite.meta.id),
        slug(&model)
    );
    let run_dir = RunDir::create(&opts.runs_root, &dir_name)?;

    // Sweep containers a crashed/cancelled earlier run left behind. Safe:
    // the app runs one bench at a time.
    docker::cleanup_stale_containers().await;

    info!(run_id, suite = %opts.suite.meta.id, model, oracle = opts.oracle, "bench run start");
    let _ = events.send(RunEvent::RunStarted {
        run_id: run_id.clone(),
        suite_id: opts.suite.meta.id.clone(),
        total: tasks.len(),
    });

    let mut reports: Vec<TaskReport> = Vec::new();
    for (i, task) in tasks.iter().enumerate() {
        let _ = events.send(RunEvent::TaskStarted {
            index: i,
            total: tasks.len(),
            task_id: task.spec.id.clone(),
            title: task.spec.title.clone(),
        });
        let report = if opts.oracle {
            run_oracle_task(task, &run_dir, &run_id, &events).await
        } else {
            run_agent_task(
                task,
                runtime.as_ref().expect("runtime present in agent mode"),
                &opts.base_agent_config,
                &run_dir,
                &run_id,
                i,
                &events,
            )
            .await
        };
        if let Err(e) = run_dir.append_task(&report) {
            tracing::warn!(error = %e.message, "could not append task report");
        }
        let _ = events.send(RunEvent::TaskFinished {
            task_id: task.spec.id.clone(),
            report: report.clone(),
        });
        reports.push(report);
    }

    let (score, strict, passed, failed, errored) = summarize_tasks(&reports);
    let summary = RunSummary {
        run_id,
        suite_id: opts.suite.meta.id.clone(),
        suite_version: opts.suite.meta.version.clone(),
        oracle: opts.oracle,
        model,
        runtime_name,
        tasks_total: reports.len(),
        tasks_passed: passed,
        tasks_failed: failed,
        tasks_errored: errored,
        score,
        strict_pass_rate: strict,
        started_at: started.to_rfc3339(),
        finished_at: Utc::now().to_rfc3339(),
        results_dir: run_dir.root.display().to_string(),
        harness_version: env!("CARGO_PKG_VERSION").into(),
    };
    if let Err(e) = run_dir.write_summary(&summary) {
        tracing::warn!(error = %e.message, "could not write run summary");
    }
    let _ = events.send(RunEvent::RunFinished {
        summary: summary.clone(),
    });
    Ok(summary)
}

/// Directory-name-safe slug.
fn slug(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect();
    out.truncate(48);
    if out.is_empty() {
        out.push('x');
    }
    out
}

/// Counters accumulated from the live event stream during the agent phase.
#[derive(Default)]
struct TurnCounters {
    rounds: u32,
    tool_calls: u32,
    tool_errors: u32,
}

fn now_rfc3339() -> String {
    Utc::now().to_rfc3339()
}

fn error_report(task: &BenchTask, started_at: String, msg: String) -> TaskReport {
    TaskReport {
        task_id: task.spec.id.clone(),
        title: task.spec.title.clone(),
        domain: task.spec.domain.clone(),
        tier: task.spec.tier,
        weight: task.spec.weight,
        status: TaskStatus::Error,
        score: 0.0,
        checks: vec![],
        agent: None,
        error: Some(msg),
        started_at,
        finished_at: now_rfc3339(),
    }
}

/// Make sure `image` exists locally, pulling it if needed.
async fn ensure_image(
    task: &BenchTask,
    events: &mpsc::UnboundedSender<RunEvent>,
) -> Result<(), LocalCodeError> {
    if docker::image_present(&task.spec.image).await {
        return Ok(());
    }
    let _ = events.send(RunEvent::TaskPhase {
        task_id: task.spec.id.clone(),
        phase: TaskPhase::PullImage,
    });
    docker::pull_image(&task.spec.image).await
}

async fn run_agent_task(
    task: &BenchTask,
    runtime: &ActiveRuntime,
    base_cfg: &AgentConfig,
    run_dir: &RunDir,
    run_id: &str,
    index: usize,
    events: &mpsc::UnboundedSender<RunEvent>,
) -> TaskReport {
    let started_at = now_rfc3339();
    let task_dir = run_dir.task_dir(&task.spec.id);
    let ws = task_dir.join("workspace");

    if let Err(e) = copy_tree(&task.workspace_dir(), &ws) {
        return error_report(task, started_at, format!("workspace copy failed: {}", e.message));
    }
    if let Err(e) = ensure_image(task, events).await {
        return error_report(task, started_at, format!("image unavailable: {}", e.message));
    }

    // -- agent phase -------------------------------------------------------
    let container = format!("lcb-{run_id}-{index}");
    let _ = events.send(RunEvent::TaskPhase {
        task_id: task.spec.id.clone(),
        phase: TaskPhase::StartContainer,
    });
    if let Err(e) =
        docker::start_task_container(&container, &task.spec.image, &ws, &task.spec.network, run_id)
            .await
    {
        return error_report(task, started_at, format!("container start failed: {}", e.message));
    }

    let _ = events.send(RunEvent::TaskPhase {
        task_id: task.spec.id.clone(),
        phase: TaskPhase::Agent,
    });
    let agent_started = Instant::now();
    let stats = run_agent_phase(task, runtime, base_cfg, &ws, &container, events).await;
    let wall_ms = agent_started.elapsed().as_millis() as u64;

    // The agent container is done — grading runs in fresh ones.
    docker::remove_container(&container).await;

    let agent_stats = AgentStats {
        wall_ms,
        rounds: stats.counters.rounds,
        tool_calls: stats.counters.tool_calls,
        tool_errors: stats.counters.tool_errors,
        timed_out: stats.timed_out,
        error: stats.error,
        final_message: tail(&stats.final_message, 400),
    };

    // -- grading -----------------------------------------------------------
    let _ = events.send(RunEvent::TaskPhase {
        task_id: task.spec.id.clone(),
        phase: TaskPhase::Grade,
    });
    let graded = task_dir.join("graded");
    if let Err(e) = assemble_tree(&graded, &[&ws, &task.hidden_dir()]) {
        let mut r = error_report(task, started_at, format!("grade tree failed: {}", e.message));
        r.agent = Some(agent_stats);
        return r;
    }
    let checks = match grade_tree(task, &graded, run_id).await {
        Ok(c) => c,
        Err(e) => {
            let mut r = error_report(task, started_at, format!("grading failed: {}", e.message));
            r.agent = Some(agent_stats);
            return r;
        }
    };

    let (score, all_passed) = score_checks(&checks);
    TaskReport {
        task_id: task.spec.id.clone(),
        title: task.spec.title.clone(),
        domain: task.spec.domain.clone(),
        tier: task.spec.tier,
        weight: task.spec.weight,
        status: if all_passed {
            TaskStatus::Passed
        } else {
            TaskStatus::Failed
        },
        score,
        checks,
        agent: Some(agent_stats),
        error: None,
        started_at,
        finished_at: now_rfc3339(),
    }
}

struct AgentPhaseOutcome {
    counters: TurnCounters,
    timed_out: bool,
    error: Option<String>,
    final_message: String,
}

/// One agent turn against `runtime`, bash routed into `container`, bounded by
/// the task wall clock. Never fails — outcomes land in the returned struct
/// and the workspace is graded regardless.
async fn run_agent_phase(
    task: &BenchTask,
    runtime: &ActiveRuntime,
    base_cfg: &AgentConfig,
    ws: &Path,
    container: &str,
    events: &mpsc::UnboundedSender<RunEvent>,
) -> AgentPhaseOutcome {
    let cfg = bench_agent_config(base_cfg, task, ws);
    let agent = CodingAgent::new(cfg).with_container_shell(ContainerShell {
        container: container.to_string(),
        mount_root: CONTAINER_WORKSPACE.to_string(),
    });
    let mut session = AgentSession::new(ws.to_path_buf());

    // Forward live tool activity as compact lines; count rounds/calls/errors.
    let counters = Arc::new(Mutex::new(TurnCounters::default()));
    let (ev_tx, mut ev_rx) = mpsc::unbounded_channel::<AgentEvent>();
    let fwd_counters = counters.clone();
    let fwd_events = events.clone();
    let fwd_task_id = task.spec.id.clone();
    let forwarder = tokio::spawn(async move {
        while let Some(ev) = ev_rx.recv().await {
            let line = match &ev {
                AgentEvent::MessageComplete => {
                    if let Ok(mut c) = fwd_counters.lock() {
                        c.rounds += 1;
                    }
                    None
                }
                AgentEvent::ToolStarted { name, args_preview } => {
                    if let Ok(mut c) = fwd_counters.lock() {
                        c.tool_calls += 1;
                    }
                    Some(format!("▶ {name} {args_preview}"))
                }
                AgentEvent::ToolFinished { name, ok, summary, .. } => {
                    if !ok {
                        if let Ok(mut c) = fwd_counters.lock() {
                            c.tool_errors += 1;
                        }
                    }
                    Some(format!("{} {name} — {summary}", if *ok { "✓" } else { "✗" }))
                }
                AgentEvent::Delta(_) | AgentEvent::ThinkingDelta(_) => None,
            };
            if let Some(line) = line {
                let _ = fwd_events.send(RunEvent::AgentActivity {
                    task_id: fwd_task_id.clone(),
                    line,
                });
            }
        }
    });

    let api_key = runtime.api_key.clone();
    let turn = agent.run_turn(
        &mut session,
        &task.spec.prompt,
        runtime,
        api_key.as_deref(),
        None, // AlwaysApprove: nothing gates; the container is the sandbox
        Some(&ev_tx),
    );
    let (timed_out, error, final_message) =
        match tokio::time::timeout(Duration::from_secs(task.spec.timeout_secs), turn).await {
            Ok(Ok(text)) => (false, None, text),
            Ok(Err(e)) => (false, Some(format!("{}: {}", e.code, e.message)), String::new()),
            Err(_) => (true, None, String::new()),
        };
    drop(ev_tx);
    let _ = forwarder.await;

    let counters = Arc::try_unwrap(counters)
        .map(|m| m.into_inner().unwrap_or_default())
        .unwrap_or_default();
    AgentPhaseOutcome {
        counters,
        timed_out,
        error,
        final_message,
    }
}

/// The agent config for one bench task: the user's tuning (stream, history
/// budgets) with the bench-critical fields pinned.
fn bench_agent_config(base: &AgentConfig, task: &BenchTask, ws: &Path) -> AgentConfig {
    let mut cfg = base.clone();
    // Unattended inside the sandbox: nothing to gate on.
    cfg.approval_mode = ApprovalMode::AlwaysApprove;
    cfg.confirm_destructive_tools = false;
    // The container is the confinement; the host text checks would reject
    // legitimate container paths.
    cfg.shell_sandbox = false;
    cfg.bash_timeout_secs = task
        .spec
        .exec_timeout_secs
        .unwrap_or(DEFAULT_EXEC_TIMEOUT_SECS);
    cfg.max_tool_rounds = task.spec.agent_max_rounds.unwrap_or(DEFAULT_MAX_ROUNDS);
    cfg.system_prompt = Some(BENCH_SYSTEM_PROMPT.into());
    // No user skills: results must not depend on what's installed locally.
    cfg.skills_dir = Some(ws.join(".no-skills").display().to_string());
    if !cfg.disabled_tools.iter().any(|t| t == "skill") {
        cfg.disabled_tools.push("skill".into());
    }
    cfg
}

/// Build `dst` fresh from ordered overlay `layers` (later layers overwrite
/// earlier ones; missing layers are skipped). Grading uses
/// `[workspace, hidden]` — the hidden overlay comes last, so any agent edit
/// to a grader-owned file is overwritten and tampering with tests can't
/// survive into grading. Oracle mode adds the solution layer in between.
fn assemble_tree(dst: &Path, layers: &[&Path]) -> Result<(), LocalCodeError> {
    crate::fsops::remove_tree_best_effort(dst);
    for layer in layers {
        if layer.is_dir() {
            copy_tree(layer, dst)?;
        }
    }
    Ok(())
}

/// Run every grader check over `tree` in fresh containers.
async fn grade_tree(
    task: &BenchTask,
    tree: &Path,
    run_id: &str,
) -> Result<Vec<CheckReport>, LocalCodeError> {
    let mut out = Vec::new();
    for check in &task.spec.grader.checks {
        let outcome = docker::run_check_container(
            &task.spec.image,
            tree,
            &task.spec.network,
            &check.cmd,
            Duration::from_secs(task.spec.grader.timeout_secs),
            run_id,
        )
        .await?;
        out.push(CheckReport {
            name: check.name.clone(),
            passed: outcome.exit_code == 0 && !outcome.timed_out,
            weight: check.weight,
            exit_code: outcome.exit_code,
            output_tail: tail(&outcome.output, CHECK_OUTPUT_TAIL),
        });
    }
    Ok(out)
}

/// `(weighted score, all passed)`.
fn score_checks(checks: &[CheckReport]) -> (f64, bool) {
    let total: f64 = checks.iter().map(|c| c.weight).sum::<f64>().max(1e-9);
    let passed: f64 = checks.iter().filter(|c| c.passed).map(|c| c.weight).sum();
    let all = !checks.is_empty() && checks.iter().all(|c| c.passed);
    (passed / total, all)
}

/// Oracle verification: the reference solution must pass, the unmodified
/// workspace must fail. Proves the task is both solvable and discriminating.
async fn run_oracle_task(
    task: &BenchTask,
    run_dir: &RunDir,
    run_id: &str,
    events: &mpsc::UnboundedSender<RunEvent>,
) -> TaskReport {
    let started_at = now_rfc3339();
    if !task.has_solution() {
        return error_report(
            task,
            started_at,
            "no solution/ directory — nothing to verify against".into(),
        );
    }
    if let Err(e) = ensure_image(task, events).await {
        return error_report(task, started_at, format!("image unavailable: {}", e.message));
    }
    let _ = events.send(RunEvent::TaskPhase {
        task_id: task.spec.id.clone(),
        phase: TaskPhase::Grade,
    });

    let task_dir = run_dir.task_dir(&task.spec.id);

    // Tree 1: workspace + solution + hidden → every check must pass.
    let sol_tree = task_dir.join("oracle-solution");
    if let Err(e) = assemble_tree(
        &sol_tree,
        &[&task.workspace_dir(), &task.solution_dir(), &task.hidden_dir()],
    ) {
        return error_report(task, started_at, format!("oracle tree failed: {}", e.message));
    }
    let sol_checks = match grade_tree(task, &sol_tree, run_id).await {
        Ok(c) => c,
        Err(e) => {
            return error_report(task, started_at, format!("grading failed: {}", e.message))
        }
    };
    let (_, sol_all) = score_checks(&sol_checks);

    // Tree 2: workspace + hidden (no solution) → at least one check must fail,
    // or the task can't tell a working model from a no-op.
    let base_tree = task_dir.join("oracle-base");
    if let Err(e) = assemble_tree(&base_tree, &[&task.workspace_dir(), &task.hidden_dir()]) {
        return error_report(task, started_at, format!("oracle tree failed: {}", e.message));
    }
    let base_checks = match grade_tree(task, &base_tree, run_id).await {
        Ok(c) => c,
        Err(e) => {
            return error_report(task, started_at, format!("grading failed: {}", e.message))
        }
    };
    let (_, base_all) = score_checks(&base_checks);

    let mut problems = Vec::new();
    if !sol_all {
        let failing: Vec<&str> = sol_checks
            .iter()
            .filter(|c| !c.passed)
            .map(|c| c.name.as_str())
            .collect();
        problems.push(format!("reference solution FAILS checks: {}", failing.join(", ")));
    }
    if base_all {
        problems.push(
            "unmodified workspace already PASSES every check — the task doesn't discriminate"
                .into(),
        );
    }

    let ok = problems.is_empty();
    TaskReport {
        task_id: task.spec.id.clone(),
        title: task.spec.title.clone(),
        domain: task.spec.domain.clone(),
        tier: task.spec.tier,
        weight: task.spec.weight,
        status: if ok { TaskStatus::Passed } else { TaskStatus::Failed },
        score: if ok { 1.0 } else { 0.0 },
        checks: sol_checks,
        agent: None,
        error: if ok { None } else { Some(problems.join("; ")) },
        started_at,
        finished_at: now_rfc3339(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::TaskStatus;

    fn check(name: &str, passed: bool, weight: f64) -> CheckReport {
        CheckReport {
            name: name.into(),
            passed,
            weight,
            exit_code: if passed { 0 } else { 1 },
            output_tail: String::new(),
        }
    }

    #[test]
    fn check_scoring_weights_and_strictness() {
        let (score, all) = score_checks(&[check("a", true, 3.0), check("b", false, 1.0)]);
        assert!((score - 0.75).abs() < 1e-9);
        assert!(!all);
        let (score, all) = score_checks(&[check("a", true, 1.0)]);
        assert!((score - 1.0).abs() < 1e-9 && all);
        // No checks: never "all passed".
        let (score, all) = score_checks(&[]);
        assert_eq!(score, 0.0);
        assert!(!all);
    }

    #[test]
    fn graded_tree_overlays_hidden_over_agent_edits() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().join("ws");
        let hidden = dir.path().join("hidden");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&hidden).unwrap();
        std::fs::write(ws.join("code.py"), "agent code").unwrap();
        // Agent tried to cheat by writing its own test file.
        std::fs::write(ws.join("test_x.py"), "assert True").unwrap();
        std::fs::write(hidden.join("test_x.py"), "real assertions").unwrap();

        let graded = dir.path().join("graded");
        assemble_tree(&graded, &[&ws, &hidden]).unwrap();
        assert_eq!(std::fs::read_to_string(graded.join("code.py")).unwrap(), "agent code");
        assert_eq!(
            std::fs::read_to_string(graded.join("test_x.py")).unwrap(),
            "real assertions"
        );
        // Re-assembly replaces a stale graded tree wholesale.
        std::fs::write(ws.join("new.py"), "later").unwrap();
        assemble_tree(&graded, &[&ws, &hidden]).unwrap();
        assert!(graded.join("new.py").exists());
        assert_eq!(
            std::fs::read_to_string(graded.join("test_x.py")).unwrap(),
            "real assertions"
        );
    }

    #[test]
    fn oracle_layers_stack_solution_between_workspace_and_hidden() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().join("ws");
        let sol = dir.path().join("sol");
        let hidden = dir.path().join("hidden");
        for d in [&ws, &sol, &hidden] {
            std::fs::create_dir_all(d).unwrap();
        }
        std::fs::write(ws.join("impl.py"), "stub").unwrap();
        std::fs::write(sol.join("impl.py"), "correct").unwrap();
        std::fs::write(hidden.join("test.py"), "tests").unwrap();
        let out = dir.path().join("out");
        assemble_tree(&out, &[&ws, &sol, &hidden]).unwrap();
        assert_eq!(std::fs::read_to_string(out.join("impl.py")).unwrap(), "correct");
        assert!(out.join("test.py").exists());
        // Missing layers are skipped, not errors.
        assemble_tree(&out, &[&ws, &dir.path().join("nope"), &hidden]).unwrap();
        assert_eq!(std::fs::read_to_string(out.join("impl.py")).unwrap(), "stub");
    }

    #[test]
    fn bench_config_pins_safety_fields_but_keeps_tuning() {
        let base = AgentConfig {
            stream: false,
            max_history_chars: 12345,
            approval_mode: ApprovalMode::AskPermission,
            shell_sandbox: true,
            ..AgentConfig::default()
        };
        let dir = tempfile::tempdir().unwrap();
        let task = BenchTask {
            spec: toml::from_str(
                r#"
title = "t"
prompt = "p"
image = "i"
agent_max_rounds = 7
exec_timeout_secs = 33
[[grader.checks]]
name = "c"
cmd = "true"
"#,
            )
            .unwrap(),
            dir: dir.path().to_path_buf(),
        };
        let cfg = bench_agent_config(&base, &task, dir.path());
        assert_eq!(cfg.approval_mode, ApprovalMode::AlwaysApprove);
        assert!(!cfg.shell_sandbox);
        assert_eq!(cfg.max_tool_rounds, 7);
        assert_eq!(cfg.bash_timeout_secs, 33);
        assert!(cfg.disabled_tools.iter().any(|t| t == "skill"));
        assert!(cfg.system_prompt.as_deref().unwrap().contains("benchmark"));
        // User tuning survives.
        assert!(!cfg.stream);
        assert_eq!(cfg.max_history_chars, 12345);
        assert_eq!(TaskStatus::Passed, TaskStatus::Passed);
    }

    #[test]
    fn slug_is_fs_safe() {
        assert_eq!(slug("Qwen/Qwen2.5-Coder:Q4_K_M"), "Qwen-Qwen2.5-Coder-Q4_K_M");
        assert_eq!(slug(""), "x");
    }
}
