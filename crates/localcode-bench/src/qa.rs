//! Legacy prompt-level QA bench: one chat completion per task, graded by
//! regex / substring checks on the reply. Kept for compatibility (publish
//! payload shape, quick endpoint smoke checks). The real agentic benchmark —
//! sandboxed Docker tasks graded by hidden tests — lives in [`crate::runner`].

use chrono::Utc;
use localcode_api_client::ApiClient;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::events::{AppEvent, EventBus};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::info;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suite {
    pub id: String,
    pub version: String,
    pub title: String,
    pub description: String,
    pub tasks: Vec<Task>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub prompt: String,
    #[serde(default)]
    pub expect_regex: Option<String>,
    #[serde(default)]
    pub expect_contains: Option<String>,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "default_weight")]
    pub weight: f64,
}

fn default_timeout() -> u64 {
    60
}
fn default_weight() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subject {
    pub hf_model_id: String,
    pub quantization: String,
    pub weight_source: String,
    pub backend: String,
    pub backend_version: String,
    pub precision_notes: String,
    pub hardware: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub passed: bool,
    pub latency_ms: u64,
    pub output: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub id: String,
    pub suite_id: String,
    pub suite_version: String,
    pub subject: Subject,
    pub tasks: Vec<TaskResult>,
    pub metrics: Metrics,
    pub started_at: String,
    pub finished_at: String,
    pub runner_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub score: f64,
    pub pass_rate: f64,
    pub latency_p50_ms: u64,
    pub latency_p95_ms: u64,
    pub tokens_per_sec: Option<f64>,
}

pub fn sample_coding_suite() -> Suite {
    Suite {
        id: "localcode-sample-coding".into(),
        version: "1.0.0".into(),
        title: "Sample Coding Suite".into(),
        description: "Tiny coding tasks for smoke benchmarks".into(),
        tasks: vec![
            Task {
                id: "hello_fn".into(),
                prompt: "Write a Python function `add(a, b)` that returns the sum of a and b. Reply with code only.".into(),
                expect_regex: Some(r"def\s+add\s*\(".into()),
                expect_contains: Some("return".into()),
                timeout_secs: 60,
                weight: 1.0,
            },
            Task {
                id: "fizzbuzz".into(),
                prompt: "Write a Python function fizzbuzz(n) that returns 'Fizz' if n divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if both, else str(n).".into(),
                expect_regex: Some(r"def\s+fizzbuzz".into()),
                expect_contains: None,
                timeout_secs: 60,
                weight: 1.0,
            },
        ],
    }
}

pub fn load_suite(path: &Path) -> Result<Suite, LocalCodeError> {
    let raw = std::fs::read_to_string(path)?;
    if path.extension().and_then(|e| e.to_str()) == Some("toml") {
        toml::from_str(&raw).map_err(|e| {
            LocalCodeError::new(ErrorCode::ConfigParseFailed, e.to_string())
                .with_cause("Invalid suite.toml")
        })
    } else {
        serde_json::from_str(&raw).map_err(|e| {
            LocalCodeError::new(ErrorCode::ConfigParseFailed, e.to_string())
                .with_cause("Invalid suite JSON")
        })
    }
}

pub fn save_suite(suite: &Suite, dir: &Path) -> Result<PathBuf, LocalCodeError> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("suite.toml");
    let raw = toml::to_string_pretty(suite).map_err(|e| {
        LocalCodeError::new(ErrorCode::ConfigSaveFailed, e.to_string())
    })?;
    std::fs::write(&path, raw)?;
    Ok(path)
}

pub struct BenchRunner {
    events: EventBus,
    http: reqwest::Client,
}

impl BenchRunner {
    pub fn new(events: EventBus) -> Self {
        Self {
            events,
            http: reqwest::Client::new(),
        }
    }

    /// Run suite against an OpenAI-compatible chat completions endpoint.
    pub async fn run(
        &self,
        suite: &Suite,
        subject: Subject,
        endpoint_base: &str,
        api_key: Option<&str>,
        model_name: &str,
    ) -> Result<RunResult, LocalCodeError> {
        let run_id = Uuid::new_v4().to_string();
        let started = Utc::now();
        let total = suite.tasks.len() as u32;
        let mut results = Vec::new();

        info!(%run_id, suite = %suite.id, "bench run start");

        for (i, task) in suite.tasks.iter().enumerate() {
            self.events.publish(AppEvent::BenchProgress {
                run_id: run_id.clone(),
                completed: i as u32,
                total,
                message: format!("Running {}", task.id),
            });

            let t0 = Instant::now();
            let output = match self
                .complete(endpoint_base, api_key, model_name, &task.prompt, task.timeout_secs)
                .await
            {
                Ok(o) => o,
                Err(e) => {
                    results.push(TaskResult {
                        task_id: task.id.clone(),
                        passed: false,
                        latency_ms: t0.elapsed().as_millis() as u64,
                        output: String::new(),
                        message: e.message.clone(),
                    });
                    continue;
                }
            };
            let latency_ms = t0.elapsed().as_millis() as u64;
            let passed = evaluate_task(task, &output);
            results.push(TaskResult {
                task_id: task.id.clone(),
                passed,
                latency_ms,
                output,
                message: if passed { "ok".into() } else { "assertion failed".into() },
            });
        }

        self.events.publish(AppEvent::BenchProgress {
            run_id: run_id.clone(),
            completed: total,
            total,
            message: "Complete".into(),
        });

        let finished = Utc::now();
        let metrics = compute_metrics(&suite.tasks, &results);

        Ok(RunResult {
            id: run_id,
            suite_id: suite.id.clone(),
            suite_version: suite.version.clone(),
            subject,
            tasks: results,
            metrics,
            started_at: started.to_rfc3339(),
            finished_at: finished.to_rfc3339(),
            runner_version: env!("CARGO_PKG_VERSION").into(),
        })
    }

    async fn complete(
        &self,
        base: &str,
        api_key: Option<&str>,
        model: &str,
        prompt: &str,
        timeout_secs: u64,
    ) -> Result<String, LocalCodeError> {
        let base = base.trim_end_matches('/');
        let url = if base.ends_with("/v1") {
            format!("{base}/chat/completions")
        } else {
            format!("{base}/v1/chat/completions")
        };

        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1024,
        });

        let mut req = self
            .http
            .post(&url)
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .json(&body);
        if let Some(k) = api_key {
            req = req.bearer_auth(k);
        }

        let resp = req.send().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::BackendNotReady, e.to_string())
                .with_cause("Inference endpoint unreachable")
                .with_hint("Deploy a model first and select it as runtime")
                .retryable(true)
        })?;

        if !resp.status().is_success() {
            let t = resp.text().await.unwrap_or_default();
            return Err(LocalCodeError::new(
                ErrorCode::AgentToolFailed,
                format!("completion failed: {t}"),
            ));
        }

        let v: serde_json::Value = resp.json().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::Internal, e.to_string())
        })?;
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        Ok(content)
    }
}

fn evaluate_task(task: &Task, output: &str) -> bool {
    if let Some(re) = &task.expect_regex {
        if let Ok(r) = regex_lite_is_match(re, output) {
            if !r {
                return false;
            }
        }
    }
    if let Some(s) = &task.expect_contains {
        if !output.contains(s) {
            return false;
        }
    }
    true
}

fn regex_lite_is_match(pattern: &str, text: &str) -> Result<bool, ()> {
    // Use simple contains for overly complex patterns if regex crate not linked — we use regex via string search fallback
    match regex::Regex::new(pattern) {
        Ok(re) => Ok(re.is_match(text)),
        Err(_) => Ok(text.contains(pattern)),
    }
}

// Prefer real regex
mod regex {
    pub use ::regex::*;
}

fn compute_metrics(tasks: &[Task], results: &[TaskResult]) -> Metrics {
    let total_w: f64 = tasks.iter().map(|t| t.weight).sum::<f64>().max(1e-9);
    let mut score = 0.0;
    for t in tasks {
        if results.iter().any(|r| r.task_id == t.id && r.passed) {
            score += t.weight;
        }
    }
    let pass_rate = if results.is_empty() {
        0.0
    } else {
        results.iter().filter(|r| r.passed).count() as f64 / results.len() as f64
    };
    let mut latencies: Vec<u64> = results.iter().map(|r| r.latency_ms).collect();
    latencies.sort_unstable();
    let p50 = percentile(&latencies, 0.50);
    let p95 = percentile(&latencies, 0.95);
    Metrics {
        score: score / total_w,
        pass_rate,
        latency_p50_ms: p50,
        latency_p95_ms: p95,
        tokens_per_sec: None,
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Publish payload — rejects if required fields missing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishPayload {
    pub hf_model_id: String,
    pub quantization: String,
    pub weight_source: String,
    pub backend: String,
    pub backend_version: String,
    pub precision_notes: String,
    pub hardware: serde_json::Value,
    pub suite_id: String,
    pub suite_version: String,
    pub metrics: Metrics,
    pub started_at: String,
    pub finished_at: String,
    pub runner_version: String,
}

impl PublishPayload {
    pub fn from_run(run: &RunResult) -> Result<Self, LocalCodeError> {
        if run.subject.hf_model_id.is_empty() {
            return Err(LocalCodeError::new(
                ErrorCode::ConfigParseFailed,
                "Publish requires hf_model_id",
            ));
        }
        if run.subject.quantization.is_empty() {
            return Err(LocalCodeError::new(
                ErrorCode::ConfigParseFailed,
                "Publish requires quantization",
            ));
        }
        Ok(Self {
            hf_model_id: run.subject.hf_model_id.clone(),
            quantization: run.subject.quantization.clone(),
            weight_source: run.subject.weight_source.clone(),
            backend: run.subject.backend.clone(),
            backend_version: run.subject.backend_version.clone(),
            precision_notes: run.subject.precision_notes.clone(),
            hardware: run.subject.hardware.clone(),
            suite_id: run.suite_id.clone(),
            suite_version: run.suite_version.clone(),
            metrics: run.metrics.clone(),
            started_at: run.started_at.clone(),
            finished_at: run.finished_at.clone(),
            runner_version: run.runner_version.clone(),
        })
    }
}

pub async fn publish_result(api: &ApiClient, run: &RunResult) -> Result<serde_json::Value, LocalCodeError> {
    let payload = PublishPayload::from_run(run)?;
    api.post_json("/v1/bench/results", &payload).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_sample() {
        let t = &sample_coding_suite().tasks[0];
        assert!(evaluate_task(t, "def add(a, b):\n  return a+b"));
        assert!(!evaluate_task(t, "print(1)"));
    }

    #[test]
    fn publish_requires_fields() {
        let mut run = RunResult {
            id: "1".into(),
            suite_id: "s".into(),
            suite_version: "1".into(),
            subject: Subject {
                hf_model_id: "".into(),
                quantization: "Q4".into(),
                weight_source: "x".into(),
                backend: "ollama".into(),
                backend_version: "1".into(),
                precision_notes: "".into(),
                hardware: serde_json::json!({}),
            },
            tasks: vec![],
            metrics: Metrics {
                score: 0.0,
                pass_rate: 0.0,
                latency_p50_ms: 0,
                latency_p95_ms: 0,
                tokens_per_sec: None,
            },
            started_at: "".into(),
            finished_at: "".into(),
            runner_version: "0.1.0".into(),
        };
        assert!(PublishPayload::from_run(&run).is_err());
        run.subject.hf_model_id = "org/model".into();
        assert!(PublishPayload::from_run(&run).is_ok());
    }
}
