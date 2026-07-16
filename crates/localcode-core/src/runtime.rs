use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeKind {
    Ollama,
    LlamaCpp,
    Vllm,
    Sglang,
    Colibri,
    ColibriHy3,
    CloudRunpod,
    CloudVast,
    CloudAkash,
    OpenAiCompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStatus {
    Starting,
    Healthy,
    Unhealthy,
    Stopping,
    Stopped,
}

/// A live local or cloud inference endpoint registered for Coding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveRuntime {
    pub id: Uuid,
    pub name: String,
    pub kind: RuntimeKind,
    pub model_id: Option<String>,
    pub quantization: Option<String>,
    pub base_url: String,
    pub api_key: Option<String>,
    /// Model context window in tokens, when known (llama.cpp `-c` / vLLM
    /// `--max-model-len` / Ollama `num_ctx` / the assistant's `local_context`).
    /// Drives the agent's auto-compaction budget so history is summarized before
    /// it overflows this model's context. `None` when the backend didn't report
    /// a fixed window (the agent then falls back to `agent.max_history_chars`).
    #[serde(default)]
    pub context_tokens: Option<u32>,
    pub status: RuntimeStatus,
    pub created_at: DateTime<Utc>,
    pub correlation_id: String,
}

impl ActiveRuntime {
    pub fn new(
        name: impl Into<String>,
        kind: RuntimeKind,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            kind,
            model_id: None,
            quantization: None,
            base_url: base_url.into(),
            api_key: None,
            context_tokens: None,
            status: RuntimeStatus::Starting,
            created_at: Utc::now(),
            correlation_id: Uuid::new_v4().to_string(),
        }
    }
}
