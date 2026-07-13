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
            status: RuntimeStatus::Starting,
            created_at: Utc::now(),
            correlation_id: Uuid::new_v4().to_string(),
        }
    }
}
