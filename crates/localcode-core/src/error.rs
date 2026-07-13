use crate::ids::CorrelationId;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fmt;
use thiserror::Error;

/// Stable machine-readable error codes (see ERROR_CODES.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    // Config
    ConfigLoadFailed,
    ConfigParseFailed,
    ConfigSaveFailed,
    // Network / HF
    HfUnreachable,
    HfRateLimited,
    HfAuthRequired,
    HfMirrorFailed,
    HfModelNotFound,
    // GPU
    GpuDetectFailed,
    GpuNoDevices,
    // Backends
    BackendNotFound,
    BackendNotReady,
    BackendPortInUse,
    BackendStartFailed,
    BackendHealthTimeout,
    BackendBinaryMissing,
    DeployDiskLow,
    DeployDownloadFailed,
    DeployOversizedWarning,
    // Agent
    AgentToolFailed,
    AgentWorkspaceMissing,
    AgentMcpFailed,
    // Cloud
    CloudKeyMissing,
    CloudProvisionFailed,
    CloudQuotaExceeded,
    CloudProviderUnavailable,
    // Payments
    PaymentConfirmRequired,
    InsufficientBalance,
    DepositFailed,
    // API / auth
    ApiUnreachable,
    AuthRequired,
    AuthFailed,
    // Self-update
    UpdateCheckFailed,
    UpdateFailed,
    // Generic
    IoError,
    Internal,
    Cancelled,
    NotImplemented,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ConfigLoadFailed => "CONFIG_LOAD_FAILED",
            Self::ConfigParseFailed => "CONFIG_PARSE_FAILED",
            Self::ConfigSaveFailed => "CONFIG_SAVE_FAILED",
            Self::HfUnreachable => "HF_UNREACHABLE",
            Self::HfRateLimited => "HF_RATE_LIMITED",
            Self::HfAuthRequired => "HF_AUTH_REQUIRED",
            Self::HfMirrorFailed => "HF_MIRROR_FAILED",
            Self::HfModelNotFound => "HF_MODEL_NOT_FOUND",
            Self::GpuDetectFailed => "GPU_DETECT_FAILED",
            Self::GpuNoDevices => "GPU_NO_DEVICES",
            Self::BackendNotFound => "BACKEND_NOT_FOUND",
            Self::BackendNotReady => "BACKEND_NOT_READY",
            Self::BackendPortInUse => "BACKEND_PORT_IN_USE",
            Self::BackendStartFailed => "BACKEND_START_FAILED",
            Self::BackendHealthTimeout => "BACKEND_HEALTH_TIMEOUT",
            Self::BackendBinaryMissing => "BACKEND_BINARY_MISSING",
            Self::DeployDiskLow => "DEPLOY_DISK_LOW",
            Self::DeployDownloadFailed => "DEPLOY_DOWNLOAD_FAILED",
            Self::DeployOversizedWarning => "DEPLOY_OVERSIZED_WARNING",
            Self::AgentToolFailed => "AGENT_TOOL_FAILED",
            Self::AgentWorkspaceMissing => "AGENT_WORKSPACE_MISSING",
            Self::AgentMcpFailed => "AGENT_MCP_FAILED",
            Self::CloudKeyMissing => "CLOUD_KEY_MISSING",
            Self::CloudProvisionFailed => "CLOUD_PROVISION_FAILED",
            Self::CloudQuotaExceeded => "CLOUD_QUOTA_EXCEEDED",
            Self::CloudProviderUnavailable => "CLOUD_PROVIDER_UNAVAILABLE",
            Self::PaymentConfirmRequired => "PAYMENT_CONFIRM_REQUIRED",
            Self::InsufficientBalance => "INSUFFICIENT_BALANCE",
            Self::DepositFailed => "DEPOSIT_FAILED",
            Self::ApiUnreachable => "API_UNREACHABLE",
            Self::AuthRequired => "AUTH_REQUIRED",
            Self::AuthFailed => "AUTH_FAILED",
            Self::UpdateCheckFailed => "UPDATE_CHECK_FAILED",
            Self::UpdateFailed => "UPDATE_FAILED",
            Self::IoError => "IO_ERROR",
            Self::Internal => "INTERNAL",
            Self::Cancelled => "CANCELLED",
            Self::NotImplemented => "NOT_IMPLEMENTED",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppErrorSource {
    pub component: String,
    pub operation: String,
}

/// Structured user-facing error with recovery hints.
#[derive(Debug, Clone, Serialize, Deserialize, Error)]
#[error("{code}: {message}")]
pub struct LocalCodeError {
    pub code: ErrorCode,
    pub message: String,
    pub causes: Vec<String>,
    pub hints: Vec<String>,
    pub correlation_id: CorrelationId,
    pub retryable: bool,
    pub origin: Option<AppErrorSource>,
    pub details: Value,
}

impl LocalCodeError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            causes: Vec::new(),
            hints: Vec::new(),
            correlation_id: CorrelationId::new(),
            retryable: false,
            origin: None,
            details: json!({}),
        }
    }

    pub fn with_cause(mut self, cause: impl Into<String>) -> Self {
        self.causes.push(cause.into());
        self
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hints.push(hint.into());
        self
    }

    pub fn retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    pub fn with_correlation(mut self, id: CorrelationId) -> Self {
        self.correlation_id = id;
        self
    }

    pub fn with_source(mut self, component: impl Into<String>, operation: impl Into<String>) -> Self {
        self.origin = Some(AppErrorSource {
            component: component.into(),
            operation: operation.into(),
        });
        self
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }

    /// Context pack for the in-app assistant.
    pub fn assistant_context(&self) -> Value {
        json!({
            "code": self.code.as_str(),
            "message": self.message,
            "causes": self.causes,
            "hints": self.hints,
            "correlation_id": self.correlation_id.to_string(),
            "retryable": self.retryable,
            "origin": self.origin,
            "details": self.details,
        })
    }
}

impl From<std::io::Error> for LocalCodeError {
    fn from(err: std::io::Error) -> Self {
        LocalCodeError::new(ErrorCode::IoError, err.to_string())
            .with_cause("Filesystem or process I/O failed")
            .with_hint("Check permissions and free disk space")
            .retryable(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_code_display() {
        assert_eq!(ErrorCode::BackendPortInUse.as_str(), "BACKEND_PORT_IN_USE");
    }

    #[test]
    fn assistant_context_has_code() {
        let e = LocalCodeError::new(ErrorCode::HfUnreachable, "down")
            .with_cause("DNS")
            .with_hint("Check network");
        let ctx = e.assistant_context();
        assert_eq!(ctx["code"], "HF_UNREACHABLE");
        assert_eq!(ctx["causes"][0], "DNS");
    }
}
