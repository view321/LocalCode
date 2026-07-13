//! Cloud hosting orchestration (RunPod, Vast.ai, Akash).
//!
//! v1 status: the provider APIs are NOT wired up yet. Every adapter returns a
//! structured `NOT_IMPLEMENTED` error instead of pretending to provision —
//! an earlier draft fabricated "provisioning" deployments with made-up
//! endpoint URLs, which is worse than no support at all.

use async_trait::async_trait;
use localcode_core::config::Config;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::runtime::ActiveRuntime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloudProvider {
    Runpod,
    Vast,
    Akash,
}

impl CloudProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Runpod => "runpod",
            Self::Vast => "vast",
            Self::Akash => "akash",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudDeployRequest {
    pub provider: CloudProvider,
    pub model_id: String,
    pub quantization: Option<String>,
    pub gpu_filter: Option<String>,
    pub estimated_usd_per_hour: Option<f64>,
    pub confirmed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudDeployment {
    pub id: String,
    pub provider: CloudProvider,
    pub status: String,
    pub endpoint: Option<String>,
    pub cost_usd_per_hour: Option<f64>,
    pub runtime: Option<ActiveRuntime>,
    pub raw_error: Option<String>,
}

#[async_trait]
pub trait CloudAdapter: Send + Sync {
    fn provider(&self) -> CloudProvider;
    async fn deploy(&self, req: CloudDeployRequest) -> Result<CloudDeployment, LocalCodeError>;
    async fn destroy(&self, id: &str) -> Result<(), LocalCodeError>;
    async fn status(&self, id: &str) -> Result<CloudDeployment, LocalCodeError>;
}

pub struct CloudOrchestrator {
    adapters: Vec<Box<dyn CloudAdapter>>,
}

impl CloudOrchestrator {
    pub fn from_config(cfg: &Config) -> Self {
        let adapters: Vec<Box<dyn CloudAdapter>> = vec![
            Box::new(StubAdapter::new(
                CloudProvider::Runpod,
                env_key(&cfg.cloud.runpod.api_key_env, "RUNPOD_API_KEY"),
            )),
            Box::new(StubAdapter::new(
                CloudProvider::Vast,
                env_key(&cfg.cloud.vast.api_key_env, "VAST_API_KEY"),
            )),
            Box::new(StubAdapter::new(
                CloudProvider::Akash,
                env_key(&cfg.cloud.akash.api_key_env, "AKASH_API_KEY"),
            )),
        ];
        Self { adapters }
    }

    pub fn get(&self, provider: CloudProvider) -> Option<&dyn CloudAdapter> {
        self.adapters
            .iter()
            .find(|a| a.provider() == provider)
            .map(|a| a.as_ref())
    }

    pub async fn deploy(&self, req: CloudDeployRequest) -> Result<CloudDeployment, LocalCodeError> {
        let adapter = self.get(req.provider).ok_or_else(|| {
            LocalCodeError::new(
                ErrorCode::CloudProviderUnavailable,
                format!("Provider {:?} not registered", req.provider),
            )
        })?;
        adapter.deploy(req).await
    }
}

fn env_key(configured: &str, fallback: &str) -> Option<String> {
    let name = if configured.is_empty() {
        fallback
    } else {
        configured
    };
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

fn not_implemented(provider: CloudProvider) -> LocalCodeError {
    LocalCodeError::new(
        ErrorCode::NotImplemented,
        format!("{} cloud deploys are not implemented yet", provider.as_str()),
    )
    .with_cause("The provider adapter is a stub in this build")
    .with_hint("Provision manually in the provider console, then add the endpoint as an OpenAI-compatible runtime")
    .with_hint("Track progress in the LocalCode repository issues")
}

/// Honest placeholder adapter: refuses to deploy rather than fabricating a
/// deployment. Keeps the key plumbing so real adapters can slot in later.
struct StubAdapter {
    provider: CloudProvider,
    api_key: Option<String>,
}

impl StubAdapter {
    fn new(provider: CloudProvider, api_key: Option<String>) -> Self {
        Self { provider, api_key }
    }
}

#[async_trait]
impl CloudAdapter for StubAdapter {
    fn provider(&self) -> CloudProvider {
        self.provider
    }

    async fn deploy(&self, req: CloudDeployRequest) -> Result<CloudDeployment, LocalCodeError> {
        // Key/confirmation validation still applies so early UX is real…
        if self.api_key.is_none() && self.provider != CloudProvider::Akash {
            return Err(LocalCodeError::new(
                ErrorCode::CloudKeyMissing,
                format!("{} API key not configured", self.provider.as_str()),
            )
            .with_hint("Export the provider API key (see Setup)"));
        }
        if let Some(cost) = req.estimated_usd_per_hour {
            if cost > 0.0 && !req.confirmed {
                return Err(LocalCodeError::new(
                    ErrorCode::PaymentConfirmRequired,
                    format!("Confirm cloud spend ~${cost:.2}/hr"),
                )
                .with_hint("Confirm in the deploy dialog"));
            }
        }
        // …but provisioning itself is not implemented.
        Err(not_implemented(self.provider))
    }

    async fn destroy(&self, _id: &str) -> Result<(), LocalCodeError> {
        Err(not_implemented(self.provider))
    }

    async fn status(&self, _id: &str) -> Result<CloudDeployment, LocalCodeError> {
        Err(not_implemented(self.provider))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cloud_deploy_is_honestly_unimplemented() {
        let orch = CloudOrchestrator::from_config(&Config::default());
        let err = orch
            .deploy(CloudDeployRequest {
                provider: CloudProvider::Akash,
                model_id: "org/model".into(),
                quantization: None,
                gpu_filter: None,
                estimated_usd_per_hour: None,
                confirmed: true,
            })
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::NotImplemented);
    }
}
