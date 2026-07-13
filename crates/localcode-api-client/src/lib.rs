//! REST client for LocalCode VPS API.

use localcode_core::config::ApiConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::debug;

#[derive(Clone)]
pub struct ApiClient {
    http: reqwest::Client,
    base_url: String,
    token: Option<String>,
}

impl ApiClient {
    pub fn new(cfg: &ApiConfig, token: Option<String>) -> Result<Self, LocalCodeError> {
        let http = reqwest::Client::builder()
            .user_agent(format!("LocalCode/{}", env!("CARGO_PKG_VERSION")))
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| LocalCodeError::new(ErrorCode::Internal, e.to_string()))?;
        // Env override resolved here, at use time — never persisted to config.
        let base_url = std::env::var("LOCALCODE_API_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| cfg.base_url.clone());
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            token,
        })
    }

    pub fn set_token(&mut self, token: Option<String>) {
        self.token = token;
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(t) = &self.token {
            req.bearer_auth(t)
        } else {
            req
        }
    }

    pub async fn health(&self) -> Result<HealthResponse, LocalCodeError> {
        self.get_json("/v1/health").await
    }

    pub async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, LocalCodeError> {
        let url = self.url(path);
        debug!(%url, "API GET");
        let req = self.auth(self.http.get(&url));
        let resp = req.send().await.map_err(map_net)?;
        parse_response(resp).await
    }

    pub async fn post_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, LocalCodeError> {
        let url = self.url(path);
        debug!(%url, "API POST");
        let req = self.auth(self.http.post(&url).json(body));
        let resp = req.send().await.map_err(map_net)?;
        parse_response(resp).await
    }

    pub async fn device_auth_start(&self) -> Result<DeviceAuthStart, LocalCodeError> {
        self.post_json("/v1/auth/device/start", &serde_json::json!({})).await
    }

    pub async fn device_auth_poll(&self, device_code: &str) -> Result<DeviceAuthPoll, LocalCodeError> {
        self.post_json(
            "/v1/auth/device/poll",
            &serde_json::json!({ "device_code": device_code }),
        )
        .await
    }

    pub async fn me(&self) -> Result<UserInfo, LocalCodeError> {
        self.get_json("/v1/me").await
    }

    pub async fn trending_models(&self) -> Result<Vec<serde_json::Value>, LocalCodeError> {
        self.get_json("/v1/models/trending?task=code").await
    }
}

fn map_net(e: reqwest::Error) -> LocalCodeError {
    LocalCodeError::new(ErrorCode::ApiUnreachable, e.to_string())
        .with_cause("Cannot reach LocalCode API")
        .with_hint("Check LOCALCODE_API_URL and network")
        .with_hint("Local features work offline without the API")
        .retryable(true)
}

async fn parse_response<T: DeserializeOwned>(
    resp: reqwest::Response,
) -> Result<T, LocalCodeError> {
    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Err(LocalCodeError::new(
            ErrorCode::AuthRequired,
            "Authentication required",
        )
        .with_hint("Sign in via Setup → Account")
        .with_cause("Missing or expired session token"));
    }
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(LocalCodeError::new(
            ErrorCode::ApiUnreachable,
            format!("API {status}: {body}"),
        )
        .retryable(status.is_server_error()));
    }
    resp.json().await.map_err(|e| {
        LocalCodeError::new(ErrorCode::ApiUnreachable, e.to_string())
            .with_cause("Invalid JSON from API")
    })
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct HealthResponse {
    pub status: String,
    #[serde(default)]
    pub version: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct DeviceAuthStart {
    pub device_code: String,
    pub user_code: String,
    pub verification_url: String,
    pub interval: u64,
    pub expires_in: u64,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct DeviceAuthPoll {
    pub status: String,
    #[serde(default)]
    pub access_token: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct UserInfo {
    pub id: String,
    #[serde(default)]
    pub email: Option<String>,
}
