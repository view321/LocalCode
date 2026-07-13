//! Hugging Face model registry client with mirror support and local cache.

mod cache;
mod quant;
mod url_builder;

pub use cache::ModelCache;
pub use quant::{discover_quants, parse_quant_from_filename, QuantFile, QuantGroup};
pub use url_builder::UrlBuilder;

use localcode_core::config::RegistryConfig;
use localcode_core::error::{ErrorCode, LocalCodeError};
use localcode_core::ids::CorrelationId;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    pub id: String,
    pub author: Option<String>,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub likes: Option<u64>,
    pub downloads: Option<u64>,
    pub last_modified: Option<String>,
    pub private: Option<bool>,
    pub gated: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetail {
    pub summary: ModelSummary,
    pub siblings: Vec<ModelFile>,
    pub card_data: Option<serde_json::Value>,
    pub sha: Option<String>,
    pub card_markdown: Option<String>,
    pub license: Option<String>,
    pub parameter_size: Option<String>,
    pub quants: Vec<QuantGroup>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub rfilename: String,
    pub size: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct HfClient {
    // Cloneable via Arc internals of reqwest::Client
    http: reqwest::Client,
    urls: UrlBuilder,
    token: Option<String>,
    cache: ModelCache,
}

impl HfClient {
    pub fn new(
        registry: &RegistryConfig,
        token: Option<String>,
        cache_dir: std::path::PathBuf,
    ) -> Result<Self, LocalCodeError> {
        let http = reqwest::Client::builder()
            .user_agent(format!("LocalCode/{}", env!("CARGO_PKG_VERSION")))
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| {
                LocalCodeError::new(ErrorCode::Internal, e.to_string())
                    .with_source("hf", "client_build")
            })?;

        Ok(Self {
            http,
            urls: UrlBuilder::new(&registry.endpoint, &registry.api_endpoint),
            token,
            cache: ModelCache::new(cache_dir),
        })
    }

    fn auth_header(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(token) = &self.token {
            req.bearer_auth(token)
        } else {
            req
        }
    }

    /// Search models with optional coding bias.
    pub async fn search(
        &self,
        query: &str,
        coding_only: bool,
        limit: u32,
        sort: &str,
    ) -> Result<Vec<ModelSummary>, LocalCodeError> {
        let cid = CorrelationId::new();
        let mut url = self.urls.api(&format!(
            "models?search={}&limit={}&sort={}&direction=-1",
            urlencoding_lite(query),
            limit,
            sort
        ));
        if coding_only {
            url = format!("{url}&filter=text-generation");
        }

        match self.fetch_search(&url, cid).await {
            Ok(models) => {
                let mut models = models;
                if coding_only {
                    models.sort_by_key(|m| {
                        let score = coding_boost(m);
                        std::cmp::Reverse(score)
                    });
                }
                let _ = self.cache.put_search(query, &models);
                Ok(models)
            }
            Err(e) => {
                if let Some(cached) = self.cache.get_search(query) {
                    warn!(error = %e, "HF search failed; serving cache");
                    Ok(cached)
                } else {
                    Err(e)
                }
            }
        }
    }

    async fn fetch_search(
        &self,
        url: &str,
        cid: CorrelationId,
    ) -> Result<Vec<ModelSummary>, LocalCodeError> {
        info!(%cid, %url, "HF search");
        let req = self.auth_header(self.http.get(url));
        let resp = req.send().await.map_err(|e| map_http_err(e, cid))?;
        handle_status(resp.status(), cid)?;
        let models: Vec<ModelSummary> = resp.json().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::HfUnreachable, e.to_string())
                .with_correlation(cid)
                .with_cause("Invalid JSON from HF models API")
        })?;
        Ok(models)
    }

    pub async fn popular_coding(&self, limit: u32) -> Result<Vec<ModelSummary>, LocalCodeError> {
        self.search("code", true, limit, "downloads").await
    }

    pub async fn trending_coding(&self, limit: u32) -> Result<Vec<ModelSummary>, LocalCodeError> {
        // HF has no official trending; use likes + coding boost as heuristic
        self.search("coder", true, limit, "likes").await
    }

    pub async fn model_info(&self, model_id: &str) -> Result<ModelDetail, LocalCodeError> {
        let cid = CorrelationId::new();
        match self.fetch_model_info(model_id, cid).await {
            Ok(detail) => {
                let _ = self.cache.put_model(model_id, &detail);
                Ok(detail)
            }
            Err(e) => {
                if let Some(cached) = self.cache.get_model(model_id) {
                    warn!(error = %e, model_id, "HF model info failed; serving cache");
                    Ok(cached)
                } else {
                    Err(e)
                }
            }
        }
    }

    async fn fetch_model_info(
        &self,
        model_id: &str,
        cid: CorrelationId,
    ) -> Result<ModelDetail, LocalCodeError> {
        let url = self.urls.api(&format!("models/{model_id}"));
        info!(%cid, %model_id, "HF model info");
        let req = self.auth_header(self.http.get(&url));
        let resp = req.send().await.map_err(|e| map_http_err(e, cid))?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(LocalCodeError::new(
                ErrorCode::HfModelNotFound,
                format!("Model not found: {model_id}"),
            )
            .with_correlation(cid)
            .with_hint("Check the model id (org/name)")
            .retryable(false));
        }
        handle_status(resp.status(), cid)?;

        #[derive(Deserialize)]
        struct RawModel {
            id: String,
            author: Option<String>,
            pipeline_tag: Option<String>,
            tags: Option<Vec<String>>,
            likes: Option<u64>,
            downloads: Option<u64>,
            #[serde(rename = "lastModified")]
            last_modified: Option<String>,
            private: Option<bool>,
            gated: Option<serde_json::Value>,
            siblings: Option<Vec<RawSibling>>,
            card_data: Option<serde_json::Value>,
            sha: Option<String>,
        }
        #[derive(Deserialize)]
        struct RawSibling {
            rfilename: String,
            size: Option<u64>,
        }

        let raw: RawModel = resp.json().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::HfUnreachable, e.to_string())
                .with_correlation(cid)
        })?;

        let siblings: Vec<ModelFile> = raw
            .siblings
            .unwrap_or_default()
            .into_iter()
            .map(|s| ModelFile {
                rfilename: s.rfilename,
                size: s.size,
            })
            .collect();

        let card_markdown = self.fetch_readme(model_id).await.ok();
        let license = extract_license(raw.card_data.as_ref(), raw.tags.as_deref());
        let parameter_size = extract_param_size(&raw.id, raw.card_data.as_ref(), raw.tags.as_deref());
        let quants = discover_quants(&siblings);

        Ok(ModelDetail {
            summary: ModelSummary {
                id: raw.id,
                author: raw.author,
                pipeline_tag: raw.pipeline_tag,
                tags: raw.tags.unwrap_or_default(),
                likes: raw.likes,
                downloads: raw.downloads,
                last_modified: raw.last_modified,
                private: raw.private,
                gated: raw.gated,
            },
            siblings,
            card_data: raw.card_data,
            sha: raw.sha,
            card_markdown,
            license,
            parameter_size,
            quants,
        })
    }

    async fn fetch_readme(&self, model_id: &str) -> Result<String, LocalCodeError> {
        let url = self.urls.resolve_file(model_id, "README.md");
        let req = self.auth_header(self.http.get(&url));
        let resp = req.send().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::HfUnreachable, e.to_string())
        })?;
        if !resp.status().is_success() {
            return Err(LocalCodeError::new(
                ErrorCode::HfUnreachable,
                format!("README status {}", resp.status()),
            ));
        }
        Ok(resp.text().await.unwrap_or_default())
    }

    /// Resolve download URL via mirror with primary fallback.
    pub fn download_url(&self, model_id: &str, filename: &str) -> String {
        self.urls.resolve_file(model_id, filename)
    }

    pub fn cache(&self) -> &ModelCache {
        &self.cache
    }
}

fn map_http_err(e: reqwest::Error, cid: CorrelationId) -> LocalCodeError {
    LocalCodeError::new(ErrorCode::HfUnreachable, e.to_string())
        .with_correlation(cid)
        .with_cause("Network error contacting Hugging Face")
        .with_hint("Check internet, proxy, or set a mirror (registry.endpoint)")
        .with_hint("Cached results will be used when available")
        .retryable(true)
}

fn handle_status(status: reqwest::StatusCode, cid: CorrelationId) -> Result<(), LocalCodeError> {
    if status.is_success() {
        return Ok(());
    }
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(LocalCodeError::new(
            ErrorCode::HfRateLimited,
            "Hugging Face rate limit exceeded",
        )
        .with_correlation(cid)
        .with_cause("Too many requests to HF API")
        .with_hint("Wait and retry; use a mirror; set HF_TOKEN for higher limits")
        .retryable(true));
    }
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(LocalCodeError::new(
            ErrorCode::HfAuthRequired,
            "Hugging Face authentication required or access denied",
        )
        .with_correlation(cid)
        .with_cause("Gated model or invalid token")
        .with_hint("Set HF_TOKEN for gated models")
        .retryable(false));
    }
    Err(LocalCodeError::new(
        ErrorCode::HfUnreachable,
        format!("HF API returned {status}"),
    )
    .with_correlation(cid)
    .retryable(true))
}

fn coding_boost(m: &ModelSummary) -> u64 {
    let mut score = m.downloads.unwrap_or(0) / 1000 + m.likes.unwrap_or(0) * 10;
    let id_lower = m.id.to_lowercase();
    if id_lower.contains("code") || id_lower.contains("coder") || id_lower.contains("codellama") {
        score += 50_000;
    }
    if m.tags.iter().any(|t| t.contains("code")) {
        score += 10_000;
    }
    score
}

fn extract_license(card: Option<&serde_json::Value>, tags: Option<&[String]>) -> Option<String> {
    if let Some(c) = card {
        if let Some(lic) = c.get("license").and_then(|v| v.as_str()) {
            return Some(lic.to_string());
        }
    }
    tags.and_then(|ts| {
        ts.iter()
            .find(|t| t.starts_with("license:"))
            .map(|t| t.trim_start_matches("license:").to_string())
    })
}

fn extract_param_size(
    id: &str,
    card: Option<&serde_json::Value>,
    tags: Option<&[String]>,
) -> Option<String> {
    if let Some(c) = card {
        if let Some(p) = c
            .get("model_name")
            .and_then(|v| v.as_str())
            .and_then(param_from_text)
        {
            return Some(p);
        }
    }
    if let Some(p) = param_from_text(id) {
        return Some(p);
    }
    tags.and_then(|ts| ts.iter().find_map(|t| param_from_text(t)))
}

fn param_from_text(s: &str) -> Option<String> {
    let re = regex::Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*([bB])\b").ok()?;
    re.captures(s).map(|c| format!("{}B", &c[1]))
}

fn urlencoding_lite(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_parse() {
        assert_eq!(param_from_text("Qwen2.5-Coder-7B-Instruct").as_deref(), Some("7B"));
        assert_eq!(param_from_text("llama-70b").as_deref(), Some("70B"));
    }
}
