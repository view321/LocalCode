//! URL construction with mirror base swapping.
//!
//! A [`UrlBuilder`] holds an ordered list of host bases (the configured
//! endpoint first, then any mirrors, then `https://huggingface.co`, and
//! finally the public `https://hf-mirror.com` mirror as a built-in last
//! resort — so HF stays reachable even with no mirrors configured). The
//! single-URL methods (`api`, `resolve_file`) return the primary (first) host
//! so existing call sites are unchanged; the `*_candidates` methods return
//! every host in order so downloaders can fall back mirror-by-mirror when one
//! is unreachable.

/// A web+api base pair for one host.
#[derive(Debug, Clone, PartialEq, Eq)]
struct HostPair {
    web: String,
    api: String,
}

const PRIMARY_WEB: &str = "https://huggingface.co";
const PRIMARY_API: &str = "https://huggingface.co/api";
/// Well-known public HF mirror, appended after the canonical host so it is
/// only consulted when `huggingface.co` (and every configured mirror) fails.
const FALLBACK_MIRROR_WEB: &str = "https://hf-mirror.com";
const FALLBACK_MIRROR_API: &str = "https://hf-mirror.com/api";

#[derive(Debug, Clone)]
pub struct UrlBuilder {
    /// Ordered, de-duplicated host bases. Never empty (the primary HF host is
    /// always appended as a fallback).
    hosts: Vec<HostPair>,
}

/// Derive the api base for a web root: `https://host` -> `https://host/api`.
fn api_for(web: &str) -> String {
    format!("{}/api", web.trim_end_matches('/'))
}

impl UrlBuilder {
    /// Single-endpoint builder (back-compat). The hardcoded primary HF host is
    /// still appended as a fallback so `*_candidates` always includes it.
    pub fn new(endpoint: &str, api_endpoint: &str) -> Self {
        Self::with_mirrors(endpoint, api_endpoint, &[])
    }

    /// Ordered builder: `endpoint` first, then each mirror (web roots; the api
    /// base is derived as `{web}/api`), then the hardcoded primary HF host,
    /// then the public hf-mirror.com fallback. Empty entries are skipped and
    /// duplicates removed while preserving order.
    pub fn with_mirrors(endpoint: &str, api_endpoint: &str, mirrors: &[String]) -> Self {
        let mut hosts: Vec<HostPair> = Vec::new();
        let mut push = |web: &str, api: &str| {
            let web = web.trim_end_matches('/').to_string();
            let api = api.trim_end_matches('/').to_string();
            if web.is_empty() {
                return;
            }
            let pair = HostPair { web, api };
            if !hosts.contains(&pair) {
                hosts.push(pair);
            }
        };

        push(endpoint, api_endpoint);
        for m in mirrors {
            let m = m.trim();
            if !m.is_empty() {
                let api = api_for(m);
                push(m, &api);
            }
        }
        // Always keep the canonical HF host reachable as a fallback, and the
        // public hf-mirror.com after it for when huggingface.co is unavailable.
        push(PRIMARY_WEB, PRIMARY_API);
        push(FALLBACK_MIRROR_WEB, FALLBACK_MIRROR_API);

        Self { hosts }
    }

    fn primary(&self) -> &HostPair {
        &self.hosts[0]
    }

    pub fn api(&self, path: &str) -> String {
        let path = path.trim_start_matches('/');
        format!("{}/{}", self.primary().api, path)
    }

    pub fn resolve_file(&self, model_id: &str, filename: &str) -> String {
        format!(
            "{}/{}/resolve/main/{}",
            self.primary().web,
            model_id,
            filename.trim_start_matches('/')
        )
    }

    /// Every host's api URL for `path`, primary first. Used for fallback.
    pub fn api_candidates(&self, path: &str) -> Vec<String> {
        let path = path.trim_start_matches('/');
        self.hosts
            .iter()
            .map(|h| format!("{}/{}", h.api, path))
            .collect()
    }

    /// Every host's file-download URL, primary first. Used for fallback.
    pub fn resolve_file_candidates(&self, model_id: &str, filename: &str) -> Vec<String> {
        let filename = filename.trim_start_matches('/');
        self.hosts
            .iter()
            .map(|h| format!("{}/{}/resolve/main/{}", h.web, model_id, filename))
            .collect()
    }

    /// Given a resolved URL on one host, produce the same path on every host in
    /// this builder (primary first). If `url` doesn't start with any known host,
    /// it is returned as the sole candidate. This lets a downloader that only
    /// has a final URL string still try mirrors.
    pub fn mirror_candidates(&self, url: &str) -> Vec<String> {
        for h in &self.hosts {
            if let Some(rest) = url.strip_prefix(&h.web) {
                return self
                    .hosts
                    .iter()
                    .map(|host| format!("{}{}", host.web, rest))
                    .collect();
            }
        }
        vec![url.to_string()]
    }

    pub fn is_mirror(&self) -> bool {
        self.primary().web != PRIMARY_WEB
    }

    pub fn primary_api(&self, path: &str) -> String {
        let path = path.trim_start_matches('/');
        format!("{PRIMARY_API}/{path}")
    }

    pub fn primary_file(&self, model_id: &str, filename: &str) -> String {
        format!(
            "{}/{}/resolve/main/{}",
            PRIMARY_WEB,
            model_id,
            filename.trim_start_matches('/')
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_urls() {
        let u = UrlBuilder::new("https://hf-mirror.com", "https://hf-mirror.com/api");
        assert!(u.is_mirror());
        assert_eq!(
            u.resolve_file("org/model", "model.gguf"),
            "https://hf-mirror.com/org/model/resolve/main/model.gguf"
        );
    }

    #[test]
    fn candidates_are_ordered_and_include_primary_fallback() {
        let u = UrlBuilder::with_mirrors(
            "https://hf-mirror.com",
            "https://hf-mirror.com/api",
            &["https://hf.example.net".into()],
        );
        let files = u.resolve_file_candidates("org/model", "m.gguf");
        assert_eq!(
            files,
            vec![
                "https://hf-mirror.com/org/model/resolve/main/m.gguf".to_string(),
                "https://hf.example.net/org/model/resolve/main/m.gguf".to_string(),
                "https://huggingface.co/org/model/resolve/main/m.gguf".to_string(),
            ]
        );
        // api candidates derive `{web}/api` for mirrors given as web roots.
        let apis = u.api_candidates("models/org/model");
        assert_eq!(apis[1], "https://hf.example.net/api/models/org/model");
        assert_eq!(apis[2], "https://huggingface.co/api/models/org/model");
    }

    #[test]
    fn default_endpoint_has_no_duplicate_fallback() {
        // When the primary already IS huggingface.co, it must appear once, with
        // the built-in hf-mirror.com fallback after it.
        let u = UrlBuilder::new("https://huggingface.co", "https://huggingface.co/api");
        assert!(!u.is_mirror());
        let files = u.resolve_file_candidates("org/model", "m.gguf");
        assert_eq!(
            files,
            vec![
                "https://huggingface.co/org/model/resolve/main/m.gguf".to_string(),
                "https://hf-mirror.com/org/model/resolve/main/m.gguf".to_string(),
            ]
        );
    }

    #[test]
    fn mirror_candidates_swaps_known_host() {
        let u = UrlBuilder::with_mirrors(
            "https://hf-mirror.com",
            "https://hf-mirror.com/api",
            &[],
        );
        let got = u.mirror_candidates("https://hf-mirror.com/org/model/resolve/main/m.gguf");
        assert_eq!(
            got,
            vec![
                "https://hf-mirror.com/org/model/resolve/main/m.gguf".to_string(),
                "https://huggingface.co/org/model/resolve/main/m.gguf".to_string(),
            ]
        );
        // Unknown host: returned as-is, single candidate.
        let other = u.mirror_candidates("https://other.example/x.gguf");
        assert_eq!(other, vec!["https://other.example/x.gguf".to_string()]);
    }
}
