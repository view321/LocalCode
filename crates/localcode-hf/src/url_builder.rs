//! URL construction with mirror base swapping.

#[derive(Debug, Clone)]
pub struct UrlBuilder {
    endpoint: String,
    api_endpoint: String,
    primary_endpoint: String,
    primary_api: String,
}

impl UrlBuilder {
    pub fn new(endpoint: &str, api_endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            api_endpoint: api_endpoint.trim_end_matches('/').to_string(),
            primary_endpoint: "https://huggingface.co".into(),
            primary_api: "https://huggingface.co/api".into(),
        }
    }

    pub fn api(&self, path: &str) -> String {
        let path = path.trim_start_matches('/');
        format!("{}/{}", self.api_endpoint, path)
    }

    pub fn resolve_file(&self, model_id: &str, filename: &str) -> String {
        format!(
            "{}/{}/resolve/main/{}",
            self.endpoint,
            model_id,
            filename.trim_start_matches('/')
        )
    }

    pub fn is_mirror(&self) -> bool {
        self.endpoint != self.primary_endpoint
    }

    pub fn primary_api(&self, path: &str) -> String {
        let path = path.trim_start_matches('/');
        format!("{}/{}", self.primary_api, path)
    }

    pub fn primary_file(&self, model_id: &str, filename: &str) -> String {
        format!(
            "{}/{}/resolve/main/{}",
            self.primary_endpoint,
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
}
