//! Simple filesystem JSON cache for HF metadata.

use crate::{ModelDetail, ModelSummary};
use serde::{de::DeserializeOwned, Serialize};
use std::path::{Path, PathBuf};
use tracing::debug;

#[derive(Debug, Clone)]
pub struct ModelCache {
    // root is PathBuf — clone is cheap
    root: PathBuf,
}

impl ModelCache {
    pub fn new(root: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&root);
        Self { root }
    }

    fn search_path(&self, query: &str) -> PathBuf {
        let key = sanitize(query);
        self.root.join(format!("search_{key}.json"))
    }

    fn model_path(&self, model_id: &str) -> PathBuf {
        let key = sanitize(model_id);
        self.root.join(format!("model_{key}.json"))
    }

    pub fn put_search(&self, query: &str, models: &[ModelSummary]) -> std::io::Result<()> {
        write_json(&self.search_path(query), models)
    }

    pub fn get_search(&self, query: &str) -> Option<Vec<ModelSummary>> {
        read_json(&self.search_path(query))
    }

    pub fn put_model(&self, model_id: &str, detail: &ModelDetail) -> std::io::Result<()> {
        write_json(&self.model_path(model_id), detail)
    }

    pub fn get_model(&self, model_id: &str) -> Option<ModelDetail> {
        read_json(&self.model_path(model_id))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .take(120)
        .collect()
}

fn write_json<T: Serialize + ?Sized>(path: &Path, value: &T) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let data = serde_json::to_vec_pretty(value)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, data)?;
    debug!(path = %path.display(), "cache write");
    Ok(())
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Option<T> {
    let data = std::fs::read(path).ok()?;
    serde_json::from_slice(&data).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn cache_roundtrip() {
        let dir = tempdir().unwrap();
        let cache = ModelCache::new(dir.path().to_path_buf());
        let models = vec![ModelSummary {
            id: "test/model".into(),
            author: Some("test".into()),
            pipeline_tag: Some("text-generation".into()),
            tags: vec!["code".into()],
            likes: Some(1),
            downloads: Some(2),
            last_modified: None,
            private: Some(false),
            gated: None,
        }];
        cache.put_search("code", &models).unwrap();
        let got = cache.get_search("code").unwrap();
        assert_eq!(got[0].id, "test/model");
    }
}
