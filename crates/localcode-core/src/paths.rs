use crate::error::{ErrorCode, LocalCodeError};
use directories::ProjectDirs;
use std::path::PathBuf;

/// Platform paths for config, data, and logs.
#[derive(Debug, Clone)]
pub struct AppPaths {
    pub home: PathBuf,
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub log_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub models_cache: PathBuf,
    pub workspaces_dir: PathBuf,
    pub sessions_dir: PathBuf,
}

impl AppPaths {
    pub fn resolve() -> Result<Self, LocalCodeError> {
        if let Ok(home) = std::env::var("LOCALCODE_HOME") {
            let home = PathBuf::from(home);
            return Ok(Self::from_home(home));
        }

        let dirs = ProjectDirs::from("dev", "LocalCode", "localcode").ok_or_else(|| {
            LocalCodeError::new(
                ErrorCode::ConfigLoadFailed,
                "Could not resolve platform project directories",
            )
            .with_hint("Set LOCALCODE_HOME to a writable directory")
        })?;

        Ok(Self {
            home: dirs.config_dir().parent().unwrap_or(dirs.config_dir()).to_path_buf(),
            config_dir: dirs.config_dir().to_path_buf(),
            data_dir: dirs.data_dir().to_path_buf(),
            log_dir: dirs
                .state_dir()
                .unwrap_or(dirs.data_local_dir())
                .join("logs"),
            cache_dir: dirs.cache_dir().to_path_buf(),
            models_cache: dirs.cache_dir().join("models"),
            workspaces_dir: dirs.data_dir().join("workspaces"),
            sessions_dir: dirs.data_dir().join("sessions"),
        })
    }

    pub fn from_home(home: PathBuf) -> Self {
        Self {
            config_dir: home.join("config"),
            data_dir: home.join("data"),
            log_dir: home.join("logs"),
            cache_dir: home.join("cache"),
            models_cache: home.join("cache").join("models"),
            workspaces_dir: home.join("data").join("workspaces"),
            sessions_dir: home.join("data").join("sessions"),
            home,
        }
    }

    pub fn config_file(&self) -> PathBuf {
        self.config_dir.join("config.toml")
    }

    /// Managed llama.cpp install root (prebuilt `llama-server` lives here).
    pub fn llamacpp_dir(&self) -> PathBuf {
        self.data_dir.join("backends").join("llamacpp")
    }

    /// Bundled local Bonsai assistant weights and state.
    pub fn assistant_dir(&self) -> PathBuf {
        self.data_dir.join("assistant")
    }

    pub fn ensure_dirs(&self) -> Result<(), LocalCodeError> {
        let assistant = self.assistant_dir();
        for dir in [
            &self.config_dir,
            &self.data_dir,
            &self.log_dir,
            &self.cache_dir,
            &self.models_cache,
            &self.workspaces_dir,
            &self.sessions_dir,
            &assistant,
        ] {
            std::fs::create_dir_all(dir).map_err(|e| {
                LocalCodeError::from(e)
                    .with_source("paths", "ensure_dirs")
                    .with_hint(format!("Create directory manually: {}", dir.display()))
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn from_home_layout() {
        let dir = tempdir().unwrap();
        let paths = AppPaths::from_home(dir.path().to_path_buf());
        assert!(paths.config_file().ends_with("config/config.toml")
            || paths.config_file().ends_with("config\\config.toml"));
        paths.ensure_dirs().unwrap();
        assert!(paths.log_dir.exists());
        assert!(paths.sessions_dir.ends_with("data/sessions")
            || paths.sessions_dir.ends_with("data\\sessions"));
        assert!(paths.sessions_dir.exists());
    }
}
