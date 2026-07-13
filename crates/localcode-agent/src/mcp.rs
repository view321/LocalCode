//! MCP server lifecycle (config-driven).

use localcode_core::error::{ErrorCode, LocalCodeError};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerStatus {
    pub name: String,
    pub healthy: bool,
    pub tools: Vec<String>,
    pub error: Option<String>,
}

pub struct McpManager {
    config_path: PathBuf,
    config: McpConfig,
    statuses: Vec<McpServerStatus>,
}

impl McpManager {
    pub fn new(config_path: PathBuf) -> Self {
        let config = load_config(&config_path).unwrap_or_default();
        Self {
            config_path,
            config,
            statuses: vec![],
        }
    }

    pub fn reload(&mut self) {
        self.config = load_config(&self.config_path).unwrap_or_default();
    }

    pub async fn connect_all(&mut self) -> Result<(), LocalCodeError> {
        self.statuses.clear();
        for server in &self.config.servers {
            let status = connect_server(server).await;
            if let Some(err) = &status.error {
                warn!(server = %server.name, %err, "MCP server failed (degraded)");
            } else {
                info!(server = %server.name, tools = status.tools.len(), "MCP connected");
            }
            self.statuses.push(status);
        }
        Ok(())
    }

    pub fn statuses(&self) -> &[McpServerStatus] {
        &self.statuses
    }

    pub fn config_path(&self) -> &PathBuf {
        &self.config_path
    }
}

fn load_config(path: &PathBuf) -> Result<McpConfig, LocalCodeError> {
    if !path.exists() {
        return Ok(McpConfig::default());
    }
    let raw = std::fs::read_to_string(path)?;
    serde_json::from_str(&raw).map_err(|e| {
        LocalCodeError::new(ErrorCode::AgentMcpFailed, e.to_string())
            .with_cause("Invalid mcp.json")
            .with_hint(format!("Fix JSON at {}", path.display()))
    })
}

async fn connect_server(server: &McpServerConfig) -> McpServerStatus {
    // v1: record config presence; full JSON-RPC MCP handshake in hardening
    if let Some(url) = &server.url {
        match reqwest::Client::new().get(url).send().await {
            Ok(r) if r.status().is_success() => McpServerStatus {
                name: server.name.clone(),
                healthy: true,
                tools: vec!["(http mcp — schema cache pending)".into()],
                error: None,
            },
            Ok(r) => McpServerStatus {
                name: server.name.clone(),
                healthy: false,
                tools: vec![],
                error: Some(format!("HTTP {}", r.status())),
            },
            Err(e) => McpServerStatus {
                name: server.name.clone(),
                healthy: false,
                tools: vec![],
                error: Some(e.to_string()),
            },
        }
    } else if server.command.is_some() {
        McpServerStatus {
            name: server.name.clone(),
            healthy: true,
            tools: vec!["(stdio mcp — started on demand)".into()],
            error: None,
        }
    } else {
        McpServerStatus {
            name: server.name.clone(),
            healthy: false,
            tools: vec![],
            error: Some("No command or url configured".into()),
        }
    }
}
