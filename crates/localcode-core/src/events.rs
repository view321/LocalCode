use crate::error::LocalCodeError;
use crate::runtime::ActiveRuntime;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// Application-wide events for UI and services.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AppEvent {
    Notification {
        severity: Severity,
        title: String,
        body: String,
        correlation_id: Option<String>,
    },
    DeployProgress {
        job_id: String,
        percent: u8,
        message: String,
    },
    DeployFinished {
        job_id: String,
        runtime: ActiveRuntime,
    },
    DeployFailed {
        job_id: String,
        error: LocalCodeError,
    },
    RuntimeUpdated {
        runtime: ActiveRuntime,
    },
    BenchProgress {
        run_id: String,
        completed: u32,
        total: u32,
        message: String,
    },
    ErrorRaised {
        error: LocalCodeError,
    },
    Status {
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warn,
    Error,
    Success,
}

/// Simple in-process event bus (cloneable, thread-safe).
#[derive(Clone, Default)]
pub struct EventBus {
    inner: Arc<Mutex<Vec<AppEvent>>>,
    listeners: Arc<Mutex<Vec<Box<dyn Fn(&AppEvent) + Send>>>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn publish(&self, event: AppEvent) {
        if let Ok(listeners) = self.listeners.lock() {
            for listener in listeners.iter() {
                listener(&event);
            }
        }
        if let Ok(mut q) = self.inner.lock() {
            q.push(event);
            // Keep a bounded history for UI
            if q.len() > 500 {
                let drain = q.len() - 500;
                q.drain(0..drain);
            }
        }
    }

    pub fn drain(&self) -> Vec<AppEvent> {
        self.inner
            .lock()
            .map(|mut q| std::mem::take(&mut *q))
            .unwrap_or_default()
    }

    pub fn recent(&self, n: usize) -> Vec<AppEvent> {
        self.inner
            .lock()
            .map(|q| q.iter().rev().take(n).cloned().collect())
            .unwrap_or_default()
    }

    pub fn subscribe<F>(&self, f: F)
    where
        F: Fn(&AppEvent) + Send + 'static,
    {
        if let Ok(mut listeners) = self.listeners.lock() {
            listeners.push(Box::new(f));
        }
    }
}

impl std::fmt::Debug for EventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventBus").finish_non_exhaustive()
    }
}
