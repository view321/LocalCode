//! LocalCode core: configuration, errors, events, and shared types.

pub mod config;
pub mod error;
pub mod events;
pub mod ids;
pub mod paths;
pub mod runtime;
pub mod theme;

pub use config::{Config, ConfigError};
pub use error::{ErrorCode, LocalCodeError};
pub use events::{AppEvent, EventBus};
pub use ids::CorrelationId;
pub use paths::AppPaths;
pub use runtime::{ActiveRuntime, RuntimeKind, RuntimeStatus};
pub use theme::{Theme, ThemeMode};
