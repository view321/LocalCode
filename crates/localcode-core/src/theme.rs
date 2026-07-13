use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThemeMode {
    #[default]
    Dark,
    Light,
    HighContrast,
}

/// Semantic theme tokens used by the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    pub mode: ThemeMode,
}

impl Theme {
    pub fn new(mode: ThemeMode) -> Self {
        Self { mode }
    }

    /// RGB for a named token. Ratatui maps these to Color::Rgb.
    pub fn token_rgb(&self, token: ThemeToken) -> (u8, u8, u8) {
        match (self.mode, token) {
            (_, ThemeToken::NavIdle) => match self.mode {
                ThemeMode::Light => (120, 120, 120),
                _ => (100, 100, 100),
            },
            (_, ThemeToken::NavHover) => (240, 240, 240),
            (_, ThemeToken::NavActive) => (255, 255, 255),
            (_, ThemeToken::Warn) => (230, 180, 40),
            (_, ThemeToken::Error) => (220, 70, 70),
            (_, ThemeToken::Ok) => (80, 200, 120),
            (_, ThemeToken::Muted) => (140, 140, 150),
            (_, ThemeToken::Accent) => (90, 160, 255),
            (ThemeMode::Light, ThemeToken::Bg) => (245, 245, 248),
            (_, ThemeToken::Bg) => (18, 18, 22),
            (ThemeMode::Light, ThemeToken::Fg) => (20, 20, 24),
            (_, ThemeToken::Fg) => (230, 230, 235),
            (_, ThemeToken::Border) => (60, 60, 70),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeToken {
    NavIdle,
    NavHover,
    NavActive,
    Warn,
    Error,
    Ok,
    Muted,
    Accent,
    Bg,
    Fg,
    Border,
}
