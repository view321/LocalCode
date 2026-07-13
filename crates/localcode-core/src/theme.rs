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

    /// RGB for a named token. Every token is mode-aware so light mode is
    /// readable on a light background and high-contrast stays high-contrast.
    pub fn token_rgb(&self, token: ThemeToken) -> (u8, u8, u8) {
        match self.mode {
            ThemeMode::Dark => match token {
                ThemeToken::NavIdle => (100, 100, 100),
                ThemeToken::NavHover => (240, 240, 240),
                ThemeToken::NavActive => (255, 255, 255),
                ThemeToken::Warn => (230, 180, 40),
                ThemeToken::Error => (220, 70, 70),
                ThemeToken::Ok => (80, 200, 120),
                ThemeToken::Muted => (140, 140, 150),
                ThemeToken::Accent => (90, 160, 255),
                ThemeToken::Bg => (18, 18, 22),
                ThemeToken::Fg => (230, 230, 235),
                ThemeToken::Border => (60, 60, 70),
            },
            ThemeMode::Light => match token {
                ThemeToken::NavIdle => (120, 120, 128),
                ThemeToken::NavHover => (40, 40, 50),
                ThemeToken::NavActive => (0, 0, 0),
                ThemeToken::Warn => (146, 100, 0),
                ThemeToken::Error => (176, 30, 30),
                ThemeToken::Ok => (18, 120, 60),
                ThemeToken::Muted => (95, 95, 108),
                ThemeToken::Accent => (26, 84, 200),
                ThemeToken::Bg => (245, 245, 248),
                ThemeToken::Fg => (25, 25, 30),
                ThemeToken::Border => (176, 176, 188),
            },
            ThemeMode::HighContrast => match token {
                ThemeToken::NavIdle => (170, 170, 170),
                ThemeToken::NavHover => (255, 255, 255),
                ThemeToken::NavActive => (255, 255, 0),
                ThemeToken::Warn => (255, 200, 0),
                ThemeToken::Error => (255, 90, 90),
                ThemeToken::Ok => (0, 255, 130),
                ThemeToken::Muted => (200, 200, 200),
                ThemeToken::Accent => (80, 190, 255),
                ThemeToken::Bg => (0, 0, 0),
                ThemeToken::Fg => (255, 255, 255),
                ThemeToken::Border => (255, 255, 255),
            },
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Light mode must not render near-white nav text on its light background.
    #[test]
    fn light_mode_nav_is_readable() {
        let t = Theme::new(ThemeMode::Light);
        let bg = t.token_rgb(ThemeToken::Bg);
        for token in [ThemeToken::NavHover, ThemeToken::NavActive, ThemeToken::Fg] {
            let fg = t.token_rgb(token);
            let dist = (bg.0 as i32 - fg.0 as i32).abs()
                + (bg.1 as i32 - fg.1 as i32).abs()
                + (bg.2 as i32 - fg.2 as i32).abs();
            assert!(dist > 300, "token {token:?} too close to background");
        }
    }
}
