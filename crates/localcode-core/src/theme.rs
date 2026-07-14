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

    /// RGB for a named token.
    ///
    /// `Dark` and `Light` are the shipped themes and are **grayscale**: emphasis
    /// is carried by brightness, never hue. The semantic tokens (`Ok`/`Warn`/
    /// `Error`) desaturate to grays too — state is conveyed with words and
    /// weight, not red/green/yellow (see the TUI redesign spec). `HighContrast`
    /// keeps its saturated palette (out of scope for the redesign).
    pub fn token_rgb(&self, token: ThemeToken) -> (u8, u8, u8) {
        match self.mode {
            // Dark (dark + gray).
            ThemeMode::Dark => match token {
                ThemeToken::Bg => (13, 13, 15),
                ThemeToken::Fg => (215, 215, 218),
                ThemeToken::Muted | ThemeToken::NavIdle => (108, 108, 114),
                ThemeToken::Border => (34, 34, 38),
                // Emphasis: selected row, active theme, user prompt.
                ThemeToken::Accent | ThemeToken::NavActive | ThemeToken::NavHover => (243, 243, 245),
                ThemeToken::Faint => (60, 60, 66),
                ThemeToken::Work => (207, 207, 212),
                // Grayscale semantics: good/active = emphasis, idle = muted,
                // error = primary text (bold at the call site).
                ThemeToken::Ok => (243, 243, 245),
                ThemeToken::Warn => (108, 108, 114),
                ThemeToken::Error => (215, 215, 218),
            },
            // Light (white + gray).
            ThemeMode::Light => match token {
                ThemeToken::Bg => (244, 244, 243),
                ThemeToken::Fg => (43, 43, 45),
                ThemeToken::Muted | ThemeToken::NavIdle => (134, 134, 138),
                ThemeToken::Border => (224, 224, 221),
                ThemeToken::Accent | ThemeToken::NavActive | ThemeToken::NavHover => (15, 15, 17),
                ThemeToken::Faint => (188, 188, 188),
                ThemeToken::Work => (58, 58, 62),
                ThemeToken::Ok => (15, 15, 17),
                ThemeToken::Warn => (134, 134, 138),
                ThemeToken::Error => (43, 43, 45),
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
                ThemeToken::Faint => (120, 120, 120),
                ThemeToken::Work => (255, 255, 255),
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
    /// Bar tracks, disabled text, `·` separators — dimmer than `Muted`.
    Faint,
    /// The animated braille glyph shown only while the agent is working.
    Work,
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
