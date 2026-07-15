use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThemeMode {
    #[default]
    Dark,
    /// Neon: light blue + white on dark gray. Replaces the old grayscale
    /// `Light` theme; the serde alias keeps existing `theme = "light"` configs
    /// loading.
    #[serde(alias = "light")]
    Neon,
    /// Neon variation: hot pink + white on dark gray.
    NeonPink,
    /// Soft green text on a gray background — the calm phosphor look.
    Sage,
    HighContrast,
}

impl ThemeMode {
    /// The themes offered by the status-bar switcher and `/theme`, in order.
    /// `HighContrast` is reachable only via the config file (accessibility
    /// escape hatch), so it is intentionally left out of the cycle.
    pub const SWITCHER: [ThemeMode; 4] =
        [ThemeMode::Dark, ThemeMode::Neon, ThemeMode::NeonPink, ThemeMode::Sage];

    /// Short label shown in the switcher and Settings.
    pub fn label(self) -> &'static str {
        match self {
            ThemeMode::Dark => "dark",
            ThemeMode::Neon => "neon",
            ThemeMode::NeonPink => "pink",
            ThemeMode::Sage => "sage",
            ThemeMode::HighContrast => "contrast",
        }
    }

    /// The next theme in the switcher cycle (used by `/theme` and the Settings
    /// toggle). Falls back to the first entry for themes outside the cycle.
    pub fn next(self) -> ThemeMode {
        let sw = Self::SWITCHER;
        let i = sw
            .iter()
            .position(|m| *m == self)
            .map(|i| (i + 1) % sw.len())
            .unwrap_or(0);
        sw[i]
    }
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
    /// `Dark` is grayscale (emphasis by brightness). The two **neon** themes
    /// (`Neon`, `NeonPink`) put white text and a saturated accent (light blue /
    /// hot pink) on a dark-gray background; they replaced the old grayscale
    /// `Light` theme. `Sage` is soft green text on a gray background.
    /// `HighContrast` keeps its own saturated palette.
    ///
    /// `SelBg` is the background of the currently-selected row/element — the
    /// only token that is meant to be used as a background fill.
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
                // A lighter gray bar behind the selected row.
                ThemeToken::SelBg => (44, 44, 54),
                // Grayscale semantics: good/active = emphasis, idle = muted,
                // error = primary text (bold at the call site).
                ThemeToken::Ok => (243, 243, 245),
                ThemeToken::Warn => (108, 108, 114),
                ThemeToken::Error => (215, 215, 218),
            },
            // Neon (light blue + white on dark gray).
            ThemeMode::Neon => match token {
                ThemeToken::Bg => (20, 22, 28),
                ThemeToken::Fg => (235, 240, 248),
                ThemeToken::Muted | ThemeToken::NavIdle => (120, 140, 165),
                ThemeToken::Border => (40, 50, 66),
                ThemeToken::Accent | ThemeToken::NavActive | ThemeToken::NavHover => (125, 211, 255),
                ThemeToken::Faint => (60, 72, 92),
                ThemeToken::Work => (125, 211, 255),
                ThemeToken::SelBg => (28, 48, 70),
                ThemeToken::Ok => (125, 211, 255),
                ThemeToken::Warn => (120, 140, 165),
                ThemeToken::Error => (235, 240, 248),
            },
            // Neon variation (hot pink + white on dark gray).
            ThemeMode::NeonPink => match token {
                ThemeToken::Bg => (22, 18, 26),
                ThemeToken::Fg => (240, 236, 246),
                ThemeToken::Muted | ThemeToken::NavIdle => (150, 120, 155),
                ThemeToken::Border => (56, 40, 60),
                ThemeToken::Accent | ThemeToken::NavActive | ThemeToken::NavHover => (255, 120, 214),
                ThemeToken::Faint => (78, 58, 84),
                ThemeToken::Work => (255, 120, 214),
                ThemeToken::SelBg => (58, 28, 54),
                ThemeToken::Ok => (255, 120, 214),
                ThemeToken::Warn => (150, 120, 155),
                ThemeToken::Error => (240, 236, 246),
            },
            // Sage (soft green text on gray).
            ThemeMode::Sage => match token {
                ThemeToken::Bg => (40, 43, 40),
                ThemeToken::Fg => (185, 210, 180),
                ThemeToken::Muted | ThemeToken::NavIdle => (124, 140, 122),
                ThemeToken::Border => (62, 66, 62),
                ThemeToken::Accent | ThemeToken::NavActive | ThemeToken::NavHover => (150, 225, 150),
                ThemeToken::Faint => (80, 88, 78),
                ThemeToken::Work => (150, 225, 150),
                ThemeToken::SelBg => (58, 70, 56),
                ThemeToken::Ok => (150, 225, 150),
                ThemeToken::Warn => (124, 140, 122),
                ThemeToken::Error => (185, 210, 180),
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
                ThemeToken::SelBg => (0, 40, 90),
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
    /// Background fill behind the currently-selected row/element.
    SelBg,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The colored themes must keep their accent/text well clear of the
    /// background (and the selection bar must sit between the two).
    #[test]
    fn colored_themes_are_readable() {
        let dist = |a: (u8, u8, u8), b: (u8, u8, u8)| {
            (a.0 as i32 - b.0 as i32).abs()
                + (a.1 as i32 - b.1 as i32).abs()
                + (a.2 as i32 - b.2 as i32).abs()
        };
        for mode in [ThemeMode::Neon, ThemeMode::NeonPink, ThemeMode::Sage] {
            let t = Theme::new(mode);
            let bg = t.token_rgb(ThemeToken::Bg);
            for token in [ThemeToken::NavHover, ThemeToken::NavActive, ThemeToken::Fg] {
                let fg = t.token_rgb(token);
                assert!(dist(bg, fg) > 300, "{mode:?} token {token:?} too close to background");
            }
            // The selection bar is distinct from both the background and the
            // accent so selected text stays legible on it.
            let sel = t.token_rgb(ThemeToken::SelBg);
            assert!(dist(bg, sel) > 20, "{mode:?} selection bar too close to background");
            assert!(dist(sel, t.token_rgb(ThemeToken::Accent)) > 150, "{mode:?} accent unreadable on selection");
        }
    }
}
