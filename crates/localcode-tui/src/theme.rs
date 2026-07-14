use localcode_core::theme::{Theme, ThemeToken};
use ratatui::style::{Color, Style};

pub fn color(theme: &Theme, token: ThemeToken) -> Color {
    let (r, g, b) = theme.token_rgb(token);
    Color::Rgb(r, g, b)
}

pub fn accent(theme: &Theme) -> Style {
    Style::default().fg(color(theme, ThemeToken::Accent))
}

pub fn muted(theme: &Theme) -> Style {
    Style::default().fg(color(theme, ThemeToken::Muted))
}

pub fn border(theme: &Theme) -> Style {
    Style::default().fg(color(theme, ThemeToken::Border))
}

/// Dimmest gray: bar tracks, `·` separators, disabled/pending markers.
pub fn faint(theme: &Theme) -> Style {
    Style::default().fg(color(theme, ThemeToken::Faint))
}

/// The color of the animated braille glyph (only shown while busy).
pub fn work(theme: &Theme) -> Style {
    Style::default().fg(color(theme, ThemeToken::Work))
}
