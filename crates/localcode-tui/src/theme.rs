use localcode_core::theme::{Theme, ThemeToken};
use ratatui::style::{Color, Modifier, Style};

pub fn color(theme: &Theme, token: ThemeToken) -> Color {
    let (r, g, b) = theme.token_rgb(token);
    Color::Rgb(r, g, b)
}

/// The background-highlight for the currently-selected row/element. Meant to be
/// used as a row's base style (per-span `fg` still shows through), or patched
/// onto individual spans, so the picked item reads as a solid bar.
pub fn selected(theme: &Theme) -> Style {
    Style::default()
        .bg(color(theme, ThemeToken::SelBg))
        .fg(color(theme, ThemeToken::Accent))
        .add_modifier(Modifier::BOLD)
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
