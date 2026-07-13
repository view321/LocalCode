use localcode_core::theme::{Theme, ThemeToken};
use ratatui::style::{Color, Modifier, Style};

pub fn color(theme: &Theme, token: ThemeToken) -> Color {
    let (r, g, b) = theme.token_rgb(token);
    Color::Rgb(r, g, b)
}

pub fn nav_idle(theme: &Theme) -> Style {
    Style::default().fg(color(theme, ThemeToken::NavIdle))
}

pub fn nav_hover(theme: &Theme) -> Style {
    Style::default().fg(color(theme, ThemeToken::NavHover))
}

pub fn nav_active(theme: &Theme) -> Style {
    Style::default()
        .fg(color(theme, ThemeToken::NavActive))
        .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
}

pub fn accent(theme: &Theme) -> Style {
    Style::default().fg(color(theme, ThemeToken::Accent))
}

pub fn warn(theme: &Theme) -> Style {
    Style::default().fg(color(theme, ThemeToken::Warn))
}

pub fn error(theme: &Theme) -> Style {
    Style::default().fg(color(theme, ThemeToken::Error))
}

#[allow(dead_code)]
pub fn ok(theme: &Theme) -> Style {
    Style::default().fg(color(theme, ThemeToken::Ok))
}

pub fn muted(theme: &Theme) -> Style {
    Style::default().fg(color(theme, ThemeToken::Muted))
}

pub fn border(theme: &Theme) -> Style {
    Style::default().fg(color(theme, ThemeToken::Border))
}
