//! Inline banner state + renderer.
//!
//! The redesign has no popups: confirms, warnings, errors and info that used to
//! be centered modals are now **inline banners** rendered at the top of the
//! working area (see the TUI redesign spec §8). The `ModalState`/`ConfirmAction`
//! semantics and click wiring are unchanged — only the presentation moved.

use localcode_backends::BackendKind;
use localcode_core::error::LocalCodeError;
use localcode_core::theme::{Theme, ThemeToken};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};
use ratatui::Frame;
use unicode_width::UnicodeWidthStr;

use crate::theme;

/// A button "surrounded" by block-drawing caps instead of plain `[ ]`.
///
/// `active` (a primary action, or the selected button in a row) fills the pill
/// with the accent colour; otherwise it renders as a light outline. The total
/// width is always [`button_width`] regardless of `active`, so click regions
/// computed from the label stay exact.
pub fn button(th: &Theme, label: &str, active: bool) -> Vec<Span<'static>> {
    let body = format!(" {label} ");
    if active {
        let accent = theme::color(th, ThemeToken::Accent);
        let bg = theme::color(th, ThemeToken::Bg);
        let cap = Style::default().fg(accent);
        vec![
            Span::styled("▐", cap),
            Span::styled(body, Style::default().bg(accent).fg(bg).add_modifier(Modifier::BOLD)),
            Span::styled("▌", cap),
        ]
    } else {
        let wall = theme::muted(th);
        let text = Style::default().fg(theme::color(th, ThemeToken::Fg));
        vec![
            Span::styled("▏", wall),
            Span::styled(body, text),
            Span::styled("▕", wall),
        ]
    }
}

/// Display width of [`button`] for `label`: two caps + two padding spaces + the
/// label itself.
pub fn button_width(label: &str) -> u16 {
    label.width() as u16 + 4
}

/// What a Confirm banner's "Confirm" button actually does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmAction {
    /// Quit the app (asked when managed runtimes would be stopped).
    Quit,
    /// Approve a destructive agent tool call (answers the pending approval).
    ToolApproval,
    /// Fetch + rebuild + swap the binary in the background.
    InstallUpdate,
    /// Run the platform installer for a backend (after showing the command).
    InstallBackend(BackendKind),
    /// Apply a diagnosed repair (after showing the exact commands it will run).
    ApplyRepair,
    /// Install the bundled local Bonsai assistant (llama.cpp + -hf Q4_1 pull).
    InstallLocalAssistant,
}

#[derive(Debug, Clone)]
pub enum ModalKind {
    Confirm {
        title: String,
        body: String,
        action: ConfirmAction,
    },
    Warning {
        title: String,
        body: String,
    },
    Error {
        error: LocalCodeError,
        /// A diagnosed, auto-applicable repair exists → show a `Fix` button.
        has_repair: bool,
    },
    Info {
        title: String,
        body: String,
    },
}

#[derive(Debug, Clone)]
pub struct ModalState {
    pub kind: ModalKind,
    pub selected: usize, // button index
}

impl ModalState {
    pub fn error(error: LocalCodeError, has_repair: bool) -> Self {
        Self {
            kind: ModalKind::Error { error, has_repair },
            selected: 0,
        }
    }

    pub fn warning(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            kind: ModalKind::Warning {
                title: title.into(),
                body: body.into(),
            },
            selected: 0,
        }
    }

    pub fn confirm(
        title: impl Into<String>,
        body: impl Into<String>,
        action: ConfirmAction,
    ) -> Self {
        Self {
            kind: ModalKind::Confirm {
                title: title.into(),
                body: body.into(),
                action,
            },
            // Default to the safe choice for confirmations.
            selected: 1,
        }
    }

    pub fn info(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            kind: ModalKind::Info {
                title: title.into(),
                body: body.into(),
            },
            selected: 0,
        }
    }

    /// Button labels. Non-retryable errors get no Retry button, so a button's
    /// label — not its index — is what handlers should match on.
    pub fn buttons(&self) -> Vec<&'static str> {
        match &self.kind {
            ModalKind::Error { error, has_repair } => {
                // Fix leads (the recommended action) when a repair is available;
                // Retry only when the operation itself is retryable.
                let mut b = Vec::new();
                if *has_repair {
                    b.push("Fix");
                }
                if error.retryable {
                    b.push("Retry");
                }
                b.extend(["Open logs", "Ask assistant", "Dismiss"]);
                b
            }
            ModalKind::Warning { .. } => vec!["Continue", "Cancel"],
            ModalKind::Confirm { .. } => vec!["Confirm", "Cancel"],
            ModalKind::Info { .. } => vec!["Dismiss"],
        }
    }

    pub fn selected_button(&self) -> &'static str {
        let buttons = self.buttons();
        buttons
            .get(self.selected.min(buttons.len().saturating_sub(1)))
            .copied()
            .unwrap_or("")
    }
}

/// Title + wrapped body content of a banner (no button row), styled per kind.
/// Errors render bold `Fg` (never red); everything else uses `Fg`/`Muted`.
fn content_rows(modal: &ModalState, th: &localcode_core::Theme) -> Vec<Line<'static>> {
    let fg = Style::default().fg(theme::color(th, ThemeToken::Fg));
    let bold = fg.add_modifier(Modifier::BOLD);
    match &modal.kind {
        ModalKind::Error { error, .. } => {
            let mut lines = vec![Line::from(Span::styled(
                format!("{}: {}", error.code, error.message),
                bold,
            ))];
            for c in &error.causes {
                lines.push(Line::from(Span::styled(format!("cause  {c}"), theme::muted(th))));
            }
            for h in &error.hints {
                lines.push(Line::from(Span::styled(format!("try    {h}"), theme::muted(th))));
            }
            lines
        }
        ModalKind::Warning { title, body } => banner_text(title, body, bold, th),
        ModalKind::Confirm { title, body, .. } => banner_text(title, body, bold, th),
        ModalKind::Info { title, body } => banner_text(title, body, bold, th),
    }
}

fn banner_text(
    title: &str,
    body: &str,
    title_style: Style,
    th: &localcode_core::Theme,
) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(title.to_string(), title_style))];
    for l in body.lines() {
        lines.push(Line::from(Span::styled(l.to_string(), theme::muted(th))));
    }
    lines
}

/// Rows the banner needs at `width`: wrapped content + one button row, capped so
/// a long error can't swallow the whole working area.
pub fn banner_height(modal: &ModalState, th: &localcode_core::Theme, width: u16) -> u16 {
    let w = width.saturating_sub(2).max(1) as usize;
    let body: usize = content_rows(modal, th)
        .iter()
        .map(|l| (l.width().max(1)).div_ceil(w))
        .sum();
    // content + blank + buttons + the bottom rule.
    (body as u16 + 3).clamp(3, 16)
}

/// Render the banner at the top of the working area. Returns each button's rect
/// with its index so the caller can register `ModalButton` click regions.
pub fn draw_inline_banner(
    f: &mut Frame,
    area: Rect,
    modal: &ModalState,
    th: &localcode_core::Theme,
) -> Vec<(Rect, usize)> {
    // A single thin rule separates the banner from the view below it.
    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_type(BorderType::Plain)
        .border_style(theme::border(th));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut rows = content_rows(modal, th);
    let button_y = inner.y + inner.height.saturating_sub(1);

    // Body fills all but the last inner row (reserved for buttons).
    let body_rect = Rect {
        x: inner.x,
        y: inner.y,
        width: inner.width,
        height: inner.height.saturating_sub(1),
    };
    // Drop content that would collide with the button row.
    let max_body = body_rect.height as usize;
    if rows.len() > max_body {
        rows.truncate(max_body);
    }
    f.render_widget(Paragraph::new(rows).wrap(Wrap { trim: false }), body_rect);

    // Inline pseudographic buttons; the selected one is filled, the rest outlined.
    let buttons = modal.buttons();
    let sel = modal.selected.min(buttons.len().saturating_sub(1));
    let mut spans: Vec<Span> = Vec::new();
    let mut hits: Vec<(Rect, usize)> = Vec::new();
    let mut x = inner.x;
    for (i, b) in buttons.iter().enumerate() {
        let label = b.to_lowercase();
        let w = button_width(&label);
        if x + w <= inner.x + inner.width {
            hits.push((Rect { x, y: button_y, width: w, height: 1 }, i));
        }
        spans.extend(button(th, &label, i == sel));
        spans.push(Span::raw(" "));
        x = x.saturating_add(w + 1);
    }
    f.render_widget(Paragraph::new(Line::from(spans)), Rect {
        x: inner.x,
        y: button_y,
        width: inner.width,
        height: 1,
    });
    hits
}
