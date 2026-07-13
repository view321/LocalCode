//! Shared TUI widgets: modal, command palette, status strip helpers.

use localcode_core::error::LocalCodeError;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap};
use ratatui::Frame;

use crate::theme;

#[derive(Debug, Clone)]
pub enum ModalKind {
    Confirm {
        title: String,
        body: String,
    },
    Warning {
        title: String,
        body: String,
    },
    Error {
        error: LocalCodeError,
    },
    Payment {
        title: String,
        body: String,
        amount: String,
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
    pub fn error(error: LocalCodeError) -> Self {
        Self {
            kind: ModalKind::Error { error },
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

    pub fn buttons(&self) -> Vec<&'static str> {
        match &self.kind {
            ModalKind::Error { .. } => vec!["Retry", "Open logs", "Ask assistant", "Dismiss"],
            ModalKind::Warning { .. } => vec!["Continue", "Cancel"],
            ModalKind::Confirm { .. } => vec!["Confirm", "Cancel"],
            ModalKind::Payment { .. } => vec!["Confirm pay", "Cancel"],
            ModalKind::Info { .. } => vec!["OK"],
        }
    }
}

pub fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

pub fn draw_modal(f: &mut Frame, area: Rect, modal: &ModalState, th: &localcode_core::Theme) {
    let rect = centered_rect(70, 60, area);
    f.render_widget(Clear, rect);

    let (title, body_lines, style) = match &modal.kind {
        ModalKind::Error { error } => {
            let mut lines = vec![
                Line::from(Span::styled(
                    format!("{}: {}", error.code, error.message),
                    theme::error(th).add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from(Span::styled("Possible causes:", theme::muted(th))),
            ];
            for c in &error.causes {
                lines.push(Line::from(format!("  • {c}")));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("What to try:", theme::muted(th))));
            for h in &error.hints {
                lines.push(Line::from(format!("  → {h}")));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(format!(
                "correlation_id: {}",
                error.correlation_id
            )));
            ("Error", lines, theme::error(th))
        }
        ModalKind::Warning { title, body } => (
            title.as_str(),
            vec![Line::from(body.as_str())],
            theme::warn(th),
        ),
        ModalKind::Confirm { title, body } => (
            title.as_str(),
            vec![Line::from(body.as_str())],
            theme::accent(th),
        ),
        ModalKind::Payment { title, body, amount } => (
            title.as_str(),
            vec![
                Line::from(body.as_str()),
                Line::from(""),
                Line::from(Span::styled(
                    format!("Amount: {amount}"),
                    theme::warn(th).add_modifier(Modifier::BOLD),
                )),
            ],
            theme::warn(th),
        ),
        ModalKind::Info { title, body } => (
            title.as_str(),
            vec![Line::from(body.as_str())],
            theme::accent(th),
        ),
    };

    let block = Block::default()
        .title(format!(" {title} "))
        .borders(Borders::ALL)
        .border_style(style);

    let inner = block.inner(rect);
    f.render_widget(block, rect);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(inner);

    f.render_widget(
        Paragraph::new(body_lines).wrap(Wrap { trim: true }),
        chunks[0],
    );

    let buttons = modal.buttons();
    let btn_line: Vec<Span> = buttons
        .iter()
        .enumerate()
        .flat_map(|(i, b)| {
            let selected = i == modal.selected;
            let st = if selected {
                Style::default()
                    .fg(theme::color(th, localcode_core::theme::ThemeToken::Bg))
                    .bg(theme::color(th, localcode_core::theme::ThemeToken::Accent))
                    .add_modifier(Modifier::BOLD)
            } else {
                theme::muted(th)
            };
            vec![
                Span::styled(format!(" [{b}] "), st),
                Span::raw(" "),
            ]
        })
        .collect();
    f.render_widget(Paragraph::new(Line::from(btn_line)), chunks[1]);
}

pub fn draw_palette(
    f: &mut Frame,
    area: Rect,
    query: &str,
    items: &[String],
    selected: usize,
    th: &localcode_core::Theme,
) {
    let rect = centered_rect(60, 50, area);
    f.render_widget(Clear, rect);
    let block = Block::default()
        .title(" Command palette (Ctrl+K) ")
        .borders(Borders::ALL)
        .border_style(theme::accent(th));
    let inner = block.inner(rect);
    f.render_widget(block, rect);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(1)])
        .split(inner);

    f.render_widget(
        Paragraph::new(format!("> {query}")).style(theme::accent(th)),
        chunks[0],
    );

    let list_items: Vec<ListItem> = items
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let style = if i == selected {
                theme::nav_active(th)
            } else {
                theme::muted(th)
            };
            ListItem::new(s.as_str()).style(style)
        })
        .collect();
    f.render_widget(List::new(list_items), chunks[1]);
}
