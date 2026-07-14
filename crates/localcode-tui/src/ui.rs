//! View rendering.
//!
//! Layout: a thin top status line (no app name/emoji), the home transcript
//! filling the middle, a one-line status/spinner, and a persistent omnibar at
//! the bottom. Panels (former tabs) open as popups over the transcript; the
//! omnibar stays visible and functional in every mode. Slash commands and a
//! command menu drive navigation.

use crate::app::{App, BusyKind, ClickRegion, ClickTarget, EntryKind, ModelsPane, Panel, ResizeBorder};
use crate::markdown;
use crate::theme;
use crate::widgets::draw_modal;
use localcode_core::events::Severity;
use localcode_core::runtime::RuntimeStatus;
use localcode_core::theme::ThemeToken;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Gauge, List, ListItem, Paragraph, Wrap};
use ratatui::Frame;
use unicode_width::UnicodeWidthStr;

const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Standard bordered pane. Focused panes get the accent border.
fn pane(app: &App, title: String, focused: bool) -> Block<'static> {
    let border = if focused {
        theme::accent(&app.theme)
    } else {
        theme::border(&app.theme)
    };
    let title_style = if focused {
        theme::accent(&app.theme).add_modifier(Modifier::BOLD)
    } else {
        theme::muted(&app.theme)
    };
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border)
        .title(Span::styled(title, title_style))
}

fn ratio_constraints(ratios: &[f32]) -> Vec<Constraint> {
    ratios
        .iter()
        .map(|r| Constraint::Ratio((r * 1000.0) as u32, 1000))
        .collect()
}

/// The area inside a pane's border.
fn inner_rect(a: Rect) -> Rect {
    Rect {
        x: a.x.saturating_add(1),
        y: a.y.saturating_add(1),
        width: a.width.saturating_sub(2),
        height: a.height.saturating_sub(2),
    }
}

fn click(app: &mut App, rect: Rect, target: ClickTarget) {
    app.click_regions.push(ClickRegion { rect, target });
}

fn border(app: &mut App, x: u16, area: Rect, view: &'static str, idx: usize) {
    app.resize_borders.push(ResizeBorder {
        x,
        y0: area.y,
        y1: area.y.saturating_add(area.height),
        view,
        idx,
        area,
    });
}

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();

    app.click_regions.clear();
    app.resize_borders.clear();

    let base = Style::default()
        .bg(theme::color(&app.theme, ThemeToken::Bg))
        .fg(theme::color(&app.theme, ThemeToken::Fg));
    f.render_widget(Block::default().style(base), area);

    if area.width < 40 || area.height < 12 {
        f.render_widget(
            Paragraph::new("Terminal too small. Resize to continue.").style(theme::warn(&app.theme)),
            area,
        );
        return;
    }

    let composer_rows = app.config.ui.composer_rows.clamp(1, 10);
    let main = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),               // top status chips
            Constraint::Min(5),                  // body (transcript / panel)
            Constraint::Length(1),               // status / spinner line
            Constraint::Length(composer_rows + 2), // omnibar
        ])
        .split(area);

    draw_topbar(f, main[0], app);
    draw_home(f, main[1], app);
    draw_statusline(f, main[2], app);
    draw_omnibar(f, main[3], app);

    // Panels open as popups filling the body area, above the omnibar.
    if let Some(panel) = app.panel {
        draw_panel(f, main[1], app, panel);
    }

    // The backends manager overlays everything except the modal layer.
    if app.backends_open {
        draw_backend_manager(f, area, app);
    }

    // The slash-command menu anchors just above the omnibar.
    if app.slash_active() {
        draw_slash_menu(f, main[1], main[3].y, app);
    }

    // Top-most overlays.
    let modal_btns = if let Some(modal) = &app.modal {
        draw_modal(f, area, modal, &app.theme)
    } else {
        vec![]
    };
    for (rect, i) in modal_btns {
        click(app, rect, ClickTarget::ModalButton(i));
    }
    if app.assistant_open {
        draw_assistant_dock(f, area, app);
    }
}

// ---------------------------------------------------------------------------
// Chrome: top status, status line, omnibar
// ---------------------------------------------------------------------------

fn runtime_glyph(status: RuntimeStatus, th: &localcode_core::Theme) -> (&'static str, Style) {
    match status {
        RuntimeStatus::Healthy => ("●", theme::ok(th)),
        RuntimeStatus::Starting => ("◐", theme::warn(th)),
        RuntimeStatus::Unhealthy => ("◑", theme::error(th)),
        RuntimeStatus::Stopping => ("◌", theme::muted(th)),
        RuntimeStatus::Stopped => ("○", theme::muted(th)),
    }
}

/// Thin top line: runtime · gpu · api chips on the left, update badge right.
/// No app name or emoji — minimal by design.
fn draw_topbar(f: &mut Frame, area: Rect, app: &App) {
    let th = &app.theme;
    let sep = Span::styled("  ", theme::muted(th));
    let mut spans: Vec<Span> = vec![Span::raw(" ")];

    match app.active_runtime() {
        Some(rt) => {
            let (glyph, style) = runtime_glyph(rt.status, th);
            spans.push(Span::styled(format!("{glyph} "), style));
            spans.push(Span::styled(
                rt.name.clone(),
                Style::default().fg(theme::color(th, ThemeToken::Fg)),
            ));
        }
        None => spans.push(Span::styled("○ no runtime", theme::muted(th))),
    }
    spans.push(sep.clone());
    spans.push(Span::styled(app.gpu.summary(), theme::muted(th)));
    spans.push(sep);
    let (api_txt, api_style) = match app.api_healthy {
        None => ("api …", theme::muted(th)),
        Some(true) => ("api ✓", theme::ok(th)),
        Some(false) => ("api ✗ local-first", theme::muted(th)),
    };
    spans.push(Span::styled(api_txt, api_style));

    f.render_widget(Paragraph::new(Line::from(spans)), area);

    // Right-aligned update badge.
    let badge: Option<(String, Style)> = if let Some(v) = &app.update_installed {
        Some((
            format!("↻ v{v} installed — restart "),
            theme::ok(th).add_modifier(Modifier::BOLD),
        ))
    } else if app.update_busy.is_some() {
        Some((" updating… ".into(), theme::accent(th)))
    } else {
        app.update_available.as_ref().map(|info| {
            (
                format!("⬆ update v{} — /update ", info.latest),
                theme::warn(th).add_modifier(Modifier::BOLD),
            )
        })
    };
    if let Some((text, style)) = badge {
        let w = text.width() as u16;
        if area.width > w + 20 {
            let rect = Rect {
                x: area.x + area.width - w,
                y: area.y,
                width: w,
                height: 1,
            };
            f.render_widget(Paragraph::new(Span::styled(text, style)), rect);
        }
    }
}

/// One-line status/spinner above the omnibar.
fn draw_statusline(f: &mut Frame, area: Rect, app: &App) {
    let th = &app.theme;
    let (text, style) = if let Some(b) = &app.busy {
        let frame = SPINNER[(b.started.elapsed().as_millis() / 80) as usize % SPINNER.len()];
        let extra = if b.kind == BusyKind::Coding {
            " — tokens streaming".to_string()
        } else {
            String::new()
        };
        (
            format!(
                " {frame} {}{extra} ({}s) — Esc cancels",
                b.label,
                b.started.elapsed().as_secs()
            ),
            theme::accent(th),
        )
    } else if let Some(b) = &app.deploy_busy {
        let frame = SPINNER[(b.started.elapsed().as_millis() / 80) as usize % SPINNER.len()];
        (
            format!(
                " {frame} {} {}% ({}s) — Esc cancels",
                b.label,
                app.deploy_progress,
                b.started.elapsed().as_secs()
            ),
            theme::accent(th),
        )
    } else if let Some(b) = &app.install_busy {
        let frame = SPINNER[(b.started.elapsed().as_millis() / 80) as usize % SPINNER.len()];
        (
            format!(" {frame} {}: {}", b.label, app.install_progress_line),
            theme::accent(th),
        )
    } else if let Some(b) = &app.update_busy {
        let frame = SPINNER[(b.started.elapsed().as_millis() / 80) as usize % SPINNER.len()];
        (
            format!(" {frame} {}: {}", b.label, app.update_progress_line),
            theme::accent(th),
        )
    } else {
        let style = if app.status_is_error {
            theme::error(th)
        } else {
            theme::muted(th)
        };
        (format!(" {}", app.status_line), style)
    };
    f.render_widget(Paragraph::new(Line::from(Span::styled(text, style))), area);
}

/// The persistent omnibar. Typing chats with the agent; a leading '/' turns it
/// into a command entry (menu drawn above). It doubles as the model-search box
/// when the Models panel is open.
fn draw_omnibar(f: &mut Frame, area: Rect, app: &mut App) {
    click(app, area, ClickTarget::Composer);
    let th = &app.theme;
    let slash = app.slash_active();
    let style = if slash {
        theme::accent(th)
    } else {
        theme::border(th)
    };

    let placeholder = match app.panel {
        Some(Panel::Models) => "search HuggingFace models…  (Enter searches)",
        _ => "message the agent…  ( / for commands )",
    };

    let title = match app.panel {
        Some(Panel::Models) => " search ".to_string(),
        Some(p) => format!(" {} · type to chat ", p.title()),
        None => " prompt ".to_string(),
    };

    let mut spans: Vec<Span> = vec![Span::styled("❯ ", theme::accent(th))];
    if app.coding_input.is_empty() {
        spans.push(Span::styled(placeholder, theme::muted(th)));
    } else {
        let chars: Vec<char> = app.coding_input.chars().collect();
        let cur = app.coding_cursor.min(chars.len());
        let before: String = chars[..cur].iter().collect();
        let at: String = chars
            .get(cur)
            .map(|c| c.to_string())
            .unwrap_or_else(|| " ".into());
        let after: String = if cur < chars.len() {
            chars[cur + 1..].iter().collect()
        } else {
            String::new()
        };
        spans.push(Span::raw(before));
        spans.push(Span::styled(at, Style::default().add_modifier(Modifier::REVERSED)));
        spans.push(Span::raw(after));
    }

    f.render_widget(
        Paragraph::new(Line::from(spans)).wrap(Wrap { trim: false }).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(title)
                .border_style(style),
        ),
        area,
    );
}

/// The slash-command menu, anchored just above the omnibar.
fn draw_slash_menu(f: &mut Frame, body: Rect, omnibar_y: u16, app: &mut App) {
    let items = app.palette_items();
    let n = items.len().max(1) as u16;
    let height = (n + 2).min(body.height.max(3));
    let width = body.width.clamp(30, 72);
    let x = body.x + (body.width.saturating_sub(width)) / 2;
    let y = omnibar_y.saturating_sub(height);
    let rect = Rect {
        x,
        y,
        width,
        height,
    };
    f.render_widget(Clear, rect);
    let th = &app.theme;
    let block = Block::default()
        .title(" commands — ↑/↓ Enter · Esc ")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::accent(th));
    let inner = block.inner(rect);
    f.render_widget(block, rect);

    let sel = app.palette_selected.min(items.len().saturating_sub(1));
    let list_items: Vec<ListItem> = items
        .iter()
        .map(|s| ListItem::new(s.as_str()))
        .collect();
    let mut hits: Vec<(Rect, usize)> = Vec::new();
    for (i, _) in items.iter().enumerate() {
        let ry = inner.y.saturating_add(i as u16);
        if ry >= inner.y.saturating_add(inner.height) {
            break;
        }
        hits.push((
            Rect {
                x: inner.x,
                y: ry,
                width: inner.width,
                height: 1,
            },
            i,
        ));
    }
    let styled: Vec<ListItem> = list_items
        .into_iter()
        .enumerate()
        .map(|(i, it)| {
            if i == sel {
                it.style(theme::nav_active(th))
            } else {
                it.style(theme::muted(th))
            }
        })
        .collect();
    f.render_widget(List::new(styled), inner);
    for (rect, i) in hits {
        click(app, rect, ClickTarget::PaletteItem(i));
    }
}

// ---------------------------------------------------------------------------
// Home surface: the coding transcript
// ---------------------------------------------------------------------------

fn draw_home(f: &mut Frame, area: Rect, app: &mut App) {
    click(app, area, ClickTarget::Transcript);
    draw_transcript(f, area, app);
}

/// Transcript from structured entries, with scroll + follow.
fn draw_transcript(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    let inner_w = area.width.saturating_sub(2).max(1) as usize;
    let inner_h = area.height.saturating_sub(2);
    let agent_running = app
        .busy
        .as_ref()
        .is_some_and(|b| b.kind == BusyKind::Coding);

    let fg = Style::default().fg(theme::color(&th, ThemeToken::Fg));
    let mut lines: Vec<Line> = Vec::new();
    for (idx, e) in app.coding_transcript.iter().enumerate() {
        if e.kind == EntryKind::You && idx > 0 {
            lines.push(Line::from(""));
        }
        let (prefix, style) = match e.kind {
            EntryKind::You => ("❯ ", theme::accent(&th).add_modifier(Modifier::BOLD)),
            EntryKind::Agent => ("", fg),
            EntryKind::Tool => ("  ", theme::muted(&th).add_modifier(Modifier::ITALIC)),
            EntryKind::System => ("· ", theme::muted(&th)),
            EntryKind::Error => ("✗ ", theme::error(&th)),
        };
        let text_style = if e.kind == EntryKind::You { style } else { fg };
        let mut first = true;
        let part_count = e.text.split('\n').count();
        for (pi, part) in e.text.split('\n').enumerate() {
            let mut spans: Vec<Span> = Vec::new();
            if first && !prefix.is_empty() {
                spans.push(Span::styled(prefix.to_string(), style));
            } else if !first && !prefix.is_empty() {
                spans.push(Span::raw(" ".repeat(prefix.width())));
            }
            let content_style = match e.kind {
                EntryKind::Tool | EntryKind::System | EntryKind::Error => style,
                _ => text_style,
            };
            spans.push(Span::styled(part.to_string(), content_style));
            if e.live && agent_running && pi + 1 == part_count {
                spans.push(Span::styled("▌", theme::accent(&th)));
            }
            lines.push(Line::from(spans));
            first = false;
        }
    }

    let total: usize = lines.iter().map(|l| (l.width().max(1)).div_ceil(inner_w)).sum();
    app.coding_total_lines = total;
    app.coding_view_height = inner_h;
    let max_scroll = total.saturating_sub(inner_h as usize);
    let offset = if app.coding_follow {
        max_scroll
    } else {
        app.coding_scroll.min(max_scroll)
    };

    let title = if app.coding_follow {
        " conversation ".to_string()
    } else {
        " conversation — scrolled · End to follow ".to_string()
    };
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((offset as u16, 0))
            .block(pane(app, title, false)),
        area,
    );
}

// ---------------------------------------------------------------------------
// Panel frame + dispatch
// ---------------------------------------------------------------------------

/// Draw the popup frame for a panel and its content inside.
fn draw_panel(f: &mut Frame, area: Rect, app: &mut App, panel: Panel) {
    f.render_widget(Clear, area);
    let th = app.theme;
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::accent(&th))
        .title(Span::styled(
            format!(" {} ", panel.title()),
            theme::accent(&th).add_modifier(Modifier::BOLD),
        ))
        .title(Span::styled(" Esc ✕ ", theme::muted(&th)));
    let inner = block.inner(area);
    f.render_widget(block, area);

    // A close hit at the top-right of the frame.
    let close = Rect {
        x: area.x + area.width.saturating_sub(4),
        y: area.y,
        width: 3,
        height: 1,
    };
    click(app, close, ClickTarget::PanelClose);

    match panel {
        Panel::Models => draw_models(f, inner, app),
        Panel::Runtimes => draw_runtimes(f, inner, app),
        Panel::Benchmarks => draw_benchmarks(f, inner, app),
        Panel::Setup => draw_setup(f, inner, app),
        Panel::Notifications => draw_notifications(f, inner, app),
        Panel::Settings => draw_settings(f, inner, app),
        Panel::Remote => draw_remote(f, inner, app),
    }
}

fn human_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0f64.powi(3)))
}

// ---------------------------------------------------------------------------
// Runtimes panel (formerly Dashboard)
// ---------------------------------------------------------------------------

fn draw_runtimes(f: &mut Frame, area: Rect, app: &mut App) {
    let ratios = app.pane_ratios("dashboard", App::pane_defaults("dashboard"));
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(ratio_constraints(&ratios))
        .split(area);

    click(app, inner_rect(cols[0]), ClickTarget::RuntimeList);
    border(app, cols[1].x, area, "dashboard", 0);

    let th = app.theme;
    let runtimes = app.all_runtimes();
    let items: Vec<ListItem> = if runtimes.is_empty() {
        vec![
            ListItem::new("No active runtimes.").style(theme::muted(&th)),
            ListItem::new(""),
            ListItem::new("Deploy one with /models, or connect a remote GPU with /remote")
                .style(theme::muted(&th)),
        ]
    } else {
        runtimes
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let sel = i == app.runtime_selected;
                let (glyph, gstyle) = runtime_glyph(r.status, &th);
                let name_style = if sel {
                    theme::nav_active(&th)
                } else {
                    Style::default().fg(theme::color(&th, ThemeToken::Fg))
                };
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {glyph} "), gstyle),
                    Span::styled(r.name.clone(), name_style),
                    Span::styled(format!("  {:?} · {}", r.status, r.base_url), theme::muted(&th)),
                ]))
            })
            .collect()
    };
    let n = runtimes.len();
    app.runtime_list_state
        .select(if n == 0 { None } else { Some(app.runtime_selected.min(n - 1)) });
    f.render_stateful_widget(
        List::new(items).block(pane(app, " Runtimes — /stop to stop ".into(), true)),
        cols[0],
        &mut app.runtime_list_state,
    );

    let th = &app.theme;
    let mut right: Vec<Line> = Vec::new();
    right.push(Line::from(Span::styled(
        "GPU",
        theme::accent(th).add_modifier(Modifier::BOLD),
    )));
    right.push(Line::from(app.gpu.summary()));
    for w in &app.gpu.warnings {
        right.push(Line::from(Span::styled(w.as_str(), theme::warn(th))));
    }
    right.push(Line::from(""));
    right.push(Line::from(Span::styled(
        "Health",
        theme::accent(th).add_modifier(Modifier::BOLD),
    )));
    right.push(Line::from(format!(
        "API: {}",
        match app.api_healthy {
            None => "checking…",
            Some(true) => "ok",
            Some(false) => "offline (local-first OK)",
        }
    )));
    right.push(Line::from(format!(
        "Assistant: {}",
        if app.assistant_configured {
            "configured"
        } else {
            "not configured (/settings)"
        }
    )));
    if !app.remote_sessions.is_empty() {
        right.push(Line::from(""));
        right.push(Line::from(Span::styled(
            "Remote servers",
            theme::accent(th).add_modifier(Modifier::BOLD),
        )));
        for s in &app.remote_sessions {
            right.push(Line::from(vec![
                Span::styled("  ● ", theme::ok(th)),
                Span::styled(s.server_name.clone(), theme::muted(th)),
                Span::styled(format!("  {}", s.gpu.summary()), theme::muted(th)),
            ]));
        }
    }
    f.render_widget(
        Paragraph::new(right).wrap(Wrap { trim: true }).block(pane(app, " Overview ".into(), false)),
        cols[1],
    );
}

// ---------------------------------------------------------------------------
// Models panel
// ---------------------------------------------------------------------------

fn draw_models(f: &mut Frame, area: Rect, app: &mut App) {
    let stale = match (&app.model_detail, &app.card_cache) {
        (Some(d), Some(c)) => c.model_id != d.summary.id || c.mode != app.theme.mode,
        (Some(_), None) => true,
        _ => false,
    };
    if stale {
        if let Some(d) = &app.model_detail {
            let md = d
                .card_markdown
                .clone()
                .unwrap_or_else(|| "*(no model card — see quants and deploy panel)*".to_string());
            app.card_cache = Some(crate::app::CardCache {
                model_id: d.summary.id.clone(),
                mode: app.theme.mode,
                lines: markdown::render(&md, &app.theme),
            });
        }
    }

    let ratios = app.pane_ratios("models", App::pane_defaults("models"));
    let panes3 = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(ratio_constraints(&ratios))
        .split(area);

    border(app, panes3[1].x, area, "models", 0);
    border(app, panes3[2].x, area, "models", 1);

    draw_models_list(f, panes3[0], app);
    draw_models_card(f, panes3[1], app);
    draw_models_deploy(f, panes3[2], app);
}

fn draw_models_list(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    click(app, inner_rect(area), ClickTarget::ModelList);

    let items: Vec<ListItem> = if app.models.is_empty() {
        vec![
            ListItem::new("No results yet.").style(theme::muted(&th)),
            ListItem::new(""),
            ListItem::new("Type a query below and press Enter").style(theme::muted(&th)),
        ]
    } else {
        app.models
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let sel = i == app.model_selected;
                let name_style = if sel {
                    theme::nav_active(&th)
                } else {
                    Style::default().fg(theme::color(&th, ThemeToken::Fg))
                };
                let gguf = m.tags.iter().any(|t| t.eq_ignore_ascii_case("gguf"));
                let mut meta = format!(
                    "  ↓{} ♥{}",
                    human_count(m.downloads.unwrap_or(0)),
                    human_count(m.likes.unwrap_or(0))
                );
                if gguf {
                    meta.push_str(" · gguf");
                }
                ListItem::new(Line::from(vec![
                    Span::styled(m.id.clone(), name_style),
                    Span::styled(meta, theme::muted(&th)),
                ]))
            })
            .collect()
    };
    app.model_list_state
        .select(if app.models.is_empty() {
            None
        } else {
            Some(app.model_selected)
        });

    let focused = app.models_focus == ModelsPane::List;
    f.render_stateful_widget(
        List::new(items).block(pane(
            app,
            format!(" Models ({}) — click to open ", app.models.len()),
            focused,
        )),
        area,
        &mut app.model_list_state,
    );
}

fn draw_models_card(f: &mut Frame, area: Rect, app: &mut App) {
    click(app, area, ClickTarget::ModelCard);
    let th = &app.theme;
    let focused = app.models_focus == ModelsPane::Card;

    let Some(d) = &app.model_detail else {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled("  Select a model and press Enter.", theme::muted(th))),
                Line::from(""),
                Line::from(Span::styled(
                    "  The full model card renders here — click it to focus, scroll to read.",
                    theme::muted(th),
                )),
            ])
            .block(pane(app, " Model card ".into(), focused)),
            area,
        );
        return;
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(6), Constraint::Min(3)])
        .split(area);

    let s = &d.summary;
    let mut chip_line1: Vec<Span> = vec![Span::styled(
        s.id.clone(),
        theme::accent(th).add_modifier(Modifier::BOLD),
    )];
    if s.gated
        .as_ref()
        .is_some_and(|g| g != &serde_json::Value::Bool(false))
    {
        chip_line1.push(Span::styled("  ⚠ gated", theme::warn(th)));
    }

    let mut chips2 = format!(
        "↓{}  ♥{}",
        human_count(s.downloads.unwrap_or(0)),
        human_count(s.likes.unwrap_or(0))
    );
    if let Some(lic) = &d.license {
        chips2.push_str(&format!("  │ {lic}"));
    }
    if let Some(p) = &d.parameter_size {
        chips2.push_str(&format!("  │ {p}"));
    }
    if let Some(pt) = &s.pipeline_tag {
        chips2.push_str(&format!("  │ {pt}"));
    }
    chips2.push_str(&format!("  │ {} files", d.siblings.len()));

    let mut tags: Vec<&str> = s.tags.iter().map(String::as_str).take(6).collect();
    if s.tags.len() > 6 {
        tags.push("…");
    }
    let updated = s
        .last_modified
        .as_deref()
        .map(|t| t.split('T').next().unwrap_or(t).to_string())
        .unwrap_or_else(|| "?".into());

    let header = vec![
        Line::from(chip_line1),
        Line::from(Span::styled(chips2, Style::default().fg(theme::color(th, ThemeToken::Fg)))),
        Line::from(Span::styled(format!("tags: {}", tags.join(", ")), theme::muted(th))),
        Line::from(Span::styled(format!("updated: {updated}"), theme::muted(th))),
    ];
    f.render_widget(
        Paragraph::new(header).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(if focused {
                    theme::accent(th)
                } else {
                    theme::border(th)
                }),
        ),
        rows[0],
    );

    let inner_w = rows[1].width.saturating_sub(2).max(1) as usize;
    let inner_h = rows[1].height.saturating_sub(2);
    let lines: Vec<Line> = app
        .card_cache
        .as_ref()
        .map(|c| c.lines.clone())
        .unwrap_or_default();
    let total: usize = lines.iter().map(|l| (l.width().max(1)).div_ceil(inner_w)).sum();
    app.card_total_lines = total;
    app.card_view_height = inner_h;
    let max_scroll = total.saturating_sub(inner_h as usize);
    let offset = app.card_scroll.min(max_scroll);

    let pct = (offset * 100).checked_div(max_scroll).unwrap_or(100);
    let title = if focused {
        format!(" Model card — scroll · {pct}% ")
    } else {
        " Model card — click to focus ".to_string()
    };
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((offset as u16, 0))
            .block(pane(app, title, focused)),
        rows[1],
    );
}

fn draw_models_deploy(f: &mut Frame, area: Rect, app: &mut App) {
    // Register all click regions up front (they need `&mut app`) before taking
    // an immutable borrow of the theme. Later-registered regions win the
    // reverse hit-scan: DeployButton (last row) over QuantList over the pane.
    click(app, area, ClickTarget::DeployPane);
    let inner = inner_rect(area);
    if inner.height > 0 {
        click(
            app,
            Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 },
            ClickTarget::BackendCycle,
        );
    }
    if inner.height > 4 {
        click(
            app,
            Rect {
                x: inner.x,
                y: inner.y + 3,
                width: inner.width,
                height: inner.height.saturating_sub(4),
            },
            ClickTarget::QuantList,
        );
    }
    if inner.height >= 1 {
        let by = inner.y + inner.height.saturating_sub(1);
        click(
            app,
            Rect { x: inner.x, y: by, width: inner.width, height: 1 },
            ClickTarget::DeployButton,
        );
    }

    let th = &app.theme;
    let deploying = app.deploy_busy.is_some();
    let chunks = if deploying {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(1)])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3)])
            .split(area)
    };

    let mut lines: Vec<Line> = vec![Line::from(vec![
        Span::styled("Backend  ", theme::muted(th)),
        Span::styled(
            app.deploy_backend.as_str().to_string(),
            theme::accent(th).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ⟳ click", theme::muted(th)),
    ])];
    lines.push(Line::from(vec![
        Span::styled("Context  ", theme::muted(th)),
        Span::styled(
            format!("{}", app.deploy_ctx),
            Style::default().fg(theme::color(th, ThemeToken::Fg)),
        ),
    ]));
    lines.push(Line::from(""));

    if let Some(d) = &app.model_detail {
        lines.push(Line::from(Span::styled(
            format!("Quants ({}) — click to cycle", d.quants.len()),
            theme::muted(th),
        )));
        let selected_idx = d
            .quants
            .iter()
            .position(|q| Some(q.label.as_str()) == app.selected_quant.as_deref())
            .unwrap_or(0);
        let max_show = (area.height.saturating_sub(12) as usize).clamp(4, 10);
        let start = selected_idx
            .saturating_sub(max_show / 2)
            .min(d.quants.len().saturating_sub(max_show));
        for (i, q) in d.quants.iter().enumerate().skip(start).take(max_show) {
            let selected = i == selected_idx;
            let mark = if selected { "▶" } else { " " };
            let style = if selected {
                theme::accent(th).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme::color(th, ThemeToken::Fg))
            };
            let size = if q.total_size == 0 {
                "size ?".to_string()
            } else {
                gib(q.total_size)
            };
            lines.push(Line::from(Span::styled(
                format!("{mark} {}  {size} ({} files)", q.label, q.files.len()),
                style,
            )));
        }
    } else {
        lines.push(Line::from(Span::styled(
            "Open a model to pick a quant.",
            theme::muted(th),
        )));
    }

    if let Some(fit) = &app.last_fit {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("VRAM fit", theme::muted(th))));
        lines.push(Line::from(format!(
            "est {}   free {} / {}",
            gib(fit.estimated_vram_bytes),
            gib(fit.free_vram_bytes),
            gib(fit.total_vram_bytes),
        )));
        if let Some(w) = &fit.warning {
            lines.push(Line::from(Span::styled(w.clone(), theme::warn(th))));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "▶ Deploy  (click, or /deploy)",
        theme::accent(th).add_modifier(Modifier::BOLD),
    )));

    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(pane(app, " Deploy ".into(), false)),
        chunks[0],
    );

    if deploying {
        f.render_widget(
            Gauge::default()
                .percent(app.deploy_progress as u16)
                .label(format!("{}%", app.deploy_progress))
                .gauge_style(theme::accent(&app.theme)),
            chunks[1],
        );
    }
}

// ---------------------------------------------------------------------------
// Remote panel
// ---------------------------------------------------------------------------

fn draw_remote(f: &mut Frame, area: Rect, app: &mut App) {
    use crate::app::REMOTE_FIELDS;
    let th = app.theme;
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(22), Constraint::Min(24)])
        .split(area);

    // Left: saved servers + a "new" affordance.
    let list_block = Block::default()
        .borders(Borders::RIGHT)
        .border_style(theme::border(&th));
    let list_inner = list_block.inner(cols[0]);
    f.render_widget(list_block, cols[0]);
    click(app, list_inner, ClickTarget::RemoteList);

    let connected: Vec<String> = app.remote_sessions.iter().map(|s| s.server_name.clone()).collect();
    let mut items: Vec<ListItem> = Vec::new();
    for (i, s) in app.config.remote.servers.iter().enumerate() {
        let sel = i == app.remote_selected;
        let live = connected.contains(&s.name);
        let connecting = app.remote_connecting == Some(i);
        let (glyph, gstyle) = if connecting {
            ("◐", theme::warn(&th))
        } else if live {
            ("●", theme::ok(&th))
        } else {
            ("○", theme::muted(&th))
        };
        let name_style = if sel {
            theme::nav_active(&th)
        } else {
            Style::default().fg(theme::color(&th, ThemeToken::Fg))
        };
        items.push(ListItem::new(Line::from(vec![
            Span::styled(format!(" {glyph} "), gstyle),
            Span::styled(s.name.clone(), name_style),
        ])));
    }
    if items.is_empty() {
        items.push(ListItem::new(" (no servers) ").style(theme::muted(&th)));
    }
    f.render_widget(List::new(items), list_inner);
    // "+ New" row at the bottom of the list column.
    let newy = cols[0].y + cols[0].height.saturating_sub(1);
    let newrect = Rect { x: cols[0].x, y: newy, width: cols[0].width, height: 1 };
    f.render_widget(
        Paragraph::new(Span::styled(" + New server ", theme::accent(&th))),
        newrect,
    );
    click(app, newrect, ClickTarget::RemoteNew);

    // Right: the editable form for the selected server + actions.
    let right = Rect {
        x: cols[1].x + 1,
        y: cols[1].y,
        width: cols[1].width.saturating_sub(2),
        height: cols[1].height,
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(REMOTE_FIELDS.len() as u16 + 1),
            Constraint::Length(2),
            Constraint::Min(1),
        ])
        .split(right);

    let has_server = !app.config.remote.servers.is_empty();
    let mut field_lines: Vec<Line> = vec![Line::from(Span::styled(
        "Server — click a field to edit",
        theme::muted(&th),
    ))];
    if has_server {
        for (i, label) in REMOTE_FIELDS.iter().enumerate() {
            let selected = i == app.remote_field;
            let editing = selected && app.remote_editing;
            let raw = if editing {
                app.remote_field_edit.clone()
            } else {
                app.remote_field_value(i)
            };
            // Mask the password field when not editing.
            let shown = if *label == "password" && !editing {
                "•".repeat(raw.chars().count().min(12))
            } else if editing {
                format!("{raw}▌")
            } else {
                raw
            };
            let mark = if selected { "▶" } else { " " };
            let label_style = if selected {
                theme::accent(&th).add_modifier(Modifier::BOLD)
            } else {
                theme::muted(&th)
            };
            let val_style = if editing {
                theme::accent(&th)
            } else {
                Style::default().fg(theme::color(&th, ThemeToken::Fg))
            };
            field_lines.push(Line::from(vec![
                Span::styled(format!("{mark} {label:<10} "), label_style),
                Span::styled(shown, val_style),
            ]));
        }
    } else {
        field_lines.push(Line::from(Span::styled(
            "No server selected. Click '+ New server' to add one.",
            theme::muted(&th),
        )));
    }
    // Field rows are clickable (row 0 is the header, fields start at row 1).
    let field_rect = Rect {
        x: rows[0].x,
        y: rows[0].y + 1,
        width: rows[0].width,
        height: rows[0].height.saturating_sub(1),
    };
    click(app, field_rect, ClickTarget::RemoteField);
    f.render_widget(Paragraph::new(field_lines), rows[0]);

    // Action buttons row.
    let buttons = [
        ("[ Connect ]", ClickTarget::RemoteConnect),
        ("[ Save ]", ClickTarget::RemoteSave),
        ("[ Disconnect ]", ClickTarget::RemoteDisconnect),
        ("[ Delete ]", ClickTarget::RemoteDelete),
    ];
    let mut x = rows[1].x;
    let mut spans: Vec<Span> = Vec::new();
    for (label, target) in buttons {
        let w = label.width() as u16;
        if x + w <= rows[1].x + rows[1].width {
            click(app, Rect { x, y: rows[1].y, width: w, height: 1 }, target);
        }
        spans.push(Span::styled(label.to_string(), theme::accent(&th).add_modifier(Modifier::BOLD)));
        spans.push(Span::raw(" "));
        x = x.saturating_add(w + 1);
    }
    f.render_widget(Paragraph::new(Line::from(spans)), rows[1]);

    // Help / password warning.
    let help = vec![
        Line::from(Span::styled(
            "Connect installs & starts Ollama on the server (mirror-aware), then",
            theme::muted(&th),
        )),
        Line::from(Span::styled(
            "tunnels its port here so the agent uses the remote GPU.",
            theme::muted(&th),
        )),
        Line::from(Span::styled(
            "⚠ passwords are stored in plaintext in config.toml — prefer key_path.",
            theme::warn(&th),
        )),
        Line::from(Span::styled(
            "Quick add: /remote add <name> <host> <user> <password>",
            theme::muted(&th),
        )),
    ];
    f.render_widget(Paragraph::new(help).wrap(Wrap { trim: true }), rows[2]);
}

// ---------------------------------------------------------------------------
// Benchmarks / Setup / Notifications / Settings (reused, minus tab hints)
// ---------------------------------------------------------------------------

fn draw_benchmarks(f: &mut Frame, area: Rect, app: &App) {
    let th = &app.theme;
    let mut lines = vec![
        Line::from(Span::styled("Benchmarks", theme::accent(th).add_modifier(Modifier::BOLD))),
        Line::from("Suite: localcode-sample-coding v1.0.0"),
        Line::from(""),
        Line::from("Enter runs the sample suite on the active runtime."),
        Line::from(""),
    ];
    match &app.last_bench_result {
        Some(r) => {
            lines.push(Line::from(Span::styled(
                "Last run",
                theme::accent(th).add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::from(format!("  score      {:.2}", r.metrics.score)));
            lines.push(Line::from(format!("  pass rate  {:.0}%", r.metrics.pass_rate * 100.0)));
            lines.push(Line::from(format!("  latency    p50 {} ms", r.metrics.latency_p50_ms)));
        }
        None => lines.push(Line::from(Span::styled(app.last_bench_summary.clone(), theme::muted(th)))),
    }
    f.render_widget(Paragraph::new(lines).block(pane(app, " Benchmarks ".into(), false)), area);
}

fn draw_setup(f: &mut Frame, area: Rect, app: &mut App) {
    click(app, area, ClickTarget::SetupBody);
    let th = &app.theme;
    let mut lines = vec![
        Line::from(Span::styled("Setup", theme::accent(th).add_modifier(Modifier::BOLD))),
        Line::from("1. GPUs detected — see /runtimes"),
        Line::from(format!(
            "2. Backends: {}",
            if app.detecting { "(detecting…)" } else { "" }
        )),
    ];
    for b in &app.backend_reports {
        let (mark, style) = if b.ready {
            ("✓", theme::ok(th))
        } else {
            ("·", theme::muted(th))
        };
        lines.push(Line::from(Span::styled(
            format!(
                "   {mark} {:?} ready={} {}",
                b.kind,
                b.ready,
                b.notes.first().cloned().unwrap_or_default()
            ),
            style,
        )));
    }
    let manage_line_idx = lines.len();
    lines.push(Line::from(Span::styled(
        "   ▶ Manage & install backends (/backends)",
        theme::accent(th).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from("3. HF token: set HF_TOKEN env for gated models"));
    lines.push(Line::from(format!("   endpoint: {}", app.config.registry.endpoint)));
    if !app.config.registry.mirrors.is_empty() {
        lines.push(Line::from(format!(
            "   mirrors: {}",
            app.config.registry.mirrors.join(", ")
        )));
    }
    lines.push(Line::from("4. Assistant: OPENROUTER_API_KEY, or an OpenAI-compatible base_url"));
    lines.push(Line::from("5. Remote GPU: /remote to connect a server over SSH"));
    lines.push(Line::from(format!(
        "6. Updates: checked on startup ({}) — /update when available",
        if app.config.updates.check_on_startup { "on" } else { "off" }
    )));
    lines.push(Line::from(""));
    lines.push(Line::from("Enter/scroll — /doctor runs diagnostics · /backends manages backends"));
    if let Some(doc) = &app.doctor_summary {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Doctor report", theme::muted(th))));
        for l in doc.lines() {
            lines.push(Line::from(l.to_string()));
        }
    }

    let inner = inner_rect(area);
    if (manage_line_idx as u16) >= app.setup_scroll {
        let ry = inner.y + (manage_line_idx as u16 - app.setup_scroll);
        if ry < inner.y.saturating_add(inner.height) {
            click(
                app,
                Rect { x: inner.x, y: ry, width: inner.width, height: 1 },
                ClickTarget::SetupManageBackends,
            );
        }
    }
    f.render_widget(
        Paragraph::new(lines).scroll((app.setup_scroll, 0)).block(pane(app, " Setup & Doctor ".into(), false)),
        area,
    );
}

fn draw_notifications(f: &mut Frame, area: Rect, app: &mut App) {
    click(app, inner_rect(area), ClickTarget::NotificationList);
    let th = app.theme;
    let items: Vec<ListItem> = if app.notifications.is_empty() {
        vec![ListItem::new("No notifications").style(theme::muted(&th))]
    } else {
        app.notifications
            .iter()
            .rev()
            .enumerate()
            .map(|(i, n)| {
                let style = match n.severity {
                    Severity::Error => theme::error(&th),
                    Severity::Warn => theme::warn(&th),
                    Severity::Success => theme::ok(&th),
                    Severity::Info => theme::muted(&th),
                };
                let style = if i == app.notif_selected {
                    style.add_modifier(Modifier::BOLD | Modifier::REVERSED)
                } else {
                    style
                };
                ListItem::new(format!(
                    "{} [{:?}] {} — {}",
                    n.at.format("%H:%M:%S"),
                    n.severity,
                    n.title,
                    n.body
                ))
                .style(style)
            })
            .collect()
    };
    app.notif_list_state.select(if app.notifications.is_empty() {
        None
    } else {
        Some(app.notif_selected.min(app.notifications.len() - 1))
    });
    f.render_stateful_widget(
        List::new(items).block(pane(app, format!(" Notifications ({}) ", app.notifications.len()), true)),
        area,
        &mut app.notif_list_state,
    );
}

fn draw_settings(f: &mut Frame, area: Rect, app: &App) {
    let th = &app.theme;
    let head = |t: &str| Line::from(Span::styled(t.to_string(), theme::accent(th).add_modifier(Modifier::BOLD)));
    let lines = vec![
        head("Appearance"),
        Line::from(format!("  Theme: {:?}   (/theme cycles)", app.config.ui.theme)),
        Line::from(format!("  Mouse capture: {}", app.config.ui.mouse)),
        Line::from(format!("  Omnibar rows: {}   (Ctrl+↑/↓)", app.config.ui.composer_rows)),
        Line::from(""),
        head("Agent"),
        Line::from(format!("  Token streaming: {}   (agent.stream)", app.config.agent.stream)),
        Line::from(format!(
            "  Confirm destructive shell commands: {}",
            app.config.agent.confirm_destructive_tools
        )),
        Line::from(format!("  Allow cloud fallback: {}", app.config.agent.allow_cloud_fallback)),
        Line::from(""),
        head("Updates"),
        Line::from(format!(
            "  Check on startup: {}   Current: v{}",
            app.config.updates.check_on_startup,
            env!("CARGO_PKG_VERSION")
        )),
        Line::from(format!("  Source: {} ({})", app.config.updates.repo_url, app.config.updates.branch)),
        Line::from(""),
        head("Services"),
        Line::from(format!("  Default backend: {}", app.config.backends.default.kind)),
        Line::from(format!("  Registry: {}", app.config.registry.api_endpoint)),
        Line::from(format!("  HF mirrors: {}", if app.config.registry.mirrors.is_empty() { "none".into() } else { app.config.registry.mirrors.join(", ") })),
        Line::from(format!("  Remote servers: {}", app.config.remote.servers.len())),
        Line::from(""),
        head("Config file"),
        Line::from(format!("  {}", app.paths.config_file().display())),
        Line::from(""),
        Line::from(Span::styled(
            "Edit the file for options without a command, then Ctrl+S. Env overrides:",
            theme::muted(th),
        )),
        Line::from(Span::styled(
            "LOCALCODE_API_URL, LOCALCODE_HF_ENDPOINT, LOCALCODE_LOG_LEVEL, HF_TOKEN, OPENROUTER_API_KEY.",
            theme::muted(th),
        )),
    ];
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(pane(app, " Settings ".into(), false)),
        area,
    );
}

fn draw_assistant_dock(f: &mut Frame, area: Rect, app: &App) {
    use crate::widgets::centered_rect;
    let rect = centered_rect(80, 70, area);
    f.render_widget(Clear, rect);
    let body = app
        .assistant_reply
        .clone()
        .unwrap_or_else(|| "Ask the assistant about the last error (/assistant).".into());
    f.render_widget(
        Paragraph::new(body)
            .wrap(Wrap { trim: true })
            .scroll((app.assistant_scroll, 0))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Assistant (Esc close · ↑/↓ scroll) ")
                    .border_style(theme::accent(&app.theme)),
            ),
        rect,
    );
}

/// The backends manager overlay (install + configure each backend).
fn draw_backend_manager(f: &mut Frame, area: Rect, app: &mut App) {
    use crate::app::BACKEND_ORDER;
    use crate::widgets::centered_rect;

    let th = app.theme;
    let rect = centered_rect(85, 82, area);
    f.render_widget(Clear, rect);
    let outer = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(" Backends — install & configure · Esc close ")
        .border_style(theme::accent(&th));
    let inner = outer.inner(rect);
    f.render_widget(outer, rect);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(18), Constraint::Min(24)])
        .split(inner);

    let list_block = Block::default()
        .borders(Borders::RIGHT)
        .border_style(theme::border(&th));
    let list_inner = list_block.inner(cols[0]);
    f.render_widget(list_block, cols[0]);
    click(app, list_inner, ClickTarget::BackendMgrItem);

    let spin = app
        .install_busy
        .as_ref()
        .map(|b| SPINNER[(b.started.elapsed().as_millis() / 80) as usize % SPINNER.len()]);
    let mut items: Vec<ListItem> = Vec::new();
    for (i, kind) in BACKEND_ORDER.iter().enumerate() {
        let report = app.backend_reports.iter().find(|r| r.kind == *kind);
        let (glyph, gstyle) = if app.installing_kind == Some(*kind) {
            (spin.unwrap_or('◐').to_string(), theme::accent(&th))
        } else if report.map(|r| r.ready).unwrap_or(false) {
            ("✓".to_string(), theme::ok(&th))
        } else if report.map(|r| r.installed).unwrap_or(false) {
            ("◐".to_string(), theme::warn(&th))
        } else {
            ("·".to_string(), theme::muted(&th))
        };
        let name_style = if i == app.backend_sel {
            theme::nav_active(&th)
        } else {
            Style::default().fg(theme::color(&th, ThemeToken::Fg))
        };
        items.push(ListItem::new(Line::from(vec![
            Span::styled(format!(" {glyph} "), gstyle),
            Span::styled(kind.as_str().to_string(), name_style),
        ])));
    }
    f.render_widget(List::new(items), list_inner);

    let kind = app.backend_sel_kind();
    let labels = App::backend_field_labels(kind);
    let nfields = labels.len() as u16;
    let right = Rect {
        x: cols[1].x + 1,
        y: cols[1].y,
        width: cols[1].width.saturating_sub(2),
        height: cols[1].height,
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Length(nfields.max(1)),
            Constraint::Length(4),
            Constraint::Length(1),
            Constraint::Min(1),
        ])
        .split(right);

    let report = app.backend_reports.iter().find(|r| r.kind == kind).cloned();
    let status = match &report {
        Some(r) if r.ready => "ready",
        Some(r) if r.installed => "installed, not running",
        Some(_) => "not installed",
        None => "detecting…",
    };
    let mut info: Vec<Line> = vec![Line::from(vec![
        Span::styled(kind.as_str().to_string(), theme::accent(&th).add_modifier(Modifier::BOLD)),
        Span::styled(format!("  — {status}"), theme::muted(&th)),
    ])];
    if let Some(r) = &report {
        if let Some(v) = &r.version {
            info.push(Line::from(Span::styled(format!("version: {v}"), theme::muted(&th))));
        }
        if let Some(p) = &r.binary_path {
            info.push(Line::from(Span::styled(format!("path: {p}"), theme::muted(&th))));
        }
        if let Some(u) = &r.base_url {
            info.push(Line::from(Span::styled(format!("url:  {u}"), theme::muted(&th))));
        }
        if let Some(n) = r.notes.first() {
            info.push(Line::from(Span::styled(n.clone(), theme::muted(&th))));
        }
    } else {
        info.push(Line::from(Span::styled("press [r] to re-detect", theme::muted(&th))));
    }
    f.render_widget(Paragraph::new(info).wrap(Wrap { trim: true }), rows[0]);

    click(app, rows[1], ClickTarget::BackendMgrField);
    let mut field_lines: Vec<Line> = Vec::new();
    for (i, label) in labels.iter().enumerate() {
        let selected = i == app.backend_field;
        let editing = selected && app.backend_editing;
        let value = if editing {
            format!("{}▌", app.backend_field_edit)
        } else {
            app.backend_field_value(kind, i)
        };
        let mark = if selected { "▶" } else { " " };
        let label_style = if selected {
            theme::accent(&th).add_modifier(Modifier::BOLD)
        } else {
            theme::muted(&th)
        };
        let val_style = if editing {
            theme::accent(&th)
        } else {
            Style::default().fg(theme::color(&th, ThemeToken::Fg))
        };
        field_lines.push(Line::from(vec![
            Span::styled(format!("{mark} {label:<9} "), label_style),
            Span::styled(value, val_style),
        ]));
    }
    f.render_widget(Paragraph::new(field_lines), rows[1]);

    let preview = if app.backend_plan_preview.is_empty() {
        "—".to_string()
    } else {
        app.backend_plan_preview.clone()
    };
    f.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled("Install plan", theme::muted(&th))),
            Line::from(Span::styled(preview, Style::default().fg(theme::color(&th, ThemeToken::Fg)))),
        ])
        .wrap(Wrap { trim: true }),
        rows[2],
    );

    let buttons = [
        ("[i] Install", ClickTarget::BackendMgrInstall),
        ("[s] Save", ClickTarget::BackendMgrSave),
        ("[r] Re-detect", ClickTarget::BackendMgrRedetect),
    ];
    let mut x = rows[3].x;
    let mut spans: Vec<Span> = Vec::new();
    for (label, target) in buttons {
        let w = label.width() as u16;
        if x + w <= rows[3].x + rows[3].width {
            click(app, Rect { x, y: rows[3].y, width: w, height: 1 }, target);
        }
        spans.push(Span::styled(label.to_string(), theme::accent(&th).add_modifier(Modifier::BOLD)));
        spans.push(Span::raw("  "));
        x = x.saturating_add(w + 2);
    }
    f.render_widget(Paragraph::new(Line::from(spans)), rows[3]);

    let hint = if let Some(b) = &app.install_busy {
        format!("{}: {}", b.label, app.install_progress_line)
    } else if app.backend_editing {
        "editing — type, Enter/Esc to commit".to_string()
    } else {
        "Tab switch · ↑/↓ field · Enter/click edit · i install · s save · r re-detect".to_string()
    };
    let hint_style = if app.install_busy.is_some() {
        theme::accent(&th)
    } else {
        theme::muted(&th)
    };
    f.render_widget(Paragraph::new(Span::styled(hint, hint_style)).wrap(Wrap { trim: true }), rows[4]);
}
