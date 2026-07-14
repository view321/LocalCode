//! View rendering for all tabs.
//!
//! Layout: header (identity + live chips) · clickable tab strip · view ·
//! status bar. Views with multiple panes take their split from
//! `App::pane_ratios` so `[`/`]` (and `{`/`}`) resizing persists.

use crate::app::{
    App, BusyKind, ClickRegion, ClickTarget, EntryKind, ModelsPane, ResizeBorder, Tab,
};
use crate::markdown;
use crate::theme;
use crate::widgets::{draw_modal, draw_palette};
use localcode_core::events::Severity;
use localcode_core::runtime::RuntimeStatus;
use localcode_core::theme::ThemeToken;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Gauge, List, ListItem, Paragraph, Wrap};
use ratatui::Frame;
use unicode_width::UnicodeWidthStr;

const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Standard bordered pane. Focused panes get the accent border so the pane
/// that owns j/k/PgUp/PgDn is always visible.
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

/// The area inside a pane's border — where list rows and content actually
/// render. Used so a click maps to the right row (row 0 sits at `area.y + 1`).
fn inner_rect(a: Rect) -> Rect {
    Rect {
        x: a.x.saturating_add(1),
        y: a.y.saturating_add(1),
        width: a.width.saturating_sub(2),
        height: a.height.saturating_sub(2),
    }
}

/// Record a clickable region for this frame.
fn click(app: &mut App, rect: Rect, target: ClickTarget) {
    app.click_regions.push(ClickRegion { rect, target });
}

/// Record a draggable vertical seam between panes `idx` and `idx + 1` of
/// `view`, spanning the full height of `area` at column `x`.
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

    // Hit regions are rebuilt every frame. Clear before the small-terminal
    // early return so a shrink can never leave stale zones that fire on click.
    app.click_regions.clear();
    app.resize_borders.clear();

    // Paint the themed background/foreground over the whole frame so light
    // and high-contrast modes don't depend on the terminal's own colors.
    let base = Style::default()
        .bg(theme::color(&app.theme, ThemeToken::Bg))
        .fg(theme::color(&app.theme, ThemeToken::Fg));
    f.render_widget(Block::default().style(base), area);

    if area.width < 40 || area.height < 12 {
        f.render_widget(
            Paragraph::new("Terminal too small. Resize to continue.")
                .style(theme::warn(&app.theme)),
            area,
        );
        return;
    }

    let main = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Length(1), // tab strip
            Constraint::Min(5),    // view
            Constraint::Length(3), // status
        ])
        .split(area);

    draw_header(f, main[0], app);
    draw_tabs(f, main[1], app);
    draw_view(f, main[2], app);
    draw_status(f, main[3], app);

    // The backends manager sits under the modal layer so an install-confirm
    // dialog overlays it (and its regions, recorded later, win the scan).
    if app.backends_open {
        draw_backend_manager(f, area, app);
    }

    // Overlays are recorded last so their click regions win the reverse scan.
    let modal_btns = if let Some(modal) = &app.modal {
        draw_modal(f, area, modal, &app.theme)
    } else {
        vec![]
    };
    for (rect, i) in modal_btns {
        click(app, rect, ClickTarget::ModalButton(i));
    }

    if app.palette_open {
        let items = app.palette_items();
        let hits = draw_palette(
            f,
            area,
            &app.palette_query,
            &items,
            app.palette_selected,
            &app.theme,
        );
        for (rect, i) in hits {
            click(app, rect, ClickTarget::PaletteItem(i));
        }
    }
    if app.assistant_open {
        draw_assistant_dock(f, area, app);
    }
}

// ---------------------------------------------------------------------------
// Chrome: header, tab strip, status
// ---------------------------------------------------------------------------

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let th = &app.theme;
    let sep = Span::styled(" │ ", theme::border(th));

    let mut left: Vec<Span> = vec![
        Span::styled(
            " ⚡ LocalCode ",
            theme::accent(th).add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("v{}", env!("CARGO_PKG_VERSION")), theme::muted(th)),
        sep.clone(),
    ];

    // Runtime chip
    match app.active_runtime() {
        Some(rt) => {
            let (glyph, style) = runtime_glyph(rt.status, th);
            left.push(Span::styled(format!("{glyph} "), style));
            left.push(Span::styled(
                rt.name.clone(),
                Style::default().fg(theme::color(th, ThemeToken::Fg)),
            ));
        }
        None => left.push(Span::styled("○ no runtime", theme::muted(th))),
    }
    left.push(sep.clone());

    // GPU chip
    left.push(Span::styled(app.gpu.summary(), theme::muted(th)));
    left.push(sep);

    // API chip
    let (api_txt, api_style) = match app.api_healthy {
        None => ("api …", theme::muted(th)),
        Some(true) => ("api ✓", theme::ok(th)),
        Some(false) => ("api ✗ (local-first ok)", theme::muted(th)),
    };
    left.push(Span::styled(api_txt, api_style));

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border(th));
    let inner = block.inner(area);
    f.render_widget(block, area);
    f.render_widget(Paragraph::new(Line::from(left)), inner);

    // Right-aligned update badge, painted after so it always wins.
    let badge: Option<(String, Style)> = if let Some(v) = &app.update_installed {
        Some((
            format!("↻ v{v} installed — restart to apply "),
            theme::ok(th).add_modifier(Modifier::BOLD),
        ))
    } else if app.update_busy.is_some() {
        Some((" updating… (Esc cancels) ".into(), theme::accent(th)))
    } else {
        app.update_available.as_ref().map(|info| {
            (
                format!("⬆ update v{} — press u ", info.latest),
                theme::warn(th).add_modifier(Modifier::BOLD),
            )
        })
    };
    if let Some((text, style)) = badge {
        let w = text.width() as u16;
        if inner.width > w + 40 {
            let rect = Rect {
                x: inner.x + inner.width - w,
                y: inner.y,
                width: w,
                height: 1,
            };
            f.render_widget(Paragraph::new(Span::styled(text, style)), rect);
        }
    }
}

fn runtime_glyph(status: RuntimeStatus, th: &localcode_core::Theme) -> (&'static str, Style) {
    match status {
        RuntimeStatus::Healthy => ("●", theme::ok(th)),
        RuntimeStatus::Starting => ("◐", theme::warn(th)),
        RuntimeStatus::Unhealthy => ("◑", theme::error(th)),
        RuntimeStatus::Stopping => ("◌", theme::muted(th)),
        RuntimeStatus::Stopped => ("○", theme::muted(th)),
    }
}

fn tab_label(tab: Tab) -> &'static str {
    match tab {
        Tab::Dashboard => "home",
        Tab::Models => "models",
        Tab::Benchmarks => "bench",
        Tab::Coding => "coding",
        Tab::Setup => "setup",
        Tab::Notifications => "alerts",
        Tab::Settings => "settings",
    }
}

/// Tab strip. Also records hit ranges on `app` so mouse clicks map to tabs.
fn draw_tabs(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    app.tab_strip_row = area.y;
    app.tab_hit.clear();

    let mut spans: Vec<Span> = vec![Span::raw(" ")];
    let mut x = area.x + 1;
    for t in Tab::all() {
        let mut label = format!(" {} ", tab_label(t));
        if t == Tab::Notifications && !app.notifications.is_empty() {
            label = format!(" {}({}) ", tab_label(t), app.notifications.len());
        }
        let w = label.width() as u16;
        let active = app.tab == t;
        let hovered = app.tab_hover == Some(t) && app.config.ui.right_rail_hover_brightens;
        let style = if active {
            Style::default()
                .fg(theme::color(&th, ThemeToken::Bg))
                .bg(theme::color(&th, ThemeToken::Accent))
                .add_modifier(Modifier::BOLD)
        } else if hovered {
            theme::nav_hover(&th)
        } else {
            theme::nav_idle(&th)
        };
        app.tab_hit.push((x, x + w, t));
        spans.push(Span::styled(label, style));
        spans.push(Span::raw(" "));
        x += w + 1;
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn tab_hint(tab: Tab) -> &'static str {
    match tab {
        Tab::Dashboard => "click a runtime · drag border to resize · scroll · x stop · ? help",
        Tab::Models => {
            "click a model to open · click card · drag borders · scroll · d deploy · ? help"
        }
        Tab::Coding => "click composer to type · scroll transcript · n new · ? help",
        Tab::Benchmarks => "r run · ? help",
        Tab::Setup => "scroll · m manage backends · d doctor · r redetect · ? help",
        Tab::Notifications => "click or scroll to select · c clear · ? help",
        Tab::Settings => "t theme · Ctrl+S save · ? help",
    }
}

fn draw_status(f: &mut Frame, area: Rect, app: &App) {
    let th = &app.theme;
    let (mut text, style) = if let Some(b) = &app.busy {
        let frame = SPINNER[(b.started.elapsed().as_millis() / 80) as usize % SPINNER.len()];
        let label = if b.kind == BusyKind::Coding {
            format!("{} — tokens stream into the transcript", b.label)
        } else {
            b.label.clone()
        };
        (
            format!(
                "{frame} {} ({}s) — Esc cancels",
                label,
                b.started.elapsed().as_secs()
            ),
            theme::accent(th),
        )
    } else if let Some(b) = &app.deploy_busy {
        let frame = SPINNER[(b.started.elapsed().as_millis() / 80) as usize % SPINNER.len()];
        (
            format!(
                "{frame} {} {}% ({}s) — Esc cancels",
                b.label,
                app.deploy_progress,
                b.started.elapsed().as_secs()
            ),
            theme::accent(th),
        )
    } else if let Some(b) = &app.update_busy {
        let frame = SPINNER[(b.started.elapsed().as_millis() / 80) as usize % SPINNER.len()];
        (
            format!(
                "{frame} {}: {} ({}s) — Esc cancels",
                b.label,
                app.update_progress_line,
                b.started.elapsed().as_secs()
            ),
            theme::accent(th),
        )
    } else if let Some(b) = &app.install_busy {
        let frame = SPINNER[(b.started.elapsed().as_millis() / 80) as usize % SPINNER.len()];
        (
            format!(
                "{frame} {}: {} ({}s) — Esc cancels",
                b.label,
                app.install_progress_line,
                b.started.elapsed().as_secs()
            ),
            theme::accent(th),
        )
    } else {
        let style = if app.status_is_error {
            theme::error(th)
        } else {
            theme::muted(th)
        };
        (app.status_line.clone(), style)
    };

    // Right-aligned per-tab key hints, left text truncated to fit.
    let hint = tab_hint(app.tab);
    let inner_w = area.width.saturating_sub(2) as usize;
    let hint_w = hint.width();
    if inner_w > hint_w + 3 {
        let max_text = inner_w - hint_w - 3;
        if text.width() > max_text {
            text = text
                .chars()
                .scan(0usize, |w, c| {
                    *w += c.to_string().width();
                    if *w > max_text.saturating_sub(1) {
                        None
                    } else {
                        Some(c)
                    }
                })
                .collect::<String>()
                + "…";
        }
        let pad = inner_w.saturating_sub(text.width() + hint_w);
        text = format!("{}{}{}", text, " ".repeat(pad), hint);
    }

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(text, style))).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(theme::border(th)),
        ),
        area,
    );
}

fn draw_view(f: &mut Frame, area: Rect, app: &mut App) {
    match app.tab {
        Tab::Dashboard => draw_dashboard(f, area, app),
        Tab::Models => draw_models(f, area, app),
        Tab::Benchmarks => draw_benchmarks(f, area, app),
        Tab::Coding => draw_coding(f, area, app),
        Tab::Setup => draw_setup(f, area, app),
        Tab::Notifications => draw_notifications(f, area, app),
        Tab::Settings => draw_settings(f, area, app),
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
// Dashboard
// ---------------------------------------------------------------------------

fn draw_dashboard(f: &mut Frame, area: Rect, app: &mut App) {
    let ratios = app.pane_ratios("dashboard", App::pane_defaults("dashboard"));
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(ratio_constraints(&ratios))
        .split(area);

    click(app, inner_rect(cols[0]), ClickTarget::RuntimeList);
    border(app, cols[1].x, area, "dashboard", 0);

    let th = app.theme;
    let items: Vec<ListItem> = if app.runtimes.is_empty() {
        vec![
            ListItem::new("No active runtimes.").style(theme::muted(&th)),
            ListItem::new(""),
            ListItem::new("Deploy a model from the models tab — press d there")
                .style(theme::muted(&th)),
        ]
    } else {
        app.runtimes
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
                let line = Line::from(vec![
                    Span::styled(format!(" {glyph} "), gstyle),
                    Span::styled(r.name.clone(), name_style),
                    Span::styled(
                        format!("  {:?} · {}", r.status, r.base_url),
                        theme::muted(&th),
                    ),
                ]);
                ListItem::new(line)
            })
            .collect()
    };
    app.runtime_list_state.select(if app.runtimes.is_empty() {
        None
    } else {
        Some(app.runtime_selected)
    });

    f.render_stateful_widget(
        List::new(items).block(pane(
            app,
            " Runtimes — click or scroll · x stop ".into(),
            true,
        )),
        cols[0],
        &mut app.runtime_list_state,
    );

    let th = &app.theme;
    let mut right: Vec<Line> = Vec::new();
    if let Some(v) = &app.update_installed {
        right.push(Line::from(Span::styled(
            format!("↻ v{v} installed — restart LocalCode to apply"),
            theme::ok(th).add_modifier(Modifier::BOLD),
        )));
        right.push(Line::from(""));
    } else if let Some(info) = &app.update_available {
        right.push(Line::from(Span::styled(
            format!("⬆ Update v{} available — press u to install", info.latest),
            theme::warn(th).add_modifier(Modifier::BOLD),
        )));
        right.push(Line::from(""));
    }
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
            "not configured (see the setup tab)"
        }
    )));
    right.push(Line::from(""));
    right.push(Line::from(Span::styled(
        "Quick actions",
        theme::accent(th).add_modifier(Modifier::BOLD),
    )));
    right.push(Line::from(
        "  models: deploy a model    coding: code with it",
    ));
    right.push(Line::from(
        "  bench: benchmark          setup: setup & doctor",
    ));
    right.push(Line::from("  Ctrl+K command palette"));
    if !app.notifications.is_empty() {
        right.push(Line::from(""));
        right.push(Line::from(Span::styled(
            "Recent",
            theme::accent(th).add_modifier(Modifier::BOLD),
        )));
        for n in app.notifications.iter().rev().take(5) {
            let style = match n.severity {
                Severity::Error => theme::error(th),
                Severity::Warn => theme::warn(th),
                Severity::Success => theme::ok(th),
                Severity::Info => theme::muted(th),
            };
            right.push(Line::from(vec![
                Span::styled("  • ", style),
                Span::styled(n.title.clone(), theme::muted(th)),
            ]));
        }
    }

    f.render_widget(
        Paragraph::new(right).wrap(Wrap { trim: true }).block(pane(
            app,
            " Overview ".into(),
            false,
        )),
        cols[1],
    );
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

fn draw_models(f: &mut Frame, area: Rect, app: &mut App) {
    // Refresh the memoized card render if the model or theme changed.
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
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(3)])
        .split(area);

    click(app, left[0], ClickTarget::ModelSearch);
    click(app, inner_rect(left[1]), ClickTarget::ModelList);

    let search_style = if app.model_search_focus {
        theme::accent(&th)
    } else {
        theme::muted(&th)
    };
    f.render_widget(
        Paragraph::new(format!(
            "{}{}",
            app.model_query,
            if app.model_search_focus { "▌" } else { "" }
        ))
        .style(search_style)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Search — click to type · p popular · t trending ")
                .border_style(search_style),
        ),
        left[0],
    );

    let items: Vec<ListItem> = if app.models.is_empty() {
        vec![
            ListItem::new("No results yet.").style(theme::muted(&th)),
            ListItem::new(""),
            ListItem::new("  / search HF   p popular   t trending").style(theme::muted(&th)),
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
    app.model_list_state.select(if app.models.is_empty() {
        None
    } else {
        Some(app.model_selected)
    });

    let focused = app.models_focus == ModelsPane::List && !app.model_search_focus;
    f.render_stateful_widget(
        List::new(items).block(pane(
            app,
            format!(" Models ({}) — click to open ", app.models.len()),
            focused,
        )),
        left[1],
        &mut app.model_list_state,
    );
}

/// Metadata chips + rendered markdown card, scrollable.
fn draw_models_card(f: &mut Frame, area: Rect, app: &mut App) {
    click(app, area, ClickTarget::ModelCard);
    let th = &app.theme;
    let focused = app.models_focus == ModelsPane::Card;

    let Some(d) = &app.model_detail else {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled(
                    "  Select a model and press Enter.",
                    theme::muted(th),
                )),
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

    // --- Chips header
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
        Line::from(Span::styled(
            chips2,
            Style::default().fg(theme::color(th, ThemeToken::Fg)),
        )),
        Line::from(Span::styled(
            format!("tags: {}", tags.join(", ")),
            theme::muted(th),
        )),
        Line::from(Span::styled(
            format!("updated: {updated}"),
            theme::muted(th),
        )),
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

    // --- Markdown body (memoized in app.card_cache)
    let inner_w = rows[1].width.saturating_sub(2).max(1) as usize;
    let inner_h = rows[1].height.saturating_sub(2);
    let lines: Vec<Line> = app
        .card_cache
        .as_ref()
        .map(|c| c.lines.clone())
        .unwrap_or_default();
    let total: usize = lines
        .iter()
        .map(|l| (l.width().max(1)).div_ceil(inner_w))
        .sum();
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
    click(app, area, ClickTarget::DeployPane);
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
        Span::styled("   [b] cycle", theme::muted(th)),
    ])];
    lines.push(Line::from(vec![
        Span::styled("Context  ", theme::muted(th)),
        Span::styled(
            format!("{}", app.deploy_ctx),
            Style::default().fg(theme::color(th, ThemeToken::Fg)),
        ),
        Span::styled("   [+/-] adjust", theme::muted(th)),
    ]));
    lines.push(Line::from(""));

    if let Some(d) = &app.model_detail {
        lines.push(Line::from(Span::styled(
            format!("Quants ({})  [,/.] pick", d.quants.len()),
            theme::muted(th),
        )));
        let selected_idx = d
            .quants
            .iter()
            .position(|q| Some(q.label.as_str()) == app.selected_quant.as_deref())
            .unwrap_or(0);
        // Window the list around the selection so long quant lists stay usable.
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
        if d.quants.len() > start + max_show {
            lines.push(Line::from(Span::styled(
                format!("  … {} more", d.quants.len() - start - max_show),
                theme::muted(th),
            )));
        }
    } else {
        lines.push(Line::from(Span::styled(
            "Open a model (Enter) to pick a quant.",
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
        "[d] one-click deploy",
        theme::accent(th).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(Span::styled(
        "[ ] and { } resize panes",
        theme::muted(th),
    )));

    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(pane(app, " Deploy ".into(), false)),
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
// Coding
// ---------------------------------------------------------------------------

fn draw_coding(f: &mut Frame, area: Rect, app: &mut App) {
    let composer_rows = app.config.ui.composer_rows.clamp(1, 10);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(composer_rows + 2),
        ])
        .split(area);

    click(app, chunks[1], ClickTarget::Transcript);
    click(app, chunks[2], ClickTarget::Composer);

    let th = &app.theme;
    let stream_chip = if app.config.agent.stream {
        Span::styled("streaming ✓", theme::ok(th))
    } else {
        Span::styled("streaming off", theme::muted(th))
    };
    let header = Line::from(vec![
        Span::styled("Runtime ", theme::muted(th)),
        Span::styled(
            app.active_runtime_name().unwrap_or_else(|| "none".into()),
            Style::default().fg(theme::color(th, ThemeToken::Fg)),
        ),
        Span::styled(" │ ", theme::border(th)),
        stream_chip,
        Span::styled(" │ ", theme::border(th)),
        Span::styled(
            format!(
                "skills {} · mcp {}",
                app.skill_count, app.mcp_status_summary
            ),
            theme::muted(th),
        ),
    ]);
    f.render_widget(
        Paragraph::new(header).block(pane(app, " Coding ".into(), false)),
        chunks[0],
    );

    draw_transcript(f, chunks[1], app);

    // Composer with a real cursor.
    let composer_style = if app.coding_input_focus {
        theme::accent(&app.theme)
    } else {
        theme::border(&app.theme)
    };
    let mut spans: Vec<Span> = vec![Span::styled("❯ ", theme::accent(&app.theme))];
    if app.coding_input_focus {
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
        spans.push(Span::styled(
            at,
            Style::default().add_modifier(Modifier::REVERSED),
        ));
        spans.push(Span::raw(after));
    } else {
        spans.push(Span::raw(app.coding_input.clone()));
    }
    f.render_widget(
        Paragraph::new(Line::from(spans))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Composer — click to type · Enter send · ↑ history ")
                    .border_style(composer_style),
            ),
        chunks[2],
    );
}

/// Transcript from structured entries, with scroll + follow. Wrapped line
/// counts are estimated so PgUp/PgDn and auto-follow know the bounds.
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
        // Breathing room before each user turn.
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
                EntryKind::Tool | EntryKind::System => style,
                EntryKind::Error => style,
                _ => text_style,
            };
            spans.push(Span::styled(part.to_string(), content_style));
            // Live streaming cursor on the entry's last line.
            if e.live && agent_running && pi + 1 == part_count {
                spans.push(Span::styled("▌", theme::accent(&th)));
            }
            lines.push(Line::from(spans));
            first = false;
        }
    }

    let total: usize = lines
        .iter()
        .map(|l| (l.width().max(1)).div_ceil(inner_w))
        .sum();
    app.coding_total_lines = total;
    app.coding_view_height = inner_h;
    let max_scroll = total.saturating_sub(inner_h as usize);
    let offset = if app.coding_follow {
        max_scroll
    } else {
        app.coding_scroll.min(max_scroll)
    };

    let title = if app.coding_follow {
        " Transcript ".to_string()
    } else {
        " Transcript — scrolled · End to follow ".to_string()
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
// Benchmarks / Setup / Notifications / Settings
// ---------------------------------------------------------------------------

fn draw_benchmarks(f: &mut Frame, area: Rect, app: &App) {
    let th = &app.theme;
    let mut lines = vec![
        Line::from(Span::styled(
            "Benchmarks",
            theme::accent(th).add_modifier(Modifier::BOLD),
        )),
        Line::from("Suite: localcode-sample-coding v1.0.0"),
        Line::from(""),
        Line::from("[r] Run sample suite on the selected runtime"),
        Line::from("[p] Publish last result (requires sign-in)"),
        Line::from(""),
    ];
    match &app.last_bench_result {
        Some(r) => {
            lines.push(Line::from(Span::styled(
                "Last run",
                theme::accent(th).add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::from(format!("  score      {:.2}", r.metrics.score)));
            lines.push(Line::from(format!(
                "  pass rate  {:.0}%",
                r.metrics.pass_rate * 100.0
            )));
            lines.push(Line::from(format!(
                "  latency    p50 {} ms",
                r.metrics.latency_p50_ms
            )));
        }
        None => lines.push(Line::from(Span::styled(
            app.last_bench_summary.clone(),
            theme::muted(th),
        ))),
    }
    f.render_widget(
        Paragraph::new(lines).block(pane(app, " Benchmarks ".into(), false)),
        area,
    );
}

fn draw_setup(f: &mut Frame, area: Rect, app: &mut App) {
    click(app, area, ClickTarget::SetupBody);
    let th = &app.theme;
    let mut lines = vec![
        Line::from(Span::styled(
            "Setup",
            theme::accent(th).add_modifier(Modifier::BOLD),
        )),
        Line::from("1. GPUs detected — see Dashboard"),
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
        "   [m] Manage & install backends — configure URL/bin/port · one-click install",
        theme::accent(th).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from("3. HF token: set HF_TOKEN env for gated models"));
    lines.push(Line::from(format!(
        "   endpoint: {}",
        app.config.registry.endpoint
    )));
    lines.push(Line::from(
        "4. Assistant: OPENROUTER_API_KEY, or an OpenAI-compatible base_url",
    ));
    lines.push(Line::from(
        "5. Cloud providers (RunPod/Vast/Akash): not implemented yet",
    ));
    lines.push(Line::from(format!(
        "6. Updates: checked on startup ({}) — press u when one is available,",
        if app.config.updates.check_on_startup {
            "on"
        } else {
            "off"
        }
    )));
    lines.push(Line::from("   or run `localcode update` in a terminal"));
    lines.push(Line::from(""));
    lines.push(Line::from(
        "[m] Manage backends   [d] Run doctor   [r] Refresh detection   PgUp/PgDn scroll",
    ));
    if let Some(doc) = &app.doctor_summary {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Doctor report", theme::muted(th))));
        for l in doc.lines() {
            lines.push(Line::from(l.to_string()));
        }
    }

    // Make the "Manage backends" line clickable (recorded after SetupBody so it
    // wins the reverse-scan for that row). Skipped when scrolled out of view.
    let inner = inner_rect(area);
    if (manage_line_idx as u16) >= app.setup_scroll {
        let ry = inner.y + (manage_line_idx as u16 - app.setup_scroll);
        if ry < inner.y.saturating_add(inner.height) {
            click(
                app,
                Rect {
                    x: inner.x,
                    y: ry,
                    width: inner.width,
                    height: 1,
                },
                ClickTarget::SetupManageBackends,
            );
        }
    }

    f.render_widget(
        Paragraph::new(lines)
            .scroll((app.setup_scroll, 0))
            .block(pane(app, " Setup ".into(), false)),
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
    app.notif_list_state
        .select(if app.notifications.is_empty() {
            None
        } else {
            Some(app.notif_selected.min(app.notifications.len() - 1))
        });
    f.render_stateful_widget(
        List::new(items).block(pane(
            app,
            format!(" Notifications ({}) — c clear ", app.notifications.len()),
            true,
        )),
        area,
        &mut app.notif_list_state,
    );
}

fn draw_settings(f: &mut Frame, area: Rect, app: &App) {
    let th = &app.theme;
    let head = |t: &str| {
        Line::from(Span::styled(
            t.to_string(),
            theme::accent(th).add_modifier(Modifier::BOLD),
        ))
    };
    let lines = vec![
        head("Appearance"),
        Line::from(format!("  Theme: {:?}   [t] cycle", app.config.ui.theme)),
        Line::from(format!("  Mouse capture: {}", app.config.ui.mouse)),
        Line::from(format!(
            "  Composer rows: {}   (Ctrl+↑/↓ on Coding)",
            app.config.ui.composer_rows
        )),
        Line::from(""),
        head("Agent"),
        Line::from(format!(
            "  Token streaming: {}   (agent.stream)",
            app.config.agent.stream
        )),
        Line::from(format!(
            "  Confirm destructive shell commands: {}",
            app.config.agent.confirm_destructive_tools
        )),
        Line::from(format!(
            "  Allow cloud fallback for Coding: {}",
            app.config.agent.allow_cloud_fallback
        )),
        Line::from(""),
        head("Updates"),
        Line::from(format!(
            "  Check on startup: {}   Current: v{}",
            app.config.updates.check_on_startup,
            env!("CARGO_PKG_VERSION")
        )),
        Line::from(format!(
            "  Source: {} ({})",
            app.config.updates.repo_url, app.config.updates.branch
        )),
        Line::from(""),
        head("Services"),
        Line::from(format!(
            "  Default backend: {}",
            app.config.backends.default.kind
        )),
        Line::from(format!("  Log level: {}", app.config.log_level())),
        Line::from(format!(
            "  Redact secrets in logs: {}",
            app.config.logging.redact_secrets
        )),
        Line::from(format!("  API: {}", app.config.api_base_url())),
        Line::from(format!("  Registry: {}", app.config.registry.api_endpoint)),
        Line::from(""),
        head("Config file"),
        Line::from(format!("  {}", app.paths.config_file().display())),
        Line::from(""),
        Line::from(Span::styled(
            "Edit the file for options without a keybinding, then Ctrl+S here",
            theme::muted(th),
        )),
        Line::from(Span::styled(
            "or restart. Env overrides: LOCALCODE_API_URL, LOCALCODE_HF_ENDPOINT,",
            theme::muted(th),
        )),
        Line::from(Span::styled(
            "LOCALCODE_LOG_LEVEL, LOCALCODE_INSTALL_DIR, HF_TOKEN, OPENROUTER_API_KEY.",
            theme::muted(th),
        )),
    ];
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(pane(
            app,
            " Settings ".into(),
            false,
        )),
        area,
    );
}

fn draw_assistant_dock(f: &mut Frame, area: Rect, app: &App) {
    use crate::widgets::centered_rect;
    use ratatui::widgets::Clear;
    let rect = centered_rect(80, 70, area);
    f.render_widget(Clear, rect);
    let body = app
        .assistant_reply
        .clone()
        .unwrap_or_else(|| "Ask assistant about the last error, or press a anywhere.".into());
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

/// The backends manager: install (with prerequisites) and configure each
/// backend. Left column lists the four backends with live status; the right
/// column shows detection detail, editable config fields, the resolved install
/// plan, and Install / Save / Re-detect actions.
fn draw_backend_manager(f: &mut Frame, area: Rect, app: &mut App) {
    use crate::app::BACKEND_ORDER;
    use crate::widgets::centered_rect;
    use ratatui::widgets::Clear;

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

    // --- Left column: the four backends with a status glyph.
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

    // --- Right column: detail for the selected backend.
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
            Constraint::Length(7),           // detection
            Constraint::Length(nfields.max(1)), // config fields
            Constraint::Length(4),           // install plan
            Constraint::Length(1),           // action buttons
            Constraint::Min(1),              // progress / hint
        ])
        .split(right);

    // Detection detail (owned clone so no borrow lingers across click()).
    let report = app.backend_reports.iter().find(|r| r.kind == kind).cloned();
    let status = match &report {
        Some(r) if r.ready => "ready",
        Some(r) if r.installed => "installed, not running",
        Some(_) => "not installed",
        None => "detecting…",
    };
    let mut info: Vec<Line> = vec![Line::from(vec![
        Span::styled(
            kind.as_str().to_string(),
            theme::accent(&th).add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("  — {status}"), theme::muted(&th)),
    ])];
    if let Some(r) = &report {
        if let Some(v) = &r.version {
            info.push(Line::from(Span::styled(
                format!("version: {v}"),
                theme::muted(&th),
            )));
        }
        if let Some(p) = &r.binary_path {
            info.push(Line::from(Span::styled(
                format!("path: {p}"),
                theme::muted(&th),
            )));
        }
        if let Some(u) = &r.base_url {
            info.push(Line::from(Span::styled(
                format!("url:  {u}"),
                theme::muted(&th),
            )));
        }
        if let Some(n) = r.notes.first() {
            info.push(Line::from(Span::styled(n.clone(), theme::muted(&th))));
        }
    } else {
        info.push(Line::from(Span::styled(
            "press [r] to re-detect",
            theme::muted(&th),
        )));
    }
    f.render_widget(Paragraph::new(info).wrap(Wrap { trim: true }), rows[0]);

    // Editable config fields — one clickable region over exactly the field rows.
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

    // Resolved install plan preview.
    let preview = if app.backend_plan_preview.is_empty() {
        "—".to_string()
    } else {
        app.backend_plan_preview.clone()
    };
    f.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled("Install plan", theme::muted(&th))),
            Line::from(Span::styled(
                preview,
                Style::default().fg(theme::color(&th, ThemeToken::Fg)),
            )),
        ])
        .wrap(Wrap { trim: true }),
        rows[2],
    );

    // Action buttons.
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
            click(
                app,
                Rect {
                    x,
                    y: rows[3].y,
                    width: w,
                    height: 1,
                },
                target,
            );
        }
        spans.push(Span::styled(
            label.to_string(),
            theme::accent(&th).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::raw("  "));
        x = x.saturating_add(w + 2);
    }
    f.render_widget(Paragraph::new(Line::from(spans)), rows[3]);

    // Progress / hint line.
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
    f.render_widget(
        Paragraph::new(Span::styled(hint, hint_style)).wrap(Wrap { trim: true }),
        rows[4],
    );
}
