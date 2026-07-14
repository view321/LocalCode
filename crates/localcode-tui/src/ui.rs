//! View rendering.
//!
//! Three chrome-light zones (redesign spec §2): a one-line **status bar**, a
//! single scrollable **working area**, and a one-line modal **omnibar**. No
//! popups or overlays — every former panel renders inline in the working area,
//! and confirms/errors are inline banners. Two grayscale themes; the only
//! animated glyph is the braille spinner, shown only while busy.

use crate::app::{App, ClickRegion, ClickTarget, EntryKind, Mode};
use crate::markdown;
use crate::theme;
use crate::widgets::{banner_height, draw_inline_banner};
use localcode_core::runtime::RuntimeStatus;
use localcode_core::theme::{ThemeMode, ThemeToken};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, List, ListItem, Paragraph, Wrap};
use ratatui::Frame;
use unicode_width::UnicodeWidthStr;

const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// A Setup checklist row: its rendered line plus an optional clickable action
/// word (`x`, `width`, target) right-aligned on that row.
type SetupRow = (Line<'static>, Option<(u16, u16, ClickTarget)>);

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn fg(th: &localcode_core::Theme) -> Style {
    Style::default().fg(theme::color(th, ThemeToken::Fg))
}

fn click(app: &mut App, rect: Rect, target: ClickTarget) {
    app.click_regions.push(ClickRegion { rect, target });
}

/// The braille frame while any task is busy, else `None` (render a space).
fn spinner_frame(app: &App) -> Option<char> {
    let started = app
        .busy
        .as_ref()
        .map(|b| b.started)
        .or_else(|| app.deploy_busy.as_ref().map(|b| b.started))
        .or_else(|| app.install_busy.as_ref().map(|b| b.started))
        .or_else(|| app.update_busy.as_ref().map(|b| b.started))?;
    let i = (started.elapsed().as_millis() / 90) as usize % SPINNER.len();
    Some(SPINNER[i])
}

/// A fixed-width inline meter: `█` (Fg) filled + `─` (faint) track. No color.
fn meter(th: &localcode_core::Theme, ratio: f64, cells: usize) -> Vec<Span<'static>> {
    let r = if ratio.is_finite() { ratio.clamp(0.0, 1.0) } else { 0.0 };
    let filled = ((r * cells as f64).round() as usize).min(cells);
    vec![
        Span::styled("█".repeat(filled), fg(th)),
        Span::styled("─".repeat(cells - filled), theme::faint(th)),
    ]
}

fn sep(th: &localcode_core::Theme) -> Span<'static> {
    Span::styled(" · ", theme::faint(th))
}

fn spans_width(spans: &[Span]) -> u16 {
    spans.iter().map(|s| s.width()).sum::<usize>() as u16
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

fn human_ctx(n: u32) -> String {
    if n >= 1_000 {
        format!("{}k", n / 1_000)
    } else {
        n.to_string()
    }
}

fn gib_f(bytes: u64) -> f64 {
    bytes as f64 / 1024.0f64.powi(3)
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", gib_f(bytes))
}

/// One column of left padding (the working area is otherwise borderless).
fn pad(area: Rect) -> Rect {
    Rect {
        x: area.x.saturating_add(1),
        y: area.y,
        width: area.width.saturating_sub(2),
        height: area.height,
    }
}

// ---------------------------------------------------------------------------
// Top-level draw
// ---------------------------------------------------------------------------

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    app.click_regions.clear();

    let base = Style::default()
        .bg(theme::color(&app.theme, ThemeToken::Bg))
        .fg(theme::color(&app.theme, ThemeToken::Fg));
    f.render_widget(Block::default().style(base), area);

    if area.width < 40 || area.height < 8 {
        f.render_widget(
            Paragraph::new("Terminal too small. Resize to continue.").style(theme::muted(&app.theme)),
            area,
        );
        return;
    }

    // Status (1) · rule (1) · working area (min) · omnibar (2: top rule + input).
    let main = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(2),
        ])
        .split(area);

    draw_status_bar(f, main[0], app);
    f.render_widget(
        Block::default()
            .borders(Borders::TOP)
            .border_type(BorderType::Plain)
            .border_style(theme::border(&app.theme)),
        main[1],
    );
    draw_working_area(f, main[2], app);
    draw_omnibar(f, main[3], app);
}

/// The working area: an inline banner (if any) at the top, the command list
/// while commanding, otherwise the current mode's view.
fn draw_working_area(f: &mut Frame, area: Rect, app: &mut App) {
    if let Some(modal) = app.modal.clone() {
        let h = banner_height(&modal, &app.theme, area.width).min(area.height);
        let banner_rect = Rect { x: area.x, y: area.y, width: area.width, height: h };
        let view_rect = Rect {
            x: area.x,
            y: area.y.saturating_add(h),
            width: area.width,
            height: area.height.saturating_sub(h),
        };
        if view_rect.height > 0 {
            draw_mode(f, view_rect, app);
        }
        let hits = draw_inline_banner(f, banner_rect, &modal, &app.theme);
        for (rect, i) in hits {
            click(app, rect, ClickTarget::ModalButton(i));
        }
        return;
    }
    if app.slash_active() {
        draw_command_list(f, area, app);
        return;
    }
    draw_mode(f, area, app);
}

fn draw_mode(f: &mut Frame, area: Rect, app: &mut App) {
    match app.mode {
        Mode::Chat => draw_chat(f, area, app),
        Mode::Models => draw_models(f, area, app),
        Mode::Runtimes => draw_runtimes(f, area, app),
        Mode::Remote => draw_remote(f, area, app),
        Mode::Backends => draw_backends(f, area, app),
        Mode::Bench => draw_bench(f, area, app),
        Mode::Setup => draw_setup(f, area, app),
        Mode::Settings => draw_settings(f, area, app),
    }
}

// ---------------------------------------------------------------------------
// Status bar (§5)
// ---------------------------------------------------------------------------

fn draw_status_bar(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    let inner = pad(area);

    // Left cluster: spinner · model · vram · ctx.
    let mut left: Vec<Span> = Vec::new();
    match spinner_frame(app) {
        Some(c) => left.push(Span::styled(format!("{c} "), theme::work(&th))),
        None => left.push(Span::raw("  ")),
    }
    match app.active_runtime() {
        Some(rt) => left.push(Span::styled(rt.name.clone(), fg(&th))),
        None => left.push(Span::styled("no runtime", theme::muted(&th))),
    }
    if !app.gpu.devices.is_empty() {
        let total = app.gpu.total_vram();
        let used = total.saturating_sub(app.gpu.free_vram());
        left.push(sep(&th));
        left.push(Span::styled("vram ", theme::muted(&th)));
        left.push(Span::styled(
            format!("{:.1}/{:.0}G", gib_f(used), gib_f(total)),
            fg(&th),
        ));
        left.push(Span::raw(" "));
        left.extend(meter(&th, used as f64 / total.max(1) as f64, 6));
    }
    left.push(sep(&th));
    left.push(Span::styled("ctx ", theme::muted(&th)));
    left.push(Span::styled(human_ctx(app.deploy_ctx), fg(&th)));
    // Transient status / feedback (set_status, raise_error). The redesign
    // dropped its dedicated row, orphaning `status_line` — so `/logs`, deploy
    // progress, "unknown command", etc. set text that nothing drew. Render it
    // here; the right cluster is drawn afterwards and clips a long message.
    if !app.status_line.is_empty() {
        left.push(sep(&th));
        let style = if app.status_is_error {
            fg(&th).add_modifier(Modifier::BOLD)
        } else {
            theme::muted(&th)
        };
        left.push(Span::styled(app.status_line.clone(), style));
    }
    f.render_widget(Paragraph::new(Line::from(left)), inner);

    // Right cluster: version/update · dark / light.
    let (ver_text, ver_style, is_update) = if let Some(v) = &app.update_installed {
        (format!("v{v} — restart"), theme::muted(&th), false)
    } else if let Some(info) = &app.update_available {
        (format!("update v{}", info.latest), fg(&th), true)
    } else {
        (format!("v{}", env!("CARGO_PKG_VERSION")), theme::muted(&th), false)
    };
    let dark_style = if th.mode == ThemeMode::Dark {
        theme::accent(&th)
    } else {
        theme::muted(&th)
    };
    let light_style = if th.mode == ThemeMode::Light {
        theme::accent(&th)
    } else {
        theme::muted(&th)
    };
    let right: Vec<Span> = vec![
        Span::styled(ver_text.clone(), ver_style),
        sep(&th),
        Span::styled("dark", dark_style),
        Span::styled(" / ", theme::faint(&th)),
        Span::styled("light", light_style),
        Span::raw(" "),
    ];
    let rw = spans_width(&right);
    if inner.width > rw + 8 {
        let rx = inner.x + inner.width - rw;
        f.render_widget(
            Paragraph::new(Line::from(right)),
            Rect { x: rx, y: inner.y, width: rw, height: 1 },
        );
        // Click regions for version badge + theme toggle.
        let mut cx = rx;
        let ver_w = ver_text.width() as u16;
        if is_update {
            click(app, Rect { x: cx, y: inner.y, width: ver_w, height: 1 }, ClickTarget::UpdateBadge);
        }
        cx += ver_w + sep(&th).width() as u16;
        click(app, Rect { x: cx, y: inner.y, width: 4, height: 1 }, ClickTarget::ThemeDark);
        cx += 4 + 3; // "dark" + " / "
        click(app, Rect { x: cx, y: inner.y, width: 5, height: 1 }, ClickTarget::ThemeLight);
    }
}

// ---------------------------------------------------------------------------
// Omnibar (§6)
// ---------------------------------------------------------------------------

fn draw_omnibar(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    f.render_widget(
        Block::default()
            .borders(Borders::TOP)
            .border_type(BorderType::Plain)
            .border_style(theme::border(&th)),
        area,
    );
    let row = Rect {
        x: area.x.saturating_add(1),
        y: area.y.saturating_add(1),
        width: area.width.saturating_sub(2),
        height: 1,
    };

    // A pending sudo password prompt takes over the omnibar: masked input, plus
    // the exact command it authorizes (re-shown from the confirm banner).
    if let Some((len, cmd)) = app.sudo_prompt() {
        let label = "sudo password: ";
        let dots = "•".repeat(len.min(64));
        let hint = " ↵ run · Esc cancel";
        let used = label.width() + dots.width() + hint.width() + 6;
        let cmd_room = (row.width as usize).saturating_sub(used);
        let cmd_clip = clip(&cmd, cmd_room);
        let spans = vec![
            Span::styled(label, theme::muted(&th)),
            Span::styled(dots, fg(&th)),
            Span::styled(format!("   {cmd_clip}"), theme::faint(&th)),
            Span::styled(hint, theme::faint(&th)),
        ];
        f.render_widget(Paragraph::new(Line::from(spans)), row);
        return;
    }

    let commanding = app.slash_active();
    let mut spans: Vec<Span> = Vec::new();
    if !commanding && app.mode != Mode::Chat {
        spans.push(Span::styled(format!("[{}] ", app.mode.tag()), theme::muted(&th)));
    }
    spans.push(Span::styled("❯ ", theme::muted(&th)));

    if app.coding_input.is_empty() {
        let placeholder = if commanding {
            "run a command — ↵ to execute, Esc to cancel"
        } else {
            app.mode.placeholder()
        };
        spans.push(Span::styled(placeholder, theme::faint(&th)));
    } else {
        let chars: Vec<char> = app.coding_input.chars().collect();
        let cur = app.coding_cursor.min(chars.len());
        let before: String = chars[..cur].iter().collect();
        let at: String = chars.get(cur).map(|c| c.to_string()).unwrap_or_else(|| " ".into());
        let after: String = if cur < chars.len() {
            chars[cur + 1..].iter().collect()
        } else {
            String::new()
        };
        spans.push(Span::styled(before, fg(&th)));
        spans.push(Span::styled(at, Style::default().add_modifier(Modifier::REVERSED)));
        spans.push(Span::styled(after, fg(&th)));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), row);

    // Right-aligned hint.
    let hint = if commanding {
        "↵ run"
    } else if app.mode == Mode::Models {
        "search"
    } else {
        "/ commands"
    };
    let hw = hint.width() as u16;
    if row.width > hw + 2 {
        f.render_widget(
            Paragraph::new(Span::styled(hint, theme::faint(&th))),
            Rect { x: row.x + row.width - hw, y: row.y, width: hw, height: 1 },
        );
    }
}

// ---------------------------------------------------------------------------
// Command list (§7.2)
// ---------------------------------------------------------------------------

fn draw_command_list(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    let inner = pad(area);
    let items = app.palette_items();
    let sel = app.palette_selected.min(items.len().saturating_sub(1));

    let header = Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("commands  ", theme::muted(&th)),
            Span::styled("↵ runs the first match", theme::faint(&th)),
        ])),
        header,
    );

    for (i, item) in items.iter().enumerate() {
        let y = inner.y + 2 + i as u16;
        if y >= inner.y + inner.height {
            break;
        }
        let (name, desc) = item.split_once("  —  ").unwrap_or((item.as_str(), ""));
        let marked = i == sel;
        let name_style = if marked {
            theme::accent(&th).add_modifier(Modifier::BOLD)
        } else {
            fg(&th)
        };
        let mark = if marked { "› " } else { "  " };
        let line = Line::from(vec![
            Span::styled(mark, theme::accent(&th)),
            Span::styled(format!("{name:<14}"), name_style),
            Span::styled(format!("  {desc}"), theme::muted(&th)),
        ]);
        let rect = Rect { x: inner.x, y, width: inner.width, height: 1 };
        f.render_widget(Paragraph::new(line), rect);
        click(app, rect, ClickTarget::CommandItem(i));
    }
}

// ---------------------------------------------------------------------------
// Chat (§7.1)
// ---------------------------------------------------------------------------

fn draw_chat(f: &mut Frame, area: Rect, app: &mut App) {
    click(app, area, ClickTarget::Transcript);
    let th = app.theme;
    let inner = pad(area);
    let inner_w = inner.width.max(1) as usize;
    let agent_running = app
        .busy
        .as_ref()
        .is_some_and(|b| b.kind == crate::app::BusyKind::Coding);

    let mut lines: Vec<Line> = Vec::new();
    for (idx, e) in app.coding_transcript.iter().enumerate() {
        if e.kind == EntryKind::You && idx > 0 {
            lines.push(Line::from(""));
        }
        let (prefix, style) = match e.kind {
            EntryKind::You => ("❯ ", theme::accent(&th).add_modifier(Modifier::BOLD)),
            EntryKind::Agent => ("", fg(&th)),
            EntryKind::Tool => ("", theme::muted(&th)),
            EntryKind::System => ("· ", theme::muted(&th)),
            EntryKind::Error => ("", fg(&th).add_modifier(Modifier::BOLD)),
        };
        let part_count = e.text.split('\n').count();
        let mut first = true;
        for (pi, part) in e.text.split('\n').enumerate() {
            let mut spans: Vec<Span> = Vec::new();
            if first && !prefix.is_empty() {
                spans.push(Span::styled(prefix.to_string(), style));
            } else if !first && !prefix.is_empty() {
                spans.push(Span::raw(" ".repeat(prefix.width())));
            }
            spans.push(Span::styled(part.to_string(), style));
            if e.live && agent_running && pi + 1 == part_count {
                spans.push(Span::styled(" ", style));
                if let Some(c) = spinner_frame(app) {
                    spans.push(Span::styled(c.to_string(), theme::work(&th)));
                }
            }
            lines.push(Line::from(spans));
            first = false;
        }
    }

    let total: usize = lines.iter().map(|l| l.width().max(1).div_ceil(inner_w)).sum();
    app.coding_total_lines = total;
    app.coding_view_height = inner.height;
    let max_scroll = total.saturating_sub(inner.height as usize);
    let offset = if app.coding_follow {
        max_scroll
    } else {
        app.coding_scroll.min(max_scroll)
    };
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((offset as u16, 0)),
        inner,
    );
}

// ---------------------------------------------------------------------------
// Models (§7.3) — two-pane, split ~42%
// ---------------------------------------------------------------------------

fn draw_models(f: &mut Frame, area: Rect, app: &mut App) {
    // Rebuild the card excerpt cache when the selection/theme changed.
    let stale = match (&app.model_detail, &app.card_cache) {
        (Some(d), Some(c)) => c.model_id != d.summary.id || c.mode != app.theme.mode,
        (Some(_), None) => true,
        _ => false,
    };
    if stale {
        if let Some(d) = &app.model_detail {
            let md = d.card_markdown.clone().unwrap_or_default();
            app.card_cache = Some(crate::app::CardCache {
                model_id: d.summary.id.clone(),
                mode: app.theme.mode,
                lines: markdown::render(&md, &app.theme),
            });
        }
    }

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Min(20)])
        .split(area);

    // Left column with a single vertical rule on its right edge.
    let th = app.theme;
    let rule = Block::default()
        .borders(Borders::RIGHT)
        .border_type(BorderType::Plain)
        .border_style(theme::border(&th));
    let left = rule.inner(cols[0]);
    f.render_widget(rule, cols[0]);
    draw_models_list(f, left, app);
    draw_models_detail(f, pad(cols[1]), app);
}

fn draw_models_list(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    let inner = pad(area);
    let header = Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 };
    let q = if app.model_query.is_empty() { "code" } else { app.model_query.as_str() };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("results  ", theme::muted(&th)),
            Span::styled(format!("\"{q}\" · {}", app.models.len()), theme::faint(&th)),
        ])),
        header,
    );

    let list_area = Rect {
        x: inner.x,
        y: inner.y + 1,
        width: inner.width,
        height: inner.height.saturating_sub(1),
    };
    click(app, list_area, ClickTarget::ModelList);

    let items: Vec<ListItem> = if app.models.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "no results — type a query, Enter to search",
            theme::muted(&th),
        )))]
    } else {
        app.models
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let selected = i == app.model_selected;
                let mark = if selected { "› " } else { "  " };
                let id_style = if selected {
                    theme::accent(&th).add_modifier(Modifier::BOLD)
                } else {
                    fg(&th)
                };
                let gguf = if m.tags.iter().any(|t| t.eq_ignore_ascii_case("gguf")) {
                    " · gguf"
                } else {
                    ""
                };
                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled(mark, theme::accent(&th)),
                        Span::styled(m.id.clone(), id_style),
                    ]),
                    Line::from(Span::styled(
                        format!(
                            "  {} dl · {} likes{gguf}",
                            human_count(m.downloads.unwrap_or(0)),
                            human_count(m.likes.unwrap_or(0)),
                        ),
                        theme::muted(&th),
                    )),
                ])
            })
            .collect()
    };
    app.model_list_state.select(if app.models.is_empty() {
        None
    } else {
        Some(app.model_selected)
    });
    f.render_stateful_widget(List::new(items), list_area, &mut app.model_list_state);
}

fn draw_models_detail(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    let Some(d) = app.model_detail.clone() else {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "select a model on the left — Enter loads it",
                theme::muted(&th),
            )))
            .wrap(Wrap { trim: true }),
            area,
        );
        return;
    };

    let s = &d.summary;
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(s.id.clone(), theme::accent(&th).add_modifier(Modifier::BOLD))));
    let mut metrics = format!(
        "{} downloads · {} likes",
        human_count(s.downloads.unwrap_or(0)),
        human_count(s.likes.unwrap_or(0)),
    );
    if let Some(l) = &d.license {
        metrics.push_str(&format!(" · {l}"));
    }
    if let Some(p) = &d.parameter_size {
        metrics.push_str(&format!(" · {p}"));
    }
    if let Some(pt) = &s.pipeline_tag {
        metrics.push_str(&format!(" · {pt}"));
    }
    lines.push(Line::from(Span::styled(metrics, theme::muted(&th))));
    lines.push(Line::from(""));

    // Two-to-three line model-card excerpt.
    if let Some(c) = &app.card_cache {
        for l in c.lines.iter().filter(|l| l.width() > 0).take(3) {
            lines.push(l.clone());
        }
    }
    lines.push(Line::from(""));

    // Quantizations — record the first row so clicks map to a quant index.
    lines.push(Line::from(vec![
        Span::styled("quantizations  ", theme::muted(&th)),
        Span::styled("click to select", theme::faint(&th)),
    ]));
    let quant_first = area.y + lines.len() as u16;
    let total_vram = app.gpu.total_vram();
    let shown = d.quants.len().min(6);
    for q in d.quants.iter().take(shown) {
        let selected = Some(q.label.as_str()) == app.selected_quant.as_deref();
        let mark = if selected { "› " } else { "  " };
        let label_style = if selected {
            theme::accent(&th).add_modifier(Modifier::BOLD)
        } else {
            fg(&th)
        };
        let ratio = if total_vram == 0 {
            0.0
        } else {
            q.total_size as f64 / total_vram as f64
        };
        let fit = if total_vram == 0 {
            "—"
        } else if ratio <= 0.9 {
            "fits"
        } else if ratio <= 1.0 {
            "tight"
        } else {
            "over"
        };
        let mut spans = vec![
            Span::styled(mark, theme::accent(&th)),
            Span::styled(format!("{:<9}", q.label), label_style),
            Span::styled(format!("{:>9}  ", gib(q.total_size)), theme::muted(&th)),
        ];
        spans.extend(meter(&th, ratio, 10));
        spans.push(Span::styled(format!(" {fit}"), theme::muted(&th)));
        lines.push(Line::from(spans));
    }
    if shown > 0 {
        click(
            app,
            Rect { x: area.x, y: quant_first, width: area.width, height: shown as u16 },
            ClickTarget::QuantList,
        );
    }
    lines.push(Line::from(""));

    // backend ⟳ · context
    let backend_row = area.y + lines.len() as u16;
    lines.push(Line::from(vec![
        Span::styled("backend ", theme::muted(&th)),
        Span::styled(format!("{} ⟳", app.deploy_backend.as_str()), theme::accent(&th)),
        Span::styled("    context ", theme::muted(&th)),
        Span::styled(app.deploy_ctx.to_string(), fg(&th)),
    ]));
    click(app, Rect { x: area.x, y: backend_row, width: 18, height: 1 }, ClickTarget::BackendCycle);

    // vram fit + a wide bar.
    if let Some(fit) = &app.last_fit {
        let ratio = fit.estimated_vram_bytes as f64 / fit.total_vram_bytes.max(1) as f64;
        lines.push(Line::from(vec![
            Span::styled("vram fit  ", theme::muted(&th)),
            Span::styled(
                format!(
                    "est {} · free {} / {}",
                    gib(fit.estimated_vram_bytes),
                    gib(fit.free_vram_bytes),
                    gib(fit.total_vram_bytes),
                ),
                fg(&th),
            ),
        ]));
        lines.push(Line::from(meter(&th, ratio, 24)));
    }
    lines.push(Line::from(""));

    // Deploy button / progress.
    let deploy_row = area.y + lines.len() as u16;
    if app.deploy_busy.is_some() {
        let mut spans = vec![Span::styled(
            format!("deploying {}%  ", app.deploy_progress),
            theme::accent(&th).add_modifier(Modifier::BOLD),
        )];
        spans.extend(meter(&th, app.deploy_progress as f64 / 100.0, 16));
        lines.push(Line::from(spans));
    } else {
        lines.push(Line::from(Span::styled(
            "[ deploy ]",
            theme::accent(&th).add_modifier(Modifier::BOLD),
        )));
        click(app, Rect { x: area.x, y: deploy_row, width: 10, height: 1 }, ClickTarget::DeployButton);
    }

    app.card_view_height = area.height;
    app.card_total_lines = lines.len();
    // No wrap: each logical line is one screen row so the click y-positions
    // computed above (quants / backend / deploy) stay exact.
    f.render_widget(Paragraph::new(lines), area);
}

// ---------------------------------------------------------------------------
// Runtimes (§7.4)
// ---------------------------------------------------------------------------

fn status_word(s: RuntimeStatus) -> &'static str {
    match s {
        RuntimeStatus::Healthy => "healthy",
        RuntimeStatus::Starting => "starting",
        RuntimeStatus::Unhealthy => "unhealthy",
        RuntimeStatus::Stopping => "stopping",
        RuntimeStatus::Stopped => "stopped",
    }
}

fn draw_runtimes(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    let inner = pad(area);
    let mut lines: Vec<Line> = vec![Line::from(Span::styled("runtimes", theme::muted(&th)))];

    let runtimes = app.all_runtimes();
    if runtimes.is_empty() {
        lines.push(Line::from(Span::styled(
            "no active runtimes — deploy one with /models, or /remote for a GPU box",
            theme::muted(&th),
        )));
    } else {
        for (i, r) in runtimes.iter().enumerate() {
            let selected = i == app.runtime_selected;
            let name_style = if selected {
                theme::accent(&th).add_modifier(Modifier::BOLD)
            } else {
                fg(&th)
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{:<22} ", r.name), name_style),
                Span::styled(format!("{:<10}", status_word(r.status)), theme::muted(&th)),
                Span::styled(format!(" {}", r.base_url), theme::faint(&th)),
            ]));
        }
    }
    let list_rows = runtimes.len().max(1) as u16;
    // Clickable region over the runtime rows (row 0 is the header).
    click(
        app,
        Rect { x: inner.x, y: inner.y + 1, width: inner.width, height: list_rows },
        ClickTarget::RuntimeList,
    );

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("gpu     ", theme::muted(&th)),
        Span::styled(app.gpu.summary(), fg(&th)),
    ]));
    lines.push(Line::from(vec![
        Span::styled("api     ", theme::muted(&th)),
        Span::styled(
            match app.api_healthy {
                None => "checking…".to_string(),
                Some(true) => "healthy".to_string(),
                Some(false) => "offline (local-first)".to_string(),
            },
            fg(&th),
        ),
    ]));
    let remote = if app.remote_sessions.is_empty() {
        "none".to_string()
    } else {
        app.remote_sessions
            .iter()
            .map(|s| s.server_name.clone())
            .collect::<Vec<_>>()
            .join(", ")
    };
    lines.push(Line::from(vec![
        Span::styled("remote  ", theme::muted(&th)),
        Span::styled(remote, fg(&th)),
    ]));
    // No wrap: runtime rows stay one-per-line for the RuntimeList click region.
    f.render_widget(Paragraph::new(lines), inner);
}

// ---------------------------------------------------------------------------
// Remote (§7.5) — two-pane, split ~30%
// ---------------------------------------------------------------------------

fn draw_remote(f: &mut Frame, area: Rect, app: &mut App) {
    use crate::app::REMOTE_FIELDS;
    let th = app.theme;
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Min(24)])
        .split(area);

    // Left: server list + "+ new server".
    let rule = Block::default()
        .borders(Borders::RIGHT)
        .border_type(BorderType::Plain)
        .border_style(theme::border(&th));
    let left = rule.inner(cols[0]);
    f.render_widget(rule, cols[0]);
    let linner = pad(left);

    let connected: Vec<String> = app.remote_sessions.iter().map(|s| s.server_name.clone()).collect();
    let mut llines: Vec<Line> = vec![Line::from(Span::styled("servers", theme::muted(&th)))];
    for (i, srv) in app.config.remote.servers.iter().enumerate() {
        let selected = i == app.remote_selected;
        let live = connected.contains(&srv.name);
        let connecting = app.remote_connecting == Some(i);
        let word = if connecting {
            "connecting"
        } else if live {
            "connected"
        } else {
            "offline"
        };
        let name_style = if selected {
            theme::accent(&th).add_modifier(Modifier::BOLD)
        } else {
            fg(&th)
        };
        llines.push(Line::from(Span::styled(srv.name.clone(), name_style)));
        llines.push(Line::from(Span::styled(
            format!("{word} · {}", srv.host),
            theme::muted(&th),
        )));
    }
    if app.config.remote.servers.is_empty() {
        llines.push(Line::from(Span::styled("(none yet)", theme::muted(&th))));
    }
    // Server rows are two lines each; the list region starts at row 1.
    let rows = (app.config.remote.servers.len() * 2).max(1) as u16;
    click(
        app,
        Rect { x: linner.x, y: linner.y + 1, width: linner.width, height: rows },
        ClickTarget::RemoteList,
    );
    f.render_widget(Paragraph::new(llines), linner);

    let newy = left.y + left.height.saturating_sub(1);
    let newrect = Rect { x: linner.x, y: newy, width: linner.width, height: 1 };
    f.render_widget(Paragraph::new(Span::styled("+ new server", theme::accent(&th))), newrect);
    click(app, newrect, ClickTarget::RemoteNew);

    // Right: editable fields + actions + connect progress.
    let right = pad(cols[1]);
    let mut lines: Vec<Line> = Vec::new();
    let title = app
        .config
        .remote
        .servers
        .get(app.remote_selected)
        .map(|s| s.name.clone())
        .unwrap_or_else(|| "no server".into());
    lines.push(Line::from(vec![
        Span::styled(title, theme::muted(&th)),
        Span::styled(" — click a field to edit", theme::faint(&th)),
    ]));
    let field_first = right.y + lines.len() as u16;
    let has_server = !app.config.remote.servers.is_empty();
    if has_server {
        for (i, label) in REMOTE_FIELDS.iter().enumerate() {
            let selected = i == app.remote_field;
            let editing = selected && app.remote_editing;
            let raw = if editing {
                app.remote_field_edit.clone()
            } else {
                app.remote_field_value(i)
            };
            let shown = if *label == "password" && !editing {
                "•".repeat(raw.chars().count().min(12))
            } else if editing {
                format!("{raw}▌")
            } else {
                raw
            };
            let val_style = if editing { theme::accent(&th) } else { fg(&th) };
            lines.push(Line::from(vec![
                Span::styled(format!("{label:<10} "), theme::muted(&th)),
                Span::styled(shown, val_style),
            ]));
        }
        click(
            app,
            Rect { x: right.x, y: field_first, width: right.width, height: REMOTE_FIELDS.len() as u16 },
            ClickTarget::RemoteField,
        );
    } else {
        lines.push(Line::from(Span::styled(
            "click '+ new server' to add one",
            theme::muted(&th),
        )));
    }

    // Action words.
    lines.push(Line::from(""));
    let actions_row = right.y + lines.len() as u16;
    let buttons = [
        ("[ connect ]", ClickTarget::RemoteConnect),
        ("[ save ]", ClickTarget::RemoteSave),
        ("[ disconnect ]", ClickTarget::RemoteDisconnect),
        ("[ delete ]", ClickTarget::RemoteDelete),
    ];
    let mut bx = right.x;
    let mut bspans: Vec<Span> = Vec::new();
    for (label, target) in buttons {
        let w = label.width() as u16;
        if bx + w <= right.x + right.width {
            click(app, Rect { x: bx, y: actions_row, width: w, height: 1 }, target);
        }
        bspans.push(Span::styled(label.to_string(), theme::accent(&th)));
        bspans.push(Span::raw(" "));
        bx += w + 1;
    }
    lines.push(Line::from(bspans));

    // Connect progress checklist (best-effort — steps aren't individually
    // reported by the SSH provisioner, so the running step shows the spinner).
    if app.remote_connecting.is_some() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("connecting…", theme::muted(&th))));
        let steps = [
            "reach host",
            "detect GPU (nvidia-smi)",
            "install & start ollama",
            "open tunnel",
        ];
        let spin = spinner_frame(app).unwrap_or('·');
        for (i, s) in steps.iter().enumerate() {
            let (mark, mstyle, tstyle) = if i == 0 {
                (spin.to_string(), theme::work(&th), fg(&th))
            } else {
                ("·".to_string(), theme::faint(&th), theme::faint(&th))
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{mark}  "), mstyle),
                Span::styled(s.to_string(), tstyle),
            ]));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "passwords stored in plaintext — prefer key_path.",
        theme::faint(&th),
    )));
    // No wrap: field/action rows keep the y-positions used for click regions.
    f.render_widget(Paragraph::new(lines), right);
}

// ---------------------------------------------------------------------------
// Backends (§7.6)
// ---------------------------------------------------------------------------

fn draw_backends(f: &mut Frame, area: Rect, app: &mut App) {
    use crate::app::BACKEND_ORDER;
    let th = app.theme;
    let inner = pad(area);
    // Rows are rendered directly at a running y so click regions map exactly,
    // regardless of terminal width or the variable-height smoke result row.
    f.render_widget(
        Paragraph::new(Span::styled("backends", theme::muted(&th))),
        Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 },
    );
    let bottom = inner.y + inner.height;
    let spin = spinner_frame(app);
    let mut y = inner.y + 2;
    for (i, kind) in BACKEND_ORDER.iter().enumerate() {
        if y >= bottom {
            break;
        }
        let report = app.backend_reports.iter().find(|r| r.kind == *kind);
        let installing = app.installing_kind == Some(*kind);
        let (status, ready) = if installing {
            ("installing".to_string(), false)
        } else if report.map(|r| r.ready).unwrap_or(false) {
            ("ready".to_string(), true)
        } else if report.map(|r| r.installed).unwrap_or(false) {
            ("installed".to_string(), true)
        } else if report.is_some() {
            ("not installed".to_string(), false)
        } else {
            ("detecting…".to_string(), false)
        };
        let selected = i == app.backend_sel;
        let mark = if selected { "› " } else { "  " };
        let name_style = if ready { fg(&th) } else { theme::muted(&th) };
        let mut spans = vec![
            Span::styled(mark, theme::accent(&th)),
            Span::styled(format!("{:<10}", kind.as_str()), name_style),
            Span::styled(format!("{status:<14}"), theme::muted(&th)),
        ];
        // Primary-row action: install (not ready) or smoke-test (ready).
        let row_action = if installing {
            if let Some(c) = spin {
                spans.push(Span::styled(format!("{c} "), theme::work(&th)));
            }
            spans.push(Span::styled(app.install_progress_line.clone(), theme::faint(&th)));
            None
        } else if !ready {
            spans.push(Span::styled("[ install ]", theme::accent(&th)));
            Some(ClickTarget::BackendInstall(i))
        } else {
            spans.push(Span::styled("[ smoke-test ]", theme::accent(&th)));
            Some(ClickTarget::BackendSmoke(i))
        };
        f.render_widget(
            Paragraph::new(Line::from(spans)),
            Rect { x: inner.x, y, width: inner.width, height: 1 },
        );
        if let Some(t) = row_action {
            click(app, Rect { x: inner.x, y, width: inner.width, height: 1 }, t);
        }
        y += 1;

        // Secondary row: the smoke result for this backend, if we have one.
        if y < bottom {
            if let Some(sr) = app.smoke_reports.iter().find(|r| r.kind == *kind) {
                let (glyph, gstyle) = if sr.ok {
                    ("✓", theme::accent(&th))
                } else {
                    ("✗", fg(&th))
                };
                let raw = sr
                    .diagnosis
                    .as_ref()
                    .map(|d| d.summary.clone())
                    .unwrap_or_else(|| if sr.ok { "passed".into() } else { sr.checked.clone() });
                let summary = clip(&raw, inner.width.saturating_sub(20) as usize);
                let prefix = format!("    {glyph} {summary}  ");
                let pw = prefix.width() as u16;
                let has_fix = sr
                    .diagnosis
                    .as_ref()
                    .and_then(|d| d.repair.as_ref())
                    .is_some();
                let mut l2 = vec![
                    Span::styled(format!("    {glyph} "), gstyle),
                    Span::styled(summary, theme::faint(&th)),
                    Span::raw("  "),
                ];
                if has_fix {
                    l2.push(Span::styled("[ fix ]", theme::accent(&th)));
                }
                f.render_widget(
                    Paragraph::new(Line::from(l2)),
                    Rect { x: inner.x, y, width: inner.width, height: 1 },
                );
                if has_fix {
                    let fw = "[ fix ]".width() as u16;
                    let fx = (inner.x + pw).min(inner.x + inner.width.saturating_sub(fw));
                    click(app, Rect { x: fx, y, width: fw, height: 1 }, ClickTarget::BackendFix(i));
                }
                y += 1;
            }
        }
    }

    y += 1;
    if y < bottom {
        let redetect = "[ re-detect ]";
        f.render_widget(
            Paragraph::new(Span::styled(redetect, theme::accent(&th))),
            Rect { x: inner.x, y, width: inner.width, height: 1 },
        );
        click(
            app,
            Rect { x: inner.x, y, width: redetect.width() as u16, height: 1 },
            ClickTarget::BackendRedetect,
        );
    }
}

/// Truncate to at most `max` display columns, adding an ellipsis when clipped.
fn clip(s: &str, max: usize) -> String {
    if max == 0 || s.width() <= max {
        return s.to_string();
    }
    let mut out = String::new();
    let mut w = 0usize;
    for c in s.chars() {
        let cw = c.to_string().width();
        if w + cw + 1 > max {
            break;
        }
        out.push(c);
        w += cw;
    }
    out.push('…');
    out
}

// ---------------------------------------------------------------------------
// Bench (§7.7)
// ---------------------------------------------------------------------------

fn draw_bench(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    let inner = pad(area);
    let target = app
        .active_runtime()
        .and_then(|r| r.model_id.clone())
        .unwrap_or_else(|| "no runtime".into());

    // Suite line + run button.
    let suite = Line::from(vec![
        Span::styled("suite  ", theme::muted(&th)),
        Span::styled("localcode-sample-coding v1.0.0", fg(&th)),
        Span::styled(format!("  · target {target}"), theme::faint(&th)),
    ]);
    f.render_widget(Paragraph::new(suite), Rect { x: inner.x, y: inner.y, width: inner.width.saturating_sub(8), height: 1 });
    let run = "[ run ]";
    let rx = inner.x + inner.width - run.width() as u16;
    f.render_widget(Paragraph::new(Span::styled(run, theme::accent(&th))), Rect { x: rx, y: inner.y, width: run.width() as u16, height: 1 });
    click(app, Rect { x: rx, y: inner.y, width: run.width() as u16, height: 1 }, ClickTarget::BenchRun);

    let mut lines: Vec<Line> = vec![Line::from("")];
    match &app.last_bench_result {
        None => {
            lines.push(Line::from(Span::styled("no runs yet — press [ run ]", theme::muted(&th))));
        }
        Some(r) => {
            let m = &r.metrics;
            // Stat grid: labels, values, bars (score/pass).
            let cells = [
                ("SCORE", format!("{:.2}", m.score), Some(m.score)),
                ("PASS", format!("{:.0}%", m.pass_rate * 100.0), Some(m.pass_rate)),
                ("P50", format!("{}ms", m.latency_p50_ms), None),
                ("P95", format!("{}ms", m.latency_p95_ms), None),
                (
                    "TOK/S",
                    m.tokens_per_sec.map(|t| format!("{t:.0}")).unwrap_or_else(|| "—".into()),
                    None,
                ),
            ];
            let mut labels: Vec<Span> = Vec::new();
            let mut values: Vec<Span> = Vec::new();
            let mut bars: Vec<Span> = Vec::new();
            for (i, (label, value, ratio)) in cells.iter().enumerate() {
                if i > 0 {
                    labels.push(Span::styled(" │ ", theme::faint(&th)));
                    values.push(Span::styled(" │ ", theme::faint(&th)));
                    bars.push(Span::styled("   ", theme::faint(&th)));
                }
                labels.push(Span::styled(format!("{label:<7}"), theme::muted(&th)));
                values.push(Span::styled(format!("{value:<7}"), theme::accent(&th).add_modifier(Modifier::BOLD)));
                match ratio {
                    Some(r) => {
                        let mut m = meter(&th, *r, 5);
                        m.push(Span::styled("  ", theme::faint(&th)));
                        bars.extend(m);
                    }
                    None => bars.push(Span::styled(format!("{:<7}", ""), theme::faint(&th))),
                }
            }
            lines.push(Line::from(labels));
            lines.push(Line::from(values));
            lines.push(Line::from(bars));
            lines.push(Line::from(""));

            lines.push(Line::from(Span::styled("tasks", theme::muted(&th))));
            for t in r.tasks.iter().take(10) {
                let (word, wstyle) = if t.passed {
                    ("pass", fg(&th))
                } else {
                    ("fail", fg(&th).add_modifier(Modifier::BOLD))
                };
                let latency = if t.passed { format!("{} ms", t.latency_ms) } else { "—".into() };
                lines.push(Line::from(vec![
                    Span::styled(format!("{:<34}", t.task_id), fg(&th)),
                    Span::styled(format!("{word:<6}"), wstyle),
                    Span::styled(latency, theme::muted(&th)),
                ]));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("recent runs", theme::muted(&th))));
            let mut hspans = vec![
                Span::styled(format!("{:<8}", "latest"), theme::muted(&th)),
                Span::styled(format!("{:<6}", format!("{:.2}", m.score)), fg(&th)),
            ];
            hspans.extend(meter(&th, m.score, 16));
            lines.push(Line::from(hspans));
        }
    }
    let body = Rect { x: inner.x, y: inner.y + 1, width: inner.width, height: inner.height.saturating_sub(1) };
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), body);
}

// ---------------------------------------------------------------------------
// Setup (§7.8)
// ---------------------------------------------------------------------------

fn draw_setup(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    let inner = pad(area);
    let w = inner.width as usize;

    let ready_backends = app.backend_reports.iter().filter(|r| r.ready).count();
    let ready_backend = app
        .backend_reports
        .iter()
        .find(|r| r.ready)
        .map(|r| format!("{:?}", r.kind).to_lowercase());
    let steps: [(bool, &str, String, &str, usize); 6] = [
        (!app.gpu.devices.is_empty(), "GPU detected", app.gpu.summary(), "recheck", 0),
        (
            ready_backends > 0,
            "Backend installed",
            ready_backend.clone().map(|k| format!("{k} ready")).unwrap_or_else(|| "install a backend".into()),
            "manage",
            1,
        ),
        (
            !app.all_runtimes().is_empty(),
            "Deploy a model",
            app.active_runtime_name().unwrap_or_else(|| "no runtime yet".into()),
            "open",
            2,
        ),
        (
            !app.remote_sessions.is_empty(),
            "Connect a remote GPU (optional)",
            "run models on a GPU box over SSH".into(),
            "add",
            3,
        ),
        (
            app.assistant_configured,
            "Configure the assistant",
            "OPENROUTER_API_KEY for the Ask helper".into(),
            "set",
            4,
        ),
        (
            app.config.updates.check_on_startup,
            "Updates",
            format!("checked on startup: {}", if app.config.updates.check_on_startup { "on" } else { "off" }),
            "settings",
            5,
        ),
    ];
    let done = steps.iter().filter(|s| s.0).count();

    // Each row: its rendered Line + an optional (action_x, action_w, target).
    let mut rows: Vec<SetupRow> = Vec::new();

    let mut header = vec![Span::styled("get started  ", theme::muted(&th))];
    header.extend(meter(&th, done as f64 / steps.len() as f64, 12));
    header.push(Span::styled(format!("  {done} of {}", steps.len()), theme::faint(&th)));
    rows.push((Line::from(header), None));
    rows.push((Line::from(""), None));

    for (ok, title, subtitle, action, idx) in &steps {
        let mark = if *ok { "[x] " } else { "[ ] " };
        let mark_style = if *ok { theme::muted(&th) } else { theme::faint(&th) };
        let title_style = if *ok { theme::muted(&th) } else { fg(&th) };
        let action_style = if *ok { theme::faint(&th) } else { theme::accent(&th) };
        let action_label = format!("[ {action} ]");
        let aw = action_label.width() as u16;
        let used = mark.width() + title.width();
        let padn = w.saturating_sub(used + aw as usize);
        let action_x = inner.x + inner.width.saturating_sub(aw);
        rows.push((
            Line::from(vec![
                Span::styled(mark, mark_style),
                Span::styled((*title).to_string(), title_style),
                Span::raw(" ".repeat(padn)),
                Span::styled(action_label, action_style),
            ]),
            Some((action_x, aw, ClickTarget::SetupStep(*idx))),
        ));
        rows.push((Line::from(Span::styled(format!("    {subtitle}"), theme::muted(&th))), None));
    }

    // Doctor block — probes derived from live state.
    rows.push((Line::from(""), None));
    let run = "[ run doctor ]";
    rows.push((
        Line::from(vec![
            Span::styled("doctor   ", theme::muted(&th)),
            Span::styled(run, theme::accent(&th)),
        ]),
        Some((inner.x + 9, run.width() as u16, ClickTarget::SetupDoctor)),
    ));
    let gpu_word = if app.gpu.devices.is_empty() { "none" } else { "ok" };
    let probes = [
        ("nvidia-smi", format!("{gpu_word} — {}", app.gpu.summary())),
        (
            "backends",
            format!("{ready_backends} ready — {}", ready_backend.unwrap_or_else(|| "none".into())),
        ),
        ("hf", format!("endpoint {}", app.config.registry.endpoint)),
        (
            "api",
            match app.api_healthy {
                None => "checking…".to_string(),
                Some(true) => "healthy".to_string(),
                Some(false) => "offline (local-first)".to_string(),
            },
        ),
    ];
    for (probe, detail) in probes {
        rows.push((
            Line::from(vec![
                Span::styled(format!("{probe:<12}"), theme::muted(&th)),
                Span::styled(detail, fg(&th)),
            ]),
            None,
        ));
    }
    rows.push((Line::from(""), None));
    rows.push((
        Line::from(Span::styled(
            format!("config {} — Ctrl+S to save", app.paths.config_file().display()),
            theme::faint(&th),
        )),
        None,
    ));

    // Render row-by-row with the scroll offset so action-word clicks stay exact.
    let scroll = app.setup_scroll as usize;
    for (i, (line, click_opt)) in rows.into_iter().enumerate() {
        if i < scroll {
            continue;
        }
        let sy = inner.y + (i - scroll) as u16;
        if sy >= inner.y + inner.height {
            break;
        }
        f.render_widget(Paragraph::new(line), Rect { x: inner.x, y: sy, width: inner.width, height: 1 });
        if let Some((ax, aw, target)) = click_opt {
            click(app, Rect { x: ax, y: sy, width: aw, height: 1 }, target);
        }
    }
}

// ---------------------------------------------------------------------------
// Settings (§7.9)
// ---------------------------------------------------------------------------

fn draw_settings(f: &mut Frame, area: Rect, app: &App) {
    let th = &app.theme;
    let inner = pad(area);
    let row = |label: &str, value: String, hint: &str| -> Line<'static> {
        let mut spans = vec![
            Span::styled(format!("{label:<16}"), theme::muted(th)),
            Span::styled(value, fg(th)),
        ];
        if !hint.is_empty() {
            spans.push(Span::styled(format!("   {hint}"), theme::faint(th)));
        }
        Line::from(spans)
    };
    let mirrors = if app.config.registry.mirrors.is_empty() {
        "none".to_string()
    } else {
        app.config.registry.mirrors.join(", ")
    };
    let lines = vec![
        Line::from(Span::styled("settings", theme::muted(th))),
        Line::from(""),
        row("theme", format!("{:?}", app.config.ui.theme), "/theme"),
        row("token streaming", app.config.agent.stream.to_string(), "agent.stream"),
        row(
            "confirm shell",
            app.config.agent.confirm_destructive_tools.to_string(),
            "destructive commands",
        ),
        row(
            "cloud fallback",
            app.config.agent.allow_cloud_fallback.to_string(),
            "local-first when off",
        ),
        row("default backend", app.config.backends.default.kind.clone(), ""),
        row("registry", app.config.registry.api_endpoint.clone(), ""),
        row("hf mirrors", mirrors, ""),
        row("remote servers", app.config.remote.servers.len().to_string(), ""),
        row("config file", app.paths.config_file().display().to_string(), "Ctrl+S"),
        Line::from(""),
        Line::from(Span::styled(
            "env overrides: LOCALCODE_API_URL, LOCALCODE_HF_ENDPOINT, HF_TOKEN, OPENROUTER_API_KEY",
            theme::faint(th),
        )),
    ];
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
}
