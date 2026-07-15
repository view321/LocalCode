//! View rendering.
//!
//! Three chrome-light zones (redesign spec §2): a bordered **status dashboard**
//! (rounded frame, matching the omnibar — collapsed 3 content lines, expands
//! to 10 on hover or click-to-pin), a single scrollable **working area**, and
//! a bordered multi-line **omnibar** anchored at the bottom. No popups or
//! overlays — every former panel renders inline in the working area, and
//! confirms/errors are inline banners. The command palette ('/') and the file
//! picker ('@') dock directly above the omnibar, where the eye already is. The
//! only animated glyph is the braille spinner, shown only while busy — during a
//! coding turn it sits next to the user message (with a 3-line backend log
//! tail) until the agent starts speaking, then moves onto the live stream line.

use crate::app::{
    App, ClickRegion, ClickTarget, DashCard, EntryKind, Mode, SettingAction, SettingsRowKind,
};
use crate::markdown;
use crate::theme;
use crate::widgets::{banner_height, button, button_width, draw_inline_banner};
use localcode_backends::human_size;
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

/// Coding turn is running but the model has not yet produced any thinking,
/// tool, or agent text after the latest user message — the "waiting" gap
/// where we park the spinner + backend log tail next to the user prompt.
fn waiting_for_agent_response(app: &App) -> bool {
    let Some(b) = app.busy.as_ref() else {
        return false;
    };
    if b.kind != crate::app::BusyKind::Coding {
        return false;
    }
    let Some(you_idx) = app
        .coding_transcript
        .iter()
        .rposition(|e| e.kind == EntryKind::You)
    else {
        return false;
    };
    !app.coding_transcript[you_idx + 1..].iter().any(|e| {
        matches!(
            e.kind,
            EntryKind::Agent | EntryKind::Thinking | EntryKind::Tool
        )
    })
}

/// Spinner lives in the chat transcript during a coding turn (next to the
/// user message while waiting, or on the live agent/thinking/tool line once
/// tokens arrive) — so the status dashboard yields it.
fn spinner_carried_to_chat(app: &App) -> bool {
    app.busy
        .as_ref()
        .is_some_and(|b| b.kind == crate::app::BusyKind::Coding)
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

/// Build a row `Line` that, when `selected`, gets a full-width selection bar:
/// the highlight background patches behind every span (per-span `fg` shows
/// through) and a trailing pad extends the bar to `width`. Use this for rows
/// that live inside a multi-line `Paragraph`, where a per-widget `.style()`
/// can't target a single row.
fn sel_line(
    th: &localcode_core::Theme,
    mut spans: Vec<Span<'static>>,
    width: u16,
    selected: bool,
) -> Line<'static> {
    if selected {
        let sel = theme::selected(th);
        for s in spans.iter_mut() {
            s.style = sel.patch(s.style);
        }
        let used: u16 = spans.iter().map(|s| s.width() as u16).sum();
        if used < width {
            spans.push(Span::styled(" ".repeat((width - used) as usize), sel));
        }
    }
    Line::from(spans)
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

    // The omnibar is a bordered multi-line composer: at least two text rows
    // (so it always reads as *the* input box), growing with the input up to
    // ui.composer_rows. The '/' command palette and the '@' file picker dock
    // directly above it. The status dashboard is a matching bordered frame:
    // collapsed = 2 metric lines + 1 latest-log line; hover/pin expands to
    // 10 content lines (2 metrics + 8 logs). While waiting for the agent,
    // the log tail is redirected under the user message (3 lines there), so
    // collapsed status is metrics-only. Outer height adds top/bottom border.
    let cap = app.config.ui.composer_rows.clamp(1, 10).max(2);
    let max_by_area = area.height.saturating_sub(8).max(1);
    let composer_h = (app.coding_input.split('\n').count().max(1) as u16)
        .max(2)
        .min(cap)
        .min(max_by_area);

    let status_content: u16 = if app.status_expanded() {
        10
    } else if waiting_for_agent_response(app) {
        2 // log lines live next to the user message while waiting
    } else {
        3
    };
    // Leave room for omnibar (composer+2), a working min-1, and picker room.
    let status_h = (status_content + 2).min(area.height.saturating_sub(composer_h + 3).max(3));

    // Bottom-docked picker band ('/' commands or '@' files), sized to its rows.
    let band_rows: u16 = if app.slash_active() {
        app.palette_items().len().clamp(1, 10) as u16
    } else if app.at_picker_active() {
        app.at_matches().len().clamp(1, 8) as u16
    } else {
        0
    };
    let band_rows = band_rows.min(area.height.saturating_sub(composer_h + status_h + 1));

    // Status frame · working area (min) · picker band · omnibar
    // (composer + top/bottom border).
    let main = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(status_h),
            Constraint::Min(1),
            Constraint::Length(band_rows),
            Constraint::Length(composer_h + 2),
        ])
        .split(area);

    draw_status_bar(f, main[0], app);
    draw_working_area(f, main[1], app);
    if main[2].height > 0 {
        draw_picker_band(f, main[2], app);
    }
    draw_omnibar(f, main[3], app);
}

/// The working area: an inline banner (if any) at the top, otherwise the
/// current mode's view. (The command list no longer replaces this area — it
/// docks above the omnibar; see [`draw_picker_band`].)
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
    draw_mode(f, area, app);
}

fn draw_mode(f: &mut Frame, area: Rect, app: &mut App) {
    match app.mode {
        Mode::Chat => draw_chat(f, area, app),
        Mode::Models => draw_models(f, area, app),
        Mode::Runtimes => draw_runtimes(f, area, app),
        Mode::Dash => draw_dash(f, area, app),
        Mode::Sessions => draw_sessions(f, area, app),
        Mode::Remote => draw_remote(f, area, app),
        Mode::Backends => draw_backends(f, area, app),
        Mode::Bench => draw_bench(f, area, app),
        Mode::Setup => draw_setup(f, area, app),
        Mode::Settings => draw_settings(f, area, app),
    }
}

// ---------------------------------------------------------------------------
// Status dashboard (§5) — bordered frame matching the omnibar
//
// Collapsed: 3 content lines (2 metric rows + 1 latest log).
// Expanded (hover or click-pinned): 10 content lines (2 metrics + 8 logs).
// ---------------------------------------------------------------------------

fn draw_status_bar(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    // Remember the outer rect so next-frame hover expand works without re-layout.
    app.status_bar_rect = area;

    let pin_mark = if app.status_pinned { " pinned" } else { "" };
    let title = if app.status_expanded() {
        format!(" status{pin_mark} · click to collapse ")
    } else {
        format!(" status{pin_mark} · hover/click for logs ")
    };

    // Full rounded pseudographic frame in the accent colour — same treatment as
    // the omnibar so the live metrics read as a fixed chrome dashboard, not a
    // free-floating text row.
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::accent(&th))
        .title(Span::styled(title, theme::muted(&th)));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width < 8 || inner.height == 0 {
        return;
    }

    // Whole dashboard is clickable to pin/unpin; finer controls (theme, approvals)
    // register afterwards so they win the reverse hit-test.
    click(app, area, ClickTarget::StatusToggle);

    // One-cell inset so content doesn't sit hard against the left border glyph.
    let content = Rect {
        x: inner.x.saturating_add(1),
        y: inner.y,
        width: inner.width.saturating_sub(1),
        height: inner.height,
    };
    if content.width == 0 || content.height == 0 {
        return;
    }

    // --- Line 0: model · GPU meters · power · tok/s ---
    let row0 = Rect {
        x: content.x,
        y: content.y,
        width: content.width,
        height: 1,
    };
    let mut line0: Vec<Span> = Vec::new();
    if !app.mouse_capture {
        // The mouse is released for terminal-native copy; F2 is the way back.
        line0.push(Span::styled(
            "SELECT",
            theme::accent(&th).add_modifier(Modifier::BOLD),
        ));
        line0.push(Span::styled(" copy text · F2 to exit", theme::muted(&th)));
        line0.push(sep(&th));
    }
    // During a coding turn the spinner is carried into the transcript (user
    // message while waiting, live stream line once the agent responds).
    if spinner_carried_to_chat(app) {
        line0.push(Span::raw("  "));
    } else {
        match spinner_frame(app) {
            Some(c) => line0.push(Span::styled(format!("{c} "), theme::work(&th))),
            None => line0.push(Span::raw("  ")),
        }
    }
    match app.active_runtime() {
        Some(rt) => line0.push(Span::styled(rt.name.clone(), fg(&th))),
        None => line0.push(Span::styled("no runtime", theme::muted(&th))),
    }
    if !app.gpu.devices.is_empty() {
        let total = app.gpu.total_vram();
        let used = total.saturating_sub(app.gpu.free_vram());
        line0.push(sep(&th));
        line0.push(Span::styled("vram ", theme::muted(&th)));
        line0.push(Span::styled(
            format!("{:.1}/{:.0}G", gib_f(used), gib_f(total)),
            fg(&th),
        ));
        line0.push(Span::raw(" "));
        line0.extend(meter(&th, used as f64 / total.max(1) as f64, 6));
        if let Some(pwr) = app.gpu.total_power_draw_w() {
            line0.push(sep(&th));
            line0.push(Span::styled("energy ", theme::muted(&th)));
            line0.push(Span::styled(format!("{pwr:.0}W"), fg(&th)));
        }
        if let Some(temp) = app.gpu.max_temperature_c() {
            line0.push(sep(&th));
            line0.push(Span::styled("temp ", theme::muted(&th)));
            line0.push(Span::styled(format!("{temp}°C"), fg(&th)));
        }
        if let Some(util) = app.gpu.avg_utilization_pct() {
            line0.push(sep(&th));
            line0.push(Span::styled("gpu ", theme::muted(&th)));
            line0.push(Span::styled(format!("{util}%"), fg(&th)));
        }
    }
    if let Some(tps) = app.tokens_per_sec {
        line0.push(sep(&th));
        line0.push(Span::styled("tok/s ", theme::muted(&th)));
        line0.push(Span::styled(format!("{tps:.0}"), fg(&th)));
    }
    f.render_widget(Paragraph::new(Line::from(line0)), row0);

    // --- Line 1: ctx · approvals · status · (right: version · themes) ---
    if content.height < 2 {
        return;
    }
    let row1 = Rect {
        x: content.x,
        y: content.y + 1,
        width: content.width,
        height: 1,
    };
    let mut line1: Vec<Span> = Vec::new();
    // Context: used/max with a meter. Used is estimated from the session
    // (chars÷4); max is the active model's context window (the assistant's
    // 128k, a deploy's `-c`, …), falling back to the deploy-form value.
    let ctx_max = app
        .active_runtime()
        .and_then(|r| r.context_tokens)
        .unwrap_or(app.deploy_ctx)
        .max(1);
    let ctx_used = app.ctx_used_tokens.min(ctx_max.saturating_mul(2));
    line1.push(Span::styled("ctx ", theme::muted(&th)));
    line1.push(Span::styled(
        format!("{}/{}", human_ctx(ctx_used), human_ctx(ctx_max)),
        fg(&th),
    ));
    line1.push(Span::raw(" "));
    line1.extend(meter(&th, ctx_used as f64 / ctx_max as f64, 6));
    // Agent approval mode — always visible (it decides what runs unprompted).
    line1.push(sep(&th));
    let approval_x = row1.x + spans_width(&line1);
    let approval_label = "approvals ";
    let approval_tag = app.config.agent.approval().tag();
    line1.push(Span::styled(approval_label, theme::muted(&th)));
    line1.push(Span::styled(approval_tag, fg(&th)));
    click(
        app,
        Rect {
            x: approval_x,
            y: row1.y,
            width: (approval_label.width() + approval_tag.width()) as u16,
            height: 1,
        },
        ClickTarget::ApprovalCycle,
    );
    // Transient status / feedback (set_status, raise_error).
    if !app.status_line.is_empty() {
        line1.push(sep(&th));
        let style = if app.status_is_error {
            fg(&th).add_modifier(Modifier::BOLD)
        } else {
            theme::muted(&th)
        };
        line1.push(Span::styled(app.status_line.clone(), style));
    }
    f.render_widget(Paragraph::new(Line::from(line1)), row1);

    // Right cluster on line 1: version/update · theme swatches.
    let (ver_text, ver_style, is_update) = if let Some(v) = &app.update_installed {
        (format!("v{v} — restart"), theme::muted(&th), false)
    } else if let Some(info) = &app.update_available {
        (format!("update v{}", info.latest), fg(&th), true)
    } else {
        (format!("v{}", env!("CARGO_PKG_VERSION")), theme::muted(&th), false)
    };
    let label_slot: u16 = 1 + ThemeMode::SWITCHER
        .iter()
        .map(|m| m.label().width() as u16)
        .max()
        .unwrap_or(0);
    let ver_w = ver_text.width() as u16;
    let dots_w = (ThemeMode::SWITCHER.len() as u16) * 2; // "◉ " per theme
    let rw = label_slot + ver_w + 3 + dots_w;
    if row1.width > rw + 8 {
        let rx = row1.x + row1.width - rw;
        let dots_x = rx + label_slot + ver_w + 3;
        let hovered: Option<ThemeMode> = app.hover.and_then(|(hc, hr)| {
            if hr != row1.y {
                return None;
            }
            ThemeMode::SWITCHER
                .iter()
                .enumerate()
                .find(|(i, _)| {
                    let x = dots_x + (*i as u16) * 2;
                    hc >= x && hc < x + 2
                })
                .map(|(_, m)| *m)
        });
        let label = hovered.map(|m| m.label()).unwrap_or("");
        let mut right: Vec<Span> = vec![
            Span::styled(
                format!("{label:>w$} ", w = label_slot.saturating_sub(1) as usize),
                theme::muted(&th),
            ),
            Span::styled(ver_text, ver_style),
            sep(&th),
        ];
        for m in ThemeMode::SWITCHER.iter() {
            let dot = if th.mode == *m { "◉" } else { "●" };
            let (r, g, b) = localcode_core::Theme::new(*m).token_rgb(ThemeToken::Accent);
            right.push(Span::styled(
                format!("{dot} "),
                Style::default().fg(ratatui::style::Color::Rgb(r, g, b)),
            ));
        }
        f.render_widget(
            Paragraph::new(Line::from(right)),
            Rect {
                x: rx,
                y: row1.y,
                width: rw,
                height: 1,
            },
        );
        if is_update {
            click(
                app,
                Rect {
                    x: rx + label_slot,
                    y: row1.y,
                    width: ver_w,
                    height: 1,
                },
                ClickTarget::UpdateBadge,
            );
        }
        for (i, m) in ThemeMode::SWITCHER.iter().enumerate() {
            let x = dots_x + (i as u16) * 2;
            click(
                app,
                Rect {
                    x,
                    y: row1.y,
                    width: 2,
                    height: 1,
                },
                ClickTarget::Theme(*m),
            );
        }
    }

    // --- Lines 2..: latest log(s) ---
    // While the agent is still waiting to speak, the log tail is shown under
    // the user message instead (see `draw_chat`). Expanded/pinned status still
    // keeps its full log view for inspection.
    if content.height < 3 {
        return;
    }
    if waiting_for_agent_response(app) && !app.status_expanded() {
        return;
    }
    let log_area = Rect {
        x: content.x,
        y: content.y + 2,
        width: content.width,
        height: content.height.saturating_sub(2),
    };
    let log_slots = log_area.height as usize;
    let log_lines = status_log_lines(app, log_slots);
    let mut rows: Vec<Line> = Vec::with_capacity(log_slots);
    for i in 0..log_slots {
        let text = log_lines.get(i).cloned().unwrap_or_default();
        // Clip to width so long JSON-ish tails don't wrap the frame.
        let clipped = clip(&text, log_area.width as usize);
        rows.push(Line::from(Span::styled(clipped, theme::muted(&th))));
    }
    f.render_widget(Paragraph::new(rows), log_area);
}

/// Pick the last `n` compact status log lines (newest last). When empty, a
/// single placeholder so the third row never looks broken.
fn status_log_lines(app: &App, n: usize) -> Vec<String> {
    if n == 0 {
        return Vec::new();
    }
    if app.status_logs.is_empty() {
        return vec!["log · (no entries yet)".into()];
    }
    let start = app.status_logs.len().saturating_sub(n);
    app.status_logs[start..].to_vec()
}

// ---------------------------------------------------------------------------
// Omnibar (§6)
// ---------------------------------------------------------------------------

fn draw_omnibar(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;

    // A full pseudographic frame in the accent colour: the input box is the one
    // element that should always be findable at a glance.
    let title = if app.sudo_prompt().is_some() {
        " sudo "
    } else if app.slash_active() {
        " command "
    } else if app.mode != Mode::Chat {
        app.mode.tag()
    } else {
        ""
    };
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::accent(&th));
    if !title.is_empty() {
        block = block.title(Span::styled(format!(" {} ", title.trim()), theme::muted(&th)));
    }
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width < 4 || inner.height == 0 {
        return;
    }
    let row = Rect {
        x: inner.x.saturating_add(1),
        y: inner.y,
        width: inner.width.saturating_sub(1),
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
            Span::styled("█", theme::accent(&th)),
            Span::styled(format!("  {cmd_clip}"), theme::faint(&th)),
            Span::styled(hint, theme::faint(&th)),
        ];
        f.render_widget(Paragraph::new(Line::from(spans)), row);
        return;
    }

    let prefix = Span::styled("❯ ", theme::accent(&th).add_modifier(Modifier::BOLD));
    let prefix_w = prefix.width();
    let caret = |s: String| Span::styled(s, Style::default().add_modifier(Modifier::REVERSED));

    // Empty: prompt + a visible caret block + the placeholder — so there is
    // always a text cursor showing where typing goes.
    if app.coding_input.is_empty() {
        let agent_busy = app
            .busy
            .as_ref()
            .is_some_and(|b| b.kind == crate::app::BusyKind::Coding);
        let placeholder = if agent_busy {
            "agent is working… Esc cancels"
        } else {
            app.mode.placeholder()
        };
        let spans = vec![
            prefix,
            caret(" ".into()),
            Span::raw(" "),
            Span::styled(placeholder, theme::faint(&th)),
        ];
        f.render_widget(Paragraph::new(Line::from(spans)), row);
        return;
    }

    // Multi-line composer: render a window of `inner.height` lines that keeps
    // the caret visible, highlighting the caret cell on its line.
    let composer_h = inner.height as usize;
    let lines: Vec<&str> = app.coding_input.split('\n').collect();
    let (cur_line, cur_col) = app.omnibar_cursor_line_col();
    let start = if cur_line >= composer_h {
        cur_line + 1 - composer_h
    } else {
        0
    };
    for (vi, li) in (start..lines.len()).take(composer_h).enumerate() {
        let y = row.y + vi as u16;
        let mut spans: Vec<Span> = Vec::new();
        if li == 0 {
            spans.push(prefix.clone());
        } else {
            spans.push(Span::raw(" ".repeat(prefix_w)));
        }
        if li == cur_line {
            let chars: Vec<char> = lines[li].chars().collect();
            let cc = cur_col.min(chars.len());
            let before: String = chars[..cc].iter().collect();
            let at: String = chars
                .get(cc)
                .map(|c| c.to_string())
                .unwrap_or_else(|| " ".into());
            let after: String = if cc < chars.len() {
                chars[cc + 1..].iter().collect()
            } else {
                String::new()
            };
            spans.push(Span::styled(before, fg(&th)));
            spans.push(caret(at));
            spans.push(Span::styled(after, fg(&th)));
        } else {
            spans.push(Span::styled(lines[li].to_string(), fg(&th)));
        }
        f.render_widget(
            Paragraph::new(Line::from(spans)),
            Rect { x: row.x, y, width: row.width, height: 1 },
        );
    }
}

// ---------------------------------------------------------------------------
// Picker band (§7.2) — commands ('/') or files ('@'), docked above the omnibar
// ---------------------------------------------------------------------------

/// One row of a picker band: full-width selection bar when picked, clickable,
/// wheel-scrollable (the wheel moves the selection; see `wheel_scroll_at`).
fn band_row(
    f: &mut Frame,
    app: &mut App,
    rect: Rect,
    spans: Vec<Span<'static>>,
    selected: bool,
    target: ClickTarget,
) {
    let th = app.theme;
    let mut all = vec![Span::styled(
        if selected { "▌ " } else { "  " },
        theme::accent(&th),
    )];
    all.extend(spans);
    let para = Paragraph::new(Line::from(all));
    f.render_widget(
        if selected {
            para.style(theme::selected(&th))
        } else {
            para
        },
        rect,
    );
    click(app, rect, target);
}

/// The window of `visible` rows that keeps `sel` in view.
fn band_window(sel: usize, len: usize, visible: usize) -> std::ops::Range<usize> {
    let start = if sel < visible { 0 } else { sel + 1 - visible };
    let start = start.min(len.saturating_sub(visible.max(1)));
    start..(start + visible).min(len)
}

/// The '/' command palette or the '@' file picker, rendered directly above the
/// omnibar so the suggestions appear where the user is already looking.
fn draw_picker_band(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    let visible = area.height as usize;
    if app.slash_active() {
        let items = app.palette_items();
        if items.is_empty() {
            f.render_widget(
                Paragraph::new(Span::styled(
                    "  no matching command — ↵ runs the text as typed",
                    theme::faint(&th),
                )),
                Rect { x: area.x, y: area.y, width: area.width, height: 1 },
            );
            return;
        }
        let sel = app.palette_selected.min(items.len() - 1);
        for (vi, i) in band_window(sel, items.len(), visible).enumerate() {
            let (name, desc) = items[i].split_once("  —  ").unwrap_or((items[i].as_str(), ""));
            let selected = i == sel;
            let name_style = if selected {
                theme::accent(&th).add_modifier(Modifier::BOLD)
            } else {
                fg(&th)
            };
            let spans = vec![
                Span::styled(format!("{name:<24}"), name_style),
                Span::raw("  "),
                Span::styled(desc.to_string(), theme::muted(&th)),
            ];
            let rect = Rect { x: area.x, y: area.y + vi as u16, width: area.width, height: 1 };
            band_row(f, app, rect, spans, selected, ClickTarget::CommandItem(i));
        }
        return;
    }

    // '@' file picker.
    let files = app.at_matches();
    if files.is_empty() {
        return;
    }
    let sel = app.at_selected.min(files.len() - 1);
    for (vi, i) in band_window(sel, files.len(), visible).enumerate() {
        let selected = i == sel;
        let (dir, name) = match files[i].rsplit_once('/') {
            Some((d, n)) => (format!("{d}/"), n.to_string()),
            None => (String::new(), files[i].clone()),
        };
        let name_style = if selected {
            theme::accent(&th).add_modifier(Modifier::BOLD)
        } else {
            fg(&th)
        };
        let mut spans = vec![
            Span::styled(dir, theme::muted(&th)),
            Span::styled(name, name_style),
        ];
        if selected {
            spans.push(Span::styled("  ↵ attach", theme::muted(&th)));
        }
        let rect = Rect { x: area.x, y: area.y + vi as u16, width: area.width, height: 1 };
        band_row(f, app, rect, spans, selected, ClickTarget::AtItem(i));
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
    let last_you_idx = app
        .coding_transcript
        .iter()
        .rposition(|e| e.kind == EntryKind::You);
    // Backend log tail + spinner sit under the latest user prompt until the
    // first thinking/tool/agent token arrives, then they clear out of the way.
    let waiting = waiting_for_agent_response(app);

    // Each logical line is tagged with the transcript entry that produced it
    // so we can register per-entry click regions after scroll is applied.
    let mut lines: Vec<Line> = Vec::new();
    let mut line_entry: Vec<Option<usize>> = Vec::new();

    for (idx, e) in app.coding_transcript.iter().enumerate() {
        if e.kind == EntryKind::You && idx > 0 {
            lines.push(Line::from(""));
            line_entry.push(None);
        }

        let body_style = match e.kind {
            EntryKind::You => theme::accent(&th).add_modifier(Modifier::BOLD),
            EntryKind::Agent => fg(&th),
            EntryKind::Thinking => theme::faint(&th).add_modifier(Modifier::ITALIC),
            EntryKind::Tool => theme::muted(&th),
            EntryKind::System => theme::muted(&th),
            EntryKind::Error => fg(&th).add_modifier(Modifier::BOLD),
        };
        let header_style = match e.kind {
            EntryKind::Thinking => theme::muted(&th).add_modifier(Modifier::ITALIC),
            EntryKind::Tool => theme::muted(&th),
            _ => body_style,
        };
        // A selected (clicked + copied) model response gets the selection
        // highlight background, keeping its own foreground so it stays readable.
        let (body_style, header_style) = if app.coding_selected == Some(idx) {
            match theme::selected(&th).bg {
                Some(bg) => (body_style.bg(bg), header_style.bg(bg)),
                None => (body_style, header_style),
            }
        } else {
            (body_style, header_style)
        };

        // Build the visible text for this entry (collapsed header vs full body).
        let display = chat_entry_display(e);
        let mut parts: Vec<&str> = display.lines().collect();
        if parts.is_empty() {
            parts.push("");
        }
        let part_count = parts.len();
        let prefix = match e.kind {
            EntryKind::You => "❯ ",
            EntryKind::System => "· ",
            _ => "",
        };

        for (pi, part) in parts.iter().enumerate() {
            let style = if pi == 0 { header_style } else { body_style };
            let mut spans: Vec<Span> = Vec::new();
            if pi == 0 && !prefix.is_empty() {
                spans.push(Span::styled(prefix.to_string(), style));
            } else if pi > 0 && !prefix.is_empty() {
                spans.push(Span::raw(" ".repeat(prefix.width())));
            }
            // Indent multi-line thinking / tool detail under the header.
            if pi > 0 && matches!(e.kind, EntryKind::Thinking | EntryKind::Tool) {
                spans.push(Span::styled("  ".to_string(), style));
            }
            spans.push(Span::styled((*part).to_string(), style));
            // Spinner: on the latest user line while waiting for first tokens;
            // on the live stream line once the agent is speaking.
            let spin_here = agent_running
                && pi + 1 == part_count
                && ((waiting && Some(idx) == last_you_idx)
                    || (e.live && !waiting));
            if spin_here {
                spans.push(Span::styled(" ", style));
                if let Some(c) = spinner_frame(app) {
                    spans.push(Span::styled(c.to_string(), theme::work(&th)));
                }
            }
            lines.push(Line::from(spans));
            // Clickable rows: toggleable entries (thinking/tool) expand; a model
            // response selects + auto-copies.
            line_entry.push(if e.can_toggle() || e.is_model_response() {
                Some(idx)
            } else {
                None
            });
        }

        // Three live backend log lines under the user message while waiting —
        // something to watch until the agent starts responding, then gone.
        if waiting && Some(idx) == last_you_idx {
            let log_lines = status_log_lines(app, 3);
            for i in 0..3 {
                let text = log_lines.get(i).map(|s| s.as_str()).unwrap_or("");
                let clipped = clip(text, inner_w.saturating_sub(2));
                lines.push(Line::from(Span::styled(
                    format!("  {clipped}"),
                    theme::muted(&th),
                )));
                line_entry.push(None);
            }
        }
    }

    // Paragraph wraps long lines; mirror that to map screen rows → entries.
    let mut wrapped_entry: Vec<Option<usize>> = Vec::new();
    for (line, entry) in lines.iter().zip(line_entry.iter()) {
        let cells = line.width().max(1);
        let rows = cells.div_ceil(inner_w).max(1);
        for _ in 0..rows {
            wrapped_entry.push(*entry);
        }
    }

    let total = wrapped_entry.len();
    app.coding_total_lines = total;
    app.coding_view_height = inner.height;
    let max_scroll = total.saturating_sub(inner.height as usize);
    let offset = if app.coding_follow {
        max_scroll
    } else {
        app.coding_scroll.min(max_scroll)
    };

    // One click region per visible screen row that belongs to a toggleable entry.
    let view_h = inner.height as usize;
    for row in 0..view_h {
        let virt = offset + row;
        if let Some(Some(entry_idx)) = wrapped_entry.get(virt) {
            click(
                app,
                Rect {
                    x: inner.x,
                    y: inner.y + row as u16,
                    width: inner.width,
                    height: 1,
                },
                ClickTarget::TranscriptEntry(*entry_idx),
            );
        }
    }

    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((offset as u16, 0)),
        inner,
    );
}

/// Visible text for a transcript entry, respecting expand/collapse.
fn chat_entry_display(e: &crate::app::TranscriptEntry) -> String {
    match e.kind {
        crate::app::EntryKind::Thinking => {
            let n = e.text.chars().count();
            let show_full = e.live || e.expanded;
            if show_full {
                let chev = if e.live {
                    ""
                } else if e.can_toggle() {
                    "  ▾"
                } else {
                    ""
                };
                if e.text.trim().is_empty() {
                    format!("thinking…{chev}")
                } else {
                    format!("thinking{chev}\n{}", e.text)
                }
            } else {
                format!("thinking  · {n} chars  ▸")
            }
        }
        crate::app::EntryKind::Tool => {
            // Header chevron is derived from expand state so click and keyboard
            // toggles stay consistent without rewriting `text`.
            let mut header = e
                .text
                .trim_end_matches(['▾', '▸', ' '])
                .to_string();
            if e.detail.as_ref().is_some_and(|d| !d.trim().is_empty()) {
                header.push_str(if e.expanded { "  ▾" } else { "  ▸" });
            }
            if e.expanded {
                if let Some(detail) = &e.detail {
                    format!("{header}\n{detail}")
                } else {
                    header
                }
            } else {
                header
            }
        }
        _ => e.text.clone(),
    }
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
                let favorite = app.config.is_favorite(&m.id);
                let downloaded = app.is_downloaded(&m.id);
                let mut id_line: Vec<Span> = vec![Span::styled(mark, theme::accent(&th))];
                if favorite {
                    id_line.push(Span::styled("★ ", theme::accent(&th)));
                }
                id_line.push(Span::styled(m.id.clone(), id_style));
                let mut meta: Vec<Span> = vec![Span::styled(
                    format!(
                        "  {} dl · {} likes{gguf}",
                        human_count(m.downloads.unwrap_or(0)),
                        human_count(m.likes.unwrap_or(0)),
                    ),
                    theme::muted(&th),
                )];
                if downloaded {
                    meta.push(Span::styled(
                        " · downloaded",
                        theme::accent(&th).add_modifier(Modifier::BOLD),
                    ));
                }
                ListItem::new(vec![Line::from(id_line), Line::from(meta)])
            })
            .collect()
    };
    app.model_list_state.select(if app.models.is_empty() {
        None
    } else {
        Some(app.model_selected)
    });
    f.render_stateful_widget(
        List::new(items).highlight_style(theme::selected(&th)),
        list_area,
        &mut app.model_list_state,
    );
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

    // Controls block (fixed, no-wrap): id, metrics, quants, backend/ctx, fit,
    // deploy. Rendered at the top so the click y-positions computed here stay
    // exact; the scrollable model card fills whatever is left below.
    let s = &d.summary;
    let mut ctrl: Vec<Line> = Vec::new();
    ctrl.push(Line::from(Span::styled(
        s.id.clone(),
        theme::accent(&th).add_modifier(Modifier::BOLD),
    )));
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
    ctrl.push(Line::from(Span::styled(metrics, theme::muted(&th))));
    ctrl.push(Line::from(""));

    // Quantizations — record the first row so clicks map to a quant index.
    ctrl.push(Line::from(vec![
        Span::styled("quantizations  ", theme::muted(&th)),
        Span::styled("click to select", theme::faint(&th)),
    ]));
    let quant_first = area.y + ctrl.len() as u16;
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
        ctrl.push(sel_line(&th, spans, area.width, selected));
    }
    if shown > 0 {
        click(
            app,
            Rect { x: area.x, y: quant_first, width: area.width, height: shown as u16 },
            ClickTarget::QuantList,
        );
    }
    ctrl.push(Line::from(""));

    // backend ⟳ — click to cycle Ollama → llama.cpp → vLLM → SGLang.
    let backend_row = area.y + ctrl.len() as u16;
    let backend_cell = format!("{} ⟳", app.deploy_backend.as_str());
    let backend_w = ("backend ".len() + backend_cell.width()) as u16;
    ctrl.push(Line::from(vec![
        Span::styled("backend ", theme::muted(&th)),
        Span::styled(backend_cell, theme::accent(&th)),
    ]));
    click(app, Rect { x: area.x, y: backend_row, width: backend_w, height: 1 }, ClickTarget::BackendCycle);

    // Editable deploy parameters, filtered to what the current backend honors.
    // Click a row to edit it inline (↵ save, Esc cancel, blank = default).
    for field in app.deploy_fields() {
        let row = area.y + ctrl.len() as u16;
        let editing = app.deploy_editing_field() == Some(field);
        let value = if editing {
            // Live edit buffer with a block cursor.
            Span::styled(
                format!("{}\u{2588}", app.deploy_field_edit_buf()),
                theme::accent(&th).add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled(app.deploy_field_display(field), fg(&th))
        };
        ctrl.push(Line::from(vec![
            Span::styled(format!("{:<11}", field.label()), theme::muted(&th)),
            value,
        ]));
        click(
            app,
            Rect { x: area.x, y: row, width: area.width, height: 1 },
            ClickTarget::DeployField(field),
        );
    }

    // vram fit + a wide bar.
    if let Some(fit) = &app.last_fit {
        let ratio = fit.estimated_vram_bytes as f64 / fit.total_vram_bytes.max(1) as f64;
        ctrl.push(Line::from(vec![
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
        ctrl.push(Line::from(meter(&th, ratio, 24)));
    }
    ctrl.push(Line::from(""));

    // Deploy button, or — while deploying — a progress meter with a Cancel
    // button on its own row (deploy is cancelled here, never with Esc).
    let deploy_row = area.y + ctrl.len() as u16;
    if app.deploy_busy.is_some() {
        let mut spans = vec![Span::styled(
            format!("deploying {}%  ", app.deploy_progress),
            theme::accent(&th).add_modifier(Modifier::BOLD),
        )];
        spans.extend(meter(&th, app.deploy_progress as f64 / 100.0, 16));
        ctrl.push(Line::from(spans));
        let cancel_row = area.y + ctrl.len() as u16;
        ctrl.push(Line::from(button(&th, "cancel", false)));
        click(
            app,
            Rect { x: area.x, y: cancel_row, width: button_width("cancel"), height: 1 },
            ClickTarget::DeployCancel,
        );
    } else {
        ctrl.push(Line::from(button(&th, "deploy", true)));
        click(
            app,
            Rect { x: area.x, y: deploy_row, width: button_width("deploy"), height: 1 },
            ClickTarget::DeployButton,
        );
    }

    // Render the controls at the top (no wrap → exact click rows).
    let controls_h = (ctrl.len() as u16).min(area.height);
    f.render_widget(
        Paragraph::new(ctrl),
        Rect { x: area.x, y: area.y, width: area.width, height: controls_h },
    );

    // The full model card fills the rest: wrapped and scrollable (PgUp/PgDn and
    // the mouse wheel drive `card_scroll`). This is the proper card view that
    // was lost in the redesign — long lines wrap instead of running off-screen.
    let avail = area.height.saturating_sub(controls_h);
    let card_top = area.y + controls_h;
    if avail >= 2 {
        let inner_w = area.width.max(1) as usize;
        let card_lines: Vec<Line> = app
            .card_cache
            .as_ref()
            .map(|c| c.lines.clone())
            .unwrap_or_default();
        let view_h = avail.saturating_sub(1); // header row
        let wrapped: usize = card_lines
            .iter()
            .map(|l| l.width().max(1).div_ceil(inner_w))
            .sum();
        app.card_view_height = view_h;
        app.card_total_lines = wrapped;
        let max_scroll = wrapped.saturating_sub(view_h as usize);
        let offset = app.card_scroll.min(max_scroll);

        let shown_to = (offset + view_h as usize).min(wrapped);
        let more = if wrapped > view_h as usize {
            format!("   ↑↓ scroll · {shown_to}/{wrapped}")
        } else {
            String::new()
        };
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("model card", theme::muted(&th)),
                Span::styled(more, theme::faint(&th)),
            ])),
            Rect { x: area.x, y: card_top, width: area.width, height: 1 },
        );
        if card_lines.is_empty() {
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    "no model card provided by the author",
                    theme::faint(&th),
                ))),
                Rect { x: area.x, y: card_top + 1, width: area.width, height: 1 },
            );
        } else {
            f.render_widget(
                Paragraph::new(card_lines)
                    .wrap(Wrap { trim: false })
                    .scroll((offset as u16, 0)),
                Rect { x: area.x, y: card_top + 1, width: area.width, height: view_h },
            );
        }
    } else {
        app.card_view_height = 0;
        app.card_total_lines = 0;
    }
}

// ---------------------------------------------------------------------------
// Runtimes (§7.4)
// ---------------------------------------------------------------------------

pub(crate) fn status_word(s: RuntimeStatus) -> &'static str {
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
            let spans = vec![
                Span::styled(format!("{:<22} ", r.name), name_style),
                Span::styled(format!("{:<10}", status_word(r.status)), theme::muted(&th)),
                Span::styled(format!(" {}", r.base_url), theme::faint(&th)),
            ];
            lines.push(sel_line(&th, spans, inner.width, selected));
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
// Dash — multi-model manager (/dash)
//
// One card per running model, in `all_runtimes()` order (so a card's index
// selects that runtime). Each card is DASH_CARD_H content lines + a gap:
//   1  ● name    backend · status · vram · tok/s · ctx        ★ active
//   2    $ launch command                                     [ copy ]
//   3    log  newest backend output
//   4    ! error (if the process exited non-zero)             [ copy error ]
//   5    [ stop ]  [ use ]
// The header carries a [ + new model ] button, and the whole thing scrolls by
// card when there are more models than fit.
// ---------------------------------------------------------------------------

/// Content lines per card (excludes the one-line gap between cards).
const DASH_CARD_H: u16 = 5;
/// Card stride including the trailing gap.
const DASH_CARD_STRIDE: u16 = DASH_CARD_H + 1;

fn draw_dash(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    let inner = pad(area);
    if inner.height == 0 || inner.width < 8 {
        return;
    }

    let cards = app.dash_cards();

    // --- Header row: title · [ + new model ] · gpu summary ---
    let header = Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 };
    let title = format!("models ({})", cards.len());
    let mut hspans = vec![Span::styled(title, theme::muted(&th).add_modifier(Modifier::BOLD))];
    hspans.push(Span::raw("  "));
    let new_label = "+ new model";
    let new_w = button_width(new_label);
    hspans.extend(button(&th, new_label, false));
    f.render_widget(Paragraph::new(Line::from(hspans)), header);
    // Button click rect (title width is stable, so recompute its x).
    let new_x = inner.x + (format!("models ({})  ", cards.len()).width() as u16);
    click(
        app,
        Rect { x: new_x, y: header.y, width: new_w, height: 1 },
        ClickTarget::DashStartNew,
    );
    // GPU summary right-aligned when there's room.
    if !app.gpu.devices.is_empty() {
        let total = app.gpu.total_vram();
        let used = total.saturating_sub(app.gpu.free_vram());
        let gtext = format!("gpu {:.1}/{:.0}G", gib_f(used), gib_f(total));
        let gw = gtext.width() as u16;
        if inner.width > new_x - inner.x + new_w + gw + 4 {
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(gtext, theme::muted(&th)))),
                Rect { x: inner.x + inner.width - gw, y: header.y, width: gw, height: 1 },
            );
        }
    }

    // Empty state: no models running yet.
    let cards_area = Rect {
        x: inner.x,
        y: inner.y.saturating_add(2),
        width: inner.width,
        height: inner.height.saturating_sub(2),
    };
    if cards.is_empty() {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    "no models running.",
                    theme::muted(&th),
                )),
                Line::from(Span::styled(
                    "the local Bonsai assistant starts automatically once accepted; \
                     click [ + new model ] or run /models to deploy another.",
                    theme::faint(&th),
                )),
            ])
            .wrap(Wrap { trim: false }),
            cards_area,
        );
        return;
    }

    // --- Scroll so the selected card is visible ---
    // Cards are reordered (favourites first, OpenWebUI pinned, downloaded appended)
    // so a card's position no longer equals its runtime index — locate the active
    // runtime's card by its stored index.
    let visible = (cards_area.height / DASH_CARD_STRIDE).max(1) as usize;
    let sel = cards
        .iter()
        .position(|c| c.runtime_index == Some(app.runtime_selected))
        .unwrap_or(0);
    let max_scroll = cards.len().saturating_sub(visible);
    if app.dash_scroll > max_scroll {
        app.dash_scroll = max_scroll;
    }
    if sel < app.dash_scroll {
        app.dash_scroll = sel;
    } else if sel >= app.dash_scroll + visible {
        app.dash_scroll = sel + 1 - visible;
    }
    let start = app.dash_scroll;

    for (row, (i, card)) in cards
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .enumerate()
    {
        let y = cards_area.y + (row as u16) * DASH_CARD_STRIDE;
        if y + DASH_CARD_H > cards_area.y + cards_area.height {
            break;
        }
        let card_rect = Rect { x: inner.x, y, width: inner.width, height: DASH_CARD_H };
        draw_dash_card(f, card_rect, app, i, card, i == sel);
    }

    // Scroll hint when more cards exist off-screen.
    if cards.len() > visible {
        let hint = format!("  {}–{} of {} · scroll / PgUp-PgDn", start + 1, (start + visible).min(cards.len()), cards.len());
        let hy = cards_area.y + cards_area.height.saturating_sub(1);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, theme::faint(&th)))),
            Rect { x: inner.x, y: hy, width: inner.width, height: 1 },
        );
    }
}

/// Render one dashboard card and register its click regions. `idx` is the
/// `all_runtimes()` index (what the buttons act on); `selected` highlights it.
fn draw_dash_card(f: &mut Frame, rect: Rect, app: &mut App, idx: usize, card: &DashCard, selected: bool) {
    let th = app.theme;
    let w = rect.width as usize;

    // The whole card is clickable to select the model; buttons registered later
    // win the reverse hit-test.
    click(app, rect, ClickTarget::DashCard(idx));

    // Line 1: glyph · name · backend · status · metrics · active marker.
    let glyph_style = if card.status_is_error {
        fg(&th).add_modifier(Modifier::BOLD)
    } else if selected {
        theme::accent(&th).add_modifier(Modifier::BOLD)
    } else {
        theme::muted(&th)
    };
    let name_style = if selected {
        theme::accent(&th).add_modifier(Modifier::BOLD)
    } else {
        fg(&th)
    };
    let mut l1: Vec<Span> = vec![Span::styled(format!("{} ", card.glyph), glyph_style)];
    // Favourite star (filled when starred) before the name.
    if card.model_id.is_some() {
        l1.push(Span::styled(
            if card.is_favorite { "★ " } else { "" },
            theme::accent(&th),
        ));
    }
    l1.extend([
        Span::styled(card.name.clone(), name_style),
        sep(&th),
        Span::styled(card.backend_label.clone(), theme::muted(&th)),
        sep(&th),
        Span::styled(
            card.status_label.clone(),
            if card.status_is_error {
                fg(&th).add_modifier(Modifier::BOLD)
            } else {
                theme::muted(&th)
            },
        ),
    ]);
    if let Some(v) = card.vram_bytes {
        l1.push(sep(&th));
        l1.push(Span::styled(format!("vram ~{:.1}G", gib_f(v)), fg(&th)));
    }
    if let Some(d) = card.disk_bytes {
        l1.push(sep(&th));
        l1.push(Span::styled(format!("disk {}", human_size(d)), theme::muted(&th)));
    }
    if let Some(t) = card.tok_per_sec {
        l1.push(sep(&th));
        l1.push(Span::styled(format!("{t:.0} tok/s"), fg(&th)));
    }
    if let Some(u) = card.ctx_used {
        l1.push(sep(&th));
        l1.push(Span::styled(
            format!("ctx {}/{}", human_ctx(u), human_ctx(card.ctx_max)),
            fg(&th),
        ));
    }
    if card.is_openwebui {
        if let Some(url) = &card.url {
            l1.push(sep(&th));
            l1.push(Span::styled(format!("→ {url}"), theme::accent(&th)));
        }
    }
    if card.is_active {
        l1.push(Span::styled("  ★ next request", theme::accent(&th)));
    }
    f.render_widget(
        Paragraph::new(sel_line(&th, l1, rect.width, selected)),
        Rect { x: rect.x, y: rect.y, width: rect.width, height: 1 },
    );

    // Line 2: launch command + [ copy ] button (right).
    let copy_label = "copy";
    let copy_w = button_width(copy_label);
    let cmd_w = w.saturating_sub(copy_w as usize + 5);
    let cmd_text = if card.command.is_empty() {
        "(command unavailable)".to_string()
    } else {
        format!("$ {}", card.command)
    };
    let l2 = Line::from(vec![
        Span::raw("  "),
        Span::styled(clip(&cmd_text, cmd_w), theme::faint(&th)),
    ]);
    f.render_widget(Paragraph::new(l2), Rect { x: rect.x, y: rect.y + 1, width: rect.width, height: 1 });
    if !card.command.is_empty() {
        let cx = rect.x + rect.width.saturating_sub(copy_w + 1);
        let crect = Rect { x: cx, y: rect.y + 1, width: copy_w, height: 1 };
        f.render_widget(Paragraph::new(Line::from(button(&th, copy_label, false))), crect);
        click(app, crect, ClickTarget::DashCopyCmd(idx));
    }

    // Line 3: newest log line.
    let log_line = card.logs.last().cloned().unwrap_or_else(|| "(no logs yet)".into());
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("  log  ", theme::faint(&th)),
            Span::styled(clip(&log_line, w.saturating_sub(8)), theme::muted(&th)),
        ])),
        Rect { x: rect.x, y: rect.y + 2, width: rect.width, height: 1 },
    );

    // Line 4: either the error (+ copy-error button) or a 2nd log line.
    let row4 = Rect { x: rect.x, y: rect.y + 3, width: rect.width, height: 1 };
    if let Some(err) = &card.error_text {
        let ce_label = "copy error";
        let ce_w = button_width(ce_label);
        let first = err.lines().next().unwrap_or("");
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("  ! ", fg(&th).add_modifier(Modifier::BOLD)),
                Span::styled(clip(first, w.saturating_sub(ce_w as usize + 6)), fg(&th).add_modifier(Modifier::BOLD)),
            ])),
            row4,
        );
        let ex = rect.x + rect.width.saturating_sub(ce_w + 1);
        let erect = Rect { x: ex, y: row4.y, width: ce_w, height: 1 };
        f.render_widget(Paragraph::new(Line::from(button(&th, ce_label, true))), erect);
        click(app, erect, ClickTarget::DashCopyErr(idx));
    } else {
        let second = if card.logs.len() >= 2 {
            card.logs[card.logs.len() - 2].clone()
        } else {
            String::new()
        };
        if !second.is_empty() {
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("  log  ", theme::faint(&th)),
                    Span::styled(clip(&second, w.saturating_sub(8)), theme::faint(&th)),
                ])),
                row4,
            );
        }
    }

    // Line 5: action buttons. Running: [stop] [use] [★]. Downloaded-only:
    // [deploy] [delete] [★]. OpenWebUI: [stop] [copy url].
    let row5 = Rect { x: rect.x, y: rect.y + 4, width: rect.width, height: 1 };
    let mut bx = rect.x + 2;
    let mut bspans: Vec<Span> = vec![Span::raw("  ")];
    // Helper to push a button + register its click region + advance the cursor.
    let push_btn =
        |app: &mut App, bspans: &mut Vec<Span>, bx: &mut u16, label: &str, filled: bool, target: ClickTarget| {
            let bw = button_width(label);
            bspans.extend(button(&th, label, filled));
            bspans.push(Span::raw(" "));
            click(app, Rect { x: *bx, y: row5.y, width: bw, height: 1 }, target);
            *bx += bw + 1;
        };

    if card.is_openwebui {
        if card.can_stop {
            push_btn(app, &mut bspans, &mut bx, "stop", false, ClickTarget::DashStop(idx));
        }
        push_btn(app, &mut bspans, &mut bx, "copy url", false, ClickTarget::DashUse(idx));
    } else if card.runtime_index.is_some() {
        if card.can_stop {
            push_btn(app, &mut bspans, &mut bx, "stop", false, ClickTarget::DashStop(idx));
        }
        push_btn(app, &mut bspans, &mut bx, "use", card.is_active, ClickTarget::DashUse(idx));
        if card.model_id.is_some() {
            let star = if card.is_favorite { "★" } else { "☆" };
            push_btn(app, &mut bspans, &mut bx, star, card.is_favorite, ClickTarget::DashFavorite(idx));
        }
    } else {
        // Downloaded-only.
        if card.can_deploy {
            push_btn(app, &mut bspans, &mut bx, "deploy", false, ClickTarget::DashDeploy(idx));
        }
        if card.can_delete {
            push_btn(app, &mut bspans, &mut bx, "delete", true, ClickTarget::DashDelete(idx));
        }
        if card.model_id.is_some() {
            let star = if card.is_favorite { "★" } else { "☆" };
            push_btn(app, &mut bspans, &mut bx, star, card.is_favorite, ClickTarget::DashFavorite(idx));
        }
    }
    f.render_widget(Paragraph::new(Line::from(bspans)), row5);
}

// ---------------------------------------------------------------------------
// Sessions — past chats for this workspace (/sessions)
//
// One row per session file, newest first:
//   2h ago  fix the deploy retry loop                        14 msgs  · current
// ↑↓/wheel move the highlight, Enter or a click resumes that chat, and the
// header carries a [ + new chat ] button (same as /new).
// ---------------------------------------------------------------------------

fn draw_sessions(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    let inner = pad(area);
    if inner.height < 3 || inner.width < 16 {
        return;
    }

    // --- Header row: title · [ + new chat ] ---
    let title = format!("past chats ({})", app.sessions.len());
    let mut hspans = vec![Span::styled(
        title.clone(),
        theme::muted(&th).add_modifier(Modifier::BOLD),
    )];
    hspans.push(Span::raw("  "));
    let new_label = "+ new chat";
    let new_w = button_width(new_label);
    hspans.extend(button(&th, new_label, false));
    f.render_widget(
        Paragraph::new(Line::from(hspans)),
        Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 },
    );
    let new_x = inner.x + (format!("{title}  ").width() as u16);
    click(
        app,
        Rect { x: new_x, y: inner.y, width: new_w, height: 1 },
        ClickTarget::SessionsNew,
    );

    let rows_area = Rect {
        x: inner.x,
        y: inner.y + 2,
        width: inner.width,
        height: inner.height.saturating_sub(2),
    };
    if app.sessions.is_empty() {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    "no saved chats for this workspace yet.",
                    theme::muted(&th),
                )),
                Line::from(Span::styled(
                    "chats save automatically as you talk — start another with [ + new chat ] \
                     or /new and switch back here any time.",
                    theme::faint(&th),
                )),
            ])
            .wrap(Wrap { trim: false }),
            rows_area,
        );
        return;
    }

    // --- Scroll so the selected row stays visible (rows are one line each) ---
    let total = app.sessions.len();
    let mut visible = rows_area.height as usize;
    if total > visible && visible > 1 {
        visible -= 1; // reserve the bottom line for the range hint
    }
    let visible = visible.max(1);
    let sel = app.session_selected.min(total - 1);
    app.session_selected = sel;
    let max_scroll = total.saturating_sub(visible);
    if app.sessions_scroll > max_scroll {
        app.sessions_scroll = max_scroll;
    }
    if sel < app.sessions_scroll {
        app.sessions_scroll = sel;
    } else if sel >= app.sessions_scroll + visible {
        app.sessions_scroll = sel + 1 - visible;
    }
    let start = app.sessions_scroll;
    let shown = visible.min(total - start);

    // Column budget: age + gap + title (flex) + gap + msgs + current tag.
    const AGE_W: usize = 9;
    const MSGS_W: usize = 9; // "1234 msgs"
    const CUR_W: usize = 11; // "  · current" for the live chat's row
    let title_w = (inner.width as usize).saturating_sub(AGE_W + 2 + MSGS_W + CUR_W + 2);

    let mut lines: Vec<Line> = Vec::with_capacity(shown);
    for (row, m) in app.sessions.iter().enumerate().skip(start).take(shown) {
        let selected = row == sel;
        let is_current = m.id == app.current_session_id;
        let name = clip(crate::app::display_title(&m.title), title_w);
        let pad_n = title_w.saturating_sub(name.width());
        let name_style = if selected {
            theme::accent(&th).add_modifier(Modifier::BOLD)
        } else {
            fg(&th)
        };
        let mut spans = vec![
            Span::styled(format!("{:>AGE_W$}  ", rel_age(m.updated_at)), theme::muted(&th)),
            Span::styled(name, name_style),
            Span::raw(" ".repeat(pad_n + 2)),
            Span::styled(format!("{:>4} msgs", m.message_count), theme::muted(&th)),
        ];
        if app.session_is_working(&m.id) {
            spans.push(Span::styled(
                "  · working…",
                theme::accent(&th).add_modifier(Modifier::BOLD),
            ));
        }
        if is_current {
            spans.push(Span::styled("  · current", theme::accent(&th)));
        }
        lines.push(sel_line(&th, spans, inner.width, selected));
    }
    // No wrap: rows stay one-per-line so the SessionList click math holds.
    f.render_widget(
        Paragraph::new(lines),
        Rect { x: rows_area.x, y: rows_area.y, width: rows_area.width, height: shown as u16 },
    );
    click(
        app,
        Rect { x: rows_area.x, y: rows_area.y, width: rows_area.width, height: shown as u16 },
        ClickTarget::SessionList,
    );

    // Range hint when more rows exist off-screen.
    if total > visible {
        let hint = format!(
            "  {}–{} of {} · scroll / PgUp-PgDn",
            start + 1,
            (start + visible).min(total),
            total
        );
        let hy = rows_area.y + rows_area.height.saturating_sub(1);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, theme::faint(&th)))),
            Rect { x: rows_area.x, y: hy, width: rows_area.width, height: 1 },
        );
    }
}

/// Compact "how long ago" for a session row; future timestamps (clock skew)
/// read as "just now".
fn rel_age(t: std::time::SystemTime) -> String {
    let secs = t.elapsed().map(|d| d.as_secs()).unwrap_or(0);
    match secs {
        0..=59 => "just now".into(),
        60..=3_599 => format!("{}m ago", secs / 60),
        3_600..=86_399 => format!("{}h ago", secs / 3_600),
        _ => format!("{}d ago", secs / 86_400),
    }
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
        llines.push(sel_line(
            &th,
            vec![Span::styled(srv.name.clone(), name_style)],
            linner.width,
            selected,
        ));
        llines.push(sel_line(
            &th,
            vec![Span::styled(format!("{word} · {}", srv.host), theme::muted(&th))],
            linner.width,
            selected,
        ));
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
    f.render_widget(Paragraph::new(Line::from(button(&th, "+ new server", true))), newrect);
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
            lines.push(sel_line(
                &th,
                vec![
                    Span::styled(format!("{label:<10} "), theme::muted(&th)),
                    Span::styled(shown, val_style),
                ],
                right.width,
                selected,
            ));
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
        ("connect", ClickTarget::RemoteConnect),
        ("save", ClickTarget::RemoteSave),
        ("disconnect", ClickTarget::RemoteDisconnect),
        ("delete", ClickTarget::RemoteDelete),
    ];
    let mut bx = right.x;
    let mut bspans: Vec<Span> = Vec::new();
    for (label, target) in buttons {
        let w = button_width(label);
        if bx + w <= right.x + right.width {
            click(app, Rect { x: bx, y: actions_row, width: w, height: 1 }, target);
        }
        bspans.extend(button(&th, label, true));
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
            spans.extend(button(&th, "install", true));
            Some(ClickTarget::BackendInstall(i))
        } else {
            spans.extend(button(&th, "smoke-test", true));
            Some(ClickTarget::BackendSmoke(i))
        };
        let row_rect = Rect { x: inner.x, y, width: inner.width, height: 1 };
        let para = Paragraph::new(Line::from(spans));
        f.render_widget(if selected { para.style(theme::selected(&th)) } else { para }, row_rect);
        if let Some(t) = row_action {
            click(app, row_rect, t);
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
                    l2.extend(button(&th, "fix", true));
                }
                f.render_widget(
                    Paragraph::new(Line::from(l2)),
                    Rect { x: inner.x, y, width: inner.width, height: 1 },
                );
                if has_fix {
                    let fw = button_width("fix");
                    let fx = (inner.x + pw).min(inner.x + inner.width.saturating_sub(fw));
                    click(app, Rect { x: fx, y, width: fw, height: 1 }, ClickTarget::BackendFix(i));
                }
                y += 1;
            }
        }
    }

    y += 1;
    if y < bottom {
        f.render_widget(
            Paragraph::new(Line::from(button(&th, "re-detect", true))),
            Rect { x: inner.x, y, width: inner.width, height: 1 },
        );
        click(
            app,
            Rect { x: inner.x, y, width: button_width("re-detect"), height: 1 },
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
    let run_w = button_width("run");
    let rx = inner.x + inner.width - run_w;
    f.render_widget(Paragraph::new(Line::from(button(&th, "run", true))), Rect { x: rx, y: inner.y, width: run_w, height: 1 });
    click(app, Rect { x: rx, y: inner.y, width: run_w, height: 1 }, ClickTarget::BenchRun);

    let mut lines: Vec<Line> = vec![Line::from("")];
    match &app.last_bench_result {
        None => {
            lines.push(Line::from(Span::styled("no runs yet — press the run button", theme::muted(&th))));
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
        let aw = button_width(action);
        let used = mark.width() + title.width();
        let padn = w.saturating_sub(used + aw as usize);
        let action_x = inner.x + inner.width.saturating_sub(aw);
        let mut line_spans = vec![
            Span::styled(mark, mark_style),
            Span::styled((*title).to_string(), title_style),
            Span::raw(" ".repeat(padn)),
        ];
        // Pending steps get a filled button; completed ones an outline.
        line_spans.extend(button(&th, action, !*ok));
        rows.push((
            Line::from(line_spans),
            Some((action_x, aw, ClickTarget::SetupStep(*idx))),
        ));
        rows.push((Line::from(Span::styled(format!("    {subtitle}"), theme::muted(&th))), None));
    }

    // Doctor block — probes derived from live state.
    rows.push((Line::from(""), None));
    let mut doctor_spans = vec![Span::styled("doctor   ", theme::muted(&th))];
    doctor_spans.extend(button(&th, "run doctor", true));
    rows.push((
        Line::from(doctor_spans),
        Some((inner.x + 9, button_width("run doctor"), ClickTarget::SetupDoctor)),
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

fn draw_settings(f: &mut Frame, area: Rect, app: &mut App) {
    let th = app.theme;
    let inner = pad(area);
    app.settings_view_height = inner.height;

    let rows = app.settings_rows();
    let editing = app.settings_editing_field();
    let sel = app.settings_sel;
    let scroll = app.settings_scroll as usize;
    let val_w = inner.width.saturating_sub(26) as usize;

    // Row-by-row with the scroll offset so click y-positions stay exact.
    let mut act_idx = 0usize;
    for (i, r) in rows.iter().enumerate() {
        let is_action = r.action.is_some();
        if i < scroll {
            if is_action {
                act_idx += 1;
            }
            continue;
        }
        let y = inner.y + (i - scroll) as u16;
        if y >= inner.y + inner.height {
            break;
        }
        let selected = is_action && act_idx == sel;
        let marker = if selected { "› " } else { "  " };
        let name_style = if selected {
            theme::accent(&th).add_modifier(Modifier::BOLD)
        } else {
            fg(&th)
        };

        let line = match &r.kind {
            SettingsRowKind::Header => Line::from(Span::styled(
                r.label.clone(),
                theme::muted(&th).add_modifier(Modifier::BOLD),
            )),
            SettingsRowKind::Toggle(on) => Line::from(vec![
                Span::styled(marker, theme::accent(&th)),
                Span::styled(
                    if *on { "[x] " } else { "[ ] " },
                    if *on { theme::accent(&th) } else { theme::faint(&th) },
                ),
                Span::styled(format!("{:<18}", clip(&r.label, 18)), name_style),
                Span::styled(clip(&r.value, val_w), theme::muted(&th)),
            ]),
            SettingsRowKind::Text => {
                let is_editing = matches!(
                    (editing, r.action),
                    (Some(a), Some(SettingAction::Edit(b))) if a == b
                );
                let value = if is_editing {
                    Span::styled(format!("{}▌", app.settings_edit_buffer()), theme::accent(&th))
                } else {
                    Span::styled(clip(&r.value, val_w), fg(&th))
                };
                Line::from(vec![
                    Span::styled(marker, theme::accent(&th)),
                    Span::styled(format!("{:<18}", clip(&r.label, 18)), name_style),
                    value,
                ])
            }
            SettingsRowKind::Action(word) => {
                let mut spans = vec![
                    Span::styled(marker, theme::accent(&th)),
                    Span::styled(format!("{:<18}", clip(&r.label, 18)), name_style),
                ];
                spans.extend(button(&th, word, true));
                spans.push(Span::styled(format!("  {}", clip(&r.value, 28)), theme::faint(&th)));
                Line::from(spans)
            }
            SettingsRowKind::Info => Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{:<18}", clip(&r.label, 18)), theme::muted(&th)),
                Span::styled(clip(&r.value, val_w), theme::faint(&th)),
            ]),
        };

        let row_rect = Rect { x: inner.x, y, width: inner.width, height: 1 };
        let para = Paragraph::new(line);
        f.render_widget(if selected { para.style(theme::selected(&th)) } else { para }, row_rect);
        if let Some(action) = r.action {
            click(app, row_rect, ClickTarget::Setting(action));
            act_idx += 1;
        }
    }
}
