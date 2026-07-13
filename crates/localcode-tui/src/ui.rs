//! View rendering for all tabs.

use crate::app::{App, Tab};
use crate::theme;
use crate::widgets::{draw_modal, draw_palette};
use localcode_core::theme::ThemeToken;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap};
use ratatui::Frame;

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    if area.width < 40 || area.height < 10 {
        f.render_widget(
            Paragraph::new("Terminal too small. Resize to continue.")
                .style(theme::warn(&app.theme)),
            area,
        );
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(20), Constraint::Length(14)])
        .split(area);

    let main = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(chunks[0]);

    draw_title(f, main[0], app);
    draw_view(f, main[1], app);
    draw_status(f, main[2], app);
    draw_rail(f, chunks[1], app);

    if let Some(modal) = &app.modal {
        draw_modal(f, area, modal, &app.theme);
    }
    if app.palette_open {
        draw_palette(
            f,
            area,
            &app.palette_query,
            &app.palette_items(),
            app.palette_selected,
            &app.theme,
        );
    }
    if app.assistant_open {
        draw_assistant_dock(f, area, app);
    }
}

fn draw_title(f: &mut Frame, area: Rect, app: &App) {
    let model = app
        .active_runtime_name()
        .unwrap_or_else(|| "no runtime".into());
    let gpu = app.gpu.summary();
    let title = format!(
        " LocalCode v{}  │  {}  │  {}  │  {} ",
        env!("CARGO_PKG_VERSION"),
        app.tab.as_str(),
        model,
        gpu
    );
    f.render_widget(
        Paragraph::new(title).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme::border(&app.theme))
                .title(" LocalCode "),
        ),
        area,
    );
}

fn draw_status(f: &mut Frame, area: Rect, app: &App) {
    let style = if app.status_is_error {
        theme::error(&app.theme)
    } else {
        theme::muted(&app.theme)
    };
    let mut text = app.status_line.clone();
    if app.last_error.is_some() {
        text.push_str("  │  [a] Ask assistant  [l] Logs");
    }
    f.render_widget(
        Paragraph::new(text).style(style).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme::border(&app.theme))
                .title(" status "),
        ),
        area,
    );
}

fn draw_rail(f: &mut Frame, area: Rect, app: &App) {
    let tabs = Tab::all();
    let items: Vec<ListItem> = tabs
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let active = *t == app.tab;
            let focused = app.rail_focus && app.rail_index == i;
            let hover_region = app.rail_hover;
            let style = if active {
                theme::nav_active(&app.theme)
            } else if focused || (hover_region && app.config.ui.right_rail_hover_brightens) {
                theme::nav_hover(&app.theme)
            } else {
                theme::nav_idle(&app.theme)
            };
            let label = if active {
                format!(" › {}", t.as_str())
            } else {
                format!("   {}", t.as_str())
            };
            ListItem::new(label).style(style)
        })
        .collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme::border(&app.theme))
        .title(" nav ");
    f.render_widget(List::new(items).block(block), area);
}

fn draw_view(f: &mut Frame, area: Rect, app: &App) {
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

fn draw_dashboard(f: &mut Frame, area: Rect, app: &App) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let runtimes: Vec<ListItem> = if app.runtimes.is_empty() {
        vec![ListItem::new("No active runtimes — deploy from Models")]
    } else {
        app.runtimes
            .iter()
            .map(|r| {
                ListItem::new(format!(
                    "• {} [{:?}] {}",
                    r.name, r.status, r.base_url
                ))
            })
            .collect()
    };

    f.render_widget(
        List::new(runtimes).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Active runtimes ")
                .border_style(theme::border(&app.theme)),
        ),
        cols[0],
    );

    let mut right = vec![
        Line::from(Span::styled("GPU", theme::accent(&app.theme).add_modifier(Modifier::BOLD))),
        Line::from(app.gpu.summary()),
    ];
    for w in &app.gpu.warnings {
        right.push(Line::from(Span::styled(w.as_str(), theme::warn(&app.theme))));
    }
    right.push(Line::from(""));
    right.push(Line::from(Span::styled(
        "Quick actions",
        theme::accent(&app.theme).add_modifier(Modifier::BOLD),
    )));
    right.push(Line::from("  [2] Models / Deploy"));
    right.push(Line::from("  [4] Coding"));
    right.push(Line::from("  [3] Benchmarks"));
    right.push(Line::from("  [5] Setup / Top-up"));
    right.push(Line::from(""));
    right.push(Line::from(format!(
        "API: {}",
        if app.api_healthy { "ok" } else { "offline (local-first OK)" }
    )));
    right.push(Line::from(format!(
        "Assistant: {}",
        if app.assistant_configured {
            "configured"
        } else {
            "not configured"
        }
    )));
    if !app.notifications.is_empty() {
        right.push(Line::from(""));
        right.push(Line::from(Span::styled(
            "Recent",
            theme::muted(&app.theme),
        )));
        for n in app.notifications.iter().rev().take(5) {
            right.push(Line::from(format!("  {}", n.title)));
        }
    }

    f.render_widget(
        Paragraph::new(right).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Dashboard ")
                .border_style(theme::border(&app.theme)),
        ),
        cols[1],
    );
}

fn draw_models(f: &mut Frame, area: Rect, app: &App) {
    let ratios = app.pane_ratios("models", &[0.3, 0.4, 0.3]);
    let panes = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio((ratios[0] * 100.0) as u32, 100),
            Constraint::Ratio((ratios[1] * 100.0) as u32, 100),
            Constraint::Ratio((ratios[2] * 100.0) as u32, 100),
        ])
        .split(area);

    // List
    let mut list_items = vec![ListItem::new(format!(
        "Search: {}{}",
        app.model_query,
        if app.model_search_focus { "▌" } else { "" }
    ))
    .style(theme::accent(&app.theme))];
    if app.models.is_empty() {
        list_items.push(ListItem::new("No results. Press / to search, p popular, t trending."));
    }
    for (i, m) in app.models.iter().enumerate() {
        let sel = i == app.model_selected;
        let style = if sel {
            theme::nav_active(&app.theme)
        } else {
            theme::muted(&app.theme)
        };
        let stats = format!(
            "↓{} ♥{}",
            m.downloads.unwrap_or(0),
            m.likes.unwrap_or(0)
        );
        list_items.push(
            ListItem::new(format!("{}  {stats}", m.id)).style(style),
        );
    }
    f.render_widget(
        List::new(list_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Models ")
                .border_style(theme::border(&app.theme)),
        ),
        panes[0],
    );

    // Detail
    let detail = if let Some(d) = &app.model_detail {
        let mut lines = vec![
            Line::from(Span::styled(
                d.summary.id.clone(),
                theme::accent(&app.theme).add_modifier(Modifier::BOLD),
            )),
            Line::from(format!(
                "license: {}  size: {}",
                d.license.as_deref().unwrap_or("?"),
                d.parameter_size.as_deref().unwrap_or("?")
            )),
            Line::from(""),
            Line::from(Span::styled("Quants", theme::muted(&app.theme))),
        ];
        for q in &d.quants {
            let gib = q.total_size as f64 / (1024.0 * 1024.0 * 1024.0);
            let mark = if Some(q.label.as_str()) == app.selected_quant.as_deref() {
                "▶"
            } else {
                " "
            };
            lines.push(Line::from(format!(
                "{mark} {}  {:.2} GiB  ({} files)",
                q.label,
                gib,
                q.files.len()
            )));
        }
        if let Some(card) = &d.card_markdown {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("Card", theme::muted(&app.theme))));
            for l in card.lines().take(20) {
                lines.push(Line::from(l.to_string()));
            }
        }
        lines
    } else {
        vec![Line::from("Select a model to view details.")]
    };
    f.render_widget(
        Paragraph::new(detail).wrap(Wrap { trim: true }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Detail ")
                .border_style(theme::border(&app.theme)),
        ),
        panes[1],
    );

    // Deploy panel
    let mut deploy = vec![
        Line::from(Span::styled(
            "Deploy panel",
            theme::accent(&app.theme).add_modifier(Modifier::BOLD),
        )),
        Line::from(format!("Backend: {}", app.deploy_backend.as_str())),
        Line::from(format!(
            "Quant: {}",
            app.selected_quant.as_deref().unwrap_or("(none)")
        )),
        Line::from(format!("Ctx: {}  Port: {:?}", app.deploy_ctx, app.deploy_port)),
        Line::from(format!(
            "Target: {}",
            if app.deploy_cloud { "Cloud" } else { "Local" }
        )),
        Line::from(""),
        Line::from("[d] one-click Deploy"),
        Line::from("[b] cycle backend  [c] cloud toggle"),
        Line::from("[Enter] open detail  [/] search"),
    ];
    if let Some(fit) = &app.last_fit {
        deploy.push(Line::from(""));
        deploy.push(Line::from(format!(
            "Est. VRAM: {:.2} GiB",
            fit.estimated_vram_bytes as f64 / (1024.0f64.powi(3))
        )));
        deploy.push(Line::from(format!(
            "Free/Total: {:.1}/{:.1} GiB",
            fit.free_vram_bytes as f64 / (1024.0f64.powi(3)),
            fit.total_vram_bytes as f64 / (1024.0f64.powi(3))
        )));
        if let Some(w) = &fit.warning {
            deploy.push(Line::from(Span::styled(w.as_str(), theme::warn(&app.theme))));
        }
    }
    if app.deploy_progress > 0 && app.deploy_progress < 100 {
        // progress shown via gauge below text — use status for message
    }
    f.render_widget(
        Paragraph::new(deploy).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Deploy ")
                .border_style(theme::border(&app.theme)),
        ),
        panes[2],
    );

    if app.deploy_progress > 0 && app.deploy_progress < 100 {
        let gauge_area = Rect {
            x: panes[2].x + 1,
            y: panes[2].y + panes[2].height.saturating_sub(3),
            width: panes[2].width.saturating_sub(2),
            height: 1,
        };
        f.render_widget(
            Gauge::default()
                .percent(app.deploy_progress as u16)
                .label(format!("{}%", app.deploy_progress))
                .gauge_style(theme::accent(&app.theme)),
            gauge_area,
        );
    }
}

fn draw_benchmarks(f: &mut Frame, area: Rect, app: &App) {
    let lines = vec![
        Line::from(Span::styled(
            "Benchmarks",
            theme::accent(&app.theme).add_modifier(Modifier::BOLD),
        )),
        Line::from("Suite: localcode-sample-coding v1.0.0"),
        Line::from(""),
        Line::from("[r] Run sample suite on active runtime"),
        Line::from("[p] Publish last result (requires sign-in)"),
        Line::from(""),
        Line::from(app.last_bench_summary.clone()),
    ];
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Benchmarks ")
                .border_style(theme::border(&app.theme)),
        ),
        area,
    );
}

fn draw_coding(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(area);

    let header = format!(
        "Runtime: {}  │  Subagents: {}  │  Skills: {}  │  MCP: {}",
        app.active_runtime_name().unwrap_or_else(|| "none".into()),
        if app.subagents_enabled { "ON" } else { "OFF" },
        app.skill_count,
        app.mcp_status_summary
    );
    f.render_widget(
        Paragraph::new(header).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Coding ")
                .border_style(theme::border(&app.theme)),
        ),
        chunks[0],
    );

    let transcript: Vec<Line> = app
        .coding_transcript
        .iter()
        .map(|m| {
            let style = if m.starts_with("you:") {
                theme::accent(&app.theme)
            } else if m.starts_with("error:") {
                theme::error(&app.theme)
            } else {
                Style::default().fg(theme::color(&app.theme, ThemeToken::Fg))
            };
            Line::from(Span::styled(m.clone(), style))
        })
        .collect();
    f.render_widget(
        Paragraph::new(transcript)
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Transcript ")
                    .border_style(theme::border(&app.theme)),
            ),
        chunks[1],
    );

    f.render_widget(
        Paragraph::new(format!(
            "> {}{}",
            app.coding_input,
            if app.coding_input_focus { "▌" } else { "" }
        ))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Composer (Enter send, s toggle subagents) ")
                .border_style(theme::border(&app.theme)),
        ),
        chunks[2],
    );
}

fn draw_setup(f: &mut Frame, area: Rect, app: &App) {
    let mut lines = vec![
        Line::from(Span::styled(
            "Setup wizard",
            theme::accent(&app.theme).add_modifier(Modifier::BOLD),
        )),
        Line::from("1. GPUs detected — see Dashboard"),
        Line::from("2. Backends:"),
    ];
    for b in &app.backend_reports {
        let mark = if b.ready { "✓" } else { "·" };
        lines.push(Line::from(format!(
            "   {mark} {:?} ready={} {}",
            b.kind,
            b.ready,
            b.notes.first().cloned().unwrap_or_default()
        )));
    }
    lines.push(Line::from("3. HF token: set HF_TOKEN env for gated models"));
    lines.push(Line::from(format!(
        "   endpoint: {}",
        app.config.registry.endpoint
    )));
    lines.push(Line::from("4. Cloud keys: RUNPOD_API_KEY, VAST_API_KEY"));
    lines.push(Line::from(
        "5. Akash: managed account + USDC top-up (Base L2) — custody disclosed",
    ));
    lines.push(Line::from(
        "6. Assistant: OPENROUTER_API_KEY or openai-compatible URL",
    ));
    lines.push(Line::from(""));
    lines.push(Line::from("[d] Run doctor diagnostics"));
    if let Some(doc) = &app.doctor_summary {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Doctor summary",
            theme::muted(&app.theme),
        )));
        for l in doc.lines().take(15) {
            lines.push(Line::from(l.to_string()));
        }
    }
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Setup ")
                .border_style(theme::border(&app.theme)),
        ),
        area,
    );
}

fn draw_notifications(f: &mut Frame, area: Rect, app: &App) {
    let items: Vec<ListItem> = if app.notifications.is_empty() {
        vec![ListItem::new("No notifications")]
    } else {
        app.notifications
            .iter()
            .rev()
            .map(|n| {
                ListItem::new(format!(
                    "[{:?}] {} — {}",
                    n.severity, n.title, n.body
                ))
            })
            .collect()
    };
    f.render_widget(
        List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Notifications ")
                .border_style(theme::border(&app.theme)),
        ),
        area,
    );
}

fn draw_settings(f: &mut Frame, area: Rect, app: &App) {
    let lines = vec![
        Line::from(Span::styled(
            "Settings",
            theme::accent(&app.theme).add_modifier(Modifier::BOLD),
        )),
        Line::from(format!("Theme: {:?}", app.config.ui.theme)),
        Line::from(format!("Mouse: {}", app.config.ui.mouse)),
        Line::from(format!(
            "Default backend: {}",
            app.config.backends.default.kind
        )),
        Line::from(format!(
            "Subagents default: {}",
            app.config.agent.subagents_enabled
        )),
        Line::from(format!(
            "Confirm destructive tools: {}",
            app.config.agent.confirm_destructive_tools
        )),
        Line::from(format!("Log level: {}", app.config.logging.level)),
        Line::from(format!(
            "Redact secrets: {}",
            app.config.logging.redact_secrets
        )),
        Line::from(format!("API: {}", app.config.api.base_url)),
        Line::from(format!(
            "Registry: {}",
            app.config.registry.api_endpoint
        )),
        Line::from(""),
        Line::from("Config file:"),
        Line::from(format!("  {}", app.paths.config_file().display())),
        Line::from(""),
        Line::from("Payments: USDC on Base (v1)"),
        Line::from("Feature flags: cloud adapters experimental"),
        Line::from(""),
        Line::from("[t] cycle theme  Ctrl+S save config"),
    ];
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Settings ")
                .border_style(theme::border(&app.theme)),
        ),
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
        .unwrap_or_else(|| "Ask assistant about the last error or type a question in Coding.".into());
    f.render_widget(
        Paragraph::new(body)
            .wrap(Wrap { trim: true })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Assistant (Esc close) ")
                    .border_style(theme::accent(&app.theme)),
            ),
        rect,
    );
}
