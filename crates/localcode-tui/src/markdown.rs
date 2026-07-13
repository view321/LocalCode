//! Minimal markdown → styled `Line`s for terminal rendering (model cards).
//!
//! Renders the subset that matters for Hugging Face READMEs: headings,
//! paragraphs, lists, code blocks, inline styles, quotes, tables, rules.
//! Raw HTML and YAML frontmatter are dropped. Wrapping is left to the
//! `Paragraph` widget, so lines here are logical lines.

use localcode_core::theme::{Theme, ThemeToken};
use pulldown_cmark::{Event, Options, Parser, Tag};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme;

/// Hard cap so a pathological README can't grow the frame unbounded.
const MAX_LINES: usize = 1500;

pub fn render(md: &str, th: &Theme) -> Vec<Line<'static>> {
    let src = strip_frontmatter(md);
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TASKLISTS);

    let mut r = Renderer::new(th);
    for ev in Parser::new_ext(src, opts) {
        if r.lines.len() > MAX_LINES {
            break;
        }
        r.event(ev);
    }
    r.finish()
}

/// Drop a leading `--- … ---` YAML block (HF model cards start with one).
fn strip_frontmatter(md: &str) -> &str {
    let s = md.trim_start_matches('\u{feff}');
    let rest = s.strip_prefix("---").filter(|r| r.starts_with(['\n', '\r']));
    if let Some(rest) = rest {
        // Find the closing fence at the start of a line.
        let mut offset = 0;
        for line in rest.split_inclusive('\n') {
            if line.trim_end() == "---" {
                return &rest[offset + line.len()..];
            }
            offset += line.len();
        }
    }
    s
}

/// What container each `Start` opened; `End` events just pop this stack, so
/// no matching on `TagEnd` payloads is needed.
enum Ctx {
    Paragraph,
    Heading,
    CodeBlock,
    List,
    Item,
    Quote,
    Emphasis,
    Strong,
    Strike,
    Link,
    Table,
    TableHead,
    TableRow,
    TableCell,
    Suppressed,
    Other,
}

#[derive(Default)]
struct TableState {
    rows: Vec<Vec<String>>,
    header_rows: usize,
    current_row: Vec<String>,
    in_cell: bool,
}

struct Renderer {
    fg: Style,
    accent: Style,
    muted: Style,
    lines: Vec<Line<'static>>,
    spans: Vec<Span<'static>>,
    stack: Vec<Ctx>,
    heading: u8,
    bold: u32,
    italic: u32,
    strike: u32,
    link: u32,
    quote: u32,
    suppress: u32,
    code_block: bool,
    list_counters: Vec<Option<u64>>,
    table: Option<TableState>,
}

impl Renderer {
    fn new(th: &Theme) -> Self {
        Self {
            fg: Style::default().fg(theme::color(th, ThemeToken::Fg)),
            accent: theme::accent(th),
            muted: theme::muted(th),
            lines: vec![],
            spans: vec![],
            stack: vec![],
            heading: 0,
            bold: 0,
            italic: 0,
            strike: 0,
            link: 0,
            quote: 0,
            suppress: 0,
            code_block: false,
            list_counters: vec![],
            table: None,
        }
    }

    fn cur_style(&self) -> Style {
        let mut s = if self.heading > 0 {
            self.accent.add_modifier(Modifier::BOLD)
        } else if self.quote > 0 {
            self.muted
        } else {
            self.fg
        };
        if self.bold > 0 {
            s = s.add_modifier(Modifier::BOLD);
        }
        if self.italic > 0 {
            s = s.add_modifier(Modifier::ITALIC);
        }
        if self.strike > 0 {
            s = s.add_modifier(Modifier::CROSSED_OUT);
        }
        if self.link > 0 {
            s = self.accent.add_modifier(Modifier::UNDERLINED);
        }
        s
    }

    fn flush(&mut self) {
        if self.spans.is_empty() {
            return;
        }
        let mut spans = std::mem::take(&mut self.spans);
        if self.quote > 0 {
            spans.insert(0, Span::styled("▎ ".to_string(), self.muted));
        }
        self.lines.push(Line::from(spans));
    }

    /// One blank spacer line, unless the output is empty or already spaced.
    fn blank(&mut self) {
        if matches!(self.lines.last(), Some(l) if !l.spans.is_empty()) {
            self.lines.push(Line::from(""));
        }
    }

    fn push_text(&mut self, text: &str) {
        if self.suppress > 0 {
            return;
        }
        if let Some(t) = &mut self.table {
            if t.in_cell {
                if let Some(cell) = t.current_row.last_mut() {
                    cell.push_str(text);
                }
                return;
            }
        }
        if self.code_block {
            let gutter = self.accent;
            let body = self.fg;
            let mut parts: Vec<&str> = text.split('\n').collect();
            // A trailing "\n" yields an empty final fragment; drop it.
            if text.ends_with('\n') {
                parts.pop();
            }
            for l in parts {
                self.lines.push(Line::from(vec![
                    Span::styled("│ ".to_string(), gutter),
                    Span::styled(l.to_string(), body),
                ]));
            }
            return;
        }
        let style = self.cur_style();
        self.spans.push(Span::styled(text.to_string(), style));
    }

    fn event(&mut self, ev: Event<'_>) {
        match ev {
            Event::Start(tag) => self.start(tag),
            Event::End(_) => self.end(),
            Event::Text(t) => self.push_text(&t),
            Event::Code(t) => {
                if self.suppress > 0 {
                    return;
                }
                if let Some(tb) = &mut self.table {
                    if tb.in_cell {
                        if let Some(cell) = tb.current_row.last_mut() {
                            cell.push_str(&t);
                        }
                        return;
                    }
                }
                self.spans
                    .push(Span::styled(t.to_string(), self.accent));
            }
            Event::SoftBreak => self.push_text(" "),
            Event::HardBreak => self.flush(),
            Event::Rule => {
                self.flush();
                self.blank();
                self.lines
                    .push(Line::from(Span::styled("─".repeat(32), self.muted)));
                self.blank();
            }
            Event::TaskListMarker(done) => {
                let mark = if done { "[x] " } else { "[ ] " };
                self.spans
                    .push(Span::styled(mark.to_string(), self.accent));
            }
            Event::FootnoteReference(name) => {
                self.spans
                    .push(Span::styled(format!("[^{name}]"), self.muted));
            }
            // Raw HTML (badges, center divs…) is noise in a terminal.
            Event::Html(_) | Event::InlineHtml(_) => {}
            _ => {}
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {
                self.flush();
                self.stack.push(Ctx::Paragraph);
            }
            Tag::Heading { level, .. } => {
                self.flush();
                self.blank();
                self.heading = level as u8;
                let marker = match self.heading {
                    1 => "◆ ",
                    2 => "▸ ",
                    _ => "· ",
                };
                self.spans.push(Span::styled(
                    marker.to_string(),
                    self.accent.add_modifier(Modifier::BOLD),
                ));
                self.stack.push(Ctx::Heading);
            }
            Tag::CodeBlock(_) => {
                self.flush();
                self.blank();
                self.code_block = true;
                self.stack.push(Ctx::CodeBlock);
            }
            Tag::List(start) => {
                self.flush();
                self.list_counters.push(start);
                self.stack.push(Ctx::List);
            }
            Tag::Item => {
                self.flush();
                let depth = self.list_counters.len().saturating_sub(1);
                let indent = "  ".repeat(depth);
                let marker = match self.list_counters.last_mut() {
                    Some(Some(n)) => {
                        let m = format!("{indent}{n}. ");
                        *n += 1;
                        m
                    }
                    _ => format!("{indent}• "),
                };
                self.spans.push(Span::styled(marker, self.accent));
                self.stack.push(Ctx::Item);
            }
            Tag::BlockQuote(_) => {
                self.flush();
                self.quote += 1;
                self.stack.push(Ctx::Quote);
            }
            Tag::Emphasis => {
                self.italic += 1;
                self.stack.push(Ctx::Emphasis);
            }
            Tag::Strong => {
                self.bold += 1;
                self.stack.push(Ctx::Strong);
            }
            Tag::Strikethrough => {
                self.strike += 1;
                self.stack.push(Ctx::Strike);
            }
            Tag::Link { .. } => {
                self.link += 1;
                self.stack.push(Ctx::Link);
            }
            Tag::Image { .. } => {
                // Keep the alt text, marked as an image.
                self.spans
                    .push(Span::styled("[img] ".to_string(), self.muted));
                self.stack.push(Ctx::Other);
            }
            Tag::Table(_) => {
                self.flush();
                self.blank();
                self.table = Some(TableState::default());
                self.stack.push(Ctx::Table);
            }
            Tag::TableHead => {
                if let Some(t) = &mut self.table {
                    t.current_row = vec![];
                }
                self.stack.push(Ctx::TableHead);
            }
            Tag::TableRow => {
                if let Some(t) = &mut self.table {
                    t.current_row = vec![];
                }
                self.stack.push(Ctx::TableRow);
            }
            Tag::TableCell => {
                if let Some(t) = &mut self.table {
                    t.current_row.push(String::new());
                    t.in_cell = true;
                }
                self.stack.push(Ctx::TableCell);
            }
            Tag::HtmlBlock | Tag::MetadataBlock(_) => {
                self.suppress += 1;
                self.stack.push(Ctx::Suppressed);
            }
            _ => self.stack.push(Ctx::Other),
        }
    }

    fn end(&mut self) {
        match self.stack.pop() {
            Some(Ctx::Paragraph) => {
                self.flush();
                self.blank();
            }
            Some(Ctx::Heading) => {
                self.flush();
                self.heading = 0;
                self.blank();
            }
            Some(Ctx::CodeBlock) => {
                self.code_block = false;
                self.blank();
            }
            Some(Ctx::List) => {
                self.list_counters.pop();
                if self.list_counters.is_empty() {
                    self.blank();
                }
            }
            Some(Ctx::Item) => self.flush(),
            Some(Ctx::Quote) => {
                self.flush();
                self.quote = self.quote.saturating_sub(1);
            }
            Some(Ctx::Emphasis) => self.italic = self.italic.saturating_sub(1),
            Some(Ctx::Strong) => self.bold = self.bold.saturating_sub(1),
            Some(Ctx::Strike) => self.strike = self.strike.saturating_sub(1),
            Some(Ctx::Link) => self.link = self.link.saturating_sub(1),
            Some(Ctx::Table) => {
                if let Some(t) = self.table.take() {
                    self.render_table(t);
                }
                self.blank();
            }
            Some(Ctx::TableHead) => {
                if let Some(t) = &mut self.table {
                    let row = std::mem::take(&mut t.current_row);
                    t.rows.push(row);
                    t.header_rows = t.rows.len();
                }
            }
            Some(Ctx::TableRow) => {
                if let Some(t) = &mut self.table {
                    let row = std::mem::take(&mut t.current_row);
                    t.rows.push(row);
                }
            }
            Some(Ctx::TableCell) => {
                if let Some(t) = &mut self.table {
                    t.in_cell = false;
                }
            }
            Some(Ctx::Suppressed) => self.suppress = self.suppress.saturating_sub(1),
            Some(Ctx::Other) | None => {}
        }
    }

    fn render_table(&mut self, t: TableState) {
        const CELL_CAP: usize = 28;
        let cols = t.rows.iter().map(|r| r.len()).max().unwrap_or(0);
        if cols == 0 {
            return;
        }
        let mut widths = vec![0usize; cols];
        for row in &t.rows {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(cell.chars().count().min(CELL_CAP));
            }
        }
        for (ri, row) in t.rows.iter().enumerate() {
            let mut text = String::new();
            for (i, w) in widths.iter().enumerate() {
                let cell = row.get(i).map(String::as_str).unwrap_or("");
                let mut c: String = cell.chars().take(CELL_CAP).collect();
                if cell.chars().count() > CELL_CAP {
                    c.pop();
                    c.push('…');
                }
                let pad = w.saturating_sub(c.chars().count());
                text.push_str(&c);
                text.push_str(&" ".repeat(pad));
                if i + 1 < cols {
                    text.push_str("  ");
                }
            }
            let style = if ri < t.header_rows {
                self.fg.add_modifier(Modifier::BOLD)
            } else {
                self.fg
            };
            self.lines.push(Line::from(Span::styled(text, style)));
            if ri + 1 == t.header_rows {
                let total: usize = widths.iter().sum::<usize>() + 2 * (cols - 1);
                self.lines
                    .push(Line::from(Span::styled("─".repeat(total), self.muted)));
            }
        }
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        self.flush();
        if self.lines.len() > MAX_LINES {
            self.lines.truncate(MAX_LINES);
            self.lines
                .push(Line::from(Span::styled("… (truncated)".to_string(), self.muted)));
        }
        // Trim leading/trailing blank lines.
        while matches!(self.lines.first(), Some(l) if l.spans.is_empty()) {
            self.lines.remove(0);
        }
        while matches!(self.lines.last(), Some(l) if l.spans.is_empty()) {
            self.lines.pop();
        }
        self.lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use localcode_core::theme::ThemeMode;

    fn text_of(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect()
    }

    #[test]
    fn frontmatter_is_stripped() {
        let md = "---\nlicense: mit\ntags:\n- code\n---\n# Title\nBody";
        let lines = render(md, &Theme::new(ThemeMode::Dark));
        let text = text_of(&lines).join("\n");
        assert!(!text.contains("license"), "{text}");
        assert!(text.contains("Title"));
        assert!(text.contains("Body"));
    }

    #[test]
    fn headings_lists_code_render() {
        let md = "# H1\n\nPara *it* **bold** `inline`.\n\n- a\n- b\n  1. one\n\n```rust\nlet x = 1;\n```\n";
        let lines = render(md, &Theme::new(ThemeMode::Dark));
        let text = text_of(&lines);
        assert!(text.iter().any(|l| l.contains("◆ H1")));
        assert!(text.iter().any(|l| l.contains("• a")));
        assert!(text.iter().any(|l| l.contains("1. one")));
        assert!(text.iter().any(|l| l.contains("│ let x = 1;")));
        assert!(text.iter().any(|l| l.contains("inline")));
    }

    #[test]
    fn tables_align_and_html_is_dropped() {
        let md = "<div align=center>\n<b>ignored</b>\n</div>\n\n| q | size |\n|---|------|\n| Q4 | 4.1G |\n";
        let lines = render(md, &Theme::new(ThemeMode::Dark));
        let text = text_of(&lines);
        assert!(text.iter().any(|l| l.starts_with("q ")), "{text:?}");
        assert!(text.iter().any(|l| l.contains("Q4")));
        assert!(!text.iter().any(|l| l.contains("<div")));
    }

    #[test]
    fn unbalanced_input_is_safe() {
        // Pathological inputs must not panic.
        for md in ["", "``` unclosed", "> quote\n> more", "|a|\n|-|"] {
            let _ = render(md, &Theme::new(ThemeMode::Dark));
        }
    }
}
