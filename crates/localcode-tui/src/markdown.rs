//! Minimal markdown → styled `Line`s for terminal rendering (model cards
//! and chat replies).
//!
//! Renders the subset that matters for Hugging Face READMEs and agent
//! answers: headings, paragraphs, lists, code blocks, inline styles, quotes,
//! tables, rules. [`render`] (cards) drops raw HTML and YAML frontmatter;
//! [`render_chat`] keeps HTML-ish tokens as literal text. Wrapping is left
//! to the caller, so lines here are logical lines.

use localcode_core::theme::{Theme, ThemeToken};
use pulldown_cmark::{Event, Options, Parser, Tag};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::theme;

/// Hard cap so a pathological README can't grow the frame unbounded.
const MAX_LINES: usize = 1500;

pub fn render(md: &str, th: &Theme) -> Vec<Line<'static>> {
    render_impl(strip_frontmatter(md), th, false)
}

/// Chat variant: a leading `---` is a rule, not YAML frontmatter, and raw
/// HTML-ish tokens are kept as literal text — local models speak in
/// `<think>`/`<answer>`-style tags, and dropping those would silently eat
/// parts of the reply.
pub fn render_chat(md: &str, th: &Theme) -> Vec<Line<'static>> {
    render_impl(md, th, true)
}

fn render_impl(src: &str, th: &Theme, keep_html: bool) -> Vec<Line<'static>> {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TASKLISTS);

    let mut r = Renderer::new(th, keep_html);
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

/// Expand tab characters to spaces on 4-column stops. ratatui filters control
/// chars (a tab renders as nothing), so tab-indented code (Go, Makefiles) would
/// otherwise lose all indentation. Columns are counted per char, which is
/// correct for the ASCII indentation tabs are used for.
fn expand_tabs(s: &str) -> String {
    if !s.contains('\t') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 4);
    let mut col = 0usize;
    for c in s.chars() {
        if c == '\t' {
            let n = 4 - (col % 4);
            for _ in 0..n {
                out.push(' ');
            }
            col += n;
        } else {
            out.push(c);
            col += 1;
        }
    }
    out
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
    /// Bytes of the current code block not yet terminated by a newline. CRLF
    /// input splits a line's text and its newline across separate `Text`
    /// events, so lines are only emitted once their terminator arrives.
    code_buf: String,
    keep_html: bool,
    list_counters: Vec<Option<u64>>,
    /// A list item's marker (`• ` / `1. `), deferred so a loose item — whose
    /// text is wrapped in a paragraph — attaches the marker to its first
    /// content line instead of flushing it alone.
    pending_marker: Option<Span<'static>>,
    table: Option<TableState>,
}

impl Renderer {
    fn new(th: &Theme, keep_html: bool) -> Self {
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
            code_buf: String::new(),
            keep_html,
            list_counters: vec![],
            pending_marker: None,
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
            // Keep any pending list marker for the item's first real content
            // line — flushing it alone here is exactly the orphaned-bullet bug.
            return;
        }
        let mut spans = std::mem::take(&mut self.spans);
        if let Some(marker) = self.pending_marker.take() {
            spans.insert(0, marker);
        }
        if self.quote > 0 {
            spans.insert(0, Span::styled("▎ ".to_string(), self.muted));
        }
        self.lines.push(Line::from(spans));
    }

    /// Emit a still-pending list marker on its own line. Used when an item's
    /// only content bypasses the span buffer (a nested code block/table/rule)
    /// or the item is empty, so the marker is neither attached nor leaked onto
    /// a later item.
    fn resolve_pending_marker(&mut self) {
        if let Some(marker) = self.pending_marker.take() {
            self.lines.push(Line::from(marker));
        }
    }

    /// Append one code line with the gutter, expanding tabs (ratatui drops the
    /// control char, which would silently lose all indentation otherwise).
    fn push_code_line(&mut self, line: &str) {
        self.lines.push(Line::from(vec![
            Span::styled("│ ".to_string(), self.accent),
            Span::styled(expand_tabs(line), self.fg),
        ]));
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
            // Buffer, then emit only complete lines; an unterminated tail is
            // held until its newline arrives in a later event. CRLF code text
            // is delivered as e.g. Text("a"), Text("\nb"), Text("\n") — the old
            // per-event split() turned each leading "\n" into a bogus blank
            // gutter line.
            self.code_buf.push_str(text);
            while let Some(nl) = self.code_buf.find('\n') {
                let line: String = self.code_buf.drain(..=nl).collect();
                let line = line.trim_end_matches('\n').trim_end_matches('\r');
                self.push_code_line(line);
            }
            return;
        }
        let style = self.cur_style();
        self.spans.push(Span::styled(expand_tabs(text), style));
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
                self.resolve_pending_marker();
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
            // Raw HTML is noise in model cards (badges, center divs…) but
            // meaningful in chat, where models emit literal `<tag>` text.
            Event::Html(t) | Event::InlineHtml(t) if self.keep_html => {
                let mut first = true;
                for seg in t.split('\n') {
                    if !first {
                        self.flush();
                    }
                    first = false;
                    if !seg.is_empty() {
                        self.push_text(seg);
                    }
                }
            }
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
                // A code block is the item's content but bypasses the span
                // buffer, so settle any pending marker before it.
                self.resolve_pending_marker();
                self.blank();
                self.code_block = true;
                self.code_buf.clear();
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
                // Defer: attach to the item's first content line in `flush`, so
                // a loose item (Item → Paragraph → text) doesn't flush the bare
                // marker on its own line ahead of the text.
                self.pending_marker = Some(Span::styled(marker, self.accent));
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
                // Keep the alt text, marked as an image. Inside a table cell the
                // marker must go into the cell string — routing it to the span
                // buffer would leak a stray "[img]" line after the table (the
                // alt text itself is already routed into the cell by push_text).
                let in_cell = self.table.as_ref().is_some_and(|t| t.in_cell);
                if in_cell {
                    if let Some(t) = &mut self.table {
                        if let Some(cell) = t.current_row.last_mut() {
                            cell.push_str("[img] ");
                        }
                    }
                } else {
                    self.spans
                        .push(Span::styled("[img] ".to_string(), self.muted));
                }
                self.stack.push(Ctx::Other);
            }
            Tag::Table(_) => {
                self.flush();
                self.resolve_pending_marker();
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
            Tag::HtmlBlock if self.keep_html => {
                self.flush();
                self.stack.push(Ctx::Other);
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
                // Emit a final line that had no trailing newline (common for the
                // last line of a fenced block, and for CRLF input).
                if !self.code_buf.is_empty() {
                    let line = std::mem::take(&mut self.code_buf);
                    let line = line.trim_end_matches('\n').trim_end_matches('\r');
                    self.push_code_line(line);
                }
                self.code_block = false;
                self.blank();
            }
            Some(Ctx::List) => {
                self.list_counters.pop();
                if self.list_counters.is_empty() {
                    self.blank();
                }
            }
            Some(Ctx::Item) => {
                self.flush();
                // An empty item never attached its marker — show it alone.
                self.resolve_pending_marker();
            }
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
        // Clip to a char cap, then measure DISPLAY width — a char count would
        // under-measure CJK/emoji cells (2 columns each) and misalign the grid.
        let clip = |cell: &str| -> String {
            let mut c: String = cell.chars().take(CELL_CAP).collect();
            if cell.chars().count() > CELL_CAP {
                c.pop();
                c.push('…');
            }
            c
        };
        let mut widths = vec![0usize; cols];
        for row in &t.rows {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(clip(cell).width());
            }
        }
        for (ri, row) in t.rows.iter().enumerate() {
            let mut text = String::new();
            for (i, w) in widths.iter().enumerate() {
                let cell = row.get(i).map(String::as_str).unwrap_or("");
                let c = clip(cell);
                let pad = w.saturating_sub(c.width());
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
            let _ = render_chat(md, &Theme::new(ThemeMode::Dark));
        }
    }

    #[test]
    fn chat_bolds_and_keeps_html_tags() {
        let md = "Call <tool>fs.read</tool> and it is **done**.";
        let lines = render_chat(md, &Theme::new(ThemeMode::Dark));
        let text = text_of(&lines).join("\n");
        assert!(text.contains("<tool>"), "{text}");
        let bold = lines.iter().flat_map(|l| &l.spans).any(|s| {
            s.content.contains("done") && s.style.add_modifier.contains(Modifier::BOLD)
        });
        assert!(bold, "{text}");
    }

    #[test]
    fn loose_lists_keep_marker_with_text() {
        // Blank-line-separated ("loose") items wrap their text in a paragraph;
        // the marker must stay on the text's line, not flush alone above it.
        let bullets = text_of(&render("- alpha\n\n- beta", &Theme::new(ThemeMode::Dark)));
        assert!(bullets.iter().any(|l| l.contains("• alpha")), "{bullets:?}");
        assert!(bullets.iter().any(|l| l.contains("• beta")), "{bullets:?}");
        assert!(!bullets.iter().any(|l| l.trim() == "•"), "orphan bullet: {bullets:?}");
        assert!(!bullets.iter().any(|l| l.trim() == "alpha"), "text split off: {bullets:?}");

        let numbered = text_of(&render("1. one\n\n2. two", &Theme::new(ThemeMode::Dark)));
        assert!(numbered.iter().any(|l| l.contains("1. one")), "{numbered:?}");
        assert!(numbered.iter().any(|l| l.contains("2. two")), "{numbered:?}");
        assert!(!numbered.iter().any(|l| l.trim() == "1."), "orphan number: {numbered:?}");
    }

    #[test]
    fn crlf_code_block_has_no_blank_gutter_lines() {
        let md = "```\r\nfoo\r\nbar\r\n```\r\n";
        let lines = text_of(&render(md, &Theme::new(ThemeMode::Dark)));
        assert!(lines.iter().any(|l| l == "│ foo"), "{lines:?}");
        assert!(lines.iter().any(|l| l == "│ bar"), "{lines:?}");
        // No spurious empty gutter line between them.
        assert!(!lines.iter().any(|l| l == "│ " || l == "│"), "blank gutter: {lines:?}");
    }

    #[test]
    fn tabs_in_code_expand_to_spaces() {
        let md = "```\n\tindented\n```";
        let lines = text_of(&render(md, &Theme::new(ThemeMode::Dark)));
        // The tab became spaces (ratatui would otherwise render it as nothing).
        assert!(lines.iter().any(|l| l.contains("    indented")), "{lines:?}");
        assert!(!lines.iter().any(|l| l.contains('\t')), "raw tab survived: {lines:?}");
    }

    #[test]
    fn table_columns_align_with_wide_glyphs() {
        // CJK cells are two display columns each; padding by char count (the old
        // bug) would misalign column 2. Aligned rows share a total width.
        let md = "| id | v |\n|----|---|\n| 日本語 | 1 |";
        let lines = text_of(&render(md, &Theme::new(ThemeMode::Dark)));
        let header = lines.iter().find(|l| l.contains("id")).expect("header row");
        let data = lines.iter().find(|l| l.contains("日本語")).expect("data row");
        assert_eq!(header.width(), data.width(), "misaligned: {header:?} vs {data:?}");
    }

    #[test]
    fn image_in_table_cell_does_not_leak() {
        let md = "| a |\n|---|\n| ![alt](u.png) |\n\nafter";
        let lines = text_of(&render(md, &Theme::new(ThemeMode::Dark)));
        // The marker stays with its alt text in the cell ("[img] alt"); the bug
        // leaked a bare "[img]" marker line after the table.
        assert!(!lines.iter().any(|l| l.trim() == "[img]"), "leaked marker: {lines:?}");
        assert!(lines.iter().any(|l| l.contains("[img] alt")), "{lines:?}");
        assert!(lines.iter().any(|l| l.contains("after")), "{lines:?}");
    }

    #[test]
    fn chat_does_not_treat_leading_dashes_as_frontmatter() {
        let md = "---\nplan: step one\n---\nthen go";
        let chat = text_of(&render_chat(md, &Theme::new(ThemeMode::Dark))).join("\n");
        assert!(chat.contains("plan: step one"), "{chat}");
        assert!(chat.contains("then go"));
        // The card renderer, by contrast, strips it as YAML.
        let card = text_of(&render(md, &Theme::new(ThemeMode::Dark))).join("\n");
        assert!(!card.contains("plan: step one"), "{card}");
    }
}
