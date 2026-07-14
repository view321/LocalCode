# LocalCode TUI — "2a" redesign spec (for Claude Code)

Implement a new TUI shell for the `localcode` binary. This is a **presentation-only**
rewrite of `crates/localcode-tui`: the agent, HF, backends, remote, bench and cloud
logic stay as-is. Only how the app is drawn and how the omnibar drives navigation change.

Visual source of truth: the interactive prototypes in this project — `2a Minimal.dc.html`
(the shell: status bar, omnibar, command list, chat) and `2a Screens.dc.html` (the finalized
Models / Remote / Bench / Setup screens, §7.3/7.5/7.7/7.8). All numbers/labels there are
illustrative — wire the real values from existing `App` state. Note: the `screens` tab strip
in `2a Screens.dc.html` is a review aid only; the shipped app has no such nav (navigation is
command-driven via the omnibar, §6).

---

## 1. Goal & principles

Reduce the whole interface to **two chrome zones** — a quiet **status bar** on top and a
**modal omnibar** on the bottom — with **one working area** between them. Hard rules:

1. **Only status bar + omnibar are chrome.** Everything else happens in the working area.
2. **No popups / no overlays.** Delete the panel-popup and centered-modal layers. Every
   former panel (models, runtimes, remote, backends, bench, setup, settings) and the
   command menu render **inline in the working area**. Confirms/errors become inline
   banners in the working area, not centered boxes.
3. **Modal omnibar.** One input, always at the bottom. Typing chats with the agent.
   Typing `/` turns the working area into a **command list**; running a command switches
   the working area's **mode**. In `models` mode the typed text is the search query.
4. **Two grayscale themes only:** `dark` (dark + gray) and `light` (white + gray).
   **No accent color.** Emphasis is carried by brightness/weight, not hue.
5. **Square corners.** No rounded borders. Zones are separated by single thin rules only.
6. **No emoji.** Tool activity, status and actions are words + plain ASCII/box glyphs.
7. **Exactly one animated glyph** — the braille spinner — shown only while the agent is
   working. Nothing else animates.

Keep terminal-native (ratatui): everything below maps to cells, `Block`, `Layout`, `Line`/`Span`.

---

## 2. Layout

Vertical `Layout` on `f.area()`, top to bottom:

| Zone         | Height        | Notes |
|--------------|---------------|-------|
| Status bar   | `Length(1)`   | one line, `Borders::BOTTOM`, `BorderType::Plain` |
| Working area | `Min(1)`      | no border; content depends on `mode`; scrollable |
| Omnibar      | `Length(2)`   | one input line + `Borders::TOP`, `BorderType::Plain` |

Drop the old 4-line split (top chips / body / status-spinner / multi-row omnibar), the
`composer_rows` growth, and the separate status/spinner line — fold the spinner into the
status bar (see §5) and the omnibar (see §6). Keep the "terminal too small" guard.

---

## 3. Color tokens

Rewrite `crates/localcode-core/src/theme.rs :: Theme::token_rgb`. Reduce `ThemeMode` to
`Dark` and `Light` (you may keep `HighContrast` untouched but it is out of scope). All
tokens are **grayscale**; `Accent`/`NavActive` collapse to a single "emphasis" gray.

**Dark (dark + gray)**

| Token            | RGB            | hex     | Use |
|------------------|----------------|---------|-----|
| `Bg`             | `(13,13,15)`   | #0d0d0f | background |
| `Fg`             | `(215,215,218)`| #d7d7da | primary text / values |
| `Muted`          | `(108,108,114)`| #6c6c72 | labels, secondary text |
| `Border`         | `(34,34,38)`   | #222226 | rules, list separators |
| `Accent`/`NavActive` | `(243,243,245)` | #f3f3f5 | emphasis (selected row, active theme, user prompt) |
| *faint*          | `(60,60,66)`   | #3c3c42 | bar tracks, disabled, `·` separators |
| *work*           | `(207,207,212)`| #cfcfd4 | the animated braille glyph |

**Light (white + gray)**

| Token            | RGB            | hex     |
|------------------|----------------|---------|
| `Bg`             | `(244,244,243)`| #f4f4f3 |
| `Fg`             | `(43,43,45)`   | #2b2b2d |
| `Muted`          | `(134,134,138)`| #86868a |
| `Border`         | `(224,224,221)`| #e0e0dd |
| `Accent`/`NavActive` | `(15,15,17)` | #0f0f11 |
| *faint*          | `(188,188,188)`| #bcbcbc |
| *work*           | `(58,58,62)`   | #3a3a3e |

`faint` and `work` aren't in the current `ThemeToken` enum — add two variants
(`Faint`, `Work`) or reuse `Border` for faint and `Fg` for work. Prefer adding them.

**Semantic tokens** (`Ok`/`Warn`/`Error`): desaturate to grayscale. Do **not** render red/
green/yellow. Convey state with words ("healthy", "stopped", "not installed", "failed")
and brightness (`Accent` for good/active, `Muted` for idle, `Fg` bold for errors).
`crates/localcode-tui/src/theme.rs` helpers keep their names but return grayscale styles.

Borders: replace every `BorderType::Rounded` with `BorderType::Plain` across
`ui.rs`/`widgets.rs`. Most panes lose their border entirely (working area is borderless).

---

## 4. The one animated glyph

Reuse the existing `SPINNER: [char;10]` braille frames in `ui.rs`
(`⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`). It is the **only** animated element.

- Advance one frame per ~90ms (drive from the existing redraw tick; index by
  `elapsed.as_millis() / 90 % 10`).
- Show it **only while the agent/task is busy** (`app.busy`/`deploy_busy`/etc). When idle,
  render a single space in its slot (do not animate anything).
- Appears in two places: the **status-bar activity slot** (§5) and inline at the end of the
  **streaming transcript line** (chat mode). Nowhere else.

---

## 5. Status bar (one line)

Left → right, all `Muted` labels / `Fg` values, `faint` `·` separators, no color:

```
{spinner} qwen2.5-coder:7b · vram 14.2/24G {bar} · tok/s 48 · ctx 12k/32k {bar}        v0.4.1 · dark / light
```

- `{spinner}`: braille frame while busy, else space (§4).
- Model name: active runtime (`app.active_runtime()`), or "no runtime" in `Muted`.
- `vram`: from `app.gpu` (used/total GiB); `tok/s`: live decode rate if available else omit;
  `ctx`: current/creation context window.
- `{bar}`: a **6-cell** inline meter, `Fg` filled cells + `faint` empty cells, e.g. fill =
  `█` and track = `─` (round `used/total*6`). No gauge widget, no color.
- Right side: version text (no `⬆` glyph — just `v0.4.1`, or `update v0.4.2` in `Fg` when an
  update is available, replacing the old badge). Then a **theme toggle**: `dark / light`
  with the active one in `Accent`, the other in `Muted`; clicking either sets the theme.

If width is tight, drop tok/s then ctx-bar then vram-bar (keep model + theme toggle).

---

## 6. Omnibar + command system (one line)

Bottom zone: `Borders::TOP`, one input row. Format:

```
[{mode}] ❯ {input or placeholder}                                   {hint}
```

- `❯` prompt in `Muted` (static; it is not the animated glyph).
- `[{mode}]` tag: shown only when `mode != Chat`, a square 1-cell-padded label in `Muted`.
- Placeholder per mode (chat: "message the agent…    / for commands"; models:
  "search models — type to filter, Enter to run"; when commanding: "run a command — ↵ to
  execute, Esc to cancel").
- `{hint}` right-aligned `faint`: "/ commands", or "search" in models, or "↵ run" while commanding.

**Behavior (replaces the slash-menu popup + panel navigation):**

- Replace `App.panel: Option<Panel>` with `App.mode: Mode` where
  `Mode = Chat | Commands | Models | Runtimes | Remote | Backends | Bench | Setup | Settings`.
- On input change: if the trimmed input starts with `/`, enter **command view** (working
  area shows the command list, §7.2) filtered by the typed prefix. Otherwise the working
  area shows the current `mode`'s view.
- On `Enter`:
  - input starts with `/`: resolve the command word by exact match, else first command whose
    name starts with the typed prefix (`/mod`→`/models`); run it (switch mode / toggle theme /
    reset chat); capture any argument (`/models qwen coder` seeds the query). Clear input.
  - `mode == Models` and non-empty: set the search query, run HF search, clear input.
  - `mode == Chat`: submit the prompt to the agent (existing streaming path).
- On `Esc`: cancel the running task if busy (keep partial output, as today), else return to
  `Chat` and clear input.
- Drop `Ctrl+↑/↓` omnibar resizing (single line now). Keep input history on `↑/↓` in chat.

**Command table** (name · target · description):

| Command | Target mode / action | Description |
|---|---|---|
| `/models [q]` | Models | search & deploy HuggingFace models |
| `/runtimes` | Runtimes | active runtimes & system overview |
| `/remote` | Remote | connect a GPU server over SSH |
| `/backends` | Backends | install & configure inference backends |
| `/bench` | Bench | run the sample benchmark suite |
| `/setup` | Setup | first-run setup & doctor |
| `/settings` | Settings | preferences & config file |
| `/theme` | toggle Dark/Light | (does not change mode) |
| `/chat`, `/new` | Chat | back to / reset the conversation |
| `/deploy` | action in Models | deploy the selected model |
| `/quit` | quit | exit |

---

## 7. Working-area views

Borderless, scrollable, `padding: 1` column. Each view is a function drawing `Line`s into
the working rect. Selection marker is `›` in `Accent`; rows separated by a `Border` rule.
Finalized layouts for Models / Remote / Bench / Setup match the prototype
`2a Screens.dc.html` — reproduce those. **Models and Remote split the working area into two
columns divided by a single vertical `Border` rule** (still one screen, still no popup):
split at ~42% (models) / ~30% (remote); each column scrolls independently.

**Bar convention** (reused from §5): a fixed-width run of cells, filled = `█` in `Fg`, track
= `─`/space in `faint`. No `Gauge`, no color. Cell counts given per bar below.

### 7.1 Chat (default)
Transcript from `app.coding_transcript`. Roles: user line prefixed `❯ ` in `Accent`;
agent text in `Fg`; tool calls as compact one-liners in `Muted` with a word verb and a
`faint` result, e.g. `read   crates/auth/src/lib.rs   142 lines` /
`write  …  +38 −21` / `shell  cargo check -p auth   ok · 3.1s`. No emoji, no card borders.
While busy, the last streaming line ends with the animated braille glyph.

### 7.2 Commands (transient, when input starts with `/`)
Filtered command list (§6 table): each row `{mark} {name}   {desc}` (`name` in `Accent`
when it's the first/selected match, else `Fg`; `desc` in `Muted`). Enter runs the top match;
rows are clickable.

### 7.3 Models  (two-pane)

```
results  "qwen coder" · 5        │ Qwen/Qwen2.5-Coder-7B-Instruct
› Qwen/Qwen2.5-Coder-7B-Instruct │ 1.2M downloads · 3.4k likes · apache-2.0 · 7.6B · text-generation
  1.2M dl · 3.4k likes · gguf    │
  Qwen/Qwen2.5-Coder-7B-GGUF     │ <two-line model-card excerpt, Fg>
  890k dl · 2.1k likes · gguf    │
  deepseek-coder-6.7b-instruct   │ quantizations   click to select
  640k dl · 1.8k likes           │ › Q5_K_M   5.13 GiB  ██████──── fits
  …                              │   Q6_K     6.25 GiB  ███████─── fits
                                 │ backend ollama ⟳   context 32768
                                 │ vram fit  est 5.9 · free 21.1 / 24.0 GiB
                                 │ ██████──────────────────
                                 │ [ deploy ]
```

- **Left column** — header `results  "{query}" · {n}` (`Muted`/`faint`). Rows from
  `app.models`: two lines each — `{mark} {id}` (selected `id` in `Accent`) then
  `{downloads} dl · {likes} likes[ · gguf]` in `Muted`. **No `♥`/emoji — spell out
  "dl"/"likes".** Row separated by a `Border` rule; click or `↑/↓` selects.
- **Right column** (detail of selected, from `app.model_detail`) —
  1. `{id}` in `Accent`.
  2. metrics line `Muted`: `{downloads} downloads · {likes} likes · {license} · {params} · {pipeline_tag}`.
  3. model-card excerpt (first ~2–3 lines, `Fg`, via `markdown.rs`).
  4. `quantizations   click to select` header, then one row per quant
     (`app.model_detail.quants`): `{mark} {label}  {size}  {10-cell fit bar}  {fits|tight|over}`.
     Bar fill = estimated size / total VRAM; selecting sets `app.selected_quant`.
  5. action line: `backend {app.deploy_backend} ⟳` (click cycles) · `context {ctx}`.
  6. `vram fit  est {est} · free {free} / {total} GiB` + a ~24-cell bar (`app.last_fit`).
  7. square `[ deploy ]` (1-cell-padded, `Accent` border) — existing deploy path; progress
     shows on the status-bar spinner + this bar. Never hard-block on VRAM (warn, per README).

### 7.4 Runtimes
List from `app.all_runtimes()`: `{name}   {status-word}   {base_url}` (active in `Accent`,
stopped in `Muted`). Below, an overview block: `gpu …`, `api …`, `remote …` from `app.gpu`,
`app.api_healthy`, `app.remote_sessions`. No right-hand pane, no border.

### 7.5 Remote  (two-pane)

```
servers                │ gpu-box — click a field to edit
[gpu-box            ]  │ host      10.8.0.2▌
 connected · 10.8.0.2  │ username  ubuntu
 workstation           │ password  ••••••••••
 offline · 192.168…    │ port      22
                       │ [ connect ] [ save ] [ disconnect ] [ delete ]
                       │ connecting…
 + new server          │ ok  reach host 10.8.0.2:22
                       │ ok  detect GPU (nvidia-smi) — RTX 3090
                       │ ⠹   install & start ollama
                       │ ·   open tunnel localhost:11434
                       │ ⚠ passwords are stored in plaintext … prefer key_path.
```

- **Left column** — `servers` header, one entry per `app.config.remote.servers`: name
  (selected in `Accent`, with a `Border`/`faint` selection block) + `{status} · {host}`
  (`Muted`); live sessions from `app.remote_sessions` (connected/connecting/offline as
  **words**, no colored dots). Bottom: `+ new server` in `Accent`.
- **Right column** — editable field set for the selected server: `host / username /
  password (masked) / port`, each a `{label}  {value}` row; the focused field shows a `▌`
  caret in `Accent` and is edited in place (click a field to focus). Then action words
  `[ connect ] [ save ] [ disconnect ] [ delete ]` (reuse `ClickTarget::Remote*`).
- **Connect progress** (only while connecting) — a checklist bound to the real
  connect/provision/tunnel steps: `reach host` → `detect GPU (nvidia-smi)` → `install &
  start ollama` → `open tunnel localhost:11434`. Done steps marked `ok` (`Fg`), the running
  step marked with the **braille spinner** (`Work`), pending steps `·` (`faint`).
- Keep the plaintext-password warning as an inline `Muted`/`faint` note. No popup.

### 7.6 Backends
Rows from `app.backend_reports`: `{kind}   {ready? "ready" : "not installed"}   {note}`; a
square `[ install ]` word for not-installed backends (runs the existing installer confirm —
rendered inline, see §8). Fold in the old backends-manager overlay here.

### 7.7 Bench  (single column)

```
suite  localcode-sample-coding v1.0.0 · 24 tasks · target qwen2.5-coder:7b   [ run ]
last run · 2h ago
┌ SCORE ┬ PASS ┬ P50   ┬ P95   ┬ TOK/S ┐
│ 0.82  │ 90%  │ 240ms │ 610ms │ 47    │
│ ████─ │ ████─│       │       │       │
└───────┴──────┴───────┴───────┴───────┘
tasks
fizzbuzz.rs                              pass    180 ms
async-refactor                           fail    —
recent runs
2h ago  0.82  ████████████████──
1d ago  0.79  ███████████████───
```

- Top line: suite name + version + task count + target runtime (`Muted`/`faint`), with a
  square `[ run ]` (existing bench path; runs on the active runtime).
- **Stat grid** from `app.last_bench_result.metrics`: 5 cells (`score`, `pass`, `p50`, `p95`,
  `tok/s`) in a single `Border`-ruled row — uppercase `Muted` label, big `Accent` value,
  and a ~10-cell bar under the ratio metrics (score, pass). Cells divided by `Border` rules.
- **Tasks** table: `{name}   {pass|fail}   {latency}` per task (fail in `Accent` bold, not
  red; pass in `Fg`; latency `Muted`).
- **Recent runs**: last N runs as `{when}  {score}  {~16-cell bar}`. All grayscale.

### 7.8 Setup  (single column)

```
get started  ██████────── 3 of 6
[x] GPU detected                              recheck
    RTX 4090 · 24 GiB
[x] Backend installed                         manage
    ollama ready · default
[ ] Connect a remote GPU (optional)           add
    run models on a GPU box over SSH
…
doctor                                        [ run doctor ]
nvidia-smi  ok — 1 gpu, driver 550.90
ollama      ok — serving on :11434
hf          reachable — endpoint huggingface.co
disk        84% used — 61 GiB free
config ~/.config/localcode/config.toml — Ctrl+S to save
```

- Header `get started` + a ~12-cell progress bar + `{done} of {total}` (`faint`).
- Checklist, one row per step (GPU / backend / model / remote / assistant / updates):
  `[x]` done (marker + title in `Muted`/dimmed) or `[ ]` todo (marker `faint`, title `Fg`);
  a `Muted` sub-line; and a per-step action word on the right (`Accent` border for todo,
  `faint` for done). **Use `[x]`/`[ ]` — no check/emoji glyphs.** Bind to the real detectors
  (`app.gpu`, `app.backend_reports`, `app.all_runtimes()`, `app.remote_sessions`,
  `app.assistant_configured`, `app.config.updates`).
- **Doctor** block: `[ run doctor ]` + diagnostic lines `{probe}  {status-word} — {detail}`
  from `app.doctor_summary` (status as words, `Fg`). Config path + `Ctrl+S` hint in `faint`.

### 7.9 Settings
Plain `label  value` lines (no border), values from `app.config`: theme, token streaming,
confirm-destructive-shell, cloud-fallback, default backend, registry endpoint + mirrors,
remote-server count, config-file path. Env-override hints in `faint`.

---

## 8. Confirms, warnings, errors (were modals)

Delete `draw_modal` centered popups and `centered_rect` usage for panels. Render these as an
**inline banner** at the top of the working area (or a 1–2 line strip just above the omnibar):
title + body in `Fg`/`Muted`, actions as inline words `[ confirm ] [ cancel ]` (or
`[ retry ] [ open logs ] [ ask ] [ dismiss ]` for errors). Keep the `ConfirmAction` semantics
and click-target wiring; only the presentation moves inline. Errors use `Fg` bold, not red.

---

## 9. Mouse & keyboard

- **Mouse:** keep and extend the `app.click_regions` / `ClickTarget` system. Everything is
  clickable: status-bar theme toggle, command rows, model rows + deploy, runtime rows, remote
  fields/actions, backend install, bench run, inline banner buttons. Remove panel-close and
  pane-resize-border targets (no panes).
- **Keyboard:** `Enter` submit/run; `Esc` cancel-or-chat; `↑/↓` select in list views and input
  history in chat; typing edits the omnibar; `Ctrl+C` quit; `Ctrl+S` save config. Drop
  `Ctrl+K` (no separate menu — `/` is the entry point) or alias it to prefill `/`.

---

## 10. Files to change

- `crates/localcode-core/src/theme.rs` — grayscale `token_rgb`; add `Faint`, `Work` tokens;
  reduce modes to Dark/Light for the toggle.
- `crates/localcode-tui/src/theme.rs` — helpers return grayscale styles; add `faint()`,
  `work()`.
- `crates/localcode-tui/src/app.rs` — replace `panel: Option<Panel>` with `mode: Mode`; add
  omnibar command parsing + prefix match; theme toggle; convert modal state to inline banner
  state; keep agent/deploy/remote/bench logic intact.
- `crates/localcode-tui/src/ui.rs` — rewrite `draw`: 3-zone layout, status bar (§5), omnibar
  (§6), per-mode working-area renderers (§7); delete `draw_panel`, `draw_slash_menu` popup,
  `draw_backend_manager` overlay, `draw_assistant_dock` overlay (fold assistant into an inline
  view or banner); `BorderType::Plain` throughout.
- `crates/localcode-tui/src/widgets.rs` — `draw_modal` → `draw_inline_banner`; drop
  `centered_rect` for panels.
- `markdown.rs` — unchanged (still used to render model-card text inline).

## 11. Preserve / out of scope

Do not touch: agent streaming + cancel, HF search/quants/deploy, backends install, remote SSH
connect/tunnel, bench runner, updates/self-update, config load/save, payments. The prototype's
metric values, model list, and button results are mock — bind the real `App` state.

## 12. Acceptance checklist

- [ ] Only a 1-line status bar and a 2-line omnibar are chrome; no popups/overlays anywhere.
- [ ] `/` shows an inline command list; Enter runs the prefix match; each command switches the
      working area's mode; `/models qwen` seeds a search.
- [ ] Models/runtimes/remote/backends/bench/setup/settings all render inline, mouse-clickable.
- [ ] `dark` and `light` are grayscale; no red/green/yellow; toggling from the status bar works.
- [ ] Square borders only; single thin rules under the status bar and above the omnibar.
- [ ] No emoji in output; state shown with words.
- [ ] The braille spinner is the only animated glyph and animates only while busy.
- [ ] Agent chat, model deploy, and remote connect still function unchanged.
