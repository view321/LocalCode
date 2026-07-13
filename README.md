# LocalCode

**Local-first Rust TUI coding agent** for developers who prefer local LLMs — with Hugging Face model discovery, GPU VRAM fit warnings, one-click deploy (Ollama / llama.cpp / vLLM / SGLang), benchmarks, optional cloud (RunPod / Vast.ai / Akash), USDC top-up on **Base**, and an in-app assistant that helps fix LocalCode itself.

Philosophy: **one-click and no errors** — guided recovery, warnings over hard blocks, always-available Ask assistant.

> Spec: [`docs/HANDOFF.md`](docs/HANDOFF.md) · Error catalog: [`docs/ERROR_CODES.md`](docs/ERROR_CODES.md)

## Requirements

- Rust 1.75+ (edition 2021)
- Optional: [Ollama](https://ollama.com), [llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server`, NVIDIA drivers + `nvidia-smi`
- Optional: `HF_TOKEN`, `OPENROUTER_API_KEY`, cloud API keys

## Install

### One-line installers

**Linux / macOS:**

```bash
curl -fsSL https://raw.githubusercontent.com/view321/LocalCode/main/scripts/install.sh | bash
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/view321/LocalCode/main/scripts/install.ps1 | iex
```

Scripts install Rust (via rustup) if needed, clone this repo, build the release binary, and place `localcode` on your PATH (`~/.local/bin` or `%USERPROFILE%\.local\bin`).

### Updates

LocalCode checks for new versions on startup (background, offline-safe) and
shows a **⬆ update available** badge in the TUI header plus a notification.
Press `u` to install in one step — it fetches the tracked branch, rebuilds in
the background (Esc cancels safely), swaps the binary, and asks you to restart.

```bash
localcode update          # same thing, headless
localcode update --check  # check only, don't install
```

Configure in `config.toml` under `[updates]`: `check_on_startup`, `repo_url`,
`branch`, `install_dir` (defaults to the installer's checkout; the
`LOCALCODE_INSTALL_DIR` env var overrides).

### From source (Cargo)

```bash
git clone https://github.com/view321/LocalCode.git
cd LocalCode
cargo install --path crates/localcode-cli
# or: cargo build --release && ./target/release/localcode
```

### Run

```bash
# TUI (default)
localcode

# Diagnostics
localcode doctor --json
```

Website (project page): open [`website/index.html`](website/index.html) or the GitHub Pages site after deploy.

### First deploy (happy path)

1. Start Ollama (`ollama serve`) if using the default backend.
2. Open **Models** (`2`), press `/` to search or `p` for popular coding models.
3. Select a model, `Enter` for detail, pick a quant with `,` / `.`, press `d` to **Deploy**.
   HF GGUF repos deploy through Ollama as `hf.co/{org}/{repo}:{quant}` automatically.
4. If VRAM fit warns, choose **Continue** — deploys are never hard-blocked on size.
5. Open **Coding** (`4`), press `i` and type a prompt. Replies **stream
   token-by-token** into the transcript, with live tool activity (`⚙ fs.read …`)
   as the agent works. `Esc` cancels a running turn and keeps the partial output.
   Streaming can be disabled with `agent.stream = false` if your runtime rejects
   `stream: true` together with tools.

Coding is **local-first**: with no runtime deployed it will not silently fall back
to a cloud provider. Set `agent.allow_cloud_fallback = true` in config.toml to
allow using the configured assistant provider instead.

### Headless CLI

```bash
localcode models search "qwen coder"
localcode models info Qwen/Qwen2.5-Coder-7B-Instruct
localcode deploy qwen2.5-coder:7b --backend ollama --force
localcode bench run sample --endpoint http://127.0.0.1:11434/v1 --model qwen2.5-coder:7b
localcode agent run "List files in the workspace" --workspace .
```

### Config

| Item | Default |
|------|---------|
| Config | platform config dir `localcode/config.toml`, or `$LOCALCODE_HOME/config/config.toml` |
| Logs | platform log/state dir `localcode/logs/` |
| Env | `LOCALCODE_HOME`, `LOCALCODE_API_URL`, `LOCALCODE_HF_ENDPOINT`, `LOCALCODE_LOG_LEVEL`, `HF_TOKEN` |

Accounts are **optional**. Local browse, deploy, coding, and benchmarks work offline after cache warm-up.

## TUI

Top bar: identity + live chips (runtime health, GPU, API) and an update badge.
Below it, a clickable tab strip:

1 home · 2 models · 3 bench · 4 coding · 5 setup · 6 alerts · 7 settings

| Shortcut | Action |
|----------|--------|
| `1`–`7`, `Tab`/`Shift+Tab` | Switch tab (or click the strip) |
| `?` | Help overlay (per-tab keys) |
| `Ctrl+K` | Command palette |
| `u` | Install the available update (background build) |
| `Esc` | Cancel the running task (search/deploy/agent turn/update) |
| `/` `p` `t` | Models: search / popular / trending |
| `←`/`→` | Models: focus the results list or the model card |
| `j`/`k`, `PgUp/PgDn`, `g`/`G` | Models: move selection / scroll the card |
| `,` `.` / `+` `-` | Models: pick quant / adjust context size |
| `b` / `d` | Models: cycle backend / one-click deploy |
| `[` `]` and `{` `}` | Resize panes (Models: list↔card and card↔deploy; Dashboard: columns) |
| `i`, `Enter` | Coding: focus composer (`↑` history, `PgUp/PgDn` scroll) |
| `Ctrl+↑`/`Ctrl+↓` (or `+`/`-`) | Coding: grow/shrink the composer |
| `n` | Coding: new session |
| `j`/`k`, `x` | Dashboard: select / stop runtime |
| `a` | Ask assistant (uses last error context) |
| `e` | Open last error details (with working Retry) |
| `c` | Notifications: clear |
| `Ctrl+S` | Save config |
| `q` | Quit (confirms if managed runtimes are running) |

**Model cards** render as formatted markdown (headings, lists, tables, code)
with metadata chips — downloads, likes, license, parameter size, gating — and
scroll independently of the results list; the focused pane has the accent
border. **Pane sizes persist** across restarts (saved with the config).

Mouse: click tabs, scroll wheel in transcript/card/setup (honors `ui.mouse`
config; disable it to keep native text selection). Long operations run in the
background — the status bar shows a spinner and elapsed time, and `Esc`
cancels.

## Server (optional VPS)

```bash
# Dev API (in-memory; set LOCALCODE_DEV_AUTO_AUTH=1 for easy tokens)
cargo run -p localcode-server
# listens on 0.0.0.0:8787

# Point client
set LOCALCODE_API_URL=http://127.0.0.1:8787
```

Production schema: `server/localcode-server/migrations/001_init.sql` (PostgreSQL).

### Payments (v1)

- **Asset:** USDC  
- **Chain:** Base (L2) — locked default  
- Top-up creates a deposit intent via API; Akash managed deploy uses ledger **holds** with explicit confirmation and custody disclosure in Setup.

## Workspace layout

```
crates/
  localcode-core/       config, errors, events
  localcode-log/        tracing + redaction
  localcode-gpu/        GPU discover + VRAM fit
  localcode-hf/         HF client, mirrors, quants, cache
  localcode-backends/   Ollama, llama.cpp, vLLM, SGLang
  localcode-cloud/      RunPod, Vast, Akash adapters
  localcode-payments/   USDC balance client
  localcode-bench/      suites + runner + publish
  localcode-agent/      coding agent, skills, MCP, tools
  localcode-assistant/  app-repair assistant
  localcode-api-client/ VPS REST client
  localcode-upgrade/    update check + self-update
  localcode-tui/        ratatui UI
  localcode-cli/        binary
server/localcode-server/
docs/
```

## Development

```bash
cargo test --workspace
cargo build -p localcode-cli
cargo build -p localcode-server
```

## License

MIT
