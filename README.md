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
3. Select a model, `Enter` for detail, pick a quant, press `d` to **Deploy**.
4. If VRAM fit warns, choose **Continue** — deploys are never hard-blocked on size.
5. Open **Coding** (`4`), type a prompt.

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

Right rail tabs (gray idle → white hover/focus; active bold+underline):

1. dashboard · 2 models · 3 benchmarks · 4 coding · 5 setup · 6 notifications · 7 settings

| Shortcut | Action |
|----------|--------|
| `1`–`7` | Switch tab |
| `Ctrl+K` | Command palette |
| `/` | Models search |
| `d` | One-click deploy |
| `a` | Ask assistant (uses last error context) |
| `s` | Toggle subagents (Coding) |
| `Ctrl+S` | Save config |
| `q` | Quit |

Mouse: hover/click rail, resize supported. Pane ratios for Models adjust with `[` `]`.

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
