# LocalCode

**Local-first Rust TUI coding agent** for developers who prefer local LLMs — with Hugging Face model discovery, GPU VRAM fit warnings, one-click deploy (Ollama / llama.cpp / vLLM / SGLang), **one-click remote GPU servers over SSH**, download **mirror fallbacks** for air-gapped networks, benchmarks, optional cloud (RunPod / Vast.ai / Akash), USDC top-up on **Base**, and an in-app assistant that helps fix LocalCode itself.

The interface is a single **omnibar**: type to chat with the agent, or press `/` for commands. Everything else (models, backends, remote servers, settings) opens as a popup above the bar — inspired by OpenCode and Pi.

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
2. Type `/models qwen coder` (or just `/models`) — the omnibar becomes the
   HuggingFace search box and results open in a popup.
3. Click a model (or `↑`/`↓` then `Enter`) to open its card, click a **Quant**
   to cycle, then click **Deploy** (or type `/deploy`).
   HF GGUF repos deploy through Ollama as `hf.co/{org}/{repo}:{quant}` automatically.
4. If VRAM fit warns, choose **Continue** — deploys are never hard-blocked on size.
5. Press `Esc` to close the popup and just **type a prompt** in the bar. Replies
   **stream token-by-token** into the transcript, with live tool activity
   (`⚙ fs.read …`) as the agent works. `Esc` cancels a running turn and keeps the
   partial output. Streaming can be disabled with `agent.stream = false` if your
   runtime rejects `stream: true` together with tools.

Coding is **local-first**: with no runtime deployed it will not silently fall back
to a cloud provider. Set `agent.allow_cloud_fallback = true` in config.toml to
allow using the configured assistant provider instead.

### Remote GPU over SSH

Code on your laptop, run the model on a GPU box — even one on an isolated LAN
that can't reach GitHub or HuggingFace directly.

1. Type `/remote` to open the servers panel.
2. Click **+ New server**, click each field to fill in **host, username,
   password** (like the AmneziaVPN one-click setup), then click **Connect** —
   or do it in one line:

   ```
   /remote add gpu-box 10.8.0.2 ubuntu mypassword
   ```

   then click **Connect** (or `/connect`).
3. LocalCode connects over SSH, detects the remote GPU (`nvidia-smi`), installs
   and starts **Ollama** if needed, and opens a tunnel
   (`localhost → remote:11434`). The agent then uses the remote GPU as if it
   were local — deploy, coding, and benchmarks all flow over the tunnel.

Auth uses a stored password (kept in `config.toml`; ⚠ plaintext — prefer
`key_path` for key-based auth). Set `auto_connect = true` to link on startup.

**Mirrors for air-gapped networks.** When the deploy target can't reach the
public internet, add fallback mirrors — tried in order after the primary:

```toml
[registry]
endpoint = "https://hf-mirror.com"          # primary HuggingFace host
mirrors  = ["https://my-internal-hf.lan"]    # fallbacks, then huggingface.co

[updates]
mirrors  = ["https://git.internal.lan/LocalCode.git"]  # self-update fallback

# per-server, in the [[remote.servers]] table:
#   [remote.servers.mirrors]
#   ollama_install = ["https://internal.lan/ollama-install.sh"]
#   hf_endpoint    = "https://hf-mirror.com"   # remote Ollama pulls via this
```

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

No tabs. A thin top line shows live chips (runtime health, GPU, API) and an
update badge; the middle is the conversation transcript; the bottom is the
**omnibar**, which is active in every mode.

- **Type** a message + `Enter` → chat with the agent.
- Type **`/`** → the command menu opens above the bar (`↑`/`↓` pick, `Enter`
  run, `Esc` close). `Ctrl+K` also opens it.
- **`Esc`** → close a panel, or cancel the running task at home.

| Key | Action |
|-----|--------|
| `Enter` | Send the prompt (or run the highlighted command) |
| `/` or `Ctrl+K` | Open the command menu |
| `↑`/`↓` | Command menu / panel navigation (input history at home) |
| `Ctrl+↑`/`Ctrl+↓` | Grow/shrink the omnibar |
| `Shift+Tab` | Cycle agent approvals (always approve → auto → approve edits → ask permission) |
| `Ctrl+S` | Save config &nbsp;·&nbsp; `Ctrl+C` quit |
| `Esc` | Close panel / cancel task |

| Command | Opens |
|---------|-------|
| `/models [query]` | Search & deploy HuggingFace models |
| `/remote` | Connect a GPU server over SSH |
| `/backends` | Install & configure inference backends |
| `/runtimes` | Active runtimes & system overview |
| `/deploy` | Deploy the selected model |
| `/mode [always\|auto\|edits\|ask]` | How much the agent asks before running tools |
| `/bench` `/setup` `/doctor` `/settings` `/theme` `/new` `/assistant` `/update` `/logs` `/quit` | … |

**Agent approvals** (`/mode`, `Shift+Tab`, or click `approvals` in the status
bar): `always approve` runs everything unprompted; `auto` asks only for
destructive shell commands; `approve edits` asks before file edits and any
shell command; `ask permission` asks before every tool call. Headless runs
(`localcode agent run`) have no prompt, so gated calls are refused there —
pass `--approvals always` for unattended runs.

**Themes**: `dark` (grayscale), `neon` (light blue), `pink` (hot pink), and
`sage` (soft green on gray) — cycle with `/theme` or click a name in the
status bar.

Panels open as **popups** above the bar. **Model cards** render as formatted
markdown with metadata chips and scroll independently; the focused pane has the
accent border. **Pane sizes persist** across restarts (drag the pane borders).

Mouse-first: click list rows, buttons (Deploy, Connect, Backend/Quant), and
fields to edit; scroll wheel in transcript/card/panels (honors `ui.mouse`;
disable it to keep native text selection). Long operations run in the
background — the status line shows a spinner and elapsed time, and `Esc`
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
  localcode-remote/     SSH remote GPU servers (russh: connect, provision, tunnel)
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
