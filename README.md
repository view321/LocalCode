# LocalCode

**Local-first Rust TUI coding agent** for developers who prefer local LLMs — with Hugging Face model discovery, GPU VRAM fit warnings, one-click deploy (Ollama / llama.cpp / vLLM / SGLang / colibrì for GLM-5.2 & Hy3 expert-streaming), **one-click remote GPU servers over SSH**, download **mirror fallbacks** for air-gapped networks, benchmarks, optional cloud (RunPod / Vast.ai / Akash), USDC top-up on **Base**, and an in-app assistant that helps fix LocalCode itself.

The interface is a single **omnibar**: type to chat with the agent, or press `/` for commands. Everything else (models, backends, remote servers, settings) opens as a popup above the bar — inspired by OpenCode and Pi.

Philosophy: **one-click and no errors** — guided recovery, warnings over hard blocks, always-available Ask assistant.

> Spec: [`docs/HANDOFF.md`](docs/HANDOFF.md) · Error catalog: [`docs/ERROR_CODES.md`](docs/ERROR_CODES.md)

## Requirements

- Rust 1.75+ (edition 2021)
- Optional: [Ollama](https://ollama.com), NVIDIA drivers + `nvidia-smi`
- **llama-server** is installed automatically by the app installer / `localcode setup` (or on first TUI launch if missing)
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

Scripts install Rust (via rustup) if needed, clone this repo, build the release binary, place `localcode` on your PATH (`~/.local/bin` or `%USERPROFILE%\.local\bin`), and run `localcode setup` to install a managed **PrismML llama-server** (source build when git+cmake are available, else Prism prebuilt) into LocalCode’s data dir, writing the absolute path to `backends.llamacpp.bin` in config. Skip with `LOCALCODE_SKIP_LLAMA=1`.

You can re-run setup anytime:

```bash
localcode setup              # ensure llama-server + update config
localcode setup --skip-llama # no-op placeholder for future steps
```

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
   (`⚙ read …`) as the agent works. `Esc` cancels a running turn and keeps the
   partial output. Streaming can be disabled with `agent.stream = false` if your
   runtime rejects `stream: true` together with tools.

Sessions persist by default (latest resumes on restart; `/new` starts fresh).
Long chats auto-compact as they approach the model's context window — for every
runtime, including the local assistant. Tools: `read`, `write`, `bash`, `ls`, `grep`, `skill`,
`hf.model_card`, `hf.search` (Pi-style + Hugging Face catalogue). Shell stays in
the workspace when `agent.shell_sandbox = true`.

Coding is **local-first**: when the local Bonsai assistant is installed it is the
**default conversation model** (no `/assistant` required). With no local runtime
and no assistant, the chat will not silently fall back to a cloud provider — set
`agent.allow_cloud_fallback = true` in config.toml to allow the hosted assistant
provider instead.

### Local Bonsai assistant (default chat)

On first launch LocalCode offers to install a **local assistant** based on
[prism-ml/Bonsai-27B-gguf](https://huggingface.co/prism-ml/Bonsai-27B-gguf)
**Q1_0** language model (`Bonsai-27B-Q1_0.gguf`, ~3.8 GB), started like the model card:

```bash
./build/bin/llama-server \
    -m Bonsai-27B-Q1_0.gguf \
    --host 127.0.0.1 --port 18080 -ngl 99
```

> The assistant loads the **Q1_0 language model only** — no Q4 weights are ever
> mapped into memory. The repo’s **Q4_1** file (`Bonsai-27B-dspark-Q4_1.gguf`) is
> a **DSpark drafter** for speculative decoding (`-md`), not a standalone model
> (loading it as `-m` alone makes `llama-server` exit with code 1), so LocalCode
> deliberately does not download or pass it.

Bonsai needs the **[PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp)**
(custom kernels) — stock ggml-org builds will not load it. LocalCode installs
that runtime automatically:

1. **Preferred (model card):** when `git` and `cmake` are on PATH, clone
   `https://github.com/PrismML-Eng/llama.cpp` and build with
   `cmake -B build -DGGML_CUDA=ON` (if `nvcc` is present) then
   `cmake --build build -j`.
2. **Fallback:** download a matching prebuilt from
   [PrismML-Eng/llama.cpp releases](https://github.com/PrismML-Eng/llama.cpp/releases).
3. **Weights:** download `Bonsai-27B-Q1_0.gguf` (Q1 only) into the
   assistant data dir.

- You can **decline** — preference is remembered (`assistant.local_preference`).
- Accept later with **`/assistant`** (or `/assistant accept`), **`/assistant install`**,
  or **Settings → Accept local assistant**.
- When ready it becomes the **default chat runtime** — type normally in the
  conversation view; you do not need to hunt for `/assistant`.
- Tools: shell + filesystem, **Hugging Face search & model cards**, doctor
  snapshot; reads model descriptions, helps launch deploys, and fixes LocalCode
  issues. Also invoked automatically on structured errors when available.

Config (`config.toml` → `[assistant]`): `prefer_local`, `local_port` (default
`18080`), `local_context` (default `131072` — 128k; long chats auto-compact
before it fills), `auto_handle_errors`, `auto_deploy_hints`, `greet_on_startup`.

> Set `HF_TOKEN` if the GGUF download is gated or rate-limited. For a CUDA build
> you need the CUDA toolkit (`nvcc` on PATH) at install time.

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
**omnibar** — a bordered multi-line composer that is active in every mode.

- **Type** a message + `Enter` → chat with the agent. `Shift+Enter` (or
  `Ctrl+Enter` / `Ctrl+J`) inserts a newline; multi-line paste just works.
- Type **`/`** → the command menu docks right above the bar (`↑`/`↓` or the
  mouse wheel pick, `Enter`/click run, `Esc` close). `Ctrl+K` also opens it.
- Type **`@`** → a file picker docks above the bar; pick a workspace file to
  attach its contents to the message (`@src/main.rs` style references).
- **`Esc`** → close a picker/panel, or cancel the running task at home.

| Key | Action |
|-----|--------|
| `Enter` | Send the prompt (or run the highlighted command / attach the file) |
| `Shift+Enter` / `Ctrl+Enter` / `Ctrl+J` | Insert a newline (multi-line composer) |
| `/` or `Ctrl+K` | Open the command menu |
| `@` | Attach a workspace file to the message |
| `↑`/`↓` | Picker / panel navigation (input history at home) |
| `Shift+Tab` | Cycle agent approvals (always approve → auto → approve edits → ask permission) |
| `Ctrl+S` | Save config &nbsp;·&nbsp; `Ctrl+C` quit |
| `F2` | Select mode: release the mouse to copy text (press again to leave) |
| `Esc` | Close picker/panel / cancel task |

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

**Themes**: `ember` (amber on dark maroon — the default), `dark` (grayscale),
`neon` (light blue), `pink` (hot pink), and `sage` (soft green on gray) —
cycle with `/theme`, or use the swatch dots at the right of the status bar
(hover a dot for the theme's name, click to switch).

Everything renders **inline** — no popups: confirms and errors are banners at
the top of the working area, pickers dock above the omnibar. **Model cards**
render as formatted markdown with metadata chips and scroll independently.

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
  localcode-backends/   Ollama, llama.cpp, vLLM, SGLang, colibrì (GLM-5.2 / Hy3)
  localcode-remote/     SSH remote GPU servers (russh: connect, provision, tunnel)
  localcode-cloud/      RunPod, Vast, Akash adapters
  localcode-payments/   USDC balance client
  localcode-bench/      suites + runner + publish
  localcode-agent/      coding agent, skills, MCP, tools
  localcode-assistant/  local Bonsai + hosted app-repair assistant
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
