# LocalCode — Complete Product & Technical Specification (Handoff)

**Document type:** Full product + technical handoff for multi-pass coding-agent implementation  
**Product name:** LocalCode  
**Primary client form:** Terminal User Interface (TUI)  
**Client language:** Rust  
**Backend:** Hosted on operator VPS; optional accounts  
**Philosophy:** **One-click and no errors** — prefer guided recovery, warnings over hard blocks, and an always-available AI assistant when something fails  
**Status:** Spec only — no implementation yet  
**Date:** 2026-07-13  

---

## 1. Executive summary

LocalCode is a **Rust TUI coding agent** for developers who prefer **local LLMs**, with first-class support for:

- Multiple local inference backends (**Ollama**, **llama.cpp**, **vLLM**, **SGLang**)
- **Hugging Face** as the primary model registry (mirrors supported)
- Model **search**, **metadata**, **popular/trending coding models**
- **GPU awareness** and **VRAM fit prediction**
- **One-click deploy** of any available quantization (warn if oversized; never hard-block)
- Community **benchmarks** (run locally, publish to a VPS backend)
- **Cloud hosting** (RunPod, Vast.ai, Akash) with API-key-driven automation
- **Crypto top-up** (minimal viable: 1–2 chains + stablecoin) for Akash deploy with auto account creation
- An in-app **AI assistant** (OpenRouter / OpenAI-compatible / self-hosted) that can diagnose and fix LocalCode itself
- Full **coding-agent** capabilities: skills, MCP servers, optional subagents

The product is local-first: core coding and local models work **without an account**. Accounts unlock publishing benchmarks, cloud orchestration, crypto balance, and synced preferences.

---

## 2. Goals, non-goals, and success criteria

### 2.1 Goals

1. Make local model discovery → fit check → deploy → code a **single happy path**.
2. Treat cloud and local as **swapable runtimes** behind one coding surface.
3. Make failures **visible, logged, recoverable**, and **assistant-invokable**.
4. Support agentic development of LocalCode itself (robust logging, structured errors, modular crates).
5. Build a trustworthy public performance dataset tied to **official HF model names + quantizations**.

### 2.2 Non-goals (v1)

- Becoming a full Hugging Face Hub clone or social network.
- Training / fine-tuning pipelines (may appear later).
- Guaranteeing every cloud provider API forever without maintenance (providers are adapters with health checks).
- Hard enforcement of “model fits VRAM” (warnings only).
- Broad multi-crypto / multi-wallet marketplace in v1 (see §12).
- Mobile clients.

### 2.3 Success criteria (acceptance-level)

| Area | Done when |
|------|-----------|
| Local deploy | User can search HF model → pick quant → one-click deploy to a configured backend with progress + logs |
| Fit prediction | App reports estimated VRAM need vs free/total GPU memory and warns if over |
| Coding agent | User can open Coding tab, chat, run tools, use skills/MCP; subagents toggle works |
| Benchmarks | User can run a benchmark suite locally and optionally publish results when signed in |
| Cloud | With valid API key, one-click deploy path exists for RunPod, Vast.ai, Akash (Akash can use in-app crypto balance) |
| Errors | Any user-visible error is structured, logged, and offers “Ask assistant” |
| TUI | All tabs listed; right-side nav gray → white on hover/focus; panes resize |
| Offline | Models browse (cached), local backends, coding with local assistant work without backend |

---

## 3. Product philosophy

### 3.1 One-click and no errors

Interpretation for implementers:

- **One-click** means: sensible defaults + preflight + single primary action. Configuration is auto-derived when possible.
- **No errors** does **not** mean “never fail.” It means:
  - Failures are rare on happy paths.
  - When assumptions fail, show **actionable** warnings/errors with **likely causes** and **next steps**.
  - Never fail silently.
  - Prefer **warn and continue** over hard stops for resource fit.
  - Always offer recovery (retry, open Setup, call AI assistant).

### 3.2 Local-first, account-optional

| Capability | Auth required? |
|------------|----------------|
| Browse HF models (live or cache) | No |
| Local GPU detect / fit predict | No |
| Deploy local backend | No |
| Coding agent (local / self-hosted assistant) | No |
| Skills / MCP / subagents | No |
| Create local benchmark runs | No |
| Publish benchmark results | Yes |
| Cloud providers (user API keys) | Optional account; keys can be local-only |
| Akash auto-account + crypto top-up + in-app balance | Yes |
| Sync settings across machines | Yes (optional) |

### 3.3 Trust and transparency

- Always show **quantization**, **approx size**, **license**, and **source registry URL**.
- Benchmark publishes require **reproducible metadata** (model id, quant, backend, hardware, suite version).
- Cloud spend and crypto balance changes require explicit confirmation screens (still “one primary button,” but not silent).

---

## 4. Personas and primary journeys

### 4.1 Personas

1. **Local-only engineer** — dual GPU workstation, Ollama/llama.cpp, no account.
2. **Benchmark contributor** — runs suites, publishes results, cares about quant parity.
3. **Cloud burster** — local machine too small; uses RunPod/Vast/Akash when needed.
4. **Crypto-native deployer** — tops up with stablecoin, deploys on Akash without manually wiring cloud consoles.

### 4.2 Core journeys

#### J1 — Discover and deploy a coding model locally

1. Open **Models**.
2. Search or open Popular/Trending (coding).
3. Open model detail (card, sizes, quants, estimated VRAM).
4. Select quantization + backend.
5. One-click **Deploy**.
6. If predicted not to fit → **warning banner** (continue allowed).
7. Progress stream in UI + logger; model appears as active runtime for Coding.

#### J2 — Code with the agent

1. Open **Coding**.
2. Select runtime (local deployed model, cloud endpoint, or assistant provider).
3. Chat / agent loop with tools, skills, MCP.
4. Toggle subagents on/off.
5. On tool/runtime failure → error panel + “Ask assistant”.

#### J3 — Publish a benchmark

1. Open **Benchmarks**.
2. Pick suite (or create custom).
3. Select local model+quant currently deployed (or path).
4. Run locally; view results.
5. Sign in if needed → **Publish** with official HF name + quant.

#### J4 — Cloud one-click

1. **Setup** → add RunPod / Vast / Akash API key (or Akash via crypto balance).
2. Models → Deploy → target **Cloud**.
3. App provisions instance, waits healthy, wires OpenAI-compatible endpoint.
4. On assumption failure → warning/error with causes (quota, region, image, GPU type unavailable, etc.).

#### J5 — Crypto top-up → Akash deploy

1. Settings/Setup → Wallet top-up (v1 stablecoin path).
2. Confirm payment; balance updates.
3. Deploy model to Akash using balance; auto-create managed Akash account if missing.

---

## 5. Information architecture & TUI UX

### 5.1 Global layout

```
┌──────────────────────────────────────────────────────────────┬────────────┐
│  Title / context bar (app name, env, active model, GPU sum)  │            │
├──────────────────────────────────────────────────────────────┤  dashboard │
│                                                              │  models    │
│                  ACTIVE VIEW (resizable panes)               │  benchmarks│
│                                                              │  coding    │
│                                                              │  setup     │
│                                                              │  notifs    │
│                                                              │  settings  │
├──────────────────────────────────────────────────────────────┤            │
│  Status / log strip (last error, progress, assistant CTA)    │            │
└──────────────────────────────────────────────────────────────┴────────────┘
```

### 5.2 Tabs (right rail)

**Canonical ordered list** (gray text by default):

1. `dashboard`
2. `models`
3. `benchmarks`
4. `coding`
5. `setup`
6. `notifications`
7. `settings`

**Visual rules:**

- Right rail labels render in **gray** (`Style::default().fg(Color::DarkGray)` or theme token `nav.idle`).
- When the pointer is over the **right rail region** (mouse hover) **or** a label is keyboard-focused, that label (or all rail labels—see below) becomes **white**.
  - **Recommended behavior:** entire rail text brightens on rail hover; **active tab** is always bold/white/underlined; hovered non-active tab is white but not bold.
- Click / Enter selects tab.
- Keyboard: `Tab`/`Shift+Tab` or `1`–`7` or `j/k` within rail; `/` focuses search when on Models.
- Mouse support **required** (hover + click + drag-to-resize).

**Note:** User originally listed six tabs and also required Settings. Settings is included as a seventh tab so configuration is first-class without overloading Setup.

### 5.3 Resizing

- Application must run in a **normal resizable terminal**; layout recomputes on `Resize` events.
- Each major view is composed of **resizable panes** (splitters):
  - Horizontal and vertical splits where the view has multiple regions (e.g., Models: list | detail | deploy panel).
  - Drag splitter with mouse; keyboard adjust with `[` `]` / `-` `=` when splitter focused.
  - Persist pane ratios per view in local config.
- Minimum pane sizes enforced; if terminal too small, show compact mode + message rather than panic.

### 5.4 Theme tokens (baseline)

| Token | Meaning |
|-------|---------|
| `nav.idle` | Gray rail text |
| `nav.hover` | White rail text |
| `nav.active` | White + bold |
| `warn` | Yellow/amber |
| `error` | Red |
| `ok` | Green |
| `muted` | Secondary metadata |
| `accent` | Primary actions |

Support a simple light/dark preference in Settings (default dark).

### 5.5 View specifications

#### 5.5.1 Dashboard

- Active runtimes (local + cloud).
- GPU cards: name, VRAM used/total, util if available.
- Recent errors (deep-link to notifications/logs).
- Quick actions: Deploy model, Open coding, Run benchmark, Top-up.
- Backend connectivity status (VPS health).
- Assistant status (configured / not).

#### 5.5.2 Models

**Panes:**

1. **Search & filters** — query, task=`code` presets, sort (downloads, likes, trending, recency), parameter size range, license filter, backend compatibility hints.
2. **Results list** — name, org, approx size, likes/downloads, trending badge.
3. **Detail** — model card (rendered markdown subset), files/quants, metadata, estimated VRAM table per quant, links.
4. **Deploy panel** — backend selector, quant selector, device map, port, one-click Deploy, cloud target toggle.

**Behaviors:**

- Hugging Face primary registry; mirror base URL configurable.
- Parse & cache model metadata (see §8).
- Popular + Trending for **coding** (heuristic tags: `text-generation`, `code`, popular coding model lists, downloads velocity).
- One-click deploy any listed quant.
- If `predicted_vram > free_vram` (or total): show **warning**, require explicit Continue, **do not hard-block**.

#### 5.5.3 Benchmarks

- Catalog of official + community + user-created suites.
- Run wizard: suite → model/quant → backend → run → results chart/table.
- Publish UI (auth-gated).
- Leaderboard browser (from VPS): filter by model family, quant, GPU class, suite.
- Create benchmark flow (see §10).

#### 5.5.4 Coding

- Session list + transcript.
- Composer with attachments (file paths).
- Tool call / agent timeline.
- Runtime picker.
- Skills browser toggle.
- MCP server status.
- Subagents: global toggle + per-session override.
- Diff/apply UX for file edits (confirm policy configurable).

#### 5.5.5 Setup

Wizard-oriented connectivity:

1. Detect GPUs.
2. Install/configure backends (Ollama / llama.cpp / vLLM / SGLang) — detect existing; print commands if missing; optional guided install where safe.
3. Hugging Face token (optional, for gated models) + mirror URL.
4. Cloud API keys (RunPod, Vast.ai, Akash).
5. Assistant provider (OpenRouter / OpenAI-compatible URL+key / self-hosted model).
6. Connectivity tests with clear pass/fail.

#### 5.5.6 Notifications

- Chronological feed: deploys, benchmarks, payments, cloud status, errors.
- Severity filters.
- Actions: dismiss, open related view, ask assistant, copy log id.

#### 5.5.7 Settings

- Theme, keybindings, mouse.
- Data directories (models cache, logs, workspaces).
- Default backend & quant preferences.
- Agent: max tools, confirmations, subagents default on/off, skill paths, MCP config path.
- Logging level, redaction.
- Account session, device name.
- Payment / balance display.
- Feature flags (experimental adapters).
- Export/import config.

### 5.6 Cross-cutting UI components

- **Command palette** (`Ctrl+K`): jump tabs, deploy last model, toggle subagents, open logs.
- **Modal system**: confirm, warning, error detail, payment confirm.
- **Assistant dock**: slide-over or bottom pane; pre-filled with error context when invoked from an error.
- **Toast/status strip**: non-blocking messages; errors pin until dismissed.

---

## 6. System architecture

### 6.1 High-level diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     LocalCode TUI (Rust)                     │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌───────────┐ │
│  │  UI    │ │ Agent  │ │ Model  │ │ Bench  │ │  Cloud    │ │
│  │(ratatui│ │ Runtime│ │Registry│ │ Runner │ │ Orchestr. │ │
│  │+crossterm)│ Skills │ │  HF    │ │        │ │           │ │
│  │        │ │ MCP    │ │ GPU    │ │        │ │ Payments  │ │
│  └────┬───┘ └───┬────┘ └───┬────┘ └───┬────┘ └─────┬─────┘ │
│       │         │          │          │            │       │
│  ┌────▼─────────▼──────────▼──────────▼────────────▼─────┐ │
│  │              Core services                             │ │
│  │  config · logger · errors · event bus · auth client    │ │
│  └───────────────────────────┬────────────────────────────┘ │
└──────────────────────────────┼──────────────────────────────┘
                               │ HTTPS
                               ▼
                    ┌──────────────────────┐
                    │  LocalCode API (VPS) │
                    │  auth · benchmarks   │
                    │  leaderboards ·      │
                    │  payments webhooks   │
                    │  model hints cache   │
                    └──────────┬───────────┘
                               │
                          PostgreSQL
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   Hugging Face (+mirrors)  Local backends     Cloud APIs
   OpenRouter / OpenAI*     Ollama/llama.cpp   RunPod/Vast/Akash
                            vLLM/SGLang        Payment rails
```

### 6.2 Process model

- **Single binary** `localcode` (TUI default when no subcommand / when `localcode tui`).
- Optional **headless CLI** subcommands for automation (same crate, not a second product):
  - `localcode doctor`
  - `localcode models search "..."`
  - `localcode deploy ...`
  - `localcode bench run ...`
  - `localcode agent run ...` (non-interactive)
- Long-running deploys/benchmarks run as **managed child processes** supervised by the app (or a small local supervisor task inside the same process using async jobs).

### 6.3 Recommended crate layout (Rust workspace)

```
localcode/
  Cargo.toml                 # workspace
  crates/
    localcode-core/          # config, error types, events, ids
    localcode-log/           # structured logging, redaction
    localcode-gpu/           # GPU discovery + VRAM fit
    localcode-hf/            # HF API client, mirrors, model parse, cards
    localcode-backends/      # ollama, llama.cpp, vllm, sglang adapters
    localcode-cloud/         # runpod, vast, akash
    localcode-payments/      # top-up client, balance
    localcode-bench/         # suites, runners, publish payloads
    localcode-agent/         # coding agent, tools, skills, MCP, subagents
    localcode-assistant/     # app-fix assistant (may share agent core)
    localcode-api-client/    # VPS REST client
    localcode-tui/           # ratatui UI
    localcode-cli/           # binary entry: TUI + subcommands
  server/                    # VPS backend (recommend Rust axum OR separate service)
    localcode-server/
  docs/
    HANDOFF.md               # this file
```

**TUI stack recommendation:** `ratatui` + `crossterm` + `tokio` + `reqwest` + `serde` + `tracing` + `sqlx` (server) + `jsonschema` for MCP.

### 6.4 Server stack recommendation

- **Language:** Rust (`axum` + `sqlx` + `postgres`) for one language across monorepo, OR TypeScript/Go if operator prefers faster HTTP iteration. **Default in this handoff: Rust axum.**
- **DB:** PostgreSQL
- **Cache:** Redis optional (rate limits, sessions); can start without
- **Object storage:** optional for large raw benchmark artifacts (S3-compatible)
- **Auth:** email magic link **or** device code + API tokens (optional OAuth later)
- **Migrations:** versioned SQL

---

## 7. Configuration model

### 7.1 Config locations (Windows + Unix)

| Item | Path |
|------|------|
| Config | `$LOCALCODE_HOME/config.toml` defaulting to platform config dir `localcode/config.toml` |
| Data | platform data dir `localcode/` (cache, db sqlite local, workspaces) |
| Logs | platform state/log dir `localcode/logs/` |
| Secrets | OS keyring preferred; fallback encrypted file with user warning |

Environment overrides:

- `LOCALCODE_HOME`
- `LOCALCODE_API_URL` (default production VPS URL)
- `LOCALCODE_HF_ENDPOINT` (mirror)
- `LOCALCODE_LOG_LEVEL`
- `HTTP_PROXY` / `HTTPS_PROXY`

### 7.2 Config schema (conceptual)

```toml
[ui]
theme = "dark"
mouse = true
right_rail_hover_brightens = true

[registry]
provider = "huggingface"
endpoint = "https://huggingface.co"
# mirror example: "https://hf-mirror.com"
api_endpoint = "https://huggingface.co/api"
token_env = "HF_TOKEN"

[backends.default]
kind = "ollama"  # ollama | llamacpp | vllm | sglang

[backends.ollama]
base_url = "http://127.0.0.1:11434"

[backends.llamacpp]
bin = "llama-server"
host = "127.0.0.1"
port = 8080

[backends.vllm]
# ...

[backends.sglang]
# ...

[assistant]
provider = "openrouter" # openrouter | openai_compatible | self_hosted
base_url = "https://openrouter.ai/api/v1"
model = "..."
# api_key in keyring

[agent]
subagents_enabled = true
skills_dir = "~/.localcode/skills"
mcp_config = "~/.localcode/mcp.json"
confirm_destructive_tools = true

[cloud.runpod]
# api_key in keyring
enabled = false

[cloud.vast]
enabled = false

[cloud.akash]
enabled = false
managed_account = true

[api]
base_url = "https://api.localcode.example"
# session token in keyring

[logging]
level = "info"
redact_secrets = true
max_files = 20
```

---

## 8. Hugging Face registry integration

### 8.1 Responsibilities

- Search models
- Fetch model info, siblings (files), tags, pipeline_tag, library_name, likes, downloads, lastModified
- Fetch README / model card markdown
- Enumerate quantizations / weight files (GGUF, GPTQ, AWQ, EXL2, safetensors shards, etc.)
- Resolve download URLs via primary or **mirror** base
- Optional auth for gated models
- Local metadata cache with ETag / last-modified

### 8.2 Mirror support

- User sets `registry.endpoint` / `registry.api_endpoint`.
- All constructed URLs must go through a **UrlBuilder** that swaps host/base.
- On mirror failure: auto-retry primary (if different) once; surface warning “mirror failed, fell back to primary” with causes (DNS, TLS, rate limit).

### 8.3 Search UX

- Debounced search (e.g., 300ms) against HF API `models` search.
- Filters: coding-oriented presets (`pipeline_tag`, tags containing `code`, known coding orgs/models boost).
- Trending: combination of HF sort + local/server curated list for coding models (server can provide `GET /v1/models/trending?task=code`).
- Popular: downloads/likes sort.

### 8.4 Metadata to parse and display

| Field | Source | Display |
|-------|--------|---------|
| model_id | id | title |
| author | namespace | subtitle |
| pipeline_tag | API | badge |
| tags | API | chips |
| likes, downloads | API | stats |
| last_modified | API | relative time |
| card/README | raw | markdown pane (sanitized) |
| license | card YAML or tag | badge |
| parameter size | card / name heuristics / safetensors index | “7B”, “70B” |
| file inventory | tree/siblings | quant table |
| quant type | filename patterns | Q4_K_M, AWQ, etc. |
| disk size | file sizes sum | GiB |
| estimated VRAM | fit engine | GiB estimate |

### 8.5 Quantization discovery rules

- Parse filenames with known patterns (`Q4_K_M`, `IQ4_XS`, `gptq`, `awq`, `exl2`, `fp16`, `bf16`).
- Group by quant family; show size per file/group.
- For multi-shard safetensors, sum shards.
- If unknown, still list file with size and allow deploy attempt with warning “unknown quant metadata”.

---

## 9. GPU awareness & fit prediction

### 9.1 Discovery

- **Windows:** DXGI / NVML if available; fallback to `nvidia-smi` parsing; try Vulkan/WMI for basic adapters.
- **Linux:** NVML, `nvidia-smi`, `/proc/driver/nvidia`, ROCm `rocm-smi` best-effort.
- Expose per-GPU: index, name, total_vram, free_vram, driver version, backend affinity (CUDA/ROCm/CPU).

### 9.2 Fit model (v1 heuristic — explicit about uncertainty)

```
estimated_vram ≈ weight_bytes * dtype_factor + kv_cache_estimate + runtime_overhead
```

Where:

- `weight_bytes` from file sizes or param_count * bytes_per_param(quant)
- `kv_cache_estimate` from context length setting (default 8k/16k/32k selectable)
- `runtime_overhead` backend-specific constants (Ollama vs vLLM differ)
- Multi-GPU: optional tensor parallel assumptions for vLLM/SGLang

**Output structure:**

```json
{
  "estimated_vram_bytes": 123,
  "free_vram_bytes": 456,
  "total_vram_bytes": 789,
  "fits_free": false,
  "fits_total": true,
  "confidence": "low|medium|high",
  "assumptions": ["ctx=8192", "backend=llamacpp", "kv_dtype=fp16"],
  "warning": "Model may exceed free VRAM; deploy may spill to RAM/CPU or fail."
}
```

### 9.3 Policy

- **Never hard-block deploy** due to fit.
- Always show warning when `fits_free == false`.
- Stronger warning when `fits_total == false`.
- Log assumptions with the deploy job for debugging.

---

## 10. Local backends

### 10.1 Adapter interface

```text
trait InferenceBackend {
  name() -> BackendKind
  detect() -> DetectReport
  ensure_ready() -> Result<()>
  list_models()
  pull_or_materialize(spec: ModelDeploySpec) -> JobHandle
  start_server(spec) -> RunningEndpoint  # OpenAI-compatible where possible
  stop(id)
  health(id) -> Health
  logs(id) -> Stream
}
```

### 10.2 Backend notes

| Backend | Deploy strategy | Endpoint style |
|---------|-----------------|----------------|
| Ollama | `ollama pull` / create from GGUF when supported | Ollama API + OpenAI compat if available |
| llama.cpp | download GGUF → `llama-server` args | OpenAI-compatible server |
| vLLM | HF repo id / local path → `vllm serve` | OpenAI-compatible |
| SGLang | similar | OpenAI-compatible |

### 10.3 One-click deploy pipeline

1. Validate network / disk space (warn if low).
2. Fit prediction warning gate (Continue).
3. Download weights (progress, resume, mirror).
4. Convert/import if required by backend.
5. Start process with generated flags.
6. Health poll until ready or timeout.
7. Register as **ActiveRuntime** for Coding.
8. On failure: structured error + causes + assistant CTA + link to logs.

**Assumption failures** must enumerate likely causes, e.g.:

- binary not on PATH
- CUDA version mismatch
- port in use
- insufficient disk
- gated model without HF token
- mirror returned 403/HTML error page

---

## 11. Cloud hosting

### 11.1 Providers (v1)

| Provider | Auth | Notes |
|----------|------|-------|
| RunPod | User API key | Prefer serverless or pod templates known-good for vLLM/Ollama |
| Vast.ai | User API key | Search offers by GPU/price; create instance |
| Akash | Managed account and/or user wallet integration | Auto-create account when using in-app balance |

### 11.2 Orchestration flow

1. User selects model+quant+provider.
2. App selects GPU SKU / bid using cost + VRAM fit heuristics.
3. Provision instance / deployment.
4. Wait for SSH/HTTP ready.
5. Install or launch inference image (prebaked images preferred).
6. Expose OpenAI-compatible URL (+ auth token).
7. Store runtime metadata; surface estimated $/hr.
8. Support **Stop** / **Destroy** to avoid runaway spend.

### 11.3 Failure UX (required)

If any assumption fails, show:

- What was attempted
- Provider error (raw + parsed)
- **Possible causes** (bullet list)
- **What to try** (rotate region, add credits, fix key scopes, change GPU type)
- Log correlation id
- **Ask assistant** button

Never partially provision without listing orphan resources and offering cleanup.

### 11.4 Security

- API keys only in keyring.
- Cloud endpoints displayed with redaction options.
- Spend confirmation when estimated hourly cost above user threshold (Settings).

---

## 12. Payments & crypto (minimal viable v1)

### 12.1 Scope

- Purpose: top up **LocalCode account balance** used primarily for **Akash deploys**.
- **v1 assets:** 1–2 chains + **stablecoin** (recommendation below).
- Fiat out of scope for v1.

### 12.2 Recommended v1 rails

**Primary:** **USDC** on **one low-fee L2** (recommend **Base** or **Arbitrum** — pick one at implementation start and document).  
**Secondary (optional same release if low cost):** USDC on Ethereum mainnet (higher fees; warn user) **or** TRON/SOL only if integration cost is already covered by provider SDK — otherwise defer.

**Integration pattern:**

- Hosted payment provider / crypto payment API (e.g. self-hosted watcher + deposit addresses, or a reputable payments processor that supports USDC).
- Server generates **deposit address** or **payment intent** tied to `user_id`.
- Watcher credits balance after N confirmations.
- TUI shows price quotes for Akash resources **fetched live** from provider + LocalCode fee schedule.

### 12.3 Balance model

- Ledger entries: `deposit`, `hold`, `capture`, `refund`, `adjust`.
- Deploy places a **hold** based on estimate; settle on destroy/stop.
- Display: available balance, holds, recent transactions.

### 12.4 Akash auto account

- On first Akash deploy with managed mode:
  - Server creates/manages provider credentials **or** app generates wallet secured by user passphrase (prefer **server-managed escrow + provider deploy** for true one-click, with clear custody disclosure).
- **Must disclose** custody model in Setup UI.
- If chain/API assumptions fail → error with causes (insufficient balance, network congestion, provider unavailable).

### 12.5 Pricing display

- Before deploy: estimated $/hr and crypto equivalent at current quote.
- Refresh quotes; if quote stale > T seconds, force refresh.
- User confirms once.

---

## 13. Benchmarks system

### 13.1 Concepts

- **Suite:** versioned set of tasks (prompts, expected patterns, metrics, timeouts).
- **Run:** execution of suite against a concrete **Subject** (model_id, quant, backend, backend_version, hardware fingerprint).
- **Result:** metrics + raw outputs (optional upload) + environment.
- **Publish:** authenticated upload to VPS; appears on leaderboards.

### 13.2 Official naming rules (publish)

Publish payload **must** include:

- `hf_model_id` (exact Hub id, e.g. `Qwen/Qwen2.5-Coder-7B-Instruct`)
- `quantization` (normalized enum + raw string)
- `weight_source` (HF file names / commit sha / etag)
- `backend` + version
- `precision_notes`
- `hardware`: GPU name(s), VRAM, CPU, RAM, OS
- `suite_id` + `suite_version`
- `metrics`: e.g. pass@k, latency_p50/p95, tokens/s, score
- `started_at`, `finished_at`, `runner_version`
- Optional: `code_hash` of suite tasks for integrity

Reject publish if required fields missing.

### 13.3 Running HF-oriented benchmarks

- Support pulling task definitions compatible with common public coding evals (document which: e.g. HumanEval-style, MBPP-style, custom JSONL).
- Local execution only by default; network tools disabled unless suite allows.
- Streaming progress in Benchmarks tab + logs.

### 13.4 Creating benchmarks

UI + schema:

1. Name, description, version, license.
2. Task editor: prompt template, files, assertions (regex, exact, model-graded optional later), timeouts.
3. Scoring: weight per task, aggregate function.
4. Validation dry-run on a tiny model/runtime.
5. Export suite package (`suite.toml` + `tasks/`).
6. Optional publish suite definition (auth) for others.

### 13.5 Server API (benchmarks)

- `POST /v1/bench/results` — publish
- `GET /v1/bench/results` — query leaderboard filters
- `GET /v1/bench/suites` — list suites
- `POST /v1/bench/suites` — publish suite
- Rate limit + spam controls; optional moderation flag

### 13.6 Trust

- Mark results `unverified` by default.
- Optional future: reproducible attestation. v1: show hardware self-report + client version.

---

## 14. Coding agent specification

### 14.1 Role

LocalCode’s **Coding** tab is a full coding agent oriented at repositories on disk.

Capabilities:

- Multi-turn chat with tool use
- Read/list/search files, apply patches, run shell commands (policy-gated)
- Skills
- MCP servers
- Subagents (toggle)

### 14.2 Tools (v1 baseline)

| Tool | Description | Risk |
|------|-------------|------|
| `fs.read` | Read file | low |
| `fs.list` | List dir | low |
| `fs.search` | ripgrep-like | low |
| `fs.write` / `fs.apply_patch` | Edit files | medium |
| `shell.exec` | Run command in workspace | high |
| `git.status` / `git.diff` | VCS awareness | low |
| `web.fetch` (optional) | If enabled | medium |
| `mcp.*` | Proxy to MCP tools | varies |
| `subagent.spawn` | Delegate task | medium |

### 14.3 Skills

- Skills are folders with `SKILL.md` (+ optional scripts).
- Discovery: user skills dir + bundled skills.
- Agent loads skill instructions when relevant (user invoke or router heuristic).
- Settings: enable/disable individual skills.

### 14.4 MCP

- Config file lists servers (command/env or HTTP).
- Lifecycle: start on demand, health, tool schema cache.
- UI: list servers, tools, errors.
- Failures must not crash agent loop; degrade with message.

### 14.5 Subagents

- Global toggle in Settings + visible toggle in Coding.
- When **off**: main agent never spawns subagents; tool hidden/disabled.
- When **on**: main agent may delegate explore/plan/review style tasks with isolated context.
- Subagent output returns as a single message to parent.
- Resource limits: max concurrent subagents, timeout.

### 14.6 Model routing

Coding runtime may be:

1. Active local backend endpoint
2. Active cloud endpoint
3. Assistant provider (OpenRouter / OpenAI-compatible / self-hosted)

User can pin model per session.

### 14.7 Workspace safety

- Default workspace root required.
- Destructive commands confirmation (configurable).
- `.localcodeignore` support.
- Secret redaction in logs/transcripts.

---

## 15. In-app AI assistant (app repair)

### 15.1 Purpose

A specialized agent that helps users **fix LocalCode issues** (config, backends, deploys, cloud keys, payments), distinct from but sharing infrastructure with the coding agent.

### 15.2 Providers

- OpenRouter
- Any OpenAI-compatible endpoint
- Self-hosted model (local backend)

### 15.3 Invocation

- Manual: command palette / shortcut / Assistant entry.
- **Automatic prompt on errors:** any `Error` modal includes:
  - Summary
  - Possible causes
  - Buttons: `Retry` | `Open logs` | `Ask assistant` | `Dismiss`
- Asking assistant attaches: error struct, recent log slice, config redacted snapshot, doctor report.

### 15.4 Permissions

- Assistant may propose config edits and run `doctor` diagnostics.
- Applying fixes requires confirmation unless user enables auto-apply for low-risk fixes.
- Cannot initiate crypto spend without explicit user confirmation.

---

## 16. Logging & error handling (agent-maintainable)

### 16.1 Goals

The codebase will be built/modified by coding agents in one or several passes. Logging and errors must make failures **diagnosable without a debugger**.

### 16.2 Logging

- Use `tracing` + JSON or logfmt files + optional pretty stderr.
- Levels: error, warn, info, debug, trace.
- Every request/job has `correlation_id` (UUID).
- Rotate logs; max size/count in Settings.
- **Redact** API keys, tokens, mnemonics, Authorization headers.
- UI “Open logs” jumps to file + filters by correlation id.

Log events of interest:

- deploy steps
- HF API failures
- backend stdout/stderr (captured, truncated with pointers)
- cloud provision state transitions
- payment ledger updates (no secrets)
- agent tool calls (args redacted)

### 16.3 Error type system

```text
LocalCodeError {
  code: ErrorCode,           // stable machine code, e.g. BACKEND_PORT_IN_USE
  message: String,           // human short
  causes: Vec<String>,       // possible causes
  hints: Vec<String>,        // what to try
  correlation_id: Uuid,
  retryable: bool,
  source: Option<AppErrorSource>,
  details: serde_json::Value // structured, non-sensitive
}
```

Rules:

- Map IO/HTTP/provider errors into `ErrorCode`.
- User-visible surfaces always show `message` + `causes` + `hints`.
- Panic hooks install → log + friendly recovery screen; never raw panic to users in release.
- Fallible startup: enter TUI with Setup focus and error notification rather than exit when possible.

### 16.4 Doctor

`localcode doctor` and Setup diagnostics:

- GPU detect
- backend binaries
- ports
- HF reachability/mirror
- API reachability
- keyring
- disk space

Produces a JSON report attachable to assistant.

---

## 17. Backend (VPS) specification

### 17.1 Responsibilities

- Auth (optional accounts)
- Benchmark suites & results storage
- Trending/popular coding model hints (cache/curated)
- Payment intents & balance ledger
- Akash managed orchestration support (if server-mediated)
- Feature flags / min client version
- Telemetry opt-in only

### 17.2 Suggested REST surface

**Auth**

- `POST /v1/auth/device/start`
- `POST /v1/auth/device/poll`
- `POST /v1/auth/logout`
- `GET /v1/me`

**Models (optional cache)**

- `GET /v1/models/trending?task=code`
- `GET /v1/models/popular?task=code`

**Benchmarks**

- `GET /v1/bench/suites`
- `POST /v1/bench/suites`
- `POST /v1/bench/results`
- `GET /v1/bench/results`
- `GET /v1/bench/results/{id}`

**Payments**

- `GET /v1/billing/balance`
- `POST /v1/billing/deposits`
- `GET /v1/billing/deposits/{id}`
- `GET /v1/billing/transactions`
- `GET /v1/billing/quotes/akash` (resource price fetch aggregation)

**Cloud assist (optional)**

- `POST /v1/cloud/akash/deployments`
- `DELETE /v1/cloud/akash/deployments/{id}`

**Meta**

- `GET /v1/health`
- `GET /v1/client/min-version`

### 17.3 Data model (PostgreSQL, simplified)

**users**

- id, email (nullable), created_at, status

**sessions / api_tokens**

- user_id, token_hash, expires_at

**bench_suites**

- id, slug, version, title, definition_json, publisher_id, created_at

**bench_results**

- id, suite_id, suite_version, user_id, hf_model_id, quantization, backend, hardware_json, metrics_json, runner_version, created_at, visibility

**ledger_accounts**

- user_id, currency, available, held

**ledger_entries**

- id, user_id, type, amount, ref_type, ref_id, created_at, metadata_json

**deposits**

- id, user_id, chain, asset, address, txid, status, amount, confirmations

**cloud_deployments** (if server-tracked)

- id, user_id, provider, status, cost_hold, endpoint, metadata_json

Indexes on `hf_model_id`, `quantization`, `suite_id`, `created_at`.

### 17.4 Security & ops

- TLS only
- Rate limits per IP and user
- CORS not required for TUI; keep tight if any web admin
- Backups for Postgres
- Migrate with zero-downtime when possible
- Admin moderation for abusive benchmark spam

---

## 18. Security & privacy

- Secrets in OS keyring whenever possible.
- Redact secrets in UI copy actions by default (show once).
- Workspace tools confined to chosen root unless user elevates.
- Payment confirmations explicit.
- Gated HF models: user supplies token; never upload HF token to LocalCode VPS.
- Cloud keys remain local unless user opts into server-mediated Akash management (disclose).
- Opt-in telemetry only; default off.
- Dependency auditing in CI.

---

## 19. Internationalization & accessibility

- v1 English UI strings centralized (fluent or simple rust i18n ready).
- High-contrast theme option.
- Full keyboard navigation; mouse optional but supported.
- Screen reader: best-effort via clear textual hierarchy (TUI limitations accepted).

---

## 20. Testing strategy

| Layer | What |
|-------|------|
| Unit | fit estimator, quant parse, error mapping, config |
| Contract | HF client against fixtures; backend adapters mocked |
| Integration | deploy dry-run; API client against testcontainers Postgres |
| TUI | snapshot tests for layouts; event-driven component tests |
| E2E (smoke) | doctor, search cached models, agent tool on temp dir |
| Load | publish benchmarks rate limits |

Golden fixtures for model cards and filename quant parsing.

---

## 21. Implementation phases (for multi-pass agents)

### Phase 0 — Skeleton

- Workspace crates, CI, `localcode` binary, tracing, error types, config load
- Empty TUI shell with right rail tabs (gray/white hover), resize, status strip
- **Exit criteria:** app runs, switches tabs, resizes without panic

### Phase 1 — Models + HF + GPU

- HF client + mirror, search, detail, card render, quant table, cache
- GPU discover + fit prediction warnings
- **Exit criteria:** browse/search models offline-cached after first fetch

### Phase 2 — Local backends one-click deploy

- Adapter for at least **Ollama** + **llama.cpp**; stubs for vLLM/SGLang
- Deploy job progress + logs
- ActiveRuntime registry
- **Exit criteria:** one-click GGUF/Ollama deploy with oversize warning path

### Phase 3 — Coding agent MVP

- Chat loop, core tools, workspace, transcripts
- Skills load, MCP connect, subagents toggle
- **Exit criteria:** agent can read repo and apply a patch with confirmation

### Phase 4 — Assistant + error UX

- Assistant providers
- Error modal → Ask assistant with context pack
- Doctor command
- **Exit criteria:** forced backend error offers assistant with prefilled logs

### Phase 5 — Benchmarks local

- Suite format, runner, results UI, create suite
- **Exit criteria:** run sample coding suite on local model

### Phase 6 — VPS API + publish + accounts

- Auth optional, publish results, leaderboards, trending endpoint
- **Exit criteria:** signed-in publish appears on GET leaderboard

### Phase 7 — Cloud adapters

- RunPod + Vast with user API keys; robust error causes
- **Exit criteria:** deploy + destroy one model endpoint (staging keys)

### Phase 8 — Payments + Akash

- USDC deposit flow, balance, Akash one-click with managed account disclosure
- **Exit criteria:** testnet or sandbox top-up + mock/real deploy per operator env

### Phase 9 — Hardening

- Redaction audit, resume downloads, flake fixes, docs, installers
- Performance pass on TUI render
- **Exit criteria:** release checklist green

---

## 22. Detailed acceptance criteria (checklist)

### TUI / UX

- [ ] Tabs: dashboard, models, benchmarks, coding, setup, notifications, settings
- [ ] Right rail gray idle; white on hover/focus; active distinct
- [ ] Window resize safe; per-view pane ratios adjustable and persisted
- [ ] Command palette and status strip present

### Models

- [ ] HF search + filters + popular/trending coding
- [ ] Metadata: size, card, quant list, license, stats
- [ ] Mirrors configurable with fallback warning
- [ ] One-click deploy any quant; oversize **warns only**

### GPU / backends

- [ ] Lists connected GPUs with VRAM
- [ ] Fit prediction with assumptions listed
- [ ] Ollama, llama.cpp, vLLM, SGLang adapters (quality may vary; all selectable)
- [ ] Structured failures with causes

### Agent

- [ ] Skills, MCP, subagents toggle
- [ ] Coding tools work in a workspace
- [ ] Assistant via OpenRouter / OpenAI-compatible / self-hosted
- [ ] Errors prompt assistant

### Benchmarks / backend

- [ ] Create + run suites locally
- [ ] Publish with official model name + quant
- [ ] DB-backed leaderboard queries

### Cloud / payments

- [ ] RunPod, Vast.ai, Akash paths
- [ ] API key setup; assumption failures explained
- [ ] Crypto top-up (USDC, 1–2 chains), prices shown, Akash auto account path

### Quality

- [ ] Robust logger (correlation ids, rotation, redaction)
- [ ] Unified error type + user hints
- [ ] Tests for core parsers and fit engine
- [ ] No secrets in logs in default config

---

## 23. Open decisions (defaults locked for implementers)

These were ambiguous; **defaults are fixed here** so agents do not block:

| Topic | Default |
|-------|---------|
| UI form | TUI (ratatui), not GUI |
| Language | Rust client; Rust axum server |
| Accounts | Optional |
| Settings | 7th tab (not buried only in setup) |
| Oversize models | Warn + continue, never hard-stop |
| Crypto v1 | USDC on a single L2 (Base **or** Arbitrum — choose in Phase 8 README) + optional second rail |
| Custody | Server-mediated Akash for one-click, with clear disclosure; advanced BYO key later |
| vLLM/SGLang on Windows | Best-effort; document Linux as preferred host for those backends |
| Headless CLI | Supported subset for CI/scripting; TUI is primary |
| Telemetry | Off by default |

If product owner overrides a default, update this table and the affected sections in the same PR.

---

## 24. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| HF API rate limits | Cache aggressively; mirrors; conditional requests |
| Provider API churn (Vast/RunPod/Akash) | Adapter layer + contract tests + version pins |
| VRAM estimates wrong | Show confidence + assumptions; never block |
| Crypto regulatory/compliance | Minimal rails; geoblocking hooks server-side if needed; ToS |
| Agent destructive actions | Confirmations, workspace root, audits in transcript |
| TUI mouse inconsistency across terminals | Keyboard-first parity; detect capabilities |
| Multi-pass agent development drift | This handoff + ErrorCode catalog + phase exit criteria |

---

## 25. Deliverables expected from implementation agents

1. Working Rust workspace as in §6.3
2. TUI meeting §5
3. Local deploy path §10
4. Agent + assistant §14–15
5. Server + migrations §17
6. README: install, doctor, first deploy, account, cloud, payments
7. `ERROR_CODES.md` generated or hand-maintained catalog
8. Phase-by-phase PRs with tests

---

## 26. Out-of-scope follow-ups (post-v1)

- Fine-tuning / LoRA studio
- GUI desktop shell
- Multi-user team orgs
- Fiat on-ramps
- Mobile companion
- Fully verified reproducible benchmark attestation
- Windows first-class vLLM (if not achieved in v1)

---

## 27. Glossary

| Term | Meaning |
|------|---------|
| ActiveRuntime | A live local or cloud inference endpoint registered for Coding |
| Quant | Weight quantization / precision variant |
| Suite | Versioned benchmark definition |
| Subject | Model + quant + backend + hardware under test |
| Assistant | In-app repair/help agent |
| Coding agent | Repo-working agent in Coding tab |
| Mirror | Alternate HF-compatible endpoint |
| Hold | Ledger reservation against balance for cloud usage |

---

## 28. Handoff checklist for the next agent

Before writing code:

1. Read this document fully.
2. Create workspace skeleton (Phase 0) only unless asked to go further.
3. Keep user-facing copy aligned with **one-click and no errors**.
4. Prefer structured `LocalCodeError` over ad-hoc strings.
5. Do not hard-block oversized model deploys.
6. Do not require account for local paths.
7. Ask the product owner only if changing a **locked default** in §23.

---

**End of specification.**
