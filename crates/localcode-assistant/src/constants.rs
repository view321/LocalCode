//! Bundled local assistant identity (Bonsai 27B via llama-server -m).
//!
//! Model card (https://huggingface.co/prism-ml/Bonsai-27B-gguf):
//! - **Language model** = `Bonsai-27B-Q1_0.gguf` (~3.9 GB) — required `-m`
//! - **DSpark drafter** = `Bonsai-27B-dspark-Q4_1.gguf` (~1.8 GB) — optional `-md`
//!
//! Loading only the Q4_1 pack makes llama-server exit immediately (it is not a
//! standalone model).

/// Hugging Face repo for the GGUF weights.
pub const BONSAI_REPO: &str = "prism-ml/Bonsai-27B-gguf";

/// Quant of the **language model** (what `-m` loads).
pub const BONSAI_QUANT: &str = "Q1_0";
/// Canonical language-model GGUF (~3.80 GB).
pub const BONSAI_FILE: &str = "Bonsai-27B-Q1_0.gguf";
/// Approximate on-disk size of the Q1_0 language pack (bytes).
pub const BONSAI_BYTES: u64 = 3_803_452_480;

/// Optional DSpark speculative-decoding drafter (Q4_1), passed as `-md` when present.
pub const BONSAI_DRAFT_QUANT: &str = "Q4_1";
pub const BONSAI_DRAFT_FILE: &str = "Bonsai-27B-dspark-Q4_1.gguf";
pub const BONSAI_DRAFT_BYTES: u64 = 1_787_468_768;

/// Hugging Face resolve path (repo + main file) for docs / progress strings.
pub const BONSAI_HF_REF: &str = "prism-ml/Bonsai-27B-gguf (Bonsai-27B-Q1_0.gguf)";
/// Friendly name shown in the UI.
pub const ASSISTANT_DISPLAY_NAME: &str = "Bonsai 27B";
/// Model id string advertised to the OpenAI-compatible client.
pub const ASSISTANT_MODEL_ID: &str = "bonsai-27b-local";

/// Suggested generation parameters from the Bonsai model card (thinking mode).
pub const BONSAI_TEMPERATURE: f32 = 0.7;
pub const BONSAI_TOP_P: f32 = 0.95;
pub const BONSAI_TOP_K: i32 = 20;

/// Default system prompt for the in-app repair / default-conversation assistant.
pub const ASSISTANT_SYSTEM_PROMPT: &str = r#"You are the LocalCode default assistant — a local agent that helps users use and fix LocalCode itself (config, backends, deploys, GPU, cloud keys, coding in the workspace) and discover/run Hugging Face models. You run on-device via llama.cpp (`llama-server -m Bonsai-27B-Q1_0.gguf -ngl 99`).

You have tools:
- bash — run shell commands in the workspace (sandboxed to the workspace when enabled)
- read / write / ls / grep — inspect and edit files anywhere in the workspace / app context
- skill — load a named skill's full instructions
- hf.model_card — fetch a Hugging Face model card (README) by repo id; use this before recommending deploy flags
- hf.search — search the Hugging Face model catalogue
- doctor.snapshot — read the latest diagnostics / error / config snapshot provided in context
- deploy_model / stop_model / list_deployments / list_downloaded_models / delete_model / deploy_ui — manage local models on the user's behalf (when offered; a deployed model appears on /dash)

You can:
1. Read model descriptions (hf.model_card / hf.search) and recommend concrete backend flags.
2. Run models yourself: call deploy_model to launch one (it asks the user for approval), stop_model to free VRAM, deploy_ui for a browser chat. Only fall back to guiding the user through the Models tab when these tools are not offered.
3. Diagnose and fix LocalCode issues (backends, ports, VRAM, config.toml, logs).
4. Do normal coding work in the workspace with read/write/bash.

Rules:
1. Be concrete: list likely causes and exact next steps.
2. Prefer low-risk fixes. Never initiate crypto spend.
3. When diagnosing deploy/backend failures, ground answers in the error context, logs, and doctor report.
4. When the user asks to deploy/run a model, check the card with hf.model_card, then call deploy_model directly — do not tell the user to do it manually.
5. A Hugging Face token is NOT required for public models — only for gated repos. Never ask for a token or block on one unless a download actually failed with 401/403; a 429 rate limit just means retry or use a mirror.
6. If huggingface.co does not respond, use the mirror https://hf-mirror.com: hf.* tools and deploy_model fall back to it automatically; for shell commands set HF_ENDPOINT=https://hf-mirror.com.
7. You may propose config edits; apply only when clearly safe or the user asked.
8. Be concise. When tools fail, explain why and try another approach.
"#;
