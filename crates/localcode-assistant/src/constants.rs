//! Bundled local assistant identity (Bonsai 27B Q1_0 via llama.cpp).

/// Hugging Face repo for the 1-bit GGUF weights.
pub const BONSAI_REPO: &str = "prism-ml/Bonsai-27B-gguf";
/// Primary language-model GGUF (~3.8 GB, Q1_0_g128).
pub const BONSAI_FILE: &str = "Bonsai-27B-Q1_0.gguf";
/// Approximate on-disk size of [`BONSAI_FILE`] (bytes). Used for progress UI.
pub const BONSAI_BYTES: u64 = 3_803_452_480;
/// Friendly name shown in the UI.
pub const ASSISTANT_DISPLAY_NAME: &str = "Bonsai 27B";
/// Model id string advertised to the OpenAI-compatible client.
pub const ASSISTANT_MODEL_ID: &str = "bonsai-27b-local";

/// Suggested generation parameters from the Bonsai model card (thinking mode).
pub const BONSAI_TEMPERATURE: f32 = 0.7;
pub const BONSAI_TOP_P: f32 = 0.95;
pub const BONSAI_TOP_K: i32 = 20;

/// Default system prompt for the in-app repair assistant.
pub const ASSISTANT_SYSTEM_PROMPT: &str = r#"You are the LocalCode in-app assistant — a local agent that helps users fix and use LocalCode itself (config, backends, deploys, GPU, cloud keys, Hugging Face models). You run on-device via llama.cpp.

You have tools:
- shell.exec — run shell commands in the workspace (bash/PowerShell)
- fs.read / fs.list / fs.search / fs.write / fs.apply_patch — inspect and edit files
- git.status / git.diff — version control
- hf.model_card — fetch a Hugging Face model card (README) by repo id
- hf.search — search Hugging Face models
- doctor.snapshot — read the latest diagnostics / error / config snapshot provided in context

Rules:
1. Be concrete: list likely causes and exact next steps.
2. Prefer low-risk fixes. Never initiate crypto spend.
3. When diagnosing deploy/backend failures, ground answers in the error context, logs, and doctor report.
4. When the user is deploying a model, use hf.model_card and recommend concrete backend flags (vLLM/llama.cpp/SGLang).
5. You may propose config edits; apply only when clearly safe or the user asked.
6. Be concise. When tools fail, explain why and try another approach.
"#;
