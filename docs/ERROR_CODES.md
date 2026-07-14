# LocalCode Error Codes

Stable machine-readable codes used in `LocalCodeError.code`. User surfaces always show **message**, **causes**, and **hints**, plus **Ask assistant**.

| Code | Meaning | Typical recovery |
|------|---------|------------------|
| `CONFIG_LOAD_FAILED` | Could not read config or paths | Set `LOCALCODE_HOME`; fix permissions |
| `CONFIG_PARSE_FAILED` | Invalid TOML/JSON | Fix file or regenerate defaults |
| `CONFIG_SAVE_FAILED` | Could not write config | Check disk/permissions |
| `HF_UNREACHABLE` | Hugging Face / mirror network failure | Check network, proxy, mirror; use cache |
| `HF_RATE_LIMITED` | HF 429 | Wait; set `HF_TOKEN`; use mirror |
| `HF_AUTH_REQUIRED` | Gated model or bad token | Set `HF_TOKEN` |
| `HF_MIRROR_FAILED` | Mirror failed (may fall back) | Fix mirror URL or use primary |
| `HF_MODEL_NOT_FOUND` | Unknown model id | Verify `org/name` |
| `GPU_DETECT_FAILED` | GPU probe error | Install drivers; `nvidia-smi` on PATH |
| `GPU_NO_DEVICES` | No GPU found | CPU fallback / cloud |
| `BACKEND_NOT_FOUND` | Unknown backend kind | Use ollama/llamacpp/vllm/sglang |
| `BACKEND_NOT_READY` | Backend not reachable | Start service; fix base_url |
| `BACKEND_PORT_IN_USE` | Port conflict | Free port or change deploy port |
| `BACKEND_START_FAILED` | Process spawn/start failed | Check binary, CUDA, logs |
| `BACKEND_HEALTH_TIMEOUT` | Health poll timed out | Smaller model; check OOM |
| `BACKEND_BINARY_MISSING` | Binary not on PATH | Install backend; set config path |
| `BACKEND_INSTALL_FAILED` | Backend/prerequisite install failed | Causes in error; run the shown command manually |
| `DEPLOY_DISK_LOW` | Low disk warning | Free space |
| `DEPLOY_DOWNLOAD_FAILED` | Weight pull/create failed | Network, path, library name |
| `DEPLOY_OVERSIZED_WARNING` | VRAM fit warning (not a hard stop) | Continue or smaller quant |
| `AGENT_TOOL_FAILED` | Tool or LLM call failed | Check workspace, runtime, policy |
| `AGENT_WORKSPACE_MISSING` | Workspace root missing | Set in Settings |
| `AGENT_MCP_FAILED` | MCP config/connect issue | Fix `mcp.json`; degrade OK |
| `CLOUD_KEY_MISSING` | Provider API key missing | Setup → cloud keys |
| `CLOUD_PROVISION_FAILED` | Provider rejected deploy | Causes in error; rotate region/GPU |
| `CLOUD_QUOTA_EXCEEDED` | Credits/quota | Add provider credits |
| `CLOUD_PROVIDER_UNAVAILABLE` | Adapter not registered | Enable provider |
| `PAYMENT_CONFIRM_REQUIRED` | Explicit confirm needed | Confirm in UI |
| `INSUFFICIENT_BALANCE` | In-app balance low | Top up USDC on Base |
| `DEPOSIT_FAILED` | Deposit intent failed | Retry; check chain |
| `API_UNREACHABLE` | VPS API down | Local features still work |
| `AUTH_REQUIRED` | Sign-in needed (publish etc.) | Device auth in Setup |
| `AUTH_FAILED` | Bad/expired token | Re-login |
| `UPDATE_CHECK_FAILED` | Version check unreachable/invalid | Check network; LocalCode works without updating |
| `UPDATE_FAILED` | Self-update fetch/build/swap failed | Causes in error; run `localcode update` for full log |
| `IO_ERROR` | Filesystem/process I/O | Permissions, disk |
| `INTERNAL` | Unexpected internal error | Logs + Ask assistant |
| `CANCELLED` | User cancelled | — |
| `NOT_IMPLEMENTED` | Stub path | Upgrade / contribute |

Policy: **never hard-block** deploy solely because `DEPLOY_OVERSIZED_WARNING`; always allow Continue.
