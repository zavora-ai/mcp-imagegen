# Changelog

## 0.1.0 (unreleased)

- Tools: `generate_image`, `edit_image`, `get_job`, `cancel_job`, `list_models`, `unload_models`.
- stable-diffusion.cpp backend: `sd-server` child process, native async job API, progress and timings parsed from output,
  pinned LoRA and working directories, orphan reaping via PID file, clean shutdown on SIGTERM/SIGINT.
- Qwen-Image-2.1 presets: fast (cfg 1 + EasyCache, ~3.9× faster than cfg 6) and hq (40 steps). Instruction editing with
  up to 3 references (vision weights loaded when present).
- Model registry with Hugging Face cache resolution, license and commercial-use flags, per-model `backend_options`.
- Single-worker job queue with cancel, TTL and idle unload; memory pre-flight (macOS-correct available memory).
- Path-jailed atomic output with sidecar JSON; jailed, format-sniffed, fingerprinted edit inputs.
- MCP Tasks for long jobs via `adk-mcp-sdk` 0.3, with legacy blocking and `wait: false` fallbacks.
