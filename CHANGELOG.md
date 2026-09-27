# Changelog

## 0.2.0 (unreleased)

- **Codex cloud backend** (feature `codex`, on by default): `generate_image` and `edit_image` on OpenAI's image model through
  the Codex CLI's built-in image tool. Runs `codex exec --json` in a read-only sandbox with the instruction on stdin, finds the
  image by the run's thread id under `$CODEX_HOME/generated_images/`, re-encodes non-PNG output, and records the Codex version,
  agent model, thread id, source file and token usage in the sidecar. Works with a ChatGPT sign-in (no API key).
- Codex errors are specific: agent model not supported with this sign-in (points at `codex.model`), not signed in, usage limit,
  no image produced (with Codex's reply), timeout. Cancelling kills the whole Codex process tree (process group / `taskkill /T`).
- **Two job lanes**: one local and one cloud job can run at once. Cloud jobs skip the memory pre-flight, never unload local
  models, and don't block `unload_models`.
- Results and sidecars: `seed` is `null` for backends without seeds; `width`/`height` are the written image's real size (with a
  warning when it differs from the request); new `runs`, `commercial_outputs` and `backend_details`. `list_models` shows `runs`,
  `provider`, `privacy` and `agent_model`.
- Registry: `codex` model section, optional `commercial_outputs`; `commercial_weights` and `est_memory_mb` are now optional.
  Config: `[backends.codex]` (`bin`, `codex_home`, `timeout_secs`, `extra_args`). Example presets `codex-image` (gpt-5.6-terra)
  and `codex-image-astra`.
- Seed, steps, CFG and sampler passed to a Codex model are ignored with a warning.
- Tests: a fake Codex binary (`fake-codex`, built only with `mock`) drives integration and protocol tests on every platform;
  `tests/codex_live.rs` is an opt-in live test.

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
