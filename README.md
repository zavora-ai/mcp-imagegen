# Image Generation MCP Server (mcp-imagegen)

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.94.1%2B-orange.svg)](https://www.rust-lang.org/)
[![MCP](https://img.shields.io/badge/MCP-2025--11--25%20%7C%202026--07--28-green.svg)](https://modelcontextprotocol.io/)
[![ADK-Rust Enterprise](https://img.shields.io/badge/ADK--Rust-Enterprise-purple.svg)](https://enterprise.adk-rust.com)
[![Registry Ready](https://img.shields.io/badge/ADK_Registry-Ready-green.svg)](https://www.zavora.ai)

Local, private image **generation and editing** over MCP. It runs open-weight models on your own machine
through [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (Metal, CUDA, Vulkan or CPU),
behind a pluggable backend interface. It was built for agents making game assets on a workstation shared with
Blender and Unreal Engine: one job at a time, a memory check before loading, and no surprises on disk.

The current model family is **[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)** (7B DiT, native 2K,
strong text rendering), for text-to-image and instruction-based editing with up to 3 reference images. The default
preset adds Viggle's 6-step turbo LoRA. It runs on **macOS, Linux and Windows**, on Metal, CUDA, Vulkan or CPU.

Optionally, **cloud models** run beside the local ones through the [Codex CLI](https://github.com/openai/codex)'s
built-in image tool (OpenAI's image model; a ChatGPT sign-in is enough, no API key). They use no local memory, take about
a minute, and are marked `runs: "cloud"` so agents know the prompt leaves the machine. See [Cloud models](#cloud-models-codex-cli).

## Example outputs

<table>
<tr>
<td align="center"><strong>generate_image</strong></td>
<td align="center"><strong>edit_image</strong></td>
</tr>
<tr>
<td><img src="https://raw.githubusercontent.com/zavora-ai/mcp-imagegen/main/docs/assets/crate_generated.png" width="360" alt="Generated wooden crate"/></td>
<td><img src="https://raw.githubusercontent.com/zavora-ai/mcp-imagegen/main/docs/assets/crate_edited.png" width="360" alt="Crate edited to red"/></td>
</tr>
<tr>
<td><sub>"a weathered wooden supply crate with iron corner brackets, studio lighting, game asset concept". Qwen-Image-2.1 fast preset, 1024×1024, seed 1234</sub></td>
<td><sub>edit of the image on the left: "paint the crate bright red, keep the dark iron corner brackets and the wood grain", seed 4321</sub></td>
</tr>
</table>

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/zavora-ai/mcp-imagegen/main/docs/assets/architecture.svg" alt="mcp-imagegen architecture" width="850"/>
</p>

- **Backends** implement one trait (`availability`, `generate` with progress and cancel, `unload`, and where they `run`). Each is
  behind a Cargo feature: `sdcpp` (default) drives an `sd-server` child process through its native async job API, `codex`
  (default) runs `codex exec` per job for cloud generation, and `mock` is for tests.
- **Model registry** (`models.toml`): adding a model on an existing backend is config only. Weights resolve from the
  Hugging Face cache by `hf_repo` + `hf_file` (newest snapshot) or an explicit `path`. The server **never downloads**.
  `list_models` prints the exact `hf download` command and size for anything missing.
- **Job manager**: two lanes (local and cloud), each with one worker and a bounded FIFO queue, so a cloud image never waits
  behind a local render. Cancel for queued or running jobs, result TTL, and idle unload of local models.
- **Safety**: output and input paths are jailed to configured roots. Writes are atomic (`.tmp` + rename) and never overwrite.
  Inputs are format-sniffed, size-limited and fingerprinted (SHA-256). A free-memory check refuses to load a local model that
  would push the machine into swap. An orphaned `sd-server` from a crashed run is reaped at startup.

## Tools (6)

| Tool | Purpose | Risk |
|------|---------|------|
| `generate_image` | Text-to-image. Returns path, seed, parameters, timings and an optional inline preview. MCP Task on 2026-07-28 clients | internal_write |
| `edit_image` | Edit 1–N existing images from an instruction (recolour, add/remove details, variations, combine references) | internal_write |
| `get_job` | Status, progress (`loading model`, `sampling 5/20`, `decoding`) and result of a job | read_only |
| `cancel_job` | Cancel a queued or running job. No partial files are left behind | internal_write |
| `list_models` | Readiness (`ready` / `missing_files` + fetch command / `backend_unavailable`), edit readiness, where it runs (`local` / `cloud` + provider), license, commercial flags | read_only |
| `unload_models` | Stop local backend processes and free memory now (cloud jobs don't block it) | internal_write |

Long jobs: `generate_image` and `edit_image` run as protocol-native **MCP Tasks** for clients that support them, block
for legacy clients (Claude Code backgrounds calls over two minutes automatically), and accept `wait: false`, which
returns a `job_id` to poll with `get_job`.

Every image gets a sidecar JSON with the model, backend, where it ran, license, prompt, seed, the image's real size, steps, CFG,
sampler, backend options, timings, backend provenance (for Codex: version, agent model, thread id, source file, token usage), and
for edits, each reference's path, size and SHA-256. Same seed + same inputs + same local backend gives identical pixels (verified
by the live tests). Cloud models have no seed (`"seed": null`).

## Install

```bash
git clone https://github.com/zavora-ai/mcp-imagegen && cd mcp-imagegen
cargo install --path .
```

Build the engine, [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (`sd-server`). Qwen-Image-2.1 needs a build
from `2f88688` (2026-09-26) or later. Pick the GPU backend for your machine:

| Platform | Backend flag |
|---|---|
| macOS (Apple Silicon) | `-DSD_METAL=ON` |
| Linux / Windows, NVIDIA | `-DSD_CUDA=ON` |
| Linux / Windows, AMD / Intel / other | `-DSD_VULKAN=ON` |
| CPU only | *(no flag)* |

```bash
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cmake -S stable-diffusion.cpp -B stable-diffusion.cpp/build -DCMAKE_BUILD_TYPE=Release -DSD_METAL=ON   # your flag
cmake --build stable-diffusion.cpp/build --config Release -j
```

The server binary ends up in `build/bin/` (`sd-server`, or `sd-server.exe` on Windows, often under `build/bin/Release/`).
Prebuilt binaries are also on the stable-diffusion.cpp releases page.

Fetch the weights with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`pip install -U huggingface_hub`).
That's about 13.4 GB, plus 0.7 GB for the turbo LoRA and 1.2 GB for editing:

```bash
hf download leejet/Qwen-Image-2.1-GGUF qwen_image_2.1-Q8_0.gguf
hf download Qwen/Qwen3-VL-8B-Instruct-GGUF Qwen3VL-8B-Instruct-Q4_K_M.gguf
hf download Comfy-Org/Qwen-Image-2.1 vae/qwen_image_2.1_vae_bf16.safetensors
hf download Viggle/Qwen-Image-2.1-viggle-turbo Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors   # turbo preset
hf download Qwen/Qwen3-VL-8B-Instruct-GGUF mmproj-Qwen3VL-8B-Instruct-F16.gguf   # editing only
```

`list_models` always shows exactly which files a model still needs, with the command to fetch each.

## Configure

```json
{
  "mcpServers": {
    "imagegen": {
      "type": "stdio",
      "command": "/Users/you/.cargo/bin/mcp-imagegen",
      "env": { "MCP_IMAGEGEN_CONFIG_DIR": "/path/to/project/.imagegen" }
    }
  }
}
```

On first run the server writes `config.toml` and `models.toml` to `$MCP_IMAGEGEN_CONFIG_DIR`, or by default
`%APPDATA%\mcp-imagegen` on Windows and `$XDG_CONFIG_HOME/mcp-imagegen` (else `~/.config/mcp-imagegen`) on macOS and Linux.
Paths may use `~`. Windows paths work with forward slashes or escaped backslashes in TOML. See [`config.example.toml`](config.example.toml) and [`models.example.toml`](models.example.toml).
At minimum, set `backends.sdcpp.server_bin` and your output roots:

```toml
default_output_dir = "~/Projects/game/assets/generated"
allowed_output_roots = ["~/Projects/game/assets"]   # edit inputs default to the same roots
memory_headroom_mb = 2048

[backends.sdcpp]
server_bin = "~/stable-diffusion.cpp/build/bin/sd-server"   # or just "sd-server" if it's on PATH; ".exe" is found automatically
```

## Models

| id | Use | Defaults | License |
|---|---|---|---|
| `qwen-image-2.1-turbo` | **Default.** Fastest and sharpest | [Viggle turbo LoRA](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo): 6 fixed steps, cfg 1.0, resolution-shifted sigma schedule | Qwen Research License |
| `qwen-image-2.1` | No LoRA | 20 steps, cfg 1.0, EasyCache step caching | Qwen Research License |
| `qwen-image-2.1-hq` | Reference quality | 40 steps (official), cfg 1.0, no caching | Qwen Research License |
| `codex-image` | Cloud, fast, strong prompt following | Codex agent `gpt-5.6-terra`, reasoning effort low | OpenAI terms (outputs assigned to the user) |
| `codex-image-astra` | Cloud, same image tool | Codex agent `gpt-6-astra` | OpenAI terms |

The Qwen presets support `edit_image` with up to 3 references once the vision weights are present; the Codex ones always do. Models can declare `loras` and a
`sigma_schedule` in `models.toml`. Because stable-diffusion.cpp uses custom sigmas verbatim, the server applies the model's
resolution-dependent time shift itself.

**License note.** Qwen-Image-2.1's weights are under the Qwen Research License (non-commercial). The Qwen team has stated
that model outputs are not part of the licensed Materials and that users keep the rights to what they generate. Check the
[model card](https://huggingface.co/Qwen/Qwen-Image-2.1) for the current terms before shipping generated assets.
`list_models` reports `commercial_weights` for every local model (and `commercial_outputs` where the terms speak about outputs) so agents can check.

## Cloud models (Codex CLI)

Install the [Codex CLI](https://github.com/openai/codex) and sign in once (`codex login`; a ChatGPT account works without an API
key). `list_models` then shows the `codex-*` models as `ready`; it checks with `codex login status`, which spends no quota.

```toml
# config.toml (defaults shown)
[backends.codex]
bin = "codex"            # codex.cmd / codex.exe are found on Windows
# codex_home = "~/.codex" # default: $CODEX_HOME, else ~/.codex
timeout_secs = 600
```

How a job runs: `codex exec --json --skip-git-repo-check -s read-only -m <agent model>` with a fixed instruction on stdin
("use your image generation tool exactly once… don't run commands or touch files") and, for edits, the reference images attached
with `--image`. The image is read from `<codex_home>/generated_images/<thread_id>/`, where the thread id comes from Codex's own
event stream, never from the agent's reply. Cancelling kills the whole Codex process tree.

- **Privacy:** prompts and reference images go to OpenAI. `list_models` says so per model (`provider`, `privacy`).
- **Cost:** it spends the signed-in account's quota. A 1536×1024 image took **45 s** and about 33k input tokens (19k cached) with
  `gpt-5.6-terra` on Codex 0.153.
- **Sizes:** the request picks square (1024×1024 requested, Codex returns 1254×1254), 2:3 portrait (1024×1536) or 3:2 landscape
  (1536×1024). The result reports the real size, with a warning when it differs.
- **Agent models:** with a ChatGPT sign-in on Codex 0.153, `gpt-6-astra` and the `gpt-5.6-*` models were accepted, and
  `gpt-6-luna`, `gpt-6-terra` and `gpt-6-sol` were rejected even though Codex lists them. The server always passes the
  registry's model, so your own Codex default doesn't matter.
- **No seed, steps or CFG:** passing them is allowed and ignored with a warning.

## Performance

Apple M4 (10-core GPU), 24 GB, stable-diffusion.cpp Metal, Q8_0 weights. Qwen-Image-2.1 turns out to be effectively
guidance-free: **cfg 1.0 halves the transformer passes with no visible loss**, and EasyCache skips about half the steps.

| Size | Preset | Sampling | Wall | Peak memory |
|---|---|---|---|---|
| 512² | cfg 6.0 (naive baseline) | 322 s | 344 s | 12.1 GB |
| 512² | cfg 1.0 | 167 s | 180 s | 12.1 GB |
| 512² | **cfg 1.0 + EasyCache** (`qwen-image-2.1`) | **82 s** | ~95 s | 12.1 GB |
| 1024² | `qwen-image-2.1` | 312 s | 379 s | 12.2 GB |
| 1024² | **`qwen-image-2.1-turbo`** (6 steps, runtime LoRA) | **232 s** | **298 s** | ~13.5 GB |
| 512² edit, 1 reference | `qwen-image-2.1-turbo` | 63 s | 71–75 s | — |
| 1024² edit, 1 reference | `qwen-image-2.1-turbo` | 577 s | 664 s | 14.5 GB |
| 512² edit, 1 reference | `qwen-image-2.1` | 75–85 s | 83–105 s | 13.3 GB |

Edits carry the reference image's tokens as well as the output's, so a 1024² edit costs several times a 1024² generation.
Edit at 512–768 px for iteration. Turbo is weaker on dense rendered text; use `qwen-image-2.1-hq` for that.
Details and methodology are in [`docs/benchmarks.md`](docs/benchmarks.md).

## Development

```bash
cargo test --all-features   # unit + protocol tests over stdio with the mock backend

# Live tests against real weights (ignored by default)
MCP_IMAGEGEN_SD_SERVER=/path/to/sd-server \
  cargo test --release --test sdcpp_live -- --ignored --nocapture

# Live Codex test (spends quota; needs `codex login`)
cargo test --test codex_live -- --ignored --nocapture
```

Codex behaviour in CI is covered by `fake-codex`, a stand-in binary built only with the `mock` feature.

CI runs fmt, clippy (`-D warnings`), tests on Linux, macOS and Windows, and an MSRV build.

## rmcp and MCP compatibility

This server is built with [`rmcp` 3.1.2](https://github.com/modelcontextprotocol/rust-sdk/releases/tag/rmcp-v3.1.2) and requires
Rust 1.94.1 or newer. It keeps legacy MCP initialization compatibility and targets MCP protocol revisions `2025-11-25` and `2026-07-28`.

## MCP 2026-07-28 rollout

This server uses `rmcp` 3.1.2 and `adk-mcp-sdk` 0.3 (pinned to `fd08c98` for `task_ttl_overrides` and task status updates).
It accepts stateless MCP 2026 requests with per-request protocol, client identity and capability metadata, and keeps the
legacy MCP 2025-11-25 initialize flow.

- **Tasks:** `generate_image`, `edit_image` (TTL 1 hour, status messages like `sampling 5/20`)
- **MRTR approvals:** none. Writes are confined to configured output roots (`writes_allowed = "gated"`).
- **Annotations:** `generate_image`, `edit_image`, `cancel_job` and `unload_models` are mutating, and `cancel_job` and `unload_models` are idempotent.
- **Caching:** `tools/list` returns a public `ttlMs` of 60,000 for MCP 2026 clients.
- **Transport:** stdio. Task records are process-local.

## License

Apache-2.0. Model weights have their own licenses (see [Models](#models)).
