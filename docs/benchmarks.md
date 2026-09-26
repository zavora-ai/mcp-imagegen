# Benchmarks

Machine: Apple M4 (10-core GPU), 24 GB unified memory, macOS. stable-diffusion.cpp `2f88688` (Metal, `--fa`).

## Qwen-Image-2.1 (Q8_0 diffusion, Qwen3-VL-8B Q4_K_M text encoder, bf16 VAE)

| Size | Steps | CFG | Extra | Load | Sampling | Decode | Wall | Notes |
|---|---|---|---|---|---|---|---|---|
| 512×512 | 20 | 6.0 | none | 8.5 s | 321.6 s (16.1 s/step) | 6.3 s | 344 s | cold start, sd-server RSS ~12 GB |
| 512×512 | 20 | 6.0 | none | 0 | 309.7 s (15.5 s/step) | 5.4 s | 315 s | warm; same seed → identical pixels |

At 20 steps with CFG that's 40 transformer passes of a 7B model, about 30 TFLOP per step at 512². The M4 GPU is near its
practical limit, so speedups have to come from fewer passes (turbo distillation, CFG 1, step caching), not engine tuning.

### Speed-ups (512×512, seed 1234, 20 steps, sd-cli)

| Variant | Sampling | Speed-up | Quality |
|---|---|---|---|
| cfg 6.0 (baseline) | 321.6 s (16.1 s/step) | 1.0× | three-quarter view, good |
| cfg 1.0 | 167.0 s (7.8 s/step) | 1.9× | excellent, most detailed. The model is effectively guidance-free |
| cfg 6.0 + EasyCache | 153.1 s (11/20 steps skipped) | 2.1× | same composition as baseline, slightly soft |
| **cfg 1.0 + EasyCache** | **82.5 s** | **3.9×** | close to cfg 1.0, touch softer. **Registry default for `qwen-image-2.1`** |

Next: Viggle's `Qwen-Image-2.1-viggle-turbo` DMD LoRA (6 passes, custom sigmas, cfg 1) should be about 6–7× faster than the baseline.

### 1024×1024 through the MCP server (`qwen-image-2.1` fast preset: cfg 1.0, EasyCache, 20 steps)

| Load | Sampling | Decode | Wall | Peak RSS |
|---|---|---|---|---|
| 0.3 s (weights mapped at spawn) | 311.9 s | 59.3 s | 379 s | 12.15 GB |

VAE decode is now ~16% of the time at 1024². Next candidates: `--vae-tiling`, or a tiny autoencoder for previews.

### Editing (`qwen-image-2.1` fast preset, 1 reference = the 1024² crate, output 512×512)

| Run | Load | Sampling | Decode | Wall | Peak RSS |
|---|---|---|---|---|---|
| 1 (cold) | 0.3 s | 85.2 s | 6.4 s | 105 s | 13.3 GB (vision weights loaded) |
| 2 (warm) | 0 | 75.1 s | 5.3 s | 82.5 s | same seed + input → identical pixels |

### Turbo preset (`qwen-image-2.1-turbo`: Viggle v0.2.1 r128 LoRA at runtime, 6 fixed shifted sigmas, cfg 1)

| Run | Size | Sampling | Decode | Wall | Peak RSS | Notes |
|---|---|---|---|---|---|---|
| generate (sd-cli) | 1024² | 231.7 s (36 s/step) | 59.6 s | 298 s | — | vs 312 / 379 s for `qwen-image-2.1`, sharper |
| generate warm | 512² | 50.5 s | 5.4 s | 56 s | — | same seed → identical pixels |
| edit, 1 ref | 512² | 63 s | 5.2 s | 71–75 s | — | same seed + input → identical pixels |
| edit, 1 ref (MCP server) | 1024² | 576.7 s | 59.7 s | 664 s | 14.53 GB | the reference doubles the token count; edit at 512–768 for speed |

`--lora-apply-mode immediately` (merging the LoRA into Q8 weights) segfaults in sd.cpp `2f88688`, so runtime application costs about 16% per step.
An offline merge into bf16 weights + re-quantization would recover it (future work).
