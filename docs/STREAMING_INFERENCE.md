# Streaming inference

Details for `inference_single_gpu.py` and `inference_dual_gpu.py`: options, memory requirements, backends and reproducibility. The [README](../README.md#inference) has the minimal command.

## Usage

`inference_single_gpu.py` generates one long video (T2V, CFG = 1) with a plain `python` launch. Frames are decoded and appended to the MP4 while the DiT is still generating. Only the model paths and the output file are required:

```bash
model_path=checkpoints/long_rolling_plain_from_ode_init
wan_path=wan_models/Wan2.1-T2V-1.3B

python inference_single_gpu.py \
    --pretrained $model_path \
    --vae_path wan_models/vae-2.1/vae \
    --t5_path $wan_path/models_t5_umt5-xxl-enc-bf16.pth \
    --tokenizer_path $wan_path/google/umt5-xxl \
    --output demo/example.mp4
```

`inference_dual_gpu.py` takes the same arguments; the two scripts differ in where the VAE runs:

| Script | Layout |
|--------|--------|
| `inference_single_gpu.py` | DiT and Wan VAE share one GPU (`--device`, default `cuda:0`); each chunk is decoded right after it is denoised |
| `inference_dual_gpu.py` | DiT on `--dit_device` (default `cuda:0`); the VAE decodes on `--decode_device` (default `cuda:1`) while the DiT generates the next chunk. Needs two visible GPUs (e.g. `CUDA_VISIBLE_DEVICES=0,1`) |

Defaults: a demo prompt, `--nframes 969` (about 60 s at 16 fps), 480x832, 4 sampling steps, `--seed 42`. Set `--prompt`, `--nframes` (the number of latent frames, `(nframes - 1) // 4 + 1`, must be a multiple of 3: 81, 249, 969, ...) and `--seed` as needed; `--switch_prompts` / `--switch_frame_indices` change the prompt mid-video (one latent-frame index per switch prompt: increasing, inside the video and multiples of `--frame_per_block`). The video is written with the `ffmpeg` executable (libx264), which must be on `PATH`.

`--prompt_file` replaces `--prompt` and the switch options and generates one video per line of the file into the `--output` directory: a `.txt` holds one prompt per line, a `.jsonl` one `{"id", "prompts", "switch_frame_indices"}` record per line, written as `<id>.mp4`. The README uses `configs/prompts/demo.txt` for short videos, `configs/prompts/project_page/standard_60s.jsonl` for long videos and `configs/prompts/interactive_60s.jsonl` for long videos with prompt switching.

**KV history cache.** NSA attends to a sliding window of recent chunks plus the history blocks it selects. `--kv_cache_policy` decides where that history is kept:

| Policy | On the DiT GPU | Everything else |
|--------|----------------|-----------------|
| `block_lru` (default) | A pool of the selected KV *blocks* (`--kv_block_pool_gb`, default 24) plus a scratch pool (`--kv_block_scratch_gb`, default 0.75) | Pinned host memory; a kernel gathers the missing blocks into the pool. A history that fits the two budgets stays on the GPU entirely |
| `chunk_lru` | The `--lru_max_size` (default 20) most recently selected *chunks* | Pinned host memory, reloaded one whole chunk at a time — or, with `--kv_tier`, the second GPU |

Keep `block_lru` unless you want the GPU tier: `--kv_cache_policy chunk_lru --kv_tier` (`inference_dual_gpu.py` only) moves evicted chunks to the decode GPU, where attention reads them in place, so no history goes through host memory. It needs peer-to-peer access between the two GPUs and most of an 80 GiB decode GPU for a 960-frame video: the tier then holds about 55 GiB of history next to the decoder, about 67 GiB allocated (73 GiB reserved) at peak (`--kv_tier_budget_gb`, default 60; evictions past the budget fall back to pinned host memory). On a single GPU, long videos under `chunk_lru` need `--park_t5 cpu`.

**Cost of the default command** (measured on H100 80 GB):

- *Host memory.* The KV history takes about 0.8 GiB per chunk of 3 latent frames: about 64 GiB for 960 frames (80 chunks), 17 GiB for 249. Unless the whole history stays on the GPU, both policies back it with page-locked (pinned) host memory sized for the whole video, and the runtime locks exactly that much: a 960-frame process has a resident set of about 67 GiB. The history stays on the GPU, and no host memory is used, under `block_lru` when it fits the pool and scratch budgets (up to 30 chunks, or 357 frames, with the defaults, which includes a 249-frame video) and with the GPU tier while it fits the chunk LRU plus the tier budget.
- *First frames.* The host memory is page-locked on the first request of a process, which takes about half a minute for 960 frames, and is reused by later requests of the same length; a request of another length releases it and page-locks what it needs. The first frames of a cold 960-frame run therefore appear after roughly 55 s (about 25 s when nothing has to be page-locked; about 15 s of either is kernel compilation, which later processes find cached on disk), while later requests in the same process deliver theirs within 1-3 s. `OMNIMEM_EXACT_PINNED_KV=0` takes the host memory from PyTorch's pinned allocator instead, which rounds every buffer up to a power of two (about 120 GiB locked and 45 s more before the first frames at 960 frames) but never gives it back, so that a process alternating between video lengths page-locks each of them only once.
- *Peak GPU memory.* 960 frames: about 60 GiB allocated (66 GiB reserved) on a single GPU; with two, about 49 GiB on the DiT GPU plus about 17 GiB reserved on the decode GPU. 249 frames: about 47 GiB (52 GiB reserved) on one GPU, about 36 GiB + 17 GiB on two.

**Smaller machines.** Lower `--kv_block_pool_gb` (a 12 GiB pool was verified at 960 frames; a 16 GiB pool is 1.5-4% slower than 24), pass `--no_kv_window_buffer` (saves about 5 GiB on the DiT GPU, noticeably slower) or `--park_t5 cpu` (moves the T5 encoder off the DiT GPU once the prompts are encoded: frees about 11 GB there and costs about 10 s of first-frame latency per request; `cuda:1` works too), or generate a shorter video. A smaller pool does not reduce the host memory, which is set by the video length; the GPU tier avoids it. Shortening the video aside, none of these changes the sampled latents.

**Other switches.** The fast VAE decode is on by default; `--no_vae_fast` restores the original decoder. `--patch_embed_linear` (off by default) is a faster patch embedding that is not guaranteed to reproduce the default latents. The DiT is compiled with `torch.compile` in mode `default`; `--compile_mode` and `--no_compile` change that (see [Reproducibility](#reproducibility)). `--preview_dir` writes PNG previews, `--metrics_json` the timings and the resolved configuration; see `--help` of either script for the rest.

## NSA backend (Triton / CUDA)

The NSA kernels (top-k selection, selection attention, pointer table) and the block-KV cache kernels run on Triton (installed with PyTorch) by default. A native CUDA extension can be selected per process with `OMNIMEM_NSA_BACKEND=cuda` (all entry points, training included) or `--nsa_backend cuda` (the two streaming scripts):

```bash
python -m omnimem.nsa.cuda_backend.build      # once; needs the CUDA toolkit (nvcc) and ninja

OMNIMEM_NSA_BACKEND=cuda python inference_single_gpu.py \
    --pretrained $model_path \
    --vae_path wan_models/vae-2.1/vae \
    --t5_path $wan_path/models_t5_umt5-xxl-enc-bf16.pth \
    --tokenizer_path $wan_path/google/umt5-xxl \
    --output demo/example_cuda.mp4
```

The build targets the visible GPUs (`TORCH_CUDA_ARCH_LIST` overrides, sm80 and later) and honours `MAX_JOBS` and `TORCH_EXTENSIONS_DIR`; without the explicit build step the extension is compiled on first use. A missing toolchain, a compile error or an unsupported input raises `omnimem.nsa.backend.NSABackendError` — there is no silent fallback to Triton. The switch only changes which kernels run: checkpoints, configs and the training recipe are the same.

## Environment variables

Each is read once per process:

| Variable | Default | Effect |
|----------|---------|--------|
| `OMNIMEM_NSA_BACKEND` | `triton` | NSA kernel backend of training and inference, see above |
| `OMNIMEM_FLASH_ATTN_VERSION` | `2` | FlashAttention build of training and inference. `auto` takes the newest one installed (4, then 3, then 2); `3` and `4` request that build and fail when it is unavailable. FlashAttention 2 is what all validation used and the only one that runs on pre-Hopper GPUs; without it the default behaves like `auto` |
| `OMNIMEM_FLEX_COMPILE_MODE` | `max-autotune-no-cudagraphs` | `torch.compile` mode of the FlexAttention kernel behind the block-causal training mask (ODE, teacher-forcing and consistency-distillation stages); inference does not use it |

## Reproducibility

The statements below were verified on H100 (sm90) with the Triton and with the CUDA backend, on PyTorch 2.9.1 and again, for the compile settings and the reference scripts, on 2.11.0. Outputs differ across GPU architectures (e.g. A100 vs H100) and PyTorch versions.

- **Layouts and switches.** For a given request, GPU architecture, backend and software stack, every layout, cache policy and switch above, except `--patch_embed_linear` and the compile settings, produces the same latents bit for bit. They match `CausalWanT2VPipeline.generate()` called under bf16 autocast, as training validation, `inference_mp.py` and `scripts/evaluate_project_page.py` call it, run under the same compile settings.
- **Compile settings.** `--no_compile` or a `--compile_mode` other than `default` gives a different, equally valid sample. With the compiled default, an occasional process also gives a different, equally valid sample, so use `--no_compile` when a video must be reproduced exactly. With the Triton backend, `--no_compile` reproduces the videos of the uncompiled reference scripts (training validation, `scripts/evaluate_project_page.py`) for the same seed and frame count. `inference_wan_causal.py` and `inference_mp.py` compile with `max-autotune-no-cudagraphs` and are not bit-reproducible from run to run.
- **Across processes.** Reproducibility across processes relies on pinned kernel launch configurations. For the Triton top-k kernel they are pinned on Hopper only: on other architectures that kernel is still autotuned per process, so cross-process bit-reproducibility is not established there.
- **Fast VAE decode.** It leaves the latents unchanged; its pixels differ from the original decoder's at about 57-58 dB PSNR on H100. `--no_vae_fast` restores the original decoder and bit-identical frames.
- **Backends.** The Triton and CUDA backends are each self-consistent, but they produce different samples from each other.

**Several requests in one process.** `torch.compile` keeps its state per process, so in a process of the streaming runtime the pipeline records what the compiled DiT is specialised on (resolution, `frame_per_block`, window and sink sizes, the patch-embedding switch, the DiT device, streaming versus a direct `generate()` call) and resets `torch.compile` before a request that differs in any of it. Every request therefore produces what it produces in a fresh process, at the cost of a recompile of a few seconds; requests that differ only in prompt, seed, length or cache settings reuse the compiled code.

Differences from the previously published code:

- Prompt switching with `recache_mode='window_only'` (`--recache_mode window_only`) re-encodes the recent history with the new prompt and then restores the cached KV of the chunks outside the attention window, so that only the window is recached. That restore now really takes effect: earlier it was a no-op for the per-head chunk storage NSA uses, so results in that mode differ from before. The default `full` mode is unaffected.
- Kernel launch configurations that used to be autotuned per process are pinned (see above), so a run of the old code that happened to pick another configuration is not bit-identical to the new code.
- FlashAttention 2 is the default where the old code took the newest build installed (`OMNIMEM_FLASH_ATTN_VERSION=auto` selects that again).
- `--kv_evict_min_gpu_chunks` is gone from `inference_wan_causal.py` and `inference_mp.py`: it was accepted but never used.

## Known limitations of the Triton kernels

The shipped configurations do not reach any of these; they matter when the operators of `omnimem.nsa.ops` are called directly.

1. The non-causal token forward (`group_size == 1`, `causal=False`) reads out of bounds for `-1` ids.
2. In the `'mask'` backward an invalid slot can clear block 0's bit in a row that also selects block 0, which drops that contribution (`'auto'` picks `'mask'` for small problems).
3. Block ids must be unique within a row: the forward and the `'inverted'` backward count every occurrence, `'mask'` counts a block once per row. Top-k output is always unique.
4. The dense backward kernels index `d_out` with the strides of `out`, so a `d_out` laid out differently gives wrong gradients. The pointer (KV cache) backward uses `d_out`'s own strides.
5. With more than 64 selected blocks the chunked scoring path ignores `q_chunk_offset` in its causal mask.
6. The tiled top-k masks with -1e6 instead of -inf, so a valid block that scores below that ranks after the invalid slots.
7. The token path needs `block_size >= 9`.
