"""Options, run loop and timing line shared by inference_single_gpu.py and inference_dual_gpu.py.

Kept out of omnimem.pipelines, whose import loads torch: run() binds to the GPU's NUMA node first.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
# omnimem's logger forces root handlers to ERROR; keep our progress on a private logger.
log = logging.getLogger("omnimem.streaming")
log.setLevel(logging.INFO)
log.propagate = False
_h = logging.StreamHandler(sys.stderr)
_h.setFormatter(logging.Formatter("%(asctime)s [streaming] %(message)s"))
log.addHandler(_h)


class _UnderscoreNegBooleanOptionalAction(argparse.BooleanOptionalAction):
    """Like ``argparse.BooleanOptionalAction`` but the negative flag is ``--no_foo``."""

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
        self.option_strings = [s.replace("--no-", "--no_", 1) if s.startswith("--no-") else s
                               for s in self.option_strings]

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith("--no_"))


def timing_summary(timings, frames_delivered: int) -> str:
    """One log line for a finished request; stages that did not happen are left out."""
    stages = (("first_preview_png_s", "first preview PNG at"),
              ("first_frames_delivered_s", "first frames delivered at"),
              ("denoise_done_s", "denoise done"), ("writer_done_s", "writer done"))
    parts = [f"{label} {timings[key]:.2f}s" for key, label in stages
             if timings.get(key) is not None]
    parts.append(f"total {timings['total_s']:.2f}s ({frames_delivered} frames, "
                 f"{timings['fps_total']:.2f} fps)")
    return ", ".join(parts)


def load_prompt_file(path: str) -> list:
    """Requests of a prompt file: (name, prompt, switch_prompts, switch_frame_indices) per line."""
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"{path} contains no prompts")
    if not path.endswith(".jsonl"):
        return [(f"prompt-{i:03d}", line, None, None) for i, line in enumerate(lines)]
    requests = []
    for i, record in enumerate(map(json.loads, lines)):
        name, prompts = record.get("id", f"prompt-{i:03d}"), record.get("prompts")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in {r[0] for r in requests}:
            raise ValueError(f"{path}: ids must be unique file names, got {name!r}")
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"{path}: {name} needs a nonempty prompts list")
        requests.append((name, prompts[0], prompts[1:] or None,
                         record.get("switch_frame_indices") or None))
    return requests


def build_parser(description: str) -> argparse.ArgumentParser:
    """Parser of the options both entry scripts take; each script adds its device options."""
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # assets
    p.add_argument("--pretrained", required=True)
    p.add_argument("--vae_path", required=True)
    p.add_argument("--t5_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    # request
    p.add_argument("--prompt", default=None,
                   help="text prompt (default: the cat/piano demo prompt)")
    p.add_argument("--prompt_file", default=None,
                   help="one video per line: a .txt of prompts, or a .jsonl of prompts and switch_frame_indices")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--nframes", type=int, default=969,
                   help="requested frames; actual = 1 + 4 * ((nframes - 1) // 4)")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--cfg", type=float, default=1.0, help="must be 1 for this entry point")
    p.add_argument("--num_inference_steps", type=int, default=4)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_prompt", default=None,
                   help="negative prompt (unused with CFG=1, kept for parity)")
    # causal / NSA runtime
    p.add_argument("--nsa_backend", default=None, choices=["triton", "cuda"],
                   help="kernel backend, triton or cuda (unset: $OMNIMEM_NSA_BACKEND, else triton)")
    p.add_argument("--frame_per_block", type=int, default=3)
    p.add_argument("--window_size", type=int, default=3)
    p.add_argument("--sink_size", type=int, default=None)
    p.add_argument("--enable_kv_evict", action=_UnderscoreNegBooleanOptionalAction, default=True,
                   help="evict KV history outside the window and sink from the DiT GPU")
    p.add_argument("--lru_max_size", type=int, default=20,
                   help="chunk LRU capacity; only with --kv_cache_policy chunk_lru")
    p.add_argument("--park_t5", default=None,
                   help="move the T5 encoder to this device after encoding (e.g. cuda:1 or cpu) to free DiT GPU memory")
    p.add_argument("--vae_fast", action=_UnderscoreNegBooleanOptionalAction, default=True,
                   help="use the faster VAE decode path; --no_vae_fast keeps the original decoder")
    p.add_argument("--patch_embed_linear", action="store_true",
                   help="use a linear patch embedding instead of Conv3d; results may differ slightly")
    p.add_argument("--no_compile", action="store_true",
                   help="run the DiT without torch.compile")
    p.add_argument("--compile_mode", default="default",
                   help="torch.compile mode of the DiT; modes other than 'default' autotune per process")
    # prompt switch
    p.add_argument("--switch_prompts", nargs="*", default=None,
                   help="prompts the video switches to, in order: one per entry of "
                        "--switch_frame_indices")
    p.add_argument("--switch_frame_indices", nargs="*", type=int, default=None,
                   help="latent frame indices of the prompt switches; increasing, multiples of --frame_per_block")
    p.add_argument("--slice_last_frames", type=int, default=21,
                   help="latent frames before a prompt switch whose KV history is recomputed under the new prompt")
    p.add_argument("--recache_mode", default="full", choices=["full", "window_only", "none"],
                   help="KV history handling at a prompt switch; window_only restores old KV outside the window, none keeps it")
    # streaming switches
    p.add_argument("--queue_depth", type=int, default=2,
                   help="capacity of the latent and frame queues, in chunks")
    p.add_argument("--kv_cache_policy", default="block_lru", choices=["block_lru", "chunk_lru"],
                   help="history cache policy: block_lru keeps selected KV blocks, chunk_lru keeps whole chunks")
    p.add_argument("--kv_block_pool_gb", type=float, default=None,
                   help="persistent block pool budget in GiB, must be > 0 (unset = 24); only with block_lru")
    p.add_argument("--kv_block_scratch_gb", type=float, default=None,
                   help="shared scratch block pool budget in GiB (unset = 0.75); only with block_lru")
    # Block store options: each only meaningful with --kv_cache_policy block_lru.
    p.add_argument("--kv_block_resident_if_fits", action=_UnderscoreNegBooleanOptionalAction,
                   default=True,
                   help="keep the whole KV history on the DiT GPU if it fits the pool budget; only with block_lru")
    p.add_argument("--kv_block_dedup", default="device", choices=["unique", "device"],
                   help="dedup of selected block keys, on the device or with torch.unique; only with block_lru")
    p.add_argument("--kv_window_buffer", action=_UnderscoreNegBooleanOptionalAction, default=True,
                   help="preallocate KV history and window buffers; faster but uses more GPU memory")
    p.add_argument("--no_numa_bind", action="store_true")
    # output
    p.add_argument("--output", required=True,
                   help="final MP4 path; with --prompt_file, the directory of the <id>.mp4 files")
    p.add_argument("--preview_dir", default=None,
                   help="directory for first-frame / periodic PNG previews")
    p.add_argument("--preview_every_frames", type=int, default=None)
    p.add_argument("--fragmented_mp4", action="store_true",
                   help="write a fragmented MP4, playable while generating (partial file: <output>.partial.<pid>.mp4)")
    p.add_argument("--metrics_json", default=None)
    p.add_argument("--repeat", type=int, default=1,
                   help="generate the request N times (repeats get a _rK suffix on output and preview paths)")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="medium")
    return p


def run(args, dit_device, decode_device=None, **tier) -> int:
    """Run the parsed request; a ``decode_device`` means two GPUs, ``tier`` are Recipe fields."""
    if not args.no_numa_bind:
        from omnimem.utils.numa_bind import bind_to_gpu_numa
        bind_to_gpu_numa(verbose=False)

    # Imported after the NUMA binding, so that the threads torch starts inherit it.
    from omnimem.pipelines.compile_state import filter_expected_dynamo_warnings
    from omnimem.pipelines.streaming_inference import (
        DEFAULT_PROMPT, NEG_PROMPT, ModelPaths, Recipe, StreamingRuntime, git_info,
        validate_request, actual_frames, check_output_target, check_layout)

    # Everything that can be refused without the weights is refused here, before the
    # ~80 s of loading: the layout, the request, the output target, the recipe.
    try:
        dit_device, decode_device = check_layout(dit_device, decode_device)
        if args.prompt_file is None:
            requests = [(None, args.prompt or DEFAULT_PROMPT, args.switch_prompts,
                         args.switch_frame_indices)]
        elif args.prompt or args.switch_prompts or args.switch_frame_indices:
            raise ValueError("--prompt_file replaces --prompt, --switch_prompts and "
                             "--switch_frame_indices")
        else:
            requests = load_prompt_file(args.prompt_file)
        for _, _, switch_prompts, switch_frame_indices in requests:
            validate_request(args.nframes, args.frame_per_block, args.cfg,
                             switch_prompts=switch_prompts,
                             switch_frame_indices=switch_frame_indices,
                             queue_depth=args.queue_depth)
        if args.repeat < 1:
            raise ValueError(f"--repeat must be >= 1, got {args.repeat}")
        outputs = [args.output if name is None else os.path.join(args.output, name + ".mp4")
                   for name, *_ in requests]
        for output in outputs:
            check_output_target(output, args.preview_dir)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    recipe = Recipe(height=args.height, width=args.width, fps=args.fps, cfg=args.cfg,
                    num_inference_steps=args.num_inference_steps, shift=args.shift,
                    frame_per_block=args.frame_per_block, window_size=args.window_size,
                    sink_size=args.sink_size, enable_kv_evict=args.enable_kv_evict,
                    lru_max_size=args.lru_max_size,
                    n_prompt=args.n_prompt or NEG_PROMPT, compile=not args.no_compile,
                    compile_mode=args.compile_mode, crf=args.crf, preset=args.preset,
                    slice_last_frames=args.slice_last_frames, recache_mode=args.recache_mode,
                    park_t5=args.park_t5,
                    kv_cache_policy=args.kv_cache_policy,
                    kv_block_pool_gb=args.kv_block_pool_gb,
                    kv_block_scratch_gb=args.kv_block_scratch_gb,
                    kv_block_resident_if_fits=args.kv_block_resident_if_fits,
                    kv_block_dedup=args.kv_block_dedup,
                    kv_window_buffer=args.kv_window_buffer,
                    vae_decode_fast=args.vae_fast,
                    patch_embed_linear=args.patch_embed_linear,
                    **tier)
    paths = ModelPaths(args.pretrained, args.vae_path, args.t5_path, args.tokenizer_path)
    if args.nsa_backend is not None:
        # Before ``load()``, which builds / loads the extensions of the active backend.
        from omnimem.nsa import backend as nsa_backend
        nsa_backend.set_backend(args.nsa_backend)
    # The runtime refuses a recipe it cannot run (the KV tier under block_lru, a block
    # pool budget under chunk_lru, ...) when it is built, before any weights are loaded.
    try:
        rt = StreamingRuntime(paths, recipe, dit_device=dit_device, decode_device=decode_device)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    log.info("layout: %s (DiT on %s, VAE decode on %s)",
             "one GPU" if rt.gpus == 1 else "two GPUs", rt.dit_device, rt.decode_device)
    # Compiled requests reach dynamo's recompile limit by design (the limit is pinned
    # and part of the reference behaviour); keep its two warnings off the log.
    filter_expected_dynamo_warnings()
    t_load0 = time.monotonic()
    try:
        load_info = rt.load()
    except ValueError as exc:       # the resolution does not fit the checkpoint
        print(f"error: {exc}", file=sys.stderr)
        return 2
    load_info["load_total_s"] = time.monotonic() - t_load0
    log.info("models loaded in %.1fs (%s)", load_info["load_total_s"],
                 {k: v for k, v in load_info.items() if k.endswith("_s")})

    results = []
    total = len(requests) * args.repeat
    for (name, prompt, switch_prompts, switch_frame_indices), output in zip(requests, outputs):
        for k in range(args.repeat):
            suffix = "" if k == 0 else f"_r{k}"
            root, ext = os.path.splitext(output)
            out_path = root + suffix + (ext or ".mp4")
            prev_dir = None
            if args.preview_dir:
                prev_dir = (os.path.join(args.preview_dir, name) if name else args.preview_dir) + suffix
            log.info("=== video %d/%d: %d requested frames (%d actual) -> %s", len(results) + 1,
                         total, args.nframes, actual_frames(args.nframes), out_path)
            res = rt.run(prompt, args.nframes, args.seed, output_path=out_path,
                         preview_dir=prev_dir, switch_prompts=switch_prompts,
                         switch_frame_indices=switch_frame_indices,
                         queue_depth=args.queue_depth, fragmented=args.fragmented_mp4,
                         preview_every_frames=args.preview_every_frames)
            log.info("%s", timing_summary(res.timings, res.frames_delivered))
            log.info("actual config: %s", res.config)
            results.append(res.to_json())

    if args.metrics_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_json)), exist_ok=True)
        with open(args.metrics_json, "w") as f:
            prompts = [prompt for _, prompt, _, _ in requests]
            json.dump({"args": vars(args), "prompt": prompts if args.prompt_file else prompts[0],
                       "recipe": recipe.__dict__,
                       "load": load_info,
                       "git": git_info(os.path.dirname(os.path.abspath(sys.argv[0]))),
                       "runs": results}, f, indent=1)
        log.info("wrote %s", args.metrics_json)
    return 0
