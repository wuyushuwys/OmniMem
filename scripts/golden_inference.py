#!/usr/bin/env python
"""Bit-exact inference goldens for the streaming runtime.

Sub-commands:
  record   run one request and save golden.pt + manifest.json in --golden_dir
  compare  rerun the request with the current code against golden.pt (exit 1 on mismatch)
  diff     compare two saved goldens offline (exit 1 on mismatch)

Example:
  python scripts/golden_inference.py record --golden_dir g --pretrained P --vae_path V \\
    --t5_path T --tokenizer_path K --prompt 'A cat.'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from omnimem.utils.streaming_cli import _UnderscoreNegBooleanOptionalAction


# The model assets of a run: required arguments of ``record`` / ``compare``.
ASSET_ARGS = ("pretrained", "vae_path", "t5_path", "tokenizer_path")
# Recipe fields that cannot change a golden payload (negative prompt, MP4 encoder settings).
RECIPE_FIELDS_OUTSIDE_A_GOLDEN = ("crf", "preset", "pix_fmt")
DEFAULT_SWITCH_PROMPT = ("The same scene at dawn, warm golden light pouring in, the music slowing to "
                         "a gentle finale.")
# request_spec keys compare checks against the golden's; a difference is warned about.
SPEC_KEYS_COMPARED = ("gpus", "kv_tier", "nframes", "seed", "lru_max_size", "compile",
                      "queue_depth", "recache_mode", "kv_cache_policy", "kv_block_pool_gb",
                      "kv_block_scratch_gb", "kv_block_resident_if_fits", "kv_block_dedup")


# ----------------------------------------------------------------------------- helpers
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(repo: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _env_info() -> Dict[str, Any]:
    import triton
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("TRITON", "TORCHINDUCTOR", "OMNIMEM", "PYTORCH_CUDA", "CUBLAS", "NVIDIA_TF32"))},
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "capability": [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())],
        "git_head": _git_head(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    }
    try:
        import omnimem
        info["omnimem_import_path"] = os.path.dirname(os.path.abspath(omnimem.__file__))
    except Exception:  # noqa: BLE001
        pass
    # The block-KV cache CUDA extension's library path (None unless it was loaded).
    from omnimem.nsa.cuda_backend import _block_kv_loader
    info["block_kv_cuda_library"] = _block_kv_loader.library_path()
    return info


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return {"tensor_shape": list(obj.shape), "dtype": str(obj.dtype)}
    return str(obj)


# ----------------------------------------------------------------------------- runtime
def recipe_kwargs(args) -> Dict[str, Any]:
    """The ``Recipe`` of this run, every field set from a switch of this tool."""
    return dict(
        height=args.height, width=args.width, fps=args.fps,
        num_inference_steps=args.num_inference_steps, shift=args.shift,
        frame_per_block=args.frame_per_block, window_size=args.window_size,
        sink_size=args.sink_size, enable_kv_evict=args.enable_kv_evict,
        lru_max_size=args.lru_max_size,
        compile=not args.no_compile, compile_mode=args.compile_mode,
        patch_embed_linear=args.patch_embed_linear,
        slice_last_frames=args.slice_last_frames, recache_mode=args.recache_mode,
        park_t5=args.park_t5,
        kv_cache_policy=args.kv_cache_policy, kv_block_pool_gb=args.kv_block_pool_gb,
        kv_block_scratch_gb=args.kv_block_scratch_gb,
        kv_block_resident_if_fits=args.kv_block_resident_if_fits,
        kv_block_dedup=args.kv_block_dedup,
        kv_tier=args.kv_tier, kv_tier_budget_gb=args.kv_tier_budget_gb,
        kv_window_buffer=args.kv_window_buffer,
        vae_decode_fast=args.vae_decode_fast,
    )


def build_recipe(args):
    """``Recipe(**recipe_kwargs(args))``, refusing a ``Recipe`` this tool does not cover."""
    import dataclasses
    from omnimem.pipelines.streaming_inference import Recipe
    kwargs = recipe_kwargs(args)
    fields = {f.name for f in dataclasses.fields(Recipe)}
    unset = sorted(fields - set(kwargs) - set(RECIPE_FIELDS_OUTSIDE_A_GOLDEN))
    if unset:
        raise RuntimeError(
            f"golden_inference.py does not set Recipe field(s) {unset}: add a switch for "
            f"them (or list them in RECIPE_FIELDS_OUTSIDE_A_GOLDEN if they cannot change "
            f"a golden) instead of inheriting the Recipe default")
    return Recipe(**kwargs)


def resolve_layout(args) -> None:
    """Resolve ``args.dit_device`` / ``args.decode_device`` to indexed devices."""
    from omnimem.utils.torch_utils import indexed_device
    from omnimem.pipelines.streaming_inference import check_layout

    try:
        one_gpu = (args.decode_device is None
                   or indexed_device(args.decode_device) == indexed_device(args.dit_device))
        dit, decode = check_layout(args.dit_device, None if one_gpu else args.decode_device)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}")
    args.dit_device, args.decode_device = str(dit), str(decode)


def make_runtime(args):
    """The runtime for ``args``, weights not loaded yet; exits on an unsupported request."""
    from omnimem.nsa import backend as nsa_backend
    from omnimem.pipelines.compile_state import filter_expected_dynamo_warnings
    from omnimem.pipelines.streaming_inference import (
        ModelPaths, StreamingRuntime, validate_request)

    resolve_layout(args)
    nsa_backend.set_backend(args.nsa_backend)
    recipe = build_recipe(args)
    paths = ModelPaths(args.pretrained, args.vae_path, args.t5_path, args.tokenizer_path)
    spec = request_spec(args)
    try:
        rt = StreamingRuntime(paths, recipe, dit_device=args.dit_device,
                              decode_device=args.decode_device)
        validate_request(spec["nframes"], recipe.frame_per_block,
                         switch_prompts=spec["switch_prompts"],
                         switch_frame_indices=spec["switch_frame_indices"],
                         queue_depth=spec["queue_depth"])
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    # The recompile limit is reached by design; keep dynamo's two warnings off the log.
    filter_expected_dynamo_warnings()
    return rt


def load_runtime(rt):
    t0 = time.monotonic()
    load_info = rt.load()
    return load_info, time.monotonic() - t0


def build_runtime(args):
    rt = make_runtime(args)
    load_info, load_s = load_runtime(rt)
    return rt, load_info, load_s


def request_spec(args) -> Dict[str, Any]:
    from omnimem.pipelines.streaming_inference import latent_frames
    from omnimem.pipelines.streaming_inference import DEFAULT_PROMPT
    prompt = args.prompt or DEFAULT_PROMPT
    switch_prompts = switch_idx = None
    if args.switch_prompt:
        sp = DEFAULT_SWITCH_PROMPT if args.switch_prompt == "default" else args.switch_prompt
        if args.switch_latent_frame is not None:
            sf = int(args.switch_latent_frame)
        else:
            n_lat = latent_frames(args.nframes)
            fpb = int(args.frame_per_block)
            sf = max(fpb, (n_lat // 2) // fpb * fpb)
        switch_prompts, switch_idx = [sp], [sf]
    return dict(gpus=1 if args.decode_device == args.dit_device else 2,
                kv_tier=bool(args.kv_tier),
                prompt=prompt, nframes=int(args.nframes), seed=int(args.seed),
                switch_prompts=switch_prompts, switch_frame_indices=switch_idx,
                trace=not args.no_trace, lru_max_size=int(args.lru_max_size),
                compile=not args.no_compile, queue_depth=int(args.queue_depth),
                # Recorded so compare can warn when history or block-store settings differ.
                recache_mode=args.recache_mode,
                kv_cache_policy=args.kv_cache_policy,
                kv_block_pool_gb=args.kv_block_pool_gb,
                kv_block_scratch_gb=args.kv_block_scratch_gb,
                kv_block_resident_if_fits=args.kv_block_resident_if_fits,
                kv_block_dedup=args.kv_block_dedup)


def run_once(rt, spec: Dict[str, Any], work_dir: Path, tag: str):
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{tag}.mp4"
    t0 = time.monotonic()
    res = rt.run(spec["prompt"], spec["nframes"], spec["seed"],
                 output_path=str(out), preview_dir=None,
                 switch_prompts=spec["switch_prompts"], switch_frame_indices=spec["switch_frame_indices"],
                 queue_depth=spec["queue_depth"], capture=True, trace_selection=spec["trace"])
    wall = time.monotonic() - t0
    try:
        out.unlink()
    except OSError:
        pass
    return res, wall


def payload_from_result(res) -> Dict[str, Any]:
    trace = None
    if res.selection_trace is not None:
        trace = []
        for e in res.selection_trace:
            layer, ids = int(e[0]), e[1].contiguous()
            tags = e[2] if len(e) > 2 else None
            trace.append((layer, ids, tags))
    return dict(
        latents=res.latents.contiguous(),
        frames=torch.from_numpy(np.ascontiguousarray(res.frames)),
        selection_trace=trace,
        generator_state_hash=res.generator_state_hash,
        frames_delivered=int(res.frames_delivered),
        actual_frames=int(res.actual_frames),
    )


# ----------------------------------------------------------------------------- compare
def _compare_traces(a, b) -> Dict[str, Any]:
    if len(a) != len(b):
        return {"exact": False, "len_a": len(a), "len_b": len(b)}
    tags_compared = 0
    for i, (ea, eb) in enumerate(zip(a, b)):
        la, ta, ga = ea[0], ea[1], (ea[2] if len(ea) > 2 else None)
        lb, tb, gb = eb[0], eb[1], (eb[2] if len(eb) > 2 else None)
        if la != lb:
            return {"exact": False, "first_mismatch_call": i, "layer_a": la, "layer_b": lb, "calls": len(a)}
        if ta.shape != tb.shape or ta.dtype != tb.dtype or not torch.equal(ta, tb):
            detail = {}
            if ta.shape == tb.shape:
                neq = ta != tb
                detail = {"n_diff": int(neq.sum()), "n_total": int(neq.numel()),
                          "a_neg1_frac": float((ta < 0).float().mean()), "b_neg1_frac": float((tb < 0).float().mean()),
                          "diff_groups": int(neq.any(-1).sum()),
                          "heads_with_diff": sorted(set(int(h) for h in neq.any(-1).any(0).any(0).nonzero().flatten().tolist())) if neq.dim() == 4 else None}
            return {"exact": False, "first_mismatch_call": i, "layer": la, "calls": len(a),
                    "shape_a": list(ta.shape), "shape_b": list(tb.shape), "tags": _jsonable(ga), **detail}
        if ga is not None and gb is not None:
            tags_compared += 1
            if ga != gb:
                return {"exact": False, "first_mismatch_call": i, "layer": la, "calls": len(a),
                        "tags_a": _jsonable(ga), "tags_b": _jsonable(gb), "reason": "tags differ"}
    return {"exact": True, "calls": len(a), "tags_compared": tags_compared}


def compare_payloads(g: Dict[str, Any], r: Dict[str, Any], check_trace: bool = True) -> Dict[str, Any]:
    """Compare two payloads; ``exact`` is the AND of every check that ran."""
    from omnimem.pipelines.streaming_inference import compare_frames, compare_latents
    out: Dict[str, Any] = {}
    out["latents"] = compare_latents(g["latents"], r["latents"])
    out["frames"] = compare_frames(g["frames"].numpy(), r["frames"].numpy())
    out["generator_state_equal"] = bool(g["generator_state_hash"] == r["generator_state_hash"])
    out["frames_delivered_equal"] = bool(g["frames_delivered"] == r["frames_delivered"])
    ok = (out["latents"]["exact"] and out["frames"]["exact"]
          and out["generator_state_equal"] and out["frames_delivered_equal"])
    ga, rb = g.get("selection_trace"), r.get("selection_trace")
    if not check_trace:
        out["selection_trace"] = {"exact": None, "note": "not compared (--no_trace)"}
    elif not ga or not rb:
        missing = [side for side, t in (("golden", ga), ("run", rb)) if not t]
        out["selection_trace"] = {
            "exact": False,
            "reason": f"trace is ON but {'/'.join(missing)} has no selection trace "
                      f"(len_golden={0 if not ga else len(ga)}, len_run={0 if not rb else len(rb)}); "
                      f"record the golden with the trace on, or compare with --no_trace",
            "len_a": 0 if ga is None else len(ga), "len_b": 0 if rb is None else len(rb),
        }
        ok = False
    else:
        out["selection_trace"] = _compare_traces(ga, rb)
        ok = ok and out["selection_trace"]["exact"]
    out["exact"] = bool(ok)
    return out


def _print_summary(name: str, cmp: Dict[str, Any]) -> None:
    lat, fr, tr = cmp["latents"], cmp["frames"], cmp["selection_trace"]
    print(f"[golden] {name}: exact={cmp['exact']} latents={lat['exact']} (max_abs={lat.get('max_abs')}) "
          f"frames={fr['exact']} (max_abs={fr.get('max_abs')}) rng={cmp['generator_state_equal']} "
          f"trace={tr.get('exact')} ({tr.get('calls', tr.get('note') or tr.get('reason'))})",
          flush=True)


# ----------------------------------------------------------------------------- commands
def _stats_of(res) -> Dict[str, Any]:
    return {"timings": _jsonable(res.timings), "memory": _jsonable(res.memory), "queues": _jsonable(res.queues),
            "cache_stats": _jsonable(res.cache_stats), "config": _jsonable(res.config)}


def _check_reload(res, expect: bool) -> Optional[str]:
    if not expect:
        return None
    rb = (res.cache_stats or {}).get("bytes")
    if rb is None:
        return f"--expect_reload: cache_stats has no 'bytes' (keys={sorted((res.cache_stats or {}).keys())})"
    if rb <= 0:
        return f"--expect_reload: reload_bytes={rb} (offload/reload path was NOT exercised)"
    return None


def _check_block_fetch(res, expect: bool) -> Optional[str]:
    """``--expect_block_fetch``: the block path moved history this repeat."""
    if not expect:
        return None
    st = res.cache_stats or {}
    if st.get("block_mode") == "resident":
        calls = st.get("block_calls")
        fast = st.get("block_fast_path_calls")
        if calls is None or fast is None:
            return (f"--expect_block_fetch: resident mode but cache_stats has no "
                    f"block_calls/block_fast_path_calls (keys={sorted(st.keys())})")
        if not (calls > 0 and fast == calls):
            return (f"--expect_block_fetch: resident mode but block_fast_path_calls="
                    f"{fast} != block_calls={calls} (every call must take the fast "
                    f"path in resident mode)")
        return None
    got = {k: st.get(f"block_{k}") for k in ("misses", "h2d_bytes", "retired_chunks")}
    if all(v is None for v in got.values()):
        return (f"--expect_block_fetch: cache_stats carries no block_* counters "
                f"(kv_cache_policy={st.get('kv_cache_policy')!r}; is the block store on?)")
    bad = [f"block_{k}={v}" for k, v in got.items() if not (v or 0) > 0]
    if bad:
        return (f"--expect_block_fetch: {', '.join(bad)} -- the block fetch/retire path was "
                f"NOT exercised (need block_misses > 0, block_h2d_bytes > 0, "
                f"block_retired_chunks > 0)")
    return None


def _check_no_chunk_reload(res, expect: bool) -> Optional[str]:
    """``--expect_no_chunk_reload``: zero whole-chunk H2D reloads, this repeat."""
    if not expect:
        return None
    st = res.cache_stats or {}
    calls, rbytes = st.get("calls"), st.get("bytes")       # whole-chunk reload counters
    whole = st.get("block_whole_chunk_reload_calls")
    if calls is None or rbytes is None:
        return (f"--expect_no_chunk_reload: cache_stats has no reload counters "
                f"(keys={sorted(st.keys())})")
    bad = []
    if (calls or 0) != 0:
        bad.append(f"reload_calls={calls}")
    if (rbytes or 0) != 0:
        bad.append(f"reload_bytes={rbytes}")
    if whole is None:
        bad.append("block_whole_chunk_reload_calls is missing (no block store?)")
    elif (whole or 0) != 0:
        bad.append(f"block_whole_chunk_reload_calls={whole}")
    if bad:
        return f"--expect_no_chunk_reload: {', '.join(bad)} (whole-chunk reload must be zero)"
    return None


def _run_checks(res, args, label: str) -> list:
    """Every per-run expectation, for THIS repeat.  Returns a list of problem strings."""
    problems = []
    for check, flag in ((_check_reload, getattr(args, "expect_reload", False)),
                        (_check_block_fetch, getattr(args, "expect_block_fetch", False)),
                        (_check_no_chunk_reload, getattr(args, "expect_no_chunk_reload", False))):
        msg = check(res, flag)
        if msg:
            problems.append(f"{label}: {msg}")
    return problems


def cmd_record(args) -> int:
    gdir = Path(args.golden_dir)
    rt = make_runtime(args)      # a refused request exits here: nothing is written yet
    gdir.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir) if args.work_dir else gdir / "work"
    load_info, load_s = load_runtime(rt)
    spec = request_spec(args)
    print(f"[golden] record {spec} load_s={load_s:.1f}", flush=True)
    runs, payloads = [], []
    problems: list = []
    for i in range(max(1, args.repeat)):
        res, wall = run_once(rt, spec, work, f"record_r{i}")
        payloads.append(payload_from_result(res))
        runs.append({"repeat": i, "wall_s": wall, **_stats_of(res)})
        print(f"[golden] run r{i}: wall={wall:.2f}s total_s={res.timings.get('total_s'):.2f} "
              f"first_frames={res.timings.get('first_frames_delivered_s')} "
              f"reload_bytes={(res.cache_stats or {}).get('bytes')} "
              f"block_h2d_bytes={(res.cache_stats or {}).get('block_h2d_bytes')}", flush=True)
        # EVERY repeat is checked: a single final-run check hides a first-run-only
        # whole-chunk reload or a repeat that never fetched a block.
        rp = _run_checks(res, args, f"r{i}")
        for msg in rp:
            print(f"[golden] FAIL {msg}", flush=True)
        problems.extend(rp)
    repeat_cmp = [compare_payloads(payloads[0], p, check_trace=spec["trace"]) for p in payloads[1:]]
    for i, c in enumerate(repeat_cmp, start=1):
        _print_summary(f"repeat r{i} vs r0", c)
    problem = "; ".join(problems) if problems else None
    torch.save(payloads[0], gdir / "golden.pt")
    if args.save_repeats:
        for i, p in enumerate(payloads[1:], start=1):
            torch.save(p, gdir / f"repeat_r{i}.pt")
    manifest = {
        "kind": "golden", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tag": args.tag,
        "spec": spec, "recipe": _jsonable(recipe_kwargs(args)),
        "assets": {k: getattr(args, k) for k in ASSET_ARGS},
        "load_s": load_s, "load_info": _jsonable(load_info), "runs": runs,
        "repeat_exact": [c["exact"] for c in repeat_cmp], "repeat_compare": repeat_cmp,
        "golden_sha256": _sha256_file(gdir / "golden.pt"), "env": _env_info(),
        "latent_shape": list(payloads[0]["latents"].shape), "frames_shape": list(payloads[0]["frames"].shape),
        "selection_calls": None if payloads[0]["selection_trace"] is None else len(payloads[0]["selection_trace"]),
        "problem": problem, "problems": problems,
        # The resolved configuration of the last run, including the real block-store geometry.
        "resolved_config": _jsonable(runs[-1]["config"]) if runs else None,
        "block_store": _jsonable((runs[-1]["config"] or {}).get("block_store")) if runs else None,
    }
    (gdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[golden] saved {gdir/'golden.pt'} sha256={manifest['golden_sha256'][:16]} "
          f"selection_calls={manifest['selection_calls']}", flush=True)
    if problem:
        print(f"[golden] FAIL: {problem}", flush=True)
        return 2
    if not all(manifest["repeat_exact"]):
        print("[golden] FAIL: in-process repeats are NOT bit-exact — baseline is not self-consistent", flush=True)
        return 3
    return 0


def cmd_compare(args) -> int:
    gdir = Path(args.golden_dir)
    golden = torch.load(gdir / "golden.pt", weights_only=False)
    gman = json.loads((gdir / "manifest.json").read_text())
    gspec = gman["spec"]
    rt, load_info, load_s = build_runtime(args)      # also resolves the device layout
    spec = request_spec(args)
    # A golden recorded under another recache_mode is not comparable; warn, not mismatch.
    for k in SPEC_KEYS_COMPARED:
        if gspec.get(k) != spec[k]:
            print(f"[golden] WARNING: spec[{k}] golden={gspec.get(k)} current={spec[k]}",
                  flush=True)
    # prompt / switch follow the golden unless explicitly overridden
    if args.prompt is None:
        spec["prompt"] = gspec["prompt"]
    if not args.switch_prompt and gspec.get("switch_prompts"):
        spec["switch_prompts"], spec["switch_frame_indices"] = gspec["switch_prompts"], gspec["switch_frame_indices"]
    work = Path(args.work_dir) if args.work_dir else gdir / "work"
    results = []
    problems: list = []
    for i in range(max(1, args.repeat)):
        res, wall = run_once(rt, spec, work, f"compare_{args.tag}_r{i}")
        p = payload_from_result(res)
        # Trace on (no --no_trace) means BOTH sides must have a trace and it must match:
        # a golden without one is a failure with a reason, not a silent skip.
        c = compare_payloads(golden, p, check_trace=spec["trace"])
        _print_summary(f"{args.tag} r{i} vs golden", c)
        results.append({"repeat": i, "wall_s": wall, "compare": c, **_stats_of(res)})
        # EVERY repeat is checked, not just the last one.
        rp = _run_checks(res, args, f"r{i}")
        for msg in rp:
            print(f"[golden] FAIL {msg}", flush=True)
        problems.extend(rp)
        if args.save_run:
            torch.save(p, gdir / f"run_{args.tag or 'run'}_r{i}.pt")
    problem = "; ".join(problems) if problems else None
    report = {"kind": "compare", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tag": args.tag,
              "golden_dir": str(gdir), "golden_sha256": gman.get("golden_sha256"), "spec": spec,
              "recipe": _jsonable(recipe_kwargs(args)), "golden_spec": gspec,
              "load_s": load_s, "results": results, "exact": all(r["compare"]["exact"] for r in results),
              "problem": problem, "problems": problems,
              "resolved_config": _jsonable(results[-1]["config"]) if results else None,
              "block_store": _jsonable((results[-1]["config"] or {}).get("block_store")) if results else None,
              "env": _env_info()}
    out = gdir / f"compare_{args.tag or 'run'}.json"
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[golden] report -> {out} exact={report['exact']}", flush=True)
    if problem:
        print(f"[golden] FAIL: {problem}", flush=True)
        return 2
    return 0 if report["exact"] else 1


def cmd_diff(args) -> int:
    a = torch.load(args.a, weights_only=False)
    b = torch.load(args.b, weights_only=False)
    c = compare_payloads(a, b, check_trace=not args.no_trace)
    _print_summary(f"{args.a} vs {args.b}", c)
    if args.report:
        Path(args.report).write_text(json.dumps(c, indent=2, default=str))
    return 0 if c["exact"] else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_run_args(p):
        p.add_argument("--golden_dir", required=True)
        for k in ASSET_ARGS:
            p.add_argument(f"--{k}", required=True)
        p.add_argument("--nframes", type=int, default=249)
        p.add_argument("--prompt", default=None)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--switch_prompt", default="", help="'' = none, 'default' = built-in, or text")
        p.add_argument("--switch_latent_frame", type=int, default=None)
        p.add_argument("--no_trace", action="store_true", help="do not capture per-layer top-k IDs")
        p.add_argument("--repeat", type=int, default=1)
        # The request (``Recipe``) -- every default below is this tool's own, not the
        # runtime's: the reference configuration (chunk LRU 7, original VAE decode).
        p.add_argument("--height", type=int, default=480)
        p.add_argument("--width", type=int, default=832)
        p.add_argument("--fps", type=int, default=16)
        p.add_argument("--num_inference_steps", type=int, default=4)
        p.add_argument("--shift", type=float, default=5.0)
        p.add_argument("--frame_per_block", type=int, default=3)
        p.add_argument("--window_size", type=int, default=3)
        p.add_argument("--sink_size", type=int, default=None)
        p.add_argument("--enable_kv_evict", action=_UnderscoreNegBooleanOptionalAction, default=True,
                       help="evict KV history outside the window and sink from the DiT GPU")
        p.add_argument("--lru_max_size", type=int, default=7)
        p.add_argument("--slice_last_frames", type=int, default=21)
        p.add_argument("--no_compile", action="store_true")
        p.add_argument("--compile_mode", default="default",
                       help="torch.compile mode of the DiT; only 'default' is reproducible across processes")
        p.add_argument("--queue_depth", type=int, default=2)
        p.add_argument("--nsa_backend", default=os.environ.get("OMNIMEM_NSA_BACKEND", "triton"),
                       choices=["triton", "cuda"],
                       help="kernel backend, triton or cuda (default: $OMNIMEM_NSA_BACKEND, else triton)")
        p.add_argument("--dit_device", default="cuda:0")
        p.add_argument("--decode_device", default=None,
                       help="VAE decode device (default: the DiT GPU; the same device means one GPU)")
        p.add_argument("--work_dir", default=None)
        p.add_argument("--tag", default="")
        p.add_argument("--expect_reload", action="store_true",
                       help="fail unless whole chunks were reloaded (cache_stats bytes > 0)")
        p.add_argument("--save_repeats", action="store_true", help="record: also save repeat payloads as repeat_r<i>.pt")
        p.add_argument("--kv_cache_policy", default="chunk_lru", choices=["chunk_lru", "block_lru"],
                       help="history cache policy: block_lru keeps selected KV blocks, chunk_lru keeps whole chunks")
        p.add_argument("--kv_block_pool_gb", type=float, default=0.0,
                       help="persistent block pool budget in GiB, must be > 0; only with block_lru, else leave 0")
        p.add_argument("--kv_block_scratch_gb", type=float, default=0.0,
                       help="shared scratch block pool budget in GiB; only with block_lru, else leave 0")
        p.add_argument("--kv_tier", action="store_true",
                       help="with --kv_cache_policy chunk_lru on two GPUs: evict KV history to the decode GPU, not host memory")
        p.add_argument("--kv_tier_budget_gb", type=float, default=48.0,
                       help="KV history budget on the decode GPU, in GiB")
        # Block store options: each only meaningful with --kv_cache_policy block_lru;
        # saved in the golden spec (compare warns on drift, same as --recache_mode).
        p.add_argument("--kv_block_resident_if_fits", action=_UnderscoreNegBooleanOptionalAction,
                       default=True,
                       help="keep the whole KV history on the DiT GPU if it fits the pool budget; only with block_lru")
        p.add_argument("--kv_block_dedup", default="device", choices=["unique", "device"],
                       help="dedup of selected block keys, on the device or with torch.unique; only with block_lru")
        # A golden run keeps the original window-buffer and VAE decode path unless flagged.
        p.add_argument("--kv_window_buffer", action="store_true",
                       help="preallocate KV history and window buffers; faster but uses more GPU memory")
        p.add_argument("--vae_fast", dest="vae_decode_fast",
                       action=_UnderscoreNegBooleanOptionalAction, default=False,
                       help="faster VAE decode path, off by default here; output is not bit-identical")
        p.add_argument("--patch_embed_linear", action="store_true",
                       help="use a linear patch embedding instead of Conv3d; results may differ slightly")
        p.add_argument("--recache_mode", default="full", choices=["full", "window_only", "none"],
                       help="KV history handling at a prompt switch; window_only restores old KV outside the window, none keeps it")
        p.add_argument("--expect_block_fetch", action="store_true",
                       help="fail unless every repeat used the block path (block fetches and retirements > 0)")
        p.add_argument("--expect_no_chunk_reload", action="store_true",
                       help="fail unless every repeat had no whole-chunk reloads")
        p.add_argument("--park_t5", default=None,
                       help="move the T5 encoder to this device after encoding (e.g. cuda:1 or cpu) to free DiT GPU memory")

    p_rec = sub.add_parser("record")
    add_run_args(p_rec)
    p_cmp = sub.add_parser("compare")
    add_run_args(p_cmp)
    p_cmp.add_argument("--report", default=None)
    p_cmp.add_argument("--save_run", action="store_true", help="also save each compared run as run_<tag>_r<i>.pt for diagnosis")
    p_diff = sub.add_parser("diff")
    p_diff.add_argument("a")
    p_diff.add_argument("b")
    p_diff.add_argument("--no_trace", action="store_true")
    p_diff.add_argument("--report", default=None)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "record":
        return cmd_record(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    return cmd_diff(args)


if __name__ == "__main__":
    sys.exit(main())
