#!/usr/bin/env python3
"""Evaluate project-page examples with independent GPU workers, without FSDP."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import traceback


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def timing_plan(record, fps=16, block=3):
    """Align latent blocks, then crop only the video tail to the stated duration."""
    seconds = record["duration_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    export_frames = round(seconds * fps)
    if not math.isclose(export_frames, seconds * fps) or export_frames < 1:
        raise ValueError("Duration must correspond to a whole number of output frames")
    latent_frames = math.ceil((math.ceil((export_frames - 1) / 4) + 1) / block) * block
    prompts, switches = record["prompts"], record["switch_times_seconds"]
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Each example needs a nonempty list of exact prompt texts")
    if not isinstance(switches, list) or len(switches) != len(prompts) - 1:
        raise ValueError("One switch time is required for each later prompt")
    if any(type(t) not in (int, float) or not math.isfinite(t) or not 0 < t < seconds for t in switches):
        raise ValueError("Switch times must lie inside the video")
    if switches != sorted(set(switches)):
        raise ValueError("Switch times must be strictly increasing")
    # The first L Wan latents decode to 4L-3 RGB frames. A switch before latent
    # index L therefore first affects RGB frame index 4L-3, with zero-based indices.
    positions = [block * math.floor((t * fps + 3) / (4 * block) + 0.5) for t in switches]
    if (positions != sorted(set(positions))
            or any(not 0 < p < latent_frames or 4 * p - 3 >= export_frames for p in positions)):
        raise ValueError("Requested switches collide or fall outside the exported block-aligned video")
    actual_frames = [4 * p - 3 for p in positions]
    return {
        "declared_duration_seconds": seconds, "fps": fps,
        "export_frames": export_frames, "generated_frames": 4 * (latent_frames - 1) + 1,
        "latent_frames": latent_frames, "frame_per_block": block,
        "requested_switch_times_seconds": switches, "switch_latent_indices": positions,
        "first_affected_rgb_frame_indices": actual_frames,
        "effective_switch_times_seconds": [p / fps for p in actual_frames],
        "switch_drift_seconds": [p / fps - t for p, t in zip(actual_frames, switches)],
        "timing_note": "Causal VAE output boundary; visible semantic transitions may take additional frames.",
    }


def require_output_under(output, root):
    """Refuse an output directory outside ``root``; ``root=None`` allows any location."""
    if root is not None and not Path(output).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"This evaluation writes outputs only under {root}")


def load_records(path, selected):
    manifest = json.loads(path.read_text())
    records = manifest["examples"]
    seen = set()
    for record in records:
        name = record["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in seen:
            raise ValueError(f"Example IDs must be unique filesystem-safe names: {name!r}")
        seen.add(name)
    if selected and not set(selected).issubset(seen):
        raise ValueError(f"Unknown example IDs: {sorted(set(selected) - seen)}")
    return [r for r in records if not selected or r["id"] in selected]


def write_video(frames, destination, *, fps, width, height):
    """Encode one frame at a time; avoid copying a 60-second clip into a tensor grid."""
    import av
    import numpy as np

    temporary = destination.with_name(destination.stem + ".partial.mp4")
    with av.open(str(temporary), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        stream.codec_context.thread_count = 2
        for index, image in enumerate(frames):
            if image.size != (width, height):
                raise ValueError(f"Unexpected output frame shape: {image.size}")
            frame = av.VideoFrame.from_ndarray(np.asarray(image.convert("RGB")), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(temporary)) as container:
        stream = container.streams.video[0]
        if stream.frames != len(frames) or stream.average_rate != fps:
            raise ValueError("Encoded video frame count/rate differs from requested output")
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--checkpoint-step", type=int, default=600)
    parser.add_argument("--checkpoint-role", choices=("model", "ema"), default="ema")
    parser.add_argument("--vae-path", type=Path)
    parser.add_argument("--t5-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        help="refuse an --output outside this directory, e.g. to keep "
                             "the videos on a local scratch disk (default: no restriction)")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1)
    parser.add_argument("--shift", type=float, default=5)
    args = parser.parse_args()
    if args.checkpoint_step < 0:
        raise ValueError("checkpoint-step must be nonnegative")
    if args.fps <= 0 or args.steps <= 0 or min(args.width, args.height) <= 0 or args.width % 16 or args.height % 16:
        raise ValueError("Positive FPS/steps and dimensions divisible by 16 are required")
    records = load_records(args.manifest, args.ids)
    planned = [(record, timing_plan(record, args.fps)) for record in records]
    rank, world, local_rank = (int(os.environ.get(name, default)) for name, default in
                               (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    if not 0 <= rank < world or local_rank < 0:
        raise ValueError("Invalid worker rank environment")
    assigned = planned[rank::world]
    if args.dry_run:
        print(json.dumps({"rank": rank, "world_size": world,
                          "examples": [{"id": r["id"], **p} for r, p in assigned]}, indent=2))
        return
    require_output_under(args.output, args.output_root)
    for name in ("pretrained", "vae_path", "t5_path", "tokenizer_path"):
        if getattr(args, name) is None or not getattr(args, name).exists():
            raise ValueError(f"A readable --{name.replace('_', '-')} is required")
    args.output.mkdir(parents=True, exist_ok=True)
    worker_path = args.output / f"worker-{rank:02d}.json"
    worker = {"rank": rank, "world_size": world, "local_rank": local_rank,
              "pid": os.getpid(), "started_at_utc": timestamp(), "completed_ids": [],
              "assigned_ids": [r["id"] for r, _ in assigned], "status": "loading"}
    atomic_json(worker_path, worker)
    try:
        if not assigned:
            worker.update(status="completed", finished_at_utc=timestamp())
            atomic_json(worker_path, worker)
            return
        # The torchrun ranks are worker assignment only: never call init_process_group.
        import torch
        # Make this worker's GPU the current CUDA device before any OmniMem model code
        # runs, so code that asks for the current device agrees with ``device`` below.
        torch.cuda.set_device(local_rank)
        from omnimem.utils.numa_bind import bind_to_gpu_numa
        from omnimem.pipelines import CausalWanT2VPipeline
        import inspect

        bind_to_gpu_numa()
        device = torch.device("cuda", local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(args.seed)
        pipeline = CausalWanT2VPipeline.from_checkpoints(
            str(args.pretrained), str(args.vae_path), str(args.t5_path), str(args.tokenizer_path),
            shift=args.shift, frame_per_block=3)
        if args.compile:
            pipeline.transformer.compile(mode="default", dynamic=False)
        pipeline.to(device=device, dtype=torch.bfloat16)
        pipeline.set_progress_bar_config(disable=True)
        inference = {"checkpoint_role": args.checkpoint_role, "checkpoint_step": args.checkpoint_step, "checkpoint_directory": str(args.pretrained.resolve()),
                     "checkpoint_config_sha256": file_sha256(args.pretrained / "config.json"),
                     "pipeline_source": inspect.getfile(CausalWanT2VPipeline),
                     "pipeline_sha256": file_sha256(inspect.getfile(CausalWanT2VPipeline)),
                     "manifest_sha256": file_sha256(args.manifest), "seed": args.seed,
                     "width": args.width, "height": args.height, "inference_steps": args.steps,
                     "guidance_scale": args.cfg, "shift": args.shift, "frame_per_block": 3,
                     "effective_sampler": "rectified_flow_pred_x0_and_re_noise", "compile": args.compile,
                     "window_size": {"kv": 3, "kv_cmp": None}, "sink_size": {"kv": None, "kv_cmp": None},
                     "lru_max_size": 7, "slice_last_frames": 21, "recache_mode": "full", "vae_decode_chunk_size": 3,
                     "generation_mode": "continuous_causal_AR_as_training_evaluation"}
        for record, timing in assigned:
            name = record["id"]
            video, metadata = args.output / f"{name}.mp4", args.output / f"{name}.json"
            request = {"example": record, "timing": timing, "inference": inference}
            if video.exists() or metadata.exists():
                previous = json.loads(metadata.read_text()) if metadata.exists() else {}
                if (args.resume and previous.get("status") == "completed" and previous.get("request") == request
                        and video.exists() and previous.get("video_sha256") == file_sha256(video)):
                    worker["completed_ids"].append(name)
                    atomic_json(worker_path, worker)
                    continue
                raise FileExistsError(f"Existing output does not match a completed resumable example: {name}")
            worker.update(status="running", current_id=name, updated_at_utc=timestamp())
            atomic_json(worker_path, worker)
            print(json.dumps({"rank": rank, "event": "start", "id": name, "at": timestamp()}), flush=True)
            start = time.monotonic()
            generator = torch.Generator(device=device).manual_seed(args.seed)
            torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = pipeline.generate(
                    input_prompt=record["prompts"][0], size=(args.width, args.height),
                    frame_num=timing["generated_frames"], shift=args.shift, sampling_steps=args.steps,
                    guide_scale=args.cfg, generator=generator, offload_model=False,
                    window_size=inference["window_size"], sink_size=inference["sink_size"],
                    enable_chunk_per_head_cache=True,
                    enable_kv_evict=True, lru_max_size=7, switch_prompts=record["prompts"][1:] or None,
                    switch_frame_indices=timing["switch_latent_indices"] or None,
                    slice_last_frames=21, recache_mode="full", vae_decode_chunk_size=3).images[0]
            if len(generated) != timing["generated_frames"]:
                raise ValueError(f"Unexpected decoded frame count: {len(generated)}")
            write_video(generated[:timing["export_frames"]], video, fps=args.fps, width=args.width, height=args.height)
            atomic_json(metadata, {"status": "completed", "request": request, "rank": rank,
                                   "completed_at_utc": timestamp(), "elapsed_seconds": time.monotonic() - start,
                                   "video": str(video), "video_bytes": video.stat().st_size,
                                   "video_sha256": file_sha256(video),
                                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)})
            del generated
            gc.collect()
            torch.cuda.empty_cache()
            worker["completed_ids"].append(name)
            atomic_json(worker_path, worker)
            print(json.dumps({"rank": rank, "event": "completed", "id": name, "at": timestamp()}), flush=True)
        worker.update(status="completed", current_id=None, finished_at_utc=timestamp())
        atomic_json(worker_path, worker)
    except BaseException as error:
        worker.update(status="failed", updated_at_utc=timestamp(), error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        atomic_json(worker_path, worker)
        raise


if __name__ == "__main__":
    main()
