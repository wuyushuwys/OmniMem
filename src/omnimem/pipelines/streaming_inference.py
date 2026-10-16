"""Single-video low-latency streaming inference runtime: DiT on one GPU, Wan VAE on the same or a second.

Shared by ``inference_single_gpu.py``, ``inference_dual_gpu.py`` and
``scripts/golden_inference.py``; :class:`Recipe` (``omnimem.utils.streaming_recipe``) holds the defaults.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from omnimem.models.cache import EXACT_PINNED_KV_ENV, exact_pinned_kv_enabled
from omnimem.pipelines import compile_state
from omnimem.pipelines.pipeline_wan_causal import validate_kv_policy
from omnimem.pipelines.streaming_decode import (
    LatentChunk,
    StreamingSession,
    WanChunkDecoder,
    decoder_thread_streams,
)
from omnimem.utils.stage_marks import StageMarks
from omnimem.utils.streaming_recipe import DEFAULT_BLOCK_POOL_GB, DEFAULT_BLOCK_SCRATCH_GB, Recipe
from omnimem.utils.streaming_video_writer import StreamingVideoWriter
from omnimem.utils.torch_utils import indexed_device
from omnimem.utils.validation_prompts import validate_prompt_schedules


def _block_store_device_bytes(store_cfg: dict) -> int:
    """Device bytes ONE configured block store actually holds (resident mode keeps no pools)."""
    if str(store_cfg.get("mode") or "") == "resident":
        return int(store_cfg.get("resident_history_bytes") or 0)
    return int(store_cfg.get("pool_bytes") or 0) + int(store_cfg.get("scratch_bytes") or 0)


NEG_PROMPT = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
              "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
              "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
              "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in "
              "the background, walking backwards")

DEFAULT_PROMPT = "A cat sitting at a grand piano, elegantly playing a classical piece with its paws."


@dataclass
class ModelPaths:
    pretrained: str
    vae_path: str
    t5_path: str
    tokenizer_path: str


#: The Wan 2.1 VAE maps 8x8 pixels to one latent; the DiT patches latents 2x2.
VAE_SPATIAL_STRIDE = 8


def latent_frames(nframes: int, temporal_stride: int = 4) -> int:
    return (nframes - 1) // temporal_stride + 1


def actual_frames(nframes: int, temporal_stride: int = 4) -> int:
    return 1 + temporal_stride * (latent_frames(nframes, temporal_stride) - 1)


def validate_request(nframes: int, frame_per_block: int, *,
                     switch_prompts: Optional[Sequence[str]] = None,
                     switch_frame_indices: Optional[Sequence[int]] = None,
                     queue_depth: int = 2) -> None:
    """Fail early (``ValueError``) for a request the streaming path cannot run."""
    if nframes < 1:
        raise ValueError(f"nframes must be >= 1, got {nframes}")
    if frame_per_block < 1:
        raise ValueError(f"frame_per_block must be >= 1, got {frame_per_block}")
    t = latent_frames(nframes)
    if t % frame_per_block != 0:
        raise ValueError(
            f"nframes={nframes} -> {t} latent frames is not a multiple of frame_per_block="
            f"{frame_per_block}; partial DiT chunks are not supported by the streaming path "
            f"(use a frame count whose latent count divides evenly, e.g. 81/249/960)")
    if queue_depth < 1:
        raise ValueError(f"queue_depth must be >= 1, got {queue_depth}")
    if not switch_prompts and not switch_frame_indices:
        return
    if (isinstance(switch_prompts, str) or not switch_prompts or not switch_frame_indices
            or len(switch_prompts) != len(switch_frame_indices)):
        raise ValueError(
            "switch_prompts and switch_frame_indices go together, one latent frame index "
            f"per switch prompt (got prompts={switch_prompts!r}, "
            f"indices={switch_frame_indices!r})")
    try:
        # The evaluation side's rules for a prompt schedule: the indices are latent
        # frames inside the video, increasing, on chunk boundaries.
        validate_prompt_schedules(
            [{"id": "request", "prompts": ["main", *switch_prompts],
              "switch_frame_indices": list(switch_frame_indices)}],
            num_frames=int(nframes), frame_per_block=int(frame_per_block))
    except ValueError as exc:
        raise ValueError(
            f"invalid prompt switch (switch_frame_indices={list(switch_frame_indices)}): "
            f"{str(exc).split(': ', 1)[-1]}; they are latent frame indices, increasing "
            f"multiples of frame_per_block={frame_per_block} inside 1..{t - 1}") from None


def check_token_grid(height: int, width: int, frame_per_block: int,
                     model_config: Dict[str, Any]) -> None:
    """``ValueError`` unless one chunk's token grid fits the checkpoint's NSA pooling."""
    pf, ph, pw = model_config.get("patch_size") or (1, 2, 2)
    grid = (frame_per_block // pf, height // (VAE_SPATIAL_STRIDE * ph),
            width // (VAE_SPATIAL_STRIDE * pw))
    pool = tuple(int(model_config.get(k) or 1) for k in ("pool_f", "pool_h", "pool_w"))
    if frame_per_block % pf or any(g % p for g, p in zip(grid, pool)):
        raise ValueError(
            f"{height}x{width} with frame_per_block={frame_per_block} is a token grid of "
            f"{grid[0]}x{grid[1]}x{grid[2]} (frames x height x width) per chunk, which this "
            f"checkpoint's NSA pooling {pool[0]}x{pool[1]}x{pool[2]} does not divide")
    group = int(model_config.get("group_q_size") or 1)
    if (grid[0] * grid[1] * grid[2]) % group:
        raise ValueError(
            f"{height}x{width} with frame_per_block={frame_per_block} is "
            f"{grid[0] * grid[1] * grid[2]} tokens per chunk, not a multiple of this "
            f"checkpoint's query group size {group}")


def _cuda_device(device, n_visible: int, what: str) -> torch.device:
    """``device`` as an indexed CUDA device that exists, else ``ValueError``."""
    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError):
        raise ValueError(f"{what}: {device!r} is not a device") from None
    if parsed.type != "cuda":
        raise ValueError(f"{what} must be a CUDA device, got {parsed}")
    if n_visible < 1:
        raise ValueError("no CUDA device is visible")
    parsed = indexed_device(parsed)
    if parsed.index >= n_visible:
        raise ValueError(f"{what} {parsed} does not exist ({n_visible} CUDA device(s) visible)")
    return parsed


def check_layout(dit_device="cuda:0", decode_device=None) -> Tuple[torch.device, torch.device]:
    """Indexed ``(dit_device, decode_device)``; with no decode device the VAE shares the DiT GPU."""
    n_visible = torch.cuda.device_count()
    dit = _cuda_device(dit_device, n_visible, "the DiT device")
    if decode_device is None:
        return dit, dit
    if n_visible < 2:
        raise ValueError(f"the two-GPU layout needs two CUDA devices, only {n_visible} is visible")
    decode = _cuda_device(decode_device, n_visible, "the decode device")
    if decode == dit:
        raise ValueError(f"the two-GPU layout needs two distinct devices, got {dit} twice")
    return dit, decode


def check_output_target(output_path: str, preview_dir: Optional[str] = None,
                        ffmpeg: str = "ffmpeg") -> None:
    """Fail early on a missing encoder or unwritable output directory; creates the directories."""
    if not output_path:
        raise ValueError("an output path is required: the frames are encoded while "
                         "they are generated")
    if shutil.which(ffmpeg) is None:
        raise FileNotFoundError(f"{ffmpeg!r} is not on PATH; it encodes the video")
    for what, directory in (("output", os.path.dirname(os.path.abspath(output_path))),
                            ("preview", preview_dir)):
        if directory is None:
            continue
        try:
            os.makedirs(directory, exist_ok=True)
            with tempfile.TemporaryFile(dir=directory):
                pass
        except OSError as exc:
            raise OSError(f"cannot write to the {what} directory {directory}: {exc}") from exc


def flash_attention_backend() -> str:
    import omnimem.models.transformers.attention as attn_mod
    if attn_mod.FLASH_ATTN_4_AVAILABLE:
        return "fa4"
    if attn_mod.FLASH_ATTN_3_AVAILABLE:
        return "fa3"
    if attn_mod.FLASH_ATTN_2_AVAILABLE:
        return "fa2"
    return "sdpa"


def _rss() -> Dict[str, float]:
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS", "VmHWM")):
                    k, v = line.split(":", 1)
                    out[k.strip().lower() + "_gb"] = float(v.strip().split()[0]) / (1024 ** 2)
    except OSError:
        pass
    return out


def _percentile(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


class VideoFileSink:
    """Writer-thread sink: incremental MP4 + previews + optional frame capture."""

    def __init__(self, writer: StreamingVideoWriter, capture: bool = False):
        self.writer = writer
        self.capture = capture
        self.captured: List[np.ndarray] = []
        self.first_delivery_time: Optional[float] = None

    def on_frames(self, frames_u8: torch.Tensor, frame_start: int, chunk_index: int) -> None:
        if not self.writer.is_open:
            self.writer.open()
        arr = frames_u8.numpy()
        self.writer.write_frames(arr)
        if self.first_delivery_time is None:
            self.first_delivery_time = time.monotonic()
        if self.capture:
            self.captured.append(arr.copy())

    def finish(self):
        if not self.writer.is_open:
            self.writer.open()
        return self.writer.finalize()

    def abort(self) -> None:
        self.writer.abort()


@dataclass
class KVHistory:
    """The recipe's KV history options resolved against the layout, as ``generate()`` takes them."""
    kv_cache_policy: str
    kv_tier_device: Optional[torch.device]
    kv_tier_budget_bytes: int
    kv_block_pool_bytes: int
    kv_block_scratch_bytes: int
    kv_block_resident_if_fits: bool
    kv_block_dedup: str
    kv_exact_pinned_host: bool


@dataclass
class RunResult:
    prompt: str
    seed: int
    requested_frames: int
    actual_frames: int
    frames_delivered: int
    output_path: Optional[str]
    timings: Dict[str, Any]
    memory: Dict[str, Any]
    queues: Dict[str, Any]
    cache_stats: Dict[str, Any]
    config: Dict[str, Any]
    preview_paths: List[str] = field(default_factory=list)
    chunk_records: List[Dict[str, float]] = field(default_factory=list)
    marks: List[Dict[str, Any]] = field(default_factory=list)
    latents: Optional[torch.Tensor] = None          # capture only (CPU float32)
    frames: Optional[np.ndarray] = None             # capture only (uint8 [T,H,W,3])
    selection_trace: Optional[list] = None          # capture only
    generator_state_hash: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("latents", "frames", "selection_trace"):
            d[k] = None if getattr(self, k) is None else "<captured>"
        return d


class StreamingRuntime:
    """Loads the models once and generates videos, one request at a time."""

    def __init__(self, paths: ModelPaths, recipe: Recipe, dit_device="cuda:0", decode_device=None):
        self.paths = paths
        self.recipe = recipe
        # Indexed, so that ``cuda`` and ``cuda:0`` are one device wherever two devices
        # are compared (the layout, the tier, the VAE placement).
        self.dit_device = indexed_device(dit_device)
        self.decode_device = (self.dit_device if decode_device is None
                              else indexed_device(decode_device))
        self.pipeline = None
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.load_info: Dict[str, Any] = {}
        self._vae_device: Optional[torch.device] = None
        self._compiled = False
        # Set when a failed request left a worker thread behind that still owns the VAE;
        # every later request is refused (see ``run``).
        self._unusable: Optional[str] = None
        self._check_recipe()
        # The decoder thread's CUDA streams (two-GPU layout): made by the first request
        # and kept for the runtime's lifetime, see ``_decoder_streams``.
        self._decoder_stream_pair: Optional[Tuple[torch.cuda.Stream, torch.cuda.Stream]] = None
        # KV history options resolved against the layout, so a bad combination fails before loading.
        self._resolve_kv_history()
        # The last resolved configuration dict (the one manifests save), kept so ``run``
        # can fill in the real block-store geometry once the lazily created cache exists.
        self._config: Dict[str, Any] = {}

    @property
    def gpus(self) -> int:
        """GPUs of the layout: 1 = the VAE shares the DiT GPU, 2 = it has its own."""
        return 1 if self.decode_device == self.dit_device else 2

    def _check_recipe(self) -> None:
        """``ValueError`` for recipe values no checkpoint can run (no model needed)."""
        r = self.recipe
        stride = 2 * VAE_SPATIAL_STRIDE
        if r.height < stride or r.width < stride or r.height % stride or r.width % stride:
            raise ValueError(
                f"height and width must be positive multiples of {stride} (the VAE's "
                f"{VAE_SPATIAL_STRIDE}x stride times the DiT's 2x2 patches), got "
                f"{r.height}x{r.width}")
        if r.frame_per_block < 1:
            raise ValueError(f"frame_per_block must be >= 1, got {r.frame_per_block}")
        if r.num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {r.num_inference_steps}")
        if r.park_t5:
            try:
                park = torch.device(r.park_t5)
            except (RuntimeError, TypeError):
                raise ValueError(f"park_t5: {r.park_t5!r} is not a device") from None
            if park.type not in ("cpu", "cuda") or (
                    park.type == "cuda" and (park.index or 0) >= torch.cuda.device_count()):
                raise ValueError(
                    f"park_t5: device {park} does not exist "
                    f"({torch.cuda.device_count()} CUDA device(s) visible; or 'cpu')")

    # ------------------------------------------------------------------ loading
    def load(self) -> Dict[str, Any]:
        from omnimem.nsa import backend as nsa_backend
        from omnimem.models.text_encoders.t5 import T5EncoderModel
        from omnimem.models.transformers.causal_wan_nsa_model import CausalWanNSAModel
        from omnimem.models.autoencoders import get_autoencoder
        from omnimem.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from omnimem.pipelines import CausalWanT2VPipeline
        from omnimem.models.transformers.wan_model import set_patch_embed_linear

        # Before anything heavy is read: the checkpoint's config alone says whether the
        # requested resolution fits its NSA pooling.
        r = self.recipe
        check_token_grid(r.height, r.width, r.frame_per_block,
                         CausalWanNSAModel.load_config(self.paths.pretrained))
        if nsa_backend.get_backend() == "cuda":
            # Build and load both CUDA extensions now so none is built inside the first request.
            from omnimem.nsa.cuda_backend import _loader, block_kv_ops
            _loader.load()
            block_kv_ops.load()
            block_kv_ops.warmup(self.dit_device)

        t0 = time.monotonic()
        scheduler = FlowUniPCMultistepScheduler(shift=r.shift)
        vae = get_autoencoder(self.paths.vae_path)
        t_vae = time.monotonic()
        text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=torch.device("cpu"),
                                      checkpoint_path=self.paths.t5_path,
                                      tokenizer_path=self.paths.tokenizer_path, shard_fn=None)
        text_encoder.model.requires_grad_(False).eval()
        t5_param_bytes = sum(p.numel() * p.element_size() for p in text_encoder.model.parameters())
        t_t5 = time.monotonic()
        transformer = CausalWanNSAModel.from_pretrained(self.paths.pretrained)
        transformer.requires_grad_(False).eval()
        vae.requires_grad_(False).eval()
        t_dit = time.monotonic()

        transformer.to(device=self.dit_device, dtype=torch.bfloat16)
        vae.to(dtype=torch.bfloat16)
        set_patch_embed_linear(r.patch_embed_linear)
        if r.compile:
            # Pin the process-wide compile settings the outputs depend on.
            compile_state.pin_compile_settings()
            transformer.compile(mode=r.compile_mode)
            self._compiled = True
        pipeline = CausalWanT2VPipeline(transformer=transformer, tokenizer=text_encoder.tokenizer,
                                        text_encoder=text_encoder, vae=vae, scheduler=scheduler,
                                        frame_per_block=r.frame_per_block)
        # Explicit placement: never pipeline.to(device) once the VAE may live elsewhere.
        self.pipeline, self.transformer, self.vae, self.text_encoder = pipeline, transformer, vae, text_encoder
        self.place_vae(self.decode_device)
        self.load_info = {
            "load_vae_s": t_vae - t0, "load_t5_s": t_t5 - t_vae, "load_dit_s": t_dit - t_t5,
            "to_device_s": time.monotonic() - t_dit,
            "flash_attention": flash_attention_backend(),
            "nsa_backend": nsa_backend.get_backend(),
            "compile": r.compile, "compile_mode": r.compile_mode if r.compile else None,
            "patch_embed_linear": r.patch_embed_linear,
            "dit_device": str(self.dit_device), "decode_device": str(self.decode_device),
            "t5_param_bytes": t5_param_bytes,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformer_config": {k: v for k, v in dict(transformer.config).items()
                                   if isinstance(v, (int, float, str, bool, type(None), list, tuple))},
        }
        return self.load_info

    def place_vae(self, device) -> None:
        """Move the VAE between runs (never during a run)."""
        device = indexed_device(device)
        if self._vae_device != device:
            self.vae.to(device)
            self.vae.clear_cache()
            self._vae_device = device

    def _resolve_kv_history(self) -> None:
        """Resolve the recipe's KV history options against the layout; raises ``ValueError``."""
        r = self.recipe
        # The GPU history tier lives on the decode GPU; the policy rules are the pipeline's.
        if r.kv_tier and self.gpus < 2:
            raise ValueError(
                f"--kv_tier needs the two-GPU layout: the KV history tier lives on the "
                f"decode GPU, and here the VAE decodes on the DiT GPU ({self.dit_device})")
        pool_gb, scratch_gb = r.kv_block_pool_gb, r.kv_block_scratch_gb
        if r.kv_cache_policy == "block_lru":
            # No explicit budget = the runtime defaults; an explicit 0 pool is rejected.
            pool_gb = DEFAULT_BLOCK_POOL_GB if pool_gb is None else pool_gb
            scratch_gb = DEFAULT_BLOCK_SCRATCH_GB if scratch_gb is None else scratch_gb
        dedup = str(r.kv_block_dedup or "device")
        pool_bytes, scratch_bytes = validate_kv_policy(
            r.kv_cache_policy, dedup, enable_kv_evict=r.enable_kv_evict, tier=bool(r.kv_tier),
            pool=pool_gb, scratch=scratch_gb, finite_window=r.window_size is not None, unit="gb")
        # Page-lock host KV at exactly its size: torch's pinned allocator rounds up to 2**n.
        exact_pinned = (True if not os.environ.get(EXACT_PINNED_KV_ENV)
                        else exact_pinned_kv_enabled())
        self._kv = KVHistory(
            kv_cache_policy=r.kv_cache_policy,
            kv_tier_device=self.decode_device if r.kv_tier else None,
            kv_tier_budget_bytes=int(r.kv_tier_budget_gb * 2 ** 30) if r.kv_tier else 0,
            kv_block_pool_bytes=pool_bytes, kv_block_scratch_bytes=scratch_bytes,
            kv_block_resident_if_fits=bool(r.kv_block_resident_if_fits),
            kv_block_dedup=dedup, kv_exact_pinned_host=exact_pinned)

    def configure(self) -> Dict[str, Any]:
        """Apply the recipe's switches and report the *actual* configuration."""
        from omnimem.models.transformers.causal_wan_nsa_model import (
            set_nsa_attention_schedule, get_nsa_attention_schedule)
        from omnimem.models.transformers.wan_model import set_patch_embed_linear
        from omnimem.nsa import backend as nsa_backend

        # Validate BEFORE touching any global state, so a bad recipe leaves the process
        # exactly as it was.  The recipe may have been replaced since the last request.
        r = self.recipe
        self._check_recipe()
        self._resolve_kv_history()
        check_token_grid(r.height, r.width, r.frame_per_block, dict(self.transformer.config))
        kv = self._kv
        block_policy = kv.kv_cache_policy == "block_lru"
        # Model-side settings the pipeline reads at request time.  (Whether and how the
        # DiT is compiled was decided by ``load``; ``config['compile']`` reports that.)
        self.pipeline.frame_per_block = r.frame_per_block
        set_patch_embed_linear(r.patch_embed_linear)
        # The streaming runtime always runs the two-stream schedule (the model
        # attribute default, which every other caller keeps, is ``serial``).
        n_set = set_nsa_attention_schedule(self.transformer, "dual_topk_first")
        self.place_vae(self.decode_device)

        # The KV cache is created lazily and reset by the pipeline each request; do not reconfigure.
        config = {
            # 1: the VAE shares the DiT GPU and each chunk is decoded in the DiT loop
            # ("sync"); 2: the decoder thread drives the VAE on the decode GPU ("async").
            "gpus": self.gpus, "decode": "sync" if self.gpus == 1 else "async",
            "kv_tier_device": str(kv.kv_tier_device) if kv.kv_tier_device else None,
            "kv_tier_budget_gb": kv.kv_tier_budget_bytes / 2 ** 30,
            "kv_block_pool_gb": kv.kv_block_pool_bytes / 2 ** 30,
            "kv_cache_policy": kv.kv_cache_policy,
            "kv_block_scratch_gb": kv.kv_block_scratch_bytes / 2 ** 30,
            "kv_block_resident_if_fits": kv.kv_block_resident_if_fits,
            "kv_block_dedup": kv.kv_block_dedup,
            "kv_window_buffer": bool(self.recipe.kv_window_buffer),
            "kv_exact_pinned_host": kv.kv_exact_pinned_host,
            "lru_max_size": "n/a (block_lru)" if block_policy else self.recipe.lru_max_size,
            "attention_schedule": get_nsa_attention_schedule(self.transformer),
            "attention_modules_set": n_set,
            "flash_attention": flash_attention_backend(),
            "nsa_backend": nsa_backend.get_backend(),
            "vae_device": str(self._vae_device), "dit_device": str(self.dit_device),
            "vae_decode_fast": self.recipe.vae_decode_fast,
            "compile": self._compiled,
        }
        if block_policy:
            # Real store geometry exists only after the cache is configured; run() fills it.
            config["block_store"] = None
        self._config = config
        return config

    def _sync_block_store_config(self, config: Dict[str, Any]) -> None:
        """Fill in the block store's resolved geometry and device cost, in place."""
        kv = getattr(self.pipeline, "_kv_cache", None)
        if kv is None or getattr(kv, "block_store", None) is None:
            return
        resolved = dict(kv.block_store_config())
        total = _block_store_device_bytes(resolved)
        config["block_store"] = resolved
        config["block_total_gpu_bytes"] = total
        config["block_total_gpu_gb"] = total / 2 ** 30

    # ------------------------------------------------------------------ helpers
    def _pipe_kwargs(self, nframes: int, switch_prompts, switch_frame_indices) -> Dict[str, Any]:
        r = self.recipe
        kw = dict(
            n_prompt=NEG_PROMPT,
            size=(r.width, r.height),
            frame_num=nframes,
            shift=r.shift,
            sample_solver="unipc",
            sampling_steps=r.num_inference_steps,
            guide_scale=1.0,
            offload_model=False,
            sink_size={"kv": r.sink_size, "kv_cmp": None},
            window_size={"kv": r.window_size, "kv_cmp": None},
            enable_chunk_per_head_cache=True,
            enable_kv_evict=r.enable_kv_evict,
            lru_max_size=r.lru_max_size,
            slice_last_frames=r.slice_last_frames,
            recache_mode=r.recache_mode,
            kv_window_buffer=r.kv_window_buffer,
            kv_resident_fast_path=self._kv.kv_cache_policy == "chunk_lru",
            **asdict(self._kv),
        )
        if r.park_t5:
            kw["text_encoder_park_device"] = r.park_t5
        if switch_prompts:
            kw["switch_prompts"] = list(switch_prompts)
            kw["switch_frame_indices"] = list(switch_frame_indices)
        return kw

    def _decoder_streams(self) -> Optional[Tuple[torch.cuda.Stream, torch.cuda.Stream]]:
        """The decoder thread's CUDA streams, one pair per pair of devices."""
        if self.gpus == 1:
            return None
        pair = self._decoder_stream_pair
        if pair is None or (pair[0].device, pair[1].device) != (self.decode_device, self.dit_device):
            pair = self._decoder_stream_pair = decoder_thread_streams(self.decode_device,
                                                                      self.dit_device)
        return pair

    def _make_writer(self, output_path: Optional[str], preview_dir: Optional[str], fragmented: bool,
                     preview_every_frames: Optional[int]) -> StreamingVideoWriter:
        r = self.recipe
        extra = None
        if fragmented:
            # A fragment closes at a keyframe: one per delivered chunk, flushed so it is readable.
            gop = str(4 * r.frame_per_block)
            extra = ["-g", gop, "-keyint_min", gop, "-flush_packets", "1"]
        return StreamingVideoWriter(
            output_path, width=r.width, height=r.height, fps=r.fps, crf=r.crf, preset=r.preset,
            pix_fmt=r.pix_fmt, fragmented=fragmented, preview_dir=preview_dir,
            preview_every_frames=preview_every_frames, extra_ffmpeg_args=extra)

    def _cache_stats_from_marks(self, marks: StageMarks) -> Dict[str, Any]:
        ends = marks.all("chunk.end")
        if not ends:
            return {}
        last = ends[-1][1]
        hit_rates = [e.get("lru_hit_rate") for _, e in ends if e.get("lru_hit_rate") is not None]
        access = sum(e.get("lru_access", 0) for _, e in ends)
        hits = sum(e.get("lru_hits", 0) for _, e in ends)
        # The chunk LRU's whole-chunk reload counters: ``calls``, ``cids``,
        # ``head_chunks``, ``bytes`` (all 0 under 'block_lru').
        out = {k[len("reload_"):]: v for k, v in last.items() if k.startswith("reload_")}
        out.update({
            "lru_hit_rate_mean_per_chunk": float(np.mean(hit_rates)) if hit_rates else None,
            "lru_hit_rate_overall": (hits / access) if access else None,
            "lru_accesses": access, "lru_hits": hits,
        })
        kv = getattr(self.pipeline, "_kv_cache", None)
        if kv is not None and getattr(kv, "_pinned_buffers", None):
            out["kv_pinned_offload_bytes"] = sum(
                t.numel() * t.element_size() for per in kv._pinned_buffers.values() for t in per.values())
        # Offload / ptr-publish counters are read live off the cache.
        off = getattr(kv, "offload_stats", None) if kv is not None else None
        if off:
            out.update({
                "d2h_bytes": off.get("d2h_bytes"),
                "d2h_calls": off.get("d2h_calls"),
                "d2h_skipped_bytes": off.get("skipped_bytes"),
                "d2h_skipped_calls": off.get("skipped_calls"),
                "d2h_slot_reuse": off.get("slot_reuse"),
                "d2h_host_waits": off.get("host_waits"),
            })
        ptr = getattr(kv, "ptr_stats", None) if kv is not None else None
        if ptr:
            out.update({
                "ptr_publishes": ptr.get("publishes"),
                "ptr_entries": ptr.get("entries"),
                "ptr_ring_waits": ptr.get("ring_waits"),
                "ptr_ring_wait_s": ptr.get("ring_wait_s"),
            })
        win = getattr(kv, "window_buf_stats", None) if kv is not None else None
        if win is not None:
            out.update({
                "winbuf_flat_writes": win.get("flat_writes"),
                "winbuf_flat_hits": win.get("flat_hits"),
                "winbuf_ring_writes": win.get("ring_writes"),
                "winbuf_ring_hits": win.get("ring_hits"),
                "winbuf_ring_misses": win.get("ring_misses"),
                "winbuf_bytes": kv.window_buffer_bytes(),
            })
        fp = getattr(kv, "fast_path_stats", None) if kv is not None else None
        if fp is not None:
            out.update({
                "fastpath_skipped": fp.get("skipped"),
                "fastpath_clamp_skipped": fp.get("clamp_skipped"),
            })
        tier = getattr(kv, "tier_stats", None) if kv is not None else None
        if tier is not None:
            out.update({
                "tier_device": str(kv.tier_device) if kv.tier_device is not None else None,
                "tier_budget_bytes": getattr(kv, "tier_budget_bytes", 0),
                "tier_evictions": tier.get("evictions"),
                "tier_bytes": tier.get("bytes"),
                "tier_fallback_host": tier.get("fallback_host"),
                "tier_used_bytes": tier.get("used_bytes"),
                "tier_peak_used_bytes": tier.get("peak_used_bytes"),
                "tier_window_copyback": tier.get("window_copyback"),
                "preallocate_skipped": getattr(kv, "preallocate_skipped", None),
                "preallocate_estimate_bytes": getattr(kv, "preallocate_estimate_bytes", None),
                "preallocate_local_budget_bytes": getattr(
                    kv, "preallocate_local_budget_bytes", None),
            })
        # Block store: every counter is exported with a block_ prefix, plus the derived hit rate.
        block = getattr(kv, "block_stats", None) if kv is not None else None
        if block:
            out.update({f"block_{k}": v for k, v in block.items()})
            # Fast-path calls are excluded from selected_unique by the store.
            out["block_hit_rate"] = kv.block_hit_rate()
            rates = [e.get("block_hit_rate") for _, e in ends if e.get("block_hit_rate") is not None]
            out["block_hit_rate_mean_per_chunk"] = float(np.mean(rates)) if rates else None
            out["kv_cache_policy"] = kv.kv_cache_policy
            # Resolved store geometry: which gather variant actually ran (one fused
            # launch vs the split ones) belongs next to the counters it explains.
            store_cfg = kv.block_store_config()
            for _k in ("persistent_slots_per_layer", "scratch_slots", "max_blocks", "block_bytes",
                      "mode", "dedup", "resident_history_bytes",
                      "resident_budget_bytes"):
                out.setdefault(f"block_{_k}", store_cfg[_k])
            # Per-layer counters (misses, persistent_hits, evictions, h2d_bytes per
            # layer); numpy arrays become plain lists so the manifest stays JSON-able.
            per_layer = kv.block_stats_per_layer()
            if per_layer:
                out["block_per_layer"] = {
                    str(k): (v.tolist() if hasattr(v, "tolist") else list(v))
                    for k, v in per_layer.items()
                }
        return out

    # ------------------------------------------------------------------ run
    def _reset_cache_counters(self) -> None:
        """Zero the request-window counters of the reusable cache."""
        kv = getattr(self.pipeline, "_kv_cache", None)
        if kv is not None:
            kv.reset_reload_stats()
            kv.reset_ptr_stats()

    def run(self, prompt: str, nframes: int, seed: int, *,
            output_path: str, preview_dir: Optional[str] = None,
            switch_prompts: Optional[Sequence[str]] = None,
            switch_frame_indices: Optional[Sequence[int]] = None,
            queue_depth: int = 2, capture: bool = False, fragmented: bool = False,
            preview_every_frames: Optional[int] = None, trace_selection: bool = False) -> RunResult:
        """Generate one video into ``output_path`` and return wall-clock metrics."""
        if self._unusable is not None:
            raise RuntimeError(
                f"this runtime cannot run another request: {self._unusable}.  Build a "
                f"new runtime, preferably in a new process")
        r = self.recipe
        validate_request(nframes, r.frame_per_block, switch_prompts=switch_prompts,
                         switch_frame_indices=switch_frame_indices, queue_depth=queue_depth)
        check_output_target(output_path, preview_dir)
        config = self.configure()
        config["queue_depth"] = queue_depth
        config["fragmented_mp4"] = fragmented
        config["writer"] = {"codec": "libx264", "crf": r.crf, "preset": r.preset, "pix_fmt": r.pix_fmt}
        n_actual = actual_frames(nframes)
        devices = {self.dit_device, self._vae_device}
        if self._kv.kv_tier_device is not None:
            # Peak allocated/reserved of the tier device must be recorded too.
            devices.add(self._kv.kv_tier_device)

        self._reset_cache_counters()
        selection_trace = [] if trace_selection else None
        if trace_selection:
            from omnimem.models.transformers.causal_wan_nsa_model import set_selection_trace
            set_selection_trace(selection_trace)

        for d in devices:
            torch.cuda.synchronize(d)
            torch.cuda.reset_peak_memory_stats(d)
        marks = StageMarks()
        generator = torch.Generator(device=self.dit_device).manual_seed(seed)
        pipe_kwargs = self._pipe_kwargs(nframes, switch_prompts, switch_frame_indices)
        captured_latents: List[torch.Tensor] = []
        t_req = time.monotonic()
        marks.t0 = t_req
        marks.mark("request.begin")

        frames_np: Optional[np.ndarray] = None
        session: Optional[StreamingSession] = None
        try:
            writer = self._make_writer(output_path, preview_dir, fragmented, preview_every_frames)
            sink = VideoFileSink(writer, capture=capture)
            decoder = WanChunkDecoder(
                self.vae, self._vae_device, autocast_dtype=torch.bfloat16,
                fast=r.vae_decode_fast)
            session = StreamingSession(
                decoder=decoder, decode_device=self._vae_device, dit_device=self.dit_device,
                sink=sink, frame_hw=(r.height, r.width),
                max_frames_per_chunk=4 * r.frame_per_block, queue_depth=queue_depth,
                decoder_streams=self._decoder_streams(), marks=marks)
            session.start()
            marks.mark("session.ready")

            def on_chunk(ch: LatentChunk):
                if capture:
                    captured_latents.append(ch.latent.clone())
                session.submit(ch)

            # Like evaluation_wan and inference_mp.py: the text encoder runs under bf16 autocast.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.pipeline.generate(input_prompt=prompt, generator=generator, marks=marks,
                                             on_latent_chunk=on_chunk, return_latents=True,
                                             **pipe_kwargs)
            marks.mark("generate.return")
            assert out.num_chunks == session._next_submit, (out.num_chunks, session._next_submit)
            session_result = session.finish()
            sink_result = session_result.sink_result
            marks.mark("writer_done")
            frames_delivered = session_result.frames_delivered
            if capture and sink.captured:
                frames_np = np.concatenate(sink.captured, axis=0)
            preview_paths = list(writer.preview_paths)
        except BaseException as exc:
            if session is not None:
                # Idempotent: stops the workers and kills the encoder wherever the
                # request failed (start, sampler, submit, finish).
                session.abort(exc)
                if session.stuck_workers:
                    # A decoder thread that wakes up later clears the VAE's causal
                    # cache -- under whatever request would be running by then.
                    self._unusable = (
                        f"the {' and the '.join(session.stuck_workers)} thread of a failed "
                        f"request did not stop within {session.join_timeout:.0f} s and "
                        f"still holds what it was working with (the decoder thread: the "
                        f"VAE)")
            raise
        finally:
            if trace_selection:
                from omnimem.models.transformers.causal_wan_nsa_model import set_selection_trace
                set_selection_trace(None)

        t_end = time.monotonic()
        for d in devices:
            torch.cuda.synchronize(d)

        # The cache is created lazily, so the resolved store geometry exists only now.
        self._sync_block_store_config(config)

        if frames_delivered != n_actual:
            raise RuntimeError(f"delivered {frames_delivered} frames, expected {n_actual}")

        # ---- timings (all relative to request start, one monotonic clock)
        rel = lambda t: None if t is None else t - t_req  # noqa: E731
        deliveries = marks.all("frames_delivered")
        delivery_times = [t for t, _ in deliveries]
        intervals = [b - a for a, b in zip(delivery_times[:-1], delivery_times[1:])]
        first_latent_submit = marks.first("latent.submitted")
        recs = session_result.chunk_records
        first_rec = recs[0] if recs else {}
        span = lambda name: ((marks.first(f"{name}.end") or 0)  # noqa: E731
                             - (marks.first(f"{name}.begin") or 0))
        timings = {
            # Moving a parked text encoder back to the DiT GPU, then the encoding itself.
            "text_encoder_reload_s": span("text_encoder.reload"),
            "text_encode_s": span("text_encode"),
            "session_ready_s": rel(marks.first("session.ready")),
            "first_latent_submit_s": rel(first_latent_submit),
            "first_latent_gpu_ready_s": rel(first_rec.get("gpu_ready")),
            "first_decode_done_s": rel(first_rec.get("decode_done")),
            "first_d2h_done_s": rel(first_rec.get("d2h_done")),
            "first_frames_delivered_s": rel(delivery_times[0]) if delivery_times else None,
            "first_preview_png_s": rel(writer.first_frame_time),
            "denoise_seconds_reported": (marks.all("denoise.end") or [(None, {})])[0][1].get("seconds"),
            "denoise_done_s": rel(marks.first("denoise.end")),
            "generate_return_s": rel(marks.first("generate.return")),
            "decoder_drained_s": rel(session_result.timeline.get("decoder_drained")),
            "sink_finished_s": rel(marks.first("sink_finished")),
            "writer_done_s": rel(marks.first("writer_done")),
            "total_s": t_end - t_req,
            "delivery_batches": len(deliveries),
            "first_batch_frames": int(deliveries[0][1].get("frames", 0)) if deliveries else 0,
            "delivery_interval_p50_s": _percentile(intervals, 50),
            "delivery_interval_p95_s": _percentile(intervals, 95),
            "delivery_interval_max_s": max(intervals) if intervals else None,
            "fps_total": n_actual / (t_end - t_req),
            "fps_after_first_delivery": ((n_actual - (int(deliveries[0][1].get("frames", 0)) if deliveries else 0))
                                         / (t_end - delivery_times[0])) if delivery_times and t_end > delivery_times[0] else None,
            "ffmpeg_seconds": getattr(sink_result, "seconds", None),
        }
        memory = {
            f"{str(d)}_peak_alloc_gb": torch.cuda.max_memory_allocated(d) / 2 ** 30 for d in devices
        }
        memory.update({f"{str(d)}_peak_reserved_gb": torch.cuda.max_memory_reserved(d) / 2 ** 30 for d in devices})
        memory.update(_rss())
        memory["d2h_pinned_pool_bytes"] = session_result.d2h_pool_bytes
        queues = session_result.queue_stats
        cache_stats = self._cache_stats_from_marks(marks)

        result = RunResult(
            prompt=prompt, seed=seed, requested_frames=nframes, actual_frames=n_actual,
            frames_delivered=frames_delivered, output_path=getattr(sink_result, "path", output_path),
            timings=timings, memory=memory, queues=queues, cache_stats=cache_stats, config=config,
            preview_paths=preview_paths, chunk_records=[{k: (rel(v) if k != "chunk_index" and k != "frames" else v)
                                                        for k, v in rec.items()} for rec in recs],
            marks=marks.relative(t_req),
            generator_state_hash=_hash_tensor(generator.get_state()),
        )
        if capture:
            if captured_latents:
                result.latents = torch.cat([c.float().cpu() for c in captured_latents], dim=2)
            result.frames = frames_np
            result.selection_trace = selection_trace
        return result


def _hash_tensor(t: torch.Tensor) -> str:
    import hashlib
    return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def compare_latents(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    if a.shape != b.shape:
        return {"exact": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    diff = (a.double() - b.double()).abs()
    first = None
    if not torch.equal(a, b):
        nz = torch.nonzero(diff > 0)
        first = [int(x) for x in nz[0]] if len(nz) else None
    return {"exact": bool(torch.equal(a, b)), "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "first_mismatch_index": first, "shape": list(a.shape)}


def compare_frames(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    if a.shape != b.shape:
        return {"exact": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    psnr = float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
    per_frame_max = d.reshape(d.shape[0], -1).max(axis=1)
    first = int(np.argmax(per_frame_max > 0)) if per_frame_max.max() > 0 else None
    return {"exact": bool(d.max() == 0), "max_abs": int(d.max()), "mean_abs": float(d.mean()),
            "psnr_db": psnr, "first_mismatch_frame": first, "frames": int(a.shape[0])}


def git_info(repo: str) -> Dict[str, str]:
    def _run(cmd):
        try:
            return subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"<{exc}>"
    return {"head": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"])}
