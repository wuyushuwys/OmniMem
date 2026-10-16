# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import contextlib
import math
import time
import gc
from collections.abc import MutableMapping
from typing import Optional, Union
from contextlib import contextmanager

import PIL
import torch
import torch.cuda.amp as amp
from torchvision.transforms.v2.functional import to_tensor

from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from omnimem.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler

from omnimem.models.cache import MMCache
from omnimem.models.transformers.causal_wan_model import CausalWanModel
from omnimem.models.transformers.causal_wan_nsa_model import CausalWanNSAModel, set_selection_trace_context
from omnimem.models.transformers.wan_model import patch_embed_linear_enabled
from omnimem.models.autoencoders import get_autoencoder
from omnimem.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from omnimem.schedulers import RectifiedFlowScheduler
from omnimem.pipelines import WanTI2VPipeline
from omnimem.utils.train_utils import pred_x0
from omnimem.pipelines.utils import masks_like
from omnimem.utils.train_utils import vae_encode
from omnimem.utils.logging_tool import get_logger
from omnimem.utils.validation_prompts import latent_frames
from omnimem.models.text_encoders.t5 import T5EncoderModel
from omnimem.pipelines import compile_state
from omnimem.pipelines.streaming_decode import (
    LatentChunk,
    LatentGenerationOutput,
    decode_latents_chunked_to_cpu,
)
from omnimem.utils.stage_marks import StageMarks
from omnimem.utils.torch_utils import unsharded_param_dtype

logger = get_logger()


def _hashable(value):
    """``value`` (nested dicts / lists / sets of plain values) as a hashable tuple tree."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _hashable(v)) for k, v in value.items()))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(str(v) for v in value))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)        # torch.device, torch.dtype


def forward_positions(num_frames: int, frame_per_block: int, switch_frames,
                      slice_last_frames: int, recache_mode: str) -> list:
    """``(start_frame, n_frames)`` of the DiT forwards of one request, in call order."""
    positions = []
    switch_frames = list(switch_frames)
    switch_idx = 0
    for start_frame in range(0, num_frames, frame_per_block):
        while switch_idx < len(switch_frames) and start_frame >= switch_frames[switch_idx]:
            if recache_mode != "none":
                num_recache = min(slice_last_frames, start_frame)
                recache_start = start_frame - num_recache
                positions.extend(
                    (recache_start + blk, min(frame_per_block, num_recache - blk))
                    for blk in range(0, num_recache, frame_per_block))
            switch_idx += 1
        positions.append((start_frame, min(frame_per_block, num_frames - start_frame)))
    return positions


# How a KV option is named in an error: as ``generate()`` takes it, or as the streaming CLI spells it.
_KV_OPTION_NAMES = {
    "bytes": dict(tier="kv_tier_device", pool="kv_block_pool_bytes",
                  scratch="kv_block_scratch_bytes", scale=1, spell="{}={}"),
    "gb": dict(tier="--kv_tier", pool="--kv_block_pool_gb",
               scratch="--kv_block_scratch_gb", scale=2 ** 30, spell="{} {}"),
}


def validate_kv_policy(policy, dedup, *, enable_kv_evict, tier, pool, scratch,
                       finite_window=True, per_head_cache=True, unit="bytes"):
    """``ValueError`` unless the KV options make one cache policy; returns the block budgets, given in ``unit``, in bytes."""
    names = _KV_OPTION_NAMES[unit]
    if policy not in ("chunk_lru", "block_lru"):
        raise ValueError(
            f"unknown kv_cache_policy {policy!r}; expected 'chunk_lru' or 'block_lru'")
    # Validated under every policy so a typo is rejected early.
    if dedup not in ("unique", "device"):
        raise ValueError(f"unknown kv_block_dedup {dedup!r}; expected 'unique' or 'device'")
    if policy == "chunk_lru":
        # The chunk LRU has no block pools: reject a budget rather than ignore it.
        for name, value, kind in ((names["pool"], pool, "persistent"),
                                  (names["scratch"], scratch, "scratch")):
            if value:
                raise ValueError(
                    f"{names['spell'].format(name, value)} needs kv_cache_policy='block_lru' "
                    f"(it is the block store's {kind} pool budget); the chunk-level policy "
                    f"'chunk_lru' has no block pool")
        return 0, 0
    # One policy per cache: the store owns the whole history path.
    if tier:
        raise ValueError(
            f"kv_cache_policy='block_lru' and {names['tier']} are mutually exclusive: the GPU "
            f"tier never writes the pinned host backing the block store fetches blocks from")
    # ``window_size=None`` is dense over all history, which the store would have to keep resident.
    if not finite_window:
        raise ValueError(
            "kv_cache_policy='block_lru' needs a finite KV window (window_size=None is "
            "dense over the whole history)")
    if not enable_kv_evict:
        raise ValueError(
            "kv_cache_policy='block_lru' needs enable_kv_evict=True: history retirement at "
            "every generation-chunk boundary replaces the chunk LRU, and without it the whole "
            "history stays protected on the DiT GPU")
    pool_bytes = int(pool * names["scale"])
    if pool_bytes <= 0:
        raise ValueError(
            f"kv_cache_policy='block_lru' needs {names['pool']} > 0 (it is the persistent "
            f"block pool budget, total across layers)")
    scratch_bytes = int(scratch * names["scale"])
    if scratch_bytes < 0:
        raise ValueError(f"{names['scratch']} must be >= 0")
    if not per_head_cache:
        raise ValueError(
            "kv_cache_policy='block_lru' requires the chunk_per_head K/V cache "
            "(enable_chunk_per_head_cache=True)")
    return pool_bytes, scratch_bytes


def _snapshot_chunks(cache: MMCache, chunk_start: int, chunk_end: int) -> dict:
    """Save references to cache chunks [chunk_start, chunk_end) for all layers, with versions."""
    snap = {}
    cids = range(chunk_start, chunk_end)
    for layer_idx, layer_cache in cache.cache.items():
        snap[layer_idx] = {}
        for cache_type, chunks in layer_cache.items():
            if not isinstance(chunks, list):
                continue
            head_major = cache_type in cache.chunk_per_head_types
            versions = None
            if head_major:
                entry = [[head[i] if i < len(head) else None for i in cids]
                         for head in chunks]
                versions = [[cache.chunk_version(layer_idx, cache_type, h, i) for i in cids]
                            for h in range(len(chunks))]
            else:
                entry = [chunks[i] if i < len(chunks) else None for i in cids]
                if cache.window_buffer_active(cache_type):
                    # The entry is a view the recache forward writes through, so clone it.
                    entry = [None if t is None else t.clone() for t in entry]
            snap[layer_idx][cache_type] = (head_major, chunk_start, entry, versions)
    if cache.block_store is not None:
        # 'block_lru': the store's own state; capturing it also freezes the host rows of the range.
        snap['block_store'] = cache.block_snapshot_range(chunk_start, chunk_end)
    return snap


def _restore_chunks(cache: MMCache, snapshot: dict):
    """Put the snapshotted chunks back and re-publish their pointers."""
    block_snap = snapshot.get('block_store')
    for layer_idx, layer_snap in snapshot.items():
        if layer_idx == 'block_store':
            continue
        ptr_entries = []
        for cache_type, (head_major, start, entry, versions) in layer_snap.items():
            chunks = cache.cache[layer_idx][cache_type]
            if head_major:
                for h, snap_head in enumerate(entry):
                    if h >= len(chunks):
                        continue
                    head = chunks[h]
                    for offset, snap_chunk in enumerate(snap_head):
                        idx = start + offset
                        if snap_chunk is None or idx >= len(head):
                            continue
                        head[idx] = snap_chunk
                        cache.set_chunk_version(layer_idx, cache_type, h, idx,
                                                versions[h][offset])
                        if block_snap is None:
                            ptr_entries.append(
                                (cache_type, h, idx, MMCache.chunk_ptr_of(snap_chunk)))
                            # GPU tier: recache replaced this slot with a local chunk; restore it.
                            cache.note_tier_entry(layer_idx, cache_type, h, idx,
                                                  snap_chunk)
                        # Window ring goes stale for this cid; the next build falls back to cat.
                        cache.invalidate_window_ring(layer_idx, idx, cache_type)
            else:
                for offset, snap_chunk in enumerate(entry):
                    idx = start + offset
                    if snap_chunk is not None and idx < len(chunks):
                        cache.restore_flat_chunk(layer_idx, cache_type, idx, snap_chunk)
        if ptr_entries:
            cache.publish_chunk_ptrs(layer_idx, ptr_entries)
    if block_snap is not None:
        # 'block_lru': one call per layer restores block addresses, versions and flags.
        cache.block_restore_range(block_snap)


class CausalWanT2VPipeline(WanTI2VPipeline):

    def __init__(
            self,
            text_encoder: T5EncoderModel,
            vae: AutoencoderKLWan,
            transformer: Union[CausalWanModel, CausalWanNSAModel],
            tokenizer: None,
            scheduler: FlowUniPCMultistepScheduler,
            frame_per_block: Optional[int] = None,
    ):
        super().__init__(
            text_encoder,
            vae,
            transformer,
            tokenizer,
            scheduler
        )

        self.frame_per_block = frame_per_block
        self._kv_cache = None
        self._kv_cache_uncond = None
        self._cache_signature = None

    @classmethod
    def from_checkpoints(cls, pretrained, vae_path, t5_path, tokenizer_path, *, shift,
                         frame_per_block, model_cls=CausalWanNSAModel, timings=None):
        """Pipeline of frozen models on the CPU (VAE, umT5 in bf16, DiT); ``timings`` gets each load's seconds."""
        t0 = time.monotonic()
        vae = get_autoencoder(vae_path)
        t1 = time.monotonic()
        text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=torch.device("cpu"),
                                      checkpoint_path=t5_path, tokenizer_path=tokenizer_path,
                                      shard_fn=None)
        t2 = time.monotonic()
        transformer = model_cls.from_pretrained(pretrained)
        for model in (vae, text_encoder.model, transformer):
            model.requires_grad_(False).eval()
        if timings is not None:
            timings.update(load_vae_s=t1 - t0, load_t5_s=t2 - t1, load_dit_s=time.monotonic() - t2)
        return cls(transformer=transformer, tokenizer=text_encoder.tokenizer,
                   text_encoder=text_encoder, vae=vae,
                   scheduler=FlowUniPCMultistepScheduler(shift=shift),
                   frame_per_block=frame_per_block)

    def _compiled_request(self, *, dit_device, latent_shape, window_size, sink_size,
                          image_conditioned, switch_frames, slice_last_frames, recache_mode):
        """Context of one compiled-DiT request (``compiled_request``); no-op if not compiled."""
        compiled_forward = getattr(self.transformer, "_compiled_call_impl", None)
        if compiled_forward is None:
            return contextlib.nullcontext()
        matmul = torch.backends.cuda.matmul
        signature = (
            id(compiled_forward),
            type(self.transformer).__qualname__, _hashable(dict(self.transformer.config)),
            self.transformer.training,
            str(self.param_dtype), str(dit_device),
            (latent_shape[0], latent_shape[2], latent_shape[3]),
            tuple(m.attention_schedule for m in self.transformer.modules()
                  if hasattr(m, "attention_schedule")),
            patch_embed_linear_enabled(),
            _hashable(window_size), _hashable(sink_size),
            bool(image_conditioned),
            (torch.get_num_threads(), str(torch.get_default_dtype()),
             matmul.allow_tf32, matmul.allow_bf16_reduced_precision_reduction,
             matmul.allow_fp16_reduced_precision_reduction,
             torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled()),
        )
        return compile_state.compiled_request(signature, forward_positions(
            latent_shape[1], self.frame_per_block, switch_frames, slice_last_frames,
            recache_mode))

    def generate(
            self,
            input_prompt,
            size=(1280, 720),
            frame_num=81,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            generator=None,
            offload_model=False,
            sink_size=None,
            window_size=None,
            img: Optional[PIL.Image.Image] = None,
            lru_max_size=10,
            enable_chunk_per_head_cache=False,
            enable_kv_evict=True,
            switch_prompts=None,
            switch_frame_indices=None,
            slice_last_frames=21,
            recache_mode="full",
            vae_decode_chunk_size=None,
            evict_every_k_chunks=1,
            _kv_cache: Optional[MMCache] = None,
            on_latent_chunk=None,
            return_latents=False,
            marks: Optional[StageMarks] = None,
            text_encoder_park_device=None,
            kv_tier_device=None,
            kv_tier_budget_bytes: int = 0,
            kv_block_pool_bytes: int = 0,
            kv_cache_policy: str = "chunk_lru",
            kv_block_scratch_bytes: int = 0,
            kv_block_resident_if_fits: bool = True,
            kv_block_dedup: str = "device",
            kv_window_buffer: bool = False,
            kv_resident_fast_path: bool = False,
            kv_exact_pinned_host: Optional[bool] = None,
    ):
        """Generate video frames from a text prompt, chunk-by-chunk with KV cache.

        Args:
            input_prompt: Text prompt.
            size: Video resolution (width, height).
            frame_num: Number of frames.
            shift: Noise schedule shift.
            sample_solver: Solver type.
            sampling_steps: Number of denoising steps.
            guide_scale: Classifier-free guidance scale.
            n_prompt: Negative prompt.
            offload_model: Offload models to CPU to save VRAM.
            on_latent_chunk: optional callback receiving a LatentChunk right after each chunk's final latents.
            return_latents: skip the final VAE decode and return a LatentGenerationOutput.
            marks: optional StageMarks recorder for coarse per-stage CPU timestamps.
            text_encoder_park_device: if set, park the text encoder on this device after encoding all prompts.
            kv_tier_device: opt-in peer CUDA device holding a GPU history tier; None keeps the pinned host path.
            kv_tier_budget_bytes: byte budget for the tier on that device.
            kv_block_pool_bytes: 'block_lru' only, persistent block pool budget; must stay 0 under 'chunk_lru'.
            kv_cache_policy: 'chunk_lru' (default) or 'block_lru' (block store with retirement per chunk).
            kv_block_scratch_bytes: 'block_lru' only, budget of the shared scratch block pool.
            kv_block_resident_if_fits: 'block_lru' only, run resident when the whole request fits. Default True.
            kv_block_dedup: 'block_lru' only, 'device' (default) or 'unique' dedup of selected block keys.
            evict_every_k_chunks: 'chunk_lru' only, chunks between flush_offload() drains.
            kv_window_buffer: opt-in preallocated history and window buffers, bit-exact.
            kv_resident_fast_path: opt-in, skip LRU touch and host sync while nothing is host-resident ('chunk_lru' only).
            kv_exact_pinned_host: page-lock exactly the bytes of each host KV buffer; None follows OMNIMEM_EXACT_PINNED_KV.
        """
        validate_kv_policy(kv_cache_policy, kv_block_dedup, enable_kv_evict=enable_kv_evict,
                           tier=kv_tier_device is not None, pool=kv_block_pool_bytes,
                           scratch=kv_block_scratch_bytes,
                           per_head_cache=enable_chunk_per_head_cache)
        block_policy = kv_cache_policy == "block_lru"
        if return_latents and offload_model:
            raise ValueError("return_latents=True does not support offload_model=True "
                             "(the streaming path never moves the VAE during generation)")

        # All DiT-side placement (text context, noise, timesteps, KV cache) follows the
        # transformer's device explicitly; the VAE may live on another device.
        dit_device = self.transformer.device

        # preprocess
        F = frame_num
        target_shape = (self.vae.z_dim, latent_frames(F, self.vae_stride[0]),
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        input_prompt = self._text_preprocessing(input_prompt)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if marks is not None:
            marks.mark("text_encoder.reload.begin")
        self.text_encoder.model.to(dit_device)
        if marks is not None:
            marks.mark("text_encoder.reload.end")
            marks.mark("text_encode.begin")
        context, attn_cond = self.text_encoder(input_prompt, dit_device)
        if guide_scale > 1:
            context_null, attn_null = self.text_encoder(n_prompt, dit_device)
        else:
            # CFG is a no-op at guide_scale<=1 (pred_image_cond is used directly in the
            # sampler loop, see below) -- skip the negative-prompt T5 encode entirely.
            context_null, attn_null = None, None
        _main_p = input_prompt[0] if isinstance(input_prompt, (list, tuple)) else input_prompt
        logger.info(f"[prompt] MAIN (start_frame=0): {_main_p}")
        # Encode all switch prompts upfront
        if switch_prompts is not None and switch_frame_indices is not None:
            if isinstance(switch_prompts, str):
                switch_prompts = [switch_prompts] * len(switch_frame_indices)
            contexts_switch = []
            for sp in switch_prompts:
                ctx, _ = self.text_encoder(sp, dit_device)
                contexts_switch.append(ctx)
            _sfi_sorted = sorted(switch_frame_indices)
            for i, sp in enumerate(switch_prompts):
                _sf = _sfi_sorted[i] if i < len(_sfi_sorted) else None
                logger.info(f"[prompt] SWITCH#{i} (triggers at start_frame>={_sf}): {sp}")
        else:
            contexts_switch = []
        if marks is not None:
            marks.mark("text_encode.end")
        if text_encoder_park_device is not None and not offload_model:
            # Opt-in: free the encoder's ~11 GB on the DiT device for KV residency once all
            # prompts (incl. switch prompts) are encoded; it is moved back at the next call.
            self.text_encoder.model.to(text_encoder_park_device)
            gc.collect()
            torch.cuda.empty_cache()
        if offload_model:
            self.text_encoder.model.cpu()

        # prepare image condition
        if img is not None:
            assert isinstance(img, PIL.Image.Image), f"{img.type=} is not pillow image"
            # reshape
            w, h = img.size
            target_w, target_h = size
            assert w <= target_w, f"{w=} > {size[0]=}"
            assert h <= target_h, f"{h=} > {size[1]=}"
            if w != target_w or h != target_h:
                img = img.resize(size)
            img_tensor = to_tensor(img)[None, :, None] * 2 - 1
            self.vae.encoder.to(dit_device)
            img_tensor = img_tensor.to(device=dit_device, dtype=self.param_dtype)
            img_latent = vae_encode(vae=self.vae, pixel_values=img_tensor)
            if offload_model:
                self.vae.encoder.cpu()
                torch.cuda.empty_cache()
        else:
            img_latent = None

        # Move full VAE (encoder + decoder) to CPU during the diffusion loop —
        # the transformer never touches it. Brought back before final decode.
        if offload_model:
            self.vae.cpu()
            torch.cuda.empty_cache()

        noise = torch.randn(
            1,
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=dit_device,
            generator=generator)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.transformer, 'no_sync', noop_no_sync)

        assert self.frame_per_block is not None, f"got invalid {self.frame_per_block=}"
        compiled_request = self._compiled_request(
            dit_device=dit_device, latent_shape=target_shape, window_size=window_size,
            sink_size=sink_size, image_conditioned=img_latent is not None,
            switch_frames=(sorted(switch_frame_indices)[:len(contexts_switch)]
                           if contexts_switch else ()),
            slice_last_frames=slice_last_frames, recache_mode=recache_mode)

        # The DiT device must be current: Triton launches on the current device.
        with torch.cuda.device(dit_device), amp.autocast(dtype=self.param_dtype), \
                torch.no_grad(), no_sync(), compiled_request:
            sample_scheduler = RectifiedFlowScheduler(num_train_timesteps=self.num_train_timesteps, shift=shift)
            sample_scheduler.set_timesteps(num_inference_steps=sampling_steps, device=dit_device, )
            timesteps = sample_scheduler.timesteps
            # Python floats for the trace tags; float(t) on a CUDA tensor per step would sync.
            timesteps_list = timesteps.tolist()

            # sample videos
            latents = noise
            mask1, mask2 = masks_like(noise, zero=True)
            if img_latent is not None:
                # replace first frame to image
                latents = (1. - mask1) * img_latent + mask1 * latents

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}
            args_c_switch = [{'context': ctx, 'seq_len': seq_len} for ctx in contexts_switch]

            frame_per_block = self.frame_per_block
            batch_size, _, num_frames, height, width = noise.shape
            seq_len_per_frame = height * width // (self.patch_size[1] * self.patch_size[2])

            cache_shape = (
                batch_size,
                seq_len,
                self.transformer.num_heads,
                self.transformer.dim // self.transformer.num_heads
            )

            kv_dtype = unsharded_param_dtype(self.transformer)
            cache_kwargs = dict(
                config=dict(
                    cache_type=["k_cache", 'v_cache'],
                    k_cache=dict(),
                    v_cache=dict(),
                ),
                seq_dim=1,
                available_shape=dict(
                    k_cache=cache_shape,
                    v_cache=cache_shape,
                ),
                device=self.transformer.device,
                dtype=kv_dtype,
                block_seqlen_config=dict(
                    k_cache=seq_len_per_frame * frame_per_block,
                    v_cache=seq_len_per_frame * frame_per_block,
                ),
                window_config=dict(
                    k_cache=window_size if not isinstance(window_size, MutableMapping) else window_size.get("kv"),
                    v_cache=window_size if not isinstance(window_size, MutableMapping) else window_size.get("kv"),
                ),
                sink_config=dict(
                    k_cache=sink_size if not isinstance(sink_size, MutableMapping) else sink_size.get("kv"),
                    v_cache=sink_size if not isinstance(sink_size, MutableMapping) else sink_size.get("kv"),
                ),
                num_layers=self.transformer.num_layers,
                chunk_per_head_types={'k_cache', 'v_cache'} if enable_chunk_per_head_cache else None,
                num_heads=self.transformer.num_heads if enable_chunk_per_head_cache else None,
                kv_block_size=(getattr(self.transformer.config, 'pool_f', 1)
                            * self.transformer.config.pool_h
                            * self.transformer.config.pool_w)
                            if enable_chunk_per_head_cache else 0,
            )

            if self.transformer.config.get("enable_nsa"):
                compress_seq_len_per_frame = seq_len_per_frame // (self.transformer.config.pool_w * self.transformer.config.pool_h)
                compress_seq_len = seq_len // (self.transformer.config.pool_w * self.transformer.config.pool_h)
                cache_compress_shape = (
                    batch_size,
                    compress_seq_len,
                    self.transformer.num_heads,
                    self.transformer.dim // self.transformer.num_heads
                )

                for cache_type in ["k_cmp_cache", "v_cmp_cache"]:
                    cache_kwargs['config']['cache_type'].append(cache_type)
                    cache_kwargs['config'][cache_type] = dict()
                    cache_kwargs['available_shape'][cache_type] = cache_compress_shape
                    cache_kwargs["block_seqlen_config"][cache_type] = compress_seq_len_per_frame * frame_per_block
                    cache_kwargs["window_config"][cache_type] = window_size if not isinstance(window_size, MutableMapping) else window_size.get("kv_cmp")
                    cache_kwargs["sink_config"][cache_type] = sink_size if not isinstance(sink_size, MutableMapping) else sink_size.get("kv_cmp")

            # 'block_lru' owns capacity and eviction itself: the chunk LRU must never
            # engage, so the cache is built with capacity 0.
            cache_lru = 0 if block_policy else lru_max_size
            # CFG keeps a second cache for the unconditional branch.
            needs_twin = guide_scale > 1
            # Everything a cache is built from; a change gets new caches.
            cache_signature = (
                _hashable(cache_kwargs), cache_lru, needs_twin,
                kv_cache_policy, int(kv_block_pool_bytes), int(kv_block_scratch_bytes),
            )
            if _kv_cache is not None:
                # External cache: caller owns lifecycle; reset to wipe prior state.
                assert guide_scale <= 1, (
                    "External _kv_cache currently only supports guide_scale <= 1; "
                    "CFG (guide_scale > 1) needs a separate uncond cache, not yet "
                    "plumbed through this kwarg."
                )
                _kv_cache.reset()
                kv_cache, kv_cache_uncond = _kv_cache, None
            else:
                if self._kv_cache is not None and self._cache_signature == cache_signature:
                    self._kv_cache.reset()
                    if needs_twin:
                        self._kv_cache_uncond.reset()
                else:
                    if self._kv_cache is not None:
                        logger.info("[cache] signature changed, rebuilding")
                        # Release old caches before allocating new ones; tracer cycles need a gc.
                        self._kv_cache = self._kv_cache_uncond = None
                        self._cache_signature = None
                        gc.collect()
                        torch.cuda.empty_cache()
                    self._kv_cache = MMCache(**cache_kwargs, lru_max_size=cache_lru)
                    if needs_twin:
                        self._kv_cache_uncond = MMCache(**cache_kwargs, lru_max_size=cache_lru)
                    self._cache_signature = cache_signature
                kv_cache, kv_cache_uncond = self._kv_cache, self._kv_cache_uncond
            caches = [c for c in (kv_cache, kv_cache_uncond) if c is not None]
            # Where host KV buffers come from; an externally owned cache keeps its own setting.
            if _kv_cache is None or kv_exact_pinned_host is not None:
                for _c in caches:
                    _c.configure_exact_pinned_host(kv_exact_pinned_host)
            # ONE budget on the tier device: the two caches of a CFG request share it.
            tier_budget_per_cache = int(kv_tier_budget_bytes) // len(caches)

            # Tier and block store are re-configured every request; disable the unwanted one first.
            if _kv_cache is None or kv_tier_device is not None or block_policy:
                for _c in caches:
                    if block_policy:
                        _c.configure_gpu_tier(None)
                        _c.configure_block_store(
                            int(kv_block_pool_bytes), int(kv_block_scratch_bytes),
                            resident_if_fits=bool(kv_block_resident_if_fits),
                            dedup=kv_block_dedup)
                        continue
                    _c.configure_block_store(0, 0)
                    if kv_tier_device is None:
                        _c.configure_gpu_tier(None)
                    else:
                        _c.configure_gpu_tier(kv_tier_device, tier_budget_per_cache)

            # Window buffers and fast path are re-applied every request; off for block_lru.
            if _kv_cache is None or kv_window_buffer or kv_resident_fast_path:
                _fast = bool(kv_resident_fast_path) and not block_policy
                for _c in caches:
                    _c.configure_window_buffers(bool(kv_window_buffer))
                    _c.configure_resident_fast_path(_fast)

            if _kv_cache is None and enable_kv_evict and enable_chunk_per_head_cache:
                total_chunks = math.ceil(num_frames / frame_per_block)
                chunk_len_for_kv = seq_len_per_frame * frame_per_block
                head_dim = self.transformer.dim // self.transformer.num_heads
                _sample = torch.empty(
                    (batch_size, chunk_len_for_kv, head_dim),
                    dtype=kv_dtype,
                    device=self.transformer.device,
                )
                # KV bytes this request produces per cache, and the part held by local residency.
                _per_head_chunk_bytes = _sample.numel() * _sample.element_size()
                _n_names = max(1, len(kv_cache.chunk_per_head_types))
                _est_bytes = (total_chunks * self.transformer.num_layers * _n_names
                              * self.transformer.num_heads * _per_head_chunk_bytes)
                _local_budget = (lru_max_size * self.transformer.num_layers
                                 * self.transformer.num_heads * 2 * _per_head_chunk_bytes)
                if block_policy:
                    # 'block_lru': pinned host backing is chunk-major, one [H,1,S,D] row per cid.
                    logger.info("[preallocate] === START (block_lru host backing) ===")
                    for _c in caches:
                        _c.preallocate_estimate_bytes = int(_est_bytes)
                        _c.preallocate_local_budget_bytes = int(_local_budget)
                        # Resident mode allocates no pool, scratch or host backing; skip prealloc.
                        _store_cfg = _c.block_store_config()
                        _resident = _store_cfg.get("mode") == "resident"
                        _c.preallocate_skipped = _resident
                        if _resident:
                            logger.info(
                                f"[preallocate] SKIPPED (resident mode: no pool/scratch/"
                                f"host backing; host_backing_bytes="
                                f"{_store_cfg.get('host_backing_bytes')})")
                            continue
                        _c.preallocate_block_host(sample=_sample, segment_cids=total_chunks)
                    logger.info("[preallocate] === DONE (block_lru host backing) ===")
                    del _sample
                else:
                    # If the whole KV fits the resident part plus the tier, skip the page-lock.
                    _fits = (kv_tier_device is not None
                             and _est_bytes <= _local_budget + tier_budget_per_cache)
                    for _c in caches:
                        _c.preallocate_skipped = bool(_fits)
                        _c.preallocate_estimate_bytes = int(_est_bytes)
                        _c.preallocate_local_budget_bytes = int(_local_budget)
                    if _fits:
                        logger.info(
                            f"[preallocate] SKIPPED: KV estimate {_est_bytes / 2**30:.1f} GiB fits "
                            f"local {_local_budget / 2**30:.1f} GiB + tier "
                            f"{tier_budget_per_cache / 2**30:.1f} GiB on {kv_tier_device}")
                    else:
                        logger.info("[preallocate] === START ===")
                        for _c in caches:
                            _c.preallocate_offload_buffer(sample=_sample,
                                                          segment_cids=total_chunks)
                        logger.info("[preallocate] === DONE ===")
                    del _sample

            gate_data = {}
            switch_idx = 0
            switch_frame_indices_sorted = sorted(switch_frame_indices) if switch_frame_indices else []
            # Chunks since the last offload flush of the cond / uncond cache ('chunk_lru').
            chunks_since_evict = [0, 0]

            def release_history(i, cache):
                """'block_lru' retires the completed history; 'chunk_lru' flushes evictions every evict_every_k_chunks."""
                if block_policy:
                    cache.retire_completed_history()
                    return
                chunks_since_evict[i] += 1
                if enable_kv_evict and chunks_since_evict[i] >= evict_every_k_chunks:
                    cache.flush_offload()
                    chunks_since_evict[i] = 0

            torch.cuda.synchronize(dit_device)
            t0 = time.perf_counter()
            with self.progress_bar(total=math.ceil(num_frames / frame_per_block)) as progress_bar:
                for start_frame in range(0, num_frames, frame_per_block):
                    while (switch_idx < len(switch_frame_indices_sorted)
                        and switch_idx < len(args_c_switch)
                        and start_frame >= switch_frame_indices_sorted[switch_idx]):
                        _sp_text = switch_prompts[switch_idx] if switch_prompts else ""
                        logger.info(
                            f"[switch] start_frame={start_frame} >= {switch_frame_indices_sorted[switch_idx]}, "
                            f"recache_mode={recache_mode}, switching to: {_sp_text}"
                        )
                        num_recache = min(slice_last_frames, start_frame)
                        recache_start = start_frame - num_recache
                        recache_latents = latents[:, :, recache_start:start_frame]
                        context_t = torch.zeros(batch_size, device=dit_device, dtype=self.param_dtype)

                        snap_cond = snap_uncond = None
                        if recache_mode == "none":
                            arg_c = args_c_switch[switch_idx]
                            switch_idx += 1
                            continue
                        if recache_mode == "window_only":
                            _wkv = (window_size if not isinstance(window_size, MutableMapping)
                                    else window_size.get("kv"))
                            _window_frames = (_wkv or 0) * frame_per_block
                            _num_outer = max(0, num_recache - _window_frames)
                            if _num_outer > 0:
                                outer_chunk_start = recache_start // frame_per_block
                                outer_chunk_end = (recache_start + _num_outer) // frame_per_block
                                snap_cond = _snapshot_chunks(kv_cache, outer_chunk_start, outer_chunk_end)
                                if guide_scale > 1:
                                    snap_uncond = _snapshot_chunks(kv_cache_uncond, outer_chunk_start, outer_chunk_end)

                        # The window_only recache freezes host rows; release them in finally.
                        _pending = []
                        if snap_cond is not None:
                            _pending.append((kv_cache, snap_cond))
                        if snap_uncond is not None:
                            _pending.append((kv_cache_uncond, snap_uncond))
                        _done = 0
                        try:
                            for blk in range(0, num_recache, frame_per_block):
                                block = recache_latents[:, :, blk:blk + frame_per_block]
                                set_selection_trace_context(kind='recache', block=blk)
                                self.transformer(block, t=context_t, **args_c_switch[switch_idx],
                                                start_frame=recache_start + blk, kv_cache=kv_cache)
                                if guide_scale > 1:
                                    set_selection_trace_context(kind='recache', block=blk, cfg='uncond')
                                    self.transformer(block, t=context_t, **arg_null,
                                                    start_frame=recache_start + blk, kv_cache=kv_cache_uncond)

                            while _done < len(_pending):
                                _c, _s = _pending[_done]
                                _restore_chunks(_c, _s)
                                _done += 1
                        finally:
                            for _c, _s in _pending[_done:]:
                                _bs = _s.get('block_store') if isinstance(_s, dict) else None
                                if _bs is not None:
                                    _c.block_release_snapshot(_bs)

                        arg_c = args_c_switch[switch_idx]
                        switch_idx += 1

                    noisy_input = noise[:, :, start_frame:start_frame + frame_per_block]
                    if marks is not None:
                        marks.mark("chunk.begin", chunk=start_frame // frame_per_block)

                    for _c in caches:
                        _c.lru_reset_stats()

                    for index, t in enumerate(timesteps):
                        progress_bar.set_description(f"{t=:>7.02f}")

                        latent_model_input = noisy_input
                        timestep = [t]

                        timestep = torch.stack(timestep)

                        if img_latent is not None and start_frame == 0:
                            timestep = mask1[:, 0, :, ::self.patch_size[1], ::self.patch_size[2]] * timestep

                        self.transformer.to(dit_device)
                        set_selection_trace_context(chunk=start_frame // frame_per_block,
                                                    start_frame=start_frame, kind='denoise',
                                                    step=index, t=timesteps_list[index])
                        noise_pred_cond = self.transformer(
                            latent_model_input,
                            t=timestep,
                            **arg_c,
                            start_frame=start_frame,
                            kv_cache=kv_cache,
                        )
                        pred_image_cond = pred_x0(noise_pred_cond, timestep, noisy_input)

                        if guide_scale > 1:
                            set_selection_trace_context(chunk=start_frame // frame_per_block,
                                                        start_frame=start_frame, kind='denoise',
                                                        step=index, t=timesteps_list[index],
                                                        cfg='uncond')
                            noise_pred_uncond = self.transformer(
                                latent_model_input,
                                t=timestep,
                                **arg_null,
                                start_frame=start_frame,
                                kv_cache=kv_cache_uncond,
                            )

                            pred_image_uncond = pred_x0(noise_pred_uncond, timestep, noisy_input)

                            pred_image = pred_image_uncond + guide_scale * (
                                    pred_image_cond - pred_image_uncond)
                        else:
                            pred_image = pred_image_cond

                        if img_latent is not None and start_frame == 0:
                            pred_image = (1. - mask2) * img_latent + mask2 * pred_image

                        if index < len(timesteps) - 1:
                            next_timestep = timesteps[index + 1].expand(noisy_input.shape[0])
                            noisy_input = sample_scheduler.add_noise(
                                pred_image,
                                torch.randn(
                                    pred_image.shape,
                                    dtype=pred_image.dtype,
                                    device=pred_image.device,
                                    generator=generator,
                                ),
                                next_timestep
                            )

                    latents[:, :, start_frame:start_frame + frame_per_block] = pred_image

                    if on_latent_chunk is not None:
                        # Owned snapshot of the written latents, taken before the refresh.
                        chunk_end = min(start_frame + frame_per_block, num_frames)
                        snapshot = latents[:, :, start_frame:chunk_end].detach().clone(
                            memory_format=torch.contiguous_format)
                        ready_event = torch.cuda.Event()
                        ready_event.record(torch.cuda.current_stream(dit_device))
                        # Marked BEFORE the callback: in the one-GPU layout the callback
                        # decodes the chunk synchronously.
                        if marks is not None:
                            marks.mark("latent.submitted", chunk=start_frame // frame_per_block)
                        on_latent_chunk(LatentChunk(
                            chunk_index=start_frame // frame_per_block,
                            latent=snapshot,
                            ready_event=ready_event,
                            submit_time=time.monotonic(),
                        ))

                    context_timestep = torch.zeros_like(timestep)
                    set_selection_trace_context(chunk=start_frame // frame_per_block,
                                                start_frame=start_frame, kind='clean')
                    self.transformer(
                        pred_image_cond,
                        t=context_timestep,
                        **arg_c,
                        start_frame=start_frame,
                        kv_cache=kv_cache,
                    )
                    release_history(0, kv_cache)

                    if guide_scale > 1:
                        set_selection_trace_context(chunk=start_frame // frame_per_block,
                                                    start_frame=start_frame, kind='clean',
                                                    cfg='uncond')
                        self.transformer(
                            pred_image_uncond,
                            t=context_timestep,
                            **arg_null,
                            start_frame=start_frame,
                            kv_cache=kv_cache_uncond,
                        )
                        release_history(1, kv_cache_uncond)

                    _block_rate = None
                    if block_policy:
                        # protected = chunk live on GPU, persistent = block already in the pool.
                        _block_rate = kv_cache.block_hit_rate()
                    if marks is not None:
                        _extra = {}
                        if block_policy:
                            _extra.update({f"block_{k}": v for k, v in kv_cache.block_stats.items()})
                            _extra["block_hit_rate"] = _block_rate
                        elif enable_kv_evict:
                            _extra["lru_hit_rate"] = kv_cache.lru_hit_rate_overall()
                            _extra["lru_access"] = kv_cache._lru_total_access
                            _extra["lru_hits"] = kv_cache._lru_total_hits
                        _rs = getattr(kv_cache, "reload_stats", None)
                        if _rs:
                            _extra.update({f"reload_{k}": v for k, v in _rs.items()})
                        marks.mark("chunk.end", chunk=start_frame // frame_per_block, **_extra)
                    alloc = torch.cuda.memory_allocated(dit_device) / 1e9
                    reserved = torch.cuda.memory_reserved(dit_device) / 1e9
                    postfix = dict(mem=f"{alloc:.1f}/{reserved:.1f}G")
                    if block_policy:
                        postfix['blk'] = "n/a" if _block_rate is None else f"{_block_rate:.1%}"
                    elif enable_kv_evict:
                        postfix['lru'] = f"{kv_cache.lru_hit_rate_overall():.1%}"
                    progress_bar.set_postfix(**postfix, refresh=False)
                    progress_bar.update()

            set_selection_trace_context()
            torch.cuda.synchronize(dit_device)
            t = time.perf_counter() - t0
            logger.info(f"Denoising Time: {t:.2f}s")
            if marks is not None:
                marks.mark("denoise.end", seconds=t)
            if return_latents:
                # Latent-only contract: no decode, no extra copy of the latent buffer.
                return LatentGenerationOutput(
                    num_chunks=math.ceil(num_frames / frame_per_block),
                )
            if offload_model:
                self.vae.to(dit_device)

            # No-op when the VAE shares the DiT device.
            latents = latents.to(self.vae.device)

            if vae_decode_chunk_size:
                # Chunked decode; release reserved-but-unallocated blocks so activations fit.
                torch.cuda.empty_cache()
                image = decode_latents_chunked_to_cpu(
                    self.vae, latents, chunk_size=int(vae_decode_chunk_size),
                    autocast_dtype=self.param_dtype)
                image = self.tensor2vid(image, self.image_processor, output_type="pil")
            else:
                latents_mean = torch.tensor(self.vae.config.latents_mean,
                                            device=self.vae.device, dtype=self.vae.dtype).reshape(1, -1, 1, 1, 1)
                latents_std = torch.tensor(self.vae.config.latents_std,
                                        device=self.vae.device, dtype=self.vae.dtype).reshape(1, -1, 1, 1, 1)
                latents = latents * latents_std + latents_mean
                image = self.vae.decode(latents).sample
                image = self.tensor2vid(image.float().cpu(), self.image_processor, output_type="pil")
            if offload_model:
                self.vae.cpu()

            gc.collect()
            torch.cuda.empty_cache()
            return ImagePipelineOutput(images=image)
