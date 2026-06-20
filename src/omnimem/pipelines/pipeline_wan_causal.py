# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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
from omnimem.models.transformers.causal_wan_nsa_model import CausalWanNSAModel
from omnimem.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from omnimem.schedulers import RectifiedFlowScheduler
from omnimem.pipelines import WanTI2VPipeline
from omnimem.utils.train_utils import pred_x0
from omnimem.pipelines.utils import masks_like
from omnimem.utils.train_utils import vae_encode
from omnimem.utils.logging_tool import get_logger
from omnimem.models.text_encoders.t5 import T5EncoderModel

logger = get_logger()


def _snapshot_chunks(cache: MMCache, chunk_start: int, chunk_end: int) -> dict:
    """Save references to cache chunks [chunk_start, chunk_end) for all layers.
    No clone needed: update_cache replaces list entries by reference assignment,
    so the original tensors are never modified in-place."""
    snap = {}
    for layer_idx, layer_cache in cache.cache.items():
        snap[layer_idx] = {}
        for cache_type, chunks in layer_cache.items():
            if not isinstance(chunks, list):
                continue
            snap[layer_idx][cache_type] = [
                chunks[i] if i < len(chunks) else None
                for i in range(chunk_start, chunk_end)
            ]
    return snap


def _restore_chunks(cache: MMCache, snapshot: dict, chunk_start: int):
    for layer_idx, layer_snap in snapshot.items():
        for cache_type, snap_chunks in layer_snap.items():
            chunks = cache.cache[layer_idx][cache_type]
            for offset, snap_chunk in enumerate(snap_chunks):
                idx = chunk_start + offset
                if snap_chunk is not None and idx < len(chunks):
                    chunks[idx] = snap_chunk


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
            enable_cpu_offload=True,
            lru_max_size=10,
            enable_block_level_cache=False,
            enable_chunk_per_head_cache=False,
            enable_kv_evict=True,
            kv_evict_min_gpu_chunks=None,
            switch_prompts=None,
            switch_frame_indices=None,
            slice_last_frames=21,
            recache_mode="full",
            vae_decode_chunk_size=None,
            evict_every_k_chunks=1,
            _kv_cache: Optional[MMCache] = None,
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
        """

        # preprocess
        F = frame_num
        target_shape = (self.vae.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        input_prompt = self._text_preprocessing(input_prompt)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        self.text_encoder.model.to(self.device)
        context, attn_cond = self.text_encoder(input_prompt, self.device)
        context_null, attn_null = self.text_encoder(n_prompt, self.device)
        _main_p = input_prompt[0] if isinstance(input_prompt, (list, tuple)) else input_prompt
        logger.info(f"[prompt] MAIN (start_frame=0): {_main_p}")
        # Encode all switch prompts upfront
        if switch_prompts is not None and switch_frame_indices is not None:
            if isinstance(switch_prompts, str):
                switch_prompts = [switch_prompts] * len(switch_frame_indices)
            contexts_switch = []
            for sp in switch_prompts:
                ctx, _ = self.text_encoder(sp, self.device)
                contexts_switch.append(ctx)
            _sfi_sorted = sorted(switch_frame_indices)
            for i, sp in enumerate(switch_prompts):
                _sf = _sfi_sorted[i] if i < len(_sfi_sorted) else None
                logger.info(f"[prompt] SWITCH#{i} (triggers at start_frame>={_sf}): {sp}")
        else:
            contexts_switch = []
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
            self.vae.encoder.to(self.device)
            img_tensor = img_tensor.to(device=self.device, dtype=self.param_dtype)
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
            device=self.device,
            generator=generator)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.transformer, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            sample_scheduler = RectifiedFlowScheduler(num_train_timesteps=self.num_train_timesteps, shift=shift)
            sample_scheduler.set_timesteps(num_inference_steps=sampling_steps, device=self.device, )
            timesteps = sample_scheduler.timesteps

            # sample videos
            latents = noise
            mask1, mask2 = masks_like(noise, zero=True)
            if img_latent is not None:
                # replace first frame to image
                latents = (1. - mask1) * img_latent + mask1 * latents

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}
            args_c_switch = [{'context': ctx, 'seq_len': seq_len} for ctx in contexts_switch]

            assert self.frame_per_block is not None, f"got invalid {self.frame_per_block=}"
            frame_per_block = self.frame_per_block
            batch_size, _, num_frames, height, width = noise.shape
            seq_len_per_frame = height * width // (self.patch_size[1] * self.patch_size[2])

            cache_shape = (
                batch_size,
                seq_len,
                self.transformer.num_heads,
                self.transformer.dim // self.transformer.num_heads
            )

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
                dtype=self.transformer.dtype,
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
                per_head_types={'k_cache', 'v_cache'} if enable_block_level_cache else None,
                chunk_per_head_types={'k_cache', 'v_cache'} if enable_chunk_per_head_cache else None,
                num_heads=self.transformer.num_heads if (enable_block_level_cache or enable_chunk_per_head_cache) else None,
                kv_block_size=(getattr(self.transformer.config, 'pool_f', 1)
                            * self.transformer.config.pool_h
                            * self.transformer.config.pool_w)
                            if (enable_block_level_cache or enable_chunk_per_head_cache) else 0,
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

            cache_signature = (
                batch_size, seq_len, self.transformer.num_heads,
                self.transformer.dim, frame_per_block,
                num_frames,  # affects total_chunks for preallocate
            )
            if _kv_cache is not None:
                # External cache: caller owns lifecycle; reset to wipe prior state.
                assert guide_scale <= 1, (
                    "External _kv_cache currently only supports guide_scale <= 1; "
                    "CFG (guide_scale > 1) needs a separate uncond cache, not yet "
                    "plumbed through this kwarg."
                )
                _kv_cache.reset()
                kv_cache = _kv_cache
            elif self._kv_cache is not None and self._cache_signature == cache_signature:
                self._kv_cache.reset()
                kv_cache = self._kv_cache
                if guide_scale > 1:
                    self._kv_cache_uncond.reset()
                    kv_cache_uncond = self._kv_cache_uncond
            else:
                if self._kv_cache is not None:
                    logger.info("[cache] signature changed, rebuilding")
                    del self._kv_cache
                    if self._kv_cache_uncond is not None:
                        del self._kv_cache_uncond
                    torch.cuda.empty_cache()

                kv_cache = MMCache(**cache_kwargs, lru_max_size=lru_max_size)
                self._kv_cache = kv_cache
                if guide_scale > 1:
                    kv_cache_uncond = MMCache(**cache_kwargs, lru_max_size=lru_max_size)
                    self._kv_cache_uncond = kv_cache_uncond

            if _kv_cache is None and enable_kv_evict and enable_chunk_per_head_cache:
                logger.info("[preallocate] === START ===")
                total_chunks = math.ceil(num_frames / frame_per_block)
                chunk_len_for_kv = seq_len_per_frame * frame_per_block
                head_dim = self.transformer.dim // self.transformer.num_heads
                _sample = torch.empty(
                    (batch_size, chunk_len_for_kv, head_dim),
                    dtype=self.transformer.dtype,
                    device=self.transformer.device,
                )
                _max_entries = self.transformer.num_layers * 2 * self.transformer.num_heads
                kv_cache.preallocate_offload_buffer(
                    sample=_sample,
                    max_entries_per_call=_max_entries,
                    segment_cids=total_chunks,
                )
                del _sample
                logger.info(f"[preallocate] === DONE ===")

            self._cache_signature = cache_signature
            gate_data = {}
            switch_idx = 0
            switch_frame_indices_sorted = sorted(switch_frame_indices) if switch_frame_indices else []
            chunks_since_evict_cond = 0
            chunks_since_evict_uncond = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with self.progress_bar(total=math.ceil(num_frames / frame_per_block)) as progress_bar:
                for start_frame in range(0, num_frames, frame_per_block):
                    if hasattr(self.transformer, "set_start_frame"):
                        self.transformer.set_start_frame(start_frame)

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
                        context_t = torch.zeros(batch_size, device=self.device, dtype=self.param_dtype)

                        snap_cond = snap_uncond = None
                        outer_chunk_start = 0
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

                        for blk in range(0, num_recache, frame_per_block):
                            block = recache_latents[:, :, blk:blk + frame_per_block]
                            self.transformer(block, t=context_t, **args_c_switch[switch_idx],
                                            start_frame=recache_start + blk, kv_cache=kv_cache)
                            if guide_scale > 1:
                                self.transformer(block, t=context_t, **arg_null,
                                                start_frame=recache_start + blk, kv_cache=kv_cache_uncond)

                        if snap_cond is not None:
                            _restore_chunks(kv_cache, snap_cond, outer_chunk_start)
                        if snap_uncond is not None:
                            _restore_chunks(kv_cache_uncond, snap_uncond, outer_chunk_start)

                        arg_c = args_c_switch[switch_idx]
                        switch_idx += 1

                    noisy_input = noise[:, :, start_frame:start_frame + frame_per_block]

                    if kv_cache is not None:
                        kv_cache.lru_reset_stats()
                    if guide_scale > 1 and kv_cache_uncond is not None:
                        kv_cache_uncond.lru_reset_stats()

                    for index, t in enumerate(timesteps):
                        progress_bar.set_description(f"{t=:>7.02f}")

                        latent_model_input = noisy_input
                        timestep = [t]

                        timestep = torch.stack(timestep)

                        if img_latent is not None and start_frame == 0:
                            timestep = mask1[:, 0, :, ::self.patch_size[1], ::self.patch_size[2]] * timestep

                        self.transformer.to(self.device)
                        noise_pred_cond = self.transformer(
                            latent_model_input,
                            t=timestep,
                            **arg_c,
                            start_frame=start_frame,
                            kv_cache=kv_cache,
                        )
                        pred_image_cond = pred_x0(noise_pred_cond, timestep, noisy_input)

                        if guide_scale > 1:
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

                    context_timestep = torch.zeros_like(timestep)
                    self.transformer(
                        pred_image_cond,
                        t=context_timestep,
                        **arg_c,
                        start_frame=start_frame,
                        kv_cache=kv_cache,
                    )
                    chunks_since_evict_cond += 1
                    if enable_kv_evict and chunks_since_evict_cond >= evict_every_k_chunks:
                        kv_cache.flush_offload()
                        chunks_since_evict_cond = 0

                    if guide_scale > 1:
                        self.transformer(
                            pred_image_uncond,
                            t=context_timestep,
                            **arg_null,
                            start_frame=start_frame,
                            kv_cache=kv_cache_uncond,
                        )
                        chunks_since_evict_uncond += 1
                        if enable_kv_evict and chunks_since_evict_uncond >= evict_every_k_chunks:
                            kv_cache_uncond.flush_offload()
                            chunks_since_evict_uncond = 0

                    alloc = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    postfix = dict(mem=f"{alloc:.1f}/{reserved:.1f}G")
                    if enable_kv_evict:
                        postfix['lru'] = f"{kv_cache.lru_hit_rate_overall():.1%}"
                    progress_bar.set_postfix(**postfix, refresh=False)
                    progress_bar.update()

            torch.cuda.synchronize()
            t = time.perf_counter() - t0
            logger.info(f"Denoising Time: {t:.2f}s")
            if offload_model:
                self.vae.to(self.device)

            latents_mean = torch.tensor(self.vae.config.latents_mean,
                                        device=self.vae.device, dtype=self.vae.dtype).reshape(1, -1, 1, 1, 1)
            latents_std = torch.tensor(self.vae.config.latents_std,
                                    device=self.vae.device, dtype=self.vae.dtype).reshape(1, -1, 1, 1, 1)
            latents = latents * latents_std + latents_mean

            if vae_decode_chunk_size:
                image = self.vae.decode_to_cpu(latents, chunk_size=vae_decode_chunk_size)
                image = self.tensor2vid(image, self.image_processor, output_type="pil")
            else:
                image = self.vae.decode(latents).sample
                image = self.tensor2vid(image.float().cpu(), self.image_processor, output_type="pil")
            if offload_model:
                self.vae.cpu()

            gc.collect()
            torch.cuda.empty_cache()
            return ImagePipelineOutput(images=image)

