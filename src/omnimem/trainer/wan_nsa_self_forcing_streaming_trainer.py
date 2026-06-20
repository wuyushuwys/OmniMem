import math
import random
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule

from omnimem.models.cache import MMCache
from omnimem.models.transformers.causal_wan_nsa_model import CausalWanNSAModel
from omnimem.schedulers import RectifiedFlowScheduler
from omnimem.utils.train_utils import pred_x0, vae_encode
from omnimem.utils.misc import reduce_dict, unwrap_model
from omnimem.utils.torch_utils import update_ema_fsdp
from omnimem.trainer.wan_nsa_self_forcing_trainer import WanNSASelfForcingTrainer


class WanNSASelfForcingStreamingTrainer(WanNSASelfForcingTrainer):
    """
    Streaming long-video trainer that generates and trains one chunk at a time with a persistent KV cache.
    Context flows through the cache; the previous chunk's tail is prepended as raw-latent overlap.
    """

    def __init__(self, config):
        self.streaming_chunk_size = config.get("streaming_chunk_size", 21)
        self.streaming_max_length = config.get("streaming_max_length", 240)
        self.streaming_min_new_frame = config.get("streaming_min_new_frame", 18)
        self.train_first_chunk = config.get("train_first_chunk", True)
        self.streaming_possible_max_length = config.get("streaming_possible_max_length", None)
        # optional VAE re-encode of the loss window's first frame
        self.vae_re_encode = config.get("vae_re_encode", False)

        super().__init__(config)

        # Round max_length up to a whole number of chunks to avoid tail waste.
        cs = self.streaming_chunk_size
        aligned = math.ceil(self.streaming_max_length / cs) * cs
        if aligned != self.streaming_max_length:
            self.logger.info(
                f"streaming_max_length {self.streaming_max_length} not a multiple of "
                f"chunk_size {cs}; aligning up to {aligned} ({aligned // cs} chunks)."
            )
            self.streaming_max_length = aligned
        if self.streaming_possible_max_length is not None:
            self.streaming_possible_max_length = [
                math.ceil(v / cs) * cs for v in self.streaming_possible_max_length
            ]

        self.streaming_active = False
        self.streaming_kv_cache: Optional[MMCache] = None
        self.streaming_state: Dict[str, Any] = {}

    def _select_max_length(self) -> int:
        if self.streaming_possible_max_length is not None:
            choices = list(self.streaming_possible_max_length)
            idx = torch.zeros(1, dtype=torch.int32, device=self.device)
            if not dist.is_initialized() or dist.get_rank() == 0:
                idx.fill_(random.randint(0, len(choices) - 1))
            if dist.is_initialized():
                dist.broadcast(idx, src=0)
            return choices[idx.item()]
        return self.streaming_max_length

    def _reset_streaming_state(self):
        self.streaming_state = {
            "current_length": 0,
            "previous_frames": None,
            "temp_max_length": None,
            "condition_dict": None,
            "uncondition_dict": None,
        }
        if self.streaming_kv_cache is not None:
            self.streaming_kv_cache.reset()
        self.streaming_active = False

    def _ensure_streaming_kv_cache(self, dummy_latents: torch.Tensor):
        """Build the persistent streaming KV cache on first call; subsequent sequences reuse it via reset()."""
        if self.streaming_kv_cache is not None:
            return
        self.streaming_kv_cache = self.build_kv_cache(
            model=self.generator, latents=dummy_latents,
        )
        if not (self.enable_kv_evict and self.enable_chunk_per_head_cache):
            return
        batch_size, _, latent_num_frames, latent_h, latent_w = dummy_latents.shape
        patch_size = self.generator.patch_size[1]
        seq_len_per_frame = latent_h * latent_w // (patch_size * patch_size)
        chunk_len_for_kv = seq_len_per_frame * self.frame_per_block
        head_dim = self.generator.dim // self.generator.num_heads
        total_chunks = math.ceil(latent_num_frames / self.frame_per_block)
        sample = torch.empty(
            (batch_size, chunk_len_for_kv, head_dim),
            dtype=self.generator.dtype, device=self.generator.device,
        )
        max_entries = self.generator.num_layers * 2 * self.generator.num_heads
        self.streaming_kv_cache.preallocate_offload_buffer(
            sample=sample,
            max_entries_per_call=max_entries,
            segment_cids=total_chunks,
        )
        del sample

    def start_new_sequence(self):
        """Sample fresh prompts and initialise a new streaming sequence."""
        self._reset_streaming_state()

        try:
            batch = next(self._streaming_data_iter)
        except StopIteration:
            self._streaming_data_iter = iter(self.train_dataloader)
            batch = next(self._streaming_data_iter)
        text = batch["prompts"]
        text_embeddings, _ = self.text_encoder(texts=text, device=self.device)
        null_text_embeddings, _ = self.text_encoder(
            texts=[self.null_prompt] * len(text), device=self.device
        )

        _, latent_num_frames, latent_height, latent_width = self.latents_shape
        patch_size = self.generator.patch_size[1]
        sequence_length = latent_num_frames * latent_height * latent_width // (patch_size * patch_size)

        condition_dict = dict(context=text_embeddings, seq_len=sequence_length)
        uncondition_dict = dict(context=null_text_embeddings, seq_len=sequence_length)

        max_length = self.streaming_max_length
        if self.streaming_possible_max_length is not None:
            max_length = max(self.streaming_possible_max_length)
        dummy_latents = torch.zeros(
            len(text),
            self.latents_shape[0],
            max_length,
            latent_height,
            latent_width,
            device=self.device, dtype=self.dtype,
        )
        self._ensure_streaming_kv_cache(dummy_latents)

        self.streaming_state.update(
            current_length=0,
            previous_frames=None,
            temp_max_length=self._select_max_length(),
            condition_dict=condition_dict,
            uncondition_dict=uncondition_dict,
        )
        self.streaming_active = True

    def can_generate_more(self) -> bool:
        cur = self.streaming_state["current_length"]
        max_len = self.streaming_state["temp_max_length"]
        return cur < max_len and (cur + self.streaming_min_new_frame) <= max_len

    @torch.no_grad()
    def _vae_re_encode_first_frame(self, frames: torch.Tensor) -> torch.Tensor:
        """Replace the loss window's first latent with a VAE decode->re-encode round-trip (loss-input only)."""
        if frames.shape[2] <= 1:
            return frames
        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=self.vae.device, dtype=self.vae.dtype,
        ).reshape(1, -1, 1, 1, 1)
        latents_std = torch.tensor(
            self.vae.config.latents_std,
            device=self.vae.device, dtype=self.vae.dtype,
        ).reshape(1, -1, 1, 1, 1)
        z = frames[:, :, :1].to(device=self.vae.device, dtype=self.vae.dtype)
        z = z * latents_std + latents_mean

        pixels = self.vae.decode(z).sample  # [B, 3, 1, H_pix, W_pix]
        anchor = vae_encode(self.vae, pixels[:, :, -1:].contiguous())  # [B, C, 1, H, W]
        anchor = anchor.to(device=frames.device, dtype=frames.dtype)
        return torch.cat([anchor, frames[:, :, 1:]], dim=2)

    def _set_generator_reshard(self, reshard_after_forward: bool, reshard_after_backward: bool):
        """Toggle FSDP reshard flags on generator blocks.

        reshard_after_backward=False is needed because with skip_fsdp_hooks=True,
        post_backward() fires mid-backward and would free params before the cycle ends."""
        if not dist.is_initialized():
            return
        for block in self.generator.blocks:
            if isinstance(block, FSDPModule):
                block.set_reshard_after_forward(reshard_after_forward)
                state = block._get_fsdp_state()
                pg = getattr(state, "_fsdp_param_group", None)
                if pg is not None:
                    pg.reshard_after_backward = reshard_after_backward

    def _register_kv_free_hooks(self, model, kv_cache):
        """No-op: streaming KV cache persists across train_steps; flushing mid-backward
        would corrupt AC recompute metadata. Offload happens at end of train_step instead."""
        self._remove_kv_free_hooks()

    def _snapshot_streaming_kv(self) -> dict:
        """Shallow-copy chunk lists before backward() so AC recompute writes don't corrupt context-pass KV.

        Per-head types use one-level-deeper copy because they are list-of-lists."""
        if self.streaming_kv_cache is None:
            return {}
        kv = self.streaming_kv_cache
        snap = {}
        for li, layer_cache in kv.cache.items():
            snap[li] = {}
            for name in kv.chunked_types:
                if name not in layer_cache:
                    continue
                if name in kv.block_level_types or name in kv.chunk_per_head_types:
                    # list-of-lists: deep-copy one level so inner lists are independent
                    snap[li][name] = [list(h_list) for h_list in layer_cache[name]]
                else:
                    snap[li][name] = list(layer_cache[name])
        return snap

    def _restore_streaming_kv(self, snap: dict):
        """Restore context-pass KV after backward(), undoing AC recompute writes."""
        if self.streaming_kv_cache is None or not snap:
            return
        for li, layer_snap in snap.items():
            for name, chunks in layer_snap.items():
                self.streaming_kv_cache.cache[li][name] = chunks

    def backward_simulation_chunk(
        self,
        transformer: CausalWanNSAModel,
        noise_scheduler: RectifiedFlowScheduler,
        num_inference_steps: int,
        noise: torch.Tensor,
        condition_dict: dict,
        start_frame: int,
        kv_cache: MMCache,
        requires_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Self-forcing simulation for one chunk; updates KV cache in-place. Returns (denoised_chunk, target_timestep)."""
        batch_size, num_channels, chunk_frames, latent_height, latent_width = noise.shape
        noisy_latents = noise

        if requires_grad:
            self._register_kv_free_hooks(transformer, kv_cache)

        noise_scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=noise.device,
            noise=noisy_latents,
        )
        inference_timesteps = noise_scheduler.timesteps

        if self.same_step_across_blocks:
            selected_step = torch.randint(
                0, inference_timesteps.shape[0],
                size=(1,), device=noise.device, dtype=torch.long,
            )
            if dist.is_initialized():
                dist.broadcast(selected_step, src=0)
        else:
            selected_step = None

        num_blocks = math.ceil(chunk_frames / self.frame_per_block)
        output_latent = torch.zeros_like(noisy_latents)
        target_timestep = torch.empty(batch_size, num_blocks, device=noise.device)

        for idx in range(num_blocks):
            blk_start = idx * self.frame_per_block
            abs_start = start_frame + blk_start
            noisy_latent = noisy_latents[:, :, blk_start:blk_start + self.frame_per_block].contiguous()

            if not self.same_step_across_blocks:
                selected_step = torch.randint(
                    0, inference_timesteps.shape[0],
                    size=(1,), device=noise.device, dtype=torch.long,
                )
                if dist.is_initialized():
                    dist.broadcast(selected_step, src=0)

            for step_idx, t in enumerate(inference_timesteps[:selected_step]):
                timestep = t.expand(noisy_latent.shape[0])
                with torch.no_grad():
                    noise_pred = transformer(
                        x=noisy_latent, t=timestep,
                        **condition_dict,
                        start_frame=abs_start,
                        kv_cache=kv_cache,
                    )
                denoised_pred = pred_x0(noise_pred, timestep, noisy_latent, noise_scheduler)
                denoised_pred = denoised_pred.contiguous()
                next_t = inference_timesteps[step_idx + 1].expand(denoised_pred.shape[0])
                noisy_latent = noise_scheduler.add_noise(
                    denoised_pred, noise=torch.randn_like(denoised_pred), timesteps=next_t
                )

            timestep = inference_timesteps[selected_step].expand(noisy_latent.shape[0])
            with torch.set_grad_enabled(requires_grad):
                noise_pred = transformer(
                    x=noisy_latent, t=timestep,
                    **condition_dict,
                    start_frame=abs_start,
                    kv_cache=kv_cache,
                )
                denoised_pred = pred_x0(noise_pred, timestep, noisy_latent, noise_scheduler)
                denoised_pred = denoised_pred.contiguous()

            output_latent[:, :, blk_start:blk_start + self.frame_per_block] = denoised_pred
            target_timestep[:, idx] = inference_timesteps[selected_step]

            with torch.no_grad():
                context_t = torch.zeros_like(timestep)
                transformer(
                    x=denoised_pred, t=context_t,
                    **condition_dict,
                    start_frame=abs_start,
                    kv_cache=kv_cache,
                )

            # Flush only in no_grad mode; in grad mode flush happens at end-of-train_step
            # to avoid mutating cache refs captured by save_for_backward during AC recompute.
            if (not requires_grad
                    and self.enable_kv_evict
                    and kv_cache is not None):
                kv_cache.flush_offload()

        return output_latent, target_timestep

    def generate_next_chunk(self, requires_grad: bool = True):
        """Generate the next streaming chunk; prepend the previous chunk's tail as raw-latent overlap."""
        if not self.streaming_active or not self.can_generate_more():
            self.start_new_sequence()
            if not self.train_first_chunk:
                self.generate_next_chunk(requires_grad=False)

        state = self.streaming_state
        current_length = state["current_length"]
        previous_frames = state["previous_frames"]
        batch_size = len(state["condition_dict"]["context"])
        fpb = self.frame_per_block
        cs = self.streaming_chunk_size

        if self.streaming_kv_cache is not None:
            self.streaming_kv_cache.lru_reset_stats()

        # Sample how many new frames to generate; overlap fills the rest of the loss window.
        if previous_frames is not None:
            max_new = min(state["temp_max_length"] - current_length, cs)
            possible = list(range(self.streaming_min_new_frame, max_new + 1, fpb))
            if not possible:
                possible = [self.streaming_min_new_frame]

            sel = torch.zeros(1, dtype=torch.int32, device=self.device)
            if not dist.is_initialized() or dist.get_rank() == 0:
                sel.fill_(random.randint(0, len(possible) - 1))
            if dist.is_initialized():
                dist.broadcast(sel, src=0)
            new_frames = possible[sel.item()]

            overlap = cs - new_frames
            if overlap <= 0 or overlap > previous_frames.shape[2]:
                overlap = 0
                new_frames = cs
        else:
            new_frames = cs
            overlap = 0

        noise = torch.randn(
            batch_size, self.latents_shape[0], new_frames,
            self.latents_shape[2], self.latents_shape[3],
            device=self.device, dtype=self.dtype,
        )

        # New frames generate at the current position; overlap KV is already in the cache.
        denoised_new, target_ts = self.backward_simulation_chunk(
            transformer=self.generator,
            noise_scheduler=self.noise_scheduler,
            num_inference_steps=self.max_distillation_steps,
            noise=noise,
            condition_dict=state["condition_dict"],
            start_frame=current_length,
            kv_cache=self.streaming_kv_cache,
            requires_grad=requires_grad,
        )

        if overlap > 0:
            full_chunk = torch.cat([previous_frames[:, :, -overlap:], denoised_new], dim=2)
        else:
            full_chunk = denoised_new

        # Carry raw latents to the next chunk before any VAE re-encode.
        frames_to_save = full_chunk[:, :, -cs:].detach().clone()

        if self.vae_re_encode and overlap > 0:
            full_chunk = self._vae_re_encode_first_frame(full_chunk)

        if overlap > 0:
            gradient_mask = torch.zeros_like(full_chunk, dtype=torch.bool)
            gradient_mask[:, :, overlap:] = True
        else:
            gradient_mask = torch.ones_like(full_chunk, dtype=torch.bool)

        state["current_length"] += new_frames
        state["previous_frames"] = frames_to_save

        info = dict(
            target_timestep=target_ts,
            gradient_mask=gradient_mask,
            new_frames=new_frames,
            overlap=overlap,
            chunk_start=current_length,
        )
        return full_chunk, info

    def generator_step(self, chunk, info):
        state = self.streaming_state
        gradient_mask = info["gradient_mask"]

        dmd_timestep = self.noise_scheduler.sample_timesteps(
            chunk.shape[0], device=self.device,
        )
        dmd_timestep = self.noise_scheduler.scale_timesteps(dmd_timestep, noise=chunk)
        dmd_timestep.clamp_(
            self.noise_scheduler.num_train_timesteps * 0.02,
            self.noise_scheduler.num_train_timesteps * 0.98,
        )

        noise = torch.randn_like(chunk)
        noisy = self.noise_scheduler.add_noise(chunk, noise, dmd_timestep)
        x0_proj_kw = dict(timestep=dmd_timestep, sample=noisy, noise_scheduler=self.noise_scheduler)

        with torch.no_grad():
            pred_fake_noise = self.fake_net(x=noisy, t=dmd_timestep, **state["condition_dict"])
            pred_fake_image = pred_x0(pred_fake_noise, **x0_proj_kw)

            pred_real_cond = self.real_net(x=noisy, t=dmd_timestep, **state["condition_dict"])
            pred_real_cond_x0 = pred_x0(pred_real_cond, **x0_proj_kw)

            pred_real_uncond = self.real_net(x=noisy, t=dmd_timestep, **state["uncondition_dict"])
            pred_real_uncond_x0 = pred_x0(pred_real_uncond, **x0_proj_kw)

            pred_real_image = pred_real_cond_x0 + (pred_real_cond_x0 - pred_real_uncond_x0) * self.real_guidance_scale

            grad = (pred_fake_image - pred_real_image)
            normalizer = torch.abs(chunk - pred_real_image).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = torch.nan_to_num(grad / normalizer)

        per_elem_loss = 0.5 * F.mse_loss(
            chunk.float(),
            (chunk.float() - grad.float()).detach(),
            reduction="none",
        )

        loss = per_elem_loss[gradient_mask].mean() if gradient_mask is not None else per_elem_loss.mean()
        log = {"g_loss/g_dmd_loss": loss.detach(), "train/g_total_loss": loss.detach()}
        return loss, log

    def critic_step(self, chunk, info):
        state = self.streaming_state
        gradient_mask = info["gradient_mask"]

        chunk = chunk.detach()
        batch_size = chunk.shape[0]

        critic_noise = torch.randn_like(chunk)
        critic_ts = self.noise_scheduler.sample_timesteps(batch_size, device=self.device)
        critic_ts = self.noise_scheduler.scale_timesteps(critic_ts, chunk)
        critic_ts.clamp_(
            self.noise_scheduler.num_train_timesteps * 0.02,
            self.noise_scheduler.num_train_timesteps * 0.98,
        )

        noisy = self.noise_scheduler.add_noise(chunk, critic_noise, critic_ts)
        target = self.noise_scheduler.get_velocity(chunk, critic_noise, critic_ts)

        fake_pred = self.fake_net(x=noisy, t=critic_ts, **state["condition_dict"])
        fake_x0 = pred_x0(fake_pred, critic_ts, noisy, self.noise_scheduler)

        if self.critic_loss_type == "flow":
            sigma_t = critic_ts.view(batch_size, 1, 1, 1, 1) / self.noise_scheduler.config.num_train_timesteps
            flow_pred = (noisy - fake_x0) / sigma_t
            per_elem_loss = F.mse_loss(flow_pred.float(), target.float(), reduction="none")
        elif self.critic_loss_type == "x0":
            per_elem_loss = F.mse_loss(fake_x0.float(), chunk.float(), reduction="none")
        elif self.critic_loss_type == "v":
            per_elem_loss = F.mse_loss(fake_pred.float(), target.float(), reduction="none")
        else:
            raise ValueError(f"Unknown critic loss type: {self.critic_loss_type}")

        loss = per_elem_loss[gradient_mask].mean() if gradient_mask is not None else per_elem_loss.mean()
        log = {"c_loss/c_flow_loss": loss.detach(), "train/critic_total_loss": loss.detach()}
        return loss, log

    def train_step(self):
        log_dict = {}
        loss_log = {}

        is_gen_step = (self.global_step % self.dfake_gen_update_ratio == 0)

        # One simulation feeds both generator and critic; critic_step detaches internally.
        if is_gen_step:
            self.set_requires_gradient_reduce(self.generator)
            self.generator_timer.tic()

            # Keep params gathered for the entire forward+backward cycle.
            self._set_generator_reshard(reshard_after_forward=False, reshard_after_backward=False)
            chunk, info = self.generate_next_chunk(requires_grad=True)

            g_loss, g_loss_log = self.generator_step(chunk, info)
            kv_snap = self._snapshot_streaming_kv()
            (g_loss / self.gradient_accumulation_steps).backward()
            self._restore_streaming_kv(kv_snap)
            self._set_generator_reshard(reshard_after_forward=True, reshard_after_backward=True)
            self._remove_kv_free_hooks()

            if self.enable_kv_evict and self.streaming_kv_cache is not None:
                self.streaming_kv_cache.flush_offload()

            if self.accumulation_index == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.generator.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                log_dict["train/g_grad_norm"] = grad_norm.to_local().item() if hasattr(grad_norm, 'to_local') else grad_norm.item()
                self.generator_optimizer.step()
                self.generator_optimizer.zero_grad(set_to_none=True)
                self.generator_lr_scheduler.step()

                if self.ema is not None and self.ema_start_step <= self.global_step:
                    update_ema_fsdp(self.ema, unwrap_model(self.generator), self.ema_decay)

            loss_log.update(g_loss_log)
            self.generator_timer.toc()
        else:
            with torch.no_grad():
                chunk, info = self.generate_next_chunk(requires_grad=False)
            if self.enable_kv_evict and self.streaming_kv_cache is not None:
                self.streaming_kv_cache.flush_offload()

        # train critic
        self.set_requires_gradient_reduce(self.fake_net)
        self.critic_timer.tic()

        c_loss, c_loss_log = self.critic_step(chunk, info)
        (c_loss / self.gradient_accumulation_steps).backward()

        if self.accumulation_index == 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.fake_net.parameters() if p.requires_grad],
                self.max_grad_norm
            )
            log_dict["train/critic_grad_norm"] = critic_grad_norm.to_local().item() if hasattr(critic_grad_norm, 'to_local') else critic_grad_norm.item()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.critic_lr_scheduler.step()

        loss_log.update(c_loss_log)
        loss_log["train/critic_loss"] = c_loss.detach()
        self.critic_timer.toc()

        reduce_dict(loss_log)
        self.loss_meter.update(loss_log)

        return log_dict

    def evaluate_init(self):
        # streaming_kv_cache is None on first eval; pipeline builds its own cache.
        super().evaluate_init()

    def evaluate(self):
        injected = self.streaming_kv_cache is not None
        if injected:
            self._eval_kwargs["_kv_cache"] = self.streaming_kv_cache
        try:
            super().evaluate()
        finally:
            if injected:
                self._eval_kwargs.pop("_kv_cache", None)
                # Eval overwrote the shared cache; reset so next train_step starts fresh.
                self.streaming_active = False

    def train(self):
        self.logger.info("***** Running streaming training *****")
        self.logger.info(
            f"  chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}, "
            f"min_new_frame={self.streaming_min_new_frame}"
        )

        self._streaming_data_iter = iter(self.train_dataloader)

        self.evaluate_init()

        # save/eval fire at sequence boundaries to avoid mid-sequence GPU pressure.
        last_save_step = self.global_step
        last_eval_step = self.global_step

        step = 0
        while self.global_step < self.max_train_steps:
            step += 1
            self.accumulation_index = step % self.gradient_accumulation_steps

            self.setup_ema_model()
            self.generator.train()
            self.fake_net.train()

            self.data_timer.toc()

            log_dict = self.train_step()

            self.data_timer.tic()

            if self.accumulation_index == 0:
                self.global_step += 1

                self._log_progress(log_dict)

                at_end = self.global_step >= self.max_train_steps
                at_boundary = (not self.streaming_active) or (not self.can_generate_more())
                save_due = (self.global_step - last_save_step) >= self.checkpointing_steps
                eval_due = (self.global_step - last_eval_step) >= self.validation_steps

                if (at_boundary or at_end) and (save_due or eval_due):
                    if self.streaming_kv_cache is not None:
                        self._reset_streaming_state()
                    torch.cuda.empty_cache()
                    if save_due:
                        self.logger.info(
                            f"Sequence finished at step {self.global_step}; firing save "
                            f"(last save at step {last_save_step}, interval={self.checkpointing_steps})"
                        )
                        self.save_models()
                        last_save_step = self.global_step
                    if eval_due:
                        self.logger.info(
                            f"Sequence finished at step {self.global_step}; firing eval "
                            f"(last eval at step {last_eval_step}, interval={self.validation_steps})"
                        )
                        self.evaluate()
                        last_eval_step = self.global_step

                if at_end:
                    self.logger.info("Finish streaming training")
                    return
