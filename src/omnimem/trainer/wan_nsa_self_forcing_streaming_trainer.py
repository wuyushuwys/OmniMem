import gc
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
from omnimem.utils.torch_utils import update_ema_fsdp, unsharded_param_dtype
from omnimem.trainer.wan_nsa_self_forcing_trainer import WanNSASelfForcingTrainer
from omnimem.trainer.streaming_checkpoint import StreamingCheckpointMixin
from omnimem.trainer.streaming_generator_schedule import generator_update_due
from omnimem.utils.ode_training_state import restore_rng_state


class WanNSASelfForcingStreamingTrainer(StreamingCheckpointMixin, WanNSASelfForcingTrainer):
    """
    Streaming long-video trainer that generates and trains one chunk at a time with a persistent KV cache.
    Re-encodes the previous window's last frames as a VAE boundary block and writes its clean KV first.
    """
    _sf_trainer_family = "nsa-self-forcing-streaming"
    streaming_boundary_mode = "vae_roundtrip_block"

    def __init__(self, config):
        self.streaming_chunk_size = config.get("streaming_chunk_size", 21)
        self.streaming_max_length = config.get("streaming_max_length", 240)
        self.streaming_min_new_frame = config.get("streaming_min_new_frame", 18)
        self.train_first_chunk = config.get("train_first_chunk", True)
        self.streaming_possible_max_length = config.get("streaming_possible_max_length", None)
        self.streaming_generator_schedule = config.get("streaming_generator_schedule", "fixed")
        # Validate before loading models or comparing a checkpoint recipe.
        generator_update_due(
            0, config.get("dfake_gen_update_ratio", 5),
            schedule=self.streaming_generator_schedule,
        )
        # Complete resume can defer saves until a rollout ends, omitting its KV.
        self.streaming_checkpoint_at_rollout_boundary = config.get(
            "streaming_checkpoint_at_rollout_boundary", False)
        if type(self.streaming_checkpoint_at_rollout_boundary) is not bool:
            raise ValueError("streaming_checkpoint_at_rollout_boundary must be a boolean")

        # Round max_length up to a whole number of chunks to avoid tail waste.
        # Normalize before the parent compares checkpoint recipes.
        cs = self.streaming_chunk_size
        fpb = config.get("frame_per_block", 3)
        if (cs <= 0 or fpb <= 0 or cs % fpb or self.streaming_min_new_frame <= 0
                or self.streaming_min_new_frame > cs - fpb or self.streaming_min_new_frame % fpb
                or self.streaming_max_length <= 0):
            raise ValueError(
                "Streaming lengths must be positive and block-aligned, with "
                "min_new_frame + frame_per_block <= chunk_size for the VAE boundary"
            )
        aligned = math.ceil(self.streaming_max_length / cs) * cs
        self.streaming_max_length = aligned
        if self.streaming_possible_max_length is not None:
            if not self.streaming_possible_max_length or any(v <= 0 for v in self.streaming_possible_max_length):
                raise ValueError("streaming_possible_max_length requires nonempty positive lengths")
            self.streaming_possible_max_length = [
                math.ceil(v / cs) * cs for v in self.streaming_possible_max_length
            ]
        allowed_lengths = self.streaming_possible_max_length or [self.streaming_max_length]
        if not self.train_first_chunk and min(allowed_lengths) < cs + fpb + self.streaming_min_new_frame:
            raise ValueError("Skipping the first chunk requires room for at least one subsequent training chunk")

        self.streaming_active = False
        self.streaming_kv_cache: Optional[MMCache] = None
        self.streaming_state: Dict[str, Any] = {}
        self._last_streaming_eval_step = 0
        self._pending_streaming_resume = None

        super().__init__(config)
        self._restore_streaming_runtime()

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

    def _streaming_kv_dtype(self) -> torch.dtype:
        """Use the generator's forward dtype before and after FSDP reshards."""
        return unsharded_param_dtype(self.generator)

    def _ensure_streaming_kv_cache(self, dummy_latents: torch.Tensor):
        """Build the persistent streaming KV cache on first call; subsequent sequences reuse it via reset()."""
        if self.streaming_kv_cache is not None:
            if self.streaming_kv_cache.available_shape["k_cache"][0] == dummy_latents.shape[0]:
                return
            # Text loaders may yield a smaller final batch. Its cache capacity
            # must follow the actual prompt batch, including after a restart.
            self.streaming_kv_cache = None
        kv_dtype = self._streaming_kv_dtype()
        self.streaming_kv_cache = self.build_kv_cache(
            model=self.generator, latents=dummy_latents, dtype=kv_dtype,
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
            dtype=kv_dtype, device=self.generator.device,
        )
        self.streaming_kv_cache.preallocate_offload_buffer(
            sample=sample,
            segment_cids=total_chunks,
        )
        del sample

    def start_new_sequence(self):
        """Sample fresh prompts and initialise a new streaming sequence."""
        self._reset_streaming_state()

        try:
            batch = next(self._streaming_data_iter)
        except StopIteration:
            if self.complete_sf_resume:
                raise RuntimeError("Streaming prompt epochs exhausted before max_train_steps") from None
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

        self._ensure_streaming_kv_cache(self._streaming_cache_shape_tensor(len(text)))

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
        boundary_cost = self.frame_per_block if self.streaming_state["previous_frames"] is not None else 0
        return cur < max_len and (cur + boundary_cost + self.streaming_min_new_frame) <= max_len

    @torch.no_grad()
    def _vae_boundary_block(
        self,
        prev_chunk_latent: torch.Tensor,
        num_boundary_latents: int,
    ) -> torch.Tensor:
        """Decode the previous window and re-encode its tail as a fresh VAE clip."""
        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=self.vae.device, dtype=self.vae.dtype,
        ).reshape(1, -1, 1, 1, 1)
        latents_std = torch.tensor(
            self.vae.config.latents_std,
            device=self.vae.device, dtype=self.vae.dtype,
        ).reshape(1, -1, 1, 1, 1)
        z = prev_chunk_latent.to(device=self.vae.device, dtype=self.vae.dtype)
        z = z * latents_std + latents_mean

        pixels = self.vae.decode(z).sample
        # Wan's first latent represents one RGB frame, subsequent latents four.
        n_pix = 4 * num_boundary_latents - 3
        if pixels.shape[2] < n_pix:
            raise ValueError(f"Previous window has {pixels.shape[2]} RGB frames; boundary requires {n_pix}")
        boundary = vae_encode(self.vae, pixels[:, :, -n_pix:].contiguous())
        expected_shape = (*prev_chunk_latent.shape[:2], num_boundary_latents, *prev_chunk_latent.shape[3:])
        if tuple(boundary.shape) != expected_shape:
            raise ValueError(f"Unexpected VAE boundary shape: {tuple(boundary.shape)} != {expected_shape}")
        return boundary.detach().to(device=prev_chunk_latent.device, dtype=prev_chunk_latent.dtype)

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
                if name in kv.chunk_per_head_types:
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
        """Generate a window, reserving a full VAE boundary block after the first window."""
        if not self.streaming_active or not self.can_generate_more():
            self.start_new_sequence()
            if not self.train_first_chunk:
                self.generate_next_chunk(requires_grad=False)
                # Re-enter the switch override after warmup, so a switch at
                # that boundary applies to the first trainable chunk too.
                return self.generate_next_chunk(requires_grad=requires_grad)

        state = self.streaming_state
        current_length = state["current_length"]
        previous_frames = state["previous_frames"]
        batch_size = len(state["condition_dict"]["context"])
        fpb = self.frame_per_block
        cs = self.streaming_chunk_size
        has_boundary = previous_frames is not None

        if self.streaming_kv_cache is not None:
            self.streaming_kv_cache.lru_reset_stats()

        if has_boundary:
            max_new = min(state["temp_max_length"] - current_length - fpb, cs - fpb)
            possible = list(range(self.streaming_min_new_frame, max_new + 1, fpb))
            if not possible:
                raise RuntimeError("Streaming window has no room for its VAE boundary and new frames")

            sel = torch.zeros(1, dtype=torch.int32, device=self.device)
            if not dist.is_initialized() or dist.get_rank() == 0:
                sel.fill_(random.randint(0, len(possible) - 1))
            if dist.is_initialized():
                dist.broadcast(sel, src=0)
            new_frames = possible[sel.item()]

            boundary_latents = self._vae_boundary_block(previous_frames, num_boundary_latents=fpb)
            # The boundary occupies new, block-aligned positions in the persistent
            # cache. Its clean context pass is detached; new generation keeps gradients.
            with torch.no_grad():
                context_t = torch.zeros(
                    batch_size, device=boundary_latents.device, dtype=boundary_latents.dtype,
                )
                self.generator(
                    x=boundary_latents, t=context_t, **state["condition_dict"],
                    start_frame=current_length, kv_cache=self.streaming_kv_cache,
                )
            if self.enable_kv_evict and self.streaming_kv_cache is not None:
                self.streaming_kv_cache.flush_offload()
            sim_start_frame = current_length + fpb
        else:
            new_frames = cs
            boundary_latents = None
            sim_start_frame = current_length

        noise = torch.randn(
            batch_size, self.latents_shape[0], new_frames,
            self.latents_shape[2], self.latents_shape[3],
            device=self.device, dtype=self.dtype,
        )

        denoised_new, target_ts = self.backward_simulation_chunk(
            transformer=self.generator,
            noise_scheduler=self.noise_scheduler,
            num_inference_steps=self.max_distillation_steps,
            noise=noise,
            condition_dict=state["condition_dict"],
            start_frame=sim_start_frame,
            kv_cache=self.streaming_kv_cache,
            requires_grad=requires_grad,
        )

        if has_boundary:
            full_chunk = torch.cat([boundary_latents, denoised_new], dim=2)
        else:
            full_chunk = denoised_new

        frames_to_save = full_chunk[:, :, -cs:].detach().clone()

        gradient_mask = torch.ones_like(full_chunk, dtype=torch.bool)
        if has_boundary:
            gradient_mask[:, :, :fpb] = False

        state["current_length"] += (fpb + new_frames) if has_boundary else new_frames
        state["previous_frames"] = frames_to_save

        info = dict(
            target_timestep=target_ts,
            gradient_mask=gradient_mask,
            new_frames=new_frames,
            boundary_frames=fpb if has_boundary else 0,
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

        # Key the decision by optimizer step, so every accumulation microstep
        # and every distributed rank trains the same side of the GAN.
        is_gen_step = generator_update_due(
            self.global_step, self.dfake_gen_update_ratio,
            schedule=getattr(self, "streaming_generator_schedule", "fixed"),
        )

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
        if self.complete_sf_resume:
            with self._preserve_streaming_evaluation_state():
                super().evaluate_init()
        else:
            super().evaluate_init()

    def evaluate(self):
        if self.complete_sf_resume:
            with self._preserve_streaming_evaluation_state():
                super().evaluate()
            return
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
        if self.complete_sf_resume:
            return self._train_with_complete_resume()
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

    def _train_with_complete_resume(self):
        if self.global_step >= self.max_train_steps:
            self.logger.info("Requested streaming training is already complete.")
            return
        if self.checkpointing_steps <= 0 or self.validation_steps <= 0:
            raise ValueError("Streaming checkpoint/validation intervals must be positive")
        self.logger.info("***** Running streaming training with complete resume *****")
        self._streaming_data_iter = iter(self._sf_training_batches())
        # Do not repeat startup evaluation after a restart. Evaluation at any
        # point preserves both training RNG and the persistent rollout cache.
        if self._resume_checkpoint is None:
            self.evaluate_init()
        if self._resume_rng_state is not None:
            restore_rng_state(self._resume_rng_state)
            self._resume_rng_state = None
        self.data_timer.tic()
        micro_step = 0
        save_pending = False
        while self.global_step < self.max_train_steps:
            micro_step += 1
            self.accumulation_index = micro_step % self.gradient_accumulation_steps
            self.setup_ema_model()
            self.generator.train()
            self.fake_net.train()
            self.data_timer.toc()
            log_dict = self.train_step()
            self.data_timer.tic()
            if self.accumulation_index != 0:
                continue
            self.global_step += 1
            self.setup_ema_model()
            self._log_progress(log_dict)
            at_end = self.global_step >= self.max_train_steps
            at_boundary = not self.streaming_active or not self.can_generate_more()
            eval_due = self.global_step - self._last_streaming_eval_step >= self.validation_steps
            if (at_boundary and eval_due) or at_end:
                if self.streaming_checkpoint_at_rollout_boundary:
                    # Release pinned host buffers before eval allocates another long-video cache.
                    self._reset_streaming_state()
                    self.streaming_kv_cache = None
                    gc.collect()
                self.evaluate()
                self._last_streaming_eval_step = self.global_step
            # Save after all state-changing work for this step, including the
            # eval cursor. The default policy preserves mid-rollout KV.
            save_due = self.global_step % self.checkpointing_steps == 0
            if self.streaming_checkpoint_at_rollout_boundary:
                save_pending = save_pending or save_due
                if at_end or (save_pending and at_boundary):
                    # Final saves also discard any unfinished rollout. A resumed
                    # completed run exits before attempting another training step.
                    self._reset_streaming_state()
                    self.save_models()
                    save_pending = False
            elif save_due or at_end:
                self.save_models()
        self.logger.info("Finish streaming training")
