import os
import math
from typing import Union

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from omnimem.models.transformers.wan_model import WanAttentionBlock
from omnimem.models.transformers.causal_wan_model import CausalWanModel, CausalWanAttentionBlock
from omnimem.models.cache import MMCache
from omnimem.pipelines import CausalWanT2VPipeline
from omnimem.schedulers import RectifiedFlowScheduler
from omnimem.utils.torch_utils import (
    create_ema_model,
    resume_model,
    load_and_broadcast_diffuser,
)
from omnimem.utils.train_utils import pred_x0
from omnimem.utils.misc import (
    unwrap_model,
)
from omnimem.evaluate import evaluation_wan
from omnimem.trainer.wan_dmd_trainer import WanDMDTrainer


class WanSelfForcingTrainer(WanDMDTrainer):
    generator: CausalWanModel

    ema: Union[None, CausalWanModel] = None

    checkpoint_modules = (WanAttentionBlock, CausalWanAttentionBlock)

    def __init__(self, config):
        self.config = config
        # Self-Forcing
        self.frame_per_block = self._get_and_record("frame_per_block", None)
        self.same_step_across_blocks = self._get_and_record("same_step_across_blocks", True)
        self.window_size = self._get_and_record("window_size", None)
        self.sink_size = self._get_and_record("sink_size", None)
        self.enable_kv_evict = self._get_and_record("enable_kv_evict", False)
        self.kv_evict_min_gpu_chunks = self._get_and_record("kv_evict_min_gpu_chunks", None)

        super().__init__(config)
        self._kv_free_hooks = []

    def build_model(self) -> CausalWanModel:
        return load_and_broadcast_diffuser(
            model_cls=CausalWanModel,
            model_name_or_path=self.config.model_path,
            device=self.device,
        )

    def setup_ema_model(self, try_resume=False):
        if self.ema_model and self.ema_start_step <= self.global_step and self.ema is None:
            self.ema, info = create_ema_model(unwrap_model(self.generator), module_cls=CausalWanModel)
            if try_resume:
                resume_model(path=os.path.join(self.output_dir, "checkpoints", 'ema'), module=self.ema)
            for block in self.ema.blocks:
                fully_shard(block, **self.fsdp_kwargs)
            fully_shard(self.ema, **self.fsdp_kwargs.root_kwargs)

    def evaluate_init(self):
        if self.original_step and self.original_cfg:
            evaluation_wan(
                transformer=self.real_net,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                vae=self.vae,
                noise_scheduler=self.noise_scheduler,
                output_dir=self.output_dir,
                global_step=self.global_step,
                global_seed=self.global_seed,
                validation_data=self.validation_data,
                original_step=self.original_step,
                original_cfg=self.original_cfg,
                n_rows=self.validation_data.get("n_rows", 4),
                tag='original/',
                device=self.device,
                s3_bucket=self.s3_bucket,
                s3_dir=self.s3_dir,
                upload=self.upload,
            )
        evaluation_wan(
            transformer=self.generator,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            noise_scheduler=self.noise_scheduler,
            output_dir=self.output_dir,
            global_step=self.global_step,
            global_seed=self.global_seed,
            validation_data=self.validation_data,
            n_rows=self.validation_data.get("n_rows", 4),
            device=self.device,
            pipeline_cls=CausalWanT2VPipeline,
            frame_per_block=self.frame_per_block,
            sink_size=self.sink_size,
            window_size=self.window_size,
            enable_block_level_cache=getattr(self, 'enable_block_level_cache', False),
            enable_chunk_per_head_cache=getattr(self, 'enable_chunk_per_head_cache', False),
            enable_kv_evict=getattr(self, 'enable_kv_evict', True),
            lru_max_size=getattr(self, 'kv_cache_lru_max_size', 10),
            commit=True,
            s3_bucket=self.s3_bucket,
            s3_dir=self.s3_dir,
            upload=self.upload,
        )
        torch.cuda.empty_cache()

    def evaluate(self):
        if self.ema is not None:
            evaluation_wan(
                transformer=self.ema,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                vae=self.vae,
                noise_scheduler=self.noise_scheduler,
                output_dir=self.output_dir,
                global_step=self.global_step,
                global_seed=self.global_seed,
                validation_data=self.validation_data,
                n_rows=self.config.validation_data.get("n_rows", 4),
                tag='ema/',
                device=self.device,
                pipeline_cls=CausalWanT2VPipeline,
                frame_per_block=self.frame_per_block,
                sink_size=self.sink_size,
                window_size=self.window_size,
                enable_block_level_cache=getattr(self, 'enable_block_level_cache', False),
                enable_chunk_per_head_cache=getattr(self, 'enable_chunk_per_head_cache', False),
                enable_kv_evict=getattr(self, 'enable_kv_evict', True),
                s3_bucket=self.s3_bucket,
                s3_dir=self.s3_dir,
                upload=self.upload,
            )
        evaluation_wan(
            transformer=self.generator,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            noise_scheduler=self.noise_scheduler,
            output_dir=self.output_dir,
            global_step=self.global_step,
            global_seed=self.global_seed,
            validation_data=self.validation_data,
            n_rows=self.validation_data.get("n_rows", 4),
            device=self.device,
            pipeline_cls=CausalWanT2VPipeline,
            frame_per_block=self.frame_per_block,
            sink_size=self.sink_size,
            window_size=self.window_size,
            enable_block_level_cache=getattr(self, 'enable_block_level_cache', False),
            enable_chunk_per_head_cache=getattr(self, 'enable_chunk_per_head_cache', False),
            enable_kv_evict=getattr(self, 'enable_kv_evict', True),
            s3_bucket=self.s3_bucket,
            s3_dir=self.s3_dir,
            upload=self.upload,
        )
        torch.cuda.empty_cache()

    def build_kv_cache(
            self,
            model,
            latents,
    ) -> MMCache:
        _, patch_size, _ = model.patch_size
        batch_size, num_channels, latent_num_frames, latent_height, latent_width = latents.shape
        seq_len_per_frame = latent_height * latent_width // (patch_size * patch_size)
        seq_len = seq_len_per_frame * latent_num_frames
        cache_shape = (
            batch_size,
            seq_len,
            model.num_heads,
            model.dim // model.num_heads
        )

        kv_cache = MMCache(
            config=dict(
                cache_type=["k_cache", 'v_cache'],
            ),
            seq_dim=1,
            available_shape=dict(
                k_cache=cache_shape,
                v_cache=cache_shape,
            ),
            num_layers=model.num_layers,
            device=model.device,
            dtype=model.dtype,
            block_seqlen_config=dict(
                k_cache=seq_len_per_frame * self.frame_per_block,
                v_cache=seq_len_per_frame * self.frame_per_block,
            ),
            window_config=dict(
                k_cache=self.window_size,
                v_cache=self.window_size,
            ),
            sink_config=dict(
                k_cache=self.sink_size,
                v_cache=self.sink_size,
            ),
        )
        return kv_cache

    def _register_kv_free_hooks(self, model, kv_cache: MMCache):
        """Register per-block backward hooks to free the last KV cache chunk."""
        self._remove_kv_free_hooks()
        for layer_idx, block in enumerate(model.blocks):
            def make_hook(li):
                def hook(module, grad_input, grad_output):
                    kv_cache.free_last_chunk(layer_idx=li)

                return hook

            h = block.self_attn.register_full_backward_hook(make_hook(layer_idx))
            self._kv_free_hooks.append(h)

    def _remove_kv_free_hooks(self):
        for h in self._kv_free_hooks:
            h.remove()
        self._kv_free_hooks.clear()

    def backward_simulation(
            self,
            transformer: CausalWanModel,
            noise_scheduler: RectifiedFlowScheduler,
            num_inference_steps,
            latents: torch.Tensor,
            condition_dict,
    ):
        batch_size, num_channels, latent_num_frames, latent_height, latent_width = latents.shape
        noisy_latents = torch.randn_like(latents)
        kv_cache = self.build_kv_cache(model=transformer, latents=latents)

        if torch.is_grad_enabled():
            self._register_kv_free_hooks(transformer, kv_cache)
        noise_scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=noisy_latents.device,
            noise=noisy_latents,
        )

        output_latent = torch.zeros_like(noisy_latents)
        inference_timesteps = noise_scheduler.timesteps
        if self.same_step_across_blocks:
            selected_step = torch.randint(0, inference_timesteps.shape[0], size=(1,),
                                          device=noisy_latents.device,
                                          dtype=torch.long)
            if dist.is_initialized():
                dist.broadcast(selected_step, src=0)
        else:
            selected_step = -1
        num_blocks = math.ceil(latent_num_frames / self.frame_per_block)
        target_timestep = torch.empty(noisy_latents.shape[0], num_blocks).to(inference_timesteps)
        _chunk_denoised_preds = []

        for idx, start_frame in enumerate(range(0, latent_num_frames, self.frame_per_block)):

            if not self.same_step_across_blocks:
                selected_step = torch.randint(0, inference_timesteps.shape[0], size=(1,),
                                              device=noisy_latents.device,
                                              dtype=torch.long)
                if dist.is_initialized():
                    dist.broadcast(selected_step, src=0)
            noisy_latent = noisy_latents[:, :, start_frame:start_frame + self.frame_per_block].contiguous()
            for index, t in enumerate(inference_timesteps[:selected_step]):
                timestep = t.expand(noisy_latent.shape[0])
                with torch.no_grad():
                    noise_pred = transformer(
                        x=noisy_latent,
                        t=timestep,
                        **condition_dict,
                        start_frame=start_frame,
                        kv_cache=kv_cache,
                    )
                denoised_pred = pred_x0(noise_pred, timestep, noisy_latent, noise_scheduler)
                denoised_pred = denoised_pred.contiguous()
                next_timestep = inference_timesteps[index + 1].expand(denoised_pred.shape[0])
                noisy_latent = noise_scheduler.add_noise(
                    denoised_pred,
                    noise=torch.randn_like(denoised_pred),
                    timesteps=next_timestep
                )
            timestep = inference_timesteps[selected_step].expand(noisy_latent.shape[0])
            noise_pred = transformer(
                x=noisy_latent,
                t=timestep,
                **condition_dict,
                start_frame=start_frame,
                kv_cache=kv_cache,
            )
            denoised_pred = pred_x0(noise_pred, timestep, noisy_latent, noise_scheduler)
            denoised_pred = denoised_pred.contiguous()

            output_latent[:, :, start_frame:start_frame + self.frame_per_block] = denoised_pred
            target_timestep[:, idx] = inference_timesteps[selected_step]
            with torch.no_grad():
                context_timestep = torch.zeros_like(timestep)
                transformer(
                    x=denoised_pred,
                    t=context_timestep,
                    **condition_dict,
                    start_frame=start_frame,
                    kv_cache=kv_cache,
                )

            pass

        return output_latent, target_timestep

    def train_step(self, *args, **kwargs):
        output = super().train_step(*args, **kwargs)
        self._remove_kv_free_hooks()
        return output
