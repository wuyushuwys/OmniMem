import os
import safetensors.torch
from typing import Union
from collections.abc import MutableMapping
import torch
from torch.distributed.fsdp import fully_shard
from diffusers.optimization import get_scheduler
import torch.distributed as dist
from omnimem.models.transformers.wan_model import WanAttentionBlock
from omnimem.models.transformers.causal_wan_nsa_model import CausalWanNSAModel, CausalWanNSAttentionBlock
from omnimem.models.cache import MMCache
from omnimem.utils.torch_utils import (
    create_ema_model,
    resume_model,
    load_and_broadcast_diffuser,
    get_fsdp_state_dict,
    save_fsdp_checkpoint,
    save_dcp_checkpoint,
    resume_dcp_checkpoint,
)
from omnimem.utils.misc import (
    unwrap_model,
    is_main_process,
)
from omnimem.utils.lora_utils import configure_lora_for_model, gather_lora_state_dict, load_lora_weights
from omnimem.trainer.wan_self_forcing_trainer import WanSelfForcingTrainer
from omnimem.pipelines import CausalWanT2VPipeline
from omnimem.evaluate import evaluation_wan


class WanNSASelfForcingTrainer(WanSelfForcingTrainer):
    generator: CausalWanNSAModel

    ema: Union[None, CausalWanNSAModel] = None

    checkpoint_modules = (WanAttentionBlock, CausalWanNSAttentionBlock)

    def __init__(self, config):
        self.config = config
        # Read resume_ckpt_path before super().__init__ so _get_resume_dir can see it.
        self.resume_ckpt_path = self._get_and_record("resume_ckpt_path", None)
        self.nsa_kwargs = self._get_and_record("nsa_kwargs")
        self.enable_gate_init = self._get_and_record("enable_gate_init", False)
        self.kv_cache_lru_max_size = self._get_and_record("kv_cache_lru_max_size", None)
        self.enable_block_level_cache = self._get_and_record("enable_block_level_cache", False)
        self.enable_chunk_per_head_cache = self._get_and_record("enable_chunk_per_head_cache", False)
        self.per_head_gather_variant = self._get_and_record("per_head_gather_variant", "grouped")
        self.adapter = self._get_and_record("adapter", None)
        self.lora_ckpt = self._get_and_record("lora_ckpt", None)
        self.is_lora_enabled = False
        self.is_fake_net_lora_enabled = False

        super().__init__(config)

        self._eval_kwargs = dict(
            pipeline_cls=CausalWanT2VPipeline,
            frame_per_block=self.frame_per_block,
            sink_size=self.sink_size,
            window_size=self.window_size,
            enable_block_level_cache=self.enable_block_level_cache,
            enable_chunk_per_head_cache=self.enable_chunk_per_head_cache,
            enable_kv_evict=self.enable_kv_evict,
            lru_max_size=self.kv_cache_lru_max_size,
        )

    def build_model(self) -> CausalWanNSAModel:
        model = load_and_broadcast_diffuser(
            model_cls=CausalWanNSAModel,
            model_name_or_path=self.config.model_path,
            device=self.device,
            low_cpu_mem_usage=False,
            **self.nsa_kwargs
        )
        if self.enable_gate_init:
            self.logger.info("Initialize g_proj weights and boardcast")
            model.init_gate()
            if dist.is_initialized():
                for name, p in model.named_parameters():
                    if 'g_proj' in name:
                        dist.broadcast(p.data.to(self.device), src=0)

        if self.adapter and self.adapter.get("type") == "lora":
            model = configure_lora_for_model(
                transformer=model,
                block_class_names=['CausalWanNSAttentionBlock'],
                lora_config=self.adapter,
                is_main_process=is_main_process(),
            )
            self.is_lora_enabled = True

        return model

    def setup_models(self):
        super().setup_models()
        if (self.adapter and self.adapter.get("type") == "lora"
                and self.adapter.get("apply_to_critic", False)):
            self.fake_net = configure_lora_for_model(
                transformer=self.fake_net,
                block_class_names=['WanAttentionBlock'],
                lora_config=self.adapter,
                is_main_process=is_main_process(),
            )
            self.is_fake_net_lora_enabled = True

    def wrap_model_fsdp(self):
        if dist.is_initialized():
            gen_blocks = (self.generator.base_model.model.blocks
                          if self.is_lora_enabled else self.generator.blocks)
            for block in gen_blocks:
                fully_shard(block, **self.fsdp_kwargs)
            fully_shard(self.generator, **self.fsdp_kwargs.root_kwargs)

            fn_blocks = (self.fake_net.base_model.model.blocks
                         if self.is_fake_net_lora_enabled else self.fake_net.blocks)
            for block in fn_blocks:
                fully_shard(block, **self.fsdp_kwargs)
            fully_shard(self.fake_net, **self.fsdp_kwargs.root_kwargs)
        else:
            self.generator.to(self.device)
            self.fake_net.to(self.device)

    def _get_resume_dir(self):
        """Prefer local checkpoint for crash-restart; fall back to resume_ckpt_path for cold start."""
        local = os.path.join(self.output_dir, "checkpoints")
        if os.path.exists(os.path.join(local, "checkpoint.pth")):
            if self.resume_ckpt_path:
                self.logger.info(
                    f"resume_ckpt_path={self.resume_ckpt_path} ignored; local "
                    f"checkpoint at {local} takes precedence."
                )
            return local
        if self.resume_ckpt_path:
            self.logger.info(f"Resuming full checkpoint from external path: {self.resume_ckpt_path}")
            return self.resume_ckpt_path
        return local

    def resume_from_checkpoint(self):
        checkpoint_path = self.resume_dir
        if self.is_lora_enabled:
            lora_path = self.lora_ckpt or os.path.join(checkpoint_path, "lora_weights.pt")
            if os.path.exists(lora_path):
                ckpt = torch.load(lora_path, map_location="cpu")
                if "generator_lora" in ckpt:
                    load_lora_weights(self.generator, ckpt["generator_lora"],
                                      is_main_process=is_main_process())
                    if self.is_fake_net_lora_enabled and "fake_net_lora" in ckpt:
                        load_lora_weights(self.fake_net, ckpt["fake_net_lora"],
                                          is_main_process=is_main_process())
                else:
                    load_lora_weights(self.generator, ckpt, is_main_process=is_main_process())
            if not self.is_fake_net_lora_enabled:
                resume_model(os.path.join(checkpoint_path, "fake_net"), self.fake_net)
        else:
            resume_model(os.path.join(checkpoint_path, "generator"), self.generator)
            resume_model(os.path.join(checkpoint_path, "fake_net"), self.fake_net)

    def save_models(self):
        output_dir = self.output_dir
        checkpoint_path = os.path.join(output_dir, "checkpoints")

        if self.is_lora_enabled:
            gen_lora_sd = gather_lora_state_dict(self.generator)
            lora_ckpt = {"generator_lora": gen_lora_sd, "step": self.global_step}

            if self.is_fake_net_lora_enabled:
                lora_ckpt["fake_net_lora"] = gather_lora_state_dict(self.fake_net)
            else:
                fake_net_sd = get_fsdp_state_dict(unwrap_model(self.fake_net))

            if is_main_process():
                self.logger.info(f"saving LoRA checkpoint to {output_dir} ...")
                torch.save(lora_ckpt, os.path.join(checkpoint_path, "lora_weights.pt"))
                self.logger.info("Save LoRA weights.")

                if not self.is_fake_net_lora_enabled:
                    save_dir = os.path.join(checkpoint_path, "fake_net")
                    self.fake_net.save_config(save_directory=save_dir)
                    safetensors.torch.save_file(
                        fake_net_sd,
                        os.path.join(save_dir, "diffusion_pytorch_model.safetensors"),
                    )
                    self.logger.info("Save fake-net.")

                save_fsdp_checkpoint(None, None, self.global_step, output_dir=checkpoint_path)
                self.logger.info("Save checkpoints.")

            save_dcp_checkpoint(
                self.generator, self.generator_optimizer, self.global_step,
                output_dir=os.path.join(checkpoint_path, "checkpoint_dcp_generator"),
            )
            save_dcp_checkpoint(
                self.fake_net, self.critic_optimizer, self.global_step,
                output_dir=os.path.join(checkpoint_path, "checkpoint_dcp_critic"),
            )
        else:
            super().save_models()

    def setup_optimizers_and_lr_scheduler(self):
        # setup generator
        self.generator_optimizer = torch.optim.AdamW(
            list(filter(lambda p: p.requires_grad, self.generator.parameters())),
            lr=self.learning_rate,
            weight_decay=self.adam_weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon,
            fused=True,
        )

        self.generator_lr_scheduler = get_scheduler(
            self.lr_scheduler_type,
            optimizer=self.generator_optimizer,
            num_warmup_steps=self.lr_warmup_steps // self.dfake_gen_update_ratio,
            num_training_steps=self.max_train_steps,
            **self.lr_scheduler_kwargs
        )

        # setup fake_net
        critic_trainable_params = list(filter(lambda p: p.requires_grad, self.fake_net.parameters()))
        self.critic_optimizer = torch.optim.AdamW(
            critic_trainable_params,
            lr=self.critic_learning_rate,
            betas=(self.critic_adam_beta1, self.critic_adam_beta2),
            weight_decay=self.critic_adam_weight_decay,
            eps=self.critic_adam_epsilon,
            fused=True,
        )

        self.critic_lr_scheduler = get_scheduler(
            self.lr_scheduler_type,
            optimizer=self.critic_optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.max_train_steps,
            **self.lr_scheduler_kwargs
        )

        checkpoint_path = self.resume_dir
        resume_dcp_checkpoint(
            self.generator, self.generator_optimizer,
            os.path.join(checkpoint_path, "checkpoint_dcp_generator"),
        )
        resume_dcp_checkpoint(
            self.fake_net, self.critic_optimizer,
            os.path.join(checkpoint_path, "checkpoint_dcp_critic"),
        )

    def model_compile(self, mode="max-autotune-no-cudagraphs"):
        self.real_net.compile(mode=mode, dynamic=False)
        self.fake_net.compile(mode=mode, dynamic=False)

    def setup_ema_model(self, try_resume=False):
        if self.is_lora_enabled:
            return
        if self.ema_model and self.ema_start_step <= self.global_step and self.ema is None:
            self.ema, info = create_ema_model(unwrap_model(self.generator), module_cls=CausalWanNSAModel)
            if try_resume:
                resume_model(path=os.path.join(self.resume_dir, 'ema'), module=self.ema)
            for block in self.ema.blocks:
                fully_shard(block, **self.fsdp_kwargs)
            fully_shard(self.ema, **self.fsdp_kwargs.root_kwargs)

    def build_kv_cache(
            self,
            model,
            latents,
    ) -> MMCache:
        _, patch_size, _ = model.patch_size
        batch_size, num_channels, latent_num_frames, latent_height, latent_width = latents.shape

        # create standard size cache
        seq_len_per_frame = latent_height * latent_width // (patch_size * patch_size)
        seq_len = seq_len_per_frame * latent_num_frames
        cache_shape = (
            batch_size,
            seq_len,
            model.num_heads,
            model.dim // model.num_heads
        )

        # create compressed size cache
        compress_seq_len_per_frame = seq_len_per_frame // (model.config.pool_w * model.config.pool_h)
        compress_seq_len = seq_len // (model.config.pool_w * model.config.pool_h)
        compress_cache_shape = (
            batch_size,
            compress_seq_len,
            model.num_heads,
            model.dim // model.num_heads
        )

        kv_cache = MMCache(
            config=dict(
                cache_type=["k_cache", 'v_cache', "k_cmp_cache", "v_cmp_cache"],
            ),
            seq_dim=1,
            available_shape=dict(
                k_cache=cache_shape,
                v_cache=cache_shape,
                k_cmp_cache=compress_cache_shape,
                v_cmp_cache=compress_cache_shape,
            ),
            num_layers=model.num_layers,
            device=model.device,
            dtype=model.dtype,
            block_seqlen_config=dict(
                k_cache=seq_len_per_frame * self.frame_per_block,
                v_cache=seq_len_per_frame * self.frame_per_block,
                k_cmp_cache=compress_seq_len_per_frame * self.frame_per_block,
                v_cmp_cache=compress_seq_len_per_frame * self.frame_per_block,
            ),
            window_config=dict(
                k_cache=self.window_size if not isinstance(self.window_size, MutableMapping) else self.window_size.get(
                    "kv"),
                v_cache=self.window_size if not isinstance(self.window_size, MutableMapping) else self.window_size.get(
                    "kv"),
                k_cmp_cache=self.window_size if not isinstance(self.window_size,
                                                               MutableMapping) else self.window_size.get("kv_cmp"),
                v_cmp_cache=self.window_size if not isinstance(self.window_size,
                                                               MutableMapping) else self.window_size.get("kv_cmp"),
            ),
            sink_config=dict(
                k_cache=self.sink_size if not isinstance(self.sink_size, MutableMapping) else self.sink_size.get("kv"),
                v_cache=self.sink_size if not isinstance(self.sink_size, MutableMapping) else self.sink_size.get("kv"),
                k_cmp_cache=self.sink_size if not isinstance(self.sink_size, MutableMapping) else self.sink_size.get(
                    "kv_cmp"),
                v_cmp_cache=self.sink_size if not isinstance(self.sink_size, MutableMapping) else self.sink_size.get(
                    "kv_cmp"),
            ),
            lru_max_size=self.kv_cache_lru_max_size if self.kv_cache_lru_max_size is not None else 0,
            per_head_types={'k_cache', 'v_cache'} if self.enable_block_level_cache else None,
            chunk_per_head_types={'k_cache', 'v_cache'} if self.enable_chunk_per_head_cache else None,
            num_heads=model.num_heads if (self.enable_block_level_cache or self.enable_chunk_per_head_cache) else None,
            kv_block_size=(getattr(model.config, 'pool_f', 1)
                           * model.config.pool_h
                           * model.config.pool_w) if self.enable_block_level_cache else 0,
        )

        return kv_cache

    def evaluate_init(self):
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
            commit=True,
            s3_bucket=self.s3_bucket,
            s3_dir=self.s3_dir,
            upload=self.upload,
            **self._eval_kwargs,
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
                n_rows=self.validation_data.get("n_rows", 4),
                tag='ema/',
                device=self.device,
                s3_bucket=self.s3_bucket,
                s3_dir=self.s3_dir,
                upload=self.upload,
                **self._eval_kwargs,
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
            s3_bucket=self.s3_bucket,
            s3_dir=self.s3_dir,
            upload=self.upload,
            **self._eval_kwargs,
        )
        torch.cuda.empty_cache()
