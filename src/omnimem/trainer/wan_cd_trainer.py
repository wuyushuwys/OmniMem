import os
import math
import random
import wandb

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.optim.lr_scheduler import LRScheduler

from peft import PeftModel
from diffusers.optimization import get_scheduler

from omnimem.models.flex_attention_utils import create_causal_block_mask_cached
from omnimem.models.transformers.causal_wan_model import CausalWanModel, CausalWanAttentionBlock
from omnimem.pipelines import CausalWanT2VPipeline
from omnimem.schedulers import RectifiedFlowScheduler
from omnimem.utils.train_utils import vae_encode
from omnimem.utils.torch_utils import (
    get_fsdp_state_dict,
    save_fsdp_checkpoint,
    load_and_broadcast_diffuser,
    resume_model,
    save_sharded_safetensors,
    update_ema_fsdp,
    resume_dcp_checkpoint,
    save_dcp_checkpoint,
)
from omnimem.utils.meter import TimerMeter
from omnimem.evaluate import evaluation_wan
from omnimem.utils.misc import (
    is_main_process,
    wait_for_everyone,
    reduce_dict,
    unwrap_model,
)

from omnimem.trainer.base_wan_trainer import BaseWanTrainer


class WanConsistencyDistillTrainer(BaseWanTrainer):
    """
    Naive Consistency Distillation trainer for causal Wan.

    Student is trained to be self-consistent across adjacent (t, t_next) pairs
    on a discretized flow-matching ODE schedule, using a frozen teacher for one-step
    Euler steps and an EMA student as the target network.
    """

    model: CausalWanModel          # student / generator
    ema: CausalWanModel            # generator_ema (target network)
    teacher: CausalWanModel        # frozen teacher

    optimizer: torch.optim.Optimizer
    lr_scheduler: LRScheduler

    forward_timer: TimerMeter
    backward_timer: TimerMeter

    checkpoint_modules = CausalWanAttentionBlock

    def __init__(self, config):
        self.config = config
        self.frame_per_block = self._get_and_record("frame_per_block", None)
        self.timestep_shift = self._get_and_record("timestep_shift", None)
        self.lora_weights = self._get_and_record("lora_weights")

        # Number of discrete CD timestep grid points.
        self.discrete_cd_N = self._get_and_record("discrete_cd_N", 48)
        self.guidance_scale = self._get_and_record("guidance_scale", 5.0)
        self.use_cfg = self._get_and_record("use_cfg", True)
        self.teacher_path = self._get_and_record("teacher_path", None)
        self.diffusion_forcing = self._get_and_record("diffusion_forcing", False)
        self.teacher_forcing = self._get_and_record("teacher_forcing", False)
        self.negative_prompt = self._get_and_record("negative_prompt", "")
        self._neg_embeddings_cache = None

        super().__init__(config)
        self.model_compile(mode="max-autotune-no-cudagraphs")

    def load_noise_scheduler(self):
        self.noise_scheduler = RectifiedFlowScheduler.from_config(
            self.scheduler_path,
            shift=self.timestep_shift,
        )
        # Pre-discretize the CD grid; clone so validation's set_timesteps doesn't overwrite it.
        self.noise_scheduler.set_timesteps(
            num_inference_steps=self.discrete_cd_N,
            device=self.device,
        )
        self.cd_sigmas = self.noise_scheduler.sigmas.detach().clone().to(self.device)
        self.cd_timesteps = self.noise_scheduler.timesteps.detach().clone().float().to(self.device)

        self.logger.info(f"scheduler path {self.scheduler_path}")
        self.logger.info(
            f"{self.noise_scheduler.__class__.__name__} - "
            f"{self.noise_scheduler.config.prediction_type} - "
            f"shift={self.timestep_shift} - "
            f"discrete_cd_N={self.discrete_cd_N}"
        )
        self.logger.info(
            f"cd_sigmas length={len(self.cd_sigmas)}, "
            f"first 3={self.cd_sigmas[:3].tolist()}, "
            f"mid={self.cd_sigmas[len(self.cd_sigmas)//2].item():.4f}, "
            f"last 3={self.cd_sigmas[-3:].tolist()}"
        )
        self.logger.info(
            f"cd_timesteps length={len(self.cd_timesteps)}, "
            f"first 3={self.cd_timesteps[:3].tolist()}, "
            f"last 3={self.cd_timesteps[-3:].tolist()}"
        )

    def build_model(self):
        model: CausalWanModel = load_and_broadcast_diffuser(
            model_cls=CausalWanModel,
            model_name_or_path=self.model_path,
            device=self.device,
            low_cpu_mem_usage=False,
        )
        self.logger.info(f"Load student {model.__class__.__name__} from {self.model_path}")

        if self.lora_weights is not None:
            model = PeftModel.from_pretrained(model, self.lora_weights)
            model.requires_grad_(True)
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False
        return model

    def build_teacher(self):
        """Build the frozen teacher; teacher_path falls back to model_path."""
        teacher_path = self.teacher_path or self.model_path
        teacher: CausalWanModel = load_and_broadcast_diffuser(
            model_cls=CausalWanModel,
            model_name_or_path=teacher_path,
            device=self.device,
            low_cpu_mem_usage=False,
        )
        teacher.requires_grad_(False)
        teacher.eval()
        self.logger.info(f"Load teacher {teacher.__class__.__name__} from {teacher_path}")

        if self.lora_weights is not None:
            pass
        return teacher

    def setup_models(self):
        model = self.build_model()
        self.load_model_checkpoints(model)
        self.possible_apply_gradient_checkpointing(model)
        self.count_model_parameter(model, name="Student")
        self.model = model

        teacher = self.build_teacher()
        self.possible_apply_gradient_checkpointing(teacher)
        self.count_model_parameter(teacher, name="Teacher")
        self.teacher = teacher

    def resume_from_checkpoint(self):
        checkpoint_path = os.path.join(self.output_dir, "checkpoints")
        resume_model(os.path.join(checkpoint_path, "model"), self.model)

    def wrap_model_fsdp(self):
        if dist.is_initialized():
            for block in self.model.blocks:
                fully_shard(block, **self.fsdp_kwargs)
            fully_shard(self.model, **self.fsdp_kwargs.root_kwargs)

            for block in self.teacher.blocks:
                fully_shard(block, **self.fsdp_kwargs)
            fully_shard(self.teacher, **self.fsdp_kwargs.root_kwargs)
        else:
            self.model.to(self.device)
            self.teacher.to(self.device)

    def create_ema_model_peft(self, model, module_cls):
        if isinstance(model, PeftModel):
            ema_config = dict(model.get_base_model().config)
        else:
            ema_config = dict(model.config)
        ema_config["_class_name"] = module_cls.__name__
        ema = module_cls.from_config(ema_config)
        state_dict = get_fsdp_state_dict(model, master_only=False)

        cleaned = {}
        for k, v in state_dict.items():
            if "lora_" in k:
                continue
            k = k.replace("base_model.model.", "")
            k = k.replace(".base_layer.", ".")
            cleaned[k] = v
        if cleaned:
            state_dict = cleaned

        info = ema.load_state_dict(state_dict, strict=False)
        return ema, info

    def setup_ema_model(self, try_resume=False):
        """EMA is mandatory for CD (used as target network at t_next)."""
        self.ema, info = self.create_ema_model_peft(
            unwrap_model(self.model), module_cls=unwrap_model(self.model).__class__
        )
        self.logger.info(
            f"EMA load info: missing={len(info.missing_keys)}, "
            f"unexpected={len(info.unexpected_keys)}"
        )
        if self.lora_weights is not None:
            self.ema = PeftModel.from_pretrained(self.ema, self.lora_weights)

        if try_resume:
            resume_model(
                path=os.path.join(self.output_dir, "checkpoints", "ema"),
                module=self.ema,
            )
        if dist.is_initialized():
            for block in self.ema.blocks:
                fully_shard(block, **self.fsdp_kwargs)
            fully_shard(self.ema, **self.fsdp_kwargs.root_kwargs)
        else:
            self.ema.to(self.device)

    def model_compile(self, mode="max-autotune-no-cudagraphs"):
        self.model.compile(mode=mode)
        self.teacher.compile(mode=mode)
    
    def setup_optimizers_and_lr_scheduler(self):
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
            fused=True,
        )
        self.lr_scheduler = get_scheduler(
            self.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.max_train_steps,
            **self.lr_scheduler_kwargs,
        )

        dcp_path = os.path.join(self.output_dir, "checkpoints", "checkpoint_dcp")
        if os.path.exists(dcp_path):
            resume_dcp_checkpoint(model=self.model, optimizer=self.optimizer, path=dcp_path)

        for _ in range(self.global_step):
            self.lr_scheduler.step()

    def setup_timers(self):
        super().setup_timers()
        self.forward_timer = TimerMeter(
            max_length=self.log_steps * self.gradient_accumulation_steps,
            wait_for_all=False,
        )
        self.backward_timer = TimerMeter(
            max_length=self.log_steps * self.gradient_accumulation_steps,
            wait_for_all=False,
        )

    @property
    def sync_time(self):
        sync_time = torch.tensor(
            [self.data_timer.mavg, self.forward_timer.mavg, self.backward_timer.mavg]
        ).to(self.device)
        if dist.is_initialized():
            dist.all_reduce(sync_time, dist.ReduceOp.AVG)
        return sync_time

    def _log_progress(self, log_dict):
        sync_time = self.sync_time
        wait_for_everyone()
        if wandb.run is not None:
            log_dict.update({
                "global_step": self.global_step,
                "learning_rate": self.lr_scheduler.get_last_lr()[0],
                "time/data": sync_time[0].item(),
                "time/forward": sync_time[1].item(),
                "time/backward": sync_time[2].item(),
            })
            log_dict.update(self.loss_meter.val)
            wandb.log(log_dict, step=int(self.global_step))

        max_train_steps = self.max_train_steps
        if self.global_step % self.log_steps == 0 or self.global_step >= max_train_steps:
            time_log = (
                f"data:{sync_time[0].item():.2f} "
                f"forward:{sync_time[1].item():.2f} "
                f"backward:{sync_time[2].item():.2f}"
            )
            self.logger.info(
                f"step: {self.global_step}/{max_train_steps}"
                f"[{self.global_step / max_train_steps:.02%}]||"
                f"time:[{time_log}]"
                f" loss: {self.loss_meter.mavg.get('train/cd_loss', 0):.04f}"
            )

    def save_models(self):
        output_dir = self.output_dir
        model_state_dict = get_fsdp_state_dict(unwrap_model(self.model))
        ema_state_dict = get_fsdp_state_dict(unwrap_model(self.ema))

        save_dir = os.path.join(output_dir, "checkpoints", "checkpoint_dcp")
        save_dcp_checkpoint(
            self.model, self.optimizer, global_step=self.global_step, output_dir=save_dir
        )

        if is_main_process():
            self.logger.info(f"saving checkpoint to {output_dir} ...")
            save_dir = os.path.join(output_dir, "checkpoints", "model")
            self.model.save_config(save_directory=save_dir)
            save_sharded_safetensors(state_dict=model_state_dict, save_dir=save_dir)
            self.logger.info("Save student model.")

            save_dir = os.path.join(output_dir, "checkpoints", "ema")
            self.ema.save_config(save_directory=save_dir)
            save_sharded_safetensors(state_dict=ema_state_dict, save_dir=save_dir)
            self.logger.info("Save EMA model.")

            save_fsdp_checkpoint(
                None, None, self.global_step,
                output_dir=os.path.join(output_dir, "checkpoints"),
            )
            self.logger.info("Save checkpoints.")

            self.s3.upload_folder(
                folder_path=os.path.join(output_dir, "checkpoints"),
                global_step=self.global_step,
            )

    def evaluate(self):
        if self.lora_weights is not None:
            self.ema.disable_adapter_layers()
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
            tag="ema/",
            timestep_shift=self.timestep_shift,
            device=self.device,
            pipeline_cls=CausalWanT2VPipeline,
            frame_per_block=self.frame_per_block,
            s3_bucket=self.s3_bucket,
            s3_dir=self.s3_dir,
            upload=self.upload,
        )
        if self.lora_weights is not None:
            self.model.disable_adapter_layers()
        evaluation_wan(
            transformer=self.model,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            noise_scheduler=self.noise_scheduler,
            output_dir=self.output_dir,
            global_step=self.global_step,
            global_seed=self.global_seed,
            validation_data=self.validation_data,
            n_rows=self.validation_data.get("n_rows", 4),
            timestep_shift=self.timestep_shift,
            device=self.device,
            pipeline_cls=CausalWanT2VPipeline,
            frame_per_block=self.frame_per_block,
            s3_bucket=self.s3_bucket,
            s3_dir=self.s3_dir,
            upload=self.upload,
        )
        torch.cuda.empty_cache()
        if self.lora_weights is not None:
            self.ema.enable_adapter_layers()
            self.model.enable_adapter_layers()

    def _get_unconditional_embeddings(self, batch_size):
        """Return cached negative-prompt embeddings, re-encoding if batch size changed."""
        if self._neg_embeddings_cache is None:
            with torch.no_grad():
                neg_emb, _ = self.text_encoder(
                    texts=[self.negative_prompt] * batch_size,
                    device=self.device,
                )
                self._neg_embeddings_cache = neg_emb.copy()
            return self._neg_embeddings_cache
        cached = self._neg_embeddings_cache
        if len(cached) == batch_size:
            return cached
        if len(cached) > batch_size:
            return cached[:batch_size]
        with torch.no_grad():
            neg_emb, _ = self.text_encoder(
                texts=[self.negative_prompt] * batch_size,
                device=self.device,
            )
            self._neg_embeddings_cache = neg_emb.copy()
        return self._neg_embeddings_cache

    @staticmethod
    def logit_normal_inverse_cdf(u, mu=0.0, sigma=1.0):
        u = torch.clamp(u, min=1e-5, max=1.0 - 1e-5)
        z = mu + sigma * torch.sqrt(torch.tensor(2.0, device=u.device)) * torch.erfinv(2 * u - 1)
        return torch.sigmoid(z)

    def sample_cd_timesteps(self, batch_size, latent_num_frames, frame_per_block):
        """Sample adjacent (t, t_next) index pairs from the discretized CD grid.

        Returns (sigma_t, sigma_t_next, timestep_t, timestep_t_next).
        """
        device = self.device
        N = self.discrete_cd_N

        if self.diffusion_forcing:
            num_chunks = math.ceil(latent_num_frames / frame_per_block)
            idx = torch.randint(
                low=0, high=N - 1,
                size=(batch_size, num_chunks),
                device=device,
                dtype=torch.long,
            )
        else:
            idx_scalar = random.randrange(N - 1)
            idx = torch.full(
                (batch_size, math.ceil(latent_num_frames / frame_per_block)),
                idx_scalar,
                device=device,
                dtype=torch.long,
            )

        sigma_t = self.cd_sigmas[idx]
        sigma_t_next = self.cd_sigmas[idx + 1]
        timestep_t = self.cd_timesteps[idx]
        timestep_t_next = self.cd_timesteps[idx + 1]

        sigma_t = sigma_t.repeat_interleave(frame_per_block, dim=-1).narrow(1, 0, latent_num_frames)
        sigma_t_next = sigma_t_next.repeat_interleave(frame_per_block, dim=-1).narrow(1, 0, latent_num_frames)
        timestep_t = timestep_t.repeat_interleave(frame_per_block, dim=-1).narrow(1, 0, latent_num_frames)
        timestep_t_next = timestep_t_next.repeat_interleave(frame_per_block, dim=-1).narrow(1, 0, latent_num_frames)

        sigma_t = sigma_t.view(batch_size, 1, latent_num_frames, 1, 1)
        sigma_t_next = sigma_t_next.view(batch_size, 1, latent_num_frames, 1, 1)

        return sigma_t, sigma_t_next, timestep_t, timestep_t_next

    def _build_block_mask(self, block_size, sequence_length):
        """Build the causal BlockMask; NSA subclasses override to add sink/window constraints."""
        return create_causal_block_mask_cached(
            block_size=block_size,
            B=None,
            H=None,
            Q_LEN=sequence_length,
            KV_LEN=sequence_length,
            use_flex_attention=True,
            torch_compile=True,
            teacher_forcing=self.teacher_forcing,
        )

    def _v_to_x0(self, noisy, v, sigma):
        """Convert v-prediction to x0 estimate: x0 = noisy - sigma * v."""
        return noisy - sigma * v

    def consistency_distill(
        self,
        student,
        ema_student,
        teacher,
        latents,
        text_embeddings,
        frame_per_block,
    ):
        patch_size = 2
        batch_size, _, latent_num_frames, latent_height, latent_width = latents.shape
        num_train_t = self.noise_scheduler.config.num_train_timesteps

        sigma_t, sigma_t_next, ts_t, ts_t_next = self.sample_cd_timesteps(
            batch_size, latent_num_frames, frame_per_block
        )

        noise = torch.randn_like(latents)
        latent_t = sigma_t * noise + (1 - sigma_t) * latents

        frame_length = (latent_height // patch_size) * (latent_width // patch_size)
        sequence_length = frame_length * latent_num_frames
        timesteps_t = ts_t.repeat_interleave(frame_length, dim=1)
        timesteps_t_next = ts_t_next.repeat_interleave(frame_length, dim=1)

        block_mask = self._build_block_mask(
            block_size=frame_per_block * frame_length,
            sequence_length=sequence_length,
        )

        # Teacher: one Euler step latent_t -> latent_t_next with CFG.
        teacher_kwargs = dict(
            context=text_embeddings,
            block_mask=block_mask,
            teacher=latents if self.teacher_forcing else None,
        )
        with torch.no_grad():
            v_cond = teacher(
                latent_t.to(teacher.dtype),
                t=timesteps_t.to(teacher.dtype),
                **teacher_kwargs,
            )
            if self.use_cfg and self.guidance_scale != 1.0:
                uncond_emb = self._get_unconditional_embeddings(batch_size)
                teacher_uncond_kwargs = dict(teacher_kwargs)
                teacher_uncond_kwargs["context"] = uncond_emb
                v_uncond = teacher(
                    latent_t.to(teacher.dtype),
                    t=timesteps_t.to(teacher.dtype),
                    **teacher_uncond_kwargs,
                )
                v_pred = v_uncond + self.guidance_scale * (v_cond - v_uncond)
            else:
                v_pred = v_cond

            dt = (ts_t - ts_t_next).view(batch_size, 1, latent_num_frames, 1, 1) / num_train_t
            dt = dt.to(latent_t.dtype)
            latent_t_next = latent_t - dt * v_pred

        student_kwargs = dict(
            context=text_embeddings,
            block_mask=block_mask,
            teacher=latents if self.teacher_forcing else None,
        )
        v_student = student(
            latent_t.to(student.dtype),
            t=timesteps_t.to(student.dtype),
            **student_kwargs,
        )
        cm_pred_t = self._v_to_x0(latent_t.to(v_student.dtype), v_student, sigma_t.to(v_student.dtype))

        ema_kwargs = dict(
            context=text_embeddings,
            block_mask=block_mask,
            teacher=latents if self.teacher_forcing else None,
        )
        with torch.no_grad():
            v_ema = ema_student(
                latent_t_next.to(ema_student.dtype),
                t=timesteps_t_next.to(ema_student.dtype),
                **ema_kwargs,
            )
            cm_pred_t_next = self._v_to_x0(
                latent_t_next.to(v_ema.dtype), v_ema, sigma_t_next.to(v_ema.dtype)
            )

        cd_loss = F.mse_loss(cm_pred_t.float(), cm_pred_t_next.float(), reduction="mean")

        loss_dict = {
            "train/cd_loss": cd_loss.detach(),
            "train/loss": cd_loss.detach(),
            "train/sigma_t_mean": sigma_t.mean().detach(),
            "train/sigma_t_next_mean": sigma_t_next.mean().detach(),
        }
        return cd_loss, loss_dict

    def train_step(self, data):
        log_dict = {}
        self.set_requires_gradient_reduce(self.model)

        self.forward_timer.tic()
        loss, loss_log = self.consistency_distill(
            student=self.model,
            ema_student=self.ema,
            teacher=self.teacher,
            latents=data["latents"],
            text_embeddings=data["text_embeddings"],
            frame_per_block=self.frame_per_block,
        )
        self.forward_timer.toc()

        self.backward_timer.tic()
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        if self.accumulation_index == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            log_dict["train/grad_norm"] = grad_norm.to_local().item()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()

            update_ema_fsdp(unwrap_model(self.ema), unwrap_model(self.model), self.ema_decay)

        log_dict.update(loss_log)
        self.backward_timer.toc()

        reduce_dict(loss_log)
        self.loss_meter.update(loss_log)
        return log_dict

    def train(self):
        self.evaluate()
        self.logger.info("***** Running CD training *****")

        for epoch in range(self.num_train_epochs):
            self.loss_meter.reset()
            self.data_timer.tic()
            for step, tuple_batch in enumerate(self.train_dataloader, start=1):
                self.accumulation_index = step % self.gradient_accumulation_steps

                if self.dataset_type == 'ode_dataset':
                    ode = tuple_batch['ode'].to(
                        device=self.device, dtype=torch.float32, non_blocking=True,
                    )
                    latents = ode[:, -1]
                    text_embeddings = [
                        t.to(device=self.device, non_blocking=True)
                        for t in tuple_batch['t5_embed']
                    ]
                else:
                    pixel_values = tuple_batch["pixel_values"].to(
                        device=self.device, dtype=torch.float32, non_blocking=True,
                    )
                    text = tuple_batch["text"]
                    latents = vae_encode(self.vae, pixel_values=pixel_values)
                    text_embeddings, _ = self.text_encoder(
                        texts=text, device=self.device,
                    )
                data = dict(latents=latents, text_embeddings=text_embeddings)
                self.data_timer.toc()

                self.model.train()
                self.teacher.eval()
                self.ema.eval()

                log_dict = self.train_step(data=data)

                if self.accumulation_index == 0:
                    self.global_step += 1
                    self._log_progress(log_dict)

                    if (self.global_step % self.checkpointing_steps == 0
                            or self.global_step >= self.max_train_steps):
                        self.save_models()

                    if (self.global_step % self.validation_steps == 0
                            or self.global_step >= self.max_train_steps):
                        self.evaluate()

                self.data_timer.tic()
                if self.global_step >= self.max_train_steps:
                    self.logger.info("Finish CD training")
                    return