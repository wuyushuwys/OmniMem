import os
import math
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


class WanCausalTrainer(BaseWanTrainer):
    model: CausalWanModel
    ema: CausalWanModel

    optimizer: torch.optim.Optimizer
    lr_scheduler: LRScheduler

    forward_timer: TimerMeter
    backward_timer: TimerMeter

    checkpoint_modules = CausalWanAttentionBlock

    def __init__(self, config):
        self.config = config
        self.frame_per_block = self._get_and_record("frame_per_block", None)
        self.timestep_shift = self._get_and_record("timestep_shift", None)
        self.teacher_forcing = self._get_and_record("teacher_forcing", False)
        super().__init__(config)
        self.model_compile(mode="max-autotune-no-cudagraphs")

    def load_noise_scheduler(self):
        self.noise_scheduler = RectifiedFlowScheduler.from_config(
            self.scheduler_path,
            shift=self.timestep_shift,
        )
        self.logger.info(f"scheduler path {self.scheduler_path}")
        self.logger.info(f"{self.noise_scheduler.__class__.__name__} - {self.noise_scheduler.config.prediction_type}")

    def build_model(self):
        model: CausalWanModel = load_and_broadcast_diffuser(
            model_cls=CausalWanModel,
            model_name_or_path=self.model_path,
            device=self.device,
            low_cpu_mem_usage=False,
        )
        self.logger.info(f"Load {model.__class__.__name__} from {self.model_path}")

        return model

    def model_compile(self, mode="max-autotune-no-cudagraphs"):
        self.model.compile(mode=mode)

    def setup_models(self):
        model = self.build_model()
        self.load_model_checkpoints(model)
        self.possible_apply_gradient_checkpointing(model)
        self.count_model_parameter(model, name="Model")
        self.model = model

    def resume_from_checkpoint(self):
        checkpoint_path = os.path.join(self.output_dir, "checkpoints")
        resume_model(os.path.join(checkpoint_path, "model"), self.model)

    def wrap_model_fsdp(self):
        if dist.is_initialized():
            for block in self.model.blocks:
                fully_shard(block, **self.fsdp_kwargs)
            fully_shard(self.model, **self.fsdp_kwargs.root_kwargs)
        else:
            self.model.to(self.device)

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
            **self.lr_scheduler_kwargs
        )

        dcp_path = os.path.join(self.output_dir, "checkpoints", 'checkpoint_dcp')
        if os.path.exists(dcp_path):
            resume_dcp_checkpoint(model=self.model, optimizer=self.optimizer, path=dcp_path)

        for _ in range(self.global_step):
            self.lr_scheduler.step()

    def setup_timers(self):
        super().setup_timers()
        self.forward_timer = TimerMeter(
            max_length=self.log_steps * self.gradient_accumulation_steps,
            wait_for_all=False
        )
        self.backward_timer = TimerMeter(
            max_length=self.log_steps * self.gradient_accumulation_steps,
            wait_for_all=False
        )

    @property
    def sync_time(self):
        sync_time = torch.tensor(
            [self.data_timer.mavg, self.forward_timer.mavg, self.backward_timer.mavg]).to(self.device)
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
            time_log = (f"data:{sync_time[0].item():.2f} forward:{sync_time[1].item():.2f} backward:{sync_time[2].item():.2f}")
            self.logger.info(f"step: {self.global_step}/{max_train_steps}[{self.global_step / max_train_steps:.02%}]||"
                             f'time:[{time_log}]'
                             f" loss: {self.loss_meter.mavg.get('train/loss', 0):.04f}")

    def save_models(self):
        output_dir = self.output_dir
        model_state_dict = get_fsdp_state_dict(unwrap_model(self.model))
        if self.ema is not None:
            ema_state_dict = get_fsdp_state_dict(unwrap_model(self.ema))
        else:
            ema_state_dict = None
        save_dir = os.path.join(output_dir, "checkpoints", "checkpoint_dcp")
        save_dcp_checkpoint(self.model, self.optimizer, global_step=self.global_step, output_dir=save_dir)
 
        if is_main_process():
            self.logger.info(f"saving checkpoint to {output_dir} ...")
            save_dir = os.path.join(output_dir, "checkpoints", "model")
            self.model.save_config(save_directory=save_dir)
            save_sharded_safetensors(state_dict=model_state_dict, save_dir=save_dir)
            self.logger.info("Save model.")
            if self.ema is not None:
                save_dir = os.path.join(output_dir, "checkpoints", "ema")
                self.ema.save_config(save_directory=save_dir)
                save_sharded_safetensors(state_dict=ema_state_dict, save_dir=save_dir)
                self.logger.info("Save EMA model.")
 
            save_fsdp_checkpoint(None, None, self.global_step, output_dir=os.path.join(output_dir, "checkpoints"))
            self.logger.info("Save checkpoints.")

            self.s3.upload_folder(
                folder_path=os.path.join(output_dir, "checkpoints"),
                global_step=self.global_step,
            )

    def create_ema_model_peft(self, model, module_cls):
        if isinstance(model, PeftModel):
            ema_config = dict(model.get_base_model().config)
        else:
            ema_config = dict(model.config)
        ema_config["_class_name"] = module_cls.__name__
        ema = module_cls.from_config(ema_config)
        state_dict = get_fsdp_state_dict(model, master_only=False)

        self.logger.info(f"State dict keys (first 5): {list(state_dict.keys())[:5]}")
        self.logger.info(f"EMA model keys (first 5): {list(ema.state_dict().keys())[:5]}")
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
        if self.ema_model:
            self.ema, info = self.create_ema_model_peft(
                unwrap_model(self.model), module_cls=CausalWanModel
            )
            self.logger.info(
                f"EMA load info: missing={len(info.missing_keys)}, "
                f"unexpected={len(info.unexpected_keys)}"
            )
            self.logger.info(f"{info}")
        else:
            self.ema = None
            return
        if try_resume:
            resume_model(path=os.path.join(self.output_dir, "checkpoints", 'ema'), module=self.ema)
        if dist.is_initialized():
            for block in self.ema.blocks:
                fully_shard(block, **self.fsdp_kwargs)
            fully_shard(self.ema, **self.fsdp_kwargs.root_kwargs)
        else:
            self.ema.to(self.device)
        
            

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
                timestep_shift=self.timestep_shift,
                device=self.device,
                pipeline_cls=CausalWanT2VPipeline,
                frame_per_block=self.frame_per_block,
                s3_bucket=self.s3_bucket,
                s3_dir=self.s3_dir,
                upload=self.upload,
            )
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

    @staticmethod
    def logit_normal_inverse_cdf(u, mu=0.0, sigma=1.0):
        u = torch.clamp(u, min=1e-5, max=1.0 - 1e-5)
        z = mu + sigma * torch.sqrt(torch.tensor(2.0, device=u.device)) * torch.erfinv(2 * u - 1)
        return torch.sigmoid(z)


    def causal_training(
            self,
            model: CausalWanModel,
            latents,
            text_embeddings,
            frame_per_block,
            ema_teacher=None,
    ):
        patch_size = 2
        batch_size, _, latent_num_frames, latent_height, latent_width = latents.shape
        noise = torch.randn_like(latents)

        num_chunks = math.ceil(latent_num_frames / frame_per_block)
        u = torch.rand(batch_size, num_chunks, device=latents.device, dtype=torch.float32)
        sigmas = self.logit_normal_inverse_cdf(u)

        sigmas = sigmas.repeat_interleave(frame_per_block, dim=-1).narrow(1, 0, latent_num_frames)
        sigmas = sigmas.view(batch_size, 1, latent_num_frames, 1, 1)
        mask = torch.ones_like(sigmas, dtype=torch.bool)

        noisy_latent = sigmas * noise + (1 - sigmas) * latents
        target = noise - latents
        mask = mask.expand_as(target)

        frame_length = (latent_height // patch_size) * (latent_width // patch_size)

        t_s = self.noise_scheduler.config.num_train_timesteps * sigmas
        t_s = t_s.view(batch_size, latent_num_frames)
        timesteps_s = t_s.repeat_interleave(frame_length, dim=1)

        sequence_length = frame_length * latent_num_frames
        block_mask = create_causal_block_mask_cached(
            block_size=frame_per_block * frame_length,
            B=None,
            H=None,
            Q_LEN=sequence_length,
            KV_LEN=sequence_length,
            use_flex_attention=True,
            torch_compile=True,
            teacher_forcing=self.teacher_forcing,
        )

        loss_dict = {}
        loss = 0
        condition_dict = dict(
            t=timesteps_s.to(model.dtype),
            context=text_embeddings,
            block_mask=block_mask,
            teacher=latents if self.teacher_forcing else None
        )
        v_pred = model(
            noisy_latent.to(model.dtype),
            **condition_dict,
        )

        fm_loss = F.mse_loss(v_pred[mask].float(), target[mask].float(), reduction='mean')
        loss += fm_loss
        loss_dict["train/fm_loss"] = fm_loss.clone().detach()
        loss_dict["train/loss"] = loss.clone().detach()

        return loss, loss_dict

    def train_step(self, data):

        log_dict = {}
        self.set_requires_gradient_reduce(self.model)
        self.forward_timer.tic()
        loss, loss_log = self.causal_training(
            model=self.model,
            latents=data["latents"],
            text_embeddings=data["text_embeddings"],
            frame_per_block=self.frame_per_block,
            ema_teacher=self.ema,
        )
        self.forward_timer.toc()

        self.backward_timer.tic()
        loss /= self.gradient_accumulation_steps

        loss.backward()

        if self.accumulation_index == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            log_dict["train/grad_norm"] = grad_norm.to_local().item()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()

        log_dict.update(loss_log)
        if self.ema is not None and self.accumulation_index == 0:
            update_ema_fsdp(unwrap_model(self.ema), unwrap_model(self.model), self.ema_decay)
        self.backward_timer.toc()

        # log loss
        reduce_dict(loss_log)
        self.loss_meter.update(loss_log)

        return log_dict

    def train(self):
        # evaluate initial model
        self.evaluate()
        # main training loop
        self.logger.info("***** Running training *****")

        for epoch in range(self.num_train_epochs):
            self.loss_meter.reset()
            self.data_timer.tic()
            for step, tuple_batch in enumerate(self.train_dataloader, start=1):
                self.accumulation_index = step % self.gradient_accumulation_steps

                # load data
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
                    pixel_values = tuple_batch['pixel_values'].to(
                        device=self.device, dtype=torch.float32, non_blocking=True,
                    )
                    text = tuple_batch['text']
                    latents = vae_encode(self.vae, pixel_values=pixel_values)
                    text_embeddings, _ = self.text_encoder(
                        texts=text,
                        device=self.device,
                    )
                data = dict(
                    latents=latents,
                    text_embeddings=text_embeddings,
                )

                self.data_timer.toc()

                self.model.train()
                log_dict = self.train_step(
                    data=data,
                )

                if self.accumulation_index == 0:
                    self.global_step += 1

                    self._log_progress(log_dict)

                    if self.global_step % self.checkpointing_steps == 0 or self.global_step >= self.max_train_steps:
                        self.save_models()

                    if self.global_step % self.validation_steps == 0 or self.global_step >= self.max_train_steps:
                        self.evaluate()

                self.data_timer.tic()
                if self.global_step >= self.max_train_steps:
                    self.logger.info("Finish training")
                    return
