import os
import math

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.optim.lr_scheduler import LRScheduler

from peft import PeftModel

from omnimem.models.flex_attention_utils import create_causal_block_mask_cached
from omnimem.models.transformers.causal_wan_nsa_model import CausalWanNSAModel, CausalWanNSAttentionBlock
from omnimem.pipelines import CausalWanT2VPipeline
from omnimem.schedulers import RectifiedFlowScheduler
from omnimem.utils.torch_utils import (
    get_fsdp_state_dict, 
    load_and_broadcast_diffuser, 
    resume_model, 
    update_ema_fsdp,
    )
from omnimem.utils.meter import TimerMeter
from omnimem.evaluate import evaluation_wan
from omnimem.utils.misc import (
    reduce_dict,
    unwrap_model,
)

from omnimem.trainer.wan_causal_trainer import WanCausalTrainer


class WanCausalNSATrainer(WanCausalTrainer):
    model: CausalWanNSAModel
    ema: CausalWanNSAModel

    optimizer: torch.optim.Optimizer
    lr_scheduler: LRScheduler

    forward_timer: TimerMeter
    backward_timer: TimerMeter

    checkpoint_modules = CausalWanNSAttentionBlock

    def __init__(self, config):
        self.config = config
        self.window_size = self._get_and_record("window_size", None)
        self.sink_size = self._get_and_record("sink_size", None)
        self.nsa_kwargs = self._get_and_record("nsa_kwargs")
        super().__init__(config)
        self._evaluate_kwargs = dict(
            pipeline_cls=CausalWanT2VPipeline,
            frame_per_block=self.frame_per_block,
            sink_size=self.sink_size,
            window_size=self.window_size,
        )

    def load_noise_scheduler(self):
        self.noise_scheduler = RectifiedFlowScheduler.from_config(
            self.scheduler_path,
            shift=self.timestep_shift,
        )
        self.logger.info(f"scheduler path {self.scheduler_path}")
        self.logger.info(f"{self.noise_scheduler.__class__.__name__} - {self.noise_scheduler.config.prediction_type}")

    def build_model(self):
        model: CausalWanNSAModel = load_and_broadcast_diffuser(
            model_cls=CausalWanNSAModel,
            model_name_or_path=self.config.model_path,
            device=self.device,
            low_cpu_mem_usage=False,
            **self.nsa_kwargs
        )
        model.init_gate()
        if dist.is_initialized():
            for name, p in model.named_parameters():
                if 'g_proj' in name:
                    dist.broadcast(p.data.to(self.device), src=0)

        return model

    def create_ema_model_peft(self, model, module_cls):
        ema_config = dict(model.config)
        ema_config["_class_name"] = module_cls.__name__
        ema = module_cls.from_config(ema_config)
        state_dict = get_fsdp_state_dict(model, master_only=False)

        self.logger.info(f"State dict keys (first 5): {list(state_dict.keys())[:5]}")
        self.logger.info(f"EMA model keys (first 5): {list(ema.state_dict().keys())[:5]}")

        info = ema.load_state_dict(state_dict, strict=False)
        return ema, info

    def setup_ema_model(self, try_resume=False):
        if self.ema_model:
            ema_module_cls = CausalWanNSAModel
            self.ema, info = self.create_ema_model_peft(
                unwrap_model(self.model), module_cls=ema_module_cls
            )
            self.logger.info(
                f"EMA load info: missing={len(info.missing_keys)}, "
                f"unexpected={len(info.unexpected_keys)}"
            )
            self.logger.info(f"{info}")
            if self.lora_weights is not None:
                self.ema = PeftModel.from_pretrained(self.ema, self.lora_weights)
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
                s3_bucket=self.s3_bucket,
                s3_dir=self.s3_dir,
                upload=self.upload,
                **self._evaluate_kwargs,
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
            s3_bucket=self.s3_bucket,
            s3_dir=self.s3_dir,
            upload=self.upload,
            **self._evaluate_kwargs,
        )
        torch.cuda.empty_cache()

    @staticmethod
    def logit_normal_inverse_cdf(u, mu=0.0, sigma=1.0):
        u = torch.clamp(u, min=1e-5, max=1.0 - 1e-5)
        z = mu + sigma * torch.sqrt(torch.tensor(2.0, device=u.device)) * torch.erfinv(2 * u - 1)
        return torch.sigmoid(z)

    def causal_training(
            self,
            model: CausalWanNSAModel,
            latents,
            text_embeddings,
            frame_per_block,
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
            window_chunks=self.window_size.get('kv', None) if self.window_size is not None else None,
            sink_chunks=(self.sink_size.get('kv') if self.sink_size else None) or 0,
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

