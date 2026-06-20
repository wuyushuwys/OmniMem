import os
import wandb
from typing import Union

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.optim.lr_scheduler import LRScheduler

import safetensors.torch
from diffusers.optimization import get_scheduler

from omnimem.models.transformers.wan_model import WanModel, WanAttentionBlock
from omnimem.schedulers import RectifiedFlowScheduler
from omnimem.utils.torch_utils import (
    get_fsdp_state_dict,
    update_ema_fsdp,
    save_fsdp_checkpoint,
    save_dcp_checkpoint,
    resume_dcp_checkpoint,
    create_ema_model,
    resume_model,
    load_and_broadcast_diffuser,
    build_sharded_model,
)
from omnimem.utils.train_utils import pred_x0
from omnimem.utils.misc import (
    is_main_process,
    wait_for_everyone,
    unwrap_model,
    reduce_dict
)
from omnimem.utils.meter import TimerMeter
from omnimem.evaluate import evaluation_wan
from omnimem.trainer.base_wan_trainer import BaseWanTrainer


class WanDMDTrainer(BaseWanTrainer):
    generator: WanModel
    real_net: WanModel
    fake_net: WanModel

    generator_optimizer: torch.optim.Optimizer
    generator_lr_scheduler: LRScheduler

    critic_optimizer: torch.optim.Optimizer
    critic_lr_scheduler: LRScheduler

    ema: Union[None, WanModel] = None

    generator_timer: TimerMeter
    critic_timer: TimerMeter

    checkpoint_modules = WanAttentionBlock

    def __init__(self, config):
        self.config = config
        self.real_model_path = self.config.real_model_path
        self.fake_model_path = self.config.fake_model_path
        self.original_step = config.get("original_step")
        self.original_cfg = config.get("original_cfg")
        self.timestep_shift = config.timestep_shift
        self.latents_shape = config.get("latents_shape", [16, 21, 60, 104])
        self.max_distillation_steps = config.get("max_distillation_steps", 4)
        self.timestep_sample_method = config.get("timestep_sample_method", 'uniform')
        self.real_guidance_scale = config.real_guidance_scale
        self.dfake_gen_update_ratio = config.dfake_gen_update_ratio
        self.critic_loss_type = config.critic_loss_type
        self.critic_learning_rate = config.critic_learning_rate
        self.critic_adam_beta1 = config.get("critic_adam_beta1", 0.9)
        self.critic_adam_beta2 = config.get("critic_adam_beta2", 0.999)
        self.critic_adam_weight_decay = config.get("critic_adam_weight_decay", 1e-02)
        self.critic_adam_epsilon = config.get("critic_adam_epsilon", 1e-8)

        super().__init__(config)
        self.model_compile(mode="max-autotune-no-cudagraphs")

    def load_noise_scheduler(self):
        self.noise_scheduler = RectifiedFlowScheduler.from_config(
            self.scheduler_path,
            sample_method=self.timestep_sample_method,
            shift=self.timestep_shift,
        )
        self.logger.info(f"Scheduler path {self.scheduler_path}")
        self.logger.info(f"{self.noise_scheduler.__class__.__name__} - {self.noise_scheduler.config.prediction_type}")

    def build_model(self):
        generator: WanModel = load_and_broadcast_diffuser(
            model_cls=WanModel,
            model_name_or_path=self.model_path,
            device=self.device,
        )
        self.logger.info(f"Load {generator.__class__.__name__} from {self.model_path}")
        return generator

    def model_compile(self, mode="max-autotune-no-cudagraphs"):
        self.generator.compile(mode=mode, dynamic=False)
        self.real_net.compile(mode=mode, dynamic=False)
        self.fake_net.compile(mode=mode, dynamic=False)

    def optimizer_compile(self):
        self.generator_optimizer.step = torch.compile(self.generator_optimizer.step, fullgraph=False)
        self.critic_optimizer.step = torch.compile(self.critic_optimizer.step, fullgraph=False)

    def setup_models(self):
        generator = self.build_model()
        self.load_model_checkpoints(generator)

        self.possible_apply_gradient_checkpointing(generator)

        self.count_model_parameter(generator, name="Generator")

        if dist.is_initialized():
            real_net = build_sharded_model(
                module_cls=WanModel,
                model_dir=self.real_model_path,
                fsdp_kwargs=self.fsdp_kwargs,
                module='blocks',
            )
        else:
            real_net = WanModel.from_pretrained(self.real_model_path)
            real_net.to(self.device)

        self.logger.info(f"load real net model from {self.real_model_path}")
        real_net.requires_grad_(False)
        self.count_model_parameter(real_net, name="RealNet")

        fake_net = load_and_broadcast_diffuser(
            WanModel,
            self.fake_model_path,
            device=self.device
        )
        self.logger.info(f"load fake net model from {self.fake_model_path}")
        fake_net.requires_grad_(True)
        self.possible_apply_gradient_checkpointing(fake_net)
        self.count_model_parameter(fake_net, name="FakeNet")

        self.generator = generator
        self.fake_net = fake_net
        self.real_net = real_net

    def resume_from_checkpoint(self):
        checkpoint_path = os.path.join(self.output_dir, "checkpoints")
        resume_model(os.path.join(checkpoint_path, "generator"), self.generator)
        resume_model(os.path.join(checkpoint_path, "fake_net"), self.fake_net)

    def wrap_model_fsdp(self):
        if dist.is_initialized():
            for block in self.generator.blocks:
                fully_shard(block, **self.fsdp_kwargs)
            fully_shard(self.generator, **self.fsdp_kwargs.root_kwargs)

            for block in self.fake_net.blocks:
                fully_shard(block, **self.fsdp_kwargs)
            fully_shard(self.fake_net, **self.fsdp_kwargs.root_kwargs)
        else:
            self.generator.to(self.device)
            self.fake_net.to(self.device)

    def setup_optimizers_and_lr_scheduler(self):
        # Separate lr group for gate params when present.
        gate_params = []
        other_params = []
        for name, p in self.generator.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ('g_proj', 'path_scale', 'path_bias', 'noise_proj')):
                gate_params.append(p)
            else:
                other_params.append(p)

        param_groups = [
            {'params': other_params, 'lr': self.learning_rate},
        ]
        if gate_params:
            gate_lr = getattr(self, 'nsa_kwargs', {}).get('gate_lr', self.learning_rate)
            param_groups.append({'params': gate_params, 'lr': gate_lr})
            print(f"[Optimizer] gate params: {len(gate_params)}, lr: {gate_lr:.2e}")
            print(f"[Optimizer] other params: {len(other_params)}, lr: {self.learning_rate:.2e}")

        self.generator_optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
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

        for _ in range(self.global_step):
            self.generator_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        checkpoint_path = os.path.join(self.output_dir, "checkpoints")
        resume_dcp_checkpoint(
            self.generator, self.generator_optimizer,
            os.path.join(checkpoint_path, "checkpoint_dcp_generator"),
        )
        resume_dcp_checkpoint(
            self.fake_net, self.critic_optimizer,
            os.path.join(checkpoint_path, "checkpoint_dcp_critic"),
        )

    def setup_ema_model(self, try_resume=False):
        if self.ema_model and self.ema_start_step <= self.global_step and self.ema is None:
            self.ema, info = create_ema_model(unwrap_model(self.generator), module_cls=WanModel)
            if try_resume:
                resume_model(path=os.path.join(self.output_dir, "checkpoints", 'ema'), module=self.ema)
            if dist.is_initialized():
                for block in self.ema.blocks:
                    fully_shard(block, **self.fsdp_kwargs)
                fully_shard(self.ema, **self.fsdp_kwargs.root_kwargs)
            else:
                self.ema.to(self.device)

    def setup_timers(self):
        super().setup_timers()
        self.generator_timer = TimerMeter(
            max_length=self.log_steps * self.gradient_accumulation_steps // self.dfake_gen_update_ratio,
            wait_for_all=False
        )
        self.critic_timer = TimerMeter(
            max_length=self.log_steps * self.gradient_accumulation_steps,
            wait_for_all=False
        )

    @property
    def sync_time(self):
        sync_time = torch.tensor(
            [self.data_timer.mavg, self.generator_timer.mavg, self.critic_timer.mavg]).to(self.device)
        if dist.is_initialized():
            dist.all_reduce(sync_time, dist.ReduceOp.AVG)
        return sync_time

    def _log_progress(self, log_dict):
        sync_time = self.sync_time
        wait_for_everyone()
        if wandb.run is not None:
            log_dict.update({
                "global_step": self.global_step,
                "learning_rate/g": self.generator_lr_scheduler.get_last_lr()[0],
                "learning_rate/c": self.critic_lr_scheduler.get_last_lr()[0],
                "time/data": sync_time[0].item(),
                "time/g": sync_time[1].item(),
                "time/c": sync_time[2].item(),
            })
            log_dict.update(self.loss_meter.val)
            wandb.log(log_dict, step=int(self.global_step))

        max_train_steps = self.max_train_steps
        if self.global_step % self.log_steps == 0 or self.global_step >= max_train_steps:
            time_log = (f"data:{sync_time[0].item():.2f} "
                        f"g:{sync_time[1].item():.2f} "
                        f"c:{sync_time[2].item():.2f}")
            self.logger.info(f"step: {self.global_step}/{max_train_steps}[{self.global_step / max_train_steps:.02%}]||"
                             f'time:[{time_log}]'
                             f" g_total_loss: {self.loss_meter.mavg.get('train/g_total_loss', 0):.04f}"
                             f" c_total_loss: {self.loss_meter.mavg.get('train/critic_total_loss', 0):.04f}")

    def save_models(self):
        output_dir = self.output_dir
        generator_state_dict = get_fsdp_state_dict(unwrap_model(self.generator))
        fake_net_state_dict = get_fsdp_state_dict(unwrap_model(self.fake_net))
        if self.ema is not None:
            ema_state_dict = get_fsdp_state_dict(unwrap_model(self.ema))
        else:
            ema_state_dict = None

        if is_main_process():
            self.logger.info(f"saving checkpoint to {output_dir} ...")

            save_dir = os.path.join(output_dir, "checkpoints", "generator")
            self.generator.save_config(save_directory=save_dir)
            weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
            safetensors.torch.save_file(generator_state_dict, weight_path)
            self.logger.info("Save generator.")

            save_dir = os.path.join(output_dir, "checkpoints", "fake_net")
            self.fake_net.save_config(save_directory=save_dir)
            weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
            safetensors.torch.save_file(fake_net_state_dict, weight_path)
            self.logger.info("Save fake-net.")

            save_fsdp_checkpoint(None, None, self.global_step, output_dir=os.path.join(output_dir, "checkpoints"))
            self.logger.info("Save checkpoints.")

            # save EMA model
            if self.ema is not None:
                save_dir = os.path.join(output_dir, "checkpoints", "ema")
                self.generator.save_config(save_directory=save_dir)
                weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
                safetensors.torch.save_file(ema_state_dict, weight_path)
                self.logger.info("Save ema model.")

            self.s3.upload_folder(
                folder_path=os.path.join(output_dir, "checkpoints"),
                global_step=self.global_step,
            )

        save_dcp_checkpoint(
            self.generator, self.generator_optimizer, self.global_step,
            output_dir=os.path.join(output_dir, "checkpoints", "checkpoint_dcp_generator"),
        )
        save_dcp_checkpoint(
            self.fake_net, self.critic_optimizer, self.global_step,
            output_dir=os.path.join(output_dir, "checkpoints", "checkpoint_dcp_critic"),
        )

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
                timestep_shift=self.timestep_shift,
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
            timestep_shift=self.timestep_shift,
            device=self.device,
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
                n_rows=self.validation_data.get("n_rows", 4),
                tag='ema/',
                timestep_shift=self.timestep_shift,
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
            timestep_shift=self.timestep_shift,
            device=self.device,
            s3_bucket=self.s3_bucket,
            s3_dir=self.s3_dir,
            upload=self.upload,
        )
        torch.cuda.empty_cache()

    @staticmethod
    def backward_simulation(
            transformer: WanModel,
            noise_scheduler: RectifiedFlowScheduler,
            num_inference_steps,
            latents: torch.Tensor,
            condition_dict,
    ):
        noisy_latent = torch.randn_like(latents)

        noise_scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=noisy_latent.device,
            noise=noisy_latent,
        )

        inference_timesteps = noise_scheduler.timesteps

        selected_step = torch.randint(0, inference_timesteps.shape[0], size=(1,),
                                      device=noisy_latent.device,
                                      dtype=torch.long)
        latent_samples = [noisy_latent]

        for index, t in enumerate(inference_timesteps, start=1):
            latent_model_input = noisy_latent
            timestep = t.expand(noisy_latent.shape[0])
            with torch.no_grad():
                noise_pred = transformer(
                    x=noisy_latent,
                    t=timestep,
                    **condition_dict,
                )
                denoised_pred = pred_x0(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=noisy_latent,
                    noise_scheduler=noise_scheduler
                )
                latent_samples.append(denoised_pred)
                if index < len(inference_timesteps):
                    next_timestep = inference_timesteps[index].expand(denoised_pred.shape[0])
                    noisy_latent = noise_scheduler.add_noise(
                        denoised_pred,
                        noise=torch.randn_like(denoised_pred),
                        timesteps=next_timestep
                    )
        generated_image = latent_samples[selected_step]
        selected_timestep = inference_timesteps[selected_step].expand(generated_image.shape[0])
        noisy_latent = noise_scheduler.add_noise(
            generated_image,
            noise=torch.randn_like(generated_image),
            timesteps=selected_timestep
        )
        noise_pred = transformer(
            x=noisy_latent,
            t=selected_timestep,
            **condition_dict
        )

        denoised_pred = pred_x0(
            model_output=noise_pred,
            timestep=selected_timestep,
            sample=noisy_latent,
            noise_scheduler=noise_scheduler
        )

        return denoised_pred, selected_step

    @staticmethod
    def compute_distribution_matching_loss(
            real_net,
            fake_net,
            noise_scheduler,
            latents,
            timesteps,
            condition_dict,
            uncondition_dict,
            real_guidance_scale,
    ):
        original_latents = latents

        noise = torch.randn_like(latents)

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        x0_proj_kwargs = dict(
            timestep=timesteps,
            sample=noisy_latents,
            noise_scheduler=noise_scheduler
        )
        with torch.no_grad():
            pred_fake_noise = fake_net(
                noisy_latents,
                t=timesteps,
                **condition_dict,
            )
            pred_fake_image = pred_x0(pred_fake_noise, **x0_proj_kwargs)

            pred_real_noise_cond = real_net(
                noisy_latents,
                t=timesteps,
                **condition_dict,
            )
            pred_real_image_cond = pred_x0(pred_real_noise_cond, **x0_proj_kwargs)

            pred_real_noise_uncond = real_net(
                noisy_latents,
                t=timesteps,
                **uncondition_dict,
            )
            pred_real_image_uncond = pred_x0(pred_real_noise_uncond, **x0_proj_kwargs)
            pred_real_noise = pred_real_noise_cond + (
                    pred_real_noise_cond - pred_real_noise_uncond) * real_guidance_scale
            pred_real_image = pred_real_image_cond + (
                    pred_real_image_cond - pred_real_image_uncond) * real_guidance_scale

            grad = (pred_fake_image - pred_real_image)
            p_real = (original_latents - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
            grad = torch.nan_to_num(grad)

        return grad, pred_real_noise

    def generator_step(
            self,
            generator,
            real_net,
            fake_net,
            noise_scheduler: RectifiedFlowScheduler,
            latents,
            condition_dict,
            uncondition_dict,
            real_guidance_scale,
    ):
        loss_dict = {}

        batch_size, _, latent_num_frames, latent_height, latent_width = latents.shape

        dmd_timestep = noise_scheduler.sample_timesteps(
            batch_size,
            device=generator.device
        )
        dmd_timestep = noise_scheduler.scale_timesteps(dmd_timestep, noise=latents)
        dmd_timestep.clamp_(noise_scheduler.num_train_timesteps * 0.02, noise_scheduler.num_train_timesteps * 0.98)

        dmd_grad, pred_real_noise = self.compute_distribution_matching_loss(
            real_net=real_net,
            fake_net=fake_net,
            noise_scheduler=noise_scheduler,
            latents=latents,
            timesteps=dmd_timestep,
            condition_dict=condition_dict,
            uncondition_dict=uncondition_dict,
            real_guidance_scale=real_guidance_scale,
        )
        dmd_loss = 0.5 * F.mse_loss(
            latents.float(),
            (latents.float() - dmd_grad.float()).detach(),
            reduction="mean"
        )
        g_total_loss = dmd_loss
        loss_dict["g_loss/g_dmd_loss"] = dmd_loss.detach().clone()

        loss_dict['train/g_total_loss'] = g_total_loss.detach().clone()
        return g_total_loss, loss_dict

    @staticmethod
    def critic_step(
            fake_net,
            noise_scheduler: RectifiedFlowScheduler,
            latents,
            condition_dict,
            critic_loss_type='flow',
    ):
        loss_dict = {}
        batch_size, _, latent_num_frames, latent_height, latent_width = latents.shape

        critic_noise_patch = torch.randn_like(latents)

        critic_timesteps = noise_scheduler.sample_timesteps(
            batch_size,
            device=fake_net.device
        )
        critic_timesteps = noise_scheduler.scale_timesteps(critic_timesteps, latents)
        critic_timesteps.clamp_(
            noise_scheduler.num_train_timesteps * 0.02,
            noise_scheduler.num_train_timesteps * 0.98
        )

        generate_noisy_latents = noise_scheduler.add_noise(latents, critic_noise_patch, critic_timesteps)

        target = noise_scheduler.get_velocity(latents, critic_noise_patch, critic_timesteps)
        fake_noise_pred = fake_net(
            x=generate_noisy_latents,
            t=critic_timesteps,
            **condition_dict
        )
        fake_score_x0 = pred_x0(fake_noise_pred, critic_timesteps, generate_noisy_latents, noise_scheduler)

        if critic_loss_type == 'flow':
            sigma_t = critic_timesteps.view(batch_size, 1, 1, 1, 1) / noise_scheduler.config.num_train_timesteps
            flow_pred = (generate_noisy_latents - fake_score_x0) / sigma_t
            critic_loss = F.mse_loss(flow_pred.float(), target.float())
        elif critic_loss_type == 'x0':
            critic_loss = F.mse_loss(fake_score_x0.float(), latents.float())
        elif critic_loss_type == 'v':
            critic_loss = F.mse_loss(fake_noise_pred.float(), target.float())
        else:
            raise ValueError(f"Unknown critic loss type: {critic_loss_type}")

        loss_dict['c_loss/c_flow_loss'] = critic_loss.detach().clone()

        critic_total_loss = critic_loss

        loss_dict['train/critic_total_loss'] = critic_total_loss.detach().clone()

        return critic_total_loss, loss_dict

    def train_step(self, latents, condition_dict, uncondition_dict):

        log_dict = {}
        loss_log = {}

        # train generator
        if self.global_step % self.dfake_gen_update_ratio == 0:
            self.set_requires_gradient_reduce(self.generator)
            self.generator_timer.tic()
            generated_latents, sample_timestep = self.backward_simulation(
                transformer=self.generator,
                noise_scheduler=self.noise_scheduler,
                num_inference_steps=self.max_distillation_steps,
                latents=latents,
                condition_dict=condition_dict,
            )
            g_total_loss, g_loss_logs = self.generator_step(
                generator=self.generator,
                real_net=self.real_net,
                fake_net=self.fake_net,
                noise_scheduler=self.noise_scheduler,
                latents=generated_latents,
                condition_dict=condition_dict,
                uncondition_dict=uncondition_dict,
                real_guidance_scale=self.real_guidance_scale,
            )
            g_total_loss /= self.gradient_accumulation_steps

            g_total_loss.backward()


            if self.accumulation_index == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)
                log_dict["train/g_grad_norm"] = grad_norm.to_local().item()
                self.generator_optimizer.step()
                self.generator_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                self.generator_lr_scheduler.step()

            loss_log.update(g_loss_logs)
            if self.ema is not None and self.ema_start_step <= self.global_step and self.accumulation_index == 0:
                update_ema_fsdp(self.ema, unwrap_model(self.generator), self.ema_decay)
            self.generator_timer.toc()

        # train critic
        self.set_requires_gradient_reduce(self.fake_net)
        self.critic_timer.tic()
        with torch.no_grad():
            generated_latents, sample_timestep = self.backward_simulation(
                transformer=self.generator,
                noise_scheduler=self.noise_scheduler,
                num_inference_steps=self.max_distillation_steps,
                latents=latents,
                condition_dict=condition_dict,
            )
        critic_loss, critic_loss_logs = self.critic_step(
            fake_net=self.fake_net,
            noise_scheduler=self.noise_scheduler,
            latents=generated_latents.detach(),
            condition_dict=condition_dict,
            critic_loss_type=self.critic_loss_type,
        )
        critic_loss /= self.gradient_accumulation_steps
        critic_loss.backward()

        if self.accumulation_index == 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.fake_net.parameters(), self.max_grad_norm)
            log_dict["train/critic_grad_norm"] = critic_grad_norm.to_local().item()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.critic_lr_scheduler.step()

        loss_log.update(critic_loss_logs)
        self.critic_timer.toc()

        # log loss
        reduce_dict(loss_log)
        self.loss_meter.update(loss_log)

        return log_dict

    def train(self):
        # evaluate initial model
        self.evaluate_init()
        # main training loop
        self.logger.info("***** Running training *****")

        for epoch in range(self.num_train_epochs):
            self.loss_meter.reset()
            self.data_timer.tic()
            for step, tuple_batch in enumerate(self.train_dataloader, start=1):
                self.accumulation_index = step % self.gradient_accumulation_steps

                self.setup_ema_model()

                # load data
                text = tuple_batch["prompts"]
                text_embeddings, seq_lens = self.text_encoder(
                    texts=text,
                    device=self.device
                )
                null_text_embeddings, null_seq_lens = self.text_encoder(
                    texts=[self.null_prompt] * len(text),
                    device=self.device
                )
                latents = torch.randn(
                    [len(text_embeddings), *self.latents_shape],
                    device=self.device,
                    dtype=self.dtype
                )
                _, latent_num_frames, latent_height, latent_width = self.latents_shape
                patch_size = self.generator.patch_size[1]
                sequence_length = latent_num_frames * latent_height * latent_width // (patch_size * patch_size)
                condition_dict = dict(
                    context=text_embeddings,
                    seq_len=sequence_length,
                )
                uncondition_dict = dict(
                    context=null_text_embeddings,
                    seq_len=sequence_length,
                )

                self.data_timer.toc()

                self.generator.train()
                self.fake_net.train()

                log_dict = self.train_step(
                    latents=latents,
                    condition_dict=condition_dict,
                    uncondition_dict=uncondition_dict,
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
