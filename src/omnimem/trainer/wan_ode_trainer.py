import os
import math
import json
import posixpath
from pathlib import Path
import wandb

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.optim.lr_scheduler import LRScheduler

from diffusers.optimization import get_scheduler

from omnimem.models.flex_attention_utils import create_causal_block_mask_cached
from omnimem.models.transformers.causal_wan_model import CausalWanModel, CausalWanAttentionBlock
from omnimem.pipelines import CausalWanT2VPipeline
from omnimem.schedulers import RectifiedFlowScheduler
from omnimem.utils.torch_utils import get_fsdp_state_dict, load_and_broadcast_diffuser, resume_model, save_sharded_safetensors
from omnimem.utils.ode_checkpoint_store import ODECheckpointStore, config_differences
from omnimem.utils.ode_training_state import (
    call_on_all_ranks, capture_rng_state, restore_rng_state, save_training_state, load_training_state,
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


class WanODETrainer(BaseWanTrainer):
    model: CausalWanModel

    optimizer: torch.optim.Optimizer
    lr_scheduler: LRScheduler

    forward_timer: TimerMeter
    backward_timer: TimerMeter

    checkpoint_modules = CausalWanAttentionBlock

    def __init__(self, config):
        self.config = config
        self.frame_per_block = self._get_and_record("frame_per_block", None)
        self.timestep_shift = self._get_and_record('timestep_shift')
        self.window_size = self._get_and_record("window_size", None)
        self.sink_size = self._get_and_record("sink_size", None)
        self.teacher_forcing = self._get_and_record("teacher_forcing")
        super().__init__(config)

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
        )
        self.logger.info(f"Load {model.__class__.__name__} from {self.model_path}")
        return model

    def model_compile(self, mode="max-autotune-no-cudagraphs"):
        self.model.compile(mode=mode)

    def optimizer_compile(self):
        self.optimizer.step = torch.compile(self.optimizer.step, fullgraph=False)

    def _check_model(self, model, tag):
        m = model.module if hasattr(model, 'module') else model
        for name, p in m.named_parameters():
            if torch.isnan(p).any():
                self.logger.warning(f"[{tag}] NaN in {name}!")
                return
        self.logger.info(f"[{tag}] {model.__class__.__name__} all clean")
        
    def setup_models(self):
        model = self.build_model()
        self._check_model(model, 'after build')
        self.load_model_checkpoints(model)
        self._check_model(model, 'after load ckpt')
        self.possible_apply_gradient_checkpointing(model)
        self.count_model_parameter(model, name="Model")
        self.model = model

    def _get_resume_dir(self):
        return self.config.get("checkpoint_dir") or os.path.join(self.output_dir, "checkpoints")

    def _checkpoint_rank0_call(self, operation):
        """Propagate coordinator I/O failures before peers enter a collective."""
        result = [None]
        if is_main_process():
            try:
                result[0] = {"value": operation(), "error": None}
            except Exception as exc:
                result[0] = {"value": None, "error": f"{type(exc).__name__}: {exc}"}
        if dist.is_initialized():
            dist.broadcast_object_list(result, src=0)
        if result[0]["error"] is not None:
            raise RuntimeError(f"ODE checkpoint operation failed: {result[0]['error']}")
        return result[0]["value"]

    def _checkpoint_all_ranks_call(self, operation):
        return call_on_all_ranks(operation, "complete the ODE checkpoint step")

    def resume_state(self):
        prefix = self.config.get("checkpoint_s3_prefix")
        client = self.s3.s3_client if self.upload else None
        if client is not None and prefix is None:
            if os.path.isabs(self.config.output_dir):
                raise ValueError("An absolute output_dir requires an explicit checkpoint_s3_prefix")
            prefix = posixpath.join(self.s3_dir, self.output_dir, "checkpoints")
        self._checkpoint_store = ODECheckpointStore(
            Path(self.resume_dir), s3_client=client,
            bucket=self.s3_bucket if client is not None else None,
            s3_prefix=prefix if client is not None else None,
        )
        self._resume_rng_state = None
        keep = self.config.get("checkpoint_keep_local")
        if keep is not None and client is not None:
            # Free space before the first save; the newest local copy is kept for resume.
            removed = self._checkpoint_rank0_call(
                lambda: [str(p) for p in self._checkpoint_store.prune_local(max(keep, 1))])
            self.logger.info(f"Pruned local checkpoints already on S3 before resume: {removed}")

        def select_checkpoint():
            checkpoint = self._checkpoint_store.latest()
            if checkpoint is None:
                if (Path(self.resume_dir) / "checkpoint.pth").exists() or (Path(self.resume_dir) / "model").exists():
                    raise RuntimeError(
                        "Legacy ODE checkpoint has no complete optimizer state. Use its model/ directory "
                        "as model_path or model_checkpoint_path with a new run name to initialize a new run."
                    )
                return None
            # latest() already verified all hashes; published directories are immutable.
            manifest = json.loads((checkpoint / "COMPLETED.json").read_text())
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            if manifest["world_size"] != world_size:
                raise ValueError("Full ODE resume requires the original number of training ranks")
            return {"path": str(checkpoint), "step": manifest["global_step"]}

        selected = self._checkpoint_rank0_call(select_checkpoint)
        self._resume_checkpoint = Path(selected["path"]) if selected is not None else None
        if selected is None:
            self.logger.info("No completed ODE checkpoint; starting a new run.")
            return 0
        self.logger.info(f"Resume complete ODE checkpoint {self._resume_checkpoint}")
        return selected["step"]

    def resume_from_checkpoint(self):
        if self._resume_checkpoint is None:
            return

        def restore_weights():
            model_dir = self._resume_checkpoint / "model"
            different = config_differences(json.loads((model_dir / "config.json").read_text()),
                                           json.loads(json.dumps(dict(self.model.config))))
            if different:
                raise ValueError(f"Model configuration differs from the resume checkpoint: {different}")
            resume_model(str(model_dir), self.model, strict=True)

        self._checkpoint_all_ranks_call(restore_weights)
        self._check_model(self.model, 'after resume ckpt')

    def wrap_model_fsdp(self):
        if dist.is_initialized():
            for _, p in self.model.state_dict().items():
                dist.broadcast(p.to(self.device), src=0)
        
            for block in self.model.blocks:
                fully_shard(block, **self.fsdp_kwargs)
            fully_shard(self.model, **self.fsdp_kwargs.root_kwargs)
        else:
            self.model.to(self.device)
        self._check_model(self.model, 'after fsdp wrap')

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

        if self._resume_checkpoint is not None:
            state = load_training_state(
                self._resume_checkpoint, self.model, self.optimizer, self.lr_scheduler,
                expected_step=self.global_step,
            )
            self._resume_rng_state = state["rng_state"]
            saved_schedule = state["extra_state"].get("scheduler_config")
            if saved_schedule != self._scheduler_config():
                raise ValueError("The LR schedule configuration differs from the completed checkpoint")
            self.logger.info("Restored optimizer and LR scheduler; ODE data stream will resample on resume.")

    def _scheduler_config(self):
        # LambdaLR restores counters, but its lambda is rebuilt from this configuration.
        from omegaconf import OmegaConf
        config = {
            "type": self.lr_scheduler_type,
            "warmup_steps": self.lr_warmup_steps,
            "kwargs": dict(self.lr_scheduler_kwargs),
        }
        if self.lr_scheduler_type not in ("constant", "constant_with_warmup"):
            config["max_train_steps"] = self.max_train_steps
        return OmegaConf.to_container(OmegaConf.create(config), resolve=True)


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
            time_log = (f"data:{sync_time[0].item():.2f} "
                        f"forward:{sync_time[1].item():.2f} "
                        f"backward:{sync_time[2].item():.2f}")
            self.logger.info(f"step: {self.global_step}/{max_train_steps}[{self.global_step / max_train_steps:.02%}]||"
                             f'time:[{time_log}]'
                             f" loss: {self.loss_meter.mavg.get('train/loss', 0):.04f}")

    def save_models(self):
        staging = Path(self._checkpoint_rank0_call(lambda: str(self._checkpoint_store.begin(self.global_step))))

        def require_shared_directory():
            if not staging.is_dir():
                raise FileNotFoundError(
                    f"Checkpoint directory {staging} is not visible on this rank. "
                    "Multi-node ODE training requires a shared checkpoint_dir."
                )

        self._checkpoint_all_ranks_call(require_shared_directory)
        model_state_dict = get_fsdp_state_dict(unwrap_model(self.model))

        def save_weights():
            save_dir = str(staging / "model")
            self.model.save_config(save_directory=save_dir)
            save_sharded_safetensors(state_dict=model_state_dict, save_dir=save_dir)

        self._checkpoint_rank0_call(save_weights)
        del model_state_dict
        save_training_state(
            staging, self.model, self.optimizer, self.lr_scheduler, self.global_step,
            extra_state={
                "scheduler_config": self._scheduler_config(),
            },
        )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        checkpoint = Path(self._checkpoint_rank0_call(
            lambda: str(self._checkpoint_store.commit(staging, self.global_step, world_size))
        ))
        self.logger.info(f"Saved complete ODE checkpoint: {checkpoint}")
        if self._checkpoint_store.s3 is not None:
            self._checkpoint_rank0_call(lambda: self._checkpoint_store.upload(checkpoint))
            keep = self.config.get("checkpoint_keep_local")
            if keep is not None:
                removed = self._checkpoint_rank0_call(lambda: [str(p) for p in self._checkpoint_store.prune_local(keep)])
                self.logger.info(f"Pruned local checkpoints already on S3 (keep {keep}): {removed}")

    def _evaluate_without_rng_change(self, initial=False):
        state = capture_rng_state()
        try:
            self.evaluate_init() if initial else self.evaluate()
        finally:
            restore_rng_state(state)

    def evaluate_init(self):
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
            commit=True,
            pipeline_cls=CausalWanT2VPipeline,
            frame_per_block=self.frame_per_block,
            sink_size=self.sink_size,
            window_size=self.window_size,
            s3_bucket=self.s3_bucket,
            s3_dir=self.s3_dir,
            upload=self.upload,
        )
        torch.cuda.empty_cache()

    def evaluate(self):
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
            sink_size=self.sink_size,
            window_size=self.window_size,
            s3_bucket=self.s3_bucket,
            s3_dir=self.s3_dir,
            upload=self.upload,
        )
        torch.cuda.empty_cache()

    def ode_regression(
            self,
            generator: CausalWanModel,
            ode,
            timesteps,
            text_embeddings,
            frame_per_block,
    ):

        batch_size, num_denoising_steps, num_channels, latent_num_frames, latent_height, latent_width = ode.shape
        if latent_num_frames // frame_per_block * frame_per_block != latent_num_frames:
            latent_num_frames = latent_num_frames // frame_per_block * frame_per_block
            ode = ode[:, :, :, :latent_num_frames]

        target = ode[:, -1]

        patch_size = 2
        frame_length = (latent_height // patch_size) * (latent_width // patch_size)
        sequence_length = frame_length * latent_num_frames
        # Sample noise that we'll add to the latents
        index = torch.randint(0, num_denoising_steps,
                              [batch_size, math.ceil(latent_num_frames / frame_per_block)],
                              device=self.device, dtype=torch.long)
        index = index.repeat_interleave(frame_per_block, 1)[:, :latent_num_frames]
        t = torch.gather(
            timesteps,
            dim=1,
            index=index,
        )
        timesteps = t.repeat_interleave(frame_length, dim=1)
        t = t.unsqueeze(dim=1)

        noisy_latent = torch.gather(
            ode,
            dim=1,
            index=(
                index
                .reshape(batch_size, 1, 1, latent_num_frames, 1, 1)
                .expand(-1, 1, num_channels, latent_num_frames, latent_height, latent_width)
            ),
        ).squeeze(1)
        kv_block_tokens = frame_per_block * frame_length
        block_mask = create_causal_block_mask_cached(
            block_size=kv_block_tokens,
            B=None,
            H=None,
            Q_LEN=sequence_length,
            KV_LEN=sequence_length,
            use_flex_attention=True,
            torch_compile=True,
            window_chunks=self.window_size.get('kv', None) if self.window_size is not None else None,
            sink_chunks=self.sink_size.get('kv', 0) if self.sink_size is not None else 0,
            teacher_forcing=self.teacher_forcing,
        )

        condition_dict = dict(
            t=timesteps,
            context=text_embeddings,
            block_mask=block_mask,
        )
        model_pred = generator(
            noisy_latent,
            **condition_dict,
        )

        proj_x0 = noisy_latent - model_pred * t.reshape((batch_size, 1, -1, 1, 1)) / 1000

        loss_dict = {}
        mask = t.reshape((batch_size, 1, -1, 1, 1)) != 0
        mask = mask.expand_as(proj_x0)
        loss = F.mse_loss(proj_x0[mask], target[mask].to(proj_x0), reduction='mean')

        loss_dict["train/loss"] = loss.detach()

        return loss, loss_dict

    def train_step(self, data):

        log_dict = {}
        self.set_requires_gradient_reduce(self.model)
        self.forward_timer.tic()
        loss, loss_log = self.ode_regression(
            generator=self.model,
            ode=data["ode"],
            timesteps=data["timesteps"],
            text_embeddings=data["text_embeddings"],
            frame_per_block=self.frame_per_block,
        )
        self.forward_timer.toc()

        self.backward_timer.tic()
        loss /= self.gradient_accumulation_steps

        loss.backward()

        if self.accumulation_index == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            log_dict["train/grad_norm"] = (
                grad_norm.to_local() if hasattr(grad_norm, "to_local") else grad_norm
            ).item()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()

        log_dict.update(loss_log)
        self.backward_timer.toc()

        # log loss
        reduce_dict(loss_log)
        self.loss_meter.update(loss_log)

        return log_dict

    def train(self):
        if self.global_step >= self.max_train_steps:
            self.logger.info(f"Training already complete at step {self.global_step}; nothing to run.")
            return
        # evaluate initial model
        self._evaluate_without_rng_change(initial=True)
        # main training loop
        self.logger.info("***** Running training *****")

        for epoch in range(self.num_train_epochs):
            self.loss_meter.reset()
            self.data_timer.tic()
            data_iterator = iter(self.train_dataloader)
            for step, tuple_batch in enumerate(data_iterator, start=1):
                # WebDataset starts its workers lazily on the first next(), not iter().
                if self._resume_rng_state is not None:
                    restore_rng_state(self._resume_rng_state)
                    self._resume_rng_state = None
                self.accumulation_index = step % self.gradient_accumulation_steps

                # load data
                data = dict(
                    ode=torch.as_tensor(tuple_batch['ode']).to(device=self.device, dtype=torch.float32, non_blocking=True),
                    timesteps=torch.as_tensor(tuple_batch['timesteps']).to(device=self.device, dtype=torch.float32, non_blocking=True),
                    text_embeddings=[
                        torch.as_tensor(embedding).to(device=self.device, dtype=self.dtype, non_blocking=True)
                        for embedding in tuple_batch['t5_embed']
                    ],
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
                        self._evaluate_without_rng_change()

                self.data_timer.tic()
                if self.global_step >= self.max_train_steps:
                    self.logger.info("Finish training")
                    return
