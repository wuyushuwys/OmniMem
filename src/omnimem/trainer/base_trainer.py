import os
import datetime
import wandb
from abc import ABC, abstractmethod
from omegaconf import OmegaConf, ListConfig

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
from torch.utils.data import DataLoader

import diffusers
import transformers
import safetensors.torch

from omnimem.utils.torch_utils import prepare_fsdp_kwargs, resume_step
from omnimem.utils.train_utils import prepare_dataloader
from omnimem.utils.checkpoint_wrapper import checkpoint_wrapper
from omnimem.utils.meter import LossesMeter, TimerMeter
from omnimem.utils.aws_handler import S3
from omnimem.utils.logging_tool import get_logger
from omnimem.utils.misc import set_seed, is_main_process, format_numel_str, get_model_numel


class BaseTrainer(ABC):
    output_dir: str
    global_seed: int
    train_dataloader: DataLoader
    data_timer: TimerMeter
    accumulation_index: int
    gradient_accumulation_steps: int = 1
    s3: S3

    def __init__(self, config):
        self.config = config
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")

        # Environment & Logging
        self.dataset_type = self._get_and_record("dataset_type")
        self.train_data = self._get_and_record("train_data")
        self.conditioning_dropout_prob = self._get_and_record("conditioning_dropout_prob", default=0)
        self.num_workers = self._get_and_record("num_workers", 8)
        self.pin_memory = self._get_and_record("pin_memory", True)
        self.validation_data = self._get_and_record("validation_data")
        self.null_prompt = self._get_and_record("null_prompt", "")
        # load model
        self.model_path = self._get_and_record("model_path")
        self.model_checkpoint_path = self._get_and_record("model_checkpoint_path", None)
        # model training config
        self.gradient_accumulation_steps = self._get_and_record("gradient_accumulation_steps", 1)
        self.max_grad_norm = self._get_and_record('max_grad_norm', 1)
        self.torch_compile = self._get_and_record("torch_compile")
        self.gradient_checkpointing = self._get_and_record('gradient_checkpointing', True)
        self.learning_rate = self._get_and_record("learning_rate")
        self.adam_beta1 = self._get_and_record("adam_beta1", 0.9)
        self.adam_beta2 = self._get_and_record("adam_beta2", 0.999)
        self.adam_weight_decay = self._get_and_record("adam_weight_decay", 1e-02)
        self.adam_epsilon = self._get_and_record("adam_epsilon", 1e-08)
        self.num_train_epochs = self._get_and_record("num_train_epochs", 100)
        # intervals
        self.max_train_steps = self._get_and_record("max_train_steps")
        self.checkpointing_steps = self._get_and_record("checkpointing_steps")
        self.log_steps = self._get_and_record("log_steps")
        self.validation_steps = self._get_and_record("validation_steps")
        # LR scheduler
        self.lr_warmup_steps = self._get_and_record("lr_warmup_steps", 0)
        self.lr_scheduler_type = self._get_and_record("lr_scheduler", "constant_with_warmup")
        self.lr_scheduler_kwargs = self._get_and_record("lr_scheduler_kwargs", dict())
        # EMA
        self.ema_model = self._get_and_record("ema_model", False)
        self.ema_decay = self._get_and_record("ema_decay", 0.99)
        self.ema_start_step = self._get_and_record("ema_start_step", 0)
        # s3 config
        self.s3_bucket = self._get_and_record("s3_bucket")
        self.s3_dir = self._get_and_record("s3_dir")
        self.upload = self._get_and_record('upload', True)

        self.setup_environment()
        self.logger = get_logger(file_path=self.output_dir)

        # Default resume dir; subclasses may override _get_resume_dir().
        self.resume_dir = self._get_resume_dir()

        self.dtype = torch.bfloat16 if self.config.get("dtype", "bf16") == 'bf16' else torch.float32
        self.fsdp_kwargs = prepare_fsdp_kwargs(
            sharding_strategy=self.config.sharding_strategy,
            dtype=self.dtype
        ) if dist.is_initialized() else None
        self.global_step = self.resume_state()
        self.setup_frozen_components()
        self.setup_dataloader()
        self.setup_models()
        self.resume_from_checkpoint()
        self.wrap_model_fsdp()
        self.setup_optimizers_and_lr_scheduler()
        self.setup_ema_model(try_resume=True)

        self.loss_meter = LossesMeter(max_length=self.log_steps)
        self.setup_timers()

        self.logger.info(f"[world_size] {torch.distributed.get_world_size()}")

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.max_train_steps}")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def _get_and_record(self, key, default=None):
        value = self.config.get(key, default)
        self.config[key] = value
        return value

    def setup_environment(self):
        """Initialize distributed training, seeds, WandB, and output directories."""
        diffusers.utils.logging.disable_progress_bar()
        transformers.utils.logging.disable_progress_bar()
        torch.backends.cuda.matmul.allow_tf32 = True

        torch.cuda.set_device(self.device)
        if not dist.is_initialized() and int(os.getenv('WORLD_SIZE', 1)) > 1:
            dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))

        self.global_seed = self.config.global_seed
        seed = self.global_seed + dist.get_rank() if dist.is_initialized() else self.global_seed
        set_seed(seed)

        self.output_dir = os.path.join(self.config.output_dir, self.config.name)
        if is_main_process():
            os.makedirs(f"{self.output_dir}/samples", exist_ok=True)
            os.makedirs(f"{self.output_dir}/checkpoints", exist_ok=True)
            os.makedirs(os.path.join('/tmp', self.config.name), exist_ok=True)
            wandb.login(key=self.config.wandb_key, host=self.config.wandb_host)
            wandb.init(
                project=self.config.wandb_project,
                dir=os.path.join('/tmp', self.config.name),
                name=self.config.name,
                config=OmegaConf.to_container(self.config, resolve=True)
            )
            # save current config
            config_name = f'config_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}.yaml'
            OmegaConf.save(self.config, os.path.join(self.output_dir, config_name))

        self.s3 = S3(bucket=self.s3_bucket, subdir=self.s3_dir)


    def setup_dataloader(self):
        if 'prompt_file' in self.validation_data:
            if isinstance(self.validation_data['prompt_file'], ListConfig):
                self.validation_data.prompts = []
                for pf in self.validation_data['prompt_file']:
                    self.validation_data.prompts.extend([p.strip() for p in open(pf, 'r').readlines()])
            else:
                self.validation_data.prompts = [p.strip() for p in
                                                open(self.validation_data.prompt_file, 'r').readlines()]

        self.train_dataloader = prepare_dataloader(
            dataset_type=self.dataset_type,
            train_data=self.train_data,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            conditioning_dropout_prob=self.conditioning_dropout_prob
        )

    @abstractmethod
    def setup_frozen_components(self):
        raise NotImplementedError("Child classes must implement frozen components.")

    @abstractmethod
    def build_model(self):
        raise NotImplementedError("Child classes must implement build_model()")

    def load_model_checkpoints(self, model):
        # load checkpoint if provided
        if self.model_checkpoint_path and os.path.exists(self.model_checkpoint_path):
            self.logger.info(f'Load model from {self.model_checkpoint_path}')
            if os.path.isfile(self.model_checkpoint_path):
                state_dict = safetensors.torch.load_file(self.model_checkpoint_path)
            elif os.path.isdir(self.model_checkpoint_path):
                state_dict = model.__class__.from_pretrained(self.model_checkpoint_path).state_dict()
            else:
                raise FileNotFoundError(f'{self.model_checkpoint_path} not found.')
            incompatible_keys = model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"missing key: {len(incompatible_keys.missing_keys)}")
            self.logger.info(f"unexpected_keys: {len(incompatible_keys.unexpected_keys)}")
            del state_dict

    def count_model_parameter(self, model, name='Model'):
        model_numel, model_numel_trainable = get_model_numel(model)
        self.logger.info(
            "[%s] Trainable params: %s, Total params: %s",
            name,
            format_numel_str(model_numel_trainable),
            format_numel_str(model_numel),
        )

    def possible_apply_gradient_checkpointing(self, model):
        """Apply activation checkpointing to modules listed in checkpoint_modules."""
        if self.gradient_checkpointing:
            assert hasattr(self, 'checkpoint_modules'), f"checkpoint_modules must be set."
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=checkpoint_wrapper,
                check_fn=lambda module: isinstance(module, self.checkpoint_modules),
            )
            self.logger.info(f"Enable gradient checkpointing for {model.__class__.__name__}")

    @abstractmethod
    def setup_models(self):
        raise NotImplementedError("Child classes must implement model initialization.")

    @abstractmethod
    def wrap_model_fsdp(self):
        raise NotImplementedError("Child classes must implement model wrapping.")

    @abstractmethod
    def setup_optimizers_and_lr_scheduler(self):
        raise NotImplementedError("Child classes must implement optimizer and lr_scheduler.")

    def setup_ema_model(self, try_resume=False):
        if self.ema_model and self.ema_start_step <= self.global_step:
            raise NotImplementedError("Child classes to implement EMA.")

    def _get_resume_dir(self):
        """Return the directory the resume helpers read from. Subclasses may override to support external checkpoints."""
        return os.path.join(self.output_dir, "checkpoints")

    def resume_state(self):
        checkpoint_path = os.path.join(self.resume_dir, 'checkpoint.pth')
        curr_step = resume_step(path=checkpoint_path, key='global_step')
        if curr_step > 0:
            self.logger.info(f"Resume from step {curr_step}")
        return curr_step

    @abstractmethod
    def resume_from_checkpoint(self):
        raise NotImplementedError("Child classes must implement checkpoint resuming.")

    @abstractmethod
    def train_step(self, *args, **kwargs):
        raise NotImplementedError("Child classes must implement the training step.")

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError("Child classes must implement evaluation.")

    @abstractmethod
    def save_models(self):
        raise NotImplementedError("Child classes must implement save model.")

    @abstractmethod
    def _log_progress(self, *args, **kwargs):
        raise NotImplementedError("Child classes must implement log progress.")

    @abstractmethod
    def setup_timers(self):
        self.data_timer = TimerMeter(
            max_length=self.log_steps * self.gradient_accumulation_steps,
            wait_for_all=False
        )

    @abstractmethod
    def train(self):
        self.logger.info("***** Running training *****")

        for epoch in range(self.config.get("num_train_epochs", 100)):
            self.data_timer.tic()
            for step, batch in enumerate(self.train_dataloader, start=1):
                self.accumulation_index = step % self.gradient_accumulation_steps

                # load data

                self.data_timer.toc()

                # Dispatch to the specific algorithm's implementation
                log_dict = self.train_step(*batch)

                if self.accumulation_index == 0:
                    self.global_step += 1

                    self._log_progress(log_dict)

                    if self.global_step % self.checkpointing_steps == 0:
                        self.save_models()

                    if self.global_step % self.validation_steps == 0:
                        self.evaluate()

                self.data_timer.tic()
                if self.global_step >= self.max_train_steps:
                    self.logger.info("Finish training")
                    return

    def set_requires_gradient_reduce(self, *models: torch.nn.Module):
        """Toggle FSDP2 all-reduce: skip on intermediate accumulation steps, enable on the final step."""
        if self.gradient_accumulation_steps <= 1:
            return
        requires_reduce = self.accumulation_index == 0
        for model in models:
            for m in model.modules():
                if isinstance(m, FSDPModule):
                    m.set_requires_all_reduce(requires_reduce)
                    m.set_requires_gradient_sync(requires_reduce)