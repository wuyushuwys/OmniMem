"""Complete model, optimizer, scheduler and RNG checkpoints shared by the ODE, causal and SF trainers."""

import json
import os
from pathlib import Path
import posixpath

from omegaconf import OmegaConf
import torch.distributed as dist

from omnimem.utils.misc import is_main_process, unwrap_model
from omnimem.utils.ode_checkpoint_store import COMPLETED, ODECheckpointStore, config_differences
from omnimem.utils.ode_training_state import (
    call_on_all_ranks, capture_rng_state, load_training_state, restore_rng_state, save_training_state,
)
from omnimem.utils.torch_utils import get_fsdp_state_dict, resume_model, save_sharded_safetensors


class CompleteCheckpointMixin:
    """Save and resume one atomic checkpoint per step; subclasses fill in the hooks below."""

    _checkpoint_label = "ODE"
    _checkpoint_store_cls = ODECheckpointStore
    _legacy_checkpoint_names = ("checkpoint.pth", "model")

    def _check_resume_recipe(self, checkpoint):
        """Raise on rank zero when ``checkpoint`` was saved with another training recipe."""

    def _checkpoint_extra_state(self):
        """Return the weights-only ``extra_state`` saved with the training state."""
        raise NotImplementedError

    def _ema_active(self):
        return False

    def _checkpoint_models(self):
        return [("model", self.model), ("ema", getattr(self, "ema", None))]

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
            raise RuntimeError(f"{self._checkpoint_label} checkpoint operation failed: {result[0]['error']}")
        return result[0]["value"]

    def _scheduler_config(self):
        # LambdaLR restores counters, but its lambda is rebuilt from this configuration.
        config = {"type": self.lr_scheduler_type, "warmup_steps": self.lr_warmup_steps,
                  "kwargs": dict(self.lr_scheduler_kwargs)}
        if self.lr_scheduler_type not in ("constant", "constant_with_warmup"):
            config["max_train_steps"] = self.max_train_steps
        return OmegaConf.to_container(OmegaConf.create(config), resolve=True)

    def _prune_local_checkpoints(self, resume=False):
        keep = self.config.get("checkpoint_keep_local")
        if keep is None or self._checkpoint_store.s3 is None:
            return
        if resume:
            # Keep the newest local copy until the resumed state has loaded.
            keep = max(keep, 1)
        # Rank zero alone deletes shared files after verifying every S3 object.
        removed = self._checkpoint_rank0_call(
            lambda: [str(path) for path in self._checkpoint_store.prune_local(keep)])
        self.logger.info(f"Pruned local {self._checkpoint_label} checkpoints already on S3 (keep {keep}): {removed}")

    def _select_checkpoint(self):
        checkpoint = self._checkpoint_store.latest()
        if checkpoint is None:
            if any((Path(self.resume_dir) / name).exists() for name in self._legacy_checkpoint_names):
                raise ValueError(f"Legacy {self._checkpoint_label} checkpoint cannot be resumed as complete state; "
                                 "initialize a new run from its weights instead")
            return None
        # latest() already verified all hashes; published directories are immutable.
        manifest = json.loads((checkpoint / COMPLETED).read_text())
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if manifest["world_size"] != world_size:
            raise ValueError(f"Full {self._checkpoint_label} resume requires the original number of training ranks")
        self._check_resume_recipe(checkpoint)
        return {"path": str(checkpoint), "step": manifest["global_step"]}

    def resume_state(self):
        client = self.s3.s3_client if self.upload else None
        prefix = self.config.get("checkpoint_s3_prefix")
        if client is not None and prefix is None:
            if os.path.isabs(self.config.output_dir):
                raise ValueError("An absolute output_dir requires an explicit checkpoint_s3_prefix")
            prefix = posixpath.join(self.s3_dir, self.output_dir, "checkpoints")
        self._checkpoint_store = self._checkpoint_store_cls(
            Path(self.resume_dir), s3_client=client,
            bucket=self.s3_bucket if client is not None else None,
            s3_prefix=prefix if client is not None else None,
        )
        self._resume_rng_state = None
        self._prune_local_checkpoints(resume=True)
        selected = self._checkpoint_rank0_call(self._select_checkpoint)
        self._resume_checkpoint = Path(selected["path"]) if selected else None
        self.logger.info(f"Resume complete {self._checkpoint_label} checkpoint {self._resume_checkpoint}" if selected
                         else f"No completed {self._checkpoint_label} checkpoint; starting a new run.")
        return selected["step"] if selected else 0

    def _restore_checkpoint_weights(self, role, module, name=None):
        def restore():
            directory = self._resume_checkpoint / role
            different = config_differences(json.loads((directory / "config.json").read_text()),
                                           json.loads(json.dumps(dict(module.config))))
            if different:
                raise ValueError(f"{name or role} configuration differs from the resume checkpoint: {different}")
            resume_model(str(directory), module, strict=True)
        call_on_all_ranks(restore, f"restore {role} weights from {self._resume_checkpoint}")

    def resume_from_checkpoint(self):
        if self._resume_checkpoint is not None:
            self._restore_checkpoint_weights("model", self.model)

    def _restore_training_state(self):
        """Restore optimizer and scheduler after FSDP wrapping and return the loaded state, if any."""
        if self._resume_checkpoint is None:
            return None
        state = load_training_state(self._resume_checkpoint, self.model, self.optimizer,
                                    self.lr_scheduler, expected_step=self.global_step)
        self._resume_rng_state = state["rng_state"]
        self.logger.info("Restored optimizer and LR scheduler; the data stream will resample on resume.")
        return state

    def _save_training_states(self, staging):
        save_training_state(staging, self.model, self.optimizer, self.lr_scheduler, self.global_step,
                            extra_state=self._checkpoint_extra_state())

    def save_models(self):
        step = self.global_step
        staging = Path(self._checkpoint_rank0_call(lambda: str(self._checkpoint_store.begin(step))))

        def require_shared():
            if not staging.is_dir():
                raise FileNotFoundError(f"Checkpoint staging {staging} must be shared by all training ranks")
            if (getattr(self, "ema", None) is not None) != self._ema_active():
                raise ValueError("EMA state differs from the configured EMA policy")
        call_on_all_ranks(require_shared, f"prepare checkpoint {staging}")
        for role, module in self._checkpoint_models():
            if module is None:
                continue
            self.logger.info(f"Checkpoint step {step}: gathering {role} weights.")
            state = get_fsdp_state_dict(unwrap_model(module))

            def save_weights():
                directory = str(staging / role)
                module.save_config(save_directory=directory)
                save_sharded_safetensors(state_dict=state, save_dir=directory)
            self._checkpoint_rank0_call(save_weights)
            del state
        self.logger.info(f"Checkpoint step {step}: saving optimizer, scheduler and RNG state.")
        self._save_training_states(staging)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        checkpoint = Path(self._checkpoint_rank0_call(
            lambda: str(self._checkpoint_store.commit(staging, step, world_size))))
        self.logger.info(f"Saved complete {self._checkpoint_label} checkpoint: {checkpoint}")
        if self._checkpoint_store.s3 is not None:
            self._checkpoint_rank0_call(lambda: self._checkpoint_store.upload(checkpoint))
            self.logger.info(f"Checkpoint step {step}: S3 upload complete.")
            self._prune_local_checkpoints()

    def _evaluate_without_rng_change(self, initial=False):
        state = capture_rng_state()
        try:
            self.evaluate_init() if initial else self.evaluate()
        finally:
            restore_rng_state(state)
