"""Complete teacher-forcing checkpoint orchestration, including EMA weights."""

import json
import os
from pathlib import Path, PurePosixPath
import posixpath

from omegaconf import OmegaConf
import torch
import torch.distributed as dist

from omnimem.utils.misc import is_main_process, unwrap_model
from omnimem.utils.ode_checkpoint_store import ODECheckpointStore
from omnimem.utils.ode_training_state import capture_rng_state, restore_rng_state, save_training_state, load_training_state
from omnimem.utils.torch_utils import get_fsdp_state_dict, resume_model, save_sharded_safetensors


def _training_metadata(path):
    metadata = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=True)
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint training metadata is not a dictionary")
    extra = metadata.get("extra_state")
    if not isinstance(extra, dict) or extra.get("trainer_family") != "causal":
        raise ValueError("Checkpoint is not a complete causal/teacher-forcing training checkpoint")
    if not isinstance(extra.get("training_config"), dict):
        raise ValueError("Checkpoint is missing its training configuration")
    return metadata


def _check_ema_payload(path):
    """EMA is mandatory when the saved training recipe enables it."""
    path = Path(path)
    metadata = _training_metadata(path)
    enabled = metadata["extra_state"]["training_config"].get("ema_model")
    if type(enabled) is not bool:
        raise ValueError("Checkpoint has no valid EMA policy")
    if not enabled:
        if (path / "ema").exists():
            raise ValueError("Checkpoint contains EMA weights but its EMA policy is disabled")
        return
    directory = path / "ema"
    saved_config = json.loads((directory / "config.json").read_text())
    model_config = json.loads((path / "model/config.json").read_text())
    if not isinstance(saved_config, dict) or not isinstance(model_config, dict):
        raise ValueError("Model and EMA configurations must be dictionaries")
    semantic = lambda config: {key: value for key, value in config.items() if not key.startswith("_")}
    if semantic(saved_config) != semantic(model_config):
        raise ValueError("EMA configuration differs from the saved model configuration")
    single = directory / "diffusion_pytorch_model.safetensors"
    index = directory / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        weights = json.loads(index.read_text()).get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("EMA shard index has no weight_map")
        for name in weights.values():
            if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
                    or PurePosixPath(name).is_absolute()
                    or any(part in ("", ".", "..") for part in name.split("/"))
                    or not name.endswith(".safetensors") or not (directory / name).is_file()):
                raise ValueError(f"EMA index references a missing or unsafe shard: {name!r}")
    elif not single.is_file():
        raise ValueError("EMA-enabled checkpoint has no EMA weights")


class CausalCheckpointStore(ODECheckpointStore):
    """Use the existing atomic store, adding required causal metadata and EMA."""

    def commit(self, staging, step, world_size):
        _check_ema_payload(staging)
        metadata = _training_metadata(staging)
        if (metadata["global_step"], metadata["world_size"]) != (step, world_size):
            raise ValueError("Training-state step/world size differs from the proposed checkpoint")
        return super().commit(staging, step, world_size)

    def validate(self, path):
        manifest = super().validate(path)
        _check_ema_payload(path)
        metadata = _training_metadata(path)
        if (metadata["global_step"], metadata["world_size"]) != (
                manifest["global_step"], manifest["world_size"]):
            raise ValueError("Training-state step/world size differs from the completion manifest")
        return manifest


class CausalCheckpointMixin:
    def _get_resume_dir(self):
        return self.config.get("checkpoint_dir") or os.path.join(self.output_dir, "checkpoints")

    def _checkpoint_rank0_call(self, operation):
        result = [None]
        if is_main_process():
            try:
                result[0] = {"value": operation(), "error": None}
            except Exception as exc:
                result[0] = {"value": None, "error": f"{type(exc).__name__}: {exc}"}
        if dist.is_initialized():
            dist.broadcast_object_list(result, src=0)
        if result[0]["error"] is not None:
            raise RuntimeError(f"Causal checkpoint operation failed: {result[0]['error']}")
        return result[0]["value"]

    def _checkpoint_all_ranks_call(self, operation):
        result, error = None, None
        try:
            result = operation()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [error]
        if dist.is_initialized():
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError(f"Causal checkpoint rank errors: {errors}")
        return result

    def _scheduler_config(self):
        config = {"type": self.lr_scheduler_type, "warmup_steps": self.lr_warmup_steps,
                  "kwargs": dict(self.lr_scheduler_kwargs)}
        if self.lr_scheduler_type not in ("constant", "constant_with_warmup"):
            config["max_train_steps"] = self.max_train_steps
        return OmegaConf.to_container(OmegaConf.create(config), resolve=True)

    def _training_config(self):
        # Exclude storage and initialization paths: nodes stage them at different
        # local locations, and the full checkpoint supersedes initialization.
        keys = ("teacher_forcing", "frame_per_block", "timestep_shift", "ema_model", "ema_decay",
                "gradient_accumulation_steps", "conditioning_dropout_prob", "max_grad_norm",
                "learning_rate", "adam_beta1", "adam_beta2", "adam_weight_decay", "adam_epsilon")
        config = {key: getattr(self, key) for key in keys}
        config.update({key: self.config.get(key) for key in
                       ("train_data", "window_size", "sink_size", "dtype")})
        config["scheduler"] = self._scheduler_config()
        return OmegaConf.to_container(OmegaConf.create(config), resolve=True)

    def resume_state(self):
        client = self.s3.s3_client if self.upload else None
        prefix = self.config.get("checkpoint_s3_prefix")
        if client is not None and prefix is None:
            if os.path.isabs(self.config.output_dir):
                raise ValueError("An absolute output_dir requires an explicit checkpoint_s3_prefix")
            prefix = posixpath.join(self.s3_dir, self.output_dir, "checkpoints")
        self._checkpoint_store = CausalCheckpointStore(
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
                if any((Path(self.resume_dir) / name).exists() for name in ("checkpoint.pth", "model", "checkpoint_dcp")):
                    raise ValueError("Legacy causal checkpoint cannot be resumed as complete state; use a new run and explicit weight initialization")
                return None
            manifest = json.loads((checkpoint / "COMPLETED.json").read_text())
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            if manifest["world_size"] != world_size:
                raise ValueError("Full causal resume requires the original number of training ranks")
            if _training_metadata(checkpoint)["extra_state"]["training_config"] != self._training_config():
                raise ValueError("Training configuration differs from the completed causal checkpoint")
            return {"path": str(checkpoint), "step": manifest["global_step"]}

        selected = self._checkpoint_rank0_call(select_checkpoint)
        self._resume_checkpoint = Path(selected["path"]) if selected else None
        self.logger.info(f"Resume complete causal checkpoint {self._resume_checkpoint}" if selected
                         else "No completed causal checkpoint; starting a new run.")
        return selected["step"] if selected else 0

    def _restore_checkpoint_weights(self, role, module):
        def restore():
            directory = self._resume_checkpoint / role
            saved = json.loads((directory / "config.json").read_text())
            current = json.loads(json.dumps(dict(module.config)))
            semantic = lambda config: {key: value for key, value in config.items() if not key.startswith("_")}
            if semantic(saved) != semantic(current):
                raise ValueError(f"{role} configuration differs from the resume checkpoint")
            resume_model(str(directory), module, strict=True)
        self._checkpoint_all_ranks_call(restore)

    def resume_from_checkpoint(self):
        if self._resume_checkpoint is not None:
            self._restore_checkpoint_weights("model", self.model)

    def _restore_training_state(self):
        if self._resume_checkpoint is not None:
            state = load_training_state(self._resume_checkpoint, self.model, self.optimizer,
                                        self.lr_scheduler, expected_step=self.global_step)
            self._resume_rng_state = state["rng_state"]
            self.logger.info("Restored causal optimizer/scheduler; data stream will resample on resume.")

    def save_models(self):
        staging = Path(self._checkpoint_rank0_call(lambda: str(self._checkpoint_store.begin(self.global_step))))

        def require_shared():
            if not staging.is_dir():
                raise FileNotFoundError(f"Checkpoint staging {staging} must be shared by all training ranks")
            if self.ema_model != (self.ema is not None):
                raise ValueError("EMA state does not match the configured EMA policy")
        self._checkpoint_all_ranks_call(require_shared)
        for role, module in (("model", self.model), ("ema", self.ema)):
            if module is None:
                continue
            state = get_fsdp_state_dict(unwrap_model(module))

            def save_weights():
                directory = str(staging / role)
                module.save_config(save_directory=directory)
                save_sharded_safetensors(state_dict=state, save_dir=directory)
            self._checkpoint_rank0_call(save_weights)
            del state
        save_training_state(staging, self.model, self.optimizer, self.lr_scheduler, self.global_step,
                            extra_state={"trainer_family": "causal", "training_config": self._training_config(),
                                         "data_resume_policy": "resample_stream"})
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        checkpoint = Path(self._checkpoint_rank0_call(
            lambda: str(self._checkpoint_store.commit(staging, self.global_step, world_size))))
        self.logger.info(f"Saved complete causal checkpoint: {checkpoint}")
        if self._checkpoint_store.s3 is not None:
            self._checkpoint_rank0_call(lambda: self._checkpoint_store.upload(checkpoint))
            keep = self.config.get("checkpoint_keep_local")
            if keep is not None:
                removed = self._checkpoint_rank0_call(lambda: [str(p) for p in self._checkpoint_store.prune_local(keep)])
                self.logger.info(f"Pruned local checkpoints already on S3 (keep {keep}): {removed}")

    def _evaluate_without_rng_change(self):
        state = capture_rng_state()
        try:
            self.evaluate()
        finally:
            restore_rng_state(state)
