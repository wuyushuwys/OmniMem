"""Complete teacher-forcing checkpoint orchestration, including EMA weights."""

from pathlib import Path

from omegaconf import OmegaConf

from omnimem.trainer.complete_checkpoint import CompleteCheckpointMixin
from omnimem.utils.ode_checkpoint_store import ODECheckpointStore, check_model_dir, config_differences
from omnimem.utils.ode_training_state import read_training_metadata


def _check_payload(path, step, world_size):
    """Require causal training metadata and the EMA weights its recipe enables."""
    metadata = read_training_metadata(path, {"causal"})
    if (metadata["global_step"], metadata["world_size"]) != (step, world_size):
        raise ValueError("Training-state step/world size differs from the checkpoint")
    enabled = metadata["extra_state"]["training_config"].get("ema_model")
    if type(enabled) is not bool:
        raise ValueError("Checkpoint has no valid EMA policy")
    if not enabled:
        if (Path(path) / "ema").exists():
            raise ValueError("Checkpoint contains EMA weights but its EMA policy is disabled")
        return
    if config_differences(check_model_dir(Path(path) / "ema"), check_model_dir(Path(path) / "model")):
        raise ValueError("EMA configuration differs from the saved model configuration")


class CausalCheckpointStore(ODECheckpointStore):
    """Use the existing atomic store, adding required causal metadata and EMA."""
    _check_payload = staticmethod(_check_payload)


class CausalCheckpointMixin(CompleteCheckpointMixin):
    _checkpoint_label = "causal"
    _checkpoint_store_cls = CausalCheckpointStore
    _legacy_checkpoint_names = ("checkpoint.pth", "model", "checkpoint_dcp")

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

    def _check_resume_recipe(self, checkpoint):
        if read_training_metadata(checkpoint, {"causal"})["extra_state"]["training_config"] != self._training_config():
            raise ValueError("Training configuration differs from the completed causal checkpoint")

    def _checkpoint_extra_state(self):
        return {"trainer_family": "causal", "training_config": self._training_config()}

    def _ema_active(self):
        return self.ema_model
