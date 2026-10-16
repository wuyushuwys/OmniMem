"""Complete checkpoints for NSA self-forcing trainers: models, EMA, schedulers, RNG, KV cache."""
import hashlib
from pathlib import Path

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from omnimem.trainer.complete_checkpoint import CompleteCheckpointMixin
from omnimem.utils.misc import unwrap_model
from omnimem.utils.ode_checkpoint_store import ODECheckpointStore, check_model_dir, config_differences
from omnimem.utils.ode_training_state import (
    call_on_all_ranks, load_training_state, read_training_metadata, save_training_state,
)
from omnimem.utils.torch_utils import create_ema_model


_SF_FAMILIES = {"nsa-self-forcing", "nsa-self-forcing-streaming", "nsa-self-forcing-switch"}


def _without_removed_options(recipe):
    """Drop ``enable_block_level_cache: False`` from old recipes; ``True`` cannot be resumed."""
    recipe = dict(recipe)
    if recipe.pop("enable_block_level_cache", False):
        raise ValueError("The self-forcing checkpoint was trained with enable_block_level_cache=True, "
                         "which has been removed; it cannot resume complete training")
    return recipe


def _check_payload_contents(path, step, world_size):
    path = Path(path)
    generator = read_training_metadata(path, _SF_FAMILIES)
    critic = read_training_metadata(path / "critic_state", _SF_FAMILIES)
    for role, metadata in (("generator", generator), ("critic", critic)):
        if (metadata["global_step"], metadata["world_size"]) != (step, world_size):
            raise ValueError(f"{role} step/world size differs from the checkpoint")
        if metadata["extra_state"].get("optimizer_role") != role:
            raise ValueError(f"Wrong optimizer role for {role}")
    extra = generator["extra_state"]
    family = extra["trainer_family"]
    if family != critic["extra_state"]["trainer_family"]:
        raise ValueError("Generator and critic trainer families differ")
    if extra["training_config"] != critic["extra_state"]["training_config"]:
        raise ValueError("Generator and critic training configurations differ")
    cursor = extra.get("data_cursor")
    if (not isinstance(cursor, dict) or set(cursor) != {"epoch", "batches_consumed"}
            or any(type(value) is not int or value < 0 for value in cursor.values())
            or critic["extra_state"].get("data_cursor") != cursor):
        raise ValueError("Missing or inconsistent SF data cursor")
    recipe = extra["training_config"]
    if type(recipe.get("ema_model")) is not bool or type(recipe.get("ema_start_step")) is not int:
        raise ValueError("Missing delayed EMA policy")
    active = recipe["ema_model"] and step >= recipe["ema_start_step"]
    if type(extra.get("ema_active")) is not bool or extra["ema_active"] != active:
        raise ValueError("EMA activation state disagrees with the completed step")
    if critic["extra_state"].get("ema_active") != active:
        raise ValueError("Critic EMA activation metadata differs")
    generator_config = check_model_dir(path / "model")
    check_model_dir(path / "fake_net")
    if active:
        if config_differences(check_model_dir(path / "ema"), generator_config):
            raise ValueError("EMA configuration differs from the generator")
    elif (path / "ema").exists():
        raise ValueError("Checkpoint contains EMA before its activation boundary")
    for state_dir in (path, path / "critic_state"):
        if not (state_dir / "optimizer/.metadata").is_file() or not any(
                p.is_file() and p.name != ".metadata" for p in (state_dir / "optimizer").rglob("*")):
            raise ValueError(f"Missing optimizer metadata/shards: {state_dir}")
        expected = {f"rank_{rank:05d}.pt" for rank in range(world_size)}
        if {p.name for p in (state_dir / "rng").glob("*.pt")} != expected:
            raise ValueError(f"Incomplete rank RNG payload: {state_dir}")
        for rank in range(world_size):
            payload = torch.load(state_dir / "rng" / f"rank_{rank:05d}.pt", map_location="cpu", weights_only=True)
            if (payload.get("global_step"), payload.get("world_size"), payload.get("rank")) != (step, world_size, rank):
                raise ValueError(f"RNG identity disagrees with checkpoint: {state_dir}, rank {rank}")
            if not isinstance(payload.get("rng_state"), dict):
                raise ValueError("Missing rank RNG state")
    streaming_dir = path / "streaming"
    if family == "nsa-self-forcing":
        if streaming_dir.exists():
            raise ValueError("Ordinary self-forcing checkpoint contains unexpected streaming state")
    else:
        # Keep ordinary SF independent of streaming runtime code and validate
        # every rank before publishing or accepting the completion marker.
        from omnimem.trainer.streaming_checkpoint import validate_streaming_checkpoint_payload

        expected = {f"rank_{rank:05d}.pt" for rank in range(world_size)}
        if {p.name for p in streaming_dir.glob("*.pt")} != expected:
            raise ValueError("Incomplete rank streaming payload")
        for rank in range(world_size):
            payload = torch.load(streaming_dir / f"rank_{rank:05d}.pt", map_location="cpu", weights_only=True)
            validate_streaming_checkpoint_payload(
                payload, step=step, world_size=world_size, rank=rank,
                trainer_family=family, training_config=recipe)


def _check_payload(path, step, world_size):
    try:
        _check_payload_contents(path, step, world_size)
    except Exception as exc:
        # Convert malformed weights-only metadata to a validation failure so
        # latest() can still select an older, complete checkpoint.
        raise ValueError(f"Invalid self-forcing checkpoint payload: {exc}") from exc


class SelfForcingCheckpointStore(ODECheckpointStore):
    _check_payload = staticmethod(_check_payload)


class SelfForcingCheckpointMixin(CompleteCheckpointMixin):
    """Opt-in complete SF state, with optional per-trainer runtime hooks."""

    _checkpoint_label = "self-forcing"
    _checkpoint_store_cls = SelfForcingCheckpointStore
    _legacy_checkpoint_names = ("checkpoint.pth", "generator", "fake_net", "model", "checkpoint_dcp_generator")
    _sf_trainer_family = "nsa-self-forcing"

    def _training_config(self):
        keys = ("frame_per_block", "same_step_across_blocks", "timestep_shift", "latents_shape",
                "max_distillation_steps", "timestep_sample_method", "real_guidance_scale",
                "dfake_gen_update_ratio", "critic_loss_type", "ema_model", "ema_decay", "ema_start_step",
                "gradient_accumulation_steps", "conditioning_dropout_prob", "max_grad_norm",
                "learning_rate", "adam_beta1", "adam_beta2", "adam_weight_decay", "adam_epsilon",
                "critic_learning_rate", "critic_adam_beta1", "critic_adam_beta2",
                "critic_adam_weight_decay", "critic_adam_epsilon", "window_size", "sink_size",
                "enable_kv_evict", "enable_chunk_per_head_cache",
                "kv_cache_lru_max_size", "per_head_gather_variant")
        config = {key: getattr(self, key) for key in keys}
        config.update({key: self.config.get(key) for key in
                       ("train_data", "dtype", "real_model_path", "nsa_kwargs", "dataset_type",
                        "null_prompt", "text_encoder_path", "tokenizer_path")})
        config["scheduler"] = self._scheduler_config()
        config["global_seed"] = self.config.get("global_seed", 42)
        prompt_path = self.config.get("train_data", {}).get("prompt_path")
        if prompt_path:
            if not hasattr(self, "_sf_prompt_sha256"):
                self._sf_prompt_sha256 = hashlib.sha256(Path(prompt_path).read_bytes()).hexdigest()
            config["prompt_sha256"] = self._sf_prompt_sha256
        return OmegaConf.to_container(OmegaConf.create(config), resolve=True)

    def _check_resume_recipe(self, checkpoint):
        extra = read_training_metadata(checkpoint, _SF_FAMILIES)["extra_state"]
        if extra["trainer_family"] != self._sf_trainer_family:
            raise ValueError("Trainer family differs from the completed self-forcing checkpoint")
        if _without_removed_options(extra["training_config"]) != self._training_config():
            raise ValueError("Training configuration differs from the completed self-forcing checkpoint")

    def _ema_active(self):
        return self.ema_model and self.global_step >= self.ema_start_step

    def _checkpoint_models(self):
        return [("model", self.generator), ("fake_net", self.fake_net), ("ema", self.ema)]

    def _prune_local_checkpoints(self, resume=False):
        keep = self.config.get("checkpoint_keep_local")
        if keep is not None and (type(keep) is not int or keep < 0):
            raise ValueError("checkpoint_keep_local must be a nonnegative integer for complete SF resume")
        if keep == 0 and self._checkpoint_store.s3 is None:
            raise ValueError("checkpoint_keep_local=0 requires upload and an S3 checkpoint store")
        super()._prune_local_checkpoints(resume)

    def resume_state(self):
        if not getattr(self, "complete_sf_resume", False):
            # Bypass the complete checkpoint mixin, preserving the legacy DMD/SF lifecycle.
            return super(CompleteCheckpointMixin, self).resume_state()
        self._sf_resume_state_loaded = False
        self._sf_data_cursor = {"epoch": 0, "batches_consumed": 0}
        return super().resume_state()

    def resume_from_checkpoint(self):
        if self._resume_checkpoint is not None:
            self._restore_checkpoint_weights("model", self.generator)
            self._restore_checkpoint_weights("fake_net", self.fake_net)

    def _restore_training_state(self):
        if self._resume_checkpoint is None:
            return None
        generator = load_training_state(self._resume_checkpoint, self.generator, self.generator_optimizer,
                                        self.generator_lr_scheduler, expected_step=self.global_step)
        load_training_state(self._resume_checkpoint / "critic_state", self.fake_net, self.critic_optimizer,
                            self.critic_lr_scheduler, expected_step=self.global_step)
        self._resume_rng_state = generator["rng_state"]
        self._sf_data_cursor = generator["extra_state"]["data_cursor"]
        restore_extra = getattr(self, "_sf_restore_extra_state", None)
        if restore_extra is not None:
            call_on_all_ranks(lambda: restore_extra(self._resume_checkpoint), "load the streaming rollout state")
        self._sf_resume_state_loaded = True
        self.logger.info(f"Restored both SF optimizers/schedulers and prompt cursor {self._sf_data_cursor}.")
        return generator

    def _sf_training_batches(self):
        """Rebuild deterministic prompt order without consuming training RNG."""
        loader = self.train_dataloader
        if len(loader) == 0:
            raise ValueError("Self-forcing prompt loader has no batches; check corpus size and world size")
        start_epoch = self._sf_data_cursor["epoch"]
        consumed = self._sf_data_cursor["batches_consumed"]
        if consumed > len(loader):
            raise ValueError("Saved SF data cursor exceeds the current epoch length")
        for epoch in range(start_epoch, self.num_train_epochs):
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            data_generator = torch.Generator().manual_seed(self.config.get("global_seed", 42) + epoch)
            # TextDataset itself is deterministic. Isolate worker base seeds and
            # the single-rank RandomSampler from model/noise RNG streams.
            loader.generator = data_generator
            if isinstance(sampler, torch.utils.data.RandomSampler):
                sampler.generator = data_generator
            for index, batch in enumerate(loader):
                if epoch == start_epoch and index < consumed:
                    continue
                self._sf_data_cursor = {"epoch": epoch, "batches_consumed": index + 1}
                yield batch
            self._sf_data_cursor = {"epoch": epoch + 1, "batches_consumed": 0}

    def setup_ema_model(self, try_resume=False):
        if self.ema_model and self.global_step >= self.ema_start_step and self.ema is None:
            self.ema, _ = create_ema_model(unwrap_model(self.generator), module_cls=self.ema_module_cls)
            if try_resume and self._resume_checkpoint is not None:
                self._restore_checkpoint_weights("ema", self.ema)
            self.ema.requires_grad_(False).eval()
            if dist.is_initialized():
                for block in self.ema.blocks:
                    fully_shard(block, **self.fsdp_kwargs)
                fully_shard(self.ema, **self.fsdp_kwargs.root_kwargs)
            else:
                self.ema.to(self.device)
        if (try_resume and self._resume_checkpoint is not None
                and getattr(self, "_sf_resume_state_loaded", False)
                and self.config.get("checkpoint_keep_local") == 0):
            # Streaming payloads are already loaded in memory; EMA is the last
            # checkpoint reader. Wait for every rank before removing shared files.
            call_on_all_ranks(lambda: None, "finish reading the resume checkpoint")
            self._prune_local_checkpoints()

    def save_models(self):
        if getattr(self, "accumulation_index", 0) != 0:
            raise ValueError("SF checkpoints require a completed gradient accumulation boundary")
        self.setup_ema_model()
        super().save_models()

    def _save_training_states(self, staging):
        extra = {"trainer_family": self._sf_trainer_family, "training_config": self._training_config(),
                 "ema_active": self.ema is not None,
                 "data_cursor": self._sf_data_cursor.copy()}
        save_training_state(staging, self.generator, self.generator_optimizer, self.generator_lr_scheduler,
                            self.global_step, extra_state={**extra, "optimizer_role": "generator"})
        save_training_state(staging / "critic_state", self.fake_net, self.critic_optimizer,
                            self.critic_lr_scheduler, self.global_step,
                            extra_state={**extra, "optimizer_role": "critic"})
        save_extra = getattr(self, "_sf_save_extra_state", None)
        if save_extra is not None:
            call_on_all_ranks(lambda: save_extra(staging), "save the streaming rollout state")
