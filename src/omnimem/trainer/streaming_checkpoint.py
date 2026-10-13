"""Per-rank rollout state for complete streaming self-forcing checkpoints."""
from contextlib import contextmanager
import hashlib
from pathlib import Path

from omegaconf import OmegaConf
import torch
import torch.distributed as dist

from omnimem.utils.ode_training_state import capture_rng_state, restore_rng_state
from omnimem.models.cache import MMCache


def _tensor_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {key: _tensor_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_tensor_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_tensor_tree(item, device) for item in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ValueError(f"Unsupported streaming state value: {type(value).__name__}")


def validate_streaming_checkpoint_payload(payload, *, step, world_size, rank,
                                          trainer_family, training_config):
    """Validate on CPU before publishing or selecting a checkpoint."""
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Invalid streaming checkpoint format")
    for key, expected in (("global_step", step), ("world_size", world_size),
                          ("rank", rank), ("trainer_family", trainer_family)):
        if payload.get(key) != expected:
            raise ValueError(f"Streaming checkpoint {key} differs from its identity")
    if type(payload.get("streaming_active")) is not bool:
        raise ValueError("Missing streaming activity flag")
    last_eval = payload.get("last_eval_step")
    if type(last_eval) is not int or not 0 <= last_eval <= step:
        raise ValueError("Invalid streaming evaluation cursor")
    state = payload.get("streaming_state")
    if not isinstance(state, dict) or "kv_cache" not in payload:
        raise ValueError("Missing streaming runtime state")
    cache = payload["kv_cache"]
    if cache is not None and not isinstance(cache, dict):
        raise ValueError("Invalid streaming KV cache")
    if cache is not None:
        MMCache.validate_checkpoint_state_dict(cache)
    if training_config.get("streaming_checkpoint_at_rollout_boundary", False) and (
            payload["streaming_active"] or cache is not None
            or any(value is not None for key, value in state.items() if key != "current_length")):
        raise ValueError("Rollout-boundary streaming checkpoint contains rollout state")
    if not payload["streaming_active"]:
        if state and (state.get("current_length") != 0 or state.get("previous_frames") is not None):
            raise ValueError("Inactive streaming checkpoint contains an unfinished rollout")
        return
    if cache is None:
        raise ValueError("Active streaming checkpoint has no KV cache")
    current, maximum = state.get("current_length"), state.get("temp_max_length")
    if type(current) is not int or type(maximum) is not int or not 0 < current <= maximum:
        raise ValueError("Invalid streaming sequence position")
    choices = training_config.get("streaming_possible_max_length")
    allowed_lengths = choices if choices is not None else [training_config["streaming_max_length"]]
    if maximum not in allowed_lengths or current % training_config["frame_per_block"]:
        raise ValueError("Streaming sequence position disagrees with its recipe")
    frames = state.get("previous_frames")
    shape = training_config["latents_shape"]
    frame_per_block = training_config["frame_per_block"]
    chunk_size = training_config["streaming_chunk_size"]
    boundary_mode = training_config.get("streaming_boundary_mode")
    # A boundary block plus fewer new blocks can form a shorter loss window.
    # Legacy raw-overlap windows always padded their history to chunk_size.
    min_window = (training_config["streaming_min_new_frame"] + frame_per_block
                  if boundary_mode == "vae_roundtrip_block" else chunk_size)
    if (not isinstance(frames, torch.Tensor) or frames.ndim != 5
            or list(frames.shape[1:2]) != list(shape[:1])
            or list(frames.shape[3:]) != list(shape[2:])
            or not min_window <= frames.shape[2] <= chunk_size
            or frames.shape[2] % frame_per_block
            or frames.shape[0] <= 0):
        raise ValueError("Invalid streaming history frames")
    if frames.shape[0] != cache["configuration"]["available_shape"]["k_cache"][0]:
        raise ValueError("Streaming frame batch differs from the KV cache batch")

    def check_condition(condition):
        if (not isinstance(condition, dict) or type(condition.get("seq_len")) is not int
                or condition["seq_len"] <= 0 or not isinstance(condition.get("context"), (list, tuple))
                or len(condition["context"]) != frames.shape[0]
                or any(not isinstance(item, torch.Tensor) for item in condition["context"])):
            raise ValueError("Invalid streaming text conditioning")

    check_condition(state.get("condition_dict"))
    check_condition(state.get("uncondition_dict"))
    if trainer_family == "nsa-self-forcing-switch":
        switch_frame = state.get("switch_frame_index")
        if type(switch_frame) is not int or not 0 < switch_frame <= maximum:
            raise ValueError("Invalid streaming switch position")
        if "switch_condition_dict" not in state:
            raise ValueError("Missing streaming switch conditioning")
        if state["switch_condition_dict"] is not None:
            check_condition(state["switch_condition_dict"])


class StreamingCheckpointMixin:
    """Extend the shared two-optimizer transaction with each rank's live rollout."""

    def _training_config(self):
        config = super()._training_config()
        keys = ("streaming_chunk_size", "streaming_max_length", "streaming_min_new_frame",
                "streaming_possible_max_length", "train_first_chunk")
        config.update({key: getattr(self, key) for key in keys})
        # Record the rollout algorithm even at inactive sequence boundaries.
        # Raw-overlap checkpoints must start a new run when adopting VAE boundaries.
        config["streaming_boundary_mode"] = self.streaming_boundary_mode
        schedule = getattr(self, "streaming_generator_schedule", "fixed")
        # Omit the legacy default to keep existing fixed-schedule recipes
        # resumable. A changed schedule is a new training run, not exact resume.
        if schedule != "fixed":
            config["streaming_generator_schedule"] = schedule
        config["kv_evict_min_gpu_chunks"] = getattr(self, "kv_evict_min_gpu_chunks", None)
        config["vae_path"] = self.config.get("vae_path")
        config["scheduler_path"] = self.config.get("scheduler_path")
        if config["scheduler_path"] is not None:
            config["noise_scheduler_sha256"] = hashlib.sha256(
                Path(config["scheduler_path"]).read_bytes()).hexdigest()
        config["gradient_checkpointing"] = self.gradient_checkpointing
        config["torch_compile"] = self.config.get("torch_compile", False)
        config["num_train_epochs"] = self.num_train_epochs
        # Preserve the default recipe; changing the enabled policy requires a new run.
        if getattr(self, "streaming_checkpoint_at_rollout_boundary", False):
            config["streaming_checkpoint_at_rollout_boundary"] = True
        if self._sf_trainer_family == "nsa-self-forcing-switch":
            config.update(switch_prompt_file=self.switch_prompt_file, switch_mode=self.switch_mode,
                          switch_choices=self.switch_choices)
            config["switch_prompt_sha256"] = (
                hashlib.sha256(Path(self.switch_prompt_file).read_bytes()).hexdigest()
                if self.switch_prompt_file is not None else None)
        return OmegaConf.to_container(OmegaConf.create(config), resolve=True)

    def _sf_save_extra_state(self, staging):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        boundary = getattr(self, "streaming_checkpoint_at_rollout_boundary", False)
        if boundary and self.streaming_active:
            raise ValueError("Rollout-boundary streaming checkpoints require a reset rollout")
        payload = dict(format_version=1, global_step=self.global_step, rank=rank, world_size=world,
                       trainer_family=self._sf_trainer_family,
                       streaming_active=self.streaming_active,
                       streaming_state=_tensor_tree(self.streaming_state, "cpu"),
                       kv_cache=(self.streaming_kv_cache.checkpoint_state_dict()
                                 if self.streaming_kv_cache is not None and not boundary else None),
                       last_eval_step=self._last_streaming_eval_step)
        validate_streaming_checkpoint_payload(
            payload, step=self.global_step, world_size=world, rank=rank,
            trainer_family=self._sf_trainer_family, training_config=self._training_config())
        directory = Path(staging) / "streaming"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(payload, directory / f"rank_{rank:05d}.pt")

    def _sf_restore_extra_state(self, checkpoint):
        rank = dist.get_rank() if dist.is_initialized() else 0
        self._pending_streaming_resume = torch.load(
            Path(checkpoint) / "streaming" / f"rank_{rank:05d}.pt",
            map_location="cpu", weights_only=True)

    def _restore_streaming_runtime(self):
        """Called after models and the streaming attributes have been initialized."""
        payload = getattr(self, "_pending_streaming_resume", None)
        if payload is None:
            return

        def restore():
            self.streaming_state = _tensor_tree(payload["streaming_state"], self.device)
            self.streaming_active = payload["streaming_active"]
            self._last_streaming_eval_step = payload["last_eval_step"]
            cache_state = payload["kv_cache"]
            if cache_state is not None:
                batch_size = cache_state["configuration"]["available_shape"]["k_cache"][0]
                self._ensure_streaming_kv_cache(self._streaming_cache_shape_tensor(batch_size))
                self.streaming_kv_cache.load_checkpoint_state_dict(cache_state)
        self._checkpoint_all_ranks_call(restore)
        self._pending_streaming_resume = None

    def _streaming_cache_shape_tensor(self, batch_size):
        # The cache builder only reads shape. A scalar expanded as a view avoids
        # allocating a full long video solely to initialize the cache.
        maximum = max(self.streaming_possible_max_length) if self.streaming_possible_max_length is not None \
            else self.streaming_max_length
        channels, _, height, width = self.latents_shape
        return torch.empty((), device=self.device, dtype=self.dtype).expand(
            batch_size, channels, maximum, height, width)

    @contextmanager
    def _preserve_streaming_evaluation_state(self):
        """Suspend the live cache while evaluation uses its own shape/CFG setup."""
        rng = capture_rng_state()
        cache = self.streaming_kv_cache
        state, active = _tensor_tree(self.streaming_state, "cpu"), self.streaming_active
        cache_state = cache.checkpoint_state_dict() if cache is not None else None
        kwargs = self._eval_kwargs
        previous = kwargs.get("_kv_cache")
        had_previous = "_kv_cache" in kwargs
        try:
            if cache is not None:
                cache.reset()
            self.streaming_kv_cache = None
            # Evaluation may have another batch size, resolution, or CFG > 1.
            # An injected training cache does not support those combinations.
            kwargs.pop("_kv_cache", None)
            yield
        finally:
            if had_previous:
                kwargs["_kv_cache"] = previous
            else:
                kwargs.pop("_kv_cache", None)
            self.streaming_kv_cache = cache
            self.streaming_state = _tensor_tree(state, self.device)
            self.streaming_active = active
            try:
                if cache is not None:
                    cache.load_checkpoint_state_dict(cache_state)
            finally:
                restore_rng_state(rng)
