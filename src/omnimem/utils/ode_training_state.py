"""Optimizer, scheduler and per-rank RNG payloads for ODE checkpoints; all ranks must call."""
from __future__ import annotations

import io
import random
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)


_FORMAT_VERSION = 1
_CHECKPOINT_WORLD_GROUP: dist.ProcessGroup | None = None
_CHECKPOINT_PROCESS_GROUP: dist.ProcessGroup | None = None


def _checkpoint_process_group() -> dist.ProcessGroup | None:
    """Return a cached CPU process group for DCP planning; NCCL gather_object can stall."""
    global _CHECKPOINT_WORLD_GROUP, _CHECKPOINT_PROCESS_GROUP
    if not dist.is_initialized():
        _CHECKPOINT_WORLD_GROUP = _CHECKPOINT_PROCESS_GROUP = None
        return None
    world_group = dist.group.WORLD
    if _CHECKPOINT_WORLD_GROUP is world_group and _CHECKPOINT_PROCESS_GROUP is not None:
        try:
            dist.get_backend(_CHECKPOINT_PROCESS_GROUP)
            return _CHECKPOINT_PROCESS_GROUP
        except ValueError:
            # The caller may explicitly destroy the metadata group between runs.
            pass
    _CHECKPOINT_PROCESS_GROUP = dist.new_group(
        backend="gloo", timeout=timedelta(minutes=5))
    _CHECKPOINT_WORLD_GROUP = world_group
    return _CHECKPOINT_PROCESS_GROUP


def _rank_and_world_size() -> tuple[int, int]:
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _raise_collective_error(
    error: str | None, operation: str, process_group: dist.ProcessGroup | None,
) -> None:
    """Keep peers out of the next collective if one rank's local I/O failed."""
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=process_group)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message]
    if failures:
        raise RuntimeError(f"Cannot {operation}: {'; '.join(failures)}")


def call_on_all_ranks(operation: Callable[[], Any], description: str,
                      process_group: dist.ProcessGroup | None = None) -> Any:
    """Run ``operation`` on every rank; if any rank fails, all ranks raise."""
    result, error = None, None
    try:
        result = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_collective_error(error, description, process_group)
    return result


def read_training_metadata(path: str | Path, families: set[str] | None = None) -> dict[str, Any]:
    """Load ``training_state.pt``; with ``families``, also require one of them and its training config."""
    metadata = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=True)
    if (not isinstance(metadata, dict) or metadata.get("format_version") != _FORMAT_VERSION
            or not isinstance(metadata.get("extra_state"), dict)):
        raise ValueError(f"Invalid training metadata: {path}")
    extra = metadata["extra_state"]
    if families is not None and (extra.get("trainer_family") not in families
                                 or not isinstance(extra.get("training_config"), dict)):
        raise ValueError(f"Checkpoint is not complete {'/'.join(sorted(families))} training state: {path}")
    return metadata


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU torch and (if already initialized) CUDA RNG state."""
    numpy_state = np.random.get_state()
    cuda_initialized = torch.cuda.is_initialized()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state().cpu() if cuda_initialized else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a rank's saved RNG stream on its current CUDA device, if any."""
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state(cuda_state.cpu(), device=torch.cuda.current_device())
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        numpy_state["keys"].cpu().numpy().astype(np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())


def _class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _check_safe_payload(payload: Any) -> None:
    """Reject metadata that would require arbitrary-object unpickling on resume."""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    torch.load(buffer, map_location="cpu", weights_only=True)


def save_training_state(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    extra_state: Any = None,
) -> None:
    """Save an all-rank payload into the staging directory; ``extra_state`` must be weights-only."""
    path = Path(path)
    rank, world_size = _rank_and_world_size()
    process_group = _checkpoint_process_group()
    error = None
    try:
        if not isinstance(global_step, int) or isinstance(global_step, bool) or global_step < 0:
            raise ValueError("global_step must be a nonnegative integer")
        optimizer_initialized = bool(optimizer.state)
        metadata = {
            "format_version": _FORMAT_VERSION,
            "global_step": global_step,
            "world_size": world_size,
            "optimizer_class": _class_name(optimizer),
            "optimizer_initialized": optimizer_initialized,
            "scheduler_class": _class_name(scheduler),
            "scheduler": scheduler.state_dict(),
            "extra_state": extra_state,
        }
        rng_payload = {
            "global_step": global_step,
            "world_size": world_size,
            "rank": rank,
            "rng_state": capture_rng_state(),
        }
        _check_safe_payload(metadata)
        _check_safe_payload(rng_payload)
        (path / "rng").mkdir(parents=True, exist_ok=True)
        torch.save(rng_payload, path / "rng" / f"rank_{rank:05d}.pt")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_collective_error(error, "prepare ODE training state", process_group)

    # PyTorch materializes an empty Adam state using a zero-LR step. Do not let
    # that implementation detail advance a live, as-yet-unused optimizer.
    try:
        error = None
        try:
            optimizer_state = get_optimizer_state_dict(model, optimizer)
            metadata["optimizer_parameter_groups"] = [
                group["params"] for group in optimizer_state["param_groups"]
            ]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _raise_collective_error(error, "collect ODE optimizer state", process_group)
        dcp.save(
            {"optimizer": optimizer_state}, checkpoint_id=path / "optimizer",
            process_group=process_group,
        )
    finally:
        if not optimizer_initialized:
            optimizer.state.clear()

    error = None
    try:
        if rank == 0:
            torch.save(metadata, path / "training_state.pt")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_collective_error(error, "finish ODE training state", process_group)


def load_training_state(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_step: int,
) -> dict[str, Any]:
    """Restore optimizer/scheduler after FSDP wrapping; return the RNG state for later restore."""
    path = Path(path)
    rank, world_size = _rank_and_world_size()
    process_group = _checkpoint_process_group()
    error = None
    try:
        metadata = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        if metadata["format_version"] != _FORMAT_VERSION:
            raise ValueError(f"Unsupported training-state format: {metadata['format_version']}")
        if metadata["global_step"] != expected_step:
            raise ValueError(
                f"Checkpoint step {metadata['global_step']} != expected step {expected_step}")
        if metadata["world_size"] != world_size:
            raise ValueError(
                f"Cannot resume with changed world size: saved {metadata['world_size']}, current {world_size}")
        if metadata["optimizer_class"] != _class_name(optimizer):
            raise ValueError("Checkpoint optimizer class differs from the current optimizer")
        if metadata["scheduler_class"] != _class_name(scheduler):
            raise ValueError("Checkpoint scheduler class differs from the current scheduler")
        rng_payload = torch.load(
            path / "rng" / f"rank_{rank:05d}.pt", map_location="cpu", weights_only=True)
        if (rng_payload["global_step"], rng_payload["world_size"], rng_payload["rank"]) != (
            expected_step, world_size, rank,
        ):
            raise ValueError("Per-rank RNG payload does not match checkpoint step/world size/rank")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_collective_error(error, "load ODE training state", process_group)

    error = None
    try:
        state = {"optimizer": get_optimizer_state_dict(model, optimizer)}
        parameter_groups = [group["params"] for group in state["optimizer"]["param_groups"]]
        if parameter_groups != metadata["optimizer_parameter_groups"]:
            raise ValueError("Checkpoint optimizer parameter groups differ from the current optimizer")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_collective_error(error, "prepare ODE optimizer restore", process_group)
    dcp.load(state, checkpoint_id=path / "optimizer", process_group=process_group)
    error = None
    try:
        set_optimizer_state_dict(model, optimizer, optim_state_dict=state["optimizer"])
        if not metadata["optimizer_initialized"]:
            optimizer.state.clear()
        scheduler.load_state_dict(metadata["scheduler"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_collective_error(error, "restore ODE optimizer and scheduler", process_group)
    return {"rng_state": rng_payload["rng_state"], "extra_state": metadata["extra_state"]}
