"""Keeps a compiled request's output independent of what the process compiled before.

compiled_request() resets dynamo when its cache entries would not serve the request like a fresh process.
"""
from __future__ import annotations

import contextlib
import logging
import threading
from typing import Hashable, Iterable, Iterator, Optional, Tuple

import torch

from omnimem.utils.logging_tool import get_logger

logger = get_logger()

__all__ = ["RECOMPILE_LIMIT", "pin_compile_settings", "compiled_request",
           "filter_expected_dynamo_warnings"]

#: dynamo's per-frame recompile limit this code base is pinned to (the torch 2.9
#: default).  Another value moves the compiled / eager boundary of every long video.
RECOMPILE_LIMIT = 8

Position = Tuple[int, int]

_lock = threading.Lock()
_pinned = False
_signature: Optional[Hashable] = None       # of the requests compiled since the last reset
_positions: Tuple[Position, ...] = ()       # their distinct positions, in order of first use
_active = 0                                 # compiled requests running right now


def pin_compile_settings() -> None:
    """Pin the process-wide ``torch.compile`` settings the outputs depend on."""
    import torch._inductor.config as inductor_config

    global _pinned
    # int module attributes (layer_idx) go symbolic after the second value.
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    # pad_mm picks padded vs unpadded matmul by timing, and the two differ bitwise.
    inductor_config.shape_padding = False
    # See the module docstring: the limit is where a video's forwards stop being
    # compiled.  Pinned so that a changed torch default cannot change the videos.
    torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT
    _pinned = True


class _ExpectedDynamoWarnings(logging.Filter):
    """Drops the two warnings every compiled NSA DiT request triggers by design; others pass."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "causal_wan_nsa_model" not in message:
            return True
        return not ("Tensor.item()" in message or "recompile_limit" in message)


_expected_dynamo_warnings = _ExpectedDynamoWarnings()


def filter_expected_dynamo_warnings() -> None:
    """Keep the two expected dynamo warnings off the log (idempotent; for entry points)."""
    for name in ("torch._dynamo.variables.tensor", "torch._dynamo.convert_frame"):
        log = logging.getLogger(name)
        if _expected_dynamo_warnings not in log.filters:
            log.addFilter(_expected_dynamo_warnings)


def _first_distinct(positions: Iterable[Position]) -> Tuple[Position, ...]:
    """The first ``RECOMPILE_LIMIT`` distinct positions, in order of first use."""
    seen = []
    for pos in positions:
        pos = (int(pos[0]), int(pos[1]))
        if pos not in seen:
            seen.append(pos)
            if len(seen) == RECOMPILE_LIMIT:
                break
    return tuple(seen)


@contextlib.contextmanager
def compiled_request(signature: Hashable, positions: Iterable[Position]) -> Iterator[bool]:
    """Run one compiled request on the dynamo state a fresh process would give it."""
    global _signature, _positions, _active
    if not _pinned:
        yield False
        return
    mine = _first_distinct(positions)
    with _lock:
        shared = min(len(mine), len(_positions))
        compatible = signature == _signature and mine[:shared] == _positions[:shared]
        reset = _signature is not None and not compatible
        if reset:
            if _active:
                raise RuntimeError(
                    "this request needs a different torch.compile state than the "
                    "compiled request that is still running in this process (dynamo's "
                    "state is per process); run the two one after the other, or in "
                    "separate processes")
            what = "configuration" if signature != _signature else "frame positions"
            logger.info(f"[compile] what this process compiled before does not serve "
                        f"this request (other {what}): resetting torch.compile, the "
                        f"request recompiles")
            torch._dynamo.reset()
            pin_compile_settings()
        if reset or len(mine) > len(_positions):
            _positions = mine
        _signature = signature
        _active += 1
    try:
        yield reset
    finally:
        with _lock:
            _active -= 1
