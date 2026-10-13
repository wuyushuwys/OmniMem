"""NSA backend selection: OMNIMEM_NSA_BACKEND (triton or cuda) or set_backend()."""
from __future__ import annotations

import contextlib
import os
import threading
from typing import Iterator, Optional

VALID_BACKENDS = ("triton", "cuda")
ENV_VAR = "OMNIMEM_NSA_BACKEND"

_lock = threading.Lock()
_backend: Optional[str] = None


class NSABackendError(RuntimeError):
    """Raised when the requested NSA backend cannot serve a request."""


class BlockKVBackendError(NSABackendError):
    """Raised when the requested block-KV cache kernel backend cannot serve a request."""


def _normalize(name: str) -> str:
    name = (name or "").strip().lower()
    if name not in VALID_BACKENDS:
        raise NSABackendError(
            f"invalid NSA backend {name!r}; expected one of {VALID_BACKENDS} "
            f"(set via {ENV_VAR} or omnimem.nsa.backend.set_backend)"
        )
    return name


def get_backend() -> str:
    """Return the active backend name (``'triton'`` or ``'cuda'``)."""
    global _backend
    if _backend is None:
        with _lock:
            if _backend is None:
                _backend = _normalize(os.environ.get(ENV_VAR, "triton"))
    return _backend


def set_backend(name: str) -> str:
    """Explicitly select the backend for this process (CLIs, tests, benchmarks)."""
    global _backend
    name = _normalize(name)
    with _lock:
        _backend = name
    return name


@contextlib.contextmanager
def use_backend(name: str) -> Iterator[str]:
    """Temporarily switch the backend inside a ``with`` block."""
    prev = get_backend()
    set_backend(name)
    try:
        yield name
    finally:
        set_backend(prev)


def resolve_block_kv_backend(name: Optional[str] = None) -> str:
    """Backend of a block store's cache kernels; an explicit name wins over the active backend."""
    if name is None:
        return get_backend()
    name = str(name).strip().lower()
    if name not in VALID_BACKENDS:
        raise BlockKVBackendError(
            f"invalid block KV backend {name!r}; expected one of {VALID_BACKENDS}")
    return name
