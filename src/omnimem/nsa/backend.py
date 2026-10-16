"""NSA backend selection: OMNIMEM_NSA_BACKEND (triton or cuda) or set_backend()."""
from __future__ import annotations

import os
import threading
from typing import Optional

VALID_BACKENDS = ("triton", "cuda")
ENV_VAR = "OMNIMEM_NSA_BACKEND"

_lock = threading.Lock()
_backend: Optional[str] = None


class NSABackendError(RuntimeError):
    """Raised when the requested NSA backend cannot serve a request."""


class BlockKVBackendError(NSABackendError):
    """Raised when the requested block-KV cache kernel backend cannot serve a request."""


def _normalize(name: str, error: type = NSABackendError, what: str = "invalid NSA backend",
               hint: str = f" (set via {ENV_VAR} or omnimem.nsa.backend.set_backend)") -> str:
    """The stripped, lower-cased name; raises ``error`` unless it is one of VALID_BACKENDS."""
    name = str(name or "").strip().lower()
    if name not in VALID_BACKENDS:
        raise error(f"{what} {name!r}; expected one of {VALID_BACKENDS}{hint}")
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


def resolve_backend(name: Optional[str] = None, error: type = ValueError,
                    what: str = "unknown NSA backend") -> str:
    """An explicit kernel backend name, checked; the active backend when it is None."""
    return get_backend() if name is None else _normalize(name, error, what, hint="")


def resolve_block_kv_backend(name: Optional[str] = None) -> str:
    """Backend of a block store's cache kernels; an explicit name wins over the active backend."""
    return resolve_backend(name, BlockKVBackendError, "invalid block KV backend")
