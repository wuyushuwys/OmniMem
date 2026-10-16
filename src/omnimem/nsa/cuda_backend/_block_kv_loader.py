"""Lazy loader of the block-KV cache CUDA extension (K1 gather, K2 range, K3 dedup)."""
from __future__ import annotations

import threading
from typing import Any, Optional

from omnimem.nsa.backend import BlockKVBackendError
from ._extension import Extension

#: The functions the module exports.
OP_NAMES = ("gather_publish_clear", "block_range", "dedup_run")

EXTENSION = Extension(
    prefix="omnimem_block_kv_cuda",
    what="block-KV CUDA extension",
    sources=("block_kv_bindings.cpp", "block_kv_gather.cu", "block_kv_range.cu",
             "block_kv_dedup.cu"),
    headers=("block_kv_ops.h", "block_kv_common.cuh", "warp_scan.cuh"),
    cuda_cflags=("-O3", "-std=c++17", "-lineinfo"),
    python_module=True,
    error=BlockKVBackendError,
)

_lock = threading.Lock()
_module: Optional[Any] = None
_load_error: Optional[Exception] = None
_library_path: Optional[str] = None


def is_loaded() -> bool:
    return _module is not None


def library_path() -> Optional[str]:
    return _library_path


def load(verbose: bool = False):
    """Compile (if needed) and load the extension; idempotent. A failure is kept and re-raised."""
    global _module, _load_error
    if _module is not None:
        return _module
    with _lock:
        if _module is not None:
            return _module
        if _load_error is not None:
            raise BlockKVBackendError(
                f"block-KV CUDA extension previously failed to load: {_load_error}"
            ) from _load_error
        try:
            mod = _do_load(verbose=verbose)
        except BlockKVBackendError:
            raise
        except Exception as e:  # noqa: BLE001 - surface build errors verbatim
            _load_error = e
            raise BlockKVBackendError(
                "failed to build/load the block-KV CUDA extension (no Triton fallback). "
                f"Original error: {type(e).__name__}: {e}") from e
        _module = mod
        return mod


def _do_load(verbose: bool):
    global _library_path
    mod, _library_path = EXTENSION.build_and_load(verbose=verbose)
    missing = [n for n in OP_NAMES if not hasattr(mod, n)]
    if missing:
        raise BlockKVBackendError(f"block-KV CUDA extension loaded but it lacks {missing}")
    return mod


def ops():
    """The loaded pybind module (builds on demand)."""
    return load()
