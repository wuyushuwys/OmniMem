"""Lazy loader of the block-KV cache CUDA extension (K1 gather, K2 range, K3 dedup)."""
from __future__ import annotations

from typing import Optional

from omnimem.nsa.backend import BlockKVBackendError
from ._extension import Extension, LazyExtension

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

_lazy = LazyExtension(EXTENSION, OP_NAMES)


def is_loaded() -> bool:
    return _lazy.is_loaded()


def library_path() -> Optional[str]:
    return _lazy.library_path


def load(verbose: bool = False):
    """Compile (if needed) and load the extension; idempotent. A failure is kept and re-raised."""
    return _lazy.load(verbose)


def ops():
    """The loaded pybind module (builds on demand)."""
    return load()
