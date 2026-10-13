"""Block-level KV cache kernels: K1 gather + publish + clear, K2 range register / clear."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from omnimem.nsa.backend import get_backend
from omnimem.nsa.ops._int_utils import cdiv

__all__ = [
    "register_block_range",
    "clear_block_range",
    "gather_publish_clear",
    "table_entry_addresses",
]

#: K1 clear-tile size; must match block_kv_kernels_triton._CLR_BLOCK and the CUDA kernel's tile.
_CLR_BLOCK = 256


def _check_table(table: torch.Tensor, n_valid: torch.Tensor):
    assert table.dtype == torch.int64 and table.is_cuda and table.is_contiguous(), \
        "table must be a contiguous int64 CUDA tensor"
    assert table.dim() == 3 and table.shape[0] == 2, \
        f"table must be [2, H, MB], got {tuple(table.shape)}"
    assert n_valid.dtype == torch.int32 and n_valid.is_cuda \
        and n_valid.shape == (table.shape[1],), \
        f"n_valid must be int32 [H={table.shape[1]}], got {n_valid.dtype} {tuple(n_valid.shape)}"


def _dispatch_block_range(backend: str, table: torch.Tensor, n_valid: torch.Tensor,
                          first_bid: int, n_blocks: int, k_base: int, v_base: int,
                          head_stride_bytes: int, block_bytes: int,
                          n_valid_value: int, write_k: bool, write_v: bool,
                          clear: bool) -> None:
    if backend == "triton":
        from .block_kv_kernels_triton import launch_block_range_triton
        launch_block_range_triton(
            table, n_valid, first_bid, n_blocks, k_base, v_base,
            head_stride_bytes, block_bytes, n_valid_value, write_k, write_v, clear)
    elif backend == "cuda":
        from omnimem.nsa.cuda_backend import block_kv_ops
        block_kv_ops.block_range(
            table, n_valid, first_bid, n_blocks, k_base, v_base,
            head_stride_bytes, block_bytes, n_valid_value, write_k, write_v, clear)
    else:
        raise ValueError(f"unknown block KV backend {backend!r}")


# K2: range register / clear
def register_block_range(table: torch.Tensor, n_valid: torch.Tensor,
                         first_bid: int, n_blocks: int,
                         k_base: int, v_base: int,
                         head_stride_bytes: int, block_bytes: int,
                         n_valid_value: int,
                         backend: Optional[str] = None) -> None:
    """table[name, h, first_bid + j] = base_name + h * head_stride + j * block_bytes; 0 skips."""
    _check_table(table, n_valid)
    _, H, MB = table.shape
    first_bid = int(first_bid)
    n_blocks = int(n_blocks)
    assert 0 <= first_bid and first_bid + n_blocks <= MB, \
        f"block range [{first_bid}, {first_bid + n_blocks}) outside [0, {MB})"
    if n_blocks <= 0:
        # Empty range still publishes n_valid (unless -1) so a stale larger n_valid cannot survive.
        if int(n_valid_value) >= 0:
            n_valid.fill_(int(n_valid_value))
        return
    assert int(k_base) != 0 or int(v_base) != 0, "nothing to register"
    backend = backend or get_backend()
    _dispatch_block_range(
        backend, table, n_valid, first_bid, n_blocks, int(k_base), int(v_base),
        int(head_stride_bytes), int(block_bytes), int(n_valid_value),
        int(k_base) != 0, int(v_base) != 0, False)


def clear_block_range(table: torch.Tensor, n_valid: torch.Tensor,
                      first_bid: int, n_blocks: int, n_valid_value: int,
                      which: str = "kv",
                      backend: Optional[str] = None) -> None:
    """Zero table[rows, :, first_bid:first_bid + n_blocks] and set n_valid (< 0 leaves it)."""
    _check_table(table, n_valid)
    _, H, MB = table.shape
    first_bid = int(first_bid)
    n_blocks = int(n_blocks)
    assert which in ("k", "v", "kv"), which
    assert 0 <= first_bid and first_bid + n_blocks <= MB, \
        f"block range [{first_bid}, {first_bid + n_blocks}) outside [0, {MB})"
    if n_blocks <= 0:
        # Still publish n_valid (e.g. free_last_chunk shrinking to a shorter length)
        # unless the caller passed the "leave unchanged" sentinel -1.
        if int(n_valid_value) >= 0:
            n_valid.fill_(int(n_valid_value))
        return
    backend = backend or get_backend()
    _dispatch_block_range(
        backend, table, n_valid, first_bid, n_blocks, 0, 0, 0, 0,
        int(n_valid_value), "k" in which, "v" in which, True)


def table_entry_addresses(table: torch.Tensor, name_idx, keys) -> "torch.Tensor":
    """Byte addresses of table[name_idx, idx // MB, idx % MB] for flat int64 keys h * MB + bid."""
    _, H, MB = table.shape
    if isinstance(keys, torch.Tensor):
        keys = keys.to(torch.int64)
    else:
        keys = np.asarray(keys, dtype=np.int64)
    # int64 arithmetic only: an int32 key tensor would wrap the byte address.
    return table.data_ptr() + (name_idx * H * MB + keys) * 8


# K1: fused gather + publish + clear
def gather_publish_clear(desc: torch.Tensor, n_fetch: int, n_clear: int,
                         block_tokens: int, head_dim: int, typed: torch.Tensor,
                         num_warps: int = 1, wide: Optional[bool] = None,
                         backend: Optional[str] = None) -> int:
    """Launch K1 over the int64 descriptor; needs pinned host sources and distinct dst ranges."""
    n_fetch = int(n_fetch)
    n_clear = int(n_clear)
    if n_fetch == 0 and n_clear == 0:
        return 0
    assert desc.dtype == torch.int64 and desc.is_cuda and desc.is_contiguous(), \
        "desc must be a contiguous int64 CUDA tensor"
    assert desc.numel() >= 6 * n_fetch + n_clear, (
        f"descriptor holds {desc.numel()} elements, need {6 * n_fetch + n_clear}")
    elem = typed.element_size()
    assert elem == 2, f"the gather kernel supports 16-bit KV dtypes only, got {typed.dtype}"
    block_bytes = int(block_tokens) * int(head_dim) * elem
    if wide is None:
        wide = block_bytes % 8 == 0 and (int(head_dim) * elem) % 8 == 0
    else:
        assert not wide or block_bytes % 8 == 0, (
            f"wide copy needs block_bytes ({block_bytes}) to be a multiple of 8")
    backend = backend or get_backend()
    if backend == "triton":
        from .block_kv_kernels_triton import launch_gather_publish_clear_triton
        launch_gather_publish_clear_triton(
            desc, typed, n_fetch, n_clear, int(block_tokens), int(head_dim),
            bool(wide), int(num_warps))
    elif backend == "cuda":
        from omnimem.nsa.cuda_backend import block_kv_ops
        block_kv_ops.gather_publish_clear(
            desc, n_fetch, n_clear, block_bytes, bool(wide), int(num_warps))
    else:
        raise ValueError(f"unknown block KV backend {backend!r}")
    return n_fetch + cdiv(n_clear, _CLR_BLOCK)


# Lazily re-exported so importing this facade never imports Triton.
_TRITON_REEXPORTS = ("_block_range_kernel", "_gather_publish_clear_kernel")


def __getattr__(name: str):
    if name in _TRITON_REEXPORTS:
        from . import block_kv_kernels_triton as _t
        value = getattr(_t, name)
        globals()[name] = value           # cache: only the first access pays the import
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
