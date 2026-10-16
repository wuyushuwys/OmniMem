"""Python adapters for the torch.ops.omnimem_nsa operators, plus fake versions for torch.compile."""
from __future__ import annotations

from typing import Sequence, Tuple

import torch

from omnimem.nsa.backend import NSABackendError
from . import _loader


def _ops():
    return _loader.ops()


def _call(name: str, *args):
    """Run the extension operator name; its RuntimeError becomes NSABackendError."""
    op = getattr(_ops(), name)
    try:
        return op(*args)
    except (torch.OutOfMemoryError, torch.AcceleratorError):
        raise
    except RuntimeError as e:
        raise NSABackendError(str(e)) from e


def _check_cuda(*ts: Tuple[torch.Tensor, str]) -> None:
    dev = None
    for t, name in ts:
        if not t.is_cuda:
            raise NSABackendError(f"{name} must be a CUDA tensor")
        if dev is None:
            dev = t.device
        elif t.device != dev:
            raise NSABackendError(f"{name} on {t.device} but expected {dev}")


_INT32_MAX = 2**31 - 1


def _check_int(name: str, v, lo: int = 0) -> int:
    """Kernel integer parameters are consumed as 32-bit; reject values outside [lo, INT32_MAX]."""
    v = int(v)
    if v < lo or v > _INT32_MAX:
        raise NSABackendError(f"{name}={v} out of the supported range [{lo}, {_INT32_MAX}]")
    return v


def _contig_meta(t: torch.Tensor) -> torch.Tensor:
    """Make small metadata tensors contiguous; the kernels read them with unit strides."""
    return t if t.is_contiguous() else t.contiguous()


# Top-K
def topk(
    q: torch.Tensor, k: torch.Tensor, topk: int, scale: float, block_size: int,
    chunk_size: int, chunk_size_k: int, causal: bool, ex_win: int = 0, ex_sink: int = 0,
    prog_excl: bool = False, tf_mask: bool = False, q_chunk_offset: int = 0,
    allow_tf32: bool = True,
) -> torch.Tensor:
    for name, v in (("block_size", block_size), ("chunk_size", chunk_size),
                    ("chunk_size_k", chunk_size_k), ("ex_win", ex_win), ("ex_sink", ex_sink),
                    ("q_chunk_offset", q_chunk_offset)):
        _check_int(name, v)
    return _call(
        "topk",
        q, k, int(topk), float(scale), int(block_size), int(chunk_size), int(chunk_size_k),
        bool(causal), int(ex_win), int(ex_sink), bool(prog_excl), bool(tf_mask),
        int(q_chunk_offset), bool(allow_tf32),
    )


# Pointer table
@torch.compiler.disable()
def build_ptr_table(
    block_indices: torch.Tensor, block_base_ptrs_k: torch.Tensor,
    block_base_ptrs_v: torch.Tensor, n_blocks_per_head: torch.Tensor, batch_stride: int,
    blocks_per_chunk: int = 0, in_chunk_block_bytes: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pointer-table build; raw int64 addresses make every pointer op an eager boundary."""
    return _call(
        "build_ptr_table",
        block_indices, _contig_meta(block_base_ptrs_k), _contig_meta(block_base_ptrs_v),
        _contig_meta(n_blocks_per_head), int(batch_stride), int(blocks_per_chunk),
        int(in_chunk_block_bytes),
    )


# Dense selection attention
def sel_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_indices: torch.Tensor,
    block_size: int, chunk_size: int, scale: float, causal: bool, group_size: int,
    allow_tf32: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_cuda((q, "q"), (k, "k"), (v, "v"), (block_indices, "block_indices"))
    _check_int("block_size", block_size, 1); _check_int("chunk_size", chunk_size, 1)
    return _call(
        "sel_fwd",
        q, k, v, block_indices, int(block_size), int(chunk_size), float(scale), bool(causal),
        int(group_size), bool(allow_tf32),
    )


def sel_bwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_indices: torch.Tensor,
    out: torch.Tensor, lse: torch.Tensor, d_out: torch.Tensor, block_size: int,
    chunk_size: int, scale: float, causal: bool, group_size: int, allow_tf32: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_cuda((q, "q"), (k, "k"), (v, "v"), (block_indices, "block_indices"),
                (out, "out"), (lse, "lse"), (d_out, "d_out"))
    if d_out.dtype != q.dtype:
        d_out = d_out.to(q.dtype)
    _check_int("block_size", block_size, 1); _check_int("chunk_size", chunk_size, 1)
    _check_int("group_size", group_size, 1)
    return _call(
        "sel_bwd",
        q, k, v, block_indices, out, lse, d_out, int(block_size), int(chunk_size),
        float(scale), bool(causal), int(group_size), bool(allow_tf32),
    )


# Pointer selection attention
@torch.compiler.disable()
def sel_fwd_ptr(
    q: torch.Tensor, ptr_table_k: torch.Tensor, ptr_table_v: torch.Tensor, block_size: int,
    group_size: int, stride_bn: int, stride_bd: int, softmax_scale: float,
    input_precision: str, owners: Sequence[torch.Tensor] = (),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pointer forward; eager boundary because K/V blocks are reached through raw addresses."""
    _check_cuda((q, "q"), (ptr_table_k, "ptr_table_k"), (ptr_table_v, "ptr_table_v"),
                *[(o, f"owners[{i}]") for i, o in enumerate(owners) if isinstance(o, torch.Tensor)])
    for o in owners:
        if isinstance(o, torch.Tensor) and o.dtype != torch.bfloat16:
            raise NSABackendError(f"pointer selection attention requires bf16 K/V storage, got {o.dtype}")
    _check_int("block_size", block_size, 1)
    _check_int("stride_bn", stride_bn); _check_int("stride_bd", stride_bd)
    return _call(
        "sel_fwd_ptr",
        q, ptr_table_k, ptr_table_v, int(block_size), int(group_size), int(stride_bn),
        int(stride_bd), float(softmax_scale), str(input_precision), list(owners),
    )


@torch.compiler.disable()
def sel_bwd_ptr(
    q: torch.Tensor, out: torch.Tensor, lse: torch.Tensor, d_out: torch.Tensor,
    block_indices: torch.Tensor, ptr_table_k: torch.Tensor, ptr_table_v: torch.Tensor,
    block_base_ptrs_k: torch.Tensor, block_base_ptrs_v: torch.Tensor,
    block_base_ptrs_dk: torch.Tensor, block_base_ptrs_dv: torch.Tensor,
    n_blocks_per_head: torch.Tensor, block_size: int, group_size: int,
    batch_stride_kv_bytes: int, batch_stride_dkv_bytes: int, stride_bn: int, stride_bd: int,
    stride_dbn: int, stride_dbd: int, blocks_per_chunk: int, in_chunk_block_bytes_kv: int,
    in_chunk_block_bytes_dkv: int, softmax_scale: float, input_precision: str,
    compute_dkv: bool, owners: Sequence[torch.Tensor] = (),
    grad_buffers: Sequence[torch.Tensor] = (),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pointer backward; arguments as in the Triton ``selection_attention_padded_ptr_bwd``."""
    _check_cuda((q, "q"), (out, "out"), (lse, "lse"), (d_out, "d_out"),
                (block_indices, "block_indices"), (ptr_table_k, "ptr_table_k"),
                (ptr_table_v, "ptr_table_v"), (block_base_ptrs_k, "block_base_ptrs_k"),
                (block_base_ptrs_v, "block_base_ptrs_v"), (block_base_ptrs_dk, "block_base_ptrs_dk"),
                (block_base_ptrs_dv, "block_base_ptrs_dv"), (n_blocks_per_head, "n_blocks_per_head"),
                *[(o, f"owners[{i}]") for i, o in enumerate(owners) if isinstance(o, torch.Tensor)],
                *[(g, f"grad_buffers[{i}]") for i, g in enumerate(grad_buffers) if isinstance(g, torch.Tensor)])
    if d_out.dtype != q.dtype:
        d_out = d_out.to(q.dtype)
    for g in grad_buffers:
        if isinstance(g, torch.Tensor) and g.dtype != torch.float32:
            raise NSABackendError(f"pointer dK/dV scratch must be float32, got {g.dtype}")
    for name, v in (("block_size", block_size), ("group_size", group_size), ("stride_bn", stride_bn),
                    ("stride_bd", stride_bd), ("stride_dbn", stride_dbn), ("stride_dbd", stride_dbd),
                    ("blocks_per_chunk", blocks_per_chunk)):
        _check_int(name, v)
    metas = map(_contig_meta, (block_base_ptrs_k, block_base_ptrs_v, block_base_ptrs_dk,
                               block_base_ptrs_dv, n_blocks_per_head))
    return _call(
        "sel_bwd_ptr",
        q, out, lse, d_out, block_indices, ptr_table_k, ptr_table_v, *metas,
        int(block_size), int(group_size), int(batch_stride_kv_bytes),
        int(batch_stride_dkv_bytes), int(stride_bn), int(stride_bd), int(stride_dbn),
        int(stride_dbd), int(blocks_per_chunk), int(in_chunk_block_bytes_kv),
        int(in_chunk_block_bytes_dkv), float(softmax_scale), str(input_precision),
        bool(compute_dkv), list(owners), list(grad_buffers),
    )


# Fake / meta implementations (shape-only)
_fakes_registered = False


def register_fakes() -> None:
    """Register meta kernels; called by the loader after the library is loaded."""
    global _fakes_registered
    if _fakes_registered:
        return
    _fakes_registered = True
    lib = torch.library

    @lib.register_fake("omnimem_nsa::topk")
    def _topk_fake(q, k, topk, scale, block_size, chunk_size, chunk_size_k, causal, ex_win,
                   ex_sink, prog_excl, tf_mask, q_chunk_offset, allow_tf32):
        B, MG, H, _ = q.shape
        return q.new_empty((B, MG, H, topk), dtype=torch.int32)

    @lib.register_fake("omnimem_nsa::build_ptr_table")
    def _ptr_fake(block_indices, base_k, base_v, n_blocks_per_head, batch_stride,
                  blocks_per_chunk, in_chunk_block_bytes):
        t = block_indices.new_empty(block_indices.shape, dtype=torch.int64)
        return t, block_indices.new_empty(block_indices.shape, dtype=torch.int64)

    @lib.register_fake("omnimem_nsa::sel_fwd")
    def _fwd_fake(q, k, v, block_indices, block_size, chunk_size, scale, causal, group_size,
                  allow_tf32):
        B, M, H, _ = q.shape
        return torch.empty_like(q), q.new_empty((B, H, M), dtype=torch.float32)

    @lib.register_fake("omnimem_nsa::sel_fwd_ptr")
    def _fwd_ptr_fake(q, ptr_k, ptr_v, block_size, group_size, stride_bn, stride_bd, scale,
                      input_precision, owners):
        B, M, H, _ = q.shape
        return torch.empty_like(q), q.new_empty((B, H, M), dtype=torch.float32)

    @lib.register_fake("omnimem_nsa::sel_bwd")
    def _bwd_fake(q, k, v, block_indices, out, lse, d_out, block_size, chunk_size, scale,
                  causal, group_size, allow_tf32):
        # dq preserves q's layout (at::empty_like); dk/dv are fresh contiguous buffers.
        return torch.empty_like(q), k.new_empty(k.shape), v.new_empty(v.shape)

    @lib.register_fake("omnimem_nsa::sel_bwd_ptr")
    def _bwd_ptr_fake(q, out, lse, d_out, block_indices, ptr_k, ptr_v, base_k, base_v, base_dk,
                      base_dv, n_blocks_per_head, block_size, group_size, batch_stride_kv_bytes,
                      batch_stride_dkv_bytes, stride_bn, stride_bd, stride_dbn, stride_dbd,
                      blocks_per_chunk, in_chunk_block_bytes_kv, in_chunk_block_bytes_dkv, scale,
                      input_precision, compute_dkv, owners, grad_buffers):
        return torch.empty_like(q), lse.new_empty(lse.shape)  # delta is a fresh contiguous buffer
