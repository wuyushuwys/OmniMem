"""
Autograd wrappers for padded-ptr selection attention (training path).
K/V chunks are passed as *args so autograd routes dK/dV back to their source.
Invariant: k_chunks_flat[i].data_ptr() == chunk_base_ptrs_k[h, cid] for each (h, cid).
Training assumes all chunks are GPU-resident.
"""
from typing import Dict, Any, List, Optional, Tuple

import torch

from omnimem.nsa.backend import get_backend

from ._dispatch import launch_ptr_fwd, launch_ptr_bwd
from .ptr_builder import build_ptr_table
from .wrapper import (
    _compute_needed_bids_per_head,
    _build_block_base_ptrs_chunk_level,
    _find_sample_block,
)


def _build_scratch_base_ptrs(
    scratch_chunks: Tuple[torch.Tensor, ...],
    n_chunks_per_head_list: List[int],
    max_n_chunks: int,
    device: torch.device,
) -> torch.Tensor:
    """Build [H, max_n_chunks] int64 base-ptr table for fp32 dK/dV scratch."""
    H = len(n_chunks_per_head_list)
    base_cpu = torch.zeros(H, max_n_chunks, dtype=torch.int64)
    flat_idx = 0
    for h, nc in enumerate(n_chunks_per_head_list):
        for cid in range(nc):
            base_cpu[h, cid] = scratch_chunks[flat_idx].data_ptr()
            flat_idx += 1
    return base_cpu.to(device, non_blocking=True)


class _SelectionAttentionPaddedPtr(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,               # [B, M, H, D] bf16
        block_indices,   # [B, MG, H, T] int32
        n_chunks_per_head,   # [H] int32
        meta,                # dict: kernel params + strides + max_chunks
        *chunks,             # (k_flat..., v_flat...) head-major bf16 tensors
    ):
        block_size = meta['block_size']
        group_size = meta['group_size']
        chunk_strides = meta['chunk_strides']
        softmax_scale = meta['softmax_scale']
        input_precision = meta['input_precision']
        blocks_per_chunk = meta['blocks_per_chunk']

        B, M, H, D = q.shape
        G = group_size
        MG = M // G
        T = block_indices.shape[-1]

        device = q.device

        batch_stride_bytes = chunk_strides['batch_stride_bytes']
        token_stride_bytes = chunk_strides['token_stride_bytes']
        stride_bn = chunk_strides['stride_bn']
        stride_bd = chunk_strides['stride_bd']
        in_chunk_block_bytes = block_size * token_stride_bytes

        n_global_blocks = n_chunks_per_head * blocks_per_chunk

        # Base tables come from this call's chunks, not the cache, which later writes rewrite.
        n_chunks_per_head_list = meta['n_chunks_per_head_list']
        n_total_k = sum(n_chunks_per_head_list)
        max_n_chunks = meta['max_chunks']
        chunk_base_ptrs_k = _build_scratch_base_ptrs(
            chunks[:n_total_k], n_chunks_per_head_list, max_n_chunks, device,
        )
        chunk_base_ptrs_v = _build_scratch_base_ptrs(
            chunks[n_total_k:], n_chunks_per_head_list, max_n_chunks, device,
        )

        # Backend captured here, for the ptr build and the forward of this call, and
        # reused by backward (never re-read from the environment).
        backend = get_backend()
        ptr_table_k, ptr_table_v = build_ptr_table(
            block_indices=block_indices,
            block_base_ptrs_k=chunk_base_ptrs_k,
            block_base_ptrs_v=chunk_base_ptrs_v,
            n_blocks_per_head=n_global_blocks,
            batch_stride=batch_stride_bytes,
            blocks_per_chunk=blocks_per_chunk,
            in_chunk_block_bytes=in_chunk_block_bytes,
            backend=backend,
        )

        out, lse = launch_ptr_fwd(
            q, ptr_table_k, ptr_table_v,
            block_size=block_size, group_size=G,
            stride_bn=stride_bn, stride_bd=stride_bd,
            softmax_scale=softmax_scale, return_lse=True,
            input_precision=input_precision,
            owners=chunks, backend=backend,
        )

        # n_chunks_per_head is cloned to avoid autograd's version check failing (the cache
        # mutates it via clamp_min_ in update_cache).
        ctx.save_for_backward(
            q, out, lse, block_indices,
            ptr_table_k, ptr_table_v,
            chunk_base_ptrs_k, chunk_base_ptrs_v,
            n_chunks_per_head.clone(), n_global_blocks,
            *chunks,
        )
        ctx.meta = meta
        ctx.backend = backend

        return out

    @staticmethod
    def backward(ctx, d_out):
        saved = ctx.saved_tensors
        q               = saved[0]
        out             = saved[1]
        lse             = saved[2]
        block_indices   = saved[3]
        ptr_table_k     = saved[4]
        ptr_table_v     = saved[5]
        chunk_base_ptrs_k = saved[6]
        chunk_base_ptrs_v = saved[7]
        n_chunks_per_head = saved[8]
        n_global_blocks   = saved[9]
        chunks            = saved[10:]

        meta = ctx.meta
        block_size            = meta['block_size']
        group_size            = meta['group_size']
        chunk_strides         = meta['chunk_strides']
        softmax_scale         = meta['softmax_scale']
        input_precision       = meta['input_precision']
        blocks_per_chunk      = meta['blocks_per_chunk']
        n_chunks_per_head_list = meta['n_chunks_per_head_list']

        n_total_k = sum(n_chunks_per_head_list)
        k_chunks = chunks[:n_total_k]
        v_chunks = chunks[n_total_k:]

        device = q.device

        dk_scratch = tuple(torch.zeros_like(k, dtype=torch.float32) for k in k_chunks)
        dv_scratch = tuple(torch.zeros_like(v, dtype=torch.float32) for v in v_chunks)

        max_n_chunks = chunk_base_ptrs_k.shape[1]
        base_dk = _build_scratch_base_ptrs(dk_scratch, n_chunks_per_head_list, max_n_chunks, device)
        base_dv = _build_scratch_base_ptrs(dv_scratch, n_chunks_per_head_list, max_n_chunks, device)

        batch_stride_kv_bytes    = chunk_strides['batch_stride_bytes']
        token_stride_bytes       = chunk_strides['token_stride_bytes']
        in_chunk_block_bytes_kv  = block_size * token_stride_bytes
        stride_bn = chunk_strides['stride_bn']
        stride_bd = chunk_strides['stride_bd']

        dk_sample = dk_scratch[0]
        elem_fp32 = dk_sample.element_size()          # 4
        batch_stride_dkv_bytes   = dk_sample.stride(0) * elem_fp32
        token_stride_dkv_bytes   = dk_sample.stride(1) * elem_fp32
        in_chunk_block_bytes_dkv = block_size * token_stride_dkv_bytes
        stride_dbn = dk_sample.stride(1)   # elements
        stride_dbd = dk_sample.stride(2)   # elements

        dq, _ = launch_ptr_bwd(
            backend=ctx.backend,
            q=q, out=out, lse=lse, d_out=d_out,
            block_indices=block_indices,
            ptr_table_k=ptr_table_k, ptr_table_v=ptr_table_v,
            block_base_ptrs_k=chunk_base_ptrs_k,
            block_base_ptrs_v=chunk_base_ptrs_v,
            block_base_ptrs_dk=base_dk,
            block_base_ptrs_dv=base_dv,
            n_blocks_per_head=n_global_blocks,
            block_size=block_size,
            group_size=group_size,
            batch_stride_kv_bytes=batch_stride_kv_bytes,
            batch_stride_dkv_bytes=batch_stride_dkv_bytes,
            stride_bn=stride_bn,
            stride_bd=stride_bd,
            stride_dbn=stride_dbn,
            stride_dbd=stride_dbd,
            blocks_per_chunk=blocks_per_chunk,
            in_chunk_block_bytes_kv=in_chunk_block_bytes_kv,
            in_chunk_block_bytes_dkv=in_chunk_block_bytes_dkv,
            softmax_scale=softmax_scale,
            input_precision=input_precision,
            compute_dkv=True,
            owners=chunks,
            grad_buffers=tuple(dk_scratch) + tuple(dv_scratch),
        )

        dk_out = tuple(dk.to(k_chunks[i].dtype) for i, dk in enumerate(dk_scratch))
        dv_out = tuple(dv.to(v_chunks[i].dtype) for i, dv in enumerate(dv_scratch))

        return (dq, None, None, None, *dk_out, *dv_out)


def selection_attention_padded_ptr_train_fast(
    q: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    group_size: int,
    chunk_len: int,
    max_chunks: int,
    n_chunks_per_head: torch.Tensor,
    chunk_strides: Dict[str, Any],
    k_chunks_flat: Tuple[torch.Tensor, ...],
    v_chunks_flat: Tuple[torch.Tensor, ...],
    n_chunks_per_head_list: List[int],
    softmax_scale: Optional[float] = None,
    input_precision: str = "tf32",
) -> torch.Tensor:
    """Training forward; gradients flow to q and each k/v_chunks_flat tensor."""
    B, M, H, D = q.shape
    if softmax_scale is None:
        softmax_scale = D ** -0.5

    blocks_per_chunk = chunk_len // block_size

    meta = {
        'block_size':            block_size,
        'group_size':            group_size,
        'chunk_len':             chunk_len,
        'chunk_strides':         chunk_strides,
        'softmax_scale':         softmax_scale,
        'input_precision':       input_precision,
        'blocks_per_chunk':      blocks_per_chunk,
        'n_chunks_per_head_list': n_chunks_per_head_list,
        'max_chunks':            int(max_chunks),
    }

    return _SelectionAttentionPaddedPtr.apply(
        q, block_indices, n_chunks_per_head, meta,
        *k_chunks_flat, *v_chunks_flat,
    )


def selection_attention_padded_ptr_train(
    q: torch.Tensor,
    k_chunks: List[List[torch.Tensor]],
    v_chunks: List[List[torch.Tensor]],
    block_indices: torch.Tensor,
    block_size: int,
    group_size: int,
    chunk_len: int,
    softmax_scale: Optional[float] = None,
    input_precision: str = "tf32",
    strict: bool = True,
) -> torch.Tensor:
    """Slow training path: builds ptr tables from scratch each call."""
    H = len(k_chunks)
    device = q.device

    needed_bids_per_head = _compute_needed_bids_per_head(block_indices)

    result = _build_block_base_ptrs_chunk_level(
        k_chunks, v_chunks, needed_bids_per_head,
        chunk_len, block_size, device, strict=strict,
    )
    base_k, base_v, n_global_blocks, batch_stride, in_chunk_block_bytes, blocks_per_chunk = result

    assert base_k is not None, "no GPU-resident chunks found in k_chunks"

    sample = _find_sample_block(k_chunks, needed_bids_per_head, chunk_len, block_size)
    assert sample is not None and sample.is_contiguous()

    elem = sample.element_size()
    chunk_strides = {
        'batch_stride_bytes':  sample.stride(0) * elem,
        'token_stride_bytes':  sample.stride(1) * elem,
        'stride_bn':           sample.stride(1),
        'stride_bd':           sample.stride(2),
    }

    n_chunks_per_head_list = [len(k_chunks[h]) for h in range(H)]
    n_chunks_per_head = torch.tensor(n_chunks_per_head_list, dtype=torch.int32, device=device)

    k_flat = tuple(chunk for h in range(H) for chunk in k_chunks[h])  # head-major flatten
    v_flat = tuple(chunk for h in range(H) for chunk in v_chunks[h])

    return selection_attention_padded_ptr_train_fast(
        q=q,
        block_indices=block_indices,
        block_size=block_size,
        group_size=group_size,
        chunk_len=chunk_len,
        max_chunks=base_k.shape[1],
        n_chunks_per_head=n_chunks_per_head,
        chunk_strides=chunk_strides,
        k_chunks_flat=k_flat,
        v_chunks_flat=v_flat,
        n_chunks_per_head_list=n_chunks_per_head_list,
        softmax_scale=softmax_scale,
        input_precision=input_precision,
    )
