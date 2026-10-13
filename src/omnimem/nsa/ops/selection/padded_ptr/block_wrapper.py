"""Selection attention over the block-level KV store (kv_cache_policy=block_lru)."""
from __future__ import annotations

from typing import Optional

import torch

from omnimem.utils.stage_marks import profile_label

from omnimem.nsa.backend import get_backend

from ._dispatch import build_ptr_table_dispatch, launch_ptr_fwd

__all__ = ["selection_attention_block_store", "build_ptr_table"]


def build_ptr_table(*, block_indices, block_base_ptrs_k, block_base_ptrs_v,
                    n_blocks_per_head, batch_stride=0, blocks_per_chunk=0,
                    in_chunk_block_bytes=0, backend=None):
    """Block-level pointer build (keyword-only so tests can spy on it)."""
    return build_ptr_table_dispatch(
        block_indices, block_base_ptrs_k, block_base_ptrs_v, n_blocks_per_head,
        batch_stride, blocks_per_chunk, in_chunk_block_bytes, backend=backend)


def selection_attention_block_store(
    q: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    group_size: int,
    kv_cache,
    layer_idx: int,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    input_precision: str = "tf32",
):
    """One selection-attention call with the block store as the only address source."""
    B, M, H, D = q.shape
    G = group_size
    MG = M // G
    T = block_indices.shape[-1]

    assert B == 1, (
        f"the block KV store requires batch size 1, got B={B}: a pool slot holds one "
        f"batch element's block and the ptr table adds no batch stride")
    assert q.is_cuda and q.dtype == torch.bfloat16, \
        f"block-store selection attention needs bf16 CUDA queries, got {q.dtype}"
    assert M % G == 0, \
        f"selection attention requires M ({M}) divisible by group_size ({G})"
    assert block_indices.shape == (B, MG, H, T), \
        f"block_indices {tuple(block_indices.shape)} != {(B, MG, H, T)}"
    assert block_indices.dtype == torch.int32, \
        f"block_indices must be int32, got {block_indices.dtype}"

    # Inference only: release_selection hands a pool slot to the next call right after forward.
    if torch.is_grad_enabled():
        raise RuntimeError(
            "the block KV cache is inference-only: selection_attention_block_store "
            "was called with grad enabled.  Wrap the forward in torch.no_grad(), or use "
            "kv_cache_policy='chunk_lru' for training")
    # Capture the backend once per call so the ptr builder and the selection forward agree.
    backend = get_backend()

    store = kv_cache.block_store
    assert store is not None, "selection_attention_block_store needs a configured store"
    assert store.block_tokens == block_size and store.head_dim == D \
        and store.num_heads == H, (
        f"store geometry (bt={store.block_tokens}, D={store.head_dim}, "
        f"H={store.num_heads}) != ({block_size}, {D}, {H})")

    if softmax_scale is None:
        softmax_scale = D ** -0.5

    with profile_label("nsa.block_prepare"):
        lease = store.prepare_selection(layer_idx, block_indices)
    try:
        addr_k, addr_v, n_valid = store.table_views(layer_idx)
        ptr_table_k, ptr_table_v = build_ptr_table(
            block_indices=block_indices,
            block_base_ptrs_k=addr_k,
            block_base_ptrs_v=addr_v,
            n_blocks_per_head=n_valid,
            batch_stride=0,
            blocks_per_chunk=0,
            backend=backend,
        )
        # owners: bf16 storages the int64 addresses point into; CUDA validates them, Triton ignores.
        out, lse = launch_ptr_fwd(
            q, ptr_table_k, ptr_table_v,
            block_size=block_size, group_size=G,
            stride_bn=D, stride_bd=1,
            softmax_scale=softmax_scale, return_lse=return_lse,
            input_precision=input_precision,
            owners=(store.pool_k[layer_idx], store.pool_v[layer_idx],
                    store.scratch_k, store.scratch_v),
            backend=backend,
        )
    finally:
        store.release_selection(lease)

    if return_lse:
        return out, lse
    return out
