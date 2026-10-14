# -*- coding: utf-8 -*-
# Top-K block selection for NSA-style sparse attention; facade over the triton and cuda kernels.
# TF mode (tf_mask=True): Q and K are [clean half | noisy half]; n2n sees its own chunk only.

import math
from typing import Optional

import torch

from omnimem.nsa.backend import get_backend

__all__ = [
    "parallel_nsa_topk",
    "parallel_nsa_topk_grouped_heads",
    "should_use_selection_attention",
]


def _launch_topk(
        backend, q, k, S, scale, block_size, chunk_size, chunk_size_k, causal,
        bt_chunk, allow_tf32, exclude_window_chunks, exclude_sink_chunks,
        progressive_exclude, tf_mask, q_chunk_offset,
):
    if backend == "triton":
        from .index_topk_triton import topk_triton
        return topk_triton(
            q, k, S, scale, block_size, chunk_size, chunk_size_k, causal,
            bt_chunk=bt_chunk, allow_tf32=allow_tf32,
            exclude_window_chunks=exclude_window_chunks,
            exclude_sink_chunks=exclude_sink_chunks,
            progressive_exclude=progressive_exclude,
            tf_mask=tf_mask, q_chunk_offset=q_chunk_offset,
        )
    if backend == "cuda":
        from omnimem.nsa.cuda_backend import ops as cuda_ops
        return cuda_ops.topk(
            q, k, S, scale, block_size, chunk_size, chunk_size_k, causal,
            ex_win=exclude_window_chunks, ex_sink=exclude_sink_chunks,
            prog_excl=progressive_exclude, tf_mask=tf_mask,
            q_chunk_offset=q_chunk_offset, allow_tf32=allow_tf32,
        )
    raise ValueError(f"unknown NSA backend {backend!r}")


def parallel_nsa_topk(
        q: torch.Tensor,       # [B, T, H, K]
        k: torch.Tensor,       # [B, TC, H, K]
        block_counts: int,
        block_size: int = 64,
        scale: Optional[float] = None,
        chunk_size: int = 64,
        causal: bool = True,
        bt_chunk: int = 512,
        allow_tf32: bool = True,
        group_size: int = 1,
        exclude_window_chunks: int = 0,
        exclude_sink_chunks: int = 0,
        progressive_exclude: bool = False,
        tf_mask: bool = False,
        q_chunk_offset: int = 0,
) -> torch.Tensor:
    """Select top-K compressed-key block indices.

    group_size pools every G queries before selection (G | chunk_size required).
    tf_mask=True enforces the 4-case teacher-forcing mask (see module docstring).
    AWE (window/sink) is restricted to the clean half so n2n is never excluded.
    q_chunk_offset shifts query chunk ids to their global position (0 for full-sequence
    passes; set during chunk-by-chunk rollout where only one chunk is present).
    """
    assert not (tf_mask and q_chunk_offset), "tf_mask assumes a full [clean|noisy] layout; q_chunk_offset must be 0"
    B, T, H, K = q.shape
    G = group_size

    if G > 1:
        # Query sharing: mean-pool every G queries before scoring.
        assert T % G == 0
        assert chunk_size % G == 0
        q = q.reshape(B, T // G, G, H, K).mean(dim=2)
        chunk_size_k = chunk_size
        chunk_size = chunk_size // G
        T = T // G
    else:
        chunk_size_k = chunk_size

    if causal:
        assert chunk_size_k % block_size == 0

    if tf_mask:
        assert T % (2 * chunk_size) == 0, (
            f"tf_mask=True requires T ({T}) divisible by 2*chunk_size ({2*chunk_size}); "
            f"Q must be laid out as [clean | noisy] with equal halves."
        )
        TC_total = k.shape[1]
        kchunk_blocks = chunk_size_k // block_size  # K-side blocks per chunk
        assert TC_total % (2 * kchunk_blocks) == 0, (
            f"tf_mask=True requires TC ({TC_total}) divisible by 2 * (chunk_size_k/block_size) "
            f"({2*kchunk_blocks}); K layout must be [clean | noisy] with equal halves."
        )

    S = block_counts
    sc = scale if scale is not None else K ** -0.5

    return _launch_topk(
        get_backend(), q, k, S, sc, block_size, chunk_size, chunk_size_k, causal,
        bt_chunk, allow_tf32, exclude_window_chunks, exclude_sink_chunks,
        progressive_exclude, tf_mask, q_chunk_offset,
    )


def parallel_nsa_topk_grouped_heads(
    q: torch.Tensor,           # [B, T, H, K]
    k: torch.Tensor,           # [B, TC, H, K]
    num_kv_head_groups: int,
    block_counts: int,
    block_size: int,
    chunk_size: int,
    causal: bool = False,
    group_size: int = 1,
    tf_mask=False,
    **topk_kwargs,
) -> torch.Tensor:
    """GQA-style head pooling: pool heads per KV group, then broadcast block_indices."""
    B, T, H, K = q.shape
    TC = k.shape[1]
    assert H % num_kv_head_groups == 0
    if num_kv_head_groups == H:
        return parallel_nsa_topk(
            q=q,
            k=k,
            block_counts=block_counts,
            block_size=block_size,
            chunk_size=chunk_size,
            causal=causal,
            group_size=group_size,
            tf_mask=tf_mask,
            **topk_kwargs,
        )
    heads_per_group = H // num_kv_head_groups

    # Head pooling happens first (then query pooling inside parallel_nsa_topk); both keep the
    # input dtype's rounding.  Do not merge or reorder these two reductions.
    q_grouped = q.view(B, T, num_kv_head_groups, heads_per_group, K).mean(dim=3)
    k_grouped = k.view(B, TC, num_kv_head_groups, heads_per_group, K).mean(dim=3)

    block_indices = parallel_nsa_topk(
        q=q_grouped,
        k=k_grouped,
        block_counts=block_counts,
        block_size=block_size,
        chunk_size=chunk_size,
        causal=causal,
        group_size=group_size,
        tf_mask=tf_mask,
        **topk_kwargs,
    )

    block_indices = (
        block_indices
        .unsqueeze(-2)
        .expand(-1, -1, -1, heads_per_group, -1)
        .reshape(block_indices.shape[0], block_indices.shape[1], H, block_indices.shape[-1])
    )

    return block_indices


def should_use_selection_attention(
        seq_len, chunk_size, exclude_window_chunks, exclude_sink_chunks,
        causal=True,
):
    total_chunks = math.ceil(seq_len / chunk_size)
    if causal:
        max_visible = total_chunks
        excluded = min(exclude_sink_chunks, max_visible) + min(exclude_window_chunks, max_visible)
        return max_visible - excluded > 0
    else:
        excluded = exclude_window_chunks + exclude_sink_chunks
        return total_chunks - excluded > 0
