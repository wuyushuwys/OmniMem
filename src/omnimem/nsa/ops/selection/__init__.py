"""Dense selection attention, backend facade; Triton autograd functions are imported lazily."""
import math
import os

import torch

from omnimem.nsa.backend import get_backend

__all__ = ["selection_attention"]


def _debug_checks_enabled() -> bool:
    return os.environ.get("OMNIMEM_NSA_DEBUG", "0") not in ("", "0", "false", "False")


def _validate_common(q, k, v, block_indices, block_size, chunk_size, group_size):
    B, M, H, D = q.shape
    N = k.shape[1]
    G = group_size
    assert q.shape == (B, M, H, D)
    assert k.shape == (B, N, H, D), f"expected k {(B, N, H, D)}, got {tuple(k.shape)}"
    assert v.shape == (B, N, H, D), f"expected v {(B, N, H, D)}, got {tuple(v.shape)}"
    assert D >= 16, f"expected D >= 16, but got {D}"
    assert M % G == 0, f"M ({M}) must be divisible by group_size ({G})"
    MG = M // G
    assert block_indices.dim() == 4 and block_indices.shape[:3] == (B, MG, H), (
        f"Expected block_indices {(B, MG, H, '*')}, got {tuple(block_indices.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if block_indices.shape[-1] == 0:
        raise ValueError("top_idx last dim T must be > 0")
    num_blocks = math.ceil(N / block_size)
    # One host sync per call, matching the Triton facade's out-of-range check.
    valid_mask = block_indices >= 0
    if valid_mask.any() and torch.any(block_indices[valid_mask] >= num_blocks):
        raise ValueError(
            f"top_idx out of range: expected in [-1, {num_blocks - 1}] "
            f"but got [{block_indices.min()}, {block_indices.max()}]"
        )
    if _debug_checks_enabled():
        _debug_check_unique(block_indices)
    return num_blocks


def _debug_check_unique(block_indices):
    """Debug-only: valid ids within a (b, g, h) row must be unique."""
    ids = block_indices.to(torch.int64)
    sorted_ids, _ = ids.sort(dim=-1)
    dup = (sorted_ids[..., 1:] == sorted_ids[..., :-1]) & (sorted_ids[..., 1:] >= 0)
    if dup.any():
        raise ValueError("duplicate valid block ids within a row are not allowed (OMNIMEM_NSA_DEBUG)")


def selection_attention(
        q, k, v,
        block_indices,
        block_count,
        block_size,
        chunk_size=1,
        scale=None,
        variant='two-pass',
        causal=False,
        return_lse=False,
        allow_tf32=True,
        bwd_method='auto',
        group_size=1,
):
    if variant not in ('one-pass', 'two-pass'):
        raise ValueError(f"Invalid variant: {variant!r}, expected 'one-pass' or 'two-pass'")
    if group_size < 1:
        raise ValueError(f"Invalid group_size: {group_size}, expected >= 1")
    if variant == 'one-pass':
        raise NotImplementedError(f"{variant=} is not supported")
    if bwd_method not in ('auto', 'mask', 'inverted'):
        raise ValueError(f"Unknown bwd_method '{bwd_method}', expected 'auto', 'inverted' or 'mask'")

    backend = get_backend()
    if backend == 'triton':
        if group_size > 1:
            from .group.selection_two_pass import SelectionAttention2p as SelectionAttention2pGroup
            return SelectionAttention2pGroup.apply(
                q, k, v, block_indices, block_size, chunk_size,
                scale, causal, return_lse, allow_tf32, bwd_method, group_size,
            )
        from .token.selection_two_pass import SelectionAttention2p as SelectionAttention2pToken
        return SelectionAttention2pToken.apply(
            q, k, v, block_indices, block_size, chunk_size,
            scale, causal, return_lse, allow_tf32, bwd_method,
        )

    if backend == 'cuda':
        from omnimem.nsa.cuda_backend.autograd import SelectionAttentionCUDA
        _validate_common(q, k, v, block_indices, block_size, chunk_size, group_size)
        if causal and group_size > 1:
            assert chunk_size % group_size == 0, (
                f"causal + group requires chunk_size ({chunk_size}) "
                f"divisible by group_size ({group_size})"
            )
        if block_indices.dtype != torch.int32:
            block_indices = block_indices.to(torch.int32)
        return SelectionAttentionCUDA.apply(
            q, k, v, block_indices, block_size, chunk_size,
            scale, causal, return_lse, allow_tf32, bwd_method, group_size,
        )

    raise ValueError(f"unknown NSA backend {backend!r}")
