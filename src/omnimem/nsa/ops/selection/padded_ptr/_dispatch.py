"""Backend-neutral launch helpers for the pointer-table selection attention."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from omnimem.nsa.backend import get_backend, resolve_backend
from omnimem.nsa.ops._int_utils import compute_tile_n, padded_pow2

__all__ = ["launch_ptr_fwd", "launch_ptr_bwd", "build_ptr_table", "get_backend"]


def build_ptr_table(
    block_indices: torch.Tensor,
    block_base_ptrs_k: torch.Tensor,
    block_base_ptrs_v: torch.Tensor,
    n_blocks_per_head: torch.Tensor,
    batch_stride: int,
    blocks_per_chunk: int = 0,
    in_chunk_block_bytes: int = 0,
    backend: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    args = (block_indices, block_base_ptrs_k, block_base_ptrs_v, n_blocks_per_head,
            batch_stride, blocks_per_chunk, in_chunk_block_bytes)
    if resolve_backend(backend or get_backend()) == "cuda":
        from omnimem.nsa.cuda_backend import ops as cuda_ops
        return cuda_ops.build_ptr_table(*args)
    from .kernels.ptr_builder import build_ptr_table_triton
    return build_ptr_table_triton(*args)


def launch_ptr_fwd(
    q: torch.Tensor,
    ptr_table_k: torch.Tensor,
    ptr_table_v: torch.Tensor,
    block_size: int,
    group_size: int,
    stride_bn: int,
    stride_bd: int,
    softmax_scale: float,
    return_lse: bool,
    input_precision: str,
    owners: Sequence[torch.Tensor] = (),
    backend: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the pointer forward kernel.  Returns ``(out, lse_or_None)``."""
    backend = resolve_backend(backend or get_backend())
    B, M, H, D = q.shape
    G = group_size
    MG = M // G
    T = ptr_table_k.shape[-1]
    device = q.device
    # The grid is (B, MG, H): rows [MG*G, M) would never be written and `out` is uninitialised.
    assert M % G == 0, f"selection attention requires M ({M}) divisible by group_size ({G})"

    if backend == "cuda":
        from omnimem.nsa.cuda_backend import ops as cuda_ops

        out, lse = cuda_ops.sel_fwd_ptr(
            q, ptr_table_k, ptr_table_v, block_size, G, stride_bn, stride_bd,
            softmax_scale, input_precision, owners,
        )
    else:
        from .kernels.select_fwd import _sel_attn_fwd_padded_ptr_kernel

        out = torch.empty_like(q)
        lse = torch.empty(B, H, M, dtype=torch.float32, device=device) if return_lse else q
        DP = padded_pow2(D)
        PADDED_BLOCK_SIZE = padded_pow2(block_size)
        PADDED_GROUP_SIZE = padded_pow2(G)
        TILE_N = compute_tile_n(block_size)
        _sel_attn_fwd_padded_ptr_kernel[(B, MG, H)](
            q, ptr_table_k, ptr_table_v, out, lse,
            softmax_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            ptr_table_k.stride(0), ptr_table_k.stride(1),
            ptr_table_k.stride(2), ptr_table_k.stride(3),
            lse.stride(0) if return_lse else 0,
            lse.stride(1) if return_lse else 0,
            lse.stride(2) if return_lse else 0,
            stride_bn, stride_bd,
            H=H, M=M, D=D, T=T, DP=DP,
            BLOCK_SIZE=block_size,
            PADDED_BLOCK_SIZE=PADDED_BLOCK_SIZE,
            GROUP_SIZE=G,
            PADDED_GROUP_SIZE=PADDED_GROUP_SIZE,
            TILE_N=TILE_N,
            RETURN_LSE=return_lse,
            INPUT_PRECISION=input_precision,
        )
    return out, (lse if return_lse else None)


def launch_ptr_bwd(
    *,
    backend: str,
    compute_dkv: bool = True,
    owners: Sequence[torch.Tensor] = (),
    grad_buffers: Sequence[torch.Tensor] = (),
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the pointer backward; returns (dq, delta), dK/dV go to the fp32 scratch.

    ``kwargs`` are ``selection_attention_padded_ptr_bwd``'s; ``compute_dkv``, ``owners`` and
    ``grad_buffers`` are for the CUDA backend only.
    """
    if resolve_backend(backend or get_backend()) == "cuda":
        from omnimem.nsa.cuda_backend import ops as cuda_ops
        return cuda_ops.sel_bwd_ptr(**kwargs, compute_dkv=compute_dkv, owners=owners,
                                    grad_buffers=grad_buffers)
    from .kernels.select_bwd import selection_attention_padded_ptr_bwd
    return selection_attention_padded_ptr_bwd(**kwargs)
