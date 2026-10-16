"""Autograd wiring for the CUDA NSA backend (dense selection attention)."""
from __future__ import annotations

import torch

from . import ops as cuda_ops


class SelectionAttentionCUDA(torch.autograd.Function):
    """Drop-in replacement for the Triton SelectionAttention2p functions."""

    @staticmethod
    def forward(ctx, q, k, v, block_indices, block_size, chunk_size=1, softmax_scale=None,
                causal=False, return_lse=False, allow_tf32=True, bwd_method='auto',
                group_size=1):
        D = q.shape[-1]
        if softmax_scale is None:
            softmax_scale = 1.0 / (D ** 0.5)
        out, lse = cuda_ops.sel_fwd(
            q, k, v, block_indices, block_size, chunk_size, softmax_scale, causal,
            group_size, allow_tf32,
        )
        ctx.save_for_backward(q, k, v, block_indices, out, lse)
        ctx.block_size = block_size
        ctx.chunk_size = chunk_size
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.allow_tf32 = allow_tf32
        ctx.group_size = group_size
        if return_lse:
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        d_out = grad_outputs[0]
        q, k, v, block_indices, out, lse = ctx.saved_tensors
        dq, dk, dv = cuda_ops.sel_bwd(
            q, k, v, block_indices, out, lse, d_out, ctx.block_size, ctx.chunk_size,
            ctx.softmax_scale, ctx.causal, ctx.group_size, ctx.allow_tf32,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None
