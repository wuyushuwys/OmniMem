# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Optional
import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention, _DEFAULT_SPARSE_BLOCK_SIZE

flex_attention = torch.compile(flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs")

_BLOCK_SIZE = _DEFAULT_SPARSE_BLOCK_SIZE

_fa_import_errors = {}

try:
    import flash_attn.cute
    FLASH_ATTN_4_AVAILABLE = True
except ImportError as exc:
    FLASH_ATTN_4_AVAILABLE = False
    _fa_import_errors[4] = exc

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ImportError as exc:
    FLASH_ATTN_3_AVAILABLE = False
    _fa_import_errors[3] = exc

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ImportError as exc:
    FLASH_ATTN_2_AVAILABLE = False
    _fa_import_errors[2] = exc

# SM120 (e.g. RTX 5090): force-disable FlashAttention; kernels aren't supported.
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
    FLASH_ATTN_4_AVAILABLE = False
    FLASH_ATTN_3_AVAILABLE = False
    FLASH_ATTN_2_AVAILABLE = False

_fa_available = [
    v for v, ok in ((4, FLASH_ATTN_4_AVAILABLE),
                    (3, FLASH_ATTN_3_AVAILABLE),
                    (2, FLASH_ATTN_2_AVAILABLE)) if ok
]
_fa_missing = [v for v in (4, 3, 2) if v in _fa_import_errors]
if _fa_missing:
    _missing_str = ", ".join(f"FA{v}: {_fa_import_errors[v]}" for v in _fa_missing)
    _avail_str = ", ".join(f"FA{v}" for v in _fa_available) if _fa_available else "none"
    warnings.warn(
        f"FlashAttention import status — available: {_avail_str}; "
        f"unavailable: {_missing_str}. Falling back to remaining attention kernels."
    )

del _fa_import_errors, _fa_available, _fa_missing

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
):
    """
    FlashAttention dispatcher selecting FA4/FA3/FA2 based on availability.

    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      Dropout probability.
    softmax_scale:  Scaling of QK^T before softmax.
    causal:         Whether to apply causal attention mask.
    window_size:    (left, right). If not (-1, -1), apply sliding window attention.
    deterministic:  If True, slightly slower and uses more memory.
    dtype:          Applied when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes, f"{dtype=} is not float16/bfloat16"
    assert q.device.type == 'cuda' and q.size(-1) <= 256, f"{q.device.type=} == 'cuda' {q.size(-1)=} <= 256"

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if FLASH_ATTN_4_AVAILABLE:
        uniform_q = bool(torch.all(q_lens == q_lens[0]).item())
        uniform_k = bool(torch.all(k_lens == k_lens[0]).item())

        if uniform_q and uniform_k:
            q_p = q.unflatten(0, (b, lq))
            k_p = k.unflatten(0, (b, lk))
            v_p = v.unflatten(0, (b, lk))
            x = flash_attn.cute.flash_attn_func(
                q=q_p, k=k_p, v=v_p,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        else:
            x = flash_attn.cute.flash_attn_varlen_func(
                q=q, k=k, v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                seqused_q=None, seqused_k=None,
                max_seqlen_q=lq, max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            ).unflatten(0, (b, lq))
    elif (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # FA3 does not support dropout_p or window_size.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE, f"{FLASH_ATTN_2_AVAILABLE=}"
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        fa_version=None,
        block_mask: Optional[BlockMask] = None,
        return_lse=False,
):
    if isinstance(block_mask, BlockMask) or return_lse:

        if return_lse:
            out, lse = flex_attention(
                query=q.transpose(2, 1).to(dtype),
                key=k.transpose(2, 1).to(dtype),
                value=v.transpose(2, 1).to(dtype),
                block_mask=block_mask,
                return_lse=True
            )
            out = out.transpose(2, 1).contiguous()
            lse = lse.contiguous()
            return out, lse
        else:
            out = flex_attention(
                query=q.transpose(2, 1).to(dtype),
                key=k.transpose(2, 1).to(dtype),
                value=v.transpose(2, 1).to(dtype),
                block_mask=block_mask,
            )
            out = out.transpose(2, 1).contiguous()
            return out

    elif isinstance(block_mask, torch.Tensor):
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=block_mask,
            is_causal=causal,
            dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous()
        return out

    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE or FLASH_ATTN_4_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
