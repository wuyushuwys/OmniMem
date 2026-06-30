# -*- coding: utf-8 -*-
# Top-K block selection for NSA-style sparse attention.
# Tiled kernel for S_padded <= TILED_BC_LIMIT, chunked fallback for larger S.
#
# TF mode (tf_mask=True): Q and K are laid out as [clean half | noisy half].
# Half boundaries are derived from runtime shapes. 4-case TF mask:
#   c2c (clean q -> clean k): block-causal within clean half
#   n2c (noisy q -> clean k): strict block-causal (k_rel < q_rel)
#   n2n (noisy q -> noisy k): same chunk only (k_rel == q_rel)
#   clean q -> noisy k:       forbidden
# AWE window/sink applies only to clean half so n2n is never excluded.

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from omnimem.nsa.ops.utils import _bitonic_merge


TILED_BC_LIMIT = 128


def _get_gpu_family() -> str:
    cap = torch.cuda.get_device_capability()
    if cap[0] == 9:
        return "hopper"
    elif cap[0] == 8 and cap[1] == 9:
        return "ada"
    return "unknown"


SCORES_CONFIGS_L40S = [
    triton.Config({'BT': 64, 'BC': 32}, num_warps=4, num_stages=3),
    triton.Config({'BT': 64, 'BC': 32}, num_warps=4, num_stages=2),
    triton.Config({'BT': 64, 'BC': 64}, num_warps=4, num_stages=2),
    triton.Config({'BT': 128, 'BC': 32}, num_warps=4, num_stages=3),
    triton.Config({'BT': 128, 'BC': 64}, num_warps=4, num_stages=2),
    triton.Config({'BT': 128, 'BC': 64}, num_warps=8, num_stages=2),
    triton.Config({'BT': 128, 'BC': 128}, num_warps=8, num_stages=2),
    triton.Config({'BT': 256, 'BC': 64}, num_warps=8, num_stages=2),
]
SCORES_CONFIGS_H100 = [
    triton.Config({'BT': 64, 'BC': 64}, num_warps=4, num_stages=4),
    triton.Config({'BT': 128, 'BC': 64}, num_warps=4, num_stages=4),
    triton.Config({'BT': 128, 'BC': 64}, num_warps=4, num_stages=3),
    triton.Config({'BT': 128, 'BC': 64}, num_warps=8, num_stages=3),
    triton.Config({'BT': 128, 'BC': 128}, num_warps=8, num_stages=3),
    triton.Config({'BT': 128, 'BC': 128}, num_warps=8, num_stages=2),
    triton.Config({'BT': 256, 'BC': 64}, num_warps=8, num_stages=3),
    triton.Config({'BT': 256, 'BC': 128}, num_warps=8, num_stages=2),
    triton.Config({'BT': 256, 'BC': 128}, num_warps=16, num_stages=2),
]
TILED_CONFIGS_H100 = [
    triton.Config({'BT': 32}, num_warps=2, num_stages=3),
    triton.Config({'BT': 32}, num_warps=4, num_stages=3),
    triton.Config({'BT': 64}, num_warps=2, num_stages=3),
    triton.Config({'BT': 64}, num_warps=4, num_stages=3),
    triton.Config({'BT': 64}, num_warps=4, num_stages=4),
    triton.Config({'BT': 128}, num_warps=4, num_stages=2),
    triton.Config({'BT': 128}, num_warps=4, num_stages=3),
    triton.Config({'BT': 128}, num_warps=8, num_stages=2),
    triton.Config({'BT': 128}, num_warps=8, num_stages=3),
    triton.Config({'BT': 256}, num_warps=8, num_stages=2),
]
TILED_CONFIGS_L40S = [
    triton.Config({'BT': 32}, num_warps=2, num_stages=2),
    triton.Config({'BT': 32}, num_warps=4, num_stages=2),
    triton.Config({'BT': 64}, num_warps=4, num_stages=2),
    triton.Config({'BT': 64}, num_warps=4, num_stages=3),
    triton.Config({'BT': 128}, num_warps=4, num_stages=2),
    triton.Config({'BT': 128}, num_warps=8, num_stages=2),
]


def get_scores_configs():
    fam = _get_gpu_family()
    if fam == "hopper":
        return SCORES_CONFIGS_H100
    elif fam == "ada":
        return SCORES_CONFIGS_L40S
    return SCORES_CONFIGS_L40S + SCORES_CONFIGS_H100


def get_tiled_configs():
    fam = _get_gpu_family()
    if fam == "hopper":
        return TILED_CONFIGS_H100
    elif fam == "ada":
        return TILED_CONFIGS_L40S
    return TILED_CONFIGS_L40S + TILED_CONFIGS_H100


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _ilog2(x: int) -> int:
    return int(math.log2(x))


@triton.autotune(
    configs=get_tiled_configs(),
    key=['TC', 'BS', 'BK', 'BC'],
)
@triton.jit
def parallel_nsa_kernel_topk_tiled(
        q, k, scale, block_indices,
        stride_qb, stride_qt, stride_qh, stride_qk,
        stride_kb, stride_kc, stride_kh, stride_kk,
        stride_ib, stride_it, stride_ih, stride_is,
        T, TC,
        H: tl.constexpr,
        K: tl.constexpr,
        S: tl.constexpr,
        BC: tl.constexpr,
        BS: tl.constexpr,
        BK: tl.constexpr,
        N_DIMS: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        CHUNK_SIZE_K: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        CS_POW2: tl.constexpr, CS_LOG2: tl.constexpr,
        CSK_POW2: tl.constexpr, CSK_LOG2: tl.constexpr,
        EX_WIN: tl.constexpr, EX_SINK: tl.constexpr,
        PROG_EXCL: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        TF_MASK: tl.constexpr,
        BT: tl.constexpr,
):
    i_bt = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    BPC: tl.constexpr = CHUNK_SIZE_K // BS

    q_half_chunks = T // (2 * CHUNK_SIZE)
    k_half_chunks = (TC * BS) // (2 * CHUNK_SIZE_K)

    # load Q tile: [BT, K]
    offs_t = i_bt * BT + tl.arange(0, BT)
    mask_t = offs_t < T

    p_q = tl.make_block_ptr(
        q + i_b * stride_qb + i_h * stride_qh,
        shape=(T, K),
        strides=(stride_qt, stride_qk),
        offsets=(i_bt * BT, 0),
        block_shape=(BT, BK),
        order=(1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)

    # top-K state: b_i = scores, o_i = 1-based block indices (BC slots for bitonic sort)
    b_i = tl.full([BT, BC], -1e6, dtype=tl.float32)
    o_i = tl.zeros([BT, BC], dtype=tl.int32)
    m_i = (tl.arange(0, BC) < BC // 2)[None, :]

    if CS_POW2:
        q_cid = offs_t >> CS_LOG2
    else:
        q_cid = offs_t // CHUNK_SIZE

    if TF_MASK:
        q_in_clean = q_cid < q_half_chunks
        q_rel = q_cid - tl.where(q_in_clean, 0, q_half_chunks)
    else:
        q_in_clean = q_cid >= 0  # placeholder
        q_rel = q_cid

    if PROG_EXCL:
        do_excl_row = q_rel >= (EX_SINK + EX_WIN)

    # Causal upper bound: take max over BT rows; per-row mask in inner loop enforces correctness.
    if IS_CAUSAL:
        if TF_MASK:
            if EX_WIN > 0:
                clean_q_ub = tl.maximum(0, q_rel - EX_WIN + 1)
            else:
                clean_q_ub = q_rel + 1
            noisy_q_ub = k_half_chunks + q_rel + 1   # n2n inclusive
            row_ub_kcid = tl.where(q_in_clean, clean_q_ub, noisy_q_ub)
            if PROG_EXCL and EX_WIN > 0:
                clean_q_ub_full = q_rel + 1
                row_ub_kcid = tl.where(
                    do_excl_row,
                    row_ub_kcid,
                    tl.where(q_in_clean, clean_q_ub_full, noisy_q_ub),
                )
            max_ub_kcid = tl.max(row_ub_kcid, axis=0)
            TC_ub = tl.minimum(max_ub_kcid * BPC, TC)
        else:
            if EX_WIN > 0:
                win_ub = tl.maximum(0, q_cid - EX_WIN + 1)
                if PROG_EXCL:
                    row_ub_chunk = tl.where(do_excl_row, win_ub, q_cid + 1)
                else:
                    row_ub_chunk = win_ub
            else:
                row_ub_chunk = q_cid + 1
            max_ub_chunk = tl.max(row_ub_chunk, axis=0)
            TC_ub = tl.minimum(max_ub_chunk * BPC, TC)
    else:
        TC_ub = TC

    # loop over K blocks; load, score, merge into top-K heap
    for i_c in range(0, TC_ub, BC):
        o_c = i_c + tl.arange(0, BC)

        # load K tile: [K, BC] (compressed key layout)
        p_k = tl.make_block_ptr(
            k + i_b * stride_kb + i_h * stride_kh,
            shape=(K, TC),
            strides=(stride_kk, stride_kc),
            offsets=(0, i_c),
            block_shape=(BK, BC),
            order=(0, 1),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)

        # QK scores: [BT, BC]
        b_s = tl.dot(b_q, b_k, allow_tf32=ALLOW_TF32) * scale

        if CSK_POW2:
            k_cid = (o_c * BS) >> CSK_LOG2
        else:
            k_cid = (o_c * BS) // CHUNK_SIZE_K

        valid = (o_c < TC)[None, :] & mask_t[:, None]

        if TF_MASK:
            k_in_clean = k_cid < k_half_chunks
            k_rel = k_cid - tl.where(k_in_clean, 0, k_half_chunks)

            c2c = q_in_clean[:, None]    & k_in_clean[None, :]    & (k_rel[None, :] <= q_rel[:, None])
            n2c = (~q_in_clean[:, None]) & k_in_clean[None, :]    & (k_rel[None, :] <  q_rel[:, None])
            n2n = (~q_in_clean[:, None]) & (~k_in_clean[None, :]) & (k_rel[None, :] == q_rel[:, None])
            valid = valid & (c2c | n2c | n2n)
        elif IS_CAUSAL:
            valid = valid & (q_cid[:, None] >= k_cid[None, :])

        if EX_SINK > 0:
            if TF_MASK:
                sink_ok = (~k_in_clean) | (k_rel >= EX_SINK)  # sink applies in clean half only; n2n untouched
                sink_ok = sink_ok[None, :]
            else:
                sink_ok = (k_cid >= EX_SINK)[None, :]
            if PROG_EXCL:
                valid = valid & (sink_ok | ~do_excl_row[:, None])
            else:
                valid = valid & sink_ok

        if EX_WIN > 0:
            if TF_MASK:
                # Window applies in clean half only; n2n always preserved.
                in_window = k_in_clean[None, :] & (
                    k_rel[None, :] >= (q_rel[:, None] - EX_WIN + 1)
                )
            else:
                in_window = k_cid[None, :] >= (q_cid[:, None] - EX_WIN + 1)
                if not IS_CAUSAL:
                    in_window = in_window & (k_cid[None, :] <= q_cid[:, None] + EX_WIN - 1)
            if PROG_EXCL:
                valid = valid & (~in_window | ~do_excl_row[:, None])
            else:
                valid = valid & ~in_window

        b_score = tl.where(valid, b_s, -1e6)
        new_idx = tl.where(valid, (o_c + 1)[None, :], 0)

        b_ip = b_i
        o_ip = o_i
        b_i = b_score
        o_i = new_idx

        for i in tl.static_range(1, N_DIMS):
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), i, 2, N_DIMS)

        if i_c != 0:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), N_DIMS, False, N_DIMS)
            b_i_new = tl.where(m_i, b_ip, b_i)
            o_i_new = tl.where(m_i, o_ip, o_i)
            b_i, o_i = _bitonic_merge(
                b_i_new, o_i_new.to(tl.int32), N_DIMS, True, N_DIMS
            )
        else:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), N_DIMS, True, N_DIMS)

    # epilogue: extract top-S from bitonic heap (convert 1-based to 0-based)
    m_top = (tl.arange(0, BC // S) == 0)[None, :, None]
    o_i_reshaped = tl.reshape(o_i - 1, [BT, BC // S, S])
    b_top = tl.sum(m_top * o_i_reshaped, axis=1)

    offs_s = tl.arange(0, S)
    idx_ptrs = (
        block_indices
        + i_b * stride_ib
        + offs_t[:, None] * stride_it
        + i_h * stride_ih
        + offs_s[None, :] * stride_is
    )
    store_mask = mask_t[:, None] & (offs_s[None, :] < S)
    tl.store(idx_ptrs, b_top.to(block_indices.dtype.element_ty), mask=store_mask)


def _topk_fused_tiled(
        q, k, S, scale, block_size, chunk_size, chunk_size_k, causal,
        allow_tf32=True,
        exclude_window_chunks=0, exclude_sink_chunks=0,
        progressive_exclude=False,
        tf_mask=False,
):
    B, T, H, K = q.shape
    TC = k.shape[1]
    BK = triton.next_power_of_2(K)
    BS = block_size

    S_padded = triton.next_power_of_2(S)
    BC = max(16, 2 * S_padded)
    assert _is_pow2(BC) and BC % S_padded == 0
    assert BC <= TILED_BC_LIMIT, (
        f"BC ({BC}) exceeds TILED_BC_LIMIT. Should have routed to chunked path."
    )
    N_DIMS = _ilog2(BC)

    cs_pow2 = _is_pow2(chunk_size)
    cs_log2 = _ilog2(chunk_size) if cs_pow2 else 0
    csk_pow2 = _is_pow2(chunk_size_k)
    csk_log2 = _ilog2(chunk_size_k) if csk_pow2 else 0

    block_indices = torch.full(
        (B, T, H, S_padded), -1, dtype=torch.int32, device=q.device
    )

    grid = lambda meta: (triton.cdiv(T, meta['BT']), B * H)
    parallel_nsa_kernel_topk_tiled[grid](
        q=q, k=k, scale=scale, block_indices=block_indices,
        stride_qb=q.stride(0), stride_qt=q.stride(1),
        stride_qh=q.stride(2), stride_qk=q.stride(3),
        stride_kb=k.stride(0), stride_kc=k.stride(1),
        stride_kh=k.stride(2), stride_kk=k.stride(3),
        stride_ib=block_indices.stride(0), stride_it=block_indices.stride(1),
        stride_ih=block_indices.stride(2), stride_is=block_indices.stride(3),
        T=T, TC=TC,
        H=H, K=K, S=S_padded,
        BC=BC, BS=BS, BK=BK, N_DIMS=N_DIMS,
        CHUNK_SIZE=chunk_size, CHUNK_SIZE_K=chunk_size_k,
        IS_CAUSAL=causal,
        CS_POW2=cs_pow2, CS_LOG2=cs_log2,
        CSK_POW2=csk_pow2, CSK_LOG2=csk_log2,
        EX_WIN=exclude_window_chunks,
        EX_SINK=exclude_sink_chunks,
        PROG_EXCL=progressive_exclude,
        ALLOW_TF32=allow_tf32,
        TF_MASK=tf_mask,
    )

    if block_indices.shape[-1] > S:
        block_indices = block_indices[..., :S].contiguous()
    return block_indices



@triton.autotune(configs=get_scores_configs(), key=['BK'])
@triton.jit
def compute_scores_sparse_kernel(
        q, k, scores,
        scale,
        T_TOTAL, T_OFFSET, T_CHUNK, TC,
        stride_qb, stride_qt, stride_qh, stride_qk,
        stride_kb, stride_kc, stride_kh, stride_kk,
        stride_sb, stride_st, stride_sh, stride_sc,
        H: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        BC: tl.constexpr,
        BS: tl.constexpr,
        BK: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        CHUNK_SIZE_K: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        CS_POW2: tl.constexpr,
        CS_LOG2: tl.constexpr,
        CSK_POW2: tl.constexpr,
        CSK_LOG2: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        EX_WIN: tl.constexpr,
        EX_SINK: tl.constexpr,
        PROG_EXCL: tl.constexpr,
        TF_MASK: tl.constexpr,
):
    i_bt, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    q_half_chunks = T_TOTAL // (2 * CHUNK_SIZE)
    k_half_chunks = (TC * BS) // (2 * CHUNK_SIZE_K)

    t_global = T_OFFSET + i_bt * BT

    p_q = tl.make_block_ptr(
        q + i_b * stride_qb + i_h * stride_qh,
        shape=(T_TOTAL, K),
        strides=(stride_qt, stride_qk),
        offsets=(t_global, 0),
        block_shape=(BT, BK), order=(1, 0)
    )
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)

    BPC: tl.constexpr = CHUNK_SIZE_K // BS

    t_last = tl.minimum(t_global + BT - 1, T_OFFSET + T_CHUNK - 1)
    if CS_POW2:
        q_chunk_first = t_global >> CS_LOG2
        q_chunk_last = t_last >> CS_LOG2
    else:
        q_chunk_first = t_global // CHUNK_SIZE
        q_chunk_last = t_last // CHUNK_SIZE

    if PROG_EXCL:
        if TF_MASK:
            q_first_in_clean = q_chunk_first < q_half_chunks
            q_first_rel = q_chunk_first - tl.where(q_first_in_clean, 0, q_half_chunks)
            do_excl_tile = q_first_rel >= (EX_SINK + EX_WIN)
        else:
            do_excl_tile = q_chunk_first >= (EX_SINK + EX_WIN)
    else:
        do_excl_tile = False

    if IS_CAUSAL:
        if TF_MASK:
            q_last_in_clean = q_chunk_last < q_half_chunks
            q_last_rel = q_chunk_last - tl.where(q_last_in_clean, 0, q_half_chunks)
            ub_chunk = k_half_chunks + q_last_rel + 1  # n2n inclusive
            TC_ub = tl.minimum(ub_chunk * BPC, TC)
            TC_fast_causal = 0   # per-row mask required, no fast loop
        else:
            if EX_WIN > 0 and (not PROG_EXCL or do_excl_tile):
                ub_chunk = tl.maximum(0, q_chunk_last - EX_WIN + 1)
            else:
                ub_chunk = q_chunk_last + 1
            TC_ub = tl.minimum(ub_chunk * BPC, TC)
            TC_fast_causal = tl.minimum((q_chunk_first + 1) * BPC, TC_ub)
    else:
        TC_ub = TC
        TC_fast_causal = TC

    if EX_SINK > 0 and (not PROG_EXCL or do_excl_tile):
        if TF_MASK:
            TC_fast_start = 0
        else:
            fast_lower = (EX_SINK * BPC // BC) * BC
            TC_fast_start = tl.minimum(fast_lower, TC_fast_causal)
    else:
        TC_fast_start = 0

    TC_fast_end = (TC_fast_causal // BC) * BC
    TC_fast_start = tl.minimum(TC_fast_start, TC_fast_end)

    # fast loop: no per-row mask (all blocks valid within causal/AWE bounds)
    for i_c in range(TC_fast_start, TC_fast_end, BC):
        p_k = tl.make_block_ptr(
            k + i_b * stride_kb + i_h * stride_kh,
            shape=(K, TC),
            strides=(stride_kk, stride_kc),
            offsets=(0, i_c), block_shape=(BK, BC), order=(0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_s = tl.dot(b_q, b_k, allow_tf32=ALLOW_TF32) * scale

        p_s = tl.make_block_ptr(
            scores + i_b * stride_sb + i_h * stride_sh,
            shape=(T_CHUNK, TC),
            strides=(stride_st, stride_sc),
            offsets=(i_bt * BT, i_c),
            block_shape=(BT, BC), order=(1, 0)
        )
        tl.store(p_s, b_s, boundary_check=(0, 1))

    # slow loop: full per-row causal / AWE / TF-mask check
    offs_t = t_global + tl.arange(0, BT)

    for i_c in range(TC_fast_end, TC_ub, BC):
        offs_c = i_c + tl.arange(0, BC)

        p_k = tl.make_block_ptr(
            k + i_b * stride_kb + i_h * stride_kh,
            shape=(K, TC),
            strides=(stride_kk, stride_kc),
            offsets=(0, i_c), block_shape=(BK, BC), order=(0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_s = tl.dot(b_q, b_k, allow_tf32=ALLOW_TF32) * scale

        valid = (offs_c[None, :] < TC) & (offs_t[:, None] < T_OFFSET + T_CHUNK)

        if CS_POW2:
            q_cid_r = offs_t[:, None] >> CS_LOG2
        else:
            q_cid_r = offs_t[:, None] // CHUNK_SIZE
        if CSK_POW2:
            k_cid_r = (offs_c[None, :] * BS) >> CSK_LOG2
        else:
            k_cid_r = (offs_c[None, :] * BS) // CHUNK_SIZE_K

        if TF_MASK:
            q_in_clean_r = q_cid_r < q_half_chunks
            q_rel_r = q_cid_r - tl.where(q_in_clean_r, 0, q_half_chunks)
            k_in_clean_r = k_cid_r < k_half_chunks
            k_rel_r = k_cid_r - tl.where(k_in_clean_r, 0, k_half_chunks)

            c2c = q_in_clean_r    & k_in_clean_r    & (k_rel_r <= q_rel_r)
            n2c = (~q_in_clean_r) & k_in_clean_r    & (k_rel_r <  q_rel_r)
            n2n = (~q_in_clean_r) & (~k_in_clean_r) & (k_rel_r == q_rel_r)
            valid = valid & (c2c | n2c | n2n)
        elif IS_CAUSAL:
            valid = valid & (q_cid_r >= k_cid_r)

        b_s = tl.where(valid, b_s, float('-inf'))

        p_s = tl.make_block_ptr(
            scores + i_b * stride_sb + i_h * stride_sh,
            shape=(T_CHUNK, TC),
            strides=(stride_st, stride_sc),
            offsets=(i_bt * BT, i_c),
            block_shape=(BT, BC), order=(1, 0)
        )
        tl.store(p_s, b_s, boundary_check=(0, 1))


def _topk_sparse_chunked(
        q, k, S, scale, block_size, chunk_size, chunk_size_k, causal,
        bt_chunk=512, allow_tf32=True,
        exclude_window_chunks=0, exclude_sink_chunks=0,
        progressive_exclude=False,
        tf_mask=False,
):
    B, TG, H, K = q.shape
    TC = k.shape[1]
    BK = triton.next_power_of_2(K)
    BS = block_size
    BPC = chunk_size_k // BS

    cs_pow2 = _is_pow2(chunk_size)
    cs_log2 = _ilog2(chunk_size) if cs_pow2 else 0
    csk_pow2 = _is_pow2(chunk_size_k)
    csk_log2 = _ilog2(chunk_size_k) if csk_pow2 else 0

    block_indices = torch.full((B, TG, H, S), -1, dtype=torch.int32, device=q.device)
    S_eff = min(S, TC)

    bt_buf = min(bt_chunk, TG)
    scores_buf = torch.empty(B, bt_buf, H, TC, dtype=torch.float32, device=q.device)

    # Python-side AWE only used in non-TF mode (TF AWE is in-kernel + below).
    need_py_mask = (not tf_mask) and (exclude_window_chunks > 0 or exclude_sink_chunks > 0)
    if need_py_mask:
        k_cids = torch.arange(TC, device=q.device) * BS // chunk_size_k

    excl_threshold = exclude_sink_chunks + exclude_window_chunks

    do_tf_awe = tf_mask and (exclude_window_chunks > 0 or exclude_sink_chunks > 0)
    if do_tf_awe:
        q_half_chunks_py = TG // (2 * chunk_size)
        k_half_chunks_py = (TC * BS) // (2 * chunk_size_k)
        k_cids_full = torch.arange(TC, device=q.device) * BS // chunk_size_k
        k_in_clean_full = k_cids_full < k_half_chunks_py
        k_rel_full = k_cids_full - torch.where(
            k_in_clean_full,
            k_cids_full.new_zeros(()),
            k_cids_full.new_full((), k_half_chunks_py),
        )

    for t_start in range(0, TG, bt_chunk):
        bt = min(bt_chunk, TG - t_start)
        scores_buf[:, :bt].fill_(float('-inf'))

        grid = lambda meta: (triton.cdiv(bt, meta['BT']), B * H)
        compute_scores_sparse_kernel[grid](
            q=q, k=k, scores=scores_buf,
            scale=scale,
            T_TOTAL=TG, T_OFFSET=t_start, T_CHUNK=bt, TC=TC,
            stride_qb=q.stride(0), stride_qt=q.stride(1),
            stride_qh=q.stride(2), stride_qk=q.stride(3),
            stride_kb=k.stride(0), stride_kc=k.stride(1),
            stride_kh=k.stride(2), stride_kk=k.stride(3),
            stride_sb=scores_buf.stride(0), stride_st=scores_buf.stride(1),
            stride_sh=scores_buf.stride(2), stride_sc=scores_buf.stride(3),
            H=H, K=K, BS=BS, BK=BK,
            CHUNK_SIZE=chunk_size, CHUNK_SIZE_K=chunk_size_k,
            IS_CAUSAL=causal,
            CS_POW2=cs_pow2, CS_LOG2=cs_log2,
            CSK_POW2=csk_pow2, CSK_LOG2=csk_log2,
            ALLOW_TF32=allow_tf32,
            EX_WIN=exclude_window_chunks,
            EX_SINK=exclude_sink_chunks,
            PROG_EXCL=progressive_exclude,
            TF_MASK=tf_mask,
        )

        scores_chunk = scores_buf[:, :bt]

        if need_py_mask:
            q_cids = torch.arange(t_start, t_start + bt, device=q.device) // chunk_size

            if progressive_exclude:
                apply_excl = q_cids >= excl_threshold

            if exclude_sink_chunks > 0:
                sink_end = exclude_sink_chunks * BPC
                if progressive_exclude:
                    scores_chunk[:, apply_excl, :, :sink_end] = float('-inf')
                else:
                    scores_chunk[:, :, :, :sink_end] = float('-inf')

            if exclude_window_chunks > 0:
                in_window = k_cids[None, :] >= (q_cids[:, None] - exclude_window_chunks + 1)
                if causal:
                    in_window = in_window & (k_cids[None, :] <= q_cids[:, None])
                else:
                    in_window = in_window & (
                        k_cids[None, :] <= q_cids[:, None] + exclude_window_chunks - 1
                    )
                if progressive_exclude:
                    in_window = in_window & apply_excl[:, None]
                scores_chunk.masked_fill_(in_window[None, :, None, :], float('-inf'))
        elif do_tf_awe:
            q_cids_t = torch.arange(t_start, t_start + bt, device=q.device) // chunk_size
            q_in_clean_t = q_cids_t < q_half_chunks_py
            q_rel_t = q_cids_t - torch.where(
                q_in_clean_t,
                q_cids_t.new_zeros(()),
                q_cids_t.new_full((), q_half_chunks_py),
            )

            if progressive_exclude:
                apply_excl = q_rel_t >= excl_threshold

            if exclude_sink_chunks > 0:
                sink_block_mask = k_in_clean_full & (k_rel_full < exclude_sink_chunks)
                if progressive_exclude:
                    eff_mask = apply_excl[:, None] & sink_block_mask[None, :]
                    scores_chunk.masked_fill_(eff_mask[None, :, None, :], float('-inf'))
                else:
                    scores_chunk[:, :, :, sink_block_mask] = float('-inf')

            if exclude_window_chunks > 0:
                in_window = k_in_clean_full[None, :] & (
                    k_rel_full[None, :] >= (q_rel_t[:, None] - exclude_window_chunks + 1)
                ) & (k_rel_full[None, :] <= q_rel_t[:, None])
                if progressive_exclude:
                    in_window = in_window & apply_excl[:, None]
                scores_chunk.masked_fill_(in_window[None, :, None, :], float('-inf'))

        topk_result = torch.topk(scores_chunk, S_eff, dim=-1)
        indices = topk_result.indices.to(torch.int32)
        indices[torch.isinf(topk_result.values) & (topk_result.values < 0)] = -1
        block_indices[:, t_start:t_start + bt, :, :S_eff] = indices

    return block_indices



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
        _chunk_size_k: Optional[int] = None,
) -> torch.Tensor:
    """Select top-K compressed-key block indices.

    group_size pools every G queries before selection (G | chunk_size required).
    tf_mask=True enforces the 4-case teacher-forcing mask (see module docstring).
    AWE (window/sink) is restricted to the clean half so n2n is never excluded.
    """
    B, T, H, K = q.shape
    G = group_size

    if G > 1:
        assert T % G == 0
        assert chunk_size % G == 0

        q_g = q.reshape(B, T // G, G, H, K).mean(dim=2)

        return parallel_nsa_topk(
            q_g, k, block_counts,
            block_size=block_size,
            scale=scale,
            chunk_size=chunk_size // G,
            causal=causal,
            bt_chunk=bt_chunk,
            allow_tf32=allow_tf32,
            group_size=1,
            exclude_window_chunks=exclude_window_chunks,
            exclude_sink_chunks=exclude_sink_chunks,
            progressive_exclude=progressive_exclude,
            tf_mask=tf_mask,
            _chunk_size_k=chunk_size,
        )

    chunk_size_k = _chunk_size_k if _chunk_size_k is not None else chunk_size

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
    S_padded = triton.next_power_of_2(S)
    sc = scale if scale is not None else K ** -0.5

    BC_tiled = max(16, 2 * S_padded)

    if BC_tiled <= TILED_BC_LIMIT:
        return _topk_fused_tiled(
            q, k, S,
            scale=sc,
            block_size=block_size,
            chunk_size=chunk_size,
            chunk_size_k=chunk_size_k,
            causal=causal,
            allow_tf32=allow_tf32,
            exclude_window_chunks=exclude_window_chunks,
            exclude_sink_chunks=exclude_sink_chunks,
            progressive_exclude=progressive_exclude,
            tf_mask=tf_mask,
        )

    return _topk_sparse_chunked(
        q, k, S, sc, block_size, chunk_size, chunk_size_k, causal,
        bt_chunk=bt_chunk, allow_tf32=allow_tf32,
        exclude_window_chunks=exclude_window_chunks,
        exclude_sink_chunks=exclude_sink_chunks,
        progressive_exclude=progressive_exclude,
        tf_mask=tf_mask,
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