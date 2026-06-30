"""
reference: https://github.com/tilde-research/nsa-impl/blob/main/nsa/selection.py
Two-pass backward: dQ then dKV via 'inverted' (counting sort) or 'mask' (dense block_mask).
TILE_N / TILE_KV set by Python based on SELECTION_BLOCK_SIZE.
"""
from typing import Union
import math
import torch
import triton
import triton.language as tl
import triton.testing

from .select_fwd import _sel_attn_fwd_kernel, _compute_tile_n

# Device-adaptive autotune configs (TILE_N / TILE_KV set by caller)
_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
_sm_version = _capability[0] * 10 + _capability[1]

base_num_warps = 1

if _sm_version >= 90:
    _sel_attn_bwd_preprocess_configs = [
        triton.Config({'BLOCK_M': 32}, num_warps=base_num_warps, num_stages=1, num_ctas=1),
        triton.Config({'BLOCK_M': 32}, num_warps=base_num_warps, num_stages=2, num_ctas=1),
        triton.Config({'BLOCK_M': 64}, num_warps=base_num_warps, num_stages=2, num_ctas=1),
        triton.Config({'BLOCK_M': 64}, num_warps=base_num_warps * 2, num_stages=2, num_ctas=1),
    ]
    _sel_attn_bwd_dq_configs = [
        triton.Config({}, num_warps=base_num_warps * 2, num_stages=1),
        triton.Config({}, num_warps=base_num_warps * 2, num_stages=2),
        triton.Config({}, num_warps=base_num_warps * 4, num_stages=1),
        triton.Config({}, num_warps=base_num_warps * 4, num_stages=2),
    ]
    _sel_attn_bwd_dkv_mask_configs = [
        triton.Config({'BLOCK_Q': 16}, num_warps=base_num_warps * 2, num_stages=1),
        triton.Config({'BLOCK_Q': 32}, num_warps=base_num_warps * 2, num_stages=1),
        triton.Config({'BLOCK_Q': 32}, num_warps=base_num_warps * 4, num_stages=1),
        triton.Config({'BLOCK_Q': 64}, num_warps=base_num_warps * 4, num_stages=1),
    ]
    _sel_attn_bwd_dkv_inv_configs = [
        triton.Config({'BLOCK_Q': 16}, num_warps=base_num_warps * 2, num_stages=1),
        triton.Config({'BLOCK_Q': 32}, num_warps=base_num_warps * 2, num_stages=1),
        triton.Config({'BLOCK_Q': 32}, num_warps=base_num_warps * 4, num_stages=1),
        triton.Config({'BLOCK_Q': 64}, num_warps=base_num_warps * 4, num_stages=1),
    ]
else:
    _sel_attn_bwd_preprocess_configs = [
        triton.Config({'BLOCK_M': 32}, num_warps=base_num_warps * 2, num_stages=1, num_ctas=1),
        triton.Config({'BLOCK_M': 32}, num_warps=base_num_warps * 2, num_stages=2, num_ctas=1),
        triton.Config({'BLOCK_M': 64}, num_warps=base_num_warps * 2, num_stages=1, num_ctas=1),
        triton.Config({'BLOCK_M': 64}, num_warps=base_num_warps * 2, num_stages=2, num_ctas=1),
    ]
    _sel_attn_bwd_dq_configs = [
        triton.Config({}, num_warps=base_num_warps * 2, num_stages=1),
        triton.Config({}, num_warps=base_num_warps * 4, num_stages=1),
        triton.Config({}, num_warps=base_num_warps * 2, num_stages=2),
    ]
    _sel_attn_bwd_dkv_mask_configs = [
        triton.Config({'BLOCK_Q': 16}, num_warps=base_num_warps * 2, num_stages=1),
        triton.Config({'BLOCK_Q': 32}, num_warps=base_num_warps * 2, num_stages=1),
        triton.Config({'BLOCK_Q': 32}, num_warps=base_num_warps * 4, num_stages=1),
        triton.Config({'BLOCK_Q': 64}, num_warps=base_num_warps * 4, num_stages=1),
    ]
    _sel_attn_bwd_dkv_inv_configs = [
        triton.Config({'BLOCK_Q': 16}, num_warps=base_num_warps * 2, num_stages=1),
        triton.Config({'BLOCK_Q': 32}, num_warps=base_num_warps * 2, num_stages=1),
        triton.Config({'BLOCK_Q': 32}, num_warps=base_num_warps * 4, num_stages=1),
        triton.Config({'BLOCK_Q': 64}, num_warps=base_num_warps * 4, num_stages=1),
    ]



def build_block_mask(block_indices, num_kv_tokens, selection_block_size, chunk_size=1, causal=False):
    B, M, H, T = block_indices.shape
    NS = math.ceil(num_kv_tokens / selection_block_size)

    valid = (block_indices >= 0) & (block_indices < NS)
    safe_idx = torch.where(valid, block_indices, 0)
    block_mask = torch.zeros(B, M, H, NS, dtype=torch.bool, device=block_indices.device)
    block_mask.scatter_(3, safe_idx.long(), valid)

    if causal:
        kv_block_ids = torch.arange(NS, device=block_indices.device)
        q_chunks = torch.arange(M, device=block_indices.device) // chunk_size
        k_chunks = (kv_block_ids * selection_block_size) // chunk_size
        causal_mask = k_chunks[None, :] <= q_chunks[:, None]
        block_mask &= causal_mask[None, :, None, :]

    return block_mask


@triton.jit
def _inverted_index_histogram_kernel(
        TopIdx, Histogram,
        stride_tb, stride_tm, stride_th, stride_tt,
        stride_hb, stride_hh, stride_hs,
        M, NS,
        T: tl.constexpr, BLOCK_M: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        SELECTION_BLOCK_SIZE: tl.constexpr, CHUNK_SIZE: tl.constexpr,
):
    i_b = tl.program_id(0)
    i_h = tl.program_id(1)
    i_m_blk = tl.program_id(2)

    offs_m = i_m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    hist_base = Histogram + i_b * stride_hb + i_h * stride_hh

    for i_t in range(T):
        idx_ptrs = (TopIdx + i_b * stride_tb + offs_m * stride_tm
                    + i_h * stride_th + i_t * stride_tt)
        b_idx = tl.load(idx_ptrs, mask=mask_m, other=0).to(tl.int32)
        valid = mask_m & (b_idx >= 0) & (b_idx < NS)

        if IS_CAUSAL:
            k_chunk = (b_idx * SELECTION_BLOCK_SIZE) // CHUNK_SIZE
            q_chunk = offs_m // CHUNK_SIZE
            valid = valid & (k_chunk <= q_chunk)

        tl.atomic_add(hist_base + b_idx * stride_hs,
                      tl.where(valid, 1, 0), mask=valid)


@triton.jit
def _inverted_index_fused_kernel(
        TopIdx, WritePos, SortedQueries,
        stride_tb, stride_tm, stride_th, stride_tt,
        stride_wb, stride_wh, stride_ws,
        stride_sb, stride_sh,
        M, NS,
        T: tl.constexpr, BLOCK_M: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        SELECTION_BLOCK_SIZE: tl.constexpr, CHUNK_SIZE: tl.constexpr,
):
    i_b = tl.program_id(0)
    i_h = tl.program_id(1)
    i_m_blk = tl.program_id(2)

    offs_m = i_m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    wp_base = WritePos + i_b * stride_wb + i_h * stride_wh
    sq_base = SortedQueries + i_b * stride_sb + i_h * stride_sh

    for i_t in range(T):
        idx_ptrs = (TopIdx + i_b * stride_tb + offs_m * stride_tm
                    + i_h * stride_th + i_t * stride_tt)
        b_idx = tl.load(idx_ptrs, mask=mask_m, other=0).to(tl.int32)
        valid = mask_m & (b_idx >= 0) & (b_idx < NS)

        if IS_CAUSAL:
            k_chunk = (b_idx * SELECTION_BLOCK_SIZE) // CHUNK_SIZE
            q_chunk = offs_m // CHUNK_SIZE
            valid = valid & (k_chunk <= q_chunk)

        pos = tl.atomic_add(wp_base + b_idx * stride_ws,
                            tl.where(valid, 1, 0), mask=valid)
        tl.store(sq_base + pos, offs_m.to(tl.int32), mask=valid)


def build_inverted_index(top_idx, NS, selection_block_size=1, chunk_size=1, causal=False):
    B, M, H, T = top_idx.shape
    device = top_idx.device
    BLOCK_M = 256

    histogram = torch.zeros(B, H, NS, dtype=torch.int32, device=device)
    grid = (B, H, triton.cdiv(M, BLOCK_M))
    _inverted_index_histogram_kernel[grid](
        top_idx, histogram,
        top_idx.stride(0), top_idx.stride(1), top_idx.stride(2), top_idx.stride(3),
        histogram.stride(0), histogram.stride(1), histogram.stride(2),
        M, NS, T, BLOCK_M,
        IS_CAUSAL=causal,
        SELECTION_BLOCK_SIZE=selection_block_size,
        CHUNK_SIZE=chunk_size,
    )

    block_offsets = torch.zeros(B, H, NS + 1, dtype=torch.int32, device=device)
    block_offsets[:, :, 1:] = histogram.cumsum(dim=-1)

    write_pos = block_offsets[:, :, :NS].clone()
    sorted_queries = torch.empty(B, H, M * T, dtype=torch.int32, device=device)

    _inverted_index_fused_kernel[grid](
        top_idx, write_pos, sorted_queries,
        top_idx.stride(0), top_idx.stride(1), top_idx.stride(2), top_idx.stride(3),
        write_pos.stride(0), write_pos.stride(1), write_pos.stride(2),
        sorted_queries.stride(0), sorted_queries.stride(1),
        M, NS, T, BLOCK_M,
        IS_CAUSAL=causal,
        SELECTION_BLOCK_SIZE=selection_block_size,
        CHUNK_SIZE=chunk_size,
    )

    return sorted_queries, block_offsets


@triton.autotune(configs=_sel_attn_bwd_preprocess_configs, key=['M', 'D', 'H'])
@triton.jit
def _sel_attn_bwd_preprocess_kernel(
        Out, DOut, Delta,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_dob, stride_doh, stride_dom, stride_dod,
        stride_db, stride_dh, stride_dm,
        B: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
        D: tl.constexpr, DP: tl.constexpr, BLOCK_M: tl.constexpr,
):
    m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H

    # base pointers
    o_base = Out + b * stride_ob + h * stride_oh
    do_base = DOut + b * stride_dob + h * stride_doh

    offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, DP)
    mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)

    # load O, dO tiles: [BLOCK_M, D]
    o = tl.load(o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
                 mask=mask, other=0.0).to(tl.float32)

    # delta = rowsum(O * dO): [BLOCK_M]
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + b * stride_db + h * stride_dh + offs_m * stride_dm,
             delta, mask=offs_m < M)


# dQ kernel: query-major, no atomics
@triton.autotune(
    configs=_sel_attn_bwd_dq_configs,
    key=['M', 'N', 'D', 'SELECTION_BLOCK_SIZE', 'T', 'causal', 'TILE_N'],
)
@triton.jit
def _sel_attn_bwd_dq_kernel(
        Q, K, V, Top_idx, Lse, DOut, Delta, Order,
        softmax_scale: tl.constexpr, causal: tl.constexpr,
        DQ,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_tb, stride_th, stride_tm, stride_tt,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_lb, stride_lh, stride_lm,
        stride_rb, stride_rm, stride_rh,
        B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
        D: tl.constexpr, T: tl.constexpr, DP: tl.constexpr,
        SELECTION_BLOCK_SIZE: tl.constexpr,
        CHUNK_SIZE: tl.constexpr, OFFSET_M: tl.constexpr, ALLOW_TF32: tl.constexpr,
        GROUP_SIZE: tl.constexpr, PADDED_GROUP_SIZE: tl.constexpr,
        TILE_N: tl.constexpr,
):
    b = tl.program_id(0)
    g_physical = tl.program_id(1) + OFFSET_M
    h = tl.program_id(2)

    # Order: sorted group position maps to original group index
    r_ptr = Order + b * stride_rb + g_physical * stride_rm + h * stride_rh
    g = tl.load(r_ptr)
    m_start = g * GROUP_SIZE

    # base pointers
    q_base = Q + b * stride_qb + m_start * stride_qm + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    t_base = Top_idx + b * stride_tb + g * stride_tm + h * stride_th
    o_base = DOut + b * stride_ob + m_start * stride_om + h * stride_oh
    l_base = Lse + b * stride_lb + m_start * stride_lm + h * stride_lh
    d_base = Delta + b * stride_lb + m_start * stride_lm + h * stride_lh
    dq_base = DQ + b * stride_qb + m_start * stride_qm + h * stride_qh

    # offsets & masks
    offs_g = tl.arange(0, PADDED_GROUP_SIZE)
    mask_g = offs_g < GROUP_SIZE
    offs_d = tl.arange(0, DP)
    mask_d = offs_d < D

    # load Q, dO tiles: [GROUP_SIZE, D]
    q_blck = tl.load(q_base + offs_g[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                     mask=mask_g[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    do_blck = tl.load(o_base + offs_g[:, None] * stride_om + offs_d[None, :] * stride_od,
                      mask=mask_g[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # load LSE and delta: [GROUP_SIZE]
    l_blck = tl.load(l_base + offs_g * stride_lm, mask=mask_g, other=float('-inf'))
    d_blck = tl.load(d_base + offs_g * stride_lm, mask=mask_g, other=0.0)

    dq_accum = tl.zeros([PADDED_GROUP_SIZE, DP], dtype=tl.float32)
    log_scale = softmax_scale * 1.44269504
    q_chunk_idx = m_start // CHUNK_SIZE if causal else N

    NUM_TILES: tl.constexpr = (SELECTION_BLOCK_SIZE + TILE_N - 1) // TILE_N

    # loop over T selected blocks
    for idx in range(T):
        top = tl.load(t_base + idx * stride_tt)
        block_start = top * SELECTION_BLOCK_SIZE

        if block_start >= 0 and (not causal or block_start // CHUNK_SIZE <= q_chunk_idx):
            for tile_idx in range(NUM_TILES):
                tile_off = tile_idx * TILE_N
                tile_start = block_start + tile_off
                offs_tile = tl.arange(0, TILE_N)
                tile_cols = tile_start + offs_tile

                tile_valid = (tile_off + offs_tile < SELECTION_BLOCK_SIZE) & (tile_cols < N)

                # load K, V tiles: [TILE_N, D]
                p_k = tl.make_block_ptr(
                    base=k_base, shape=(N, D), strides=(stride_kn, stride_kd),
                    offsets=(tile_start, 0), block_shape=(TILE_N, DP), order=(1, 0)
                )
                k_tile = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)

                p_v = tl.make_block_ptr(
                    base=v_base, shape=(N, D), strides=(stride_vn, stride_vd),
                    offsets=(tile_start, 0), block_shape=(TILE_N, DP), order=(1, 0)
                )
                v_tile = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)

                # QK^T -> scores: [GROUP_SIZE, TILE_N]; recompute softmax weights
                qk = tl.dot(q_blck, tl.trans(k_tile),
                            allow_tf32=ALLOW_TF32) * log_scale

                if causal:
                    causal_mask = (tile_cols // CHUNK_SIZE) <= q_chunk_idx
                    kv_mask = causal_mask & tile_valid
                else:
                    kv_mask = tile_valid
                combined_mask = kv_mask[None, :] & mask_g[:, None]
                qk = tl.where(combined_mask, qk, -1.0e6)

                l2 = l_blck * 1.44269504
                exp_qk = tl.math.exp2(qk - l2[:, None])
                exp_qk = tl.where(combined_mask & (l_blck[:, None] > -1.0e6),
                                  exp_qk, 0.0)

                # ds = exp_qk * (dp - delta) * scale; dQ += ds @ K
                dp = tl.dot(do_blck, tl.trans(v_tile), allow_tf32=ALLOW_TF32)
                ds = exp_qk * (dp - d_blck[:, None]) * softmax_scale

                dq_accum = tl.dot(ds.to(q_blck.dtype), k_tile,
                                  acc=dq_accum, allow_tf32=ALLOW_TF32)

    # epilogue: write dQ
    dq_ptrs = dq_base + offs_g[:, None] * stride_qm + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, dq_accum, mask=mask_g[:, None] & mask_d[None, :])


# dKV kernels: grid = i_s * NUM_KV_TILES + kv_tile_idx

@triton.autotune(
    configs=_sel_attn_bwd_dkv_mask_configs,
    key=['M', 'D', 'SELECTION_BLOCK_SIZE', 'TILE_KV'],
)
@triton.jit
def _sel_attn_bwd_dkv_mask_kernel(
        Q, K, V, Lse, DOut, Delta,
        BlockMask,
        softmax_scale: tl.constexpr, causal: tl.constexpr,
        DK, DV,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_lb, stride_lh, stride_lm,
        stride_bmb, stride_bmm, stride_bmh, stride_bms,
        B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
        D: tl.constexpr, DP: tl.constexpr, NS: tl.constexpr,
        SELECTION_BLOCK_SIZE: tl.constexpr,
        CHUNK_SIZE: tl.constexpr, ALLOW_TF32: tl.constexpr,
        BLOCK_Q: tl.constexpr, TILE_KV: tl.constexpr,
):
    NUM_KV_TILES: tl.constexpr = (SELECTION_BLOCK_SIZE + TILE_KV - 1) // TILE_KV
    i_s_tile = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)

    i_s = i_s_tile // NUM_KV_TILES
    kv_tile_idx = i_s_tile % NUM_KV_TILES

    if i_s >= NS:
        return

    # KV tile columns
    col_start = i_s * SELECTION_BLOCK_SIZE + kv_tile_idx * TILE_KV
    offs_d = tl.arange(0, DP)
    mask_d = offs_d < D
    offs_kv = tl.arange(0, TILE_KV)
    tile_valid = (kv_tile_idx * TILE_KV + offs_kv < SELECTION_BLOCK_SIZE)
    cols = col_start + offs_kv
    mask_n = cols < N
    kv_mask = mask_n & tile_valid

    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh

    # load K, V tiles: [TILE_KV, D]
    k_blck = tl.load(k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                     mask=kv_mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    v_blck = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                     mask=kv_mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    b_dk = tl.zeros([TILE_KV, DP], dtype=tl.float32)
    b_dv = tl.zeros([TILE_KV, DP], dtype=tl.float32)
    log_scale = softmax_scale * 1.44269504

    # iterate over Q tiles; block_mask indexed by KV block (i_s), not tile
    bm_base = BlockMask + b * stride_bmb + h * stride_bmh + i_s * stride_bms

    for m_start in range(0, M, BLOCK_Q):
        offs_m = m_start + tl.arange(0, BLOCK_Q)
        mask_m = offs_m < M

        bm_ptrs = bm_base + offs_m * stride_bmm
        b_mask = tl.load(bm_ptrs, mask=mask_m, other=False)

        if tl.sum(b_mask.to(tl.int32)) > 0:
            # load Q, dO tiles: [BLOCK_Q, D]
            q_ptrs = (Q + b * stride_qb + h * stride_qh
                      + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
            q_blck = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :],
                             other=0.0).to(tl.float32)

            do_ptrs = (DOut + b * stride_ob + h * stride_oh
                       + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
            do_blck = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :],
                              other=0.0).to(tl.float32)

            # load LSE and delta: [BLOCK_Q]
            l_vals = tl.load(Lse + b * stride_lb + h * stride_lh + offs_m * stride_lm,
                             mask=mask_m, other=0.0)
            d_vals = tl.load(Delta + b * stride_lb + h * stride_lh + offs_m * stride_lm,
                             mask=mask_m, other=0.0)

            # QK^T -> scores: [BLOCK_Q, TILE_KV]; recompute softmax weights
            qk = tl.dot(q_blck, tl.trans(k_blck), allow_tf32=ALLOW_TF32) * log_scale

            base_mask = kv_mask[None, :]
            if causal:
                q_ci = offs_m // CHUNK_SIZE
                k_ci = cols // CHUNK_SIZE
                base_mask = base_mask & (k_ci[None, :] <= q_ci[:, None])
            qk = tl.where(base_mask, qk, -1.0e6)

            l2 = l_vals * 1.44269504
            exp_qk = tl.math.exp2(qk - l2[:, None])
            exp_qk = tl.where(b_mask[:, None] & mask_m[:, None] & (l_vals[:, None] > -1e6),
                              exp_qk, 0.0)

            # dV += exp_qk^T @ dO; dK += ds^T @ Q
            b_dv += tl.dot(tl.trans(exp_qk), do_blck, allow_tf32=ALLOW_TF32)

            dp = tl.dot(do_blck, tl.trans(v_blck), allow_tf32=ALLOW_TF32)
            ds = exp_qk * (dp - d_vals[:, None]) * softmax_scale
            b_dk += tl.dot(tl.trans(ds), q_blck, allow_tf32=ALLOW_TF32)

    # epilogue: write dK, dV
    write_mask = kv_mask[:, None] & mask_d[None, :]
    dk_base = DK + b * stride_kb + h * stride_kh
    dv_base = DV + b * stride_vb + h * stride_vh
    tl.store(dk_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd, b_dk, mask=write_mask)
    tl.store(dv_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd, b_dv, mask=write_mask)


@triton.autotune(
    configs=_sel_attn_bwd_dkv_inv_configs,
    key=['D', 'SELECTION_BLOCK_SIZE', 'TILE_KV'],
)
@triton.jit
def _sel_attn_bwd_dkv_inv_kernel(
        Q, K, V, Lse, DOut, Delta,
        SortedQueries, BlockOffsets,
        softmax_scale: tl.constexpr, causal: tl.constexpr,
        DK, DV,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_lb, stride_lh, stride_lm,
        stride_sqb, stride_sqh,
        stride_bob, stride_boh,
        B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
        D: tl.constexpr, DP: tl.constexpr, NS: tl.constexpr,
        SELECTION_BLOCK_SIZE: tl.constexpr,
        CHUNK_SIZE: tl.constexpr, ALLOW_TF32: tl.constexpr,
        BLOCK_Q: tl.constexpr, TILE_KV: tl.constexpr,
):
    NUM_KV_TILES: tl.constexpr = (SELECTION_BLOCK_SIZE + TILE_KV - 1) // TILE_KV
    i_s_tile = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)

    i_s = i_s_tile // NUM_KV_TILES
    kv_tile_idx = i_s_tile % NUM_KV_TILES

    if i_s >= NS:
        return

    # KV tile columns
    col_start = i_s * SELECTION_BLOCK_SIZE + kv_tile_idx * TILE_KV
    offs_d = tl.arange(0, DP)
    mask_d = offs_d < D
    offs_kv = tl.arange(0, TILE_KV)
    tile_valid = (kv_tile_idx * TILE_KV + offs_kv < SELECTION_BLOCK_SIZE)
    cols = col_start + offs_kv
    mask_n = cols < N
    kv_mask = mask_n & tile_valid

    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh

    # load K, V tiles: [TILE_KV, D]
    k_blck = tl.load(k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                     mask=kv_mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    v_blck = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                     mask=kv_mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    b_dk = tl.zeros([TILE_KV, DP], dtype=tl.float32)
    b_dv = tl.zeros([TILE_KV, DP], dtype=tl.float32)
    log_scale = softmax_scale * 1.44269504

    # CSR range per block (i_s), shared across tiles
    bo_ptr = BlockOffsets + b * stride_bob + h * stride_boh
    csr_start = tl.load(bo_ptr + i_s).to(tl.int32)
    csr_end = tl.load(bo_ptr + i_s + 1).to(tl.int32)
    count = csr_end - csr_start

    sq_base = SortedQueries + b * stride_sqb + h * stride_sqh + csr_start

    # loop over query tiles from inverted index
    num_tiles = tl.cdiv(count, BLOCK_Q)
    for i_tile in range(num_tiles):
        i_off = i_tile * BLOCK_Q
        offs_q = tl.arange(0, BLOCK_Q)
        mask_q = (i_off + offs_q) < count

        # load Q indices from sorted_queries
        m_indices = tl.load(sq_base + i_off + offs_q, mask=mask_q, other=0).to(tl.int32)

        # load Q, dO tiles: [BLOCK_Q, D]
        q_ptrs = (Q + b * stride_qb + h * stride_qh
                  + m_indices[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        q_blck = tl.load(q_ptrs, mask=mask_q[:, None] & mask_d[None, :],
                         other=0.0).to(tl.float32)

        do_ptrs = (DOut + b * stride_ob + h * stride_oh
                   + m_indices[:, None] * stride_om + offs_d[None, :] * stride_od)
        do_blck = tl.load(do_ptrs, mask=mask_q[:, None] & mask_d[None, :],
                          other=0.0).to(tl.float32)

        # load LSE and delta: [BLOCK_Q]
        l_vals = tl.load(Lse + b * stride_lb + h * stride_lh + m_indices * stride_lm,
                         mask=mask_q, other=0.0)
        d_vals = tl.load(Delta + b * stride_lb + h * stride_lh + m_indices * stride_lm,
                         mask=mask_q, other=0.0)

        # QK^T -> scores: [BLOCK_Q, TILE_KV]; recompute softmax weights
        qk = tl.dot(q_blck, tl.trans(k_blck), allow_tf32=ALLOW_TF32) * log_scale

        base_mask = kv_mask[None, :] & mask_q[:, None]
        if causal:
            q_ci = m_indices // CHUNK_SIZE
            k_ci = cols // CHUNK_SIZE
            base_mask = base_mask & (k_ci[None, :] <= q_ci[:, None])
        qk = tl.where(base_mask, qk, -1.0e6)

        l2 = l_vals * 1.44269504
        exp_qk = tl.math.exp2(qk - l2[:, None])
        exp_qk = tl.where(mask_q[:, None] & (l_vals[:, None] > -1e6), exp_qk, 0.0)

        # dV += exp_qk^T @ dO; dK += ds^T @ Q
        b_dv += tl.dot(tl.trans(exp_qk), do_blck, allow_tf32=ALLOW_TF32)

        dp = tl.dot(do_blck, tl.trans(v_blck), allow_tf32=ALLOW_TF32)
        ds = exp_qk * (dp - d_vals[:, None]) * softmax_scale
        b_dk += tl.dot(tl.trans(ds), q_blck, allow_tf32=ALLOW_TF32)

    # epilogue: write dK, dV
    write_mask = kv_mask[:, None] & mask_d[None, :]
    dk_base = DK + b * stride_kb + h * stride_kh
    dv_base = DV + b * stride_vb + h * stride_vh
    tl.store(dk_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd, b_dk, mask=write_mask)
    tl.store(dv_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd, b_dv, mask=write_mask)


class SelectionAttention2p(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            q, k, v, top_idx,
            selection_block_size,
            chunk_size=1,
            softmax_scale=None,
            causal=False,
            return_lse=False,
            allow_tf32=True,
            bwd_method='auto',
            group_size=1,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        B, M, H, D = q.shape
        _, N, _, _ = k.shape
        G = group_size
        MG = M // G
        _, _, _, T = top_idx.shape

        assert q.shape == (B, M, H, D)
        assert k.shape == (B, N, H, D)
        assert v.shape == (B, N, H, D)
        assert top_idx.shape == (B, MG, H, T), f"Expected top_idx {(B, MG, H, T)}, got {top_idx.shape}"
        assert D >= 16, f"expected D >= 16, but got {D}"
        assert M % G == 0, f"M ({M}) must be divisible by group_size ({G})"

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if causal and G > 1:
            assert chunk_size % G == 0, (
                f"causal + group requires chunk_size ({chunk_size}) "
                f"divisible by group_size ({G})"
            )
        if top_idx.shape[-1] == 0:
            raise ValueError("top_idx last dim T must be > 0")

        num_blocks = math.ceil(N / selection_block_size)
        valid_mask = top_idx >= 0
        if valid_mask.any() and torch.any(top_idx[valid_mask] >= num_blocks):
            raise ValueError(
                f"top_idx out of range: expected in [-1, {num_blocks - 1}] "
                f"but got [{top_idx.min()}, {top_idx.max()}]"
            )

        if softmax_scale is None:
            softmax_scale = 1.0 / (D ** 0.5)

        order = torch.argsort(top_idx[:, :, :, 0], dim=1).to(torch.int32)

        out = torch.zeros_like(q)
        lse = torch.full((B, H, M), float('-inf'), device=q.device, dtype=torch.float32)

        DP = triton.next_power_of_2(D)
        OFFSET_M = 0
        padded_group_size = max(16, triton.next_power_of_2(G))
        tile_n = _compute_tile_n(selection_block_size, max_tile=128)
        tile_kv = _compute_tile_n(selection_block_size, max_tile=64)

        grid = (B, MG - OFFSET_M, H)

        _sel_attn_fwd_kernel[grid](
            q, k, v, top_idx, order,
            softmax_scale, causal, out, lse,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            top_idx.stride(0), top_idx.stride(2), top_idx.stride(1), top_idx.stride(3),
            out.stride(0), out.stride(2), out.stride(1), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            order.stride(0), order.stride(1), order.stride(2),
            B, H, M, N, D, T, DP,
            SELECTION_BLOCK_SIZE=selection_block_size,
            CHUNK_SIZE=chunk_size,
            OFFSET_M=OFFSET_M,
            ALLOW_TF32=allow_tf32,
            GROUP_SIZE=G,
            PADDED_GROUP_SIZE=padded_group_size,
            TILE_N=tile_n,
        )

        if bwd_method == 'auto':
            ratio = T / num_blocks
            n = M * T
            use_inverted = ratio < 0.5 and (n > 200_000 or num_blocks > 256)
            bwd_method = 'inverted' if use_inverted else 'mask'

        ctx.save_for_backward(q, k, v, top_idx, out, lse, order)
        ctx.selection_block_size = selection_block_size
        ctx.chunk_size = chunk_size
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.allow_tf32 = allow_tf32
        ctx.bwd_method = bwd_method
        ctx.group_size = G
        ctx.padded_group_size = padded_group_size
        ctx.tile_n = tile_n
        ctx.tile_kv = tile_kv

        if return_lse:
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        d_out = grad_outputs[0]

        q, k, v, top_idx, out, lse, order = ctx.saved_tensors
        B, M, H, D = q.shape
        _, N, _, _ = k.shape
        _, _, _, T = top_idx.shape
        G = ctx.group_size
        MG = M // G

        selection_block_size = ctx.selection_block_size
        softmax_scale = ctx.softmax_scale
        chunk_size = ctx.chunk_size
        causal = ctx.causal
        allow_tf32 = ctx.allow_tf32
        bwd_method = ctx.bwd_method
        padded_group_size = ctx.padded_group_size
        tile_n = ctx.tile_n
        tile_kv = ctx.tile_kv

        DP = triton.next_power_of_2(D)
        OFFSET_M = 0
        NS = math.ceil(N / selection_block_size)

        delta = torch.empty_like(lse)

        def grid_preprocess(META):
            return (triton.cdiv(M, META['BLOCK_M']), B * H)

        _sel_attn_bwd_preprocess_kernel[grid_preprocess](
            out, d_out, delta,
            out.stride(0), out.stride(2), out.stride(1), out.stride(3),
            d_out.stride(0), d_out.stride(2), d_out.stride(1), d_out.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            B, H, M, D, DP,
        )

        dq = torch.empty_like(q, dtype=q.dtype)

        _sel_attn_bwd_dq_kernel[(B, MG - OFFSET_M, H)](
            q, k, v, top_idx, lse, d_out, delta, order,
            softmax_scale, causal, dq,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            top_idx.stride(0), top_idx.stride(2), top_idx.stride(1), top_idx.stride(3),
            out.stride(0), out.stride(2), out.stride(1), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            order.stride(0), order.stride(1), order.stride(2),
            B, H, M, N, D, T, DP,
            SELECTION_BLOCK_SIZE=selection_block_size,
            CHUNK_SIZE=chunk_size,
            OFFSET_M=OFFSET_M,
            ALLOW_TF32=allow_tf32,
            GROUP_SIZE=G,
            PADDED_GROUP_SIZE=padded_group_size,
            TILE_N=tile_n,
        )

        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        # Expand grouped top_idx [B, MG, H, T] to per-token [B, M, H, T]
        if G > 1:
            top_idx_expanded = top_idx.repeat_interleave(G, dim=1)
        else:
            top_idx_expanded = top_idx

        num_kv_tiles = math.ceil(selection_block_size / tile_kv)
        grid_dkv = (NS * num_kv_tiles, B, H)

        if bwd_method == 'inverted':
            sorted_queries, block_offsets = build_inverted_index(
                top_idx_expanded, NS,
                selection_block_size=selection_block_size,
                chunk_size=chunk_size,
                causal=causal,
            )

            _sel_attn_bwd_dkv_inv_kernel[grid_dkv](
                q, k, v, lse, d_out, delta,
                sorted_queries, block_offsets,
                softmax_scale, causal,
                dk, dv,
                q.stride(0), q.stride(2), q.stride(1), q.stride(3),
                k.stride(0), k.stride(2), k.stride(1), k.stride(3),
                v.stride(0), v.stride(2), v.stride(1), v.stride(3),
                out.stride(0), out.stride(2), out.stride(1), out.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                sorted_queries.stride(0), sorted_queries.stride(1),
                block_offsets.stride(0), block_offsets.stride(1),
                B, H, M, N, D, DP, NS,
                SELECTION_BLOCK_SIZE=selection_block_size,
                CHUNK_SIZE=chunk_size,
                ALLOW_TF32=allow_tf32,
                TILE_KV=tile_kv,
            )

        elif bwd_method == 'mask':
            block_mask = build_block_mask(
                top_idx_expanded, num_kv_tokens=N,
                selection_block_size=selection_block_size,
                chunk_size=chunk_size, causal=causal,
            )

            _sel_attn_bwd_dkv_mask_kernel[grid_dkv](
                q, k, v, lse, d_out, delta,
                block_mask,
                softmax_scale, causal,
                dk, dv,
                q.stride(0), q.stride(2), q.stride(1), q.stride(3),
                k.stride(0), k.stride(2), k.stride(1), k.stride(3),
                v.stride(0), v.stride(2), v.stride(1), v.stride(3),
                out.stride(0), out.stride(2), out.stride(1), out.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                block_mask.stride(0), block_mask.stride(1),
                block_mask.stride(2), block_mask.stride(3),
                B, H, M, N, D, DP, NS,
                SELECTION_BLOCK_SIZE=selection_block_size,
                CHUNK_SIZE=chunk_size,
                ALLOW_TF32=allow_tf32,
                TILE_KV=tile_kv,
            )
        else:
            raise ValueError(f"Unknown bwd_method '{bwd_method}', expected 'inverted' or 'mask'")

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None, None, None, None, None