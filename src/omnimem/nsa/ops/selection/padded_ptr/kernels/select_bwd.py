"""
Two-pass backward for padded-ptr selection attention.

Pass 1: delta = rowsum(O * dO)  [B, H, M]
Pass 2: dQ  — query-major, walks T selected blocks via PtrTable, no atomics
Pass 3: build CSR inverted index from block_indices (expand to token-level via repeat_interleave(G))
Pass 4: dK/dV — KV-major, resolves bid → physical addr (chunk-level or block-level base),
         writes fp32 scratch directly (no atomics); caller zero-initializes and casts back to bf16

Strides: stride_bn/bd in ELEMENTS for K/V (bf16); stride_dbn/dbd for dK/dV (fp32).
Base == 0 means chunk is offloaded; kernel skips those blocks.
"""
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .select_fwd import _compute_tile_n


_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
_sm_version = _capability[0] * 10 + _capability[1]


if _sm_version >= 90:
    _bwd_preprocess_configs = [
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
    ]
    _bwd_dq_configs = [
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
    ]
    _bwd_dkv_configs = [
        triton.Config({'BLOCK_Q': 16}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_Q': 32}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_Q': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_Q': 64}, num_warps=4, num_stages=1),
    ]
else:
    _bwd_preprocess_configs = [
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=2),
    ]
    _bwd_dq_configs = [
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
    ]
    _bwd_dkv_configs = [
        triton.Config({'BLOCK_Q': 16}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_Q': 32}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_Q': 32}, num_warps=4, num_stages=1),
    ]


# preprocess: delta[m] = rowsum(O * dO)  [B, H, M]
@triton.autotune(configs=_bwd_preprocess_configs, key=['M', 'D', 'H'])
@triton.jit
def _sel_attn_bwd_preprocess_kernel(
    Out,              # [B, M, H, D] bf16
    DOut,             # [B, M, H, D] bf16
    Delta,            # [B, H, M] fp32 (out)
    stride_ob, stride_om, stride_oh, stride_od,
    stride_dob, stride_dom, stride_doh, stride_dod,
    stride_db, stride_dh, stride_dm,
    B: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
    D: tl.constexpr, DP: tl.constexpr,
    BLOCK_M: tl.constexpr,
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

    # load O and dO tiles: [BLOCK_M, D]
    o = tl.load(
        o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=mask, other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        mask=mask, other=0.0,
    ).to(tl.float32)

    # delta = rowsum(O * dO): [BLOCK_M]
    delta = tl.sum(o * do, axis=1)
    tl.store(
        Delta + b * stride_db + h * stride_dh + offs_m * stride_dm,
        delta, mask=offs_m < M,
    )


# dQ kernel: query-major, no atomics
@triton.autotune(
    configs=_bwd_dq_configs,
    key=['M', 'D', 'BLOCK_SIZE', 'T', 'GROUP_SIZE', 'TILE_N'],
)
@triton.jit
def _sel_attn_bwd_dq_padded_ptr_kernel(
    Q,                # [B, M, H, D] bf16
    PtrTableK,        # [B, MG, H, T] int64 (bf16 block addrs; 0 = skip)
    PtrTableV,        # [B, MG, H, T] int64
    Lse,              # [B, H, M] fp32
    DOut,             # [B, M, H, D] bf16
    Delta,            # [B, H, M] fp32
    DQ,               # [B, M, H, D] bf16 (out)
    softmax_scale: tl.constexpr,
    # Q / DOut / DQ strides — all share [B, M, H, D] layout
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_dob, stride_dom, stride_doh, stride_dod,
    stride_dqb, stride_dqm, stride_dqh, stride_dqd,
    # PtrTable strides
    stride_ptb, stride_ptg, stride_pth, stride_ptt,
    # Lse / Delta strides ([B, H, M])
    stride_lb, stride_lh, stride_lm,
    # Block-internal strides for K / V (in elements)
    stride_bn, stride_bd,
    H: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    DP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PADDED_BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PADDED_GROUP_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    h = tl.program_id(2)

    m_start = g * GROUP_SIZE

    # base pointers
    q_base = Q + b * stride_qb + m_start * stride_qm + h * stride_qh
    do_base = DOut + b * stride_dob + m_start * stride_dom + h * stride_doh
    dq_base = DQ + b * stride_dqb + m_start * stride_dqm + h * stride_dqh

    pt_base_k = PtrTableK + b * stride_ptb + g * stride_ptg + h * stride_pth
    pt_base_v = PtrTableV + b * stride_ptb + g * stride_ptg + h * stride_pth

    l_base = Lse + b * stride_lb + h * stride_lh + m_start * stride_lm
    d_base = Delta + b * stride_lb + h * stride_lh + m_start * stride_lm

    # offsets & masks
    offs_g = tl.arange(0, PADDED_GROUP_SIZE)
    mask_g = offs_g < GROUP_SIZE
    offs_d = tl.arange(0, DP)
    mask_d = offs_d < D

    # load Q, dO tiles: [GROUP_SIZE, D]
    q_blck = tl.load(
        q_base + offs_g[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=mask_g[:, None] & mask_d[None, :], other=0.0,
    ).to(tl.float32)
    do_blck = tl.load(
        do_base + offs_g[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        mask=mask_g[:, None] & mask_d[None, :], other=0.0,
    ).to(tl.float32)

    # load LSE and delta: [GROUP_SIZE]
    l_blck = tl.load(l_base + offs_g * stride_lm, mask=mask_g, other=float('-inf'))
    d_blck = tl.load(d_base + offs_g * stride_lm, mask=mask_g, other=0.0)

    dq_accum = tl.zeros([PADDED_GROUP_SIZE, DP], dtype=tl.float32)
    log_scale = softmax_scale * 1.44269504

    NUM_TILES: tl.constexpr = (BLOCK_SIZE + TILE_N - 1) // TILE_N

    # loop over T selected blocks via ptr table
    for t_idx in range(T):
        k_block_ptr = tl.load(pt_base_k + t_idx * stride_ptt)
        v_block_ptr = tl.load(pt_base_v + t_idx * stride_ptt)

        if k_block_ptr != 0:  # 0 = invalid sentinel
            k_blk = k_block_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
            v_blk = v_block_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)

            for tile_idx in range(NUM_TILES):
                tile_off = tile_idx * TILE_N
                offs_tile = tl.arange(0, TILE_N)
                tile_valid = (tile_off + offs_tile) < BLOCK_SIZE

                # load K, V tiles: [TILE_N, D]
                k_tile = tl.load(
                    k_blk + (tile_off + offs_tile)[:, None] * stride_bn
                    + offs_d[None, :] * stride_bd,
                    mask=tile_valid[:, None] & mask_d[None, :], other=0.0,
                ).to(tl.float32)
                v_tile = tl.load(
                    v_blk + (tile_off + offs_tile)[:, None] * stride_bn
                    + offs_d[None, :] * stride_bd,
                    mask=tile_valid[:, None] & mask_d[None, :], other=0.0,
                ).to(tl.float32)

                # QK^T -> scores: [GROUP_SIZE, TILE_N]
                qk = tl.dot(
                    q_blck, tl.trans(k_tile),
                    input_precision=INPUT_PRECISION,
                ) * log_scale

                combined_mask = tile_valid[None, :] & mask_g[:, None]
                qk = tl.where(combined_mask, qk, -1.0e6)

                # recompute softmax weights from LSE
                l2 = l_blck * 1.44269504
                exp_qk = tl.math.exp2(qk - l2[:, None])
                exp_qk = tl.where(
                    combined_mask & (l_blck[:, None] > -1.0e6),
                    exp_qk, 0.0,
                )

                # dp = dO @ V^T; ds = exp_qk * (dp - delta) * scale
                dp = tl.dot(
                    do_blck, tl.trans(v_tile),
                    input_precision=INPUT_PRECISION,
                )
                ds = exp_qk * (dp - d_blck[:, None]) * softmax_scale

                # dQ += ds @ K
                dq_accum = tl.dot(
                    ds.to(q_blck.dtype), k_tile, acc=dq_accum,
                    input_precision=INPUT_PRECISION,
                )

    # epilogue: write dQ
    dq_ptrs = dq_base + offs_g[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_accum, mask=mask_g[:, None] & mask_d[None, :])


@triton.jit
def _inverted_index_histogram_kernel(
    TopIdx,           # [B, M, H, T] int32 (token-level, after group expansion)
    NBlocksPerHead,   # [H] int32
    Histogram,        # [B, H, MAX_N] int32 (zero-initialized)
    stride_tb, stride_tm, stride_th, stride_tt,
    stride_hb, stride_hh, stride_hs,
    M,
    T: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    i_b = tl.program_id(0)
    i_h = tl.program_id(1)
    i_m_blk = tl.program_id(2)

    offs_m = i_m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    hist_base = Histogram + i_b * stride_hb + i_h * stride_hh

    n_h = tl.load(NBlocksPerHead + i_h)

    for i_t in range(T):
        idx_ptrs = (TopIdx + i_b * stride_tb + offs_m * stride_tm
                    + i_h * stride_th + i_t * stride_tt)
        b_idx = tl.load(idx_ptrs, mask=mask_m, other=-1).to(tl.int32)
        valid = mask_m & (b_idx >= 0) & (b_idx < n_h)

        tl.atomic_add(
            hist_base + b_idx * stride_hs,
            tl.where(valid, 1, 0), mask=valid,
        )


@triton.jit
def _inverted_index_fused_kernel(
    TopIdx,           # [B, M, H, T] int32
    NBlocksPerHead,   # [H] int32
    WritePos,         # [B, H, MAX_N] int32 (initialized to block_offsets[:, :, :MAX_N])
    SortedQueries,    # [B, H, M * T] int32 (out)
    stride_tb, stride_tm, stride_th, stride_tt,
    stride_wb, stride_wh, stride_ws,
    stride_sb, stride_sh,
    M,
    T: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    i_b = tl.program_id(0)
    i_h = tl.program_id(1)
    i_m_blk = tl.program_id(2)

    offs_m = i_m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    wp_base = WritePos + i_b * stride_wb + i_h * stride_wh
    sq_base = SortedQueries + i_b * stride_sb + i_h * stride_sh

    n_h = tl.load(NBlocksPerHead + i_h)

    for i_t in range(T):
        idx_ptrs = (TopIdx + i_b * stride_tb + offs_m * stride_tm
                    + i_h * stride_th + i_t * stride_tt)
        b_idx = tl.load(idx_ptrs, mask=mask_m, other=-1).to(tl.int32)
        valid = mask_m & (b_idx >= 0) & (b_idx < n_h)

        pos = tl.atomic_add(
            wp_base + b_idx * stride_ws,
            tl.where(valid, 1, 0), mask=valid,
        )
        tl.store(sq_base + pos, offs_m.to(tl.int32), mask=valid)


def build_inverted_index_padded_ptr(
    top_idx_expanded: torch.Tensor,    # [B, M, H, T] int32 (token-level)
    n_blocks_per_head: torch.Tensor,   # [H] int32
    max_n: int,
):
    """Build CSR inverted index (counting sort). Returns (sorted_queries, block_offsets).

    block_offsets [B, H, max_n+1] int32 — cumulative count per bid
    sorted_queries [B, H, M*T] int32   — token indices grouped by bid
    """
    B, M, H, T = top_idx_expanded.shape
    device = top_idx_expanded.device
    BLOCK_M = 256

    # pass 1: count queries per KV block → histogram
    histogram = torch.zeros(B, H, max_n, dtype=torch.int32, device=device)
    grid = (B, H, triton.cdiv(M, BLOCK_M))
    _inverted_index_histogram_kernel[grid](
        top_idx_expanded, n_blocks_per_head, histogram,
        top_idx_expanded.stride(0), top_idx_expanded.stride(1),
        top_idx_expanded.stride(2), top_idx_expanded.stride(3),
        histogram.stride(0), histogram.stride(1), histogram.stride(2),
        M, T, BLOCK_M,
    )

    # exclusive prefix sum → CSR offsets
    block_offsets = torch.zeros(B, H, max_n + 1, dtype=torch.int32, device=device)
    block_offsets[:, :, 1:] = histogram.cumsum(dim=-1)

    # pass 2: scatter query indices into CSR buckets
    write_pos = block_offsets[:, :, :max_n].clone()  # atomic write counter per block
    sorted_queries = torch.empty(B, H, M * T, dtype=torch.int32, device=device)

    _inverted_index_fused_kernel[grid](
        top_idx_expanded, n_blocks_per_head, write_pos, sorted_queries,
        top_idx_expanded.stride(0), top_idx_expanded.stride(1),
        top_idx_expanded.stride(2), top_idx_expanded.stride(3),
        write_pos.stride(0), write_pos.stride(1), write_pos.stride(2),
        sorted_queries.stride(0), sorted_queries.stride(1),
        M, T, BLOCK_M,
    )

    return sorted_queries, block_offsets


@triton.autotune(
    configs=_bwd_dkv_configs,
    key=['D', 'BLOCK_SIZE', 'TILE_KV', 'IS_CHUNK_BASE'],
)
@triton.jit
def _sel_attn_bwd_dkv_padded_ptr_kernel(
    Q,                # [B, M, H, D] bf16
    BlockBasePtrsK,   # [H, MAX_BASE] int64
    BlockBasePtrsV,
    BlockBasePtrsDK,  # [H, MAX_BASE] int64 (fp32 dK/dV base addrs)
    BlockBasePtrsDV,
    NBlocksPerHead,   # [H] int32 (global-bid bound)
    Lse,              # [B, H, M] fp32
    DOut,             # [B, M, H, D] bf16
    Delta,            # [B, H, M] fp32
    SortedQueries,    # [B, H, M * T] int32
    BlockOffsets,     # [B, H, MAX_N + 1] int32
    softmax_scale: tl.constexpr,
    # Q / DOut strides ([B, M, H, D])
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_dob, stride_dom, stride_doh, stride_dod,
    # BlockBasePtrs strides ([H, MAX_BASE]; same for K, V, DK, DV)
    stride_bbph, stride_bbpb,
    # Lse / Delta strides ([B, H, M])
    stride_lb, stride_lh, stride_lm,
    # SortedQueries strides ([B, H, M*T])
    stride_sqb, stride_sqh,
    # BlockOffsets strides ([B, H, MAX_N+1])
    stride_bob, stride_boh,
    # Block-internal strides — IN ELEMENTS
    stride_bn, stride_bd,        # K / V (bf16)
    stride_dbn, stride_dbd,      # dK / dV (fp32)
    # Byte strides for ptr arithmetic
    batch_stride_kv_bytes: tl.constexpr,
    batch_stride_dkv_bytes: tl.constexpr,
    in_chunk_block_bytes_kv: tl.constexpr,
    in_chunk_block_bytes_dkv: tl.constexpr,
    BLOCKS_PER_CHUNK: tl.constexpr,
    IS_CHUNK_BASE: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PADDED_BLOCK_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    TILE_KV: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    NUM_KV_TILES: tl.constexpr = (BLOCK_SIZE + TILE_KV - 1) // TILE_KV

    i_s_tile = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)

    bid = i_s_tile // NUM_KV_TILES
    kv_tile_idx = i_s_tile % NUM_KV_TILES

    n_h = tl.load(NBlocksPerHead + h)
    if bid >= n_h:
        return

    # resolve K/V/dK/dV base addresses for this (h, bid)
    if IS_CHUNK_BASE:
        cid = bid // BLOCKS_PER_CHUNK
        in_chunk = bid - cid * BLOCKS_PER_CHUNK
        in_chunk_off_kv = in_chunk.to(tl.int64) * in_chunk_block_bytes_kv
        in_chunk_off_dkv = in_chunk.to(tl.int64) * in_chunk_block_bytes_dkv
        bbp_off = cid.to(tl.int64) * stride_bbpb
    else:
        in_chunk_off_kv = tl.zeros([], dtype=tl.int64)
        in_chunk_off_dkv = tl.zeros([], dtype=tl.int64)
        bbp_off = bid.to(tl.int64) * stride_bbpb

    base_k = tl.load(BlockBasePtrsK + h * stride_bbph + bbp_off)
    base_v = tl.load(BlockBasePtrsV + h * stride_bbph + bbp_off)
    base_dk = tl.load(BlockBasePtrsDK + h * stride_bbph + bbp_off)
    base_dv = tl.load(BlockBasePtrsDV + h * stride_bbph + bbp_off)

    if base_k == 0:  # chunk offloaded; skip
        return
    if base_dk == 0:
        return  # scratch not allocated

    batch_off_kv = b.to(tl.int64) * batch_stride_kv_bytes
    batch_off_dkv = b.to(tl.int64) * batch_stride_dkv_bytes

    k_addr = base_k + batch_off_kv + in_chunk_off_kv
    v_addr = base_v + batch_off_kv + in_chunk_off_kv
    dk_addr = base_dk + batch_off_dkv + in_chunk_off_dkv
    dv_addr = base_dv + batch_off_dkv + in_chunk_off_dkv

    k_blk = k_addr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
    v_blk = v_addr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
    dk_blk = dk_addr.to(tl.pointer_type(tl.float32), bitcast=True)
    dv_blk = dv_addr.to(tl.pointer_type(tl.float32), bitcast=True)

    # offsets & masks for this KV tile
    tile_off = kv_tile_idx * TILE_KV
    offs_kv = tl.arange(0, TILE_KV)
    tile_valid = (tile_off + offs_kv) < BLOCK_SIZE
    offs_d = tl.arange(0, DP)
    mask_d = offs_d < D

    # load K, V tiles: [TILE_KV, D]
    k_blck = tl.load(
        k_blk + (tile_off + offs_kv)[:, None] * stride_bn + offs_d[None, :] * stride_bd,
        mask=tile_valid[:, None] & mask_d[None, :], other=0.0,
    ).to(tl.float32)
    v_blck = tl.load(
        v_blk + (tile_off + offs_kv)[:, None] * stride_bn + offs_d[None, :] * stride_bd,
        mask=tile_valid[:, None] & mask_d[None, :], other=0.0,
    ).to(tl.float32)

    b_dk = tl.zeros([TILE_KV, DP], dtype=tl.float32)  # this program owns these rows — no atomics
    b_dv = tl.zeros([TILE_KV, DP], dtype=tl.float32)

    log_scale = softmax_scale * 1.44269504

    # CSR range for this (b, h, bid) from the inverted index
    bo_base = BlockOffsets + b * stride_bob + h * stride_boh
    csr_start = tl.load(bo_base + bid).to(tl.int32)
    csr_end = tl.load(bo_base + bid + 1).to(tl.int32)
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
        q_blck = tl.load(
            q_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0,
        ).to(tl.float32)

        do_ptrs = (DOut + b * stride_dob + h * stride_doh
                   + m_indices[:, None] * stride_dom + offs_d[None, :] * stride_dod)
        do_blck = tl.load(
            do_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0,
        ).to(tl.float32)

        # load LSE and delta: [BLOCK_Q]
        l_vals = tl.load(
            Lse + b * stride_lb + h * stride_lh + m_indices * stride_lm,
            mask=mask_q, other=0.0,
        )
        d_vals = tl.load(
            Delta + b * stride_lb + h * stride_lh + m_indices * stride_lm,
            mask=mask_q, other=0.0,
        )

        # QK^T -> scores: [BLOCK_Q, TILE_KV]
        qk = tl.dot(
            q_blck, tl.trans(k_blck),
            input_precision=INPUT_PRECISION,
        ) * log_scale

        base_mask = tile_valid[None, :] & mask_q[:, None]
        qk = tl.where(base_mask, qk, -1.0e6)

        # recompute softmax weights from LSE
        l2 = l_vals * 1.44269504
        exp_qk = tl.math.exp2(qk - l2[:, None])
        exp_qk = tl.where(mask_q[:, None] & (l_vals[:, None] > -1e6), exp_qk, 0.0)

        # dV += exp_qk^T @ dO
        b_dv = tl.dot(
            tl.trans(exp_qk), do_blck, acc=b_dv,
            input_precision=INPUT_PRECISION,
        )

        # ds = exp_qk * (dp - delta) * scale; dK += ds^T @ Q
        dp = tl.dot(
            do_blck, tl.trans(v_blck),
            input_precision=INPUT_PRECISION,
        )
        ds = exp_qk * (dp - d_vals[:, None]) * softmax_scale

        b_dk = tl.dot(
            tl.trans(ds), q_blck, acc=b_dk,
            input_precision=INPUT_PRECISION,
        )

    # epilogue: write dK, dV to fp32 scratch
    write_mask = tile_valid[:, None] & mask_d[None, :]
    dk_ptrs = dk_blk + (tile_off + offs_kv)[:, None] * stride_dbn + offs_d[None, :] * stride_dbd
    dv_ptrs = dv_blk + (tile_off + offs_kv)[:, None] * stride_dbn + offs_d[None, :] * stride_dbd
    tl.store(dk_ptrs, b_dk, mask=write_mask)
    tl.store(dv_ptrs, b_dv, mask=write_mask)



def selection_attention_padded_ptr_bwd(
    q: torch.Tensor,                       # [B, M, H, D] bf16
    out: torch.Tensor,                     # [B, M, H, D] bf16 (saved from fwd)
    lse: torch.Tensor,                     # [B, H, M] fp32 (saved from fwd)
    d_out: torch.Tensor,                   # [B, M, H, D] bf16
    block_indices: torch.Tensor,           # [B, MG, H, T] int32 (saved from fwd input)
    ptr_table_k: torch.Tensor,             # [B, MG, H, T] int64 (saved from fwd)
    ptr_table_v: torch.Tensor,
    block_base_ptrs_k: torch.Tensor,       # [H, MAX_BASE] int64
    block_base_ptrs_v: torch.Tensor,
    block_base_ptrs_dk: torch.Tensor,      # [H, MAX_BASE] int64 (fp32 dK/dV)
    block_base_ptrs_dv: torch.Tensor,
    n_blocks_per_head: torch.Tensor,       # [H] int32 (global bid bound)
    block_size: int,
    group_size: int,
    batch_stride_kv_bytes: int,
    batch_stride_dkv_bytes: int,
    stride_bn: int,
    stride_bd: int,
    stride_dbn: Optional[int] = None,
    stride_dbd: Optional[int] = None,
    blocks_per_chunk: int = 0,             # > 0 → chunk-level base, == 0 → block-level
    in_chunk_block_bytes_kv: int = 0,
    in_chunk_block_bytes_dkv: int = 0,
    softmax_scale: Optional[float] = None,
    input_precision: str = "tf32",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-pass backward. Returns (dq, delta).
    dK/dV are written in-place into caller-owned fp32 scratch (zero-initialized).
    stride_dbn/dbd default to stride_bn/bd when None (same shape, fp32 dtype).
    blocks_per_chunk=0 for block-level base; delta is for debugging.
    """
    B, M, H, D = q.shape
    G = group_size
    MG = M // G
    T = block_indices.shape[-1]

    assert q.is_cuda and q.dtype == torch.bfloat16
    assert d_out.shape == q.shape and d_out.dtype == q.dtype
    assert out.shape == q.shape
    assert lse.shape == (B, H, M) and lse.dtype == torch.float32
    assert block_indices.shape == (B, MG, H, T) and block_indices.dtype == torch.int32
    assert ptr_table_k.shape == (B, MG, H, T) and ptr_table_k.dtype == torch.int64
    assert ptr_table_v.shape == ptr_table_k.shape
    assert block_base_ptrs_k.dim() == 2 and block_base_ptrs_k.shape[0] == H
    assert block_base_ptrs_v.shape == block_base_ptrs_k.shape
    assert block_base_ptrs_dk.shape == block_base_ptrs_k.shape
    assert block_base_ptrs_dv.shape == block_base_ptrs_k.shape
    assert n_blocks_per_head.shape == (H,) and n_blocks_per_head.dtype == torch.int32

    is_chunk_base = blocks_per_chunk > 0
    if is_chunk_base:
        assert in_chunk_block_bytes_kv > 0
        assert in_chunk_block_bytes_dkv > 0

    if softmax_scale is None:
        softmax_scale = D ** -0.5
    if stride_dbn is None:
        stride_dbn = stride_bn
    if stride_dbd is None:
        stride_dbd = stride_bd

    DP = max(16, triton.next_power_of_2(D))

    delta = torch.empty_like(lse)

    def grid_preprocess(META):
        return (triton.cdiv(M, META['BLOCK_M']), B * H)

    _sel_attn_bwd_preprocess_kernel[grid_preprocess](
        out, d_out, delta,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        d_out.stride(0), d_out.stride(1), d_out.stride(2), d_out.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        B, H, M, D, DP,
    )

    dq = torch.empty_like(q)

    PADDED_BLOCK_SIZE = max(16, triton.next_power_of_2(block_size))
    PADDED_GROUP_SIZE = max(16, triton.next_power_of_2(G))
    TILE_N = _compute_tile_n(block_size, max_tile=128)

    grid_dq = (B, MG, H)
    _sel_attn_bwd_dq_padded_ptr_kernel[grid_dq](
        q, ptr_table_k, ptr_table_v,
        lse, d_out, delta, dq,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        d_out.stride(0), d_out.stride(1), d_out.stride(2), d_out.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        ptr_table_k.stride(0), ptr_table_k.stride(1),
        ptr_table_k.stride(2), ptr_table_k.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        stride_bn, stride_bd,
        H=H, M=M, D=D, T=T, DP=DP,
        BLOCK_SIZE=block_size,
        PADDED_BLOCK_SIZE=PADDED_BLOCK_SIZE,
        GROUP_SIZE=G,
        PADDED_GROUP_SIZE=PADDED_GROUP_SIZE,
        TILE_N=TILE_N,
        INPUT_PRECISION=input_precision,
    )

    if G > 1:
        block_indices_expanded = block_indices.repeat_interleave(G, dim=1).contiguous()
    else:
        block_indices_expanded = block_indices.contiguous()

    max_n = int(n_blocks_per_head.max().item())
    sorted_queries, block_offsets = build_inverted_index_padded_ptr(
        block_indices_expanded, n_blocks_per_head, max_n,
    )

    TILE_KV = _compute_tile_n(block_size, max_tile=64)
    num_kv_tiles = (block_size + TILE_KV - 1) // TILE_KV

    grid_dkv = (max_n * num_kv_tiles, B, H)
    _sel_attn_bwd_dkv_padded_ptr_kernel[grid_dkv](
        q,
        block_base_ptrs_k, block_base_ptrs_v,
        block_base_ptrs_dk, block_base_ptrs_dv,
        n_blocks_per_head,
        lse, d_out, delta,
        sorted_queries, block_offsets,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        d_out.stride(0), d_out.stride(1), d_out.stride(2), d_out.stride(3),
        block_base_ptrs_k.stride(0), block_base_ptrs_k.stride(1),
        lse.stride(0), lse.stride(1), lse.stride(2),
        sorted_queries.stride(0), sorted_queries.stride(1),
        block_offsets.stride(0), block_offsets.stride(1),
        stride_bn, stride_bd,
        stride_dbn, stride_dbd,
        batch_stride_kv_bytes=batch_stride_kv_bytes,
        batch_stride_dkv_bytes=batch_stride_dkv_bytes,
        in_chunk_block_bytes_kv=in_chunk_block_bytes_kv if is_chunk_base else 0,
        in_chunk_block_bytes_dkv=in_chunk_block_bytes_dkv if is_chunk_base else 0,
        BLOCKS_PER_CHUNK=blocks_per_chunk if is_chunk_base else 1,
        IS_CHUNK_BASE=is_chunk_base,
        H=H, M=M, D=D, DP=DP,
        BLOCK_SIZE=block_size,
        PADDED_BLOCK_SIZE=PADDED_BLOCK_SIZE,
        TILE_KV=TILE_KV,
        INPUT_PRECISION=input_precision,
    )

    return dq, delta