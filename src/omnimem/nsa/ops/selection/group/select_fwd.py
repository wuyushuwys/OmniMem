"""
reference: https://github.com/tilde-research/nsa-impl/blob/main/nsa/selection.py
Tiled forward kernel; TILE_N = next_pow2(block_size) if <= 128, else 128 (multi-tile).
"""
import math
import torch
import triton
import triton.language as tl
import triton.testing

# Device-adaptive autotune configs (TILE_N set by caller, not autotuned)
_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
_sm_version = _capability[0] * 10 + _capability[1]

base_num_warps = 1

if _sm_version >= 90:
    _sel_attn_fwd_configs = [
        triton.Config({}, num_warps=base_num_warps * 2, num_stages=1),
        triton.Config({}, num_warps=base_num_warps * 2, num_stages=2),
        triton.Config({}, num_warps=base_num_warps * 4, num_stages=1),
        triton.Config({}, num_warps=base_num_warps * 4, num_stages=2),
    ]
else:
    _sel_attn_fwd_configs = [
        triton.Config({}, num_warps=base_num_warps * 2, num_stages=1),
        triton.Config({}, num_warps=base_num_warps * 4, num_stages=1),
        triton.Config({}, num_warps=base_num_warps * 2, num_stages=2),
    ]


def _compute_tile_n(selection_block_size, max_tile=128):
    """Pick TILE_N: use one tile if it fits, otherwise chunk into max_tile."""
    padded = max(16, triton.next_power_of_2(selection_block_size))
    return padded if padded <= max_tile else max_tile


@triton.autotune(
    configs=_sel_attn_fwd_configs,
    key=['M', 'N', 'D', 'SELECTION_BLOCK_SIZE', 'T', 'causal', 'TILE_N'],
)
@triton.jit
def _sel_attn_fwd_kernel(
        Q: tl.tensor,
        K: tl.tensor,
        V: tl.tensor,
        Top_idx: tl.tensor,  # [B, MG, H, T]
        Order: tl.tensor,    # [B, MG, H]
        softmax_scale: tl.constexpr,
        causal: tl.constexpr,
        Out: tl.tensor,
        Lse: tl.tensor,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_tb, stride_th, stride_tm, stride_tt,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_lb, stride_lh, stride_lm,
        stride_rb, stride_rm, stride_rh,
        B: tl.constexpr,
        H: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        T: tl.constexpr,
        DP: tl.constexpr,
        SELECTION_BLOCK_SIZE: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        OFFSET_M: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        PADDED_GROUP_SIZE: tl.constexpr,
        TILE_N: tl.constexpr,
):
    b = tl.program_id(0)
    g_physical = tl.program_id(1) + OFFSET_M
    h = tl.program_id(2)

    # Order: sorted group position → original group index
    r_ptr = Order + b * stride_rb + g_physical * stride_rm + h * stride_rh
    g = tl.load(r_ptr)
    m_start = g * GROUP_SIZE

    # base pointers
    q_base = Q + b * stride_qb + m_start * stride_qm + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    t_base = Top_idx + b * stride_tb + g * stride_tm + h * stride_th
    o_base = Out + b * stride_ob + m_start * stride_om + h * stride_oh
    l_base = Lse + b * stride_lb + m_start * stride_lm + h * stride_lh

    # offsets & masks
    offs_g = tl.arange(0, PADDED_GROUP_SIZE)
    mask_g = offs_g < GROUP_SIZE
    offs_d = tl.arange(0, DP)
    mask_d = offs_d < D

    # load Q tile: [GROUP_SIZE, D]
    q_ptrs = q_base + offs_g[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_blck = tl.load(q_ptrs, mask=mask_g[:, None] & mask_d[None, :],
                     other=0.0).to(tl.float32)

    # online softmax accumulators
    max_log = tl.full([PADDED_GROUP_SIZE], float('-inf'), dtype=tl.float32)
    sum_exp = tl.zeros([PADDED_GROUP_SIZE], dtype=tl.float32)
    accum = tl.zeros([PADDED_GROUP_SIZE, DP], dtype=tl.float32)

    q_chunk_idx = m_start // CHUNK_SIZE if causal else N

    NUM_TILES: tl.constexpr = (SELECTION_BLOCK_SIZE + TILE_N - 1) // TILE_N  # tiles per selection block

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

                # load K tile: [TILE_N, D]
                p_k = tl.make_block_ptr(
                    base=k_base, shape=(N, D), strides=(stride_kn, stride_kd),
                    offsets=(tile_start, 0), block_shape=(TILE_N, DP), order=(1, 0)
                )
                k_tile = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)

                # load V tile: [TILE_N, D]
                p_v = tl.make_block_ptr(
                    base=v_base, shape=(N, D), strides=(stride_vn, stride_vd),
                    offsets=(tile_start, 0), block_shape=(TILE_N, DP), order=(1, 0)
                )
                v_tile = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)

                # QK^T -> scores: [GROUP_SIZE, TILE_N]
                qk = tl.dot(q_blck, tl.trans(k_tile),
                            allow_tf32=ALLOW_TF32) * softmax_scale

                if causal:
                    causal_mask = (tile_cols // CHUNK_SIZE) <= q_chunk_idx
                    combined_mask = causal_mask[None, :] & tile_valid[None, :] & mask_g[:, None]
                else:
                    combined_mask = tile_valid[None, :] & mask_g[:, None]
                qk = tl.where(combined_mask, qk, -1.0e6)

                # online softmax: running max / sum
                new_max = tl.maximum(max_log, tl.max(qk, axis=1))
                exp_qk = tl.math.exp(qk - new_max[:, None])
                exp_qk = tl.where(combined_mask, exp_qk, 0.0)
                sum_qk = tl.sum(exp_qk, axis=1)

                alpha = tl.math.exp(max_log - new_max)
                alpha = tl.where(max_log > -1.0e6, alpha, 0.0)
                sum_exp = sum_exp * alpha + sum_qk
                accum = accum * alpha[:, None]

                # accumulate: exp_qk @ V
                accum = tl.dot(exp_qk, v_tile, accum, allow_tf32=ALLOW_TF32)
                max_log = new_max

    # epilogue: normalize and write output + LSE
    fin_log = tl.where(sum_exp > 0.0,
                       max_log + tl.math.log(sum_exp),
                       float('-inf'))
    out_vals = tl.where(sum_exp[:, None] > 0.0,
                        accum / sum_exp[:, None],
                        0.0)

    o_ptrs = o_base + offs_g[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out_vals, mask=mask_g[:, None] & mask_d[None, :])

    l_ptrs = l_base + offs_g * stride_lm
    tl.store(l_ptrs, fin_log, mask=mask_g)