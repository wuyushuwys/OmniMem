"""
reference: https://github.com/tilde-research/nsa-impl/blob/main/nsa/selection.py
"""
import math
import torch
import triton
import triton.language as tl
import triton.testing

# Device-adaptive autotune configs: SM >= 90 (H100) uses more warps/stages
_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
_sm_version = _capability[0] * 10 + _capability[1]  # e.g. 89 for L40S, 90 for H100

base_num_warps = 1

if _sm_version >= 90:
    _sel_attn_fwd_configs = [
        triton.Config({}, num_warps=base_num_warps, num_stages=2),
        triton.Config({}, num_warps=base_num_warps, num_stages=3),
        triton.Config({}, num_warps=base_num_warps * 2, num_stages=2),
        triton.Config({}, num_warps=base_num_warps * 2, num_stages=3),
        triton.Config({}, num_warps=base_num_warps * 2, num_stages=4),
    ]
else:
    _sel_attn_fwd_configs = [
        triton.Config({}, num_warps=base_num_warps, num_stages=1),
        triton.Config({}, num_warps=base_num_warps, num_stages=2),
        triton.Config({}, num_warps=base_num_warps * 2, num_stages=1),
        triton.Config({}, num_warps=base_num_warps * 2, num_stages=2),
        triton.Config({}, num_warps=base_num_warps * 4, num_stages=1),
    ]



@triton.autotune(
    configs=_sel_attn_fwd_configs,
    key=['M', 'N', 'D', 'SELECTION_BLOCK_SIZE', 'T', 'causal'],
)
@triton.jit
def _sel_attn_fwd_kernel(
        Q: tl.tensor,
        K: tl.tensor,
        V: tl.tensor,
        Top_idx: tl.tensor,
        Order: tl.tensor,
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
        PADDED_BLOCK_SIZE: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        OFFSET_M: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
):
    b = tl.program_id(0)
    m_physical = tl.program_id(1) + OFFSET_M
    h = tl.program_id(2)

    # Order: sorted token position maps to original token index
    r_ptr = Order + b * stride_rb + m_physical * stride_rm + h * stride_rh
    m = tl.load(r_ptr)

    # base pointers
    q_base = Q + b * stride_qb + m * stride_qm + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    t_base = Top_idx + b * stride_tb + m * stride_tm + h * stride_th
    o_base = Out + b * stride_ob + m * stride_om + h * stride_oh
    l_base = Lse + b * stride_lb + m * stride_lm + h * stride_lh

    # offsets & masks
    offs_d = tl.arange(0, DP)
    mask_d = offs_d < D
    offs_p = tl.arange(0, PADDED_BLOCK_SIZE)
    valid_mask = offs_p < SELECTION_BLOCK_SIZE

    # load Q token: [1, D]
    q_ptrs = q_base + offs_d[None, :] * stride_qd
    q_blck = tl.load(q_ptrs, mask=mask_d[None, :], other=0.0).to(tl.float32)

    # online softmax accumulators
    max_log = tl.full([1], float('-inf'), dtype=tl.float32)
    sum_exp = tl.zeros([1], dtype=tl.float32)
    accum = tl.zeros([1, DP], dtype=tl.float32)

    q_chunk_idx = m // CHUNK_SIZE if causal else N

    # loop over T selected blocks
    for idx in range(T):
        top = tl.load(t_base + idx * stride_tt)

        col = top * SELECTION_BLOCK_SIZE
        col = tl.multiple_of(col, SELECTION_BLOCK_SIZE)

        if not causal or (col // CHUNK_SIZE <= q_chunk_idx and col >= 0):
            cols = col + offs_p
            mask_n = cols < N

            # load K block: [BLOCK_SIZE, D]
            k_ptrs = k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k_blck = tl.load(k_ptrs, mask=mask_n[:, None] & valid_mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            # load V block: [BLOCK_SIZE, D]
            v_ptrs = v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v_blck = tl.load(v_ptrs, mask=mask_n[:, None] & valid_mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            # QK^T -> scores: [1, BLOCK_SIZE]
            qk = tl.dot(
                q_blck, tl.trans(k_blck),
                allow_tf32=ALLOW_TF32
            ) * softmax_scale

            # online softmax: running max / sum
            causal_mask = (cols // CHUNK_SIZE) <= q_chunk_idx
            qk = tl.where(causal_mask[None, :] & valid_mask[None, :] & mask_n[None, :], qk, -1.0e6)
            new_max = tl.maximum(max_log, tl.max(qk, axis=1))
            exp_qk = tl.math.exp(qk - new_max[:, None])
            sum_qk = tl.sum(exp_qk, axis=1)

            alpha = tl.math.exp(max_log - new_max)
            alpha = tl.where(max_log > -1.0e6, alpha, 0.0)
            sum_exp = sum_exp * alpha + sum_qk
            accum = accum * alpha[:, None]

            # accumulate: exp_qk @ V: [1, D]
            accum = tl.dot(
                exp_qk,
                v_blck,
                accum,
                allow_tf32=ALLOW_TF32
            )
            max_log = new_max

    # epilogue: normalize and write output + LSE
    safe_sum = tl.where(sum_exp > 0.0, sum_exp, 1.0)
    fin_log = max_log + tl.math.log(safe_sum)
    fin_log = tl.where(sum_exp > 0.0, fin_log, float('-inf'))
    safe_sum = tl.where(sum_exp > 0.0, sum_exp, 1.0)
    out_vals = accum / safe_sum[:, None]
    out_vals = tl.where(sum_exp[:, None] > 0.0, out_vals, 0.0)

    o_ptrs = o_base + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out_vals, mask=mask_d[None, :])

    offs_1 = tl.arange(0, 1)
    l_ptrs = l_base + offs_1 * stride_lh
    tl.store(l_ptrs, fin_log)