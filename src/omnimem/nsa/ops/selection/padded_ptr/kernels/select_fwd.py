"""
Padded ptr table selection attention forward kernel.

Q reads KV blocks via ptr table (pointer chasing: load int64 addr → load [BLOCK_SIZE, D] bf16 → online softmax).
PtrTableK/V: [B, MG, H, T] int64; 0 = invalid sentinel (byte addresses to [BLOCK_SIZE, D] bf16 regions).
stride_bn / stride_bd: element counts (Triton scales by element size on typed ptrs).
"""
import torch
import triton
import triton.language as tl


_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
_sm_version = _capability[0] * 10 + _capability[1]

# num_stages=1: ptr-chasing breaks software pipelining (next-iter addr unknown until current load completes).
if _sm_version >= 90:
    _fwd_configs = [
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ]
else:
    _fwd_configs = [
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
    ]


def _compute_tile_n(block_size, max_tile=128):
    """Pick TILE_N: one tile if it fits, otherwise chunk into max_tile."""
    padded = max(16, triton.next_power_of_2(block_size))
    return padded if padded <= max_tile else max_tile


@triton.autotune(
    configs=_fwd_configs,
    key=['M', 'D', 'BLOCK_SIZE', 'T', 'GROUP_SIZE', 'TILE_N'],
)
@triton.jit
def _sel_attn_fwd_padded_ptr_kernel(
    Q,                # [B, M, H, D] bf16
    PtrTableK,        # [B, MG, H, T] int64
    PtrTableV,        # [B, MG, H, T] int64
    Out,              # [B, M, H, D] bf16
    Lse,              # [B, H, M] fp32
    softmax_scale: tl.constexpr,
    # Q / Out strides
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_ob, stride_om, stride_oh, stride_od,
    # PtrTable strides
    stride_ptb, stride_ptg, stride_pth, stride_ptt,
    # Lse strides
    stride_lb, stride_lh, stride_lm,
    # Block-internal strides — IN ELEMENTS (not bytes!)
    stride_bn,        # token stride within a block
    stride_bd,        # dim stride within a block (usually 1)
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
    RETURN_LSE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)        # group index in [0, MG)
    h = tl.program_id(2)

    m_start = g * GROUP_SIZE

    # base pointers
    q_base = Q + b * stride_qb + m_start * stride_qm + h * stride_qh
    o_base = Out + b * stride_ob + m_start * stride_om + h * stride_oh
    pt_base_k = PtrTableK + b * stride_ptb + g * stride_ptg + h * stride_pth
    pt_base_v = PtrTableV + b * stride_ptb + g * stride_ptg + h * stride_pth

    # offsets & masks
    offs_g = tl.arange(0, PADDED_GROUP_SIZE)
    mask_g = offs_g < GROUP_SIZE
    offs_d = tl.arange(0, DP)
    mask_d = offs_d < D

    # load Q tile: [GROUP_SIZE, D]
    q_ptrs = q_base + offs_g[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_blck = tl.load(
        q_ptrs, mask=mask_g[:, None] & mask_d[None, :], other=0.0
    ).to(tl.float32)

    # online softmax accumulators
    max_log = tl.full([PADDED_GROUP_SIZE], float('-inf'), dtype=tl.float32)
    sum_exp = tl.zeros([PADDED_GROUP_SIZE], dtype=tl.float32)
    accum = tl.zeros([PADDED_GROUP_SIZE, DP], dtype=tl.float32)

    NUM_TILES: tl.constexpr = (BLOCK_SIZE + TILE_N - 1) // TILE_N

    # loop over T selected blocks via ptr table
    for t_idx in range(T):
        k_block_ptr = tl.load(pt_base_k + t_idx * stride_ptt)  # int64 byte addr
        v_block_ptr = tl.load(pt_base_v + t_idx * stride_ptt)

        if k_block_ptr != 0:  # 0 = invalid sentinel
            # Reinterpret int64 byte address as typed bf16 ptr (bitcast keeps raw addr).
            k_blk = k_block_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
            v_blk = v_block_ptr.to(tl.pointer_type(tl.bfloat16), bitcast=True)

            for tile_idx in range(NUM_TILES):
                tile_off = tile_idx * TILE_N
                offs_tile = tl.arange(0, TILE_N)
                tile_valid = (tile_off + offs_tile) < BLOCK_SIZE

                # load K tile: [TILE_N, D] (element offsets on typed ptr)
                k_tile_ptrs = (
                    k_blk
                    + (tile_off + offs_tile)[:, None] * stride_bn
                    + offs_d[None, :] * stride_bd
                )
                k_tile = tl.load(
                    k_tile_ptrs,
                    mask=tile_valid[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)

                # load V tile: [TILE_N, D]
                v_tile_ptrs = (
                    v_blk
                    + (tile_off + offs_tile)[:, None] * stride_bn
                    + offs_d[None, :] * stride_bd
                )
                v_tile = tl.load(
                    v_tile_ptrs,
                    mask=tile_valid[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)

                # QK^T -> scores: [GROUP_SIZE, TILE_N]
                qk = tl.dot(
                    q_blck, tl.trans(k_tile),
                    input_precision=INPUT_PRECISION,
                ) * softmax_scale

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
                accum = tl.dot(
                    exp_qk, v_tile, accum,
                    input_precision=INPUT_PRECISION,
                )
                max_log = new_max

    # epilogue: normalize and write output
    safe_sum = tl.where(sum_exp > 0.0, sum_exp, 1.0)
    out_vals = tl.where(
        sum_exp[:, None] > 0.0,
        accum / safe_sum[:, None],
        0.0,
    )

    o_ptrs = o_base + offs_g[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out_vals.to(tl.bfloat16), mask=mask_g[:, None] & mask_d[None, :])

    if RETURN_LSE:
        l_base = Lse + b * stride_lb + h * stride_lh + m_start * stride_lm
        fin_log = tl.where(
            sum_exp > 0.0,
            max_log + tl.math.log(safe_sum),
            float('-inf'),
        )
        l_ptrs = l_base + offs_g * stride_lm
        tl.store(l_ptrs, fin_log, mask=mask_g)