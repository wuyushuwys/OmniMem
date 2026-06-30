"""
GPU-side ptr table builder. Two modes (IS_CHUNK_BASE constexpr):

  Block-level (IS_CHUNK_BASE=False): base[h, bid] gives final ptr; adds batch_stride only.
  Chunk-level (IS_CHUNK_BASE=True):  base[h, cid] gives chunk ptr; kernel adds in-chunk offset:
    cid = bid // BLOCKS_PER_CHUNK; off = (bid % BLOCKS_PER_CHUNK) * IN_CHUNK_BLOCK_BYTES

Critical: base == 0 means chunk is NOT GPU-resident; kernel must produce ptr == 0
so the fwd kernel skips it (base + batch_off would yield a non-zero invalid addr).
"""
import torch
import triton
import triton.language as tl


_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
_sm_version = _capability[0] * 10 + _capability[1]

if _sm_version >= 90:
    _build_configs = [
        triton.Config({'BLOCK_T': 16}, num_warps=2),
        triton.Config({'BLOCK_T': 32}, num_warps=2),
        triton.Config({'BLOCK_T': 32}, num_warps=4),
        triton.Config({'BLOCK_T': 64}, num_warps=4),
    ]
else:
    _build_configs = [
        triton.Config({'BLOCK_T': 16}, num_warps=2),
        triton.Config({'BLOCK_T': 32}, num_warps=4),
    ]


@triton.autotune(configs=_build_configs, key=['T', 'H', 'IS_CHUNK_BASE'])
@triton.jit
def _build_ptr_table_kernel(
    BlockIndices,        # [B, MG, H, T] int32 — global bid
    BlockBasePtrsK,      # [H, max_n] int64
    BlockBasePtrsV,      # [H, max_n] int64
    NBlocksPerHead,      # [H] int32 — bound for global bid
    PtrTableK,           # [B, MG, H, T] int64 (out)
    PtrTableV,           # [B, MG, H, T] int64 (out)
    stride_bib, stride_big, stride_bih, stride_bit,
    stride_bbph, stride_bbpb,
    stride_ptb, stride_ptg, stride_pth, stride_ptt,
    batch_stride: tl.constexpr,
    BLOCKS_PER_CHUNK: tl.constexpr,
    IN_CHUNK_BLOCK_BYTES: tl.constexpr,
    IS_CHUNK_BASE: tl.constexpr,
    MG: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    h = tl.program_id(2)

    n_h = tl.load(NBlocksPerHead + h)

    # base pointers into BlockIndices and PtrTable for this (b, g, h)
    bi_base = BlockIndices + b * stride_bib + g * stride_big + h * stride_bih
    pk_base = PtrTableK + b * stride_ptb + g * stride_ptg + h * stride_pth
    pv_base = PtrTableV + b * stride_ptb + g * stride_ptg + h * stride_pth
    bbpk_base = BlockBasePtrsK + h * stride_bbph
    bbpv_base = BlockBasePtrsV + h * stride_bbph

    batch_off = b * batch_stride  # byte offset for batch dimension

    # loop over T slots in steps of BLOCK_T
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T

        # load block indices; filter invalid (-1) and out-of-range
        bids = tl.load(bi_base + offs_t * stride_bit, mask=mask_t, other=-1).to(tl.int32)
        valid = mask_t & (bids >= 0) & (bids < n_h)

        if IS_CHUNK_BASE:
            # chunk-level: base[h, cid] + in-chunk offset + batch
            cids = bids // BLOCKS_PER_CHUNK
            in_chunk = bids - cids * BLOCKS_PER_CHUNK
            in_chunk_off = in_chunk.to(tl.int64) * IN_CHUNK_BLOCK_BYTES
            safe_cids = tl.where(valid, cids, 0).to(tl.int64)
            base_k = tl.load(bbpk_base + safe_cids * stride_bbpb, mask=valid, other=0)
            base_v = tl.load(bbpv_base + safe_cids * stride_bbpb, mask=valid, other=0)
            # base==0: chunk not GPU-resident; ptr must be 0 so fwd kernel's `if ptr != 0` skip works.
            base_valid_k = valid & (base_k != 0)
            base_valid_v = valid & (base_v != 0)
            ptr_k = tl.where(base_valid_k, base_k + batch_off + in_chunk_off, 0)
            ptr_v = tl.where(base_valid_v, base_v + batch_off + in_chunk_off, 0)
        else:
            # block-level: base[h, bid] + batch
            safe_bids = tl.where(valid, bids, 0).to(tl.int64)
            base_k = tl.load(bbpk_base + safe_bids * stride_bbpb, mask=valid, other=0)
            base_v = tl.load(bbpv_base + safe_bids * stride_bbpb, mask=valid, other=0)
            base_valid_k = valid & (base_k != 0)
            base_valid_v = valid & (base_v != 0)
            ptr_k = tl.where(base_valid_k, base_k + batch_off, 0)
            ptr_v = tl.where(base_valid_v, base_v + batch_off, 0)

        tl.store(pk_base + offs_t * stride_ptt, ptr_k, mask=mask_t)
        tl.store(pv_base + offs_t * stride_ptt, ptr_v, mask=mask_t)


def build_ptr_table(
    block_indices: torch.Tensor,
    block_base_ptrs_k: torch.Tensor,
    block_base_ptrs_v: torch.Tensor,
    n_blocks_per_head: torch.Tensor,
    batch_stride: int,
    blocks_per_chunk: int = 0,
    in_chunk_block_bytes: int = 0,
):
    B, MG, H, T = block_indices.shape
    device = block_indices.device

    assert block_indices.dtype == torch.int32
    assert block_base_ptrs_k.dtype == torch.int64
    assert block_base_ptrs_v.dtype == torch.int64
    assert n_blocks_per_head.dtype == torch.int32
    assert block_base_ptrs_k.shape == block_base_ptrs_v.shape
    assert block_base_ptrs_k.shape[0] == H
    assert n_blocks_per_head.shape == (H,)

    is_chunk_base = blocks_per_chunk > 0
    if is_chunk_base:
        assert in_chunk_block_bytes > 0, "in_chunk_block_bytes required for chunk-level base"

    ptr_table_k = torch.empty(B, MG, H, T, dtype=torch.int64, device=device)
    ptr_table_v = torch.empty(B, MG, H, T, dtype=torch.int64, device=device)

    grid = (B, MG, H)
    _build_ptr_table_kernel[grid](
        block_indices,
        block_base_ptrs_k, block_base_ptrs_v,
        n_blocks_per_head,
        ptr_table_k, ptr_table_v,
        block_indices.stride(0), block_indices.stride(1),
        block_indices.stride(2), block_indices.stride(3),
        block_base_ptrs_k.stride(0), block_base_ptrs_k.stride(1),
        ptr_table_k.stride(0), ptr_table_k.stride(1),
        ptr_table_k.stride(2), ptr_table_k.stride(3),
        batch_stride=batch_stride,
        BLOCKS_PER_CHUNK=blocks_per_chunk if is_chunk_base else 1,
        IN_CHUNK_BLOCK_BYTES=in_chunk_block_bytes if is_chunk_base else 0,
        IS_CHUNK_BASE=is_chunk_base,
        MG=MG, H=H, T=T,
    )

    return ptr_table_k, ptr_table_v