"""
Selection attention via gather-to-varlen-dense (streaming trainer with LRU offload).

Avoids saving live cache refs in save_for_backward (which trips on device equality under AC).
Steps: 1) gather unique (h, cid) chunks to dense K_gather/V_gather; build remapped ptr table.
       2) run existing Triton fwd with remapped indices; save only small stable tensors.
       3) backward re-gathers K/V from cache (cache content is stable across fwd and bwd); returns dq only.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import triton

from .kernels.select_fwd import _sel_attn_fwd_padded_ptr_kernel, _compute_tile_n
from .kernels.select_bwd import selection_attention_padded_ptr_bwd
from .ptr_builder import build_ptr_table



def _decode_block_indices_per_head(
    block_indices: torch.Tensor,
    blocks_per_chunk: int,
    n_chunks_cap: int,
) -> Tuple[List[List[int]], torch.Tensor, List[int], List[int], int]:
    """Decode block_indices to per-head unique cids and remapped block indices.

    Returns: (head_cid_lists, block_indices_remapped, head_offsets, n_chunks_per_head_list, n_total)
    block_indices_remapped: bid = local_pos * blocks_per_chunk + in_chunk_offset; -1 for invalid.
    """
    bi_np = block_indices.cpu().numpy()
    B, MG, H, T = bi_np.shape
    cid_np = bi_np // blocks_per_chunk
    in_chunk_np = bi_np - cid_np * blocks_per_chunk
    valid_all = (bi_np >= 0) & (cid_np < n_chunks_cap)

    head_cid_lists: List[List[int]] = []
    head_offsets: List[int] = [0]
    n_chunks_per_head_list: List[int] = []
    block_indices_remapped_np = np.full_like(bi_np, -1, dtype=np.int32)

    # build per-head unique cid list and remap block indices to gather-space
    for h in range(H):
        h_valid = valid_all[:, :, h, :]
        head_cids_flat = cid_np[:, :, h, :][h_valid]
        if head_cids_flat.size == 0:
            head_cid_lists.append([])
            n_chunks_per_head_list.append(0)
            head_offsets.append(head_offsets[-1])
            continue
        unique_cids = np.unique(head_cids_flat)
        head_cid_lists.append(unique_cids.tolist())
        n_chunks_per_head_list.append(int(unique_cids.size))
        head_offsets.append(head_offsets[-1] + int(unique_cids.size))

        h_cid = cid_np[:, :, h, :]
        h_in = in_chunk_np[:, :, h, :]
        safe_cid = np.where(h_valid, h_cid, 0)  # safe index for searchsorted
        local_pos = np.searchsorted(unique_cids, safe_cid).astype(np.int32)
        bid_remapped = local_pos * blocks_per_chunk + h_in.astype(np.int32)
        block_indices_remapped_np[:, :, h, :] = np.where(h_valid, bid_remapped, -1)

    block_indices_remapped = torch.from_numpy(
        block_indices_remapped_np
    ).to(block_indices.device, non_blocking=True).contiguous()

    return (head_cid_lists, block_indices_remapped, head_offsets,
            n_chunks_per_head_list, head_offsets[-1])


def _gather_kv_to_dense(
    k_store: List[List[torch.Tensor]],
    v_store: List[List[torch.Tensor]],
    head_cid_lists: List[List[int]],
    head_offsets: List[int],
    n_total: int,
    chunk_len: int,
    head_dim: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Gather selected (h, cid) chunks into a flat dense (B, n_total, chunk_len, D) bf16 tensor.
    CPU pinned slots are reloaded transiently; cache list is NOT mutated.
    Returns: (K_gather, V_gather, chunk_base_ptrs_k, chunk_base_ptrs_v, max_chunks)
    """
    H = len(head_cid_lists)
    max_chunks = max((len(L) for L in head_cid_lists), default=0)
    if max_chunks == 0:  # at least 1 column for the ptr table
        max_chunks = 1

    # allocate dense gather buffer: [B, n_total, chunk_len, D]
    K_gather = torch.empty(
        (batch_size, n_total if n_total > 0 else 1, chunk_len, head_dim),
        dtype=dtype, device=device,
    )
    V_gather = torch.empty_like(K_gather)
    # n_total==0: 1-slot dummy; ptr table all zeros; kernel skips all blocks.

    K_base = K_gather.data_ptr()
    V_base = V_gather.data_ptr()
    elem_size = K_gather.element_size()
    chunk_bytes = chunk_len * head_dim * elem_size  # byte stride per chunk slot in K_gather

    chunk_base_ptrs_k_cpu = torch.zeros((H, max_chunks), dtype=torch.int64)
    chunk_base_ptrs_v_cpu = torch.zeros((H, max_chunks), dtype=torch.int64)

    # gather selected (h, cid) chunks into K_gather / V_gather; build base ptr table
    for h in range(H):
        cids = head_cid_lists[h]
        if not cids:
            continue
        head_start = head_offsets[h]
        for local_pos, cid in enumerate(cids):
            slot_idx = head_start + local_pos
            # copy_ handles both pinned CPU (H2D) and GPU (D2D) sources
            K_gather[:, slot_idx].copy_(k_store[h][cid], non_blocking=True)
            V_gather[:, slot_idx].copy_(v_store[h][cid], non_blocking=True)
            chunk_base_ptrs_k_cpu[h, local_pos] = K_base + slot_idx * chunk_bytes
            chunk_base_ptrs_v_cpu[h, local_pos] = V_base + slot_idx * chunk_bytes

    # move base ptr table to GPU
    chunk_base_ptrs_k = chunk_base_ptrs_k_cpu.to(device, non_blocking=True)
    chunk_base_ptrs_v = chunk_base_ptrs_v_cpu.to(device, non_blocking=True)
    return K_gather, V_gather, chunk_base_ptrs_k, chunk_base_ptrs_v, max_chunks


class _SelectionAttentionVarlenDense(torch.autograd.Function):
    # save_for_backward: q, out, lse, block_indices_remapped, n_chunks_per_head_gpu
    # ctx: kv_cache, layer_idx, head_cid_lists, head_offsets, n_chunks_per_head_list, meta

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,                       # [B, M, H, D] bf16
        block_indices_remapped: torch.Tensor,  # [B, MG, H, T] int32, in gather-space
        n_chunks_per_head: torch.Tensor,       # [H] int32 on device, = |head_cid_lists[h]|
        K_gather: torch.Tensor,                # [B, n_total or 1, chunk_len, D] bf16
        V_gather: torch.Tensor,
        chunk_base_ptrs_k: torch.Tensor,       # [H, max_chunks] int64, into K_gather
        chunk_base_ptrs_v: torch.Tensor,
        meta: Dict[str, Any],
        # Non-tensor: stashed on ctx for backward's re-gather.
        kv_cache,
        layer_idx: int,
        head_cid_lists: List[List[int]],
        head_offsets: List[int],
        n_chunks_per_head_list: List[int],
    ):
        block_size = meta['block_size']
        group_size = meta['group_size']
        chunk_len = meta['chunk_len']
        blocks_per_chunk = chunk_len // block_size
        softmax_scale = meta['softmax_scale']
        input_precision = meta['input_precision']

        B, M, H, D = q.shape
        G = group_size
        MG = M // G
        T = block_indices_remapped.shape[-1]
        device = q.device

        elem = K_gather.element_size()
        n_chunks_in_gather = K_gather.shape[1]
        batch_stride_bytes = n_chunks_in_gather * chunk_len * D * elem  # batch stride in BYTES
        token_stride_bytes = D * elem
        stride_bn = D  # in elements
        stride_bd = 1
        in_chunk_block_bytes = block_size * token_stride_bytes

        n_global_blocks = n_chunks_per_head * blocks_per_chunk  # bound for remapped block indices

        ptr_table_k, ptr_table_v = build_ptr_table(
            block_indices=block_indices_remapped,
            block_base_ptrs_k=chunk_base_ptrs_k,
            block_base_ptrs_v=chunk_base_ptrs_v,
            n_blocks_per_head=n_global_blocks,
            batch_stride=batch_stride_bytes,
            blocks_per_chunk=blocks_per_chunk,
            in_chunk_block_bytes=in_chunk_block_bytes,
        )

        out = torch.empty_like(q)
        lse = torch.empty(B, H, M, dtype=torch.float32, device=device)

        DP = max(16, triton.next_power_of_2(D))
        PADDED_BLOCK_SIZE = max(16, triton.next_power_of_2(block_size))
        PADDED_GROUP_SIZE = max(16, triton.next_power_of_2(G))
        TILE_N = _compute_tile_n(block_size)

        _sel_attn_fwd_padded_ptr_kernel[(B, MG, H)](
            q, ptr_table_k, ptr_table_v, out, lse,
            softmax_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
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
            RETURN_LSE=True,
            INPUT_PRECISION=input_precision,
        )

        ctx.save_for_backward(
            q, out, lse,
            block_indices_remapped,
            n_chunks_per_head.clone(),
        )
        ctx.kv_cache = kv_cache
        ctx.layer_idx = layer_idx
        ctx.head_cid_lists = head_cid_lists
        ctx.head_offsets = head_offsets
        ctx.n_chunks_per_head_list = n_chunks_per_head_list
        ctx.meta = meta
        ctx.batch_size = B
        ctx.head_dim = D
        ctx.dtype = q.dtype
        # K_gather/V_gather intentionally NOT saved; backward re-gathers on demand.
        return out

    @staticmethod
    def backward(ctx, d_out):
        q, out, lse, block_indices_remapped, n_chunks_per_head = ctx.saved_tensors
        meta = ctx.meta
        block_size = meta['block_size']
        group_size = meta['group_size']
        chunk_len = meta['chunk_len']
        blocks_per_chunk = chunk_len // block_size
        softmax_scale = meta['softmax_scale']
        input_precision = meta['input_precision']

        device = q.device
        B, M, H, D = q.shape
        dtype = ctx.dtype

        k_store = ctx.kv_cache.cache[ctx.layer_idx]['k_cache']
        v_store = ctx.kv_cache.cache[ctx.layer_idx]['v_cache']

        # re-gather K/V to dense (cache content is stable across fwd and bwd)
        K_gather, V_gather, chunk_base_ptrs_k, chunk_base_ptrs_v, _ = _gather_kv_to_dense(
            k_store, v_store,
            ctx.head_cid_lists, ctx.head_offsets,
            n_total=ctx.head_offsets[-1] if ctx.head_offsets[-1] > 0 else 0,
            chunk_len=chunk_len,
            head_dim=ctx.head_dim,
            batch_size=ctx.batch_size,
            device=device,
            dtype=dtype,
        )

        elem = K_gather.element_size()
        n_chunks_in_gather = K_gather.shape[1]
        batch_stride_kv_bytes = n_chunks_in_gather * chunk_len * D * elem
        token_stride_bytes = D * elem
        stride_bn = D
        stride_bd = 1
        in_chunk_block_bytes_kv = block_size * token_stride_bytes

        n_global_blocks = n_chunks_per_head * blocks_per_chunk

        # rebuild ptr table from fresh K_gather addresses
        ptr_table_k, ptr_table_v = build_ptr_table(
            block_indices=block_indices_remapped,
            block_base_ptrs_k=chunk_base_ptrs_k,
            block_base_ptrs_v=chunk_base_ptrs_v,
            n_blocks_per_head=n_global_blocks,
            batch_stride=batch_stride_kv_bytes,
            blocks_per_chunk=blocks_per_chunk,
            in_chunk_block_bytes=in_chunk_block_bytes_kv,
        )

        # allocate fp32 dK/dV scratch buffers (same layout as K_gather)
        dk_scratch = torch.zeros_like(K_gather, dtype=torch.float32)
        dv_scratch = torch.zeros_like(V_gather, dtype=torch.float32)
        elem_fp32 = dk_scratch.element_size()
        batch_stride_dkv_bytes = n_chunks_in_gather * chunk_len * D * elem_fp32
        token_stride_dkv_bytes = D * elem_fp32
        in_chunk_block_bytes_dkv = block_size * token_stride_dkv_bytes
        stride_dbn = D  # in fp32 elements
        stride_dbd = 1

        # build dK/dV base ptr table (CPU to GPU)
        H_n = len(ctx.head_cid_lists)
        max_chunks = chunk_base_ptrs_k.shape[1]
        dk_base = dk_scratch.data_ptr()
        dv_base = dv_scratch.data_ptr()
        dk_chunk_bytes = chunk_len * D * elem_fp32
        base_dk_cpu = torch.zeros((H_n, max_chunks), dtype=torch.int64)
        base_dv_cpu = torch.zeros_like(base_dk_cpu)
        for h, cids in enumerate(ctx.head_cid_lists):
            head_start = ctx.head_offsets[h]
            for local_pos in range(len(cids)):
                slot = head_start + local_pos
                base_dk_cpu[h, local_pos] = dk_base + slot * dk_chunk_bytes
                base_dv_cpu[h, local_pos] = dv_base + slot * dk_chunk_bytes
        base_dk = base_dk_cpu.to(device, non_blocking=True)
        base_dv = base_dv_cpu.to(device, non_blocking=True)

        dq, _ = selection_attention_padded_ptr_bwd(
            q=q, out=out, lse=lse, d_out=d_out,
            block_indices=block_indices_remapped,
            ptr_table_k=ptr_table_k, ptr_table_v=ptr_table_v,
            block_base_ptrs_k=chunk_base_ptrs_k,
            block_base_ptrs_v=chunk_base_ptrs_v,
            block_base_ptrs_dk=base_dk,
            block_base_ptrs_dv=base_dv,
            n_blocks_per_head=n_global_blocks,
            block_size=block_size,
            group_size=group_size,
            batch_stride_kv_bytes=batch_stride_kv_bytes,
            batch_stride_dkv_bytes=batch_stride_dkv_bytes,
            stride_bn=stride_bn,
            stride_bd=stride_bd,
            stride_dbn=stride_dbn,
            stride_dbd=stride_dbd,
            blocks_per_chunk=blocks_per_chunk,
            in_chunk_block_bytes_kv=in_chunk_block_bytes_kv,
            in_chunk_block_bytes_dkv=in_chunk_block_bytes_dkv,
            softmax_scale=softmax_scale,
            input_precision=input_precision,
        )

        # All non-q inputs are non-differentiable (non-grad tensors or non-Tensors)
        return (
            dq,   # q
            None, # block_indices_remapped
            None, # n_chunks_per_head
            None, # K_gather
            None, # V_gather
            None, # chunk_base_ptrs_k
            None, # chunk_base_ptrs_v
            None, # meta
            None, # kv_cache
            None, # layer_idx
            None, # head_cid_lists
            None, # head_offsets
            None, # n_chunks_per_head_list
        )


def selection_attention_varlen_dense(
    q: torch.Tensor,                  # [B, M, H, D] bf16
    block_indices: torch.Tensor,      # [B, MG, H, T] int32 (cache-space)
    kv_cache,                         # MMCache
    layer_idx: int,
    n_chunks_cap: int,                # current_chunk_id + 1
    block_size: int,
    group_size: int,
    chunk_len: int,
    softmax_scale: Optional[float] = None,
    input_precision: str = "tf32",
) -> torch.Tensor:
    """Gather selected (h, cid) chunks, run padded_ptr Triton fwd/bwd, return per-query output.
    For streaming trainer's LRU grad path; cache content is stable within one train_step."""
    blocks_per_chunk = chunk_len // block_size

    head_cid_lists, block_indices_remapped, head_offsets, n_chunks_per_head_list, n_total = \
        _decode_block_indices_per_head(
            block_indices=block_indices,
            blocks_per_chunk=blocks_per_chunk,
            n_chunks_cap=n_chunks_cap,
        )

    B, M, H, D = q.shape

    k_store = kv_cache.cache[layer_idx]['k_cache']
    v_store = kv_cache.cache[layer_idx]['v_cache']

    K_gather, V_gather, chunk_base_ptrs_k, chunk_base_ptrs_v, _ = _gather_kv_to_dense(
        k_store, v_store,
        head_cid_lists, head_offsets,
        n_total=n_total,
        chunk_len=chunk_len,
        head_dim=D,
        batch_size=B,
        device=q.device,
        dtype=q.dtype,
    )

    n_chunks_per_head = torch.tensor(
        n_chunks_per_head_list, dtype=torch.int32, device=q.device,
    )

    if softmax_scale is None:
        softmax_scale = D ** -0.5

    meta = {
        'block_size': block_size,
        'group_size': group_size,
        'chunk_len': chunk_len,
        'softmax_scale': softmax_scale,
        'input_precision': input_precision,
    }

    return _SelectionAttentionVarlenDense.apply(
        q,
        block_indices_remapped,
        n_chunks_per_head,
        K_gather,
        V_gather,
        chunk_base_ptrs_k,
        chunk_base_ptrs_v,
        meta,
        kv_cache,
        layer_idx,
        head_cid_lists,
        head_offsets,
        n_chunks_per_head_list,
    )
