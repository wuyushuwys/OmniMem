"""
Plain (no-autograd) wrappers for padded-ptr selection attention.
Ptr arithmetic: base entries are int64 BYTE addresses; batch_stride in BYTES;
stride_bn/bd in ELEMENTS (Triton scales by element size on typed ptrs).
"""
from typing import List, Optional, Dict, Any
from collections import defaultdict
import torch
import triton

from .kernels.select_fwd import _sel_attn_fwd_padded_ptr_kernel, _compute_tile_n
from .ptr_builder import build_ptr_table


def _compute_needed_bids_per_head(block_indices: torch.Tensor) -> List[List[int]]:
    """Derive needed bids per head. Single GPU→CPU sync via whole-tensor move."""
    B, MG, H, T = block_indices.shape
    flat_cpu = block_indices.permute(2, 0, 1, 3).reshape(H, -1).cpu()
    needed = []
    for h in range(H):
        row = flat_cpu[h]
        bids = torch.unique(row[row >= 0]).tolist()
        needed.append(bids)
    return needed


def _build_block_base_ptrs_block_level(
    k_blocks: List[List[torch.Tensor]],
    v_blocks: List[List[torch.Tensor]],
    needed_bids_per_head: List[List[int]],
    device: torch.device,
    strict: bool = True,
):
    """Block-level storage (each block [B, BS, D]). Returns (base_k, base_v, n_blocks, batch_stride)."""
    H = len(k_blocks)
    n_blocks_list = [len(k_blocks[h]) for h in range(H)]
    max_n = max(n_blocks_list) if n_blocks_list else 0
    if max_n == 0:
        return None, None, None, 0

    base_k_cpu = torch.zeros(H, max_n, dtype=torch.int64)
    base_v_cpu = torch.zeros(H, max_n, dtype=torch.int64)
    n_blocks_cpu = torch.tensor(n_blocks_list, dtype=torch.int32)

    sample = None
    for h in range(H):
        bids = needed_bids_per_head[h]
        if not bids:
            continue
        n_h = n_blocks_list[h]
        k_head = k_blocks[h]
        v_head = v_blocks[h]
        valid_bids = []
        k_ptrs = []
        v_ptrs = []
        for bid in bids:
            if bid < 0 or bid >= n_h:
                continue
            k_t = k_head[bid]
            v_t = v_head[bid]
            if not k_t.is_cuda or not v_t.is_cuda:
                if strict:
                    raise RuntimeError(
                        f"k/v_blocks[{h}][{bid}] is not on GPU. "
                        f"Reload before calling selection_attention_padded_ptr, "
                        f"or pass strict=False to silently skip."
                    )
                continue
            valid_bids.append(bid)
            k_ptrs.append(k_t.data_ptr())
            v_ptrs.append(v_t.data_ptr())
            if sample is None:
                sample = k_t

        if valid_bids:
            bids_t = torch.tensor(valid_bids, dtype=torch.long)
            base_k_cpu[h].index_copy_(
                0, bids_t, torch.tensor(k_ptrs, dtype=torch.int64)
            )
            base_v_cpu[h].index_copy_(
                0, bids_t, torch.tensor(v_ptrs, dtype=torch.int64)
            )

    if sample is None:
        return None, None, None, 0

    batch_stride = sample.stride(0) * sample.element_size()
    return (
        base_k_cpu.to(device, non_blocking=True),
        base_v_cpu.to(device, non_blocking=True),
        n_blocks_cpu.to(device, non_blocking=True),
        batch_stride,
    )


def _build_block_base_ptrs_chunk_level(
    k_chunks: List[List[torch.Tensor]],
    v_chunks: List[List[torch.Tensor]],
    needed_bids_per_head: List[List[int]],
    chunk_len: int,
    block_size: int,
    device: torch.device,
    strict: bool = True,
):
    """Chunk-level storage; base[h, cid] = chunk ptr (no in-chunk offset).
    Returns (base_k, base_v, n_blocks_per_head, batch_stride, in_chunk_block_bytes, blocks_per_chunk)."""
    H = len(k_chunks)
    blocks_per_chunk = chunk_len // block_size
    n_chunks_list = [len(k_chunks[h]) for h in range(H)]
    max_n_chunks = max(n_chunks_list) if n_chunks_list else 0
    if max_n_chunks == 0:
        return None, None, None, 0, 0, 0

    base_k_cpu = torch.zeros(H, max_n_chunks, dtype=torch.int64)
    base_v_cpu = torch.zeros(H, max_n_chunks, dtype=torch.int64)
    n_global_blocks = torch.tensor(
        [nc * blocks_per_chunk for nc in n_chunks_list], dtype=torch.int32
    )

    sample = None
    token_stride_bytes = None
    batch_stride = None

    for h in range(H):
        bids = needed_bids_per_head[h]
        if not bids:
            continue
        n_global_h = n_chunks_list[h] * blocks_per_chunk
        needed_cids = sorted(set(
            bid // blocks_per_chunk for bid in bids
            if 0 <= bid < n_global_h
        ))
        if not needed_cids:
            continue

        k_head = k_chunks[h]
        v_head = v_chunks[h]
        valid_cids = []
        k_ptrs = []
        v_ptrs = []
        for cid in needed_cids:
            if cid >= len(k_head):
                continue
            k_chunk = k_head[cid]
            v_chunk = v_head[cid]
            if not k_chunk.is_cuda or not v_chunk.is_cuda:
                if strict:
                    raise RuntimeError(
                        f"k/v_chunks[{h}][{cid}] is not on GPU. "
                        f"Reload before calling selection_attention_padded_ptr, "
                        f"or pass strict=False to silently skip."
                    )
                continue
            if sample is None:
                sample = k_chunk
                elem = sample.element_size()
                batch_stride = sample.stride(0) * elem
                token_stride_bytes = sample.stride(1) * elem
            valid_cids.append(cid)
            k_ptrs.append(k_chunk.data_ptr())
            v_ptrs.append(v_chunk.data_ptr())

        if valid_cids:
            cids_t = torch.tensor(valid_cids, dtype=torch.long)
            base_k_cpu[h].index_copy_(
                0, cids_t, torch.tensor(k_ptrs, dtype=torch.int64)
            )
            base_v_cpu[h].index_copy_(
                0, cids_t, torch.tensor(v_ptrs, dtype=torch.int64)
            )

    if sample is None:
        return None, None, None, 0, 0, 0

    in_chunk_block_bytes = block_size * token_stride_bytes
    return (
        base_k_cpu.to(device, non_blocking=True),
        base_v_cpu.to(device, non_blocking=True),
        n_global_blocks.to(device, non_blocking=True),
        batch_stride,
        in_chunk_block_bytes,
        blocks_per_chunk,
    )


def _find_sample_block(
    k_blocks: List[List[torch.Tensor]],
    needed_bids_per_head: List[List[int]],
    chunk_len: Optional[int],
    block_size: int,
):
    """Find any GPU tensor for stride probing."""
    H = len(k_blocks)
    blocks_per_chunk = chunk_len // block_size if chunk_len is not None else None
    for h in range(H):
        for bid in needed_bids_per_head[h]:
            if chunk_len is None:
                if bid < 0 or bid >= len(k_blocks[h]):
                    continue
                t = k_blocks[h][bid]
            else:
                cid = bid // blocks_per_chunk
                if cid < 0 or cid >= len(k_blocks[h]):
                    continue
                t = k_blocks[h][cid]
            if t.is_cuda:
                return t
    return None


def _build_fresh_chunk_base_ptrs(
    template_k: torch.Tensor,
    template_v: torch.Tensor,
    kv_cache,
    layer_idx: int,
    device: torch.device,
):
    """Build fresh [H, max_n_chunks] int64 ptr table from live cache chunks (race-free).
    CPU-resident chunks get ptr=0; caller should reload via _touch_lru_and_detect_missing."""
    H, max_n = template_k.shape
    chunks_k = kv_cache.cache[layer_idx]['k_cache']
    chunks_v = kv_cache.cache[layer_idx]['v_cache']

    base_k_cpu = torch.zeros(H, max_n, dtype=torch.int64)
    base_v_cpu = torch.zeros(H, max_n, dtype=torch.int64)
    for h in range(H):
        head_k = chunks_k[h]
        head_v = chunks_v[h]
        n = min(len(head_k), len(head_v))
        for cid in range(n):
            k_t = head_k[cid]
            v_t = head_v[cid]
            if k_t.is_cuda and v_t.is_cuda:
                base_k_cpu[h, cid] = k_t.data_ptr()
                base_v_cpu[h, cid] = v_t.data_ptr()
    return (
        base_k_cpu.to(device, non_blocking=True),
        base_v_cpu.to(device, non_blocking=True),
    )


def _touch_lru_and_detect_missing(
    block_indices: torch.Tensor,
    chunk_base_ptrs_k: torch.Tensor,
    chunk_base_ptrs_v: torch.Tensor,
    blocks_per_chunk: int,
    kv_cache,
    layer_idx: int,
    device: torch.device,
    detect_missing: bool,
):
    """Touch LRU for all (h, cid) in block_indices; reload missing GPU chunks if detect_missing.
    Returns True if anything was reloaded (caller should rebuild ptr table)."""
    H = block_indices.shape[2]
    max_n = chunk_base_ptrs_k.shape[1]

    cids = (block_indices // blocks_per_chunk).clamp_(min=0).to(torch.int64)
    h_idx = (
        torch.arange(H, device=device, dtype=torch.int64)
        .view(1, 1, H, 1).expand_as(block_indices)
    )
    keys = h_idx * max_n + cids

    valid_mask = block_indices >= 0
    unique_keys = keys[valid_mask].unique()

    if detect_missing:
        ptrs_k = chunk_base_ptrs_k.flatten()[unique_keys]
        ptrs_v = chunk_base_ptrs_v.flatten()[unique_keys]
        missing_local = (ptrs_k == 0) | (ptrs_v == 0)
        any_missing = missing_local.any().item()
    else:
        any_missing = False

    unique_keys_cpu = unique_keys.cpu().tolist()

    for key in unique_keys_cpu:
        h = int(key // max_n)
        cid = int(key % max_n)
        kv_cache.lru_touch_per_head(layer_idx, 'k_cache', h, cid)

    if not any_missing:
        return False

    missing_local_cpu = missing_local.cpu().tolist()
    chunks_k = kv_cache.cache[layer_idx]['k_cache']
    chunks_v = kv_cache.cache[layer_idx]['v_cache']

    by_cid = defaultdict(list)
    for key, is_missing in zip(unique_keys_cpu, missing_local_cpu):
        if not is_missing:
            continue
        h = int(key // max_n)
        cid = int(key % max_n)
        if cid >= len(chunks_k[h]):
            continue
        assert chunks_k[h][cid].is_cuda == chunks_v[h][cid].is_cuda, \
            f"k/v desync at L{layer_idx} h={h} cid={cid}: " \
            f"k_cuda={chunks_k[h][cid].is_cuda} v_cuda={chunks_v[h][cid].is_cuda}, " \
            f"k_ptr={chunk_base_ptrs_k[h, cid].item()}, v_ptr={chunk_base_ptrs_v[h, cid].item()}, " \
            f"in_lru={kv_cache.lru_contains_per_head(layer_idx, 'k_cache', h, cid)}, " \
            f"k_chunks_len={len(chunks_k[h])}, v_chunks_len={len(chunks_v[h])}"
        by_cid[cid].append(h)

    if not by_cid:
        return False

    if hasattr(kv_cache, '_wait_offload'):
        kv_cache._wait_offload()

    all_h = []
    all_cid = []
    all_pk = []
    all_pv = []
    for cid, heads in by_cid.items():
        needs_reload = []
        for h in heads:
            k_t = chunks_k[h][cid]
            v_t = chunks_v[h][cid]
            if not k_t.is_cuda or not v_t.is_cuda:
                needs_reload.append(h)
            else:
                all_h.append(h)
                all_cid.append(cid)
                all_pk.append(k_t.data_ptr())
                all_pv.append(v_t.data_ptr())

        if not needs_reload:
            continue

        cpu_stack_k = torch.stack([chunks_k[h][cid] for h in needs_reload], dim=0)
        gpu_stack_k = cpu_stack_k.to(device, non_blocking=True)
        cpu_stack_v = torch.stack([chunks_v[h][cid] for h in needs_reload], dim=0)
        gpu_stack_v = cpu_stack_v.to(device, non_blocking=True)
        for idx, h in enumerate(needs_reload):
            k_view = gpu_stack_k[idx].clone()
            v_view = gpu_stack_v[idx].clone()
            chunks_k[h][cid] = k_view
            chunks_v[h][cid] = v_view
            all_h.append(h)
            all_cid.append(cid)
            all_pk.append(k_view.data_ptr())
            all_pv.append(v_view.data_ptr())

    h_t = torch.tensor(all_h, dtype=torch.long, device=device)
    c_t = torch.tensor(all_cid, dtype=torch.long, device=device)
    pk_t = torch.tensor(all_pk, dtype=torch.int64, device=device)
    pv_t = torch.tensor(all_pv, dtype=torch.int64, device=device)
    chunk_base_ptrs_k[h_t, c_t] = pk_t
    chunk_base_ptrs_v[h_t, c_t] = pv_t
    return True


def selection_attention_padded_ptr(
    q: torch.Tensor,
    k_blocks: List[List[torch.Tensor]],
    v_blocks: List[List[torch.Tensor]],
    block_indices: torch.Tensor,
    block_size: int,
    group_size: int = 1,
    chunk_len: Optional[int] = None,
    needed_bids_per_head: Optional[List[List[int]]] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    strict: bool = True,
    input_precision: str = "tf32",
):
    """Selection attention with padded ptr table (per-(b, mg, h), no concat/compact buffer)."""
    B, M, H, D = q.shape
    G = group_size
    MG = M // G
    T = block_indices.shape[-1]

    assert q.is_cuda
    assert q.dtype == torch.bfloat16, f"q must be bf16, got {q.dtype}"
    assert M % G == 0, f"M ({M}) must be divisible by group_size ({G})"
    assert block_indices.shape == (B, MG, H, T), \
        f"block_indices shape mismatch: got {block_indices.shape}, expected ({B}, {MG}, {H}, {T})"
    assert block_indices.dtype == torch.int32

    if softmax_scale is None:
        softmax_scale = D ** -0.5

    device = q.device

    if needed_bids_per_head is None:
        needed_bids_per_head = _compute_needed_bids_per_head(block_indices)

    blocks_per_chunk = 0
    in_chunk_block_bytes = 0
    if chunk_len is None:
        base_k, base_v, n_blocks_per_head, batch_stride = _build_block_base_ptrs_block_level(
            k_blocks, v_blocks, needed_bids_per_head, device, strict=strict,
        )
    else:
        assert chunk_len % block_size == 0, \
            f"chunk_len ({chunk_len}) must be multiple of block_size ({block_size})"
        result = _build_block_base_ptrs_chunk_level(
            k_blocks, v_blocks, needed_bids_per_head, chunk_len, block_size, device,
            strict=strict,
        )
        base_k, base_v, n_blocks_per_head, batch_stride, in_chunk_block_bytes, blocks_per_chunk = result

    if base_k is None:
        out = torch.zeros_like(q)
        if return_lse:
            lse = torch.full((B, H, M), float('-inf'), dtype=torch.float32, device=device)
            return out, lse
        return out

    ptr_table_k, ptr_table_v = build_ptr_table(
        block_indices=block_indices,
        block_base_ptrs_k=base_k,
        block_base_ptrs_v=base_v,
        n_blocks_per_head=n_blocks_per_head,
        batch_stride=batch_stride,
        blocks_per_chunk=blocks_per_chunk,
        in_chunk_block_bytes=in_chunk_block_bytes,
    )

    sample = _find_sample_block(k_blocks, needed_bids_per_head, chunk_len, block_size)
    assert sample is not None, "no GPU tensor found in k_blocks for stride probing"
    assert sample.is_contiguous(), (
        f"KV block/chunk must be contiguous; got shape={tuple(sample.shape)} "
        f"strides={tuple(sample.stride())}. Call .contiguous() before storing."
    )
    stride_bn = sample.stride(1)   # token stride (elements)
    stride_bd = sample.stride(2)   # dim stride (elements, typically 1)

    out = torch.empty_like(q)
    lse = torch.empty(B, H, M, dtype=torch.float32, device=device) if return_lse else q

    DP = max(16, triton.next_power_of_2(D))
    PADDED_BLOCK_SIZE = max(16, triton.next_power_of_2(block_size))
    PADDED_GROUP_SIZE = max(16, triton.next_power_of_2(G))
    TILE_N = _compute_tile_n(block_size)

    grid = (B, MG, H)

    _sel_attn_fwd_padded_ptr_kernel[grid](
        q, ptr_table_k, ptr_table_v, out, lse,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        ptr_table_k.stride(0), ptr_table_k.stride(1),
        ptr_table_k.stride(2), ptr_table_k.stride(3),
        lse.stride(0) if return_lse else 0,
        lse.stride(1) if return_lse else 0,
        lse.stride(2) if return_lse else 0,
        stride_bn, stride_bd,
        H=H, M=M, D=D, T=T, DP=DP,
        BLOCK_SIZE=block_size,
        PADDED_BLOCK_SIZE=PADDED_BLOCK_SIZE,
        GROUP_SIZE=G,
        PADDED_GROUP_SIZE=PADDED_GROUP_SIZE,
        TILE_N=TILE_N,
        RETURN_LSE=return_lse,
        INPUT_PRECISION=input_precision,
    )

    if return_lse:
        return out, lse
    return out


def selection_attention_padded_ptr_fast(
    q: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    group_size: int,
    chunk_len: int,
    chunk_base_ptrs_k: torch.Tensor,
    chunk_base_ptrs_v: torch.Tensor,
    n_chunks_per_head: torch.Tensor,
    chunk_strides: Dict[str, Any],
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    input_precision: str = "tf32",
    verify_complete: bool = False,
    kv_cache=None,
    layer_idx: Optional[int] = None,
):
    B, M, H, D = q.shape
    G = group_size
    MG = M // G
    T = block_indices.shape[-1]

    assert q.is_cuda and q.dtype == torch.bfloat16
    assert M % G == 0
    assert block_indices.shape == (B, MG, H, T)
    assert block_indices.dtype == torch.int32
    assert chunk_base_ptrs_k.dtype == torch.int64
    assert chunk_base_ptrs_v.dtype == torch.int64
    assert chunk_base_ptrs_k.shape[0] == H
    assert n_chunks_per_head.dtype == torch.int32
    assert n_chunks_per_head.shape == (H,)
    assert chunk_len % block_size == 0

    if softmax_scale is None:
        softmax_scale = D ** -0.5

    device = q.device
    blocks_per_chunk = chunk_len // block_size
    n_global_blocks = n_chunks_per_head * blocks_per_chunk

    batch_stride_bytes = chunk_strides['batch_stride_bytes']
    token_stride_bytes = chunk_strides['token_stride_bytes']
    stride_bn = chunk_strides['stride_bn']
    stride_bd = chunk_strides['stride_bd']
    in_chunk_block_bytes = block_size * token_stride_bytes

    # race-free rebuild: MMCache's async H2D can leave stale ptrs; rebuild from live chunks
    if kv_cache is not None and layer_idx is not None:
        chunk_base_ptrs_k, chunk_base_ptrs_v = _build_fresh_chunk_base_ptrs(
            chunk_base_ptrs_k, chunk_base_ptrs_v, kv_cache, layer_idx, device,
        )

    ptr_table_k, ptr_table_v = build_ptr_table(
        block_indices=block_indices,
        block_base_ptrs_k=chunk_base_ptrs_k,
        block_base_ptrs_v=chunk_base_ptrs_v,
        n_blocks_per_head=n_global_blocks,
        batch_stride=batch_stride_bytes,
        blocks_per_chunk=blocks_per_chunk,
        in_chunk_block_bytes=in_chunk_block_bytes,
    )

    if kv_cache is not None and layer_idx is not None:
        reloaded = _touch_lru_and_detect_missing(
            block_indices, chunk_base_ptrs_k, chunk_base_ptrs_v,
            blocks_per_chunk, kv_cache, layer_idx, device,
            detect_missing=verify_complete,
        )
        if reloaded:
            ptr_table_k, ptr_table_v = build_ptr_table(
                block_indices=block_indices,
                block_base_ptrs_k=chunk_base_ptrs_k,
                block_base_ptrs_v=chunk_base_ptrs_v,
                n_blocks_per_head=n_global_blocks,
                batch_stride=batch_stride_bytes,
                blocks_per_chunk=blocks_per_chunk,
                in_chunk_block_bytes=in_chunk_block_bytes,
            )

    out = torch.empty_like(q)
    lse = torch.empty(B, H, M, dtype=torch.float32, device=device) if return_lse else q

    DP = max(16, triton.next_power_of_2(D))
    PADDED_BLOCK_SIZE = max(16, triton.next_power_of_2(block_size))
    PADDED_GROUP_SIZE = max(16, triton.next_power_of_2(G))
    TILE_N = _compute_tile_n(block_size)

    grid = (B, MG, H)
    _sel_attn_fwd_padded_ptr_kernel[grid](
        q, ptr_table_k, ptr_table_v, out, lse,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        ptr_table_k.stride(0), ptr_table_k.stride(1),
        ptr_table_k.stride(2), ptr_table_k.stride(3),
        lse.stride(0) if return_lse else 0,
        lse.stride(1) if return_lse else 0,
        lse.stride(2) if return_lse else 0,
        stride_bn, stride_bd,
        H=H, M=M, D=D, T=T, DP=DP,
        BLOCK_SIZE=block_size,
        PADDED_BLOCK_SIZE=PADDED_BLOCK_SIZE,
        GROUP_SIZE=G,
        PADDED_GROUP_SIZE=PADDED_GROUP_SIZE,
        TILE_N=TILE_N,
        RETURN_LSE=return_lse,
        INPUT_PRECISION=input_precision,
    )

    if return_lse:
        return out, lse
    return out
