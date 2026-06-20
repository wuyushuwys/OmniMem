from typing import Optional
import math
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, _DEFAULT_SPARSE_BLOCK_SIZE, BlockMask, \
    and_masks

flex_attention = torch.compile(flex_attention, mode='max-autotune-no-cudagraphs')


def select_attention_torch(
        q: torch.Tensor,  # [B, M, H, D]
        k: torch.Tensor,  # [B, N, H, D] (Uncompressed keys)
        v: torch.Tensor,  # [B, N, H, D] (Uncompressed values)
        token_idx: torch.Tensor,  # [B, M, H, S] (Inflated token indices)
        scale: float = None,
        causal: bool = False,
) -> torch.Tensor:
    r"""Non-chunked PyTorch selective attention (memory-intensive; materializes full selected K/V).

    Args:
        q: [B, M, H, D]
        k: [B, N, H, D]
        v: [B, N, H, D]
        token_idx: [B, M, H, S] token indices to attend to per query
        scale: softmax scale (default 1/sqrt(D))
        causal: not implemented
    """

    batch_size, seq_len, num_heads, head_dims = q.size()
    if scale is None:
        scale = head_dims ** -0.5

    if causal:
        raise NotImplementedError(f"{causal=} not implemented")

    q, k, v, token_idx = map(lambda x: x.permute(0, 2, 1, 3).contiguous(), (q, k, v, token_idx))
    gather_index = token_idx[..., None].expand(-1, -1, -1, -1, head_dims)
    selected_k, selected_v = map(
        lambda kv: torch.gather(
            input=kv.unsqueeze(2).expand(-1, -1, seq_len, -1, -1),
            dim=3,
            index=gather_index,
        ),
        [k, v]
    )
    qk = torch.einsum('bhqd,bhqkd->bhqk', q.to(torch.float32) * scale, selected_k.to(torch.float32))
    attn_weights = qk.softmax(-1)
    o = torch.einsum('bhqk,bhqkd->bhqd', attn_weights, selected_v.to(torch.float32))
    o = o.permute(0, 2, 1, 3).contiguous().to(q.dtype)
    return o


def select_attention_torch_chunk(
        q: torch.Tensor,  # [B, M, H, D]
        k: torch.Tensor,  # [B, N, H, D] (Uncompressed keys)
        v: torch.Tensor,  # [B, N, H, D] (Uncompressed values)
        token_idx: torch.Tensor,  # [B, M, H, S] (Inflated token indices)
        scale: float = None,
        causal: bool = False,
        q_chunk_size: int = 32,
        kv_chunk_size: int = None,
        chunk_size=None,
) -> torch.Tensor:
    r"""Memory-efficient chunked PyTorch selective attention with online softmax.

    Args:
        q, k, v: [B, M/N, H, D]
        token_idx: [B, M, H, S] token indices to attend to per query
        scale: softmax scale (default 1/sqrt(D))
        q_chunk_size: query chunk size (default 32)
        kv_chunk_size: kv chunk size (default full select_len)
        chunk_size: required when causal=True
    """

    batch_size, q_len, num_heads, head_dims = q.size()
    select_len = token_idx.size(-1)
    if kv_chunk_size is None:
        kv_chunk_size = k.size(1)
    if scale is None:
        scale = head_dims ** -0.5

    if causal:
        assert chunk_size is not None, "chunk_size must be specified for causal mode"

    q, k, v, token_idx = map(lambda x: x.permute(0, 2, 1, 3).contiguous(), (q, k, v, token_idx))
    o = torch.empty_like(q)

    for q_start in range(0, q_len, q_chunk_size):
        q_end = min(q_start + q_chunk_size, q_len)
        q_chunk_len = q_end - q_start
        q_chunk = q[:, :, q_start:q_end, :]
        token_idx_i = token_idx[:, :, q_start:q_end, :]

        m_i = torch.full(
            (batch_size, num_heads, q_chunk_len),
            -torch.inf,
            device=q.device,
            dtype=torch.float32
        )
        l_i = torch.zeros(
            (batch_size, num_heads, q_chunk_len),
            device=q.device,
            dtype=torch.float32
        )
        o_i = torch.zeros(
            (batch_size, num_heads, q_chunk_len, head_dims),
            device=q.device,
            dtype=torch.float32
        )

        for kv_start in range(0, select_len, kv_chunk_size):
            kv_end = min(kv_start + kv_chunk_size, select_len)
            kv_chunk_len = kv_end - kv_start

            token_idx_kv = token_idx_i[..., kv_start: kv_end]
            gather_index = token_idx_kv[..., None].expand(-1, -1, -1, -1, head_dims)
            selected_k, selected_v = map(
                lambda x: torch.gather(
                    input=x.unsqueeze(2).expand(-1, -1, q_chunk_len, -1, -1),
                    dim=3,
                    index=gather_index
                ),
                [k, v]
            )

            qk = torch.einsum(
                "bhqd,bhqkd->bhqk",
                q_chunk.to(torch.float32) * scale,
                selected_k.to(torch.float32)
            )
            if causal:
                q_pos = torch.arange(q_start, q_end, device=q.device).view(1, 1, -1, 1)
                kv_pos = token_idx_kv
                causal_mask = (kv_pos // chunk_size) <= (q_pos // chunk_size)
                qk.masked_fill_(~causal_mask, float('-inf'))

            block_max = qk.max(dim=-1).values
            m_new = torch.maximum(m_i, block_max)
            alpha = torch.exp(m_i - m_new)
            p = torch.exp(qk - m_new.unsqueeze(-1))
            l_i = l_i * alpha + p.sum(dim=-1)
            o_i = o_i * alpha.unsqueeze(-1) + torch.einsum("bhqk,bhqkd->bhqd", p, selected_v.to(torch.float32))

            m_i = m_new

        o[:, :, q_start:q_end, :] = (o_i / l_i.unsqueeze(-1)).to(q.dtype)

    o = o.permute(0, 2, 1, 3).contiguous()
    return o


def top_idx_to_blockmask(
        top_idx: torch.Tensor,  # [B, H, M, T], compression KV block indices
        q_len: int,
        kv_len: int,
        router_kv_block_size: int,
        flex_block_size: int = _DEFAULT_SPARSE_BLOCK_SIZE,
        mask_mod=None,
        compute_q_blocks: bool = True,
) -> BlockMask:
    B, H, M, T = top_idx.shape
    device = top_idx.device
    top_idx = top_idx.to(torch.int64)

    num_q_blocks = math.ceil(q_len / flex_block_size)
    num_kv_blocks = math.ceil(kv_len / flex_block_size)

    # --- Step 1: compression block -> flex KV tile range ---
    first_flex = torch.div(top_idx * router_kv_block_size, flex_block_size, rounding_mode="floor")
    last_flex = torch.div(top_idx * router_kv_block_size + router_kv_block_size - 1,
                          flex_block_size, rounding_mode="floor").clamp_max(num_kv_blocks - 1)

    # --- Step 2: expand to all overlapping flex tiles ---
    max_expand = int((last_flex - first_flex + 1).max().item())
    offsets = torch.arange(max_expand, device=device, dtype=torch.int64)
    kv_tiles = first_flex.unsqueeze(-1) + offsets  # [B, H, M, T, max_expand]
    valid = kv_tiles <= last_flex.unsqueeze(-1)
    kv_tiles = torch.where(valid, kv_tiles, num_kv_blocks)  # sentinel
    kv_tiles = kv_tiles.reshape(B, H, M, T * max_expand)  # [B, H, M, L]

    # --- Step 3: scatter into dense grid [B*H*QB, num_kv_blocks+1] ---
    q_blk = torch.arange(M, device=device) // flex_block_size
    row = (
            torch.arange(B, device=device).view(B, 1, 1, 1) * (H * num_q_blocks)
            + torch.arange(H, device=device).view(1, H, 1, 1) * num_q_blocks
            + q_blk.view(1, 1, M, 1)
    )

    total_rows = B * H * num_q_blocks
    grid = torch.zeros(total_rows, num_kv_blocks + 1, device=device, dtype=torch.bool)
    grid[row.expand_as(kv_tiles).reshape(-1), kv_tiles.reshape(-1)] = True
    grid = grid[:, :num_kv_blocks]
    grid = grid.view(B, H, num_q_blocks, num_kv_blocks)

    # --- Step 4: extract kv_num_blocks & kv_indices ---
    kv_num_blocks = grid.sum(dim=-1, dtype=torch.int32)  # [B, H, QB]
    max_kept = max(int(kv_num_blocks.max().item()), 1)

    _, kv_indices = grid.to(torch.int32).topk(max_kept, dim=-1, sorted=False)
    kv_indices = kv_indices.to(torch.int32)

    # --- Fix: pad kv_indices last dim to num_kv_blocks ---
    # PyTorch's from_kv_blocks -> _ordered_to_dense uses kv_indices.shape[-1]
    # as dense matrix column count. If max_kept < num_kv_blocks, index OOB.
    if max_kept < num_kv_blocks:
        pad = torch.zeros(
            B, H, num_q_blocks, num_kv_blocks - max_kept,
            device=device, dtype=torch.int32
        )
        kv_indices = torch.cat([kv_indices, pad], dim=-1)

    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        BLOCK_SIZE=flex_block_size,
        mask_mod=mask_mod,
        seq_lengths=(q_len, kv_len),
        compute_q_blocks=compute_q_blocks,
    )


def selected_attention_flex(
        q: torch.Tensor,  # [B, M, H, D]
        k: torch.Tensor,  # [B, N, H, D] (Uncompressed keys)
        v: torch.Tensor,  # [B, N, H, D] (Uncompressed values)
        token_mask: torch.Tensor,  # [B, H, M, N//block_size]
        block_size: int,
        scale: float = None,
        causal: bool = False,
        topk_indices: Optional[torch.Tensor] = None,
        enable_ste=False,
        block_mask: BlockMask = None,
        _BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE
):
    r"""Selective attention via flex_attention; supports hard masking or STE soft masking.

    Args:
        q, k, v: [B, M/N, H, D]
        token_mask: [B, H, M, N//block_size] block-level mask (bool for hard, float for STE)
        block_size: KV block size
        enable_ste: use Straight-Through Estimator for differentiable routing
    """
    batch_size, q_len, num_heads, head_dims = q.shape
    kv_len = k.size(1)
    # q, k, v expected as [B, H, L, D]
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()

    if enable_ste:
        token_mask = token_mask.to(q.dtype)

        def score_mod(score, b, h, q_idx, kv_idx):
            block_idx = kv_idx // block_size
            router_weight = token_mask[b, h, q_idx, block_idx]  # --> 0 mask; 1 unmask
            ste_bias = torch.log(router_weight + 1.0e-6)  # --> log(0 + 1e-8) -> -inf; log(1) -> 0
            return score + ste_bias
    else:
        hard_mask = token_mask.to(torch.bool)

        def mask_mod(b, h, q_idx, kv_idx):
            block_idx = kv_idx // block_size
            return hard_mask[b, h, q_idx, block_idx]

        if block_mask is not None:
            merged_mask_mod = and_masks(block_mask.mask_mod, mask_mod)
        else:
            merged_mask_mod = mask_mod
        block_mask = create_block_mask(merged_mask_mod, B=batch_size, H=num_heads, Q_LEN=q_len, KV_LEN=kv_len,
                                       _compile=True)
        score_mod = None
    o = flex_attention(
        query=q,
        key=k,
        value=v,
        block_mask=block_mask,
        score_mod=score_mod,
        scale=scale,
    )
    o = o.permute(0, 2, 1, 3).contiguous()

    return o


if __name__ == "__main__":
    import torch
    from torch.nn.attention.flex_attention import flex_attention

    from omnimem.nsa.naive.index_selection import index_selection_topk, inflate_indices, index_selection_topk_mask

    flex_attention = torch.compile(flex_attention, mode='max-autotune-no-cudagraphs')


    BATCH = 1
    HEADS = 12
    SEQ_LEN = 4680
    BLOCK_SIZE = 4 * 6
    CMP_LEN = SEQ_LEN // BLOCK_SIZE
    HEAD_DIM = 128
    DTYPE = torch.float32  # torch.float32 for verify numerical accuracy
    DEVICE = 'cuda'
    topk = 24
    print(f"{BLOCK_SIZE=}")
    print(f"{CMP_LEN=}")
    q = torch.rand(BATCH, SEQ_LEN, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k = torch.rand(BATCH, SEQ_LEN, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v = torch.rand(BATCH, SEQ_LEN, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    k_cmp = k[:, :, None].reshape(BATCH, CMP_LEN, BLOCK_SIZE, HEADS, HEAD_DIM).mean(dim=2)
    v_cmp = v[:, :, None].reshape(BATCH, CMP_LEN, BLOCK_SIZE, HEADS, HEAD_DIM).mean(dim=2)

    print(q.shape, v.shape, k_cmp.shape)

    o1 = index_selection_topk(q, k_cmp, None, block_count=topk)

    print(o1.shape)
    indices = inflate_indices(o1, block_size=BLOCK_SIZE)
    print(indices.shape, o1.shape)

    indices_mask = index_selection_topk_mask(q=q, k_cmp=k_cmp, lse_cmp=None, block_count=topk)
    print(indices_mask.shape, indices_mask.dtype)

    indices_mask = indices_mask.to(torch.bool)


    def mask_mod(b, h, q_idx, kv_idx):
        block_idx = kv_idx // BLOCK_SIZE
        return indices_mask[b, h, q_idx, block_idx]


    block_mask1 = top_idx_to_blockmask(
        top_idx=o1.permute(0, 2, 1, 3),
        q_len=SEQ_LEN,
        kv_len=SEQ_LEN,
        router_kv_block_size=BLOCK_SIZE,
        mask_mod=mask_mod,
    )

    block_mask2 = create_block_mask(mask_mod, B=BATCH, H=HEADS, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, _compile=True)

    print("is mask correct", torch.equal(block_mask1.to_dense(), block_mask2.to_dense()))
    print("is mask correct", (block_mask1.to_dense().bool() != block_mask2.to_dense().bool()).sum())

