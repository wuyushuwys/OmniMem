from typing import Union
import torch
from omnimem.nsa.naive.compress_attention import compress_attention_torch


def index_selection_topk(
        q: torch.Tensor,  # [B, M, H, D]
        k_cmp: torch.Tensor,  # [B, N//block_size, G, D]
        lse_cmp: torch.Tensor,  # [B, M, H]
        block_count: int,  # 'T' blocks to select
        scale: float = None,
        causal: bool = False,
        chunk_size=None,
        block_size=None,
        return_mask=False,
        enable_ste=False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """TopK index selection. Returns top_idx [B, M, G, block_count]."""
    B, T, HQ, D = q.shape
    _, TC, H, _ = k_cmp.shape

    if scale is None:
        scale = q.size(-1) ** -0.5

    q, k_cmp = map(lambda x: x.permute(0, 2, 1, 3).contiguous(), (q, k_cmp))
    score = q @ k_cmp.transpose(-2, -1) * scale

    if causal:
        q_idx = torch.arange(T, device=q.device).view(1, 1, -1, 1)
        k_idx = torch.arange(TC, device=q.device).view(1, 1, 1, -1)

        q_chunk = q_idx // chunk_size
        k_chunk = (k_idx * block_size) // chunk_size

        mask = q_chunk >= k_chunk
        score.masked_fill_(~mask, float('-inf'))

    score = score.permute(0, 2, 1, 3)

    top_idx = torch.topk(score, block_count, dim=-1).indices
    if return_mask:
        hard_mask = torch.zeros_like(score, dtype=torch.bool)
        hard_mask.scatter_(dim=-1, index=top_idx, value=True)

        if enable_ste:
            soft_mask = score.softmax(dim=-1)
            mask = hard_mask.float().detach() - soft_mask.detach() + soft_mask
        else:
            mask = hard_mask
        return top_idx, mask
    else:
        top_idx, _ = torch.sort(top_idx, dim=-1)
        top_idx = top_idx.contiguous().to(torch.int32)
        return top_idx


def index_selection_topk_mask(
        q: torch.Tensor,  # [B, M, H, D]
        k_cmp: torch.Tensor,  # [B, N//block_size, G, D]
        lse_cmp: torch.Tensor,  # [B, M, H]
        block_count: int,  # 'T' blocks to select
        scale: float = None,
        causal: bool = False,
        enable_ste: bool = False,
        chunk_size=None,
        block_size=None,
) -> torch.Tensor:
    """TopK index selection returning block mask [B, H, M, N//block_size]."""
    B, T, HQ, D = q.shape
    _, TC, H, _ = k_cmp.shape

    if scale is None:
        scale = q.size(-1) ** -0.5

    score = torch.einsum("bqhd,bkhd->bhqk", q.to(torch.float32) * scale, k_cmp.to(torch.float32))

    if causal:
        q_idx = torch.arange(T, device=q.device).view(1, 1, -1, 1)
        k_idx = torch.arange(TC, device=q.device).view(1, 1, 1, -1)

        q_chunk = q_idx // chunk_size
        k_chunk = (k_idx * block_size) // chunk_size

        mask = q_chunk >= k_chunk
        score.masked_fill_(~mask, float('-inf'))

    assert block_count <= score.size(-1), f"{score.size(-1)=} < {block_count=}"
    top_idx = torch.topk(score, block_count, dim=-1).indices

    hard_mask = torch.zeros_like(score, dtype=torch.bool)
    hard_mask.scatter_(dim=-1, index=top_idx, value=True)

    if enable_ste:
        soft_mask = score.softmax(dim=-1)
        mask = hard_mask.float().detach() - soft_mask.detach() + soft_mask
    else:
        mask = hard_mask

    return mask


def index_selection_topk_mask_from_score(
        score: torch.Tensor,  # [B, N//block_size, G, D]
        block_count: int,  # 'T' blocks to select
        scale: float = None,
        causal: bool = False,
        enable_ste: bool = False,
) -> torch.Tensor:
    """TopK selection from a precomputed score tensor. Returns block mask."""
    if causal:
        raise NotImplementedError(f"causal not implemented yet")

    assert block_count <= score.size(-1), f"{score.size(-1)=} < {block_count=}"
    top_idx = torch.topk(score, block_count, dim=-1).indices

    hard_mask = torch.zeros_like(score, dtype=torch.bool)
    hard_mask.scatter_(dim=-1, index=top_idx, value=True)

    if enable_ste:
        soft_mask = score.softmax(dim=-1)
        mask = hard_mask.float().detach() - soft_mask.detach() + soft_mask
    else:
        mask = hard_mask

    return mask


def index_selection_topk_torch(
        q: torch.Tensor,  # [B, M, H, D]
        k_cmp: torch.Tensor,  # [B, N//block_size, G, D]
        v_cmp: torch.Tensor,  # [B, N//block_size, G, D]
        block_count: int,  # 'T' blocks to select
        lse_cmp: torch.Tensor = None,  # [B, M, H]
        scale: float = None,
        causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TopK selection with attention computation. Returns (o_cmp, top_idx)."""
    o_cmp, _, attn_weights = compress_attention_torch(
        q=q,
        k_cmp=k_cmp,
        v_cmp=v_cmp,
        scale=scale,
        causal=causal,
        return_score=True,
        return_lse=False,
    )

    top_idx = torch.topk(attn_weights, block_count, dim=-1).indices
    top_idx, _ = torch.sort(top_idx, dim=-1)
    top_idx = top_idx.permute(0, 2, 1, 3).contiguous().to(torch.int32)

    return o_cmp, top_idx


def inflate_indices(
        top_idx: torch.Tensor,  # [B, M, H, T]
        block_size: int
) -> torch.Tensor:
    """Inflate block-level indices to token-level. Returns [B, M, H, T * block_size]."""
    batch_size, seq_len, num_heads, num_blocks = top_idx.shape
    offsets = torch.arange(block_size, device=top_idx.device)
    start_idx = (top_idx * block_size).unsqueeze(-1)
    token_idx = start_idx + offsets
    token_idx = token_idx.view(batch_size, seq_len, num_heads, num_blocks * block_size)

    return token_idx


if __name__ == "__main__":
    BATCH = 1
    HEADS = 12
    SEQ_LEN = 8192
    BLOCK_SIZE = 128
    CMP_LEN = SEQ_LEN // BLOCK_SIZE
    HEAD_DIM = 128
    DTYPE = torch.float32
    DEVICE = 'cuda'
    topk = 16
    print(f"{CMP_LEN=}")
    q = torch.randn(BATCH, SEQ_LEN, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k_cmp = torch.randn(BATCH, CMP_LEN, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_cmp = torch.randn(BATCH, CMP_LEN, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    o1 = index_selection_topk(q, k_cmp, None, block_count=topk)
    _, o2 = index_selection_topk_torch(q, k_cmp, v_cmp, block_count=topk)
    is_identical = torch.equal(o1, o2)
    print(f"{is_identical=}")
    print(f"{o1.shape}")
