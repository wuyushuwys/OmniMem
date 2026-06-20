import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from typing import Optional
flex_attention = torch.compile(flex_attention, mode='max-autotune-no-cudagraphs')


def compress_attention_torch(
        q: torch.Tensor,  # [B, M, H, D]
        k_cmp: torch.Tensor,  # [B, N//block_size, H, D] - Pooled keys
        v_cmp: torch.Tensor,  # [B, N//block_size, H, D] - Pooled values
        block_mask: Optional[BlockMask] = None,
        scale: float = None,
        causal: bool = False,
        return_score=False,
        return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    r"""Native PyTorch compress attention. Returns (o_cmp, lse, attn_weights)."""
    if scale is None:
        scale = q.size(-1) ** -0.5

    if causal:
        q_perm = q.permute(0, 2, 1, 3)
        k_perm = k_cmp.permute(0, 2, 1, 3)
        v_perm = v_cmp.permute(0, 2, 1, 3)
        if return_score:
            qk = torch.einsum("bqhd,bkhd->bhqk", q * scale, k_cmp).contiguous()
            attend_mask = block_mask.to_dense()[:, :, :q.shape[1], :k_cmp.shape[1]]
            qk = qk.masked_fill(~attend_mask, float('-inf'))
            attn_weights = qk.softmax(-1)
            o_cmp = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v_cmp).contiguous()
            return o_cmp, None, attn_weights
        if return_lse:
            o_cmp, lse = flex_attention(q_perm, k_perm, v_perm,
                                        block_mask=block_mask, return_lse=True)
            return o_cmp.permute(0, 2, 1, 3), lse, None
        else:
            o_cmp = flex_attention(q_perm, k_perm, v_perm, block_mask=block_mask)
            return o_cmp.permute(0, 2, 1, 3), None, None
    if return_score:
        qk = torch.einsum("bqhd,bkhd->bhqk", q * scale, k_cmp).contiguous()
        if return_lse:
            lse = qk.logsumexp(dim=-1).transpose(-1, -2).contiguous()
        else:
            lse = None
        attn_weights = qk.softmax(-1)
        o_cmp = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v_cmp).contiguous()

        return o_cmp, lse, (attn_weights if return_score else None)
    else:
        q = q.permute(0, 2, 1, 3).contiguous()
        k_cmp = k_cmp.permute(0, 2, 1, 3).contiguous()
        v_cmp = v_cmp.permute(0, 2, 1, 3).contiguous()
        if return_lse:
            o_cmp, lse = flex_attention(q, k_cmp, v_cmp, return_lse=return_lse)
            o_cmp = o_cmp.permute(0, 2, 1, 3).contiguous()
            return o_cmp, lse, None
        else:
            o_cmp = flex_attention(q, k_cmp, v_cmp)
            o_cmp = o_cmp.permute(0, 2, 1, 3).contiguous()
            return o_cmp, None, None



