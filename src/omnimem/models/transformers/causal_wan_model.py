from typing import Optional, Union, Dict

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask

from diffusers.configuration_utils import register_to_config

from omnimem.models.cache import MMCache
from .wan_model import (
    WanModel,
    rope_params,
    rope_apply,
    WanSelfAttention,
    WanCrossAttention,
    WanLayerNorm,
    Head,
    sinusoidal_embedding_1d
)

from .attention import attention

__all__ = ['CausalWanModel', 'CausalWanAttentionBlock']


@torch.compiler.disable()
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """Apply RoPE with temporal offset start_frame for causal chunk generation."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalSelfAttention(WanSelfAttention):
    """Self-attention with causal RoPE and optional KV-cache for chunk-by-chunk generation."""

    def forward(
            self,
            x,
            seq_lens,
            grid_sizes,
            freqs,
            start_frame=0,
            kv_cache: MMCache = None,
            layer_idx=0,
            block_mask: Optional[Union[BlockMask, torch.Tensor]] = None,
    ):
        """
        x: input tensor; block_mask=BlockMask means training (standard RoPE + flex_attention);
        block_mask=None means inference (causal RoPE + KV-cache update).
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if isinstance(block_mask, BlockMask) or kv_cache is None:
            if seq_lens[0] < x.shape[1]:
                # teacher-forcing: apply RoPE to each half separately
                roped_query = torch.cat([ rope_apply(_q, grid_sizes, freqs) for _q in torch.chunk(q, 2, dim=1)], dim=1)
                roped_key = torch.cat([ rope_apply(_k, grid_sizes, freqs) for _k in torch.chunk(k, 2, dim=1)], dim=1)
            else:
                roped_query = rope_apply(q, grid_sizes, freqs)
                roped_key = rope_apply(k, grid_sizes, freqs)
        elif block_mask is None:
            roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=start_frame).type_as(v)
            roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=start_frame).type_as(v)
            h = grid_sizes[0, 1].to(torch.long)
            w = grid_sizes[0, 2].to(torch.long)
            start_frame_t = torch.as_tensor(start_frame, device=grid_sizes.device, dtype=torch.long)
            start_id = h * w * start_frame_t  # 0-dim int64 tensor
            roped_key = kv_cache.update_cache(name='k_cache', hidden_state=roped_key, layer_idx=layer_idx,
                                            start_id=start_id)
            v = kv_cache.update_cache(name='v_cache', hidden_state=v, layer_idx=layer_idx, start_id=start_id)
        else:
            raise NotImplementedError
        x = attention(
            q=roped_query,
            k=roped_key,
            v=v,
            block_mask=block_mask,
        )
        x = x.flatten(2)
        # output
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):
    """Transformer block with causal self-attention, cross-attention, FFN, and time modulation."""

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 layer_idx=None
                 ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
        self.layer_idx = layer_idx

    def forward(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            start_frame=None,
            kv_cache: MMCache = None,
            block_mask: Optional[Union[BlockMask, torch.Tensor]] = None,
    ):
        """x: [B,L,C]; e: [B,L1,6,C] time modulation; block_mask for attention."""
        # modulation
        e = (self.modulation[None] + e).unbind(-2)

        # self-attention
        y = self.self_attn(
            (self.norm1(x).float() * (1 + e[1]) + e[0]).to(x.dtype),
            seq_lens, grid_sizes,
            freqs,
            start_frame=start_frame,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
            block_mask=block_mask,
        )
        x = x + y * e[2]

        # cross-attention & ffn
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                (self.norm2(x).float() * (1 + e[4]) + e[3]).to(x.dtype))
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class CausalWanModel(WanModel):
    r"""Causal Wan backbone with CausalWanAttentionBlock for KV-cached chunk-by-chunk generation."""

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['CausalWanAttentionBlock']

    @register_to_config
    def __init__(
            self,
            model_type='t2v',
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=16,
            dim=2048,
            ffn_dim=8192,
            freq_dim=256,
            text_dim=4096,
            out_dim=16,
            num_heads=16,
            num_layers=32,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6
    ):
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(
                dim=dim, ffn_dim=ffn_dim, num_heads=num_heads,
                window_size=window_size, qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm, eps=eps,
                layer_idx=layer_idx
            ) for layer_idx in range(num_layers)
        ])

        self.init_weights()

    def forward(
            self,
            x,
            t,
            context,
            seq_len=None,
            start_frame=0,
            kv_cache: MMCache = None,
            block_mask: Optional[Union[BlockMask, torch.Tensor]] = None,
            y=None,
            teacher=None,
    ):
        """
        x: list of [C_in, F, H, W]; t: [B]; context: list of [L, C];
        start_frame: temporal offset for causal generation; teacher: optional teacher-forcing input.
        """
        if self.model_type == 'i2v':
            assert y is not None
        device = self.patch_embedding.weight.device
        self.freqs = self.get_rope_params(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long).expand(x.shape[0], -1)  # [b, 3] --> [[f, h, w],...]

        x = x.flatten(2).transpose(1, 2).contiguous()

        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)

        teacher_forcing = teacher is not None
        if teacher_forcing:
            x_t = self.patch_embedding(teacher)
            x_t = x_t.flatten(2).transpose(1, 2).contiguous()
            x = torch.cat([x_t, x], dim=1)

        # time embeddings
        batch_size = x.shape[0]
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(x.dtype))
        e = e.reshape(batch_size, -1, e.shape[-1])
        e0 = self.time_projection(e).reshape(batch_size, -1, 6, self.dim)
        
        if teacher_forcing:
            # embed teacher at timestep 0
            e_t = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, torch.zeros_like(t.flatten())).to(x.dtype))
            e_t = e_t.reshape(batch_size, -1, e_t.shape[-1])
            e0_t = self.time_projection(e_t).reshape(batch_size, -1, 6, self.dim)
            e = torch.cat([e_t, e], dim=1)
            e0 = torch.cat([e0_t, e0], dim=1)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )

        causal_kwargs = dict(
            start_frame=start_frame,
            kv_cache=kv_cache,
            block_mask=block_mask,
        )
        for layer_idx, block in enumerate(self.blocks):
            x = block(
                x, **kwargs,
                **causal_kwargs,
            )

        # head
        x = self.head(x, e)

        if teacher_forcing:
            x = x[:, x.shape[1]//2:]

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x