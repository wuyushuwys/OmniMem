from typing import Optional, Union
import time
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask
from diffusers.configuration_utils import register_to_config
from einops import rearrange
from omnimem.models.cache import MMCache
from omnimem.models.flex_attention_utils import create_causal_block_mask_cached
from omnimem.nsa.ops import (
    selection_attention,
    parallel_nsa_topk_grouped_heads,
    should_use_selection_attention,
)
from omnimem.nsa.ops.selection.padded_ptr import (
    selection_attention_padded_ptr_fast,
    selection_attention_padded_ptr_train,
    selection_attention_padded_ptr_train_fast,
    selection_attention_varlen_dense,
)
from omnimem.utils.logging_tool import get_logger
from .wan_model import (
    WanSelfAttention,
    WanCrossAttention,
    WanLayerNorm,
    sinusoidal_embedding_1d,
)
from .causal_wan_model import CausalWanModel

from .attention import attention

logger = get_logger()

__all__ = ['CausalWanNSAModel', 'CausalWanNSAttentionBlock']


def token_reorder(x, f, h, w, pool_f=1, pool_h=1, pool_w=1) -> torch.Tensor:
    """Reorder tokens to group spatial blocks for NSA pooling."""
    if pool_f == 1 and pool_h == 1 and pool_w == 1:
        return x
    else:
        x = rearrange(
            x,
            "b (f fs h hs w ws) c -> b (f h w fs hs ws) c",
            f=f // pool_f,
            h=h // pool_h,
            w=w // pool_w,
            fs=pool_f,
            hs=pool_h,
            ws=pool_w)
        return x


def token_restore_order(x, f, h, w, pool_f=1, pool_h=1, pool_w=1) -> torch.Tensor:
    """Inverse of token_reorder."""
    if pool_f == 1 and pool_h == 1 and pool_w == 1:
        return x
    else:
        x = rearrange(
            x,
            "b (f h w fs hs ws) c -> b (f fs h hs w ws) c ",
            f=f // pool_f,
            h=h // pool_h,
            w=w // pool_w,
            fs=pool_f,
            hs=pool_h,
            ws=pool_w
        )
        return x


@torch.compiler.disable()
def causal_reorder_rope_apply(x, grid_sizes, freqs, start_frame=0, pool_f=1, pool_h=1, pool_w=1):
    """Apply RoPE to reordered tokens with temporal offset start_frame."""
    n, c = x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    # loop over samples
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))

        # Build 3D freq grid then rearrange to match reordered token layout.
        freqs_grid = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1)
        freqs_i = rearrange(
            freqs_grid,
            '(f fs) (h hs) (w ws) d -> (f h w fs hs ws) 1 d',
            fs=pool_f,
            hs=pool_h,
            ws=pool_w
        )
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalNSSelfAttention(WanSelfAttention):

    def __init__(self,
                 pool_f=1,
                 pool_h=1,
                 pool_w=1,
                 num_selected_blocks=10,
                 gate_mode='sigmoid',
                 layer_idx=0,
                 chunk_length: Optional[int] = None,
                 group_q_size=1,
                 exclude_window_chunks=0,
                 exclude_sink_chunks=0,
                 progressive_exclude=False,
                 num_kv_head_groups=None,
                 skip_module=None,
                 tf=False,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        if len(args) >= 1:
            dim = args[0]
        else:
            dim = kwargs["dim"]
        if len(args) >= 2:
            num_heads = args[1]
        else:
            num_heads = kwargs["num_heads"]

        self.pool_f = pool_f
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.kv_block_size = int(pool_f * pool_h * pool_w)
        # Selection parameters
        self.num_selected_blocks = num_selected_blocks
        self.g_proj = nn.Linear(dim, 3 * num_heads)
        self.gate_mode = gate_mode
        self.chunk_length = chunk_length
        self.group_q_size = group_q_size
        self.exclude_window_chunks = exclude_window_chunks
        self.exclude_sink_chunks = exclude_sink_chunks
        self.use_triton = True
        self.layer_idx = layer_idx
        self.progressive_exclude = progressive_exclude
        self.num_kv_head_groups = num_kv_head_groups if num_kv_head_groups else self.num_heads
        self.skip_module = skip_module
        self.tf = tf

    def initialize_gate(self):
        torch.nn.init.normal_(self.g_proj.weight, mean=0., std=1.0e-3)
        target_prob = 0.5
        logit_val = torch.log(torch.tensor(target_prob / (1 - target_prob))).item()
        torch.nn.init.constant_(self.g_proj.bias, logit_val)

    def create_cmp_block(self, q_len, cmp_kv_len, cmp_kv_stride):
        return create_causal_block_mask_cached(
            block_size=self.chunk_length,
            B=None,
            H=None,
            Q_LEN=q_len if not self.tf else q_len // 2,
            KV_LEN=cmp_kv_len if not self.tf else cmp_kv_len // 2,
            use_flex_attention=True,
            torch_compile=True,
            kv_stride=cmp_kv_stride,
            teacher_forcing=self.tf,
        )

    def get_slc_valid_mask(self, seq_len, device):
        t = torch.arange(seq_len, device=device)
        chunk_ids = t // self.chunk_length
        valid = chunk_ids >= (self.exclude_window_chunks + self.exclude_sink_chunks)
        return valid[None, :, None, None]

    @torch.compiler.disable()
    def slc_attention_padded_ptr(
        self,
        roped_query,
        roped_key_cmp,
        kv_cache: MMCache,
    ):
        """
        Selection attention via GPU-resident chunk pointer metadata in MMCache.
        Returns zeros when sequence too short or metadata not yet populated (cold start).
        """
        s = roped_query.shape[1]

        if not (self.progressive_exclude or should_use_selection_attention(
            seq_len=roped_key_cmp.shape[1] * self.kv_block_size,
            chunk_size=s,
            exclude_window_chunks=self.exclude_window_chunks,
            exclude_sink_chunks=self.exclude_sink_chunks,
        )):
            return torch.zeros_like(roped_query)

        block_indices = parallel_nsa_topk_grouped_heads(
            q=roped_query, k=roped_key_cmp,
            num_kv_head_groups=self.num_kv_head_groups,
            block_counts=self.num_selected_blocks,
            block_size=self.kv_block_size,
            chunk_size=s,
            causal=False,
            group_size=self.group_q_size,
            exclude_window_chunks=self.exclude_window_chunks,
            exclude_sink_chunks=self.exclude_sink_chunks,
            progressive_exclude=self.progressive_exclude,
        ).to(torch.int32).contiguous()

        meta_k = kv_cache.get_chunk_metadata(self.layer_idx, 'k_cache')
        meta_v = kv_cache.get_chunk_metadata(self.layer_idx, 'v_cache')
        if meta_k is None or meta_v is None:
            return torch.zeros_like(roped_query)

        ptrs_k, n_chunks_k, strides = meta_k
        ptrs_v, _, _ = meta_v

        return selection_attention_padded_ptr_fast(
            q=roped_query,
            block_indices=block_indices,
            block_size=self.kv_block_size,
            group_size=self.group_q_size,
            chunk_len=strides['chunk_len'],
            chunk_base_ptrs_k=ptrs_k,
            chunk_base_ptrs_v=ptrs_v,
            n_chunks_per_head=n_chunks_k,
            chunk_strides=strides,
            verify_complete=kv_cache._has_cpu_chunks,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
        )

    def _slc_train_padded_ptr(self, roped_query, block_indices, kv_cache, n_chunks):
        """
        Training selection attention via padded_ptr.

        Uses selective CPU-to-GPU reload (only (h, cid) pairs actually targeted by
        block_indices) and lru_touch_per_head to prevent LRU thrash.
        n_chunks caps per-head chunk count to current_chunk_id+1, which is
        required for gradient-checkpoint metadata consistency across recompute.
        Falls back to selection_attention_padded_ptr_train on cold start (no ptr table yet).
        """
        k_store = kv_cache.cache[self.layer_idx]['k_cache']
        v_store = kv_cache.cache[self.layer_idx]['v_cache']
        H_n = len(k_store)
        device = roped_query.device

        # LRU path: use varlen-dense gather to avoid save_for_backward device-mismatch
        # across BSC blocks (cache mutates between gen-fwd and recompute).
        if getattr(kv_cache, 'lru_max_size', 0) > 0:
            chunk_len = next(
                (k_store[h][0].shape[1] for h in range(H_n) if k_store[h]),
                self.chunk_length,
            )
            return selection_attention_varlen_dense(
                q=roped_query,
                block_indices=block_indices,
                kv_cache=kv_cache,
                layer_idx=self.layer_idx,
                n_chunks_cap=n_chunks,
                block_size=self.kv_block_size,
                group_size=self.group_q_size,
                chunk_len=chunk_len,
            )

        meta_k = kv_cache.get_chunk_metadata(self.layer_idx, 'k_cache')
        meta_v = kv_cache.get_chunk_metadata(self.layer_idx, 'v_cache')

        if meta_k is not None and meta_v is not None:
            ptrs_k, n_chunks_k, strides = meta_k
            ptrs_v, _, _ = meta_v
            chunk_len = strides['chunk_len']
            blocks_per_chunk = chunk_len // self.kv_block_size

            import numpy as np
            bi_np = block_indices.cpu().numpy()           # [B, MG, H, T] int32
            cid_np = bi_np // blocks_per_chunk
            valid = (bi_np >= 0)
            reloaded = []
            for h in range(H_n):
                head_valid = valid[:, :, h, :]
                head_cids = cid_np[:, :, h, :][head_valid]
                if head_cids.size == 0:
                    continue
                unique_cids = np.unique(head_cids)
                cap = min(len(k_store[h]), n_chunks)
                for cid in unique_cids:
                    cid = int(cid)
                    if cid >= cap:
                        continue
                    if not k_store[h][cid].is_cuda:
                        k_store[h][cid] = k_store[h][cid].to(device, non_blocking=True)
                        v_store[h][cid] = v_store[h][cid].to(device, non_blocking=True)
                        reloaded.append((h, cid))
                    # LRU-touch to prevent critic-mode flush_offload from evicting immediately.
                    kv_cache.lru_touch_per_head(self.layer_idx, 'k_cache', h, cid)
                    kv_cache.lru_touch_per_head(self.layer_idx, 'v_cache', h, cid)

            if reloaded:
                kv_cache.notify_chunks_reloaded(self.layer_idx, 'k_cache', reloaded)
                kv_cache.notify_chunks_reloaded(self.layer_idx, 'v_cache', reloaded)
                ptrs_k, n_chunks_k, strides = kv_cache.get_chunk_metadata(self.layer_idx, 'k_cache')
                ptrs_v, _, _ = kv_cache.get_chunk_metadata(self.layer_idx, 'v_cache')

            n_chunks_per_head_list = [min(len(k_store[h]), n_chunks) for h in range(H_n)]
            k_flat = tuple(
                k_store[h][cid]
                for h in range(H_n)
                for cid in range(n_chunks_per_head_list[h])
            )
            v_flat = tuple(
                v_store[h][cid]
                for h in range(H_n)
                for cid in range(n_chunks_per_head_list[h])
            )

            return selection_attention_padded_ptr_train_fast(
                q=roped_query,
                block_indices=block_indices,
                block_size=self.kv_block_size,
                group_size=self.group_q_size,
                chunk_len=strides['chunk_len'],
                chunk_base_ptrs_k=ptrs_k,
                chunk_base_ptrs_v=ptrs_v,
                n_chunks_per_head=n_chunks_k,
                chunk_strides=strides,
                k_chunks_flat=k_flat,
                v_chunks_flat=v_flat,
                n_chunks_per_head_list=n_chunks_per_head_list,
            )
        else:
            # Cold start: ptr table not yet built — fall back to list-of-chunks API.
            chunk_len = next(
                (k_store[h][0].shape[1] for h in range(H_n) if k_store[h]),
                self.chunk_length,
            )
            return selection_attention_padded_ptr_train(
                q=roped_query,
                k_chunks=k_store,
                v_chunks=v_store,
                block_indices=block_indices,
                block_size=self.kv_block_size,
                group_size=self.group_q_size,
                chunk_len=chunk_len,
            )

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
            temb=None,
            timestep=None,
    ):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        g = self.g_proj(x)
        g = rearrange(g, '... (h d) -> ... h d', d=3)
        g = torch.clamp(g, min=-10.0, max=10.0)
        if self.gate_mode == 'sigmoid':
            g_cmp, g_slc, g_swa = g.sigmoid().unbind(3)
        elif self.gate_mode == 'softmax':
            g_cmp, g_slc, g_swa = g.softmax(dim=-1).unbind(3)
        else:
            raise NotImplementedError(f"{self.gate_mode=} not implemented")

        if getattr(self, '_log_gate', False):
            self._gate_cache = (g_cmp.detach(), g_slc.detach(), g_swa.detach())

        if isinstance(block_mask, BlockMask):
            assert self.chunk_length is not None, f"{self.chunk_length=} is not set"
            reorder_kwargs = dict(pool_f=self.pool_f, pool_h=self.pool_h, pool_w=self.pool_w)
            if seq_lens[0] < x.shape[1]:
                # teacher-forcing: apply RoPE to each half separately
                roped_query = torch.cat([ causal_reorder_rope_apply(_q, grid_sizes, freqs, start_frame=start_frame, **reorder_kwargs) for _q in torch.chunk(q, 2, dim=1)], dim=1)
                roped_key = torch.cat([ causal_reorder_rope_apply(_k, grid_sizes, freqs, start_frame=start_frame, **reorder_kwargs) for _k in torch.chunk(k, 2, dim=1)], dim=1)
            else:
                roped_query = causal_reorder_rope_apply(
                    q, grid_sizes, freqs, start_frame=start_frame,
                    **reorder_kwargs
                ).type_as(v)
                roped_key = causal_reorder_rope_apply(
                    k, grid_sizes, freqs, start_frame=start_frame,
                    **reorder_kwargs
                ).type_as(v)

            roped_key_cmp = roped_key.view(b, s // self.kv_block_size, self.kv_block_size, n, d).mean(2)
            v_cmp = v.view(b, s // self.kv_block_size, self.kv_block_size, n, d).mean(2)

            # sliding window attention
            o_swa = attention(q=roped_query, k=roped_key, v=v, block_mask=block_mask)

            # topk selection
            block_mask_cmp = self.create_cmp_block(
                q_len=roped_query.shape[1],
                cmp_kv_len=roped_key_cmp.shape[1],
                cmp_kv_stride=self.kv_block_size,
            )
            o_cmp = attention(
                q=roped_query,
                k=roped_key_cmp,
                v=v_cmp,
                block_mask=block_mask_cmp,
            )
            block_indices = parallel_nsa_topk_grouped_heads(
                q=roped_query, k=roped_key_cmp,
                num_kv_head_groups=self.num_kv_head_groups,
                block_counts=self.num_selected_blocks,
                block_size=self.kv_block_size,
                chunk_size=self.chunk_length,
                causal=True,
                group_size=self.group_q_size,
                exclude_window_chunks=self.exclude_window_chunks,
                exclude_sink_chunks=self.exclude_sink_chunks,
                progressive_exclude=self.progressive_exclude,
                tf_mask=self.tf,
            ).to(torch.int32).contiguous()
            o_slc = selection_attention(
                q=roped_query,
                k=roped_key,
                v=v,
                block_indices=block_indices,
                block_count=self.num_selected_blocks,
                chunk_size=self.chunk_length,
                block_size=self.kv_block_size,
                causal=True,
                variant='two-pass',
                group_size=self.group_q_size,
            )
                
            
            if self.skip_module == 'cmp':
                x = g_slc.unsqueeze(-1) * o_slc + g_swa.unsqueeze(-1) * o_swa
            elif self.skip_module == 'slc':
                x = g_cmp.unsqueeze(-1) * o_cmp + g_swa.unsqueeze(-1) * o_swa
            else:
                x = g_cmp.unsqueeze(-1) * o_cmp + g_slc.unsqueeze(-1) * o_slc + g_swa.unsqueeze(-1) * o_swa


        elif block_mask is None:
            reorder_kwargs = dict(pool_f=self.pool_f, pool_h=self.pool_h, pool_w=self.pool_w)
            roped_query = causal_reorder_rope_apply(
                q, grid_sizes, freqs, start_frame=start_frame,
                **reorder_kwargs
            ).type_as(v)
            roped_key = causal_reorder_rope_apply(
                k, grid_sizes, freqs, start_frame=start_frame,
                **reorder_kwargs
            ).type_as(v)
            h = grid_sizes[0, 1].to(torch.long)
            w = grid_sizes[0, 2].to(torch.long)
            start_frame_t = torch.as_tensor(start_frame, device=grid_sizes.device, dtype=torch.long)
            start_id = h * w * start_frame_t

            # Compressed (mean-pooled) K/V for global selection
            roped_key_cmp = kv_cache.update_cache(
                name='k_cmp_cache',
                hidden_state=roped_key.view(b, s // self.kv_block_size, self.kv_block_size, n, d).mean(2),
                layer_idx=layer_idx,
                start_id=start_id // self.kv_block_size
            )
            v_cmp = kv_cache.update_cache(
                name='v_cmp_cache',
                hidden_state=v.view(b, s // self.kv_block_size, self.kv_block_size, n, d).mean(2),
                layer_idx=layer_idx,
                start_id=start_id // self.kv_block_size
            )

            # Full K/V for sliding-window attention
            roped_key_swa = kv_cache.update_cache(
                name='k_cache',
                hidden_state=roped_key,
                layer_idx=layer_idx,
                start_id=start_id
            )
            v_swa = kv_cache.update_cache(
                name='v_cache',
                hidden_state=v,
                layer_idx=layer_idx,
                start_id=start_id
            )

            o_cmp = attention(q=roped_query, k=roped_key_cmp, v=v_cmp)

            o_swa = attention(q=roped_query, k=roped_key_swa, v=v_swa)
            if self.skip_module != 'slc':
                if not self.training:
                    o_slc = self.slc_attention_padded_ptr(
                        roped_query=roped_query,
                        roped_key_cmp=roped_key_cmp,
                        kv_cache=kv_cache,
                    )
                else:
                    block_indices = parallel_nsa_topk_grouped_heads(
                        q=roped_query,
                        k=roped_key_cmp,
                        num_kv_head_groups=self.num_kv_head_groups,
                        block_counts=self.num_selected_blocks,
                        block_size=self.kv_block_size,
                        chunk_size=s,
                        causal=False,
                        group_size=self.group_q_size,
                        exclude_window_chunks=self.exclude_window_chunks,
                        exclude_sink_chunks=self.exclude_sink_chunks,
                        progressive_exclude=self.progressive_exclude,
                    ).to(torch.int32).contiguous()
                    block_seqlen_k = kv_cache.block_seqlens['k_cache']
                    start_id_int = (start_id.item() if isinstance(start_id, torch.Tensor)
                                    else int(start_id))
                    n_chunks = start_id_int // block_seqlen_k + 1
                    o_slc = self._slc_train_padded_ptr(
                        roped_query=roped_query,
                        block_indices=block_indices,
                        kv_cache=kv_cache,
                        n_chunks=n_chunks,
                    )

            if self.skip_module == 'cmp':
                x = g_slc.unsqueeze(-1) * o_slc + g_swa.unsqueeze(-1) * o_swa
            elif self.skip_module == 'slc':
                x = g_cmp.unsqueeze(-1) * o_cmp + g_swa.unsqueeze(-1) * o_swa
            else:
                x = g_cmp.unsqueeze(-1) * o_cmp + g_slc.unsqueeze(-1) * o_slc + g_swa.unsqueeze(-1) * o_swa
        else:
            raise NotImplementedError

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanNSAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 layer_idx=None,
                 pool_f=1,
                 pool_h=6,
                 pool_w=4,
                 num_selected_blocks=10,
                 gate_mode='sigmoid',
                 chunk_length=None,
                 group_q_size=1,
                 exclude_window_chunks=0,
                 exclude_sink_chunks=0,
                 progressive_exclude=False,
                 num_kv_head_groups=None,
                 skip_module=None,
                 tf=False,
                 ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalNSSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps,
            pool_f=pool_f,
            pool_h=pool_h,
            pool_w=pool_w,
            num_selected_blocks=num_selected_blocks,
            gate_mode=gate_mode,
            layer_idx=layer_idx,
            chunk_length=chunk_length,
            group_q_size=group_q_size,
            exclude_window_chunks=exclude_window_chunks,
            exclude_sink_chunks=exclude_sink_chunks,
            progressive_exclude=progressive_exclude,
            num_kv_head_groups=num_kv_head_groups,
            skip_module=skip_module,
            tf=tf,
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
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
            temb=None,
            timestep=None,
    ):
        """x: [B,L,C]; e: [B,L1,6,C] time modulation."""
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
            temb=temb,
            timestep=timestep,
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


class CausalWanNSAModel(CausalWanModel):
    r"""Causal Wan backbone with NSA (compress + select + sliding-window) attention."""

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
            eps=1e-6,
            pool_f=1,
            pool_h=15,
            pool_w=2,
            num_selected_blocks=10,
            enable_nsa=True,
            gate_mode='sigmoid',
            chunk_length=None,
            group_q_size=1,
            exclude_window_chunks=0,
            exclude_sink_chunks=0,
            progressive_exclude=False,
            num_kv_head_groups=None,
            skip_module=None,
            tf=False,
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
        # nsa attributes
        self.pool_f = pool_f
        self.pool_h = pool_h
        self.pool_w = pool_w
        # re-initialize new causal blocks
        self.blocks = nn.ModuleList([
            CausalWanNSAttentionBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                window_size=window_size,
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
                layer_idx=layer_idx,
                pool_f=pool_f,
                pool_h=pool_h,
                pool_w=pool_w,
                num_selected_blocks=num_selected_blocks,
                gate_mode=gate_mode,
                chunk_length=chunk_length,
                group_q_size=group_q_size,
                exclude_window_chunks=exclude_window_chunks,
                exclude_sink_chunks=exclude_sink_chunks,
                progressive_exclude=progressive_exclude,
                num_kv_head_groups=num_kv_head_groups,
                skip_module=skip_module,
                tf=tf,
            ) for layer_idx in range(num_layers)
        ])
        self.init_weights()
        self.init_gate()

    def init_gate(self):
        for m in self.modules():
            if isinstance(m, CausalNSSelfAttention):
                m.initialize_gate()

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
        start_frame: temporal offset; teacher: optional teacher-forcing input.
        """
        if kv_cache is not None:
            assert 'k_cmp_cache' in kv_cache.available_type
            assert 'v_cmp_cache' in kv_cache.available_type
        if self.model_type == 'i2v':
            assert y is not None
        device = self.patch_embedding.weight.device
        self.freqs = self.get_rope_params(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long).expand(x.shape[0], -1)
        f, h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = token_reorder(
            x,
            f=f,
            h=h,
            w=w,
            pool_f=self.pool_f,
            pool_h=self.pool_h,
            pool_w=self.pool_w
        )

        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)

        teacher_forcing = teacher is not None
        if teacher_forcing:
            # embed teacher
            x_t = self.patch_embedding(teacher)
            x_t = x_t.flatten(2).transpose(1, 2).contiguous()
            x_t = token_reorder(
                x_t,
                f=f,
                h=h,
                w=w,
                pool_f=self.pool_f,
                pool_h=self.pool_h,
                pool_w=self.pool_w
            )
            x = torch.cat([x_t, x], dim=1)

        # time embeddings
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(-1, x.size(1))

        batch_size = x.shape[0]
        temb = sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(x.dtype)
        e = self.time_embedding(temb)
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
            temb=temb,
            timestep=t,
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
        x = token_restore_order(
            x,
            f=f,
            h=h,
            w=w,
            pool_f=self.pool_f,
            pool_h=self.pool_h,
            pool_w=self.pool_w
        )
        x = self.unpatchify(x, grid_sizes)
        return x