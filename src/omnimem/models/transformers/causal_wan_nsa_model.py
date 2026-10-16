from typing import Optional, Union
from collections import OrderedDict
import itertools
import weakref
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from omnimem.nsa.ops.selection.padded_ptr.block_wrapper import (
    selection_attention_block_store,
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
    patch_embed_linear,
    patch_embed_linear_enabled,
    rope_finish,
    rope_tail,
)
from .causal_wan_model import CausalWanModel

from .attention import attention

logger = get_logger()

__all__ = ['CausalWanNSAModel', 'CausalWanNSAttentionBlock',
           'ATTENTION_SCHEDULES', 'set_nsa_attention_schedule', 'get_nsa_attention_schedule',
           'set_selection_trace', 'set_selection_trace_context',
           'causal_reorder_rope_apply']

# Inference-only attention schedule: serial (default) or dual_topk_first.
ATTENTION_SCHEDULES = ("serial", "dual_topk_first")
_AUX_STREAMS = {}

# Diagnostic only: when set to a list, every top-k result is appended to it.
_SELECTION_TRACE = None

# Diagnostic only: tags attached to every trace entry; never read by the forward path.
_SELECTION_TRACE_CONTEXT: dict = {}


def set_selection_trace(trace):
    """Enable (list) or disable (None) recording of selected block indices."""
    global _SELECTION_TRACE
    _SELECTION_TRACE = trace


def set_selection_trace_context(**tags):
    """Replace the tag dict attached to subsequent selection-trace entries."""
    global _SELECTION_TRACE_CONTEXT
    _SELECTION_TRACE_CONTEXT = dict(tags)


def _aux_stream_for(device: torch.device) -> torch.cuda.Stream:
    key = (device.type, device.index)
    stream = _AUX_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _AUX_STREAMS[key] = stream
    return stream


def set_nsa_attention_schedule(model: nn.Module, schedule: str) -> int:
    """Set the schedule on every NSA self-attention module; returns how many were set."""
    if schedule not in ATTENTION_SCHEDULES:
        raise ValueError(f"invalid attention schedule {schedule!r}; expected one of {ATTENTION_SCHEDULES}")
    n = 0
    for m in model.modules():
        if isinstance(m, CausalNSSelfAttention):
            m.attention_schedule = schedule
            n += 1
    return n


def get_nsa_attention_schedule(model: nn.Module) -> str:
    seen = {m.attention_schedule for m in model.modules() if isinstance(m, CausalNSSelfAttention)}
    if not seen:
        return "n/a"
    return seen.pop() if len(seen) == 1 else "mixed"


def block_mean(x, fixed_order: bool):
    """x.mean(2) of [b, blocks, block_size, n, d]; fixed_order pins the sum order."""
    if not fixed_order:
        return x.mean(2)
    b, blocks, block_size, n, d = x.shape
    size = 1 << max(block_size - 1, 1).bit_length()
    y = F.pad(x.float(), (0, 0, 0, 0, 0, size - block_size))
    y = y.view(b, blocks, 2, size // 2, n, d)
    half = size // 2
    while half > 1:
        half //= 2
        y = y[:, :, :, :half] + y[:, :, :, half:]
    return ((y[:, :, 0, 0] + y[:, :, 1, 0]) * (1.0 / block_size)).to(x.dtype)


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


# RoPE position-factor cache keyed on (freqs version, c, f, h, w, start_frame, pool).
_ROPE_FACTOR_CACHE_CAPACITY = 16                    # entries
_ROPE_FACTOR_CACHE_MAX_BYTES = 128 * 1024 * 1024    # bytes
_ROPE_FACTOR_CACHE = OrderedDict()

# id(freqs tensor) -> version, kept honest by a weakref callback that drops the
# entry while the dying tensor is still the owner of that id.
_FREQS_VERSIONS = {}
_FREQS_VERSION_REFS = {}
_FREQS_VERSION_SEQ = itertools.count(1)


def _freqs_version(freqs: torch.Tensor) -> int:
    """A process-unique version for this ``freqs`` tensor *object*."""
    key = id(freqs)
    version = _FREQS_VERSIONS.get(key)
    if version is not None:
        return version
    version = next(_FREQS_VERSION_SEQ)
    try:
        def _drop(_ref, _key=key):
            _FREQS_VERSIONS.pop(_key, None)
            _FREQS_VERSION_REFS.pop(_key, None)
        _FREQS_VERSION_REFS[key] = weakref.ref(freqs, _drop)
    except TypeError:  # pragma: no cover - torch.Tensor is weak-referenceable
        return version  # never memoise: every call gets a fresh, non-matching version
    _FREQS_VERSIONS[key] = version
    return version


def _build_rope_position_factor(freqs, c, f, h, w, start_frame, pool_f, pool_h, pool_w):
    """The original in-loop grid build + rearrange, unchanged."""
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # Build 3D freq grid then rearrange to match reordered token layout.
    freqs_grid = torch.cat([
        freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ],
        dim=-1)
    return rearrange(
        freqs_grid,
        '(f fs) (h hs) (w ws) d -> (f h w fs hs ws) 1 d',
        fs=pool_f,
        hs=pool_h,
        ws=pool_w
    )


def _rope_position_factor(freqs, c, f, h, w, start_frame, pool_f, pool_h, pool_w):
    """Cached ``freqs_i`` for one sample; identical tensor on every hit."""
    device = freqs.device
    key = (_freqs_version(freqs), device.type, device.index, c,
           f, h, w, start_frame, pool_f, pool_h, pool_w)
    cached = _ROPE_FACTOR_CACHE.get(key)
    if cached is not None:
        _ROPE_FACTOR_CACHE.move_to_end(key)
        return cached
    value = _build_rope_position_factor(freqs, c, f, h, w, start_frame,
                                        pool_f, pool_h, pool_w)
    nbytes = value.numel() * value.element_size()
    if nbytes > _ROPE_FACTOR_CACHE_MAX_BYTES:
        # A single factor larger than the whole budget is never cached; the
        # caller still gets a correct, freshly built tensor.
        return value
    _ROPE_FACTOR_CACHE[key] = value
    total = nbytes + sum(v.numel() * v.element_size()
                         for k, v in _ROPE_FACTOR_CACHE.items() if k != key)
    while len(_ROPE_FACTOR_CACHE) > 1 and (
            len(_ROPE_FACTOR_CACHE) > _ROPE_FACTOR_CACHE_CAPACITY
            or total > _ROPE_FACTOR_CACHE_MAX_BYTES):
        _, victim = _ROPE_FACTOR_CACHE.popitem(last=False)
        total -= victim.numel() * victim.element_size()
    return value


@torch.compiler.disable()
def causal_reorder_rope_apply(x, grid_sizes, freqs, start_frame=0, pool_f=1, pool_h=1, pool_w=1):
    """Apply RoPE to reordered tokens with temporal offset start_frame."""
    n, c = x.size(2), x.size(3) // 2
    # Cache unless start_frame is not a plain int.
    position_factor = (_rope_position_factor if isinstance(start_frame, int)
                       else _build_rope_position_factor)
    output = []
    # loop over samples
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))

        freqs_i = position_factor(freqs, c, f, h, w, start_frame, pool_f, pool_h, pool_w)
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # ``rope_tail`` / ``rope_finish`` elide the empty-tail cat and the
        # single-element stack (exact, see wan_model).
        x_i = rope_tail(x_i, x, i, seq_len)
        # append to collection
        output.append(x_i)
    return rope_finish(output, x)


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
        # inference-only schedule, see set_nsa_attention_schedule()
        self.attention_schedule = 'serial'

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
        q_chunk_offset: int = 0,
    ):
        """
        Selection attention via GPU-resident chunk pointer metadata in MMCache.
        Returns zeros when sequence too short or metadata not yet populated (cold start).
        """
        block_indices = self._slc_topk_indices(roped_query, roped_key_cmp, q_chunk_offset)
        if block_indices is None:
            return torch.zeros_like(roped_query)
        return self._slc_from_indices(roped_query, block_indices, kv_cache)

    @torch.compiler.disable()
    def _slc_topk_indices(self, roped_query, roped_key_cmp, q_chunk_offset: int = 0):
        """Grouped-head top-k block selection; None while the sequence is too short."""
        s = roped_query.shape[1]

        if not (self.progressive_exclude or should_use_selection_attention(
            seq_len=roped_key_cmp.shape[1] * self.kv_block_size,
            chunk_size=s,
            exclude_window_chunks=self.exclude_window_chunks,
            exclude_sink_chunks=self.exclude_sink_chunks,
        )):
            return None

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
            q_chunk_offset=q_chunk_offset,
        ).to(torch.int32).contiguous()
        if _SELECTION_TRACE is not None:
            _SELECTION_TRACE.append((self.layer_idx, block_indices.to("cpu", copy=True),
                                     dict(_SELECTION_TRACE_CONTEXT)))
        return block_indices

    @torch.compiler.disable()
    def _slc_from_indices(self, roped_query, block_indices, kv_cache: MMCache):
        """Pointer-table selection attention (LRU touch, host inspection, reload, SLC)."""
        if kv_cache.block_store is not None:
            if self.training or torch.is_grad_enabled():
                # Block store is inference only; the chunk-LRU path would see all-zero pointers.
                raise RuntimeError(
                    f"L{self.layer_idx}: kv_cache_policy='block_lru' is inference-only "
                    f"(training={self.training}, grad={torch.is_grad_enabled()}).  Use "
                    f"torch.no_grad() + module.eval(), or configure_block_store(0, 0) "
                    f"to fall back to the chunk-LRU policy")
            out = selection_attention_block_store(
                q=roped_query,
                block_indices=block_indices,
                block_size=self.kv_block_size,
                group_size=self.group_q_size,
                kv_cache=kv_cache,
                layer_idx=self.layer_idx,
            )
            return out

        meta_k = kv_cache.get_chunk_metadata(self.layer_idx, 'k_cache')
        meta_v = kv_cache.get_chunk_metadata(self.layer_idx, 'v_cache')
        if meta_k is None or meta_v is None:
            return torch.zeros_like(roped_query)

        ptrs_k, n_chunks_k, strides = meta_k
        ptrs_v, _, _ = meta_v

        out = selection_attention_padded_ptr_fast(
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
        return out

    def _dual_stream_active(self, roped_query) -> bool:
        return (self.attention_schedule != 'serial'
                and not self.training
                and not torch.is_grad_enabled()
                and roped_query.is_cuda
                and self.skip_module != 'slc')

    @torch.compiler.disable()
    def _nsa_branches_dual(self, roped_query, roped_key_cmp, v_cmp, roped_key_swa, v_swa,
                           kv_cache: MMCache, q_chunk_offset: int):
        """Inference-only two-stream schedule of the three NSA branches."""
        device = roped_query.device
        main = torch.cuda.current_stream(device)
        aux = _aux_stream_for(device)

        shared_ready = torch.cuda.Event()
        shared_ready.record(main)
        block_indices = self._slc_topk_indices(roped_query, roped_key_cmp, q_chunk_offset)

        dense_done = torch.cuda.Event()
        aux.wait_event(shared_ready)
        try:
            with torch.cuda.stream(aux):
                o_cmp = attention(q=roped_query, k=roped_key_cmp, v=v_cmp)
                o_swa = attention(q=roped_query, k=roped_key_swa, v=v_swa)

            if block_indices is None:
                o_slc = torch.zeros_like(roped_query)
            else:
                o_slc = self._slc_from_indices(roped_query, block_indices, kv_cache)
        finally:
            # Nothing is dispatched to the aux stream after the two attentions, so this
            # marks the end of the dense work wherever in the block control left it.
            dense_done.record(aux)
            main.wait_event(dense_done)
        o_cmp.record_stream(main)
        o_swa.record_stream(main)
        return o_cmp, o_swa, o_slc

    def _slc_train_padded_ptr(self, roped_query, block_indices, kv_cache, n_chunks):
        """
        Training selection attention via padded_ptr.

        Uses selective CPU-to-GPU reload (only (h, cid) pairs actually targeted by
        block_indices) and lru_touch_per_head to prevent LRU thrash.
        n_chunks caps per-head chunk count to current_chunk_id+1, which is
        required for gradient-checkpoint metadata consistency across recompute.
        Falls back to selection_attention_padded_ptr_train on cold start (no ptr table yet).
        """
        if getattr(kv_cache, 'block_store', None) is not None:
            # Training never enters the block store: its chunk base table would be all zero.
            raise RuntimeError(
                f"L{self.layer_idx}: kv_cache_policy='block_lru' is inference-only; the "
                f"training selection path needs the chunk-LRU ptr tables.  Call "
                f"configure_block_store(0, 0) before training with this cache")
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
                # One publish batch for K and V together (shared ring slot, H2D
                # and scatter) instead of two per-cid staged notifications.
                kv_cache.publish_chunk_ptrs(self.layer_idx, (
                    (name, h, cid, MMCache.chunk_ptr_of(store[h][cid]))
                    for name, store in (('k_cache', k_store), ('v_cache', v_store))
                    for h, cid in reloaded
                ))
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
                max_chunks=ptrs_k.shape[1],
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
            start_id_int = int(start_id)
            cmp_start_id = start_id // self.kv_block_size
            q_chunk_offset = start_id_int // s

            # Compressed (mean-pooled) K/V for global selection.
            fixed_order = (torch.compiler.is_compiling() and not self.training
                           and not torch.is_grad_enabled())
            roped_key_cmp = kv_cache.update_cache(
                name='k_cmp_cache',
                hidden_state=block_mean(
                    roped_key.view(b, s // self.kv_block_size, self.kv_block_size, n, d), fixed_order),
                layer_idx=layer_idx,
                start_id=cmp_start_id
            )
            v_cmp = kv_cache.update_cache(
                name='v_cmp_cache',
                hidden_state=block_mean(
                    v.view(b, s // self.kv_block_size, self.kv_block_size, n, d), fixed_order),
                layer_idx=layer_idx,
                start_id=cmp_start_id
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

            if self._dual_stream_active(roped_query):
                o_cmp, o_swa, o_slc = self._nsa_branches_dual(
                    roped_query, roped_key_cmp, v_cmp, roped_key_swa, v_swa,
                    kv_cache, q_chunk_offset,
                )
            else:
                o_cmp = attention(q=roped_query, k=roped_key_cmp, v=v_cmp)

                o_swa = attention(q=roped_query, k=roped_key_swa, v=v_swa)
            if self.skip_module != 'slc' and not self._dual_stream_active(roped_query):
                if not self.training:
                    o_slc = self.slc_attention_padded_ptr(
                        roped_query=roped_query,
                        roped_key_cmp=roped_key_cmp,
                        kv_cache=kv_cache,
                        q_chunk_offset=q_chunk_offset,
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
                        q_chunk_offset=q_chunk_offset,
                    ).to(torch.int32).contiguous()
                    block_seqlen_k = kv_cache.block_seqlens['k_cache']
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
        if patch_embed_linear_enabled() and x.dim() == 5:
            x = patch_embed_linear(self.patch_embedding, x)
        else:
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
            if patch_embed_linear_enabled() and teacher.dim() == 5:
                x_t = patch_embed_linear(self.patch_embedding, teacher)
            else:
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