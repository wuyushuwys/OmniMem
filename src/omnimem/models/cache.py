from typing import Dict, Union, Optional, List
from collections import OrderedDict, defaultdict, deque

import torch


class MMCache:
    """
    Multi-modal cache with chunked + per-head storage and GPU-resident ptr tables
    for selection-attention fast path.

    Storage modes (per cache_type):
      - state          : single dense tensor, in-place updated
      - chunk_buffer   : single dense tensor, chunk-wise update API (fast path)
      - chunked        : List[Tensor]                          [shared across heads]
      - per_head       : List[List[Tensor]]                    [outer = chunk, inner = head]
      - block_level    : List[List[Tensor]]                    [outer = head,  inner = block]
      - chunk_per_head : List[List[Tensor]]                    [outer = head,  inner = chunk]

    Performance notes:
      - Hot-path tensor creation (e.g. ptr table writes) uses pre-allocated host
        pinned staging buffers, avoiding torch.tensor(..., device=cuda) calls
        which are implicit syncs.
      - chunk_per_head writes use a single permute+contiguous instead of a
        per-head loop, collapsing num_heads small kernel launches into one.

    GPU ptr table (chunk_per_head only):
      chunk_base_ptrs[layer][name] : [H, max_chunks] int64 GPU
        data_ptr() of chunks[h][cid] when GPU-resident, 0 otherwise.

    LRU offload (chunk_per_head only):
      lru_touch_per_head maintains per-(layer, name, head) active set.
      When LRU is full, the displaced cid is enqueued in self._to_offload.
      flush_offload picks them up at chunk boundary, batches D2H copies on
      the default stream, and replaces GPU refs with CPU pinned slices.

      k_cache and v_cache always evict together: only k_cache is LRU-touched
      (on write + on selection access), but _mark_for_offload also enqueues
      the v_cache entry so flush_offload offloads them in lockstep.
    """

    SEGMENT_CIDS = 10

    def __init__(self,
                 config,
                 seq_dim,
                 available_shape: Dict,
                 device='cuda',
                 dtype=torch.float32,
                 block_seqlen_config=None,
                 window_config=None,
                 sink_config=None,
                 num_layers=None,
                 lru_max_size=10,
                 per_head_types=None,
                 chunk_per_head_types=None,
                 chunk_buffer_types=None,
                 num_heads=None,
                 kv_block_size=0,
                 ):
        super().__init__()

        self.available_type = config["cache_type"]
        self.seq_dim = seq_dim
        self.device = device
        self.dtype = dtype
        self._has_cpu_chunks = False

        self.per_head_types = set(per_head_types) if per_head_types else set()
        self.chunk_per_head_types = set(chunk_per_head_types) if chunk_per_head_types else set()
        self.chunk_buffer_types = set(chunk_buffer_types) if chunk_buffer_types else set()
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.kv_block_size = kv_block_size
        self.block_level_types = (self.per_head_types.copy() if kv_block_size > 0 else set())

        for cache_type in self.available_type:
            if cache_type not in config:
                config[cache_type] = {}

        block_seqlen_config = block_seqlen_config or {}
        window_config = window_config or {}
        sink_config = sink_config or {}

        self.chunked_types = set()
        self.state_types = set()
        self.block_seqlens = {}
        self.max_chunks = {}

        for cache_type in self.available_type:
            if cache_type in self.chunk_buffer_types:
                if cache_type in block_seqlen_config:
                    blk = block_seqlen_config[cache_type]
                    self.block_seqlens[cache_type] = blk
                    total_seq = available_shape[cache_type][seq_dim]
                    self.max_chunks[cache_type] = (total_seq + blk - 1) // blk
            elif cache_type in block_seqlen_config:
                self.chunked_types.add(cache_type)
                blk = block_seqlen_config[cache_type]
                self.block_seqlens[cache_type] = blk
                total_seq = available_shape[cache_type][seq_dim]
                self.max_chunks[cache_type] = (total_seq + blk - 1) // blk
            else:
                self.state_types.add(cache_type)

        self.cache: Dict[int, Dict[str, Union[list, torch.Tensor]]] = {}
        for cache_type in self.available_type:
            cache_config = config[cache_type]
            default_layers = range(num_layers) if num_layers is not None else []
            for layer_idx in cache_config.get("layer", default_layers):
                if layer_idx not in self.cache:
                    self.cache[layer_idx] = {}

                if cache_type in self.chunk_buffer_types:
                    self.cache[layer_idx][cache_type] = torch.zeros(
                        *available_shape[cache_type],
                        device=device,
                        dtype=dtype,
                    )
                elif cache_type in self.chunked_types:
                    if (cache_type in self.block_level_types
                            or cache_type in self.chunk_per_head_types):
                        self.cache[layer_idx][cache_type] = [[] for _ in range(num_heads)]
                    else:
                        self.cache[layer_idx][cache_type] = []
                else:
                    self.cache[layer_idx][cache_type] = torch.zeros(
                        *available_shape[cache_type],
                        device=device,
                        dtype=dtype,
                    )

        # Host-side bool flag — reading is free, no GPU sync.
        self._is_update = False

        self.window_blocks = {}
        self.sink_blocks = {}
        for cache_type in self.available_type:
            window_size = window_config.get(cache_type)
            is_chunkable = (cache_type in self.chunked_types
                            or cache_type in self.chunk_buffer_types)
            if window_size is not None and is_chunkable:
                self.window_blocks[cache_type] = window_size
                sink_size = sink_config.get(cache_type)
                if sink_size is not None:
                    assert sink_size >= 0
                    self.sink_blocks[cache_type] = sink_size
                else:
                    self.sink_blocks[cache_type] = 0
            else:
                self.window_blocks[cache_type] = None
                self.sink_blocks[cache_type] = None

        self.lru_max_size = lru_max_size
        self._lru: Dict[tuple, OrderedDict] = {}
        self._lru_hits: Dict[tuple, int] = {}
        self._lru_misses: Dict[tuple, int] = {}
        self._lru_cold_misses: Dict[tuple, int] = {}
        self._lru_total_access = 0
        self._lru_total_hits = 0
        self._lru_total_cold_misses = 0
        self._lru_recent_window: deque = deque(maxlen=10000)

        self.chunk_base_ptrs: Dict[int, Dict[str, torch.Tensor]] = {}
        self.n_chunks_per_head_gpu: Dict[int, Dict[str, torch.Tensor]] = {}
        self.chunk_base_strides: Dict[int, Dict[str, dict]] = {}

        # Pre-allocated host pinned staging buffers for hot-path H2D writes.
        #   _ptr_stage_h : [H] int64 — full per-head ptr column
        #   _idx_stage_h : [H] long  — head index list (subset gathers)
        # Lazily allocated on first use (need num_heads).
        self._ptr_stage_h: Optional[torch.Tensor] = None
        self._idx_stage_h: Optional[torch.Tensor] = None

        # flush_offload reuses these for the ptr-zero step. Grown on demand.
        self._flush_h_stage: Optional[torch.Tensor] = None
        self._flush_i_stage: Optional[torch.Tensor] = None
        self._flush_stage_capacity: int = 0

        self._to_offload: Dict[tuple, set] = {}

        # Pinned offload buffers, allocated by preallocate_offload_buffer().
        # Structure mirrors the cache itself: one tensor per (name, layer)
        # of shape (num_heads, segment_cids, *sample_shape). Each
        # (layer, name, head, cid) tuple owns a fixed slot, so re-evicting
        # the same tuple reuses its slot rather than appending past an
        # offset. Pinned host RAM stays at the upfront high-water mark.
        self._pinned_buffers: Optional[Dict[str, Dict[int, torch.Tensor]]] = None
        self._pinned_segment_cids: int = 0
        self._pinned_sample_shape: tuple = ()
        self._pinned_sample_dtype = None

    # ──────────────────────────────────────────────────────────
    # Staging buffer helpers
    # ──────────────────────────────────────────────────────────

    def _ensure_ptr_stage(self):
        """Lazily allocate per-head ptr staging buffers (size H)."""
        if self._ptr_stage_h is None and self.num_heads is not None:
            self._ptr_stage_h = torch.empty(
                self.num_heads, dtype=torch.int64,
                device='cpu', pin_memory=True,
            )
            self._idx_stage_h = torch.empty(
                self.num_heads, dtype=torch.long,
                device='cpu', pin_memory=True,
            )

    def _ensure_flush_stage(self, n: int):
        """Grow flush staging buffers if needed."""
        if n <= self._flush_stage_capacity:
            return
        new_cap = max(n * 2, 1024)
        self._flush_h_stage = torch.empty(
            new_cap, dtype=torch.long, device='cpu', pin_memory=True,
        )
        self._flush_i_stage = torch.empty(
            new_cap, dtype=torch.long, device='cpu', pin_memory=True,
        )
        self._flush_stage_capacity = new_cap

    # ──────────────────────────────────────────────────────────
    # Chunk ptr table maintenance
    # ──────────────────────────────────────────────────────────

    def _ensure_chunk_ptr_tables(self, layer_idx: int, name: str):
        if name not in self.chunk_per_head_types:
            return
        if (layer_idx in self.chunk_base_ptrs
                and name in self.chunk_base_ptrs[layer_idx]):
            return
        max_n = self.max_chunks.get(name, 0)
        if max_n == 0:
            return
        if layer_idx not in self.chunk_base_ptrs:
            self.chunk_base_ptrs[layer_idx] = {}
            self.n_chunks_per_head_gpu[layer_idx] = {}
        self.chunk_base_ptrs[layer_idx][name] = torch.zeros(
            self.num_heads, max_n,
            dtype=torch.int64, device=self.device,
        )
        self.n_chunks_per_head_gpu[layer_idx][name] = torch.zeros(
            self.num_heads, dtype=torch.int32, device=self.device,
        )

    def _record_chunk_strides(self, layer_idx: int, name: str, sample: torch.Tensor):
        if name not in self.chunk_per_head_types:
            return
        if layer_idx not in self.chunk_base_strides:
            self.chunk_base_strides[layer_idx] = {}
        if name in self.chunk_base_strides[layer_idx]:
            return
        elem = sample.element_size()
        self.chunk_base_strides[layer_idx][name] = {
            'batch_stride_bytes': sample.stride(0) * elem,
            'token_stride_bytes': sample.stride(1) * elem,
            'stride_bn': sample.stride(1),
            'stride_bd': sample.stride(2),
            'chunk_len': sample.shape[1],
        }

    def _has_ptr_table(self, layer_idx: int, name: str) -> bool:
        return (layer_idx in self.chunk_base_ptrs
                and name in self.chunk_base_ptrs[layer_idx])

    def _update_chunk_ptrs_at_cid(
        self, layer_idx: int, name: str, cid: int,
        tensors_per_head: List[Optional[torch.Tensor]],
    ):
        """Update ptrs for ALL H heads at one cid via pinned staging buffer.
        Old code did `torch.tensor(ptrs, device=cuda)` per call (fresh alloc
        + sync). New code writes into pre-pinned host tensor, single async H2D.
        """
        if name not in self.chunk_per_head_types:
            return
        self._ensure_chunk_ptr_tables(layer_idx, name)
        if not self._has_ptr_table(layer_idx, name):
            return
        self._ensure_ptr_stage()
        stage = self._ptr_stage_h
        for h, t in enumerate(tensors_per_head):
            stage[h] = t.data_ptr() if (t is not None and t.is_cuda) else 0
        # Single async H2D into pre-allocated GPU column
        self.chunk_base_ptrs[layer_idx][name][:, cid].copy_(stage, non_blocking=True)

    def _update_chunk_ptrs_heads_at_cid(
        self, layer_idx: int, name: str, heads: List[int], cid: int,
        tensors: List[Optional[torch.Tensor]],
    ):
        """Update ptrs for a subset of heads at one cid via pinned staging."""
        if name not in self.chunk_per_head_types or not heads:
            return
        if not self._has_ptr_table(layer_idx, name):
            return
        self._ensure_ptr_stage()
        n = len(heads)
        idx_stage = self._idx_stage_h[:n]
        ptr_stage = self._ptr_stage_h[:n]
        for j, (h, t) in enumerate(zip(heads, tensors)):
            idx_stage[j] = h
            ptr_stage[j] = t.data_ptr() if (t is not None and t.is_cuda) else 0
        idx_gpu = idx_stage.to(self.device, non_blocking=True)
        ptr_gpu = ptr_stage.to(self.device, non_blocking=True)
        self.chunk_base_ptrs[layer_idx][name][idx_gpu, cid] = ptr_gpu

    def _rebuild_chunk_ptrs(self, layer_idx: int, name: str):
        if name not in self.chunk_per_head_types:
            return
        chunks = self.cache.get(layer_idx, {}).get(name)
        if not chunks:
            return
        self._ensure_chunk_ptr_tables(layer_idx, name)
        H = self.num_heads
        max_n = self.chunk_base_ptrs[layer_idx][name].shape[1]
        ptrs_cpu = torch.zeros(H, max_n, dtype=torch.int64)
        n_cpu = torch.zeros(H, dtype=torch.int32)
        sample = None
        for h in range(H):
            n_cpu[h] = len(chunks[h])
            for cid, t in enumerate(chunks[h]):
                if t is not None and t.is_cuda:
                    ptrs_cpu[h, cid] = t.data_ptr()
                    if sample is None:
                        sample = t
        self.chunk_base_ptrs[layer_idx][name].copy_(
            ptrs_cpu.to(self.device, non_blocking=True)
        )
        self.n_chunks_per_head_gpu[layer_idx][name].copy_(
            n_cpu.to(self.device, non_blocking=True)
        )
        if sample is not None:
            self.chunk_base_strides.setdefault(layer_idx, {}).pop(name, None)
            self._record_chunk_strides(layer_idx, name, sample)

    # ──────────────────────────────────────────────────────────
    # Public accessors
    # ──────────────────────────────────────────────────────────

    def get_chunk_metadata(self, layer_idx: int, name: str):
        if not self._has_ptr_table(layer_idx, name):
            return None
        return (
            self.chunk_base_ptrs[layer_idx][name],
            self.n_chunks_per_head_gpu[layer_idx][name],
            self.chunk_base_strides[layer_idx][name],
        )

    def notify_chunks_reloaded(self, layer_idx: int, name: str,
                               h_cid_pairs: List[tuple]):
        if name not in self.chunk_per_head_types or not h_cid_pairs:
            return
        self._wait_offload()
        chunks = self.cache.get(layer_idx, {}).get(name)
        if chunks is None:
            return
        by_cid = defaultdict(list)
        for h, cid in h_cid_pairs:
            by_cid[cid].append(h)
        for cid, heads in by_cid.items():
            tensors = [chunks[h][cid] if cid < len(chunks[h]) else None for h in heads]
            self._update_chunk_ptrs_heads_at_cid(layer_idx, name, heads, cid, tensors)

    # ──────────────────────────────────────────────────────────
    # Device / state management
    # ──────────────────────────────────────────────────────────

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype

        for layer_idx in self.cache:
            for cache_type in self.cache[layer_idx]:
                entry = self.cache[layer_idx][cache_type]
                if isinstance(entry, list):
                    if (cache_type in self.block_level_types
                            or cache_type in self.chunk_per_head_types):
                        self.cache[layer_idx][cache_type] = [
                            [t.to(device=device, dtype=dtype) for t in head_list]
                            for head_list in entry
                        ]
                    else:
                        self.cache[layer_idx][cache_type] = [
                            t.to(device=device, dtype=dtype) for t in entry
                        ]
                else:
                    self.cache[layer_idx][cache_type] = entry.to(
                        device=device, dtype=dtype
                    )

        if device is not None:
            for layer_idx in list(self.chunk_base_ptrs.keys()):
                for name in list(self.chunk_base_ptrs[layer_idx].keys()):
                    self.chunk_base_ptrs[layer_idx][name] = (
                        self.chunk_base_ptrs[layer_idx][name].to(device)
                    )
                    self.n_chunks_per_head_gpu[layer_idx][name] = (
                        self.n_chunks_per_head_gpu[layer_idx][name].to(device)
                    )
                    self._rebuild_chunk_ptrs(layer_idx, name)

        return self

    # ──────────────────────────────────────────────────────────
    # Chunk read helpers
    # ──────────────────────────────────────────────────────────

    def _cat_chunks(self, layer_idx: int, name: str) -> Optional[torch.Tensor]:
        chunks = self.cache[layer_idx][name]
        if not chunks:
            return None

        if name in self.block_level_types or name in self.chunk_per_head_types:
            H = self.num_heads
            per_head = []
            for h in range(H):
                head_items = [c for c in chunks[h] if c is not None]
                if not head_items:
                    return None
                per_head.append(
                    torch.cat(head_items, dim=self.seq_dim) if len(head_items) > 1 else head_items[0]
                )
            return torch.stack(per_head, dim=2)
        elif name in self.per_head_types:
            valid = [c for c in chunks if c is not None]
            if not valid:
                return None
            stacked = [torch.stack(c, dim=1) for c in valid]
            cat = torch.cat(stacked, dim=2) if len(stacked) > 1 else stacked[0]
            return cat.permute(0, 2, 1, 3)
        else:
            if len(chunks) == 1:
                return chunks[0]
            return torch.cat(chunks, dim=self.seq_dim)

    @torch.compiler.disable()
    def update_cache(
            self,
            name: str,
            hidden_state: torch.Tensor,
            layer_idx: int,
            start_id: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        if name in self.chunk_buffer_types:
            return self._update_chunk_buffer(name, hidden_state, layer_idx, start_id)

        assert name in self.chunked_types, f"{name} is a state type, use set_state"
        is_block_level = name in self.block_level_types
        is_chunk_per_head = name in self.chunk_per_head_types

        block_seqlen = self.block_seqlens[name]
        # start_id is almost always a Python int — only call .item() if Tensor.
        # `type() is int` is the cheapest check (no MRO walk like isinstance).
        if type(start_id) is int:
            start_id_int = start_id
        elif isinstance(start_id, torch.Tensor):
            start_id_int = start_id.item()
        else:
            start_id_int = int(start_id)
        chunks = self.cache[layer_idx][name]

        if is_block_level:
            BS = self.kv_block_size
            s = hidden_state.shape[1]
            num_blocks = s // BS
            first_block_id = start_id_int // BS
            for bid in range(num_blocks):
                global_bid = first_block_id + bid
                for h in range(self.num_heads):
                    block_tensor = hidden_state[:, bid * BS:(bid + 1) * BS, h, :].contiguous()
                    if global_bid < len(chunks[h]):
                        chunks[h][global_bid] = block_tensor
                    else:
                        chunks[h].append(block_tensor)

        elif is_chunk_per_head:
            chunk_id = start_id_int // block_seqlen

            # Old: per-head loop calling hidden_state[:, :, h, :].contiguous()
            #      → num_heads small kernel launches.
            # New: one permute+contiguous gives [H, B, S, D] in a single kernel,
            #      then per-head views (zero-copy, share storage).
            stacked = hidden_state.permute(2, 0, 1, 3).contiguous()
            new_tensors = []
            for h in range(self.num_heads):
                head_slice = stacked[h]  # view, no copy
                if chunk_id < len(chunks[h]):
                    chunks[h][chunk_id] = head_slice
                else:
                    chunks[h].append(head_slice)
                new_tensors.append(head_slice)

            self._ensure_chunk_ptr_tables(layer_idx, name)
            self._record_chunk_strides(layer_idx, name, new_tensors[0])
            self._update_chunk_ptrs_at_cid(layer_idx, name, chunk_id, new_tensors)
            self.n_chunks_per_head_gpu[layer_idx][name].clamp_min_(chunk_id + 1)

            if name == 'k_cache':
                for h in range(self.num_heads):
                    self.lru_touch_per_head(layer_idx, 'k_cache', h, chunk_id)

        else:
            chunk_id = start_id_int // block_seqlen
            is_per_head = name in self.per_head_types
            if is_per_head:
                store = [hidden_state[:, :, h, :].contiguous() for h in range(self.num_heads)]
            else:
                store = hidden_state
            if chunk_id < len(chunks):
                chunks[chunk_id] = store
            else:
                chunks.append(store)

        if is_chunk_per_head:
            return self._build_window_chunk_per_head(name, chunks, chunk_id + 1, hidden_state.device)
        elif is_block_level:
            num_valid = first_block_id + num_blocks
            return self._build_window_block_level(name, chunks, num_valid, hidden_state.device)
        else:
            return self._build_window_flat(name, chunks, chunk_id + 1, is_per_head, hidden_state.device)

    def _update_chunk_buffer(
            self,
            name: str,
            hidden_state: torch.Tensor,
            layer_idx: int,
            start_id: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        buf = self.cache[layer_idx][name]
        if type(start_id) is int:
            start = start_id
        elif isinstance(start_id, torch.Tensor):
            start = start_id.item()
        else:
            start = int(start_id)
        s = hidden_state.shape[self.seq_dim]
        cur_len = start + s

        if self._is_update:
            buf.narrow(self.seq_dim, start, s).copy_(hidden_state)

        wb = self.window_blocks.get(name)
        if wb is None:
            return buf.narrow(self.seq_dim, 0, cur_len)

        block_seqlen = self.block_seqlens[name]
        cur_chunks = (cur_len + block_seqlen - 1) // block_seqlen
        sink_end_c, win_start_c, _ = self._resolve_window_range(name, cur_chunks)
        sink_end_tok = sink_end_c * block_seqlen
        win_start_tok = win_start_c * block_seqlen
        win_len_tok = cur_len - win_start_tok

        if sink_end_c == 0:
            return buf.narrow(self.seq_dim, win_start_tok, win_len_tok)

        sink = buf.narrow(self.seq_dim, 0, sink_end_tok)
        win = buf.narrow(self.seq_dim, win_start_tok, win_len_tok)
        return torch.cat([sink, win], dim=self.seq_dim)

    def _resolve_window_range(self, name: str, num_valid: int, scale: int = 1):
        wb = self.window_blocks.get(name)
        sb = self.sink_blocks.get(name) or 0
        wb_eff = wb * scale if wb is not None else None
        sb_eff = sb * scale

        if wb_eff is None:
            return 0, 0, num_valid
        if sb_eff == 0:
            return 0, max(0, num_valid - wb_eff), num_valid
        effect_sink = min(sb_eff, num_valid)
        win_start = max(effect_sink, num_valid - wb_eff)
        return effect_sink, win_start, num_valid

    def _build_window_chunk_per_head(self, name, chunks, num_valid, device):
        sink_end, win_start, _ = self._resolve_window_range(name, num_valid)
        per_head_seqs = []
        for h in range(self.num_heads):
            if self.window_blocks.get(name) is None:
                head_rel = chunks[h][:num_valid]
            else:
                head_rel = chunks[h][:sink_end] + chunks[h][win_start:num_valid]
            head_rel = [c.to(device, non_blocking=True) if not c.is_cuda else c for c in head_rel]
            per_head_seqs.append(
                torch.cat(head_rel, dim=self.seq_dim) if len(head_rel) > 1 else head_rel[0]
            )
        return torch.stack(per_head_seqs, dim=2)

    def _build_window_block_level(self, name, chunks, num_valid, device):
        BS = self.kv_block_size
        block_seqlen = self.block_seqlens[name]
        scale = block_seqlen // BS
        sink_end, win_start, _ = self._resolve_window_range(name, num_valid, scale=scale)
        per_head_seqs = []
        for h in range(self.num_heads):
            if self.window_blocks.get(name) is None:
                head_rel = chunks[h][:num_valid]
            else:
                head_rel = chunks[h][:sink_end] + chunks[h][win_start:num_valid]
            head_rel = [b.to(device, non_blocking=True) if not b.is_cuda else b for b in head_rel]
            per_head_seqs.append(
                torch.cat(head_rel, dim=self.seq_dim) if len(head_rel) > 1 else head_rel[0]
            )
        return torch.stack(per_head_seqs, dim=2)

    def _build_window_flat(self, name, chunks, num_valid, is_per_head, device):
        sink_end, win_start, _ = self._resolve_window_range(name, num_valid)
        if self.window_blocks.get(name) is None:
            relevant = chunks[:num_valid]
        else:
            relevant = chunks[:sink_end] + chunks[win_start:num_valid]

        if is_per_head:
            per_head_seqs = []
            for h in range(self.num_heads):
                head_chunks = [r[h] for r in relevant]
                head_chunks = [c.to(device, non_blocking=True) if not c.is_cuda else c for c in head_chunks]
                per_head_seqs.append(
                    torch.cat(head_chunks, dim=self.seq_dim) if len(head_chunks) > 1 else head_chunks[0]
                )
            return torch.stack(per_head_seqs, dim=2)

        if any(not c.is_cuda for c in relevant):
            relevant = [c.to(device, non_blocking=True) for c in relevant]
        if len(relevant) == 1:
            return relevant[0]
        return torch.cat(relevant, dim=self.seq_dim)

    @torch.compiler.disable()
    def load_history(
            self,
            name: str,
            layer_idx: int,
            context_length: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        if name in self.chunked_types:
            all_kv = self._cat_chunks(layer_idx, name)
            return all_kv.narrow(self.seq_dim, 0, context_length)
        return self.cache[layer_idx][name].narrow(self.seq_dim, 0, context_length)

    # ──────────────────────────────────────────────────────────
    # LRU
    # ──────────────────────────────────────────────────────────

    def _ensure_lru(self, key: tuple) -> OrderedDict:
        if key not in self._lru:
            self._lru[key] = OrderedDict()
        if key not in self._lru_hits:
            self._lru_hits[key] = 0
        if key not in self._lru_misses:
            self._lru_misses[key] = 0
        if key not in self._lru_cold_misses:
            self._lru_cold_misses[key] = 0
        return self._lru[key]

    def _is_chunk_on_gpu(self, layer_idx: int, name: str,
                         head: Optional[int], cid: int) -> bool:
        chunks = self.cache.get(layer_idx, {}).get(name)
        if chunks is None:
            return False
        if (name in self.chunk_per_head_types
                or name in self.block_level_types):
            if head is None or head >= len(chunks) or cid >= len(chunks[head]):
                return False
            t = chunks[head][cid]
        else:
            if cid >= len(chunks):
                return False
            t = chunks[cid]
        return t is not None and t.is_cuda

    def lru_touch_per_head(self, layer_idx: int, name: str, head: int, cid: int):
        if self.lru_max_size <= 0:
            return
        key = (layer_idx, name, head)
        lru = self._ensure_lru(key)

        if cid in lru:
            lru.move_to_end(cid)
            self._lru_total_access += 1
            self._lru_hits[key] += 1
            self._lru_total_hits += 1
            self._lru_recent_window.append(1)
        else:
            cold = self._is_chunk_on_gpu(layer_idx, name, head, cid)

            if len(lru) >= self.lru_max_size:
                evicted_cid, _ = lru.popitem(last=False)
                self._mark_for_offload(layer_idx, name, head, evicted_cid)
            lru[cid] = None

            if cold:
                self._lru_cold_misses[key] += 1
                self._lru_total_cold_misses += 1
            else:
                self._lru_total_access += 1
                self._lru_misses[key] += 1
                self._lru_recent_window.append(0)

    def lru_contains(self, layer_idx: int, name: str, cid: int) -> bool:
        if self.lru_max_size <= 0:
            return False
        lru = self._lru.get((layer_idx, name))
        return lru is not None and cid in lru

    def lru_contains_per_head(self, layer_idx: int, name: str, head: int, cid: int) -> bool:
        if self.lru_max_size <= 0:
            return False
        lru = self._lru.get((layer_idx, name, head))
        return lru is not None and cid in lru

    def lru_hit_rate_overall(self) -> float:
        if self._lru_total_access == 0:
            return 0.0
        return self._lru_total_hits / self._lru_total_access

    def lru_reset_stats(self):
        for key in self._lru_hits:
            self._lru_hits[key] = 0
            self._lru_misses[key] = 0
        for key in self._lru_cold_misses:
            self._lru_cold_misses[key] = 0
        self._lru_total_access = 0
        self._lru_total_hits = 0
        self._lru_total_cold_misses = 0
        self._lru_recent_window.clear()

    def lru_clear(self):
        for lru in self._lru.values():
            lru.clear()
        for k in self._lru_hits:
            self._lru_hits[k] = 0
        for k in self._lru_misses:
            self._lru_misses[k] = 0
        for k in self._lru_cold_misses:
            self._lru_cold_misses[k] = 0
        self._lru_total_access = 0
        self._lru_total_hits = 0
        self._lru_total_cold_misses = 0
        self._lru_recent_window.clear()

    # ──────────────────────────────────────────────────────────
    # Chunk list management
    # ──────────────────────────────────────────────────────────

    def free_last_chunk(self, layer_idx: Optional[int] = None):
        layers = [layer_idx] if layer_idx is not None else list(self.cache.keys())
        for li in layers:
            for name in self.chunked_types:
                if name not in self.cache[li]:
                    continue
                chunks = self.cache[li][name]
                if name in self.block_level_types or name in self.chunk_per_head_types:
                    for h in range(self.num_heads):
                        if chunks[h]:
                            chunks[h].pop()
                    if name in self.chunk_per_head_types and self._has_ptr_table(li, name):
                        cid_popped = len(chunks[0])
                        if cid_popped < self.chunk_base_ptrs[li][name].shape[1]:
                            self.chunk_base_ptrs[li][name][:, cid_popped] = 0
                        self.n_chunks_per_head_gpu[li][name].clamp_max_(cid_popped)
                else:
                    if chunks:
                        chunks.pop()

    # ──────────────────────────────────────────────────────────
    # Lifecycle: reset / release
    # ──────────────────────────────────────────────────────────

    def reset(self):
        for li in self.cache:
            for cache_type in self.cache[li]:
                if cache_type in self.chunk_buffer_types:
                    self.cache[li][cache_type].zero_()
                elif cache_type in self.chunked_types:
                    if (cache_type in self.block_level_types
                            or cache_type in self.chunk_per_head_types):
                        self.cache[li][cache_type] = [[] for _ in range(self.num_heads)]
                    else:
                        self.cache[li][cache_type] = []
                else:
                    self.cache[li][cache_type].zero_()

        for li in self.chunk_base_ptrs:
            for name in self.chunk_base_ptrs[li]:
                self.chunk_base_ptrs[li][name].zero_()
                self.n_chunks_per_head_gpu[li][name].zero_()
        self.chunk_base_strides.clear()

        if hasattr(self, '_last_evict_processed'):
            self._last_evict_processed.clear()
        if hasattr(self, '_lru_saved_cids'):
            self._lru_saved_cids.clear()

        self.lru_clear()
        self._to_offload.clear()

        # Pinned buffers stay alive — they're sized for the worst-case
        # sequence and slots are reused. Cache list entries that pointed
        # into them are about to be cleared above; next flush_offload()
        # overwrites the slots in place.

        self._has_cpu_chunks = False

    # ──────────────────────────────────────────────────────────
    # Window-based eviction (sync, used as fallback)
    # ──────────────────────────────────────────────────────────

    def evict_out_of_window(self, min_gpu_chunks=4):
        evicted = False
        for li in self.cache:
            for name in self.chunked_types:
                wb = self.window_blocks.get(name)
                sb = self.sink_blocks.get(name, 0) or 0
                if wb is None:
                    continue

                chunks = self.cache[li][name]

                if name in self.block_level_types:
                    evicted |= self._evict_block_level(li, name, chunks, wb, sb)
                elif name in self.chunk_per_head_types:
                    evicted |= self._evict_chunk_per_head(li, name, chunks, wb, sb, min_gpu_chunks)
                elif name in self.per_head_types:
                    evicted |= self._evict_per_head(li, name, chunks, wb, sb, min_gpu_chunks)
                else:
                    evicted |= self._evict_flat(li, name, chunks, wb, sb, min_gpu_chunks)

        if evicted:
            torch.cuda.current_stream().synchronize()
            self._has_cpu_chunks = True

    def _evict_block_level(self, li, name, chunks, wb, sb):
        BS = self.kv_block_size
        block_seqlen = self.block_seqlens[name]
        blocks_per_chunk = block_seqlen // BS
        num_valid = len(chunks[0]) if chunks else 0
        wb_blocks = wb * blocks_per_chunk
        sb_blocks = sb * blocks_per_chunk
        gpu_keep = sb_blocks + wb_blocks
        evict_before = max(sb_blocks, num_valid - (gpu_keep - sb_blocks))

        evicted = False
        for h in range(self.num_heads):
            for i in range(sb_blocks, evict_before):
                if i >= len(chunks[h]):
                    continue
                t = chunks[h][i]
                if t is None or self.lru_contains_per_head(li, name, h, i):
                    continue
                if t.is_cuda:
                    pinned = torch.empty(t.shape, dtype=t.dtype, device='cpu', pin_memory=True)
                    pinned.copy_(t, non_blocking=True)
                    chunks[h][i] = pinned
                    evicted = True
        return evicted

    def _evict_chunk_per_head(self, li, name, chunks, wb, sb, min_gpu_chunks):
        num_valid = len(chunks[0]) if chunks else 0
        gpu_keep = max(sb + wb, min_gpu_chunks) if min_gpu_chunks else sb + wb
        evict_before = max(sb, num_valid - (gpu_keep - sb))
        lru_name = 'k_cache' if name in ('k_cache', 'v_cache') else name

        if not hasattr(self, '_last_evict_processed'):
            self._last_evict_processed = {}
            self._lru_saved_cids = {}
        key = (li, name)
        last = self._last_evict_processed.get(key, sb - 1)

        cids_to_check = list(range(max(sb, last + 1), evict_before))
        saved = self._lru_saved_cids.get(key)
        if saved:
            for c in saved:
                if sb <= c < evict_before and c <= last:
                    cids_to_check.append(c)

        if not cids_to_check:
            self._last_evict_processed[key] = max(last, evict_before - 1)
            return False

        cids_to_check.sort()
        still_saved = set()
        evicted = False

        for i in cids_to_check:
            to_evict = []
            has_lru = False
            for h in range(self.num_heads):
                if i >= len(chunks[h]):
                    continue
                t = chunks[h][i]
                if t is None or not t.is_cuda:
                    continue
                if self.lru_contains_per_head(li, lru_name, h, i):
                    has_lru = True
                    continue
                to_evict.append(h)

            if has_lru:
                still_saved.add(i)
            if not to_evict:
                continue

            stacked = torch.stack([chunks[h][i] for h in to_evict], dim=0)
            pinned = torch.empty(stacked.shape, dtype=stacked.dtype, device='cpu', pin_memory=True)
            pinned.copy_(stacked, non_blocking=True)
            for idx, h in enumerate(to_evict):
                chunks[h][i] = pinned[idx]
            if self._has_ptr_table(li, name):
                # Use pinned staging instead of fresh torch.tensor
                self._ensure_ptr_stage()
                n = len(to_evict)
                idx_stage = self._idx_stage_h[:n]
                for j, h in enumerate(to_evict):
                    idx_stage[j] = h
                idx_gpu = idx_stage.to(self.device, non_blocking=True)
                self.chunk_base_ptrs[li][name][idx_gpu, i] = 0
            evicted = True

        self._last_evict_processed[key] = max(last, evict_before - 1)
        self._lru_saved_cids[key] = still_saved
        return evicted

    def _evict_per_head(self, li, name, chunks, wb, sb, min_gpu_chunks):
        num_valid = len(chunks)
        gpu_keep = max(sb + wb, min_gpu_chunks) if min_gpu_chunks else sb + wb
        evict_before = max(sb, num_valid - (gpu_keep - sb))

        evicted = False
        for i in range(sb, evict_before):
            head_chunks = chunks[i]
            if head_chunks is None:
                continue
            to_evict = [h for h in range(self.num_heads)
                        if not self.lru_contains_per_head(li, name, h, i)
                        and head_chunks[h] is not None
                        and head_chunks[h].is_cuda]
            if not to_evict:
                continue
            stacked = torch.stack([head_chunks[h] for h in to_evict], dim=0)
            pinned = torch.empty(stacked.shape, dtype=stacked.dtype, device='cpu', pin_memory=True)
            pinned.copy_(stacked, non_blocking=True)
            for idx, h in enumerate(to_evict):
                head_chunks[h] = pinned[idx]
            evicted = True
        return evicted

    def _evict_flat(self, li, name, chunks, wb, sb, min_gpu_chunks):
        num_valid = len(chunks)
        gpu_keep = max(sb + wb, min_gpu_chunks) if min_gpu_chunks else sb + wb
        evict_before = max(sb, num_valid - (gpu_keep - sb))

        evicted = False
        for i in range(sb, evict_before):
            if self.lru_contains(li, name, i):
                continue
            chunk = chunks[i]
            if chunk is not None and chunk.is_cuda:
                pinned = torch.empty(chunk.shape, dtype=chunk.dtype, device='cpu', pin_memory=True)
                pinned.copy_(chunk, non_blocking=True)
                chunks[i] = pinned
                evicted = True
        return evicted

    # ──────────────────────────────────────────────────────────
    # LRU-driven async offload
    # ──────────────────────────────────────────────────────────

    def preallocate_offload_buffer(
        self,
        sample: torch.Tensor,
        max_entries_per_call: int,
        segment_cids: int = 10,
    ):
        """Allocate per-(name, layer) pinned host buffers for LRU offload.

        For each `name` in chunk_per_head_types and each existing layer, this
        creates one pinned tensor of shape `(num_heads, segment_cids, *sample.shape)`.
        Each (layer, name, head, cid) tuple owns the fixed slot
        `_pinned_buffers[name][layer][head, cid]`. Re-evicting the same tuple
        reuses its slot, so the buffers never need to grow — eliminating the
        segment fragmentation that previously forced `_alloc_new_segment`
        calls on long sequences with low LRU hit rate.

        `max_entries_per_call` is retained for API compatibility with the old
        linear-segment design; sizing is now derived from `chunk_per_head_types`,
        `num_layers`, `num_heads`, and `segment_cids`.

        Required: `segment_cids` must be >= the highest chunk id ever evicted,
        otherwise `flush_offload()` will raise at the offending cid."""
        del max_entries_per_call  # legacy arg; unused in per-(name, layer) layout
        per_slot_shape = tuple(sample.shape)

        self._pinned_sample_shape = per_slot_shape
        self._pinned_sample_dtype = sample.dtype
        self._pinned_segment_cids = segment_cids

        if (not self.chunk_per_head_types
                or self.num_heads is None
                or not self.cache):
            # Nothing offloadable through the slot-table path.
            self._pinned_buffers = {}
            return

        layer_ids = sorted(self.cache.keys())
        self._pinned_buffers = {}
        for name in sorted(self.chunk_per_head_types):
            self._pinned_buffers[name] = {}
            for li in layer_ids:
                self._pinned_buffers[name][li] = torch.empty(
                    (self.num_heads, segment_cids) + per_slot_shape,
                    dtype=sample.dtype,
                    device='cpu',
                    pin_memory=True,
                )

    def _mark_for_offload(self, layer_idx: int, name: str, head: int, cid: int):
        if name not in self.chunk_per_head_types:
            return

        names_to_mark = [name]
        if name == 'k_cache' and 'v_cache' in self.chunk_per_head_types:
            names_to_mark.append('v_cache')

        for n in names_to_mark:
            key = (layer_idx, n)
            if key not in self._to_offload:
                self._to_offload[key] = set()
            self._to_offload[key].add((head, cid))

    def flush_offload(self, layer_idx: Optional[int] = None):
        """Drain `_to_offload` queue, copying each marked (li, name, h, cid) to
        its fixed slot `_pinned_buffers[name][li][h, cid]`. Re-evictions of
        the same tuple reuse the same slot, so the pinned host RAM stays at
        the upfront high-water mark forever — no `_alloc_new_segment` calls,
        no segment fragmentation.

        If `layer_idx` is given, only entries for that layer are drained —
        other layers' queued evictions stay pending. Used by per-layer
        backward hooks so each layer flushes exactly when its save_for_backward
        refs are released."""
        if not self._to_offload:
            return

        # Phase 1: collect (optionally filtered by layer)
        to_evict = []
        keys_drained = []
        for (li, name), pairs in self._to_offload.items():
            if layer_idx is not None and li != layer_idx:
                continue
            keys_drained.append((li, name))
            chunks = self.cache[li][name]
            lru_name = 'k_cache' if name in ('k_cache', 'v_cache') else name
            # Sink + sliding-window protection: keep cids in [0, sb) and
            # [num_valid - wb, num_valid) GPU-resident regardless of LRU
            # state. Mirrors `_evict_chunk_per_head`'s policy so SWA reads
            # never need a CPU reload. The LRU active-set check below is
            # a separate, additive guard (skips currently-LRU-hot cids).
            wb = self.window_blocks.get(name)
            sb = self.sink_blocks.get(name, 0) or 0
            for h, i in pairs:
                if i >= len(chunks[h]):
                    continue
                t = chunks[h][i]
                if t is None or not t.is_cuda:
                    continue
                if wb is not None:
                    num_valid = len(chunks[h])
                    win_start = max(sb, num_valid - wb)
                    if i < sb or i >= win_start:
                        continue   # sink or window — never flush
                lru = self._lru.get((li, lru_name, h))
                if lru is not None and i in lru:
                    continue
                to_evict.append((li, name, h, i, t))
        for key in keys_drained:
            del self._to_offload[key]
        if not to_evict:
            return

        # Phase 2: ensure pinned buffers exist (lazy fallback for callers that
        # didn't preallocate).
        if self._pinned_buffers is None:
            sample = to_evict[0][4]
            self.preallocate_offload_buffer(
                sample=sample,
                max_entries_per_call=0,
                segment_cids=self.SEGMENT_CIDS,
            )

        seg_cids = self._pinned_segment_cids

        # Phase 3: D2H into fixed (h, cid) slot; track (h, cid) per (li, name)
        # for the batched ptr-table zero in Phase 4.
        ptr_zero_groups = {}
        for li, name, h, i, t in to_evict:
            if i >= seg_cids:
                raise RuntimeError(
                    f"chunk id {i} exceeds preallocated segment_cids="
                    f"{seg_cids} for cache type '{name}'. "
                    f"Increase segment_cids in preallocate_offload_buffer()."
                )
            full_slot = self._pinned_buffers[name][li][h, i]
            if t.shape == full_slot.shape:
                full_slot.copy_(t, non_blocking=True)
                slot = full_slot
            else:
                # Partial chunk (e.g. the last cache chunk when the sequence
                # length isn't a multiple of frame_per_block). The slot is
                # sized for the max chunk; we copy only t's actual shape and
                # store a same-shape view into the cache so reload pulls
                # back exactly what was offloaded.
                view_slices = tuple(slice(0, s) for s in t.shape)
                slot = full_slot[view_slices]
                slot.copy_(t, non_blocking=True)
            self.cache[li][name][h][i] = slot
            ln_key = (li, name)
            if ln_key not in ptr_zero_groups:
                ptr_zero_groups[ln_key] = ([], [])
            ptr_zero_groups[ln_key][0].append(h)
            ptr_zero_groups[ln_key][1].append(i)

        # Phase 4: batched ptr-table zero via pinned staging.
        max_group = max((len(v[0]) for v in ptr_zero_groups.values()), default=0)
        if max_group > 0:
            self._ensure_flush_stage(max_group)

        for (li, name), (h_list, i_list) in ptr_zero_groups.items():
            if (li in self.chunk_base_ptrs
                    and name in self.chunk_base_ptrs[li]):
                gn = len(h_list)
                h_pin = self._flush_h_stage[:gn]
                i_pin = self._flush_i_stage[:gn]
                for j in range(gn):
                    h_pin[j] = h_list[j]
                    i_pin[j] = i_list[j]
                h_gpu = h_pin.to(self.device, non_blocking=True)
                i_gpu = i_pin.to(self.device, non_blocking=True)
                self.chunk_base_ptrs[li][name][h_gpu, i_gpu] = 0

        self._has_cpu_chunks = True

    def _wait_offload(self):
        """No-op. Offload runs on the default stream, so the next default-
        stream operation serializes after it implicitly. Kept as a method
        so reload-path callers don't need to be edited.
        """
        return