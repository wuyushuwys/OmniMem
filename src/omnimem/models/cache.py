import contextlib
import math
import mmap
import os
import sys
import time
import warnings
from typing import Dict, Union, Optional, List, Iterable, Set, Tuple
from collections import OrderedDict, deque

import torch

from omnimem.nsa.backend import resolve_block_kv_backend


# Exact-size pinned host buffers (cudaHostRegister) avoid the allocator's power-of-two rounding.
EXACT_PINNED_KV_ENV = 'OMNIMEM_EXACT_PINNED_KV'

# Never run teardown in a fork()ed child: MADV_DONTFORK leaves the range unmapped there.
_forked_mappings: list = []


def exact_pinned_kv_enabled() -> bool:
    """Read the switch at call time; anything but unset/0/1 is an error."""
    value = os.environ.get(EXACT_PINNED_KV_ENV, '')
    if value in ('', '0'):
        return False
    if value == '1':
        return True
    raise ValueError(f'{EXACT_PINNED_KV_ENV} must be unset, 0 or 1, got {value!r}')


def _cuda_device_index(device) -> int:
    device = torch.device(device)
    if device.type == 'cuda' and device.index is not None:
        return device.index
    return torch.cuda.current_device()


class _HostRegisteredMapping(mmap.mmap):
    """Anonymous private mapping page-locked with cudaHostRegister."""

    def _register(self, ptr: int, nbytes: int, device_index: int):
        with torch.cuda.device(device_index):
            err = torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0)
        if err != torch.cuda.cudart().cudaError.success:
            raise RuntimeError(
                f'cudaHostRegister({nbytes} bytes) failed: '
                f'{torch.cuda.cudart().cudaGetErrorString(err)}')
        self._registration = (ptr, nbytes, device_index, os.getpid())

    def __del__(self):
        registration = getattr(self, '_registration', None)
        self._registration = None
        if registration is None or sys is None or sys.is_finalizing():
            return
        ptr, nbytes, device_index, pid = registration
        if pid != os.getpid():
            _forked_mappings.append(self)  # resurrect: skip unregister and munmap
            return
        try:
            torch.cuda.synchronize(device_index)
        finally:
            err = torch.cuda.cudart().cudaHostUnregister(ptr)
        if err != torch.cuda.cudart().cudaError.success:
            warnings.warn(f'cudaHostUnregister({nbytes} bytes) failed: '
                          f'{torch.cuda.cudart().cudaGetErrorString(err)}')


def exact_pinned_empty(shape, dtype: torch.dtype, device='cuda') -> torch.Tensor:
    """Uninitialized page-locked CPU tensor of exactly ``shape`` (no rounding)."""
    shape = tuple(shape)
    numel = math.prod(shape)
    nbytes = numel * dtype.itemsize
    if nbytes == 0:
        return torch.empty(shape, dtype=dtype, device='cpu', pin_memory=True)
    mapping = _HostRegisteredMapping(
        -1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    mapping.madvise(mmap.MADV_DONTFORK)
    flat = torch.frombuffer(mapping, dtype=dtype, count=numel)
    # Registration page-locks and zero-fills every page up front, like cudaHostAlloc.
    mapping._register(flat.data_ptr(), nbytes, _cuda_device_index(device))
    del mapping
    return flat.view(shape)


class _PtrStagingRing:
    """Event-protected ring of pinned host staging slots for ptr-table publishes."""

    __slots__ = ('slots', '_capacity', '_host', '_np', '_events', '_pending', '_cursor')

    def __init__(self, slots: int = 8, capacity: int = 4096):
        assert slots >= 1 and capacity >= 1
        self.slots = int(slots)
        self._capacity = int(capacity)
        self._host: List[Optional[torch.Tensor]] = [None] * self.slots
        self._np: List[object] = [None] * self.slots
        self._events: List[Optional[torch.cuda.Event]] = [None] * self.slots
        self._pending: List[bool] = [False] * self.slots
        self._cursor = 0

    def drain(self, stats: Optional[dict] = None) -> None:
        """Wait for every outstanding H2D, then clear the pending state."""
        for i in range(self.slots):
            ev = self._events[i]
            if self._pending[i] and ev is not None and not ev.query():
                t0 = time.perf_counter()
                ev.synchronize()
                if stats is not None:
                    stats['ring_waits'] += 1
                    stats['ring_wait_s'] += time.perf_counter() - t0
            self._pending[i] = False
        self._cursor = 0

    def _wait(self, i: int, stats: Optional[dict]) -> None:
        ev = self._events[i]
        if self._pending[i] and ev is not None and not ev.query():
            t0 = time.perf_counter()
            ev.synchronize()
            if stats is not None:
                stats['ring_waits'] += 1
                stats['ring_wait_s'] += time.perf_counter() - t0
        self._pending[i] = False

    def acquire(self, n: int, stats: dict):
        """Return ``(slot_id, host_tensor, numpy_view)`` for a batch of ``n`` entries."""
        i = self._cursor
        self._cursor = (self._cursor + 1) % self.slots
        self._wait(i, stats)
        host = self._host[i]
        # Safe to reallocate only *after* acquire() proved the old buffer is idle.
        if host is None or host.numel() < 2 * n:
            if host is not None:
                stats['ring_grows'] += 1
            cap = max(n, self._capacity)
            host = torch.empty(2 * cap, dtype=torch.int64, device='cpu', pin_memory=True)
            self._host[i] = host
            self._np[i] = host.numpy()
        return i, host, self._np[i]

    def acquire_elems(self, n_elems: int, stats: dict):
        """Like :meth:`acquire`, but sized in elements."""
        i = self._cursor
        self._cursor = (self._cursor + 1) % self.slots
        self._wait(i, stats)
        host = self._host[i]
        n_elems = int(n_elems)
        # Safe to reallocate only *after* _wait() proved the old buffer is idle.
        if host is None or host.numel() < n_elems:
            if host is not None:
                stats['ring_grows'] += 1
            cap = max(n_elems, self._capacity)
            host = torch.empty(cap, dtype=torch.int64, device='cpu', pin_memory=True)
            self._host[i] = host
            self._np[i] = host.numpy()
        return i, host, self._np[i]

    def record(self, slot_id: int, stream) -> None:
        ev = self._events[slot_id]
        if ev is None:
            ev = torch.cuda.Event()
            self._events[slot_id] = ev
        ev.record(stream)
        self._pending[slot_id] = True


class MMCache:
    """
    Multi-modal cache with chunked + per-head storage and GPU-resident ptr tables
    for selection-attention fast path.

    Storage modes (per cache_type):
      - state          : single dense tensor, in-place updated
      - chunk_buffer   : single dense tensor, chunk-wise update API (fast path)
      - chunked        : List[Tensor]                          [shared across heads]
      - chunk_per_head : List[List[Tensor]]                    [outer = head,  inner = chunk]

    chunk_base_ptrs[layer][name][h, cid] is the chunk's data_ptr() if CUDA-resident, else 0.
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
                 chunk_per_head_types=None,
                 chunk_buffer_types=None,
                 num_heads=None,
                 kv_block_size=0,
                 ):
        super().__init__()

        self.available_type = config["cache_type"]
        self.available_shape = {name: tuple(shape) for name, shape in available_shape.items()}
        self.seq_dim = seq_dim
        self.device = device
        self.dtype = dtype
        self._has_cpu_chunks = False

        self.chunk_per_head_types = set(chunk_per_head_types) if chunk_per_head_types else set()
        self.chunk_buffer_types = set(chunk_buffer_types) if chunk_buffer_types else set()
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.kv_block_size = kv_block_size

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
                    if cache_type in self.chunk_per_head_types:
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

        # chunk_base_ptrs[layer][name] views _chunk_ptr_backing[layer], [N_names, H, max_chunks].
        self.chunk_base_ptrs: Dict[int, Dict[str, torch.Tensor]] = {}
        self.n_chunks_per_head_gpu: Dict[int, Dict[str, torch.Tensor]] = {}
        self.chunk_base_strides: Dict[int, Dict[str, dict]] = {}
        self._chunk_ptr_backing: Dict[int, torch.Tensor] = {}
        self._ptr_names: List[str] = sorted(self.chunk_per_head_types)
        self._ptr_name_index: Dict[str, int] = {n: i for i, n in enumerate(self._ptr_names)}

        # Event-protected pinned staging ring for every ptr-table publish.
        self._ptr_ring: Optional[_PtrStagingRing] = None

        # Publish/ring diagnostics; runtime-only, never serialized.
        self.ptr_stats: Dict[str, float] = self._fresh_ptr_stats()

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
        # Source of page-locked host KV buffers; None = process-wide switch.
        self._exact_pinned_host: Optional[bool] = None

        # Inference-only reload diagnostics, never part of state_dict / checkpoints.
        self.reload_stats: Dict[str, int] = self._fresh_reload_stats()

        # _version_counter is never reset, so a slot version is never reused for different content.
        self._version_counter: int = 0
        # _chunk_version parallels cache[...]; -1 = unknown version, always forces the D2H.
        self._chunk_version: Dict[int, Dict[str, List[List[int]]]] = {}
        # _slot_meta[name][layer][head][cid] = (version, shape, dtype) of the pinned slot, or None.
        self._slot_meta: Dict[str, Dict[int, List[List[Optional[tuple]]]]] = {}
        # Recorded after each flush's D2H batch; host reads of a slot must wait on it.
        self._offload_event: Optional[torch.cuda.Event] = None
        self._offload_event_stream = None
        self._offload_event_pending: bool = False
        # Runtime-only, like reload_stats/ptr_stats: never serialized.
        self.offload_stats: Dict[str, int] = self._fresh_offload_stats()

        # Opt-in peer-GPU tier: evictions that fit the budget stay readable over P2P.
        self.tier_device: Optional[torch.device] = None
        self.tier_budget_bytes: int = 0
        self.tier_used_bytes: int = 0
        # Device the tier tensors live on; kept until the byte map is empty.
        self._tier_resident_device: Optional[torch.device] = None
        self._tier_bytes: Dict[tuple, int] = {}
        self._tier_stream = None
        self.tier_stats: Dict[str, int] = self._fresh_tier_stats()

        # Block-level KV cache (kv_cache_policy='block_lru'); None means 'chunk_lru'.
        self._block_store = None
        self._block_store_budgets: Tuple[int, int] = (0, 0)
        # Options the store was built with; carried across device/dtype change and checkpoint load.
        self._block_store_opts: Dict[str, object] = {}

        # Opt-in contiguous window buffers (flat or mirrored ring); no-grad only.
        self._window_buf_enabled: bool = False
        # to() / checkpoint load replace the chunk tensors: seal the buffers until the next reset().
        self._window_buf_sealed: bool = False
        self._flat_buf: Dict[tuple, torch.Tensor] = {}
        self._flat_buf_tokens: Dict[tuple, int] = {}
        self._flat_buf_last_cid: Dict[tuple, int] = {}
        # Leading chunks of the list that live in the buffer (contiguous from cid 0).
        self._flat_buf_count: Dict[tuple, int] = {}
        self._flat_buf_short: Dict[tuple, int] = {}
        self._flat_buf_blocked: Set[tuple] = set()
        self._win_ring: Dict[tuple, torch.Tensor] = {}
        self._win_ring_valid: Dict[tuple, Set[int]] = {}
        self._win_ring_blocked: Set[tuple] = set()
        self.window_buf_stats: Dict[str, int] = self._fresh_window_buf_stats()

        # Opt-in: skip selection LRU bookkeeping when nothing is host-resident.
        self._resident_fast_path: bool = False
        # Largest len(lru) any key has reached; grows only on insertion.
        self._lru_maxlen: int = 0
        # Chunks the request can ever have: an LRU at least this large never evicts.
        self._lru_capacity_needed: int = max(
            (self.max_chunks.get(n, 0) for n in self.chunk_per_head_types), default=0)
        # Host mirror of n_chunks_per_head_gpu; dropped by _rebuild_chunk_ptrs.
        self._n_chunks_host: Dict[tuple, int] = {}
        self.fast_path_stats: Dict[str, int] = self._fresh_fast_path_stats()

        # Diagnostics only, set by the pipeline; nothing in the cache reads them.
        self.preallocate_skipped: bool = False
        self.preallocate_estimate_bytes: int = 0
        self.preallocate_local_budget_bytes: int = 0

    @staticmethod
    def _fresh_reload_stats() -> Dict[str, int]:
        return dict(calls=0, cids=0, head_chunks=0, bytes=0)

    def reset_reload_stats(self) -> None:
        self.reload_stats = self._fresh_reload_stats()
        # The offload counters share the request window, so they are reset here too.
        self.reset_offload_stats()

    @staticmethod
    def _fresh_offload_stats() -> Dict[str, int]:
        return dict(d2h_calls=0, d2h_bytes=0, skipped_calls=0, skipped_bytes=0,
                    slot_reuse=0, host_waits=0)

    @staticmethod
    def _fresh_tier_stats() -> Dict[str, int]:
        return dict(evictions=0, bytes=0, fallback_host=0, used_bytes=0,
                    peak_used_bytes=0, window_copyback=0)

    @staticmethod
    def _fresh_window_buf_stats() -> Dict[str, int]:
        return dict(flat_writes=0, flat_hits=0, flat_misses=0, flat_bytes=0,
                    ring_writes=0, ring_hits=0, ring_misses=0, ring_bytes=0)

    @staticmethod
    def _fresh_fast_path_stats() -> Dict[str, int]:
        return dict(skipped=0, clamp_skipped=0, clamp_published=0)

    def reset_offload_stats(self) -> None:
        self.offload_stats = self._fresh_offload_stats()
        # Tier counters are zeroed with the offload ones; residency survives.
        self.tier_stats = self._fresh_tier_stats()
        self.tier_stats['used_bytes'] = self.tier_used_bytes
        self.tier_stats['peak_used_bytes'] = self.tier_used_bytes
        # Block store counters are zeroed too; pools, address table and block LRU survive.
        if self._block_store is not None:
            self._block_store.reset_stats()
        # Window-buffer / fast-path counters are zeroed; the buffers themselves survive.
        live_flat = sum(t.numel() * t.element_size() for t in self._flat_buf.values())
        live_ring = sum(t.numel() * t.element_size() for t in self._win_ring.values())
        self.window_buf_stats = self._fresh_window_buf_stats()
        self.window_buf_stats['flat_bytes'] = live_flat
        self.window_buf_stats['ring_bytes'] = live_ring
        self.fast_path_stats = self._fresh_fast_path_stats()

    @staticmethod
    def _fresh_ptr_stats() -> Dict[str, float]:
        return dict(publishes=0, entries=0, ring_waits=0, ring_wait_s=0.0, ring_grows=0)

    def reset_ptr_stats(self) -> None:
        self.ptr_stats = self._fresh_ptr_stats()

    def _cache_device(self) -> torch.device:
        """``self.device`` as a concrete device (``'cuda'`` -> the current ordinal)."""
        dev = torch.device(self.device)
        if dev.type == 'cuda' and dev.index is None:
            dev = torch.device('cuda', torch.cuda.current_device())
        return dev

    def configure_gpu_tier(self, device, budget_bytes: int = 0) -> None:
        """Enable (or disable, with ``device=None``) the peer-GPU history tier; idempotent."""
        if device is None:
            self.tier_device = None
            self.tier_budget_bytes = 0
            return

        if self._block_store is not None:
            raise ValueError(
                "the GPU history tier and the block KV cache are mutually exclusive; "
                "call configure_block_store(0, 0) first")
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError(f"GPU tier device must be a CUDA device, got {device}")
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        src = self._cache_device()
        if src.type != 'cuda':
            raise ValueError(
                f"GPU tier needs a CUDA cache device, but this cache lives on {src}")
        if device == src:
            raise ValueError(
                f"GPU tier device {device} is the cache's own device; the tier is a "
                f"second GPU's residency tier, not a no-op")
        if (self._tier_resident_device is not None
                and self._tier_resident_device != device):
            raise ValueError(
                f"cannot move the GPU tier to {device}: {self.tier_used_bytes} bytes are "
                f"still resident on {self._tier_resident_device} (reset() first)")
        if self.tier_device != device:
            if not torch.cuda.can_device_access_peer(src, device):
                raise RuntimeError(
                    f"GPU tier requires peer access between {src} and {device}, "
                    f"which this topology does not support")
            self._force_peer_access(src, device)
        self.tier_device = device
        self.tier_budget_bytes = max(0, int(budget_bytes))

    @staticmethod
    def _force_peer_access(src: torch.device, dst: torch.device) -> None:
        """Make PyTorch enable peer access now (it does so lazily, per direction)."""
        a = torch.zeros(1, dtype=torch.uint8, device=src)
        b = torch.zeros(1, dtype=torch.uint8, device=dst)
        b.copy_(a)      # src -> dst
        a.copy_(b)      # dst -> src

    def _tier_copy_stream(self):
        """Dedicated stream on the tier device so the copy barrier stalls nothing else."""
        if self._tier_stream is None or self._tier_stream.device != self.tier_device:
            self._tier_stream = torch.cuda.Stream(device=self.tier_device)
        return self._tier_stream

    def _on_tier(self, t: Optional[torch.Tensor]) -> bool:
        """``t`` is CUDA-resident on the peer tier, not on this cache's device."""
        dev = self._tier_resident_device or self.tier_device
        return dev is not None and t is not None and t.is_cuda and t.device == dev

    def _tier_track(self, key: tuple, nbytes: int) -> None:
        prev = self._tier_bytes.get(key, 0)
        self._tier_bytes[key] = nbytes
        self.tier_used_bytes += nbytes - prev
        stats = self.tier_stats
        stats['used_bytes'] = self.tier_used_bytes
        if self.tier_used_bytes > stats['peak_used_bytes']:
            stats['peak_used_bytes'] = self.tier_used_bytes

    def _tier_untrack(self, key: tuple) -> None:
        prev = self._tier_bytes.pop(key, None)
        if prev is None:
            return
        self.tier_used_bytes -= prev
        self.tier_stats['used_bytes'] = self.tier_used_bytes
        if not self._tier_bytes:
            self._tier_resident_device = None

    def _tier_untrack_cid(self, layer_idx: int, name: str, cid: int) -> None:
        """Forget every head of one ``(layer, name, cid)`` (overwrite / pop)."""
        if not self._tier_bytes:
            return
        for h in range(self.num_heads or 0):
            self._tier_untrack((layer_idx, name, h, cid))

    def note_tier_entry(self, layer_idx: int, name: str, head: int, cid: int,
                        tensor: Optional[torch.Tensor]) -> None:
        """Re-sync the tier byte map after a cache entry was replaced elsewhere."""
        key = (layer_idx, name, head, cid)
        if self._on_tier(tensor):
            self._tier_track(key, tensor.numel() * tensor.element_size())
        else:
            self._tier_untrack(key)

    def recompute_tier_usage(self) -> int:
        """Rebuild the byte map from the actual chunk lists; returns the total."""
        self._tier_bytes = {}
        total = 0
        for li, entries in self.cache.items():
            for name in self.chunk_per_head_types:
                chunks = entries.get(name)
                if not chunks:
                    continue
                for h, head in enumerate(chunks):
                    for cid, t in enumerate(head):
                        if self._on_tier(t):
                            n = t.numel() * t.element_size()
                            self._tier_bytes[(li, name, h, cid)] = n
                            total += n
        self.tier_used_bytes = total
        self.tier_stats['used_bytes'] = total
        if total > self.tier_stats['peak_used_bytes']:
            self.tier_stats['peak_used_bytes'] = total
        if not self._tier_bytes:
            self._tier_resident_device = None
        return total

    def _align_chunk_device(self, items: List[torch.Tensor], device) -> List[torch.Tensor]:
        """Bring window/history chunks onto ``device`` before a ``cat``."""
        out = []
        for c in items:
            if c.device != device:
                if c.is_cuda:
                    self.tier_stats['window_copyback'] += 1
                c = c.to(device, non_blocking=True)
            out.append(c)
        return out

    @property
    def block_store(self):
        """The :class:`BlockKVStore`, or ``None`` under the chunk-LRU policy."""
        return self._block_store

    @property
    def kv_cache_policy(self) -> str:
        return 'block_lru' if self._block_store is not None else 'chunk_lru'

    def configure_block_store(self, persistent_bytes: int, scratch_bytes: int,
                              resident_if_fits: bool = True,
                              dedup: str = 'device',
                              block_kv_backend: str | None = None) -> None:
        """Enable the block-level KV cache ((0, 0) disables); only valid before any K/V exists."""
        persistent_bytes = max(0, int(persistent_bytes))
        scratch_bytes = max(0, int(scratch_bytes))
        opts = dict(resident_if_fits=bool(resident_if_fits),
                    dedup=str(dedup),
                    block_kv_backend=resolve_block_kv_backend(block_kv_backend))
        if persistent_bytes == 0 and scratch_bytes == 0:
            self._drop_block_store()
            return
        if self._block_store is not None \
                and self._block_store_budgets == (persistent_bytes, scratch_bytes) \
                and self._block_store_opts == opts:
            return

        if self.tier_device is not None:
            raise ValueError(
                "the block KV cache and the GPU history tier are mutually exclusive; "
                "call configure_gpu_tier(None) first")
        if any(len(head) for entries in self.cache.values()
               for name in ('k_cache', 'v_cache') if name in self.chunk_per_head_types
               for head in entries.get(name, ())):
            raise ValueError(
                "the block KV cache can only be configured on a cache without K/V "
                "history (its address table would not know the chunks already "
                "written); call reset() first")
        store = self._new_block_store(persistent_bytes, scratch_bytes, opts)
        old, self._block_store = self._block_store, store
        self._block_store_budgets = (persistent_bytes, scratch_bytes)
        self._block_store_opts = opts
        if old is not None:
            old.drop()

    def _new_block_store(self, persistent_bytes: int, scratch_bytes: int, opts: dict):
        """Validate this cache's geometry for a block store and build one (no swap)."""
        for name in ('k_cache', 'v_cache'):
            if name not in self.chunk_per_head_types:
                raise ValueError(
                    f"the block KV cache needs chunk_per_head '{name}' storage "
                    f"(got {sorted(self.chunk_per_head_types)})")
        shape = self.available_shape.get('k_cache')
        if not shape or len(shape) != 4:
            raise ValueError(f"unexpected k_cache shape for the block store: {shape}")
        if int(shape[0]) != 1:
            raise ValueError(
                f"the block KV cache requires batch size 1, got {shape[0]}: a pool "
                f"slot holds ONE batch element's block and the block-level ptr table "
                f"adds no batch stride")
        if self.kv_block_size <= 0:
            raise ValueError("the block KV cache needs kv_block_size > 0")
        chunk_len = self.block_seqlens.get('k_cache', 0)
        if chunk_len <= 0 or chunk_len % self.kv_block_size:
            raise ValueError(
                f"k_cache chunk length {chunk_len} must be a positive multiple of "
                f"kv_block_size {self.kv_block_size}")
        total_seq = int(shape[self.seq_dim])
        if total_seq % self.kv_block_size:
            raise ValueError(
                f"the block KV cache needs a block-aligned request length: "
                f"k_cache seq {total_seq} % kv_block_size {self.kv_block_size} != 0")
        # Both NSA backends serve the block store; get_backend() raises on an invalid name.
        from omnimem.nsa.backend import get_backend
        get_backend()
        if self.window_blocks.get('k_cache') is None:
            raise ValueError(
                "the block KV cache requires a FINITE sliding window on k_cache "
                "(window_blocks['k_cache'] is None means the whole history is dense "
                "and nothing may be retired)")
        max_chunks = int(self.max_chunks.get('k_cache', 0))
        if max_chunks <= 0:
            raise ValueError("the block KV cache needs a positive k_cache max_chunks")

        from omnimem.models.block_kv_store import BlockKVStore

        return BlockKVStore(
            layers=sorted(self.cache.keys()),
            num_heads=int(self.num_heads),
            block_tokens=int(self.kv_block_size),
            head_dim=int(shape[3]),
            chunk_len=int(chunk_len),
            max_chunks=max_chunks,
            batch=int(shape[0]),
            device=self._cache_device(),
            dtype=self.dtype,
            persistent_bytes=persistent_bytes,
            scratch_bytes=scratch_bytes,
            # The request's full K+V history in bytes, from the requested total k_cache length.
            history_bytes=(len(self.cache) * 2 * int(self.num_heads) * total_seq
                           * int(shape[3]) * self._elem_size()),
            **opts,
        )

    def _drop_block_store(self) -> None:
        store = self._block_store
        self._block_store = None
        self._block_store_budgets = (0, 0)
        self._block_store_opts = {}
        if store is not None:
            store.drop()

    def _elem_size(self) -> int:
        return torch.empty(0, dtype=self.dtype).element_size()

    def block_store_config(self) -> dict:
        """Resolved geometry/budget of the store (empty dict under the chunk-LRU policy)."""
        store = self._block_store
        return store.config() if store is not None else {}

    @property
    def block_stats(self) -> Dict[str, float]:
        """The block store's counters, or ``{}`` under the chunk-LRU policy."""
        store = self._block_store
        return store.stats if store is not None else {}

    def block_hit_rate(self) -> float:
        """``(protected_hits + persistent_hits) / selected_unique`` (0 when off)."""
        store = self._block_store
        return store.block_hit_rate() if store is not None else 0.0

    def block_stats_per_layer(self) -> Dict[str, list]:
        """The block store's per-layer counters, or ``{}`` under the chunk-LRU policy."""
        store = self._block_store
        return store.stats_per_layer() if store is not None else {}

    def preallocate_block_host(self, sample: torch.Tensor, segment_cids: int) -> None:
        """Pinned chunk-major host backing for the block store's retirement path."""
        store = self._block_store
        if store is None:
            raise RuntimeError(
                "preallocate_block_host needs a configured block store "
                "(call configure_block_store first)")
        if sample.dim() != 3:
            raise ValueError(
                f"expected a [B, S, D] sample chunk, got {tuple(sample.shape)}")
        B, S, D = (int(x) for x in sample.shape)
        if B != store.batch or D != store.head_dim or S != store.chunk_len:
            raise ValueError(
                f"sample {tuple(sample.shape)} does not match the store's "
                f"[B={store.batch}, S={store.chunk_len}, D={store.head_dim}]")
        if sample.dtype != store.dtype:
            raise ValueError(
                f"sample dtype {sample.dtype} != store dtype {store.dtype}")
        if store.preallocate_host(int(segment_cids), exact_pinned=self.exact_pinned_host()):
            # Re-bind chunk lists into the new pinned buffer after a regrow; the old one is freed.
            store.rebind_host_views(self)

    def retire_completed_history(self) -> int:
        """Back completed, out-of-window history up to pinned host rows; returns the count."""
        store = self._block_store
        if store is None or store.resident:
            return 0
        retired = 0
        for li in store.layers:
            entries = self.cache.get(li)
            if not entries:
                continue
            chunks_k = entries.get('k_cache')
            chunks_v = entries.get('v_cache')
            if not chunks_k or not chunks_v or not chunks_k[0]:
                continue
            num_valid = len(chunks_k[0])
            sink_end, win_start, _ = self._resolve_window_range('k_cache', num_valid)
            cids = store.retire_candidates(li, sink_end, win_start)
            if cids.size == 0:
                continue
            views = store.retire(li, cids)
            for name, per_cid in views.items():
                chunks = entries.get(name)
                if chunks is None:
                    continue
                for cid, heads in per_cid.items():
                    for h, view in enumerate(heads):
                        chunks[h][cid] = view
            retired += int(cids.size)
        if retired:
            self._has_cpu_chunks = True
        return retired

    def block_snapshot_range(self, cid_lo: int, cid_hi: int):
        """Freeze + capture the store's state for ``[cid_lo, cid_hi)`` in every layer."""
        store = self._block_store
        if store is None:
            return None
        return [store.snapshot_range(li, cid_lo, cid_hi) for li in store.layers]

    def block_restore_range(self, snapshot) -> None:
        store = self._block_store
        if store is None or not snapshot:
            return
        for snap in snapshot:
            store.restore_range(snap)

    def block_release_snapshot(self, snapshot) -> None:
        store = self._block_store
        if store is None or not snapshot:
            return
        for snap in snapshot:
            store.release_snapshot(snap)

    def _reconfigure_block_store(self) -> None:
        """Drop + re-create + rebuild after a device/dtype change or checkpoint load."""
        store = self._block_store
        if store is None:
            return
        budgets = self._block_store_budgets
        opts = dict(self._block_store_opts)
        # Release the old store first so old and new pools never coexist.
        self._drop_block_store()
        self._block_store = self._new_block_store(*budgets, opts)
        self._block_store_budgets = budgets
        self._block_store_opts = opts
        self._block_store.rebuild_all(self)

    def configure_window_buffers(self, enabled: bool = False) -> None:
        """Enable (``False`` disables) the preallocated window / history buffers; idempotent."""
        enabled = bool(enabled)
        if enabled == self._window_buf_enabled:
            return
        self._window_buf_enabled = enabled
        if not enabled:
            self._drop_window_buffers()

    def _drop_window_buffers(self) -> None:
        """Forget every buffer.  Chunk-list views keep their storage alive."""
        self._flat_buf.clear()
        self._flat_buf_tokens.clear()
        self._flat_buf_last_cid.clear()
        self._flat_buf_count.clear()
        self._flat_buf_short.clear()
        self._flat_buf_blocked.clear()
        self._win_ring.clear()
        self._win_ring_valid.clear()
        self._win_ring_blocked.clear()
        self.window_buf_stats['flat_bytes'] = 0
        self.window_buf_stats['ring_bytes'] = 0

    def _seal_window_buffers(self) -> None:
        """``to()`` / checkpoint load rebuilt the chunks: no buffer describes them."""
        self._window_buf_sealed = True
        self._drop_window_buffers()

    def window_buffer_bytes(self) -> int:
        return (self.window_buf_stats.get('flat_bytes', 0)
                + self.window_buf_stats.get('ring_bytes', 0))

    def window_buffer_active(self, name: str) -> bool:
        """True when some layer of ``name`` currently serves chunks out of a buffer."""
        if not self._window_buf_enabled or self._window_buf_sealed:
            return False
        return any(k[1] == name for k in self._flat_buf)

    def restore_flat_chunk(self, layer_idx: int, name: str, cid: int,
                           tensor: Optional[torch.Tensor]) -> None:
        """Put snapshotted content back for a flat (chunk-major) cache entry."""
        if tensor is None:
            return
        chunks = self.cache.get(layer_idx, {}).get(name)
        if chunks is None or cid >= len(chunks):
            return
        cur = chunks[cid]
        buf = self._flat_buf.get((layer_idx, name))
        if (buf is not None and isinstance(cur, torch.Tensor)
                and cur.untyped_storage().data_ptr() == buf.untyped_storage().data_ptr()):
            cur.copy_(tensor)
            return
        chunks[cid] = tensor

    def invalidate_window_ring(self, layer_idx: int, cid: int,
                               name: Optional[str] = None) -> None:
        """Drop ``cid`` from the window ring's valid set (content no longer matches)."""
        if not self._win_ring_valid:
            return
        names = (name,) if name is not None else tuple(self.chunk_per_head_types)
        for nm in names:
            valid = self._win_ring_valid.get((layer_idx, nm))
            if valid is not None:
                valid.discard(cid)

    def _flat_buffer_write(self, layer_idx: int, name: str, cid: int,
                           hidden_state: torch.Tensor) -> Optional[torch.Tensor]:
        """Copy one chunk into the flat history buffer; None if not buffered."""
        key = (layer_idx, name)
        if self._window_buf_sealed or key in self._flat_buf_blocked:
            return None
        if torch.is_grad_enabled():
            # Inference / no-grad only: the chunk must keep its own autograd identity.
            self._flat_buf_blocked.add(key)
            return None
        shape = self.available_shape.get(name)
        blk = self.block_seqlens.get(name)
        if (self.window_blocks.get(name) is not None
                or not blk or shape is None or len(shape) != 4 or shape[0] != 1
                or self.seq_dim != 1 or hidden_state.dim() != 4
                or hidden_state.shape[0] != 1
                or tuple(hidden_state.shape[2:]) != tuple(shape[2:])):
            self._flat_buf_blocked.add(key)
            return None
        s = hidden_state.shape[1]
        short = self._flat_buf_short.get(key)
        if short is not None and cid > short:
            # Only the last chunk of a request is short; a later cid would break ``cid * blk``.
            self._flat_buf_blocked.add(key)
            return None
        tok0 = cid * blk
        if s > blk or tok0 + s > shape[1]:
            self._flat_buf_blocked.add(key)
            return None
        count = self._flat_buf_count.get(key, 0)
        if cid > count:
            # An earlier cid is missing from the buffer, so the span would cover unwritten bytes.
            self._flat_buf_blocked.add(key)
            return None
        buf = self._flat_buf.get(key)
        if buf is None:
            if cid != 0:
                # The buffer must cover the whole history it is read as.
                self._flat_buf_blocked.add(key)
                return None
            buf = torch.empty((shape[1], shape[2], shape[3]),
                              dtype=hidden_state.dtype, device=hidden_state.device)
            self._flat_buf[key] = buf
            self.window_buf_stats['flat_bytes'] += buf.numel() * buf.element_size()
        elif buf.device != hidden_state.device or buf.dtype != hidden_state.dtype:
            self._flat_buf_blocked.add(key)
            return None
        view = buf.narrow(0, tok0, s)
        view.copy_(hidden_state[0])
        if s != blk:
            self._flat_buf_short[key] = cid
        self._flat_buf_tokens[key] = tok0 + s
        self._flat_buf_last_cid[key] = cid
        if cid == count:
            self._flat_buf_count[key] = cid + 1
        self.window_buf_stats['flat_writes'] += 1
        # unsqueeze(0) restores the contiguous [1, S, H, D] metadata.
        return view.unsqueeze(0)

    def _flat_buffer_view(self, layer_idx: int, name: str, num_valid: int,
                          device) -> Optional[torch.Tensor]:
        """The ``[1, total_tokens, H, D]`` view that replaces ``cat(chunks[:n])``."""
        key = (layer_idx, name)
        if self._window_buf_sealed or key in self._flat_buf_blocked:
            return None
        buf = self._flat_buf.get(key)
        if buf is None or buf.device != device:
            return None
        if self._flat_buf_last_cid.get(key) != num_valid - 1:
            # The build must follow the newest chunk's store so the total matches the chunks.
            return None
        total = self._flat_buf_tokens.get(key)
        if not total:
            return None
        self.window_buf_stats['flat_hits'] += 1
        return buf.narrow(0, 0, total).unsqueeze(0)

    def _window_ring_write(self, layer_idx: int, name: str, cid: int,
                           hidden_state: torch.Tensor) -> None:
        """Mirror one chunk into the sliding-window ring (both of its two slots)."""
        key = (layer_idx, name)
        if self._window_buf_sealed or key in self._win_ring_blocked:
            return
        if torch.is_grad_enabled():
            # Inference / no-grad only: the window build must hand out the chunk list's own tensors.
            self._win_ring_blocked.add(key)
            self._win_ring_valid.pop(key, None)
            return
        wb = self.window_blocks.get(name)
        blk = self.block_seqlens.get(name)
        shape = self.available_shape.get(name)
        if (not wb or not blk or (self.sink_blocks.get(name) or 0) != 0
                or shape is None or len(shape) != 4 or shape[0] != 1
                or self.seq_dim != 1 or hidden_state.dim() != 4
                or hidden_state.shape[0] != 1
                or tuple(hidden_state.shape[2:]) != tuple(shape[2:])):
            self._win_ring_blocked.add(key)
            return
        if hidden_state.shape[1] != blk:
            # A short chunk breaks the fixed ``blk`` slot stride; stop using the ring here.
            self._win_ring_blocked.add(key)
            self._win_ring_valid.pop(key, None)
            return
        ring = self._win_ring.get(key)
        if ring is None:
            ring = torch.empty((2 * wb * blk, shape[2], shape[3]),
                               dtype=hidden_state.dtype, device=hidden_state.device)
            self._win_ring[key] = ring
            self._win_ring_valid[key] = set()
            self.window_buf_stats['ring_bytes'] += ring.numel() * ring.element_size()
        elif ring.device != hidden_state.device or ring.dtype != hidden_state.dtype:
            self._win_ring_blocked.add(key)
            return
        src = hidden_state[0]
        pos = (cid % wb) * blk
        ring.narrow(0, pos, blk).copy_(src)
        ring.narrow(0, pos + wb * blk, blk).copy_(src)
        valid = self._win_ring_valid[key]
        # Both copies of the slot pair now hold ``cid``; whatever shared them is gone.
        if valid:
            res = cid % wb
            for c in [c for c in valid if c != cid and c % wb == res]:
                valid.discard(c)
        valid.add(cid)
        self.window_buf_stats['ring_writes'] += 1

    def _window_ring_view(self, layer_idx: int, name: str, num_valid: int,
                          device) -> Optional[torch.Tensor]:
        """The ``[1, win_tokens, H, D]`` view that replaces per-head cat + stack."""
        key = (layer_idx, name)
        if self._window_buf_sealed or key in self._win_ring_blocked:
            return None
        ring = self._win_ring.get(key)
        if ring is None or ring.device != device:
            return None
        wb = self.window_blocks.get(name)
        blk = self.block_seqlens.get(name)
        if not wb or not blk:
            return None
        sink_end, win_start, _ = self._resolve_window_range(name, num_valid)
        if sink_end != 0 or num_valid <= win_start or num_valid - win_start > wb:
            return None
        valid = self._win_ring_valid.get(key)
        if valid is None:
            return None
        for cid in range(win_start, num_valid):
            if cid not in valid:
                self.window_buf_stats['ring_misses'] += 1
                return None
        self.window_buf_stats['ring_hits'] += 1
        return ring.narrow(0, (win_start % wb) * blk,
                           (num_valid - win_start) * blk).unsqueeze(0)

    def configure_resident_fast_path(self, enabled: bool = False) -> None:
        self._resident_fast_path = bool(enabled)

    def selection_resident_fast_path(self) -> bool:
        """True when this layer-step needs no LRU touch and no host sync."""
        if not self._resident_fast_path or self._has_cpu_chunks:
            return False
        if self._block_store is not None:
            return False
        if self._tier_bytes:
            return False
        if self.lru_max_size <= 0:
            return True
        need = self._lru_capacity_needed
        if need and self.lru_max_size >= need:
            return True
        if need and need - self.lru_max_size > 1:
            return False
        return self._lru_maxlen + 2 < self.lru_max_size

    def _next_version(self) -> int:
        self._version_counter += 1
        return self._version_counter

    def _version_rows(self, layer_idx: int, name: str) -> List[List[int]]:
        per_layer = self._chunk_version.get(layer_idx)
        if per_layer is None:
            per_layer = {}
            self._chunk_version[layer_idx] = per_layer
        rows = per_layer.get(name)
        if rows is None:
            rows = [[] for _ in range(self.num_heads or 0)]
            per_layer[name] = rows
        return rows

    def _bump_chunk_version(self, layer_idx: int, name: str, cid: int) -> int:
        """Assign one fresh content version to every head of ``(layer, name, cid)``."""
        v = self._next_version()
        for row in self._version_rows(layer_idx, name):
            if cid >= len(row):
                row.extend([-1] * (cid + 1 - len(row)))
            row[cid] = v
        return v

    def chunk_version(self, layer_idx: int, name: str, head: int, cid: int) -> int:
        """Content version of one chunk, or ``-1`` when it is unknown."""
        rows = self._chunk_version.get(layer_idx, {}).get(name)
        if rows is None or not (0 <= head < len(rows)):
            return -1
        row = rows[head]
        return row[cid] if 0 <= cid < len(row) else -1

    def set_chunk_version(self, layer_idx: int, name: str, head: int, cid: int,
                          version: int) -> None:
        """Reinstate a previously observed version (prompt-recache restore)."""
        rows = self._version_rows(layer_idx, name)
        if not (0 <= head < len(rows)):
            return
        row = rows[head]
        if cid >= len(row):
            row.extend([-1] * (cid + 1 - len(row)))
        row[cid] = int(version)

    def _truncate_chunk_versions(self, layer_idx: int, name: str) -> None:
        rows = self._chunk_version.get(layer_idx, {}).get(name)
        chunks = self.cache.get(layer_idx, {}).get(name)
        if rows is None or chunks is None:
            return
        for h, row in enumerate(rows):
            n = len(chunks[h]) if h < len(chunks) else 0
            if len(row) > n:
                del row[n:]

    def _reset_slot_meta(self) -> None:
        """Rebuild all-invalid slot metadata matching ``_pinned_buffers``."""
        self._slot_meta = {
            name: {li: [[None] * int(buf.shape[1]) for _ in range(int(buf.shape[0]))]
                   for li, buf in per_layer.items()}
            for name, per_layer in (self._pinned_buffers or {}).items()
        }

    def _ensure_slot_meta(self) -> None:
        """No-op when the metadata already matches the buffers; rebuild otherwise."""
        bufs = self._pinned_buffers or {}
        meta = self._slot_meta
        if set(meta) != set(bufs):
            self._reset_slot_meta()
            return
        for name, per_layer in bufs.items():
            rows_by_layer = meta[name]
            if set(rows_by_layer) != set(per_layer):
                self._reset_slot_meta()
                return
            for li, buf in per_layer.items():
                rows = rows_by_layer[li]
                if len(rows) != int(buf.shape[0]) or any(
                        len(r) != int(buf.shape[1]) for r in rows):
                    self._reset_slot_meta()
                    return

    def _invalidate_slot(self, layer_idx: int, name: str, cid: int,
                         head: Optional[int] = None) -> None:
        rows = self._slot_meta.get(name, {}).get(layer_idx)
        if rows is None:
            return
        heads = range(len(rows)) if head is None else (
            (head,) if 0 <= head < len(rows) else ())
        for h in heads:
            row = rows[h]
            if 0 <= cid < len(row):
                row[cid] = None

    def _rebuild_chunk_versions(self) -> None:
        """Assign fresh versions to every restored chunk (checkpoint load); slots stay invalid."""
        self._chunk_version = {}
        self._reset_slot_meta()
        buffers = self._pinned_buffers or {}
        for li, entries in self.cache.items():
            for name in self.chunk_per_head_types:
                chunks = entries.get(name)
                if not chunks:
                    continue
                rows = self._version_rows(li, name)
                buf = buffers.get(name, {}).get(li)
                buf_storage = buf.untyped_storage().data_ptr() if buf is not None else None
                slot_rows = self._slot_meta.get(name, {}).get(li)
                for h, head in enumerate(chunks):
                    if h >= len(rows):
                        break
                    row = rows[h]
                    if len(head) > len(row):
                        row.extend([-1] * (len(head) - len(row)))
                    for cid, t in enumerate(head):
                        if t is None:
                            continue
                        v = self._next_version()
                        row[cid] = v
                        if (buf is None or slot_rows is None or t.is_cuda
                                or h >= int(buf.shape[0]) or cid >= int(buf.shape[1])
                                or t.dtype != buf.dtype
                                or t.untyped_storage().data_ptr() != buf_storage
                                or t.data_ptr() != buf[h, cid].data_ptr()):
                            continue
                        slot_rows[h][cid] = (v, tuple(t.shape), t.dtype)

    def wait_offload_host(self) -> None:
        """Block the CPU until the last ``flush_offload`` D2H batch has landed."""
        if self._block_store is not None:
            self._block_store.wait_host()
        ev = self._offload_event
        if ev is None or not self._offload_event_pending:
            return
        self._offload_event_pending = False
        self.offload_stats['host_waits'] += 1
        ev.synchronize()

    def _record_offload_event(self, device) -> None:
        stream = torch.cuda.current_stream(device)
        ev = self._offload_event
        if ev is None or self._offload_event_stream != stream:
            if ev is not None and self._offload_event_pending:
                # Consume the pending point first, or the previous flush's D2H becomes unobservable.
                self.wait_offload_host()
            ev = torch.cuda.Event()
            self._offload_event = ev
            self._offload_event_stream = stream
        ev.record(stream)
        self._offload_event_pending = True

    # ──────────────────────────────────────────────────────────
    # Chunk ptr table maintenance
    # ──────────────────────────────────────────────────────────

    def _ptr_backing_max_n(self) -> int:
        """Column count of the shared backing: the widest chunk_per_head table."""
        return max((self.max_chunks.get(n, 0) for n in self._ptr_names), default=0)

    def _ensure_chunk_ptr_tables(self, layer_idx: int, name: str):
        if name not in self.chunk_per_head_types:
            return
        if (layer_idx in self.chunk_base_ptrs
                and name in self.chunk_base_ptrs[layer_idx]):
            return
        if self.max_chunks.get(name, 0) == 0:
            return
        max_n = self._ptr_backing_max_n()
        if max_n == 0:
            return
        backing = self._chunk_ptr_backing.get(layer_idx)
        if backing is None:
            backing = torch.zeros(
                len(self._ptr_names), self.num_heads, max_n,
                dtype=torch.int64, device=self.device,
            )
            self._chunk_ptr_backing[layer_idx] = backing
        if layer_idx not in self.chunk_base_ptrs:
            self.chunk_base_ptrs[layer_idx] = {}
            self.n_chunks_per_head_gpu[layer_idx] = {}
        # A view, not a copy: publishes write the backing, readers see them.
        self.chunk_base_ptrs[layer_idx][name] = backing[self._ptr_name_index[name]]
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

    @staticmethod
    def chunk_ptr_of(t: Optional[torch.Tensor]) -> int:
        """The value the ptr table must hold for one chunk slot."""
        return t.data_ptr() if (t is not None and t.is_cuda) else 0

    def publish_chunk_ptrs(self, layer_idx: int, entries) -> int:
        """Batch update of this layer's chunk ptr table from ``(name, head, cid, ptr)``."""
        backing = self._chunk_ptr_backing.get(layer_idx)
        if backing is None:
            return 0
        tables = self.chunk_base_ptrs.get(layer_idx)
        if not tables:
            return 0
        _, H, max_n = backing.shape

        merged: Dict[int, int] = {}
        for name, head, cid, ptr in entries:
            if name not in tables:
                continue
            if not (0 <= head < H and 0 <= cid < max_n):
                # A slot the table cannot address would be silently skipped by the selection kernel.
                raise RuntimeError(
                    f"publish_chunk_ptrs: L{layer_idx}/{name} (head={head}, "
                    f"cid={cid}) is outside the [{H}, {max_n}] ptr table; "
                    f"increase the cache's configured sequence length")
            merged[(self._ptr_name_index[name] * H + head) * max_n + cid] = int(ptr)
        if not merged:
            return 0

        n = len(merged)
        stats = self.ptr_stats
        stats['publishes'] += 1
        stats['entries'] += n
        flat = backing.view(-1)

        if backing.device.type != 'cuda':
            flat[torch.tensor(list(merged.keys()), dtype=torch.int64)] = \
                torch.tensor(list(merged.values()), dtype=torch.int64)
            return n

        if self._ptr_ring is None:
            self._ptr_ring = _PtrStagingRing()
        slot_id, host, host_np = self._ptr_ring.acquire(n, stats)
        host_np[:n] = list(merged.keys())
        host_np[n:2 * n] = list(merged.values())
        staged = host[:2 * n].to(backing.device, non_blocking=True).view(2, n)
        flat[staged[0]] = staged[1]
        self._ptr_ring.record(slot_id, torch.cuda.current_stream(backing.device))
        return n

    def _rebuild_chunk_ptrs(self, layer_idx: int, name: str):
        if name not in self.chunk_per_head_types:
            return
        chunks = self.cache.get(layer_idx, {}).get(name)
        if not chunks:
            return
        self._ensure_chunk_ptr_tables(layer_idx, name)
        if not self._has_ptr_table(layer_idx, name):
            return
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
        # Per-head counts can differ, so drop the uniform host mirror.
        self._n_chunks_host.pop((layer_idx, name), None)
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

    # ──────────────────────────────────────────────────────────
    # Device / state management
    # ──────────────────────────────────────────────────────────

    def _checkpoint_configuration(self):
        """Architecture/placement contract; CUDA ordinal is rank-local."""
        return {
            'available_type': list(self.available_type),
            'available_shape': dict(self.available_shape),
            'seq_dim': self.seq_dim,
            'device_type': torch.device(self.device).type,
            'dtype': str(self.dtype),
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'kv_block_size': self.kv_block_size,
            'lru_max_size': self.lru_max_size,
            'chunk_per_head_types': sorted(self.chunk_per_head_types),
            'chunk_buffer_types': sorted(self.chunk_buffer_types),
            'chunked_types': sorted(self.chunked_types),
            'state_types': sorted(self.state_types),
            'block_seqlens': dict(self.block_seqlens),
            'max_chunks': dict(self.max_chunks),
            'window_blocks': dict(self.window_blocks),
            'sink_blocks': dict(self.sink_blocks),
            'layers': {li: sorted(entries) for li, entries in self.cache.items()},
            'dense_shapes': {
                (li, name): list(entry.shape)
                for li, entries in self.cache.items()
                for name, entry in entries.items() if isinstance(entry, torch.Tensor)
            },
        }

    @staticmethod
    def _without_removed_storage_modes(config):
        """Accept checkpoints listing the removed per_head / block_level modes only if empty."""
        if not isinstance(config, dict):
            return config
        config = dict(config)
        for key in ('per_head_types', 'block_level_types'):
            names = config.pop(key, None)
            if names:
                raise ValueError(
                    f"MMCache checkpoint stores {names!r} with the removed '{key}' "
                    f"K/V storage mode (enable_block_level_cache); it cannot be loaded")
        return config

    def checkpoint_state_dict(self):
        """Snapshot a completed training chunk using weights-only-safe values."""
        if torch.device(self.device).type == 'cuda':
            torch.cuda.synchronize(self.device)
        # Consume the offload event so later host reads are not charged for it twice.
        self.wait_offload_host()

        def snapshot(entry):
            if isinstance(entry, list):
                return [snapshot(item) for item in entry]
            if entry is None:
                return None
            if not isinstance(entry, torch.Tensor):
                raise TypeError(f'Unsupported MMCache entry: {type(entry)}')
            return {
                'tensor': entry.detach().to(device='cpu', copy=True),
                'device_type': entry.device.type,
                'pinned': entry.is_pinned(),
                'stride': list(entry.stride()),
            }

        pinned_buffers = None
        if self._pinned_buffers is not None:
            pinned_buffers = {
                name: {
                    li: {'shape': list(buf.shape), 'dtype': str(buf.dtype),
                         'pinned': buf.is_pinned()}
                    for li, buf in layers.items()
                }
                for name, layers in self._pinned_buffers.items()
            }

        return {
            'schema_version': 1,
            'configuration': self._checkpoint_configuration(),
            'cache': {li: {name: snapshot(entry) for name, entry in entries.items()}
                      for li, entries in self.cache.items()},
            'is_update': self._is_update,
            'has_cpu_chunks': self._has_cpu_chunks,
            'lru': {key: list(order) for key, order in self._lru.items()},
            'lru_hits': dict(self._lru_hits),
            'lru_misses': dict(self._lru_misses),
            'lru_cold_misses': dict(self._lru_cold_misses),
            'lru_total_access': self._lru_total_access,
            'lru_total_hits': self._lru_total_hits,
            'lru_total_cold_misses': self._lru_total_cold_misses,
            'lru_recent_window': list(self._lru_recent_window),
            'lru_recent_window_maxlen': self._lru_recent_window.maxlen,
            'to_offload': {key: sorted(pairs) for key, pairs in self._to_offload.items()},
            'last_evict_processed': dict(getattr(self, '_last_evict_processed', {})),
            'lru_saved_cids': {key: sorted(cids) for key, cids in
                               getattr(self, '_lru_saved_cids', {}).items()},
            'valid_counts': {
                li: {name: counts.detach().to(device='cpu', copy=True)
                     for name, counts in entries.items()}
                for li, entries in self.n_chunks_per_head_gpu.items()
            },
            'chunk_strides': {
                li: {name: dict(strides) for name, strides in entries.items()}
                for li, entries in self.chunk_base_strides.items()
            },
            'pinned_buffers': pinned_buffers,
            'pinned_segment_cids': self._pinned_segment_cids,
            'pinned_sample_shape': list(self._pinned_sample_shape),
            'pinned_sample_dtype': (str(self._pinned_sample_dtype)
                                    if self._pinned_sample_dtype is not None else None),
        }

    @staticmethod
    def validate_checkpoint_state_dict(state):
        """Validate a CPU-loaded checkpoint without allocating cache buffers."""
        required = {
            'schema_version', 'configuration', 'cache', 'is_update', 'has_cpu_chunks',
            'lru', 'lru_hits', 'lru_misses', 'lru_cold_misses', 'lru_total_access',
            'lru_total_hits', 'lru_total_cold_misses', 'lru_recent_window',
            'lru_recent_window_maxlen', 'to_offload', 'last_evict_processed',
            'lru_saved_cids', 'valid_counts', 'chunk_strides', 'pinned_buffers',
            'pinned_segment_cids', 'pinned_sample_shape', 'pinned_sample_dtype',
        }
        if not isinstance(state, dict) or not required.issubset(state):
            raise ValueError('Incomplete MMCache checkpoint state')
        if state['schema_version'] != 1:
            raise ValueError('Unsupported MMCache checkpoint schema')
        config = MMCache._without_removed_storage_modes(state['configuration'])
        config_fields = {
            'available_type', 'available_shape', 'seq_dim', 'device_type', 'dtype',
            'num_heads', 'num_layers', 'kv_block_size', 'lru_max_size',
            'chunk_per_head_types', 'chunk_buffer_types',
            'chunked_types', 'state_types', 'block_seqlens', 'max_chunks',
            'window_blocks', 'sink_blocks', 'layers', 'dense_shapes',
        }
        if not isinstance(config, dict) or not config_fields.issubset(config):
            raise ValueError('Incomplete MMCache checkpoint configuration')

        def validate_tensor(entry):
            if not isinstance(entry, dict) or not {
                    'tensor', 'device_type', 'pinned', 'stride'}.issubset(entry):
                raise ValueError('Malformed MMCache checkpoint tensor')
            tensor, stride = entry['tensor'], entry['stride']
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                    or not isinstance(stride, (tuple, list)) or len(stride) != tensor.ndim
                    or any(type(s) is not int or s < 0 for s in stride)
                    or entry['device_type'] not in ('cpu', config['device_type'])
                    or type(entry['pinned']) is not bool):
                raise ValueError('Malformed MMCache checkpoint tensor metadata')

        cache = state['cache']
        if not isinstance(cache, dict) or set(cache) != set(config['layers']):
            raise ValueError('MMCache checkpoint layer topology differs')
        for li, entries in cache.items():
            if not isinstance(entries, dict) or set(entries) != set(config['layers'][li]):
                raise ValueError('MMCache checkpoint cache topology differs')
            for name, entry in entries.items():
                if name not in config['chunked_types']:
                    validate_tensor(entry)
                    if list(entry['tensor'].shape) != config['dense_shapes'].get((li, name)):
                        raise ValueError('MMCache checkpoint dense shape differs')
                    continue
                if not isinstance(entry, list):
                    raise ValueError('Malformed MMCache checkpoint chunk list')
                head_major = name in config['chunk_per_head_types']
                if head_major and len(entry) != config['num_heads']:
                    raise ValueError('MMCache checkpoint head count differs')
                max_items = config['max_chunks'][name]
                if not head_major and len(entry) > max_items:
                    raise ValueError('MMCache checkpoint exceeds chunk capacity')
                for chunk in entry:
                    if chunk is None and not head_major:
                        continue
                    if head_major:
                        if not isinstance(chunk, list):
                            raise ValueError('Malformed MMCache checkpoint head/chunk list')
                        if len(chunk) > max_items:
                            raise ValueError('MMCache checkpoint exceeds chunk capacity')
                        for tensor in chunk:
                            if tensor is not None:
                                validate_tensor(tensor)
                    else:
                        validate_tensor(chunk)
        for name in ('lru', 'lru_hits', 'lru_misses', 'lru_cold_misses', 'to_offload',
                     'last_evict_processed', 'lru_saved_cids', 'valid_counts', 'chunk_strides'):
            if not isinstance(state[name], dict):
                raise ValueError(f'Malformed MMCache checkpoint {name}')
        for li, entries in state['valid_counts'].items():
            if li not in cache or not isinstance(entries, dict):
                raise ValueError('Malformed MMCache checkpoint valid counts')
            for name, counts in entries.items():
                if (name not in config['chunk_per_head_types'] or name not in cache[li]
                        or not isinstance(counts, torch.Tensor) or counts.device.type != 'cpu'
                        or counts.dtype != torch.int32 or tuple(counts.shape) != (config['num_heads'],)):
                    raise ValueError('Malformed MMCache checkpoint valid counts')
                if any(n < 0 or n > len(chunks) for n, chunks in
                       zip(counts.tolist(), cache[li][name])):
                    raise ValueError('MMCache checkpoint valid count exceeds chunk storage')
        if state['pinned_buffers'] is not None and not isinstance(state['pinned_buffers'], dict):
            raise ValueError('Malformed MMCache checkpoint offload buffers')
        for name, layers in (state['pinned_buffers'] or {}).items():
            if name not in config['chunk_per_head_types'] or not isinstance(layers, dict):
                raise ValueError('Malformed MMCache checkpoint offload buffer topology')
            for li, desc in layers.items():
                if li not in cache or not isinstance(desc, dict) or not {
                        'shape', 'dtype', 'pinned'}.issubset(desc):
                    raise ValueError('Malformed MMCache checkpoint offload buffer descriptor')
                shape = desc['shape']
                if (not isinstance(shape, (tuple, list)) or len(shape) < 2
                        or any(type(n) is not int or n < 0 for n in shape)
                        or shape[0] != config['num_heads']
                        or shape[1] != state['pinned_segment_cids']
                        or not isinstance(desc['dtype'], str)
                        or not isinstance(getattr(torch, desc['dtype'].removeprefix('torch.'), None), torch.dtype)
                        or type(desc['pinned']) is not bool):
                    raise ValueError('Malformed MMCache checkpoint offload buffer metadata')
                for chunks in cache[li].get(name, []):
                    for cid, tensor in enumerate(chunks):
                        if tensor is None or tensor['device_type'] != 'cpu':
                            continue
                        tensor_shape = tensor['tensor'].shape
                        if (cid >= shape[1] or len(tensor_shape) != len(shape) - 2
                                or any(n > capacity for n, capacity in zip(tensor_shape, shape[2:]))):
                            raise ValueError('MMCache host chunk exceeds its offload slot')

    def load_checkpoint_state_dict(self, state):
        """Restore into an identically configured cache on this rank's device."""
        self.validate_checkpoint_state_dict(state)
        if (self._without_removed_storage_modes(state.get('configuration'))
                != self._checkpoint_configuration()):
            raise ValueError('MMCache checkpoint configuration differs from the current cache')
        if torch.device(self.device).type == 'cuda':
            torch.cuda.synchronize(self.device)
        # ``restore`` writes into pinned slots from the CPU.
        self.wait_offload_host()

        def dtype_from_name(name):
            dtype = getattr(torch, name.removeprefix('torch.'), None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f'Invalid MMCache checkpoint dtype: {name}')
            return dtype

        pinned_buffers = None
        if state['pinned_buffers'] is not None:
            pinned_buffers = {}
            for name, layers in state['pinned_buffers'].items():
                pinned_buffers[name] = {}
                for li, desc in layers.items():
                    dtype = dtype_from_name(desc['dtype'])
                    existing = (self._pinned_buffers or {}).get(name, {}).get(li)
                    if (existing is not None and list(existing.shape) == desc['shape']
                            and existing.dtype == dtype and existing.is_pinned() == desc['pinned']):
                        buf = existing
                    else:
                        buf = self._alloc_offload_buffer(desc['shape'], dtype=dtype,
                                                         pin_memory=desc['pinned'])
                    pinned_buffers[name][li] = buf

        def restore(entry, slot=None):
            if isinstance(entry, list):
                return [restore(item, slot[i] if slot is not None and i < len(slot) else None)
                        for i, item in enumerate(entry)]
            if entry is None:
                return None
            tensor = entry['tensor']
            device = 'cpu' if entry['device_type'] == 'cpu' else self.device
            if entry['device_type'] not in ('cpu', torch.device(self.device).type):
                raise ValueError('MMCache checkpoint contains an incompatible tensor device')
            if entry['device_type'] == 'cpu' and slot is not None:
                if tensor.ndim != slot.ndim or any(
                        size > capacity for size, capacity in zip(tensor.shape, slot.shape)):
                    raise ValueError('MMCache host chunk exceeds its offload slot shape')
                restored = slot[tuple(slice(0, size) for size in tensor.shape)]
                restored.copy_(tensor)
                return restored
            restored = torch.empty_strided(
                tensor.shape, entry['stride'], dtype=tensor.dtype, device=device,
                pin_memory=entry['pinned'] if torch.device(device).type == 'cpu' else False,
            )
            restored.copy_(tensor)
            return restored

        restored_cache = {
            li: {name: restore(entry, (pinned_buffers or {}).get(name, {}).get(li))
                 for name, entry in entries.items()}
            for li, entries in state['cache'].items()
        }

        self.cache = restored_cache
        self._pinned_buffers = pinned_buffers
        self._pinned_segment_cids = state['pinned_segment_cids']
        self._pinned_sample_shape = tuple(state['pinned_sample_shape'])
        self._pinned_sample_dtype = (dtype_from_name(state['pinned_sample_dtype'])
                                     if state['pinned_sample_dtype'] is not None else None)
        self._is_update = state['is_update']
        self._has_cpu_chunks = state['has_cpu_chunks']
        # Loaded chunks own their storage; no window buffer describes them.
        self._seal_window_buffers()
        self._n_chunks_host.clear()
        self._lru = {key: OrderedDict.fromkeys(order) for key, order in state['lru'].items()}
        # Not serialized: re-derive the fast path's countdown from the restored LRUs.
        self._lru_maxlen = max((len(lru) for lru in self._lru.values()), default=0)
        self._lru_hits = dict(state['lru_hits'])
        self._lru_misses = dict(state['lru_misses'])
        self._lru_cold_misses = dict(state['lru_cold_misses'])
        self._lru_total_access = state['lru_total_access']
        self._lru_total_hits = state['lru_total_hits']
        self._lru_total_cold_misses = state['lru_total_cold_misses']
        self._lru_recent_window = deque(state['lru_recent_window'],
                                        maxlen=state['lru_recent_window_maxlen'])
        self._to_offload = {key: set(map(tuple, pairs)) for key, pairs in state['to_offload'].items()}
        self._last_evict_processed = dict(state['last_evict_processed'])
        self._lru_saved_cids = {key: set(cids) for key, cids in state['lru_saved_cids'].items()}
        self._rebuild_chunk_versions()
        # Tier placement is runtime-only: restored CUDA chunks land on ``self.device``.
        self.recompute_tier_usage()

        # Every prior H2D is done; drop the ring and old backing, whose addresses are freed.
        self._ptr_ring = None
        self.reset_ptr_stats()
        self.chunk_base_ptrs = {}
        self.n_chunks_per_head_gpu = {}
        self._chunk_ptr_backing = {}
        self.chunk_base_strides = {
            li: {name: dict(strides) for name, strides in entries.items()}
            for li, entries in state['chunk_strides'].items()
        }
        for li, entries in self.cache.items():
            for name in self.chunk_per_head_types:
                if name not in entries:
                    continue
                chunks = entries[name]
                if not any(chunks) and name not in state['valid_counts'].get(li, {}):
                    continue
                self._rebuild_chunk_ptrs(li, name)
                counts = state['valid_counts'].get(li, {}).get(name)
                if counts is not None:
                    self.n_chunks_per_head_gpu[li][name].copy_(counts)
        # Runtime metadata, never serialized: rebuild pools and address table.
        self._reconfigure_block_store()
        return self

    def to(self, device=None, dtype=None):
        if device is None and dtype is None:
            return self
        # Finish in-flight copies on the cache device's streams before reading host/tier chunks.
        if torch.device(self.device).type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.wait_offload_host()
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype

        for layer_idx in self.cache:
            for cache_type in self.cache[layer_idx]:
                entry = self.cache[layer_idx][cache_type]
                if isinstance(entry, list):
                    if cache_type in self.chunk_per_head_types:
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
            # Move the shared backing and re-derive the views; the ring belongs to the old device.
            if self._ptr_ring is not None:
                self._ptr_ring.drain(self.ptr_stats)
                self._ptr_ring = None
            for layer_idx in list(self._chunk_ptr_backing.keys()):
                self._chunk_ptr_backing[layer_idx] = (
                    self._chunk_ptr_backing[layer_idx].to(device)
                )
            for layer_idx in list(self.chunk_base_ptrs.keys()):
                backing = self._chunk_ptr_backing.get(layer_idx)
                for name in list(self.chunk_base_ptrs[layer_idx].keys()):
                    if backing is not None:
                        self.chunk_base_ptrs[layer_idx][name] = (
                            backing[self._ptr_name_index[name]]
                        )
                    self.n_chunks_per_head_gpu[layer_idx][name] = (
                        self.n_chunks_per_head_gpu[layer_idx][name].to(device)
                    )

        if device is not None or dtype is not None:
            # A dtype-only conversion reallocates every chunk, so recorded addresses are stale.
            for layer_idx in list(self.chunk_base_ptrs.keys()):
                for name in list(self.chunk_base_ptrs[layer_idx].keys()):
                    self._rebuild_chunk_ptrs(layer_idx, name)
            # No slot may claim validity after chunks were reallocated; chunk versions stay.
            self._reset_slot_meta()
            # to(device=...) empties the tier and a dtype change resizes tier tensors, so rescan.
            self.recompute_tier_usage()
            # Block store pools, backing and address table describe the old device/dtype: rebuild.
            self._reconfigure_block_store()
            # The window buffers describe tensors that no longer exist: seal until reset().
            self._seal_window_buffers()
            self._n_chunks_host.clear()

        return self

    # ──────────────────────────────────────────────────────────
    # Chunk read helpers
    # ──────────────────────────────────────────────────────────

    def _cat_chunks(self, layer_idx: int, name: str) -> Optional[torch.Tensor]:
        chunks = self.cache[layer_idx][name]
        if not chunks:
            return None

        # A tier chunk is on the peer GPU; torch.cat needs one device.
        device = self._cache_device() if self._tier_bytes else None

        if name in self.chunk_per_head_types:
            H = self.num_heads
            per_head = []
            for h in range(H):
                head_items = [c for c in chunks[h] if c is not None]
                if not head_items:
                    return None
                if device is not None:
                    head_items = self._align_chunk_device(head_items, device)
                per_head.append(
                    torch.cat(head_items, dim=self.seq_dim) if len(head_items) > 1 else head_items[0]
                )
            return torch.stack(per_head, dim=2)
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

        if is_chunk_per_head:
            chunk_id = start_id_int // block_seqlen

            # One permute+contiguous gives [H, B, S, D] in a single kernel, then per-head views.
            stacked = hidden_state.permute(2, 0, 1, 3).contiguous()
            # Overwriting a cid that lives on the GPU tier drops its bytes from the tier budget.
            self._tier_untrack_cid(layer_idx, name, chunk_id)
            new_tensors = []
            for h in range(self.num_heads):
                head_slice = stacked[h]  # view, no copy
                if chunk_id < len(chunks[h]):
                    chunks[h][chunk_id] = head_slice
                else:
                    chunks[h].append(head_slice)
                new_tensors.append(head_slice)

            # Any write is new content: the fresh version invalidates slots holding the old one.
            version = self._bump_chunk_version(layer_idx, name, chunk_id)

            # Sliding-window ring: mirror the chunk so the window build is one ``narrow``.
            if self._window_buf_enabled:
                self._window_ring_write(layer_idx, name, chunk_id, hidden_state)

            # Kept in both policies so other readers keep their tensor shapes and strides.
            self._ensure_chunk_ptr_tables(layer_idx, name)
            self._record_chunk_strides(layer_idx, name, new_tensors[0])

            if self._block_store is not None:
                # 'block_lru': the block address table is the authority; never touch the chunk LRU.
                self._block_store.on_chunk_written(
                    layer_idx, name, chunk_id, stacked, version)
            else:
                # Publish every head of the written cid now: the table must be current on return.
                self.publish_chunk_ptrs(layer_idx, (
                    (name, h, chunk_id, self.chunk_ptr_of(t))
                    for h, t in enumerate(new_tensors)
                ))
                if self._has_ptr_table(layer_idx, name):
                    # ``clamp_min_`` only raises the counter, so a covering host mirror is a no-op.
                    nkey = (layer_idx, name)
                    known = self._n_chunks_host.get(nkey)
                    covered = known is not None and known >= chunk_id + 1
                    if covered and self._resident_fast_path:
                        self.fast_path_stats['clamp_skipped'] += 1
                    else:
                        self.n_chunks_per_head_gpu[layer_idx][name].clamp_min_(chunk_id + 1)
                        if self._resident_fast_path:
                            self.fast_path_stats['clamp_published'] += 1
                    if not covered:
                        self._n_chunks_host[nkey] = chunk_id + 1

                if name == 'k_cache':
                    for h in range(self.num_heads):
                        self.lru_touch_per_head(layer_idx, 'k_cache', h, chunk_id)

        else:
            chunk_id = start_id_int // block_seqlen
            store = hidden_state
            if self._window_buf_enabled:
                # The chunk lands in its fixed history-buffer slice; the list holds that view.
                buffered = self._flat_buffer_write(
                    layer_idx, name, chunk_id, hidden_state)
                if buffered is not None:
                    store = buffered
            if chunk_id < len(chunks):
                chunks[chunk_id] = store
            else:
                chunks.append(store)

        if is_chunk_per_head:
            return self._build_window_chunk_per_head(name, chunks, chunk_id + 1, hidden_state.device,
                                                     layer_idx=layer_idx)
        else:
            return self._build_window_flat(name, chunks, chunk_id + 1, hidden_state.device,
                                           layer_idx=layer_idx)

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

    def _resolve_window_range(self, name: str, num_valid: int):
        wb = self.window_blocks.get(name)
        sb = self.sink_blocks.get(name) or 0

        if wb is None:
            return 0, 0, num_valid
        if sb == 0:
            return 0, max(0, num_valid - wb), num_valid
        effect_sink = min(sb, num_valid)
        win_start = max(effect_sink, num_valid - wb)
        return effect_sink, win_start, num_valid

    def _build_window_chunk_per_head(self, name, chunks, num_valid, device, layer_idx=None):
        if self._window_buf_enabled and layer_idx is not None:
            view = self._window_ring_view(layer_idx, name, num_valid, device)
            if view is not None:
                return view
        sink_end, win_start, _ = self._resolve_window_range(name, num_valid)
        per_head_seqs = []
        for h in range(self.num_heads):
            if self.window_blocks.get(name) is None:
                head_rel = chunks[h][:num_valid]
            else:
                head_rel = chunks[h][:sink_end] + chunks[h][win_start:num_valid]
            head_rel = self._align_chunk_device(head_rel, device)
            per_head_seqs.append(
                torch.cat(head_rel, dim=self.seq_dim) if len(head_rel) > 1 else head_rel[0]
            )
        return torch.stack(per_head_seqs, dim=2)

    def _build_window_flat(self, name, chunks, num_valid, device, layer_idx=None):
        if (self._window_buf_enabled and layer_idx is not None
                and self.window_blocks.get(name) is None):
            view = self._flat_buffer_view(layer_idx, name, num_valid, device)
            if view is not None:
                return view
        sink_end, win_start, _ = self._resolve_window_range(name, num_valid)
        if self.window_blocks.get(name) is None:
            relevant = chunks[:num_valid]
        else:
            relevant = chunks[:sink_end] + chunks[win_start:num_valid]

        if any(c.device != device for c in relevant):
            relevant = self._align_chunk_device(relevant, device)
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
        if name in self.chunk_per_head_types:
            if head is None or head >= len(chunks) or cid >= len(chunks[head]):
                return False
            t = chunks[head][cid]
        else:
            if cid >= len(chunks):
                return False
            t = chunks[cid]
        # is_cuda is right: a tier chunk is read in place over P2P, so a miss is as cold as local.
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
            if len(lru) > self._lru_maxlen:
                self._lru_maxlen = len(lru)

            if cold:
                self._lru_cold_misses[key] += 1
                self._lru_total_cold_misses += 1
            else:
                self._lru_total_access += 1
                self._lru_misses[key] += 1
                self._lru_recent_window.append(0)

    def lru_touch_many(self, layer_idx: int, name: str, pairs: Iterable[Tuple[int, int]]):
        """Same as looping lru_touch_per_head over the pairs, with the key lookup hoisted."""
        if self.lru_max_size <= 0:
            return
        cur_h = None
        key = None
        lru = None
        for h, cid in pairs:
            if h != cur_h:
                cur_h = h
                key = (layer_idx, name, h)
                lru = self._ensure_lru(key)

            if cid in lru:
                lru.move_to_end(cid)
                self._lru_total_access += 1
                self._lru_hits[key] += 1
                self._lru_total_hits += 1
                self._lru_recent_window.append(1)
            else:
                cold = self._is_chunk_on_gpu(layer_idx, name, h, cid)

                if len(lru) >= self.lru_max_size:
                    evicted_cid, _ = lru.popitem(last=False)
                    self._mark_for_offload(layer_idx, name, h, evicted_cid)
                lru[cid] = None
                if len(lru) > self._lru_maxlen:
                    self._lru_maxlen = len(lru)

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
        self._lru_maxlen = 0
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
                if name in self.chunk_per_head_types:
                    for h in range(self.num_heads):
                        if chunks[h]:
                            chunks[h].pop()
                    cid_popped = len(chunks[0]) if chunks else 0
                    # The cid is gone; its slot must not be inherited by whatever is appended next.
                    self._truncate_chunk_versions(li, name)
                    self._invalidate_slot(li, name, cid_popped)
                    # A popped tier chunk released its bytes with its refs.
                    self._tier_untrack_cid(li, name, cid_popped)
                    # Same for the block store: pool copies, host versions and block range go away.
                    if self._block_store is not None:
                        self._block_store.on_chunk_freed(li, name, cid_popped)
                    # Same for the window ring: no reads of the cid's slot until a store refills it.
                    self.invalidate_window_ring(li, cid_popped, name)
                    if self._has_ptr_table(li, name):
                        cid_popped = len(chunks[0])
                        if cid_popped < self.chunk_base_ptrs[li][name].shape[1]:
                            self.chunk_base_ptrs[li][name][:, cid_popped] = 0
                        self.n_chunks_per_head_gpu[li][name].clamp_max_(cid_popped)
                        nkey = (li, name)
                        known = self._n_chunks_host.get(nkey)
                        if known is not None:
                            self._n_chunks_host[nkey] = min(known, cid_popped)
                else:
                    if chunks:
                        chunks.pop()
                        fkey = (li, name)
                        if fkey in self._flat_buf:
                            n_left = len(chunks)
                            blk = self.block_seqlens.get(name) or 0
                            self._flat_buf_last_cid[fkey] = n_left - 1
                            self._flat_buf_tokens[fkey] = max(0, n_left * blk)
                            self._flat_buf_count[fkey] = min(
                                self._flat_buf_count.get(fkey, 0), n_left)
                            if self._flat_buf_short.get(fkey, -1) >= n_left:
                                self._flat_buf_short.pop(fkey, None)

    # ──────────────────────────────────────────────────────────
    # Lifecycle: reset / release
    # ──────────────────────────────────────────────────────────

    def reset(self):
        for li in self.cache:
            for cache_type in self.cache[li]:
                if cache_type in self.chunk_buffer_types:
                    self.cache[li][cache_type].zero_()
                elif cache_type in self.chunked_types:
                    if cache_type in self.chunk_per_head_types:
                        self.cache[li][cache_type] = [[] for _ in range(self.num_heads)]
                    else:
                        self.cache[li][cache_type] = []
                else:
                    self.cache[li][cache_type].zero_()

        # Drain before zeroing, or an in-flight publish revives a dead address after zero_().
        if self._ptr_ring is not None:
            self._ptr_ring.drain(self.ptr_stats)
        for backing in self._chunk_ptr_backing.values():
            backing.zero_()
        for li in self.chunk_base_ptrs:
            for name in self.chunk_base_ptrs[li]:
                self.n_chunks_per_head_gpu[li][name].zero_()
        self.chunk_base_strides.clear()

        if hasattr(self, '_last_evict_processed'):
            self._last_evict_processed.clear()
        if hasattr(self, '_lru_saved_cids'):
            self._lru_saved_cids.clear()

        self.lru_clear()
        self._to_offload.clear()

        # Pinned buffers are reused; every chunk is gone so no slot may claim live content.
        self._chunk_version.clear()
        self._reset_slot_meta()

        # Tier tensors are released and the budget returns to zero; the configuration survives.
        self._tier_bytes.clear()
        self.tier_used_bytes = 0
        self._tier_resident_device = None
        self.tier_stats['used_bytes'] = 0

        # The block store keeps its allocations but forgets every mapping, version, flag and epoch.
        if self._block_store is not None:
            self._block_store.reset()

        # Window buffers keep their allocations but forget every chunk.
        self._flat_buf_tokens.clear()
        self._flat_buf_last_cid.clear()
        self._flat_buf_count.clear()
        self._flat_buf_short.clear()
        self._flat_buf_blocked.clear()
        self._win_ring_valid = {k: set() for k in self._win_ring}
        self._win_ring_blocked.clear()
        self._window_buf_sealed = False
        self._n_chunks_host.clear()

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

                if name in self.chunk_per_head_types:
                    evicted |= self._evict_chunk_per_head(li, name, chunks, wb, sb, min_gpu_chunks)
                else:
                    evicted |= self._evict_flat(li, name, chunks, wb, sb, min_gpu_chunks)

        if evicted:
            # The D2H copies run on the cache device's current stream, not the thread's device.
            torch.cuda.current_stream(self._cache_device()).synchronize()
            self._has_cpu_chunks = True

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
        ptr_zeros = []

        for i in cids_to_check:
            to_evict = []
            has_lru = False
            for h in range(self.num_heads):
                if i >= len(chunks[h]):
                    continue
                t = chunks[h][i]
                if t is None or not t.is_cuda:
                    continue
                if self._on_tier(t):
                    # Already on the peer tier: re-evicting it would copy peer->host for nothing.
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
            ptr_zeros.extend((name, h, i, 0) for h in to_evict)
            evicted = True

        # One publish batch for every cid this eviction pass touched.
        if ptr_zeros and self._has_ptr_table(li, name):
            self.publish_chunk_ptrs(li, ptr_zeros)

        self._last_evict_processed[key] = max(last, evict_before - 1)
        self._lru_saved_cids[key] = still_saved
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
            if chunk is not None and chunk.is_cuda and not self._on_tier(chunk):
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
        segment_cids: int = 10,
    ):
        """Allocate per-(name, layer) pinned host buffers, (num_heads, segment_cids, *sample)."""
        per_slot_shape = tuple(sample.shape)

        if (not self.chunk_per_head_types
                or self.num_heads is None
                or not self.cache):
            # Nothing offloadable through the slot-table path.
            self._pinned_sample_shape = per_slot_shape
            self._pinned_sample_dtype = sample.dtype
            self._pinned_segment_cids = segment_cids
            self._pinned_buffers = {}
            self._reset_slot_meta()
            return

        layer_ids = sorted(self.cache.keys())
        if self._pinned_buffers_match(per_slot_shape, sample.dtype, segment_cids, layer_ids):
            # Same layout as the held buffers: reuse them instead of page-locking again.
            self._ensure_slot_meta()
            return

        self._pinned_sample_shape = per_slot_shape
        self._pinned_sample_dtype = sample.dtype
        self._pinned_segment_cids = segment_cids
        self._pinned_buffers = {}
        for name in sorted(self.chunk_per_head_types):
            self._pinned_buffers[name] = {}
            for li in layer_ids:
                self._pinned_buffers[name][li] = self._alloc_offload_buffer(
                    (self.num_heads, segment_cids) + per_slot_shape,
                    dtype=sample.dtype,
                    pin_memory=True,
                )
        # Brand-new (uninitialised) storage: no slot holds anyone's content.
        self._reset_slot_meta()

    def configure_exact_pinned_host(self, enabled: Optional[bool] = None) -> None:
        """Choose the source of page-locked host KV buffers: exact-size, torch allocator, or env."""
        self._exact_pinned_host = None if enabled is None else bool(enabled)

    def exact_pinned_host(self) -> bool:
        """Whether a host KV buffer allocated now is an exact-size registered mapping."""
        if self._exact_pinned_host is None:
            return exact_pinned_kv_enabled()
        return self._exact_pinned_host

    def _alloc_offload_buffer(self, shape, dtype, pin_memory: bool) -> torch.Tensor:
        """Allocate an offload buffer without host-allocator rounding when enabled."""
        if pin_memory and self.exact_pinned_host():
            return exact_pinned_empty(shape, dtype, self.device)
        return torch.empty(shape, dtype=dtype, device='cpu', pin_memory=pin_memory)

    def _pinned_buffers_match(self, per_slot_shape, dtype, segment_cids, layer_ids) -> bool:
        bufs = self._pinned_buffers
        if not bufs:
            return False
        if (self._pinned_sample_shape != per_slot_shape or self._pinned_sample_dtype != dtype
                or self._pinned_segment_cids != segment_cids):
            return False
        expected = (self.num_heads, segment_cids) + per_slot_shape
        for name in sorted(self.chunk_per_head_types):
            per_layer = bufs.get(name)
            if per_layer is None or sorted(per_layer.keys()) != list(layer_ids):
                return False
            for t in per_layer.values():
                if tuple(t.shape) != expected or t.dtype != dtype or not t.is_pinned():
                    return False
        return True

    def _lazy_segment_cids(self) -> int:
        """``segment_cids`` for a pinned buffer created inside ``flush_offload``: ``max_chunks``."""
        widest = max((self.max_chunks.get(n, 0) for n in self.chunk_per_head_types),
                     default=0)
        return max(self.SEGMENT_CIDS, int(widest))

    def _plan_tier_placement(self, to_evict: List[tuple]) -> set:
        """Indices of ``to_evict`` whose chunk goes to the GPU tier; K and V must land together."""
        if self.tier_device is None or self.tier_budget_bytes <= 0:
            return set()
        groups: Dict[tuple, List[int]] = {}
        for idx, (li, _name, h, i, _t) in enumerate(to_evict):
            groups.setdefault((li, h, i), []).append(idx)
        chosen = set()
        used = self.tier_used_bytes
        budget = self.tier_budget_bytes
        for idxs in groups.values():
            total = 0
            for j in idxs:
                t = to_evict[j][4]
                total += t.numel() * t.element_size()
            if used + total <= budget:
                used += total
                chosen.update(idxs)
        return chosen

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
        """Move each queued (layer, name, head, cid) to the GPU tier if it fits, else to host."""
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
                if self._on_tier(t):
                    # Already off this card: re-evicting it would copy peer->host for nothing.
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
        if not to_evict:
            for key in keys_drained:
                del self._to_offload[key]
            return

        # Phase 2: split the batch between the GPU tier and the pinned host path.
        tier_idx = self._plan_tier_placement(to_evict)
        host_idx = [j for j in range(len(to_evict)) if j not in tier_idx]

        # Pinned buffers are only for the host fallback; lazy sizing must cover the whole cache.
        if host_idx and self._pinned_buffers is None:
            self.preallocate_offload_buffer(
                sample=to_evict[host_idx[0]][4],
                segment_cids=self._lazy_segment_cids(),
            )

        seg_cids = self._pinned_segment_cids
        for j in host_idx:
            _, name, _, i, _ = to_evict[j]
            if i >= seg_cids:
                # Refused before anything moved (and with the queue intact).
                raise RuntimeError(
                    f"chunk id {i} exceeds preallocated segment_cids="
                    f"{seg_cids} for cache type '{name}'. "
                    f"Increase segment_cids in preallocate_offload_buffer()."
                )
        if host_idx:
            self._ensure_slot_meta()
            if self.tier_device is not None and self.tier_budget_bytes > 0:
                self.tier_stats['fallback_host'] += len(host_idx)
        stats = self.offload_stats
        tier_stats = self.tier_stats

        # Phase 3: tier entries go peer-to-peer; host entries go D2H unless the slot is current.
        ptr_by_layer = {}
        copied_from = None
        host_evicted = False
        tier_device = self.tier_device
        tier_ctx = (torch.cuda.stream(self._tier_copy_stream())
                    if tier_idx else contextlib.nullcontext())
        try:
            with tier_ctx:
                for idx, (li, name, h, i, t) in enumerate(to_evict):
                    nbytes = t.numel() * t.element_size()

                    if idx in tier_idx:
                        dst = torch.empty_like(t, device=tier_device)
                        dst.copy_(t, non_blocking=True)
                        self.cache[li][name][h][i] = dst
                        self._tier_resident_device = tier_device
                        self._tier_track((li, name, h, i), nbytes)
                        tier_stats['evictions'] += 1
                        tier_stats['bytes'] += nbytes
                        # NON-zero: a tier chunk is never missing, so no reload is triggered for it.
                        ptr_by_layer.setdefault(li, []).append((name, h, i, dst.data_ptr()))
                        continue

                    full_slot = self._pinned_buffers[name][li][h, i]
                    shape = tuple(t.shape)
                    meta_row = self._slot_meta.get(name, {}).get(li)
                    meta = meta_row[h][i] if meta_row is not None else None
                    chunk_v = self.chunk_version(li, name, h, i)

                    if (meta is not None and chunk_v >= 0 and meta[0] == chunk_v
                            and meta[1] == shape and meta[2] == t.dtype):
                        # Same bytes already backed up: skip the D2H, keep the recorded shape.
                        slot = (full_slot if shape == tuple(full_slot.shape)
                                else full_slot[tuple(slice(0, s) for s in shape)])
                        stats['skipped_calls'] += 1
                        stats['skipped_bytes'] += nbytes
                        stats['slot_reuse'] += 1
                    else:
                        if t.shape == full_slot.shape:
                            full_slot.copy_(t, non_blocking=True)
                            slot = full_slot
                        else:
                            # Partial chunk: copy only t's actual shape and store a same-shape view.
                            slot = full_slot[tuple(slice(0, s) for s in shape)]
                            slot.copy_(t, non_blocking=True)
                        if chunk_v < 0:
                            # No recorded version: give it one so a later eviction can skip.
                            chunk_v = self._next_version()
                            self.set_chunk_version(li, name, h, i, chunk_v)
                        if meta_row is not None:
                            meta_row[h][i] = (chunk_v, shape, slot.dtype)
                        stats['d2h_calls'] += 1
                        stats['d2h_bytes'] += nbytes
                        if t.is_cuda:
                            copied_from = t.device
                    self.cache[li][name][h][i] = slot
                    host_evicted = True
                    ptr_by_layer.setdefault(li, []).append((name, h, i, 0))
        finally:
            # In a ``finally``: chunks moved before an interruption have still left this card.
            if copied_from is not None:
                self._record_offload_event(copied_from)

            # Phase 4: one publish batch per layer; the table is authoritative on return.
            for li, entries in ptr_by_layer.items():
                self.publish_chunk_ptrs(li, entries)

            if host_evicted:
                # Only a host eviction makes a chunk missing; a tier-only flush leaves all valid.
                self._has_cpu_chunks = True

        for key in keys_drained:
            del self._to_offload[key]
