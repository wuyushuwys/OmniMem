"""Block-level KV cache store (``kv_cache_policy=block_lru``), owned by MMCache; inference only.
The block is the unit of hit/refill/eviction; one device table [2, H, MB] of int64 byte
addresses feeds the pointer builder.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from omnimem.models.cache import _HostCopyEvent, _PtrStagingRing, pinned_host_empty
from omnimem.nsa.backend import resolve_block_kv_backend
from omnimem.nsa.ops.selection.padded_ptr.kernels.block_kv_kernels import (
    clear_block_range,
    gather_publish_clear,
    register_block_range,
    table_entry_addresses,
)

__all__ = ["BlockKVStore", "BlockSelectionLease", "fresh_block_stats",
           "fresh_per_layer_stats", "PER_LAYER_STAT_KEYS"]


# ``k``/``v`` are the store's internal names; ``k_cache``/``v_cache`` are MMCache's.
NAMES: Tuple[str, str] = ("k", "v")
CACHE_NAME: Dict[str, str] = {"k": "k_cache", "v": "v_cache"}
STORE_NAME: Dict[str, str] = {"k_cache": "k", "v_cache": "v"}

_EMPTY_I64 = np.empty(0, dtype=np.int64)
_EMPTY_I32 = np.empty(0, dtype=np.int32)
_EMPTY_I8 = np.empty(0, dtype=np.int8)

#: ``n_valid_value`` sentinel of the K2 kernels: "leave the valid extent unchanged".
_KEEP_EXTENT = -1

# Above this epoch count the counting-sort shortcut in _pick_victims allocates more than it saves.
_VICTIM_BINS_MAX = 1 << 17


def fresh_block_stats() -> Dict[str, float]:
    """All store-wide counters at zero (capacities are filled in by the store)."""
    return dict(
        calls=0, fast_path_calls=0, selected_unique=0, protected_hits=0,
        persistent_hits=0, misses=0, persistent_evictions=0, scratch_blocks=0,
        scratch_calls=0, peak_scratch_used=0, refetch_evicted=0, refetch_version=0,
        h2d_bytes=0, d2h_bytes=0, d2d_bytes=0, gather_launches=0,
        clear_only_launches=0, register_launches=0, clear_launches=0,
        dedup_launches=0,
        copy_api_calls=0,
        desc_h2d_bytes=0, desc_h2d_calls=0, meta_d2h_bytes=0, meta_d2h_calls=0,
        publish_entries=0, clear_entries=0, ring_waits=0, ring_wait_s=0.0,
        ring_grows=0, policy_cpu_s=0.0, prepare_s=0.0, meta_d2h_s=0.0,
        stream_switches=0, retired_chunks=0, whole_chunk_reload_calls=0,
        persistent_capacity=0, scratch_capacity=0, pool_bytes=0, scratch_bytes=0,
        host_backing_bytes=0,
    )


# Per-layer counters; an eviction is charged to the layer that caused it and lost the slot.
PER_LAYER_STAT_KEYS: Tuple[str, ...] = (
    "misses", "persistent_hits", "evictions", "h2d_bytes")


def fresh_per_layer_stats(n_layers: int) -> Dict[str, np.ndarray]:
    return {k: np.zeros(max(0, int(n_layers)), dtype=np.int64)
            for k in PER_LAYER_STAT_KEYS}


class BlockSelectionLease:
    """What one selection call borrowed from the shared scratch pool."""

    __slots__ = ("layer_idx", "scratch_keys", "n_fetch")

    def __init__(self, layer_idx: int = -1, scratch_keys: np.ndarray = _EMPTY_I64,
                 n_fetch: int = 0):
        self.layer_idx = layer_idx
        self.scratch_keys = scratch_keys
        self.n_fetch = n_fetch

    @property
    def n_scratch(self) -> int:
        return int(self.scratch_keys.size)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"BlockSelectionLease(layer={self.layer_idx}, n_fetch={self.n_fetch}, "
                f"n_scratch={self.n_scratch})")


_EMPTY_LEASE = BlockSelectionLease()


class BlockKVStore(_HostCopyEvent):
    """Per-``MMCache`` block pools, authoritative address table and block LRU."""

    def __init__(self, *, layers: Sequence[int], num_heads: int, block_tokens: int,
                 head_dim: int, chunk_len: int, max_chunks: int, batch: int,
                 device, dtype, persistent_bytes: int, scratch_bytes: int,
                 resident_if_fits: bool = True, history_bytes: int = 0,
                 dedup: str = 'device',
                 block_kv_backend: str | None = None):
        self.layers: List[int] = sorted(int(x) for x in layers)
        if not self.layers:
            raise ValueError("the block KV store needs at least one layer")
        self.num_heads = self.H = int(num_heads)
        self.block_tokens = self.bt = int(block_tokens)
        self.head_dim = self.D = int(head_dim)
        self.chunk_len = self.S_max = int(chunk_len)
        self.MC = int(max_chunks)
        self.batch = self.B = int(batch)
        if self.B != 1:
            raise ValueError(f"the block KV store requires batch size 1, got {self.B}")
        if self.bt <= 0 or self.S_max <= 0 or self.S_max % self.bt:
            raise ValueError(
                f"chunk length {self.S_max} must be a positive multiple of "
                f"kv_block_size {self.bt}")
        self.bpc = self.S_max // self.bt
        self.MB = self.MC * self.bpc
        if self.MB <= 0:
            raise ValueError("the block KV store needs a positive block count")

        self.device = torch.device(device)
        self.dtype = dtype
        self.elem = torch.empty(0, dtype=dtype).element_size()
        if self.elem != 2:
            raise ValueError(
                f"the block KV store supports 16-bit KV dtypes only, got {dtype}")
        self.block_bytes = self.bt * self.D * self.elem
        self.host_head_stride = self.B * self.S_max * self.D * self.elem
        #: Request generation; bumped by every ``reset()``.
        self.generation = 0

        if dedup not in ('unique', 'device'):
            raise ValueError(f"dedup must be 'unique' or 'device', got {dedup!r}")
        self.dedup = str(dedup)
        self.resident_if_fits = bool(resident_if_fits)
        # Resolved once at construction and passed to every K1/K2/K3 launch.
        self.block_kv_backend = resolve_block_kv_backend(block_kv_backend)
        if self.block_kv_backend == "cuda":
            # Surface a build failure at construction, not mid-request.
            from omnimem.nsa.cuda_backend import block_kv_ops
            block_kv_ops.load()
            # CUDA loads kernel modules lazily; take the first-launch stall here.
            block_kv_ops.warmup(device)

        self.persistent_bytes = max(0, int(persistent_bytes))
        self.scratch_bytes = max(0, int(scratch_bytes))
        n_layers = len(self.layers)
        pair = 2 * self.block_bytes
        # Resident-if-fits: if the whole K+V history fits the GPU budget, allocate nothing.
        self.resident_history_bytes = max(0, int(history_bytes))
        self.resident_budget_bytes = self.persistent_bytes + self.scratch_bytes
        self.resident = bool(
            self.resident_if_fits and self.resident_history_bytes > 0
            and self.resident_history_bytes <= self.resident_budget_bytes)
        self.mode = 'resident' if self.resident else 'lru'
        if self.resident:
            self.C = self.Sc = 0
        else:
            self.C = int(self.persistent_bytes // (n_layers * pair))
            self.Sc = int(self.scratch_bytes // pair)
        # Keys per chunk id; a cid owns ``[cid * _cid_stride, (cid + 1) * _cid_stride)``.
        self._cid_stride = self.H * self.bpc

        self.stats: Dict[str, float] = fresh_block_stats()
        self._layer_pos: Dict[int, int] = {L: i for i, L in enumerate(self.layers)}
        self._per_layer: Dict[str, np.ndarray] = fresh_per_layer_stats(n_layers)

        self.table: Dict[int, torch.Tensor] = {}
        self.n_valid: Dict[int, torch.Tensor] = {}
        self.pool_k: Dict[int, torch.Tensor] = {}
        self.pool_v: Dict[int, torch.Tensor] = {}
        self._pool_ptr: Dict[int, Tuple[int, int]] = {}
        for L in self.layers:
            self.table[L] = torch.zeros(2, self.H, self.MB, dtype=torch.int64,
                                        device=self.device)
            self.n_valid[L] = torch.zeros(self.H, dtype=torch.int32, device=self.device)
            pk = torch.empty((self.C, self.bt, self.D), dtype=dtype, device=self.device)
            pv = torch.empty_like(pk)
            self.pool_k[L], self.pool_v[L] = pk, pv
            self._pool_ptr[L] = (pk.data_ptr(), pv.data_ptr())
        self.scratch_k = torch.empty((self.Sc, self.bt, self.D), dtype=dtype,
                                     device=self.device)
        self.scratch_v = torch.empty_like(self.scratch_k)
        self._scratch_ptr = (self.scratch_k.data_ptr(), self.scratch_v.data_ptr())
        # Only the element type is used by K1; one element keeps a clear-only launch working.
        self._typed = torch.empty(1, dtype=dtype, device=self.device)
        # Fixed device descriptor buffer, copy_'d into per call; resident mode gets none.
        self.desc_dev = torch.empty(
            0 if self.resident else 8 * (self.C + self.Sc) + 64,
            dtype=torch.int64, device=self.device)
        self._h_idx = (torch.arange(self.H, dtype=torch.int64, device=self.device)
                       .view(1, 1, self.H, 1))
        self._sentinel = self.H * self.MB
        # O(C) scratch for the destination-uniqueness check (``_slot_pos`` / ``_iota``).
        self._slot_pos = np.zeros(max(1, self.C), dtype=np.int64)
        self._iota = np.arange(max(1, self.C), dtype=np.int64)
        # Preallocated O(C) boolean work buffers for the planner's free / candidate scans.
        self._mask_a = np.empty(max(1, self.C), dtype=bool)
        self._mask_b = np.empty(max(1, self.C), dtype=bool)

        self._ring = _PtrStagingRing()
        # Keyed deferred clears: parallel (layer, name_row, key) arrays.
        self._pc_layer: np.ndarray = _EMPTY_I32
        self._pc_row: np.ndarray = _EMPTY_I8
        self._pc_key: np.ndarray = _EMPTY_I64
        self._scratch_event: Optional[torch.cuda.Event] = None
        self._scratch_stream = None
        self._owner_stream = None
        # K3 dedup object: built lazily from the first call's ``block_indices`` shape.
        self._dedup = None

        self.cver: Dict[int, Dict[str, np.ndarray]] = {}
        self.host_ver: Dict[int, Dict[str, np.ndarray]] = {}
        self.prot: Dict[int, Dict[str, np.ndarray]] = {}
        self.gpu_chunk: Dict[int, Dict[str, List[Optional[torch.Tensor]]]] = {}
        self.host_base: Dict[int, Dict[str, np.ndarray]] = {}
        self.nblk: Dict[int, np.ndarray] = {}
        self.retired: Dict[int, np.ndarray] = {}
        self.frozen: Dict[int, np.ndarray] = {}
        self.pslot: Dict[int, np.ndarray] = {}
        self.slot_key: Dict[int, np.ndarray] = {}
        self.slot_touch: Dict[int, np.ndarray] = {}
        self.fetched_once: Dict[int, np.ndarray] = {}
        self.fetched_ever: Dict[int, np.ndarray] = {}
        self._epoch: Dict[int, int] = {}
        self._n_valid_val: Dict[int, int] = {}
        self._host_active: Dict[int, bool] = {}
        for L in self.layers:
            self._init_layer_meta(L)

        # Pinned chunk-major host backing, allocated by ``preallocate_host``.
        self.backing: Dict[int, Dict[str, torch.Tensor]] = {}
        self._host_rows = 0
        self._refresh_capacity_stats()

    def _init_layer_meta(self, L: int) -> None:
        MC, H, MB, C = self.MC, self.H, self.MB, self.C
        self.cver[L] = {n: np.full((H, MC), -1, dtype=np.int64) for n in NAMES}
        self.host_ver[L] = {n: np.full((H, MC), -1, dtype=np.int64) for n in NAMES}
        self.prot[L] = {n: np.zeros(MC, dtype=bool) for n in NAMES}
        self.gpu_chunk[L] = {n: [None] * MC for n in NAMES}
        self.host_base[L] = {n: np.zeros(MC, dtype=np.int64) for n in NAMES}
        self.nblk[L] = np.zeros(MC, dtype=np.int32)
        self.retired[L] = np.zeros(MC, dtype=bool)
        self.frozen[L] = np.zeros(MC, dtype=bool)
        # Flat and indexed by the host-order key, so one cid is ONE slice.
        self.pslot[L] = np.full(H * MB, -1, dtype=np.int32)
        self.slot_key[L] = np.full(C, -1, dtype=np.int64)
        self.slot_touch[L] = np.full(C, -1, dtype=np.int64)
        self.fetched_once[L] = np.zeros(H * MB, dtype=bool)
        self.fetched_ever[L] = np.zeros(H * MB, dtype=bool)
        self._epoch[L] = 0
        self._n_valid_val[L] = 0
        self._host_active[L] = False

    def decode_key(self, keys):
        """``keys = (cid * H + h) * bpc + loc`` -> ``(h, cid, loc, bid)`` as int64 arrays."""
        keys = np.asarray(keys, dtype=np.int64)
        rest = keys // self.bpc
        loc = keys - rest * self.bpc
        cid = rest // self.H
        h = rest - cid * self.H
        return h, cid, loc, cid * self.bpc + loc

    def _table_index(self, keys):
        """Host-order key -> the flat ``[H, MB]`` index ``h * MB + bid``."""
        h, cid, loc, _ = self.decode_key(keys)
        return h * self.MB + cid * self.bpc + loc

    def cid_key_range(self, cid: int) -> Tuple[int, int]:
        """The one contiguous key slice of chunk ``cid``."""
        return int(cid) * self._cid_stride, (int(cid) + 1) * self._cid_stride

    def _refresh_capacity_stats(self) -> None:
        st = self.stats
        st['persistent_capacity'] = self.C
        st['scratch_capacity'] = self.Sc
        st['pool_bytes'] = self.pool_bytes()
        st['scratch_bytes'] = 2 * self.Sc * self.block_bytes
        st['host_backing_bytes'] = self.host_backing_bytes()

    def reset_stats(self) -> None:
        """Zero the request-window counters; the static capacities stay."""
        self.stats = fresh_block_stats()
        self._per_layer = fresh_per_layer_stats(len(self.layers))
        self._refresh_capacity_stats()

    def pool_bytes(self) -> int:
        """Device bytes of the persistent pools (one per layer)."""
        return len(self.layers) * 2 * self.C * self.block_bytes

    def stats_per_layer(self) -> Dict[str, list]:
        """Per-``layers`` ``misses / persistent_hits / evictions / h2d_bytes``."""
        out: Dict[str, list] = {'layers': list(self.layers)}
        out.update({k: self._per_layer[k].tolist() for k in PER_LAYER_STAT_KEYS})
        return out

    def config(self) -> Dict[str, object]:
        """The resolved configuration: mode, options, backends, budgets and geometry."""
        from omnimem.nsa.backend import get_backend as _get_nsa_backend
        return dict(
            mode=self.mode,
            dedup=self.dedup,
            block_kv_backend=self.block_kv_backend,
            nsa_backend=_get_nsa_backend(),
            resident_history_bytes=self.resident_history_bytes,
            resident_budget_bytes=self.resident_budget_bytes,
            persistent_slots_per_layer=self.C,
            scratch_slots=self.Sc,
            pool_bytes=self.pool_bytes(),
            scratch_bytes=2 * self.Sc * self.block_bytes,
            host_backing_bytes=self.host_backing_bytes(),
            dedup_device_bytes=(self._dedup.device_bytes()
                                if self._dedup is not None else 0),
            max_blocks=self.MB,
            block_bytes=self.block_bytes,
            generation=self.generation,
        )

    def host_backing_bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for per_name in self.backing.values() for t in per_name.values())

    def preallocate_host(self, rows: int, exact_pinned: bool = False) -> bool:
        """Pinned chunk-major backing ``[rows, H, B, S_max, D]`` per (layer, name)."""
        if self.resident:
            return False
        rows = int(rows)
        if rows <= 0:
            raise ValueError(f"preallocate_host needs rows >= 1, got {rows}")
        if self.backing and self._host_rows >= rows:
            return False
        old_backing, old_rows = self.backing, self._host_rows
        rows = max(rows, old_rows)
        self.wait_host()
        self.backing = {}
        shape = (rows, self.H, self.B, self.S_max, self.D)
        for L in self.layers:
            self.backing[L] = {n: pinned_host_empty(shape, self.dtype, self.device, exact_pinned)
                               for n in NAMES}
            prev = old_backing.get(L)
            if prev is not None and old_rows > 0:
                keep = min(old_rows, rows)
                for n in NAMES:
                    self.backing[L][n][:keep].copy_(prev[n][:keep])
        self._host_rows = rows
        self._refresh_host_base()
        self._refresh_capacity_stats()
        return True

    def rebind_host_views(self, cache) -> int:
        """Re-point the cache's host chunk views at the current backing rows; returns the count."""
        n_bound = 0
        for L in self.layers:
            entries = cache.cache.get(L)
            if not entries:
                continue
            for n in NAMES:
                chunks = entries.get(CACHE_NAME[n])
                backing = self.backing.get(L, {}).get(n)
                if not chunks or backing is None:
                    continue
                rows = int(backing.shape[0])
                for cid in np.flatnonzero(self.retired[L]).tolist():
                    if cid >= rows or cid >= len(chunks[0]):
                        continue
                    S = int(self.nblk[L][cid]) * self.bt
                    row = backing[cid]
                    for h in range(self.H):
                        chunks[h][cid] = row[h, :, :S, :]
                    n_bound += 1
        return n_bound

    def _refresh_host_base(self) -> None:
        """Re-derive every ``host_base[L][name][cid]`` from the live backing tensors."""
        for L in self.layers:
            per_name = self.backing.get(L)
            for n in NAMES:
                base = self.host_base[L][n]
                base.fill(0)
                if per_name is None:
                    continue
                buf = per_name[n]
                rows = min(int(buf.shape[0]), self.MC)
                if rows <= 0:
                    continue
                stride_row = buf.stride(0) * self.elem
                base[:rows] = (buf.data_ptr()
                               + np.arange(rows, dtype=np.int64) * stride_row)

    def _refresh_host_active(self, L: int) -> None:
        """Recompute "a selection may need the host" from the arrays, not from an event."""
        if bool(self.retired[L].any()):
            self._host_active[L] = True
            return
        ok = np.ones(self.MC, dtype=bool)
        for n in NAMES:
            hv, cv = self.host_ver[L][n], self.cver[L][n]
            ok &= ((hv == cv) & (cv >= 0)).all(axis=0)
        unprot = ~(self.prot[L]['k'] & self.prot[L]['v'])
        self._host_active[L] = bool((ok & unprot).any())

    def _recompute_n_valid(self, L: int) -> int:
        nb = self.nblk[L]
        idx = np.flatnonzero(nb > 0)
        nvv = int((idx.astype(np.int64) * self.bpc + nb[idx]).max()) if idx.size else 0
        self._n_valid_val[L] = nvv
        return nvv

    def _publish_n_valid(self, L: int, nvv: int) -> None:
        self.n_valid[L].fill_(int(nvv))

    def _clear_range(self, L: int, first_bid: int, n_blocks: int,
                     n_valid_value: int = _KEEP_EXTENT, which: str = 'kv') -> None:
        """Counted ``clear_block_range``: every K2 clear launch shows up in the stats."""
        if n_blocks <= 0 and n_valid_value < 0:
            return
        clear_block_range(self.table[L], self.n_valid[L], first_bid, n_blocks,
                          n_valid_value, which, backend=self.block_kv_backend)
        if n_blocks > 0:
            self.stats['clear_launches'] += 1

    def _invalidate_pool_cids(self, L: int, cid_lo: int, cid_hi: int) -> None:
        """Free every persistent slot holding a key of chunks ``[cid_lo, cid_hi)``."""
        klo = max(0, int(cid_lo)) * self._cid_stride
        khi = min(self.MC, int(cid_hi)) * self._cid_stride
        if khi <= klo or self.C == 0:
            return
        window = self.pslot[L][klo:khi]
        occupied = window[window >= 0]
        if occupied.size:
            slots = occupied.astype(np.int64)
            self.slot_key[L][slots] = -1
            self.slot_touch[L][slots] = -1
            window[:] = -1

    def _purge_pending(self, L: int, cid_lo: int, cid_hi: int) -> None:
        """Drop deferred clears of chunks ``[cid_lo, cid_hi)`` in layer ``L``."""
        if self._pc_key.size == 0:
            return
        cid = self._pc_key // self._cid_stride
        drop = (self._pc_layer == L) & (cid >= int(cid_lo)) & (cid < int(cid_hi))
        if not drop.any():
            return
        keep = ~drop
        self._pc_layer = self._pc_layer[keep]
        self._pc_row = self._pc_row[keep]
        self._pc_key = self._pc_key[keep]

    def _drop_stale_copies(self, L: int, cid_lo: int, cid_hi: int) -> None:
        """Chunks ``[cid_lo, cid_hi)`` changed: drop their pool copies, deferred clears and fetch records."""
        self._invalidate_pool_cids(L, cid_lo, cid_hi)
        self._purge_pending(L, cid_lo, cid_hi)
        # The cid range is ONE key slice.
        self.fetched_once[L][cid_lo * self._cid_stride:cid_hi * self._cid_stride] = False

    def _register_cid(self, L: int, cid: int, nb: int, k_base: int, v_base: int,
                      row_bytes: int, n_valid_value: int = _KEEP_EXTENT) -> None:
        """Point the first ``nb`` blocks of ``cid`` at their GPU owner (a 0 base keeps that name) and zero the rest."""
        lo = cid * self.bpc
        if k_base or v_base:
            register_block_range(self.table[L], self.n_valid[L], lo, nb, k_base, v_base,
                                 row_bytes, self.block_bytes, n_valid_value,
                                 backend=self.block_kv_backend)
            self.stats['register_launches'] += 1
        if nb < self.bpc:
            # A short chunk: every entry past its real extent must read 0 so the block is skipped.
            self._clear_range(L, lo + nb, self.bpc - nb)

    def _clear_pending_all(self) -> None:
        self._pc_layer, self._pc_row, self._pc_key = _EMPTY_I32, _EMPTY_I8, _EMPTY_I64

    def _pending_addresses(self) -> np.ndarray:
        """Byte addresses of the deferred clears that are still stale."""
        if self._pc_key.size == 0:
            return _EMPTY_I64
        out: List[np.ndarray] = []
        pc_layer = self._pc_layer
        # Usually one ``release_selection``; ``np.unique`` sorts, the equality test does not.
        first = int(pc_layer[0])
        layer_ids = ((first,) if bool((pc_layer == first).all())
                     else tuple(np.unique(pc_layer).tolist()))
        for L in layer_ids:
            idx = np.flatnonzero(pc_layer == L)
            keys = self._pc_key[idx]
            cid = keys // self._cid_stride
            stale_gone = ((self.prot[L]['k'][cid] & self.prot[L]['v'][cid])
                          | (self.pslot[L][keys] >= 0))
            sel = idx[~stale_gone]
            if sel.size:
                out.append(table_entry_addresses(
                    self.table[L], self._pc_row[sel].astype(np.int64),
                    self._table_index(self._pc_key[sel])))
        self._clear_pending_all()
        if not out:
            return _EMPTY_I64
        return out[0] if len(out) == 1 else np.concatenate(out)

    def _pending_len(self) -> int:
        return int(self._pc_key.size)

    def _ensure_desc(self, total: int) -> None:
        if self.desc_dev.numel() >= total:
            return
        old = self.desc_dev
        old.record_stream(torch.cuda.current_stream(self.device))
        self.desc_dev = torch.empty(max(total, 2 * old.numel()), dtype=torch.int64,
                                    device=self.device)

    def _launch(self, rows: Sequence[np.ndarray], n_fetch: int,
                clear: np.ndarray) -> None:
        """One pinned-ring stage, ONE descriptor H2D, ONE fused K1 launch."""
        n_fetch = int(n_fetch)
        n_clear = int(clear.size)
        if n_fetch == 0 and n_clear == 0:
            return
        if self.resident:
            # Resident mode has no pool or scratch, so reaching here is a lifecycle bug.
            raise RuntimeError(
                f"block KV store: resident mode must never launch the gather kernel "
                f"(n_fetch={n_fetch}, n_clear={n_clear})")
        # The previous call's kernels may still read the buffer and slots written below.
        self._wait_prev_consumer()
        self._stage_one(rows, n_fetch, clear)

    def _stage_one(self, rows: Sequence[np.ndarray], n_fetch: int,
                   clear: np.ndarray) -> None:
        n_clear = int(clear.size)
        total = 6 * int(n_fetch) + n_clear
        if total == 0:
            return
        st = self.stats
        self._ensure_desc(total)
        slot_id, host, host_np = self._ring.acquire_elems(total, st)
        if n_fetch:
            for i, arr in enumerate(rows):
                host_np[i * n_fetch:(i + 1) * n_fetch] = arr
        if n_clear:
            host_np[6 * n_fetch:total] = clear
        self.desc_dev[:total].copy_(host[:total], non_blocking=True)
        gather_publish_clear(self.desc_dev, n_fetch, n_clear, self.bt, self.D,
                             self._typed, backend=self.block_kv_backend)
        self._ring.record(slot_id, torch.cuda.current_stream(self.device))
        st['desc_h2d_bytes'] += total * 8
        st['desc_h2d_calls'] += 1
        st['copy_api_calls'] += 1
        if n_fetch:
            st['gather_launches'] += 1
        else:
            st['clear_only_launches'] += 1
        st['publish_entries'] += 2 * n_fetch
        st['clear_entries'] += n_clear

    def flush_pending_clears(self) -> int:
        """Zero every still-stale deferred table entry now.  Returns how many."""
        clear = self._pending_addresses()
        if clear.size == 0:
            return 0
        self._launch((), 0, clear)
        return int(clear.size)

    @staticmethod
    def _pick_victims(cand: np.ndarray, slot_touch: np.ndarray, k: int,
                      epoch: Optional[int] = None) -> np.ndarray:
        """``k`` candidates with the smallest ``slot_touch``, ties by slot index."""
        if k <= 0:
            return _EMPTY_I64
        if k >= cand.size:
            return cand.astype(np.int64, copy=False)
        tv = slot_touch[cand]
        thr = None
        if epoch is not None and 0 < int(epoch) <= _VICTIM_BINS_MAX:
            try:
                cum = np.cumsum(np.bincount(tv))
            except (ValueError, TypeError):   # negative / non-integer touch values
                cum = None
            if cum is not None and int(cum[-1]) == tv.size:
                thr = np.int64(np.searchsorted(cum, k + 1, side='left'))
        if thr is None:
            thr = tv[np.argpartition(tv, k)[k]]
        # One pass over ``tv <= thr`` already holds every ``tv < thr`` and tie position, ascending.
        le = np.flatnonzero(tv <= thr)
        lt_m = tv[le] < thr
        lt = le[lt_m]
        need = k - int(lt.size)
        if need > 0:
            sel = np.concatenate([lt, le[np.flatnonzero(~lt_m)[:need]]])
        else:
            sel = lt
        out = cand[sel].astype(np.int64, copy=True)
        out.sort()
        return out

    @staticmethod
    def _first_free(mask: np.ndarray, k: int, block: int = 8192) -> np.ndarray:
        """``np.flatnonzero(mask)[:k]`` that stops at the k-th set entry."""
        if k <= 0:
            return _EMPTY_I64
        n = int(mask.size)
        if k >= n:
            return np.flatnonzero(mask)
        out: List[np.ndarray] = []
        got = i = 0
        while i < n and got < k:
            idx = np.flatnonzero(mask[i:i + block])
            if idx.size:
                idx += i
                out.append(idx)
                got += int(idx.size)
            i += block
        if not out:
            return _EMPTY_I64
        res = out[0] if len(out) == 1 else np.concatenate(out)
        return res[:k] if res.size > k else res

    def table_views(self, layer_idx: int):
        """``(addr_k, addr_v, n_valid)`` -- exactly what ``build_ptr_table`` consumes."""
        tab = self.table[layer_idx]
        return tab[0], tab[1], self.n_valid[layer_idx]

    def _unique_keys(self, block_indices: torch.Tensor) -> np.ndarray:
        """The ONE host round-trip: unique in-range host-order keys, ascending."""
        if self.dedup == 'device':
            return self._unique_keys_device(block_indices)
        st = self.stats
        sent = self._sentinel
        bi = block_indices.to(torch.int64)
        cid = torch.div(bi, self.bpc, rounding_mode='floor')
        loc = bi - cid * self.bpc
        keys = (cid * self.H + self._h_idx) * self.bpc + loc
        keys = torch.where((bi >= 0) & (bi < self.MB),
                           keys, torch.full_like(keys, sent))
        u = torch.unique(keys)
        u_np = u.cpu().numpy()
        st['meta_d2h_bytes'] += int(u_np.nbytes)
        st['meta_d2h_calls'] += 1
        if u_np.size and u_np[-1] == sent:
            u_np = u_np[:-1]
        return u_np

    def _ensure_dedup(self, block_indices: torch.Tensor):
        """The device-dedup (K3) object for this selection shape (built on demand)."""
        from omnimem.nsa.ops.selection.padded_ptr.kernels.block_kv_dedup import (
            DeviceBlockDedup)
        t = int(block_indices.shape[-1])
        mg = int(block_indices.shape[1])
        # A call never yields more unique keys than entries or than the key space: the D2H capacity.
        cap = min(mg * self.H * t, self.H * self.MB)
        ded = self._dedup
        if ded is not None and ded.T == t and ded.cap >= cap:
            return ded
        self._dedup = DeviceBlockDedup(
            heads=self.H, mb=self.MB, blocks_per_chunk=self.bpc,
            blocks_per_group=t, cap=cap, device=self.device,
            backend=self.block_kv_backend)
        return self._dedup

    def _unique_keys_device(self, block_indices: torch.Tensor) -> np.ndarray:
        """K3 dedup: 4 tiny launches + ONE fixed-capacity async D2H + ONE event wait."""
        st = self.stats
        if not block_indices.is_contiguous():
            block_indices = block_indices.contiguous()
        ded = self._ensure_dedup(block_indices)
        launches_before = ded.launches
        keys, n, overflow = ded.unique_keys(block_indices)
        st['dedup_launches'] += ded.launches - launches_before
        st['meta_d2h_bytes'] += ded.d2h_bytes_per_call
        st['meta_d2h_calls'] += 1
        if overflow:
            raise RuntimeError(
                f"block KV store: the device dedup buffer holds {ded.cap} keys but "
                f"this call selected {n} unique blocks "
                f"(block_indices {tuple(block_indices.shape)}); nothing was modified")
        return keys.astype(np.int64)

    def prepare_selection(self, layer_idx: int,
                          block_indices: torch.Tensor) -> BlockSelectionLease:
        """Plan and execute this call's block residency; returns the scratch lease."""
        st = self.stats
        t0 = time.perf_counter()
        st['calls'] += 1
        L = layer_idx
        if self.resident or (not self._host_active[L] and self._pending_len() == 0):
            # All-protected fast path: nothing is host-only, so no D2H or launch.
            self._epoch[L] += 1
            st['fast_path_calls'] += 1
            st['prepare_s'] += time.perf_counter() - t0
            return _EMPTY_LEASE

        self._note_stream()
        u_np = self._unique_keys(block_indices)
        t1 = time.perf_counter()
        st['meta_d2h_s'] += t1 - t0
        try:
            lease = self._plan(L, u_np)
        finally:
            st['policy_cpu_s'] += time.perf_counter() - t1
            st['prepare_s'] += time.perf_counter() - t0
        return lease

    def _note_stream(self) -> None:
        stream = torch.cuda.current_stream(self.device)
        if self._owner_stream is None:
            self._owner_stream = stream
        elif self._owner_stream != stream:
            self.stats['stream_switches'] += 1
            self._owner_stream = stream

    def _plan(self, L: int, u_np: np.ndarray) -> BlockSelectionLease:
        st = self.stats
        H, bpc, MB = self.H, self.bpc, self.MB
        nblk = self.nblk[L]
        cver_k, cver_v = self.cver[L]['k'], self.cver[L]['v']
        host_k, host_v = self.host_ver[L]['k'], self.host_ver[L]['v']
        prot_k, prot_v = self.prot[L]['k'], self.prot[L]['v']
        pslot = self.pslot[L]                      # flat [H * MB], indexed by key
        slot_key, slot_touch = self.slot_key[L], self.slot_touch[L]
        tab = self.table[L]
        nvv = self._n_valid_val[L]
        epoch = self._epoch[L] + 1

        # Read-only classification; ascending key order is already ascending host-address order.
        rest = u_np // bpc
        loc = u_np - rest * bpc
        cid = rest // H
        bid = cid * bpc + loc
        # Out-of-extent ids are not an error: the pointer builder filters them to a 0 pointer.
        partial = np.flatnonzero(nblk < bpc)
        if cid.size and (partial.size == 0 or int(cid.max()) < int(partial[0])):
            # One ``max`` reduction answers "anything out of extent?" without building the mask.
            in_extent = None if int(bid.max()) < nvv else bid < nvv
        else:
            in_extent = (bid < nvv) & (loc < nblk[cid])
        if in_extent is not None and not in_extent.all():
            keep_sel = np.flatnonzero(in_extent)
            u_np = u_np[keep_sel]
            cid, loc, rest = cid[keep_sel], loc[keep_sel], rest[keep_sel]
        # Counters stay in locals until after the last raise so a failed call changes nothing.
        n_selected = int(u_np.size)

        # One flatnonzero + integer takes instead of a boolean mask index per array.
        prot_hit = (prot_k & prot_v)[cid]
        n_prot_hit = int(np.count_nonzero(prot_hit))
        if n_prot_hit:
            rest_i = np.flatnonzero(~prot_hit)
            ur = u_np[rest_i]
        else:
            rest_i, ur = None, u_np
        ps = pslot[ur] if ur.size else _EMPTY_I32
        hit_i = np.flatnonzero(ps >= 0)
        touched = ps[hit_i].astype(np.int64)
        n_touched = int(hit_i.size)
        if n_touched == ps.size:
            keys_miss = hm = cm = lm = _EMPTY_I64
            n_miss = 0
        else:
            miss_i = np.flatnonzero(ps < 0)
            sel_i = miss_i if rest_i is None else rest_i[miss_i]
            keys_miss = u_np[sel_i]
            cm, lm = cid[sel_i], loc[sel_i]
            hm = rest[sel_i] - cm * H
            n_miss = int(miss_i.size)

        n_persist = n_scratch = 0
        victims = assign = _EMPTY_I64
        src_k = src_v = dst_k = dst_v = _EMPTY_I64
        if n_miss:
            # invariant 3/4: fetchable means both host rows hold both current versions.
            flat_hc = hm * self.MC + cm
            cvk = np.take(cver_k, flat_hc)
            hvk = np.take(host_k, flat_hc)
            cvv = np.take(cver_v, flat_hc)
            hvv_ = np.take(host_v, flat_hc)
            ok = (hvk == cvk) & (cvk >= 0) & (hvv_ == cvv) & (cvv >= 0)
            if not bool(ok.all()):
                i = int(np.flatnonzero(~ok)[0])
                raise RuntimeError(
                    f"block KV store: L{L} selected (head={int(hm[i])}, "
                    f"bid={int(cm[i]) * bpc + int(lm[i])}, cid={int(cm[i])}) is neither "
                    f"GPU-protected nor pooled and has no usable host backing "
                    f"(cver k/v={int(cver_k[hm[i], cm[i]])}/"
                    f"{int(cver_v[hm[i], cm[i]])}, host_ver k/v="
                    f"{int(host_k[hm[i], cm[i]])}/{int(host_v[hm[i], cm[i]])}) "
                    f"-- lifecycle bug")
            # Gathered once and reused as the source base below (``host_base`` is non-negative).
            hbase_k = np.take(self.host_base[L]['k'], cm)
            hbase_v = np.take(self.host_base[L]['v'], cm)
            if int(hbase_k.min()) == 0 or int(hbase_v.min()) == 0:
                raise RuntimeError(
                    f"block KV store: L{L} needs host backing rows that were never "
                    f"allocated (call MMCache.preallocate_block_host first)")

            C = self.C
            freem = self._mask_a[:C]
            np.less(slot_key, 0, out=freem)
            n_free = int(np.count_nonzero(freem))
            # ``n_from_free`` is ``min(n_free, n_miss)`` in both branches; take free slots first.
            n_from_free = n_free if n_free < n_miss else n_miss
            free_head = self._first_free(freem, n_from_free)
            if n_free >= n_miss:
                # Every miss fits in a free slot, so no victim is possible; skip the candidate scan.
                cand = _EMPTY_I64
                n_persist = n_miss
            else:
                candm = self._mask_b[:C]
                # "occupied" is the exact complement of "free": one pass over the small bitmap.
                np.logical_not(freem, out=candm)
                np.less(slot_touch, epoch, out=freem)   # freem is consumed above
                np.logical_and(candm, freem, out=candm)
                if n_touched:
                    candm[touched] = False
                cand = np.flatnonzero(candm)
                n_persist = min(n_miss, n_free + int(cand.size))
            n_scratch = n_miss - n_persist
            if n_scratch > self.Sc:
                raise RuntimeError(
                    f"block KV store: L{L} needs {n_miss} blocks this call but only "
                    f"{n_free + int(cand.size)} persistent slots are reclaimable "
                    f"(capacity {self.C}) and the shared scratch pool holds "
                    f"{self.Sc} -- {n_scratch} blocks do not fit.  Raise "
                    f"kv_block_pool_gb / kv_block_scratch_gb; nothing was modified.")
            victims = self._pick_victims(cand, slot_touch, n_persist - n_from_free,
                                         epoch)
            assign = (np.concatenate([free_head, victims])
                      if n_persist else _EMPTY_I64).astype(np.int64, copy=False)
            self._validate_slots(assign, n_scratch)

            # Descriptor addresses are computed before any mutation so the checks fail cleanly.
            head_off = hm * self.host_head_stride + lm * self.block_bytes
            src_k = hbase_k + head_off
            src_v = hbase_v + head_off
            dst_k = np.empty(n_miss, dtype=np.int64)
            dst_v = np.empty(n_miss, dtype=np.int64)
            if n_persist:
                pool_k_ptr, pool_v_ptr = self._pool_ptr[L]
                off = assign * self.block_bytes
                dst_k[:n_persist] = pool_k_ptr + off
                dst_v[:n_persist] = pool_v_ptr + off
            if n_scratch:
                off = np.arange(n_scratch, dtype=np.int64) * self.block_bytes
                dst_k[n_persist:] = self._scratch_ptr[0] + off
                dst_v[n_persist:] = self._scratch_ptr[1] + off
            self._validate_destinations(L, dst_k, dst_v, n_persist)
            self._validate_source_order(L, src_k, src_v)

        pos = self._layer_pos[L]
        pl = self._per_layer
        st['selected_unique'] += n_selected
        st['protected_hits'] += n_prot_hit
        st['persistent_hits'] += n_touched
        st['misses'] += n_miss
        pl['persistent_hits'][pos] += n_touched
        pl['misses'][pos] += n_miss
        self._epoch[L] = epoch
        if n_touched:
            slot_touch[touched] = epoch

        if n_miss == 0:
            # All-hit: clear-only launch if something is pending; the lease still records the event.
            self._launch((), 0, self._pending_addresses())
            return BlockSelectionLease(L, _EMPTY_I64, 0)

        if victims.size:
            vkeys = slot_key[victims]
            pslot[vkeys] = -1
            slot_key[victims] = -1
            st['persistent_evictions'] += int(victims.size)
            pl['evictions'][pos] += int(victims.size)
            vrest = vkeys // bpc
            vloc = vkeys - vrest * bpc
            vcid = vrest // H
            vt = (vrest - vcid * H) * MB + vcid * bpc + vloc
            # Sorted here so ``clear_v`` is already sorted and duplicate-free (see the merge below).
            vt.sort()
            clear_v = np.concatenate([table_entry_addresses(tab, 0, vt),
                                      table_entry_addresses(tab, 1, vt)])
        else:
            clear_v = _EMPTY_I64

        pk_keys = keys_miss[:n_persist]
        if n_persist:
            pslot[pk_keys] = assign.astype(np.int32)
            slot_key[assign] = pk_keys
            slot_touch[assign] = epoch
        scratch_keys = keys_miss[n_persist:].astype(np.int64, copy=True)

        fo = self.fetched_once[L]
        fe = self.fetched_ever[L]
        was_once = fo[keys_miss]
        was_ever = fe[keys_miss]
        st['refetch_evicted'] += int(np.count_nonzero(was_once))
        st['refetch_version'] += int(np.count_nonzero(was_ever & ~was_once))
        fo[keys_miss] = True
        fe[keys_miss] = True

        # ``(hm, cm, lm)`` already decode ``keys_miss``, so the table index is two multiply-adds.
        tidx = hm * MB + cm * bpc + lm
        pub_k = table_entry_addresses(tab, 0, tidx)
        pub_v = table_entry_addresses(tab, 1, tidx)

        clear = self._pending_addresses()
        if clear.size:
            if clear_v.size:
                clear = np.concatenate([clear, clear_v])
            # Publish wins: a key re-fetched this call must not be zeroed again.
            clear = np.setdiff1d(clear, np.concatenate([pub_k, pub_v]))
        elif clear_v.size:
            # Same array without the ``setdiff1d`` sorts: victim and published keys are disjoint.
            clear = clear_v

        # ``_launch`` orders this against the previous consumer and the retirement D2H.
        self._launch((src_k, src_v, dst_k, dst_v, pub_k, pub_v), n_miss, clear)
        st['h2d_bytes'] += 2 * n_miss * self.block_bytes
        pl['h2d_bytes'][pos] += 2 * n_miss * self.block_bytes
        if n_scratch:
            st['scratch_blocks'] += n_scratch
            st['scratch_calls'] += 1
            st['peak_scratch_used'] = max(int(st['peak_scratch_used']), n_scratch)
        return BlockSelectionLease(L, scratch_keys, n_miss)

    def _validate_slots(self, assign: np.ndarray, n_scratch: int) -> None:
        """Destinations are in range and pairwise distinct within this call."""
        n = int(assign.size)
        if n:
            lo, hi = int(assign.min()), int(assign.max())
            if lo < 0 or hi >= self.C:
                raise RuntimeError(
                    f"block KV store: persistent slot {lo}..."
                    f"{hi} outside [0, {self.C})")
            if n > self._iota.size:                   # only a pathological n > C
                self._iota = np.arange(n, dtype=np.int64)
            if n > self._slot_pos.size:
                self._slot_pos = np.zeros(n, dtype=np.int64)
            rows = self._iota[:n]
            self._slot_pos[assign] = rows
            if not np.array_equal(self._slot_pos[assign], rows):
                raise RuntimeError(
                    "block KV store: the same persistent slot was assigned twice in "
                    "one call (would silently overwrite a fetched block)")
        if n_scratch > self.Sc:                       # already checked; belt and braces
            raise RuntimeError("block KV store: scratch overflow slipped the budget check")

    def _validate_destinations(self, L: int, dst_k: np.ndarray, dst_v: np.ndarray,
                               n_persist: int) -> None:
        """Every destination byte range lies inside a pool or the shared scratch."""
        pool_k_ptr, pool_v_ptr = self._pool_ptr[L]
        bb = self.block_bytes
        for dst, base, cap in ((dst_k[:n_persist], pool_k_ptr, self.C),
                               (dst_v[:n_persist], pool_v_ptr, self.C),
                               (dst_k[n_persist:], self._scratch_ptr[0], self.Sc),
                               (dst_v[n_persist:], self._scratch_ptr[1], self.Sc)):
            if not dst.size:
                continue
            hi = base + cap * bb
            if int(dst.min()) < base or int(dst.max()) + bb > hi:
                raise RuntimeError(
                    f"block KV store: L{L} gather destination outside its pool "
                    f"allocation [{base}, {base + cap * bb})")

    def _validate_source_order(self, L: int, src_k: np.ndarray,
                               src_v: np.ndarray) -> None:
        """The K1 descriptor must be sorted by HOST source address, K and V alike."""
        for name, src in (('k', src_k), ('v', src_v)):
            if src.size > 1 and not bool((src[1:] > src[:-1]).all()):
                i = int(np.flatnonzero(src[1:] <= src[:-1])[0])
                raise RuntimeError(
                    f"block KV store: L{L} gather descriptor ({name}) is not sorted by "
                    f"host source address at row {i} ({int(src[i])} -> {int(src[i + 1])}); "
                    f"the key order was broken")

    def _wait_prev_consumer(self) -> None:
        """Order this call's device writes after the lease and retirement D2H events."""
        stream = None
        ev, recorded_on = self._scratch_event, self._scratch_stream
        if ev is not None and recorded_on is not None:
            stream = torch.cuda.current_stream(self.device)
            if stream != recorded_on:
                stream.wait_event(ev)
        if self._host_event_pending and self._host_event_stream is not None:
            stream = stream or torch.cuda.current_stream(self.device)
            if stream != self._host_event_stream:
                stream.wait_event(self._host_event)

    def release_selection(self, lease: Optional[BlockSelectionLease]) -> None:
        """Called right after the selection kernel is enqueued; defers scratch clears."""
        if lease is None or lease.layer_idx < 0:
            return
        if lease.layer_idx not in self.table:       # the store was dropped under us
            return
        sk = lease.scratch_keys
        if sk.size:
            n = int(sk.size)
            self._pc_layer = np.concatenate([
                self._pc_layer, np.full(2 * n, int(lease.layer_idx), dtype=np.int32)])
            self._pc_row = np.concatenate([
                self._pc_row, np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])
            self._pc_key = np.concatenate([self._pc_key, sk, sk])
        stream = torch.cuda.current_stream(self.device)
        ev = self._scratch_event
        if ev is None:
            ev = torch.cuda.Event()
            self._scratch_event = ev
        ev.record(stream)
        self._scratch_stream = stream

    def on_chunk_written(self, layer_idx: int, name: str, cid: int,
                         stacked: torch.Tensor, version: int) -> None:
        """A chunk's K or V storage became the protected GPU owner; only V advances ``n_valid``."""
        L = layer_idx
        n = STORE_NAME.get(name, name)
        if n not in NAMES:
            raise ValueError(f"the block KV store only manages k/v_cache, got {name!r}")
        if not (0 <= cid < self.MC):
            raise ValueError(f"chunk id {cid} outside [0, {self.MC}) for L{L}/{name}")
        if stacked.dim() != 4:
            raise ValueError(f"expected [H, B, S, D] storage, got {tuple(stacked.shape)}")
        H, B, S, D = (int(x) for x in stacked.shape)
        if H != self.H or B != self.B or D != self.D:
            raise ValueError(
                f"L{L}/{name} chunk {tuple(stacked.shape)} does not match the store's "
                f"[H={self.H}, B={self.B}, *, D={self.D}] geometry")
        if S <= 0 or S % self.bt:
            # configure_block_store rejects non-aligned lengths; this guards a hand-built chunk.
            raise ValueError(
                f"L{L}/{name} chunk length {S} is not a multiple of kv_block_size "
                f"{self.bt}; the block KV cache requires block-aligned chunks")
        if S > self.S_max:
            raise ValueError(f"L{L}/{name} chunk length {S} exceeds {self.S_max}")
        if stacked.device != self.device:
            raise ValueError(
                f"L{L}/{name} chunk lives on {stacked.device}, store is on {self.device}")
        if stacked.dtype != self.dtype:
            raise ValueError(
                f"L{L}/{name} chunk dtype {stacked.dtype} != store dtype {self.dtype}")

        nb = S // self.bt
        # New content: every pool copy, fetch record and deferred clear of this cid is stale.
        self._drop_stale_copies(L, cid, cid + 1)
        self.cver[L][n][:, cid] = int(version)
        self.prot[L][n][cid] = True
        self.gpu_chunk[L][n][cid] = stacked
        self.nblk[L][cid] = nb
        if self.retired[L][cid]:
            self.retired[L][cid] = False
        self._refresh_host_active(L)
        nvv = self._recompute_n_valid(L)
        base = stacked.data_ptr()
        self._register_cid(L, cid, nb, base if n == 'k' else 0, base if n == 'v' else 0,
                           stacked.stride(0) * self.elem, nvv if n == 'v' else _KEEP_EXTENT)

    def on_chunk_freed(self, layer_idx: int, name: str, cid: int) -> None:
        """``free_last_chunk`` popped ``cid``: forget it completely."""
        L = layer_idx
        n = STORE_NAME.get(name, name)
        if n not in NAMES or not (0 <= cid < self.MC):
            return
        lo = cid * self.bpc
        klo, khi = self.cid_key_range(cid)            # ONE contiguous key slice
        self._drop_stale_copies(L, cid, cid + 1)
        self.fetched_ever[L][klo:khi] = False
        self.cver[L][n][:, cid] = -1
        self.host_ver[L][n][:, cid] = -1
        self.prot[L][n][cid] = False
        self.gpu_chunk[L][n][cid] = None
        self.nblk[L][cid] = 0
        self.retired[L][cid] = False
        self.frozen[L][cid] = False
        nvv = self._recompute_n_valid(L)
        self._refresh_host_active(L)
        self._clear_range(L, lo, self.bpc, nvv, n)

    def retire_candidates(self, layer_idx: int, sink_end: int,
                          win_start: int) -> np.ndarray:
        """Completed, unfrozen, fully protected cids strictly outside sink+window."""
        L = layer_idx
        if self.resident:
            return _EMPTY_I64
        lo, hi = max(0, int(sink_end)), min(self.MC, int(win_start))
        if hi <= lo:
            return _EMPTY_I64
        sl = slice(lo, hi)
        ok = (~self.retired[L][sl] & ~self.frozen[L][sl]
              & self.prot[L]['k'][sl] & self.prot[L]['v'][sl]
              & (self.nblk[L][sl] > 0))
        return np.flatnonzero(ok).astype(np.int64) + lo

    def retire(self, layer_idx: int, cids: np.ndarray):
        """Back ``cids`` up to the pinned host rows and release their GPU owners."""
        L = layer_idx
        st = self.stats
        if cids.size == 0:
            return {}
        if self.resident:
            # Resident mode never retires; MMCache short-circuits before getting here.
            return {}
        backing = self.backing.get(L)
        if backing is None:
            raise RuntimeError(
                "block KV store: retirement needs the pinned host backing; call "
                "MMCache.preallocate_block_host(sample, segment_cids) first")
        views: Dict[str, Dict[int, List[torch.Tensor]]] = {
            CACHE_NAME[n]: {} for n in NAMES}
        for cid in cids.tolist():
            cid = int(cid)
            if cid >= int(backing['k'].shape[0]):
                raise RuntimeError(
                    f"block KV store: no host backing row for cid {cid} "
                    f"(rows={int(backing['k'].shape[0])})")
            S = int(self.nblk[L][cid]) * self.bt
            for n in NAMES:
                src = self.gpu_chunk[L][n][cid]
                if src is None:
                    raise RuntimeError(
                        f"block KV store: L{L}/{n} cid {cid} is marked protected "
                        f"but has no GPU owner")
                row = backing[n][cid]
                row[:, :, :S, :].copy_(src, non_blocking=True)
                st['d2h_bytes'] += src.numel() * src.element_size()
                st['copy_api_calls'] += 1
                self.host_ver[L][n][:, cid] = self.cver[L][n][:, cid]
                self.prot[L][n][cid] = False
                self.gpu_chunk[L][n][cid] = None
                views[CACHE_NAME[n]][cid] = [row[h, :, :S, :]
                                             for h in range(self.H)]
            self.retired[L][cid] = True
        st['retired_chunks'] += int(cids.size)
        self._refresh_host_active(L)

        for run_lo, run_hi in _runs(cids):
            self._clear_range(L, run_lo * self.bpc, (run_hi - run_lo) * self.bpc)
        self._record_host_event(self.device)
        return views

    def wait_host(self) -> None:
        """Block the CPU until the last retirement D2H batch has landed."""
        self._wait_host_event()

    def _forget_mappings(self) -> None:
        """Wait for the in-flight copies, then empty every table and its metadata."""
        self._ring.drain(self.stats)
        self.wait_host()
        for L in self.layers:
            self.table[L].zero_()
            self.n_valid[L].zero_()
            self._init_layer_meta(L)
        self._refresh_host_base()
        self._clear_pending_all()
        self._scratch_event = None
        self._scratch_stream = None

    def reset(self) -> None:
        """New request: forget every mapping.  Pools, scratch, backing, desc survive."""
        self._forget_mappings()
        self._owner_stream = None
        self.generation += 1

    def drop(self) -> None:
        """Release every allocation; the store object must not be used afterwards."""
        self._ring.drain(None)
        self.wait_host()
        # A selection kernel may still read the pools/scratch being freed; wait on the lease event.
        if self._scratch_event is not None:
            self._scratch_event.synchronize()
        self._clear_pending_all()
        self.table = {}
        self.n_valid = {}
        self.pool_k = {}
        self.pool_v = {}
        self._pool_ptr = {}
        self.scratch_k = self.scratch_v = None
        self.desc_dev = None
        self.backing = {}
        self._host_rows = 0
        for L in self.layers:
            self.gpu_chunk[L] = {n: [None] * self.MC for n in NAMES}
            for n in NAMES:
                self.host_base[L][n].fill(0)
            self.pslot[L].fill(-1)
            self.slot_key[L].fill(-1)
            self.slot_touch[L].fill(-1)
        self._dedup = None
        self._scratch_event = None
        self._scratch_stream = None
        self._owner_stream = None
        self._host_event = None
        self._host_event_pending = False

    def rebuild_all(self, cache) -> int:
        """Canonicalise every chunk of ``cache`` into the store's layout."""
        H = self.H
        self.flush_pending_clears()
        self._forget_mappings()

        plan: Dict[int, Dict[int, dict]] = {}
        rows = 0
        needs_host = False
        for L in self.layers:
            per_cid: Dict[int, dict] = {}
            for n in NAMES:
                cname = CACHE_NAME[n]
                chunks = cache.cache.get(L, {}).get(cname)
                if not chunks or len(chunks) != H or not chunks[0]:
                    continue
                rows = max(rows, len(chunks[0]))
                for cid in range(len(chunks[0])):
                    col = [chunks[h][cid] if cid < len(chunks[h]) else None
                           for h in range(H)]
                    if any(t is None for t in col):
                        continue
                    if cid >= self.MC:
                        raise RuntimeError(
                            f"block KV store: L{L}/{cname} has cid {cid} but only "
                            f"{self.MC} chunks are addressable")
                    S = int(col[0].shape[1])
                    if S % self.bt:
                        raise ValueError(
                            f"L{L}/{cname} cid {cid} length {S} is not block aligned")
                    if S > self.S_max:
                        raise ValueError(
                            f"L{L}/{cname} cid {cid} length {S} exceeds {self.S_max}")
                    entry = per_cid.setdefault(cid, {'S': S, 'cols': {}, 'host': False})
                    if entry['S'] != S:
                        raise ValueError(
                            f"block KV store: L{L} cid {cid} has K/V lengths "
                            f"{entry['S']} != {S}")
                    entry['cols'][n] = col
                    if not all(t.is_cuda and t.device == self.device for t in col):
                        # One byte off-device: K/V are normalised together.
                        entry['host'] = not self.resident
            plan[L] = per_cid
            needs_host |= any(e['host'] for e in per_cid.values())
        if needs_host:
            self.preallocate_host(max(1, min(rows, self.MC)),
                                  exact_pinned=cache.exact_pinned_host())

        total = 0
        for L in self.layers:
            pending: List[Tuple[str, int, int, int, int]] = []
            for cid, entry in sorted(plan[L].items()):
                S, to_host = entry['S'], entry['host']
                nb = S // self.bt
                self.nblk[L][cid] = nb
                for n, col in entry['cols'].items():
                    cname = CACHE_NAME[n]
                    chunks = cache.cache[L][cname]
                    # cver is the sole version authority in block mode; re-bump after a load.
                    ver = cache._bump_chunk_version(L, cname, cid)
                    self.cver[L][n][:, cid] = int(ver)
                    if to_host:
                        row = self.backing[L][n][cid]
                        # torch.stack makes a pageable / foreign-device source one CPU tensor.
                        host_stack = torch.stack(
                            [t.detach().to('cpu') if t.is_cuda else t.detach()
                             for t in col], dim=0)
                        row[:, :, :S, :].copy_(host_stack)
                        for h in range(H):
                            chunks[h][cid] = row[h, :, :S, :]
                        self.prot[L][n][cid] = False
                        self.gpu_chunk[L][n][cid] = None
                        self.host_ver[L][n][:, cid] = int(ver)
                    else:
                        # ``.to`` lifts a host chunk onto the device in resident mode; else a no-op.
                        stacked = torch.stack(
                            [t if (t.is_cuda and t.device == self.device)
                             else t.detach().to(self.device) for t in col],
                            dim=0).contiguous()
                        for h in range(H):
                            chunks[h][cid] = stacked[h]
                        self.prot[L][n][cid] = True
                        self.gpu_chunk[L][n][cid] = stacked
                        self.host_ver[L][n][:, cid] = -1
                        pending.append((n, cid, nb, stacked.data_ptr(),
                                        stacked.stride(0) * self.elem))
                    total += 1
                # retired only once BOTH names are host-backed with their live version
                self.retired[L][cid] = bool(
                    to_host and set(entry['cols']) == set(NAMES))
            nvv = self._recompute_n_valid(L)
            for n, cid, nb, base, hs in pending:
                self._register_cid(L, cid, nb, base if n == 'k' else 0, base if n == 'v' else 0, hs)
            self._publish_n_valid(L, nvv)
            self._refresh_host_active(L)
        return total

    def snapshot_range(self, layer_idx: int, cid_lo: int, cid_hi: int) -> dict:
        """Capture ``[cid_lo, cid_hi)`` and freeze their host rows against retirement."""
        L = layer_idx
        lo = max(0, int(cid_lo))
        hi = min(self.MC, int(cid_hi))
        frozen: List[int] = []
        for cid in range(lo, hi):
            if ((self.host_ver[L]['k'][:, cid] >= 0).any()
                    or (self.host_ver[L]['v'][:, cid] >= 0).any()):
                if not self.frozen[L][cid]:
                    self.frozen[L][cid] = True
                    frozen.append(cid)
        return dict(
            layer=L, lo=lo, hi=hi, frozen=frozen,
            gpu_chunk={n: list(self.gpu_chunk[L][n][lo:hi]) for n in NAMES},
            cver={n: self.cver[L][n][:, lo:hi].copy() for n in NAMES},
            host_ver={n: self.host_ver[L][n][:, lo:hi].copy() for n in NAMES},
            prot={n: self.prot[L][n][lo:hi].copy() for n in NAMES},
            retired=self.retired[L][lo:hi].copy(),
            nblk=self.nblk[L][lo:hi].copy(),
        )

    def restore_range(self, snap: dict) -> None:
        """Put a snapshot back and rebuild the block addresses of its range."""
        L = snap['layer']
        lo, hi = snap['lo'], snap['hi']
        if hi > lo:
            self._drop_stale_copies(L, lo, hi)
            for n in NAMES:
                self.cver[L][n][:, lo:hi] = snap['cver'][n]
                self.host_ver[L][n][:, lo:hi] = snap['host_ver'][n]
                self.prot[L][n][lo:hi] = snap['prot'][n]
                self.gpu_chunk[L][n][lo:hi] = snap['gpu_chunk'][n]
            self.retired[L][lo:hi] = snap['retired']
            self.nblk[L][lo:hi] = snap['nblk']
            nvv = self._recompute_n_valid(L)
            for cid in range(lo, hi):
                self._republish_cid(L, cid)
            self._publish_n_valid(L, nvv)
            self._refresh_host_active(L)
        self.release_snapshot(snap)

    def release_snapshot(self, snap: dict) -> None:
        """Commit without restoring: only the freeze is undone."""
        L = snap['layer']
        for cid in snap['frozen']:
            self.frozen[L][cid] = False
        snap['frozen'] = []

    def _republish_cid(self, L: int, cid: int) -> None:
        """Re-derive this cid's table entries from the (restored) CPU truth."""
        lo = cid * self.bpc
        nb = int(self.nblk[L][cid])
        if nb <= 0:
            self._clear_range(L, lo, self.bpc)
            return
        owners = {n: (self.gpu_chunk[L][n][cid] if self.prot[L][n][cid] else None)
                  for n in NAMES}
        kb = owners['k'].data_ptr() if owners['k'] is not None else 0
        vb = owners['v'].data_ptr() if owners['v'] is not None else 0
        ref = owners['k'] if owners['k'] is not None else owners['v']
        self._register_cid(L, cid, nb, kb, vb, 0 if ref is None else ref.stride(0) * self.elem)
        for n in NAMES:
            if owners[n] is None:
                self._clear_range(L, lo, nb, which=n)

    def block_hit_rate(self) -> float:
        """``(protected + persistent) / selected_unique`` over non-fast-path calls."""
        st = self.stats
        n = float(st['selected_unique'])
        if n <= 0:
            return 0.0
        return (float(st['protected_hits']) + float(st['persistent_hits'])) / n


def _runs(cids: np.ndarray):
    """Contiguous ``[lo, hi)`` runs of an ascending cid array."""
    out = []
    lo = prev = None
    for c in cids.tolist():
        c = int(c)
        if lo is None:
            lo = prev = c
        elif c == prev + 1:
            prev = c
        else:
            out.append((lo, prev + 1))
            lo = prev = c
    if lo is not None:
        out.append((lo, prev + 1))
    return out
