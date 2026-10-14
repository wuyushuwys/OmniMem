"""Device-side dedup of selected block keys: ascending unique store keys, one async D2H."""
from __future__ import annotations

from typing import Optional

import torch

from omnimem.nsa.backend import resolve_block_kv_backend
from omnimem.nsa.ops._int_utils import cdiv

__all__ = ["DeviceBlockDedup"]

#: Scan tile size; must match the CUDA kernel's tile.
_SCAN_BLOCK = 1024


class DeviceBlockDedup:
    """Fixed-capacity device dedup; unique_keys returns (keys_int32_view, n_unique, overflow)."""

    def __init__(self, heads: int, mb: int, blocks_per_group: int, cap: int,
                 device, *, blocks_per_chunk: int, backend: Optional[str] = None):
        self.H, self.MB, self.T = int(heads), int(mb), int(blocks_per_group)
        self.bpc = int(blocks_per_chunk)
        assert self.bpc > 0 and self.MB % self.bpc == 0, (
            f"blocks_per_chunk {self.bpc} must divide MB {self.MB} for the key to be "
            f"a bijection")
        self.n_keys = self.H * self.MB
        self.cap = int(cap)
        self.backend = resolve_block_kv_backend(backend)
        self.stamp = torch.full((self.n_keys,), -1, dtype=torch.int32, device=device)
        # The device the buffers really live on (an index, even for a bare 'cuda').
        self.device = device = self.stamp.device
        self.n_tiles = cdiv(self.n_keys, _SCAN_BLOCK)
        assert self.n_tiles <= 2 ** 15, "the scan is a single block; raise _SCAN_BLOCK"
        self.counts = torch.zeros(self.n_tiles, dtype=torch.int32, device=device)
        self.offsets = torch.zeros(self.n_tiles, dtype=torch.int32, device=device)
        # out[0] = count, out[1:] = ascending keys; one contiguous D2H
        self.out = torch.zeros(1 + self.cap, dtype=torch.int32, device=device)
        self.pinned = torch.empty(1 + self.cap, dtype=torch.int32, pin_memory=True)
        self._pin_np = self.pinned.numpy()
        self.event = torch.cuda.Event()
        self.epoch = 0
        self.launches = 0
        self.d2h_bytes = 0
        self._in_flight = False
        if self.backend == "cuda":
            # Surface a build failure at construction, not mid-request.
            from omnimem.nsa.cuda_backend import block_kv_ops
            block_kv_ops.load()
            block_kv_ops.warmup(device)       # lazy-module-load stall, once per process

    # ---------------------------------------------------------------- budget
    @property
    def d2h_bytes_per_call(self) -> int:
        """The fixed-capacity D2H cost of one call, in bytes."""
        return self.pinned.numel() * 4

    def device_bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for t in (self.stamp, self.counts, self.offsets, self.out))

    # ----------------------------------------------------------------- calls
    def enqueue(self, block_indices: torch.Tensor) -> None:
        """Enqueue the dedup launches and the async D2H; the previous one must be wait()-ed."""
        if self._in_flight:
            raise RuntimeError(
                "DeviceBlockDedup.enqueue() called again before the previous "
                "enqueue's wait(); at most one call may be in flight")
        assert block_indices.dtype == torch.int32 and block_indices.is_cuda
        assert block_indices.is_contiguous(), "flat entry -> head math assumes [B,MG,H,T]"
        assert block_indices.shape[-2] == self.H and block_indices.shape[-1] == self.T, (
            f"expected [..., H={self.H}, T={self.T}], got {tuple(block_indices.shape)}")
        assert block_indices.device == self.device, (
            f"block_indices live on {block_indices.device}, the dedup buffers on "
            f"{self.device}")
        self._in_flight = True      # set BEFORE any launch; cleared by wait()
        self.epoch += 1
        if self.epoch >= 2 ** 31 - 1:                # never reached in a request
            self.stamp.fill_(-1)
            self.epoch = 1
        flat = block_indices.view(-1)
        if self.backend == "triton":
            from .block_kv_dedup_triton import launch_dedup_triton
            n_launched = launch_dedup_triton(
                flat, self.stamp, self.counts, self.offsets, self.out,
                self.epoch, self.MB, self.H, self.T, self.bpc, self.cap)
        elif self.backend == "cuda":
            from omnimem.nsa.cuda_backend import block_kv_ops
            n_launched = block_kv_ops.dedup_run(
                flat, self.stamp, self.counts, self.offsets, self.out,
                self.epoch, self.MB, self.H, self.T, self.bpc, self.cap)
        else:
            raise ValueError(f"unknown block KV backend {self.backend!r}")
        # The D2H is enqueued on the current stream of ``self.device``; the event must
        # be recorded on that very stream to be its completion point.
        self.pinned.copy_(self.out, non_blocking=True)
        self.event.record(torch.cuda.current_stream(self.device))
        self.launches += n_launched
        self.d2h_bytes += self.d2h_bytes_per_call

    def wait(self):
        """The ONE host wait.  Returns (ascending keys view, n_unique, overflow)."""
        try:
            self.event.synchronize()
        finally:
            self._in_flight = False
        n = int(self._pin_np[0])                     # host read, no device sync
        return self._pin_np[1:1 + min(n, self.cap)], n, n > self.cap

    def unique_keys(self, block_indices):
        self.enqueue(block_indices)
        return self.wait()
