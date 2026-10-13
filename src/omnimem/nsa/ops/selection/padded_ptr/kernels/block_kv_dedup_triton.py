"""Triton JIT bodies for the device-side dedup of the selected block keys."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["launch_dedup_triton", "_MARK_BLOCK", "_SCAN_BLOCK"]

_MARK_BLOCK = 1024
_SCAN_BLOCK = 1024


@triton.jit(do_not_specialize=["n_entries", "epoch"])
def _mark_epoch_kernel(
    BlockIdx,           # int32 [n_entries] flattened [B, MG, H, T]
    Stamp,              # int32 [H * MB]
    n_entries, epoch,
    MB: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    BPC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    e = pid * BLOCK + tl.arange(0, BLOCK)
    m = e < n_entries
    bid = tl.load(BlockIdx + e, mask=m, other=-1)
    # [B, MG, H, T] contiguous -> the head of flat entry e is (e // T) % H
    h = (e // T) % H
    ok = m & (bid >= 0) & (bid < MB)
    # Clamp BEFORE the division so a -1 pad can never produce a negative key
    # (integer division of a negative value is implementation-defined in LLVM).
    b64 = tl.where(ok, bid, 0).to(tl.int64)
    cid = b64 // BPC
    loc = b64 - cid * BPC
    key = (cid * H + h.to(tl.int64)) * BPC + loc          # host-order key
    stamp_v = tl.zeros([BLOCK], dtype=tl.int32) + epoch.to(tl.int32)
    tl.store(Stamp + key, stamp_v, mask=ok)


@triton.jit(do_not_specialize=["n_keys", "epoch"])
def _count_epoch_kernel(Stamp, Counts, n_keys, epoch, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n_keys
    s = tl.load(Stamp + i, mask=m, other=-1)
    hit = m & (s == epoch.to(tl.int32))
    tl.store(Counts + pid, tl.sum(hit.to(tl.int32), axis=0))


@triton.jit(do_not_specialize=["n_tiles"])
def _scan_counts_kernel(Counts, Offsets, Out, n_tiles, NT_P: tl.constexpr):
    """Single block: exclusive scan of the tile counts + the grand total into Out[0]."""
    i = tl.arange(0, NT_P)
    m = i < n_tiles
    c = tl.load(Counts + i, mask=m, other=0)
    tl.store(Offsets + i, tl.cumsum(c, axis=0) - c, mask=m)
    tl.store(Out, tl.sum(c, axis=0))


@triton.jit(do_not_specialize=["n_keys", "cap", "epoch"])
def _compact_epoch_kernel(
    Stamp, Offsets, Out,
    n_keys, cap, epoch,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n_keys
    s = tl.load(Stamp + i, mask=m, other=-1)
    hit = m & (s == epoch.to(tl.int32))
    h32 = hit.to(tl.int32)
    rank = tl.cumsum(h32, axis=0) - h32              # ascending within the tile
    pos = tl.load(Offsets + pid) + rank              # ascending across tiles
    keep = hit & (pos < cap)
    tl.store(Out + 1 + pos, i.to(tl.int32), mask=keep)


def launch_dedup_triton(flat: torch.Tensor, stamp: torch.Tensor, counts: torch.Tensor,
                        offsets: torch.Tensor, out: torch.Tensor,
                        epoch: int, mb: int, heads: int, t: int, bpc: int,
                        cap: int) -> int:
    """Issue the mark/count/scan/compact launches.  Returns 4 (the launch count)."""
    n_entries = flat.numel()
    n_keys = heads * mb
    n_tiles = triton.cdiv(n_keys, _SCAN_BLOCK)
    _mark_epoch_kernel[(triton.cdiv(n_entries, _MARK_BLOCK),)](
        flat, stamp, n_entries, epoch,
        MB=mb, H=heads, T=t, BPC=bpc, BLOCK=_MARK_BLOCK, num_warps=4)
    _count_epoch_kernel[(n_tiles,)](
        stamp, counts, n_keys, epoch, BLOCK=_SCAN_BLOCK, num_warps=4)
    _scan_counts_kernel[(1,)](
        counts, offsets, out, n_tiles,
        NT_P=triton.next_power_of_2(n_tiles), num_warps=4)
    _compact_epoch_kernel[(n_tiles,)](
        stamp, offsets, out, n_keys, cap, epoch, BLOCK=_SCAN_BLOCK, num_warps=4)
    return 4
