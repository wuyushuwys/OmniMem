"""Python adapters for the block-KV CUDA extension (K1 gather, K2 range, K3 dedup)."""
from __future__ import annotations

import torch

from omnimem.nsa.backend import BlockKVBackendError
from . import _block_kv_loader
from .ops import _INT32_MAX

__all__ = ["gather_publish_clear", "block_range", "dedup_run",
           "is_loaded", "load", "warmup"]


def _ops():
    return _block_kv_loader.ops()


def _call(name: str, *args):
    """Run the extension function name; its RuntimeError becomes BlockKVBackendError."""
    fn = getattr(_ops(), name)
    try:
        return fn(*args)
    except (torch.OutOfMemoryError, torch.AcceleratorError):
        raise
    except RuntimeError as e:
        raise BlockKVBackendError(str(e)) from e


def is_loaded() -> bool:
    return _block_kv_loader.is_loaded()


def load(verbose: bool = False):
    return _block_kv_loader.load(verbose=verbose)


def _check_cuda_i64(t: torch.Tensor, name: str, contiguous: bool = True) -> None:
    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        raise BlockKVBackendError(f"{name} must be a CUDA tensor")
    if t.dtype != torch.int64:
        raise BlockKVBackendError(f"{name} must be int64, got {t.dtype}")
    if contiguous and not t.is_contiguous():
        raise BlockKVBackendError(f"{name} must be contiguous")


def _check_warps(num_warps: int) -> int:
    num_warps = int(num_warps)
    if num_warps < 1 or num_warps > 32:
        raise BlockKVBackendError(f"num_warps={num_warps} outside [1, 32]")
    return num_warps


# K1
@torch.compiler.disable()
def gather_publish_clear(desc: torch.Tensor, n_fetch: int, n_clear: int, block_bytes: int,
                         wide: bool, num_warps: int = 1) -> int:
    """Launch K1 on ``desc``'s current stream.  Returns the CTA count (0 = no launch)."""
    n_fetch, n_clear, block_bytes = int(n_fetch), int(n_clear), int(block_bytes)
    if n_fetch == 0 and n_clear == 0:
        return 0
    _check_cuda_i64(desc, "desc")
    if n_fetch < 0 or n_clear < 0:
        raise BlockKVBackendError("n_fetch / n_clear must be >= 0")
    if desc.numel() < 6 * n_fetch + n_clear:
        raise BlockKVBackendError(
            f"descriptor holds {desc.numel()} elements, need {6 * n_fetch + n_clear}")
    if block_bytes <= 0 or block_bytes % 2:
        raise BlockKVBackendError(f"block_bytes={block_bytes} must be a positive even count")
    if wide and block_bytes % 8:
        raise BlockKVBackendError(
            f"wide copy needs block_bytes ({block_bytes}) to be a multiple of 8")
    return int(_call("gather_publish_clear", desc, n_fetch, n_clear, block_bytes, bool(wide),
                     _check_warps(num_warps)))


# K2
@torch.compiler.disable()
def block_range(table: torch.Tensor, n_valid: torch.Tensor, first_bid: int, n_blocks: int,
                k_base: int, v_base: int, head_stride_bytes: int, block_bytes: int,
                n_valid_value: int, write_k: bool, write_v: bool, clear: bool) -> None:
    """Register (clear=False) or zero (clear=True) table[rows, :, range]."""
    _check_cuda_i64(table, "table")
    if table.dim() != 3 or table.shape[0] != 2:
        raise BlockKVBackendError(f"table must be [2, H, MB], got {tuple(table.shape)}")
    if not n_valid.is_cuda or n_valid.dtype != torch.int32 \
            or n_valid.shape != (table.shape[1],) or not n_valid.is_contiguous():
        raise BlockKVBackendError(
            f"n_valid must be a contiguous int32 CUDA [H={table.shape[1]}] tensor")
    if n_valid.device != table.device:
        raise BlockKVBackendError("table and n_valid on different devices")
    first_bid, n_blocks = int(first_bid), int(n_blocks)
    if n_blocks <= 0 or first_bid < 0 or first_bid + n_blocks > table.shape[2]:
        raise BlockKVBackendError(
            f"block range [{first_bid}, {first_bid + n_blocks}) outside [0, {table.shape[2]})")
    if not clear:
        if not (write_k or write_v):
            raise BlockKVBackendError("register_block_range: nothing to register")
        if (write_k and int(k_base) == 0) or (write_v and int(v_base) == 0):
            raise BlockKVBackendError(
                "register_block_range: a row is selected for writing but its base is 0")
    _call("block_range", table, n_valid, first_bid, n_blocks, int(k_base), int(v_base),
          int(head_stride_bytes), int(block_bytes), int(n_valid_value),
          bool(write_k), bool(write_v), bool(clear))


# K3
@torch.compiler.disable()
def dedup_run(block_indices_flat: torch.Tensor, stamp: torch.Tensor, counts: torch.Tensor,
              offsets: torch.Tensor, out: torch.Tensor, epoch: int, mb: int, heads: int,
              t: int, bpc: int, cap: int) -> int:
    """Enqueue mark/count/scan/compact.  Returns the number of launches issued."""
    for name, ten in (("block_indices", block_indices_flat), ("stamp", stamp),
                      ("counts", counts), ("offsets", offsets), ("out", out)):
        if not ten.is_cuda or ten.dtype != torch.int32 or not ten.is_contiguous():
            raise BlockKVBackendError(f"{name} must be a contiguous int32 CUDA tensor")
        if ten.device != stamp.device:
            raise BlockKVBackendError(f"{name} on {ten.device}, stamp on {stamp.device}")
    if block_indices_flat.dim() != 1:
        raise BlockKVBackendError("block_indices must be flattened to 1-D")
    mb, heads, t, bpc, cap, epoch = int(mb), int(heads), int(t), int(bpc), int(cap), int(epoch)
    if mb <= 0 or heads <= 0 or t <= 0 or bpc <= 0 or mb % bpc:
        raise BlockKVBackendError("mb / heads / t / bpc must be positive with bpc | mb")
    n_keys = heads * mb
    if n_keys > _INT32_MAX or stamp.numel() != n_keys:
        raise BlockKVBackendError(f"stamp must hold n_keys={n_keys} int32 entries")
    n_tiles = -(-n_keys // 1024)
    if n_tiles > 2 ** 15 or counts.numel() < n_tiles or offsets.numel() < n_tiles:
        raise BlockKVBackendError(
            f"n_tiles={n_tiles} exceeds the single-block scan (2**15) or the buffers")
    if cap < 0 or out.numel() < 1 + cap:
        raise BlockKVBackendError(f"out must hold 1 + cap={cap} entries")
    if epoch <= 0 or epoch > _INT32_MAX:
        raise BlockKVBackendError(f"epoch={epoch} must be in [1, INT32_MAX]")
    if block_indices_flat.numel() > _INT32_MAX:
        raise BlockKVBackendError("too many entries for an int32 index")
    return int(_call("dedup_run", block_indices_flat, stamp, counts, offsets, out,
                     epoch, mb, heads, t, bpc, cap))


# Warm-up: the first launch of each kernel loads its module lazily and stalls the stream.
_warmed: set = set()


@torch.compiler.disable()
def warmup(device) -> int:
    """Launch every kernel variant once on tiny private tensors; idempotent per device."""
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device(device.type, index)
    key = (device.type, index)
    if key in _warmed:
        return 0
    load()
    n = 0
    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            # K1: one fetch (device source -> device dst) + one clear, wide and narrow.
            src = torch.ones(2, 16, dtype=torch.int16, device=device)        # 2 blocks x 32 B
            dst = torch.zeros(2, 16, dtype=torch.int16, device=device)
            table = torch.full((4,), -1, dtype=torch.int64, device=device)   # pub_k, pub_v, clear x2
            a = table.data_ptr()
            desc = torch.tensor(
                [src[0].data_ptr(), src[1].data_ptr(), dst[0].data_ptr(), dst[1].data_ptr(),
                 a, a + 8, a + 16, a + 24], dtype=torch.int64, device=device)
            for wide in (True, False):
                n += int(_ops().gather_publish_clear(desc, 1, 2, 32, wide, 1) > 0)
            # K2: register + clear.
            tab = torch.zeros(2, 1, 4, dtype=torch.int64, device=device)
            nv = torch.zeros(1, dtype=torch.int32, device=device)
            _ops().block_range(tab, nv, 0, 4, 4096, 8192, 64, 16, 4, True, True, False)
            _ops().block_range(tab, nv, 0, 4, 0, 0, 0, 0, -1, True, True, True)
            n += 2
            # K3: mark / count / scan / compact.
            ids = torch.tensor([0, -1, 3, 3], dtype=torch.int32, device=device)
            stamp = torch.full((4,), -1, dtype=torch.int32, device=device)
            cnt = torch.zeros(1, dtype=torch.int32, device=device)
            off = torch.zeros(1, dtype=torch.int32, device=device)
            out = torch.zeros(1 + 4, dtype=torch.int32, device=device)
            n += int(_ops().dedup_run(ids, stamp, cnt, off, out, 1, 4, 1, 4, 4, 4))
        stream.synchronize()
    _warmed.add(key)
    return n
