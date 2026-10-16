"""Python adapters for the block-KV CUDA extension (K1 gather, K2 range, K3 dedup)."""
from __future__ import annotations

import torch

from omnimem.nsa.backend import BlockKVBackendError
from omnimem.utils.torch_utils import indexed_device
from . import _block_kv_loader

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


# K1
@torch.compiler.disable()
def gather_publish_clear(desc: torch.Tensor, n_fetch: int, n_clear: int, block_bytes: int,
                         wide: bool, num_warps: int = 1) -> int:
    """Launch K1 on ``desc``'s current stream.  Returns the CTA count (0 = no launch)."""
    n_fetch, n_clear = int(n_fetch), int(n_clear)
    if n_fetch == 0 and n_clear == 0:
        return 0
    return int(_call("gather_publish_clear", desc, n_fetch, n_clear, int(block_bytes),
                     bool(wide), int(num_warps)))


# K2
@torch.compiler.disable()
def block_range(table: torch.Tensor, n_valid: torch.Tensor, first_bid: int, n_blocks: int,
                k_base: int, v_base: int, head_stride_bytes: int, block_bytes: int,
                n_valid_value: int, write_k: bool, write_v: bool, clear: bool) -> None:
    """Register (clear=False) or zero (clear=True) table[rows, :, range]."""
    _call("block_range", table, n_valid, int(first_bid), int(n_blocks), int(k_base),
          int(v_base), int(head_stride_bytes), int(block_bytes), int(n_valid_value),
          bool(write_k), bool(write_v), bool(clear))


# K3
@torch.compiler.disable()
def dedup_run(block_indices_flat: torch.Tensor, stamp: torch.Tensor, counts: torch.Tensor,
              offsets: torch.Tensor, out: torch.Tensor, epoch: int, mb: int, heads: int,
              t: int, bpc: int, cap: int) -> int:
    """Enqueue mark/count/scan/compact.  Returns the number of launches issued."""
    if block_indices_flat.dim() != 1:
        raise BlockKVBackendError("block_indices must be flattened to 1-D")
    return int(_call("dedup_run", block_indices_flat, stamp, counts, offsets, out,
                     int(epoch), int(mb), int(heads), int(t), int(bpc), int(cap)))


# Warm-up: the first launch of each kernel loads its module lazily and stalls the stream.
_warmed: set = set()


@torch.compiler.disable()
def warmup(device) -> int:
    """Launch every kernel variant once on tiny private tensors; idempotent per device."""
    device = indexed_device(device)
    key = (device.type, device.index)
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
