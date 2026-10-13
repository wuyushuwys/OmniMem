"""Triton JIT bodies for the block-level KV cache kernels K1 and K2."""
from __future__ import annotations

import triton
import triton.language as tl

__all__ = [
    "launch_block_range_triton",
    "launch_gather_publish_clear_triton",
    "_block_range_kernel",
    "_gather_publish_clear_kernel",
    "_CLR_BLOCK",
    "_RANGE_BLOCK",
]

_CLR_BLOCK = 256
_RANGE_BLOCK = 128


# K2: range register / clear
@triton.jit(do_not_specialize=["first_bid", "n_blocks", "k_base", "v_base",
                               "head_stride_bytes", "block_bytes", "n_valid_value"])
def _block_range_kernel(
    Table,              # int64 [2, H, MB] contiguous
    NValid,             # int32 [H]
    first_bid, n_blocks,
    k_base, v_base,     # int64 byte addresses; ignored in CLEAR mode
    head_stride_bytes, block_bytes,
    n_valid_value,
    H: tl.constexpr,
    MB: tl.constexpr,
    WRITE_K: tl.constexpr,
    WRITE_V: tl.constexpr,
    CLEAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    h = tl.program_id(0)
    jb = tl.program_id(1)
    j = jb * BLOCK + tl.arange(0, BLOCK)
    mask = j < n_blocks
    j64 = j.to(tl.int64)
    col = first_bid.to(tl.int64) + j64
    h64 = h.to(tl.int64)
    if CLEAR:
        val_k = tl.zeros([BLOCK], dtype=tl.int64)
        val_v = tl.zeros([BLOCK], dtype=tl.int64)
    else:
        off = h64 * head_stride_bytes.to(tl.int64) + j64 * block_bytes.to(tl.int64)
        val_k = k_base.to(tl.int64) + off
        val_v = v_base.to(tl.int64) + off
    if WRITE_K:
        tl.store(Table + (0 * H + h64) * MB + col, val_k, mask=mask)
    if WRITE_V:
        tl.store(Table + (1 * H + h64) * MB + col, val_v, mask=mask)
    # n_valid_value < 0 leaves the valid extent alone, so an unpaired K row is never valid.
    if jb == 0:
        if n_valid_value >= 0:
            tl.store(NValid + h, n_valid_value.to(tl.int32))


def launch_block_range_triton(table, n_valid, first_bid: int, n_blocks: int,
                              k_base: int, v_base: int, head_stride_bytes: int,
                              block_bytes: int, n_valid_value: int,
                              write_k: bool, write_v: bool, clear: bool) -> None:
    """Issue one K2 launch; the facade guarantees n_blocks > 0 and an in-range window."""
    _, H, MB = table.shape
    grid = (H, triton.cdiv(int(n_blocks), _RANGE_BLOCK))
    _block_range_kernel[grid](
        table, n_valid, int(first_bid), int(n_blocks), int(k_base), int(v_base),
        int(head_stride_bytes), int(block_bytes), int(n_valid_value),
        H=H, MB=MB, WRITE_K=bool(write_k), WRITE_V=bool(write_v),
        CLEAR=bool(clear), BLOCK=_RANGE_BLOCK, num_warps=2,
    )


# K1: fused gather + publish + clear
@triton.jit(do_not_specialize=["n_fetch", "n_clear"])
def _gather_publish_clear_kernel(
    Desc,               # int64 [6 * n_fetch + n_clear]
    Typed,              # any tensor of the KV dtype; only its element type is used
    n_fetch, n_clear,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    ROWS_P: tl.constexpr,
    NWORDS: tl.constexpr,
    NWORDS_P: tl.constexpr,
    WIDE: tl.constexpr,
    CLR_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < n_fetch:
        n64 = n_fetch.to(tl.int64)
        p64 = pid.to(tl.int64)
        src_k = tl.load(Desc + 0 * n64 + p64)
        src_v = tl.load(Desc + 1 * n64 + p64)
        dst_k = tl.load(Desc + 2 * n64 + p64)
        dst_v = tl.load(Desc + 3 * n64 + p64)
        pub_k = tl.load(Desc + 4 * n64 + p64)
        pub_v = tl.load(Desc + 5 * n64 + p64)
        if WIDE:
            # Copy as int64 words: dst is a bitcast int64, so Triton cannot vectorise 16-bit stores.
            sk64 = src_k.to(tl.pointer_type(tl.int64), bitcast=True)
            sv64 = src_v.to(tl.pointer_type(tl.int64), bitcast=True)
            dk64 = dst_k.to(tl.pointer_type(tl.int64), bitcast=True)
            dv64 = dst_v.to(tl.pointer_type(tl.int64), bitcast=True)
            w = tl.arange(0, NWORDS_P)
            wmask = w < NWORDS
            tl.store(dk64 + w, tl.load(sk64 + w, mask=wmask, other=0), mask=wmask)
            tl.store(dv64 + w, tl.load(sv64 + w, mask=wmask, other=0), mask=wmask)
        else:
            sk = src_k.to(tl.pointer_type(Typed.dtype.element_ty), bitcast=True)
            sv = src_v.to(tl.pointer_type(Typed.dtype.element_ty), bitcast=True)
            dk = dst_k.to(tl.pointer_type(Typed.dtype.element_ty), bitcast=True)
            dv = dst_v.to(tl.pointer_type(Typed.dtype.element_ty), bitcast=True)
            r = tl.arange(0, ROWS_P)[:, None]
            c = tl.arange(0, D)[None, :]
            mask = r < ROWS
            off = r * D + c
            vals_k = tl.load(sk + off, mask=mask, other=0.0)
            vals_v = tl.load(sv + off, mask=mask, other=0.0)
            tl.store(dk + off, vals_k, mask=mask)
            tl.store(dv + off, vals_v, mask=mask)
        # Publish after the data of this program is written (same program, same
        # thread block: program order + the stream order of the next kernel).
        pk = pub_k.to(tl.pointer_type(tl.int64), bitcast=True)
        pv = pub_v.to(tl.pointer_type(tl.int64), bitcast=True)
        tl.store(pk, dst_k)
        tl.store(pv, dst_v)
    else:
        n64 = n_fetch.to(tl.int64)
        cb = (pid - n_fetch).to(tl.int64)
        i = cb * CLR_BLOCK + tl.arange(0, CLR_BLOCK).to(tl.int64)
        cmask = i < n_clear.to(tl.int64)
        addrs = tl.load(Desc + 6 * n64 + i, mask=cmask, other=0)
        ptrs = addrs.to(tl.pointer_type(tl.int64), bitcast=True)
        tl.store(ptrs, tl.zeros([CLR_BLOCK], dtype=tl.int64), mask=cmask)


def launch_gather_publish_clear_triton(desc, typed, n_fetch: int, n_clear: int,
                                       block_tokens: int, head_dim: int,
                                       wide: bool, num_warps: int) -> int:
    """Issue one K1 launch; the facade guarantees the descriptor / dtype contract."""
    elem = typed.element_size()
    block_bytes = int(block_tokens) * int(head_dim) * elem
    grid = (n_fetch + triton.cdiv(n_clear, _CLR_BLOCK),)
    _gather_publish_clear_kernel[grid](
        desc, typed, n_fetch, n_clear,
        ROWS=int(block_tokens), D=int(head_dim),
        ROWS_P=max(16, triton.next_power_of_2(int(block_tokens))),
        NWORDS=block_bytes // 8,
        NWORDS_P=max(16, triton.next_power_of_2(max(1, block_bytes // 8))),
        WIDE=bool(wide),
        CLR_BLOCK=_CLR_BLOCK, num_warps=num_warps,
    )
    return grid[0]
