"""Pointer-table builder; base == 0 means the chunk is not GPU-resident and the pointer is 0."""
from typing import Optional

import torch

from ._dispatch import build_ptr_table_dispatch

__all__ = ["build_ptr_table"]


def build_ptr_table(
    block_indices: torch.Tensor,
    block_base_ptrs_k: torch.Tensor,
    block_base_ptrs_v: torch.Tensor,
    n_blocks_per_head: torch.Tensor,
    batch_stride: int,
    blocks_per_chunk: int = 0,
    in_chunk_block_bytes: int = 0,
    backend: Optional[str] = None,
):
    """backend is 'triton' / 'cuda' / None (process default), captured for this call."""
    return build_ptr_table_dispatch(
        block_indices, block_base_ptrs_k, block_base_ptrs_v, n_blocks_per_head,
        batch_stride, blocks_per_chunk, in_chunk_block_bytes,
        backend=backend,
    )
