"""Padded-ptr selection-attention kernels; lazy so importing the package never imports Triton."""
from __future__ import annotations

from omnimem.nsa.ops._int_utils import compute_tile_n      # Triton-free

__all__ = [
    "_sel_attn_fwd_padded_ptr_kernel",
    "compute_tile_n",
    "_sel_attn_bwd_preprocess_kernel",
    "_sel_attn_bwd_dq_padded_ptr_kernel",
    "_sel_attn_bwd_dkv_padded_ptr_kernel",
    "build_inverted_index_padded_ptr",
    "selection_attention_padded_ptr_bwd",
    "build_ptr_table_triton",
]

#: name -> submodule (relative to this package) that defines it.
_SOURCE = {
    "_sel_attn_fwd_padded_ptr_kernel": "select_fwd",
    "_sel_attn_bwd_preprocess_kernel": "select_bwd",
    "_sel_attn_bwd_dq_padded_ptr_kernel": "select_bwd",
    "_sel_attn_bwd_dkv_padded_ptr_kernel": "select_bwd",
    "build_inverted_index_padded_ptr": "select_bwd",
    "selection_attention_padded_ptr_bwd": "select_bwd",
    "build_ptr_table_triton": "ptr_builder",
}


def __getattr__(name: str):
    mod_name = _SOURCE.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(f".{mod_name}", __name__)
    value = getattr(mod, name)
    globals()[name] = value            # cache: only the first access pays the import
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
