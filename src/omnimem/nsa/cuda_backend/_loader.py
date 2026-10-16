"""Lazy loader of the NSA CUDA extension (torch.ops.omnimem_nsa.*); no Triton fallback."""
from __future__ import annotations

from typing import Optional

import torch

from omnimem.nsa.backend import NSABackendError
from ._extension import Extension, LazyExtension

LIB_NAMESPACE = "omnimem_nsa"
#: The operators the library registers under ``torch.ops.omnimem_nsa``.
OP_NAMES = ("topk", "build_ptr_table", "sel_fwd", "sel_fwd_ptr", "sel_bwd", "sel_bwd_ptr")

EXTENSION = Extension(
    prefix="omnimem_nsa_cuda",
    what="NSA CUDA extension",
    sources=("bindings.cpp", "topk.cu", "ptr_table.cu", "selection_fwd.cu",
             "selection_bwd.cu", "inverted_index.cu"),
    headers=("ops.h", "common.cuh", "bwd_common.cuh", "warp_scan.cuh"),
    cuda_cflags=(
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ),
    python_module=False,
    error=NSABackendError,
)


def _register_fakes() -> None:
    from . import ops as _ops_mod  # noqa: WPS433 - register the fake kernels once loaded
    _ops_mod.register_fakes()


_lazy = LazyExtension(EXTENSION, OP_NAMES, namespace=LIB_NAMESPACE, on_load=_register_fakes)


def is_loaded() -> bool:
    return _lazy.is_loaded()


def library_path() -> Optional[str]:
    return _lazy.library_path


def load(verbose: bool = False) -> None:
    """Compile (if needed) and load the extension; idempotent."""
    _lazy.load(verbose)


def ops():
    """Return the ``torch.ops.omnimem_nsa`` namespace, loading the extension on demand."""
    load()
    return getattr(torch.ops, LIB_NAMESPACE)
