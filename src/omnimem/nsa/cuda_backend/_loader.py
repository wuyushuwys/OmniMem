"""Lazy loader of the NSA CUDA extension (torch.ops.omnimem_nsa.*); no Triton fallback."""
from __future__ import annotations

import threading
from typing import Optional

import torch

from omnimem.nsa.backend import NSABackendError
from ._extension import Extension

LIB_NAMESPACE = "omnimem_nsa"
#: The operators the library registers under ``torch.ops.omnimem_nsa``.
OP_NAMES = ("topk", "build_ptr_table", "sel_fwd", "sel_fwd_ptr", "sel_bwd", "sel_bwd_ptr")

EXTENSION = Extension(
    prefix="omnimem_nsa_cuda",
    what="NSA CUDA extension",
    sources=("bindings.cpp", "topk.cu", "ptr_table.cu", "selection_fwd.cu",
             "selection_bwd.cu", "inverted_index.cu"),
    headers=("ops.h", "common.cuh", "bwd_common.cuh"),
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

_lock = threading.Lock()
_loaded = False
_load_error: Optional[Exception] = None
_library_path: Optional[str] = None


def is_loaded() -> bool:
    return _loaded


def library_path() -> Optional[str]:
    return _library_path


def load(verbose: bool = False) -> None:
    """Compile (if needed) and load the extension; idempotent."""
    global _loaded, _load_error
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        if _load_error is not None:
            raise NSABackendError(
                f"NSA CUDA extension previously failed to load: {_load_error}"
            ) from _load_error
        try:
            _do_load(verbose=verbose)
        except NSABackendError:
            raise
        except Exception as e:  # noqa: BLE001 - surface build errors verbatim
            _load_error = e
            raise NSABackendError(
                "failed to build/load the NSA CUDA extension (no Triton fallback). "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
        _loaded = True


def _do_load(verbose: bool) -> None:
    global _library_path
    _, _library_path = EXTENSION.build_and_load(verbose=verbose)
    namespace = getattr(torch.ops, LIB_NAMESPACE)
    missing = [n for n in OP_NAMES if not hasattr(namespace, n)]
    if missing:
        raise NSABackendError(
            f"NSA CUDA extension loaded but torch.ops.{LIB_NAMESPACE} lacks {missing}")
    from . import ops as _ops_mod  # noqa: WPS433 - register the fake kernels once loaded
    _ops_mod.register_fakes()


def ops():
    """Return the ``torch.ops.omnimem_nsa`` namespace, loading the extension on demand."""
    load()
    return getattr(torch.ops, LIB_NAMESPACE)
