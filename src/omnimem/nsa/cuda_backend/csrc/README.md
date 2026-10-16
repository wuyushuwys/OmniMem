# OmniMem CUDA backend sources

C++/CUDA sources of the two extensions behind `OMNIMEM_NSA_BACKEND=cuda`, built with
`torch.utils.cpp_extension` for sm_80 and newer. Build ahead of time with:

    python -m omnimem.nsa.cuda_backend.build

NSA operators (`torch.ops.omnimem_nsa.*`):

- `bindings.cpp`: operator schemas and CUDA dispatcher registration.
- `ops.h`: declarations shared by the bindings and the kernels.
- `common.cuh`: strided views, KV block accessors, dtype dispatch, TF32 helpers.
- `bwd_common.cuh`: backward helpers (dK/dV accessors, TF32 mma, cp.async, tile plans).
- `topk.cu`: block top-k selection (register bitonic merge, or CUB sort for large topk).
- `ptr_table.cu`: block id to int64 byte address table.
- `selection_fwd.cu`: selection attention forward, dense and pointer KV.
- `selection_bwd.cu`: selection attention backward (delta, dQ, dK/dV).
- `inverted_index.cu`: CSR index from block to query groups, used by dK/dV.

Block-KV cache kernels (pybind11 module):

- `block_kv_bindings.cpp`: module bindings.
- `block_kv_ops.h`: contract of the three kernels.
- `block_kv_common.cuh`: shared constants and the CTA-wide byte copy.
- `block_kv_gather.cu`: K1, gather, publish and clear.
- `block_kv_range.cu`: K2, register or clear a range of the address table.
- `block_kv_dedup.cu`: K3, device dedup of selected (head, block) entries.
