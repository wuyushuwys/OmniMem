// pybind11 module, not TORCH_LIBRARY: K1 writes through raw addresses no schema expresses.
#include <torch/extension.h>

#include "block_kv_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "OmniMem block-KV cache CUDA kernels (K1 gather/publish/clear, K2 range "
            "register/clear, K3 device dedup)";
  m.def("gather_publish_clear", &omnimem_block_kv::gather_publish_clear,
        py::arg("desc"), py::arg("n_fetch"), py::arg("n_clear"), py::arg("block_bytes"),
        py::arg("wide"), py::arg("num_warps"));
  m.def("block_range", &omnimem_block_kv::block_range,
        py::arg("table"), py::arg("n_valid"), py::arg("first_bid"), py::arg("n_blocks"),
        py::arg("k_base"), py::arg("v_base"), py::arg("head_stride_bytes"),
        py::arg("block_bytes"), py::arg("n_valid_value"), py::arg("write_k"),
        py::arg("write_v"), py::arg("clear"));
  m.def("dedup_run", &omnimem_block_kv::dedup_run,
        py::arg("block_indices"), py::arg("stamp"), py::arg("counts"), py::arg("offsets"),
        py::arg("out"), py::arg("epoch"), py::arg("mb"), py::arg("heads"), py::arg("t"),
        py::arg("bpc"), py::arg("cap"));
}
