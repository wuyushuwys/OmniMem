// Operator schemas + CUDA dispatcher registration for the OmniMem NSA backend.
// No CUDA code lives here; kernels are in the .cu translation units.
#include <torch/library.h>

#include "ops.h"

TORCH_LIBRARY(omnimem_nsa, m) {
  m.def(
      "topk(Tensor q, Tensor k, int topk, float scale, int block_size, int chunk_size, "
      "int chunk_size_k, bool causal, int ex_win, int ex_sink, bool prog_excl, bool tf_mask, "
      "int q_chunk_offset, bool allow_tf32) -> Tensor");
  m.def(
      "build_ptr_table(Tensor block_indices, Tensor base_k, Tensor base_v, "
      "Tensor n_blocks_per_head, int batch_stride, int blocks_per_chunk, "
      "int in_chunk_block_bytes) -> (Tensor, Tensor)");
  m.def(
      "sel_fwd(Tensor q, Tensor k, Tensor v, Tensor block_indices, int block_size, "
      "int chunk_size, float scale, bool causal, int group_size, bool allow_tf32) "
      "-> (Tensor, Tensor)");
  m.def(
      "sel_fwd_ptr(Tensor q, Tensor ptr_k, Tensor ptr_v, int block_size, int group_size, "
      "int stride_bn, int stride_bd, float scale, str input_precision, Tensor[] owners) "
      "-> (Tensor, Tensor)");
  m.def(
      "sel_bwd(Tensor q, Tensor k, Tensor v, Tensor block_indices, Tensor out, Tensor lse, "
      "Tensor d_out, int block_size, int chunk_size, float scale, bool causal, "
      "int group_size, bool allow_tf32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "sel_bwd_ptr(Tensor q, Tensor out, Tensor lse, Tensor d_out, Tensor block_indices, "
      "Tensor ptr_k, Tensor ptr_v, Tensor base_k, Tensor base_v, Tensor base_dk, "
      "Tensor base_dv, Tensor n_blocks_per_head, int block_size, int group_size, "
      "int batch_stride_kv_bytes, int batch_stride_dkv_bytes, int stride_bn, int stride_bd, "
      "int stride_dbn, int stride_dbd, int blocks_per_chunk, int in_chunk_block_bytes_kv, "
      "int in_chunk_block_bytes_dkv, float scale, str input_precision, bool compute_dkv, "
      "Tensor[] owners, Tensor(a!)[] grad_buffers) -> (Tensor, Tensor)");
  m.def(
      "inverted_index(Tensor block_indices, Tensor n_blocks_per_head, int num_blocks, "
      "int block_size, int chunk_size, bool causal, int group_size) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(omnimem_nsa, CUDA, m) {
  m.impl("topk", &omnimem_nsa::topk_cuda);
  m.impl("build_ptr_table", &omnimem_nsa::build_ptr_table_cuda);
  m.impl("sel_fwd", &omnimem_nsa::sel_fwd_cuda);
  m.impl("sel_fwd_ptr", &omnimem_nsa::sel_fwd_ptr_cuda);
  m.impl("sel_bwd", &omnimem_nsa::sel_bwd_cuda);
  m.impl("sel_bwd_ptr", &omnimem_nsa::sel_bwd_ptr_cuda);
  m.impl("inverted_index", &omnimem_nsa::inverted_index_cuda);
}
