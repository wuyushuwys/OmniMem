// Declarations shared by bindings.cpp and the CUDA translation units; registered on the CUDA key.
#pragma once

#include <ATen/ATen.h>
#include <c10/util/string_view.h>

#include <tuple>

namespace omnimem_nsa {

// Top-K block selection: q [B, MG, H, D] (pooled), k [B, NC, H, D] -> int32 [B, MG, H, topk],
// -1 = invalid, valid slots sorted by descending score.
at::Tensor topk_cuda(
    const at::Tensor& q, const at::Tensor& k, int64_t topk, double scale,
    int64_t block_size, int64_t chunk_size, int64_t chunk_size_k, bool causal,
    int64_t ex_win, int64_t ex_sink, bool prog_excl, bool tf_mask,
    int64_t q_chunk_offset, bool allow_tf32);

// Pointer table: block id -> int64 byte address (0 = invalid / not resident).
std::tuple<at::Tensor, at::Tensor> build_ptr_table_cuda(
    const at::Tensor& block_indices, const at::Tensor& base_k, const at::Tensor& base_v,
    const at::Tensor& n_blocks_per_head, int64_t batch_stride, int64_t blocks_per_chunk,
    int64_t in_chunk_block_bytes);

// Selection forward, dense K/V: (out [B,M,H,D] q.dtype, lse [B,H,M] f32).
std::tuple<at::Tensor, at::Tensor> sel_fwd_cuda(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& block_indices, int64_t block_size, int64_t chunk_size, double scale,
    bool causal, int64_t group_size, bool allow_tf32);

// Selection attention forward, pointer-table K/V (bf16 blocks addressed by int64 bytes).
std::tuple<at::Tensor, at::Tensor> sel_fwd_ptr_cuda(
    const at::Tensor& q, const at::Tensor& ptr_k, const at::Tensor& ptr_v,
    int64_t block_size, int64_t group_size, int64_t stride_bn, int64_t stride_bd,
    double scale, c10::string_view input_precision, at::TensorList owners);

// Selection backward, dense: (dq q.dtype, dk k.dtype, dv v.dtype).
std::tuple<at::Tensor, at::Tensor, at::Tensor> sel_bwd_cuda(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& block_indices, const at::Tensor& out, const at::Tensor& lse,
    const at::Tensor& d_out, int64_t block_size, int64_t chunk_size, double scale,
    bool causal, int64_t group_size, bool allow_tf32);

// Selection backward, pointer-table: (dq, delta); dK/dV go to the fp32 scratch at base_dk/base_dv.
std::tuple<at::Tensor, at::Tensor> sel_bwd_ptr_cuda(
    const at::Tensor& q, const at::Tensor& out, const at::Tensor& lse, const at::Tensor& d_out,
    const at::Tensor& block_indices, const at::Tensor& ptr_k, const at::Tensor& ptr_v,
    const at::Tensor& base_k, const at::Tensor& base_v,
    const at::Tensor& base_dk, const at::Tensor& base_dv,
    const at::Tensor& n_blocks_per_head,
    int64_t block_size, int64_t group_size,
    int64_t batch_stride_kv_bytes, int64_t batch_stride_dkv_bytes,
    int64_t stride_bn, int64_t stride_bd, int64_t stride_dbn, int64_t stride_dbd,
    int64_t blocks_per_chunk, int64_t in_chunk_block_bytes_kv, int64_t in_chunk_block_bytes_dkv,
    double scale, c10::string_view input_precision, bool compute_dkv,
    at::TensorList owners, at::TensorList grad_buffers);

// Inverted index (CSR) over query groups: (sorted_groups [B,H,MG*T], offsets [B,H,NS+1]);
// empty n_blocks_per_head means num_blocks for every head.
std::tuple<at::Tensor, at::Tensor> inverted_index_cuda(
    const at::Tensor& block_indices, const at::Tensor& n_blocks_per_head, int64_t num_blocks,
    int64_t block_size, int64_t chunk_size, bool causal, int64_t group_size);

}  // namespace omnimem_nsa
