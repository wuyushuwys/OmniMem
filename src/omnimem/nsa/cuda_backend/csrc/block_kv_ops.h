// Block-KV cache kernels: byte copies and int32/int64 metadata only.
// Each function runs on the current stream of its first tensor's device and never syncs.
// Only tensor-level properties are validated here; the Python facade owns the rest.
#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace omnimem_block_kv {

// K1: gather + publish + clear; returns CTAs launched, n_fetch + ceil(n_clear / 256).
//   desc: int64 CUDA, SoA [src_k|src_v|dst_k|dst_v|pub_k|pub_v] x N, then clear x M
//   wide needs block_bytes % 8 == 0 and 8-byte aligned addresses; blockDim.x = 32 * num_warps.
int64_t gather_publish_clear(const at::Tensor& desc, int64_t n_fetch, int64_t n_clear,
                             int64_t block_bytes, bool wide, int64_t num_warps);

// K2: range register / clear of the address table int64 [2, H, MB]; n_valid int32 [H].
//   table[name, h, first_bid + j] = base_name + h * head_stride_bytes + j * block_bytes.
//   n_valid_value >= 0 sets n_valid[:], < 0 leaves it; n_blocks > 0 and the range is inside MB.
void block_range(at::Tensor& table, at::Tensor& n_valid, int64_t first_bid, int64_t n_blocks,
                 int64_t k_base, int64_t v_base, int64_t head_stride_bytes, int64_t block_bytes,
                 int64_t n_valid_value, bool write_k, bool write_v, bool clear);

// K3: device dedup of selected (head, bid) entries into ascending host-order keys.
//   block_indices: int32 [n_entries], flattened [B, MG, H, T]; key < n_keys = heads * mb.
//   stamp: int32 [n_keys], persistent; counts/offsets: int32 [n_tiles <= 2^15].
//   out: int32 [1 + cap]; out[0] = unique total (may exceed cap), then ascending keys.
int64_t dedup_run(const at::Tensor& block_indices, at::Tensor& stamp, at::Tensor& counts,
                  at::Tensor& offsets, at::Tensor& out, int64_t epoch, int64_t mb,
                  int64_t heads, int64_t t, int64_t bpc, int64_t cap);

}  // namespace omnimem_block_kv
