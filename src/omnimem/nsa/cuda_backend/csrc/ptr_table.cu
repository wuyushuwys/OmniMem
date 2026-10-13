// Pointer-table build: ptr_x = base_x[h, cid] + b*batch_stride + off, or 0 if bid/base is invalid.
// bpc > 0: cid = bid / bpc, off = (bid % bpc) * in_chunk_block_bytes; else cid = bid, off = 0.
// int64 byte arithmetic; the base is tested before any offset is added.
#include "common.cuh"
#include "ops.h"

#include <algorithm>

namespace omnimem_nsa {
namespace {

__global__ void build_ptr_table_kernel(
    const int32_t* __restrict__ ids, int64_t ids_sb, int64_t ids_sg, int64_t ids_sh,
    int64_t ids_st, const int64_t* __restrict__ base_k, const int64_t* __restrict__ base_v,
    int64_t base_sh, int64_t base_sc, int max_chunks, const int32_t* __restrict__ n_per_head,
    int64_t* __restrict__ ptr_k, int64_t* __restrict__ ptr_v, int MG, int H, int T,
    int64_t batch_stride, int blocks_per_chunk, int64_t in_chunk_block_bytes, int64_t total) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    // idx -> (b, g, h, t) over the contiguous [B, MG, H, T] output
    const int t = static_cast<int>(idx % T);
    const int64_t r1 = idx / T;
    const int h = static_cast<int>(r1 % H);
    const int64_t r2 = r1 / H;
    const int g = static_cast<int>(r2 % MG);
    const int b = static_cast<int>(r2 / MG);

    const int32_t bid = ids[b * ids_sb + g * ids_sg + h * ids_sh + t * ids_st];
    int64_t pk = 0;
    int64_t pv = 0;
    if (bid >= 0 && bid < n_per_head[h]) {
      int64_t cid;
      int64_t off;
      if (blocks_per_chunk > 0) {
        cid = static_cast<int64_t>(bid) / blocks_per_chunk;
        off = static_cast<int64_t>(bid % blocks_per_chunk) * in_chunk_block_bytes;
      } else {
        cid = static_cast<int64_t>(bid);
        off = 0;
      }
      // Metadata larger than the base table must not read outside it: the block is skipped.
      if (cid < max_chunks) {
        const int64_t boff = h * base_sh + cid * base_sc;
        const int64_t bk = base_k[boff];
        const int64_t bv = base_v[boff];
        const int64_t delta = static_cast<int64_t>(b) * batch_stride + off;
        pk = (bk != 0) ? bk + delta : 0;
        pv = (bv != 0) ? bv + delta : 0;
      }
    }
    ptr_k[idx] = pk;
    ptr_v[idx] = pv;
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> build_ptr_table_cuda(
    const at::Tensor& block_indices, const at::Tensor& base_k, const at::Tensor& base_v,
    const at::Tensor& n_blocks_per_head, int64_t batch_stride, int64_t blocks_per_chunk,
    int64_t in_chunk_block_bytes) {
  TORCH_CHECK(block_indices.is_cuda() && base_k.is_cuda() && base_v.is_cuda() &&
                  n_blocks_per_head.is_cuda(),
              "omnimem_nsa::build_ptr_table: all inputs must be CUDA tensors");
  TORCH_CHECK(block_indices.device() == base_k.device() &&
                  block_indices.device() == base_v.device() &&
                  block_indices.device() == n_blocks_per_head.device(),
              "omnimem_nsa::build_ptr_table: inputs on different devices");
  TORCH_CHECK(block_indices.dim() == 4,
              "omnimem_nsa::build_ptr_table: block_indices must be [B,MG,H,T], got ",
              block_indices.sizes());
  TORCH_CHECK(block_indices.scalar_type() == at::kInt,
              "omnimem_nsa::build_ptr_table: block_indices must be int32, got ",
              block_indices.scalar_type());
  TORCH_CHECK(base_k.dim() == 2 && base_v.dim() == 2,
              "omnimem_nsa::build_ptr_table: base_k/base_v must be [H, max_chunks]");
  TORCH_CHECK(base_k.sizes() == base_v.sizes(),
              "omnimem_nsa::build_ptr_table: base_k/base_v shape mismatch ", base_k.sizes(),
              " vs ", base_v.sizes());
  TORCH_CHECK(base_k.scalar_type() == at::kLong && base_v.scalar_type() == at::kLong,
              "omnimem_nsa::build_ptr_table: base tables must be int64 byte addresses");
  TORCH_CHECK(base_k.stride(0) == base_v.stride(0) && base_k.stride(1) == base_v.stride(1),
              "omnimem_nsa::build_ptr_table: base_k/base_v must share strides");
  TORCH_CHECK(n_blocks_per_head.dim() == 1 && n_blocks_per_head.scalar_type() == at::kInt,
              "omnimem_nsa::build_ptr_table: n_blocks_per_head must be int32 [H]");
  TORCH_CHECK(n_blocks_per_head.is_contiguous(),
              "omnimem_nsa::build_ptr_table: n_blocks_per_head must be contiguous");

  const int B = static_cast<int>(block_indices.size(0));
  const int MG = static_cast<int>(block_indices.size(1));
  const int H = static_cast<int>(block_indices.size(2));
  const int T = static_cast<int>(block_indices.size(3));
  TORCH_CHECK(base_k.size(0) == H, "omnimem_nsa::build_ptr_table: base table has ",
              base_k.size(0), " heads but block_indices has ", H);
  TORCH_CHECK(n_blocks_per_head.size(0) == H,
              "omnimem_nsa::build_ptr_table: n_blocks_per_head has ", n_blocks_per_head.size(0),
              " entries but block_indices has ", H, " heads");
  TORCH_CHECK(blocks_per_chunk >= 0,
              "omnimem_nsa::build_ptr_table: blocks_per_chunk must be >= 0");
  TORCH_CHECK(blocks_per_chunk == 0 || in_chunk_block_bytes > 0,
              "omnimem_nsa::build_ptr_table: in_chunk_block_bytes required for a chunk-level "
              "base table");
  TORCH_CHECK(blocks_per_chunk <= std::numeric_limits<int32_t>::max(),
              "omnimem_nsa::build_ptr_table: blocks_per_chunk out of range");

  at::cuda::CUDAGuard guard(block_indices.device());
  const auto opt = block_indices.options().dtype(at::kLong);
  at::Tensor ptr_k = at::empty({B, MG, H, T}, opt);
  at::Tensor ptr_v = at::empty({B, MG, H, T}, opt);
  const int64_t total = static_cast<int64_t>(B) * MG * H * T;
  if (total == 0) return std::make_tuple(ptr_k, ptr_v);
  TORCH_CHECK(base_k.size(1) > 0, "omnimem_nsa::build_ptr_table: empty base table");

  const int threads = 256;
  const int64_t want = (total + threads - 1) / threads;
  const unsigned blocks = static_cast<unsigned>(std::min<int64_t>(want, 65535));
  build_ptr_table_kernel<<<blocks, threads, 0, current_stream(block_indices)>>>(
      block_indices.data_ptr<int32_t>(), block_indices.stride(0), block_indices.stride(1),
      block_indices.stride(2), block_indices.stride(3), base_k.data_ptr<int64_t>(),
      base_v.data_ptr<int64_t>(), base_k.stride(0), base_k.stride(1),
      static_cast<int>(base_k.size(1)), n_blocks_per_head.data_ptr<int32_t>(),
      ptr_k.data_ptr<int64_t>(), ptr_v.data_ptr<int64_t>(), MG, H, T, batch_stride,
      static_cast<int>(blocks_per_chunk), in_chunk_block_bytes, total);
  OMNIMEM_CHECK_LAUNCH();
  return std::make_tuple(ptr_k, ptr_v);
}

}  // namespace omnimem_nsa
