// Inverted (block -> query group) CSR index: histogram, per-(b, h) scan, atomic scatter.
// block_indices [B, MG, H, T] -> sorted_groups [B, H, MG*T], offsets [B, H, NS+1]; bad ids dropped.
// Bucket order is atomic arrival order, so dK/dV sums are reproducible only up to FP32 rounding.
#include "bwd_common.cuh"
#include "common.cuh"
#include "ops.h"

namespace omnimem_nsa {
namespace {

constexpr int kIdxThreads = 256;

// One thread per (group, slot) pair of a (b, h) row.
__device__ __forceinline__ bool pair_bucket(
    const int32_t* __restrict__ ids, int64_t ids_sb, int64_t ids_sg, int64_t ids_sh,
    int64_t ids_st, const int32_t* __restrict__ nbh, int b, int g, int h, int t, int NS,
    int block_size, int chunk_size, bool causal, int group_size, int& bucket) {
  const int32_t bid = ids[static_cast<int64_t>(b) * ids_sb + static_cast<int64_t>(g) * ids_sg +
                          static_cast<int64_t>(h) * ids_sh + static_cast<int64_t>(t) * ids_st];
  if (bid < 0) return false;
  int bound = NS;
  if (nbh != nullptr) {
    const int nb = nbh[h];
    bound = nb < NS ? nb : NS;
  }
  if (bid >= bound) return false;
  if (causal) {
    const int64_t k_chunk = (static_cast<int64_t>(bid) * block_size) / chunk_size;
    const int64_t q_chunk = (static_cast<int64_t>(g) * group_size) / chunk_size;
    if (k_chunk > q_chunk) return false;
  }
  bucket = bid;
  return true;
}

// SCATTER == false counts each bucket into `counts`; true places each group at the cursor in it.
template <bool SCATTER>
__global__ void inv_pass_kernel(
    const int32_t* __restrict__ ids, int64_t ids_sb, int64_t ids_sg, int64_t ids_sh,
    int64_t ids_st, const int32_t* __restrict__ nbh, int32_t* __restrict__ counts,
    int32_t* __restrict__ groups, int64_t groups_row, int MG, int T, int H, int NS,
    int block_size, int chunk_size, bool causal, int group_size) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= MG * T) return;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int g = pair / T;
  const int t = pair - g * T;
  int bucket = 0;
  if (!pair_bucket(ids, ids_sb, ids_sg, ids_sh, ids_st, nbh, b, g, h, t, NS, block_size,
                   chunk_size, causal, group_size, bucket))
    return;
  const int64_t row = static_cast<int64_t>(b) * H + h;
  if (SCATTER) {
    const int pos = atomicAdd(&counts[row * NS + bucket], 1);
    groups[row * groups_row + pos] = g;
  } else {
    atomicAdd(&counts[row * NS + bucket], 1);
  }
}

// Per-(b, h) exclusive scan: wpos gets the scatter cursors, offsets the CSR boundaries.
__global__ void inv_scan_kernel(const int32_t* __restrict__ hist, int32_t* __restrict__ offsets,
                                int32_t* __restrict__ wpos, int NS) {
  constexpr int NW = kIdxThreads / kWarpSize;
  const int row = blockIdx.x;
  const int64_t hbase = static_cast<int64_t>(row) * NS;
  const int64_t obase = static_cast<int64_t>(row) * (NS + 1);
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int wid = threadIdx.x >> 5;

  __shared__ int warp_tot[NW];
  __shared__ int running;
  if (threadIdx.x == 0) {
    offsets[obase] = 0;
    running = 0;
  }
  __syncthreads();

  for (int base = 0; base < NS; base += kIdxThreads) {
    const int i = base + threadIdx.x;
    const int v = (i < NS) ? hist[hbase + i] : 0;
    const int x = warp_inclusive_scan(v);
    if (lane == kWarpSize - 1) warp_tot[wid] = x;
    __syncthreads();
    if (wid == 0) {  // scan the NW warp totals in one warp
      const int t = warp_inclusive_scan((lane < NW) ? warp_tot[lane] : 0);
      if (lane < NW) warp_tot[lane] = t;
    }
    __syncthreads();
    const int excl = running + ((wid == 0) ? 0 : warp_tot[wid - 1]) + (x - v);
    const int tile_tot = warp_tot[NW - 1];
    if (i < NS) {
      wpos[hbase + i] = excl;
      offsets[obase + i + 1] = excl + v;
    }
    __syncthreads();
    if (threadIdx.x == 0) running += tile_tot;
    __syncthreads();
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> build_inverted_index_impl(
    const at::Tensor& block_indices, const at::Tensor& n_blocks_per_head, int64_t num_blocks,
    int64_t block_size, int64_t chunk_size, bool causal, int64_t group_size) {
  TORCH_CHECK(block_indices.is_cuda(), "inverted_index: block_indices must be a CUDA tensor");
  TORCH_CHECK(block_indices.dim() == 4, "inverted_index: block_indices must be [B, MG, H, T]");
  TORCH_CHECK(block_indices.scalar_type() == at::kInt,
              "inverted_index: block_indices must be int32, got ", block_indices.scalar_type());
  TORCH_CHECK(num_blocks >= 0, "inverted_index: num_blocks must be >= 0");
  TORCH_CHECK(block_size > 0, "inverted_index: block_size must be > 0");
  TORCH_CHECK(chunk_size > 0, "inverted_index: chunk_size must be > 0");
  TORCH_CHECK(group_size >= 1, "inverted_index: group_size must be >= 1");

  const int64_t B = block_indices.size(0);
  const int64_t MG = block_indices.size(1);
  const int64_t H = block_indices.size(2);
  const int64_t T = block_indices.size(3);
  const int64_t NS = num_blocks;

  const bool has_bound = n_blocks_per_head.numel() != 0;
  if (has_bound) {
    TORCH_CHECK(n_blocks_per_head.is_cuda() && n_blocks_per_head.dim() == 1 &&
                    n_blocks_per_head.size(0) == H,
                "inverted_index: n_blocks_per_head must be an empty tensor or int32 [H]");
    TORCH_CHECK(n_blocks_per_head.scalar_type() == at::kInt,
                "inverted_index: n_blocks_per_head must be int32");
  }

  // int32 capacity: a row holds at most MG*T entries.
  const int64_t capacity = B * H * MG * T;
  const int64_t row_cap = MG * T;
  constexpr int64_t kI32Max = 2147483647LL;
  TORCH_CHECK(row_cap <= kI32Max, "inverted_index: MG*T = ", row_cap,
              " does not fit in int32 CSR offsets");
  TORCH_CHECK(capacity <= kI32Max, "inverted_index: B*H*MG*T = ", capacity,
              " exceeds the int32 CSR capacity");
  TORCH_CHECK((NS + 1) * B * H <= kI32Max, "inverted_index: offsets tensor too large");

  const at::cuda::CUDAGuard guard(block_indices.device());
  const auto stream = current_stream(block_indices);
  const auto i32 = block_indices.options().dtype(at::kInt);

  at::Tensor groups = at::empty({B, H, row_cap}, i32);
  at::Tensor offsets = at::zeros({B, H, NS + 1}, i32);
  if (NS == 0 || row_cap == 0 || B == 0 || H == 0) return std::make_tuple(groups, offsets);

  at::Tensor hist = at::zeros({B, H, NS}, i32);
  at::Tensor wpos = at::empty({B, H, NS}, i32);

  const int32_t* nbh = has_bound ? n_blocks_per_head.data_ptr<int32_t>() : nullptr;
  const dim3 pair_grid(static_cast<unsigned>(cdiv64(row_cap, kIdxThreads)),
                       static_cast<unsigned>(H), static_cast<unsigned>(B));
  TORCH_CHECK(H <= 65535 && B <= 65535, "inverted_index: B/H exceed the CUDA grid limit");

  auto pass = [&](auto kernel, at::Tensor& counts, int32_t* groups_out) {
    kernel<<<pair_grid, kIdxThreads, 0, stream>>>(
        block_indices.data_ptr<int32_t>(), block_indices.stride(0), block_indices.stride(1),
        block_indices.stride(2), block_indices.stride(3), nbh, counts.data_ptr<int32_t>(),
        groups_out, row_cap, static_cast<int>(MG), static_cast<int>(T), static_cast<int>(H),
        static_cast<int>(NS), static_cast<int>(block_size), static_cast<int>(chunk_size),
        causal, static_cast<int>(group_size));
    OMNIMEM_CHECK_LAUNCH();
  };
  pass(inv_pass_kernel<false>, hist, nullptr);

  inv_scan_kernel<<<static_cast<unsigned>(B * H), kIdxThreads, 0, stream>>>(
      hist.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(), wpos.data_ptr<int32_t>(),
      static_cast<int>(NS));
  OMNIMEM_CHECK_LAUNCH();

  pass(inv_pass_kernel<true>, wpos, groups.data_ptr<int32_t>());

  return std::make_tuple(groups, offsets);
}

}  // namespace omnimem_nsa
