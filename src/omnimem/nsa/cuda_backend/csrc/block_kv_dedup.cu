// K3: device dedup of selected (head, bid) entries into strictly ascending host-order keys.
// key = (cid * heads + head) * bpc + loc with cid = bid / bpc, loc = bid % bpc.
// out[0] is the true unique total even when it exceeds cap; only the first cap keys are written.
#include "block_kv_common.cuh"
#include "block_kv_ops.h"

#include <algorithm>

namespace omnimem_block_kv {
namespace {

constexpr int kMarkBlock = 256;
constexpr int kMarkMaxCtas = 8192;

// Duplicates store the same value, so atomicExch races are benign; live iff stamp[key] == epoch.
__global__ void mark_epoch_kernel(const int32_t* __restrict__ block_indices,
                                  int32_t* __restrict__ stamp, int32_t n_entries,
                                  int32_t epoch, int32_t mb, int32_t heads, int32_t t,
                                  int32_t bpc) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       e < static_cast<int64_t>(n_entries); e += stride) {
    const int32_t bid = block_indices[e];
    if (bid < 0 || bid >= mb) continue;  // -1 pad / out-of-range: never produces a key
    // head of flat entry e is (e / T) % H; e fits 32 bits, so unsigned 32-bit division is exact.
    const uint32_t e32 = static_cast<uint32_t>(e);
    const int32_t h = static_cast<int32_t>((e32 / static_cast<uint32_t>(t)) %
                                           static_cast<uint32_t>(heads));
    const int32_t cid = bid / bpc;
    const int32_t loc = bid - cid * bpc;
    // the facade guarantees heads * mb <= INT32_MAX, so key is a valid int32 table index.
    const int64_t key = (static_cast<int64_t>(cid) * heads + h) * bpc + loc;
    atomicExch(stamp + static_cast<int32_t>(key), epoch);
  }
}

// Count/scan/compact assume exactly 32 warps per tile (kWarpSize-entry shared array).
static_assert(kDedupTile == kWarpSize * kWarpSize, "K3 tile must be 32 warps x 32 lanes");

// Per-tile hit count, one key per thread.
__global__ void count_epoch_kernel(const int32_t* __restrict__ stamp,
                                   int32_t* __restrict__ counts, int32_t n_keys,
                                   int32_t epoch) {
  const int32_t key =
      static_cast<int32_t>(blockIdx.x) * kDedupTile + static_cast<int32_t>(threadIdx.x);
  const bool hit = (key < n_keys) && (stamp[key] == epoch);
  // Every thread of the CTA reaches this barrier (no early return above).
  const int32_t n_hit = __syncthreads_count(hit);
  if (threadIdx.x == 0) counts[blockIdx.x] = n_hit;
}

// Inclusive warp scan of `v` over a full, converged warp.
__device__ __forceinline__ int32_t warp_inclusive_scan(int32_t v) {
  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
#pragma unroll
  for (int d = 1; d < kWarpSize; d <<= 1) {
    const int32_t n = __shfl_up_sync(0xffffffffu, v, d);
    if (lane >= d) v += n;
  }
  return v;
}

// Single-CTA exclusive scan of the tile counts; total goes to out[0].
__global__ void scan_counts_kernel(const int32_t* __restrict__ counts,
                                   int32_t* __restrict__ offsets, int32_t* __restrict__ out,
                                   int32_t n_tiles, int32_t per_thread) {
  __shared__ int32_t s_warp[kWarpSize];
  __shared__ int32_t s_total;

  const int32_t begin = static_cast<int32_t>(threadIdx.x) * per_thread;
  const int32_t end = min(begin + per_thread, n_tiles);

  int32_t local = 0;
  for (int32_t j = begin; j < end; ++j) local += counts[j];

  // Block-wide exclusive scan of the 1024 per-thread sums.
  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int32_t incl = warp_inclusive_scan(local);
  if (lane == kWarpSize - 1) s_warp[warp] = incl;
  __syncthreads();
  if (threadIdx.x < kWarpSize) {
    const int32_t wv = s_warp[threadIdx.x];
    const int32_t wi = warp_inclusive_scan(wv);
    s_warp[threadIdx.x] = wi - wv;  // exclusive over the warp totals
  }
  __syncthreads();
  const int32_t excl = incl - local + s_warp[warp];

  if (threadIdx.x == blockDim.x - 1) s_total = excl + local;

  // Scatter the exclusive offsets of this thread's run.
  int32_t running = excl;
  for (int32_t j = begin; j < end; ++j) {
    offsets[j] = running;
    running += counts[j];
  }

  __syncthreads();
  // out[0] is the TRUE unique total, even when it exceeds cap.
  if (threadIdx.x == 0) out[0] = s_total;
}

// pos = offsets[tile] + in-tile rank gives ascending keys with no atomic append.
__global__ void compact_epoch_kernel(const int32_t* __restrict__ stamp,
                                     const int32_t* __restrict__ offsets,
                                     int32_t* __restrict__ out, int32_t n_keys, int64_t cap,
                                     int32_t epoch) {
  __shared__ int32_t s_warp[kWarpSize];

  const int32_t key =
      static_cast<int32_t>(blockIdx.x) * kDedupTile + static_cast<int32_t>(threadIdx.x);
  const bool hit = (key < n_keys) && (stamp[key] == epoch);

  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  // Full-warp ballot: all 32 lanes of every warp reach this point (blockDim == kDedupTile).
  const uint32_t ballot = __ballot_sync(0xffffffffu, hit);
  const int32_t rank_in_warp = __popc(ballot & ((1u << lane) - 1u));
  if (lane == 0) s_warp[warp] = __popc(ballot);
  __syncthreads();
  if (threadIdx.x < kWarpSize) {
    const int32_t wv = s_warp[threadIdx.x];
    const int32_t wi = warp_inclusive_scan(wv);
    s_warp[threadIdx.x] = wi - wv;  // exclusive over the 32 warp hit counts
  }
  __syncthreads();

  const int64_t pos = static_cast<int64_t>(offsets[blockIdx.x]) + s_warp[warp] + rank_in_warp;
  if (hit && pos < cap) out[1 + pos] = key;
}

inline void check_i32_cuda(const at::Tensor& t, const char* name, const at::Tensor& ref) {
  TORCH_CHECK(t.is_cuda(), "omnimem_block_kv::dedup_run: ", name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kInt, "omnimem_block_kv::dedup_run: ", name,
              " must be int32, got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), "omnimem_block_kv::dedup_run: ", name,
              " must be contiguous");
  TORCH_CHECK(t.device() == ref.device(), "omnimem_block_kv::dedup_run: ", name, " on ",
              t.device(), ", stamp on ", ref.device());
}

}  // namespace

int64_t dedup_run(const at::Tensor& block_indices, at::Tensor& stamp, at::Tensor& counts,
                  at::Tensor& offsets, at::Tensor& out, int64_t epoch, int64_t mb,
                  int64_t heads, int64_t t, int64_t bpc, int64_t cap) {
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();

  check_i32_cuda(stamp, "stamp", stamp);
  check_i32_cuda(block_indices, "block_indices", stamp);
  check_i32_cuda(counts, "counts", stamp);
  check_i32_cuda(offsets, "offsets", stamp);
  check_i32_cuda(out, "out", stamp);

  TORCH_CHECK(mb > 0 && heads > 0 && t > 0 && bpc > 0 && mb % bpc == 0,
              "omnimem_block_kv::dedup_run: need positive mb/heads/t/bpc with bpc | mb, got "
              "mb=", mb, " heads=", heads, " t=", t, " bpc=", bpc);
  const int64_t n_keys = heads * mb;
  TORCH_CHECK(n_keys <= kInt32Max,
              "omnimem_block_kv::dedup_run: n_keys=", n_keys, " does not fit int32");
  TORCH_CHECK(stamp.numel() == n_keys,
              "omnimem_block_kv::dedup_run: stamp holds ", stamp.numel(),
              " entries, need n_keys=", n_keys);

  const int64_t n_tiles = cdiv64(n_keys, kDedupTile);
  TORCH_CHECK(n_tiles <= kDedupMaxTiles,
              "omnimem_block_kv::dedup_run: n_tiles=", n_tiles,
              " exceeds the single-block scan limit ", static_cast<int64_t>(kDedupMaxTiles));
  TORCH_CHECK(counts.numel() >= n_tiles && offsets.numel() >= n_tiles,
              "omnimem_block_kv::dedup_run: counts/offsets hold ", counts.numel(), "/",
              offsets.numel(), " entries, need n_tiles=", n_tiles);

  TORCH_CHECK(cap >= 0 && out.numel() >= 1 + cap,
              "omnimem_block_kv::dedup_run: out holds ", out.numel(),
              " entries, need 1 + cap = ", 1 + cap);
  TORCH_CHECK(epoch >= 1 && epoch <= kInt32Max,
              "omnimem_block_kv::dedup_run: epoch=", epoch, " outside [1, INT32_MAX]");

  const int64_t n_entries = block_indices.numel();
  TORCH_CHECK(n_entries <= kInt32Max,
              "omnimem_block_kv::dedup_run: n_entries=", n_entries, " does not fit int32");

  const at::cuda::CUDAGuard guard(stamp.device());
  const cudaStream_t stream = current_stream(stamp);

  const int32_t epoch_i32 = static_cast<int32_t>(epoch);
  const int32_t n_keys_i32 = static_cast<int32_t>(n_keys);
  const int32_t n_tiles_i32 = static_cast<int32_t>(n_tiles);
  int32_t* stamp_ptr = stamp.data_ptr<int32_t>();

  // Grid-strided; a 1-CTA launch keeps the launch count at 4 when n_entries == 0.
  const int mark_ctas = static_cast<int>(
      std::min<int64_t>(std::max<int64_t>(cdiv64(n_entries, kMarkBlock), 1), kMarkMaxCtas));
  mark_epoch_kernel<<<mark_ctas, kMarkBlock, 0, stream>>>(
      block_indices.data_ptr<int32_t>(), stamp_ptr, static_cast<int32_t>(n_entries),
      epoch_i32, static_cast<int32_t>(mb), static_cast<int32_t>(heads),
      static_cast<int32_t>(t), static_cast<int32_t>(bpc));
  OMNIMEM_BLOCK_KV_CHECK_LAUNCH();

  count_epoch_kernel<<<n_tiles_i32, kDedupTile, 0, stream>>>(
      stamp_ptr, counts.data_ptr<int32_t>(), n_keys_i32, epoch_i32);
  OMNIMEM_BLOCK_KV_CHECK_LAUNCH();

  const int32_t per_thread = static_cast<int32_t>(cdiv64(n_tiles, kDedupTile));
  scan_counts_kernel<<<1, kDedupTile, 0, stream>>>(
      counts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(), out.data_ptr<int32_t>(),
      n_tiles_i32, per_thread);
  OMNIMEM_BLOCK_KV_CHECK_LAUNCH();

  compact_epoch_kernel<<<n_tiles_i32, kDedupTile, 0, stream>>>(
      stamp_ptr, offsets.data_ptr<int32_t>(), out.data_ptr<int32_t>(), n_keys_i32, cap,
      epoch_i32);
  OMNIMEM_BLOCK_KV_CHECK_LAUNCH();

  return 4;
}

}  // namespace omnimem_block_kv
