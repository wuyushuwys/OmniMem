// Shared helpers for the block-KV cache CUDA kernels (block_kv_*.cu).
// Kept separate from common.cuh so the cache extension compiles only what it needs.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <limits>

#include "warp_scan.cuh"

namespace omnimem_block_kv {

using omnimem_cuda::warp_inclusive_scan;

#define OMNIMEM_BLOCK_KV_CHECK_LAUNCH() C10_CUDA_KERNEL_LAUNCH_CHECK()

constexpr int kWarpSize = 32;
// Clear CTA granularity of the gather kernel: 256 int64 table entries per CTA.
constexpr int kClearEntriesPerCta = 256;
// Device dedup (K3) scan tile: 1024 keys per tile, n_tiles <= 2^15 (single-block scan).
constexpr int kDedupTile = 1024;
constexpr int kDedupMaxTiles = 1 << 15;

inline cudaStream_t current_stream(const at::Tensor& t) {
  return at::cuda::getCurrentCUDAStream(t.device().index()).stream();
}

__host__ __device__ __forceinline__ constexpr int64_t cdiv64(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

// CTA-wide bit copy (WIDE: uint64_t words, else uint16_t); src is __restrict__.
template <typename T, int UNROLL>
__device__ __forceinline__ void copy_elems_cta(const T* __restrict__ s, T* __restrict__ d,
                                               int64_t n) {
  const int64_t stride = blockDim.x;
  // Each batch issues up to UNROLL loads before its first store to overlap PCIe round trips.
  for (int64_t i = threadIdx.x; i < n; i += UNROLL * stride) {
    T v[UNROLL] = {};   // value-initialised so a masked-off lane never reads an indeterminate value
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int64_t j = i + u * stride;
      if (j < n) v[u] = s[j];
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int64_t j = i + u * stride;
      if (j < n) d[j] = v[u];
    }
  }
}

template <bool WIDE>
__device__ __forceinline__ void copy_block_cta(int64_t src, int64_t dst, int64_t nbytes) {
  if (WIDE) {
    copy_elems_cta<uint64_t, 8>(reinterpret_cast<const uint64_t*>(src),
                                reinterpret_cast<uint64_t*>(dst), nbytes >> 3);
  } else {
    copy_elems_cta<uint16_t, 16>(reinterpret_cast<const uint16_t*>(src),
                                 reinterpret_cast<uint16_t*>(dst), nbytes >> 1);
  }
}

}  // namespace omnimem_block_kv
