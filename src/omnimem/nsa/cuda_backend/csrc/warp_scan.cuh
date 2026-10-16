// Warp-wide scan shared by the NSA and the block-KV extensions.
#pragma once

#include <cstdint>

namespace omnimem_cuda {

// Inclusive warp scan of `v` over a full, converged warp.
__device__ __forceinline__ int32_t warp_inclusive_scan(int32_t v) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int32_t n = __shfl_up_sync(0xffffffffu, v, d);
    if (lane >= d) v += n;
  }
  return v;
}

}  // namespace omnimem_cuda
