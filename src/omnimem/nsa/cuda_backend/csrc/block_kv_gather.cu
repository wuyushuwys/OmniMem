// K1: byte-moving gather over raw int64 addresses, bit-exact, then publish and clear.
// Caller guarantees disjoint dst ranges and publish/clear addresses; wide needs 8-byte alignment.
// __syncthreads orders data stores before the publish stores.
#include "block_kv_common.cuh"
#include "block_kv_ops.h"

namespace omnimem_block_kv {
namespace {

// K1: n_fetch gather CTAs, then ceil(n_clear / 256) clear CTAs.
// desc is SoA, stride n_fetch: [src_k|src_v|dst_k|dst_v|pub_k|pub_v] x N, then clear x M
// desc has no __restrict__: the kernel stores through addresses read out of it.
template <bool WIDE>
__global__ void gather_publish_clear_kernel(const int64_t* desc, int64_t n_fetch,
                                            int64_t n_clear, int64_t block_bytes) {
  const int64_t pid = static_cast<int64_t>(blockIdx.x);
  if (pid < n_fetch) {
    const int64_t src_k = desc[0 * n_fetch + pid];
    const int64_t src_v = desc[1 * n_fetch + pid];
    const int64_t dst_k = desc[2 * n_fetch + pid];
    const int64_t dst_v = desc[3 * n_fetch + pid];
    const int64_t pub_k = desc[4 * n_fetch + pid];
    const int64_t pub_v = desc[5 * n_fetch + pid];
    copy_block_cta<WIDE>(src_k, dst_k, block_bytes);
    copy_block_cta<WIDE>(src_v, dst_v, block_bytes);
    // Publish only after every lane has written its share of K and V.
    __syncthreads();
    if (threadIdx.x == 0) {
      *reinterpret_cast<int64_t*>(pub_k) = dst_k;
      *reinterpret_cast<int64_t*>(pub_v) = dst_v;
    }
  } else {
    // Clear CTA cb zeroes entries [cb*256, cb*256+256); lanes re-check n_clear.
    const int64_t cb = pid - n_fetch;
    const int64_t base = cb * static_cast<int64_t>(kClearEntriesPerCta);
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < kClearEntriesPerCta;
         i += static_cast<int64_t>(blockDim.x)) {
      const int64_t idx = base + i;
      if (idx < n_clear) {
        const int64_t addr = desc[6 * n_fetch + idx];
        *reinterpret_cast<int64_t*>(addr) = 0;
      }
    }
  }
}

inline int block_threads(int64_t num_warps) {
  TORCH_CHECK(num_warps >= 1 && num_warps <= 32,
              "block-KV gather: num_warps=", num_warps, " outside [1, 32]");
  return static_cast<int>(num_warps) * kWarpSize;
}

inline void check_addr_tensor(const at::Tensor& t, const char* name, int64_t need) {
  TORCH_CHECK(t.is_cuda(), "block-KV gather: ", name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kLong, "block-KV gather: ", name,
              " must be int64, got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), "block-KV gather: ", name, " must be contiguous");
  TORCH_CHECK(t.numel() >= need, "block-KV gather: ", name, " holds ", t.numel(),
              " elements, need ", need);
}

inline void check_block_bytes(int64_t block_bytes, bool wide) {
  TORCH_CHECK(block_bytes > 0 && block_bytes % 2 == 0,
              "block-KV gather: block_bytes=", block_bytes,
              " must be a positive multiple of 2");
  TORCH_CHECK(!wide || block_bytes % 8 == 0,
              "block-KV gather: wide copy needs block_bytes (", block_bytes,
              ") to be a multiple of 8");
}

inline unsigned int check_grid_x(int64_t ctas, const char* what) {
  TORCH_CHECK(ctas <= static_cast<int64_t>(2147483647),
              "block-KV gather: ", what, " needs ", ctas,
              " CTAs, which exceeds the gridDim.x limit");
  return static_cast<unsigned int>(ctas);
}

}  // namespace

int64_t gather_publish_clear(const at::Tensor& desc, int64_t n_fetch, int64_t n_clear,
                             int64_t block_bytes, bool wide, int64_t num_warps) {
  // Nothing to do: launch nothing and skip validating the unused descriptor.
  if (n_fetch == 0 && n_clear == 0) return 0;
  TORCH_CHECK(n_fetch >= 0 && n_clear >= 0,
              "block-KV K1: n_fetch=", n_fetch, " / n_clear=", n_clear, " must be >= 0");
  check_addr_tensor(desc, "desc", 6 * n_fetch + n_clear);
  check_block_bytes(block_bytes, wide);
  const int threads = block_threads(num_warps);

  const int64_t ctas = n_fetch + cdiv64(n_clear, kClearEntriesPerCta);
  const dim3 grid(check_grid_x(ctas, "K1"));
  const dim3 block(static_cast<unsigned int>(threads));

  at::cuda::CUDAGuard guard(desc.device());
  const cudaStream_t stream = current_stream(desc);
  const int64_t* desc_ptr = desc.data_ptr<int64_t>();
  if (wide) {
    gather_publish_clear_kernel<true><<<grid, block, 0, stream>>>(desc_ptr, n_fetch,
                                                                  n_clear, block_bytes);
  } else {
    gather_publish_clear_kernel<false><<<grid, block, 0, stream>>>(desc_ptr, n_fetch,
                                                                   n_clear, block_bytes);
  }
  OMNIMEM_BLOCK_KV_CHECK_LAUNCH();
  return ctas;
}

}  // namespace omnimem_block_kv
