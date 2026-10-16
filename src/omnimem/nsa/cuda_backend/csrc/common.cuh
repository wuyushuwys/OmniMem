// Shared device helpers for the OmniMem NSA CUDA backend.
// Strides are in elements; pointer tables hold int64 byte addresses (0 = skip); accumulate in FP32.
// LSE is natural-log FP32, -inf for rows with no valid key.
#pragma once

// The selection and top-k kernels use TF32 `mma.sync` and `cp.async` unconditionally.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "the OmniMem NSA CUDA backend needs sm_80 or newer (set TORCH_CUDA_ARCH_LIST >= 8.0)"
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include <cstdint>
#include <limits>

#include "warp_scan.cuh"

namespace omnimem_nsa {

using omnimem_cuda::warp_inclusive_scan;

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kWarpSize = 32;

#define OMNIMEM_CHECK_LAUNCH() C10_CUDA_KERNEL_LAUNCH_CHECK()

template <typename T> struct ScalarTraits;

template <> struct ScalarTraits<float> {
  __device__ __forceinline__ static float to_float(float x) { return x; }
  __device__ __forceinline__ static float from_float(float x) { return x; }
};
template <> struct ScalarTraits<c10::Half> {
  __device__ __forceinline__ static float to_float(c10::Half x) {
    return __half2float(*reinterpret_cast<const __half*>(&x));
  }
  __device__ __forceinline__ static c10::Half from_float(float x) {
    __half h = __float2half_rn(x);
    return *reinterpret_cast<c10::Half*>(&h);
  }
};
template <> struct ScalarTraits<c10::BFloat16> {
  __device__ __forceinline__ static float to_float(c10::BFloat16 x) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
  }
  __device__ __forceinline__ static c10::BFloat16 from_float(float x) {
    __nv_bfloat16 h = __float2bfloat16_rn(x);
    return *reinterpret_cast<c10::BFloat16*>(&h);
  }
};

template <typename T>
__device__ __forceinline__ float to_f32(T x) { return ScalarTraits<T>::to_float(x); }
template <typename T>
__device__ __forceinline__ T from_f32(float x) { return ScalarTraits<T>::from_float(x); }

// Dispatch on q dtype: bf16 / fp16 / fp32.
#define OMNIMEM_DISPATCH_FLOAT_TYPES(TYPE, NAME, ...)                                  \
  [&] {                                                                              \
    switch (TYPE) {                                                                  \
      case at::ScalarType::Float: { using scalar_t = float; return __VA_ARGS__(); }  \
      case at::ScalarType::Half: { using scalar_t = c10::Half; return __VA_ARGS__(); } \
      case at::ScalarType::BFloat16: { using scalar_t = c10::BFloat16; return __VA_ARGS__(); } \
      default:                                                                       \
        TORCH_CHECK(false, NAME, ": unsupported dtype ", TYPE,                       \
                    " (expected bfloat16, float16 or float32)");                     \
    }                                                                                \
  }()

__device__ __forceinline__ float fast_exp(float x) { return exp2f(x * kLog2e); }

// exp(x - m) with the convention exp(-inf - -inf) == 0.
__device__ __forceinline__ float safe_exp_diff(float x, float m) {
  return (x == kNegInf) ? 0.f : exp2f((x - m) * kLog2e);
}

template <typename T>
__device__ __forceinline__ T warp_sum(T v) {
#pragma unroll
  for (int o = kWarpSize / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
template <typename T>
__device__ __forceinline__ T warp_max(T v) {
#pragma unroll
  for (int o = kWarpSize / 2; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__host__ __device__ __forceinline__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
__host__ __device__ __forceinline__ constexpr int64_t cdiv64(int64_t a, int64_t b) { return (a + b - 1) / b; }

__host__ inline int next_pow2(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

// [B, M, H, D]-style strided view (element strides).
template <typename T>
struct Strided4 {
  T* ptr;
  int64_t s0, s1, s2, s3;
  __device__ __forceinline__ T* at(int64_t i0, int64_t i1, int64_t i2, int64_t i3 = 0) const {
    return ptr + i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3;
  }
};
template <typename T>
inline Strided4<T> make_strided4(const at::Tensor& t) {
  TORCH_CHECK(t.dim() == 4, "expected a 4-D tensor, got dim=", t.dim());
  return Strided4<T>{t.data_ptr<T>(), t.stride(0), t.stride(1), t.stride(2), t.stride(3)};
}
template <typename T>
inline Strided4<const T> make_strided4_const(const at::Tensor& t) {
  TORCH_CHECK(t.dim() == 4, "expected a 4-D tensor, got dim=", t.dim());
  return Strided4<const T>{t.data_ptr<T>(), t.stride(0), t.stride(1), t.stride(2), t.stride(3)};
}

// [B, H, M]-style view (lse / delta).
template <typename T>
struct Strided3 {
  T* ptr;
  int64_t s0, s1, s2;
  __device__ __forceinline__ T* at(int64_t i0, int64_t i1, int64_t i2) const {
    return ptr + i0 * s0 + i1 * s1 + i2 * s2;
  }
};
template <typename T>
inline Strided3<T> make_strided3(const at::Tensor& t) {
  TORCH_CHECK(t.dim() == 3, "expected a 3-D tensor, got dim=", t.dim());
  return Strided3<T>{t.data_ptr<T>(), t.stride(0), t.stride(1), t.stride(2)};
}

// Resolved KV block: token t is at k + t*stride_n + d*stride_d; token0 is -1 without causal mask.
template <typename T>
struct BlockView {
  const T* k;
  const T* v;
  int64_t stride_n;
  int64_t stride_d;
  int n_valid;
  int64_t token0;
  __device__ __forceinline__ bool valid() const { return k != nullptr; }
};

// Dense accessor: block_indices [B, MG, H, T] int32 (strided), K/V [B, N, H, D].
template <typename T>
struct DenseKV {
  Strided4<const T> k;
  Strided4<const T> v;
  const int32_t* ids;
  int64_t ids_sb, ids_sg, ids_sh, ids_st;
  int N;
  int block_size;
  int chunk_size;
  bool causal;

  __device__ __forceinline__ int32_t id(int b, int g, int h, int t) const {
    return ids[b * ids_sb + g * ids_sg + h * ids_sh + t * ids_st];
  }
  // q_chunk: chunk id of the query group (only used when causal).
  __device__ __forceinline__ BlockView<T> block(int b, int g, int h, int t, int q_chunk) const {
    BlockView<T> bv{nullptr, nullptr, 0, 0, 0, -1};
    const int32_t bid = id(b, g, h, t);
    if (bid < 0) return bv;
    const int64_t token0 = static_cast<int64_t>(bid) * block_size;
    if (token0 >= N) return bv;
    if (causal && (token0 / chunk_size) > q_chunk) return bv;
    bv.k = k.at(b, token0, h, 0);
    bv.v = v.at(b, token0, h, 0);
    bv.stride_n = k.s1;
    bv.stride_d = k.s3;  // V with distinct strides: v_stride_n()/v_stride_d()
    { const int64_t rem = static_cast<int64_t>(N) - token0; bv.n_valid = static_cast<int>(rem < block_size ? rem : block_size); }
    bv.token0 = causal ? token0 : -1;
    return bv;
  }
  __device__ __forceinline__ int64_t v_stride_n() const { return v.s1; }
  __device__ __forceinline__ int64_t v_stride_d() const { return v.s3; }
};

// Pointer accessor: ptr tables [B, MG, H, T] int64 byte addresses.
template <typename T>
struct PtrKV {
  const int64_t* ptr_k;
  const int64_t* ptr_v;
  int64_t sb, sg, sh, st;
  int64_t stride_n;  // elements
  int64_t stride_d;  // elements
  int block_size;

  __device__ __forceinline__ BlockView<T> block(int b, int g, int h, int t, int /*q_chunk*/) const {
    BlockView<T> bv{nullptr, nullptr, stride_n, stride_d, block_size, -1};
    const int64_t off = b * sb + g * sg + h * sh + t * st;
    const int64_t ak = ptr_k[off];
    if (ak == 0) { bv.k = nullptr; return bv; }
    const int64_t av = ptr_v[off];
    if (av == 0) { bv.k = nullptr; return bv; }
    bv.k = reinterpret_cast<const T*>(static_cast<uintptr_t>(ak));
    bv.v = reinterpret_cast<const T*>(static_cast<uintptr_t>(av));
    return bv;
  }
  __device__ __forceinline__ int64_t v_stride_n() const { return stride_n; }
  __device__ __forceinline__ int64_t v_stride_d() const { return stride_d; }
};

// Chunk-causal visibility of key token `tok` for query chunk `q_chunk`.
__device__ __forceinline__ bool chunk_visible(int64_t tok, int chunk_size, int q_chunk) {
  return (tok / chunk_size) <= q_chunk;
}

enum class Precision : int { kTF32 = 0, kIEEE = 1, kTF32x3 = 2 };

inline Precision parse_precision(c10::string_view s) {
  if (s == "tf32") return Precision::kTF32;
  if (s == "ieee") return Precision::kIEEE;
  if (s == "tf32x3") return Precision::kTF32x3;
  TORCH_CHECK(false, "unsupported input_precision '", std::string(s),
              "' (expected 'tf32', 'ieee' or 'tf32x3')");
}

// Returns the current stream for tensor t's device (respects torch.cuda.stream()).
inline cudaStream_t current_stream(const at::Tensor& t) {
  return at::cuda::getCurrentCUDAStream(t.device().index()).stream();
}

// Smallest pitch >= n with pitch % 8 == 4: conflict free for lane-indexed rows and mma A fragments.
__host__ __device__ __forceinline__ int pitch_4mod8(int n) { return ((n + 7) & ~7) + 4; }

// 8-wide dim tiles of a register accumulator for 32 <= D <= 128 (template value).
__host__ __device__ __forceinline__ int dn_tiles(int D) {
  return (D <= 64) ? 8 : (D <= 96) ? 12 : 16;
}

#define OMNIMEM_DISPATCH_DN(D, ...)                                        \
  [&] {                                                                    \
    if ((D) <= 64) { constexpr int DN_MAX = 8; return __VA_ARGS__(); }     \
    if ((D) <= 96) { constexpr int DN_MAX = 12; return __VA_ARGS__(); }    \
    { constexpr int DN_MAX = 16; return __VA_ARGS__(); }                   \
  }()

// TF32 mma.m16n8k8 (sm_80+), operands rounded with cvt.rna.tf32; gid = lane >> 2, t4 = lane & 3:
//   A [16 x 8] : a0=(gid,t4)   a1=(gid+8,t4)   a2=(gid,t4+4)   a3=(gid+8,t4+4)
//   B [ 8 x 8] : b0=(t4,gid)   b1=(t4+4,gid)
//   C [16 x 8] : c0=(gid,2*t4) c1=(gid,2*t4+1) c2=(gid+8,2*t4) c3=(gid+8,2*t4+1)

__device__ __forceinline__ uint32_t f2tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}

// bf16 is the top 16 bits of fp32 and TF32 keeps 19, so bf16 -> TF32 is a shift.
template <typename T>
__device__ __forceinline__ uint32_t to_tf32(T x) {
  return f2tf32(to_f32(x));
}
template <>
__device__ __forceinline__ uint32_t to_tf32<c10::BFloat16>(c10::BFloat16 x) {
  return static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&x)) << 16;
}

__device__ __forceinline__ void mma_m16n8k8_tf32(float (&d)[4], const uint32_t (&a)[4],
                                                 const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Remap an mma C fragment to the A fragment of the next mma, rounded to TF32 (t4 = lane & 3).
__device__ __forceinline__ void c_to_a_tf32(const float (&c)[4], int lane, int t4,
                                            uint32_t (&a)[4]) {
  const int s1 = (lane & ~3) + (t4 >> 1), s2 = s1 + 2;
  const bool odd = (t4 & 1) != 0;
  const float x0 = __shfl_sync(0xffffffffu, c[0], s1);
  const float x1 = __shfl_sync(0xffffffffu, c[1], s1);
  const float y0 = __shfl_sync(0xffffffffu, c[2], s1);
  const float y1 = __shfl_sync(0xffffffffu, c[3], s1);
  const float z0 = __shfl_sync(0xffffffffu, c[0], s2);
  const float z1 = __shfl_sync(0xffffffffu, c[1], s2);
  const float w0 = __shfl_sync(0xffffffffu, c[2], s2);
  const float w1 = __shfl_sync(0xffffffffu, c[3], s2);
  a[0] = f2tf32(odd ? x1 : x0);
  a[1] = f2tf32(odd ? y1 : y0);
  a[2] = f2tf32(odd ? z1 : z0);
  a[3] = f2tf32(odd ? w1 : w0);
}

// cp.async copies global -> shared without registers; src_bytes < 16 zero-fills the rest.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :
               : "r"(s), "l"(src), "r"(src_bytes)
               : "memory");
}
__device__ __forceinline__ void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// Round a shared-memory cursor up to 16 bytes, as cp.async needs aligned source and destination.
template <typename T>
__device__ __forceinline__ T* align16(void* p) {
  return reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(p) + 15u) & ~static_cast<uintptr_t>(15));
}

// Elements per 16-byte cp.async chunk.
template <typename T> __host__ __device__ constexpr int cp_vec() { return 16 / sizeof(T); }

// Tokens of a block a query chunk may see; chunk-causal visibility is a prefix of the block.
template <typename T>
__device__ __forceinline__ int visible_prefix(const BlockView<T>& bx, int q_chunk, int chunk_size) {
  if (bx.token0 < 0) return bx.n_valid;
  const int64_t lim = static_cast<int64_t>(q_chunk + 1) * chunk_size - bx.token0;
  if (lim <= 0) return 0;
  return (lim < bx.n_valid) ? static_cast<int>(lim) : bx.n_valid;
}

// Advances t to the first block in [t, Tb) with visible tokens (bv / vis describe it).
template <typename T, typename ACC>
__device__ __forceinline__ bool find_visible(const ACC& acc, int b, int g, int h, int Tb,
                                             int q_chunk, int chunk_size, int& t,
                                             BlockView<T>& bv, int& vis) {
  for (; t < Tb; ++t) {
    bv = acc.block(b, g, h, t, q_chunk);
    vis = bv.valid() ? visible_prefix(bv, q_chunk, chunk_size) : 0;
    if (vis > 0) return true;
  }
  return false;
}

// This thread's (row, chunk) in a CTA-wide copy of `cpr` chunks per row, decoded once.
struct RowChunk {
  int rpp;     // rows per pass; 0 when a row has more chunks than the CTA has threads
  int row;     // first row of this thread (`tile` for idle tail threads)
  int chunk;
};
__device__ __forceinline__ RowChunk row_chunk(int cpr, int tile) {
  const int rpp = (cpr > 0 && cpr <= static_cast<int>(blockDim.x))
                      ? static_cast<int>(blockDim.x) / cpr : 0;
  const int crow = (rpp > 0) ? static_cast<int>(threadIdx.x) / cpr : 0;
  const int cchunk = (rpp > 0) ? static_cast<int>(threadIdx.x) - crow * cpr : 0;
  return RowChunk{rpp, (rpp > 0 && crow < rpp) ? crow : tile, cchunk};
}

// Query group served by slot gs: an optional argsort of the first selected block, for L2 reuse.
__device__ __forceinline__ int group_of(const int32_t* __restrict__ order, int64_t s0, int64_t s1,
                                        int64_t s2, int b, int gs, int h) {
  return (order == nullptr) ? gs
                            : order[static_cast<int64_t>(b) * s0 + static_cast<int64_t>(gs) * s1 +
                                    static_cast<int64_t>(h) * s2];
}
// group_of(blockIdx.x); reading blockIdx.x in each branch keeps the forward kernels' codegen.
__device__ __forceinline__ int cta_group(const int32_t* __restrict__ order, int64_t s0, int64_t s1,
                                         int64_t s2, int b, int h) {
  return (order == nullptr) ? static_cast<int>(blockIdx.x)
                            : order[b * s0 + static_cast<int64_t>(blockIdx.x) * s1 + h * s2];
}
// Host side of group_of's (order, s0, s1, s2) kernel arguments.
struct GroupOrder {
  const int32_t* p;
  int64_t s0, s1, s2;
};
inline GroupOrder make_group_order(const at::Tensor& order) {
  if (!order.defined()) return GroupOrder{nullptr, 0, 0, 0};
  return GroupOrder{order.data_ptr<int32_t>(), order.stride(0), order.stride(1), order.stride(2)};
}

}  // namespace omnimem_nsa
