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

namespace omnimem_nsa {

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

// floor(a / b) for possibly negative a (b > 0).
__device__ __forceinline__ int floordiv(int a, int b) {
  int q = a / b;
  return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
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

}  // namespace omnimem_nsa
