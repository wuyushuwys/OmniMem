// Selection attention forward (dense + pointer KV); launch_sel_fwd picks one kernel per call:
// TF32 mma (G > 1, 32 <= D <= 128), FP32 SIMT group kernel (other G > 1), token kernel (G == 1).
// Softmax state and accumulators are FP32; rows with no visible token give O = 0 and LSE = -inf.
#include "common.cuh"
#include "ops.h"

#include <algorithm>


namespace omnimem_nsa {
namespace {

constexpr int kTileN = 32;            // KV tokens per tile == warp size
constexpr int kQPW = 4;               // queries per warp: amortises the shared K/V tile reads
constexpr int kMaxWarpsGroup = 16;    // <= 512 threads/CTA, Gc <= kQPW*kMaxWarpsGroup
constexpr int kTokenWarps = 8;        // warps (== queries) per CTA in the G == 1 kernel
constexpr size_t kSmemBudget = 96 * 1024;

__host__ __device__ __forceinline__ int round4(int x) { return (x + 3) & ~3; }
__host__ __device__ __forceinline__ int round8(int x) { return (x + 7) & ~7; }

// The K tile is indexed by lane, so it is bank-conflict free iff pitch_in_elements % 8 == 4.
__host__ __device__ __forceinline__ int ldk_for(int D) { return round8(D) + 4; }

// Number of 8-wide dim tiles the TF32 path carries in the O accumulator (template value).
__host__ __device__ __forceinline__ int dn_for(int D) {
  if (D < 32 || D > 128) return (D + 7) >> 3;          // SIMT-only range
  return (D <= 64) ? 8 : (D <= 96) ? 12 : 16;
}
// V pitch: the TF32 path reads mma B fragments, which is conflict-free iff pitch % 32 == 8.
__host__ __device__ __forceinline__ int ldv_for(int D) {
  int ld = dn_for(D) * 8;
  if (ld < round8(D)) ld = round8(D);
  while ((ld & 31) != 8) ld += 8;
  return ld;
}

// K/V tiles are staged in the input dtype, which halves shared-memory footprint and bandwidth.
template <typename T>
__device__ __forceinline__ float4 load4_smem(const T* p);

template <>
__device__ __forceinline__ float4 load4_smem<float>(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
template <>
__device__ __forceinline__ float4 load4_smem<c10::BFloat16>(const c10::BFloat16* p) {
  const uint2 r = *reinterpret_cast<const uint2*>(p);
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.y));
  return make_float4(a.x, a.y, b.x, b.y);
}
template <>
__device__ __forceinline__ float4 load4_smem<c10::Half>(const c10::Half* p) {
  const uint2 r = *reinterpret_cast<const uint2*>(p);
  const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&r.x));
  const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&r.y));
  return make_float4(a.x, a.y, b.x, b.y);
}

template <typename T, int N>
__device__ __forceinline__ void loadN_smem(const T* p, float (&dst)[N]) {
  if (N % 4 == 0) {
#pragma unroll
    for (int i = 0; i < N; i += 4) {
      const float4 f = load4_smem<T>(p + i);
      dst[i] = f.x; dst[i + 1] = f.y; dst[i + 2] = f.z; dst[i + 3] = f.w;
    }
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) dst[i] = to_f32<T>(p[i]);
  }
}

// Read N contiguous elements as floats; p must be aligned to load_vecN_align<T, N>() bytes.
template <typename T, int N>
__host__ __device__ constexpr int load_vecN_align() {
  return (N * sizeof(T) == 16 || N * sizeof(T) == 8 || N * sizeof(T) == 4)
             ? static_cast<int>(N * sizeof(T))
             : static_cast<int>(sizeof(T));
}
template <typename T, int N>
__device__ __forceinline__ void load_vecN(const T* __restrict__ p, float (&dst)[N]) {
  constexpr int VB = N * static_cast<int>(sizeof(T));
  if (VB == 16) {
    const uint4 r = *reinterpret_cast<const uint4*>(p);
    const T* e = reinterpret_cast<const T*>(&r);
#pragma unroll
    for (int i = 0; i < N; ++i) dst[i] = to_f32<T>(e[i]);
  } else if (VB == 8) {
    const uint2 r = *reinterpret_cast<const uint2*>(p);
    const T* e = reinterpret_cast<const T*>(&r);
#pragma unroll
    for (int i = 0; i < N; ++i) dst[i] = to_f32<T>(e[i]);
  } else if (VB == 4) {
    const unsigned r = *reinterpret_cast<const unsigned*>(p);
    const T* e = reinterpret_cast<const T*>(&r);
#pragma unroll
    for (int i = 0; i < N; ++i) dst[i] = to_f32<T>(e[i]);
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) dst[i] = to_f32<T>(p[i]);
  }
}

// Widest legal vector copy in bytes for a [rows, D] region into pitch ld; 0 = scalar.
template <typename T>
__device__ __forceinline__ int vec_width(const T* base, int64_t sn, int64_t sd, int D, int ld) {
  if (sd != 1) return 0;
  constexpr size_t es = sizeof(T);
  const uintptr_t a = reinterpret_cast<uintptr_t>(base);
  const size_t rowb = static_cast<size_t>(D) * es;
  const size_t pitchb = static_cast<size_t>(ld) * es;
  const size_t stepb = static_cast<size_t>(sn) * es;
  if ((a % 16 == 0) && (rowb % 16 == 0) && (pitchb % 16 == 0) && (stepb % 16 == 0)) return 16;
  if ((a % 8 == 0) && (rowb % 8 == 0) && (pitchb % 8 == 0) && (stepb % 8 == 0)) return 8;
  return 0;
}

// Copy one KV row into a shared row of pitch ld, zero-filling the pitch tail.
template <typename T>
__device__ __forceinline__ void copy_row(const T* __restrict__ src, T* __restrict__ dst, int D,
                                         int ld, int64_t sd, int vw, int lane) {
  if (vw == 16) {
    const int n = static_cast<int>((static_cast<size_t>(D) * sizeof(T)) >> 4);
    for (int j = lane; j < n; j += 32)
      reinterpret_cast<uint4*>(dst)[j] = reinterpret_cast<const uint4*>(src)[j];
  } else if (vw == 8) {
    const int n = static_cast<int>((static_cast<size_t>(D) * sizeof(T)) >> 3);
    for (int j = lane; j < n; j += 32)
      reinterpret_cast<uint2*>(dst)[j] = reinterpret_cast<const uint2*>(src)[j];
  } else if (sd == 1) {
    for (int d = lane; d < D; d += 32) dst[d] = src[d];
  } else {
    for (int d = lane; d < D; d += 32) dst[d] = src[static_cast<int64_t>(d) * sd];
  }
  const T zero = from_f32<T>(0.f);
  for (int d = D + lane; d < ld; d += 32) dst[d] = zero;
}

template <typename T>
__device__ __forceinline__ void zero_row(T* __restrict__ dst, int ld, int lane) {
  const T zero = from_f32<T>(0.f);
  for (int d = lane; d < ld; d += 32) dst[d] = zero;
}

// Shared-memory plan: [sQ fp32][sP fp32] | [sK T][sV T] (T region 16B aligned); sP by (warp, slot).
__host__ __device__ __forceinline__ int p_rows(int Gc) {
  return ((Gc + kQPW - 1) / kQPW) * kQPW;
}
__host__ __device__ __forceinline__ size_t group_smem_f32_elems(int Gc, int D) {
  return static_cast<size_t>(Gc) * round4(D) + static_cast<size_t>(p_rows(Gc)) * kTileN;
}
__host__ __device__ __forceinline__ size_t group_smem_bytes(int Gc, int D, size_t esize) {
  const size_t f32b = (group_smem_f32_elems(Gc, D) * 4 + 15) & ~static_cast<size_t>(15);
  return f32b + esize * static_cast<size_t>(kTileN) * (ldk_for(D) + ldv_for(D));
}

// Vector width (bytes) for reading a contiguous [.., D] row straight from global memory.
template <typename T>
__device__ __forceinline__ int gvec_width(const T* base, int64_t sn, int64_t sd, int D, int unit) {
  if (sd != 1) return 0;
  constexpr size_t es = sizeof(T);
  const uintptr_t a = reinterpret_cast<uintptr_t>(base);
  const size_t rowb = static_cast<size_t>(D) * es;
  const size_t stepb = static_cast<size_t>(sn) * es;
  const size_t unitb = static_cast<size_t>(unit) * es;   // granularity actually consumed
  if ((a % 16 == 0) && (rowb % 16 == 0) && (stepb % 16 == 0) && (unitb % 16 == 0)) return 16;
  if ((a % 8 == 0) && (rowb % 8 == 0) && (stepb % 8 == 0) && (unitb % 8 == 0)) return 8;
  return 0;
}

// G > 1: one CTA per (b, g, h) with shared-memory staged K/V tiles.
template <typename T, typename ACC, int DPL>
__global__ void __launch_bounds__(512) sel_fwd_group_kernel(
    const Strided4<const T> q, const Strided4<T> out, const Strided3<float> lse, const ACC kv,
    const int H, const int D, const int G, const int Gc, const int Tb, const int chunk_size,
    const float scale, const int32_t* __restrict__ order, const int64_t os0,
    const int64_t os1, const int64_t os2) {
  extern __shared__ __align__(16) char smem_raw[];
  const int Dv = round4(D);
  const int ldk = ldk_for(D);
  const int ldv = ldv_for(D);
  const int ldq = Dv;
  float* sQ = reinterpret_cast<float*>(smem_raw);             // [Gc][ldq]      fp32
  float* sP = sQ + static_cast<size_t>(Gc) * ldq;             // [p_rows(Gc)][kTileN] fp32
  T* sK = reinterpret_cast<T*>(smem_raw + ((group_smem_f32_elems(Gc, D) * 4 + 15) & ~size_t(15)));
  T* sV = sK + static_cast<size_t>(kTileN) * ldk;             // [kTileN][ldv]

  const int bh = blockIdx.z;
  const int b = bh / H;
  const int h = bh - b * H;
  // Visit groups in ascending first-selected-block order so concurrent CTAs hit in L2.
  const int g = (order == nullptr)
                    ? static_cast<int>(blockIdx.x)
                    : order[b * os0 + static_cast<int64_t>(blockIdx.x) * os1 + h * os2];
  const int q0 = blockIdx.y * Gc;
  const int nq = ::min(Gc, G - q0);
  if (nq <= 0) return;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nw = blockDim.x >> 5;
  const int64_t m_base = static_cast<int64_t>(g) * G + q0;

  // stage Q (fp32, read once)
  {
    const int64_t qsd = q.s3;
    for (int i = warp; i < nq; i += nw) {
      const T* qp = q.at(b, m_base + i, h, 0);
      float* dst = sQ + static_cast<size_t>(i) * ldq;
      if (qsd == 1) {
        for (int d = lane; d < D; d += 32) dst[d] = to_f32<T>(qp[d]);
      } else {
        for (int d = lane; d < D; d += 32) dst[d] = to_f32<T>(qp[static_cast<int64_t>(d) * qsd]);
      }
      for (int d = D + lane; d < ldq; d += 32) dst[d] = 0.f;
    }
  }

  int myq[kQPW], qrow[kQPW];
  float m_i[kQPW], l_i[kQPW], av[kQPW][DPL];
#pragma unroll
  for (int qi = 0; qi < kQPW; ++qi) {
    myq[qi] = warp + qi * nw;
    // Inactive slots read row 0 so the hot loops stay branch free; their result is dropped.
    qrow[qi] = (myq[qi] < nq) ? myq[qi] : 0;
    m_i[qi] = kNegInf;
    l_i[qi] = 0.f;
#pragma unroll
    for (int i = 0; i < DPL; ++i) av[qi][i] = 0.f;
  }

  const int d_lo = lane * DPL;
  const bool d_active = d_lo < D;
  const bool d_full = (d_lo + DPL) <= D;
  const int q_chunk = static_cast<int>((static_cast<int64_t>(g) * G) / chunk_size);
  __syncthreads();

  for (int t = 0; t < Tb; ++t) {
    const BlockView<T> bv = kv.block(b, g, h, t, q_chunk);
    if (!bv.valid()) continue;
    const int nvalid = bv.n_valid;
    const int64_t vsn = kv.v_stride_n();
    const int64_t vsd = kv.v_stride_d();
    const int kvw = vec_width<T>(bv.k, bv.stride_n, bv.stride_d, D, ldk);
    const int vvw = vec_width<T>(bv.v, vsn, vsd, D, ldv);

    for (int tok0 = 0; tok0 < nvalid; tok0 += kTileN) {
      const int ntile = ::min(kTileN, nvalid - tok0);

      // stage K / V tiles (one row per warp)
      for (int r = warp; r < kTileN; r += nw) {
        T* dk = sK + static_cast<size_t>(r) * ldk;
        T* dv = sV + static_cast<size_t>(r) * ldv;
        if (r < ntile) {
          const int64_t tr = static_cast<int64_t>(tok0 + r);
          copy_row<T>(bv.k + tr * bv.stride_n, dk, D, ldk, bv.stride_d, kvw, lane);
          copy_row<T>(bv.v + tr * vsn, dv, D, ldv, vsd, vvw, lane);
        } else {
          zero_row<T>(dk, ldk, lane);
          zero_row<T>(dv, ldv, lane);
        }
      }
      __syncthreads();

      // QK: lane == token
      bool ok = lane < ntile;
      if (ok && bv.token0 >= 0) ok = ((bv.token0 + tok0 + lane) / chunk_size) <= q_chunk;
      float sc[kQPW];
      {
        const T* krow = sK + static_cast<size_t>(lane) * ldk;
#pragma unroll
        for (int qi = 0; qi < kQPW; ++qi) sc[qi] = 0.f;
#pragma unroll 4
        for (int d = 0; d < Dv; d += 4) {
          const float4 kk = load4_smem<T>(krow + d);
#pragma unroll
          for (int qi = 0; qi < kQPW; ++qi) {
            const float4 qq =
                *reinterpret_cast<const float4*>(sQ + static_cast<size_t>(qrow[qi]) * ldq + d);
            sc[qi] += qq.x * kk.x + qq.y * kk.y + qq.z * kk.z + qq.w * kk.w;
          }
        }
      }

      // online softmax (FP32 max / sum / accumulator)
#pragma unroll
      for (int qi = 0; qi < kQPW; ++qi) {
        const int mq = myq[qi];
        const float s = (ok && mq < nq) ? sc[qi] * scale : kNegInf;
        const float tmax = warp_max(s);
        float p = 0.f;
        if (mq < nq && tmax > kNegInf) {
          const float mnew = fmaxf(m_i[qi], tmax);
          const float corr = (m_i[qi] == kNegInf) ? 0.f : exp2f((m_i[qi] - mnew) * kLog2e);
          p = safe_exp_diff(s, mnew);
          const float ssum = warp_sum(p);
          l_i[qi] = l_i[qi] * corr + ssum;
#pragma unroll
          for (int i = 0; i < DPL; ++i) av[qi][i] *= corr;
          m_i[qi] = mnew;
        }
        sP[static_cast<size_t>(mq) * kTileN + lane] = p;   // mq < p_rows(Gc) always
      }
      __syncwarp();

      // PV: lane == dim slice, P (kQPW x 32) times V (32 x D)
      if (d_active) {
        const int ntile4 = (ntile + 3) & ~3;
        for (int tk0 = 0; tk0 < ntile4; tk0 += 4) {
          float4 p4[kQPW];
#pragma unroll
          for (int qi = 0; qi < kQPW; ++qi)
            p4[qi] = *reinterpret_cast<const float4*>(sP + static_cast<size_t>(myq[qi]) * kTileN + tk0);
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const T* vrow = sV + static_cast<size_t>(tk0 + u) * ldv + d_lo;
            float vv[DPL];
            if (d_full) {
              loadN_smem<T, DPL>(vrow, vv);
            } else {
#pragma unroll
              for (int i = 0; i < DPL; ++i) vv[i] = (d_lo + i < D) ? to_f32<T>(vrow[i]) : 0.f;
            }
#pragma unroll
            for (int qi = 0; qi < kQPW; ++qi) {
              const float p = reinterpret_cast<const float*>(&p4[qi])[u];
#pragma unroll
              for (int i = 0; i < DPL; ++i) av[qi][i] += p * vv[i];
            }
          }
        }
      }
      __syncthreads();
    }
  }

  // epilogue
  const int64_t osd = out.s3;
#pragma unroll
  for (int qi = 0; qi < kQPW; ++qi) {
    const int mq = myq[qi];
    if (mq >= nq) continue;
    const float sum = l_i[qi];
    const float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
    T* op = out.at(b, m_base + mq, h, 0);
#pragma unroll
    for (int i = 0; i < DPL; ++i) {
      const int d = d_lo + i;
      if (d < D) op[static_cast<int64_t>(d) * osd] = from_f32<T>(av[qi][i] * inv);
    }
    if (lane == 0)
      *lse.at(b, h, m_base + mq) = (sum > 0.f) ? (m_i[qi] + logf(sum)) : kNegInf;
  }
}

// G == 1: one warp per query, kTokenWarps queries per CTA, no shared K/V staging.
template <typename T, typename ACC, int DPL>
__global__ void __launch_bounds__(kTokenWarps * 32) sel_fwd_token_kernel(
    const Strided4<const T> q, const Strided4<T> out, const Strided3<float> lse, const ACC kv,
    const int H, const int D, const int MG, const int Tb, const int chunk_size,
    const float scale, const int32_t* __restrict__ order, const int64_t os0,
    const int64_t os1, const int64_t os2) {
  extern __shared__ __align__(16) float smem[];
  const int Dv = round4(D);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nw = blockDim.x >> 5;
  float* sQ = smem + static_cast<size_t>(warp) * Dv;
  float* sP = smem + static_cast<size_t>(nw) * Dv + static_cast<size_t>(warp) * kTileN;

  const int bh = blockIdx.y;
  const int b = bh / H;
  const int h = bh - b * H;
  int g = blockIdx.x * nw + warp;         // group == query (G == 1)
  if (g >= MG) return;                    // warp-uniform; this kernel has no __syncthreads()
  // Visit queries in ascending first-selected-block order so neighbouring warps share KV lines.
  if (order != nullptr) g = order[b * os0 + static_cast<int64_t>(g) * os1 + h * os2];

  {
    const T* qp = q.at(b, g, h, 0);
    const int64_t qsd = q.s3;
    if (qsd == 1) {
      for (int d = lane; d < D; d += 32) sQ[d] = to_f32<T>(qp[d]);
    } else {
      for (int d = lane; d < D; d += 32) sQ[d] = to_f32<T>(qp[static_cast<int64_t>(d) * qsd]);
    }
    for (int d = D + lane; d < Dv; d += 32) sQ[d] = 0.f;
  }
  __syncwarp();

  const int q_chunk = g / chunk_size;
  const int d_lo = lane * DPL;
  const bool d_active = d_lo < D;
  const bool d_full = (d_lo + DPL) <= D;
  float m_i = kNegInf, l_i = 0.f;
  float av[DPL];
#pragma unroll
  for (int i = 0; i < DPL; ++i) av[i] = 0.f;
  constexpr int VE16 = 16 / static_cast<int>(sizeof(T));
  constexpr int VE8 = 8 / static_cast<int>(sizeof(T));

  for (int t = 0; t < Tb; ++t) {
    const BlockView<T> bv = kv.block(b, g, h, t, q_chunk);
    if (!bv.valid()) continue;
    const int64_t vsn = kv.v_stride_n();
    const int64_t vsd = kv.v_stride_d();
    // K: one whole row per lane, 16B vector loads when stride_d == 1 and aligned.
    const int kvw = gvec_width<T>(bv.k, bv.stride_n, bv.stride_d, D, D);
    // V: lane == dim slice; load_vecN only when rows are aligned for it.
    const bool v_vec =
        d_full && gvec_width<T>(bv.v, vsn, vsd, D, DPL) >= load_vecN_align<T, DPL>();

    for (int tok0 = 0; tok0 < bv.n_valid; tok0 += kTileN) {
      const int ntile = ::min(kTileN, bv.n_valid - tok0);
      const int tok = tok0 + lane;
      bool ok = lane < ntile;
      if (ok && bv.token0 >= 0) ok = ((bv.token0 + tok) / chunk_size) <= q_chunk;

      // QK: lane == token
      float s = 0.f;
      if (ok) {
        const T* krow = bv.k + static_cast<int64_t>(tok) * bv.stride_n;
        if (kvw == 16) {
          const uint4* k4 = reinterpret_cast<const uint4*>(krow);
          const int n = D / VE16;
          for (int j = 0; j < n; ++j) {
            const uint4 r = k4[j];
            const T* e = reinterpret_cast<const T*>(&r);
#pragma unroll
            for (int u = 0; u < VE16; ++u) s += sQ[j * VE16 + u] * to_f32<T>(e[u]);
          }
        } else if (kvw == 8) {
          const uint2* k2 = reinterpret_cast<const uint2*>(krow);
          const int n = D / VE8;
          for (int j = 0; j < n; ++j) {
            const uint2 r = k2[j];
            const T* e = reinterpret_cast<const T*>(&r);
#pragma unroll
            for (int u = 0; u < VE8; ++u) s += sQ[j * VE8 + u] * to_f32<T>(e[u]);
          }
        } else if (bv.stride_d == 1) {
          for (int d = 0; d < D; ++d) s += sQ[d] * to_f32<T>(krow[d]);
        } else {
          for (int d = 0; d < D; ++d)
            s += sQ[d] * to_f32<T>(krow[static_cast<int64_t>(d) * bv.stride_d]);
        }
      }

      const float sv = ok ? s * scale : kNegInf;
      const float tmax = warp_max(sv);
      float p = 0.f;
      if (tmax > kNegInf) {
        const float mnew = fmaxf(m_i, tmax);
        const float corr = (m_i == kNegInf) ? 0.f : exp2f((m_i - mnew) * kLog2e);
        p = safe_exp_diff(sv, mnew);
        const float ssum = warp_sum(p);
        l_i = l_i * corr + ssum;
#pragma unroll
        for (int i = 0; i < DPL; ++i) av[i] *= corr;
        m_i = mnew;
      }
      sP[lane] = p;
      __syncwarp();

      // PV: lane == dim slice
      if (d_active) {
        for (int tk = 0; tk < ntile; ++tk) {
          const float pp = sP[tk];
          if (pp == 0.f) continue;   // warp-uniform: masked or underflowed token
          const T* vrow = bv.v + static_cast<int64_t>(tok0 + tk) * vsn + d_lo;
          if (v_vec) {
            float vv[DPL];
            load_vecN<T, DPL>(vrow, vv);
#pragma unroll
            for (int i = 0; i < DPL; ++i) av[i] += pp * vv[i];
          } else if (vsd == 1) {
#pragma unroll
            for (int i = 0; i < DPL; ++i)
              if (d_lo + i < D) av[i] += pp * to_f32<T>(vrow[i]);
          } else {
            const T* vr = bv.v + static_cast<int64_t>(tok0 + tk) * vsn;
#pragma unroll
            for (int i = 0; i < DPL; ++i) {
              const int d = d_lo + i;
              if (d < D) av[i] += pp * to_f32<T>(vr[static_cast<int64_t>(d) * vsd]);
            }
          }
        }
      }
      __syncwarp();
    }
  }

  const float inv = (l_i > 0.f) ? (1.f / l_i) : 0.f;
  T* op = out.at(b, g, h, 0);
  const int64_t osd = out.s3;
#pragma unroll
  for (int i = 0; i < DPL; ++i) {
    const int d = d_lo + i;
    if (d < D) op[static_cast<int64_t>(d) * osd] = from_f32<T>(av[i] * inv);
  }
  if (lane == 0) *lse.at(b, h, g) = (l_i > 0.f) ? (m_i + logf(l_i)) : kNegInf;
}

// TF32 mma path (sm_80), used when allow_tf32; fragment layout as in bwd_common.cuh.
// CTA = one (b, group, h); warp (mt, nt) owns 16 queries x 8 tokens of every KV tile.
__device__ __forceinline__ uint32_t to_tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}

// bf16 is the top half of fp32, so bf16 -> tf32 is an exact shift; fp16 needs a real convert.
template <typename T>
__device__ __forceinline__ uint32_t elem_to_tf32(T x);
template <>
__device__ __forceinline__ uint32_t elem_to_tf32<c10::BFloat16>(c10::BFloat16 x) {
  return static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&x)) << 16;
}
template <>
__device__ __forceinline__ uint32_t elem_to_tf32<c10::Half>(c10::Half x) {
  return to_tf32(to_f32<c10::Half>(x));
}
template <>
__device__ __forceinline__ uint32_t elem_to_tf32<float>(float x) { return to_tf32(x); }

// sm_80 async global->shared copy: every chunk of a KV tile is in flight at once.
__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::
               "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void mma_m16n8k8(float (&d)[4], const uint32_t (&a)[4],
                                            const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Reduction over the 4 lanes of a fragment row group (t4 = 0..3).
__device__ __forceinline__ float grp_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float grp_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

__host__ __device__ __forceinline__ int mma_nm(int Gc) { return (Gc + 15) >> 4; }
__host__ __device__ __forceinline__ int mma_ldo(int D) { return dn_for(D) * 8 + 4; }
__host__ __device__ __forceinline__ size_t mma_f32_bytes(int Gc, int D) {
  const size_t nmq = static_cast<size_t>(mma_nm(Gc)) * 16;
  const size_t n = nmq * ldk_for(D)                    // sQ
                 + static_cast<size_t>(mma_nm(Gc)) * 4 * 16 * 2   // sM, sL
                 + nmq * 2;                            // sMt, sLt
  return (n * 4 + 15) & ~static_cast<size_t>(15);
}
__host__ __device__ __forceinline__ size_t mma_smem_bytes(int Gc, int D, size_t esize) {
  const size_t kvb = esize * static_cast<size_t>(kTileN) * (ldk_for(D) + ldv_for(D));
  const size_t ob = static_cast<size_t>(mma_nm(Gc)) * 16 * mma_ldo(D) * 4;
  return mma_f32_bytes(Gc, D) + (kvb > ob ? kvb : ob);
}

// 8-byte cp.async needs stride_d == 1 and 8-byte alignment of row base, row step and pitch.
template <typename T>
__device__ __forceinline__ bool async_ok(const T* base, int64_t sn, int64_t sd, int D, int ld) {
  constexpr size_t es = sizeof(T);
  return sd == 1 && ((static_cast<size_t>(D) * es) % 8 == 0) &&
         ((reinterpret_cast<uintptr_t>(base) & 7) == 0) &&
         (((static_cast<size_t>(sn) * es) & 7) == 0) &&
         (((static_cast<size_t>(ld) * es) & 7) == 0);
}

template <typename T, typename ACC, int DN>
__global__ void __launch_bounds__(512) sel_fwd_group_mma_kernel(
    const Strided4<const T> q, const Strided4<T> out, const Strided3<float> lse, const ACC kv,
    const int H, const int D, const int G, const int Gc, const int Tb, const int chunk_size,
    const float scale, const int32_t* __restrict__ order, const int64_t os0,
    const int64_t os1, const int64_t os2) {
  extern __shared__ __align__(16) char smem_raw[];
  const int ldq = ldk_for(D), ldk = ldq, ldv = ldv_for(D), ldo = DN * 8 + 4;
  const int nm = mma_nm(Gc), nmq = nm * 16;
  uint32_t* sQ = reinterpret_cast<uint32_t*>(smem_raw);           // [nmq][ldq] tf32 bits
  float* sM = reinterpret_cast<float*>(sQ + static_cast<size_t>(nmq) * ldq);  // [nm*4][16]
  float* sL = sM + static_cast<size_t>(nm) * 64;                  // [nm*4][16]
  float* sMt = sL + static_cast<size_t>(nm) * 64;                 // [nmq]
  float* sLt = sMt + nmq;                                         // [nmq]
  const size_t f32b = mma_f32_bytes(Gc, D);
  const size_t bufel = static_cast<size_t>(kTileN) * (ldk + ldv);
  T* sKV = reinterpret_cast<T*>(smem_raw + f32b);                 // [32][ldk] + [32][ldv]
  float* sO = reinterpret_cast<float*>(smem_raw + f32b);          // overlays the tiles at the end

  const int bh = blockIdx.z;
  const int b = bh / H, h = bh - b * H;
  // Visit groups in ascending first-selected-block order so concurrent CTAs hit in L2.
  const int g = (order == nullptr)
                    ? static_cast<int>(blockIdx.x)
                    : order[b * os0 + static_cast<int64_t>(blockIdx.x) * os1 + h * os2];
  const int q0 = blockIdx.y * Gc;
  const int nq = ::min(Gc, G - q0);
  if (nq <= 0) return;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nw = blockDim.x >> 5;
  const int mt = warp >> 2, nt = warp & 3;
  const int gid = lane >> 2, t4 = lane & 3;
  const int t0 = nt * 8;
  const int64_t m_base = static_cast<int64_t>(g) * G + q0;
  const int KN = (D + 7) >> 3;

  // stage Q as TF32 bits (rows >= nq and the pitch tail zeroed)
  {
    const int64_t qsd = q.s3;
    for (int i = warp; i < nmq; i += nw) {
      uint32_t* dst = sQ + static_cast<size_t>(i) * ldq;
      if (i < nq) {
        const T* qp = q.at(b, m_base + i, h, 0);
        if (qsd == 1) {
          for (int d = lane; d < D; d += 32) dst[d] = elem_to_tf32<T>(qp[d]);
        } else {
          for (int d = lane; d < D; d += 32)
            dst[d] = elem_to_tf32<T>(qp[static_cast<int64_t>(d) * qsd]);
        }
        for (int d = D + lane; d < ldq; d += 32) dst[d] = 0u;
      } else {
        for (int d = lane; d < ldq; d += 32) dst[d] = 0u;
      }
    }
  }

  float m_i[2] = {kNegInf, kNegInf};
  float l_i[2] = {0.f, 0.f};
  float acc[DN][4];
#pragma unroll
  for (int n = 0; n < DN; ++n) {
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[n][e] = 0.f;
  }
  const int q_chunk = static_cast<int>((static_cast<int64_t>(g) * G) / chunk_size);
  // Zero the tile once so rows past ntile stay finite: masked columns give 0*finite, not 0*NaN.
  {
    const T z = from_f32<T>(0.f);
    for (size_t i = threadIdx.x; i < bufel; i += blockDim.x) sKV[i] = z;
  }
  const int cpr = static_cast<int>((static_cast<size_t>(D) * sizeof(T)) >> 3);
  const int rpp = (cpr > 0 && cpr <= static_cast<int>(blockDim.x))
                      ? static_cast<int>(blockDim.x) / cpr : 0;   // rows staged per pass
  const int crow = (rpp > 0) ? static_cast<int>(threadIdx.x) / cpr : 0;
  const int cchunk = (rpp > 0) ? static_cast<int>(threadIdx.x) - crow * cpr : 0;
  const int crow_act = (rpp > 0 && crow < rpp) ? crow : kTileN;   // idle tail threads
  __syncthreads();

  // Chunk-causal visibility is a prefix of the block, so one division per block gives the count.
  auto vis_of = [&](const BlockView<T>& bx) -> int {
    if (bx.token0 < 0) return bx.n_valid;
    const int64_t lim = static_cast<int64_t>(q_chunk + 1) * chunk_size - bx.token0;
    if (lim <= 0) return 0;
    return (lim < bx.n_valid) ? static_cast<int>(lim) : bx.n_valid;
  };

  // Copy one KV tile into buffer bi; cp.async unless the layout cannot be addressed by it.
  auto issue = [&](const BlockView<T>& bx, int tk0, int bi, int vis) {
    T* dK = sKV + static_cast<size_t>(bi) * bufel;
    T* dV = dK + static_cast<size_t>(kTileN) * ldk;
    const int64_t vsn = kv.v_stride_n(), vsd = kv.v_stride_d();
    const int nt2 = ::min(kTileN, vis - tk0);
    const bool ka = async_ok<T>(bx.k, bx.stride_n, bx.stride_d, D, ldk);
    const bool va = async_ok<T>(bx.v, vsn, vsd, D, ldv);
    // (row, chunk) is decoded once per thread; a per-chunk division costs more than the copies.
    if (ka && rpp > 0) {
      for (int r = crow_act; r < nt2; r += rpp)
        cp_async8(reinterpret_cast<char*>(dK + static_cast<size_t>(r) * ldk) + cchunk * 8,
                  reinterpret_cast<const char*>(
                      bx.k + static_cast<int64_t>(tk0 + r) * bx.stride_n) + cchunk * 8);
    }
    if (va && rpp > 0) {
      for (int r = crow_act; r < nt2; r += rpp)
        cp_async8(reinterpret_cast<char*>(dV + static_cast<size_t>(r) * ldv) + cchunk * 8,
                  reinterpret_cast<const char*>(
                      bx.v + static_cast<int64_t>(tk0 + r) * vsn) + cchunk * 8);
    }
    if ((ka || va) && rpp == 0) {   // cpr > blockDim.x: one row at a time
      for (int r = 0; r < nt2; ++r) {
        for (int j = threadIdx.x; j < cpr; j += blockDim.x) {
          if (ka) cp_async8(reinterpret_cast<char*>(dK + static_cast<size_t>(r) * ldk) + j * 8,
                            reinterpret_cast<const char*>(
                                bx.k + static_cast<int64_t>(tk0 + r) * bx.stride_n) + j * 8);
          if (va) cp_async8(reinterpret_cast<char*>(dV + static_cast<size_t>(r) * ldv) + j * 8,
                            reinterpret_cast<const char*>(
                                bx.v + static_cast<int64_t>(tk0 + r) * vsn) + j * 8);
        }
      }
    }
    if (!ka || !va) {
      const int kvw = ka ? 0 : vec_width<T>(bx.k, bx.stride_n, bx.stride_d, D, ldk);
      const int vvw = va ? 0 : vec_width<T>(bx.v, vsn, vsd, D, ldv);
      for (int r = warp; r < nt2; r += nw) {
        const int64_t tr = static_cast<int64_t>(tk0 + r);
        if (!ka) copy_row<T>(bx.k + tr * bx.stride_n, dK + static_cast<size_t>(r) * ldk, D, ldk,
                             bx.stride_d, kvw, lane);
        if (!va) copy_row<T>(bx.v + tr * vsn, dV + static_cast<size_t>(r) * ldv, D, ldv, vsd,
                             vvw, lane);
      }
    }
    cp_async_commit();
  };

  int t = 0, tok0 = 0, bvis = 0;
  constexpr int buf = 0;
  BlockView<T> bv{nullptr, nullptr, 0, 0, 0, -1};
  bool have = false;
  for (; t < Tb; ++t) {
    bv = kv.block(b, g, h, t, q_chunk);
    bvis = bv.valid() ? vis_of(bv) : 0;
    if (bvis > 0) { have = true; break; }
  }
  if (have) issue(bv, 0, buf, bvis);

  while (have) {
    // locate the next tile (rest of this block, else the next visible block)
    BlockView<T> nbv = bv;
    int nt_ = t, ntok0 = tok0 + kTileN, nvis = bvis;
    bool nhave = ntok0 < bvis;
    if (!nhave) {
      ntok0 = 0;
      for (nt_ = t + 1; nt_ < Tb; ++nt_) {
        nbv = kv.block(b, g, h, nt_, q_chunk);
        nvis = nbv.valid() ? vis_of(nbv) : 0;
        if (nvis > 0) { nhave = true; break; }
      }
    }
    cp_async_wait<0>();
    __syncthreads();

    {
      const int ntile = ::min(kTileN, bvis - tok0);
      const T* sK = sKV + static_cast<size_t>(buf) * bufel;
      const T* sV = sK + static_cast<size_t>(kTileN) * ldk;

      // S = Q K^T over this warp's 8 tokens
      float sfrag[4] = {0.f, 0.f, 0.f, 0.f};
      {
        const uint32_t* qrow0 = sQ + static_cast<size_t>(mt * 16 + gid) * ldq;
        const uint32_t* qrow1 = qrow0 + static_cast<size_t>(8) * ldq;
        const T* krow = sK + static_cast<size_t>(t0 + gid) * ldk;
        for (int kc = 0; kc < KN; ++kc) {
          const int d0 = kc * 8;
          uint32_t a[4], bf[2];
          a[0] = qrow0[d0 + t4];              // already tf32 bits: no conversion here
          a[1] = qrow1[d0 + t4];
          a[2] = qrow0[d0 + t4 + 4];
          a[3] = qrow1[d0 + t4 + 4];
          bf[0] = elem_to_tf32<T>(krow[d0 + t4]);
          bf[1] = elem_to_tf32<T>(krow[d0 + t4 + 4]);
          mma_m16n8k8(sfrag, a, bf);
        }
      }

      // mask + online softmax (rows gid and gid+8)
      float sv[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        // ntile counts only visible tokens, so the per-element test is a plain bound check.
        const int loc = t0 + 2 * t4 + (e & 1);
        sv[e] = (loc < ntile) ? sfrag[e] * scale : kNegInf;
      }
      float pr[4];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float tmax = grp_max(fmaxf(sv[2 * r], sv[2 * r + 1]));
        const float mnew = fmaxf(m_i[r], tmax);
        // branch free: every lane of the group runs the same shuffles
        const float corr = (m_i[r] == kNegInf) ? 0.f
                         : ((mnew == kNegInf) ? 1.f : exp2f((m_i[r] - mnew) * kLog2e));
        pr[2 * r] = (mnew == kNegInf) ? 0.f : safe_exp_diff(sv[2 * r], mnew);
        pr[2 * r + 1] = (mnew == kNegInf) ? 0.f : safe_exp_diff(sv[2 * r + 1], mnew);
        l_i[r] = l_i[r] * corr + grp_sum(pr[2 * r] + pr[2 * r + 1]);
#pragma unroll
        for (int n = 0; n < DN; ++n) {
          acc[n][2 * r] *= corr;
          acc[n][2 * r + 1] *= corr;
        }
        m_i[r] = mnew;
      }

      // O += P V; the C -> A fragment remap takes one 8-shuffle permutation per tile
      {
        const int s1 = (lane & ~3) + (t4 >> 1), s2 = s1 + 2;
        const bool odd = (t4 & 1) != 0;
        const float x0 = __shfl_sync(0xffffffffu, pr[0], s1);
        const float x1 = __shfl_sync(0xffffffffu, pr[1], s1);
        const float y0 = __shfl_sync(0xffffffffu, pr[2], s1);
        const float y1 = __shfl_sync(0xffffffffu, pr[3], s1);
        const float z0 = __shfl_sync(0xffffffffu, pr[0], s2);
        const float z1 = __shfl_sync(0xffffffffu, pr[1], s2);
        const float w0 = __shfl_sync(0xffffffffu, pr[2], s2);
        const float w1 = __shfl_sync(0xffffffffu, pr[3], s2);
        uint32_t pa[4];
        pa[0] = to_tf32(odd ? x1 : x0);
        pa[1] = to_tf32(odd ? y1 : y0);
        pa[2] = to_tf32(odd ? z1 : z0);
        pa[3] = to_tf32(odd ? w1 : w0);
        const T* vr0 = sV + static_cast<size_t>(t0 + t4) * ldv;
        const T* vr1 = sV + static_cast<size_t>(t0 + t4 + 4) * ldv;
#pragma unroll
        for (int n = 0; n < DN; ++n) {
          uint32_t vb[2];
          vb[0] = elem_to_tf32<T>(vr0[n * 8 + gid]);
          vb[1] = elem_to_tf32<T>(vr1[n * 8 + gid]);
          mma_m16n8k8(acc[n], pa, vb);
        }
      }
    }
    __syncthreads();           // everyone is done reading the tile
    if (!nhave) break;
    issue(nbv, ntok0, buf, nvis);
    bv = nbv; t = nt_; tok0 = ntok0; bvis = nvis;
  }

  // merge the 4 n-tile partial states of each m-tile
  if (t4 == 0) {
    float* mr = sM + static_cast<size_t>(mt * 4 + nt) * 16;
    float* lr = sL + static_cast<size_t>(mt * 4 + nt) * 16;
    mr[gid] = m_i[0];  mr[gid + 8] = m_i[1];
    lr[gid] = l_i[0];  lr[gid + 8] = l_i[1];
  }
  for (int i = threadIdx.x; i < nmq * ldo; i += blockDim.x) sO[i] = 0.f;
  __syncthreads();

  float wgt[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = gid + 8 * r;
    float mm = kNegInf, ls = 0.f;
#pragma unroll
    for (int u = 0; u < 4; ++u) mm = fmaxf(mm, sM[static_cast<size_t>(mt * 4 + u) * 16 + row]);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float mu = sM[static_cast<size_t>(mt * 4 + u) * 16 + row];
      const float lu = sL[static_cast<size_t>(mt * 4 + u) * 16 + row];
      ls += (mu == kNegInf) ? 0.f : lu * exp2f((mu - mm) * kLog2e);
    }
    wgt[r] = (m_i[r] == kNegInf) ? 0.f : exp2f((m_i[r] - mm) * kLog2e);
    if (nt == 0 && t4 == 0) {
      sMt[mt * 16 + row] = mm;
      sLt[mt * 16 + row] = ls;
    }
  }

  for (int u = 0; u < 4; ++u) {
    if (nt == u) {
#pragma unroll
      for (int n = 0; n < DN; ++n) {
        const int d0 = n * 8 + 2 * t4;
        if (d0 < D) {
          float* o0 = sO + static_cast<size_t>(mt * 16 + gid) * ldo + d0;
          float* o1 = sO + static_cast<size_t>(mt * 16 + gid + 8) * ldo + d0;
          o0[0] += acc[n][0] * wgt[0];
          o1[0] += acc[n][2] * wgt[1];
          if (d0 + 1 < D) {
            o0[1] += acc[n][1] * wgt[0];
            o1[1] += acc[n][3] * wgt[1];
          }
        }
      }
    }
    __syncthreads();
  }

  // normalised, coalesced write-out
  const int64_t osd = out.s3;
  for (int row = warp; row < nq; row += nw) {
    const float l = sLt[row];
    const float inv = (l > 0.f) ? (1.f / l) : 0.f;
    T* op = out.at(b, m_base + row, h, 0);
    const float* orow = sO + static_cast<size_t>(row) * ldo;
    for (int d = lane; d < D; d += 32)
      op[static_cast<int64_t>(d) * osd] = from_f32<T>(orow[d] * inv);
    if (lane == 0)
      *lse.at(b, h, m_base + row) = (l > 0.f) ? (sMt[row] + logf(l)) : kNegInf;
  }
}

#define OMNIMEM_DISPATCH_DN(D, ...)                                        \
  [&] {                                                                    \
    if ((D) <= 64) { constexpr int DN_MAX = 8; return __VA_ARGS__(); }     \
    if ((D) <= 96) { constexpr int DN_MAX = 12; return __VA_ARGS__(); }    \
    { constexpr int DN_MAX = 16; return __VA_ARGS__(); }                   \
  }()

#define OMNIMEM_DISPATCH_DPL(D, ...)                                       \
  [&] {                                                                    \
    if ((D) <= 64) { constexpr int DPL_MAX = 2; return __VA_ARGS__(); }    \
    if ((D) <= 128) { constexpr int DPL_MAX = 4; return __VA_ARGS__(); }   \
    { constexpr int DPL_MAX = 8; return __VA_ARGS__(); }                   \
  }()

// The TF32 kernel is only used for 32 <= D <= 128 (O accumulator in registers).
inline bool mma_eligible(bool tf32, int G, int D) {
  return tf32 && G > 1 && D >= 32 && D <= 128;
}

template <typename T, typename ACC>
void launch_sel_fwd(const at::Tensor& q, at::Tensor& out, at::Tensor& lse, const ACC& kv,
                    int64_t B, int64_t M, int64_t H, int64_t D, int G, int Tb, int chunk_size,
                    double scale, bool tf32, const at::Tensor& order, cudaStream_t stream) {
  const int MG = static_cast<int>(M / G);
  const auto qs = make_strided4_const<T>(q);
  const auto os = make_strided4<T>(out);
  const auto ls = make_strided3<float>(lse);
  const float sc = static_cast<float>(scale);
  const int64_t BH = B * H;
  TORCH_CHECK(BH <= 65535, "sel_fwd: B*H = ", BH, " exceeds the grid.z limit");
  const int Di = static_cast<int>(D);

  const int32_t* ord = order.defined() ? order.data_ptr<int32_t>() : nullptr;
  const int64_t o0 = order.defined() ? order.stride(0) : 0;
  const int64_t o1 = order.defined() ? order.stride(1) : 0;
  const int64_t o2 = order.defined() ? order.stride(2) : 0;

  if (G == 1) {
    const int nw = kTokenWarps;
    const size_t smem = static_cast<size_t>(nw) * (round4(Di) + kTileN) * sizeof(float);
    const dim3 grid(static_cast<unsigned>((MG + nw - 1) / nw), static_cast<unsigned>(BH));
    OMNIMEM_DISPATCH_DPL(Di, [&] {
      sel_fwd_token_kernel<T, ACC, DPL_MAX><<<grid, nw * 32, smem, stream>>>(
          qs, os, ls, kv, static_cast<int>(H), Di, MG, Tb, chunk_size, sc, ord, o0, o1, o2);
    });
    OMNIMEM_CHECK_LAUNCH();
    return;
  }

  if (mma_eligible(tf32, G, Di)) {
    int Gc = ::std::min(G, 64);
    while (Gc > 16 && mma_smem_bytes(Gc, Di, sizeof(T)) > kSmemBudget) Gc -= 16;
    if (mma_smem_bytes(Gc, Di, sizeof(T)) <= kSmemBudget) {
      const int nw = mma_nm(Gc) * 4;
      const int nsplit = (G + Gc - 1) / Gc;
      const size_t smem = mma_smem_bytes(Gc, Di, sizeof(T));
      const dim3 grid(static_cast<unsigned>(MG), static_cast<unsigned>(nsplit),
                      static_cast<unsigned>(BH));
      OMNIMEM_DISPATCH_DN(Di, [&] {
        if (smem > 48 * 1024) {
          C10_CUDA_CHECK(cudaFuncSetAttribute(
              reinterpret_cast<const void*>(&sel_fwd_group_mma_kernel<T, ACC, DN_MAX>),
              cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        }
        sel_fwd_group_mma_kernel<T, ACC, DN_MAX><<<grid, nw * 32, smem, stream>>>(
            qs, os, ls, kv, static_cast<int>(H), Di, G, Gc, Tb, chunk_size, sc, ord, o0, o1, o2);
      });
      OMNIMEM_CHECK_LAUNCH();
      return;
    }
  }

  int Gc = ::std::min(G, kQPW * kMaxWarpsGroup);
  while (Gc > 1 && group_smem_bytes(Gc, Di, sizeof(T)) > kSmemBudget) --Gc;
  TORCH_CHECK(group_smem_bytes(Gc, Di, sizeof(T)) <= kSmemBudget,
              "sel_fwd: D=", D, " needs too much shared memory");
  const int nw = (Gc + kQPW - 1) / kQPW;
  const int nsplit = (G + Gc - 1) / Gc;
  const size_t smem = group_smem_bytes(Gc, Di, sizeof(T));
  const dim3 grid(static_cast<unsigned>(MG), static_cast<unsigned>(nsplit),
                  static_cast<unsigned>(BH));
  OMNIMEM_DISPATCH_DPL(Di, [&] {
    if (smem > 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          reinterpret_cast<const void*>(&sel_fwd_group_kernel<T, ACC, DPL_MAX>),
          cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    }
    sel_fwd_group_kernel<T, ACC, DPL_MAX><<<grid, nw * 32, smem, stream>>>(
        qs, os, ls, kv, static_cast<int>(H), Di, G, Gc, Tb, chunk_size, sc, ord, o0, o1, o2);
  });
  OMNIMEM_CHECK_LAUNCH();
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> sel_fwd_cuda(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& block_indices, int64_t block_size, int64_t chunk_size, double scale,
    bool causal, int64_t group_size, bool allow_tf32) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && block_indices.is_cuda(),
              "sel_fwd: all tensors must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "sel_fwd: q/k/v must be 4-D");
  TORCH_CHECK(k.sizes() == v.sizes(), "sel_fwd: k and v must have the same shape");
  const int64_t B = q.size(0), M = q.size(1), H = q.size(2), D = q.size(3);
  const int64_t N = k.size(1);
  TORCH_CHECK(k.size(0) == B && k.size(2) == H && k.size(3) == D,
              "sel_fwd: expected k/v [B,N,H,D] compatible with q");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
              "sel_fwd: q/k/v must share one dtype, got ", q.scalar_type(), "/", k.scalar_type(),
              "/", v.scalar_type());
  TORCH_CHECK(block_indices.scalar_type() == at::kInt, "sel_fwd: block_indices must be int32");
  TORCH_CHECK(block_indices.dim() == 4, "sel_fwd: block_indices must be [B,MG,H,T]");
  TORCH_CHECK(D >= 16 && D <= 256, "sel_fwd: D must be in [16, 256], got ", D);
  TORCH_CHECK(group_size >= 1 && group_size <= 128, "sel_fwd: group_size must be in [1,128]");
  TORCH_CHECK(M % group_size == 0, "sel_fwd: M (", M, ") must be divisible by group_size (",
              group_size, ")");
  TORCH_CHECK(chunk_size > 0, "sel_fwd: chunk_size must be > 0");
  TORCH_CHECK(block_size > 0, "sel_fwd: block_size must be > 0");
  const int64_t MG = M / group_size;
  TORCH_CHECK(block_indices.size(0) == B && block_indices.size(1) == MG &&
                  block_indices.size(2) == H,
              "sel_fwd: block_indices must be [", B, ",", MG, ",", H, ",T], got ",
              block_indices.sizes());

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  auto lse = at::empty({B, H, M}, q.options().dtype(at::kFloat));
  if (B == 0 || M == 0 || H == 0) return std::make_tuple(out, lse);

  const int64_t T = block_indices.size(3);
  auto stream = current_stream(q);
  // Sort groups by first selected block (small argsort) for L2 hits; only for many groups.
  at::Tensor order;
  if (group_size == 1 && MG >= 256 && T > 0)
    order = at::argsort(block_indices.select(3, 0), /*dim=*/static_cast<int64_t>(1), /*descending=*/false).to(at::kInt).contiguous();
  OMNIMEM_DISPATCH_FLOAT_TYPES(q.scalar_type(), "sel_fwd", [&] {
    DenseKV<scalar_t> kv{make_strided4_const<scalar_t>(k),
                         make_strided4_const<scalar_t>(v),
                         block_indices.data_ptr<int32_t>(),
                         block_indices.stride(0), block_indices.stride(1),
                         block_indices.stride(2), block_indices.stride(3),
                         static_cast<int>(N), static_cast<int>(block_size),
                         static_cast<int>(chunk_size), causal};
    launch_sel_fwd<scalar_t, DenseKV<scalar_t>>(q, out, lse, kv, B, M, H, D,
                                                static_cast<int>(group_size),
                                                static_cast<int>(T),
                                                static_cast<int>(chunk_size), scale,
                                                allow_tf32, order, stream);
  });
  return std::make_tuple(out, lse);
}

std::tuple<at::Tensor, at::Tensor> sel_fwd_ptr_cuda(
    const at::Tensor& q, const at::Tensor& ptr_k, const at::Tensor& ptr_v, int64_t block_size,
    int64_t group_size, int64_t stride_bn, int64_t stride_bd, double scale,
    c10::string_view input_precision, at::TensorList owners) {
  // "tf32" permits the TF32 tensor-core kernel; "ieee" / "tf32x3" keep FP32 operands.
  const bool tf32 = (parse_precision(input_precision) == Precision::kTF32);
  TORCH_CHECK(q.is_cuda() && ptr_k.is_cuda() && ptr_v.is_cuda(),
              "sel_fwd_ptr: all tensors must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4, "sel_fwd_ptr: q must be [B,M,H,D]");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "sel_fwd_ptr: q must be bfloat16, got ",
              q.scalar_type());
  TORCH_CHECK(ptr_k.scalar_type() == at::kLong && ptr_v.scalar_type() == at::kLong,
              "sel_fwd_ptr: pointer tables must be int64 byte addresses");
  TORCH_CHECK(ptr_k.dim() == 4 && ptr_v.sizes() == ptr_k.sizes() &&
                  ptr_v.strides() == ptr_k.strides(),
              "sel_fwd_ptr: ptr_k / ptr_v must be [B,MG,H,T] with identical layout");
  const int64_t B = q.size(0), M = q.size(1), H = q.size(2), D = q.size(3);
  TORCH_CHECK(D >= 16 && D <= 256, "sel_fwd_ptr: D must be in [16, 256], got ", D);
  TORCH_CHECK(group_size >= 1 && group_size <= 128,
              "sel_fwd_ptr: group_size must be in [1,128]");
  TORCH_CHECK(M % group_size == 0, "sel_fwd_ptr: M (", M, ") must be divisible by group_size (",
              group_size, ")");
  TORCH_CHECK(block_size > 0, "sel_fwd_ptr: block_size must be > 0");
  const int64_t MG = M / group_size;
  TORCH_CHECK(ptr_k.size(0) == B && ptr_k.size(1) == MG && ptr_k.size(2) == H,
              "sel_fwd_ptr: ptr tables must be [", B, ",", MG, ",", H, ",T], got ",
              ptr_k.sizes());

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  auto lse = at::empty({B, H, M}, q.options().dtype(at::kFloat));
  if (B == 0 || M == 0 || H == 0) return std::make_tuple(out, lse);

  const int64_t T = ptr_k.size(3);
  auto stream = current_stream(q);
  at::Tensor order;
  if (group_size == 1 && MG >= 256 && T > 0)
    order = at::argsort(ptr_k.select(3, 0), /*dim=*/static_cast<int64_t>(1), /*descending=*/false).to(at::kInt).contiguous();
  using scalar_t = c10::BFloat16;
  PtrKV<scalar_t> kv{ptr_k.data_ptr<int64_t>(), ptr_v.data_ptr<int64_t>(),
                     ptr_k.stride(0), ptr_k.stride(1), ptr_k.stride(2), ptr_k.stride(3),
                     stride_bn, stride_bd, static_cast<int>(block_size)};
  launch_sel_fwd<scalar_t, PtrKV<scalar_t>>(q, out, lse, kv, B, M, H, D,
                                            static_cast<int>(group_size), static_cast<int>(T),
                                            /*chunk_size=*/1, scale, tf32, order, stream);
  return std::make_tuple(out, lse);
}

}  // namespace omnimem_nsa
