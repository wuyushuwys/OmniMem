// Selection attention backward: delta, dQ (group-major), dK/dV (block-major).
// Kernel choice (TF32 allowed = allow_tf32); accumulators are FP32, MMA operands round to TF32:
//   dQ     TF32 allowed, G >= 8, 32 <= D <= 128   sel_bwd_dq_wmma_kernel   (TF32 MMA)
//          TF32 allowed, G >= 8, any other D      sel_bwd_dq_mma_kernel    (TF32 MMA)
//          TF32 allowed, G <= 7                   sel_bwd_dq_warp_kernel   (FP32)
//          otherwise                              sel_bwd_dq_kernel        (FP32 SIMT)
//   dK/dV  TF32 allowed                           sel_bwd_dkv_mma_kernel   (TF32 MMA)
//          otherwise                              sel_bwd_dkv_kernel       (FP32 SIMT)
#include "bwd_common.cuh"
#include "common.cuh"
#include "ops.h"

namespace omnimem_nsa {
namespace {

constexpr int kBwdThreads = 256;

// dQ: one CTA per (b, group g, h, row-tile) stages Q/dO/dQ and one KV tile, recomputes P from LSE.
template <typename scalar_t, typename AccT>
__global__ void sel_bwd_dq_kernel(
    Strided4<const scalar_t> q, Strided4<const scalar_t> o, Strided4<const scalar_t> dout,
    Strided4<scalar_t> dq, Strided3<const float> lse, Strided3<float> delta,
    const int32_t* __restrict__ order, int64_t ord_sb, int64_t ord_sg, int64_t ord_sh,
    AccT acc, int G, int T, int D, int chunk_size, float scale, int BG, int TN, int pitch,
    int n_row_tiles) {
  extern __shared__ float smem[];
  const int tid = threadIdx.x;
  const int NT = blockDim.x;
  const int gs = blockIdx.x / n_row_tiles;
  const int rt = blockIdx.x - gs * n_row_tiles;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  // order only permutes which group a CTA serves (bit-identical); CTAs then share KV in L2.
  const int g = (order == nullptr)
                    ? gs
                    : order[static_cast<int64_t>(b) * ord_sb + static_cast<int64_t>(gs) * ord_sg +
                            static_cast<int64_t>(h) * ord_sh];

  const int r0 = rt * BG;
  if (r0 >= G) return;
  const int rows = min(BG, G - r0);
  const int rows2 = rows + (rows & 1);  // padded to a pair; BG is even so rows2 <= BG
  const int64_t m0 = static_cast<int64_t>(g) * G + r0;

  float* sQ = smem;
  float* sDO = sQ + static_cast<size_t>(BG) * pitch;
  float* sDQ = sDO + static_cast<size_t>(BG) * pitch;
  float* sK = sDQ + static_cast<size_t>(BG) * pitch;
  float* sV = sK + static_cast<size_t>(TN) * pitch;
  float* sDS = sV + static_cast<size_t>(TN) * pitch;
  float* sLse = sDS + static_cast<size_t>(BG) * TN;
  float* sDel = sLse + BG;

  for (int idx = tid; idx < rows2 * D; idx += NT) {
    const int i = idx / D;
    const int d = idx - i * D;
    float qv = 0.f, ov = 0.f;
    if (i < rows) {
      const int64_t m = m0 + i;
      qv = to_f32(*q.at(b, m, h, d));
      ov = to_f32(*dout.at(b, m, h, d));
    }
    sQ[i * pitch + d] = from_f32<scalar_t>(qv);
    sDO[i * pitch + d] = from_f32<scalar_t>(ov);
    sDQ[i * pitch + d] = 0.f;
  }
  for (int i = tid; i < rows2; i += NT) {  // padding rows: LSE = -inf => P = 0
    sLse[i] = kNegInf;
    sDel[i] = 0.f;
  }
  __syncthreads();

  // delta prologue: one warp per query row, FP32 row reduction of out * dO.
  {
    const int lane = tid & (kWarpSize - 1);
    const int wid = tid >> 5;
    const int NW = NT >> 5;
    for (int i = wid; i < rows; i += NW) {
      const int64_t m = m0 + i;
      float acc_d = 0.f;
      for (int d = lane; d < D; d += kWarpSize)
        acc_d += to_f32(*o.at(b, m, h, d)) * to_f32(sDO[i * pitch + d]);
      acc_d = warp_sum(acc_d);
      if (lane == 0) {
        sDel[i] = acc_d;
        sLse[i] = *lse.at(b, h, m);
        *delta.at(b, h, m) = acc_d;
      }
    }
  }
  __syncthreads();

  const int q_chunk = static_cast<int>((static_cast<int64_t>(g) * G) / chunk_size);
  const int64_t vsn = acc.v_stride_n();
  const int64_t vsd = acc.v_stride_d();

  for (int t = 0; t < T; ++t) {
    const BlockView<scalar_t> bv = acc.block(b, g, h, t, q_chunk);
    if (!bv.valid()) continue;  // -1 / out-of-range / block-level causal: skip
    for (int tile0 = 0; tile0 < bv.n_valid; tile0 += TN) {
      const int ntok = min(TN, bv.n_valid - tile0);
      __syncthreads();
      for (int idx = tid; idx < TN * D; idx += NT) {
        const int j = idx / D;
        const int d = idx - j * D;
        float kv = 0.f, vv = 0.f;
        if (j < ntok) {
          const int64_t tk = tile0 + j;
          kv = to_f32(bv.k[tk * bv.stride_n + static_cast<int64_t>(d) * bv.stride_d]);
          vv = to_f32(bv.v[tk * vsn + static_cast<int64_t>(d) * vsd]);
        }
        sK[j * pitch + d] = kv;
        sV[j * pitch + d] = vv;
      }
      __syncthreads();
      // Two query rows per thread with independent accumulator chains.
      for (int idx = tid; idx < (rows2 >> 1) * TN; idx += NT) {
        const int ib = idx / TN;
        const int j = idx - ib * TN;
        const int i0 = 2 * ib, i1 = i0 + 1;
        float ds0 = 0.f, ds1 = 0.f;
        bool vis = j < ntok;
        if (vis && bv.token0 >= 0)  // dense + causal: per-token chunk mask
          vis = chunk_visible(bv.token0 + tile0 + j, chunk_size, q_chunk);
        if (vis) {
          const float l0 = sLse[i0], l1 = sLse[i1];
          float s0 = 0.f, s1 = 0.f, dp0 = 0.f, dp1 = 0.f;
          for (int d = 0; d < D; ++d) {  // S = Q K^T, dP = dO V^T
            const float kk = sK[j * pitch + d];
            const float vv = sV[j * pitch + d];
            s0 += sQ[i0 * pitch + d] * kk;
            s1 += sQ[i1 * pitch + d] * kk;
            dp0 += sDO[i0 * pitch + d] * vv;
            dp1 += sDO[i1 * pitch + d] * vv;
          }
          if (l0 > kNegInf) ds0 = fast_exp(s0 * scale - l0) * (dp0 - sDel[i0]) * scale;
          if (l1 > kNegInf) ds1 = fast_exp(s1 * scale - l1) * (dp1 - sDel[i1]) * scale;
        }
        sDS[i0 * TN + j] = ds0;
        sDS[i1 * TN + j] = ds1;
      }
      __syncthreads();
      for (int idx = tid; idx < (rows2 >> 1) * D; idx += NT) {
        const int ib = idx / D;
        const int d = idx - ib * D;
        const int i0 = 2 * ib, i1 = i0 + 1;
        float a0 = 0.f, a1 = 0.f;  // dQ += dS K
        for (int j = 0; j < TN; ++j) {
          const float kk = sK[j * pitch + d];
          a0 += sDS[i0 * TN + j] * kk;
          a1 += sDS[i1 * TN + j] * kk;
        }
        sDQ[i0 * pitch + d] += a0;
        sDQ[i1 * pitch + d] += a1;
      }
    }
  }

  __syncthreads();
  for (int idx = tid; idx < rows * D; idx += NT) {
    const int i = idx / D;
    const int d = idx - i * D;
    *dq.at(b, m0 + i, h, d) = from_f32<scalar_t>(sDQ[i * pitch + d]);
  }
}

// dK/dV: one CTA per (b, h, kv-block, kv-tile) owns its tile (no atomics), walks the CSR bucket.
template <typename scalar_t, typename ResT>
__global__ void sel_bwd_dkv_kernel(
    Strided4<const scalar_t> q, Strided4<const scalar_t> dout, Strided3<const float> lse,
    Strided3<const float> delta, const int32_t* __restrict__ groups, int64_t groups_sb,
    int64_t groups_sh, const int32_t* __restrict__ offsets, int64_t off_sb, int64_t off_sh,
    ResT res, int G, int D, int chunk_size, float scale, int num_tiles, int TKV, int BQ,
    int pitch) {
  extern __shared__ float smem[];
  const int tid = threadIdx.x;
  const int NT = blockDim.x;
  const int bid = blockIdx.x / num_tiles;
  const int tile = blockIdx.x - bid * num_tiles;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  KVGradRef<scalar_t> r;
  if (!res.resolve(b, h, bid, r)) return;  // not resident / out of range / base == 0
  const int tile0 = tile * TKV;
  if (tile0 >= r.n_valid) return;
  const int ntok = min(TKV, r.n_valid - tile0);

  float* sK = smem;
  float* sV = sK + static_cast<size_t>(TKV) * pitch;
  float* sDK = sV + static_cast<size_t>(TKV) * pitch;
  float* sDV = sDK + static_cast<size_t>(TKV) * pitch;
  float* sQ = sDV + static_cast<size_t>(TKV) * pitch;
  float* sDO = sQ + static_cast<size_t>(BQ) * pitch;
  float* sP = sDO + static_cast<size_t>(BQ) * pitch;
  float* sDS = sP + static_cast<size_t>(BQ) * TKV;
  float* sLse = sDS + static_cast<size_t>(BQ) * TKV;
  float* sDel = sLse + BQ;
  int* sQidx = reinterpret_cast<int*>(sDel + BQ);
  int* sQchunk = sQidx + BQ;

  for (int idx = tid; idx < TKV * D; idx += NT) {
    const int j = idx / D;
    const int d = idx - j * D;
    float kv = 0.f, vv = 0.f;
    if (j < ntok) {
      const int64_t tk = tile0 + j;
      kv = to_f32(r.k[tk * r.k_sn + static_cast<int64_t>(d) * r.k_sd]);
      vv = to_f32(r.v[tk * r.v_sn + static_cast<int64_t>(d) * r.v_sd]);
    }
    sK[j * pitch + d] = from_f32<scalar_t>(kv);
    sV[j * pitch + d] = from_f32<scalar_t>(vv);
    sDK[j * pitch + d] = 0.f;
    sDV[j * pitch + d] = 0.f;
  }

  const int64_t obase = static_cast<int64_t>(b) * off_sb + static_cast<int64_t>(h) * off_sh;
  const int start = offsets[obase + bid];
  const int cnt = offsets[obase + bid + 1] - start;
  const int64_t gbase =
      static_cast<int64_t>(b) * groups_sb + static_cast<int64_t>(h) * groups_sh + start;
  const int64_t total_q = static_cast<int64_t>(cnt) * G;

  for (int64_t qt = 0; qt < total_q; qt += BQ) {
    const int64_t left = total_q - qt;
    const int nq = static_cast<int>(left < BQ ? left : BQ);
    __syncthreads();
    for (int i = tid; i < BQ; i += NT) {
      int m = -1, qc = 0;
      float l = kNegInf, dl = 0.f;
      if (i < nq) {
        const int64_t f = qt + i;
        const int64_t e = f / G;
        const int gid = groups[gbase + e];
        m = static_cast<int>(static_cast<int64_t>(gid) * G + (f - e * G));
        qc = static_cast<int>((static_cast<int64_t>(gid) * G) / chunk_size);
        l = *lse.at(b, h, m);
        dl = *delta.at(b, h, m);
      }
      sQidx[i] = m;
      sQchunk[i] = qc;
      sLse[i] = l;
      sDel[i] = dl;
    }
    __syncthreads();
    for (int idx = tid; idx < BQ * D; idx += NT) {
      const int i = idx / D;
      const int d = idx - i * D;
      const int m = sQidx[i];
      float qv = 0.f, ov = 0.f;
      if (m >= 0) {
        qv = to_f32(*q.at(b, m, h, d));
        ov = to_f32(*dout.at(b, m, h, d));
      }
      sQ[i * pitch + d] = qv;
      sDO[i * pitch + d] = ov;
    }
    __syncthreads();
    // Two query rows per thread (rows beyond nq carry LSE = -inf, so P = dS = 0).
    for (int idx = tid; idx < (BQ >> 1) * TKV; idx += NT) {
      const int ib = idx / TKV;
      const int j = idx - ib * TKV;
      const int i0 = 2 * ib, i1 = i0 + 1;
      float p0 = 0.f, p1 = 0.f, ds0 = 0.f, ds1 = 0.f;
      if (j < ntok) {
        const int tok = tile0 + j;
        bool v0 = true, v1 = true;
        if (r.token0 >= 0) {  // dense + causal: (block*BS + tok)//C <= (g*G)//C
          v0 = chunk_visible(r.token0 + tok, chunk_size, sQchunk[i0]);
          v1 = chunk_visible(r.token0 + tok, chunk_size, sQchunk[i1]);
        }
        const float l0 = sLse[i0], l1 = sLse[i1];
        v0 = v0 && (l0 > kNegInf);
        v1 = v1 && (l1 > kNegInf);
        if (v0 || v1) {
          float s0 = 0.f, s1 = 0.f, dp0 = 0.f, dp1 = 0.f;
          for (int d = 0; d < D; ++d) {  // S = Q K^T, dP = dO V^T
            const float kk = sK[j * pitch + d];
            const float vv = sV[j * pitch + d];
            s0 += sQ[i0 * pitch + d] * kk;
            s1 += sQ[i1 * pitch + d] * kk;
            dp0 += sDO[i0 * pitch + d] * vv;
            dp1 += sDO[i1 * pitch + d] * vv;
          }
          if (v0) { p0 = fast_exp(s0 * scale - l0); ds0 = p0 * (dp0 - sDel[i0]) * scale; }
          if (v1) { p1 = fast_exp(s1 * scale - l1); ds1 = p1 * (dp1 - sDel[i1]) * scale; }
        }
      }
      sP[i0 * TKV + j] = p0;
      sP[i1 * TKV + j] = p1;
      sDS[i0 * TKV + j] = ds0;
      sDS[i1 * TKV + j] = ds1;
    }
    __syncthreads();
    // Two KV rows per thread: dO/Q words are reused across both, accumulators independent.
    for (int idx = tid; idx < (TKV >> 1) * D; idx += NT) {
      const int jb = idx / D;
      const int d = idx - jb * D;
      const int j0 = 2 * jb, j1 = j0 + 1;
      float av0 = 0.f, av1 = 0.f, ak0 = 0.f, ak1 = 0.f;
      // dV += P^T dO and dK += dS^T Q
      for (int i = 0; i < BQ; ++i) {
        const float ov = sDO[i * pitch + d];
        const float qv = sQ[i * pitch + d];
        av0 += sP[i * TKV + j0] * ov;
        av1 += sP[i * TKV + j1] * ov;
        ak0 += sDS[i * TKV + j0] * qv;
        ak1 += sDS[i * TKV + j1] * qv;
      }
      sDV[j0 * pitch + d] += av0;
      sDV[j1 * pitch + d] += av1;
      sDK[j0 * pitch + d] += ak0;
      sDK[j1 * pitch + d] += ak1;
    }
  }

  __syncthreads();
  for (int idx = tid; idx < TKV * D; idx += NT) {
    const int j = idx / D;
    const int d = idx - j * D;
    if (j < ntok) {
      const int64_t tk = tile0 + j;
      r.dk[tk * r.dk_sn + static_cast<int64_t>(d) * r.dk_sd] = sDK[j * pitch + d];
      r.dv[tk * r.dv_sn + static_cast<int64_t>(d) * r.dv_sd] = sDV[j * pitch + d];
    }
  }
}

// dQ (MMA): one CTA per (b, sorted group slot, h, row-tile), 8 warps, used when TF32 is permitted.
// Visible tokens of all T blocks are packed into one column list, so there is no inner mask.
template <typename scalar_t, typename AccT>
__global__ __launch_bounds__(256) void sel_bwd_dq_mma_kernel(
    Strided4<const scalar_t> q, Strided4<const scalar_t> o, Strided4<const scalar_t> dout,
    Strided4<scalar_t> dq, Strided3<const float> lse, Strided3<float> delta,
    const int32_t* __restrict__ order, int64_t ord_sb, int64_t ord_sg, int64_t ord_sh,
    AccT acc, int64_t k_sn, int64_t k_sd, int64_t v_sn, int64_t v_sd, int G, int T, int D,
    int DP, int chunk_size, float scale, int BGP, int TN, int pitch, int dsp,
    int n_row_tiles, bool kv_vec) {
  extern __shared__ char smem_raw[];
  const int tid = threadIdx.x;
  const int NT = blockDim.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid >> 5;
  const int nwarps = NT >> 5;

  const int gs = blockIdx.x / n_row_tiles;
  const int rt = blockIdx.x - gs * n_row_tiles;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int g = (order == nullptr)
                    ? gs
                    : order[static_cast<int64_t>(b) * ord_sb + static_cast<int64_t>(gs) * ord_sg +
                            static_cast<int64_t>(h) * ord_sh];
  const int r0 = rt * BGP;
  if (r0 >= G) return;
  const int rows = min(BGP, G - r0);
  const int64_t m0 = static_cast<int64_t>(g) * G + r0;
  const int q_chunk = static_cast<int>((static_cast<int64_t>(g) * G) / chunk_size);

  const scalar_t** sKp = reinterpret_cast<const scalar_t**>(smem_raw);
  const scalar_t** sVp = sKp + T;
  int* sStart = reinterpret_cast<int*>(sVp + T);   // [T+1] counts -> exclusive prefix
  int* sColT = sStart + (T + 1);                   // [TN] block slot of each column
  int* sColTok = sColT + TN;                       // [TN] token inside that block
  __shared__ unsigned s_misalign;                  // OR of the resolved base pointers
  __shared__ int s_first_valid;                    // any valid slot: cp.async needs a legal source
  if (tid == 0) {
    s_misalign = 0u;
    s_first_valid = -1;
  }
  __syncthreads();
  float* sDQ = reinterpret_cast<float*>(sColTok + TN);          // FP32 accumulator
  float* sS = sDQ + static_cast<size_t>(BGP) * pitch;
  float* sDP = sS + static_cast<size_t>(BGP) * dsp;
  float* sLse = sDP + static_cast<size_t>(BGP) * dsp;
  float* sDel = sLse + BGP;
  scalar_t* sQ = align16<scalar_t>(sDel + BGP);                 // staged in input dtype
  scalar_t* sDO = sQ + static_cast<size_t>(BGP) * pitch;
  scalar_t* sK = sDO + static_cast<size_t>(BGP) * pitch;
  scalar_t* sV = sK + static_cast<size_t>(TN) * pitch;

  for (int idx = tid; idx < BGP * DP; idx += NT) {
    const int i = idx / DP;
    const int d = idx - i * DP;
    float qv = 0.f, ov = 0.f;
    if (i < rows && d < D) {
      const int64_t m = m0 + i;
      qv = to_f32(*q.at(b, m, h, d));
      ov = to_f32(*dout.at(b, m, h, d));
    }
    sQ[i * pitch + d] = from_f32<scalar_t>(qv);
    sDO[i * pitch + d] = from_f32<scalar_t>(ov);
    sDQ[i * pitch + d] = 0.f;
  }
  for (int i = tid; i < BGP; i += NT) {
    sLse[i] = kNegInf;
    sDel[i] = 0.f;
  }
  for (int t = tid; t < T; t += NT) {
    const BlockView<scalar_t> bv = acc.block(b, g, h, t, q_chunk);
    int nv = 0;
    if (bv.valid()) {
      nv = bv.n_valid;
      if (bv.token0 >= 0) {  // chunk-causal: visible tokens are the prefix below the limit
        int64_t lim = static_cast<int64_t>(q_chunk + 1) * chunk_size - bv.token0;
        if (lim < 0) lim = 0;
        if (lim < nv) nv = static_cast<int>(lim);
      }
      sKp[t] = bv.k;
      sVp[t] = bv.v;
      if (nv > 0) {
        atomicOr(&s_misalign, static_cast<unsigned>(
                                  (reinterpret_cast<uintptr_t>(bv.k) |
                                   reinterpret_cast<uintptr_t>(bv.v)) & 15));
        atomicMax(&s_first_valid, t);
      }
    } else {
      sKp[t] = nullptr;
      sVp[t] = nullptr;
    }
    sStart[t] = nv;
  }
  __syncthreads();
  const bool vec = kv_vec && (s_misalign == 0u) && (s_first_valid >= 0);
  const scalar_t* kfall = (s_first_valid >= 0) ? sKp[s_first_valid] : nullptr;
  const scalar_t* vfall = (s_first_valid >= 0) ? sVp[s_first_valid] : nullptr;

  for (int i = warp; i < rows; i += nwarps) {  // delta = sum_d out*dO, FP32
    const int64_t m = m0 + i;
    float acc_d = 0.f;
    for (int d = lane; d < D; d += kWarpSize)
      acc_d += to_f32(*o.at(b, m, h, d)) * to_f32(sDO[i * pitch + d]);
    acc_d = warp_sum(acc_d);
    if (lane == 0) {
      sDel[i] = acc_d;
      sLse[i] = *lse.at(b, h, m);
      *delta.at(b, h, m) = acc_d;
    }
  }
  if (warp == 0) {  // exclusive scan of the per-block visible counts
    int run = 0;
    for (int base = 0; base < T; base += kWarpSize) {
      const int t = base + lane;
      const int v = (t < T) ? sStart[t] : 0;
      int x = v;
#pragma unroll
      for (int off = 1; off < kWarpSize; off <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, off);
        if (lane >= off) x += y;
      }
      const int excl = run + x - v;
      run += __shfl_sync(0xffffffffu, x, kWarpSize - 1);
      __syncwarp();
      if (t < T) sStart[t] = excl;
      __syncwarp();
    }
    if (lane == 0) sStart[T] = run;
  }
  __syncthreads();
  const int total = sStart[T];

  for (int c0 = 0; c0 < total; c0 += TN) {
    const int ncol = min(TN, total - c0);
    __syncthreads();
    for (int j = tid; j < TN; j += NT) {  // column -> (block slot, token) by binary search
      int tt = -1, tok = 0;
      if (j < ncol) {
        const int c = c0 + j;
        int lo = 0, hi = T - 1;
        while (lo < hi) {
          const int mid = (lo + hi + 1) >> 1;
          if (sStart[mid] <= c) lo = mid; else hi = mid - 1;
        }
        tt = lo;
        tok = c - sStart[lo];
      }
      sColT[j] = tt;
      sColTok[j] = tok;
    }
    __syncthreads();
    if (vec) {
      // 16-byte cp.async per chunk; (row, chunk) advances incrementally, so no integer division.
      constexpr int kVec = cp_vec<scalar_t>();
      const int cpr = DP / kVec;
      const int adv_j = NT / cpr, adv_d = NT - (NT / cpr) * cpr;
      int j = tid / cpr, dc = tid - (tid / cpr) * cpr;
      for (int c = tid; c < TN * cpr; c += NT) {
        const int d = dc * kVec;
        const int tt = sColT[j];
        const scalar_t* ks = kfall;
        const scalar_t* vs = vfall;
        int nb = 0;
        if (tt >= 0) {
          const int64_t tok = sColTok[j];
          ks = sKp[tt] + tok * k_sn + d;
          vs = sVp[tt] + tok * v_sn + d;
          nb = (D - d) * static_cast<int>(sizeof(scalar_t));
          if (nb > 16) nb = 16;
          if (nb < 0) nb = 0;
        }
        cp_async16(&sK[j * pitch + d], ks, nb);
        cp_async16(&sV[j * pitch + d], vs, nb);
        dc += adv_d;
        j += adv_j;
        if (dc >= cpr) { dc -= cpr; ++j; }
      }
      cp_async_fence();
      cp_async_wait_all();
    } else {
      for (int idx = tid; idx < TN * DP; idx += NT) {
        const int j = idx / DP;
        const int d = idx - j * DP;
        float kv = 0.f, vv = 0.f;
        const int tt = sColT[j];
        if (tt >= 0 && d < D) {
          const int64_t tok = sColTok[j];
          kv = to_f32(sKp[tt][tok * k_sn + static_cast<int64_t>(d) * k_sd]);
          vv = to_f32(sVp[tt][tok * v_sn + static_cast<int64_t>(d) * v_sd]);
        }
        sK[j * pitch + d] = from_f32<scalar_t>(kv);
        sV[j * pitch + d] = from_f32<scalar_t>(vv);
      }
    }
    __syncthreads();
    // S = Q K^T and dP = dO V^T ; B is read n-major straight out of the [TN, pitch] tiles.
    cta_mma_gemm<false>(sS, dsp, sQ, pitch, 1, sK, 1, pitch, BGP, TN, DP, warp, nwarps, lane);
    cta_mma_gemm<false>(sDP, dsp, sDO, pitch, 1, sV, 1, pitch, BGP, TN, DP, warp, nwarps, lane);
    __syncthreads();
    for (int idx = tid; idx < BGP * TN; idx += NT) {
      const int i = idx / TN;
      const int j = idx - i * TN;
      float ds = 0.f;
      const float li = sLse[i];
      if (j < ncol && li > kNegInf)
        ds = fast_exp(sS[i * dsp + j] * scale - li) * (sDP[i * dsp + j] - sDel[i]) * scale;
      sS[i * dsp + j] = ds;
    }
    __syncthreads();
    // dQ += dS K ; here K is read k-major (k = column index) from the same tile.
    cta_mma_gemm<true>(sDQ, pitch, sS, dsp, 1, sK, pitch, 1, BGP, DP, TN, warp, nwarps, lane);
  }

  __syncthreads();
  for (int idx = tid; idx < rows * D; idx += NT) {
    const int i = idx / D;
    const int d = idx - i * D;
    *dq.at(b, m0 + i, h, d) = from_f32<scalar_t>(sDQ[i * pitch + d]);
  }
}

// dQ (warp-private): warp (mt, nt) owns queries [16mt, 16mt+16) x tokens [8nt, 8nt+8).
// sK serves two B patterns; mma_pitch (== 8 mod 16) is conflict free for both at 2-byte elements.
template <typename scalar_t, typename AccT, int DN>
__global__ __launch_bounds__(512) void sel_bwd_dq_wmma_kernel(
    Strided4<const scalar_t> q, Strided4<const scalar_t> o, Strided4<const scalar_t> dout,
    Strided4<scalar_t> dq, Strided3<const float> lse, Strided3<float> delta,
    const int32_t* __restrict__ order, int64_t ord_sb, int64_t ord_sg, int64_t ord_sh,
    AccT acc, const int H, const int D, const int G, const int Gc, const int Tb,
    const int chunk_size, const float scale) {
  extern __shared__ __align__(16) char smem_raw[];
  constexpr int TILE = 32;
  const int ldq = mma_pitch_a(D);        // A-fragment pitch for Q / dO
  const int ldkv = mma_pitch((D + 7) & ~7);  // B-fragment pitch for K / V
  const int ldo = DN * 8 + 4;
  const int nm = (Gc + 15) >> 4, nmq = nm * 16;

  float* sLse = reinterpret_cast<float*>(smem_raw);           // [nmq]
  float* sDel = sLse + nmq;                                   // [nmq]
  scalar_t* sQ = align16<scalar_t>(sDel + nmq);               // [nmq][ldq]
  scalar_t* sDO = sQ + static_cast<size_t>(nmq) * ldq;        // [nmq][ldq]
  scalar_t* sK = align16<scalar_t>(sDO + static_cast<size_t>(nmq) * ldq);
  scalar_t* sV = sK + static_cast<size_t>(TILE) * ldkv;
  float* sDQ = reinterpret_cast<float*>(sK);                  // overlays the tiles at the end

  const int bh = blockIdx.z;
  const int b = bh / H, h = bh - b * H;
  const int gs = blockIdx.x;
  const int g = (order == nullptr)
                    ? gs
                    : order[static_cast<int64_t>(b) * ord_sb +
                            static_cast<int64_t>(gs) * ord_sg + static_cast<int64_t>(h) * ord_sh];
  const int q0 = blockIdx.y * Gc;
  const int nq = min(Gc, G - q0);
  if (nq <= 0) return;

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int mt = warp >> 2, nt = warp & 3, t0 = nt * 8;
  const int gid = lane >> 2, t4 = lane & 3;
  const int64_t m_base = static_cast<int64_t>(g) * G + q0;
  const int KN = (D + 7) >> 3;
  const int q_chunk = static_cast<int>((static_cast<int64_t>(g) * G) / chunk_size);

  // stage Q / dO in input dtype (bf16 -> TF32 is a shift at fragment-load time)
  for (int i = warp; i < nmq; i += nw) {
    scalar_t* dQr = sQ + static_cast<size_t>(i) * ldq;
    scalar_t* dOr = sDO + static_cast<size_t>(i) * ldq;
    if (i < nq) {
      const int64_t m = m_base + i;
      for (int d = lane; d < D; d += 32) {
        dQr[d] = *q.at(b, m, h, d);
        dOr[d] = *dout.at(b, m, h, d);
      }
      for (int d = D + lane; d < ldq; d += 32) {
        dQr[d] = from_f32<scalar_t>(0.f);
        dOr[d] = from_f32<scalar_t>(0.f);
      }
    } else {
      for (int d = lane; d < ldq; d += 32) {
        dQr[d] = from_f32<scalar_t>(0.f);
        dOr[d] = from_f32<scalar_t>(0.f);
      }
    }
  }
  for (int i = threadIdx.x; i < nmq; i += blockDim.x) {
    sLse[i] = kNegInf;
    sDel[i] = 0.f;
  }
  {  // zero the tiles once so masked columns see finite (stale at worst) K / V
    const scalar_t z = from_f32<scalar_t>(0.f);
    for (int i = threadIdx.x; i < 2 * TILE * ldkv; i += blockDim.x) sK[i] = z;
  }
  __syncthreads();

  // delta = sum_d out*dO in FP32, still written to the [B,H,M] buffer the dK/dV kernel reads
  for (int i = warp; i < nq; i += nw) {
    const int64_t m = m_base + i;
    float dacc = 0.f;
    for (int d = lane; d < D; d += kWarpSize)
      dacc += to_f32(*o.at(b, m, h, d)) * to_f32(sDO[static_cast<size_t>(i) * ldq + d]);
    dacc = warp_sum(dacc);
    if (lane == 0) {
      sDel[i] = dacc;
      sLse[i] = *lse.at(b, h, m);
      *delta.at(b, h, m) = dacc;
    }
  }

  float dqa[DN][4];
#pragma unroll
  for (int n = 0; n < DN; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dqa[n][e] = 0.f;

  constexpr int kVec = cp_vec<scalar_t>();
  const int DPad = (D + 7) & ~7;
  const int cpr = DPad / kVec;
  const int rpp = (cpr > 0 && cpr <= static_cast<int>(blockDim.x))
                      ? static_cast<int>(blockDim.x) / cpr : 0;
  const int crow = (rpp > 0) ? static_cast<int>(threadIdx.x) / cpr : 0;
  const int cchunk = (rpp > 0) ? static_cast<int>(threadIdx.x) - crow * cpr : 0;
  const int crow_a = (rpp > 0 && crow < rpp) ? crow : TILE;
  __syncthreads();

  // Chunk-causal visibility is a prefix, so one division per block gives the visible count.
  auto vis_of = [&](const BlockView<scalar_t>& bx) -> int {
    if (bx.token0 < 0) return bx.n_valid;
    const int64_t lim = static_cast<int64_t>(q_chunk + 1) * chunk_size - bx.token0;
    if (lim <= 0) return 0;
    return (lim < bx.n_valid) ? static_cast<int>(lim) : bx.n_valid;
  };

  auto issue = [&](const BlockView<scalar_t>& bx, int tk0, int vis) {
    const int64_t vsn = acc.v_stride_n(), vsd = acc.v_stride_d();
    const int nt2 = min(TILE, vis - tk0);
    const bool va16 = (bx.stride_d == 1) && (vsd == 1) &&
                      ((reinterpret_cast<uintptr_t>(bx.k) & 15) == 0) &&
                      ((reinterpret_cast<uintptr_t>(bx.v) & 15) == 0) &&
                      (((bx.stride_n * static_cast<int64_t>(sizeof(scalar_t))) & 15) == 0) &&
                      (((vsn * static_cast<int64_t>(sizeof(scalar_t))) & 15) == 0);
    if (va16 && rpp > 0) {
      for (int r = crow_a; r < nt2; r += rpp) {
        const int d = cchunk * kVec;
        int nb = (D - d) * static_cast<int>(sizeof(scalar_t));
        nb = nb < 0 ? 0 : (nb > 16 ? 16 : nb);
        cp_async16(&sK[static_cast<size_t>(r) * ldkv + d],
                   bx.k + static_cast<int64_t>(tk0 + r) * bx.stride_n + d, nb);
        cp_async16(&sV[static_cast<size_t>(r) * ldkv + d],
                   bx.v + static_cast<int64_t>(tk0 + r) * vsn + d, nb);
      }
      cp_async_fence();
    } else {
      for (int idx = threadIdx.x; idx < nt2 * DPad; idx += blockDim.x) {
        const int r = idx / DPad, d = idx - r * DPad;
        const int64_t tr = static_cast<int64_t>(tk0 + r);
        float kv = 0.f, vv = 0.f;
        if (d < D) {
          kv = to_f32(bx.k[tr * bx.stride_n + static_cast<int64_t>(d) * bx.stride_d]);
          vv = to_f32(bx.v[tr * vsn + static_cast<int64_t>(d) * vsd]);
        }
        sK[static_cast<size_t>(r) * ldkv + d] = from_f32<scalar_t>(kv);
        sV[static_cast<size_t>(r) * ldkv + d] = from_f32<scalar_t>(vv);
      }
      cp_async_fence();
    }
  };

  int t = 0, tok0 = 0, bvis = 0;
  BlockView<scalar_t> bv{nullptr, nullptr, 0, 0, 0, -1};
  bool have = false;
  for (; t < Tb; ++t) {
    bv = acc.block(b, g, h, t, q_chunk);
    bvis = bv.valid() ? vis_of(bv) : 0;
    if (bvis > 0) { have = true; break; }
  }
  if (have) issue(bv, 0, bvis);

  while (have) {
    BlockView<scalar_t> nbv = bv;
    int nt_ = t, ntok0 = tok0 + TILE, nvis = bvis;
    bool nhave = ntok0 < bvis;
    if (!nhave) {
      ntok0 = 0;
      for (nt_ = t + 1; nt_ < Tb; ++nt_) {
        nbv = acc.block(b, g, h, nt_, q_chunk);
        nvis = nbv.valid() ? vis_of(nbv) : 0;
        if (nvis > 0) { nhave = true; break; }
      }
    }
    cp_async_wait_all();
    __syncthreads();
    {
      const int ntile = min(TILE, bvis - tok0);   // visible prefix only
      // S = Q Kt and dP = dO Vt over this warp's 8 tokens
      float sf[4] = {0.f, 0.f, 0.f, 0.f}, pf[4] = {0.f, 0.f, 0.f, 0.f};
      {
        const scalar_t* qr0 = sQ + static_cast<size_t>(mt * 16 + gid) * ldq;
        const scalar_t* qr1 = qr0 + static_cast<size_t>(8) * ldq;
        const scalar_t* or0 = sDO + static_cast<size_t>(mt * 16 + gid) * ldq;
        const scalar_t* or1 = or0 + static_cast<size_t>(8) * ldq;
        const scalar_t* kr = sK + static_cast<size_t>(t0 + gid) * ldkv;
        const scalar_t* vr = sV + static_cast<size_t>(t0 + gid) * ldkv;
        for (int kc = 0; kc < KN; ++kc) {
          const int d0 = kc * 8;
          uint32_t a[4], bK[2], c[4], bV[2];
          a[0] = to_tf32(qr0[d0 + t4]);       a[1] = to_tf32(qr1[d0 + t4]);
          a[2] = to_tf32(qr0[d0 + t4 + 4]);   a[3] = to_tf32(qr1[d0 + t4 + 4]);
          bK[0] = to_tf32(kr[d0 + t4]);       bK[1] = to_tf32(kr[d0 + t4 + 4]);
          mma_m16n8k8_tf32(sf, a, bK);
          c[0] = to_tf32(or0[d0 + t4]);       c[1] = to_tf32(or1[d0 + t4]);
          c[2] = to_tf32(or0[d0 + t4 + 4]);   c[3] = to_tf32(or1[d0 + t4 + 4]);
          bV[0] = to_tf32(vr[d0 + t4]);       bV[1] = to_tf32(vr[d0 + t4 + 4]);
          mma_m16n8k8_tf32(pf, c, bV);
        }
      }
      // dS = P * (dP - delta) * scale, P recomputed from the saved LSE
      float ds[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = mt * 16 + gid + 8 * (e >> 1);
        const int loc = t0 + 2 * t4 + (e & 1);
        const bool ok = loc < ntile;   // `ntile` already counts only visible tokens
        const float li = sLse[row];
        const float pv = (ok && li > kNegInf) ? fast_exp(sf[e] * scale - li) : 0.f;
        ds[e] = pv * (pf[e] - sDel[row]) * scale;
      }
      // dQ += dS K (C -> A lane permutation, 8 shuffles per tile)
      {
        const int s1 = (lane & ~3) + (t4 >> 1), s2 = s1 + 2;
        const bool odd = (t4 & 1) != 0;
        const float x0 = __shfl_sync(0xffffffffu, ds[0], s1);
        const float x1 = __shfl_sync(0xffffffffu, ds[1], s1);
        const float y0 = __shfl_sync(0xffffffffu, ds[2], s1);
        const float y1 = __shfl_sync(0xffffffffu, ds[3], s1);
        const float z0 = __shfl_sync(0xffffffffu, ds[0], s2);
        const float z1 = __shfl_sync(0xffffffffu, ds[1], s2);
        const float w0 = __shfl_sync(0xffffffffu, ds[2], s2);
        const float w1 = __shfl_sync(0xffffffffu, ds[3], s2);
        uint32_t sa[4];
        sa[0] = f2tf32(odd ? x1 : x0);
        sa[1] = f2tf32(odd ? y1 : y0);
        sa[2] = f2tf32(odd ? z1 : z0);
        sa[3] = f2tf32(odd ? w1 : w0);
        const scalar_t* kr0 = sK + static_cast<size_t>(t0 + t4) * ldkv;
        const scalar_t* kr1 = sK + static_cast<size_t>(t0 + t4 + 4) * ldkv;
#pragma unroll
        for (int n = 0; n < DN; ++n) {
          uint32_t bk[2];
          bk[0] = to_tf32(kr0[n * 8 + gid]);
          bk[1] = to_tf32(kr1[n * 8 + gid]);
          mma_m16n8k8_tf32(dqa[n], sa, bk);
        }
      }
    }
    __syncthreads();
    if (!nhave) break;
    issue(nbv, ntok0, nvis);
    bv = nbv; t = nt_; tok0 = ntok0; bvis = nvis;
  }

  // sum the 4 partial dQ tiles of each m-tile, store in q.dtype
  for (int i = threadIdx.x; i < nmq * ldo; i += blockDim.x) sDQ[i] = 0.f;
  __syncthreads();
  for (int u = 0; u < 4; ++u) {
    if (nt == u) {
#pragma unroll
      for (int n = 0; n < DN; ++n) {
        const int d0 = n * 8 + 2 * t4;
        if (d0 < D) {
          float* r0 = sDQ + static_cast<size_t>(mt * 16 + gid) * ldo + d0;
          float* r1 = sDQ + static_cast<size_t>(mt * 16 + gid + 8) * ldo + d0;
          r0[0] += dqa[n][0];
          r1[0] += dqa[n][2];
          if (d0 + 1 < D) { r0[1] += dqa[n][1]; r1[1] += dqa[n][3]; }
        }
      }
    }
    __syncthreads();
  }
  for (int row = warp; row < nq; row += nw) {
    const float* sr = sDQ + static_cast<size_t>(row) * ldo;
    for (int d = lane; d < D; d += 32)
      *dq.at(b, m_base + row, h, d) = from_f32<scalar_t>(sr[d]);
  }
}

// dQ (warp-per-query, small G): one warp owns a whole query group and never synchronises.
// Phase A: lanes = tokens compute S and dP; phase B: lanes = dim slices accumulate dQ.
template <typename scalar_t, typename AccT, int GMAX>
__global__ __launch_bounds__(256) void sel_bwd_dq_warp_kernel(
    Strided4<const scalar_t> q, Strided4<const scalar_t> o, Strided4<const scalar_t> dout,
    Strided4<scalar_t> dq, Strided3<const float> lse, Strided3<float> delta,
    const int32_t* __restrict__ order, int64_t ord_sb, int64_t ord_sg, int64_t ord_sh,
    AccT acc, int64_t k_sn, int64_t k_sd, int64_t v_sn, int64_t v_sd, int MG, int G, int T,
    int D, int chunk_size, float scale, int wpb, bool row_vec, int warp_bytes) {
  extern __shared__ float wsmem[];
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;
  const int gs = blockIdx.x * wpb + warp;
  if (gs >= MG) return;  // warp-uniform: this kernel has no __syncthreads at all
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int g = (order == nullptr)
                    ? gs
                    : order[static_cast<int64_t>(b) * ord_sb + static_cast<int64_t>(gs) * ord_sg +
                            static_cast<int64_t>(h) * ord_sh];

  // Per-warp slice: pointers first (8-byte aligned), then ints, then the FP32 rows.
  char* wbase = reinterpret_cast<char*>(wsmem) + static_cast<size_t>(warp) * warp_bytes;
  const scalar_t** sKp = reinterpret_cast<const scalar_t**>(wbase);          // [T]
  const scalar_t** sVp = sKp + T;                                            // [T]
  const scalar_t** sColK = sVp + T;                                          // [32]
  int* sStart = reinterpret_cast<int*>(sColK + kWarpSize);                   // [T+1]
  float* sQ = reinterpret_cast<float*>(sStart + T + 1);
  float* sDO = sQ + G * D;
  float* sDQ = sDO + G * D;
  const int64_t m0 = static_cast<int64_t>(g) * G;
  const int q_chunk = static_cast<int>(m0 / chunk_size);

  for (int idx = lane; idx < G * D; idx += kWarpSize) {
    const int i = idx / D;
    const int d = idx - i * D;
    sQ[idx] = to_f32(*q.at(b, m0 + i, h, d));
    sDO[idx] = to_f32(*dout.at(b, m0 + i, h, d));
    sDQ[idx] = 0.f;
  }
  __syncwarp();

  float lse_r[GMAX], del_r[GMAX];
#pragma unroll
  for (int i = 0; i < GMAX; ++i) { lse_r[i] = kNegInf; del_r[i] = 0.f; }
  for (int i = 0; i < G; ++i) {  // delta = sum_d out * dO, FP32, one warp reduction per row
    float a = 0.f;
    for (int d = lane; d < D; d += kWarpSize)
      a += to_f32(*o.at(b, m0 + i, h, d)) * sDO[i * D + d];
    a = warp_sum(a);
    del_r[i] = a;
    lse_r[i] = *lse.at(b, h, m0 + i);
    if (lane == 0) *delta.at(b, h, m0 + i) = a;
  }

  constexpr int kVec = 16 / sizeof(scalar_t);
  // Resolve every selected block once and warp-scan visible counts into one dense column list.
  unsigned misalign = 0u;
  int total = 0;
  for (int base = 0; base < T; base += kWarpSize) {
    const int t = base + lane;
    int nv = 0;
    if (t < T) {
      const BlockView<scalar_t> bv = acc.block(b, g, h, t, q_chunk);
      if (bv.valid()) {
        nv = bv.n_valid;
        if (bv.token0 >= 0) {  // chunk-causal: visible tokens are a prefix
          int64_t lim = static_cast<int64_t>(q_chunk + 1) * chunk_size - bv.token0;
          if (lim < 0) lim = 0;
          if (lim < nv) nv = static_cast<int>(lim);
        }
        if (nv > 0)
          misalign |= static_cast<unsigned>((reinterpret_cast<uintptr_t>(bv.k) |
                                             reinterpret_cast<uintptr_t>(bv.v)) & 15);
        sKp[t] = bv.k;
        sVp[t] = bv.v;
      } else {
        nv = 0;
        sKp[t] = nullptr;
        sVp[t] = nullptr;
      }
    }
    int x = nv;
#pragma unroll
    for (int off = 1; off < kWarpSize; off <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, x, off);
      if (lane >= off) x += y;
    }
    if (t < T) sStart[t] = total + x - nv;
    total += __shfl_sync(0xffffffffu, x, kWarpSize - 1);
  }
  if (lane == 0) sStart[T] = total;
#pragma unroll
  for (int off = 1; off < kWarpSize; off <<= 1)
    misalign |= __shfl_xor_sync(0xffffffffu, misalign, off);
  const bool vec = row_vec && misalign == 0u;
  __syncwarp();

  for (int c0 = 0; c0 < total; c0 += kWarpSize) {
    const int c = c0 + lane;
    const bool live = c < total;
    const scalar_t* krow = nullptr;
    const scalar_t* vrow = nullptr;
    if (live) {  // column -> (block, token) by binary search over the prefix table
      int lo = 0, hi = T - 1;
      while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (sStart[mid] <= c) lo = mid; else hi = mid - 1;
      }
      const int64_t tok = c - sStart[lo];
      krow = sKp[lo] + tok * k_sn;
      vrow = sVp[lo] + tok * v_sn;
    }
    sColK[lane] = krow;  // phase B broadcasts the row pointers back out
    __syncwarp();

    float sv[GMAX], dpv[GMAX];
#pragma unroll
    for (int i = 0; i < GMAX; ++i) { sv[i] = 0.f; dpv[i] = 0.f; }
    if (live) {  // phase A: lanes = tokens, 16-byte K/V row loads
      int d = 0;
      if (vec) {
        for (; d + kVec <= D; d += kVec) {
          const uint4 kk = *reinterpret_cast<const uint4*>(krow + d);
          const uint4 vv = *reinterpret_cast<const uint4*>(vrow + d);
          const scalar_t* ke = reinterpret_cast<const scalar_t*>(&kk);
          const scalar_t* ve = reinterpret_cast<const scalar_t*>(&vv);
#pragma unroll
          for (int u = 0; u < kVec; ++u) {
            const float kf = to_f32(ke[u]);
            const float vf = to_f32(ve[u]);
            for (int i = 0; i < G; ++i) {
              sv[i] += sQ[i * D + d + u] * kf;
              dpv[i] += sDO[i * D + d + u] * vf;
            }
          }
        }
      }
      for (; d < D; ++d) {
        const float kf = to_f32(krow[d * k_sd]);
        const float vf = to_f32(vrow[d * v_sd]);
        for (int i = 0; i < G; ++i) {
          sv[i] += sQ[i * D + d] * kf;
          dpv[i] += sDO[i * D + d] * vf;
        }
      }
    }
    // phase B: dQ += dS K, lanes = head dim, dS broadcast from the owning lane.
    const int ntok = min(kWarpSize, total - c0);
    for (int i = 0; i < G; ++i) {
      float ds = 0.f;
      if (live && lse_r[i] > kNegInf)
        ds = fast_exp(sv[i] * scale - lse_r[i]) * (dpv[i] - del_r[i]) * scale;
      for (int jj = 0; jj < ntok; ++jj) {
        const float dsj = __shfl_sync(0xffffffffu, ds, jj);
        if (dsj == 0.f) continue;  // warp-uniform
        const scalar_t* kr = sColK[jj];
        for (int d = lane; d < D; d += kWarpSize)
          sDQ[i * D + d] += dsj * to_f32(kr[d * k_sd]);
      }
    }
    __syncwarp();
  }

  for (int idx = lane; idx < G * D; idx += kWarpSize) {
    const int i = idx / D;
    const int d = idx - i * D;
    *dq.at(b, m0 + i, h, d) = from_f32<scalar_t>(sDQ[idx]);
  }
}

// dK/dV (MMA): same tile ownership as the SIMT kernel; P and dS round to TF32 as operands.
template <typename scalar_t, typename ResT>
__global__ __launch_bounds__(256) void sel_bwd_dkv_mma_kernel(
    Strided4<const scalar_t> q, Strided4<const scalar_t> dout, Strided3<const float> lse,
    Strided3<const float> delta, const int32_t* __restrict__ groups, int64_t groups_sb,
    int64_t groups_sh, const int32_t* __restrict__ offsets, int64_t off_sb, int64_t off_sh,
    ResT res, int G, int D, int DP, int chunk_size, float scale, int num_tiles, int TKV,
    int BQ, int pitch, int sp, bool kv_vec, bool q_vec) {
  extern __shared__ char smem_raw[];
  const int tid = threadIdx.x;
  const int NT = blockDim.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid >> 5;
  const int nwarps = NT >> 5;
  const int bid = blockIdx.x / num_tiles;
  const int tile = blockIdx.x - bid * num_tiles;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  KVGradRef<scalar_t> r;
  if (!res.resolve(b, h, bid, r)) return;
  const int tile0 = tile * TKV;
  if (tile0 >= r.n_valid) return;
  const int ntok = min(TKV, r.n_valid - tile0);

  int* sQidx = reinterpret_cast<int*>(smem_raw);
  int* sQchunk = sQidx + BQ;
  float* sDK = reinterpret_cast<float*>(sQchunk + BQ);          // FP32 accumulators
  float* sDV = sDK + static_cast<size_t>(TKV) * pitch;
  float* sS = sDV + static_cast<size_t>(TKV) * pitch;
  float* sDP = sS + static_cast<size_t>(BQ) * sp;
  float* sLse = sDP + static_cast<size_t>(BQ) * sp;
  float* sDel = sLse + BQ;
  scalar_t* sK = align16<scalar_t>(sDel + BQ);                  // staged in input dtype
  scalar_t* sV = sK + static_cast<size_t>(TKV) * pitch;
  scalar_t* sQ = sV + static_cast<size_t>(TKV) * pitch;
  scalar_t* sDO = sQ + static_cast<size_t>(BQ) * pitch;

  constexpr int kVec = cp_vec<scalar_t>();
  const bool kvec = kv_vec && r.k_sd == 1 && r.v_sd == 1 &&
                    ((reinterpret_cast<uintptr_t>(r.k) | reinterpret_cast<uintptr_t>(r.v)) & 15) == 0;
  for (int idx = tid; idx < TKV * DP; idx += NT) {
    const int j = idx / DP;
    const int d = idx - j * DP;
    sDK[j * pitch + d] = 0.f;
    sDV[j * pitch + d] = 0.f;
  }
  if (kvec) {
    const int cpr = DP / kVec;
    const int adv_j = NT / cpr, adv_d = NT - (NT / cpr) * cpr;
    int j = tid / cpr, dc = tid - (tid / cpr) * cpr;
    for (int c = tid; c < TKV * cpr; c += NT) {
      const int d = dc * kVec;
      const bool live = j < ntok;
      const int64_t tk = live ? (tile0 + j) : tile0;
      int nb = live ? (D - d) * static_cast<int>(sizeof(scalar_t)) : 0;
      if (nb > 16) nb = 16;
      if (nb < 0) nb = 0;
      cp_async16(&sK[j * pitch + d], r.k + tk * r.k_sn + (live ? d : 0), nb);
      cp_async16(&sV[j * pitch + d], r.v + tk * r.v_sn + (live ? d : 0), nb);
      dc += adv_d;
      j += adv_j;
      if (dc >= cpr) { dc -= cpr; ++j; }
    }
    cp_async_fence();
    cp_async_wait_all();
  } else {
    for (int idx = tid; idx < TKV * DP; idx += NT) {
      const int j = idx / DP;
      const int d = idx - j * DP;
      float kv = 0.f, vv = 0.f;
      if (j < ntok && d < D) {
        const int64_t tk = tile0 + j;
        kv = to_f32(r.k[tk * r.k_sn + static_cast<int64_t>(d) * r.k_sd]);
        vv = to_f32(r.v[tk * r.v_sn + static_cast<int64_t>(d) * r.v_sd]);
      }
      sK[j * pitch + d] = from_f32<scalar_t>(kv);
      sV[j * pitch + d] = from_f32<scalar_t>(vv);
    }
  }

  const int64_t obase = static_cast<int64_t>(b) * off_sb + static_cast<int64_t>(h) * off_sh;
  const int start = offsets[obase + bid];
  const int cnt = offsets[obase + bid + 1] - start;
  const int64_t gbase =
      static_cast<int64_t>(b) * groups_sb + static_cast<int64_t>(h) * groups_sh + start;
  const int64_t total_q = static_cast<int64_t>(cnt) * G;

  for (int64_t qt = 0; qt < total_q; qt += BQ) {
    const int64_t left = total_q - qt;
    const int nq = static_cast<int>(left < BQ ? left : BQ);
    __syncthreads();
    for (int i = tid; i < BQ; i += NT) {
      int m = -1, qc = 0;
      float l = kNegInf, dl = 0.f;
      if (i < nq) {
        const int64_t f = qt + i;
        const int64_t e = f / G;
        const int gid = groups[gbase + e];
        m = static_cast<int>(static_cast<int64_t>(gid) * G + (f - e * G));
        qc = static_cast<int>((static_cast<int64_t>(gid) * G) / chunk_size);
        l = *lse.at(b, h, m);
        dl = *delta.at(b, h, m);
      }
      sQidx[i] = m;
      sQchunk[i] = qc;
      sLse[i] = l;
      sDel[i] = dl;
    }
    __syncthreads();
    if (q_vec) {
      const int cpr = DP / kVec;
      const int adv_i = NT / cpr, adv_d = NT - (NT / cpr) * cpr;
      int i = tid / cpr, dc = tid - (tid / cpr) * cpr;
      for (int c = tid; c < BQ * cpr; c += NT) {
        const int d = dc * kVec;
        const int m = sQidx[i];
        const int mm = (m >= 0) ? m : 0;
        int nb = (m >= 0) ? (D - d) * static_cast<int>(sizeof(scalar_t)) : 0;
        if (nb > 16) nb = 16;
        if (nb < 0) nb = 0;
        cp_async16(&sQ[i * pitch + d], q.at(b, mm, h, 0) + ((m >= 0) ? d : 0), nb);
        cp_async16(&sDO[i * pitch + d], dout.at(b, mm, h, 0) + ((m >= 0) ? d : 0), nb);
        dc += adv_d;
        i += adv_i;
        if (dc >= cpr) { dc -= cpr; ++i; }
      }
      cp_async_fence();
      cp_async_wait_all();
    } else {
      for (int idx = tid; idx < BQ * DP; idx += NT) {
        const int i = idx / DP;
        const int d = idx - i * DP;
        const int m = sQidx[i];
        float qv = 0.f, ov = 0.f;
        if (m >= 0 && d < D) {
          qv = to_f32(*q.at(b, m, h, d));
          ov = to_f32(*dout.at(b, m, h, d));
        }
        sQ[i * pitch + d] = from_f32<scalar_t>(qv);
        sDO[i * pitch + d] = from_f32<scalar_t>(ov);
      }
    }
    __syncthreads();
    cta_mma_gemm<false>(sS, sp, sQ, pitch, 1, sK, 1, pitch, BQ, TKV, DP, warp, nwarps, lane);
    cta_mma_gemm<false>(sDP, sp, sDO, pitch, 1, sV, 1, pitch, BQ, TKV, DP, warp, nwarps, lane);
    __syncthreads();
    for (int idx = tid; idx < BQ * TKV; idx += NT) {
      const int i = idx / TKV;
      const int j = idx - i * TKV;
      float p = 0.f, ds = 0.f;
      if (j < ntok) {
        bool vis = true;
        if (r.token0 >= 0)
          vis = chunk_visible(r.token0 + tile0 + j, chunk_size, sQchunk[i]);
        const float li = sLse[i];
        if (vis && li > kNegInf) {
          p = fast_exp(sS[i * sp + j] * scale - li);
          ds = p * (sDP[i * sp + j] - sDel[i]) * scale;
        }
      }
      sS[i * sp + j] = p;
      sDP[i * sp + j] = ds;
    }
    __syncthreads();
    // dV += P^T dO and dK += dS^T Q: A is read transposed straight out of [BQ, sp].
    cta_mma_gemm<true>(sDV, pitch, sS, 1, sp, sDO, pitch, 1, TKV, DP, BQ, warp, nwarps, lane);
    cta_mma_gemm<true>(sDK, pitch, sDP, 1, sp, sQ, pitch, 1, TKV, DP, BQ, warp, nwarps, lane);
  }

  __syncthreads();
  for (int idx = tid; idx < TKV * D; idx += NT) {
    const int j = idx / D;
    const int d = idx - j * D;
    if (j < ntok) {
      const int64_t tk = tile0 + j;
      r.dk[tk * r.dk_sn + static_cast<int64_t>(d) * r.dk_sd] = sDK[j * pitch + d];
      r.dv[tk * r.dv_sn + static_cast<int64_t>(d) * r.dv_sd] = sDV[j * pitch + d];
    }
  }
}

inline void check_grid(int64_t x, int64_t y, int64_t z, const char* what) {
  TORCH_CHECK(x <= 2147483647LL && y <= 65535 && z <= 65535, what, ": grid ", x, "x", y, "x", z,
              " exceeds the CUDA limits");
}

template <typename scalar_t, typename AccT>
void launch_dq(const at::Tensor& q, const at::Tensor& o, const at::Tensor& d_out,
               at::Tensor& dq, const at::Tensor& lse, at::Tensor& delta,
               const at::Tensor& order, AccT acc, int64_t B, int64_t H, int64_t MG, int64_t G,
               int64_t T, int64_t D, int64_t chunk_size, double scale, int64_t block_size,
               cudaStream_t stream, const char* what) {
  const TilePlan plan = plan_dq(static_cast<int>(D), static_cast<int>(G),
                                static_cast<int>(block_size));
  const int pitch = smem_row_pitch(static_cast<int>(D));
  const int n_row_tiles = static_cast<int>(cdiv64(G, plan.tile_q));
  check_grid(MG * n_row_tiles, H, B, what);
  auto kernel = sel_bwd_dq_kernel<scalar_t, AccT>;
  enable_large_smem(kernel, plan.smem, what);
  const dim3 grid(static_cast<unsigned>(MG * n_row_tiles), static_cast<unsigned>(H),
                  static_cast<unsigned>(B));
  kernel<<<grid, kBwdThreads, plan.smem, stream>>>(
      make_strided4_const<scalar_t>(q), make_strided4_const<scalar_t>(o),
      make_strided4_const<scalar_t>(d_out), make_strided4<scalar_t>(dq),
      Strided3<const float>{lse.data_ptr<float>(), lse.stride(0), lse.stride(1), lse.stride(2)},
      make_strided3<float>(delta), order.defined() ? order.data_ptr<int32_t>() : nullptr,
      order.defined() ? order.stride(0) : 0, order.defined() ? order.stride(1) : 0,
      order.defined() ? order.stride(2) : 0, acc, static_cast<int>(G), static_cast<int>(T),
      static_cast<int>(D), static_cast<int>(chunk_size), static_cast<float>(scale),
      plan.tile_q, plan.tile_n, pitch, n_row_tiles);
  OMNIMEM_CHECK_LAUNCH();
}

template <typename scalar_t, typename ResT>
void launch_dkv(const at::Tensor& q, const at::Tensor& d_out, const at::Tensor& lse,
                const at::Tensor& delta, const at::Tensor& groups, const at::Tensor& offsets,
                ResT res, int64_t B, int64_t H, int64_t NS, int64_t G, int64_t D,
                int64_t chunk_size, double scale, int64_t block_size, cudaStream_t stream,
                const char* what) {
  if (NS == 0) return;
  const TilePlan plan = plan_dkv(static_cast<int>(D), static_cast<int>(block_size));
  const int pitch = smem_row_pitch(static_cast<int>(D));
  const int num_tiles = static_cast<int>(cdiv64(block_size, plan.tile_n));
  check_grid(NS * num_tiles, H, B, what);
  auto kernel = sel_bwd_dkv_kernel<scalar_t, ResT>;
  enable_large_smem(kernel, plan.smem, what);
  const dim3 grid(static_cast<unsigned>(NS * num_tiles), static_cast<unsigned>(H),
                  static_cast<unsigned>(B));
  kernel<<<grid, kBwdThreads, plan.smem, stream>>>(
      make_strided4_const<scalar_t>(q), make_strided4_const<scalar_t>(d_out),
      Strided3<const float>{lse.data_ptr<float>(), lse.stride(0), lse.stride(1), lse.stride(2)},
      Strided3<const float>{delta.data_ptr<float>(), delta.stride(0), delta.stride(1),
                            delta.stride(2)},
      groups.data_ptr<int32_t>(), groups.stride(0), groups.stride(1),
      offsets.data_ptr<int32_t>(), offsets.stride(0), offsets.stride(1), res,
      static_cast<int>(G), static_cast<int>(D), static_cast<int>(chunk_size),
      static_cast<float>(scale), num_tiles, plan.tile_n, plan.tile_q, pitch);
  OMNIMEM_CHECK_LAUNCH();
}

// order = argsort(ids[..., 0]) along the group axis: a pure CTA -> group permutation for L2.
__host__ inline at::Tensor group_order(const at::Tensor& block_indices, int64_t MG,
                                       int64_t T) {
  if (MG <= 1 || T <= 0) return at::Tensor();
  return at::argsort(block_indices.select(3, 0), /*stable=*/false, /*dim=*/1,
                     /*descending=*/false)
      .to(at::kInt);
}

// Shared-memory plan for the warp-private dQ kernel (mirrors the forward's layout).
__host__ inline size_t dq_wmma_bytes(int Gc, int D, size_t esz) {
  const int nmq = ((Gc + 15) >> 4) * 16;
  const int ldq = mma_pitch_a(D), ldkv = mma_pitch((D + 7) & ~7);
  const int ldo = dq_dn_for(D) * 8 + 4;
  const size_t head = ((static_cast<size_t>(2 * nmq) * sizeof(float)) + 15) & ~size_t(15);
  const size_t stage = ((static_cast<size_t>(2 * nmq) * ldq * esz) + 15) & ~size_t(15);
  const size_t tiles = 2 * static_cast<size_t>(32) * ldkv * esz;
  const size_t dqb = static_cast<size_t>(nmq) * ldo * sizeof(float);
  return head + stage + (tiles > dqb ? tiles : dqb);
}

#define OMNIMEM_DISPATCH_DQDN(D, ...)                                     \
  [&] {                                                                   \
    if ((D) <= 64) { constexpr int DN_ = 8; return __VA_ARGS__(); }       \
    if ((D) <= 96) { constexpr int DN_ = 12; return __VA_ARGS__(); }      \
    { constexpr int DN_ = 16; return __VA_ARGS__(); }                     \
  }()

template <typename scalar_t, typename AccT>
void launch_dq_wmma(const at::Tensor& q, const at::Tensor& o, const at::Tensor& d_out,
                    at::Tensor& dq, const at::Tensor& lse, at::Tensor& delta,
                    const at::Tensor& order, AccT acc, int64_t B, int64_t H, int64_t MG,
                    int64_t G, int64_t T, int64_t D, int64_t chunk_size, double scale,
                    cudaStream_t stream, const char* what) {
  const int Di = static_cast<int>(D), Gi = static_cast<int>(G);
  int Gc = std::min(Gi, 64);
  while (Gc > 16 && dq_wmma_bytes(Gc, Di, sizeof(scalar_t)) > kMmaSmemBudget) Gc -= 16;
  const int nsplit = (Gi + Gc - 1) / Gc;
  const int nw = ((Gc + 15) >> 4) * 4;
  const size_t smem = dq_wmma_bytes(Gc, Di, sizeof(scalar_t));
  check_grid(MG, nsplit, B * H, what);
  const bool has_ord = order.defined();
  const dim3 grid(static_cast<unsigned>(MG), static_cast<unsigned>(nsplit),
                  static_cast<unsigned>(B * H));
  OMNIMEM_DISPATCH_DQDN(Di, [&] {
    auto kernel = sel_bwd_dq_wmma_kernel<scalar_t, AccT, DN_>;
    enable_large_smem(kernel, smem, what);
    kernel<<<grid, nw * 32, smem, stream>>>(
        make_strided4_const<scalar_t>(q), make_strided4_const<scalar_t>(o),
        make_strided4_const<scalar_t>(d_out), make_strided4<scalar_t>(dq),
        Strided3<const float>{lse.data_ptr<float>(), lse.stride(0), lse.stride(1),
                              lse.stride(2)},
        make_strided3<float>(delta), has_ord ? order.data_ptr<int32_t>() : nullptr,
        has_ord ? order.stride(0) : 0, has_ord ? order.stride(1) : 0,
        has_ord ? order.stride(2) : 0, acc, static_cast<int>(H), Di, Gi, Gc,
        static_cast<int>(T), static_cast<int>(chunk_size), static_cast<float>(scale));
  });
  OMNIMEM_CHECK_LAUNCH();
}

template <typename scalar_t, typename AccT>
void launch_dq_mma(const at::Tensor& q, const at::Tensor& o, const at::Tensor& d_out,
                   at::Tensor& dq, const at::Tensor& lse, at::Tensor& delta,
                   const at::Tensor& order, AccT acc, int64_t k_sn, int64_t k_sd,
                   int64_t v_sn, int64_t v_sd, int64_t B, int64_t H, int64_t MG, int64_t G,
                   int64_t T, int64_t D, int64_t chunk_size, double scale,
                   cudaStream_t stream, const char* what) {
  if (dq_wmma_eligible(static_cast<int>(G), static_cast<int>(D))) {
    launch_dq_wmma<scalar_t, AccT>(q, o, d_out, dq, lse, delta, order, acc, B, H, MG, G, T, D,
                                   chunk_size, scale, stream, what);
    return;
  }
  const MmaDqPlan plan = plan_dq_mma(static_cast<int>(D), static_cast<int>(G),
                                     static_cast<int>(T), static_cast<int>(sizeof(scalar_t)));
  const int n_row_tiles = static_cast<int>(cdiv64(G, plan.BGP));
  check_grid(MG * n_row_tiles, H, B, what);
  auto kernel = sel_bwd_dq_mma_kernel<scalar_t, AccT>;
  enable_large_smem(kernel, plan.smem, what);
  const dim3 grid(static_cast<unsigned>(MG * n_row_tiles), static_cast<unsigned>(H),
                  static_cast<unsigned>(B));
  const bool has_ord = order.defined();
  // Needed for 16-byte cp.async staging of K/V; per-block base alignment is checked in the kernel.
  const int vecn = cp_vec<scalar_t>();
  const bool kv_vec = (k_sd == 1 && v_sd == 1 && k_sn % vecn == 0 && v_sn % vecn == 0);
  kernel<<<grid, kMmaThreads, plan.smem, stream>>>(
      make_strided4_const<scalar_t>(q), make_strided4_const<scalar_t>(o),
      make_strided4_const<scalar_t>(d_out), make_strided4<scalar_t>(dq),
      Strided3<const float>{lse.data_ptr<float>(), lse.stride(0), lse.stride(1), lse.stride(2)},
      make_strided3<float>(delta), has_ord ? order.data_ptr<int32_t>() : nullptr,
      has_ord ? order.stride(0) : 0, has_ord ? order.stride(1) : 0,
      has_ord ? order.stride(2) : 0, acc, k_sn, k_sd, v_sn, v_sd, static_cast<int>(G),
      static_cast<int>(T), static_cast<int>(D), plan.DP, static_cast<int>(chunk_size),
      static_cast<float>(scale), plan.BGP, plan.TN, plan.pitch, plan.dsp, n_row_tiles,
      kv_vec);
  OMNIMEM_CHECK_LAUNCH();
}

template <typename scalar_t, typename AccT>
void launch_dq_warp(const at::Tensor& q, const at::Tensor& o, const at::Tensor& d_out,
                    at::Tensor& dq, const at::Tensor& lse, at::Tensor& delta,
                    const at::Tensor& order, AccT acc, int64_t k_sn, int64_t k_sd,
                    int64_t v_sn, int64_t v_sd, int64_t B, int64_t H, int64_t MG, int64_t G,
                    int64_t T, int64_t D, int64_t chunk_size, double scale,
                    cudaStream_t stream, const char* what) {
  // per warp: K/V/column row pointers, the prefix table, then Q/dO/dQ in FP32
  size_t per_warp = static_cast<size_t>(2 * T + 32) * sizeof(void*) +
                    static_cast<size_t>(T + 1) * sizeof(int) +
                    static_cast<size_t>(3 * G * D) * sizeof(float);
  per_warp = (per_warp + 15u) & ~static_cast<size_t>(15);
  int wpb = 8;
  while (wpb > 1 && per_warp * wpb > 48u * 1024u) wpb >>= 1;
  const size_t smem = per_warp * wpb;
  const int64_t nblocks = cdiv64(MG, wpb);
  check_grid(nblocks, H, B, what);
  auto kernel = (G == 1) ? sel_bwd_dq_warp_kernel<scalar_t, AccT, 1>
                         : sel_bwd_dq_warp_kernel<scalar_t, AccT, 8>;
  enable_large_smem(kernel, smem, what);
  const int vecn = cp_vec<scalar_t>();
  const bool row_vec = k_sd == 1 && v_sd == 1 && k_sn % vecn == 0 && v_sn % vecn == 0 &&
                       D >= vecn;
  const bool has_ord = order.defined();
  const dim3 grid(static_cast<unsigned>(nblocks), static_cast<unsigned>(H),
                  static_cast<unsigned>(B));
  kernel<<<grid, 32 * wpb, smem, stream>>>(
      make_strided4_const<scalar_t>(q), make_strided4_const<scalar_t>(o),
      make_strided4_const<scalar_t>(d_out), make_strided4<scalar_t>(dq),
      Strided3<const float>{lse.data_ptr<float>(), lse.stride(0), lse.stride(1), lse.stride(2)},
      make_strided3<float>(delta), has_ord ? order.data_ptr<int32_t>() : nullptr,
      has_ord ? order.stride(0) : 0, has_ord ? order.stride(1) : 0,
      has_ord ? order.stride(2) : 0, acc, k_sn, k_sd, v_sn, v_sd, static_cast<int>(MG),
      static_cast<int>(G), static_cast<int>(T), static_cast<int>(D),
      static_cast<int>(chunk_size), static_cast<float>(scale), wpb, row_vec,
      static_cast<int>(per_warp));
  OMNIMEM_CHECK_LAUNCH();
}

template <typename scalar_t, typename ResT>
void launch_dkv_mma(const at::Tensor& q, const at::Tensor& d_out, const at::Tensor& lse,
                    const at::Tensor& delta, const at::Tensor& groups,
                    const at::Tensor& offsets, ResT res, int64_t B, int64_t H, int64_t NS,
                    int64_t G, int64_t D, int64_t chunk_size, double scale,
                    int64_t block_size, bool kv_stride_ok, cudaStream_t stream,
                    const char* what) {
  if (NS == 0) return;
  const MmaDkvPlan plan = plan_dkv_mma(static_cast<int>(D), static_cast<int>(block_size),
                                      static_cast<int>(sizeof(scalar_t)));
  const int num_tiles = static_cast<int>(cdiv64(block_size, plan.TKV));
  check_grid(NS * num_tiles, H, B, what);
  auto kernel = sel_bwd_dkv_mma_kernel<scalar_t, ResT>;
  enable_large_smem(kernel, plan.smem, what);
  // Q / dO are gathered row by row, so their layout decides cp.async eligibility here.
  const int vecn = cp_vec<scalar_t>();
  const bool q_vec = cp_async_ok(q, vecn) && cp_async_ok(d_out, vecn);
  const dim3 grid(static_cast<unsigned>(NS * num_tiles), static_cast<unsigned>(H),
                  static_cast<unsigned>(B));
  kernel<<<grid, kMmaThreads, plan.smem, stream>>>(
      make_strided4_const<scalar_t>(q), make_strided4_const<scalar_t>(d_out),
      Strided3<const float>{lse.data_ptr<float>(), lse.stride(0), lse.stride(1), lse.stride(2)},
      Strided3<const float>{delta.data_ptr<float>(), delta.stride(0), delta.stride(1),
                            delta.stride(2)},
      groups.data_ptr<int32_t>(), groups.stride(0), groups.stride(1),
      offsets.data_ptr<int32_t>(), offsets.stride(0), offsets.stride(1), res,
      static_cast<int>(G), static_cast<int>(D), plan.DP, static_cast<int>(chunk_size),
      static_cast<float>(scale), num_tiles, plan.TKV, plan.BQ, plan.pitch, plan.sp,
      kv_stride_ok, q_vec);
  OMNIMEM_CHECK_LAUNCH();
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> sel_bwd_cuda(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& block_indices, const at::Tensor& out, const at::Tensor& lse,
    const at::Tensor& d_out, int64_t block_size, int64_t chunk_size, double scale, bool causal,
    int64_t group_size, bool allow_tf32) {
  // allow_tf32 == false selects the FP32 SIMT kernels, true the TF32 MMA ones.
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "sel_bwd: q/k/v must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "sel_bwd: q/k/v must be 4-D");
  TORCH_CHECK(out.dim() == 4 && d_out.dim() == 4, "sel_bwd: out/d_out must be 4-D");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() &&
                  q.scalar_type() == out.scalar_type() && q.scalar_type() == d_out.scalar_type(),
              "sel_bwd: q/k/v/out/d_out must share one dtype");
  TORCH_CHECK(out.sizes() == q.sizes() && d_out.sizes() == q.sizes(),
              "sel_bwd: out/d_out must have q's shape");
  TORCH_CHECK(k.sizes() == v.sizes(), "sel_bwd: k and v must have the same shape");
  TORCH_CHECK(block_indices.scalar_type() == at::kInt && block_indices.dim() == 4,
              "sel_bwd: block_indices must be int32 [B, MG, H, T]");
  TORCH_CHECK(lse.dim() == 3 && lse.scalar_type() == at::kFloat, "sel_bwd: lse must be fp32 [B,H,M]");
  TORCH_CHECK(block_size > 0 && chunk_size > 0, "sel_bwd: block_size/chunk_size must be > 0");
  TORCH_CHECK(group_size >= 1, "sel_bwd: group_size must be >= 1");

  const int64_t B = q.size(0), M = q.size(1), H = q.size(2), D = q.size(3);
  const int64_t N = k.size(1);
  const int64_t G = group_size;
  TORCH_CHECK(M % G == 0, "sel_bwd: M (", M, ") must be divisible by group_size (", G, ")");
  const int64_t MG = M / G;
  const int64_t T = block_indices.size(3);
  TORCH_CHECK(k.size(0) == B && k.size(2) == H && k.size(3) == D,
              "sel_bwd: k must be [B, N, H, D]");
  TORCH_CHECK(block_indices.size(0) == B && block_indices.size(1) == MG &&
                  block_indices.size(2) == H,
              "sel_bwd: block_indices must be [B, ", MG, ", ", H, ", T]");
  TORCH_CHECK(lse.size(0) == B && lse.size(1) == H && lse.size(2) == M,
              "sel_bwd: lse must be [B, H, M]");
  TORCH_CHECK(D >= 1 && D <= 256, "sel_bwd: head dim must be in [1, 256], got ", D);
  TORCH_CHECK(M <= 2147483647LL && N <= 2147483647LL, "sel_bwd: M/N exceed the int32 index range");

  const at::cuda::CUDAGuard guard(q.device());
  const auto stream = current_stream(q);

  at::Tensor dq = at::empty_like(q);
  at::Tensor dk32 = at::zeros({B, N, H, D}, q.options().dtype(at::kFloat));
  at::Tensor dv32 = at::zeros({B, N, H, D}, q.options().dtype(at::kFloat));
  at::Tensor delta = at::empty({B, H, M}, q.options().dtype(at::kFloat));
  if (B == 0 || H == 0 || M == 0)
    return std::make_tuple(dq, dk32.to(k.scalar_type()), dv32.to(v.scalar_type()));

  const int64_t NS = cdiv64(N, block_size);
  const at::Tensor empty_bound = at::empty({0}, block_indices.options());
  const bool use_mma = allow_tf32;
  const at::Tensor order = group_order(block_indices, MG, T);

  OMNIMEM_DISPATCH_FLOAT_TYPES(q.scalar_type(), "sel_bwd", [&] {
    DenseKV<scalar_t> acc;
    acc.k = make_strided4_const<scalar_t>(k);
    acc.v = make_strided4_const<scalar_t>(v);
    acc.ids = block_indices.data_ptr<int32_t>();
    acc.ids_sb = block_indices.stride(0);
    acc.ids_sg = block_indices.stride(1);
    acc.ids_sh = block_indices.stride(2);
    acc.ids_st = block_indices.stride(3);
    acc.N = static_cast<int>(N);
    acc.block_size = static_cast<int>(block_size);
    acc.chunk_size = static_cast<int>(chunk_size);
    acc.causal = causal;
    if (use_mma && dq_mma_profitable(static_cast<int>(G))) {
      launch_dq_mma<scalar_t>(q, out, d_out, dq, lse, delta, order, acc, k.stride(1),
                              k.stride(3), v.stride(1), v.stride(3), B, H, MG, G, T, D,
                              chunk_size, scale, stream, "sel_bwd dQ (tf32)");
    } else if (use_mma && G <= 7) {
      launch_dq_warp<scalar_t>(q, out, d_out, dq, lse, delta, order, acc, k.stride(1),
                               k.stride(3), v.stride(1), v.stride(3), B, H, MG, G, T, D,
                               chunk_size, scale, stream, "sel_bwd dQ (warp)");
    } else {
      launch_dq<scalar_t>(q, out, d_out, dq, lse, delta, order, acc, B, H, MG, G, T, D,
                          chunk_size, scale, block_size, stream, "sel_bwd dQ");
    }

    at::Tensor groups, offsets;
    std::tie(groups, offsets) = build_inverted_index_impl(block_indices, empty_bound, NS,
                                                          block_size, chunk_size, causal, G);
    DenseKVGrad<scalar_t> res;
    res.k = acc.k;
    res.v = acc.v;
    res.dk = dk32.data_ptr<float>();
    res.dv = dv32.data_ptr<float>();
    res.d_sb = dk32.stride(0);
    res.d_sn = dk32.stride(1);
    res.d_sh = dk32.stride(2);
    res.d_sd = dk32.stride(3);
    res.N = static_cast<int>(N);
    res.block_size = static_cast<int>(block_size);
    res.chunk_size = static_cast<int>(chunk_size);
    res.causal = causal;
    if (use_mma) {
      const bool kv_ok = k.stride(3) == 1 && v.stride(3) == 1 &&
                         k.stride(1) % cp_vec<scalar_t>() == 0 &&
                         v.stride(1) % cp_vec<scalar_t>() == 0;
      launch_dkv_mma<scalar_t>(q, d_out, lse, delta, groups, offsets, res, B, H, NS, G, D,
                               chunk_size, scale, block_size, kv_ok, stream,
                               "sel_bwd dK/dV (tf32)");
    } else {
      launch_dkv<scalar_t>(q, d_out, lse, delta, groups, offsets, res, B, H, NS, G, D,
                           chunk_size, scale, block_size, stream, "sel_bwd dK/dV");
    }
  });

  return std::make_tuple(dq, dk32.to(k.scalar_type()), dv32.to(v.scalar_type()));
}

std::tuple<at::Tensor, at::Tensor> sel_bwd_ptr_cuda(
    const at::Tensor& q, const at::Tensor& out, const at::Tensor& lse, const at::Tensor& d_out,
    const at::Tensor& block_indices, const at::Tensor& ptr_k, const at::Tensor& ptr_v,
    const at::Tensor& base_k, const at::Tensor& base_v, const at::Tensor& base_dk,
    const at::Tensor& base_dv, const at::Tensor& n_blocks_per_head, int64_t block_size,
    int64_t group_size, int64_t batch_stride_kv_bytes, int64_t batch_stride_dkv_bytes,
    int64_t stride_bn, int64_t stride_bd, int64_t stride_dbn, int64_t stride_dbd,
    int64_t blocks_per_chunk, int64_t in_chunk_block_bytes_kv, int64_t in_chunk_block_bytes_dkv,
    double scale, c10::string_view input_precision, bool compute_dkv, at::TensorList owners,
    at::TensorList grad_buffers) {
  (void)owners;        // dependency lists only; the kernels use raw int64 pointers
  (void)grad_buffers;
  // `tf32` permits the TF32 MMA kernels; `ieee` / `tf32x3` stay on the FP32 SIMT path.
  const Precision prec = parse_precision(input_precision);

  TORCH_CHECK(q.is_cuda(), "sel_bwd_ptr: q must be a CUDA tensor");
  TORCH_CHECK(q.dim() == 4 && out.dim() == 4 && d_out.dim() == 4, "sel_bwd_ptr: q/out/d_out must be 4-D");
  TORCH_CHECK(q.scalar_type() == out.scalar_type() && q.scalar_type() == d_out.scalar_type(),
              "sel_bwd_ptr: q/out/d_out must share one dtype");
  TORCH_CHECK(out.sizes() == q.sizes() && d_out.sizes() == q.sizes(),
              "sel_bwd_ptr: out/d_out must have q's shape");
  TORCH_CHECK(ptr_k.scalar_type() == at::kLong && ptr_v.scalar_type() == at::kLong,
              "sel_bwd_ptr: ptr tables must be int64 byte addresses");
  TORCH_CHECK(ptr_k.dim() == 4 && ptr_v.sizes() == ptr_k.sizes() &&
                  ptr_v.strides() == ptr_k.strides(),
              "sel_bwd_ptr: ptr_k / ptr_v must share shape and strides");
  TORCH_CHECK(block_indices.scalar_type() == at::kInt && block_indices.dim() == 4,
              "sel_bwd_ptr: block_indices must be int32 [B, MG, H, T]");
  TORCH_CHECK(lse.dim() == 3 && lse.scalar_type() == at::kFloat,
              "sel_bwd_ptr: lse must be fp32 [B, H, M]");
  TORCH_CHECK(block_size > 0 && group_size >= 1, "sel_bwd_ptr: bad block_size / group_size");

  const int64_t B = q.size(0), M = q.size(1), H = q.size(2), D = q.size(3);
  const int64_t G = group_size;
  TORCH_CHECK(M % G == 0, "sel_bwd_ptr: M (", M, ") must be divisible by group_size (", G, ")");
  const int64_t MG = M / G;
  const int64_t T = ptr_k.size(3);
  TORCH_CHECK(ptr_k.size(0) == B && ptr_k.size(1) == MG && ptr_k.size(2) == H,
              "sel_bwd_ptr: ptr tables must be [B, MG, H, T]");
  TORCH_CHECK(block_indices.size(0) == B && block_indices.size(1) == MG &&
                  block_indices.size(2) == H && block_indices.size(3) == T,
              "sel_bwd_ptr: block_indices must match the ptr table shape");
  TORCH_CHECK(lse.size(0) == B && lse.size(1) == H && lse.size(2) == M,
              "sel_bwd_ptr: lse must be [B, H, M]");
  TORCH_CHECK(D >= 1 && D <= 256, "sel_bwd_ptr: head dim must be in [1, 256], got ", D);
  TORCH_CHECK(M <= 2147483647LL, "sel_bwd_ptr: M exceeds the int32 index range");

  const at::cuda::CUDAGuard guard(q.device());
  const auto stream = current_stream(q);

  at::Tensor dq = at::empty_like(q);
  at::Tensor delta = at::empty({B, H, M}, q.options().dtype(at::kFloat));
  if (B == 0 || H == 0 || M == 0) return std::make_tuple(dq, delta);
  const bool use_mma = (prec == Precision::kTF32);
  const at::Tensor order = group_order(block_indices, MG, T);

  OMNIMEM_DISPATCH_FLOAT_TYPES(q.scalar_type(), "sel_bwd_ptr", [&] {
    PtrKV<scalar_t> acc;
    acc.ptr_k = ptr_k.data_ptr<int64_t>();
    acc.ptr_v = ptr_v.data_ptr<int64_t>();
    acc.sb = ptr_k.stride(0);
    acc.sg = ptr_k.stride(1);
    acc.sh = ptr_k.stride(2);
    acc.st = ptr_k.stride(3);
    acc.stride_n = stride_bn;
    acc.stride_d = stride_bd;
    acc.block_size = static_cast<int>(block_size);
    // chunk_size == 1, causal == false: the pointer path has no token ids.
    if (use_mma && dq_mma_profitable(static_cast<int>(G))) {
      launch_dq_mma<scalar_t>(q, out, d_out, dq, lse, delta, order, acc, stride_bn, stride_bd,
                              stride_bn, stride_bd, B, H, MG, G, T, D, /*chunk_size=*/1, scale,
                              stream, "sel_bwd_ptr dQ (tf32)");
    } else if (use_mma && G <= 7) {
      launch_dq_warp<scalar_t>(q, out, d_out, dq, lse, delta, order, acc, stride_bn, stride_bd,
                               stride_bn, stride_bd, B, H, MG, G, T, D, /*chunk_size=*/1,
                               scale, stream, "sel_bwd_ptr dQ (warp)");
    } else {
      launch_dq<scalar_t>(q, out, d_out, dq, lse, delta, order, acc, B, H, MG, G, T, D,
                          /*chunk_size=*/1, scale, block_size, stream, "sel_bwd_ptr dQ");
    }
  });

  if (!compute_dkv) return std::make_tuple(dq, delta);  // LRU dq-only training path

  TORCH_CHECK(base_k.dim() == 2 && base_k.size(0) == H, "sel_bwd_ptr: base_k must be [H, MAXB]");
  TORCH_CHECK(base_v.sizes() == base_k.sizes() && base_dk.sizes() == base_k.sizes() &&
                  base_dv.sizes() == base_k.sizes(),
              "sel_bwd_ptr: base_v / base_dk / base_dv must match base_k's shape");
  TORCH_CHECK(base_k.scalar_type() == at::kLong && base_v.scalar_type() == at::kLong &&
                  base_dk.scalar_type() == at::kLong && base_dv.scalar_type() == at::kLong,
              "sel_bwd_ptr: base tables must be int64");
  TORCH_CHECK(base_v.strides() == base_k.strides() && base_dk.strides() == base_k.strides() &&
                  base_dv.strides() == base_k.strides(),
              "sel_bwd_ptr: base tables must share strides");
  const bool has_bound = n_blocks_per_head.numel() != 0;
  if (has_bound)
    TORCH_CHECK(n_blocks_per_head.scalar_type() == at::kInt && n_blocks_per_head.dim() == 1 &&
                    n_blocks_per_head.size(0) == H,
                "sel_bwd_ptr: n_blocks_per_head must be int32 [H]");
  if (blocks_per_chunk > 0)
    TORCH_CHECK(in_chunk_block_bytes_kv > 0 && in_chunk_block_bytes_dkv > 0,
                "sel_bwd_ptr: chunk-level bases need in_chunk_block_bytes_kv/dkv > 0");
  TORCH_CHECK(stride_dbn != 0 || D == 1, "sel_bwd_ptr: stride_dbn must be the fp32 scratch stride");

  // Upper bound on the global block id, derived from tensor shapes only (never .item()).
  const int64_t max_base = base_k.size(1);
  const int64_t max_n = blocks_per_chunk > 0 ? max_base * blocks_per_chunk : max_base;

  at::Tensor groups, offsets;
  std::tie(groups, offsets) = build_inverted_index_impl(
      block_indices, n_blocks_per_head, max_n, block_size, /*chunk_size=*/1, /*causal=*/false, G);

  OMNIMEM_DISPATCH_FLOAT_TYPES(q.scalar_type(), "sel_bwd_ptr", [&] {
    PtrKVGrad<scalar_t> res;
    res.base_k = base_k.data_ptr<int64_t>();
    res.base_v = base_v.data_ptr<int64_t>();
    res.base_dk = base_dk.data_ptr<int64_t>();
    res.base_dv = base_dv.data_ptr<int64_t>();
    res.base_sh = base_k.stride(0);
    res.base_sb = base_k.stride(1);
    res.max_base = static_cast<int>(max_base);
    res.n_blocks_per_head = has_bound ? n_blocks_per_head.data_ptr<int32_t>() : nullptr;
    res.batch_kv_bytes = batch_stride_kv_bytes;
    res.batch_dkv_bytes = batch_stride_dkv_bytes;
    res.in_chunk_kv_bytes = in_chunk_block_bytes_kv;
    res.in_chunk_dkv_bytes = in_chunk_block_bytes_dkv;
    res.blocks_per_chunk = static_cast<int>(blocks_per_chunk);
    res.block_size = static_cast<int>(block_size);
    res.s_bn = stride_bn;
    res.s_bd = stride_bd;
    res.s_dbn = stride_dbn;
    res.s_dbd = stride_dbd;
    if (use_mma) {
      const bool kv_ok = stride_bd == 1 && stride_bn % cp_vec<scalar_t>() == 0;
      launch_dkv_mma<scalar_t>(q, d_out, lse, delta, groups, offsets, res, B, H, max_n, G, D,
                               /*chunk_size=*/1, scale, block_size, kv_ok, stream,
                               "sel_bwd_ptr dK/dV (tf32)");
    } else {
      launch_dkv<scalar_t>(q, d_out, lse, delta, groups, offsets, res, B, H, max_n, G, D,
                           /*chunk_size=*/1, scale, block_size, stream, "sel_bwd_ptr dK/dV");
    }
  });

  return std::make_tuple(dq, delta);
}

}  // namespace omnimem_nsa
