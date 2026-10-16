// Top-K block selection: each CTA scores MROWS query rows x BC keys, then merges.
// Path A (topk <= kPathAMax): register bitonic merge of PS pairs; Path B: CUB segmented sort.
// Ordering is (score descending, id ascending); masked scores are -inf with id -1.
#include "common.cuh"
#include "ops.h"

#include <cub/cub.cuh>
#include <mma.h>

#include <algorithm>

namespace omnimem_nsa {
namespace {

struct TopkMask {
  int block_size;
  int chunk_size;
  int chunk_size_k;
  int q_chunk_offset;
  int q_half;   // tf only
  int k_half;   // tf only
  int ex_win;
  int ex_sink;
  bool causal;
  bool tf_mask;
  bool prog_excl;
};

struct RowMask {
  int q_cid;
  int q_rel;
  bool q_in_clean;
  bool do_excl;
};

__device__ __forceinline__ RowMask make_row_mask(const TopkMask& m, int t) {
  RowMask r;
  r.q_cid = t / m.chunk_size + m.q_chunk_offset;
  if (m.tf_mask) {
    r.q_in_clean = r.q_cid < m.q_half;
    r.q_rel = r.q_in_clean ? r.q_cid : (r.q_cid - m.q_half);
  } else {
    r.q_in_clean = true;
    r.q_rel = r.q_cid;
  }
  r.do_excl = m.prog_excl ? (r.q_rel >= (m.ex_sink + m.ex_win)) : true;
  return r;
}

// Only the chunk id is cached per column; k_rel / k_in_clean stay out of registers.
__device__ __forceinline__ int make_col_cid(const TopkMask& m, int c) {
  return static_cast<int>((static_cast<int64_t>(c) * m.block_size) / m.chunk_size_k);
}

__device__ __forceinline__ bool visible(const TopkMask& m, const RowMask& r, int k_cid) {
  const bool k_in_clean = m.tf_mask ? (k_cid < m.k_half) : true;
  const int k_rel = (m.tf_mask && !k_in_clean) ? (k_cid - m.k_half) : k_cid;
  bool valid;
  if (m.tf_mask) {
    valid = (r.q_in_clean && k_in_clean && (k_rel <= r.q_rel))          // c2c
         || (!r.q_in_clean && k_in_clean && (k_rel < r.q_rel))          // n2c
         || (!r.q_in_clean && !k_in_clean && (k_rel == r.q_rel));       // n2n
  } else {
    valid = m.causal ? (k_cid <= r.q_cid) : true;
  }
  if (m.ex_sink > 0) {
    const bool sink_ok = (!k_in_clean) || (k_rel >= m.ex_sink);
    valid = valid && (sink_ok || !r.do_excl);
  }
  if (m.ex_win > 0) {
    bool in_window;
    if (m.tf_mask) {
      in_window = k_in_clean && (k_rel >= r.q_rel - m.ex_win + 1);
    } else {
      in_window = (k_cid >= r.q_cid - m.ex_win + 1);
      if (!m.causal) in_window = in_window && (k_cid <= r.q_cid + m.ex_win - 1);
    }
    valid = valid && (!in_window || !r.do_excl);
  }
  return valid;
}

// Exclusive upper bound on the visible column index (chunk-causal / TF bound only); saves work.
__device__ __forceinline__ int row_col_ub(const TopkMask& m, const RowMask& r, int NC) {
  int chunk_ub;
  if (m.tf_mask) {
    chunk_ub = r.q_in_clean ? (r.q_rel + 1) : (m.k_half + r.q_rel + 1);
  } else if (m.causal) {
    chunk_ub = r.q_cid + 1;
  } else {
    return NC;
  }
  if (chunk_ub <= 0) return 0;
  // k_cid(c) = (c*BS)/CSK <= chunk_ub-1  <=>  c < ceil(chunk_ub*CSK / BS)
  const int64_t ub = cdiv64(static_cast<int64_t>(chunk_ub) * m.chunk_size_k, m.block_size);
  return static_cast<int>(ub < NC ? ub : NC);
}

// Warp-distributed bitonic network over 32*L (score, id) pairs; lane l owns [l*L, l*L + L).
__device__ __forceinline__ bool tk_gt(float sa, int ia, float sb, int ib) {
  return (sa > sb) || ((sa == sb) && (ia < ib));
}

// One compare-exchange stage with distance DD; KK is the bitonic length, ignored when DESC.
template <int L, int KK, int DD, bool DESC>
__device__ __forceinline__ void tk_stage(float (&s)[L], int (&id)[L], int lane) {
  if constexpr (DD >= L) {
    constexpr unsigned lmask = static_cast<unsigned>(DD / L);
#pragma unroll
    for (int j = 0; j < L; ++j) {
      const float ps = __shfl_xor_sync(0xffffffffu, s[j], lmask);
      const int pi = __shfl_xor_sync(0xffffffffu, id[j], lmask);
      const int i = lane * L + j;
      const bool keep_low = ((i & DD) == 0);
      const bool asc = DESC ? false : ((i & KK) == 0);
      const bool take_greater = (keep_low != asc);
      if (tk_gt(s[j], id[j], ps, pi) != take_greater) {
        s[j] = ps;
        id[j] = pi;
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < L; ++j) {
      const int jp = j ^ DD;
      if (jp > j) {
        const int i = lane * L + j;
        const bool asc = DESC ? false : ((i & KK) == 0);
        const bool take_greater = !asc;  // j is the lower index of the pair
        if (tk_gt(s[j], id[j], s[jp], id[jp]) != take_greater) {
          const float ts = s[j];
          const int ti = id[j];
          s[j] = s[jp];
          id[j] = id[jp];
          s[jp] = ts;
          id[jp] = ti;
        }
      }
    }
  }
  if constexpr (DD > 1) tk_stage<L, KK, DD / 2, DESC>(s, id, lane);
}

template <int L, int KK>
__device__ __forceinline__ void tk_phase(float (&s)[L], int (&id)[L], int lane) {
  tk_stage<L, KK, KK / 2, false>(s, id, lane);
  if constexpr (KK < 32 * L) tk_phase<L, 2 * KK>(s, id, lane);
}

// Full ascending sort of the 32*L element array.
template <int L>
__device__ __forceinline__ void tk_sort_asc(float (&s)[L], int (&id)[L], int lane) {
  tk_phase<L, 2>(s, id, lane);
}

// Full descending merge of a *bitonic* 32*L element array.
template <int L>
__device__ __forceinline__ void tk_merge_desc(float (&s)[L], int (&id)[L], int lane) {
  tk_stage<L, 0, 16 * L, true>(s, id, lane);
}

// Order-preserving float -> uint32 map; -inf maps to 0x007fffff, above the reserved 0.
__device__ __forceinline__ uint32_t f32_to_ord(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// key = (ord(score) << 32) | (NC-1-c): descending = (score desc, id asc); 0 = no entry.
__device__ __forceinline__ uint64_t pack_key(float score, int c, int NC) {
  return (static_cast<uint64_t>(f32_to_ord(score)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(NC - 1 - c));
}
__device__ __forceinline__ int unpack_id(uint64_t key, int NC) {
  return NC - 1 - static_cast<int32_t>(static_cast<uint32_t>(key));
}

// Score tile producer: ssh[r * SLD + c] = raw dot(q row0+r, k c0+c), 0 for rows >= MG, cols >= NC.
// Whole-CTA entry (has __syncthreads()); ssh may alias ksh because the K tile is dead by then.
template <typename scalar_t, int BC, int MROWS, int WARPS, int DC, int DCP, int SLD,
          bool USE_MMA>
__device__ void score_tile(const Strided4<const scalar_t>& q,
                           const Strided4<const scalar_t>& k, float* __restrict__ qsh,
                           float* __restrict__ ksh, float* __restrict__ ssh, int b, int h,
                           int row0, int c0, int MG, int D, int NC, int tid, int lane,
                           int warp) {
  constexpr int L = BC / 32;
  constexpr int ROWS = MROWS / WARPS;

  constexpr int MT = MROWS / 16;
  constexpr int NT = BC / 16;
  constexpr int NFRAG = (MT * NT + WARPS - 1) / WARPS;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> cfrag[USE_MMA ? NFRAG : 1];
  float acc[USE_MMA ? 1 : ROWS][USE_MMA ? 1 : L];
  if constexpr (USE_MMA) {
#pragma unroll
    for (int f = 0; f < NFRAG; ++f) nvcuda::wmma::fill_fragment(cfrag[f], 0.f);
  } else {
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int j = 0; j < L; ++j) acc[r][j] = 0.f;
  }

  for (int d0 = 0; d0 < D; d0 += DC) {
    const int dcur = min(DC, D - d0);
    __syncthreads();  // previous round's qsh/ksh fully consumed
    // K tile: one warp per column, lanes along head dim (coalesced for stride_d == 1).
    const scalar_t* kbase = k.at(b, c0, h, d0);
    const scalar_t* qbase = q.at(b, row0, h, d0);
    for (int col = warp; col < BC; col += WARPS) {
      float* dst = ksh + col * DCP;
      if (c0 + col < NC) {
        const scalar_t* kp = kbase + static_cast<int64_t>(col) * k.s1;
        for (int dd = lane; dd < DC; dd += 32)
          dst[dd] = (dd < dcur) ? to_f32(kp[static_cast<int64_t>(dd) * k.s3]) : 0.f;
      } else {
        for (int dd = lane; dd < DC; dd += 32) dst[dd] = 0.f;
      }
    }
    for (int r = warp; r < MROWS; r += WARPS) {
      float* dst = qsh + r * DCP;
      if (row0 + r < MG) {
        const scalar_t* qp = qbase + static_cast<int64_t>(r) * q.s1;
        for (int dd = lane; dd < DC; dd += 32)
          dst[dd] = (dd < dcur) ? to_f32(qp[static_cast<int64_t>(dd) * q.s3]) : 0.f;
      } else {
        for (int dd = lane; dd < DC; dd += 32) dst[dd] = 0.f;
      }
    }
    __syncthreads();

    if constexpr (USE_MMA) {
      // TF32 QK: C[16x16] += A[16x8] * B[8x16], A = Q rows (row major), B = K columns (col major)
#pragma unroll
      for (int f = 0; f < NFRAG; ++f) {
        const int tile = f * WARPS + warp;
        if (tile >= MT * NT) break;  // warp-uniform
        const int mt = tile / NT;
        const int nt = tile - mt * NT;
#pragma unroll
        for (int kk = 0; kk < DC; kk += 8) {
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8,
                                 nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> af;
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8,
                                 nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> bf;
          nvcuda::wmma::load_matrix_sync(af, qsh + (mt * 16) * DCP + kk, DCP);
          nvcuda::wmma::load_matrix_sync(bf, ksh + (nt * 16) * DCP + kk, DCP);
#pragma unroll
          for (int t = 0; t < af.num_elements; ++t)
            af.x[t] = nvcuda::wmma::__float_to_tf32(af.x[t]);
#pragma unroll
          for (int t = 0; t < bf.num_elements; ++t)
            bf.x[t] = nvcuda::wmma::__float_to_tf32(bf.x[t]);
          nvcuda::wmma::mma_sync(cfrag[f], af, bf, cfrag[f]);
        }
      }
    } else {
      // FP32 SIMT; lanes are DCP (= DC+1, odd) floats apart, so ksh reads are conflict-free.
      for (int dd = 0; dd < dcur; ++dd) {
        float qv[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) qv[r] = qsh[(warp * ROWS + r) * DCP + dd];
#pragma unroll
        for (int j = 0; j < L; ++j) {
          const float kv = ksh[(j * 32 + lane) * DCP + dd];
#pragma unroll
          for (int r = 0; r < ROWS; ++r) acc[r][j] = fmaf(qv[r], kv, acc[r][j]);
        }
      }
    }
  }

  __syncthreads();  // ssh may alias nothing, but the readers below need all writes visible
  if constexpr (USE_MMA) {
#pragma unroll
    for (int f = 0; f < NFRAG; ++f) {
      const int tile = f * WARPS + warp;
      if (tile >= MT * NT) break;
      const int mt = tile / NT;
      const int nt = tile - mt * NT;
      nvcuda::wmma::store_matrix_sync(ssh + (mt * 16) * SLD + nt * 16, cfrag[f], SLD,
                                      nvcuda::wmma::mem_row_major);
    }
  } else {
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int j = 0; j < L; ++j) ssh[(warp * ROWS + r) * SLD + j * 32 + lane] = acc[r][j];
  }
  __syncthreads();
}

// Path A: fused scoring + register top-k merge; SPLIT writes packed keys to part[row*nsplit+split].
template <typename scalar_t, int BC, int PS, int MROWS, int WARPS, int DC, bool USE_MMA,
          bool SPLIT>
__global__ __launch_bounds__(WARPS * 32) void topk_tile_kernel(
    const Strided4<const scalar_t> q, const Strided4<const scalar_t> k,
    int32_t* __restrict__ out, uint64_t* __restrict__ part, int nsplit, int MG, int H,
    int D, int NC, int topk, float scale, const TopkMask mk) {
  constexpr int L = PS / 32;          // state slots per lane (PS = next_pow2(topk) >= 32)
  constexpr int ROUNDS = BC / PS;     // merge rounds per staged K tile
  constexpr int ROWS = MROWS / WARPS;
  constexpr int DCP = USE_MMA ? (DC + 8) : (DC + 1);
  constexpr int SLD = BC + 4;
  // the score tile lives in the (by then dead) K tile
  constexpr int KSH = (BC * DCP > MROWS * SLD) ? (BC * DCP) : (MROWS * SLD);

  __shared__ __align__(16) float qsh[MROWS * DCP];
  __shared__ __align__(16) float ksh[KSH];
  float* const ssh = ksh;
  __shared__ int sh_ub;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int split = SPLIT ? static_cast<int>(blockIdx.y) : 0;
  const int bh = static_cast<int>(blockIdx.z);
  const int b = bh / H;
  const int h = bh - b * H;
  const int row0 = static_cast<int>(blockIdx.x) * MROWS;

  int rows[ROWS];
  RowMask rm[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = row0 + warp * ROWS + r;
    rm[r] = make_row_mask(mk, rows[r] < MG ? rows[r] : 0);
  }

  // CTA-uniform causal upper bound (max over the rows of this CTA).
  if (tid == 0) sh_ub = 0;
  __syncthreads();
  if (lane == 0) {
    int u = 0;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      if (rows[r] < MG) {
        const int v = row_col_ub(mk, rm[r], NC);
        u = v > u ? v : u;
      }
    }
    if (u > 0) atomicMax(&sh_ub, u);
  }
  __syncthreads();
  const int ub = sh_ub;

  // Column range owned by this CTA (BC-aligned so the tiles stay aligned).
  int c_begin = 0, c_end = ub;
  if (SPLIT) {
    const int span = cdiv(cdiv(ub, nsplit), BC) * BC;
    c_begin = split * span;
    c_end = min(c_begin + span, ub);
  }

  float bs[ROWS][L];
  int bi[ROWS][L];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
#pragma unroll
    for (int j = 0; j < L; ++j) {
      bs[r][j] = kNegInf;
      bi[r][j] = -1;
    }
  }

  for (int c0 = c_begin; c0 < c_end; c0 += BC) {
    score_tile<scalar_t, BC, MROWS, WARPS, DC, DCP, SLD, USE_MMA>(
        q, k, qsh, ksh, ssh, b, h, row0, c0, MG, D, NC, tid, lane, warp);

#pragma unroll
    for (int rd = 0; rd < ROUNDS; ++rd) {
      const int cbase = c0 + rd * PS;
      int cid[L];
      int kcid[L];
#pragma unroll
      for (int j = 0; j < L; ++j) {
        cid[j] = cbase + j * 32 + lane;
        kcid[j] = make_col_cid(mk, cid[j] < NC ? cid[j] : 0);
      }

#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        float cs[L];
        int ci[L];
        float cmax = kNegInf;
#pragma unroll
        for (int j = 0; j < L; ++j) {
          const bool vis = (cid[j] < NC) && (rows[r] < MG) && visible(mk, rm[r], kcid[j]);
          cs[j] = vis ? ssh[(warp * ROWS + r) * SLD + rd * PS + j * 32 + lane] * scale
                      : kNegInf;
          ci[j] = vis ? cid[j] : -1;
          cmax = fmaxf(cmax, cs[j]);
        }
        // Running topk-th best (monotone, warp min); rounds that cannot reach it are skipped.
        float thr = INFINITY;
#pragma unroll
        for (int j = 0; j < L; ++j) {
          if (lane * L + j < topk) thr = fminf(thr, bs[r][j]);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
          thr = fminf(thr, __shfl_xor_sync(0xffffffffu, thr, o));
        if (!__any_sync(0xffffffffu, (cmax > kNegInf) && (cmax >= thr))) continue;

        // Batcher: state descending ++ candidates ascending is bitonic, so max is the top-PS.
        tk_sort_asc<L>(cs, ci, lane);
#pragma unroll
        for (int j = 0; j < L; ++j) {
          if (tk_gt(cs[j], ci[j], bs[r][j], bi[r][j])) {
            bs[r][j] = cs[j];
            bi[r][j] = ci[j];
          }
        }
        tk_merge_desc<L>(bs[r], bi[r], lane);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    if (rows[r] >= MG) continue;
    const int64_t row = (static_cast<int64_t>(b * MG + rows[r]) * H + h);
    if constexpr (SPLIT) {
      uint64_t* p = part + (row * nsplit + split) * topk;
#pragma unroll
      for (int j = 0; j < L; ++j) {
        const int i = lane * L + j;
        if (i < topk) p[i] = (bi[r][j] >= 0) ? pack_key(bs[r][j], bi[r][j], NC) : 0ull;
      }
    } else {
      int32_t* op = out + row * topk;
#pragma unroll
      for (int j = 0; j < L; ++j) {
        const int i = lane * L + j;
        if (i < topk) op[i] = bi[r][j];
      }
    }
  }
}

// nsplit-way merge of per-split sorted lists, one warp per row, lane l holds the head of list l.
__global__ void topk_merge_kernel(const uint64_t* __restrict__ part, int32_t* __restrict__ out,
                                  int nsplit, int topk, int NC, int64_t rows_total) {
  const int64_t gw = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  if (gw >= rows_total) return;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const uint64_t* base = part + gw * nsplit * topk;
  int32_t* op = out + gw * topk;

  int pos = 0;
  uint64_t head = (lane < nsplit) ? base[static_cast<int64_t>(lane) * topk] : 0ull;
  for (int i = 0; i < topk; ++i) {
    uint64_t m = head;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const uint64_t other = __shfl_xor_sync(0xffffffffu, m, o);
      m = other > m ? other : m;
    }
    if (m == 0ull) {
      for (int s = i + lane; s < topk; s += 32) op[s] = -1;
      return;
    }
    const unsigned ball = __ballot_sync(0xffffffffu, head == m);
    const int owner = __ffs(static_cast<int>(ball)) - 1;
    if (lane == 0) op[i] = unpack_id(m, NC);
    if (lane == owner) {
      ++pos;
      head = (pos < topk) ? base[static_cast<int64_t>(lane) * topk + pos] : 0ull;
    }
  }
}

// Path B: scores -> (uint32 key, int32 id) -> CUB segmented radix sort -> gather.
template <typename scalar_t, int BC, int MROWS, int WARPS, int DC, bool USE_MMA>
__global__ __launch_bounds__(WARPS * 32) void topk_keys_kernel(
    const Strided4<const scalar_t> q, const Strided4<const scalar_t> k,
    uint32_t* __restrict__ keys, int32_t* __restrict__ vals, int g0, int gpb, int MG, int H,
    int D, int NC, float scale, const TopkMask mk) {
  constexpr int L = BC / 32;
  constexpr int ROWS = MROWS / WARPS;
  constexpr int DCP = USE_MMA ? (DC + 8) : (DC + 1);
  constexpr int SLD = BC + 4;
  constexpr int KSH = (BC * DCP > MROWS * SLD) ? (BC * DCP) : (MROWS * SLD);

  __shared__ __align__(16) float qsh[MROWS * DCP];
  __shared__ __align__(16) float ksh[KSH];
  float* const ssh = ksh;
  __shared__ int sh_ub;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int grel = static_cast<int>(blockIdx.x);
  const int g = g0 + grel;
  const int bh = g / gpb;
  const int b = bh / H;
  const int h = bh - b * H;
  const int row0 = (g - bh * gpb) * MROWS;

  int rows[ROWS];
  RowMask rm[ROWS];
  uint32_t* krow[ROWS];
  int32_t* vrow[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = row0 + warp * ROWS + r;
    rm[r] = make_row_mask(mk, rows[r] < MG ? rows[r] : 0);
    const int64_t wrow = static_cast<int64_t>(grel) * MROWS + warp * ROWS + r;
    krow[r] = keys + wrow * NC;
    vrow[r] = vals + wrow * NC;
  }

  if (tid == 0) sh_ub = 0;
  __syncthreads();
  if (lane == 0) {
    int u = 0;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      if (rows[r] < MG) {
        const int v = row_col_ub(mk, rm[r], NC);
        u = v > u ? v : u;
      }
    }
    if (u > 0) atomicMax(&sh_ub, u);
  }
  __syncthreads();
  const int ub = sh_ub;

  for (int c0 = 0; c0 < NC; c0 += BC) {
    if (c0 >= ub) {  // CTA-uniform: everything from here on is invisible
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        for (int c = c0 + lane; c < NC; c += 32) {
          krow[r][c] = 0u;
          vrow[r][c] = -1;
        }
      }
      break;
    }
    score_tile<scalar_t, BC, MROWS, WARPS, DC, DCP, SLD, USE_MMA>(
        q, k, qsh, ksh, ssh, b, h, row0, c0, MG, D, NC, tid, lane, warp);

    int cid[L];
    int kcid[L];
#pragma unroll
    for (int j = 0; j < L; ++j) {
      cid[j] = c0 + j * 32 + lane;
      kcid[j] = make_col_cid(mk, cid[j] < NC ? cid[j] : 0);
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
      for (int j = 0; j < L; ++j) {
        if (cid[j] >= NC) continue;
        const bool vis = (rows[r] < MG) && visible(mk, rm[r], kcid[j]);
        const float s = ssh[(warp * ROWS + r) * SLD + j * 32 + lane] * scale;
        krow[r][cid[j]] = vis ? f32_to_ord(s) : 0u;
        vrow[r][cid[j]] = vis ? cid[j] : -1;
      }
    }
  }
}

template <int MROWS>
__global__ void topk_gather_kernel(const int32_t* __restrict__ vals_sorted,
                                   int32_t* __restrict__ out, int g0, int gpb, int MG, int H,
                                   int NC, int topk, int rows_in_chunk) {
  const int rw = static_cast<int>(blockIdx.x);
  if (rw >= rows_in_chunk) return;
  const int g = g0 + rw / MROWS;
  const int i = rw - (rw / MROWS) * MROWS;
  const int bh = g / gpb;
  const int t = (g - bh * gpb) * MROWS + i;
  if (t >= MG) return;
  const int b = bh / H;
  const int h = bh - b * H;
  const int32_t* row = vals_sorted + static_cast<int64_t>(rw) * NC;
  int32_t* op = out + (static_cast<int64_t>(b * MG + t) * H + h) * topk;
  for (int s = static_cast<int>(threadIdx.x); s < topk; s += static_cast<int>(blockDim.x))
    op[s] = (s < NC) ? row[s] : -1;
}

// Largest topk served by the register (path A) merge; larger values use the CUB sort.
constexpr int kPathAMax = 128;

// Path A launch parameters.
constexpr int kMRowsA = 32;   // query rows per CTA
constexpr int kWarpsA = 8;    // warps per CTA, each owning kMRowsA / kWarpsA rows
constexpr int kDCA = 64;      // head-dim chunk staged per pass
// Split the NC range so the grid has at least this many CTAs per SM.
constexpr int kCtaPerSmA = 8;
// `topk_merge_kernel` keeps one list head per lane.
constexpr int kMaxSplitA = kWarpSize;
// Bound on the split workspace: rows * nsplit * topk * 8 bytes.
constexpr int64_t kPartBudgetA = 256LL << 20;

// Number of column ranges the NC keys are split into (1 = no split); output does not depend on it.
template <int BC>
int path_a_nsplit(int topk, int B, int MG, int H, int NC) {
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int target = kCtaPerSmA * nsm;
  const int64_t rows_total = static_cast<int64_t>(B) * MG * H;
  const int64_t base = cdiv64(MG, kMRowsA) * B * H;
  const int col_tiles = static_cast<int>(cdiv64(NC, BC));
  int nsplit = static_cast<int>(
      std::min<int64_t>(cdiv64(target, std::max<int64_t>(base, 1)), kMaxSplitA));
  nsplit = std::max(nsplit, 1);
  // keep at least 2 column tiles of work per split, otherwise the merge dominates
  nsplit = std::min(nsplit, std::max(1, col_tiles / 2));
  while (nsplit > 1 && rows_total * nsplit * topk * 8 > kPartBudgetA) nsplit >>= 1;
  return nsplit;
}

// BC = columns staged per shared-memory K tile, PS = state size (see the file header).
template <typename scalar_t, int BC, int PS, bool USE_MMA>
void launch_path_a(const at::Tensor& q, const at::Tensor& k, at::Tensor& out, int B, int MG,
                   int H, int D, int NC, int topk, float scale, const TopkMask& mk,
                   cudaStream_t stream) {
  const int nsplit = path_a_nsplit<BC>(topk, B, MG, H, NC);
  const dim3 grid(static_cast<unsigned>((MG + kMRowsA - 1) / kMRowsA),
                  static_cast<unsigned>(nsplit), static_cast<unsigned>(B * H));
  int32_t* op = out.data_ptr<int32_t>();
  if (nsplit <= 1) {
    topk_tile_kernel<scalar_t, BC, PS, kMRowsA, kWarpsA, kDCA, USE_MMA, false>
        <<<grid, kWarpsA * 32, 0, stream>>>(make_strided4_const<scalar_t>(q),
                                            make_strided4_const<scalar_t>(k), op, nullptr, 1,
                                            MG, H, D, NC, topk, scale, mk);
    OMNIMEM_CHECK_LAUNCH();
    return;
  }
  const int64_t rows_total = static_cast<int64_t>(B) * MG * H;
  at::Tensor part = at::empty({rows_total * nsplit * topk}, q.options().dtype(at::kLong));
  topk_tile_kernel<scalar_t, BC, PS, kMRowsA, kWarpsA, kDCA, USE_MMA, true>
      <<<grid, kWarpsA * 32, 0, stream>>>(
          make_strided4_const<scalar_t>(q), make_strided4_const<scalar_t>(k), nullptr,
          reinterpret_cast<uint64_t*>(part.data_ptr<int64_t>()), nsplit, MG, H, D, NC, topk,
          scale, mk);
  OMNIMEM_CHECK_LAUNCH();
  constexpr int kMergeThreads = 128;
  const int64_t merge_blocks = cdiv64(rows_total, kMergeThreads / 32);
  topk_merge_kernel<<<static_cast<unsigned>(merge_blocks), kMergeThreads, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(part.data_ptr<int64_t>()), op, nsplit, topk, NC,
      rows_total);
  OMNIMEM_CHECK_LAUNCH();
}

// One (BC, PS) per topk range; all fit 48 KB static shared memory (score tile aliases K tile).
static_assert(kPathAMax == 128, "path A instantiates states of at most 128 entries");

template <typename scalar_t, bool USE_MMA>
void dispatch_path_a(const at::Tensor& q, const at::Tensor& k, at::Tensor& out, int B, int MG,
                     int H, int D, int NC, int topk, float scale, const TopkMask& mk,
                     cudaStream_t stream) {
  if (topk <= 32)
    launch_path_a<scalar_t, 64, 32, USE_MMA>(q, k, out, B, MG, H, D, NC, topk, scale, mk, stream);
  else if (topk <= 64)
    launch_path_a<scalar_t, 128, 64, USE_MMA>(q, k, out, B, MG, H, D, NC, topk, scale, mk, stream);
  else
    launch_path_a<scalar_t, 128, 128, USE_MMA>(q, k, out, B, MG, H, D, NC, topk, scale, mk, stream);
}

}  // namespace

at::Tensor topk_cuda(
    const at::Tensor& q, const at::Tensor& k, int64_t topk, double scale,
    int64_t block_size, int64_t chunk_size, int64_t chunk_size_k, bool causal,
    int64_t ex_win, int64_t ex_sink, bool prog_excl, bool tf_mask,
    int64_t q_chunk_offset, bool allow_tf32) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda(), "omnimem_nsa::topk: q and k must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device(), "omnimem_nsa::topk: q and k on different devices");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4,
              "omnimem_nsa::topk: expected q [B,MG,H,D] and k [B,NC,H,D]");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(),
              "omnimem_nsa::topk: q and k must share a dtype, got ", q.scalar_type(), " vs ",
              k.scalar_type());
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2) && q.size(3) == k.size(3),
              "omnimem_nsa::topk: incompatible shapes q=", q.sizes(), " k=", k.sizes());
  TORCH_CHECK(topk > 0 && topk <= std::numeric_limits<int32_t>::max(),
              "omnimem_nsa::topk: topk must be in [1, 2^31), got ", topk);
  TORCH_CHECK(block_size > 0 && chunk_size > 0 && chunk_size_k > 0,
              "omnimem_nsa::topk: block_size / chunk_size / chunk_size_k must be > 0");
  TORCH_CHECK(ex_win >= 0 && ex_sink >= 0,
              "omnimem_nsa::topk: ex_win / ex_sink must be >= 0");
  TORCH_CHECK(q_chunk_offset >= 0, "omnimem_nsa::topk: q_chunk_offset must be >= 0");
  TORCH_CHECK(!(tf_mask && q_chunk_offset != 0),
              "omnimem_nsa::topk: tf_mask assumes a full [clean|noisy] layout, "
              "q_chunk_offset must be 0");
  const int64_t D64 = q.size(3);
  TORCH_CHECK(D64 >= 1 && D64 <= 256, "omnimem_nsa::topk: head dim must be in [1, 256], got ",
              D64);

  const int B = static_cast<int>(q.size(0));
  const int MG = static_cast<int>(q.size(1));
  const int H = static_cast<int>(q.size(2));
  const int D = static_cast<int>(D64);
  const int NC = static_cast<int>(k.size(1));
  const int tk = static_cast<int>(topk);
  TORCH_CHECK(static_cast<int64_t>(B) * H <= 65535,
              "omnimem_nsa::topk: B*H must be <= 65535, got ", static_cast<int64_t>(B) * H);

  at::cuda::CUDAGuard guard(q.device());
  at::Tensor out = at::empty({B, MG, H, tk}, q.options().dtype(at::kInt));
  if (out.numel() == 0) return out;
  if (NC == 0) {
    out.fill_(-1);
    return out;
  }

  TopkMask mk;
  mk.block_size = static_cast<int>(block_size);
  mk.chunk_size = static_cast<int>(chunk_size);
  mk.chunk_size_k = static_cast<int>(chunk_size_k);
  mk.q_chunk_offset = static_cast<int>(q_chunk_offset);
  mk.q_half = tf_mask ? static_cast<int>(MG / (2 * chunk_size)) : 0;
  mk.k_half =
      tf_mask ? static_cast<int>((static_cast<int64_t>(NC) * block_size) / (2 * chunk_size_k)) : 0;
  mk.ex_win = static_cast<int>(ex_win);
  mk.ex_sink = static_cast<int>(ex_sink);
  mk.causal = causal;
  mk.tf_mask = tf_mask;
  mk.prog_excl = prog_excl;

  const cudaStream_t stream = current_stream(q);
  // TF32 is exact for bf16/fp16 so it is always used there; for fp32 it is gated on allow_tf32.
  const bool use_mma = allow_tf32 || q.scalar_type() != at::kFloat;

  // Path A
  if (tk <= kPathAMax) {
    OMNIMEM_DISPATCH_FLOAT_TYPES(q.scalar_type(), "omnimem_nsa::topk", [&] {
      if (use_mma) {
        dispatch_path_a<scalar_t, true>(q, k, out, B, MG, H, D, NC, tk,
                                        static_cast<float>(scale), mk, stream);
      } else {
        dispatch_path_a<scalar_t, false>(q, k, out, B, MG, H, D, NC, tk,
                                         static_cast<float>(scale), mk, stream);
      }
    });
    return out;
  }

  // Path B
  constexpr int kBCB = 64;
  constexpr int kWarpsB = 8;
  constexpr int kMRowsB = 32;   // query rows per CTA / per workspace group
  constexpr int kDCB = 64;

  const int gpb = (MG + kMRowsB - 1) / kMRowsB;             // row groups per (b, h)
  const int64_t n_groups = static_cast<int64_t>(B) * H * gpb;
  // Workspace budget: (uint32 key + int32 value) x 2 (CUB double buffer) = 16 B per item.
  constexpr int64_t kBudgetBytes = 64LL << 20;
  int64_t groups_per_chunk =
      std::max<int64_t>(1, (kBudgetBytes / (static_cast<int64_t>(NC) * 16)) / kMRowsB);
  groups_per_chunk = std::min<int64_t>(groups_per_chunk, n_groups);
  const int64_t rows_per_chunk = groups_per_chunk * kMRowsB;
  const int64_t items_per_chunk = rows_per_chunk * NC;
  TORCH_CHECK(items_per_chunk <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
              "omnimem_nsa::topk: NC=", NC, " too large for the large-topk path");

  const auto opt_i32 = q.options().dtype(at::kInt);
  at::Tensor keys_in = at::empty({items_per_chunk}, opt_i32);
  at::Tensor keys_out = at::empty({items_per_chunk}, opt_i32);
  at::Tensor vals_in = at::empty({items_per_chunk}, opt_i32);
  at::Tensor vals_out = at::empty({items_per_chunk}, opt_i32);
  at::Tensor offsets = at::arange(0, rows_per_chunk + 1, opt_i32).mul_(NC);

  auto* d_kin = reinterpret_cast<const uint32_t*>(keys_in.data_ptr<int32_t>());
  auto* d_kout = reinterpret_cast<uint32_t*>(keys_out.data_ptr<int32_t>());
  const int32_t* d_vin = vals_in.data_ptr<int32_t>();
  int32_t* d_vout = vals_out.data_ptr<int32_t>();
  const int32_t* d_off = offsets.data_ptr<int32_t>();

  size_t temp_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr, temp_bytes, d_kin, d_kout, d_vin, d_vout, static_cast<int>(items_per_chunk),
      static_cast<int>(rows_per_chunk), d_off, d_off + 1, 0, 32, stream));
  // never allocate 0 bytes: CUB treats a null d_temp_storage as another size query.
  at::Tensor temp = at::empty({static_cast<int64_t>(temp_bytes) + 1},
                              q.options().dtype(at::kByte));

  OMNIMEM_DISPATCH_FLOAT_TYPES(q.scalar_type(), "omnimem_nsa::topk", [&] {
    const auto qs = make_strided4_const<scalar_t>(q);
    const auto ks = make_strided4_const<scalar_t>(k);
    int32_t* op = out.data_ptr<int32_t>();
    for (int64_t g0 = 0; g0 < n_groups; g0 += groups_per_chunk) {
      const int ng = static_cast<int>(std::min(groups_per_chunk, n_groups - g0));
      const int rows_here = ng * kMRowsB;
      auto launch_keys = [&](auto mma_tag) {
        constexpr bool kM = decltype(mma_tag)::value;
        topk_keys_kernel<scalar_t, kBCB, kMRowsB, kWarpsB, kDCB, kM>
            <<<static_cast<unsigned>(ng), kWarpsB * 32, 0, stream>>>(
                qs, ks, reinterpret_cast<uint32_t*>(keys_in.data_ptr<int32_t>()),
                vals_in.data_ptr<int32_t>(), static_cast<int>(g0), gpb, MG, H, D, NC,
                static_cast<float>(scale), mk);
      };
      if (use_mma) launch_keys(std::true_type{});
      else launch_keys(std::false_type{});
      OMNIMEM_CHECK_LAUNCH();
      size_t tb = temp_bytes + 1;
      C10_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
          temp.data_ptr(), tb, d_kin, d_kout, d_vin, d_vout, rows_here * NC, rows_here,
          d_off, d_off + 1, 0, 32, stream));
      topk_gather_kernel<kMRowsB><<<static_cast<unsigned>(rows_here), 128, 0, stream>>>(
          d_vout, op, static_cast<int>(g0), gpb, MG, H, NC, tk, rows_here);
      OMNIMEM_CHECK_LAUNCH();
    }
  });
  return out;
}

}  // namespace omnimem_nsa
