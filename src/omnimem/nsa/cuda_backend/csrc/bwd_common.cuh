// Helpers shared by selection_bwd.cu and inverted_index.cu; header only.
// Accumulation is FP32; pointer dK/dV are FP32 scratch blocks, never the bf16 K/V ones.
#pragma once

#include "common.cuh"

#include <tuple>

namespace omnimem_nsa {

// Defined in inverted_index.cu; called directly instead of going through the dispatcher.
std::tuple<at::Tensor, at::Tensor> build_inverted_index_impl(
    const at::Tensor& block_indices, const at::Tensor& n_blocks_per_head, int64_t num_blocks,
    int64_t block_size, int64_t chunk_size, bool causal, int64_t group_size);

// One resolved KV block plus the FP32 gradient tile the owning CTA writes; token0 >= 0 = causal.
template <typename T>
struct KVGradRef {
  const T* k;
  const T* v;
  float* dk;
  float* dv;
  int64_t k_sn, k_sd;
  int64_t v_sn, v_sd;
  int64_t dk_sn, dk_sd;
  int64_t dv_sn, dv_sd;
  int n_valid;
  int64_t token0;
};

// Dense: K/V are strided [B, N, H, D]; dK/dV are contiguous FP32 [B, N, H, D].
template <typename T>
struct DenseKVGrad {
  Strided4<const T> k;
  Strided4<const T> v;
  float* dk;
  float* dv;
  int64_t d_sb, d_sn, d_sh, d_sd;  // element strides of the fp32 dk/dv buffers
  int N;
  int block_size;
  int chunk_size;
  bool causal;

  __device__ __forceinline__ bool resolve(int b, int h, int bid, KVGradRef<T>& r) const {
    if (bid < 0) return false;
    const int64_t tok0 = static_cast<int64_t>(bid) * block_size;
    if (tok0 >= N) return false;  // out-of-range id (bid >= ceil(N/BS)) -> nothing to write
    r.k = k.at(b, tok0, h, 0);
    r.v = v.at(b, tok0, h, 0);
    r.k_sn = k.s1; r.k_sd = k.s3;
    r.v_sn = v.s1; r.v_sd = v.s3;
    const int64_t off = static_cast<int64_t>(b) * d_sb + tok0 * d_sn + static_cast<int64_t>(h) * d_sh;
    r.dk = dk + off;
    r.dv = dv + off;
    r.dk_sn = d_sn; r.dk_sd = d_sd;
    r.dv_sn = d_sn; r.dv_sd = d_sd;
    const int64_t rem = static_cast<int64_t>(N) - tok0;
    r.n_valid = static_cast<int>(rem < block_size ? rem : block_size);
    r.token0 = causal ? tok0 : -1;
    return true;
  }
};

// Pointer: base tables [H, MAXB] of int64 byte addresses (chunk-level or block-level).
template <typename T>
struct PtrKVGrad {
  const int64_t* base_k;
  const int64_t* base_v;
  const int64_t* base_dk;
  const int64_t* base_dv;
  int64_t base_sh, base_sb;  // element strides of the [H, MAXB] tables
  int max_base;
  const int32_t* n_blocks_per_head;  // may be nullptr
  int64_t batch_kv_bytes;
  int64_t batch_dkv_bytes;
  int64_t in_chunk_kv_bytes;
  int64_t in_chunk_dkv_bytes;
  int blocks_per_chunk;  // 0 => block-level bases
  int block_size;
  int64_t s_bn, s_bd;    // K/V element strides (bf16 blocks)
  int64_t s_dbn, s_dbd;  // dK/dV element strides (fp32 scratch) -- never the bf16 ones

  __device__ __forceinline__ bool resolve(int b, int h, int bid, KVGradRef<T>& r) const {
    if (bid < 0) return false;
    if (n_blocks_per_head != nullptr && bid >= n_blocks_per_head[h]) return false;
    int64_t idx, off_kv, off_dkv;
    if (blocks_per_chunk > 0) {
      idx = bid / blocks_per_chunk;
      const int64_t in_chunk = bid - idx * blocks_per_chunk;
      off_kv = in_chunk * in_chunk_kv_bytes;
      off_dkv = in_chunk * in_chunk_dkv_bytes;
    } else {
      idx = bid;
      off_kv = 0;
      off_dkv = 0;
    }
    if (idx >= max_base) return false;
    const int64_t o = static_cast<int64_t>(h) * base_sh + idx * base_sb;
    // Check the bases BEFORE adding any offset (a 0 base must never become a valid pointer).
    const int64_t bk = base_k[o];
    const int64_t bv = base_v[o];
    const int64_t bdk = base_dk[o];
    const int64_t bdv = base_dv[o];
    // All four bases are required: an asymmetric table must never dereference a null
    // pointer.
    if (bk == 0 || bv == 0 || bdk == 0 || bdv == 0) return false;
    const int64_t boff_kv = static_cast<int64_t>(b) * batch_kv_bytes + off_kv;
    const int64_t boff_dkv = static_cast<int64_t>(b) * batch_dkv_bytes + off_dkv;
    r.k = reinterpret_cast<const T*>(static_cast<uintptr_t>(bk + boff_kv));
    r.v = reinterpret_cast<const T*>(static_cast<uintptr_t>(bv + boff_kv));
    r.dk = reinterpret_cast<float*>(static_cast<uintptr_t>(bdk + boff_dkv));
    r.dv = reinterpret_cast<float*>(static_cast<uintptr_t>(bdv + boff_dkv));
    r.k_sn = s_bn; r.k_sd = s_bd;
    r.v_sn = s_bn; r.v_sd = s_bd;
    r.dk_sn = s_dbn; r.dk_sd = s_dbd;
    r.dv_sn = s_dbn; r.dv_sd = s_dbd;
    r.n_valid = block_size;
    r.token0 = -1;  // the pointer path carries no global token ids: ids define visibility
    return true;
  }
};

// Bytes of a 16-byte cp.async chunk that hold data; the rest is zero-filled.
__device__ __forceinline__ int cp_bytes16(int nb) {
  if (nb > 16) nb = 16;
  if (nb < 0) nb = 0;
  return nb;
}

// Stage `rows` rows of two tiles with 16-byte cp.async; (row, chunk) advances without division.
// src(row, d, a, b) sets both sources and returns the bytes to copy, in [0, 16].
template <typename T, typename Src>
__device__ __forceinline__ void cp_async_rows16(T* dA, T* dB, int pitch, int rows, int DP,
                                                int tid, int NT, Src src) {
  constexpr int kVec = cp_vec<T>();
  const int cpr = DP / kVec;
  const int adv_r = NT / cpr, adv_d = NT - (NT / cpr) * cpr;
  int r = tid / cpr, dc = tid - (tid / cpr) * cpr;
  for (int c = tid; c < rows * cpr; c += NT) {
    const int d = dc * kVec;
    const T* a;
    const T* b;
    const int nb = src(r, d, a, b);
    cp_async16(&dA[r * pitch + d], a, nb);
    cp_async16(&dB[r * pitch + d], b, nb);
    dc += adv_d;
    r += adv_r;
    if (dc >= cpr) { dc -= cpr; ++r; }
  }
  cp_async_fence();
  cp_async_wait_all();
}

// Element-wise fallback of cp_async_rows16: load(row, d, a, b) sets the values it has (else 0).
template <typename T, typename Load>
__device__ __forceinline__ void copy_rows(T* dA, T* dB, int pitch, int rows, int DP, int tid,
                                          int NT, Load load) {
  for (int idx = tid; idx < rows * DP; idx += NT) {
    const int r = idx / DP;
    const int d = idx - r * DP;
    float a = 0.f, b = 0.f;
    load(r, d, a, b);
    dA[r * pitch + d] = from_f32<T>(a);
    dB[r * pitch + d] = from_f32<T>(b);
  }
}

// CSR bucket of KV block `bid`: groups[gbase + e] for e < total_q / G list its query groups.
struct CsrBucket {
  int64_t gbase;
  int64_t total_q;
};
__device__ __forceinline__ CsrBucket csr_bucket(const int32_t* offsets, int64_t off_sb,
                                                int64_t off_sh, int64_t groups_sb,
                                                int64_t groups_sh, int b, int h, int bid, int G) {
  const int64_t obase = static_cast<int64_t>(b) * off_sb + static_cast<int64_t>(h) * off_sh;
  const int start = offsets[obase + bid];
  const int cnt = offsets[obase + bid + 1] - start;
  return CsrBucket{
      static_cast<int64_t>(b) * groups_sb + static_cast<int64_t>(h) * groups_sh + start,
      static_cast<int64_t>(cnt) * G};
}

// Stage index, chunk, LSE and delta of the nq bucket queries starting at flat query qt.
__device__ __forceinline__ void stage_bucket_queries(
    const int32_t* groups, int64_t gbase, int64_t qt, int nq, int BQ, int G, int chunk_size,
    const Strided3<const float>& lse, const Strided3<const float>& delta, int b, int h, int tid,
    int NT, int* sQidx, int* sQchunk, float* sLse, float* sDel) {
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
}

// cp.async staging needs D fastest, strides multiples of the chunk, and a 16-byte aligned base.
__host__ inline bool cp_async_ok(const at::Tensor& t, int vec) {
  return t.stride(3) == 1 && t.stride(0) % vec == 0 && t.stride(1) % vec == 0 &&
         t.stride(2) % vec == 0 &&
         (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0;
}

// C[Mt x Nt] = (or +=) A[Mt x Kt] * B[Kt x Nt] in shared memory; Mt % 16, Nt % 8, Kt % 8 == 0.
//   A(m,k) = A[m*as0 + k*as1] ; B(k,n) = B[k*bs0 + n*bs1] ; C(m,n) = C[m*cs0 + n]
template <bool ACC, int TPW, typename TA, typename TB>
__device__ __forceinline__ void cta_mma_gemm_tpw(
    float* __restrict__ C, int cs0, const TA* __restrict__ A, int as0, int as1,
    const TB* __restrict__ B, int bs0, int bs1, int Mt, int Nt, int Kt, int warp,
    int nwarps, int lane) {
  const int gid = lane >> 2;
  const int t4 = lane & 3;
  const int nt_total = Nt >> 3;
  const int ngroups = (nt_total + TPW - 1) / TPW;
  const int total_groups = (Mt >> 4) * ngroups;
  for (int gi = warp; gi < total_groups; gi += nwarps) {
    const int mt = gi / ngroups;
    const int nt0 = (gi - mt * ngroups) * TPW;
    const int m0 = mt << 4;
    float acc[TPW][4];
#pragma unroll
    for (int u = 0; u < TPW; ++u) {
      acc[u][0] = 0.f; acc[u][1] = 0.f; acc[u][2] = 0.f; acc[u][3] = 0.f;
    }
    const int arow0 = (m0 + gid) * as0;
    const int arow1 = (m0 + gid + 8) * as0;
    for (int k0 = 0; k0 < Kt; k0 += 8) {
      const int ka = (k0 + t4) * as1;
      const int kb = (k0 + t4 + 4) * as1;
      uint32_t a[4] = {to_tf32(A[arow0 + ka]), to_tf32(A[arow1 + ka]),
                       to_tf32(A[arow0 + kb]), to_tf32(A[arow1 + kb])};
      const int kb0 = (k0 + t4) * bs0;
      const int kb1 = (k0 + t4 + 4) * bs0;
#pragma unroll
      for (int u = 0; u < TPW; ++u) {
        if (nt0 + u < nt_total) {
          const int nb = (((nt0 + u) << 3) + gid) * bs1;
          uint32_t bfrag[2] = {to_tf32(B[kb0 + nb]), to_tf32(B[kb1 + nb])};
          mma_m16n8k8_tf32(acc[u], a, bfrag);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < TPW; ++u) {
      if (nt0 + u < nt_total) {
        const int n0 = ((nt0 + u) << 3) + (t4 << 1);
        float* p0 = C + (m0 + gid) * cs0 + n0;
        float* p1 = C + (m0 + gid + 8) * cs0 + n0;
        if (ACC) {
          p0[0] += acc[u][0]; p0[1] += acc[u][1];
          p1[0] += acc[u][2]; p1[1] += acc[u][3];
        } else {
          p0[0] = acc[u][0]; p0[1] = acc[u][1];
          p1[0] = acc[u][2]; p1[1] = acc[u][3];
        }
      }
    }
  }
}

// Pick the widest n-group that still keeps every warp busy.
template <bool ACC, typename TA, typename TB>
__device__ __forceinline__ void cta_mma_gemm(
    float* __restrict__ C, int cs0, const TA* __restrict__ A, int as0, int as1,
    const TB* __restrict__ B, int bs0, int bs1, int Mt, int Nt, int Kt, int warp,
    int nwarps, int lane) {
  const int tiles = (Mt >> 4) * (Nt >> 3);
  if (tiles >= nwarps * 4)
    cta_mma_gemm_tpw<ACC, 4>(C, cs0, A, as0, as1, B, bs0, bs1, Mt, Nt, Kt, warp, nwarps, lane);
  else if (tiles >= nwarps * 2)
    cta_mma_gemm_tpw<ACC, 2>(C, cs0, A, as0, as1, B, bs0, bs1, Mt, Nt, Kt, warp, nwarps, lane);
  else
    cta_mma_gemm_tpw<ACC, 1>(C, cs0, A, as0, as1, B, bs0, bs1, Mt, Nt, Kt, warp, nwarps, lane);
}

// Shared-memory row pitch in floats, kept odd so tile-on-lane accesses are bank-conflict free.
__host__ inline int smem_row_pitch(int D) { return (D % 2 == 0) ? (D + 1) : (D + 2); }

__host__ inline size_t dq_smem_bytes(int pitch, int BG, int TN) {
  // sQ + sDO + sDQ  [BG, pitch] ; sK + sV [TN, pitch] ; sDS [BG, TN] ; sLse + sDel [BG].
  // BG is always even: the kernels process query rows in pairs (2-way register blocking).
  return static_cast<size_t>((3 * BG + 2 * TN) * pitch + BG * TN + 2 * BG) * sizeof(float);
}

__host__ inline size_t dkv_smem_bytes(int pitch, int TKV, int BQ) {
  // sK + sV + sDK + sDV [TKV, pitch] ; sQ + sDO [BQ, pitch] ; sP + sDS [BQ, TKV] ;
  // sLse + sDel [BQ] floats ; sQidx + sQchunk [BQ] ints
  return static_cast<size_t>((4 * TKV + 2 * BQ) * pitch + 2 * BQ * TKV + 4 * BQ) * sizeof(float);
}

// Largest KV tile we stage (also the dK/dV ownership granularity).
__host__ inline int base_kv_tile(int block_size, int cap) {
  int t = next_pow2(block_size);
  if (t > cap) t = cap;
  if (t < 16) t = 16;
  return t;
}

// Tiles are sized for ~47 KiB of shared memory (3 CTAs/SM); enable_large_smem opts in for large D.
constexpr size_t kBwdSmemBudget = 47 * 1024;

struct TilePlan {
  int tile_n;  // KV tile (also the dK/dV ownership granularity)
  int tile_q;  // query rows per CTA pass; always even (rows are processed in pairs)
  size_t smem;
};

__host__ inline TilePlan plan_dq(int D, int G, int block_size) {
  const int pitch = smem_row_pitch(D);
  int tn = base_kv_tile(block_size, 64);
  int bg = G + (G & 1);
  while (dq_smem_bytes(pitch, bg, tn) > kBwdSmemBudget && tn > 16) tn >>= 1;
  while (dq_smem_bytes(pitch, bg, tn) > kBwdSmemBudget && bg > 2) {
    bg = (bg + 1) / 2;
    bg += (bg & 1);
  }
  return TilePlan{tn, bg, dq_smem_bytes(pitch, bg, tn)};
}

__host__ inline TilePlan plan_dkv(int D, int block_size) {
  const int pitch = smem_row_pitch(D);
  int tkv = base_kv_tile(block_size, 64);
  int bq = 32;
  while (dkv_smem_bytes(pitch, tkv, bq) > kBwdSmemBudget && bq > 8) bq >>= 1;
  while (dkv_smem_bytes(pitch, tkv, bq) > kBwdSmemBudget && tkv > 16) tkv >>= 1;
  while (dkv_smem_bytes(pitch, tkv, bq) > kBwdSmemBudget && bq > 2) bq >>= 1;
  return TilePlan{tkv, bq, dkv_smem_bytes(pitch, tkv, bq)};
}

// The warp-private dQ kernel needs 32 <= D <= 128 (accumulator in registers) and G >= 8.
__host__ inline bool dq_wmma_eligible(int G, int D) {
  return G >= 8 && D >= 32 && D <= 128;   // G >= 8: see dq_mma_profitable below
}

// Row pitch for MMA tiles: P % 16 == 8 keeps k-major fragment loads conflict free.
__host__ __device__ inline int mma_pitch(int n) {
  int p = n + 8;
  if ((p & 15) != 8) p += 8;
  return p;
}
__host__ inline int round_up_i(int a, int b) { return (a + b - 1) / b * b; }

constexpr size_t kMmaSmemBudget = 96 * 1024;
constexpr int kMmaThreads = 256;  // 8 warps: the MMA kernels are latency-bound on staging loads

// The dQ MMA kernel pads query rows to a multiple of 16, so it only wins from G >= 8.
__host__ inline bool dq_mma_profitable(int G) { return G >= 8; }

struct MmaDqPlan {
  int BGP;    // query rows per CTA pass, multiple of 16 (MMA m)
  int TN;     // packed key columns per pass, multiple of 8 (MMA n, then k)
  int DP;     // head dim padded to 8 (MMA k, then n)
  int pitch;  // row pitch of the [., DP] tiles
  int dsp;    // row pitch of the [BGP, TN] score tiles
  size_t smem;
};

// sQ + sDO + sK + sV are staged in `esz`-byte input precision; sDQ, sS and sDP are FP32.
__host__ inline size_t mma_dq_bytes(int BGP, int TN, int DP, int pitch, int T, int esz) {
  const size_t stage = static_cast<size_t>(2 * BGP + 2 * TN) * pitch * esz;
  const size_t f32 = (static_cast<size_t>(BGP) * pitch +
                      static_cast<size_t>(2 * BGP) * mma_pitch(TN) + 2 * BGP) * sizeof(float);
  const size_t aux = static_cast<size_t>(2 * T) * sizeof(void*) +
                     static_cast<size_t>(T + 1 + 2 * TN) * sizeof(int);
  return stage + f32 + aux + 64;
}

__host__ inline MmaDqPlan plan_dq_mma(int D, int G, int T, int esz) {
  MmaDqPlan p{};
  p.DP = round_up_i(D, 8);
  p.pitch = mma_pitch(p.DP);
  int BGP = round_up_i(G, 16);
  if (BGP > 64) BGP = 64;
  int TN = 128;
  auto fits = [&](int bg, int tn) {
    return mma_dq_bytes(bg, tn, p.DP, p.pitch, T, esz) <= kMmaSmemBudget;
  };
  while (!fits(BGP, TN) && TN > 32) TN >>= 1;
  while (!fits(BGP, TN) && BGP > 16) BGP -= 16;
  while (!fits(BGP, TN) && TN > 16) TN >>= 1;
  p.BGP = BGP;
  p.TN = TN;
  p.dsp = mma_pitch(TN);
  p.smem = mma_dq_bytes(BGP, TN, p.DP, p.pitch, T, esz);
  return p;
}

struct MmaDkvPlan {
  int TKV;    // KV rows owned by the CTA, multiple of 16 (MMA n, then m)
  int BQ;     // query rows per pass, multiple of 16 (MMA m, then k)
  int DP, pitch, sp;
  size_t smem;
};

// sK + sV + sQ + sDO staged in input precision; sDK, sDV, sS, sDP are FP32 accumulators.
__host__ inline size_t mma_dkv_bytes(int TKV, int BQ, int pitch, int esz) {
  const size_t stage = static_cast<size_t>(2 * TKV + 2 * BQ) * pitch * esz;
  const size_t f32 = (static_cast<size_t>(2 * TKV) * pitch +
                      static_cast<size_t>(2 * BQ) * mma_pitch(TKV) + 2 * BQ) * sizeof(float);
  return stage + f32 + static_cast<size_t>(2 * BQ) * sizeof(int) + 64;
}

__host__ inline MmaDkvPlan plan_dkv_mma(int D, int block_size, int esz) {
  MmaDkvPlan p{};
  p.DP = round_up_i(D, 8);
  p.pitch = mma_pitch(p.DP);
  // TKV=16, BQ=32 is slower: with cp.async these kernels are tile-size bound.
  int TKV = round_up_i(block_size, 16);
  if (TKV > 64) TKV = 64;
  int BQ = 64;
  auto fits = [&](int tkv, int bq) {
    return mma_dkv_bytes(tkv, bq, p.pitch, esz) <= kMmaSmemBudget;
  };
  // Shrink BQ first: a bigger TKV halves the CTA count per KV block and the gathered Q/dO traffic.
  while (!fits(TKV, BQ) && BQ > 32) BQ >>= 1;
  while (!fits(TKV, BQ) && TKV > 16) TKV -= 16;
  while (!fits(TKV, BQ) && BQ > 16) BQ >>= 1;
  p.TKV = TKV;
  p.BQ = BQ;
  p.sp = mma_pitch(TKV);
  p.smem = mma_dkv_bytes(TKV, BQ, p.pitch, esz);
  return p;
}

// Opt into >48 KiB dynamic shared memory for one kernel instantiation.
template <typename KernelT>
__host__ inline void enable_large_smem(KernelT kernel, size_t smem, const char* what) {
  if (smem <= 48u * 1024u) return;
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(smem <= static_cast<size_t>(props->sharedMemPerBlockOptin), what,
              ": needs ", smem, " B of shared memory but the device allows at most ",
              props->sharedMemPerBlockOptin, " B");
  C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)));
}

}  // namespace omnimem_nsa
