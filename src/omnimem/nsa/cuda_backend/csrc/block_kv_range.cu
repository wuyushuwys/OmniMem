// K2: range register / clear of the block address table.
// n_valid_value < 0 leaves n_valid unchanged, so a K-only register advertises no unpaired row.
// Addresses are generated in int64 and never dereferenced.
#include "block_kv_common.cuh"
#include "block_kv_ops.h"

namespace omnimem_block_kv {
namespace {

constexpr int kRangeBlock = 128;
constexpr int64_t kMaxGridY = 65535;

__global__ void block_range_kernel(int64_t* table, int32_t* n_valid, int64_t first_bid,
                                   int64_t n_blocks, int64_t k_base, int64_t v_base,
                                   int64_t head_stride_bytes, int64_t block_bytes,
                                   int64_t n_valid_value, int64_t heads, int64_t mb,
                                   bool write_k, bool write_v, bool clear) {
  const int64_t h = static_cast<int64_t>(blockIdx.x);
  const int64_t jb = static_cast<int64_t>(blockIdx.y);
  const int64_t j = jb * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  if (j < n_blocks) {
    const int64_t col = first_bid + j;
    int64_t val_k = 0;
    int64_t val_v = 0;
    if (!clear) {
      const int64_t off = h * head_stride_bytes + j * block_bytes;
      val_k = k_base + off;
      val_v = v_base + off;
    }
    if (write_k) table[(0 * heads + h) * mb + col] = val_k;
    if (write_v) table[(1 * heads + h) * mb + col] = val_v;
  }
  // One store per head from the first column tile, independent of the column mask.
  if (jb == 0 && threadIdx.x == 0 && n_valid_value >= 0) {
    n_valid[h] = static_cast<int32_t>(n_valid_value);
  }
}

}  // namespace

void block_range(at::Tensor& table, at::Tensor& n_valid, int64_t first_bid,
                 int64_t n_blocks, int64_t k_base, int64_t v_base,
                 int64_t head_stride_bytes, int64_t block_bytes, int64_t n_valid_value,
                 bool write_k, bool write_v, bool clear) {
  TORCH_CHECK(table.is_cuda(), "block-KV K2: table must be a CUDA tensor");
  TORCH_CHECK(table.scalar_type() == at::kLong, "block-KV K2: table must be int64, got ",
              table.scalar_type());
  TORCH_CHECK(table.is_contiguous(), "block-KV K2: table must be contiguous");
  TORCH_CHECK(table.dim() == 3 && table.size(0) == 2,
              "block-KV K2: table must be [2, H, MB], got ", table.sizes());
  const int64_t heads = table.size(1);
  const int64_t mb = table.size(2);
  TORCH_CHECK(n_valid.is_cuda(), "block-KV K2: n_valid must be a CUDA tensor");
  TORCH_CHECK(n_valid.scalar_type() == at::kInt,
              "block-KV K2: n_valid must be int32, got ", n_valid.scalar_type());
  TORCH_CHECK(n_valid.is_contiguous(), "block-KV K2: n_valid must be contiguous");
  TORCH_CHECK(n_valid.dim() == 1 && n_valid.size(0) == heads,
              "block-KV K2: n_valid must be [H=", heads, "], got ", n_valid.sizes());
  TORCH_CHECK(n_valid.device() == table.device(),
              "block-KV K2: table on ", table.device(), ", n_valid on ",
              n_valid.device());
  TORCH_CHECK(n_blocks > 0 && first_bid >= 0 && first_bid + n_blocks <= mb,
              "block-KV K2: block range [", first_bid, ", ", first_bid + n_blocks,
              ") outside [0, ", mb, ") -- the empty range is the facade's business");
  TORCH_CHECK(n_valid_value <= static_cast<int64_t>(2147483647),
              "block-KV K2: n_valid_value=", n_valid_value, " does not fit int32");
  if (!clear) {
    // A selected name with a NULL base would publish offsets the selection kernel reads as valid.
    TORCH_CHECK(write_k || write_v, "block-KV K2: nothing to register");
    TORCH_CHECK(!write_k || k_base != 0,
                "block-KV K2: write_k with k_base == 0 (use write_k=false to skip the "
                "K row, or clear=true to zero it)");
    TORCH_CHECK(!write_v || v_base != 0,
                "block-KV K2: write_v with v_base == 0 (use write_v=false to skip the "
                "V row, or clear=true to zero it)");
  }

  const int64_t tiles = cdiv64(n_blocks, kRangeBlock);
  TORCH_CHECK(tiles <= kMaxGridY, "block-KV K2: n_blocks=", n_blocks, " needs ", tiles,
              " column tiles, which exceeds the gridDim.y limit of ", kMaxGridY);
  const dim3 grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(tiles));
  const dim3 block(static_cast<unsigned int>(kRangeBlock));

  at::cuda::CUDAGuard guard(table.device());
  const cudaStream_t stream = current_stream(table);
  block_range_kernel<<<grid, block, 0, stream>>>(
      table.data_ptr<int64_t>(), n_valid.data_ptr<int32_t>(), first_bid, n_blocks, k_base,
      v_base, head_stride_bytes, block_bytes, n_valid_value, heads, mb, write_k, write_v,
      clear);
  OMNIMEM_BLOCK_KV_CHECK_LAUNCH();
}

}  // namespace omnimem_block_kv
