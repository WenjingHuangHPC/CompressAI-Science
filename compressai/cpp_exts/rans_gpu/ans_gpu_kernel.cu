// compressai/cpp_exts/rans_gpu/ans_gpu_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>

#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>

#include "rans64_gpu.cuh"
#include "rans64dec_gpu.cuh"

constexpr int precision = 16;
constexpr uint16_t bypass_precision = 4;
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;
constexpr int P_MAX = 512;

__device__ __forceinline__ int32_t cdf_find_symbol_bs(
    const int32_t* __restrict__ cdf, int32_t cdf_size, uint32_t cum_freq
) {
  int32_t lo = 0;
  int32_t hi = cdf_size - 2;
  while (lo < hi) {
    int32_t mid = (lo + hi) >> 1;
    if ((uint32_t)cdf[mid + 1] > cum_freq) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

__host__ __device__ __forceinline__ int64_t align16_i64(int64_t x) {
  return (x + 15) & ~((int64_t)15);
}

// ============================================================
// Stage1 temp-slot encoder (same idea as your packed version)
// ============================================================

__global__ void encode_chunks_into_arena_kernel(
    const int32_t* __restrict__ symbols_bxn,   // [B,N]
    const int32_t* __restrict__ indexes_bxn,   // [B,N]
    int B, int N,
    const int32_t* __restrict__ cdfs_mxl,      // [M,Lmax]
    int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,   // [M]
    const int32_t* __restrict__ offsets_m,     // [M]
    int P,
    int chunk_len,
    uint8_t* __restrict__ arena_u8,            // [B*stride]
    int64_t stride,
    int64_t header_bytes_padded,
    int cap_words_chunk,
    int32_t* __restrict__ used_words_flat      // [B*P]
) {
  int b = (int)blockIdx.x;
  int c = (int)blockIdx.y;
  if (b >= B || c >= P) return;
  if (threadIdx.x != 0) return;

  int start = c * chunk_len;
  if (start >= N) {
    used_words_flat[b * P + c] = 0;
    return;
  }
  int end = start + chunk_len;
  if (end > N) end = N;

  const int32_t* sym = symbols_bxn + (int64_t)b * N;
  const int32_t* idx = indexes_bxn + (int64_t)b * N;

  uint8_t* stream_base = arena_u8 + (int64_t)b * stride;
  uint8_t* payload_base = stream_base + header_bytes_padded;

  int cap_bytes_chunk = cap_words_chunk * 4;
  uint8_t* slot = payload_base + (int64_t)c * cap_bytes_chunk;

  uint32_t* base_u32 = reinterpret_cast<uint32_t*>(slot);
  uint32_t* ptr = base_u32 + cap_words_chunk;

  Rans64State r;
  Rans64EncInit(&r);

  int32_t last_cdf_idx = -1;
  const int32_t* last_cdf = nullptr;
  int32_t last_cdf_size = 0;
  int32_t last_offsetv = 0;

  for (int i = end - 1; i >= start; --i) {
    int32_t cdf_idx = idx[i];

    const int32_t* cdf;
    int32_t cdf_size;
    int32_t offsetv;
    if (cdf_idx == last_cdf_idx) {
      cdf = last_cdf;
      cdf_size = last_cdf_size;
      offsetv = last_offsetv;
    } else {
      cdf = cdfs_mxl + (int64_t)cdf_idx * Lmax;
      cdf_size = cdf_sizes_m[cdf_idx];
      offsetv = offsets_m[cdf_idx];
      last_cdf_idx = cdf_idx;
      last_cdf = cdf;
      last_cdf_size = cdf_size;
      last_offsetv = offsetv;
    }

    int32_t max_value = cdf_size - 2;
    int32_t value = sym[i] - offsetv;

    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = (uint32_t)(-2 * value - 1);
      value = max_value;
    } else if (value >= max_value) {
      raw_val = (uint32_t)(2 * (value - max_value));
      value = max_value;
    }

    if (value == max_value) {
      int32_t n_bypass = 0;
      if (raw_val) {
        int msb = 31 - __clz(raw_val);
        n_bypass = (msb / bypass_precision) + 1;
      }

      for (int32_t j = n_bypass - 1; j >= 0; --j) {
        uint32_t v = (raw_val >> (j * bypass_precision)) & max_bypass_val;
        Rans64EncPutBits(&r, &ptr, v, bypass_precision);
      }

      int32_t val = n_bypass;
      int32_t n_full = 0;
      while (val >= (int32_t)max_bypass_val) { val -= max_bypass_val; n_full++; }
      Rans64EncPutBits(&r, &ptr, (uint32_t)val, bypass_precision);
      for (int k = 0; k < n_full; ++k) {
        Rans64EncPutBits(&r, &ptr, (uint32_t)max_bypass_val, bypass_precision);
      }
    }

    uint32_t start_c = (uint32_t)cdf[value];
    uint32_t freq = (uint32_t)(cdf[value + 1] - cdf[value]);
    Rans64EncPut(&r, &ptr, start_c, freq, precision);
  }

  Rans64EncFlush(&r, &ptr);

  int used_words = (int)(base_u32 + cap_words_chunk - ptr);
  used_words_flat[b * P + c] = used_words;
}

// used_words -> sizes_u16(bytes) with overflow flag
__global__ void used_words_to_sizes_u16_kernel(
    const int32_t* __restrict__ used_words_flat, // [B*P]
    uint16_t* __restrict__ sizes_u16,            // [B*P]
    int32_t* __restrict__ overflow_flag,         // [1]
    int n
) {
  int i = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int bytes = used_words_flat[i] * 4;
  if (bytes > 65535) {
    if (overflow_flag) atomicExch(overflow_flag, 1);
    bytes = 65535; // safety; real behavior is to error on host
  }
  sizes_u16[i] = (uint16_t)bytes;
}

__global__ void u16_to_i32_kernel(
    const uint16_t* __restrict__ in_u16,
    int32_t* __restrict__ out_i32,
    int n
) {
  int i = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out_i32[i] = (int32_t)in_u16[i];
}

// Write global header once:
// header = [u32 N][u32 chunk_len][u16 P][u16 B][u16 sizes[B*P]] + pad
__global__ void write_tight_global_header_kernel(
    uint8_t* __restrict__ packed_u8,
    int N, int chunk_len, int P, int B,
    const uint16_t* __restrict__ sizes_u16, // [B*P]
    int n_sizes
) {
  // one block is enough; but safe if launched with one block
  if (blockIdx.x != 0) return;

  if (threadIdx.x == 0) {
    uint32_t* h32 = reinterpret_cast<uint32_t*>(packed_u8);
    h32[0] = (uint32_t)N;
    h32[1] = (uint32_t)chunk_len;

    uint16_t* h16 = reinterpret_cast<uint16_t*>(packed_u8 + 8);
    h16[0] = (uint16_t)P;
    h16[1] = (uint16_t)B;
  }

  // sizes start at packed_u8 + 12
  uint16_t* out_sizes = reinterpret_cast<uint16_t*>(packed_u8 + 12);
  for (int i = threadIdx.x; i < n_sizes; i += blockDim.x) {
    out_sizes[i] = sizes_u16[i];
  }
}

// Pack payload from temp slots into tight payload area
__global__ void pack_tight_payload_kernel(
    const uint8_t* __restrict__ temp_arena_u8,
    int64_t temp_stride,
    int64_t temp_header_padded,
    int cap_bytes_chunk,
    int B, int P,
    const uint16_t* __restrict__ sizes_u16,        // [B*P]
    const int32_t* __restrict__ chunk_offsets_i32, // [B*P], exclusive
    uint8_t* __restrict__ packed_u8,
    int64_t header_bytes
) {
  int b = (int)blockIdx.x;
  int c = (int)blockIdx.y;
  if (b >= B || c >= P) return;

  int k = b * P + c;
  uint16_t used = sizes_u16[k];
  if (used == 0) return;

  const uint8_t* stream_base = temp_arena_u8 + (int64_t)b * temp_stride;
  const uint8_t* payload_base = stream_base + temp_header_padded;
  const uint8_t* slot = payload_base + (int64_t)c * cap_bytes_chunk;
  const uint8_t* src = slot + (cap_bytes_chunk - (int)used);

  uint8_t* out_payload = packed_u8 + header_bytes;
  uint8_t* dst = out_payload + (int64_t)chunk_offsets_i32[k];

  for (int i = threadIdx.x; i < (int)used; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// ============================================================
// Tight decode kernel: one block per stream, one thread per chunk
// Uses chunk_offsets (computed on GPU) + sizes_u16
// ============================================================

__global__ void decode_streams_tight_kernel(
    const uint8_t* __restrict__ packed_u8,
    int64_t header_bytes,
    const uint16_t* __restrict__ sizes_u16_flat,        // [B*P]
    const int32_t* __restrict__ chunk_offsets_i32_flat, // [B*P]
    int B, int P, int N, int chunk_len,
    const int32_t* __restrict__ indexes_bxn,   // [B,N]
    const int32_t* __restrict__ cdfs_mxl,      // [M,Lmax]
    int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,   // [M]
    const int32_t* __restrict__ offsets_m,     // [M]
    int32_t* __restrict__ out_symbols_bxn      // [B,N]
) {
  int b = (int)blockIdx.x;
  if (b >= B) return;

  int c = (int)threadIdx.x;
  if (c >= P) return;

  int start = c * chunk_len;
  if (start >= N) return;
  int end = start + chunk_len;
  if (end > N) end = N;

  int k = b * P + c;
  uint16_t used = sizes_u16_flat[k];
  if (used == 0) return;

  const uint8_t* payload = packed_u8 + header_bytes;
  const uint8_t* chunk_bs = payload + (int64_t)chunk_offsets_i32_flat[k];
  const uint32_t* p = reinterpret_cast<const uint32_t*>(chunk_bs);

  const int32_t* idx = indexes_bxn + (int64_t)b * N;
  int32_t* out = out_symbols_bxn + (int64_t)b * N;

  Rans64State r;
  Rans64DecInit(&r, &p);

  int32_t last_cdf_idx = -1;
  const int32_t* last_cdf = nullptr;
  int32_t last_cdf_size = 0;
  int32_t last_max_value = 0;
  int32_t last_offsetv = 0;

  for (int i = start; i < end; ++i) {
    int32_t cdf_idx = idx[i];

    const int32_t* cdf;
    int32_t cdf_size, max_value, offsetv;

    if (cdf_idx == last_cdf_idx) {
      cdf = last_cdf;
      cdf_size = last_cdf_size;
      max_value = last_max_value;
      offsetv = last_offsetv;
    } else {
      cdf = cdfs_mxl + (int64_t)cdf_idx * Lmax;
      cdf_size = cdf_sizes_m[cdf_idx];
      max_value = cdf_size - 2;
      offsetv = offsets_m[cdf_idx];
      last_cdf_idx = cdf_idx;
      last_cdf = cdf;
      last_cdf_size = cdf_size;
      last_max_value = max_value;
      last_offsetv = offsetv;
    }

    uint32_t cum_freq = Rans64DecGet(&r, precision);
    int32_t s = cdf_find_symbol_bs(cdf, cdf_size, cum_freq);

    uint32_t start_c = (uint32_t)cdf[s];
    uint32_t freq = (uint32_t)(cdf[s + 1] - cdf[s]);
    Rans64DecAdvance(&r, &p, start_c, freq, precision);

    int32_t value = s;

    if (value == max_value) {
      int32_t val = (int32_t)Rans64DecGetBits(&r, &p, bypass_precision);
      int32_t n_bypass = val;
      while (val == (int32_t)max_bypass_val) {
        val = (int32_t)Rans64DecGetBits(&r, &p, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = (int32_t)Rans64DecGetBits(&r, &p, bypass_precision);
        raw_val |= (val << (j * bypass_precision));
      }

      value = raw_val >> 1;
      if (raw_val & 1) value = -value - 1;
      else value += max_value;
    }

    out[i] = value + offsetv;
  }
}

// ============================================================
// Public APIs (tight)
// ============================================================

std::vector<torch::Tensor> encode_with_indexes_tight_cuda(
    torch::Tensor symbols_bxn,   // CUDA int32 [B,N]
    torch::Tensor indexes_bxn,   // CUDA int32 [B,N]
    torch::Tensor cdfs_mxl,      // CUDA int32 [M,Lmax]
    torch::Tensor cdf_sizes_m,   // CUDA int32 [M]
    torch::Tensor offsets_m,     // CUDA int32 [M]
    int64_t P_in
) {
  TORCH_CHECK(symbols_bxn.is_cuda(), "symbols_bxn must be CUDA");
  TORCH_CHECK(indexes_bxn.is_cuda(), "indexes_bxn must be CUDA");
  TORCH_CHECK(cdfs_mxl.is_cuda(), "cdfs_mxl must be CUDA");
  TORCH_CHECK(cdf_sizes_m.is_cuda(), "cdf_sizes_m must be CUDA");
  TORCH_CHECK(offsets_m.is_cuda(), "offsets_m must be CUDA");

  const int B = (int)symbols_bxn.size(0);
  const int N = (int)symbols_bxn.size(1);
  const int Lmax = (int)cdfs_mxl.size(1);

  int P = (int)P_in;
  if (P < 1) P = 1;
  if (P > N) P = N;
  if (P > P_MAX) P = P_MAX;

  const int chunk_len = (N + P - 1) / P;

  // temp slot capacity estimate (same as your packed version)
  const int cap_words_chunk = chunk_len * 12 + 8;
  const int cap_bytes_chunk = cap_words_chunk * 4;

  // temp arena layout: minimal header (we don't use it, but keep padding)
  const int64_t temp_header_raw = (int64_t)4 * (P + 2);
  const int64_t temp_header_padded = align16_i64(temp_header_raw);
  const int64_t temp_stride = temp_header_padded + (int64_t)P * (int64_t)cap_bytes_chunk;

  auto dev = symbols_bxn.device();
  auto opts_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
  auto opts_u16 = torch::TensorOptions().dtype(torch::kUInt16).device(dev);

  auto temp_arena_u8 = torch::empty({(long long)(temp_stride * (int64_t)B)}, opts_u8);
  auto used_words_flat = torch::zeros({B * P}, opts_i32);

  dim3 grid(B, P, 1);
  encode_chunks_into_arena_kernel<<<grid, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
      symbols_bxn.data_ptr<int32_t>(),
      indexes_bxn.data_ptr<int32_t>(),
      B, N,
      cdfs_mxl.data_ptr<int32_t>(),
      Lmax,
      cdf_sizes_m.data_ptr<int32_t>(),
      offsets_m.data_ptr<int32_t>(),
      P,
      chunk_len,
      temp_arena_u8.data_ptr<uint8_t>(),
      temp_stride,
      temp_header_padded,
      cap_words_chunk,
      used_words_flat.data_ptr<int32_t>());

  // used_words -> sizes_u16
  auto sizes_u16_flat = torch::empty({B * P}, opts_u16);
  auto overflow_flag = torch::zeros({1}, opts_i32);

  {
    int n = B * P;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    used_words_to_sizes_u16_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        used_words_flat.data_ptr<int32_t>(),
        sizes_u16_flat.data_ptr<uint16_t>(),
        overflow_flag.data_ptr<int32_t>(),
        n);
  }

  // Check overflow (host sync on 1 int32)
  int32_t overflow = overflow_flag.cpu().item<int32_t>();
  TORCH_CHECK(overflow == 0, "Chunk compressed size overflow (>65535 bytes). Use smaller P or larger chunk_len, or switch sizes to uint32.");

  // sizes_u16 -> sizes_i32
  auto sizes_i32_flat = torch::empty({B * P}, opts_i32);
  {
    int n = B * P;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    u16_to_i32_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        sizes_u16_flat.data_ptr<uint16_t>(),
        sizes_i32_flat.data_ptr<int32_t>(),
        n);
  }

  // chunk_offsets = exclusive scan(sizes_i32) length B*P
  auto chunk_offsets_i32_flat = torch::empty({B * P}, opts_i32);
  auto stream = at::cuda::getDefaultCUDAStream();

  size_t scan_temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr, scan_temp_bytes,
      (const int32_t*)nullptr,
      (int32_t*)nullptr,
      B * P,
      stream.stream());

  auto scan_temp = torch::empty({(long long)scan_temp_bytes},
                                torch::TensorOptions().dtype(torch::kUInt8).device(dev));

  cub::DeviceScan::ExclusiveSum(
      scan_temp.data_ptr(),
      scan_temp_bytes,
      sizes_i32_flat.data_ptr<int32_t>(),
      chunk_offsets_i32_flat.data_ptr<int32_t>(),
      B * P,
      stream.stream());

  // total_payload = last_offset + last_size (host sync on 2 ints)
  int32_t last_off = chunk_offsets_i32_flat.index({B * P - 1}).cpu().item<int32_t>();
  int32_t last_sz  = sizes_i32_flat.index({B * P - 1}).cpu().item<int32_t>();
  int32_t total_payload_bytes = last_off + last_sz;

  // tight header bytes
  // header_raw = [u32 N][u32 chunk_len] + [u16 P][u16 B] + [u16 sizes(B*P)]
  int64_t header_raw = 12 + (int64_t)2 * (int64_t)(B * P);
  int64_t header_bytes = align16_i64(header_raw);

  int64_t total_bytes = header_bytes + (int64_t)total_payload_bytes;

  auto packed_u8 = torch::empty({(long long)total_bytes}, opts_u8);

  // write header (one block)
  write_tight_global_header_kernel<<<1, 256, 0, at::cuda::getDefaultCUDAStream()>>>(
      packed_u8.data_ptr<uint8_t>(),
      N, chunk_len, P, B,
      sizes_u16_flat.data_ptr<uint16_t>(),
      B * P);

  // pack payload
  pack_tight_payload_kernel<<<dim3(B, P, 1), 256, 0, at::cuda::getDefaultCUDAStream()>>>(
      temp_arena_u8.data_ptr<uint8_t>(),
      temp_stride,
      temp_header_padded,
      cap_bytes_chunk,
      B, P,
      sizes_u16_flat.data_ptr<uint16_t>(),
      chunk_offsets_i32_flat.data_ptr<int32_t>(),
      packed_u8.data_ptr<uint8_t>(),
      header_bytes);

  // return sizes as [B,P] for convenience + header_bytes_cpu + chunk_len_cpu + P_cpu (optional)
  auto sizes_u16 = sizes_u16_flat.view({B, P});

  auto header_bytes_cpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  header_bytes_cpu[0] = header_bytes;

  auto chunk_len_cpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  chunk_len_cpu[0] = chunk_len;

  auto P_cpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  P_cpu[0] = P;

  return {packed_u8, sizes_u16, header_bytes_cpu, chunk_len_cpu, P_cpu};
}

torch::Tensor decode_with_indexes_tight_cuda(
    torch::Tensor packed_u8,      // CUDA uint8 [total]
    torch::Tensor sizes_u16,      // CUDA uint16 [B,P]
    torch::Tensor header_bytes_cpu,// CPU int64 [1]
    torch::Tensor chunk_len_cpu,  // CPU int32 [1]
    torch::Tensor P_cpu,          // CPU int32 [1]
    torch::Tensor indexes_bxn,    // CUDA int32 [B,N]
    torch::Tensor cdfs_mxl,       // CUDA int32 [M,Lmax]
    torch::Tensor cdf_sizes_m,    // CUDA int32 [M]
    torch::Tensor offsets_m       // CUDA int32 [M]
) {
  TORCH_CHECK(packed_u8.is_cuda(), "packed_u8 must be CUDA");
  TORCH_CHECK(sizes_u16.is_cuda(), "sizes_u16 must be CUDA");
  TORCH_CHECK(indexes_bxn.is_cuda(), "indexes_bxn must be CUDA");
  TORCH_CHECK(cdfs_mxl.is_cuda(), "cdfs_mxl must be CUDA");
  TORCH_CHECK(cdf_sizes_m.is_cuda(), "cdf_sizes_m must be CUDA");
  TORCH_CHECK(offsets_m.is_cuda(), "offsets_m must be CUDA");

  int B = (int)indexes_bxn.size(0);
  int N = (int)indexes_bxn.size(1);
  int Lmax = (int)cdfs_mxl.size(1);

  int64_t header_bytes = header_bytes_cpu[0].item<int64_t>();
  int chunk_len = chunk_len_cpu[0].item<int32_t>();
  int P = P_cpu[0].item<int32_t>();

  TORCH_CHECK(P >= 1 && P <= P_MAX, "Invalid P");
  TORCH_CHECK(sizes_u16.size(0) == B && sizes_u16.size(1) == P, "sizes_u16 shape must be [B,P]");

  auto dev = packed_u8.device();
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);

  // sizes_u16 -> sizes_i32_flat
  auto sizes_u16_flat = sizes_u16.contiguous().view({B * P});
  auto sizes_i32_flat = torch::empty({B * P}, opts_i32);
  {
    int n = B * P;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    u16_to_i32_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        sizes_u16_flat.data_ptr<uint16_t>(),
        sizes_i32_flat.data_ptr<int32_t>(),
        n);
  }

  // chunk_offsets = exclusive scan(sizes_i32) length B*P
  auto chunk_offsets_i32_flat = torch::empty({B * P}, opts_i32);
  auto stream = at::cuda::getDefaultCUDAStream();

  size_t scan_temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr, scan_temp_bytes,
      (const int32_t*)nullptr,
      (int32_t*)nullptr,
      B * P,
      stream.stream());

  auto scan_temp = torch::empty({(long long)scan_temp_bytes},
                                torch::TensorOptions().dtype(torch::kUInt8).device(dev));

  cub::DeviceScan::ExclusiveSum(
      scan_temp.data_ptr(),
      scan_temp_bytes,
      sizes_i32_flat.data_ptr<int32_t>(),
      chunk_offsets_i32_flat.data_ptr<int32_t>(),
      B * P,
      stream.stream());

  // output
  auto out = torch::empty({B, N}, opts_i32);

  // one block per stream, threads=512
  decode_streams_tight_kernel<<<B, 512, 0, at::cuda::getDefaultCUDAStream()>>>(
      packed_u8.data_ptr<uint8_t>(),
      header_bytes,
      sizes_u16_flat.data_ptr<uint16_t>(),
      chunk_offsets_i32_flat.data_ptr<int32_t>(),
      B, P, N, chunk_len,
      indexes_bxn.data_ptr<int32_t>(),
      cdfs_mxl.data_ptr<int32_t>(),
      Lmax,
      cdf_sizes_m.data_ptr<int32_t>(),
      offsets_m.data_ptr<int32_t>(),
      out.data_ptr<int32_t>());

  return out;
}
