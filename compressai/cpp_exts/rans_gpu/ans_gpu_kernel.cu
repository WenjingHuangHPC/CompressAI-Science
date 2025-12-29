// compressai/cpp_exts/rans_gpu/ans_gpu_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>

#include "rans64_gpu.cuh"
#include "rans64dec_gpu.cuh"

constexpr int precision = 16;
constexpr uint16_t bypass_precision = 4;
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

// ---- Chunked (P-way) stream header format ----
// header words (uint32):
//   hdr[0] = P
//   hdr[1] = chunk_len
//   hdr[2 + c] = chunk_size_bytes[c]  for c in [0,P)
// header_bytes = 4 * (P + 2)

__global__ void words_to_bytes_kernel_flat(
    const int32_t* __restrict__ used_words_flat,  // [B*P]
    int64_t* __restrict__ sizes_bytes_flat,       // [B*P]
    int total
) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < total) sizes_bytes_flat[t] = (int64_t)used_words_flat[t] * 4;
}

__global__ void sum_chunk_sizes_kernel(
    const int64_t* __restrict__ chunk_sizes_flat, // [B*P]
    int64_t* __restrict__ stream_sizes,           // [B]
    int B, int P, int64_t header_bytes
) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  const int64_t* row = chunk_sizes_flat + (int64_t)b * P;
  int64_t s = header_bytes;
  for (int c = 0; c < P; ++c) s += row[c];
  stream_sizes[b] = s;
}

__global__ void compute_chunk_payload_offsets_kernel(
    const int64_t* __restrict__ stream_offsets,       // [B]
    const int64_t* __restrict__ chunk_sizes_flat,     // [B*P]
    int64_t* __restrict__ chunk_payload_offsets_flat, // [B*P]
    int B, int P, int64_t header_bytes
) {
  int b = blockIdx.x;
  if (b >= B) return;
  int64_t base = stream_offsets[b] + header_bytes;
  const int64_t* row = chunk_sizes_flat + (int64_t)b * P;
  int64_t* out = chunk_payload_offsets_flat + (int64_t)b * P;
  int64_t acc = 0;
  for (int c = 0; c < P; ++c) {
    out[c] = base + acc;
    acc += row[c];
  }
}

__global__ void write_headers_kernel(
    uint8_t* __restrict__ out_bytes,
    const int64_t* __restrict__ stream_offsets,     // [B]
    const int64_t* __restrict__ chunk_sizes_flat,   // [B*P]
    int B, int P, int chunk_len
) {
  int b = blockIdx.x;
  if (b >= B) return;
  uint8_t* base = out_bytes + stream_offsets[b];
  uint32_t* hdr = reinterpret_cast<uint32_t*>(base);
  hdr[0] = (uint32_t)P;
  hdr[1] = (uint32_t)chunk_len;
  const int64_t* row = chunk_sizes_flat + (int64_t)b * P;
  for (int c = 0; c < P; ++c) {
    hdr[2 + c] = (uint32_t)row[c];
  }
}

__global__ void words_to_bytes_kernel(
    const int32_t* __restrict__ used_words,
    int64_t* __restrict__ sizes_bytes,
    int B
) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < B) sizes_bytes[b] = (int64_t)used_words[b] * 4;
}

// One CUDA block handles one chunk of one stream (one batch item), single thread for correctness.
__global__ void encode_chunks_kernel(
    const int32_t* __restrict__ symbols_bxn,   // [B,N]
    const int32_t* __restrict__ indexes_bxn,   // [B,N]
    int B, int N,
    const int32_t* __restrict__ cdfs_mxl,      // [M,Lmax]
    int M, int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,   // [M]
    const int32_t* __restrict__ offsets_m,     // [M]
    int P,
    int chunk_len,
    uint32_t* __restrict__ out_words_flat,     // [B*P, cap_words_chunk]
    int cap_words_chunk,
    int32_t* __restrict__ out_used_words_flat  // [B*P]
) {
  int b = (int)blockIdx.x;
  int c = (int)blockIdx.y;
  if (b >= B || c >= P) return;
  if (threadIdx.x != 0) return;

  int start = c * chunk_len;
  if (start >= N) {
    // empty chunk
    out_used_words_flat[b * P + c] = 0;
    return;
  }
  int end = start + chunk_len;
  if (end > N) end = N;

  const int32_t* sym = symbols_bxn + (int64_t)b * N;
  const int32_t* idx = indexes_bxn + (int64_t)b * N;

  int flat = b * P + c;
  uint32_t* base = out_words_flat + (int64_t)flat * cap_words_chunk;
  uint32_t* ptr = base + cap_words_chunk;

  Rans64State r;
  Rans64EncInit(&r);

  for (int i = end - 1; i >= start; --i) {
    int32_t cdf_idx = idx[i];
    const int32_t* cdf = cdfs_mxl + (int64_t)cdf_idx * Lmax;
    int32_t cdf_size = cdf_sizes_m[cdf_idx];
    int32_t max_value = cdf_size - 2;
    int32_t value = sym[i] - offsets_m[cdf_idx];

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
      while ((raw_val >> (n_bypass * bypass_precision)) != 0) ++n_bypass;

      for (int32_t j = n_bypass - 1; j >= 0; --j) {
        uint32_t v = (raw_val >> (j * bypass_precision)) & max_bypass_val;
        Rans64EncPutBits(&r, &ptr, v, bypass_precision);
      }

      int32_t val = n_bypass;
      int32_t n_full = 0;
      while (val >= max_bypass_val) { val -= max_bypass_val; n_full++; }
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

  int used_words = (int)(base + cap_words_chunk - ptr);
  out_used_words_flat[flat] = used_words;
}


__global__ void pack_chunk_bytes_kernel(
    const uint32_t* __restrict__ in_words_flat,   // [B*P, cap_words_chunk]
    int cap_words_chunk,
    const int32_t* __restrict__ used_words_flat,  // [B*P]
    const int64_t* __restrict__ chunk_payload_offsets_flat, // [B*P] byte offsets into out_bytes
    uint8_t* __restrict__ out_bytes,
    int B, int P
) {
  int b = (int)blockIdx.x;
  int c = (int)blockIdx.y;
  if (b >= B || c >= P) return;

  int flat = b * P + c;
  int uw = used_words_flat[flat];
  if (uw <= 0) return;

  int start_word = cap_words_chunk - uw;

  const uint8_t* src = reinterpret_cast<const uint8_t*>(
      in_words_flat + (int64_t)flat * cap_words_chunk + start_word);

  uint8_t* dst = out_bytes + chunk_payload_offsets_flat[flat];

  int nbytes = uw * 4;

  for (int i = threadIdx.x; i < nbytes; i += blockDim.x) {
    dst[i] = src[i];
  }
}

std::vector<torch::Tensor> encode_with_indexes_cuda(
    torch::Tensor symbols_bxn,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int64_t P_in
) {
  const auto B = (int)symbols_bxn.size(0);
  const auto N = (int)symbols_bxn.size(1);
  const auto M = (int)cdfs_mxl.size(0);
  const auto Lmax = (int)cdfs_mxl.size(1);

  int P = (int)P_in;
  if (P < 1) P = 1;
  if (P > N) P = N; // avoid too many empty chunks

  int chunk_len = (N + P - 1) / P; // ceil
  int cap_words_chunk = chunk_len * 12 + 8;

  auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(symbols_bxn.device());
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(symbols_bxn.device());
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(symbols_bxn.device());
  auto opts_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(symbols_bxn.device());

  // Flatten chunks: total_chunks = B * P
  int total_chunks = B * P;

  auto out_words = torch::empty({total_chunks, cap_words_chunk}, opts_u32);
  auto used_words = torch::zeros({total_chunks}, opts_i32);

  // Encode each chunk in parallel: grid(B, P)
  dim3 grid(B, P, 1);
  encode_chunks_kernel<<<grid, 1>>>(
      symbols_bxn.data_ptr<int32_t>(),
      indexes_bxn.data_ptr<int32_t>(),
      B, N,
      cdfs_mxl.data_ptr<int32_t>(),
      M, Lmax,
      cdf_sizes_m.data_ptr<int32_t>(),
      offsets_m.data_ptr<int32_t>(),
      P,
      chunk_len,
      out_words.data_ptr<uint32_t>(),
      cap_words_chunk,
      used_words.data_ptr<int32_t>());

  // chunk_sizes_bytes_flat: int64[total_chunks]
  auto chunk_sizes_bytes = torch::empty({total_chunks}, opts_i64);
  {
    int threads = 256;
    int blocks = (total_chunks + threads - 1) / threads;
    words_to_bytes_kernel_flat<<<blocks, threads>>>(
        used_words.data_ptr<int32_t>(),
        chunk_sizes_bytes.data_ptr<int64_t>(),
        total_chunks);
  }

  // stream_sizes_bytes: int64[B] = header_bytes + sum chunk sizes
  int64_t header_bytes = (int64_t)4 * (P + 2);
  auto stream_sizes_bytes = torch::empty({B}, opts_i64);
  {
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    sum_chunk_sizes_kernel<<<blocks, threads>>>(
        chunk_sizes_bytes.data_ptr<int64_t>(),
        stream_sizes_bytes.data_ptr<int64_t>(),
        B, P, header_bytes);
  }

  // stream_offsets_bytes: int64[B] = exclusive scan(stream_sizes_bytes)
  auto stream_offsets_bytes = torch::empty({B}, opts_i64);

  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr, temp_storage_bytes,
      stream_sizes_bytes.data_ptr<int64_t>(),
      stream_offsets_bytes.data_ptr<int64_t>(),
      B);

  auto temp = torch::empty({(long long)temp_storage_bytes},
                           torch::TensorOptions().dtype(torch::kUInt8).device(symbols_bxn.device()));
  cub::DeviceScan::ExclusiveSum(
      temp.data_ptr(),
      temp_storage_bytes,
      stream_sizes_bytes.data_ptr<int64_t>(),
      stream_offsets_bytes.data_ptr<int64_t>(),
      B);

  // total_bytes = offsets[B-1] + sizes[B-1] (sync via item() is acceptable for now)
  int64_t last_off = stream_offsets_bytes[B - 1].item<int64_t>();
  int64_t last_sz  = stream_sizes_bytes[B - 1].item<int64_t>();
  int64_t total_bytes = last_off + last_sz;

  auto out_bytes = torch::empty({total_bytes}, opts_u8);

  // Write per-stream headers
  write_headers_kernel<<<B, 1>>>(
      out_bytes.data_ptr<uint8_t>(),
      stream_offsets_bytes.data_ptr<int64_t>(),
      chunk_sizes_bytes.data_ptr<int64_t>(),
      B, P, chunk_len);

  // Compute chunk payload offsets inside out_bytes
  auto chunk_payload_offsets = torch::empty({total_chunks}, opts_i64);
  compute_chunk_payload_offsets_kernel<<<B, 1>>>(
      stream_offsets_bytes.data_ptr<int64_t>(),
      chunk_sizes_bytes.data_ptr<int64_t>(),
      chunk_payload_offsets.data_ptr<int64_t>(),
      B, P, header_bytes);

  // Pack each chunk payload
  pack_chunk_bytes_kernel<<<grid, 256>>>(
      out_words.data_ptr<uint32_t>(),
      cap_words_chunk,
      used_words.data_ptr<int32_t>(),
      chunk_payload_offsets.data_ptr<int64_t>(),
      out_bytes.data_ptr<uint8_t>(),
      B, P);

  // Return per-stream sizes as int32
  auto out_sizes_i32 = stream_sizes_bytes.to(torch::kInt32);
  return {out_bytes, out_sizes_i32};
}

__device__ __forceinline__ int32_t cdf_find_symbol_bs(
    const int32_t* __restrict__ cdf, int32_t cdf_size, uint32_t cum_freq
) {
  // search in [0, cdf_size-2]
  int32_t lo = 0;
  int32_t hi = cdf_size - 2;
  while (lo < hi) {
    int32_t mid = (lo + hi) >> 1;
    if ((uint32_t)cdf[mid + 1] > cum_freq) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

__global__ void decode_streams_kernel(
    const uint8_t* __restrict__ merged_bytes,   // [total_bytes]
    const int64_t* __restrict__ in_offsets,     // [B] byte offsets
    const int32_t* __restrict__ in_sizes,       // [B] byte sizes (total per stream, includes header)
    int B,
    const int32_t* __restrict__ indexes_bxn,    // [B,N]
    int N,
    const int32_t* __restrict__ cdfs_mxl,       // [M,Lmax]
    int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,    // [M]
    const int32_t* __restrict__ offsets_m,      // [M]
    int32_t* __restrict__ out_symbols_bxn       // [B,N]
) {
  int b = (int)blockIdx.x;
  if (b >= B) return;
  if (threadIdx.x != 0) return;

  const uint8_t* base = merged_bytes + in_offsets[b];
  int total_nbytes = in_sizes[b];
  (void)total_nbytes;

  // Parse header (uint32 aligned)
  const uint32_t* hdr = reinterpret_cast<const uint32_t*>(base);
  int P = (int)hdr[0];
  int chunk_len = (int)hdr[1];
  if (P < 1) return;

  int64_t header_bytes = (int64_t)4 * (P + 2);

  const int32_t* idx = indexes_bxn + (int64_t)b * N;
  int32_t* out = out_symbols_bxn + (int64_t)b * N;

  const uint8_t* payload = base + header_bytes;

  int64_t prefix = 0;
  for (int c = 0; c < P; ++c) {
    uint32_t chunk_bytes_u32 = hdr[2 + c];
    int chunk_bytes = (int)chunk_bytes_u32;

    int start = c * chunk_len;
    if (start >= N) {
      // still advance prefix but should be zero usually
      prefix += chunk_bytes;
      continue;
    }
    int end = start + chunk_len;
    if (end > N) end = N;

    if (chunk_bytes <= 0) {
      // Empty chunk should only happen if start>=N; but tolerate
      prefix += chunk_bytes;
      continue;
    }

    const uint8_t* chunk_bs = payload + prefix;
    prefix += chunk_bytes;

    // uint32 aligned
    const uint32_t* p = reinterpret_cast<const uint32_t*>(chunk_bs);

    Rans64State r;
    Rans64DecInit(&r, &p);

    for (int i = start; i < end; ++i) {
      int32_t cdf_idx = idx[i];
      const int32_t* cdf = cdfs_mxl + (int64_t)cdf_idx * Lmax;
      int32_t cdf_size = cdf_sizes_m[cdf_idx];
      int32_t max_value = cdf_size - 2;
      int32_t offset = offsets_m[cdf_idx];

      uint32_t cum_freq = Rans64DecGet(&r, precision);

      int32_t s = cdf_find_symbol_bs(cdf, cdf_size, cum_freq);

      uint32_t start_c = (uint32_t)cdf[s];
      uint32_t freq = (uint32_t)(cdf[s + 1] - cdf[s]);
      Rans64DecAdvance(&r, &p, start_c, freq, precision);

      int32_t value = s;

      if (value == max_value) {
        int32_t val = (int32_t)Rans64DecGetBits(&r, &p, bypass_precision);
        int32_t n_bypass = val;

        while (val == max_bypass_val) {
          val = (int32_t)Rans64DecGetBits(&r, &p, bypass_precision);
          n_bypass += val;
        }

        int32_t raw_val = 0;
        for (int j = 0; j < n_bypass; ++j) {
          val = (int32_t)Rans64DecGetBits(&r, &p, bypass_precision);
          raw_val |= (val << (j * bypass_precision));
        }

        value = raw_val >> 1;
        if (raw_val & 1) {
          value = -value - 1;
        } else {
          value += max_value;
        }
      }

      out[i] = value + offset;
    }
  }
}

torch::Tensor decode_with_indexes_cuda(
    torch::Tensor merged_bytes_u8,   // [total_bytes] uint8 CUDA
    torch::Tensor offsets_i64,        // [B] int64 CUDA
    torch::Tensor sizes_i32,          // [B] int32 CUDA
    torch::Tensor indexes_bxn,        // [B,N] int32 CUDA
    torch::Tensor cdfs_mxl,           // [M,Lmax] int32 CUDA
    torch::Tensor cdf_sizes_m,        // [M] int32 CUDA
    torch::Tensor offsets_m           // [M] int32 CUDA
) {
  int B = (int)offsets_i64.size(0);
  int N = (int)indexes_bxn.size(1);
  int Lmax = (int)cdfs_mxl.size(1);

  auto out = torch::empty({B, N}, torch::TensorOptions().dtype(torch::kInt32).device(indexes_bxn.device()));

  decode_streams_kernel<<<B, 32>>>(
      merged_bytes_u8.data_ptr<uint8_t>(),
      offsets_i64.data_ptr<int64_t>(),
      sizes_i32.data_ptr<int32_t>(),
      B,
      indexes_bxn.data_ptr<int32_t>(),
      N,
      cdfs_mxl.data_ptr<int32_t>(),
      Lmax,
      cdf_sizes_m.data_ptr<int32_t>(),
      offsets_m.data_ptr<int32_t>(),
      out.data_ptr<int32_t>());

  return out;
}

