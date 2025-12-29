// compressai/cpp_exts/rans_gpu/ans_gpu_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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

// --------------------------
// Packed GPU-only layout (fixed-stride arena, no prefix-sum, no packing pass)
// For each stream b in [0,B):
//   stream_base = arena + b*stride
//   header_raw = 4*(P+2) bytes
//   header_padded = align16(header_raw)
//   payload_base = stream_base + header_padded
//
// Header (uint32) stored at stream_base:
//   hdr[0]=P, hdr[1]=chunk_len, hdr[2+c]=used_bytes_of_chunk_c
//
// Payload is P fixed slots, slot c has size cap_bytes_chunk
//   slot_c = payload_base + c*cap_bytes_chunk
// rANS writes backward into slot, so bitstream starts at slot_end - used_bytes
//
// Encoder returns (arena_u8 CUDA, sizes_i32 CUDA, stride_i64 CPU tensor(1))
// Decoder consumes the same without any host repack / List[bytes].
// --------------------------

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

  int used_words = (int)(base_u32 + cap_words_chunk - ptr);
  used_words_flat[b * P + c] = used_words;
}

__global__ void write_headers_from_used_words_kernel(
    uint8_t* __restrict__ arena_u8,
    int64_t stride,
    int B, int P, int chunk_len,
    const int32_t* __restrict__ used_words_flat  // [B*P]
) {
  int b = (int)blockIdx.x;
  if (b >= B) return;

  uint8_t* stream_base = arena_u8 + (int64_t)b * stride;
  uint32_t* hdr = reinterpret_cast<uint32_t*>(stream_base);

  hdr[0] = (uint32_t)P;
  hdr[1] = (uint32_t)chunk_len;

  const int32_t* row = used_words_flat + (int64_t)b * P;
  for (int c = threadIdx.x; c < P; c += blockDim.x) {
    hdr[2 + c] = (uint32_t)(row[c] * 4);
  }
}

__global__ void compute_stream_sizes_from_used_words_kernel(
    const int32_t* __restrict__ used_words_flat, // [B*P]
    int32_t* __restrict__ sizes_i32,             // [B]
    int B, int P,
    int64_t header_bytes_padded
) {
  int b = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;

  const int32_t* row = used_words_flat + (int64_t)b * P;
  int64_t sum = header_bytes_padded;
  for (int c = 0; c < P; ++c) sum += (int64_t)row[c] * 4;
  sizes_i32[b] = (int32_t)sum;
}

std::vector<torch::Tensor> encode_with_indexes_packed_cuda(
    torch::Tensor symbols_bxn,   // CUDA int32 [B,N]
    torch::Tensor indexes_bxn,   // CUDA int32 [B,N]
    torch::Tensor cdfs_mxl,      // CUDA int32 [M,Lmax]
    torch::Tensor cdf_sizes_m,   // CUDA int32 [M]
    torch::Tensor offsets_m,     // CUDA int32 [M]
    int64_t P_in
) {
  const int B = (int)symbols_bxn.size(0);
  const int N = (int)symbols_bxn.size(1);
  const int Lmax = (int)cdfs_mxl.size(1);

  int P = (int)P_in;
  if (P < 1) P = 1;
  if (P > N) P = N;
  if (P > P_MAX) P = P_MAX;

  int chunk_len = (N + P - 1) / P;

  int cap_words_chunk = chunk_len * 12 + 8;
  int64_t cap_bytes_chunk = (int64_t)cap_words_chunk * 4;

  int64_t header_raw = (int64_t)4 * (P + 2);
  int64_t header_padded = align16_i64(header_raw);

  int64_t stride = header_padded + (int64_t)P * cap_bytes_chunk;

  auto dev = symbols_bxn.device();
  auto opts_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);

  auto arena_u8 = torch::empty({(long long)(stride * (int64_t)B)}, opts_u8);
  auto used_words = torch::zeros({B * P}, opts_i32);
  auto sizes_i32 = torch::empty({B}, opts_i32);

  dim3 grid(B, P, 1);
  encode_chunks_into_arena_kernel<<<grid, 1>>>(
      symbols_bxn.data_ptr<int32_t>(),
      indexes_bxn.data_ptr<int32_t>(),
      B, N,
      cdfs_mxl.data_ptr<int32_t>(),
      Lmax,
      cdf_sizes_m.data_ptr<int32_t>(),
      offsets_m.data_ptr<int32_t>(),
      P,
      chunk_len,
      arena_u8.data_ptr<uint8_t>(),
      stride,
      header_padded,
      cap_words_chunk,
      used_words.data_ptr<int32_t>());

  write_headers_from_used_words_kernel<<<B, 256>>>(
      arena_u8.data_ptr<uint8_t>(),
      stride,
      B, P, chunk_len,
      used_words.data_ptr<int32_t>());

  {
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    compute_stream_sizes_from_used_words_kernel<<<blocks, threads>>>(
        used_words.data_ptr<int32_t>(),
        sizes_i32.data_ptr<int32_t>(),
        B, P,
        header_padded);
  }

  auto stride_cpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  stride_cpu[0] = stride;

  return {arena_u8, sizes_i32, stride_cpu};
}

// --------------------------
// Packed decode: no offsets tensor, just stride
// One block per stream, threads=512 (one thread per chunk).
// --------------------------

__global__ void decode_streams_packed_kernel(
    const uint8_t* __restrict__ arena_u8,      // [B*stride]
    int64_t stride,
    const int32_t* __restrict__ sizes_i32,     // [B] (kept for safety; not required for layout)
    int B,
    const int32_t* __restrict__ indexes_bxn,   // [B,N]
    int N,
    const int32_t* __restrict__ cdfs_mxl,      // [M,Lmax]
    int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,   // [M]
    const int32_t* __restrict__ offsets_m,     // [M]
    int32_t* __restrict__ out_symbols_bxn      // [B,N]
) {
  int b = (int)blockIdx.x;
  if (b >= B) return;

  const uint8_t* base = arena_u8 + (int64_t)b * stride;
  const uint32_t* hdr = reinterpret_cast<const uint32_t*>(base);

  int P = (int)hdr[0];
  int chunk_len = (int)hdr[1];
  if (P < 1) return;
  if (P > P_MAX) return;

  int64_t header_raw = (int64_t)4 * (P + 2);
  int64_t header_padded = align16_i64(header_raw);

  int cap_words_chunk = chunk_len * 12 + 8;
  int cap_bytes_chunk = cap_words_chunk * 4;

  const uint8_t* payload = base + header_padded;

  int c = (int)threadIdx.x;
  if (c >= P) return;

  int start = c * chunk_len;
  if (start >= N) return;
  int end = start + chunk_len;
  if (end > N) end = N;

  int used_bytes = (int)hdr[2 + c];
  if (used_bytes <= 0) return;

  const uint8_t* slot = payload + (int64_t)c * cap_bytes_chunk;
  const uint8_t* chunk_bs = slot + (cap_bytes_chunk - used_bytes);
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
      if (raw_val & 1) value = -value - 1;
      else value += max_value;
    }

    out[i] = value + offsetv;
  }
}

torch::Tensor decode_with_indexes_packed_cuda(
    torch::Tensor arena_u8,      // CUDA uint8 [B*stride]
    torch::Tensor sizes_i32,     // CUDA int32 [B]
    torch::Tensor stride_cpu,    // CPU int64 [1]
    torch::Tensor indexes_bxn,   // CUDA int32 [B,N]
    torch::Tensor cdfs_mxl,      // CUDA int32 [M,Lmax]
    torch::Tensor cdf_sizes_m,   // CUDA int32 [M]
    torch::Tensor offsets_m      // CUDA int32 [M]
) {
  int B = (int)sizes_i32.size(0);
  int N = (int)indexes_bxn.size(1);
  int Lmax = (int)cdfs_mxl.size(1);

  int64_t stride = stride_cpu[0].item<int64_t>();

  auto out = torch::empty({B, N}, torch::TensorOptions().dtype(torch::kInt32).device(indexes_bxn.device()));

  int threads = 512;
  decode_streams_packed_kernel<<<B, threads>>>(
      arena_u8.data_ptr<uint8_t>(),
      stride,
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
