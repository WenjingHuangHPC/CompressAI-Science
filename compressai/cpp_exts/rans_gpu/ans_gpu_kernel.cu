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

struct Token {
  uint16_t start;
  uint16_t range;
  uint8_t bypass;
};

// One CUDA block handles one stream (one batch item), single thread for M1 correctness
__global__ void encode_streams_kernel(
    const int32_t* __restrict__ symbols_bxn,   // [B,N]
    const int32_t* __restrict__ indexes_bxn,   // [B,N]
    int B, int N,
    const int32_t* __restrict__ cdfs_mxl,      // [M,Lmax]
    int M, int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,   // [M]
    const int32_t* __restrict__ offsets_m,     // [M]
    uint32_t* __restrict__ out_words,          // [B, cap_words]
    int cap_words,
    int32_t* __restrict__ out_used_words       // [B]
) {
  int b = blockIdx.x;
  if (b >= B) return;
  if (threadIdx.x != 0) return;

  const int32_t* sym = symbols_bxn + (int64_t)b * N;
  const int32_t* idx = indexes_bxn + (int64_t)b * N;

  // local token buffer in global memory is avoided in M1:
  // we directly emulate CPU behavior: first build tokens logically, but encode requires reverse order.
  // CPU pushes tokens in forward order into _syms, then flush pops back (reverse).
  // Equivalent: we can iterate symbols in forward order, generate tokens, but we must encode in reverse token order.
  // M1 simplest: generate tokens into a temporary token array in out_words? Not possible.
  // So: generate tokens into a small local array per symbol then push to a token stack in a global Token buffer.
  // For M1, we encode in two passes within one thread:
  //   pass1: count tokens to know total
  //   pass2: regenerate tokens and write into a Token array, then encode from back to front.
  // Token array placed in global memory after out_words region using cap_words? We'll allocate a separate Token tensor later (M2).
  //
  // To keep M1 minimal, we instead store tokens in a fixed-size local array using new/delete is not allowed.
  // Therefore for M1 we take a pragmatic approach:
  //   We encode "on the fly" by traversing symbols in REVERSE order and generating tokens in the same order
  //   they would be popped in flush().
  //
  // Let's reason:
  // CPU: for i=0..N-1 push tokens for symbols[i] (main token then bypass tokens)
  // flush pops tokens from back => encoding order is: tokens of symbols[N-1] (including its bypass tokens), then symbols[N-2], ...
  //
  // So we can do: for i=N-1..0 generate tokens for symbols[i] and immediately encode them in the order they would be popped:
  // BUT within one symbol, CPU pushed main token first, then bypass tokens; pop reverses that, so encoding order for a sentinel symbol is:
  //   last raw chunk ... first raw chunk, then last n_bypass chunk ... first n_bypass chunk, then main token.
  //
  // We'll generate in that popped order directly.

  // Prepare output word ptr: write backwards in [b*cap_words, (b+1)*cap_words)
  uint32_t* base = out_words + (int64_t)b * cap_words;
  uint32_t* ptr = base + cap_words; // one past end

  Rans64State r;
  Rans64EncInit(&r);

  for (int i = N - 1; i >= 0; --i) {
    int32_t cdf_idx = idx[i];
    // assume valid
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

    // If sentinel, encode bypass tokens first (because flush pops them before main token)
    if (value == max_value) {
      // compute n_bypass
      int32_t n_bypass = 0;
      while ((raw_val >> (n_bypass * bypass_precision)) != 0) ++n_bypass;

      // raw chunks were pushed in increasing j, so popped in decreasing j => encode j=n_bypass-1..0
      for (int32_t j = n_bypass - 1; j >= 0; --j) {
        uint32_t v = (raw_val >> (j * bypass_precision)) & max_bypass_val;
        Rans64EncPutBits(&r, &ptr, v, bypass_precision);
      }

      // encode number of bypasses: CPU pushes repeated max_bypass_val then final val,
      // pop reverses: final val first, then max chunks. We need popped order.
      int32_t val = n_bypass;
      int32_t n_full = 0;
      while (val >= max_bypass_val) { val -= max_bypass_val; n_full++; }
      // popped order: final val token first
      Rans64EncPutBits(&r, &ptr, (uint32_t)val, bypass_precision);
      // then n_full times max_bypass_val
      for (int k = 0; k < n_full; ++k) {
        Rans64EncPutBits(&r, &ptr, (uint32_t)max_bypass_val, bypass_precision);
      }
    }

    // finally encode main token (popped last)
    uint32_t start = (uint32_t)cdf[value];
    uint32_t freq = (uint32_t)(cdf[value + 1] - cdf[value]);
    Rans64EncPut(&r, &ptr, start, freq, precision);
  }

  Rans64EncFlush(&r, &ptr);

  int used_words = (int)(base + cap_words - ptr);
  out_used_words[b] = used_words;

  // ptr points to the beginning of valid words. We keep words in-place;
  // later pack kernel will copy bytes from ptr region.
  // For convenience, store ptr offset (in words) into first word slot? We'll instead reconstruct as: start = cap_words - used_words.
}

__global__ void pack_bytes_kernel(
    const uint32_t* __restrict__ in_words,   // [B, cap_words]
    int B, int cap_words,
    const int32_t* __restrict__ used_words,  // [B]
    const int64_t* __restrict__ out_offsets, // [B] byte offsets into out_bytes
    uint8_t* __restrict__ out_bytes          // [total_bytes]
) {
  int b = blockIdx.x;
  if (b >= B) return;

  int uw = used_words[b];
  int start_word = cap_words - uw;

  const uint8_t* src = (const uint8_t*)(in_words + (int64_t)b * cap_words + start_word);
  uint8_t* dst = out_bytes + out_offsets[b];

  int nbytes = uw * 4;

  // copy in 4-byte chunks (single thread is fine for M1)
  if (threadIdx.x == 0) {
    for (int i = 0; i < nbytes; ++i) dst[i] = src[i];
  }
}

std::vector<torch::Tensor> encode_with_indexes_cuda(
    torch::Tensor symbols_bxn,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m
) {
  const auto B = (int)symbols_bxn.size(0);
  const auto N = (int)symbols_bxn.size(1);
  const auto M = (int)cdfs_mxl.size(0);
  const auto Lmax = (int)cdfs_mxl.size(1);

  // cap_words heuristic for M1
  int cap_words = N * 12 + 8;

  auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(symbols_bxn.device());
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(symbols_bxn.device());
  auto out_words = torch::empty({B, cap_words}, opts_u32);
  auto used_words = torch::zeros({B}, opts_i32);

  // encode each stream
  encode_streams_kernel<<<B, 32>>>(
      symbols_bxn.data_ptr<int32_t>(),
      indexes_bxn.data_ptr<int32_t>(),
      B, N,
      cdfs_mxl.data_ptr<int32_t>(),
      M, Lmax,
      cdf_sizes_m.data_ptr<int32_t>(),
      offsets_m.data_ptr<int32_t>(),
      out_words.data_ptr<uint32_t>(),
      cap_words,
      used_words.data_ptr<int32_t>());

  // compute byte sizes and offsets on CPU for M1 simplicity (later we move to GPU scan)
  auto used_words_cpu = used_words.to(torch::kCPU);
  auto used_acc = used_words_cpu.accessor<int32_t, 1>();

  std::vector<int64_t> offsets(B, 0);
  std::vector<int32_t> sizes(B, 0);
  int64_t total = 0;
  for (int b = 0; b < B; ++b) {
    int32_t nbytes = used_acc[b] * 4;
    sizes[b] = nbytes;
    offsets[b] = total;
    total += nbytes;
  }

  auto out_sizes = torch::from_blob(sizes.data(), {B}, torch::TensorOptions().dtype(torch::kInt32)).clone().to(symbols_bxn.device());
  auto out_offsets = torch::from_blob(offsets.data(), {B}, torch::TensorOptions().dtype(torch::kInt64)).clone().to(symbols_bxn.device());

  auto out_bytes = torch::empty({total}, torch::TensorOptions().dtype(torch::kUInt8).device(symbols_bxn.device()));

  pack_bytes_kernel<<<B, 32>>>(
      out_words.data_ptr<uint32_t>(),
      B, cap_words,
      used_words.data_ptr<int32_t>(),
      out_offsets.data_ptr<int64_t>(),
      out_bytes.data_ptr<uint8_t>());

  return {out_bytes, out_sizes};
}


__global__ void decode_streams_kernel(
    const uint8_t* __restrict__ merged_bytes,   // [total_bytes]
    const int64_t* __restrict__ in_offsets,     // [B] byte offsets
    const int32_t* __restrict__ in_sizes,       // [B] byte sizes
    int B,
    const int32_t* __restrict__ indexes_bxn,    // [B,N]
    int N,
    const int32_t* __restrict__ cdfs_mxl,       // [M,Lmax]
    int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,    // [M]
    const int32_t* __restrict__ offsets_m,      // [M]
    int32_t* __restrict__ out_symbols_bxn       // [B,N]
) {
  int b = blockIdx.x;
  if (b >= B) return;
  if (threadIdx.x != 0) return;

  const uint8_t* bs = merged_bytes + in_offsets[b];
  int nbytes = in_sizes[b];

  // stream is uint32-aligned in your encoder output (multiple of 4 bytes)
  const uint32_t* ptr32 = reinterpret_cast<const uint32_t*>(bs);
  const uint32_t* p = ptr32;

  Rans64State r;
  Rans64DecInit(&r, &p);

  const int32_t* idx = indexes_bxn + (int64_t)b * N;
  int32_t* out = out_symbols_bxn + (int64_t)b * N;

  for (int i = 0; i < N; ++i) {
    int32_t cdf_idx = idx[i];
    const int32_t* cdf = cdfs_mxl + (int64_t)cdf_idx * Lmax;
    int32_t cdf_size = cdf_sizes_m[cdf_idx];
    int32_t max_value = cdf_size - 2;
    int32_t offset = offsets_m[cdf_idx];

    uint32_t cum_freq = Rans64DecGet(&r, precision);

    // find s such that cdf[s] <= cum_freq < cdf[s+1]
    // CPU uses std::find_if; here linear search for M1 correctness
    int32_t s = 0;
    // valid range: [0, cdf_size-2] (since last is sentinel end)
    for (int32_t j = 0; j < cdf_size - 1; ++j) {
      if ((uint32_t)cdf[j + 1] > cum_freq) { s = j; break; }
    }

    uint32_t start = (uint32_t)cdf[s];
    uint32_t freq = (uint32_t)(cdf[s + 1] - cdf[s]);
    Rans64DecAdvance(&r, &p, start, freq, precision);

    int32_t value = s;

    if (value == max_value) {
      // bypass decoding mode (match CPU)
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

