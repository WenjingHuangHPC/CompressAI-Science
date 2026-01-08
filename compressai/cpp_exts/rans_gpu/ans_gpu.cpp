#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> encode_with_indexes_tight_cuda(
    torch::Tensor symbols_bxn,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int64_t P_in
);

torch::Tensor decode_with_indexes_tight_cuda(
    torch::Tensor packed_u8,
    torch::Tensor sizes_u16,
    torch::Tensor header_bytes_cpu,
    torch::Tensor chunk_len_cpu,
    torch::Tensor P_cpu,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_with_indexes_tight", &encode_with_indexes_tight_cuda, "ANS encode tight (GPU packed)");
    m.def("decode_with_indexes_tight", &decode_with_indexes_tight_cuda, "ANS decode tight (GPU packed)");
}
