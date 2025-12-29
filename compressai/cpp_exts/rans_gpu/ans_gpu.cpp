#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> encode_with_indexes_packed_cuda(
    torch::Tensor symbols_bxn,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int64_t P_in
);

torch::Tensor decode_with_indexes_packed_cuda(
    torch::Tensor arena_u8,
    torch::Tensor sizes_i32,
    torch::Tensor stride_cpu,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_with_indexes_packed", &encode_with_indexes_packed_cuda, "ANS encode packed (GPU-only)");
    m.def("decode_with_indexes_packed", &decode_with_indexes_packed_cuda, "ANS decode packed (GPU-only)");
}
