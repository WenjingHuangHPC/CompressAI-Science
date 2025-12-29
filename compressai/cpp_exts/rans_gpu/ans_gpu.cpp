// compressai/cpp_exts/rans_gpu/ans_gpu.cpp
#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

std::vector<torch::Tensor> encode_with_indexes_cuda(
    torch::Tensor symbols_bxn,   // int32 CUDA [B,N]
    torch::Tensor indexes_bxn,   // int32 CUDA [B,N]
    torch::Tensor cdfs_mxl,      // int32 CUDA [M,Lmax]
    torch::Tensor cdf_sizes_m,   // int32 CUDA [M]
    torch::Tensor offsets_m,     // int32 CUDA [M]
    int64_t P                    // NEW: chunk parallelism
);

torch::Tensor decode_with_indexes_cuda(
    torch::Tensor merged_bytes_u8,
    torch::Tensor offsets_i64,
    torch::Tensor sizes_i32,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m);

// Python interface (with checks + default P=1)
std::vector<torch::Tensor> encode_with_indexes(
    torch::Tensor symbols_bxn,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int64_t P = 1
) {
    TORCH_CHECK(P >= 1, "P must be >= 1");

    TORCH_CHECK(symbols_bxn.is_cuda(), "symbols must be CUDA tensor");
    TORCH_CHECK(indexes_bxn.is_cuda(), "indexes must be CUDA tensor");
    TORCH_CHECK(cdfs_mxl.is_cuda(), "cdfs must be CUDA tensor");
    TORCH_CHECK(cdf_sizes_m.is_cuda(), "cdf_sizes must be CUDA tensor");
    TORCH_CHECK(offsets_m.is_cuda(), "offsets must be CUDA tensor");

    TORCH_CHECK(symbols_bxn.dtype() == torch::kInt32, "symbols must be int32");
    TORCH_CHECK(indexes_bxn.dtype() == torch::kInt32, "indexes must be int32");
    TORCH_CHECK(cdfs_mxl.dtype() == torch::kInt32, "cdfs must be int32");
    TORCH_CHECK(cdf_sizes_m.dtype() == torch::kInt32, "cdf_sizes must be int32");
    TORCH_CHECK(offsets_m.dtype() == torch::kInt32, "offsets must be int32");

    TORCH_CHECK(symbols_bxn.is_contiguous(), "symbols must be contiguous");
    TORCH_CHECK(indexes_bxn.is_contiguous(), "indexes must be contiguous");
    TORCH_CHECK(cdfs_mxl.is_contiguous(), "cdfs must be contiguous");
    TORCH_CHECK(cdf_sizes_m.is_contiguous(), "cdf_sizes must be contiguous");
    TORCH_CHECK(offsets_m.is_contiguous(), "offsets must be contiguous");

    TORCH_CHECK(symbols_bxn.dim() == 2, "symbols must be [B,N]");
    TORCH_CHECK(indexes_bxn.sizes() == symbols_bxn.sizes(), "indexes must match symbols shape");
    TORCH_CHECK(cdfs_mxl.dim() == 2, "cdfs must be [M,Lmax]");
    TORCH_CHECK(cdf_sizes_m.dim() == 1, "cdf_sizes must be [M]");
    TORCH_CHECK(offsets_m.dim() == 1, "offsets must be [M]");
    TORCH_CHECK(cdf_sizes_m.size(0) == offsets_m.size(0), "cdf_sizes and offsets must match");
    TORCH_CHECK(cdfs_mxl.size(0) == cdf_sizes_m.size(0), "cdfs first dim must match cdf_sizes");

    return encode_with_indexes_cuda(symbols_bxn, indexes_bxn, cdfs_mxl, cdf_sizes_m, offsets_m, P);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_with_indexes", &encode_with_indexes, "GPU rANS encode_with_indexes (chunked, P adjustable)",
            py::arg("symbols_bxn"),
            py::arg("indexes_bxn"),
            py::arg("cdfs_mxl"),
            py::arg("cdf_sizes_m"),
            py::arg("offsets_m"),
            py::arg("P") = 1);
    m.def("decode_with_indexes", &decode_with_indexes_cuda, "GPU rANS decode_with_indexes (auto-parse chunk header)");
}
