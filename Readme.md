
---

## Legend Extensions (Added Features)

This project extends **CompressAI** with GPU-oriented acceleration and lossless compression capabilities, enabling efficient **end-to-end compression and decompression** with comprehensive benchmarking.

### 1) Low-Precision Acceleration for CNN-based Compression Models

We introduce low-precision inference acceleration based on **TensorRT (TRT)** and a **model optimization / quantization framework**, supporting multiple numeric formats:

* **FP32 / FP16 / FP8**
* End-to-end compression and decompression acceleration
* Unified benchmarking for throughput and reconstruction quality

These optimizations significantly improve inference throughput while maintaining acceptable reconstruction quality (see Benchmarks below).

### 2) Integrated GPU Lossless Compression Pipeline

A GPU-based lossless compression algorithm is integrated and organized into three stages:

1. **Normalizer**
   Normalizes or transforms input data to facilitate quantization and entropy coding.

2. **Quantizer**
   Performs discretization / quantization to generate symbols suitable for lossless encoding.

3. **Lossless Encoding**
   Applies entropy coding on GPU to produce a compact bitstream.
   The decoding process is fully symmetric and lossless.

This pipeline can be used independently or combined with learned compression models as a pre-processing or post-processing stage.

---

## Benchmarks

### Metrics

The following metrics are reported for all experiments:

* **Throughput (Enc / Dec)**
  Compression and decompression throughput (e.g., images/s or GB/s).
* **RMSE / NRMSE**
  Reconstruction error metrics.
* **PSNR (dB)**
  Peak Signal-to-Noise Ratio.
* **CR (Compression Ratio)**
  Defined as `original size / compressed size`.

> **Note:** For fair comparison, the dataset, input resolution, batch size, GPU model, and software stack should remain consistent across all precision modes.

---

## Experimental Results 

### A) End-to-End Performance and Quality

| Precision | Encoding Throughput(GB/s) | Decoding Throughput(GB/s) |  RMSE  |  NRMSE  |  PSNR (dB)  |   maxe  |    CR    |
| --------- | ------------------------: | ------------------------: | -----: | ------: | ----------: | ------: | -------: |
| FP32      |                      5.36 |                      4.91 | 0.1017 | 0.10205 |        13.8 |  0.8845 |  79.217  |
| FP16      |                     15.97 |                     19.43 | 0.1017 | 0.10206 |        13.8 |  0.8871 |  79.266  |
| FP8       |                     20.12 |                      9.12 | 0.1018 | 0.10219 |       13.79 |  0.8900 |  77.393  |

### B) Experimental Setup

| Item                     | Value                |
| ------------------------ | -------------------- |
| Dataset / Input          | NYX                  |
| Input Resolution / Shape | 512X3X128X128        |
| GPU                      | H100                 |
| CUDA / Driver Version    | 12.6                 |
| TensorRT Version         | 10.13.2.6            |
| Quantization Framework   | Model Optimization   |

---

## Reproducibility

Example commands for building TensorRT engines and running end-to-end benchmarks:

```bash

# Run end-to-end benchmark
python CompressAI-Science/compressai/runtime/examples/runtime_cnn_trt_fp8.py
```

---
