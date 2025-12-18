# compressai/runtime/engines/cnn_engine.py
from __future__ import annotations

from typing import Dict, Any, Optional
import torch

from .base import Engine
from ..adapters.base import ModelAdapter


class CnnEngine(Engine):
    """
    Orchestrates:
      y = ga_runner(x)
      pack = codec.compress(y_cast_fp32)            # via adapter.compress_glue
      y_hat = codec.decompress(strings, state)      # via adapter.decompress_glue
      x_hat = gs_runner(y_hat_cast_to_gs_dtype)

    Dtype rules (per your corrected requirements):
      - compress: ALWAYS cast y -> FP32 before codec.compress (avoid lossless errors)
      - decompress: cast y_hat -> gs_input_dtype before gs_runner (match gs precision)
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        ga_runner,
        gs_runner,
        *,
        codec_input_dtype: torch.dtype = torch.float32,
        gs_input_dtype: torch.dtype = torch.float16,
        # Optional: cast x into what ga_runner expects (e.g. TRT prefers FP16 input)
        ga_input_dtype: Optional[torch.dtype] = None,
    ):
        self.adapter = adapter
        self.ga_runner = ga_runner
        self.gs_runner = gs_runner

        self.codec_input_dtype = codec_input_dtype
        self.gs_input_dtype = gs_input_dtype
        self.ga_input_dtype = ga_input_dtype

    @staticmethod
    def _ensure_cuda_contiguous(x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Runtime expects CUDA tensors for accelerated execution.")
        return x.contiguous() if not x.is_contiguous() else x

    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        x = self._ensure_cuda_contiguous(x)

        # Optional: adapt x dtype for ga_runner backend
        if self.ga_input_dtype is not None and x.dtype != self.ga_input_dtype:
            x = x.to(self.ga_input_dtype)

        y = self.ga_runner(x)
        if not isinstance(y, torch.Tensor):
            raise TypeError("ga_runner must return a torch.Tensor.")
        y = self._ensure_cuda_contiguous(y)

        # REQUIRED: codec.compress must see FP32 latent
        if y.dtype != self.codec_input_dtype:
            y = y.to(self.codec_input_dtype)

        pack = self.adapter.compress_glue(y)
        return pack

    def decompress(self, pack: Dict[str, Any]) -> torch.Tensor:
        y_hat = self.adapter.decompress_glue(pack)
        if not isinstance(y_hat, torch.Tensor):
            raise TypeError("adapter.decompress_glue(pack) must return a torch.Tensor.")
        y_hat = self._ensure_cuda_contiguous(y_hat)

        # REQUIRED: match gs precision expectation
        if y_hat.dtype != self.gs_input_dtype:
            y_hat = y_hat.to(self.gs_input_dtype)

        x_hat = self.gs_runner(y_hat)
        if not isinstance(x_hat, torch.Tensor):
            raise TypeError("gs_runner must return a torch.Tensor.")
        x_hat = self._ensure_cuda_contiguous(x_hat)
        return x_hat
