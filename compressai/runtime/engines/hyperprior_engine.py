# compressai/runtime/engines/hyperprior_engine.py
from __future__ import annotations

from typing import Dict, Any, Optional
import torch

from .base import CnnEngine
from ..codecs.compress_packed_gpu import GpuPackedEntropyCodec
from compressai.zoo import bmshj2018_hyperprior


class HyperpriorEngine(CnnEngine):
    """
    Hyperprior pipeline:
      y = ga(x)
      z = ha(y)
      z_pack = codec.compress(z_fp32)
      z_hat = codec.decompress(z_pack)
      params = hs(z_hat)
      y_pack = codec.compress(y_fp32, params=params)
      (return {"y": y_pack, "z": z_pack})

    Decompress:
      z_hat = codec.decompress(z_pack)
      params = hs(z_hat)
      y_hat = codec.decompress(y_pack, params=params)
      x_hat = gs(y_hat)
    """

    def __init__(
        self,
        net: bmshj2018_hyperprior,
        codec: GpuPackedEntropyCodec,
        runners: Dict[str, Any],
        *,
        codec_input_dtype: torch.dtype = torch.float32,
        gs_input_dtype: torch.dtype = torch.float16,
        ga_input_dtype: Optional[torch.dtype] = None,
        ha_input_dtype: Optional[torch.dtype] = None,
        hs_input_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        self.codec = codec
        self.runners = runners  # expects keys: ga, gs, ha, hs

        self.codec_input_dtype = codec_input_dtype
        self.gs_input_dtype = gs_input_dtype
        self.ga_input_dtype = ga_input_dtype
        self.ha_input_dtype = ha_input_dtype
        self.hs_input_dtype = hs_input_dtype

        for k in ("ga", "gs", "ha", "hs"):
            if k not in self.runners:
                raise ValueError(f"HyperpriorEngine requires runner '{k}'")

    def _cast(self, x: torch.Tensor, dt: Optional[torch.dtype]) -> torch.Tensor:
        if dt is not None and x.dtype != dt:
            return x.to(dt)
        return x

    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        x = self._ensure_cuda_contiguous(x)

        # ga
        x = self._cast(x, self.ga_input_dtype)
        y = self.runners["ga"](x)
        if not isinstance(y, torch.Tensor):
            raise TypeError("ga runner must return torch.Tensor")
        y = self._ensure_cuda_contiguous(y)

        # ha
        y_ha = self._cast(y, self.ha_input_dtype)
        z = self.runners["ha"](torch.abs(y_ha))
        if not isinstance(z, torch.Tensor):
            raise TypeError("ha runner must return torch.Tensor")
        z = self._ensure_cuda_contiguous(z)

        # codec(z) requires FP32
        if z.dtype != self.codec_input_dtype:
            z_fp32 = z.to(self.codec_input_dtype)
        else:
            z_fp32 = z
        z_pack = self.codec.compress(z_fp32)

        # z_hat
        z_hat = self.codec.decompress(z_pack)
        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # hs -> params
        z_hat_hs = self._cast(z_hat, self.hs_input_dtype)
        scales_hat = self.runners["hs"](z_hat_hs)

        # codec(y | params) requires FP32 y
        if y.dtype != self.codec_input_dtype:
            y_fp32 = y.to(self.codec_input_dtype)
        else:
            y_fp32 = y
        params = self.codec.gaussian_conditional.build_indexes(scales_hat)
        y_pack = self.codec.compress(y_fp32, params=params)

        return {"y": y_pack, "z": z_pack}

    def decompress(self, pack: Dict[str, Any]) -> torch.Tensor:
        if "y" not in pack or "z" not in pack:
            raise KeyError("Hyperprior pack must contain keys: 'y' and 'z'")

        # z_hat
        z_hat = self.codec.decompress(pack["z"])
        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # hs -> params
        z_hat_hs = self._cast(z_hat, self.hs_input_dtype)
        scales_hat = self.runners["hs"](z_hat_hs)
        indexes = self.codec.gaussian_conditional.build_indexes(scales_hat)

        # y_hat
        y_hat = self.codec.decompress(pack["y"], params=indexes, dtype=z_hat.dtype)
        if not isinstance(y_hat, torch.Tensor):
            raise TypeError("codec.decompress(y_pack) must return torch.Tensor")
        y_hat = self._ensure_cuda_contiguous(y_hat)

        # gs
        if y_hat.dtype != self.gs_input_dtype:
            y_hat = y_hat.to(self.gs_input_dtype)
        x_hat = self.runners["gs"](y_hat).clamp_(0, 1)
        if not isinstance(x_hat, torch.Tensor):
            raise TypeError("gs runner must return torch.Tensor")
        return self._ensure_cuda_contiguous(x_hat)
