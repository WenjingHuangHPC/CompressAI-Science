# compressai/runtime/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional
import torch

Precision = Literal["fp32", "fp16", "fp8"]

@dataclass(frozen=True)
class RuntimeConfig:
    model_name: str = "bmshj2018_factorized"
    trt_engines: Optional[Dict[str, str]] = None

    # Optional: whether to require engines exist (recommended True)
    require_engine: bool = True
    ga_input_dtype: torch.dtype = torch.float32
    gs_input_dtype: torch.dtype = torch.float32
    ha_input_dtype: torch.dtype = torch.float32
    hs_input_dtype: torch.dtype = torch.float32
    codec_input_dtype: torch.dtype = torch.float32
