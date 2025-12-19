# compressai/runtime/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional
import torch

Precision = Literal["fp32", "fp16", "fp8"]

@dataclass(frozen=True)
class RuntimeConfig:
    mode: Literal["torch", "trt"] = "trt"
    precision: Precision = "fp16"

    # Engine paths for TRT mode: per-subgraph
    # e.g. {"ga": "/path/to/ga.plan", "gs": "/path/to/gs.plan"}
    trt_engines: Optional[Dict[str, str]] = None

    # Optional: whether to require engines exist (recommended True)
    require_engine: bool = True
    ga_input_dtype: torch.dtype = torch.float32
    gs_input_dtype: torch.dtype = torch.float32
