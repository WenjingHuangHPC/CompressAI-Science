# compressai/runtime/__init__.py
from __future__ import annotations

import torch
from typing import Any

from .config import RuntimeConfig
from .adapters.cnn_simple import SimpleCnnAdapter
from .engines.cnn_engine import CnnEngine

def build_runtime(net, codec: Any, config: RuntimeConfig):
    adapter = SimpleCnnAdapter(net, codec)

    if config.mode == "torch":
        # Torch runners
        def make_torch_runner(module):
            module.eval().cuda()
            @torch.no_grad()
            def _run(x):
                return module(x)
            return _run

        ga_runner = make_torch_runner(adapter.subgraphs()["ga"])
        gs_runner = make_torch_runner(adapter.subgraphs()["gs"])

        # In torch mode, if you want fp32 gs, set gs_input_dtype accordingly outside or infer here.
        engine = CnnEngine(
            adapter,
            ga_runner,
            gs_runner,
            codec_input_dtype=torch.float32,
            gs_input_dtype=config.gs_input_dtype,     # gs engine boundary dtype
            ga_input_dtype=config.ga_input_dtype,
        )
        return engine

    if config.mode == "trt":
        if not config.trt_engines:
            raise ValueError("TRT mode requires config.trt_engines={'ga':..., 'gs':...}")

        from .backends.trt_loader import TRTRunnerFactory, TRTEngineSpec

        fac = TRTRunnerFactory()

        ga_spec = TRTEngineSpec(engine_path=config.trt_engines["ga"])
        gs_spec = TRTEngineSpec(engine_path=config.trt_engines["gs"])
        ga_mod = fac.load(ga_spec)
        gs_mod = fac.load(gs_spec)

        ga_runner = lambda x: ga_mod(x)
        gs_runner = lambda y: gs_mod(y)

        # TRT typically consumes/produces FP16 tensors at the boundary
        engine = CnnEngine(
            adapter,
            ga_runner,
            gs_runner,
            codec_input_dtype=torch.float32,  # per your rule: codec always FP32
            gs_input_dtype=config.gs_input_dtype,     # gs engine boundary dtype
            ga_input_dtype=config.ga_input_dtype,     # ga engine boundary dtype
        )
        return engine

    raise ValueError(f"Unknown mode: {config.mode}")
