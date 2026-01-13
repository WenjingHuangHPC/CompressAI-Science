# compressai/runtime/codecs/eb_packed_gpu.py
from __future__ import annotations

import pickle
from typing import Dict
import torch

from .base import Codec, Pack

def _packed_payload_bytes(packed) -> int:
    return int(packed.packed.numel())

def _bytes_of_state(obj) -> int:
    if obj is None:
        return 0
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return 0

class GpuPackedEntropyCodec(Codec):
    """
    GPU PackedANS codec for CompressAI EntropyModel.
    - Enforces FP32 input latent for lossless coding.
    - Handles stream synchronization internally (TRT stream <-> default stream),
      since EB implementations often assume default stream.
    """

    def __init__(self, entropy_bottleneck, P: int = 64):
        self.eb = entropy_bottleneck
        self.P = int(P)
        self.eb.use_gpu_ans = True
        self.eb.gpu_ans_parallelism = int(P)

    @torch.no_grad()
    def compress(self, y_fp32: torch.Tensor) -> Pack:
        if y_fp32.dtype != torch.float32:
            raise ValueError("GpuPackedEntropyBottleneckCodec expects FP32 latent for lossless coding")
        if not y_fp32.is_cuda:
            raise ValueError("GpuPackedEntropyBottleneckCodec expects CUDA tensor")

        trt_stream = torch.cuda.current_stream(device=y_fp32.device)
        default_stream = torch.cuda.default_stream(device=y_fp32.device)

        # Ensure upstream (possibly TRT) writes to y are complete
        default_stream.wait_stream(trt_stream)

        with torch.cuda.stream(default_stream):
            y_fp32 = y_fp32.contiguous()
            packed = self.eb.compress(y_fp32)
            state = {"size_hw": tuple(y_fp32.shape[-2:])}

        # Ensure packed/state visible to downstream on current stream
        trt_stream.wait_stream(default_stream)

        return {"strings": packed, "state": state}

    @torch.no_grad()
    def decompress(self, pack: Pack) -> torch.Tensor:
        packed = pack["strings"]
        state = pack["state"]

        # Best-effort device inference
        dev = getattr(getattr(packed, "packed", None), "device", None)
        if dev is None:
            dev = torch.device("cuda")

        trt_stream = torch.cuda.current_stream(device=dev)
        default_stream = torch.cuda.default_stream(device=dev)

        default_stream.wait_stream(trt_stream)

        with torch.cuda.stream(default_stream):
            size_hw = state["size_hw"]
            y_hat = self.eb.decompress(packed, size_hw)
            if not y_hat.is_contiguous():
                y_hat = y_hat.contiguous()

        trt_stream.wait_stream(default_stream)
        return y_hat

    def pack_bytes(self, pack: Pack) -> Dict[str, float]:
        payload = float(_packed_payload_bytes(pack.get("strings", None)))
        state_bytes = float(_bytes_of_state(pack.get("state", None)))
        return {"strings_bytes": payload, "state_bytes": state_bytes}
