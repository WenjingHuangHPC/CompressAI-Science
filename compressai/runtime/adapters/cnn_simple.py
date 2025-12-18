# compressai/runtime/adapters/cnn_simple.py
from __future__ import annotations

from typing import Dict, Any
import torch
import torch.nn as nn

from .base import ModelAdapter, Pack


class SimpleCnnAdapter(ModelAdapter):
    """
    Pure CNN adapter:
      y = net.g_a(x)
      pack = codec.compress(y)  # must contain "strings" and "state"
      y_hat = codec.decompress(pack["strings"], pack["state"])
      x_hat = net.g_s(y_hat)
    """

    def __init__(self, net: nn.Module, codec: Any):
        self.net = net
        self.codec = codec

        if not hasattr(net, "g_a") or not hasattr(net, "g_s"):
            raise AttributeError("Expected net to have g_a and g_s submodules.")

        self.ga = net.g_a
        self.gs = net.g_s

    def subgraphs(self) -> Dict[str, nn.Module]:
        return {"ga": self.ga, "gs": self.gs}

    def compress_glue(self, y: torch.Tensor) -> Pack:
        pack = self.codec.compress(y)
        if not isinstance(pack, dict):
            raise TypeError("codec.compress(y) must return a dict-like pack.")
        if "strings" not in pack or "state" not in pack:
            raise KeyError('pack must contain keys: "strings" and "state".')
        return pack

    def decompress_glue(self, pack: Pack) -> torch.Tensor:
        strings = pack["strings"]
        state = pack["state"]
        y_hat = self.codec.decompress(strings, state)
        if not isinstance(y_hat, torch.Tensor):
            raise TypeError("codec.decompress(strings, state) must return a torch.Tensor.")
        return y_hat
