# compressai/runtime/engines/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict
import torch


class Engine(ABC):
    @abstractmethod
    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def decompress(self, pack: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError
