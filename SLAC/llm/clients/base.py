from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
