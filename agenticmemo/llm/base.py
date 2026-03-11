"""Abstract LLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import LLMResponse, Message


class LLMBackend(ABC):
    """Provider-agnostic LLM interface.

    All backends must implement `complete`. The agent only talks to this
    interface, so swapping providers requires zero changes upstream.
    """

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 4096) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Send messages and return a normalised LLMResponse."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of texts (optional; may raise NotImplementedError)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
