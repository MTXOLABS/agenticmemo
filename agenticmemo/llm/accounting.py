"""Per-subsystem token accounting.

Optimization needs visibility: a single "total_tokens" can't tell you whether
the planner, the verifier, or memory injection is eating the budget. Every
Agent subsystem gets its LLM handle wrapped in a TaggedLLM that reports usage
into a shared TokenLedger under a purpose tag ("planner", "executor",
"verifier", ...). The Agent snapshots the ledger around each run and stores
the delta in ``Trajectory.metadata["token_breakdown"]``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..types import LLMResponse
from .base import LLMBackend


class TokenLedger:
    """Cumulative token/call counts keyed by purpose tag."""

    def __init__(self) -> None:
        self._tokens: dict[str, int] = defaultdict(int)
        self._calls: dict[str, int] = defaultdict(int)

    def record(self, purpose: str, tokens: int) -> None:
        self._tokens[purpose] += tokens
        self._calls[purpose] += 1

    def snapshot(self) -> dict[str, int]:
        return dict(self._tokens)

    def delta_since(self, snapshot: dict[str, int]) -> dict[str, int]:
        return {
            k: v - snapshot.get(k, 0)
            for k, v in self._tokens.items()
            if v - snapshot.get(k, 0) > 0
        }

    def breakdown(self) -> dict[str, dict[str, int]]:
        return {
            k: {"tokens": self._tokens[k], "calls": self._calls[k]}
            for k in sorted(self._tokens)
        }

    @property
    def total(self) -> int:
        return sum(self._tokens.values())


class TaggedLLM(LLMBackend):
    """Delegating wrapper that attributes usage to a purpose tag.

    Transparent to callers: ``complete``/``embed``/``format_tools`` all
    delegate to the wrapped backend, so it can wrap any LLMBackend —
    including other wrappers (e.g. a benchmark token meter).
    """

    def __init__(self, inner: LLMBackend, purpose: str, ledger: TokenLedger) -> None:
        super().__init__(model=inner.model)
        self._inner = inner
        self._purpose = purpose
        self._ledger = ledger

    async def complete(self, messages, tools=None, system=None) -> LLMResponse:
        resp = await self._inner.complete(messages, tools=tools, system=system)
        self._ledger.record(self._purpose, resp.input_tokens + resp.output_tokens)
        return resp

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.embed(texts)

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Must delegate: provider backends convert schemas (e.g. Anthropic's
        # input_schema format) and the executor probes for this method.
        if hasattr(self._inner, "format_tools"):
            return self._inner.format_tools(tools)
        return tools
