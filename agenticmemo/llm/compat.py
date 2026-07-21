"""Provider/model compatibility quirks.

Some models have trained-in tool expectations or output behaviors that break
naive integrations. Centralizing them here means every Agent benefits, not
just whichever benchmark discovered the quirk.

Known quirks handled:
- gpt-oss family (Groq, local): trained with a native built-in ``python``
  tool. On code-heavy tasks they call it by that name; if only a
  ``python_repl`` function is declared, the provider 400s the request
  ("python tool not enabled"). Naming our tool ``python`` makes the model's
  instinct land on the declared function.
- Reasoning families (gpt-5*, gpt-oss, o-series): "thinking" consumes the
  completion budget invisibly. Output budgets under ~2000 tokens routinely
  truncate mid-reasoning, yielding empty responses that look like model
  failures but are configuration errors.
"""

from __future__ import annotations

import warnings

from ..tools.base import Tool
from .base import LLMBackend

_REASONING_MARKERS = ("gpt-5", "gpt-oss", "o1", "o3", "o4")
_MIN_REASONING_OUTPUT = 2000


def preferred_python_tool_name(model: str) -> str | None:
    """Return the tool name a model expects for python execution, if quirky."""
    if "gpt-oss" in model:
        return "python"
    return None


def apply_tool_quirks(tool: Tool, model: str) -> Tool:
    """Adjust a tool in place for model-specific naming expectations."""
    preferred = preferred_python_tool_name(model)
    if preferred and tool.name == "python_repl":
        tool.name = preferred
    return tool


def warn_if_output_budget_low(llm: LLMBackend) -> None:
    """Warn when a reasoning model has too little output budget to answer."""
    model = (llm.model or "").lower()
    max_tokens = getattr(llm, "max_tokens", None)
    if max_tokens is None:
        return
    if any(m in model for m in _REASONING_MARKERS) and max_tokens < _MIN_REASONING_OUTPUT:
        warnings.warn(
            f"Model '{llm.model}' is a reasoning model but max_tokens={max_tokens}. "
            f"Reasoning consumes the completion budget; below ~{_MIN_REASONING_OUTPUT} "
            "responses often truncate to empty. Raise max_tokens.",
            stacklevel=3,
        )
