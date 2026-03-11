"""Abstract Tool interface and @tool decorator."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, get_type_hints

from ..types import ToolResult


class Tool(ABC):
    """Base class for all agent tools.

    Subclass this and implement `execute`. The `schema` property returns
    the JSON schema used for LLM function-calling.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool and return a ToolResult."""

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"


# ---------------------------------------------------------------------------
# @tool decorator — turn any async function into a Tool
# ---------------------------------------------------------------------------

def tool(
    name: str | None = None,
    description: str = "",
    parameters: dict[str, Any] | None = None,
) -> Callable:
    """Decorator that wraps an async function as a Tool.

    Example::

        @tool(name="add", description="Add two numbers")
        async def add(a: int, b: int) -> int:
            return a + b
    """
    def decorator(fn: Callable) -> Tool:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip()

        # Auto-infer parameters schema from function signature
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        props: dict[str, Any] = {}
        required = []

        _type_map = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "kwargs"):
                continue
            ann = hints.get(param_name, str)
            json_type = _type_map.get(ann, "string")
            props[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        inferred_params = parameters or {
            "type": "object",
            "properties": props,
            "required": required,
        }

        _fn = fn  # capture for closure

        class _FnTool(Tool):
            name = tool_name
            description = tool_desc
            parameters = inferred_params

            async def execute(self, **kwargs: Any) -> ToolResult:
                import uuid  # noqa: PLC0415
                call_id = str(uuid.uuid4())
                try:
                    output = await _fn(**kwargs)
                    return ToolResult(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        output=output,
                    )
                except Exception as e:
                    return ToolResult(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        output=None,
                        error=str(e),
                    )

        return _FnTool()

    return decorator
