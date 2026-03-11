"""Tool registry — maps tool names to Tool instances."""

from __future__ import annotations

from typing import Any

from ..exceptions import ToolError
from ..types import ToolCall, ToolResult
from .base import Tool


class ToolRegistry:
    """Central registry for all tools available to an agent.

    Usage::

        registry = ToolRegistry()
        registry.register(MyTool())
        result = await registry.call(tool_call)
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> "ToolRegistry":
        """Register a tool. Returns self for chaining."""
        self._tools[tool.name] = tool
        return self

    def register_many(self, *tools: Tool) -> "ToolRegistry":
        for t in tools:
            self.register(t)
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        """Return JSON schemas for all tools (for LLM function calling)."""
        return [t.schema for t in self._tools.values()]

    async def call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        tool = self._tools.get(tool_call.name)
        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                output=None,
                error=f"Unknown tool: '{tool_call.name}'. Available: {list(self._tools)}",
            )
        try:
            result = await tool.execute(**tool_call.arguments)
            result.tool_call_id = tool_call.id
            return result
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                output=None,
                error=str(e),
            )

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry({list(self._tools)})"
