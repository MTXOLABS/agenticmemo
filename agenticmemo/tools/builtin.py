"""Built-in tools bundled with AgentMemento."""

from __future__ import annotations

import ast
import io
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

from .base import Tool
from ..types import ToolResult


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo (no API key required)."""

    name = "web_search"
    description = "Search the web for up-to-date information."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Max results to return", "default": 5},
        },
        "required": ["query"],
    }

    async def execute(self, query: str, max_results: int = 5, **_: Any) -> ToolResult:
        import uuid  # noqa: PLC0415
        call_id = str(uuid.uuid4())
        try:
            from duckduckgo_search import DDGS  # noqa: PLC0415
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            return ToolResult(tool_call_id=call_id, tool_name=self.name, output=results)
        except ImportError:
            return ToolResult(
                tool_call_id=call_id, tool_name=self.name, output=None,
                error="duckduckgo-search not installed. Run: pip install duckduckgo-search",
            )
        except Exception as e:
            return ToolResult(tool_call_id=call_id, tool_name=self.name, output=None, error=str(e))


class PythonReplTool(Tool):
    """Execute Python code in an isolated namespace and return stdout/result."""

    name = "python_repl"
    description = "Execute Python code and return the output."
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
        },
        "required": ["code"],
    }

    def __init__(self) -> None:
        super().__init__()
        self._namespace: dict[str, Any] = {}

    async def execute(self, code: str, **_: Any) -> ToolResult:
        import uuid  # noqa: PLC0415
        call_id = str(uuid.uuid4())
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        start = time.time()
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(code, "<agent>", "exec"), self._namespace)  # noqa: S102
            output = stdout_buf.getvalue()
            err = stderr_buf.getvalue()
            duration_ms = (time.time() - start) * 1000
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                output=output or "(no output)",
                error=err or None,
                duration_ms=duration_ms,
            )
        except Exception:
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                output=None,
                error=traceback.format_exc(),
                duration_ms=(time.time() - start) * 1000,
            )


class FileReadTool(Tool):
    """Read a file from the local filesystem."""

    name = "file_read"
    description = "Read the contents of a local file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to the file"},
        },
        "required": ["path"],
    }

    async def execute(self, path: str, **_: Any) -> ToolResult:
        import uuid  # noqa: PLC0415
        call_id = str(uuid.uuid4())
        try:
            content = Path(path).read_text(encoding="utf-8")
            return ToolResult(tool_call_id=call_id, tool_name=self.name, output=content)
        except Exception as e:
            return ToolResult(tool_call_id=call_id, tool_name=self.name, output=None, error=str(e))


class FileWriteTool(Tool):
    """Write content to a local file."""

    name = "file_write"
    description = "Write text content to a local file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write the file"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    }

    async def execute(self, path: str, content: str, **_: Any) -> ToolResult:
        import uuid  # noqa: PLC0415
        call_id = str(uuid.uuid4())
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(
                tool_call_id=call_id, tool_name=self.name,
                output=f"Written {len(content)} chars to {path}",
            )
        except Exception as e:
            return ToolResult(tool_call_id=call_id, tool_name=self.name, output=None, error=str(e))
