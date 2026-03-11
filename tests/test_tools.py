"""Tests for the tools system."""

import pytest

from agenticmemo.tools.base import Tool, tool
from agenticmemo.tools.registry import ToolRegistry
from agenticmemo.tools.builtin import PythonReplTool, FileReadTool, FileWriteTool
from agenticmemo.types import ToolCall


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_decorator_basic():
    @tool(name="add", description="Add two numbers")
    async def add(a: int, b: int) -> int:
        return a + b

    result = await add.execute(a=2, b=3)
    assert result.output == 5
    assert result.error is None

@pytest.mark.asyncio
async def test_tool_decorator_schema():
    @tool(name="greet")
    async def greet(name: str) -> str:
        """Greet a person by name."""
        return f"Hello, {name}!"

    assert greet.name == "greet"
    assert "name" in greet.schema["parameters"]["properties"]

@pytest.mark.asyncio
async def test_tool_decorator_captures_exception():
    @tool(name="fail_tool")
    async def fail_tool(x: int) -> int:
        raise ValueError("intentional failure")

    result = await fail_tool.execute(x=1)
    assert result.error is not None
    assert "intentional failure" in result.error


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_registry_register_and_call():
    registry = ToolRegistry()

    @tool(name="multiply")
    async def multiply(a: int, b: int) -> int:
        return a * b

    registry.register(multiply)
    tc = ToolCall(id="tc1", name="multiply", arguments={"a": 4, "b": 5})
    result = await registry.call(tc)
    assert result.output == 20

@pytest.mark.asyncio
async def test_registry_unknown_tool():
    registry = ToolRegistry()
    tc = ToolCall(id="tc2", name="nonexistent", arguments={})
    result = await registry.call(tc)
    assert result.error is not None
    assert "nonexistent" in result.error

def test_registry_schemas():
    registry = ToolRegistry()
    registry.register(PythonReplTool())
    schemas = registry.schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "python_repl"

def test_registry_contains():
    registry = ToolRegistry()
    registry.register(FileWriteTool())
    assert "file_write" in registry
    assert "unknown" not in registry


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_python_repl_basic():
    repl = PythonReplTool()
    result = await repl.execute(code="print(2 + 2)")
    assert result.error is None
    assert "4" in str(result.output)

@pytest.mark.asyncio
async def test_python_repl_persistent_namespace():
    repl = PythonReplTool()
    await repl.execute(code="x = 42")
    result = await repl.execute(code="print(x)")
    assert "42" in str(result.output)

@pytest.mark.asyncio
async def test_python_repl_captures_syntax_error():
    repl = PythonReplTool()
    result = await repl.execute(code="def bad(: pass")
    assert result.error is not None

@pytest.mark.asyncio
async def test_file_write_and_read(tmp_path):
    write_tool = FileWriteTool()
    read_tool = FileReadTool()
    path = str(tmp_path / "test.txt")

    write_result = await write_tool.execute(path=path, content="hello memento")
    assert write_result.error is None

    read_result = await read_tool.execute(path=path)
    assert read_result.error is None
    assert read_result.output == "hello memento"

@pytest.mark.asyncio
async def test_file_read_missing_file():
    read_tool = FileReadTool()
    result = await read_tool.execute(path="/nonexistent/path/file.txt")
    assert result.error is not None
