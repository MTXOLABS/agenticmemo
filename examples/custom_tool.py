"""Example: Defining a custom tool with the @tool decorator.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/custom_tool.py
"""

import asyncio
import os

from agenticmemo import Agent
from agenticmemo.tools import tool, PythonReplTool


# ---------------------------------------------------------------------------
# Define a custom tool using the @tool decorator
# ---------------------------------------------------------------------------

@tool(
    name="celsius_to_fahrenheit",
    description="Convert temperature from Celsius to Fahrenheit",
)
async def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


@tool(
    name="word_count",
    description="Count the number of words in a text string",
)
async def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Run agent with custom tools
# ---------------------------------------------------------------------------

async def main() -> None:
    agent = Agent.from_anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        cfg=None,
    )
    agent.add_tools(celsius_to_fahrenheit, word_count, PythonReplTool())

    result = await agent.run(
        "Convert 100 degrees Celsius to Fahrenheit, then count the words "
        "in the phrase 'the quick brown fox jumps over the lazy dog'."
    )
    print(f"Answer: {result.final_answer}")
    print(f"Status: {result.status.value}")


if __name__ == "__main__":
    asyncio.run(main())
