"""Escape example: Inspect memory across multiple tasks.

This example runs the same agent on similar tasks and reports its steps, tokens,
stored cases, and GRPO statistics. It uses persistent memory. These observations
do not establish a quality or performance improvement; that requires a controlled
evaluation against the same agent without memory.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/multi_task_memory.py
"""

import asyncio
import os
import time

from agenticmemo import Agent, AgentConfig, MemoryConfig, RetrievalConfig
from agenticmemo.tools import PythonReplTool

TASKS = [
    # Round 1: warm up
    "Write a Python function to reverse a string.",
    "Write a Python function to check if a string is a palindrome.",
    # Round 2: similar tasks can retrieve earlier stored experience
    "Write a Python function to count vowels in a string.",
    "Write a Python function that returns the longest word in a sentence.",
    # Round 3: another task in the same domain
    "Write a Python function that removes duplicate characters from a string.",
]


async def main() -> None:
    agent = Agent.from_anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        cfg=AgentConfig(
            verbose=False,
            memory=MemoryConfig(persist_path="./string_tasks_memory.json"),
            retrieval=RetrievalConfig(top_k=3),
        ),
    )
    agent.add_tool(PythonReplTool())

    print("Escape — Memory accumulation demo")
    print("="*55)

    for i, task in enumerate(TASKS, 1):
        t0 = time.time()
        result = await agent.run(task)
        elapsed = time.time() - t0
        mem_size = await agent.memory_size()

        print(f"\nTask {i}: {task[:60]}")
        print(f"  Status : {result.status.value}")
        print(f"  Steps  : {result.num_steps}")
        print(f"  Tokens : {result.total_tokens}")
        print(f"  Time   : {elapsed:.2f}s")
        print(f"  Memory : {mem_size} cases stored")

    print("\n" + "="*55)
    print("Final GRPO stats:", agent.grpo_stats())

    # Show top cases in memory
    recent = await agent.recent_cases(3)
    print("\nTop 3 recent cases in memory:")
    for c in recent:
        print(f"  [{c.outcome.reward:.2f}] {c.task[:70]}")


if __name__ == "__main__":
    asyncio.run(main())
