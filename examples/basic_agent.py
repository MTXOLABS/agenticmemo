"""Basic AgentMemento example — coding assistant with memory.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/basic_agent.py
"""

import asyncio
import os

from agenticmemo import Agent, AgentConfig, MemoryConfig
from agenticmemo.tools import PythonReplTool, FileWriteTool


async def main() -> None:
    # Build agent
    agent = Agent.from_anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        cfg=AgentConfig(
            verbose=True,
            memory=MemoryConfig(persist_path="./case_bank.json"),
        ),
    )
    agent.add_tools(PythonReplTool(), FileWriteTool())

    tasks = [
        "Write a Python function that checks whether a number is prime, then test it with 17 and 18.",
        "Write a Python function to compute the nth Fibonacci number recursively, then call it for n=10.",
        "Create a Python class called Stack with push, pop, and peek methods, then demonstrate its usage.",
    ]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print('='*60)
        result = await agent.run(task)
        print(f"\nANSWER:\n{result.final_answer}")
        print(f"\nStatus: {result.status.value} | Steps: {result.num_steps} | Tokens: {result.total_tokens}")

    print(f"\n{'='*60}")
    print(f"Memory size after {len(tasks)} tasks: {await agent.memory_size()} cases")
    print(f"GRPO stats: {agent.grpo_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
