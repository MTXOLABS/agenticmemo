"""Tool-based Executor.

The Executor performs Stage 2 of the Planner-Executor loop:
  1. Run the LLM in a ReAct (Reason + Act) loop.
  2. On each step: LLM reasons → calls a tool → observes result → continues.
  3. Stops when LLM produces a final answer (no tool call) or max_steps reached.
  4. Returns a completed Trajectory.

AgenticMemo v2 — Dynamic Mid-Execution Retrieval (DMER):
  At configurable step intervals the executor re-queries memory using the
  CURRENT execution state (task + recent observations) as the query.
  Retrieved hints are injected into the conversation as a SYSTEM-level
  memory refresh, giving the agent real-time guidance mid-task.

  Why this matters: vanilla ReAct agents receive NO memory guidance after
  the initial plan. On hard multi-step problems this means the agent is
  flying blind from step 3 onward. DMER closes this gap.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from ..config import AgentConfig
from ..llm.base import LLMBackend
from ..tools.registry import ToolRegistry
from ..types import (
    Message,
    MessageRole,
    Step,
    TaskStatus,
    Trajectory,
    ToolResult,
)

if TYPE_CHECKING:
    from ..retrieval.ensemble import EnsembleRetriever


_EXEC_SYSTEM = """\
You are a highly capable AI agent. Execute the given task by calling tools step by step.

PLAN TO FOLLOW:
{plan}

Rules:
- Follow the plan unless you discover a better approach.
- Call one tool per step; observe the result before proceeding.
- When you have enough information, provide a final answer WITHOUT calling any tool.
- Be concise in your reasoning.
- If a tool returns an error, try an alternative approach.
"""

_MEMORY_REFRESH_TEMPLATE = """\
[MEMORY REFRESH at step {step}]
You have executed {step} steps so far. Here is relevant experience from memory
based on your current execution state:

{cases}

Apply these insights if helpful. Continue executing the task."""


class Executor:
    """ReAct-style execution loop with Dynamic Mid-Execution Retrieval (DMER).

    DMER re-queries memory every `memory_refresh_every` steps using the
    current execution context (task + recent tool observations) as the
    query. This gives the agent real-time memory guidance mid-task,
    closing the blind-spot that exists after planning.
    """

    def __init__(
        self,
        llm: LLMBackend,
        tools: ToolRegistry,
        cfg: AgentConfig | None = None,
        retriever: "EnsembleRetriever | None" = None,
        memory_refresh_every: int = 3,
        memory_refresh_top_k: int = 2,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._cfg = cfg or AgentConfig()
        self._retriever = retriever               # DMER: optional mid-exec retriever
        self._refresh_every = memory_refresh_every
        self._refresh_top_k = memory_refresh_top_k

    async def execute(
        self,
        task: str,
        plan: str,
        system_prefix: str | None = None,
    ) -> Trajectory:
        """Run the execution loop and return a Trajectory.

        Args:
            task:          The original task string.
            plan:          Plan text from the Planner.
            system_prefix: Optional extra system context (e.g. reflexion).
        """
        system = _EXEC_SYSTEM.format(plan=plan)
        if system_prefix:
            system = system_prefix + "\n\n" + system

        trajectory = Trajectory(task=task)
        messages: list[Message] = [
            Message(role=MessageRole.USER, content=f"Execute this task: {task}"),
        ]
        tool_schemas = self._tools.schemas()

        # Format tool schemas for the specific LLM provider
        fmt_tools = None
        if hasattr(self._llm, "format_tools") and tool_schemas:
            fmt_tools = self._llm.format_tools(tool_schemas)  # type: ignore[attr-defined]
        else:
            fmt_tools = tool_schemas or None

        start_time = time.time()
        total_tokens = 0

        for step_idx in range(self._cfg.max_steps):
            # DMER: inject memory refresh every N steps (not on step 0 — planner
            # already retrieved cases). This gives mid-task memory guidance.
            if (
                self._retriever is not None
                and step_idx > 0
                and step_idx % self._refresh_every == 0
            ):
                memory_hint = await self._build_memory_refresh(
                    task, messages, step_idx
                )
                if memory_hint:
                    messages.append(Message(
                        role=MessageRole.USER,
                        content=memory_hint,
                    ))

            resp = await self._llm.complete(
                messages=messages,
                tools=fmt_tools,
                system=system,
            )
            total_tokens += resp.input_tokens + resp.output_tokens

            if not resp.has_tool_calls:
                # Final answer
                trajectory.final_answer = resp.content
                trajectory.status = (
                    TaskStatus.SUCCESS if resp.content.strip() else TaskStatus.PARTIAL
                )
                step = Step(
                    index=step_idx,
                    thought=resp.content,
                    observation="[Final answer produced]",
                )
                trajectory.add_step(step)
                break

            # Process tool calls (take first for single-step execution)
            tool_call = resp.tool_calls[0]
            tool_result: ToolResult = await self._tools.call(tool_call)

            step = Step(
                index=step_idx,
                thought=resp.content or f"Calling {tool_call.name}",
                tool_call=tool_call,
                tool_result=tool_result,
                observation=self._format_result(tool_result),
            )
            trajectory.add_step(step)

            # Add assistant turn — only include the ONE tool call we executed,
            # not all of resp.tool_calls. Both Anthropic and OpenAI require that
            # every tool_call_id in the assistant message has a matching tool result.
            messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=resp.content or "",
                tool_calls=[tool_call],
            ))
            messages.append(Message(
                role=MessageRole.TOOL,
                content=step.observation,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
            ))

        else:
            trajectory.status = TaskStatus.PARTIAL
            trajectory.final_answer = "Max steps reached without final answer."

        trajectory.total_tokens = total_tokens
        trajectory.duration_ms = (time.time() - start_time) * 1000
        return trajectory

    async def _build_memory_refresh(
        self,
        task: str,
        messages: list[Message],
        step_idx: int,
    ) -> str:
        """Build a mid-execution memory refresh message (DMER).

        Query = task + last N tool observations to capture current execution state.
        """
        assert self._retriever is not None
        # Build execution-state query: task + recent observations
        recent_obs = []
        for msg in messages[-6:]:
            if msg.role == MessageRole.TOOL:
                recent_obs.append(msg.content[:200])
        if not recent_obs:
            return ""
        state_query = f"{task}\nCurrent observations: {' | '.join(recent_obs)}"
        try:
            results = await self._retriever.retrieve(
                state_query,
                top_k=self._refresh_top_k,
                min_score=0.3,
            )
        except Exception:
            return ""
        if not results:
            return ""
        case_blocks = [c.to_prompt_block() for c, _ in results]
        return _MEMORY_REFRESH_TEMPLATE.format(
            step=step_idx,
            cases="\n".join(case_blocks),
        )

    @staticmethod
    def _format_result(result: ToolResult) -> str:
        if result.error:
            return f"ERROR: {result.error}"
        output = result.output
        if isinstance(output, (list, dict)):
            import json  # noqa: PLC0415
            text = json.dumps(output, ensure_ascii=False)
        else:
            text = str(output)
        return text[:2000]  # truncate very long outputs
