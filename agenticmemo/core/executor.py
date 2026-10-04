"""Tool-based Executor.

The Executor performs Stage 2 of the Planner-Executor loop:
  1. Run the LLM in a ReAct (Reason + Act) loop.
  2. On each step: LLM reasons → calls a tool → observes result → continues.
  3. Stops when LLM produces a final answer (no tool call) or max_steps reached.
  4. Returns a completed Trajectory.

Escape v2 — Dynamic Mid-Execution Retrieval (DMER):
  At configurable step intervals the executor re-queries memory using the
  CURRENT execution state (task + recent observations) as the query.
  Retrieved hints are injected into the conversation as a SYSTEM-level
  memory refresh, giving the agent real-time guidance mid-task.

  Why this matters: vanilla ReAct agents receive NO memory guidance after
  the initial plan. On hard multi-step problems this means the agent is
  flying blind from step 3 onward. DMER closes this gap.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from ..config import AgentConfig
from ..exceptions import LLMError
from ..llm.base import LLMBackend
from ..tools.registry import ToolRegistry
from ..types import (
    Message,
    MessageRole,
    Step,
    TaskStatus,
    ToolResult,
    Trajectory,
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
- Keep each tool call small: short code blocks (under ~40 lines); split long
  computations across multiple steps. Large single calls often fail to parse.
- When you have enough information, provide a final answer WITHOUT calling any tool.
- Be concise in your reasoning.
- If a tool returns an error, try an alternative approach.
"""

_MALFORMED_CALL_MSG = """\
Your previous response could not be processed (the tool call failed to parse as \
valid JSON — this usually happens with very long code arguments). Try again: \
call ONE tool with a SHORTER argument (code under 40 lines), or give your final \
answer as plain text with no tool call."""

_GUIDE_TEMPLATE = (
    "[REFERENCE TRAJECTORY] In the solved similar task, "
    "the next action here was: {action}"
)

_ERROR_STREAK_MSG = """\
[COURSE CORRECTION] Two consecutive tool calls errored. Change your approach: \
simplify the code, split it into smaller pieces, and print intermediate values \
so you can see where it breaks."""

_REPEATED_CALL_MSG = """\
[COURSE CORRECTION] You repeated the exact same tool call. Repeating it will \
produce the same result — change the code or the approach."""

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
        retriever: EnsembleRetriever | None = None,
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
        exemplar_steps: list[str] | None = None,
    ) -> Trajectory:
        """Run the execution loop and return a Trajectory.

        Args:
            task:           The original task string.
            plan:           Plan text from the Planner.
            system_prefix:  Optional extra system context (e.g. reflexion).
            exemplar_steps: One-line actions from a solved similar trajectory;
                            injected one per turn as a sequential scaffold
                            (Phase 3.2 — combats mid-trajectory drift).
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
        llm_error_strikes = 0
        tool_error_streak = 0
        last_call_sig: tuple[str, str] | None = None

        for step_idx in range(self._cfg.max_steps):
            # Phase 3.2: sequential scaffold — show only the NEXT reference
            # action, so small models keep the thread without prompt bloat.
            if exemplar_steps and step_idx < len(exemplar_steps):
                messages.append(Message(
                    role=MessageRole.USER,
                    content=_GUIDE_TEMPLATE.format(action=exemplar_steps[step_idx]),
                ))
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

            # Fail-soft on provider errors (e.g. Groq `tool_use_failed` when a
            # model emits malformed tool-call JSON): nudge the model to retry
            # with a smaller call instead of killing the whole task. Two
            # consecutive provider errors end the task as FAILURE.
            try:
                resp = await self._llm.complete(
                    messages=messages,
                    tools=fmt_tools,
                    system=system,
                )
                llm_error_strikes = 0
            except LLMError as e:
                llm_error_strikes += 1
                if llm_error_strikes >= 2:
                    trajectory.status = TaskStatus.FAILURE
                    trajectory.final_answer = ""
                    trajectory.add_step(Step(
                        index=step_idx,
                        thought="[LLM provider error]",
                        observation=str(e)[:300],
                    ))
                    break
                messages.append(Message(
                    role=MessageRole.USER,
                    content=_MALFORMED_CALL_MSG,
                ))
                continue
            total_tokens += resp.input_tokens + resp.output_tokens

            if not resp.has_tool_calls:
                # Final answer. SUCCESS here is tentative — the Agent runs an
                # OutcomeVerifier pass afterwards to confirm or downgrade it.
                trajectory.final_answer = resp.content
                trajectory.status = (
                    TaskStatus.SUCCESS if resp.content.strip() else TaskStatus.FAILURE
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

            # Phase 3.3: mid-trajectory sanity checks — nudge early instead
            # of discovering a doomed trajectory only at the end.
            sig = (tool_call.name, json.dumps(tool_call.arguments, sort_keys=True, default=str))
            if tool_result.error:
                tool_error_streak += 1
            else:
                tool_error_streak = 0
            if tool_error_streak == 2:
                messages.append(Message(role=MessageRole.USER, content=_ERROR_STREAK_MSG))
            elif sig == last_call_sig:
                messages.append(Message(role=MessageRole.USER, content=_REPEATED_CALL_MSG))
            last_call_sig = sig

        else:
            trajectory.status = TaskStatus.FAILURE
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
