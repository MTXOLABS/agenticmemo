"""Outcome Verification — ground TaskStatus in an actual judgment.

The Executor cannot judge correctness: it only knows whether the LLM
produced a final answer. Without verification, any non-empty answer counts
as SUCCESS, which inflates rewards, starves the FailureMiner (CFM) of real
failures, and flattens the GRPO advantage signal.

The OutcomeVerifier closes this gap with an LLM-as-judge pass after each
execution attempt:
  - "success"  → the answer directly and correctly addresses the task
  - "partial"  → incomplete, unverified, or only partially addresses it
  - "failure"  → wrong, off-topic, or the agent gave up

Fail-soft: on any LLM/parse error the executor's tentative status is kept.
"""

from __future__ import annotations

import json

from ..config import LearningConfig
from ..llm.base import LLMBackend
from ..types import Message, MessageRole, TaskStatus, Trajectory

_VERIFY_PROMPT = """\
You are a strict evaluator of an AI agent's task execution.

TASK: {task}

EXECUTION SUMMARY:
{trajectory}

FINAL ANSWER:
{answer}

Judge whether the final answer actually completes the task:
- "success": the answer directly and correctly addresses the task
- "partial": the answer is incomplete, unverified, or only partially addresses the task
- "failure": the answer is wrong, off-topic, or the agent gave up

Respond ONLY with JSON: {{"verdict": "success" | "partial" | "failure", "reason": "<one sentence>"}}
"""

_VERDICT_MAP = {
    "success": TaskStatus.SUCCESS,
    "partial": TaskStatus.PARTIAL,
    "failure": TaskStatus.FAILURE,
}


class OutcomeVerifier:
    """LLM-as-judge verification of a completed trajectory.

    Usage:
        verifier = OutcomeVerifier(llm, cfg)
        trajectory.status = await verifier.verify(task, trajectory)
    """

    def __init__(self, llm: LLMBackend, cfg: LearningConfig | None = None) -> None:
        self._llm = llm
        self._cfg = cfg or LearningConfig()

    async def verify(self, task: str, trajectory: Trajectory) -> TaskStatus:
        """Return the verified TaskStatus for a trajectory.

        Keeps the existing status when verification is disabled or the
        judge call fails; an empty final answer is always a FAILURE.
        """
        if not self._cfg.enable_verification:
            return trajectory.status
        if not trajectory.final_answer.strip():
            return TaskStatus.FAILURE

        prompt = _VERIFY_PROMPT.format(
            task=task,
            trajectory=self._summarize(trajectory),
            answer=trajectory.final_answer[:4000],
        )
        try:
            resp = await self._llm.complete(
                messages=[Message(role=MessageRole.USER, content=prompt)]
            )
            raw = resp.content
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return trajectory.status
            data = json.loads(raw[start:end])
            verdict = str(data.get("verdict", "")).lower().strip()
            return _VERDICT_MAP.get(verdict, trajectory.status)
        except Exception:
            return trajectory.status

    @staticmethod
    def _summarize(traj: Trajectory) -> str:
        # The judge needs to SEE the evidence: multi-deliverable tasks print
        # their results in tool observations, and over-truncating them forces
        # a systematic "partial" verdict (can't confirm what isn't shown).
        lines = []
        for s in traj.steps[:14]:
            line = f"Step {s.index}: {s.thought[:150]}"
            if s.tool_call:
                line += f" → {s.tool_call.name}(...)"
            if s.observation:
                line += f"\n  obs: {s.observation[:500]}"
            lines.append(line)
        if len(traj.steps) > 14:
            lines.append(f"... +{len(traj.steps) - 14} more steps")
        return "\n".join(lines) or "(no steps)"
