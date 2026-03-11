"""Reflexion Engine — learn from failure via self-reflection.

Inspired by Shinn et al. (Reflexion, 2023) and MAR (Multi-Agent Reflexion, 2024).

On task failure:
  1. Ask the LLM to diagnose what went wrong.
  2. Generate a concrete correction plan.
  3. Attach (failure, reflection, fix) triple to the Case for future retrieval.
  4. Optionally retry the task with the reflection as additional context.

The reflection is stored in Case.outcome.reflection so future retrieval
surfaces both "what worked" AND "what failed and why".
"""

from __future__ import annotations

from ..config import LearningConfig
from ..llm.base import LLMBackend
from ..types import Message, MessageRole, TaskStatus, Trajectory


_REFLECT_PROMPT = """\
You are an expert AI agent trainer reviewing a failed task execution.

TASK: {task}

FAILED TRAJECTORY:
{trajectory}

FAILURE REASON: The agent did not produce a correct answer.

Your job:
1. Diagnose exactly why this trajectory failed (1-2 sentences).
2. Identify the key mistake(s): wrong tool, wrong reasoning, missing step.
3. Write a concrete correction strategy that a future agent should follow.

Format your response as:
DIAGNOSIS: <what went wrong>
MISTAKES: <list the specific errors>
CORRECTION: <what to do differently next time>
"""

_RETRY_SYSTEM = """\
You are a highly capable AI agent. A previous attempt at this task failed.
Use the reflection below to guide your improved attempt.

REFLECTION FROM FAILED ATTEMPT:
{reflection}

Apply the CORRECTION strategy. Do not repeat the same mistakes.
"""


class ReflexionEngine:
    """Self-reflection loop for failed trajectories.

    Usage:
        engine = ReflexionEngine(llm, cfg)
        reflection = await engine.reflect(task, failed_trajectory)
        # reflection is a string; attach to case.outcome.reflection
    """

    def __init__(self, llm: LLMBackend, cfg: LearningConfig | None = None) -> None:
        self._llm = llm
        self._cfg = cfg or LearningConfig()

    async def reflect(self, task: str, trajectory: Trajectory) -> str:
        """Generate a reflection string for a failed/partial trajectory."""
        if not self._cfg.enable_reflexion:
            return ""
        if trajectory.status == TaskStatus.SUCCESS:
            return ""

        traj_text = self._format_trajectory(trajectory)
        prompt = _REFLECT_PROMPT.format(task=task, trajectory=traj_text)

        try:
            resp = await self._llm.complete(
                messages=[Message(role=MessageRole.USER, content=prompt)]
            )
            return resp.content.strip()
        except Exception:
            return "Reflection unavailable."

    def build_retry_system(self, reflection: str) -> str:
        """Return a system prompt that injects the reflection for a retry."""
        return _RETRY_SYSTEM.format(reflection=reflection)

    def should_retry(self, trajectory: Trajectory, attempt: int) -> bool:
        """Decide whether to attempt reflexion-guided retry."""
        if not self._cfg.enable_reflexion:
            return False
        if attempt >= self._cfg.max_reflexion_retries:
            return False
        return trajectory.status != TaskStatus.SUCCESS

    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_trajectory(traj: Trajectory) -> str:
        lines = []
        for s in traj.steps:
            line = f"Step {s.index}: {s.thought}"
            if s.tool_call:
                line += f" → tool={s.tool_call.name}, args={s.tool_call.arguments}"
            if s.observation:
                line += f" → obs={s.observation[:200]}"
            lines.append(line)
        lines.append(f"Final answer: {traj.final_answer[:300]}")
        return "\n".join(lines)
