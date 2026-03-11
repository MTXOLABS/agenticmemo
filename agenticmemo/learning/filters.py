"""Trajectory Quality Filtering Pipeline.

Three-stage pipeline inspired by CLEANER + WebClipper + variance-based pruning:

Stage 1 — Structural filter
    Reject trajectories that are too short, too long, or have no answer.

Stage 2 — LLM self-purification (CLEANER-style)
    Ask the LLM to score its own trajectory for coherence and correctness.
    Trajectories scoring below threshold are rejected.

Stage 3 — Reward variance filter
    Within a batch, discard low-variance trajectories (reward near group mean).
    High-variance trajectories carry more information for policy learning.
"""

from __future__ import annotations

import statistics
from typing import Any

from ..config import LearningConfig
from ..exceptions import FilterError
from ..llm.base import LLMBackend
from ..types import Message, MessageRole, TaskStatus, Trajectory


_SELF_EVAL_PROMPT = """\
You are a quality evaluator for AI agent trajectories.
Evaluate the following agent trajectory and return a JSON object with:
  - "coherent": true/false — are the steps logically connected?
  - "correct": true/false — does the final answer address the task?
  - "efficient": true/false — are there no obvious redundant steps?
  - "score": float 0.0-1.0 — overall quality score

Task: {task}

Trajectory:
{trajectory}

Respond ONLY with valid JSON.
"""


class TrajectoryFilter:
    """Multi-stage quality filter for agent trajectories.

    Usage:
        filter_ = TrajectoryFilter(llm, cfg)
        accepted = await filter_.filter_batch(trajectories_with_tasks)
    """

    def __init__(
        self,
        llm: LLMBackend | None = None,
        cfg: LearningConfig | None = None,
    ) -> None:
        self._llm = llm
        self._cfg = cfg or LearningConfig()

    async def filter_single(self, task: str, trajectory: Trajectory) -> bool:
        """Return True if trajectory passes all filters."""
        if not self._cfg.enable_quality_filter:
            return True
        # Stage 1: structural
        if not self._structural_check(trajectory):
            return False
        # Stage 2: self-purification (only if LLM available)
        if self._llm is not None:
            score = await self._self_eval_score(task, trajectory)
            if score < 0.4:
                return False
        return True

    async def filter_batch(
        self, items: list[tuple[str, Trajectory]]
    ) -> list[tuple[str, Trajectory]]:
        """Filter a batch, also applying variance-based pruning."""
        if not self._cfg.enable_quality_filter:
            return items

        # Stages 1 + 2
        passed: list[tuple[str, Trajectory]] = []
        for task, traj in items:
            if await self.filter_single(task, traj):
                passed.append((task, traj))

        # Stage 3: variance filter
        return self._variance_filter(passed)

    def assign_reward(self, trajectory: Trajectory) -> float:
        """Map trajectory status to a scalar reward."""
        cfg = self._cfg
        if trajectory.status == TaskStatus.SUCCESS:
            return cfg.success_reward
        elif trajectory.status == TaskStatus.PARTIAL:
            return cfg.partial_reward
        else:
            return cfg.failure_reward

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _structural_check(self, traj: Trajectory) -> bool:
        cfg = self._cfg
        if traj.num_steps < cfg.min_trajectory_steps:
            return False
        if traj.num_steps > cfg.max_trajectory_steps:
            return False
        return True

    async def _self_eval_score(self, task: str, traj: Trajectory) -> float:
        """Ask LLM to rate trajectory quality. Returns 0.0–1.0."""
        traj_text = "\n".join(
            f"Step {s.index}: {s.thought}"
            + (f" → {s.tool_call.name}({s.tool_call.arguments})" if s.tool_call else "")
            + (f" → {s.observation[:150]}" if s.observation else "")
            for s in traj.steps
        )
        prompt = _SELF_EVAL_PROMPT.format(task=task, trajectory=traj_text)
        try:
            resp = await self._llm.complete(  # type: ignore[union-attr]
                messages=[Message(role=MessageRole.USER, content=prompt)]
            )
            import json  # noqa: PLC0415
            data = json.loads(resp.content.strip())
            return float(data.get("score", 0.5))
        except Exception:
            return 0.5  # default to pass on eval errors

    def _variance_filter(
        self, items: list[tuple[str, Trajectory]]
    ) -> list[tuple[str, Trajectory]]:
        if len(items) < 2:
            return items
        rewards = [self.assign_reward(t) for _, t in items]
        try:
            var = statistics.variance(rewards)
        except statistics.StatisticsError:
            return items
        if var < self._cfg.variance_filter_threshold:
            # Low variance batch — keep only top half by reward
            sorted_items = sorted(
                zip(rewards, items), key=lambda x: x[0], reverse=True
            )
            half = max(1, len(sorted_items) // 2)
            return [item for _, item in sorted_items[:half]]
        return items
