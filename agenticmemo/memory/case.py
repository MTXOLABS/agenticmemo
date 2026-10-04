"""Case — the atomic unit stored in Escape's memory."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from ..types import MemoryDomain, TaskStatus, Trajectory


class CaseOutcome(BaseModel):
    """Structured outcome attached to a Case."""
    status: TaskStatus
    reward: float = 0.0
    answer: str = ""
    reflection: str = ""          # filled by ReflexionEngine on failure
    error_type: str | None = None


class Case(BaseModel):
    """A stored experience: task + trajectory + outcome.

    Cases are the fundamental knowledge units in the Case Bank.
    They encode *what was tried*, *how it went*, and *what was learned*
    so the retrieval system can surface relevant past experience.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str
    domain: MemoryDomain = MemoryDomain.GENERAL
    category: str = ""            # sub-category within domain
    trajectory: Trajectory
    outcome: CaseOutcome
    embedding: list[float] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    related_case_ids: list[str] = Field(default_factory=list)
    # Distilled working artifact (e.g. the code that solved the task).
    # Injecting a PROVEN solution lets a weaker model adapt instead of
    # re-derive — the main mechanism behind memory's step reduction.
    solution: str = ""
    # The plan that led to this outcome — reused verbatim (inputs adapted) on
    # strong retrieval hits so weak planners don't have to re-plan (Phase 3.1).
    plan: str = ""
    q_value: float = 0.0          # soft Q-value for retrieval policy
    access_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #

    @property
    def age_days(self) -> float:
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.total_seconds() / 86_400

    @property
    def is_success(self) -> bool:
        return self.outcome.status == TaskStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        return self.outcome.status == TaskStatus.FAILURE

    def touch(self) -> None:
        """Record an access (used for staleness tracking)."""
        self.access_count += 1
        self.updated_at = datetime.now(timezone.utc)

    def summary(self) -> str:
        """One-line human-readable summary for prompt injection."""
        status_icon = "✓" if self.is_success else "✗"
        steps = self.trajectory.num_steps
        return (
            f"[{status_icon}] Task: {self.task[:80]} | "
            f"Steps: {steps} | Reward: {self.outcome.reward:.2f} | "
            f"Answer: {self.outcome.answer[:100]}"
        )

    def to_prompt_block(self, max_steps: int = 4, obs_chars: int = 120) -> str:
        """Render as a compact prompt block for the planner.

        Compact by design: injected cases multiply across top_k retrievals and
        DMER refreshes, so an uncapped trajectory dump bloats every request —
        it can even exceed provider per-minute token windows. The first steps
        plus the outcome carry most of the reusable signal.
        """
        lines = [
            f"--- Past Case (id={self.id[:8]}, reward={self.outcome.reward:.2f}) ---",
            f"Task: {self.task[:300]}",
            f"Status: {self.outcome.status.value}",
        ]
        for step in self.trajectory.steps[:max_steps]:
            line = f"  Step {step.index}: {step.thought[:150]}"
            if step.tool_call:
                line += f" → {step.tool_call.name}"
            lines.append(line)
            if step.observation:
                lines.append(f"    Obs: {step.observation[:obs_chars]}")
        hidden = len(self.trajectory.steps) - max_steps
        if hidden > 0:
            lines.append(f"  ... +{hidden} more steps")
        # When a proven solution exists, show the CODE and hide the printed
        # answer: models near their capability edge copy visible reference
        # outputs instead of executing (observed on three model families).
        if self.solution:
            lines.append(
                "Proven solution — adapt to current inputs and EXECUTE it; "
                f"the reference output is not shown on purpose:\n{self.solution[:1200]}"
            )
        else:
            lines.append(f"Answer: {self.outcome.answer[:200]}")
        if self.outcome.reflection:
            lines.append(f"Reflection: {self.outcome.reflection[:200]}")
        lines.append("---")
        return "\n".join(lines)


def extract_solution(trajectory: Trajectory, max_chars: int = 2500) -> str:
    """Distill the working artifact from a successful trajectory.

    Collects the arguments of tool calls whose execution produced no error —
    for code tasks this is the actual working code. Later calls overwrite the
    budget of earlier ones (the final, corrected version matters most).
    """
    if trajectory.status != TaskStatus.SUCCESS:
        return ""
    parts: list[str] = []
    for step in trajectory.steps:
        if not step.tool_call or step.tool_result is None or step.tool_result.error:
            continue
        args = step.tool_call.arguments
        # Code-like payloads carry the real solution; other args add context
        payload = args.get("code") or args.get("query") or ""
        if payload:
            parts.append(f"# via {step.tool_call.name}\n{payload}")
    solution = "\n\n".join(parts)
    return solution[-max_chars:] if solution else ""
