"""Case — the atomic unit stored in AgentMemento's memory."""

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

    def to_prompt_block(self) -> str:
        """Render as a structured prompt block for the planner."""
        lines = [
            f"--- Past Case (id={self.id[:8]}, reward={self.outcome.reward:.2f}) ---",
            f"Task: {self.task}",
            f"Status: {self.outcome.status.value}",
        ]
        for step in self.trajectory.steps:
            lines.append(f"  Step {step.index}: {step.thought}")
            if step.tool_call:
                lines.append(f"    Tool: {step.tool_call.name}({step.tool_call.arguments})")
            if step.observation:
                lines.append(f"    Obs: {step.observation[:200]}")
        lines.append(f"Answer: {self.outcome.answer[:200]}")
        if self.outcome.reflection:
            lines.append(f"Reflection: {self.outcome.reflection}")
        lines.append("---")
        return "\n".join(lines)
