"""Episodic-to-Semantic Memory Consolidation — AgenticMemo v2.

Converts accumulated episodic (raw trajectory) memories into compact,
reusable SEMANTIC SKILLS — abstract procedural knowledge that transfers
across surface-level task variations.

Key insight: After seeing 15 cases of "implement graph BFS", the agent
doesn't need to store all 15 raw trajectories. It can distill them into:
  SKILL: graph_traversal_bfs
  TRIGGER: tasks involving finding paths or connected components in graphs
  STRATEGY: Initialize queue with start node → BFS with visited set → ...
  EFFICIENCY: avg 3.2 steps, 85% success rate across 15 cases

This is the hippocampal-cortical transfer analogy from cognitive neuroscience:
episodic memories (specific experiences) → semantic memories (general knowledge).

Why novel:
- No existing agent memory framework implements episodic→semantic consolidation
- Reduces memory footprint while IMPROVING retrieval precision
- Enables zero-shot transfer to genuinely new task variants
- Two-tier retrieval: Skills (abstract) + Cases (specific) blend

Reference: inspired by Memory Consolidation in cognitive neuroscience
(McClelland et al. 1995) adapted for agentic AI systems.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..llm.base import LLMBackend
from ..memory.case import Case
from ..types import MemoryDomain


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Skill:
    """An abstract procedural skill distilled from multiple episodic cases.

    A skill is more compact than a raw case and more generalizable.
    It captures the ESSENCE of how to solve a CLASS of problems.
    """
    id: str
    name: str                    # Short human-readable name
    domain: str
    trigger: str                 # When to apply this skill
    strategy: str                # How to apply it (step-by-step approach)
    key_tools: list[str]         # Which tools are typically used
    avg_steps: float             # Expected number of steps
    avg_reward: float            # Average reward across source cases
    success_rate: float          # Success rate across source cases
    support: int                 # Number of source cases
    source_case_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_prompt_block(self) -> str:
        """Render as a positive experience block for the planner."""
        return (
            f"SKILL: {self.name} (domain={self.domain}, "
            f"success_rate={self.success_rate:.0%}, avg_steps={self.avg_steps:.1f}, "
            f"support={self.support} cases)\n"
            f"  WHEN: {self.trigger}\n"
            f"  HOW : {self.strategy}\n"
            f"  TOOLS: {', '.join(self.key_tools) if self.key_tools else 'varies'}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "trigger": self.trigger,
            "strategy": self.strategy,
            "key_tools": self.key_tools,
            "avg_steps": self.avg_steps,
            "avg_reward": self.avg_reward,
            "success_rate": self.success_rate,
            "support": self.support,
            "source_case_ids": self.source_case_ids,
        }


class SkillLibrary:
    """Persistent store of distilled skills."""

    def __init__(self, persist_path: str | None = None) -> None:
        self._skills: dict[str, Skill] = {}           # id → skill
        self._domain_index: dict[str, list[str]] = {} # domain → [ids]
        self._persist_path = persist_path
        if persist_path:
            self._load(persist_path)

    def add(self, skill: Skill) -> None:
        self._skills[skill.id] = skill
        self._domain_index.setdefault(skill.domain, [])
        if skill.id not in self._domain_index[skill.domain]:
            self._domain_index[skill.domain].append(skill.id)
        if self._persist_path:
            self._save()

    def update(self, skill: Skill) -> None:
        """Update existing skill (when consolidating more cases)."""
        if skill.id in self._skills:
            old = self._skills[skill.id]
            # Merge support
            skill.support = old.support + skill.support
            skill.source_case_ids = list(set(old.source_case_ids + skill.source_case_ids))
            skill.last_updated = datetime.utcnow().isoformat()
        self.add(skill)

    def by_domain(self, domain: str, top_n: int = 3) -> list[Skill]:
        ids = self._domain_index.get(domain, [])
        skills = [self._skills[i] for i in ids if i in self._skills]
        return sorted(skills, key=lambda s: s.success_rate * s.support, reverse=True)[:top_n]

    def top(self, top_n: int = 5) -> list[Skill]:
        return sorted(
            self._skills.values(),
            key=lambda s: s.success_rate * s.support,
            reverse=True
        )[:top_n]

    def to_prompt_block(self, domain: str | None = None, top_n: int = 3) -> str:
        if domain:
            skills = self.by_domain(domain, top_n)
        else:
            skills = self.top(top_n)
        if not skills:
            return ""
        lines = ["CONSOLIDATED SKILLS (distilled from past experience):"]
        for s in skills:
            lines.append(s.to_prompt_block())
            lines.append("")
        return "\n".join(lines)

    def size(self) -> int:
        return len(self._skills)

    def all_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def _save(self) -> None:
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        p = Path(self._persist_path)  # type: ignore[arg-type]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([s.to_dict() for s in self._skills.values()], default=str))

    def _load(self, path: str) -> None:
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        p = Path(path)
        if not p.exists():
            return
        try:
            for d in json.loads(p.read_text()):
                s = Skill(**d)
                self._skills[s.id] = s
                self._domain_index.setdefault(s.domain, [])
                if s.id not in self._domain_index[s.domain]:
                    self._domain_index[s.domain].append(s.id)
        except Exception:
            pass  # corrupt file — start fresh


# ─────────────────────────────────────────────────────────────────────────────
# Consolidator
# ─────────────────────────────────────────────────────────────────────────────

_CONSOLIDATE_SYSTEM = """\
You are an expert AI agent trainer. Your job is to distill a set of similar successful
agent trajectories into a compact, generalizable SKILL.

A skill should:
- Identify the CLASS of problems it applies to (not just the specific tasks)
- Capture the STRATEGIC APPROACH (not low-level details)
- Be written so another agent can apply it to NEW similar tasks
- Include which tools are key and the expected workflow

Avoid being too specific. A good skill applies to 10+ similar tasks, not just one.
"""

_CONSOLIDATE_USER = """\
Analyze these {n} successful trajectories (all in domain: {domain}) and distill them
into {k} generalizable skills.

TRAJECTORIES:
{trajectories}

Return EXACTLY {k} skills as JSON:
[
  {{
    "name": "short_snake_case_skill_name",
    "trigger": "Description of when to apply this skill",
    "strategy": "Step-by-step approach description (2-4 sentences)",
    "key_tools": ["tool1", "tool2"]
  }},
  ...
]

Only return valid JSON.
"""


class SkillConsolidator:
    """Distills episodic cases into semantic skills.

    Algorithm:
    1. Group successful cases by domain when count >= min_cases_per_skill
    2. Call LLM to extract abstract skills from the group
    3. Store skills in SkillLibrary (for fast retrieval at planning time)
    4. Cases can optionally be pruned after consolidation to save memory

    Call consolidate() periodically (e.g., every 20 new successes).
    """

    def __init__(
        self,
        llm: LLMBackend,
        library: SkillLibrary | None = None,
        min_cases_per_skill: int = 5,
        skills_per_batch: int = 2,
    ) -> None:
        self._llm = llm
        self._library = library or SkillLibrary()
        self._min_cases = min_cases_per_skill
        self._k = skills_per_batch
        self._successes_since_consolidate: int = 0

    @property
    def library(self) -> SkillLibrary:
        return self._library

    def record_success(self) -> None:
        self._successes_since_consolidate += 1

    def should_consolidate(self, every_n: int = 10) -> bool:
        return self._successes_since_consolidate >= every_n

    async def consolidate(self, cases: list[Case]) -> list[Skill]:
        """Consolidate successful cases into skills.

        Returns list of newly added/updated Skill objects.
        """
        from collections import defaultdict  # noqa: PLC0415

        # Filter to successes
        successes = [c for c in cases if c.outcome.status.value == "success"]
        if len(successes) < self._min_cases:
            return []

        by_domain: dict[str, list[Case]] = defaultdict(list)
        for c in successes:
            by_domain[c.domain.value].append(c)

        new_skills: list[Skill] = []
        for domain, domain_cases in by_domain.items():
            if len(domain_cases) < self._min_cases:
                continue
            batch = domain_cases[-12:]  # Most recent successes
            skills = await self._consolidate_batch(batch, domain)
            for skill in skills:
                self._library.update(skill)
                new_skills.append(skill)

        self._successes_since_consolidate = 0
        return new_skills

    async def _consolidate_batch(self, cases: list[Case], domain: str) -> list[Skill]:
        """Run LLM consolidation on a batch of cases."""
        from ..types import Message, MessageRole  # noqa: PLC0415

        traj_blocks = []
        for i, c in enumerate(cases, 1):
            steps_text = self._summarize_trajectory(c)
            traj_blocks.append(
                f"--- SUCCESS {i} ---\n"
                f"Task: {c.task[:150]}\n"
                f"Reward: {c.outcome.reward:.2f}, Steps: {len(c.trajectory.steps)}\n"
                f"Steps:\n{steps_text}"
            )

        k = min(self._k, max(1, len(cases) // 4))
        user_msg = _CONSOLIDATE_USER.format(
            n=len(cases),
            domain=domain,
            k=k,
            trajectories="\n\n".join(traj_blocks)[:4000],
        )

        try:
            resp = await self._llm.complete(
                messages=[Message(role=MessageRole.USER, content=user_msg)],
                system=_CONSOLIDATE_SYSTEM,
            )
            raw = resp.content.strip()
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return []
            data = json.loads(raw[start:end])

            # Compute stats from source cases
            avg_steps = sum(len(c.trajectory.steps) for c in cases) / max(1, len(cases))
            avg_reward = sum(c.outcome.reward for c in cases) / max(1, len(cases))
            success_rate = sum(1 for c in cases if c.outcome.status.value == "success") / max(1, len(cases))

            skills = []
            for item in data:
                skill_id = hashlib.md5(
                    (domain + item.get("name", "") + item.get("trigger", "")).encode()
                ).hexdigest()[:12]
                skills.append(Skill(
                    id=skill_id,
                    name=item.get("name", "unnamed_skill"),
                    domain=domain,
                    trigger=item.get("trigger", ""),
                    strategy=item.get("strategy", ""),
                    key_tools=item.get("key_tools", []),
                    avg_steps=avg_steps,
                    avg_reward=avg_reward,
                    success_rate=success_rate,
                    support=len(cases),
                    source_case_ids=[c.id for c in cases],
                ))
            return skills
        except (json.JSONDecodeError, KeyError, Exception):
            return []

    def _summarize_trajectory(self, case: Case) -> str:
        lines = []
        for step in case.trajectory.steps[:5]:
            if step.tool_call:
                args_preview = str(step.tool_call.arguments)[:60]
                lines.append(f"  [{step.index}] → {step.tool_call.name}({args_preview})")
            else:
                lines.append(f"  [{step.index}] final answer")
        return "\n".join(lines)
