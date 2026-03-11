"""Causal Failure Mining (CFM) — AgenticMemo v2.

Systematically mines failed trajectories for recurring failure patterns,
builds an anti-case bank, and generates actionable failure warnings
injected into the planner's context.

Key insight: Most agent frameworks store successes and treat failures as
one-off events. CFM clusters failures by structural similarity, extracts
the ROOT CAUSE and the COUNTERFACTUAL FIX, and injects these as hard
constraints during planning — preventing the same class of failure from
recurring.

This is novel because:
1. We extract CAUSAL chains from failures (not just descriptions)
2. We cluster failures ACROSS cases (cross-episode pattern mining)
3. We inject anti-patterns as NEGATIVE examples with counterfactuals
4. Patterns self-update as new failures arrive (online mining)

Reference: inspired by the failure case bank concept in ExpeL (Zhao et al. 2024)
but extended with causal chain extraction and cross-episode clustering.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..llm.base import LLMBackend
from ..memory.case import Case
from ..types import TaskStatus


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FailurePattern:
    """A generalized failure pattern extracted from multiple failed cases.

    A pattern captures:
    - TRIGGER: what task/situation leads to this failure
    - MISTAKE: what the agent did wrong (the causal error)
    - CONSEQUENCE: what outcome it produced
    - FIX: the counterfactual correction (what to do instead)
    - EVIDENCE: how many cases support this pattern
    """
    id: str
    domain: str
    trigger: str            # "Tasks involving dynamic programming recursion"
    mistake: str            # "Agent calls python_repl without initializing memo dict"
    consequence: str        # "RecursionError / exponential time complexity"
    fix: str                # "Always declare memoization dict before recursion"
    support: int = 1        # Number of supporting failure cases
    avg_depth: float = 0.0  # Average step at which failure occurs
    source_case_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_prompt_block(self) -> str:
        """Render as a negative example for the planner."""
        return (
            f"⚠ FAILURE PATTERN (seen {self.support}x, avg at step {self.avg_depth:.1f}):\n"
            f"  TRIGGER   : {self.trigger}\n"
            f"  MISTAKE   : {self.mistake}\n"
            f"  CONSEQUENCE: {self.consequence}\n"
            f"  FIX       : {self.fix}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "trigger": self.trigger,
            "mistake": self.mistake,
            "consequence": self.consequence,
            "fix": self.fix,
            "support": self.support,
            "avg_depth": self.avg_depth,
            "source_case_ids": self.source_case_ids,
        }


class FailurePatternBank:
    """Persistent store for extracted failure patterns.

    Provides:
    - Storage and retrieval of FailurePattern objects
    - Domain-filtered lookup
    - Prompt block generation for planner injection
    - Deduplication via fingerprinting
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self._patterns: dict[str, FailurePattern] = {}   # id → pattern
        self._fingerprints: dict[str, str] = {}           # fingerprint → id
        self._domain_index: dict[str, list[str]] = defaultdict(list)  # domain → [ids]
        self._persist_path = persist_path
        if persist_path:
            self._load(persist_path)

    def add(self, pattern: FailurePattern) -> bool:
        """Add a pattern. Returns False if near-duplicate already exists."""
        fp = self._fingerprint(pattern)
        if fp in self._fingerprints:
            existing_id = self._fingerprints[fp]
            if existing_id in self._patterns:
                self._patterns[existing_id].support += pattern.support
                self._patterns[existing_id].last_updated = datetime.utcnow().isoformat()
            if self._persist_path:
                self._save()
            return False
        self._patterns[pattern.id] = pattern
        self._fingerprints[fp] = pattern.id
        self._domain_index[pattern.domain].append(pattern.id)
        if self._persist_path:
            self._save()
        return True

    def by_domain(self, domain: str, top_n: int = 5) -> list[FailurePattern]:
        """Return top-N most-supported patterns for a domain."""
        ids = self._domain_index.get(domain, []) + self._domain_index.get("general", [])
        patterns = [self._patterns[i] for i in ids if i in self._patterns]
        return sorted(patterns, key=lambda p: p.support, reverse=True)[:top_n]

    def top(self, top_n: int = 5) -> list[FailurePattern]:
        """Return top-N most-supported patterns across all domains."""
        return sorted(self._patterns.values(), key=lambda p: p.support, reverse=True)[:top_n]

    def to_prompt_block(self, domain: str | None = None, top_n: int = 4) -> str:
        """Render failure patterns as a planning constraint block."""
        if domain:
            patterns = self.by_domain(domain, top_n)
        else:
            patterns = self.top(top_n)
        if not patterns:
            return ""
        lines = ["KNOWN FAILURE PATTERNS — AVOID THESE MISTAKES:"]
        for p in patterns:
            lines.append(p.to_prompt_block())
            lines.append("")
        return "\n".join(lines)

    def size(self) -> int:
        return len(self._patterns)

    def all_patterns(self) -> list[FailurePattern]:
        return list(self._patterns.values())

    def _save(self) -> None:
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        p = Path(self._persist_path)  # type: ignore[arg-type]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([pt.to_dict() for pt in self._patterns.values()], default=str))

    def _load(self, path: str) -> None:
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        p = Path(path)
        if not p.exists():
            return
        try:
            for d in json.loads(p.read_text()):
                pt = FailurePattern(**d)
                fp = self._fingerprint(pt)
                self._patterns[pt.id] = pt
                self._fingerprints[fp] = pt.id
                self._domain_index[pt.domain].append(pt.id)
        except Exception:
            pass  # corrupt file — start fresh

    @staticmethod
    def _fingerprint(p: FailurePattern) -> str:
        """MD5 of normalized (trigger + mistake) for dedup."""
        raw = (p.trigger.lower().strip()[:80] + "|" + p.mistake.lower().strip()[:80])
        return hashlib.md5(raw.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Miner
# ─────────────────────────────────────────────────────────────────────────────

_MINE_SYSTEM = """\
You are an expert AI agent trainer specializing in failure analysis.
Your job is to analyze FAILED agent trajectories and extract generalizable failure patterns.

A failure pattern captures:
- TRIGGER: what TYPE of task/situation causes this failure (generalize beyond the specific task)
- MISTAKE: the specific wrong action or wrong reasoning the agent took
- CONSEQUENCE: what went wrong as a result
- FIX: what the agent SHOULD have done instead (the counterfactual correction)

Good patterns are GENERAL (apply to many tasks), CAUSAL (explain WHY it failed), and ACTIONABLE (give a clear fix).

Bad patterns are too specific ("For task X, the agent did Y") — they won't generalize.
"""

_MINE_USER = """\
Analyze these {n} failed trajectories and extract {k} generalizable failure patterns.

FAILED TRAJECTORIES:
{trajectories}

Return EXACTLY {k} patterns in this JSON format (an array):
[
  {{
    "trigger": "...",
    "mistake": "...",
    "consequence": "...",
    "fix": "..."
  }},
  ...
]

Only return valid JSON. No explanation text outside the JSON.
"""


class FailureMiner:
    """Extracts generalizable failure patterns from clusters of failed Cases.

    Algorithm:
    1. Group failed cases by domain
    2. For each group with >= min_cases failures, call LLM to extract patterns
    3. Store patterns in the FailurePatternBank
    4. Deduplicate via fingerprinting

    Call mine() periodically (e.g., every 10 new failures stored).
    """

    def __init__(
        self,
        llm: LLMBackend,
        bank: FailurePatternBank | None = None,
        min_cases_to_mine: int = 3,
        patterns_per_batch: int = 3,
        max_trajectory_chars: int = 600,
    ) -> None:
        self._llm = llm
        self._bank = bank or FailurePatternBank()
        self._min_cases = min_cases_to_mine
        self._k = patterns_per_batch
        self._max_chars = max_trajectory_chars
        self._failures_since_mine: int = 0

    @property
    def bank(self) -> FailurePatternBank:
        return self._bank

    def record_failure(self) -> None:
        """Call each time a failure case is stored."""
        self._failures_since_mine += 1

    def should_mine(self, every_n: int = 5) -> bool:
        return self._failures_since_mine >= every_n

    async def mine(self, cases: list[Case]) -> list[FailurePattern]:
        """Mine failure patterns from a list of Cases.

        Returns list of newly added FailurePattern objects.
        """
        # Filter to failures only
        failed = [c for c in cases if c.outcome.status in (TaskStatus.FAILURE, TaskStatus.PARTIAL)]
        if len(failed) < self._min_cases:
            return []

        # Group by domain
        by_domain: dict[str, list[Case]] = defaultdict(list)
        for c in failed:
            by_domain[c.domain.value].append(c)

        all_new_patterns: list[FailurePattern] = []

        for domain, domain_cases in by_domain.items():
            if len(domain_cases) < self._min_cases:
                # Pool into general batch
                by_domain["general"].extend(domain_cases)
                continue

            batch = domain_cases[-10:]  # Most recent failures
            patterns = await self._mine_batch(batch, domain)
            for p in patterns:
                if self._bank.add(p):
                    all_new_patterns.append(p)

        # Process any pooled general cases
        general_cases = by_domain.get("general", [])
        if len(general_cases) >= self._min_cases:
            patterns = await self._mine_batch(general_cases[-10:], "general")
            for p in patterns:
                if self._bank.add(p):
                    all_new_patterns.append(p)

        self._failures_since_mine = 0
        return all_new_patterns

    async def _mine_batch(self, cases: list[Case], domain: str) -> list[FailurePattern]:
        """Run LLM failure mining on a single domain batch."""
        from ..types import Message, MessageRole  # noqa: PLC0415

        # Build trajectory summaries (truncated)
        traj_blocks = []
        for i, c in enumerate(cases, 1):
            steps_text = self._summarize_trajectory(c)
            traj_blocks.append(
                f"--- FAILURE {i} (domain={domain}) ---\n"
                f"Task: {c.task[:200]}\n"
                f"Outcome: {c.outcome.status.value} (reward={c.outcome.reward:.2f})\n"
                f"Reflection: {c.outcome.reflection[:200] if c.outcome.reflection else 'None'}\n"
                f"Steps:\n{steps_text}"
            )

        trajectories_text = "\n\n".join(traj_blocks)
        k = min(self._k, max(1, len(cases) // 2))

        user_msg = _MINE_USER.format(
            n=len(cases),
            k=k,
            trajectories=trajectories_text[:4000],
        )

        try:
            resp = await self._llm.complete(
                messages=[Message(role=MessageRole.USER, content=user_msg)],
                system=_MINE_SYSTEM,
            )
            raw = resp.content.strip()
            # Extract JSON array
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return []
            data = json.loads(raw[start:end])
            patterns = []
            for item in data:
                avg_depth = sum(
                    len(c.trajectory.steps) / 2 for c in cases
                ) / max(1, len(cases))
                p = FailurePattern(
                    id=hashlib.md5(
                        (domain + item.get("trigger", "") + item.get("mistake", "")).encode()
                    ).hexdigest()[:12],
                    domain=domain,
                    trigger=item.get("trigger", ""),
                    mistake=item.get("mistake", ""),
                    consequence=item.get("consequence", ""),
                    fix=item.get("fix", ""),
                    support=len(cases),
                    avg_depth=avg_depth,
                    source_case_ids=[c.id for c in cases],
                )
                patterns.append(p)
            return patterns
        except (json.JSONDecodeError, KeyError, Exception):
            return []

    def _summarize_trajectory(self, case: Case) -> str:
        """Create a compact trajectory summary."""
        lines = []
        for step in case.trajectory.steps[:6]:  # First 6 steps
            if step.tool_call:
                lines.append(
                    f"  [{step.index}] {step.thought[:80]} → {step.tool_call.name}(...)\n"
                    f"       obs: {step.observation[:100]}"
                )
            else:
                lines.append(f"  [{step.index}] {step.thought[:80]}")
        if len(case.trajectory.steps) > 6:
            lines.append(f"  ... +{len(case.trajectory.steps) - 6} more steps")
        return "\n".join(lines)[:self._max_chars]
