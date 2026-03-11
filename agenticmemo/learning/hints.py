"""Hints Internalization — Phase 4: Beyond pure case retrieval.

Inspired by "Memento No More: Coaching AI Agents to Master Multiple Tasks
via Hints Internalization" (arXiv 2502.01562).

The idea: rather than always injecting raw cases into the prompt, the agent
can *internalize* repeated patterns from the case bank into compact, reusable
"hint policies" — distilled procedural knowledge.

Two mechanisms:
  1. HintExtractor  — scans the case bank and extracts recurring patterns as
                      natural-language hints (e.g. "always check edge cases
                      before returning").
  2. HintLibrary    — stores and manages extracted hints; serves them to the
                      planner as compact priors instead of (or alongside) cases.

This is a lighter, inference-time alternative to weight fine-tuning and
complements the episodic case bank by capturing *general* strategies that
hold across many tasks, not just specific past trajectories.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ..llm.base import LLMBackend
from ..memory.case import Case
from ..types import Message, MessageRole


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Hint(BaseModel):
    """A distilled procedural hint extracted from multiple cases."""

    id: str
    domain: str
    text: str                          # the actionable hint sentence
    source_case_ids: list[str] = Field(default_factory=list)
    support: int = 1                   # number of cases this was inferred from
    avg_reward: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HintLibrary(BaseModel):
    """Persistent store of extracted hints."""

    hints: list[Hint] = Field(default_factory=list)

    def add(self, hint: Hint) -> None:
        # Avoid exact duplicates
        existing_texts = {h.text.lower().strip() for h in self.hints}
        if hint.text.lower().strip() not in existing_texts:
            self.hints.append(hint)

    def by_domain(self, domain: str) -> list[Hint]:
        return [h for h in self.hints if h.domain == domain or h.domain == "general"]

    def top(self, n: int = 5, domain: str | None = None) -> list[Hint]:
        pool = self.by_domain(domain) if domain else self.hints
        return sorted(pool, key=lambda h: (h.support, h.avg_reward), reverse=True)[:n]

    def to_prompt_block(self, domain: str | None = None, n: int = 5) -> str:
        top = self.top(n, domain)
        if not top:
            return ""
        lines = ["--- Internalized Hints (distilled from past experience) ---"]
        for i, h in enumerate(top, 1):
            lines.append(f"{i}. {h.text}")
        lines.append("---")
        return "\n".join(lines)

    def size(self) -> int:
        return len(self.hints)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
You are analysing a batch of successful AI agent task executions to extract
reusable strategic hints.

Below are {n} successful cases with their tasks, approaches, and outcomes.

{cases_text}

Your job:
Extract 3-5 SHORT, GENERAL, ACTIONABLE hints that apply broadly to many tasks
in this domain (not just these specific tasks).

Good hints:
- "Always verify tool output before using it in the next step."
- "For coding tasks, test with edge cases (empty input, zero, None)."
- "Break complex tasks into sub-steps; complete one before starting the next."

Bad hints (too specific):
- "For task X, call function Y."
- "Use tool Z when input is a CSV file."

Respond with a JSON array of strings, e.g.:
["hint 1", "hint 2", "hint 3"]
"""


class HintExtractor:
    """Extracts generalised hints from a batch of successful cases.

    Call `extract` periodically (e.g. every 20 cases stored) to
    update the HintLibrary with newly distilled knowledge.
    """

    def __init__(self, llm: LLMBackend, library: HintLibrary | None = None) -> None:
        self._llm = llm
        self.library = library or HintLibrary()
        self._processed_ids: set[str] = set()

    async def extract(
        self,
        cases: list[Case],
        domain: str = "general",
        min_reward: float = 0.5,
        max_cases: int = 10,
    ) -> list[Hint]:
        """Extract hints from a list of cases.

        Args:
            cases:      Case pool to analyse.
            domain:     Domain label for the extracted hints.
            min_reward: Only use successful cases above this reward.
            max_cases:  Max cases to include in a single extraction call.

        Returns:
            Newly extracted Hint objects (also added to self.library).
        """
        # Filter: successful, not yet processed
        eligible = [
            c for c in cases
            if c.outcome.reward >= min_reward and c.id not in self._processed_ids
        ][:max_cases]

        if len(eligible) < 2:
            return []

        cases_text = "\n\n".join(
            f"Task: {c.task}\n"
            f"Steps: {c.trajectory.num_steps}\n"
            f"Key tools: {', '.join(c.trajectory.tool_names_used) or 'none'}\n"
            f"Answer: {c.outcome.answer[:200]}"
            for c in eligible
        )
        prompt = _EXTRACT_PROMPT.format(n=len(eligible), cases_text=cases_text)

        try:
            resp = await self._llm.complete(
                messages=[Message(role=MessageRole.USER, content=prompt)]
            )
            raw = resp.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            hint_texts: list[str] = json.loads(raw)
        except Exception:
            return []

        avg_reward = sum(c.outcome.reward for c in eligible) / len(eligible)
        new_hints = []
        for text in hint_texts:
            if not isinstance(text, str) or not text.strip():
                continue
            hint = Hint(
                id=f"hint_{domain}_{len(self.library.hints)}",
                domain=domain,
                text=text.strip(),
                source_case_ids=[c.id for c in eligible],
                support=len(eligible),
                avg_reward=avg_reward,
            )
            self.library.add(hint)
            new_hints.append(hint)

        # Mark as processed
        for c in eligible:
            self._processed_ids.add(c.id)

        return new_hints
