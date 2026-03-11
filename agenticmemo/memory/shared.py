"""Shared Memory Pool — Phase 4: Multi-agent memory sharing.

In multi-agent systems, agents can operate with:
  - Private memory: role-specific cases only this agent sees.
  - Shared pool:    global cases all agents read from and write to.
  - Consensus layer: high-confidence cases propagated across all agents.

Architecture:
    SharedMemoryPool
    ├── global_memory: HierarchicalMemory  (all agents read/write)
    ├── agent_memories: dict[agent_id, HierarchicalMemory]  (private per agent)
    └── consensus_threshold: float  (min reward to promote to consensus)

Cross-agent deduplication runs automatically before storing in the shared pool,
preventing the same solution from being indexed multiple times under slight
rephrasing (which inflates retrieval noise).

Reference: Intrinsic Memory Agents (arXiv 2508.08997)
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ..config import MemoryConfig, RetrievalConfig
from ..types import MemoryDomain
from .case import Case
from .hierarchical import HierarchicalMemory


class SharedMemoryPool:
    """Multi-agent shared + private memory manager.

    Usage::

        pool = SharedMemoryPool()
        pool.register_agent("planner")
        pool.register_agent("executor")

        # Store to shared pool (with auto-dedup)
        await pool.store_shared(case)

        # Store to a specific agent's private memory
        await pool.store_private("planner", case)

        # Retrieve from merged view (private + shared)
        cases = await pool.all_cases_for("planner")
    """

    def __init__(
        self,
        shared_cfg: MemoryConfig | None = None,
        consensus_threshold: float = 0.8,
        dedup_similarity_threshold: float = 0.92,
    ) -> None:
        self._shared = HierarchicalMemory(shared_cfg or MemoryConfig())
        self._agent_memories: dict[str, HierarchicalMemory] = {}
        self._consensus: dict[str, Case] = {}           # case_id → Case
        self.consensus_threshold = consensus_threshold
        self.dedup_threshold = dedup_similarity_threshold
        self._fingerprints: dict[str, str] = {}         # fingerprint → case_id

    # ------------------------------------------------------------------ #
    # Agent registration
    # ------------------------------------------------------------------ #

    def register_agent(
        self, agent_id: str, cfg: MemoryConfig | None = None
    ) -> "SharedMemoryPool":
        """Register a new agent with its own private memory."""
        if agent_id not in self._agent_memories:
            self._agent_memories[agent_id] = HierarchicalMemory(cfg or MemoryConfig())
        return self

    def agent_ids(self) -> list[str]:
        return list(self._agent_memories.keys())

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #

    async def store_shared(self, case: Case) -> bool:
        """Store a case in the shared pool.

        Returns True if stored, False if duplicate was detected and skipped.
        """
        fp = self._fingerprint(case)
        if fp in self._fingerprints:
            return False  # duplicate detected
        self._fingerprints[fp] = case.id
        await self._shared.store(case)

        # Promote to consensus if reward is high enough
        if case.outcome.reward >= self.consensus_threshold:
            self._consensus[case.id] = case
        return True

    async def store_private(self, agent_id: str, case: Case) -> None:
        """Store a case in an agent's private memory."""
        mem = self._agent_memories.get(agent_id)
        if not mem:
            raise KeyError(f"Agent '{agent_id}' not registered. Call register_agent() first.")
        await mem.store(case)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    async def all_cases_for(self, agent_id: str) -> list[Case]:
        """Return merged view: agent's private cases + all shared cases."""
        private = self._agent_memories.get(agent_id)
        private_cases = await private.all_cases() if private else []
        shared_cases = await self._shared.all_cases()

        # Merge, dedup by id
        seen: set[str] = set()
        merged = []
        for c in private_cases + shared_cases:
            if c.id not in seen:
                seen.add(c.id)
                merged.append(c)
        return merged

    async def shared_cases(self) -> list[Case]:
        return await self._shared.all_cases()

    def consensus_cases(self) -> list[Case]:
        """Return only high-reward consensus cases."""
        return list(self._consensus.values())

    async def shared_size(self) -> int:
        return await self._shared.size()

    async def private_size(self, agent_id: str) -> int:
        mem = self._agent_memories.get(agent_id)
        return await mem.size() if mem else 0

    # ------------------------------------------------------------------ #
    # Cross-agent deduplication
    # ------------------------------------------------------------------ #

    async def dedup_shared(self) -> int:
        """Scan shared pool and remove near-duplicate cases.

        Two cases are considered duplicates if:
        - Their task fingerprints match (fast path), OR
        - Their answer texts are > dedup_threshold similar (slow path)

        Returns number of cases removed.
        """
        cases = await self._shared.all_cases()
        removed = 0
        seen_fingerprints: set[str] = set()
        to_remove: list[str] = []

        for case in cases:
            fp = self._fingerprint(case)
            if fp in seen_fingerprints:
                to_remove.append(case.id)
            else:
                seen_fingerprints.add(fp)

        for cid in to_remove:
            await self._shared.delete(cid)
            removed += 1

        return removed

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fingerprint(case: Case) -> str:
        """Fast content fingerprint for near-duplicate detection."""
        # Normalise task + answer → hash
        text = (case.task.lower().strip() + case.outcome.answer.lower().strip())
        # Remove whitespace variations
        text = " ".join(text.split())
        return hashlib.md5(text.encode()).hexdigest()  # noqa: S324
