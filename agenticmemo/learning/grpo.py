"""GRPO — Group Relative Policy Optimisation for Case Retrieval.

Replaces soft Q-learning (original Memento) with a preference-based policy
that learns *which combination of cases* is most useful, not just individual
case scores.

Algorithm:
  For each query q, sample G candidate case sets {S_1, ..., S_G}.
  Execute (or simulate) each set → collect rewards {r_1, ..., r_G}.
  Compute group-relative advantage: A_i = (r_i - mean(r)) / (std(r) + ε)
  Update case Q-values: q_i += lr * A_i  for cases in winning sets

Key insight: learning relative preferences is more stable than learning
absolute Q-values, and it requires no separate critic model.

Reference: arXiv 2510.08191 (Training-Free GRPO) and DeepSeek-R1.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from ..config import RetrievalConfig
from ..memory.case import Case


@dataclass
class GRPOState:
    """Mutable policy state persisted across updates."""
    step: int = 0
    total_updates: int = 0
    avg_reward: float = 0.0
    reward_history: list[float] = field(default_factory=list)


class GRPOPolicy:
    """Group Relative Policy Optimisation for case-set retrieval.

    The policy is implemented as a thin layer on top of the ensemble scores:
    it learns per-case Q-values (additive bonuses) that shift the raw
    ensemble ranking toward combinations that historically produced better
    outcomes.

    This is intentionally lightweight: no neural network, no backprop.
    The Q-values are stored directly on Case objects and updated online.
    """

    def __init__(self, cfg: RetrievalConfig | None = None) -> None:
        self._cfg = cfg or RetrievalConfig()
        self._state = GRPOState()
        self._pending: list[tuple[list[str], float]] = []  # (case_ids, reward)

    # ------------------------------------------------------------------ #
    # Policy application
    # ------------------------------------------------------------------ #

    def rerank(
        self,
        scored: list[tuple[Case, float]],
        temperature: float = 0.1,
    ) -> list[tuple[Case, float]]:
        """Add Q-value bonus to ensemble scores and re-sort.

        Args:
            scored:      List of (case, ensemble_score) from EnsembleRetriever.
            temperature: Controls Q-value influence (lower = more exploitation).
        """
        if not self._cfg.enable_grpo or not scored:
            return scored

        reranked = []
        for case, base_score in scored:
            bonus = math.tanh(case.q_value) * temperature
            reranked.append((case, base_score + bonus))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def sample_candidate_sets(
        self,
        pool: list[tuple[Case, float]],
        top_k: int,
        n_samples: int | None = None,
    ) -> list[list[tuple[Case, float]]]:
        """Sample G diverse candidate sets from the ranked pool.

        Each set is a top-k selection with slight perturbation so the policy
        can explore different combinations and learn relative preferences.
        """
        n_samples = n_samples or self._cfg.grpo_sample_size
        if len(pool) <= top_k:
            return [pool] * n_samples

        sets = []
        for _ in range(n_samples):
            # Soft-sample: weight by score with temperature
            weights = [max(s, 1e-6) for _, s in pool]
            selected = random.choices(pool, weights=weights, k=top_k)
            sets.append(selected)
        return sets

    # ------------------------------------------------------------------ #
    # Policy update
    # ------------------------------------------------------------------ #

    def record_outcome(self, case_ids: list[str], reward: float) -> None:
        """Record which cases were used and what reward was achieved."""
        self._pending.append((case_ids, reward))
        self._state.step += 1
        self._state.reward_history.append(reward)
        if len(self._state.reward_history) > 1000:
            self._state.reward_history.pop(0)

    def update(self, cases: dict[str, Case]) -> int:
        """Run GRPO update on accumulated pending outcomes.

        Returns number of cases updated.
        """
        cfg = self._cfg
        if not cfg.enable_grpo:
            return 0
        if self._state.step % cfg.grpo_update_every != 0:
            return 0
        if not self._pending:
            return 0

        rewards = [r for _, r in self._pending]
        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards)) + 1e-8

        updated = set()
        for case_ids, reward in self._pending:
            advantage = (reward - mean_r) / std_r
            for cid in case_ids:
                case = cases.get(cid)
                if case:
                    case.q_value += cfg.grpo_lr * advantage
                    # Clip to avoid runaway values
                    case.q_value = max(-5.0, min(5.0, case.q_value))
                    updated.add(cid)

        self._pending.clear()
        self._state.total_updates += 1
        self._state.avg_reward = mean_r
        return len(updated)

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> GRPOState:
        return self._state

    def stats(self) -> dict[str, float]:
        h = self._state.reward_history
        return {
            "total_updates": self._state.total_updates,
            "avg_reward": float(np.mean(h)) if h else 0.0,
            "reward_std": float(np.std(h)) if len(h) > 1 else 0.0,
            "pending_outcomes": len(self._pending),
        }
