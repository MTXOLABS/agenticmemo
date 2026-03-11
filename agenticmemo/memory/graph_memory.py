"""Temporal Knowledge Graph Memory.

Replaces the flat Case Bank with a NetworkX-based directed graph where:
- Each node is a Case
- Edges encode relationships (similar task, sequential, contradicts)
- Nodes carry temporal metadata for staleness-aware retrieval
- Hybrid retrieval: semantic + keyword + graph proximity + temporal weight
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

from ..exceptions import MemoryError
from ..types import MemoryDomain
from .base import MemoryBackend
from .case import Case


class EdgeType:
    SIMILAR = "similar"        # semantically similar tasks
    SEQUENTIAL = "sequential"  # B was attempted after A in the same session
    CONTRADICTS = "contradicts"  # B's approach contradicts A's
    REFINES = "refines"        # B is a better version of A's solution


class TemporalGraphMemory(MemoryBackend):
    """In-memory temporal knowledge graph for Case storage.

    Key advantages over flat storage:
    - Graph-proximity scores surface structurally related cases even when
      embedding similarity is low.
    - Temporal decay: stale cases are down-weighted, not deleted.
    - Relationship edges let the planner trace chains of reasoning.
    - Optionally persists to disk as JSON.
    """

    def __init__(
        self,
        max_cases: int = 10_000,
        temporal_decay_rate: float = 0.01,
        max_edges_per_node: int = 20,
        persist_path: str | None = None,
    ) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._cases: dict[str, Case] = {}
        self.max_cases = max_cases
        self.decay_rate = temporal_decay_rate
        self.max_edges = max_edges_per_node
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    # ------------------------------------------------------------------ #
    # MemoryBackend interface
    # ------------------------------------------------------------------ #

    async def store(self, case: Case) -> None:
        if len(self._cases) >= self.max_cases:
            await self._evict_oldest()

        self._cases[case.id] = case
        self._graph.add_node(case.id, **self._node_attrs(case))

        if self._persist_path:
            self._save()

    async def get(self, case_id: str) -> Case | None:
        case = self._cases.get(case_id)
        if case:
            case.touch()
        return case

    async def delete(self, case_id: str) -> bool:
        if case_id not in self._cases:
            return False
        del self._cases[case_id]
        self._graph.remove_node(case_id)
        return True

    async def all_cases(self) -> list[Case]:
        return list(self._cases.values())

    async def size(self) -> int:
        return len(self._cases)

    # ------------------------------------------------------------------ #
    # Graph-specific operations
    # ------------------------------------------------------------------ #

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        edge_type: str = EdgeType.SIMILAR,
        weight: float = 1.0,
    ) -> None:
        if src_id not in self._graph or dst_id not in self._graph:
            return
        # Limit degree
        if self._graph.out_degree(src_id) >= self.max_edges:
            return
        self._graph.add_edge(src_id, dst_id, type=edge_type, weight=weight)

    def neighbors(self, case_id: str, edge_type: str | None = None) -> list[Case]:
        """Return cases connected to the given case (1-hop)."""
        result = []
        for nbr_id in self._graph.successors(case_id):
            edge_data = self._graph.get_edge_data(case_id, nbr_id, default={})
            if edge_type is None or edge_data.get("type") == edge_type:
                case = self._cases.get(nbr_id)
                if case:
                    result.append(case)
        return result

    def graph_proximity_score(self, case_id: str, candidate_id: str) -> float:
        """Estimate proximity between two nodes via shortest path length."""
        try:
            path_len = nx.shortest_path_length(self._graph, case_id, candidate_id)
            return 1.0 / (1.0 + path_len)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0.0

    def temporal_weight(self, case: Case) -> float:
        """Exponential decay based on case age. Recent cases score higher."""
        return math.exp(-self.decay_rate * case.age_days)

    def pagerank_scores(self) -> dict[str, float]:
        """PageRank over the case graph — highly referenced cases score higher."""
        if len(self._graph) == 0:
            return {}
        return nx.pagerank(self._graph, weight="weight")

    # ------------------------------------------------------------------ #
    # Domain filtering
    # ------------------------------------------------------------------ #

    def cases_by_domain(self, domain: MemoryDomain) -> list[Case]:
        return [c for c in self._cases.values() if c.domain == domain]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cases": [c.model_dump(mode="json") for c in self._cases.values()],
            "edges": [
                {"src": u, "dst": v, **d}
                for u, v, d in self._graph.edges(data=True)
            ],
        }
        self._persist_path.write_text(json.dumps(data, default=str))

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text())  # type: ignore[union-attr]
            for case_data in data.get("cases", []):
                case = Case.model_validate(case_data)
                self._cases[case.id] = case
                self._graph.add_node(case.id, **self._node_attrs(case))
            for edge in data.get("edges", []):
                src, dst = edge.pop("src"), edge.pop("dst")
                self._graph.add_edge(src, dst, **edge)
        except Exception as e:
            raise MemoryError(f"Failed to load memory from {self._persist_path}: {e}") from e

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _node_attrs(case: Case) -> dict[str, Any]:
        return {
            "domain": case.domain.value,
            "reward": case.outcome.reward,
            "status": case.outcome.status.value,
            "created_at": case.created_at.isoformat(),
        }

    async def _evict_oldest(self) -> None:
        """Remove the oldest, least-accessed case when at capacity."""
        if not self._cases:
            return
        oldest = min(
            self._cases.values(),
            key=lambda c: (c.access_count, c.created_at),
        )
        await self.delete(oldest.id)
