"""Temporal Knowledge Graph Memory.

Replaces the flat Case Bank with a NetworkX-based directed graph where:
- Each node is a Case
- Edges encode relationships (similar task, sequential, contradicts)
- Nodes carry temporal metadata for staleness-aware retrieval
- Hybrid retrieval: semantic + keyword + graph proximity + temporal weight
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable
from pathlib import Path
from threading import RLock
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
        self._pagerank_cache: dict[str, float] | None = None
        self._persist_path = Path(persist_path) if persist_path else None
        # SQLite backend (path ending in .db): incremental single-row writes
        # instead of rewriting the whole JSON on every store — O(1) vs O(N).
        self._sqlite = None
        self._mutation_lock = asyncio.Lock()
        self._state_lock = RLock()
        if self._persist_path and self._persist_path.suffix == ".db":
            self._sqlite = self._open_sqlite(self._persist_path)
        if self._persist_path:
            if self._sqlite is not None:
                self._load_sqlite()
            elif self._persist_path.exists():
                self._load()

    # ------------------------------------------------------------------ #
    # MemoryBackend interface
    # ------------------------------------------------------------------ #

    async def store(self, case: Case) -> None:
        async with self._mutation_lock:
            if case.id not in self._cases and len(self._cases) >= self.max_cases:
                await self._evict_oldest()

            with self._state_lock:
                self._cases[case.id] = case
                self._graph.add_node(case.id, **self._node_attrs(case))
                self._pagerank_cache = None

            if self._sqlite is not None:
                await self._run_persistence(self._save_sqlite_case, case)
            elif self._persist_path:
                await self._run_persistence(self._save)

    async def get(self, case_id: str) -> Case | None:
        with self._state_lock:
            case = self._cases.get(case_id)
            if case:
                case.touch()
            return case

    async def delete(self, case_id: str) -> bool:
        async with self._mutation_lock:
            return await self._delete_case(case_id)

    async def _delete_case(self, case_id: str) -> bool:
        with self._state_lock:
            if case_id not in self._cases:
                return False
            del self._cases[case_id]
            self._graph.remove_node(case_id)
            self._pagerank_cache = None
        if self._sqlite is not None:
            await self._run_persistence(self._delete_sqlite_case, case_id)
        elif self._persist_path:
            await self._run_persistence(self._save)
        return True

    async def clear(self) -> None:
        async with self._mutation_lock:
            with self._state_lock:
                self._cases.clear()
                self._graph.clear()
                self._pagerank_cache = None
            if self._sqlite is not None:
                await self._run_persistence(self._clear_sqlite)
            elif self._persist_path:
                await self._run_persistence(self._save)

    async def all_cases(self) -> list[Case]:
        with self._state_lock:
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
        with self._state_lock:
            if src_id not in self._graph or dst_id not in self._graph:
                return
            # Limit degree, without preventing an existing relationship update.
            if (
                not self._graph.has_edge(src_id, dst_id)
                and self._graph.out_degree(src_id) >= self.max_edges
            ):
                return
            self._graph.add_edge(src_id, dst_id, type=edge_type, weight=weight)
            self._pagerank_cache = None
            if self._sqlite is not None:
                self._sqlite.execute(
                    "INSERT OR REPLACE INTO edges (src, dst, type, weight) VALUES (?, ?, ?, ?)",
                    (src_id, dst_id, edge_type, weight),
                )
                self._sqlite.commit()
            elif self._persist_path:
                # Preserve the synchronous public edge API, matching SQLite's
                # existing immediate durability behavior.
                self._save()

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
        """PageRank over the case graph — highly referenced cases score higher.

        Cached until the graph mutates: retrieval runs on every agent step
        (plus DMER refreshes), and recomputing PageRank each time was the
        hot-path cost at scale (Phase 4.3).
        """
        if len(self._graph) == 0:
            return {}
        if self._pagerank_cache is None:
            self._pagerank_cache = nx.pagerank(self._graph, weight="weight")
        return self._pagerank_cache

    # ------------------------------------------------------------------ #
    # Domain filtering
    # ------------------------------------------------------------------ #

    def cases_by_domain(self, domain: MemoryDomain) -> list[Case]:
        return [c for c in self._cases.values() if c.domain == domain]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    @staticmethod
    def _open_sqlite(path: Path):
        import sqlite3  # noqa: PLC0415

        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, data TEXT)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS edges "
            "(src TEXT, dst TEXT, type TEXT, weight REAL, PRIMARY KEY (src, dst))"
        )
        conn.commit()
        return conn

    def _load_sqlite(self) -> None:
        try:
            for (data,) in self._sqlite.execute("SELECT data FROM cases"):
                case = Case.model_validate(json.loads(data))
                self._cases[case.id] = case
                self._graph.add_node(case.id, **self._node_attrs(case))
            for src, dst, etype, weight in self._sqlite.execute(
                "SELECT src, dst, type, weight FROM edges"
            ):
                if src in self._graph and dst in self._graph:
                    self._graph.add_edge(src, dst, type=etype, weight=weight)
        except Exception as e:
            raise MemoryError(
                f"Failed to load memory from {self._persist_path}: {e}"
            ) from e

    def _save(self) -> None:
        if not self._persist_path:
            return
        with self._state_lock:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "cases": [c.model_dump(mode="json") for c in self._cases.values()],
                "edges": [
                    {"src": u, "dst": v, **d}
                    for u, v, d in self._graph.edges(data=True)
                ],
            }
            # Atomic write: a crash mid-write must not corrupt the memory file.
            tmp = self._persist_path.with_suffix(self._persist_path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, default=str))
            tmp.replace(self._persist_path)

    @staticmethod
    async def _run_persistence(operation: Callable[..., None], *args: Any) -> None:
        """Keep mutation ownership until a worker settles, even after cancellation."""
        worker = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError as cancellation:
            try:
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                worker.result()
            except Exception:
                raise cancellation from None
            raise

    def _save_sqlite_case(self, case: Case) -> None:
        with self._state_lock:
            self._sqlite.execute(
                "INSERT OR REPLACE INTO cases (id, data) VALUES (?, ?)",
                (case.id, json.dumps(case.model_dump(mode="json"), default=str)),
            )
            self._sqlite.commit()

    def _delete_sqlite_case(self, case_id: str) -> None:
        with self._state_lock:
            self._sqlite.execute("DELETE FROM cases WHERE id = ?", (case_id,))
            self._sqlite.execute(
                "DELETE FROM edges WHERE src = ? OR dst = ?", (case_id, case_id)
            )
            self._sqlite.commit()

    def _clear_sqlite(self) -> None:
        with self._state_lock:
            self._sqlite.execute("DELETE FROM cases")
            self._sqlite.execute("DELETE FROM edges")
            self._sqlite.commit()

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text())  # type: ignore[union-attr]
            for case_data in data.get("cases", []):
                case = Case.model_validate(case_data)
                self._cases[case.id] = case
                self._graph.add_node(case.id, **self._node_attrs(case))
            for edge in data.get("edges", []):
                src, dst = edge.pop("src"), edge.pop("dst")
                if src in self._cases and dst in self._cases:
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
        """Quality-aware eviction (Phase 4.2).

        Evict lowest-value first: failures before successes, low reward
        before high, then least-accessed/oldest. Solution-bearing successes
        are the most valuable memory (they power exemplar injection) and are
        only evicted when nothing else is left.
        """
        if not self._cases:
            return
        victim = min(
            self._cases.values(),
            key=lambda c: (
                c.is_success and bool(c.solution),   # protected class last
                c.outcome.reward,
                c.access_count,
                c.created_at,
            ),
        )
        await self._delete_case(victim.id)
