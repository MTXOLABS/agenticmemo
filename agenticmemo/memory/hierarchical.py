"""Hierarchical Memory — H-MEM style 4-layer case organisation.

Layers (coarse → fine):
  1. Domain      — e.g. coding, math, research, web
  2. Category    — e.g. "debugging", "algebra", "literature_review"
  3. Trace       — problem-pattern cluster within a category
  4. Episode     — individual Case objects

Retrieval proceeds top-down: prune search space at each layer before
descending, giving effectively O(log N) retrieval instead of O(N).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterator

from ..config import MemoryConfig
from ..types import MemoryDomain
from .base import MemoryBackend
from .case import Case
from .graph_memory import TemporalGraphMemory


# ---------------------------------------------------------------------------
# Domain auto-classifier (keyword-based, zero-cost)
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[MemoryDomain, list[str]] = {
    MemoryDomain.CODING: [
        "code", "function", "bug", "debug", "python", "javascript", "class",
        "error", "compile", "runtime", "import", "library", "implement",
    ],
    MemoryDomain.MATH: [
        "calculate", "equation", "solve", "integral", "derivative", "matrix",
        "probability", "statistics", "proof", "theorem", "algebra", "geometry",
    ],
    MemoryDomain.RESEARCH: [
        "research", "paper", "study", "literature", "review", "analysis",
        "survey", "findings", "hypothesis", "experiment", "methodology",
    ],
    MemoryDomain.WEB: [
        "search", "website", "url", "browse", "scrape", "http", "web",
        "page", "crawl", "fetch", "download", "internet",
    ],
    MemoryDomain.DATA: [
        "data", "dataset", "csv", "json", "database", "sql", "query",
        "transform", "pipeline", "etl", "schema", "table", "pandas",
    ],
    MemoryDomain.REASONING: [
        "reason", "logic", "argue", "conclude", "infer", "deduce", "think",
        "chain", "step-by-step", "explain", "why", "because",
    ],
    MemoryDomain.FINANCE: [
        "stock", "equity", "bond", "portfolio", "investment", "return", "yield",
        "dividend", "valuation", "dcf", "eps", "pe", "ratio", "revenue",
        "profit", "loss", "ebitda", "cash flow", "balance sheet", "income",
        "interest rate", "inflation", "risk", "volatility", "hedge", "option",
        "derivative", "forex", "currency", "crypto", "bitcoin", "market cap",
        "financial", "fund", "etf", "ipo", "earnings", "fiscal", "quarter",
        "asset", "liability", "shareholder", "capital", "liquidity", "credit",
    ],
    MemoryDomain.REAL_ESTATE: [
        "property", "real estate", "house", "apartment", "condo", "rent",
        "mortgage", "lease", "tenant", "landlord", "listing", "mls", "zoning",
        "appraisal", "valuation", "cap rate", "noi", "roi", "cash flow",
        "vacancy", "occupancy", "sqft", "square feet", "bedroom", "bathroom",
        "commercial", "residential", "industrial", "land", "acre", "lot",
        "building", "development", "renovation", "flip", "reit", "closing",
        "down payment", "lender", "broker", "agent", "neighborhood", "location",
        "price per sqft", "comparable", "comps", "market", "listing price",
    ],
}


def classify_domain(task: str) -> MemoryDomain:
    """Classify a task into a MemoryDomain using keyword matching."""
    task_lower = task.lower()
    scores: dict[MemoryDomain, int] = defaultdict(int)
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", task_lower):
                scores[domain] += 1
    if not scores:
        return MemoryDomain.GENERAL
    return max(scores, key=lambda d: scores[d])


def extract_keywords(task: str, max_kw: int = 10) -> list[str]:
    """Extract simple keywords from a task string."""
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "of", "for", "with", "this", "that", "it", "be", "do",
            "how", "what", "why", "when", "where", "which"}
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{2,}\b", task)
    seen: set[str] = set()
    keywords = []
    for t in tokens:
        tl = t.lower()
        if tl not in stop and tl not in seen:
            seen.add(tl)
            keywords.append(tl)
        if len(keywords) >= max_kw:
            break
    return keywords


# ---------------------------------------------------------------------------
# HierarchicalMemory
# ---------------------------------------------------------------------------


class HierarchicalMemory(MemoryBackend):
    """4-layer hierarchical wrapper around TemporalGraphMemory.

    Internally delegates all storage to a TemporalGraphMemory instance
    while maintaining index structures for layer-by-layer pruning.
    """

    def __init__(self, cfg: MemoryConfig | None = None) -> None:
        cfg = cfg or MemoryConfig()
        self._graph = TemporalGraphMemory(
            max_cases=cfg.max_cases,
            temporal_decay_rate=cfg.temporal_decay_rate,
            max_edges_per_node=cfg.max_edges_per_node,
            persist_path=cfg.persist_path,
        )
        self._auto_classify = cfg.domain_auto_classify

        # Layer indices
        # domain  →  category  →  trace_id  →  [case_ids]
        self._index: dict[MemoryDomain, dict[str, dict[str, list[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

    # ------------------------------------------------------------------ #
    # MemoryBackend interface
    # ------------------------------------------------------------------ #

    async def store(self, case: Case) -> None:
        # Auto-classify if needed
        if self._auto_classify and case.domain == MemoryDomain.GENERAL:
            case.domain = classify_domain(case.task)
        if not case.category:
            case.category = case.domain.value
        if not case.keywords:
            case.keywords = extract_keywords(case.task)

        # Derive a coarse trace key from the first 2 keywords
        trace_key = "_".join(case.keywords[:2]) if case.keywords else "default"

        await self._graph.store(case)
        self._index[case.domain][case.category][trace_key].append(case.id)

    async def get(self, case_id: str) -> Case | None:
        return await self._graph.get(case_id)

    async def delete(self, case_id: str) -> bool:
        case = await self._graph.get(case_id)
        if not case:
            return False
        trace_key = "_".join(case.keywords[:2]) if case.keywords else "default"
        try:
            self._index[case.domain][case.category][trace_key].remove(case_id)
        except ValueError:
            pass
        return await self._graph.delete(case_id)

    async def all_cases(self) -> list[Case]:
        return await self._graph.all_cases()

    async def size(self) -> int:
        return await self._graph.size()

    # ------------------------------------------------------------------ #
    # Hierarchical retrieval helpers
    # ------------------------------------------------------------------ #

    async def search_by_domain(self, domain: MemoryDomain) -> list[Case]:
        ids: list[str] = []
        for cat_dict in self._index[domain].values():
            for trace_list in cat_dict.values():
                ids.extend(trace_list)
        cases = [await self._graph.get(i) for i in ids]
        return [c for c in cases if c is not None]

    async def search_by_category(self, domain: MemoryDomain, category: str) -> list[Case]:
        ids: list[str] = []
        for trace_list in self._index[domain][category].values():
            ids.extend(trace_list)
        cases = [await self._graph.get(i) for i in ids]
        return [c for c in cases if c is not None]

    def iter_domains(self) -> Iterator[MemoryDomain]:
        yield from self._index.keys()

    def iter_categories(self, domain: MemoryDomain) -> Iterator[str]:
        yield from self._index[domain].keys()

    # ------------------------------------------------------------------ #
    # Expose graph for retrieval engine
    # ------------------------------------------------------------------ #

    @property
    def graph(self) -> TemporalGraphMemory:
        return self._graph
