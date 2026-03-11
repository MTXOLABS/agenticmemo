"""BM25 keyword index for lexical retrieval."""

from __future__ import annotations

import re

from ..memory.case import Case


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())


class BM25Index:
    """Lightweight BM25 index over Case tasks.

    Provides lexical retrieval that complements semantic embeddings —
    crucial for entity-heavy queries (names, error codes, URLs).
    """

    def __init__(self) -> None:
        self._case_ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25 = None
        self._dirty = True

    def add(self, case: Case) -> None:
        tokens = _tokenize(case.task + " " + " ".join(case.keywords))
        self._case_ids.append(case.id)
        self._corpus_tokens.append(tokens)
        self._dirty = True

    def remove(self, case_id: str) -> None:
        if case_id not in self._case_ids:
            return
        idx = self._case_ids.index(case_id)
        self._case_ids.pop(idx)
        self._corpus_tokens.pop(idx)
        self._dirty = True

    def _build(self) -> None:
        if not self._dirty:
            return
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("rank-bm25 not installed. Run: pip install rank-bm25") from e
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        self._dirty = False

    def scores(self, query: str) -> dict[str, float]:
        """Return a {case_id: bm25_score} dict for all indexed cases."""
        if not self._case_ids:
            return {}
        self._build()
        tokens = _tokenize(query)
        if not tokens or self._bm25 is None:
            return {cid: 0.0 for cid in self._case_ids}
        raw = self._bm25.get_scores(tokens)
        max_score = float(raw.max()) if len(raw) > 0 else 1.0
        if max_score == 0:
            max_score = 1.0
        return {cid: float(s) / max_score for cid, s in zip(self._case_ids, raw)}

    def __len__(self) -> int:
        return len(self._case_ids)
