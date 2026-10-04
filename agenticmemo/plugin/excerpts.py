"""Bounded lexical excerpts for long reference documents."""

from __future__ import annotations

import re
from collections import deque

_WORDS = re.compile(r"\w+", re.UNICODE)
_BOUNDARY = re.compile(r"(?:[.!?]\s+|\n+)")


def select_excerpt(content: str, query_terms: set[str], limit: int = 1600) -> str:
    """Keep the compact passage covering the most distinct query terms.

    A sliding window caps the distance between matches at 400 characters. Repeated
    terms contribute once, so a repeated heading cannot outweigh a specific clause.
    Ties favor the shortest matching span and then its earliest occurrence. Matching
    and window maintenance are linear in document length, including repeated terms.

    This remains lexical selection, not answer extraction. With no matching terms,
    the document prefix is retained. Context-budget enforcement belongs to the caller.
    """
    if limit <= 0:
        return ""
    if len(content) <= limit:
        return content
    terms = {term.casefold() for term in query_terms}
    if not terms:
        return content[:limit]

    window: deque[tuple[str, int, int]] = deque()
    counts: dict[str, int] = {}
    best_score = (0, 0)
    anchor = 0
    width = min(400, limit)

    def discard_first() -> None:
        term, _, _ = window.popleft()
        counts[term] -= 1
        if not counts[term]:
            del counts[term]

    for match in _WORDS.finditer(content):
        term = match.group().casefold()
        if term not in terms:
            continue
        window.append((term, match.start(), match.end()))
        counts[term] = counts.get(term, 0) + 1
        while len(window) > 1 and match.end() - window[0][1] > width:
            discard_first()
        # Discard repeated leading terms without changing this window's coverage.
        while counts[window[0][0]] > 1:
            discard_first()
        score = (len(counts), -(match.end() - window[0][1]))
        if score > best_score:
            best_score = score
            anchor = window[0][1]

    if not best_score[0]:
        return content[:limit]

    # Keep a short lead-in and prefer a sentence boundary when one is nearby.
    # The selected clause must appear early enough to survive later budget fitting.
    start = max(0, anchor - min(80, limit // 4))
    for boundary in _BOUNDARY.finditer(content, start, anchor):
        start = boundary.end()
    prefix = "…" if start else ""
    return prefix + content[start:start + limit - len(prefix)]
