"""Regressions for relevant clauses in long reference documents."""

import pytest

from agenticmemo import AgentMemory, KnowledgeRecord
from agenticmemo.plugin.excerpts import select_excerpt

REFUND_CLAUSE = "The product refund window is 45 calendar days after delivery. "


def handbook() -> str:
    return (
        "AcmeCorporation employee handbook. "
        + "Shipping labels and packing guidance are operational instructions. " * 80
        + REFUND_CLAUSE
        + "Warehouse safety requirements continue. " * 50
    )


def test_brand_name_in_heading_does_not_hide_refund_clause():
    excerpt = select_excerpt(handbook(), {"acmecorporation", "product", "refund", "window"})
    assert REFUND_CLAUSE.strip() in excerpt
    assert excerpt.index("45 calendar days") < 180
    assert len(excerpt) <= 1600


def test_repeated_keyword_does_not_outweigh_specific_clause():
    content = "Refund requests. " * 4000 + REFUND_CLAUSE + "Appendix. " * 20
    excerpt = select_excerpt(content, {"refund", "product", "window"})
    assert REFUND_CLAUSE.strip() in excerpt
    assert excerpt.index("45 calendar days") < 180


def test_compact_clause_wins_over_scattered_query_words():
    content = (
        "Product " + "operational " * 9 + "refund " + "operational " * 9 + "window. "
        + "Unrelated policies. " * 150 + REFUND_CLAUSE + "Appendix. " * 30
    )
    excerpt = select_excerpt(content, {"refund", "product", "window"})
    assert "45 calendar days" in excerpt
    assert excerpt.index("45 calendar days") < 180


def test_single_repeated_term_handles_maximum_reference_length():
    content = ("refund " * 14_300)[:100_000]
    assert select_excerpt(content, {"refund"}) == content[:1600]


def test_unicode_terms_preserve_clause_and_casefold_matching():
    clause = "返品 期間 は 45日 です。"
    content = "会社概要。\n" + "倉庫の案内。\n" * 400 + clause + "\nその他。" * 400
    excerpt = select_excerpt(content, {"返品", "期間"})
    assert clause in excerpt
    assert len(excerpt) <= 1600
    german = "Einleitung. " * 300 + "STRAẞE Kosten betragen 45 Euro. " + "Anhang. " * 300
    assert "45 Euro" in select_excerpt(german, {"strasse", "kosten"})


@pytest.mark.parametrize("terms", [set(), {"nonexistent"}, {"fund"}])
def test_no_whole_word_match_keeps_prefix(terms):
    content = "Reference introduction. " + "refund " * 400
    assert select_excerpt(content, terms) == content[:1600]


@pytest.mark.parametrize("limit", [1, 40, 100, 1600])
def test_excerpt_respects_character_limit(limit):
    excerpt = select_excerpt(handbook(), {"refund", "window"}, limit=limit)
    assert len(excerpt) <= limit


def test_short_content_is_preserved():
    assert select_excerpt(REFUND_CLAUSE, {"refund"}) == REFUND_CLAUSE
    assert select_excerpt("", {"refund"}) == ""
    assert select_excerpt(REFUND_CLAUSE, {"refund"}, limit=0) == ""


@pytest.mark.parametrize("query", [
    "What is the AcmeCorporation product refund window?",
    "What is the product refund window?",
])
@pytest.mark.parametrize("budget", [400, 1200])
async def test_public_context_keeps_late_refund_fact_within_budget(tmp_path, query, budget):
    memory = await AgentMemory.open(tmp_path / "references.db")
    try:
        await memory.ingest_knowledge([
            KnowledgeRecord(id="handbook", source="company/handbook", content=handbook())
        ], scope="company")
        context = await memory.before_task(
            query, scope="company", run_id="refund-question", token_budget=budget
        )
        assert [hit.record_id for hit in context.hits] == ["handbook"]
        assert "45 calendar days" in context.text
        assert len(context.text.encode("utf-8")) <= budget
        assert not context.degraded
    finally:
        await memory.close()
