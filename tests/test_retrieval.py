"""Tests for retrieval system."""

import pytest
import numpy as np

from agenticmemo.memory.case import Case, CaseOutcome
from agenticmemo.memory.hierarchical import HierarchicalMemory
from agenticmemo.retrieval.bm25 import BM25Index
from agenticmemo.retrieval.embeddings import SentenceTransformerEmbeddings
from agenticmemo.retrieval.ensemble import EnsembleRetriever
from agenticmemo.config import RetrievalConfig
from agenticmemo.types import TaskStatus, Trajectory


def _make_case(task: str, reward: float = 1.0) -> Case:
    traj = Trajectory(task=task, status=TaskStatus.SUCCESS, final_answer="done")
    return Case(
        task=task,
        trajectory=traj,
        outcome=CaseOutcome(status=TaskStatus.SUCCESS, reward=reward),
    )


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def test_bm25_basic_scoring():
    idx = BM25Index()
    c1 = _make_case("write a Python function to sort numbers")
    c2 = _make_case("solve algebra equation for unknowns")
    # BM25 IDF requires N > n; add extra unrelated docs for non-zero IDF
    for i in range(6):
        idx.add(_make_case(f"cooking recipe step {i} boil water add salt"))
    idx.add(c1)
    idx.add(c2)
    scores = idx.scores("Python sort algorithm")
    assert scores[c1.id] > scores[c2.id]

def test_bm25_empty_index():
    idx = BM25Index()
    scores = idx.scores("anything")
    assert scores == {}

def test_bm25_remove():
    idx = BM25Index()
    c = _make_case("test task")
    idx.add(c)
    idx.remove(c.id)
    assert len(idx) == 0

def test_bm25_normalised_max_1():
    idx = BM25Index()
    for t in ["python code", "python script", "debug python error"]:
        idx.add(_make_case(t))
    scores = idx.scores("python")
    assert all(0.0 <= v <= 1.0 for v in scores.values())


# ---------------------------------------------------------------------------
# Embeddings (uses local model, may be slow first run)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embeddings_shape():
    embedder = SentenceTransformerEmbeddings()
    texts = ["hello world", "foo bar baz"]
    vecs = await embedder.encode(texts)
    assert vecs.shape[0] == 2
    assert vecs.shape[1] > 0

@pytest.mark.asyncio
async def test_embeddings_cosine_same():
    embedder = SentenceTransformerEmbeddings()
    vecs = await embedder.encode(["hello world"])
    sim = embedder.cosine_similarity(vecs[0], vecs[0])
    assert abs(sim - 1.0) < 1e-4

@pytest.mark.asyncio
async def test_embeddings_similar_higher_than_dissimilar():
    embedder = SentenceTransformerEmbeddings()
    vecs = await embedder.encode([
        "write Python code",
        "Python programming script",
        "cook a pasta recipe",
    ])
    sim_similar = embedder.cosine_similarity(vecs[0], vecs[1])
    sim_dissimilar = embedder.cosine_similarity(vecs[0], vecs[2])
    assert sim_similar > sim_dissimilar

@pytest.mark.asyncio
async def test_embeddings_empty():
    embedder = SentenceTransformerEmbeddings()
    vecs = await embedder.encode([])
    assert vecs.shape[0] == 0


# ---------------------------------------------------------------------------
# EnsembleRetriever
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensemble_returns_top_k():
    mem = HierarchicalMemory()
    retriever = EnsembleRetriever(mem, RetrievalConfig(top_k=2))

    cases = [
        _make_case("write Python function to compute fibonacci"),
        _make_case("solve differential equation"),
        _make_case("search the web for news"),
        _make_case("debug Python import error"),
    ]
    for c in cases:
        await mem.store(c)
        await retriever.index_case(c)

    results = await retriever.retrieve("Python programming task", top_k=2)
    assert len(results) == 2
    # Python-related cases should rank higher
    top_tasks = [c.task for c, _ in results]
    assert any("Python" in t for t in top_tasks)

@pytest.mark.asyncio
async def test_ensemble_empty_memory():
    mem = HierarchicalMemory()
    retriever = EnsembleRetriever(mem)
    results = await retriever.retrieve("some task")
    assert results == []

@pytest.mark.asyncio
async def test_ensemble_scores_normalised():
    mem = HierarchicalMemory()
    retriever = EnsembleRetriever(mem, RetrievalConfig(min_similarity=0.0))
    c = _make_case("example task")
    await mem.store(c)
    await retriever.index_case(c)
    results = await retriever.retrieve("example task")
    assert len(results) == 1
    _, score = results[0]
    assert score >= 0.0

@pytest.mark.asyncio
async def test_ensemble_rebuild_index():
    mem = HierarchicalMemory()
    retriever = EnsembleRetriever(mem)
    for i in range(3):
        c = _make_case(f"task number {i}")
        await mem.store(c)
    await retriever.rebuild_index()
    results = await retriever.retrieve("task number 1")
    assert len(results) > 0
