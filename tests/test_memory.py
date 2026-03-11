"""Tests for the memory system."""

import pytest

from agenticmemo.memory.case import Case, CaseOutcome
from agenticmemo.memory.graph_memory import TemporalGraphMemory, EdgeType
from agenticmemo.memory.hierarchical import HierarchicalMemory, classify_domain, extract_keywords
from agenticmemo.types import MemoryDomain, TaskStatus, Trajectory


def _make_case(task: str = "test task", status: TaskStatus = TaskStatus.SUCCESS) -> Case:
    traj = Trajectory(task=task, status=status, final_answer="42")
    outcome = CaseOutcome(status=status, reward=1.0 if status == TaskStatus.SUCCESS else -0.2)
    return Case(task=task, trajectory=traj, outcome=outcome)


# ---------------------------------------------------------------------------
# Domain classifier
# ---------------------------------------------------------------------------

def test_classify_domain_coding():
    assert classify_domain("debug this Python function") == MemoryDomain.CODING

def test_classify_domain_math():
    assert classify_domain("solve the integral of x^2") == MemoryDomain.MATH

def test_classify_domain_web():
    assert classify_domain("search the web for latest news") == MemoryDomain.WEB

def test_classify_domain_general():
    assert classify_domain("hello") == MemoryDomain.GENERAL

def test_extract_keywords():
    kws = extract_keywords("write a Python function to compute fibonacci numbers")
    assert "python" in kws
    assert "fibonacci" in kws
    assert "function" in kws


# ---------------------------------------------------------------------------
# TemporalGraphMemory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_graph_memory_store_and_get():
    mem = TemporalGraphMemory()
    case = _make_case("write hello world in Python")
    await mem.store(case)
    fetched = await mem.get(case.id)
    assert fetched is not None
    assert fetched.id == case.id
    assert fetched.task == case.task

@pytest.mark.asyncio
async def test_graph_memory_delete():
    mem = TemporalGraphMemory()
    case = _make_case()
    await mem.store(case)
    deleted = await mem.delete(case.id)
    assert deleted is True
    assert await mem.get(case.id) is None

@pytest.mark.asyncio
async def test_graph_memory_size():
    mem = TemporalGraphMemory()
    assert await mem.size() == 0
    for i in range(5):
        await mem.store(_make_case(f"task {i}"))
    assert await mem.size() == 5

@pytest.mark.asyncio
async def test_graph_memory_eviction():
    mem = TemporalGraphMemory(max_cases=3)
    for i in range(5):
        await mem.store(_make_case(f"task {i}"))
    assert await mem.size() == 3

def test_graph_edges():
    mem = TemporalGraphMemory()
    import asyncio
    c1 = _make_case("task A")
    c2 = _make_case("task B")
    asyncio.get_event_loop().run_until_complete(mem.store(c1))
    asyncio.get_event_loop().run_until_complete(mem.store(c2))
    mem.add_edge(c1.id, c2.id, EdgeType.SIMILAR, weight=0.9)
    nbrs = mem.neighbors(c1.id, EdgeType.SIMILAR)
    assert any(n.id == c2.id for n in nbrs)

def test_temporal_weight_recent():
    mem = TemporalGraphMemory(temporal_decay_rate=0.01)
    case = _make_case()
    # Fresh case should have weight close to 1.0
    w = mem.temporal_weight(case)
    assert 0.99 < w <= 1.0

def test_graph_proximity_disconnected():
    mem = TemporalGraphMemory()
    import asyncio
    c1 = _make_case("A")
    c2 = _make_case("B")
    asyncio.get_event_loop().run_until_complete(mem.store(c1))
    asyncio.get_event_loop().run_until_complete(mem.store(c2))
    score = mem.graph_proximity_score(c1.id, c2.id)
    assert score == 0.0  # no edge = disconnected


# ---------------------------------------------------------------------------
# HierarchicalMemory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hierarchical_store_and_retrieve():
    mem = HierarchicalMemory()
    case = _make_case("write a Python script to sort a list")
    await mem.store(case)
    assert await mem.size() == 1
    # Should be classified as coding
    stored = await mem.get(case.id)
    assert stored is not None
    assert stored.domain == MemoryDomain.CODING

@pytest.mark.asyncio
async def test_hierarchical_domain_search():
    mem = HierarchicalMemory()
    c1 = _make_case("debug Python code")
    c2 = _make_case("compute the integral")
    await mem.store(c1)
    await mem.store(c2)
    coding_cases = await mem.search_by_domain(MemoryDomain.CODING)
    math_cases = await mem.search_by_domain(MemoryDomain.MATH)
    assert any(c.id == c1.id for c in coding_cases)
    assert any(c.id == c2.id for c in math_cases)

@pytest.mark.asyncio
async def test_hierarchical_delete():
    mem = HierarchicalMemory()
    case = _make_case("search the internet")
    await mem.store(case)
    assert await mem.delete(case.id) is True
    assert await mem.size() == 0


# ---------------------------------------------------------------------------
# Case model
# ---------------------------------------------------------------------------

def test_case_summary():
    case = _make_case("solve this problem")
    summary = case.summary()
    assert "✓" in summary
    assert "solve this problem" in summary

def test_case_prompt_block():
    case = _make_case("write code")
    block = case.to_prompt_block()
    assert "write code" in block
    assert "---" in block

def test_case_age_days():
    case = _make_case()
    assert case.age_days < 0.01  # just created

def test_case_is_failure():
    case = _make_case(status=TaskStatus.FAILURE)
    assert case.is_failure
    assert not case.is_success
