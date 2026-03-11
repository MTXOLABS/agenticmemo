"""Tests for Phase 4: SharedMemoryPool, HintLibrary, HintExtractor."""

import pytest

from agenticmemo.memory.case import Case, CaseOutcome
from agenticmemo.memory.shared import SharedMemoryPool
from agenticmemo.learning.hints import Hint, HintLibrary
from agenticmemo.types import MemoryDomain, TaskStatus, Trajectory


def _make_case(task: str = "test", reward: float = 1.0, answer: str = "done") -> Case:
    traj = Trajectory(task=task, status=TaskStatus.SUCCESS, final_answer=answer)
    return Case(
        task=task,
        trajectory=traj,
        outcome=CaseOutcome(status=TaskStatus.SUCCESS, reward=reward, answer=answer),
    )


# ---------------------------------------------------------------------------
# SharedMemoryPool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_shared_pool_register_and_store():
    pool = SharedMemoryPool()
    pool.register_agent("planner")
    pool.register_agent("executor")

    case = _make_case("write Python code")
    stored = await pool.store_shared(case)
    assert stored is True
    assert await pool.shared_size() == 1

@pytest.mark.asyncio
async def test_shared_pool_dedup_prevents_duplicates():
    pool = SharedMemoryPool()
    case = _make_case("write Python code", answer="done")
    # Same task + answer → same fingerprint → second store rejected
    await pool.store_shared(case)
    second = _make_case("write Python code", answer="done")
    stored = await pool.store_shared(second)
    assert stored is False
    assert await pool.shared_size() == 1

@pytest.mark.asyncio
async def test_shared_pool_different_cases_both_stored():
    pool = SharedMemoryPool()
    c1 = _make_case("write Python code", answer="result A")
    c2 = _make_case("solve math problem", answer="result B")
    await pool.store_shared(c1)
    await pool.store_shared(c2)
    assert await pool.shared_size() == 2

@pytest.mark.asyncio
async def test_shared_pool_private_storage():
    pool = SharedMemoryPool()
    pool.register_agent("agent1")
    case = _make_case("private task")
    await pool.store_private("agent1", case)
    assert await pool.private_size("agent1") == 1
    assert await pool.shared_size() == 0

@pytest.mark.asyncio
async def test_shared_pool_private_unregistered_agent_raises():
    pool = SharedMemoryPool()
    with pytest.raises(KeyError):
        await pool.store_private("unknown_agent", _make_case())

@pytest.mark.asyncio
async def test_shared_pool_merged_view():
    pool = SharedMemoryPool()
    pool.register_agent("agent1")

    shared = _make_case("shared task", answer="shared answer")
    private = _make_case("private task", answer="private answer")

    await pool.store_shared(shared)
    await pool.store_private("agent1", private)

    all_cases = await pool.all_cases_for("agent1")
    task_names = [c.task for c in all_cases]
    assert "shared task" in task_names
    assert "private task" in task_names

@pytest.mark.asyncio
async def test_shared_pool_consensus_promotion():
    pool = SharedMemoryPool(consensus_threshold=0.8)
    high = _make_case("great task", reward=0.9, answer="great answer")
    low = _make_case("bad task", reward=0.2, answer="bad answer")
    await pool.store_shared(high)
    await pool.store_shared(low)
    consensus = pool.consensus_cases()
    assert any(c.task == "great task" for c in consensus)
    assert not any(c.task == "bad task" for c in consensus)

@pytest.mark.asyncio
async def test_shared_pool_dedup_shared():
    """dedup_shared() removes exact duplicate fingerprints found in the store."""
    pool = SharedMemoryPool()
    # Bypass store_shared (which rejects dups) and insert directly
    c1 = _make_case("task A", answer="answer A")
    c2 = _make_case("task A", answer="answer A")  # same fingerprint
    await pool._shared.store(c1)
    await pool._shared.store(c2)
    # Both are in the store; dedup_shared scans and finds 1 duplicate → removes 1
    removed = await pool.dedup_shared()
    assert removed == 1
    assert await pool.shared_size() == 1

@pytest.mark.asyncio
async def test_shared_pool_agent_ids():
    pool = SharedMemoryPool()
    pool.register_agent("alpha")
    pool.register_agent("beta")
    assert set(pool.agent_ids()) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# HintLibrary
# ---------------------------------------------------------------------------

def test_hint_library_add_and_retrieve():
    lib = HintLibrary()
    h = Hint(id="h1", domain="coding", text="Always test edge cases", support=5, avg_reward=0.9)
    lib.add(h)
    assert lib.size() == 1
    assert lib.by_domain("coding")[0].text == "Always test edge cases"

def test_hint_library_no_exact_duplicates():
    lib = HintLibrary()
    h1 = Hint(id="h1", domain="coding", text="Test edge cases", support=3, avg_reward=0.8)
    h2 = Hint(id="h2", domain="coding", text="Test edge cases", support=2, avg_reward=0.7)
    lib.add(h1)
    lib.add(h2)
    assert lib.size() == 1  # duplicate text rejected

def test_hint_library_top_by_support():
    lib = HintLibrary()
    for i in range(5):
        lib.add(Hint(id=f"h{i}", domain="general", text=f"Hint {i}", support=i, avg_reward=0.5))
    top = lib.top(3)
    assert len(top) == 3
    assert top[0].support >= top[1].support  # sorted descending

def test_hint_library_domain_filter():
    lib = HintLibrary()
    lib.add(Hint(id="c1", domain="coding", text="coding hint", support=1, avg_reward=0.5))
    lib.add(Hint(id="m1", domain="math", text="math hint", support=1, avg_reward=0.5))
    coding = lib.by_domain("coding")
    assert len(coding) == 1
    assert coding[0].domain == "coding"

def test_hint_library_to_prompt_block_empty():
    lib = HintLibrary()
    assert lib.to_prompt_block() == ""

def test_hint_library_to_prompt_block_with_hints():
    lib = HintLibrary()
    lib.add(Hint(id="h1", domain="general", text="Check outputs before proceeding", support=3, avg_reward=0.9))
    block = lib.to_prompt_block()
    assert "Check outputs before proceeding" in block
    assert "Internalized Hints" in block
