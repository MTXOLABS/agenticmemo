"""Tests for the memory system."""

import pytest

from agenticmemo.memory.case import Case, CaseOutcome
from agenticmemo.memory.graph_memory import EdgeType, TemporalGraphMemory
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


# ---------------------------------------------------------------------------
# Solution distillation (execution-time exemplars)
# ---------------------------------------------------------------------------

def _traj_with_tool_steps(status: TaskStatus = TaskStatus.SUCCESS) -> Trajectory:
    from agenticmemo.types import Step, ToolCall, ToolResult
    traj = Trajectory(task="compute wacc", status=status, final_answer="WACC=8.1%")
    traj.add_step(Step(
        index=0, thought="write model",
        tool_call=ToolCall(id="t1", name="python_repl",
                           arguments={"code": "wacc = 0.081\nprint(wacc)"}),
        tool_result=ToolResult(tool_call_id="t1", tool_name="python_repl", output="0.081"),
        observation="0.081",
    ))
    traj.add_step(Step(
        index=1, thought="bad step",
        tool_call=ToolCall(id="t2", name="python_repl", arguments={"code": "1/0"}),
        tool_result=ToolResult(tool_call_id="t2", tool_name="python_repl",
                               output=None, error="ZeroDivisionError"),
        observation="ERROR",
    ))
    return traj


def test_extract_solution_collects_working_code_only():
    from agenticmemo.memory.case import extract_solution
    sol = extract_solution(_traj_with_tool_steps())
    assert "wacc = 0.081" in sol
    assert "1/0" not in sol          # errored calls excluded


def test_extract_solution_empty_for_failures():
    from agenticmemo.memory.case import extract_solution
    assert extract_solution(_traj_with_tool_steps(TaskStatus.FAILURE)) == ""


def test_case_prompt_block_includes_solution():
    traj = _traj_with_tool_steps()
    case = Case(
        task="compute wacc",
        trajectory=traj,
        outcome=CaseOutcome(status=TaskStatus.SUCCESS, reward=1.0),
        solution="wacc = 0.081",
    )
    block = case.to_prompt_block()
    assert "Proven solution" in block
    assert "wacc = 0.081" in block


def test_step_penalty_rewards_efficiency():
    from agenticmemo.config import LearningConfig
    from agenticmemo.learning.filters import TrajectoryFilter
    f = TrajectoryFilter(cfg=LearningConfig(step_penalty=0.05))
    short = _traj_with_tool_steps()          # 2 steps
    long = _traj_with_tool_steps()
    for i in range(2, 12):                   # pad to 12 steps
        long.add_step(long.steps[0].model_copy(update={"index": i}))
    assert f.assign_reward(short) > f.assign_reward(long)
    assert f.assign_reward(long) >= 0.5      # floored at half base reward


# ---------------------------------------------------------------------------
# Phase 2: memory packs, anti-copy guarantee, provider compat
# ---------------------------------------------------------------------------

class _NullLLM:
    model = "openai/gpt-oss-120b"
    max_tokens = 4096

    async def complete(self, messages, tools=None, system=None):
        from agenticmemo.types import LLMResponse
        return LLMResponse(content="ok")

    async def embed(self, texts):
        raise NotImplementedError


class _FakeEmb:
    async def encode(self, texts):
        import numpy as np
        vs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t[:20])) % 2**32)
            v = rng.standard_normal(8).astype(np.float32)
            vs.append(v / np.linalg.norm(v))
        return np.stack(vs)

    def batch_cosine_similarity(self, q, corpus):
        return corpus @ q


@pytest.mark.asyncio
async def test_memory_pack_roundtrip(tmp_path):
    from agenticmemo import Agent, AgentConfig

    pack = tmp_path / "pack.json"
    pack.write_text(
        '[{"task": "compute wacc for firm", "domain": "finance", '
        '"code": "# --- INPUTS ---\\nx=1\\n# --- LOGIC ---\\nprint(x)", '
        '"answer": "SECRET_OUTPUT_42"}]'
    )
    agent = Agent(_NullLLM(), AgentConfig())
    agent._retriever._embedder = _FakeEmb()

    n = await agent.load_memory_pack(str(pack))
    assert n == 1 and await agent.memory_size() == 1

    out = tmp_path / "exported.json"
    assert await agent.export_memory_pack(str(out)) == 1

    agent2 = Agent(_NullLLM(), AgentConfig())
    agent2._retriever._embedder = _FakeEmb()
    assert await agent2.load_memory_pack(str(out)) == 1


@pytest.mark.asyncio
async def test_pack_case_never_leaks_answer_into_prompts(tmp_path):
    from agenticmemo import Agent, AgentConfig

    pack = tmp_path / "pack.json"
    pack.write_text(
        '[{"task": "compute wacc", "code": "print(1)", "answer": "SECRET_OUTPUT_42"}]'
    )
    agent = Agent(_NullLLM(), AgentConfig())
    agent._retriever._embedder = _FakeEmb()
    await agent.load_memory_pack(str(pack))

    case = (await agent._memory.all_cases())[0]
    block = case.to_prompt_block()
    assert "SECRET_OUTPUT_42" not in block          # case block: code, not answer
    assert "Proven solution" in block
    for step in case.trajectory.steps:              # observations withheld too
        assert "SECRET_OUTPUT_42" not in step.observation


def test_gpt_oss_tool_rename_quirk():
    from agenticmemo import Agent, AgentConfig
    from agenticmemo.tools import PythonReplTool

    agent = Agent(_NullLLM(), AgentConfig())        # model is gpt-oss
    agent.add_tool(PythonReplTool())
    assert "python" in agent._tools and "python_repl" not in agent._tools


def test_reasoning_output_budget_warning():
    import warnings as w

    from agenticmemo.llm.compat import warn_if_output_budget_low

    class Tiny:
        model = "gpt-5.4-mini"
        max_tokens = 500

    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        warn_if_output_budget_low(Tiny())
    assert caught and "max_tokens" in str(caught[0].message)


# ---------------------------------------------------------------------------
# Phase 4: dedup, quality eviction, cached PageRank, SQLite, LRU
# ---------------------------------------------------------------------------

def _case_with(task, status=TaskStatus.SUCCESS, reward=1.0, solution="", answer=""):
    traj = Trajectory(task=task, status=status, final_answer=answer)
    return Case(task=task, trajectory=traj, solution=solution,
                outcome=CaseOutcome(status=status, reward=reward, answer=answer))


@pytest.mark.asyncio
async def test_dedup_on_store_skips_identical():
    mem = HierarchicalMemory()
    await mem.store(_case_with("same task", answer="same answer"))
    await mem.store(_case_with("same task", answer="same answer"))
    assert await mem.size() == 1
    await mem.store(_case_with("same task", answer="different answer"))
    assert await mem.size() == 2


@pytest.mark.asyncio
async def test_quality_aware_eviction_protects_solutions():
    mem = TemporalGraphMemory(max_cases=2)
    keeper = _case_with("valuable", reward=1.0, solution="print(1)")
    junk = _case_with("junk", status=TaskStatus.FAILURE, reward=-0.2)
    await mem.store(keeper)
    await mem.store(junk)
    await mem.store(_case_with("new arrival", reward=0.5))   # forces eviction
    remaining = {c.task for c in await mem.all_cases()}
    assert "valuable" in remaining          # solution-bearing success protected
    assert "junk" not in remaining          # failure evicted first


@pytest.mark.asyncio
async def test_pagerank_cache_invalidated_on_mutation():
    mem = TemporalGraphMemory()
    a, b = _case_with("a"), _case_with("b")
    await mem.store(a)
    await mem.store(b)
    first = mem.pagerank_scores()
    assert mem.pagerank_scores() is first          # cached object returned
    mem.add_edge(a.id, b.id)
    assert mem._pagerank_cache is None             # mutation invalidates
    assert mem.pagerank_scores() is not first


@pytest.mark.asyncio
async def test_sqlite_backend_roundtrip(tmp_path):
    db = str(tmp_path / "mem.db")
    mem = TemporalGraphMemory(persist_path=db)
    a, b = _case_with("alpha task", answer="42"), _case_with("beta task")
    await mem.store(a)
    await mem.store(b)
    mem.add_edge(a.id, b.id)

    mem2 = TemporalGraphMemory(persist_path=db)    # fresh load from disk
    assert await mem2.size() == 2
    loaded = await mem2.get(a.id)
    assert loaded.outcome.answer == "42"
    assert mem2.neighbors(a.id)                    # edge survived

    await mem2.delete(a.id)
    mem3 = TemporalGraphMemory(persist_path=db)
    assert await mem3.size() == 1                  # delete persisted


@pytest.mark.asyncio
async def test_embed_cache_lru_capped():
    import numpy as np

    from agenticmemo.config import RetrievalConfig
    from agenticmemo.retrieval.ensemble import EnsembleRetriever
    r = EnsembleRetriever(HierarchicalMemory(), RetrievalConfig())
    r._embed_cache_max = 3
    for i in range(6):
        r._cache_put(f"id{i}", np.zeros(4))
    assert len(r._embed_cache) == 3
    assert "id5" in r._embed_cache and "id0" not in r._embed_cache
