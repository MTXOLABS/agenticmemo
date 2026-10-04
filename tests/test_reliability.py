"""Offline regressions for durable memory and conservative outcome validation."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest

from agenticmemo import Agent, AgentConfig
from agenticmemo.config import LearningConfig, MemoryConfig
from agenticmemo.learning.failure_miner import FailurePattern
from agenticmemo.learning.filters import TrajectoryFilter
from agenticmemo.learning.hints import Hint
from agenticmemo.learning.skill_consolidator import Skill
from agenticmemo.learning.verifier import OutcomeVerifier
from agenticmemo.llm.base import LLMBackend
from agenticmemo.memory.case import Case, CaseOutcome
from agenticmemo.memory.graph_memory import EdgeType, TemporalGraphMemory
from agenticmemo.memory.hierarchical import HierarchicalMemory
from agenticmemo.retrieval.ensemble import EnsembleRetriever
from agenticmemo.tools.base import tool
from agenticmemo.types import (
    LLMResponse,
    MemoryDomain,
    Step,
    TaskStatus,
    ToolCall,
    ToolResult,
    Trajectory,
)


class _ScriptedLLM(LLMBackend):
    def __init__(self, replies):
        super().__init__(model="offline-fake")
        self.replies = list(replies)
        self.calls = 0

    async def complete(self, messages, tools=None, system=None):
        self.calls += 1
        if not self.replies:
            raise AssertionError("Unexpected LLM call")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, LLMResponse) else LLMResponse(content=reply)

    async def embed(self, texts):
        raise AssertionError("Provider embeddings must not be called in offline tests")


class _FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def encode(self, texts):
        self.calls.append(tuple(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    @staticmethod
    def batch_cosine_similarity(query, corpus):
        return np.ones(len(corpus), dtype=np.float32)


def _trajectory(task="calculate result", status=TaskStatus.SUCCESS, answer="42"):
    trajectory = Trajectory(task=task, status=status, final_answer=answer)
    for i in range(2):
        call = ToolCall(id=f"call-{i}", name="local_echo", arguments={"code": f"value = {i}"})
        trajectory.add_step(Step(
            index=i,
            thought="Inspect the local result",
            tool_call=call,
            tool_result=ToolResult(
                tool_call_id=call.id, tool_name=call.name, output=str(i),
            ),
            observation=str(i),
        ))
    trajectory.add_step(Step(index=2, thought=answer))
    return trajectory


def _case(case_id, task="debug python code", domain=MemoryDomain.CODING,
          category="debugging", answer="42", status=TaskStatus.SUCCESS, reward=1.0,
          verified=False):
    case = Case(
        id=case_id,
        task=task,
        domain=domain,
        category=category,
        trajectory=_trajectory(task, status, answer),
        outcome=CaseOutcome(status=status, reward=reward, answer=answer),
    )
    if verified:
        case.trajectory.metadata["verification"] = {
            "verdict": status.value, "reason": "Offline fixture evidence",
        }
    return case


def _close(memory):
    graph = memory.graph if isinstance(memory, HierarchicalMemory) else memory
    if graph._sqlite is not None:
        graph._sqlite.close()


def _agent(judge_reply='{"verdict":"success","reason":"The answer is correct"}',
           memory_path=None, learning_reply=None, runs=1, judge_replies=None):
    replies = []
    for _ in range(runs):
        replies.append(LLMResponse(content="Use local evidence and report the result"))
        for i in range(2):
            replies.append(LLMResponse(content="Inspect", tool_calls=[
                ToolCall(id=f"call-{i}", name="local_echo", arguments={"code": f"value = {i}"}),
            ]))
        replies.append(LLMResponse(content="The final answer is 42"))
    if learning_reply is not None:
        replies.append(learning_reply)
    primary = _ScriptedLLM(replies)
    judge = _ScriptedLLM(judge_replies if judge_replies is not None else [judge_reply])
    agent = Agent(primary, AgentConfig(
        max_steps=4,
        max_retries=0,
        memory=MemoryConfig(persist_path=str(memory_path) if memory_path else None),
        learning=LearningConfig(enable_reflexion=False),
    ), judge_llm=judge)
    agent._retriever._embedder = _FakeEmbeddings()

    @tool(name="local_echo")
    async def local_echo(code: str) -> str:
        return code

    agent._tools.register(local_echo)
    return agent


@pytest.mark.parametrize("suffix", [".json", ".db"])
async def test_restart_restores_hierarchy_and_duplicate_fingerprints(tmp_path, suffix):
    cfg = MemoryConfig(persist_path=str(tmp_path / f"memory{suffix}"))
    original = HierarchicalMemory(cfg)
    case = _case("original")
    await original.store(case)
    _close(original)

    restored = HierarchicalMemory(cfg)
    assert [c.id for c in await restored.search_by_domain(MemoryDomain.CODING)] == [case.id]
    assert [c.id for c in await restored.search_by_category(
        MemoryDomain.CODING, "debugging",
    )] == [case.id]
    await restored.store(case.model_copy(deep=True, update={"id": "duplicate"}))
    assert await restored.size() == 1
    assert await restored.get("duplicate") is None
    _close(restored)


async def test_concurrent_json_stores_preserve_every_case_on_restart(tmp_path):
    cfg = MemoryConfig(persist_path=str(tmp_path / "concurrent.json"))
    memory = HierarchicalMemory(cfg)
    cases = [_case(f"case-{i}", answer=str(i)) for i in range(12)]
    await asyncio.gather(*(memory.store(case) for case in cases))
    expected = {case.id for case in cases}
    assert {case.id for case in await memory.all_cases()} == expected
    restored = HierarchicalMemory(cfg)
    assert {case.id for case in await restored.all_cases()} == expected
    assert {case.id for case in await restored.search_by_domain(MemoryDomain.CODING)} == expected


@pytest.mark.parametrize("suffix", [".json", ".db"])
async def test_concurrent_identical_experiences_store_only_once(tmp_path, suffix):
    cfg = MemoryConfig(persist_path=str(tmp_path / f"dedup{suffix}"))
    memory = HierarchicalMemory(cfg)
    cases = [_case(f"duplicate-{i}", verified=True) for i in range(12)]
    await asyncio.gather(*(memory.store(case) for case in cases))
    assert await memory.size() == 1
    assert len(await memory.search_by_domain(MemoryDomain.CODING)) == 1
    assert len(memory._fingerprints) == 1
    _close(memory)
    restored = HierarchicalMemory(cfg)
    assert await restored.size() == 1
    assert len(await restored.search_by_category(MemoryDomain.CODING, "debugging")) == 1
    _close(restored)


@pytest.mark.parametrize("suffix", [".json", ".db"])
async def test_concurrent_capacity_eviction_keeps_indices_and_fingerprints_consistent(
    tmp_path, suffix,
):
    cfg = MemoryConfig(max_cases=1, persist_path=str(tmp_path / f"capacity{suffix}"))
    memory = HierarchicalMemory(cfg)
    domains = [MemoryDomain.CODING, MemoryDomain.MATH, MemoryDomain.RESEARCH]
    cases = [_case(
        f"case-{i}", task=f"task number {i}", domain=domains[i % len(domains)],
        category=f"category-{i}", answer=str(i),
    ) for i in range(12)]
    await asyncio.gather(*(memory.store(case) for case in cases))
    survivors = await memory.all_cases()
    assert len(survivors) == 1
    survivor = survivors[0]
    indexed_ids = {
        case_id for categories in memory._index.values() for traces in categories.values()
        for ids in traces.values() for case_id in ids
    }
    assert indexed_ids == {survivor.id}
    assert set(memory._fingerprints) == {memory._fingerprint(survivor)}
    assert memory._fingerprints[memory._fingerprint(survivor)] == {survivor.id}
    for domain in domains:
        assert {case.id for case in await memory.search_by_domain(domain)} == (
            {survivor.id} if domain == survivor.domain else set()
        )
    evicted = next(case for case in cases if case.id != survivor.id)
    reingested = evicted.model_copy(deep=True, update={"id": "reingested"})
    await memory.store(reingested)
    assert [case.id for case in await memory.all_cases()] == [reingested.id]
    assert set(memory._fingerprints) == {memory._fingerprint(reingested)}
    assert memory._fingerprints[memory._fingerprint(reingested)] == {reingested.id}
    _close(memory)
    restored = HierarchicalMemory(cfg)
    assert [case.id for case in await restored.search_by_category(
        reingested.domain, reingested.category,
    )] == [reingested.id]
    _close(restored)


@pytest.mark.parametrize("hierarchical", [False, True])
async def test_cancelled_sqlite_worker_finishes_before_newer_same_id_write(
    monkeypatch, tmp_path, hierarchical,
):
    path = tmp_path / "cancelled.db"
    cfg = MemoryConfig(persist_path=str(path))
    memory = (HierarchicalMemory(cfg) if hierarchical else
              TemporalGraphMemory(persist_path=str(path)))
    graph = memory.graph if hierarchical else memory
    old_started = threading.Event()
    release_old = threading.Event()
    new_started = threading.Event()
    new_attempted = asyncio.Event()
    save = graph._save_sqlite_case

    def gated_save(case):
        if case.outcome.answer == "old":
            old_started.set()
            if not release_old.wait(timeout=5):
                raise TimeoutError("Test did not release the SQLite worker")
        else:
            new_started.set()
        save(case)

    monkeypatch.setattr(graph, "_save_sqlite_case", gated_save)
    old = _case("same-id", answer="old", domain=MemoryDomain.CODING)
    newer = _case(
        "same-id", task="solve algebra", answer="new", domain=MemoryDomain.MATH,
        category="algebra",
    )
    old_task = asyncio.create_task(memory.store(old))
    new_task = None

    async def store_newer():
        new_attempted.set()
        await memory.store(newer)

    try:
        assert await asyncio.wait_for(asyncio.to_thread(old_started.wait, 3), timeout=4)
        old_task.cancel()
        await asyncio.sleep(0)
        old_task.cancel()
        await asyncio.sleep(0)
        new_task = asyncio.create_task(store_newer())
        await asyncio.wait_for(new_attempted.wait(), timeout=2)
        assert not await asyncio.wait_for(asyncio.to_thread(new_started.wait, 0.1), timeout=2)
        release_old.set()
        with pytest.raises(asyncio.CancelledError):
            await old_task
        await asyncio.wait_for(new_task, timeout=3)
        stored = await memory.get(newer.id)
        assert stored.outcome.answer == "new"
        assert stored.domain == MemoryDomain.MATH
        if hierarchical:
            assert await memory.search_by_domain(MemoryDomain.CODING) == []
            assert [case.id for case in await memory.search_by_category(
                MemoryDomain.MATH, "algebra",
            )] == [newer.id]
    finally:
        release_old.set()
        tasks = [old_task] + ([new_task] if new_task is not None else [])
        await asyncio.gather(*tasks, return_exceptions=True)
        _close(memory)

    restored = (HierarchicalMemory(cfg) if hierarchical else
                TemporalGraphMemory(persist_path=str(path)))
    persisted = await restored.get(newer.id)
    assert persisted.outcome.answer == "new"
    assert persisted.domain == MemoryDomain.MATH
    _close(restored)


@pytest.mark.parametrize("suffix", [".json", ".db"])
async def test_delete_is_durable_and_allows_reingestion_then_clear(tmp_path, suffix):
    cfg = MemoryConfig(persist_path=str(tmp_path / f"memory{suffix}"))
    memory = HierarchicalMemory(cfg)
    case = _case("first")
    await memory.store(case)
    await memory.store(_case("second", task="solve an integral", domain=MemoryDomain.MATH))
    assert await memory.delete(case.id)
    assert not await memory.delete(case.id)
    await memory.store(case.model_copy(deep=True, update={"id": "reingested"}))
    assert await memory.get("reingested") is not None
    _close(memory)

    restored = HierarchicalMemory(cfg)
    assert await restored.get("first") is None
    assert await restored.size() == 2
    await restored.clear()
    assert await restored.size() == 0
    assert await restored.search_by_domain(MemoryDomain.CODING) == []
    _close(restored)

    cleared = HierarchicalMemory(cfg)
    assert await cleared.size() == 0
    await cleared.store(case.model_copy(deep=True, update={"id": "after-clear"}))
    assert await cleared.get("after-clear") is not None
    _close(cleared)


@pytest.mark.parametrize("suffix", [".json", ".db"])
async def test_same_id_update_at_capacity_replaces_indices_without_eviction(tmp_path, suffix):
    cfg = MemoryConfig(max_cases=2, persist_path=str(tmp_path / f"memory{suffix}"))
    memory = HierarchicalMemory(cfg)
    original = _case("changing")
    untouched = _case("untouched", task="solve algebra", domain=MemoryDomain.MATH)
    await memory.store(original)
    await memory.store(untouched)
    updated = original.model_copy(deep=True, update={
        "task": "research a literature survey",
        "domain": MemoryDomain.RESEARCH,
        "category": "literature",
        "keywords": ["literature", "survey"],
        "outcome": CaseOutcome(status=TaskStatus.SUCCESS, reward=1.0, answer="new evidence"),
    })
    await memory.store(updated)
    assert await memory.size() == 2
    assert await memory.get(untouched.id) is not None
    assert await memory.search_by_domain(MemoryDomain.CODING) == []
    assert await memory.search_by_category(MemoryDomain.CODING, "debugging") == []
    assert [c.id for c in await memory.search_by_category(
        MemoryDomain.RESEARCH, "literature",
    )] == [updated.id]
    await memory.store(updated.model_copy(deep=True, update={"id": "duplicate-update"}))
    assert await memory.get("duplicate-update") is None
    await memory.store(original.model_copy(deep=True, update={"id": "old-experience"}))
    assert await memory.get("old-experience") is not None
    _close(memory)


async def test_json_graph_edges_and_deletion_survive_restart(tmp_path):
    path = str(tmp_path / "graph.json")
    memory = TemporalGraphMemory(persist_path=path)
    await memory.store(_case("source"))
    await memory.store(_case("target", task="calculate matrix", domain=MemoryDomain.MATH))
    memory.add_edge("source", "target", EdgeType.REFINES, weight=0.7)

    restored = TemporalGraphMemory(persist_path=path)
    assert [c.id for c in restored.neighbors("source", EdgeType.REFINES)] == ["target"]
    assert restored._graph["source"]["target"]["weight"] == pytest.approx(0.7)
    await restored.delete("target")
    deleted = TemporalGraphMemory(persist_path=path)
    assert await deleted.get("target") is None
    assert deleted.neighbors("source") == []


@pytest.mark.parametrize("suffix", [".json", ".db"])
async def test_eviction_removes_hierarchy_and_duplicate_fingerprint(tmp_path, suffix):
    cfg = MemoryConfig(max_cases=1, persist_path=str(tmp_path / f"memory{suffix}"))
    memory = HierarchicalMemory(cfg)
    first = _case("evicted", status=TaskStatus.FAILURE, reward=-0.2)
    await memory.store(first)
    await memory.store(_case("replacement", task="calculate matrix", domain=MemoryDomain.MATH))
    assert await memory.get(first.id) is None
    assert await memory.search_by_domain(MemoryDomain.CODING) == []
    await memory.store(first.model_copy(deep=True, update={"id": "returning"}))
    assert await memory.get("returning") is not None
    assert await memory.size() == 1
    _close(memory)


async def test_rebuild_after_clear_removes_bm25_and_embeddings_without_encoding():
    memory = HierarchicalMemory()
    embedder = _FakeEmbeddings()
    retriever = EnsembleRetriever(memory, embedder=embedder)
    case = _case("indexed")
    await memory.store(case)
    await retriever.index_case(case)
    assert len(retriever._bm25) == 1
    assert case.id in retriever._embed_cache
    calls = len(embedder.calls)
    await memory.clear()
    await retriever.rebuild_index()
    assert len(retriever._bm25) == 0
    assert not retriever._embed_cache
    assert await retriever.retrieve("debug python") == []
    assert len(embedder.calls) == calls


async def test_explicit_zero_top_k_does_not_encode_nonempty_memory():
    memory = HierarchicalMemory()
    await memory.store(_case("candidate"))
    embedder = _FakeEmbeddings()
    retriever = EnsembleRetriever(memory, embedder=embedder)
    assert await retriever.retrieve("debug python", top_k=0) == []
    assert embedder.calls == []


@pytest.mark.parametrize("relative", ["memory.db", "memory", "directory.json/memory.json"])
async def test_agent_learning_sidecars_are_distinct_and_preserve_main_memory(tmp_path, relative):
    path = tmp_path / relative
    agent = _agent(memory_path=path)
    libraries = [agent._hint_library, agent._skill_library, agent._failure_bank]
    sidecars = [Path(lib.persist_path if hasattr(lib, "persist_path") else lib._persist_path)
                for lib in libraries]
    assert len(set(sidecars + [path])) == 4
    assert all(p.parent == path.parent for p in sidecars)
    assert sidecars == [path.with_name(f"{path.stem}_{kind}.json")
                        for kind in ["hints", "skills", "failures"]]
    await agent._memory.store(_case("main-case"))
    agent._hint_library.add(Hint(id="hint", domain="coding", text="Check evidence"))
    agent._skill_library.add(Skill(
        id="skill", name="inspect", domain="coding", trigger="debugging",
        strategy="Inspect then test", key_tools=[], avg_steps=3.0, avg_reward=1.0,
        success_rate=1.0, support=1,
    ))
    agent._failure_bank.add(FailurePattern(
        id="failure", domain="coding", trigger="missing input", mistake="skip inspection",
        consequence="wrong result", fix="inspect input",
    ))
    if path.suffix == ".db":
        assert path.read_bytes().startswith(b"SQLite format 3\x00")
    _close(agent._memory)

    restored = _agent(memory_path=path)
    assert await restored._memory.get("main-case") is not None
    assert all(lib.size() == 1 for lib in [
        restored._hint_library, restored._skill_library, restored._failure_bank,
    ])
    _close(restored._memory)


@pytest.mark.parametrize("verdict", list(TaskStatus))
async def test_detailed_verification_distinguishes_known_verdicts(verdict):
    reason = "Checked against execution evidence"
    llm = _ScriptedLLM([f'{{"verdict":"{verdict.value}","reason":"{reason}"}}'])
    result = await OutcomeVerifier(llm).verify_detailed("task", _trajectory())
    assert result.verdict == verdict
    assert result.reason == reason
    assert result.verified
    assert result.passed == (verdict == TaskStatus.SUCCESS)
    assert result.model_dump(mode="json")["verdict"] == verdict.value


@pytest.mark.parametrize("reply", [
    "not JSON", "{broken JSON}", '{"verdict":"uncertain"}',
    '{"reason":"No verdict provided"}', RuntimeError("judge unavailable"),
    '{"verdict":"success","reason":[]}',
    '{"verdict":"success","reason":null}',
    '{"verdict":"success","reason":false}',
])
async def test_judge_errors_and_malformed_verdicts_are_unknown_and_partial(reply):
    llm = _ScriptedLLM([reply, reply])
    verifier = OutcomeVerifier(llm)
    result = await verifier.verify_detailed("task", _trajectory())
    assert result.verdict is None
    assert not result.verified
    assert not result.passed
    assert result.reason
    assert await verifier.verify("task", _trajectory()) == TaskStatus.PARTIAL


async def test_missing_verification_reason_preserves_valid_success_compatibility():
    verifier = OutcomeVerifier(_ScriptedLLM(['{"verdict":"success"}']))
    result = await verifier.verify_detailed("task", _trajectory())
    assert result.verdict == TaskStatus.SUCCESS
    assert result.reason == ""
    assert result.verified
    assert result.passed


async def test_disabled_verification_is_unknown_and_does_not_call_judge():
    llm = _ScriptedLLM([])
    verifier = OutcomeVerifier(llm, LearningConfig(enable_verification=False))
    result = await verifier.verify_detailed("task", _trajectory())
    assert result.verdict is None
    assert not result.verified
    assert not result.passed
    assert await verifier.verify("task", _trajectory()) == TaskStatus.PARTIAL
    assert llm.calls == 0


@pytest.mark.parametrize("enabled", [True, False])
async def test_empty_answer_is_known_failure_without_judge_call(enabled):
    llm = _ScriptedLLM([])
    verifier = OutcomeVerifier(llm, LearningConfig(enable_verification=enabled))
    result = await verifier.verify_detailed("task", _trajectory(answer="  "))
    assert result.verdict == TaskStatus.FAILURE
    assert result.verified
    assert not result.passed
    assert llm.calls == 0


async def test_quality_evaluation_exception_rejects_unverified_trajectory():
    llm = _ScriptedLLM([RuntimeError("evaluation unavailable")])
    filter_ = TrajectoryFilter(llm)
    assert not await filter_.filter_single("task", _trajectory(), verified=False)
    assert await filter_.filter_single("task", _trajectory(), verified=True)
    assert llm.calls == 1


@pytest.mark.parametrize("judge_reply", [
    "not JSON", RuntimeError("judge unavailable"),
    '{"verdict":"success","reason":[]}',
    '{"verdict":"success","reason":null}',
    '{"verdict":"success","reason":false}',
])
async def test_agent_unknown_judgment_preserves_answer_without_positive_learning(judge_reply):
    agent = _agent(judge_reply=judge_reply)
    trajectory = await agent.run("calculate the local result", domain=MemoryDomain.MATH)
    assert trajectory.final_answer == "The final answer is 42"
    assert trajectory.num_steps == 3
    assert trajectory.status == TaskStatus.PARTIAL
    assert trajectory.reward == 0.0
    assert trajectory.metadata["verification"]["verdict"] is None
    assert await agent._memory.all_cases() == []
    assert agent._cases_since_hint_extract == 0
    assert agent._skill_consolidator._successes_since_consolidate == 0
    assert agent._hint_library.size() == 0
    assert agent._skill_library.size() == 0


async def test_agent_disabled_verification_keeps_answer_neutral_without_judge_calls():
    agent = _agent()
    agent._cfg.learning.enable_verification = False
    judge = _ScriptedLLM([])
    agent._verifier = OutcomeVerifier(judge, agent._cfg.learning)
    trajectory = await agent.run("calculate the local result", domain=MemoryDomain.MATH)
    assert judge.calls == 0
    assert trajectory.final_answer == "The final answer is 42"
    assert trajectory.status == TaskStatus.PARTIAL
    assert trajectory.reward == 0.0
    assert trajectory.metadata["verification"]["verdict"] is None
    assert agent._cases_since_hint_extract == 0
    assert agent._skill_consolidator._successes_since_consolidate == 0
    assert agent.grpo_stats()["pending_outcomes"] == 0
    assert await agent._memory.all_cases() == []


@pytest.mark.parametrize("suffix", [".json", ".db"])
async def test_unknown_then_verified_success_is_stored_once_and_reusable(tmp_path, suffix):
    agent = _agent(
        memory_path=tmp_path / f"memory{suffix}", runs=2,
        judge_replies=["not JSON", '{"verdict":"success","reason":"Verified result"}'],
    )
    task = "calculate the local result"
    unknown = await agent.run(task, domain=MemoryDomain.MATH)
    assert unknown.status == TaskStatus.PARTIAL
    assert unknown.final_answer == "The final answer is 42"
    assert await agent._memory.size() == 0

    verified = await agent.run(task, domain=MemoryDomain.MATH)
    assert verified.status == TaskStatus.SUCCESS
    assert verified.final_answer == unknown.final_answer
    cases = await agent._memory.all_cases()
    assert len(cases) == 1
    assert cases[0].is_success
    assert cases[0].plan
    assert cases[0].solution
    assert cases[0].trajectory.metadata["verification"]["verdict"] == "success"
    calls = agent._llm.calls
    plan, retrieved = await agent._planner.plan(task, domain=MemoryDomain.MATH)
    assert "PLAN REUSED" in plan
    assert cases[0].id in {case.id for case in retrieved}
    assert agent._llm.calls == calls
    _close(agent._memory)


@pytest.mark.parametrize("suffix", [".json", ".db"])
async def test_failed_then_verified_success_keeps_both_outcomes_for_same_answer(tmp_path, suffix):
    agent = _agent(
        memory_path=tmp_path / f"memory{suffix}", runs=2,
        judge_replies=[
            '{"verdict":"failure","reason":"The result was incorrect"}',
            '{"verdict":"success","reason":"Verified against corrected evidence"}',
        ],
    )
    task = "calculate the local result"
    failed = await agent.run(task, domain=MemoryDomain.MATH)
    assert failed.status == TaskStatus.FAILURE
    assert await agent._memory.size() == 1
    verified = await agent.run(task, domain=MemoryDomain.MATH)
    assert verified.status == TaskStatus.SUCCESS
    assert verified.final_answer == failed.final_answer
    cases = await agent._memory.all_cases()
    assert len(cases) == 2
    assert {case.outcome.status for case in cases} == {TaskStatus.FAILURE, TaskStatus.SUCCESS}
    successful = next(case for case in cases if case.is_success)
    assert successful.plan
    assert successful.solution
    assert all(case.outcome.answer == verified.final_answer for case in cases)
    _close(agent._memory)
    restored = HierarchicalMemory(agent._cfg.memory)
    assert {case.outcome.status for case in await restored.all_cases()} == {
        TaskStatus.FAILURE, TaskStatus.SUCCESS,
    }
    _close(restored)


async def test_agent_disabled_grpo_does_not_queue_validated_outcomes():
    agent = _agent()
    agent._cfg.retrieval.enable_grpo = False
    await agent._memory.store(_case(
        "reference", task="calculate a prior local result", domain=MemoryDomain.MATH,
        verified=True,
    ))
    trajectory = await agent.run("calculate the local result", domain=MemoryDomain.MATH)
    assert trajectory.status == TaskStatus.SUCCESS
    assert trajectory.reward > 0.0
    assert agent.grpo_stats()["pending_outcomes"] == 0


async def test_agent_storage_error_preserves_successful_answer(monkeypatch):
    agent = _agent()

    async def broken_store(case):
        raise OSError("sensitive path must not be exposed")

    monkeypatch.setattr(agent._memory, "store", broken_store)
    trajectory = await agent.run("calculate the local result")
    assert trajectory.status == TaskStatus.SUCCESS
    assert trajectory.final_answer == "The final answer is 42"
    assert trajectory.metadata["learning_errors"]
    assert "OSError" in str(trajectory.metadata["learning_errors"])
    assert "sensitive path" not in str(trajectory.metadata)


@pytest.mark.parametrize("kind", ["hints", "skills"])
async def test_periodic_library_save_error_preserves_successful_answer(monkeypatch, tmp_path, kind):
    import agenticmemo.core.agent as agent_module

    reply = ('["Inspect evidence before answering"]' if kind == "hints" else
             '[{"name":"inspect","trigger":"calculation","strategy":"Inspect then compute",'
             '"key_tools":["local_echo"]}]')
    agent = _agent(memory_path=tmp_path / "memory.json", learning_reply=reply)
    count = 2 if kind == "hints" else 4
    for i in range(count):
        await agent._memory.store(_case(
            f"seed-{i}", task=f"calculate prior result {i}", domain=MemoryDomain.MATH,
            answer=str(i), verified=True,
        ))
    library = agent._hint_library if kind == "hints" else agent._skill_library

    def broken_save(self):
        raise OSError("sensitive persistence detail")

    monkeypatch.setattr(type(library), "_save", broken_save)
    if kind == "hints":
        agent._cases_since_hint_extract = agent_module._HINT_EXTRACT_EVERY - 1
    else:
        agent._skill_consolidator._successes_since_consolidate = (
            agent_module._SKILL_CONSOLIDATE_EVERY - 1
        )
    trajectory = await agent.run("calculate the local result", domain=MemoryDomain.MATH)
    assert trajectory.status == TaskStatus.SUCCESS
    assert trajectory.final_answer == "The final answer is 42"
    assert trajectory.metadata["learning_errors"]
    assert "OSError" in str(trajectory.metadata["learning_errors"])
    assert "sensitive persistence detail" not in str(trajectory.metadata)
    assert agent._llm.calls == 5


async def test_unvalidated_success_labels_do_not_enter_periodic_positive_learning():
    import agenticmemo.core.agent as agent_module

    agent = _agent()
    for i in range(5):
        await agent._memory.store(_case(
            f"unvalidated-{i}", task=f"calculate prior result {i}",
            domain=MemoryDomain.MATH, answer=str(i),
        ))
    agent._cases_since_hint_extract = agent_module._HINT_EXTRACT_EVERY - 1
    agent._skill_consolidator._successes_since_consolidate = (
        agent_module._SKILL_CONSOLIDATE_EVERY - 1
    )
    trajectory = await agent.run("calculate the local result", domain=MemoryDomain.MATH)
    assert trajectory.status == TaskStatus.SUCCESS
    assert trajectory.final_answer == "The final answer is 42"
    assert agent._hint_library.size() == 0
    assert agent._skill_library.size() == 0
    assert agent._llm.calls == 4  # Only planning and three execution steps.
