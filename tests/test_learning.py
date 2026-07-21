"""Tests for learning components: GRPO, Reflexion, Filters."""

import pytest

from agenticmemo.config import LearningConfig, RetrievalConfig
from agenticmemo.learning.filters import TrajectoryFilter
from agenticmemo.learning.grpo import GRPOPolicy
from agenticmemo.memory.case import Case, CaseOutcome
from agenticmemo.types import Step, TaskStatus, Trajectory


def _make_trajectory(
    task: str = "test",
    status: TaskStatus = TaskStatus.SUCCESS,
    n_steps: int = 3,
    final_answer: str = "42",
) -> Trajectory:
    traj = Trajectory(task=task, status=status, final_answer=final_answer)
    for i in range(n_steps):
        traj.add_step(Step(index=i, thought=f"step {i}", observation=f"obs {i}"))
    return traj


def _make_case(task: str = "test", reward: float = 1.0) -> Case:
    traj = _make_trajectory(task)
    return Case(
        task=task,
        trajectory=traj,
        outcome=CaseOutcome(status=TaskStatus.SUCCESS, reward=reward),
        q_value=0.0,
    )


# ---------------------------------------------------------------------------
# TrajectoryFilter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_filter_accepts_good_trajectory():
    f = TrajectoryFilter(cfg=LearningConfig())
    traj = _make_trajectory(n_steps=3)
    assert await f.filter_single("task", traj) is True

@pytest.mark.asyncio
async def test_filter_rejects_empty_trajectory():
    f = TrajectoryFilter(cfg=LearningConfig(min_trajectory_steps=1))
    traj = _make_trajectory(n_steps=0)
    assert await f.filter_single("task", traj) is False

@pytest.mark.asyncio
async def test_filter_rejects_too_long():
    f = TrajectoryFilter(cfg=LearningConfig(max_trajectory_steps=5))
    traj = _make_trajectory(n_steps=10)
    assert await f.filter_single("task", traj) is False

@pytest.mark.asyncio
async def test_filter_disabled():
    f = TrajectoryFilter(cfg=LearningConfig(enable_quality_filter=False))
    traj = _make_trajectory(n_steps=0)  # would normally fail
    assert await f.filter_single("task", traj) is True

def test_filter_reward_assignment():
    f = TrajectoryFilter(cfg=LearningConfig(
        success_reward=1.0, partial_reward=0.3, failure_reward=-0.2
    ))
    assert f.assign_reward(_make_trajectory(status=TaskStatus.SUCCESS)) == 1.0
    assert f.assign_reward(_make_trajectory(status=TaskStatus.PARTIAL)) == 0.3
    assert f.assign_reward(_make_trajectory(status=TaskStatus.FAILURE)) == -0.2

@pytest.mark.asyncio
async def test_filter_batch_variance_prune():
    """Low-variance batch (all same reward) should be pruned to half."""
    f = TrajectoryFilter(cfg=LearningConfig(variance_filter_threshold=0.1))
    items = [("task", _make_trajectory(status=TaskStatus.SUCCESS)) for _ in range(6)]
    # All rewards = 1.0 → variance = 0 < threshold → half kept
    result = await f.filter_batch(items)
    assert len(result) <= 3


# ---------------------------------------------------------------------------
# GRPOPolicy
# ---------------------------------------------------------------------------

def test_grpo_rerank_does_not_change_order_with_zero_q():
    policy = GRPOPolicy(RetrievalConfig(enable_grpo=True))
    cases = [_make_case(f"task {i}", reward=float(i)) for i in range(5)]
    scored = [(c, float(i)) for i, c in enumerate(cases)]
    reranked = policy.rerank(scored)
    # With zero Q-values, order should be same
    assert [c.id for c, _ in reranked] == [c.id for c, _ in scored][::-1]  # sorted desc

def test_grpo_update_changes_q_values():
    policy = GRPOPolicy(RetrievalConfig(enable_grpo=True, grpo_update_every=1))
    cases = [_make_case(f"task {i}") for i in range(3)]
    case_map = {c.id: c for c in cases}

    # Record diverse outcomes (high variance)
    policy.record_outcome([cases[0].id], 1.0)
    policy.record_outcome([cases[1].id], -0.5)
    policy.record_outcome([cases[2].id], 0.8)

    updated = policy.update(case_map)
    assert updated > 0
    # Winning case should have positive Q-value, losing negative
    assert cases[0].q_value > 0
    assert cases[1].q_value < 0

def test_grpo_q_value_clipped():
    policy = GRPOPolicy(RetrievalConfig(enable_grpo=True, grpo_lr=100.0, grpo_update_every=1))
    case = _make_case()
    policy.record_outcome([case.id], 1.0)
    policy.record_outcome([case.id], 1.0)
    policy.update({case.id: case})
    assert case.q_value <= 5.0

def test_grpo_disabled():
    policy = GRPOPolicy(RetrievalConfig(enable_grpo=False))
    cases = [_make_case(f"task {i}") for i in range(3)]
    scored = [(c, 1.0) for c in cases]
    reranked = policy.rerank(scored)
    # Disabled → no change
    assert reranked == scored

def test_grpo_stats():
    policy = GRPOPolicy()
    policy.record_outcome(["id1"], 0.9)
    stats = policy.stats()
    assert "avg_reward" in stats
    assert "total_updates" in stats
    assert stats["pending_outcomes"] == 1

def test_grpo_sample_candidate_sets():
    policy = GRPOPolicy()
    cases = [_make_case(f"task {i}") for i in range(10)]
    scored = [(c, float(i)) for i, c in enumerate(cases)]
    sets = policy.sample_candidate_sets(scored, top_k=3, n_samples=5)
    assert len(sets) == 5
    for s in sets:
        assert len(s) == 3


# ---------------------------------------------------------------------------
# Phase 0/1 upgrades: token accounting, single judgment, lazy learning
# ---------------------------------------------------------------------------

class _CountingLLM:
    """Minimal fake backend counting complete() calls."""
    model = "fake"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, tools=None, system=None):
        from agenticmemo.types import LLMResponse
        self.calls += 1
        return LLMResponse(content='{"score": 0.9}', input_tokens=10, output_tokens=5)

    async def embed(self, texts):
        raise NotImplementedError


async def test_filter_skips_self_eval_when_verified():
    from agenticmemo.config import LearningConfig
    llm = _CountingLLM()
    f = TrajectoryFilter(llm, LearningConfig())
    traj = _make_trajectory(status=TaskStatus.SUCCESS)
    assert await f.filter_single("t", traj, verified=True)
    assert llm.calls == 0                      # no LLM call when verified
    assert await f.filter_single("t", traj, verified=False)
    assert llm.calls == 1                      # unverified path still evaluates


def test_token_ledger_breakdown_and_delta():
    from agenticmemo.llm.accounting import TokenLedger
    ledger = TokenLedger()
    ledger.record("planner", 100)
    snap = ledger.snapshot()
    ledger.record("executor", 50)
    ledger.record("planner", 25)
    delta = ledger.delta_since(snap)
    assert delta == {"planner": 25, "executor": 50}
    assert ledger.breakdown()["planner"] == {"tokens": 125, "calls": 2}
    assert ledger.total == 175


async def test_tagged_llm_records_usage():
    from agenticmemo.llm.accounting import TaggedLLM, TokenLedger
    ledger = TokenLedger()
    tagged = TaggedLLM(_CountingLLM(), "verifier", ledger)
    await tagged.complete([])
    assert ledger.breakdown() == {"verifier": {"tokens": 15, "calls": 1}}


# ---------------------------------------------------------------------------
# Phase 3: plan reuse, stepwise guidance, sanity nudges
# ---------------------------------------------------------------------------

class _RecordingLLM:
    """Fake backend that records every message list it receives."""
    model = "fake"

    def __init__(self, script=None):
        self.seen = []
        self.calls = 0
        self._script = script or []

    async def complete(self, messages, tools=None, system=None):
        from agenticmemo.types import LLMResponse
        self.seen.append([m.content for m in messages])
        self.calls += 1
        if self._script:
            return self._script.pop(0)
        return LLMResponse(content="done")

    async def embed(self, texts):
        raise NotImplementedError


class _StubRetriever:
    def __init__(self, results):
        self._results = results

    async def retrieve(self, query, top_k=None, domain=None, min_score=None):
        return self._results


def _solved_case(plan="Step 1: run the model"):
    traj = Trajectory(task="wacc dcf", status=TaskStatus.SUCCESS, final_answer="ok")
    return Case(
        task="wacc dcf", trajectory=traj,
        outcome=CaseOutcome(status=TaskStatus.SUCCESS, reward=1.0),
        plan=plan, solution="print(1)",
    )


@pytest.mark.asyncio
async def test_plan_reuse_skips_planner_llm():
    from agenticmemo.config import AgentConfig
    from agenticmemo.core.planner import Planner

    llm = _RecordingLLM()
    planner = Planner(llm, _StubRetriever([(_solved_case(), 0.9)]), AgentConfig())
    plan, cases = await planner.plan("compute wacc for another firm")
    assert "PLAN REUSED" in plan and "Step 1: run the model" in plan
    assert llm.calls == 0                      # no LLM call — the whole point


@pytest.mark.asyncio
async def test_plan_reuse_disabled_below_threshold():
    from agenticmemo.config import AgentConfig
    from agenticmemo.core.planner import Planner

    llm = _RecordingLLM()
    planner = Planner(llm, _StubRetriever([(_solved_case(), 0.5)]), AgentConfig())
    plan, _ = await planner.plan("unrelated task")
    assert "PLAN REUSED" not in plan
    assert llm.calls == 1                      # fell through to normal planning


@pytest.mark.asyncio
async def test_stepwise_guidance_injected_per_turn():
    from agenticmemo.config import AgentConfig
    from agenticmemo.core.executor import Executor
    from agenticmemo.tools.registry import ToolRegistry
    from agenticmemo.types import LLMResponse

    llm = _RecordingLLM(script=[LLMResponse(content="final answer")])
    ex = Executor(llm, ToolRegistry(), AgentConfig(max_steps=3))
    await ex.execute("t", "plan", exemplar_steps=["compute inputs → python"])
    first_turn = llm.seen[0]
    assert any("[REFERENCE TRAJECTORY]" in m and "compute inputs" in m for m in first_turn)


@pytest.mark.asyncio
async def test_sanity_nudge_on_repeated_identical_call():
    from agenticmemo.config import AgentConfig
    from agenticmemo.core.executor import Executor
    from agenticmemo.tools.registry import ToolRegistry
    from agenticmemo.types import LLMResponse, ToolCall

    def same_call(i):
        return LLMResponse(content="", tool_calls=[
            ToolCall(id=f"c{i}", name="missing_tool", arguments={"x": 1})])
    llm = _RecordingLLM(script=[same_call(1), same_call(2),
                                LLMResponse(content="stopping")])
    ex = Executor(llm, ToolRegistry(), AgentConfig(max_steps=5))
    await ex.execute("t", "plan")
    third_turn = llm.seen[2]
    joined = " ".join(third_turn)
    assert "COURSE CORRECTION" in joined       # streak or repeat nudge fired


@pytest.mark.asyncio
async def test_no_stepwise_guidance_from_single_step_exemplar():
    """A 1-step pack exemplar must NOT produce guidance — its first hint
    would be 'final answer', teaching premature termination (measured)."""
    from agenticmemo import Agent, AgentConfig

    class _Emb:
        async def encode(self, texts):
            import numpy as np
            return np.ones((len(texts), 4), dtype="float32")

        def batch_cosine_similarity(self, q, corpus):
            import numpy as np
            return np.ones(len(corpus))

    llm = _RecordingLLM(script=None)
    agent = Agent(llm, AgentConfig(max_retries=0))
    agent._retriever._embedder = _Emb()
    # seed a 1-step solved case directly
    case = _solved_case()
    await agent._memory.store(case)
    await agent._retriever.index_case(case)

    await agent.run("wacc dcf again")
    for turn in llm.seen:
        assert not any("[REFERENCE TRAJECTORY]" in m for m in turn)
