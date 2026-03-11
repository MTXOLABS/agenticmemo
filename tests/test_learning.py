"""Tests for learning components: GRPO, Reflexion, Filters."""

import pytest

from agenticmemo.learning.filters import TrajectoryFilter
from agenticmemo.learning.grpo import GRPOPolicy
from agenticmemo.memory.case import Case, CaseOutcome
from agenticmemo.config import LearningConfig, RetrievalConfig
from agenticmemo.types import Step, TaskStatus, Trajectory, ToolCall, ToolResult


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
