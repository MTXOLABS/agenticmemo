"""Reference changes must not silently recycle previously validated answers."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from agenticmemo import AgentMemory, ExperienceInput, KnowledgeRecord, ValidationResult

TASK = "What is the product refund window?"


def policy(**changes):
    values = dict(id="policy", source="handbook", content="Product refund window is 30 days.")
    values.update(changes)
    return KnowledgeRecord(**values)


async def record(memory, run_id, *, verdict=None, answer="30 days", scope="a", budget=4000):
    context = await memory.before_task(TASK, scope=scope, run_id=run_id, token_budget=budget)
    receipt = await memory.after_task(
        context.handle, experience=ExperienceInput(task=TASK, answer=answer), validation=verdict,
    )
    return context, receipt


def passed():
    return ValidationResult.passed("policy-check", ["Compared against approved policy"])


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"content": "Product refund window is 14 days."},
    {"version": "2"},
    {"source": "replacement-handbook"},
    {"metadata": {"revision": 2}},
])
async def test_reference_changes_exclude_old_experience_and_survive_restart(tmp_path, change):
    path = tmp_path / "memory.db"
    async with await AgentMemory.open(path) as memory:
        await memory.ingest_knowledge([policy()], scope="a")
        _, receipt = await record(memory, "old", verdict=passed())
        assert receipt.eligible
        await memory.ingest_knowledge([policy(**change)], scope="a")
        old = await memory.get(receipt.id, scope="a")
        assert old.stale and not old.eligible
        assert old.validation.status == "passed"  # Keep the historical evidence.
        context = await memory.before_task(TASK, scope="a", run_id="next", token_budget=4000)
        assert [hit.kind for hit in context.hits] == ["knowledge"]
    async with await AgentMemory.open(path) as memory:
        assert (await memory.get(receipt.id, scope="a")).stale
        eligible = await memory.inspect(scope="a", eligible_only=True)
        assert all(r.kind == "knowledge" for r in eligible)
        _, fresh = await record(memory, "new", verdict=passed(), answer="current policy checked")
        assert fresh.eligible


@pytest.mark.asyncio
async def test_duplicate_knowledge_and_other_scopes_do_not_invalidate(tmp_path):
    async with await AgentMemory.open(tmp_path / "memory.db") as memory:
        await memory.ingest_knowledge([policy()], scope="a")
        _, receipt = await record(memory, "old", verdict=passed())
        assert (await memory.ingest_knowledge([policy()], scope="a")).duplicates == 1
        await memory.ingest_knowledge([policy(content="Different policy")], scope="b")
        assert (await memory.get(receipt.id, scope="a")).eligible
        await memory.clear(scope="b")
        assert (await memory.get(receipt.id, scope="a")).eligible


@pytest.mark.asyncio
async def test_deletion_and_recreation_keep_old_experience_ineligible(tmp_path):
    async with await AgentMemory.open(tmp_path / "memory.db") as memory:
        await memory.ingest_knowledge([policy()], scope="a")
        _, receipt = await record(memory, "old", verdict=passed())
        assert await memory.delete("policy", scope="a")
        assert (await memory.get(receipt.id, scope="a")).stale
        await memory.ingest_knowledge([policy()], scope="a")
        assert (await memory.get(receipt.id, scope="a")).stale


@pytest.mark.asyncio
async def test_expired_reference_also_expires_its_experience(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    async with await AgentMemory.open(tmp_path / "memory.db") as memory:
        await memory.ingest_knowledge([policy(valid_until=now + timedelta(days=1))], scope="a")
        _, receipt = await record(memory, "old", verdict=passed())
        monkeypatch.setattr("agenticmemo.plugin.models.utc_now", lambda: now + timedelta(days=2))
        context = await memory.before_task(TASK, scope="a", run_id="later")
        assert not context.hits
        assert (await memory.get(receipt.id, scope="a")).stale


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [0, 4000])
async def test_update_while_host_runs_cannot_make_old_result_eligible(tmp_path, budget):
    async with await AgentMemory.open(tmp_path / "memory.db") as memory:
        await memory.ingest_knowledge([policy()], scope="a")
        context = await memory.before_task(TASK, scope="a", run_id="running", token_budget=budget)
        await memory.ingest_knowledge(
            [policy(content="Product refund window is 14 days.")], scope="a"
        )
        receipt = await memory.after_task(
            context.handle, experience=ExperienceInput(task=TASK, answer="30 days"),
            validation=passed(),
        )
        assert receipt.durable and not receipt.eligible and not receipt.indexed
        assert (await memory.get(receipt.id, scope="a")).stale


@pytest.mark.asyncio
async def test_late_promotion_never_rebases_an_old_run(tmp_path):
    async with await AgentMemory.open(tmp_path / "memory.db") as memory:
        await memory.ingest_knowledge([policy()], scope="a")
        _, receipt = await record(memory, "old")
        await memory.ingest_knowledge([policy(version="2")], scope="a")
        _, promoted = await record(memory, "old", verdict=passed())
        assert promoted.status == "promoted" and not promoted.eligible
        assert (await memory.get(receipt.id, scope="a")).stale


@pytest.mark.asyncio
async def test_reference_update_during_embedding_discards_obsolete_context(tmp_path, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_scores(task, records, scores):
        entered.set()
        await release.wait()
        return scores

    async with await AgentMemory.open(tmp_path / "memory.db") as memory:
        await memory.ingest_knowledge([policy()], scope="a")
        memory._embedder = object()
        monkeypatch.setattr(memory, "_semantic_scores", delayed_scores)
        pending = asyncio.create_task(memory.before_task(TASK, scope="a", run_id="running"))
        await entered.wait()
        await memory._store.upsert_knowledge("a", policy(version="2"))
        release.set()
        context = await pending
        assert context.degraded and not context.hits and not context.text
        receipt = await memory.after_task(
            context.handle, experience=ExperienceInput(task=TASK), validation=passed(),
        )
        assert not receipt.eligible


@pytest.mark.asyncio
async def test_missing_provenance_on_old_records_is_conservative(tmp_path):
    async with await AgentMemory.open(tmp_path / "memory.db") as memory:
        await memory.ingest_knowledge([policy()], scope="a")
        _, receipt = await record(memory, "old", verdict=passed())
        stored = await memory.get(receipt.id, scope="a")
        stored.knowledge_revision = None  # Simulate a pre-provenance envelope.
        await memory._store._run(lambda: memory._store._put(stored))
        assert (await memory.get(receipt.id, scope="a")).stale
        eligible = await memory.inspect(scope="a", eligible_only=True)
        assert all(r.kind == "knowledge" for r in eligible)


@pytest.mark.asyncio
@pytest.mark.parametrize("update_during_embedding", [False, True])
async def test_receipt_refreshes_eligibility_after_concurrent_update(
    tmp_path, monkeypatch, update_during_embedding,
):
    async with await AgentMemory.open(tmp_path / "memory.db") as memory:
        await memory.ingest_knowledge([policy()], scope="a")
        context = await memory.before_task(TASK, scope="a", run_id="running")

        if update_during_embedding:
            async def changing_vector(record):
                await memory._store.upsert_knowledge("a", policy(version="2"))
            memory._embedder = object()
            monkeypatch.setattr(memory, "_vector", changing_vector)
        else:
            original = memory._store.record_experience

            async def changing_record(*args, **kwargs):
                receipt = await original(*args, **kwargs)
                await memory._store.upsert_knowledge("a", policy(version="2"))
                return receipt
            monkeypatch.setattr(memory._store, "record_experience", changing_record)

        receipt = await memory.after_task(
            context.handle, experience=ExperienceInput(task=TASK, answer="30 days"),
            validation=passed(),
        )
        assert receipt.durable and not receipt.eligible and not receipt.indexed
        assert (await memory.get(receipt.id, scope="a")).stale


@pytest.mark.asyncio
@pytest.mark.parametrize("clear_scope", [False, True])
async def test_experience_deleted_during_embedding_is_not_returned(
    tmp_path, monkeypatch, clear_scope,
):
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_scores(task, records, scores):
        entered.set()
        await release.wait()
        return scores

    async with await AgentMemory.open(tmp_path / "memory.db") as memory:
        _, receipt = await record(memory, "old", verdict=passed())
        memory._embedder = object()
        monkeypatch.setattr(memory, "_semantic_scores", delayed_scores)
        pending = asyncio.create_task(memory.before_task(TASK, scope="a", run_id="running"))
        await entered.wait()
        if clear_scope:
            await memory.clear(scope="a")
        else:
            await memory.delete(receipt.id, scope="a")
        release.set()
        context = await pending
        assert not context.hits and not context.text


@pytest.mark.asyncio
async def test_receipt_does_not_claim_eligibility_when_record_deleted_during_indexing(
    tmp_path, monkeypatch,
):
    async with await AgentMemory.open(tmp_path / "memory.db") as memory:
        context = await memory.before_task(TASK, scope="a", run_id="running")

        async def deleting_vector(record):
            await memory.delete(record.id, scope="a")

        memory._embedder = object()
        monkeypatch.setattr(memory, "_vector", deleting_vector)
        receipt = await memory.after_task(
            context.handle, experience=ExperienceInput(task=TASK), validation=passed(),
        )
        assert receipt.durable and not receipt.eligible and not receipt.indexed
        assert receipt.error == "index pending: LookupError"
