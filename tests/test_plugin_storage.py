"""Durability and ownership contracts for the independent SQLite memory store."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import threading
from contextlib import closing
from datetime import timedelta

import pytest
import pytest_asyncio

from agenticmemo.plugin.models import (
    ExperienceInput,
    KnowledgeRecord,
    MemoryConfigurationError,
    ValidationResult,
    utc_now,
)
from agenticmemo.plugin.storage import (
    SQLiteStore,
    StoreClosedError,
    StoreCorruptionError,
    StoreInUseError,
)


@pytest_asyncio.fixture
async def store(tmp_path):
    instance = await SQLiteStore.open(tmp_path / "memory.sqlite")
    try:
        yield instance
    finally:
        await instance.close()


def knowledge(id="policy", **changes):
    return KnowledgeRecord(id=id, content="Refunds require approval", source="policy-v1", **changes)


def experience(**changes):
    return ExperienceInput(task="Handle a refund", answer="Approval requested", **changes)


def passed():
    return ValidationResult.passed("refund-check", ["approval ticket verified"])


async def test_knowledge_is_durable_and_preserves_source_version_metadata(store):
    supplied = knowledge(version="3", metadata={"department": "support"})
    receipt = await store.upsert_knowledge("tenant-a", supplied)
    assert receipt.status == "stored"
    assert receipt.durable and receipt.eligible and not receipt.indexed
    path = store.path
    await store.close()
    reopened = await SQLiteStore.open(path)
    try:
        record = await reopened.get("tenant-a", supplied.id)
        assert record.knowledge == supplied
        assert record.scope == "tenant-a" and record.kind == "knowledge"
        assert record.experience is None and record.run_id is None
        assert len(await reopened.list_records("tenant-a", eligible_only=True)) == 1
    finally:
        await reopened.close()


async def test_knowledge_duplicate_and_changed_version_updates_same_identity(store):
    original = knowledge()
    first = await store.upsert_knowledge("tenant-a", original)
    before = await store.get("tenant-a", first.id)
    duplicate = await store.upsert_knowledge("tenant-a", original)
    assert duplicate.status == "duplicate" and duplicate.id == first.id
    assert (await store.get("tenant-a", first.id)).updated_at == before.updated_at
    changed = original.model_copy(
        update={"content": "Refunds require manager approval", "version": "2"}
    )
    updated = await store.upsert_knowledge("tenant-a", changed)
    after = await store.get("tenant-a", first.id)
    assert updated.status == "stored" and updated.id == first.id
    assert after.knowledge == changed and after.created_at == before.created_at
    assert len(await store.list_records("tenant-a")) == 1


async def test_expired_knowledge_is_inspectable_but_ineligible(store):
    await store.upsert_knowledge(
        "tenant-a", knowledge(valid_until=utc_now() - timedelta(seconds=1))
    )
    assert len(await store.list_records("tenant-a")) == 1
    assert await store.list_records("tenant-a", eligible_only=True) == []


async def test_knowledge_exact_data_distinguishes_boolean_and_number_metadata(store):
    await store.upsert_knowledge("tenant-a", knowledge(metadata={"value": 1}))
    changed = await store.upsert_knowledge("tenant-a", knowledge(metadata={"value": True}))
    assert changed.status == "stored"
    assert (await store.get("tenant-a", "policy")).knowledge.metadata["value"] is True


async def test_scopes_isolate_ids_run_ids_reads_and_deletion(store):
    await store.upsert_knowledge("tenant-a", knowledge())
    await store.upsert_knowledge("tenant-b", knowledge())
    a = await store.record_experience("tenant-a", "run-1", experience(), passed())
    b = await store.record_experience("tenant-b", "run-1", experience(), passed())
    assert a.id != b.id
    assert await store.get("tenant-b", a.id) is None
    assert not await store.delete("tenant-b", a.id)
    assert await store.clear("tenant-a") == 2
    assert await store.list_records("tenant-a") == []
    assert len(await store.list_records("tenant-b")) == 2


async def test_unknown_experience_promotes_once_and_does_not_demote(store):
    supplied = experience(actions=[{"tool": "ticket", "result": "created"}], metadata={"ticket": 7})
    first = await store.record_experience("tenant-a", "run-1", supplied, ValidationResult.unknown())
    assert first.status == "stored" and first.durable and not first.eligible
    before = await store.get("tenant-a", first.id)
    assert await store.list_records("tenant-a", eligible_only=True) == []
    duplicate = await store.record_experience(
        "tenant-a", "run-1", supplied, ValidationResult.unknown()
    )
    assert duplicate.status == "duplicate" and duplicate.id == first.id
    promoted = await store.record_experience("tenant-a", "run-1", supplied, passed())
    assert promoted.status == "promoted" and promoted.id == first.id and promoted.eligible
    again = await store.record_experience("tenant-a", "run-1", supplied, passed())
    assert again.status == "duplicate"
    unknown = await store.record_experience(
        "tenant-a", "run-1", supplied, ValidationResult.unknown()
    )
    failed = await store.record_experience(
        "tenant-a", "run-1", supplied, ValidationResult.failed("late-check", "failed")
    )
    assert unknown.eligible and failed.eligible
    record = await store.get("tenant-a", first.id)
    assert record.validation.status == "passed" and record.created_at == before.created_at
    assert len(await store.list_records("tenant-a", eligible_only=True)) == 1


async def test_failed_experience_is_retained_without_implicit_promotion(store):
    receipt = await store.record_experience(
        "tenant-a", "run-1", experience(), ValidationResult.failed("checker", "missing approval")
    )
    assert receipt.durable and not receipt.eligible
    repeated = await store.record_experience("tenant-a", "run-1", experience(), passed())
    assert repeated.status == "duplicate" and not repeated.eligible
    assert (await store.get("tenant-a", receipt.id)).validation.status == "failed"


async def test_same_run_changed_payload_is_rejected_and_preserves_original(store):
    first = await store.record_experience("tenant-a", "run-1", experience(), passed())
    changed = ExperienceInput(task="Handle a refund", answer="Refund sent without approval")
    rejected = await store.record_experience("tenant-a", "run-1", changed, passed())
    assert rejected.status == "rejected" and rejected.id == first.id
    assert not rejected.durable and not rejected.retryable
    assert (await store.get("tenant-a", first.id)).experience == experience()
    assert len(await store.list_records("tenant-a")) == 1


async def test_experience_idempotence_and_validation_survive_restart(store):
    first = await store.record_experience("tenant-a", "run-1", experience(), passed())
    path = store.path
    await store.close()
    reopened = await SQLiteStore.open(path)
    try:
        duplicate = await reopened.record_experience("tenant-a", "run-1", experience(), passed())
        assert duplicate.status == "duplicate" and duplicate.id == first.id and duplicate.eligible
        assert len(await reopened.list_records("tenant-a", eligible_only=True)) == 1
    finally:
        await reopened.close()


async def test_delete_is_durable_and_allows_reingestion(store):
    known = await store.upsert_knowledge("tenant-a", knowledge())
    learned = await store.record_experience("tenant-a", "run-1", experience(), passed())
    assert await store.delete("tenant-a", known.id)
    assert await store.delete("tenant-a", learned.id)
    assert not await store.delete("tenant-a", learned.id)
    path = store.path
    await store.close()
    reopened = await SQLiteStore.open(path)
    try:
        assert await reopened.list_records("tenant-a") == []
        assert (await reopened.upsert_knowledge("tenant-a", knowledge())).status == "stored"
        fresh = await reopened.record_experience("tenant-a", "run-1", experience(), passed())
        assert fresh.status == "stored" and fresh.id != learned.id
    finally:
        await reopened.close()


async def test_clear_is_durable_and_preserves_other_scopes(store):
    await store.upsert_knowledge("tenant-a", knowledge())
    await store.record_experience("tenant-a", "run-1", experience(), passed())
    await store.upsert_knowledge("tenant-b", knowledge())
    assert await store.clear("tenant-a") == 2
    assert await store.clear("tenant-a") == 0
    path = store.path
    await store.close()
    reopened = await SQLiteStore.open(path)
    try:
        assert await reopened.list_records("tenant-a") == []
        assert len(await reopened.list_records("tenant-b")) == 1
        assert (await reopened.record_experience(
            "tenant-a", "run-1", experience(), passed()
        )).status == "stored"
    finally:
        await reopened.close()


async def test_knowledge_cannot_overwrite_an_experience_record_id(store):
    recorded = await store.record_experience("tenant-a", "run-1", experience(), passed())
    rejected = await store.upsert_knowledge("tenant-a", knowledge(id=recorded.id))
    assert rejected.status == "rejected" and not rejected.durable
    assert (await store.get("tenant-a", recorded.id)).kind == "experience"


async def test_open_rejects_independent_owner_and_close_releases_lock(store):
    with pytest.raises(StoreInUseError):
        await SQLiteStore.open(store.path)
    with pytest.raises(StoreInUseError):
        await SQLiteStore.open(store.path.parent / "." / store.path.name)
    await store.close()
    await store.close()
    reopened = await SQLiteStore.open(store.path)
    await reopened.close()
    assert store.path.with_name(store.path.name + ".lock").exists()
    with pytest.raises(StoreClosedError):
        await store.list_records("tenant-a")
    with pytest.raises(StoreClosedError):
        await store.upsert_knowledge("tenant-a", knowledge())


async def test_owner_lock_blocks_a_separate_process(store):
    script = (
        "import fcntl, sys\n"
        "with open(sys.argv[1], 'r+') as lock:\n"
        "    try:\n"
        "        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "    except BlockingIOError:\n"
        "        sys.exit(0)\n"
        "    sys.exit(1)\n"
    )
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script, str(store.path) + ".lock"],
        capture_output=True, text=True, check=False, timeout=5,
    )
    assert result.returncode == 0, result.stderr


async def test_cancelled_open_releases_ownership_after_initialization(tmp_path, monkeypatch):
    original = SQLiteStore._initialize
    started = threading.Event()
    finish = threading.Event()

    def slow_initialize(connection):
        started.set()
        assert finish.wait(timeout=5)
        original(connection)

    monkeypatch.setattr(SQLiteStore, "_initialize", staticmethod(slow_initialize))
    path = tmp_path / "cancelled.sqlite"
    opening = asyncio.create_task(SQLiteStore.open(path))
    assert await asyncio.to_thread(started.wait, 5)
    opening.cancel()
    await asyncio.sleep(0)
    opening.cancel()
    await asyncio.sleep(0)
    assert not opening.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await opening
    reopened = await SQLiteStore.open(path)
    await reopened.close()


async def test_concurrent_recording_is_serialized_and_idempotent(store):
    receipts = await asyncio.gather(*(
        store.record_experience("tenant-a", "run-1", experience(), passed()) for _ in range(20)
    ))
    assert sum(r.status == "stored" for r in receipts) == 1
    assert sum(r.status == "duplicate" for r in receipts) == 19
    assert len({r.id for r in receipts}) == 1
    await asyncio.gather(*(
        store.upsert_knowledge("tenant-a", knowledge(id=f"policy-{i}")) for i in range(20)
    ))
    assert len(await store.list_records("tenant-a")) == 21


async def test_transaction_failure_rolls_back_and_receipt_is_sanitized(store, monkeypatch):
    original = store._put

    def fail_after_insert(record, fingerprint=None):
        original(record, fingerprint)
        raise sqlite3.OperationalError("simulated database error with private content")

    monkeypatch.setattr(store, "_put", fail_after_insert)
    receipt = await store.upsert_knowledge("tenant-a", knowledge())
    assert receipt.status == "error" and not receipt.durable and receipt.retryable
    assert "private" not in receipt.error
    assert await store.list_records("tenant-a") == []
    monkeypatch.setattr(store, "_put", original)
    assert (await store.upsert_knowledge("tenant-a", knowledge())).status == "stored"


async def test_cancellation_preserves_transaction_serialization(store, monkeypatch):
    original = store._put
    started = threading.Event()
    finish = threading.Event()

    def slow_put(record, fingerprint=None):
        started.set()
        assert finish.wait(timeout=5)
        original(record, fingerprint)

    monkeypatch.setattr(store, "_put", slow_put)
    writing = asyncio.create_task(store.upsert_knowledge("tenant-a", knowledge()))
    assert await asyncio.to_thread(started.wait, 5)
    writing.cancel()
    await asyncio.sleep(0)
    writing.cancel()
    await asyncio.sleep(0)
    assert not writing.done()
    reading = asyncio.create_task(store.list_records("tenant-a"))
    await asyncio.sleep(0)
    assert not reading.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await writing
    assert len(await reading) == 1


async def test_schema_and_json_envelopes_are_versioned(store):
    await store.upsert_knowledge("tenant-a", knowledge())
    metadata = store._connection.execute(
        "SELECT value FROM store_metadata WHERE key = 'schema_version'"
    ).fetchone()
    payload = store._connection.execute("SELECT payload_json FROM memory_records").fetchone()
    assert metadata[0] == "1" and json.loads(payload[0])["json_version"] == 1


async def test_unsupported_schema_rejected_without_leaking_ownership(store):
    path = store.path
    await store.close()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("UPDATE store_metadata SET value = '999' WHERE key = 'schema_version'")
    with pytest.raises(MemoryConfigurationError, match="schema version"):
        await SQLiteStore.open(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("UPDATE store_metadata SET value = '1' WHERE key = 'schema_version'")
    reopened = await SQLiteStore.open(path)
    await reopened.close()


async def test_legacy_database_rejected_without_changing_bytes_or_journal_mode(tmp_path):
    path = tmp_path / "legacy.sqlite"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("CREATE TABLE cases(id TEXT PRIMARY KEY, payload TEXT)")
        connection.execute("INSERT INTO cases VALUES('old-case', 'original legacy data')")
    original = path.read_bytes()
    with pytest.raises(MemoryConfigurationError, match="new path"):
        await SQLiteStore.open(path)
    assert path.read_bytes() == original
    with closing(sqlite3.connect(path)) as connection, connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute("SELECT * FROM cases").fetchall() == [
            ("old-case", "original legacy data")
        ]
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall() == [("cases",)]


async def test_unrelated_or_non_sqlite_file_rejected_without_modification(tmp_path):
    unrelated = tmp_path / "unrelated.sqlite"
    with closing(sqlite3.connect(unrelated)) as connection, connection:
        connection.execute("CREATE VIEW application_view AS SELECT 1 AS value")
        connection.execute("CREATE TABLE sqlitex_application_table(value TEXT)")
    original = unrelated.read_bytes()
    with pytest.raises(MemoryConfigurationError):
        await SQLiteStore.open(unrelated)
    assert unrelated.read_bytes() == original
    other = tmp_path / "other.bin"
    other.write_bytes(b"not a sqlite database")
    with pytest.raises(MemoryConfigurationError):
        await SQLiteStore.open(other)
    assert other.read_bytes() == b"not a sqlite database"


async def test_corrupt_json_is_explicitly_detected(store):
    await store.upsert_knowledge("tenant-a", knowledge())
    store._connection.execute(
        "UPDATE memory_records SET payload_json = '{}' WHERE scope = 'tenant-a'"
    )
    with pytest.raises(StoreCorruptionError):
        await store.list_records("tenant-a")


@pytest.mark.parametrize("scope", ["", " "])
async def test_empty_scope_is_configuration_error(store, scope):
    with pytest.raises(MemoryConfigurationError):
        await store.upsert_knowledge(scope, knowledge())
    with pytest.raises(MemoryConfigurationError):
        await store.list_records(scope)


async def test_blank_run_id_is_configuration_error(store):
    with pytest.raises(MemoryConfigurationError):
        await store.record_experience("tenant-a", " ", experience(), passed())
