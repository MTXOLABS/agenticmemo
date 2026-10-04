"""Opening a database must validate its durable schema without changing it."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from agenticmemo.plugin.models import (
    ExperienceInput,
    KnowledgeRecord,
    MemoryConfigurationError,
    ValidationResult,
)
from agenticmemo.plugin.storage import SQLiteStore

_METADATA = "CREATE TABLE store_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
_COLUMNS = (
    "scope TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL, run_id TEXT, "
    "payload_json TEXT NOT NULL, fingerprint TEXT"
)
_RECORDS = (
    f"CREATE TABLE memory_records({_COLUMNS}, PRIMARY KEY(scope, id), UNIQUE(scope, run_id), "
    "CHECK(kind IN ('knowledge', 'experience')), "
    "CHECK((kind = 'knowledge' AND run_id IS NULL) OR "
    "(kind = 'experience' AND run_id IS NOT NULL)))"
)


def _copy_committed_wal(origin: Path, destination: Path) -> None:
    """Copy a quiescent writer's files, simulating a stopped process before checkpoint."""
    shutil.copyfile(origin, destination)
    shutil.copyfile(str(origin) + "-wal", str(destination) + "-wal")
    assert Path(str(destination) + "-wal").stat().st_size > 32


def _data_files(path: Path) -> tuple[bytes, bytes]:
    return path.read_bytes(), Path(str(path) + "-wal").read_bytes()


@pytest.mark.parametrize("schema", ["legacy", "future"])
async def test_rejected_committed_wal_preserves_main_and_wal_bytes(tmp_path, monkeypatch, schema):
    origin = tmp_path / "writer.sqlite"
    # URI metacharacters in filenames must still resolve to the intended file.
    path = tmp_path / "snapshot #1?.sqlite"
    writer = sqlite3.connect(origin)
    try:
        if schema == "future":
            writer.execute(_METADATA)
            writer.execute(_RECORDS)
            writer.execute("INSERT INTO store_metadata VALUES('schema_version', '1')")
            writer.commit()
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        if schema == "legacy":
            writer.execute("CREATE TABLE cases(id TEXT PRIMARY KEY, data TEXT)")
            writer.execute("INSERT INTO cases VALUES('old-case', 'original data')")
        else:
            # The main file says version 1; only the WAL says unsupported version.
            writer.execute("UPDATE store_metadata SET value='999'")
        writer.commit()
        _copy_committed_wal(origin, path)
    finally:
        writer.close()
    before = _data_files(path)
    connect = sqlite3.connect
    attempts = []

    def checked_connect(database, *args, **kwargs):
        attempts.append((database, kwargs))
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", checked_connect)
    # Repeating rejection also checks that the failed open releases ownership.
    for _ in range(2):
        with pytest.raises(MemoryConfigurationError):
            await SQLiteStore.open(path)
        assert _data_files(path) == before
    assert all(kwargs.get("uri") and "mode=ro" in str(db) for db, kwargs in attempts)


@pytest.mark.parametrize("records,metadata", [
    (f"CREATE TABLE memory_records({_COLUMNS})", _METADATA),
    (f"CREATE TABLE memory_records({_COLUMNS}, PRIMARY KEY(scope, id))", _METADATA),
    (f"CREATE TABLE memory_records({_COLUMNS}, UNIQUE(scope, run_id))", _METADATA),
    (f"CREATE TABLE memory_records({_COLUMNS}, PRIMARY KEY(id, scope), "
     "UNIQUE(scope, run_id))", _METADATA),
    (f"CREATE TABLE memory_records({_COLUMNS}, PRIMARY KEY(scope, id), "
     "UNIQUE(scope COLLATE NOCASE, run_id))", _METADATA),
    (_RECORDS.replace("scope TEXT NOT NULL", "scope TEXT"), _METADATA),
    (_RECORDS, "CREATE TABLE store_metadata(key TEXT, value TEXT NOT NULL)"),
    (_RECORDS, "CREATE TABLE store_metadata(key TEXT PRIMARY KEY, value TEXT)"),
])
async def test_malformed_constraints_rejected_before_writable_open(
    tmp_path, monkeypatch, records, metadata
):
    path = tmp_path / "invalid-schema.sqlite"
    creator = sqlite3.connect(path)
    try:
        creator.execute(metadata)
        creator.execute(records)
        creator.execute("INSERT INTO store_metadata VALUES('schema_version', '1')")
        creator.commit()
    finally:
        creator.close()
    before = path.read_bytes()
    connect = sqlite3.connect
    modes = []

    def checked_connect(database, *args, **kwargs):
        modes.append(kwargs.get("uri") and "mode=ro" in str(database))
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", checked_connect)
    with pytest.raises(MemoryConfigurationError, match="schema"):
        await SQLiteStore.open(path)
    assert modes == [True]
    assert path.read_bytes() == before
    assert not Path(str(path) + "-wal").exists()


async def test_supported_database_recovers_records_committed_only_in_wal(tmp_path):
    origin = tmp_path / "writer.sqlite"
    path = tmp_path / "recovered.sqlite"
    initial = await SQLiteStore.open(origin)
    try:
        await initial.upsert_knowledge(
            "scope", KnowledgeRecord(id="policy", content="first value", source="fixture")
        )
    finally:
        await initial.close()
    writer = sqlite3.connect(origin)
    try:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "UPDATE memory_records SET payload_json = "
            "replace(payload_json, 'first value', 'committed value')"
        )
        writer.commit()
        _copy_committed_wal(origin, path)
    finally:
        writer.close()
    recovered = await SQLiteStore.open(path)
    try:
        record = await recovered.get("scope", "policy")
        assert record.knowledge.content == "committed value"
        receipt = await recovered.upsert_knowledge(
            "scope", KnowledgeRecord(id="second", content="new value", source="fixture")
        )
        assert receipt.durable
    finally:
        await recovered.close()
    reopened = await SQLiteStore.open(path)
    try:
        assert len(await reopened.list_records("scope")) == 2
    finally:
        await reopened.close()


async def test_new_write_revision_does_not_decode_an_unrelated_corrupt_experience(tmp_path):
    store = await SQLiteStore.open(tmp_path / "memory.sqlite")
    try:
        await store.upsert_knowledge(
            "scope", KnowledgeRecord(id="policy", content="current policy", source="fixture")
        )
        first = await store.record_experience(
            "scope", "broken-run", ExperienceInput(task="first task"), ValidationResult.unknown()
        )
        store._connection.execute(
            "UPDATE memory_records SET payload_json='{}' WHERE id=?", (first.id,)
        )
        revision = await store.knowledge_revision("scope")
        receipt = await store.record_experience(
            "scope", "valid-run", ExperienceInput(task="new task"),
            ValidationResult.passed("checker", ["checked"]), knowledge_revision=revision,
        )
        assert receipt.durable and receipt.eligible
        assert (await store.get("scope", receipt.id)).knowledge_revision == revision
    finally:
        await store.close()


async def test_promotion_and_duplicate_never_rebase_the_original_revision(tmp_path):
    store = await SQLiteStore.open(tmp_path / "memory.sqlite")
    try:
        await store.upsert_knowledge(
            "scope", KnowledgeRecord(id="policy", content="30 days", source="fixture")
        )
        original_revision = await store.knowledge_revision("scope")
        experience = ExperienceInput(task="refund policy", answer="30 days")
        original = await store.record_experience(
            "scope", "run", experience, ValidationResult.unknown()
        )
        await store.upsert_knowledge(
            "scope", KnowledgeRecord(id="policy", content="14 days", source="fixture")
        )
        current_revision = await store.knowledge_revision("scope")
        assert original_revision != current_revision
        verdict = ValidationResult.passed("checker", ["checked original answer"])
        promoted = await store.record_experience(
            "scope", "run", experience, verdict, knowledge_revision=current_revision,
        )
        assert promoted.status == "promoted" and not promoted.eligible
        duplicate = await store.record_experience(
            "scope", "run", experience, verdict, knowledge_revision=current_revision,
        )
        assert duplicate.status == "duplicate" and not duplicate.eligible
        record = await store.get("scope", original.id)
        assert record.knowledge_revision == original_revision and record.stale
        assert not record.eligible
        assert [r.kind for r in await store.list_records("scope", eligible_only=True)] == [
            "knowledge"
        ]
    finally:
        await store.close()


async def test_recording_with_outdated_context_is_durable_but_ineligible(tmp_path):
    store = await SQLiteStore.open(tmp_path / "memory.sqlite")
    try:
        old_revision = await store.knowledge_revision("scope")
        await store.upsert_knowledge(
            "scope", KnowledgeRecord(id="policy", content="new policy", source="fixture")
        )
        receipt = await store.record_experience(
            "scope", "run", ExperienceInput(task="task with older reference state"),
            ValidationResult.passed("checker", ["checked"]), knowledge_revision=old_revision,
        )
        assert receipt.durable and not receipt.eligible
        record = await store.get("scope", receipt.id)
        assert record.stale and record.knowledge_revision == old_revision
        # Staleness is recalculated from the current knowledge, not persisted as a verdict.
        payload = store._connection.execute(
            "SELECT payload_json FROM memory_records WHERE id=?", (receipt.id,)
        ).fetchone()[0]
        assert '"stale"' not in payload
    finally:
        await store.close()


async def test_case_insensitive_scope_column_rejected_despite_binary_indexes(tmp_path):
    path = tmp_path / "unsafe-scope.sqlite"
    schema = _RECORDS.replace("scope TEXT NOT NULL", "scope TEXT COLLATE NOCASE NOT NULL")
    schema = schema.replace("PRIMARY KEY(scope, id)", "PRIMARY KEY(scope COLLATE BINARY, id)")
    schema = schema.replace("UNIQUE(scope, run_id)", "UNIQUE(scope COLLATE BINARY, run_id)")
    creator = sqlite3.connect(path)
    try:
        creator.execute(_METADATA)
        creator.execute(schema)
        creator.execute("INSERT INTO store_metadata VALUES('schema_version', '1')")
        creator.execute(
            "INSERT INTO memory_records(scope, id, kind, payload_json) "
            "VALUES('TenantA', 'policy', 'knowledge', '{}')"
        )
        creator.commit()
        # Establish the actual isolation defect that index inspection cannot detect.
        assert creator.execute(
            "SELECT scope FROM memory_records WHERE scope='tenanta'"
        ).fetchall() == [("TenantA",)]
        for index in creator.execute("PRAGMA index_list(memory_records)"):
            assert all(row[4] == "BINARY" for row in creator.execute(
                "SELECT * FROM pragma_index_xinfo(?)", (index[1],)
            ))
    finally:
        creator.close()
    before = path.read_bytes()
    with pytest.raises(MemoryConfigurationError, match="schema"):
        await SQLiteStore.open(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("validation", [
    ValidationResult.unknown(), ValidationResult.passed("checker", ["checked result"]),
])
async def test_corrupt_reference_allows_durable_capture_but_never_eligible_experience(
    tmp_path, validation
):
    store = await SQLiteStore.open(tmp_path / "memory.sqlite")
    try:
        await store.upsert_knowledge(
            "scope", KnowledgeRecord(id="policy", content="policy", source="fixture")
        )
        store._connection.execute("UPDATE memory_records SET payload_json='{}' WHERE id='policy'")
        unavailable = await store.knowledge_revision("scope")
        receipt = await store.record_experience(
            "scope", "run", ExperienceInput(task="host still completed"), validation,
            knowledge_revision=unavailable,
        )
        assert receipt.durable and not receipt.eligible
        record = await store.get("scope", receipt.id)
        assert record.stale and not record.eligible
        assert await store.delete("scope", "policy")
        # Repairing/deleting the reference cannot invent provenance for this run.
        record = await store.get("scope", receipt.id)
        assert record.stale and not record.eligible
        assert await store.list_records("scope", eligible_only=True) == []
    finally:
        await store.close()
