"""Authoritative SQLite storage for the agent-independent memory plug-in.

Each database has one owner at a time, enforced with an advisory file lock on
macOS/Linux. All operations on that owner's connection are serialized and run
off the event loop. The persistent lock sidecar must not be removed while any
owner or opener may be using it. Other SQLite clients bypassing this lock are
unsupported. Retrieval indexes are derived data and are not stored here.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from .models import (
    ExperienceInput,
    KnowledgeRecord,
    MemoryConfigurationError,
    MemoryRecord,
    RecordingReceipt,
    ValidationResult,
    utc_now,
)
from .provenance import knowledge_revision as _knowledge_revision

try:
    import fcntl
except ImportError:  # pragma: no cover - explicitly unsupported on Windows
    fcntl = None  # type: ignore[assignment]

_SCHEMA_VERSION = 1
_JSON_VERSION = 1
_T = TypeVar("_T")
_UNAVAILABLE_REVISION = "unavailable"
_TABLE_SQL = {
    "store_metadata": "CREATE TABLE store_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "memory_records": (
        "CREATE TABLE memory_records ("
        "scope TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL, run_id TEXT, "
        "payload_json TEXT NOT NULL, fingerprint TEXT, PRIMARY KEY(scope, id), "
        "UNIQUE(scope, run_id), CHECK(kind IN ('knowledge', 'experience')), "
        "CHECK((kind = 'knowledge' AND run_id IS NULL) OR "
        "(kind = 'experience' AND run_id IS NOT NULL)))"
    ),
}


def _schema_tokens(sql: str) -> tuple[str, ...]:
    # This is an allowlist of the schema we create, not a general SQL parser.
    # Ignore formatting/keyword case, preserving string literals verbatim.
    return tuple(
        token if token.startswith("'") else token.casefold()
        for token in re.findall(r"'(?:''|[^'])*'|[A-Za-z_][A-Za-z_0-9]*|[^\s]", sql)
    )


class StoreInUseError(MemoryConfigurationError):
    """Another independently opened store already owns this database."""


class StoreClosedError(RuntimeError):
    """An operation was attempted after explicit store closure."""


class StoreCorruptionError(RuntimeError):
    """A durable record or its version envelope cannot be read safely."""


def _nonblank(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MemoryConfigurationError(f"{name} must be a nonblank string")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(experience: ExperienceInput) -> str:
    return hashlib.sha256(_canonical(experience.model_dump(mode="json")).encode()).hexdigest()


def _encode(record: MemoryRecord) -> str:
    return _canonical({
        "json_version": _JSON_VERSION,
        "record": record.model_dump(mode="json", exclude={"stale"}),
    })


def _decode(row: sqlite3.Row) -> MemoryRecord:
    try:
        envelope = json.loads(row["payload_json"])
        if envelope["json_version"] != _JSON_VERSION:
            raise ValueError("unsupported JSON version")
        record = MemoryRecord.model_validate(envelope["record"])
        if (record.scope, record.id, record.kind, record.run_id) != (
            row["scope"], row["id"], row["kind"], row["run_id"]
        ):
            raise ValueError("record envelope does not match its key")
        return record
    except (KeyError, TypeError, ValueError) as exc:
        raise StoreCorruptionError("Stored memory record is invalid or unsupported") from exc


def _receipt(record: MemoryRecord, status: str) -> RecordingReceipt:
    return RecordingReceipt(
        id=record.id, status=status, durable=True, indexed=False, eligible=record.eligible
    )


async def _settle_worker(task: asyncio.Task[_T]) -> _T:
    """Finish cleanup even if the caller repeatedly requests cancellation.

    Used only after the first cancellation has already been caught. Awaiting a
    to_thread task without shielding would cancel its future while the thread
    continues, losing its result and allowing another connection operation.
    """
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


class SQLiteStore:
    """Durable, scoped records with idempotent recording and explicit ownership.

    Open using ``await SQLiteStore.open(path)`` and always ``await store.close()``.
    An instance may be shared by concurrent tasks in one event loop. Independent
    instances/processes must close the current owner before opening this file.
    """

    def __init__(self, path: Path, connection: sqlite3.Connection, lock_fd: int) -> None:
        self.path = path
        self._connection = connection
        self._lock_fd = lock_fd
        self._operation_lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def open(cls, path: str | Path) -> SQLiteStore:
        """Open a disk-backed store; reject concurrent owners and unknown schemas."""
        task = asyncio.create_task(asyncio.to_thread(cls._open_sync, path))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            # A worker thread keeps running after cancellation. Reap its ownership.
            try:
                store = await _settle_worker(task)
            except Exception:
                # _open_sync already released ownership on initialization failure.
                raise cancellation from None
            else:
                await _settle_worker(asyncio.create_task(store.close()))
            raise

    @classmethod
    def _open_sync(cls, path: str | Path) -> SQLiteStore:
        if fcntl is None:
            raise MemoryConfigurationError("SQLiteStore requires macOS/Linux file locking")
        if not str(path).strip() or str(path) == ":memory:":
            raise MemoryConfigurationError("SQLiteStore requires a durable database file path")
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(str(resolved) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        connection: sqlite3.Connection | None = None
        locked = False
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise StoreInUseError("Database already has an open SQLiteStore owner") from exc
                raise
            if resolved.exists():
                # A writable connection can checkpoint an existing WAL when it
                # closes, even if no write was requested. Validate through a
                # read-only connection so rejection preserves the source files.
                # Do not use immutable=1: it would hide committed WAL contents.
                inspection = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True)
                try:
                    cls._check_existing_schema(inspection)
                finally:
                    inspection.close()
            connection = sqlite3.connect(resolved, check_same_thread=False, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            cls._initialize(connection)
            return cls(resolved, connection, lock_fd)
        except BaseException:
            if connection is not None:
                connection.close()
            if locked:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            raise

    @staticmethod
    def _check_existing_schema(connection: sqlite3.Connection) -> None:
        try:
            objects = connection.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE name NOT GLOB 'sqlite_*'"
            ).fetchall()
            if not objects:
                return  # A new or deliberately empty database may be initialized.
            if {(row[0], row[1]) for row in objects} != {
                ("table", "store_metadata"), ("table", "memory_records")
            }:
                raise MemoryConfigurationError(
                    "Existing database is not a SQLiteStore; use a new path and import legacy data"
                )
            version = connection.execute(
                "SELECT value FROM store_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if version is None or version[0] != str(_SCHEMA_VERSION):
                raise MemoryConfigurationError("Unsupported SQLiteStore schema version")
            # Column collations are not exposed by table_info. A NOCASE scope
            # column can leak another scope even when its unique indexes use
            # BINARY. Require our known DDL, including CHECK constraints, before
            # allowing reads or writes through this store.
            for _, name, sql in objects:
                if _schema_tokens(sql) != _schema_tokens(_TABLE_SQL[name]):
                    raise MemoryConfigurationError("Unsupported SQLiteStore record schema")
            expected_columns = {
                "store_metadata": {
                    ("key", "TEXT", 0, None, 1), ("value", "TEXT", 1, None, 0),
                },
                "memory_records": {
                    ("scope", "TEXT", 1, None, 1), ("id", "TEXT", 1, None, 2),
                    ("kind", "TEXT", 1, None, 0), ("run_id", "TEXT", 0, None, 0),
                    ("payload_json", "TEXT", 1, None, 0),
                    ("fingerprint", "TEXT", 0, None, 0),
                },
            }
            for table, expected in expected_columns.items():
                columns = {
                    (row[1], row[2].upper(), row[3], row[4], row[5])
                    for row in connection.execute("SELECT * FROM pragma_table_info(?)", (table,))
                }
                if columns != expected:
                    raise MemoryConfigurationError("Unsupported SQLiteStore record schema")
            # Column names alone do not establish idempotence or scope isolation.
            # In particular ON CONFLICT(scope, id) requires the composite key,
            # and recording one result per run requires UNIQUE(scope, run_id).
            unique_keys = {
                tuple((column[2], column[4], column[3]) for column in connection.execute(
                    "SELECT * FROM pragma_index_xinfo(?) ORDER BY seqno", (index[1],)
                ) if column[5])
                for index in connection.execute("PRAGMA index_list(memory_records)")
                if index[2] and not index[4]
            }
            if unique_keys != {
                (("scope", "BINARY", 0), ("id", "BINARY", 0)),
                (("scope", "BINARY", 0), ("run_id", "BINARY", 0)),
            }:
                raise MemoryConfigurationError("Unsupported SQLiteStore record schema")
        except sqlite3.DatabaseError as exc:
            raise MemoryConfigurationError("Existing file is not a supported SQLiteStore") from exc

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                _TABLE_SQL["store_metadata"].replace(
                    "CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1
                )
            )
            version = connection.execute(
                "SELECT value FROM store_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if version is not None and version[0] != str(_SCHEMA_VERSION):
                raise MemoryConfigurationError("Unsupported SQLiteStore schema version")
            connection.execute(
                _TABLE_SQL["memory_records"].replace(
                    "CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1
                )
            )
            connection.execute(
                "INSERT OR IGNORE INTO store_metadata(key, value) VALUES('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    async def _run(self, operation: Callable[[], _T]) -> _T:
        async with self._operation_lock:
            if self._closed:
                raise StoreClosedError("SQLiteStore is closed")
            task = asyncio.create_task(asyncio.to_thread(operation))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError as cancellation:
                # Keep ownership until the worker has finished its transaction.
                try:
                    await _settle_worker(task)
                except Exception:
                    # The worker has rolled back; preserve the caller's cancellation.
                    raise cancellation from None
                raise

    async def _write(self, operation: Callable[[], RecordingReceipt]) -> RecordingReceipt:
        try:
            return await self._run(operation)
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            return RecordingReceipt(
                status="error",
                error="Memory storage write failed",
                retryable=isinstance(exc, (sqlite3.OperationalError, OSError)),
            )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def _put(self, record: MemoryRecord, fingerprint: str | None = None) -> None:
        self._connection.execute(
            "INSERT INTO memory_records(scope, id, kind, run_id, payload_json, fingerprint) "
            "VALUES(?, ?, ?, ?, ?, ?) ON CONFLICT(scope, id) DO UPDATE SET "
            "payload_json = excluded.payload_json, fingerprint = excluded.fingerprint",
            (record.scope, record.id, record.kind, record.run_id, _encode(record), fingerprint),
        )

    async def upsert_knowledge(self, scope: str, knowledge: KnowledgeRecord) -> RecordingReceipt:
        _nonblank(scope, "scope")
        # Snapshot caller-owned mutable payloads before scheduling a worker.
        knowledge = knowledge.model_copy(deep=True)

        def operation() -> RecordingReceipt:
            with self._transaction():
                row = self._connection.execute(
                    "SELECT * FROM memory_records WHERE scope = ? AND id = ?", (scope, knowledge.id)
                ).fetchone()
                previous = _decode(row) if row is not None else None
                if previous is not None and previous.kind != "knowledge":
                    return RecordingReceipt(
                        id=knowledge.id, status="rejected", error="Record ID belongs to experience"
                    )
                if previous is not None and _canonical(
                    previous.knowledge.model_dump(mode="json")
                ) == _canonical(knowledge.model_dump(mode="json")):
                    return _receipt(previous, "duplicate")
                now = utc_now()
                record = MemoryRecord(
                    id=knowledge.id, scope=scope, kind="knowledge", knowledge=knowledge,
                    created_at=previous.created_at if previous is not None else now, updated_at=now,
                )
                self._put(record)
            return _receipt(record, "stored")

        return await self._write(operation)

    async def record_experience(
        self, scope: str, run_id: str, experience: ExperienceInput, validation: ValidationResult,
        *, knowledge_revision: str | None = None,
    ) -> RecordingReceipt:
        _nonblank(scope, "scope")
        _nonblank(run_id, "run_id")
        experience = experience.model_copy(deep=True)
        validation = validation.model_copy(deep=True)

        def operation() -> RecordingReceipt:
            fingerprint = _fingerprint(experience)
            with self._transaction():
                revision, has_knowledge = self._knowledge_state_sync(scope)
                row = self._connection.execute(
                    "SELECT * FROM memory_records WHERE scope = ? AND run_id = ?", (scope, run_id)
                ).fetchone()
                if row is not None:
                    previous = self._derive_stale(_decode(row), revision, has_knowledge)
                    if row["fingerprint"] != fingerprint:
                        return RecordingReceipt(
                            id=previous.id, status="rejected",
                            error="Run ID already has a different experience payload",
                        )
                    if previous.validation.status == "unknown" and validation.status == "passed":
                        record = previous.model_copy(
                            update={"validation": validation, "updated_at": utc_now()}, deep=True
                        )
                        self._put(record, fingerprint)
                        result = _receipt(record, "promoted")
                    else:
                        result = _receipt(previous, "duplicate")
                else:
                    now = utc_now()
                    record = MemoryRecord(
                        id=str(uuid.uuid4()), scope=scope, kind="experience", experience=experience,
                        validation=validation, run_id=run_id, created_at=now, updated_at=now,
                        knowledge_revision=(
                            knowledge_revision if knowledge_revision is not None else revision
                        ),
                    )
                    self._derive_stale(record, revision, has_knowledge)
                    self._put(record, fingerprint)
                    result = _receipt(record, "stored")
            return result

        return await self._write(operation)

    def _knowledge_state_sync(self, scope: str) -> tuple[str, bool]:
        # An unrelated damaged experience must not prevent a new valid write.
        rows = self._connection.execute(
            "SELECT * FROM memory_records WHERE scope = ? AND kind = 'knowledge' ORDER BY id",
            (scope,),
        ).fetchall()
        try:
            return _knowledge_revision(_decode(row) for row in rows), bool(rows)
        except StoreCorruptionError:
            # Preserve the host's new experience even when a reference cannot
            # be decoded, but never claim its reference provenance is verified.
            return _UNAVAILABLE_REVISION, bool(rows)

    async def knowledge_revision(self, scope: str) -> str:
        """Return the current reference state without reading experience payloads."""
        _nonblank(scope, "scope")
        return await self._run(lambda: self._knowledge_state_sync(scope)[0])

    @staticmethod
    def _derive_stale(record: MemoryRecord, revision: str, has_knowledge: bool) -> MemoryRecord:
        if record.kind == "experience":
            # Older envelopes have no provenance. They remain usable only when
            # there are no references on which their result could depend.
            record.stale = (
                revision == _UNAVAILABLE_REVISION
                or record.knowledge_revision == _UNAVAILABLE_REVISION
                or (record.knowledge_revision != revision
                    if record.knowledge_revision is not None else has_knowledge)
            )
        return record

    async def list_records(self, scope: str, eligible_only: bool = False) -> list[MemoryRecord]:
        _nonblank(scope, "scope")

        def operation() -> list[MemoryRecord]:
            rows = self._connection.execute(
                "SELECT * FROM memory_records WHERE scope = ? ORDER BY id", (scope,)
            ).fetchall()
            records = [_decode(row) for row in rows]
            revision = _knowledge_revision(records)
            has_knowledge = any(record.kind == "knowledge" for record in records)
            for record in records:
                self._derive_stale(record, revision, has_knowledge)
            return [r for r in records if r.eligible] if eligible_only else records

        return await self._run(operation)

    async def get(self, scope: str, id: str) -> MemoryRecord | None:
        _nonblank(scope, "scope")
        _nonblank(id, "id")

        def operation() -> MemoryRecord | None:
            row = self._connection.execute(
                "SELECT * FROM memory_records WHERE scope = ? AND id = ?", (scope, id)
            ).fetchone()
            if row is None:
                return None
            record = _decode(row)
            if record.kind == "experience":
                revision, has_knowledge = self._knowledge_state_sync(scope)
                self._derive_stale(record, revision, has_knowledge)
            return record

        return await self._run(operation)

    async def delete(self, scope: str, id: str) -> bool:
        _nonblank(scope, "scope")
        _nonblank(id, "id")

        def operation() -> bool:
            with self._transaction():
                cursor = self._connection.execute(
                    "DELETE FROM memory_records WHERE scope = ? AND id = ?", (scope, id)
                )
            return cursor.rowcount > 0

        return await self._run(operation)

    async def clear(self, scope: str) -> int:
        _nonblank(scope, "scope")

        def operation() -> int:
            with self._transaction():
                cursor = self._connection.execute(
                    "DELETE FROM memory_records WHERE scope = ?", (scope,)
                )
            return cursor.rowcount

        return await self._run(operation)

    async def close(self) -> None:
        """Finish queued work, close the connection, and release ownership once."""
        async with self._operation_lock:
            if self._closed:
                return
            self._closed = True

            def operation() -> None:
                try:
                    self._connection.close()
                finally:
                    try:
                        if fcntl is not None:
                            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                    finally:
                        os.close(self._lock_fd)

            task = asyncio.create_task(asyncio.to_thread(operation))
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as cancellation:
                try:
                    await _settle_worker(task)
                except Exception:
                    raise cancellation from None
                raise
