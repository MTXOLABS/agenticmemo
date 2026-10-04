"""Public contracts for memory attached to an existing agent."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeRecord(BaseModel):
    """Reference material supplied by the application, not a completed task."""

    model_config = ConfigDict(revalidate_instances="always")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1)
    content: str = Field(min_length=1, max_length=100_000)
    source: str = Field(min_length=1)
    version: str = "1"
    metadata: dict[str, Any] = Field(default_factory=dict)
    valid_until: datetime | None = None

    @field_validator("id", "content", "source")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("valid_until")
    @classmethod
    def timezone_required(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("valid_until must include a timezone")
        return value


class ExperienceInput(BaseModel):
    """Application-approved result and observable actions; no private reasoning required."""

    # Callers may mutate a model, its action list, or construct it without validation.
    # Recheck all fields whenever it crosses an adapter or persistence boundary.
    model_config = ConfigDict(revalidate_instances="always")

    task: str = Field(min_length=1)
    answer: str = ""
    actions: list[dict[str, Any]] = Field(default_factory=list)
    solution: str = ""
    plan: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """A verdict from an application-owned validator, never the agent's self-report."""

    model_config = ConfigDict(revalidate_instances="always")

    status: Literal["passed", "failed", "unknown"] = "unknown"
    validator: str = ""
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def positive_evidence_required(self) -> ValidationResult:
        if self.status == "passed" and (
            not self.validator.strip() or not any(e.strip() for e in self.evidence)
        ):
            raise ValueError("passed validation requires a validator and evidence")
        return self

    @classmethod
    def passed(cls, validator: str, evidence: list[str], reason: str = "") -> ValidationResult:
        return cls(status="passed", validator=validator, evidence=evidence, reason=reason)

    @classmethod
    def failed(cls, validator: str, reason: str, evidence: list[str] | None = None
               ) -> ValidationResult:
        return cls(status="failed", validator=validator, reason=reason, evidence=evidence or [])

    @classmethod
    def unknown(cls, reason: str = "No validation supplied") -> ValidationResult:
        return cls(reason=reason)


class MemoryRecord(BaseModel):
    """Versioned storage envelope; knowledge and experience have distinct payloads."""

    id: str
    scope: str
    kind: Literal["knowledge", "experience"]
    knowledge: KnowledgeRecord | None = None
    experience: ExperienceInput | None = None
    validation: ValidationResult = Field(default_factory=ValidationResult.unknown)
    run_id: str | None = None
    knowledge_revision: str | None = None
    stale: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def matching_payload(self) -> MemoryRecord:
        if self.kind == "knowledge" and (self.knowledge is None or self.experience is not None):
            raise ValueError("knowledge records require only a knowledge payload")
        if self.kind == "experience" and (self.experience is None or self.knowledge is not None):
            raise ValueError("experience records require only an experience payload")
        return self

    @property
    def eligible(self) -> bool:
        if self.knowledge is not None:
            return self.knowledge.valid_until is None or self.knowledge.valid_until > utc_now()
        return self.validation.status == "passed" and not self.stale


class RecordingReceipt(BaseModel):
    id: str | None = None
    status: Literal["stored", "duplicate", "promoted", "rejected", "error", "deleted"]
    durable: bool = False
    indexed: bool = False
    eligible: bool = False
    error: str | None = None
    retryable: bool = False


class IngestReport(BaseModel):
    receipts: list[RecordingReceipt] = Field(default_factory=list)

    @property
    def inserted(self) -> int:
        return sum(r.status == "stored" for r in self.receipts)

    @property
    def duplicates(self) -> int:
        return sum(r.status == "duplicate" for r in self.receipts)

    @property
    def rejected(self) -> int:
        return sum(r.status in {"rejected", "error"} for r in self.receipts)


class ContextHandle(BaseModel):
    scope: str
    run_id: str
    task: str
    record_ids: list[str] = Field(default_factory=list)
    knowledge_revision: str | None = None


class MemoryHit(BaseModel):
    record_id: str
    kind: Literal["knowledge", "experience"]
    score: float
    text: str
    source: str


class MemoryContext(BaseModel):
    text: str = ""
    hits: list[MemoryHit] = Field(default_factory=list)
    handle: ContextHandle
    degraded: bool = False
    diagnostics: list[str] = Field(default_factory=list)


class MemoryConfigurationError(ValueError):
    """Invalid integration input; do not silently turn this into degraded retrieval."""
