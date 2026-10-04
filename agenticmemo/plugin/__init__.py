"""Standalone memory for existing agents; no bundled execution loop required."""

from .adapter import MemoryAgent, WrappedResult
from .memory import AgentMemory
from .models import (
    ContextHandle,
    ExperienceInput,
    IngestReport,
    KnowledgeRecord,
    MemoryConfigurationError,
    MemoryContext,
    MemoryHit,
    MemoryRecord,
    RecordingReceipt,
    ValidationResult,
)

__all__ = [
    "AgentMemory", "ContextHandle", "ExperienceInput", "IngestReport", "KnowledgeRecord",
    "MemoryConfigurationError", "MemoryContext", "MemoryHit", "MemoryRecord",
    "RecordingReceipt", "ValidationResult",
    "MemoryAgent", "WrappedResult",
]
