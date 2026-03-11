"""Core type definitions for AgentMemento."""

from __future__ import annotations

from enum import Enum
from typing import Any
from datetime import datetime, timezone

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    """Outcome of a completed task."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class MemoryDomain(str, Enum):
    """High-level domain for hierarchical memory classification."""
    CODING = "coding"
    MATH = "math"
    RESEARCH = "research"
    WEB = "web"
    DATA = "data"
    REASONING = "reasoning"
    FINANCE = "finance"
    REAL_ESTATE = "real_estate"
    GENERAL = "general"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ---------------------------------------------------------------------------
# Message primitives
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in a conversation."""
    role: MessageRole
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    # Populated on ASSISTANT messages that contain tool calls (needed for Anthropic/OpenAI)
    tool_calls: list["ToolCall"] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool primitives
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A request to execute a tool."""
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Result from a tool execution."""
    tool_call_id: str
    tool_name: str
    output: Any
    error: str | None = None
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Trajectory — one step in a task execution
# ---------------------------------------------------------------------------


class Step(BaseModel):
    """One atomic step: thought → tool call → result."""
    index: int
    thought: str
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    observation: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Trajectory(BaseModel):
    """Full execution trace for a task."""
    task: str
    steps: list[Step] = Field(default_factory=list)
    final_answer: str = ""
    status: TaskStatus = TaskStatus.PARTIAL
    total_tokens: int = 0
    duration_ms: float = 0.0
    reflection: str = ""           # filled by ReflexionEngine on failure
    reward: float = 0.0            # filled by outcome evaluator

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def tool_names_used(self) -> list[str]:
        return [s.tool_call.name for s in self.steps if s.tool_call]


# ---------------------------------------------------------------------------
# LLM response
# ---------------------------------------------------------------------------


class LLMResponse(BaseModel):
    """Normalised response from any LLM backend."""
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    stop_reason: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
