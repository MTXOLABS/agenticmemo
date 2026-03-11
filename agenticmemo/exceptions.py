"""AgentMemento custom exceptions."""


class AgentMementoError(Exception):
    """Base exception for all AgentMemento errors."""


class LLMError(AgentMementoError):
    """Raised when an LLM call fails."""


class MemoryError(AgentMementoError):
    """Raised when a memory operation fails."""


class RetrievalError(AgentMementoError):
    """Raised when case retrieval fails."""


class ToolError(AgentMementoError):
    """Raised when a tool execution fails."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class PlannerError(AgentMementoError):
    """Raised when planning fails."""


class ExecutorError(AgentMementoError):
    """Raised when execution fails."""


class FilterError(AgentMementoError):
    """Raised when trajectory filtering fails."""


class EmbeddingError(AgentMementoError):
    """Raised when embedding computation fails."""


class PolicyError(AgentMementoError):
    """Raised when policy update fails."""


class ConfigError(AgentMementoError):
    """Raised on invalid configuration."""
