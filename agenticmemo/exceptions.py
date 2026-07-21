"""AgenticMemo custom exceptions."""


class AgenticMemoError(Exception):
    """Base exception for all AgenticMemo errors."""


class LLMError(AgenticMemoError):
    """Raised when an LLM call fails."""


class MemoryError(AgenticMemoError):
    """Raised when a memory operation fails."""


class RetrievalError(AgenticMemoError):
    """Raised when case retrieval fails."""


class ToolError(AgenticMemoError):
    """Raised when a tool execution fails."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class PlannerError(AgenticMemoError):
    """Raised when planning fails."""


class ExecutorError(AgenticMemoError):
    """Raised when execution fails."""


class FilterError(AgenticMemoError):
    """Raised when trajectory filtering fails."""


class EmbeddingError(AgenticMemoError):
    """Raised when embedding computation fails."""


class PolicyError(AgenticMemoError):
    """Raised when policy update fails."""


class ConfigError(AgenticMemoError):
    """Raised on invalid configuration."""


# Backward-compat alias for the pre-rebrand name
AgentMementoError = AgenticMemoError
