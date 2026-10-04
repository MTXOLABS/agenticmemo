"""Escape — Advanced agentic learning without LLM fine-tuning.

Key innovations over original Memento:
  - Temporal Knowledge Graph memory (vs flat Case Bank)
  - Hierarchical 4-layer memory organisation (H-MEM style)
  - GRPO retrieval policy (vs soft Q-learning)
  - Ensemble retrieval: semantic + BM25 + graph + temporal
  - Reflexion failure loop (CLEANER + Reflexion)
  - Trajectory quality filtering pipeline

Quick start::

    import asyncio
    from agenticmemo import Agent
    from agenticmemo.tools import PythonReplTool

    async def main():
        agent = Agent.from_anthropic(api_key="sk-ant-...")
        agent.add_tool(PythonReplTool())
        result = await agent.run("Write a Python function to check if a number is prime")
        print(result.final_answer)

    asyncio.run(main())
"""

from .config import AgentConfig, LearningConfig, MemoryConfig, RetrievalConfig
from .core import Agent, Executor, Planner
from .exceptions import (
    AgenticMemoError,
    LLMError,
    RetrievalError,
    ToolError,
)
from .learning import GRPOPolicy, OutcomeVerifier, ReflexionEngine, TrajectoryFilter
from .llm import AnthropicLLM, LLMBackend, OpenAILLM
from .memory import Case, CaseOutcome, HierarchicalMemory, TemporalGraphMemory
from .plugin import AgentMemory, ExperienceInput, KnowledgeRecord, ValidationResult
from .retrieval import EnsembleRetriever, SentenceTransformerEmbeddings
from .tools import (
    FileReadTool,
    FileWriteTool,
    PythonReplTool,
    Tool,
    ToolRegistry,
    WebSearchTool,
    tool,
)
from .types import (
    LLMResponse,
    MemoryDomain,
    Message,
    MessageRole,
    Step,
    TaskStatus,
    ToolCall,
    ToolResult,
    Trajectory,
)
from .version import __author__, __license__, __version__

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__license__",
    # Config
    "AgentConfig",
    "AgentMemory",
    "ExperienceInput",
    "KnowledgeRecord",
    "ValidationResult",
    "MemoryConfig",
    "RetrievalConfig",
    "LearningConfig",
    # Types
    "Message",
    "MessageRole",
    "ToolCall",
    "ToolResult",
    "Step",
    "Trajectory",
    "LLMResponse",
    "TaskStatus",
    "MemoryDomain",
    # Exceptions
    "AgenticMemoError",
    "LLMError",
    "ToolError",
    "RetrievalError",
    # LLM
    "LLMBackend",
    "AnthropicLLM",
    "OpenAILLM",
    # Memory
    "Case",
    "CaseOutcome",
    "HierarchicalMemory",
    "TemporalGraphMemory",
    # Retrieval
    "EnsembleRetriever",
    "SentenceTransformerEmbeddings",
    # Learning
    "TrajectoryFilter",
    "ReflexionEngine",
    "GRPOPolicy",
    "OutcomeVerifier",
    # Tools
    "Tool",
    "tool",
    "ToolRegistry",
    "WebSearchTool",
    "PythonReplTool",
    "FileReadTool",
    "FileWriteTool",
    # Core
    "Agent",
    "Planner",
    "Executor",
]
