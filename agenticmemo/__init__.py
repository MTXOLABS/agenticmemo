"""AgentMemento — Advanced agentic learning without LLM fine-tuning.

Key innovations over original Memento:
  - Temporal Knowledge Graph memory (vs flat Case Bank)
  - Hierarchical 4-layer memory organisation (H-MEM style)
  - GRPO retrieval policy (vs soft Q-learning)
  - Ensemble retrieval: semantic + BM25 + graph + temporal
  - Reflexion failure loop (CLEANER + Reflexion)
  - Trajectory quality filtering pipeline

Quick start::

    import asyncio
    from agentmemento import Agent
    from agenticmemo.tools import PythonReplTool

    async def main():
        agent = Agent.from_anthropic(api_key="sk-ant-...")
        agent.add_tool(PythonReplTool())
        result = await agent.run("Write a Python function to check if a number is prime")
        print(result.final_answer)

    asyncio.run(main())
"""

from .version import __version__, __author__, __license__
from .config import AgentConfig, MemoryConfig, RetrievalConfig, LearningConfig
from .types import (
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
    Step,
    Trajectory,
    LLMResponse,
    TaskStatus,
    MemoryDomain,
)
from .exceptions import (
    AgentMementoError,
    LLMError,
    ToolError,
    RetrievalError,
)
from .llm import LLMBackend, AnthropicLLM, OpenAILLM
from .memory import Case, CaseOutcome, HierarchicalMemory, TemporalGraphMemory
from .retrieval import EnsembleRetriever, SentenceTransformerEmbeddings
from .learning import TrajectoryFilter, ReflexionEngine, GRPOPolicy
from .tools import Tool, tool, ToolRegistry
from .tools import WebSearchTool, PythonReplTool, FileReadTool, FileWriteTool
from .core import Agent, Planner, Executor

__all__ = [
    # Version
    "__version__",
    # Config
    "AgentConfig",
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
    "AgentMementoError",
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
