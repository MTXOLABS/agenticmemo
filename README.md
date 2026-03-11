<div align="center">

<h1>AgenticMemo</h1>

<p><strong>LLM agents that learn from experience — no fine-tuning required.</strong></p>

<p>
  <a href="https://pypi.org/project/agenticmemo"><img src="https://img.shields.io/pypi/v/agenticmemo?color=blue&style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/agenticmemo"><img src="https://img.shields.io/pypi/pyversions/agenticmemo?style=flat-square" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
  <a href="https://github.com/agenticmemo/agenticmemo/actions"><img src="https://img.shields.io/badge/tests-70%20passed-brightgreen?style=flat-square" alt="Tests"></a>
  <a href="https://github.com/agenticmemo/agenticmemo"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs Welcome"></a>
</p>

<p>
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a>
</p>

</div>

---

AgenticMemo is a production-ready Python framework for building **self-improving LLM agents**. It implements a complete four-phase learning system — temporal knowledge graph memory, GRPO-based retrieval, Reflexion failure loops, and multi-agent memory sharing — all without ever updating the underlying model weights.

Agents trained with AgenticMemo get better at their tasks simply by running them. Every execution is stored, filtered for quality, and made retrievable, so future tasks automatically benefit from past experience.

```python
import asyncio
from agenticmemo import Agent
from agenticmemo.tools import PythonReplTool

async def main():
    agent = Agent.from_anthropic(api_key="sk-ant-...")
    agent.add_tool(PythonReplTool())

    # First run — no prior experience
    result = await agent.run("Write a binary search function in Python")
    print(result.final_answer)

    # Subsequent runs — agent retrieves and applies past experience
    result = await agent.run("Implement merge sort with the same style")
    print(result.final_answer)   # benefits from the binary search case

asyncio.run(main())
```

---

## Why AgenticMemo?

Traditional agent frameworks treat every task independently. AgenticMemo agents **accumulate knowledge** across tasks through a principled four-phase system:

| | Original Memento | **AgenticMemo** |
|---|---|---|
| Memory storage | Flat case bank | **Temporal Knowledge Graph** |
| Memory structure | None | **4-layer H-MEM hierarchy** |
| Retrieval policy | Soft Q-learning | **GRPO (no critic, no backprop)** |
| Similarity signal | Cosine only | **Semantic + BM25 + Graph + Temporal** |
| Failure learning | None | **Reflexion self-correction loop** |
| Trajectory quality | Store everything | **3-stage quality filter** |
| Multi-agent | None | **Shared memory pool + consensus** |
| Strategy distillation | None | **Hints internalization** |

---

## Installation

```bash
# Minimal install (uses local sentence-transformers embeddings)
pip install agenticmemo

# With Anthropic / Claude support
pip install agenticmemo[anthropic]

# With OpenAI / GPT support
pip install agenticmemo[openai]

# Full install (all providers + dev tools)
pip install agenticmemo[all]
```

**Requirements:** Python 3.10+

---

## Quick Start

### Basic agent

```python
import asyncio
from agenticmemo import Agent
from agenticmemo.tools import PythonReplTool, WebSearchTool

async def main():
    agent = Agent.from_anthropic(api_key="sk-ant-...")
    agent.add_tools(PythonReplTool(), WebSearchTool())

    result = await agent.run("Find the top 3 Python web frameworks and compare them")
    print(result.final_answer)
    print(f"Steps taken : {result.num_steps}")
    print(f"Tokens used : {result.total_tokens}")
    print(f"Cases in memory: {await agent.memory_size()}")

asyncio.run(main())
```

### Custom tools

Use the `@tool` decorator to wrap any async function:

```python
from agenticmemo.tools import tool

@tool(name="unit_converter", description="Convert between units of measurement")
async def convert(value: float, from_unit: str, to_unit: str) -> float:
    conversions = {"km_to_miles": 0.621371, "miles_to_km": 1.60934}
    factor = conversions.get(f"{from_unit}_to_{to_unit}", 1.0)
    return value * factor

agent.add_tool(convert)
```

Or subclass `Tool` for full control over schema and execution:

```python
from agenticmemo import Tool, ToolResult
import uuid

class DatabaseTool(Tool):
    name        = "query_db"
    description = "Execute a read-only SQL query and return results"
    parameters  = {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "SQL SELECT query"},
        },
        "required": ["sql"],
    }

    async def execute(self, sql: str, **_) -> ToolResult:
        rows = await self.db.fetch(sql)   # your database client
        return ToolResult(
            tool_call_id=str(uuid.uuid4()),
            tool_name=self.name,
            output=rows,
        )
```

### Multi-agent memory sharing

```python
from agenticmemo import Agent
from agenticmemo.memory import SharedMemoryPool

# Shared pool all agents read from and write to
pool = SharedMemoryPool(consensus_threshold=0.8)
pool.register_agent("researcher")
pool.register_agent("coder")

researcher = Agent.from_anthropic(api_key="...")
coder      = Agent.from_anthropic(api_key="...")

# Both agents share the same global case bank
# High-reward cases are automatically promoted to the consensus layer
```

### Persistent memory across sessions

```python
from agenticmemo import Agent, AgentConfig, MemoryConfig

cfg = AgentConfig(
    memory=MemoryConfig(persist_path="./agent_memory.json")
)

# Session 1 — agent learns from tasks
agent = Agent.from_anthropic(api_key="...", cfg=cfg)
await agent.run("...")

# Session 2 — agent loads and applies experience from session 1 automatically
agent = Agent.from_anthropic(api_key="...", cfg=cfg)
print(await agent.memory_size())  # non-zero from previous session
```

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │           AgenticMemo Agent              │
                         │                                          │
  Task ─────────────────►│  HintLibrary + EnsembleRetriever         │
                         │           │                              │
                         │           ▼                              │
                         │        Planner ──────────────────────►  │
                         │    (cases + hints)        Executor       │
                         │                              │           │
                         │   QualityFilter ◄────────────┘           │
                         │         │          ReflexionEngine       │
                         │         │          (on failure)          │
                         │         ▼                                │
                         │  HierarchicalMemory                      │
                         │  ├── TemporalGraphMemory                 │
                         │  └── 4-layer H-MEM index                 │
                         │         │                                │
                         │    GRPOPolicy (Q-value updates)          │
                         │    HintExtractor (periodic)              │
                         └─────────────────────────────────────────┘
```

### Four learning phases

#### Phase 1 — Ensemble Retrieval
Every case retrieval combines four independent signals to maximise relevance:

```
score(query, case) = 0.5 × cosine(embed(query), embed(case))   # semantic
                   + 0.3 × BM25(query, case)                    # keyword
                   + 0.1 × temporal_weight(case)                 # recency
                   + 0.1 × pagerank(case)                        # graph centrality
```

Compared to cosine-only retrieval, this ensemble approach yields ~40% better precision on diverse task distributions.

#### Phase 2 — GRPO Policy + Reflexion
**GRPO** (Group Relative Policy Optimisation) replaces the original soft Q-learning. Rather than learning absolute case values, it samples G candidate case sets per query, computes group-relative advantage from outcomes, and updates per-case Q-values. No critic model. No backpropagation.

**Reflexion** handles task failures: the LLM diagnoses what went wrong, generates a concrete correction strategy, and stores the `(failure, reflection, fix)` triple in the case. Future retrievals surface both successful and corrective cases.

#### Phase 3 — Temporal Knowledge Graph Memory
Cases are stored in a NetworkX-based directed graph instead of a flat list:
- **Nodes** — Case objects with temporal metadata and outcome scores
- **Edges** — Relationships between cases (similar, sequential, contradicts, refines)
- **Temporal decay** — Stale cases are down-weighted, not deleted
- **PageRank** — Frequently referenced cases score higher in retrieval

Cases are also organised in a **4-layer H-MEM hierarchy** (Domain → Category → Trace → Episode) for O(log N) retrieval instead of O(N).

#### Phase 4 — Multi-Agent Memory + Hints Internalization
**SharedMemoryPool** lets multiple agents share a global case bank with role-specific private memories, content-based deduplication, and a consensus layer for high-reward cases.

**HintExtractor** periodically scans the case bank and distils recurring successful strategies into compact, reusable `Hint` objects. These are injected into the Planner alongside retrieved cases, providing general procedural knowledge that complements specific episodic memory.

---

## Documentation

### Configuration reference

```python
from agenticmemo import Agent, AgentConfig, MemoryConfig, RetrievalConfig, LearningConfig

agent = Agent.from_anthropic(
    api_key="sk-ant-...",
    cfg=AgentConfig(
        # LLM settings
        llm_model="claude-opus-4-6",
        llm_temperature=0.0,
        llm_max_tokens=4096,

        # Execution
        max_steps=20,          # ReAct loop limit per task
        max_retries=3,         # Reflexion retry limit
        verbose=True,          # rich console output

        memory=MemoryConfig(
            max_cases=50_000,              # hard cap on stored cases
            persist_path="./memory.json",  # disk persistence (None = in-memory)
            temporal_decay_rate=0.005,     # staleness decay per day
            min_reward_to_store=0.0,       # discard cases below this reward
        ),

        retrieval=RetrievalConfig(
            # Embedding backend
            embedding_backend="sentence-transformers",   # or "openai"
            embedding_model="all-MiniLM-L6-v2",

            # Ensemble weights (should sum to 1.0)
            weight_semantic=0.5,
            weight_bm25=0.3,
            weight_graph=0.1,
            weight_temporal=0.1,

            top_k=4,           # cases returned per query

            # GRPO policy
            enable_grpo=True,
            grpo_lr=1e-3,
            grpo_update_every=10,
        ),

        learning=LearningConfig(
            # Reflexion
            enable_reflexion=True,
            max_reflexion_retries=2,

            # Trajectory quality filter
            enable_quality_filter=True,
            min_trajectory_steps=1,
            max_trajectory_steps=50,

            # Reward shaping
            success_reward=1.0,
            partial_reward=0.3,
            failure_reward=-0.2,
        ),
    ),
)
```

### LLM providers

```python
# Anthropic / Claude
agent = Agent.from_anthropic(api_key="sk-ant-...", model="claude-opus-4-6")

# OpenAI / GPT
agent = Agent.from_openai(api_key="sk-...", model="gpt-4o")

# Custom provider — implement two methods
from agenticmemo import LLMBackend, LLMResponse

class MyLLM(LLMBackend):
    async def complete(self, messages, tools=None, system=None) -> LLMResponse:
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

agent = Agent(MyLLM(model="my-model"))
```

### Built-in tools

| Tool | Description | Extra dependency |
|---|---|---|
| `PythonReplTool` | Executes Python with a persistent namespace | — |
| `WebSearchTool` | DuckDuckGo web search, no API key needed | `duckduckgo-search` |
| `FileReadTool` | Reads any local file | — |
| `FileWriteTool` | Writes content to a local file | — |

### Inspecting agent state

```python
# Memory
size   = await agent.memory_size()
recent = await agent.recent_cases(n=5)

# GRPO learning stats
stats  = agent.grpo_stats()
# {"avg_reward": 0.82, "reward_std": 0.14, "total_updates": 7, ...}

# Internalized hints
library = agent.hint_library()
print(library.to_prompt_block(domain="coding"))

# Manually trigger hint extraction
new_hints = await agent.extract_hints_now(domain="coding")
```

---

## Performance

Research-backed improvements over the original Memento baseline:

| Component | Improvement | Source |
|---|---|---|
| Ensemble retrieval | ~40% better retrieval precision | USMB benchmark |
| Reflexion failure loop | >18% accuracy on long-horizon tasks | EMNLP 2024 |
| CLEANER trajectory filter | ~6% on hard reasoning tasks | arXiv 2601.15141 |
| GRPO vs soft Q-learning | Faster convergence, no critic overhead | arXiv 2510.08191 |
| Hierarchical memory | Better OOD generalisation | H-MEM paper |
| Temporal graph memory | +1.4% deep retrieval accuracy | Zep arXiv 2501.13956 |

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository and create a feature branch
2. Install the development dependencies:
   ```bash
   pip install agenticmemo[dev]
   ```
3. Make your changes and add tests
4. Ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```
5. Open a pull request with a clear description of your changes

Please open an issue first for significant changes so the approach can be discussed before implementation.

---

## License

Released under the [MIT License](LICENSE).
