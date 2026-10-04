<div align="center">

<h1>Escape</h1>

<p><strong>Reference knowledge and validated experience for your agents.</strong></p>

<p>
  <a href="https://pypi.org/project/agenticmemo"><img src="https://img.shields.io/pypi/v/agenticmemo?color=blue&style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/agenticmemo"><img src="https://img.shields.io/pypi/pyversions/agenticmemo?style=flat-square" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
  <a href="https://github.com/MTXOLABS/agenticmemo"><img src="https://img.shields.io/badge/status-alpha-orange?style=flat-square" alt="Alpha"></a>
  <a href="https://github.com/MTXOLABS/agenticmemo"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs Welcome"></a>
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

Escape is an **alpha Python memory backend** for existing agents. Its `AgentMemory`
API retrieves bounded reference context and records experiences approved by your
application's validator. Your agent retains its own execution loop, model, tools,
permissions, and result format.

**Escape is the new product name for AgenticMemo.** The distribution name, Python
imports, and repository URL remain `agenticmemo` for compatibility. No import or
database migration is required for the branding change.

The repository also contains a bundled research agent runtime with graph memory,
ensemble retrieval, verification, reflection, and learning components. Those are
separate from the memory plug-in. Neither API guarantees better answers or lower
costs; evaluate it on your own held-out tasks with independent validation.

---

## What Escape provides

The memory plug-in supports:

| Capability | Behavior |
|---|---|
| Knowledge and experience | Separate reference records from completed host attempts |
| Retrieval eligibility | Exclude failed, unknown, expired, and stale experience or references as applicable |
| Reference changes | Preserve older experience for inspection while removing it from recall |
| Retrieval | Local lexical search by default; optional application-supplied embeddings |
| Context limits | Budgeted excerpts with source provenance and an optional model tokenizer |
| Integration | Explicit before/after hooks or a callable adapter that invokes the host once |
| Persistence | Scoped SQLite records, idempotent recording, durable receipts, and restart recovery |
| Management | Inspect, delete, clear, rebuild indexes, and explicitly import legacy data |

Current limits: the disk store supports one owner on macOS/Linux, with concurrent
tasks sharing that owner. Scopes are application-authorized labels, not access
control. Storage is plaintext. This is not a distributed service; real-LLM quality
and enterprise-scale performance still need evaluation. See the
[integration guide](docs/PLUGIN_MEMORY.md) and [security policy](SECURITY.md).

---

## Installation

```bash
git clone https://github.com/MTXOLABS/agenticmemo.git
cd agenticmemo
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

**Requirements:** Python 3.10+; macOS or Linux for the `AgentMemory` disk store.
Install from the checkout to use the API documented here. Published `agenticmemo`
releases may predate the plug-in and the Escape branding. The `all` extra installs
both supported provider SDKs; it does not configure credentials or make API calls.

---

## Quick Start

### Add memory to your existing agent

`AgentMemory` supplies reference knowledge and validated experience while your
agent keeps its own execution loop, tools, permissions, and result format.
Adapt its supported context input to `invoke(task, extra_context=..., **kwargs)`,
provide `normalize(result) -> ExperienceInput`, and optionally supply an
independent `validator(task, result) -> ValidationResult`.

```python
from agenticmemo import AgentMemory

async def run_existing(invoke, normalize, validator, task, *, scope, run_id):
    memory = await AgentMemory.open("application-memory.sqlite")
    try:
        attached = memory.attach(invoke, normalize=normalize, validator=validator)
        wrapped = await attached.run_with_receipt(task, scope=scope, run_id=run_id)
        return wrapped.result, wrapped.receipt  # original host result + recording status
    finally:
        await memory.close()
```

The default retriever is local lexical search. A custom embedder is optional;
there are no default paid calls, model downloads, or automatic learning steps.
Only experiences with passing application validation and evidence are eligible
for retrieval. Supply your model's `token_counter` for its exact token budget;
the default counts UTF-8 bytes conservatively. Scopes are application-authorized
partition labels. See the [two hooks, adapter, and migration guide](docs/PLUGIN_MEMORY.md)
and run the [offline example](examples/plugin_memory.py):

```bash
.venv/bin/python -m examples.plugin_memory
```

### Bundled agent runtime

These examples use the separate `Agent` runtime and require a configured model
provider. Provider calls may incur costs. The optional tools execute with the
process's permissions; read [SECURITY.md](SECURITY.md) before enabling them.

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

### Shared memory building block

```python
from agenticmemo.memory import SharedMemoryPool

# Shared pool all agents read from and write to
pool = SharedMemoryPool(consensus_threshold=0.8)
pool.register_agent("researcher")
pool.register_agent("coder")

# In your orchestration code, explicitly write and read cases:
# await pool.store_shared(case)
# cases = await pool.all_cases_for("researcher")
```

Registering names in a pool does not attach it to `Agent` instances. Your
orchestration code must connect those reads and writes to the agents.

### Persistent memory across sessions

```python
from agenticmemo import Agent, AgentConfig, MemoryConfig

cfg = AgentConfig(
    memory=MemoryConfig(persist_path="./agent_memory.json")
)

# Session 1 — eligible verified cases may be stored
agent = Agent.from_anthropic(api_key="...", cfg=cfg)
await agent.run("...")

# Session 2 — agent loads stored cases for retrieval
agent = Agent.from_anthropic(api_key="...", cfg=cfg)
print(await agent.memory_size())  # depends on which earlier outcomes were stored
```

---

## Architecture

The existing-agent plug-in has a small integration surface:

```text
Task -> before_task -> bounded reference context -> your agent
                                                    |
                       your normalizer + validator <-+
                                   |
                              after_task
                                   |
                           scoped SQLite store
```

The bundled research runtime has a separate architecture:

```
                         ┌─────────────────────────────────────────┐
                         │              Escape Agent                │
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

### Bundled runtime components

#### Phase 1 — Ensemble Retrieval
The ensemble retriever combines four signals using configurable weights. With
the default weights:

```
score(query, case) = 0.5 × cosine(embed(query), embed(case))   # semantic
                   + 0.3 × BM25(query, case)                    # keyword
                   + 0.1 × temporal_weight(case)                 # recency
                   + 0.1 × pagerank(case)                        # graph centrality
```

These weights describe implementation behavior, not a measured accuracy gain.

#### Phase 2 — GRPO Policy + Reflexion
**GRPO** (Group Relative Policy Optimisation) replaces the original soft Q-learning. Rather than learning absolute case values, it samples G candidate case sets per query, computes group-relative advantage from outcomes, and updates per-case Q-values. No critic model. No backpropagation.

**Reflexion** handles task failures: the LLM diagnoses what went wrong, generates a concrete correction strategy, and stores the `(failure, reflection, fix)` triple in the case. Future retrievals surface both successful and corrective cases.

**Outcome verification** grounds the success signal: after each execution attempt, an LLM-as-judge pass grades whether the final answer actually solves the task (`success` / `partial` / `failure`). Verified failures feed Reflexion retries, GRPO rewards, and the failure-pattern miner.

#### Phase 3 — Temporal Knowledge Graph Memory
Cases are stored in a NetworkX-based directed graph instead of a flat list:
- **Nodes** — Case objects with temporal metadata and outcome scores
- **Edges** — Relationships between cases (similar, sequential, contradicts, refines)
- **Temporal decay** — Stale cases are down-weighted, not deleted
- **PageRank** — Frequently referenced cases score higher in retrieval

Cases are also organised in a **4-layer H-MEM hierarchy** (Domain → Category →
Trace → Episode). The ensemble retriever still scores its candidate cases;
the hierarchy does not establish an O(log N) end-to-end retrieval guarantee.

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
            min_reward_to_store=-1.0,      # keep failures as anti-cases for CFM
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
            # Outcome verification (LLM-as-judge grades each attempt;
            # without it any non-empty answer would count as success)
            enable_verification=True,

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

## Evaluation

The automated suite exercises integration contracts, validation, persistence,
retrieval, and failure recovery with scripted hosts and local embeddings. It does
not establish a real-model success rate or an enterprise throughput target.
Compare the same host with no memory, an empty store, and a frozen store learned
from separate tasks. Grade held-out results independently and include all model,
embedding, validation, and storage overhead in the comparison. See the
[acceptance guide](docs/PLUGIN_MEMORY.md#integration-acceptance).

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository and create a feature branch
2. Install the development dependencies:
   ```bash
   pip install -e ".[all,dev]"
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
