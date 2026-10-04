# Escape memory packs — reference solutions for the bundled agent

A **memory pack** is a JSON file of reference solutions that the bundled Escape
`Agent` can load as experience. A human or model authors the solutions, and the
application should verify them before loading. Retrieval can then supply them as
examples for similar tasks. The package and imports remain `agenticmemo`.

Packs do not establish that a new answer is correct or guarantee lower costs.
Evaluate their effect on separate held-out tasks with independent validation.
The `AgentMemory` plug-in has a separate [legacy import path](PLUGIN_MEMORY.md#legacy-data)
that records imported experience as unknown until the application validates it.

## Quick start

```python
import asyncio
from agenticmemo import Agent
from agenticmemo.tools import PythonReplTool

async def main():
    agent = Agent.from_openai(api_key="sk-...", model="gpt-5.4-mini")
    agent.add_tool(PythonReplTool())

    n = await agent.load_memory_pack("finance_pack.json")
    print(f"loaded {n} expert solutions")

    # Tasks similar to packed solutions now retrieve them as exemplars
    result = await agent.run("Build a DCF valuation: equity $800M, beta 1.1, ...")
    print(result.final_answer)

asyncio.run(main())
```

## Pack format

```json
[
  {
    "task": "Build a full WACC model then DCF valuation: Equity=$600M ...",
    "domain": "finance",
    "code": "# --- INPUTS ---\nE, D = 600e6, 400e6\n...\n# --- LOGIC (do not modify) ---\n...",
    "answer": "WACC: 9.396% ..."
  }
]
```

- **task** — the task the solution solves, written the way users phrase it.
  Retrieval matches on this text, so realistic phrasing matters.
- **domain** *(optional)* — one of Escape's memory domains (`finance`,
  `real_estate`, `coding`, ...). Defaults to `general`.
- **code** — the verified working solution. Use the template convention below.
- **answer** *(optional)* — the solution's output. Stored for bookkeeping but
  **never shown to the model**: models near their limit copy visible reference
  answers instead of executing. Escape withholds outputs by design in the bundled
  runtime's reference rendering.

## The template convention (strongly recommended)

Structure pack code with two markers:

```python
# --- INPUTS ---
principal = 1_000_000
rate = 0.075

# --- LOGIC (do not modify) ---
...computation and printing...
```

When a template is detected, the agent instructs the model to **edit only the
INPUTS block and execute**. The application still needs to validate the resulting
artifact and calculations. Without these markers, the model adapts the code
free-form.

## Authoring rules that matter (learned from measurement)

1. **Verify before packing.** Run the code; wrong packed solutions are worse
   than none — the agent trusts them.
2. **Print every deliverable.** The model reports from its own execution
   output; anything the code doesn't print tends to be omitted from answers.
3. **One task family per entry.** Retrieval matches whole entries; a grab-bag
   entry matches nothing well.
4. **Export packs from a strong agent.** `await agent.export_memory_pack(path)`
   turns any agent's accumulated successful solutions into a pack — run your
   workflow once on a frontier model, export, and load into cheap models.

## Sharing and versioning

Packs are plain JSON: commit them, review them in PRs, ship them with your app.
Treat them like code — they effectively are (they get executed after adaptation,
so only load packs you trust).
