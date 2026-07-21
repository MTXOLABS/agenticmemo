# Memory Packs — teach small models with expert solutions

A **memory pack** is a JSON file of verified solutions that any AgenticMemo agent
can load as pre-built experience. An expert — a human or a frontier model — solves
a class of tasks *once*; every agent that loads the pack retrieves those solutions
as proven exemplars and adapts them instead of deriving from scratch.

Measured effect (gpt-5.4-mini and gpt-4.1 on hard finance tasks, verification on):
**40–56% fewer steps and 13–39% fewer tokens** on tasks similar to packed
solutions. Memory packs make repeat work dramatically cheaper; they do not raise
a model's ceiling on tasks far beyond its ability.

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
- **domain** *(optional)* — one of AgenticMemo's memory domains (`finance`,
  `real_estate`, `coding`, ...). Defaults to `general`.
- **code** — the verified working solution. Use the template convention below.
- **answer** *(optional)* — the solution's output. Stored for bookkeeping but
  **never shown to the model**: models near their limit copy visible reference
  answers instead of executing. AgenticMemo withholds outputs by design.

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
INPUTS block and execute** — turning adaptation into a trivial edit that even
nano-class models perform reliably. Blob code without markers still works; the
model adapts it free-form.

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
