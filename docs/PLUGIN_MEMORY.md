# Escape memory for an existing agent

Escape is the project's product name. The distribution and imports remain
`agenticmemo` for compatibility. Install from the current repository checkout as
shown in the [README](../README.md#installation); an older published package may
not contain this API.

`AgentMemory` supplies bounded reference context and records application-approved
experience. Your existing agent keeps its model, tools, execution loop, retries,
permissions, and result format. The integration supports two explicit hooks and
a callable adapter built on those hooks. It does not require the bundled `Agent`
runtime and does not perform automatic LLM learning, reflection, GRPO updates,
skill mining, or tool execution.

## Run the offline example

From the repository root:

```bash
.venv/bin/python -m examples.plugin_memory
```

The example supplies deterministic local embeddings and a scripted invoice
reporter. It ingests reference knowledge, records a validated result through the
two hooks, then attaches memory to the reporter. Later calls recompute totals for
new amounts. Everything runs locally without model downloads or API keys. This
demonstrates API behavior, not a measured improvement on real agent tasks.

## Two explicit hooks

Supply a stable run ID for a completed host attempt. Normalize only the task,
answer, observable actions, plan, solution, and metadata that the application
approves for storage. Private reasoning is not required.

```python
from agenticmemo.plugin import (
    AgentMemory, ExperienceInput, KnowledgeRecord, ValidationResult,
)

async def report(task, *, extra_context):
    # Replace this with your existing agent's supported context input.
    return {"task": task, "answer": "Invoice total: 40", "total": 40}

async def use_memory():
    memory = await AgentMemory.open("application-memory.sqlite")
    try:
        await memory.ingest_knowledge(
            [KnowledgeRecord(
                id="invoice-policy", source="approved-reporting-policy", version="1",
                content="Invoice reports must sum current amounts and check the source total.",
            )], scope="reporting-team",
        )
        task = "Create invoice total report"
        context = await memory.before_task(
            task, scope="reporting-team", run_id="report-001", token_budget=1200,
        )
        result = await report(task, extra_context=context.text)
        experience = ExperienceInput(task=task, answer=result["answer"])
        # This illustrative fixture is independently known. In production,
        # check the actual artifact against trusted current source data.
        verdict = (
            ValidationResult.passed("source-total-check", ["Source total equals 40"])
            if result["total"] == 40
            else ValidationResult.failed("source-total-check", "Total differs from source")
        )
        receipt = await memory.after_task(
            context.handle, experience=experience, validation=verdict,
        )
        return result, receipt, context
    finally:
        await memory.close()
```

The host decides where to place `context.text`. The context includes a reference
data preamble and record provenance. Treat its contents as untrusted data and
retain the host's permission checks. Referenced plans or code are text; this API
never executes them.

## Callable adapter

The adapter accepts these application-owned callbacks:

| Callback | Supported contract |
| --- | --- |
| `invoke` | `invoke(task, extra_context=context.text, **kwargs)` returns the original host result. |
| `normalize` | `normalize(result)` returns an `ExperienceInput` or a compatible validated mapping. Its task must equal the requested task. |
| `validator` | Optional `validator(task, result)` returns a `ValidationResult` or compatible mapping. |

All three callbacks may be async or synchronous. Synchronous callbacks run in
worker threads; a synchronous callback may also return an awaitable. Attach the
adapter after supplying those functions:

```python
attached = memory.attach(
    invoke, normalize=normalize, validator=validator, validation_timeout=10.0,
)

# The same object returned by invoke is returned here.
result = await attached.run(task, scope="reporting-team", run_id="report-002")

# Request diagnostics per call, including during concurrent use.
wrapped = await attached.run_with_receipt(
    task, scope="reporting-team", run_id="report-003", token_budget=1200,
)
result = wrapped.result
receipt = wrapped.receipt
context = wrapped.context
```

`await attached(...)` also returns the original result. There is no shared
`last_receipt` attribute: each `run_with_receipt` result carries its own context
and receipt. Additional keyword arguments go to `invoke`; `extra_context` is
reserved for the adapter.

The adapter invokes the host once per call and never retries it. Host exceptions
and cancellation propagate, with no synthesized successful experience. Invalid
configuration, blank task/scope/run IDs, invalid budgets, and mismatched context
handles fail explicitly. Runtime retrieval or recording errors degrade memory
while preserving the host result. Normalization errors skip recording. Validator
errors or timeouts record an **unknown** verdict; they never become a pass.
Adapter diagnostics report exception types without copying arbitrary exception
messages. Cancellation cannot stop a synchronous callback already running in a
worker thread, and a database transaction already underway may finish.

## What can be retrieved

Reference knowledge and execution experience have separate payloads:

| Kind | Eligibility |
| --- | --- |
| `KnowledgeRecord` | Application-supplied reference text, while `valid_until` has not passed. Source and version accompany it. |
| `ExperienceInput` with `passed` validation | Eligible after a named validator supplies nonblank evidence, while its reference knowledge is current. |
| Experience with `failed` or `unknown` validation | Stored for inspection, excluded from retrieval. |

An agent saying that it succeeded does not supply validation. The application
owns the validator and its evidence. No validator means unknown experience.
Passing evidence is a declaration from the trusted application boundary; the
library cannot establish whether that declaration is truthful.

Each task handle captures a revision of its scope's reference knowledge.
Experience retains that revision when recorded. Adding, editing, deleting, or
expiring reference knowledge makes experience from the older revision **stale**:
it remains inspectable with its original verdict and evidence, but is excluded
from recall and indexing. `get` and `inspect` expose `stale` and `eligible`.
This deliberately invalidates experience across the whole scope, including
indirect dependencies through earlier experience. Identical knowledge ingests
and changes in another scope do not invalidate it.

Use the handle returned by `before_task`; it also protects against a policy
change while the host is running. Passing validation later for an old run does
not move it to the current revision. Execute and validate a new run against the
current references. Existing records without provenance are conservatively
excluded when their scope contains reference knowledge. Manually constructed
handles without a revision retain the trusted application's responsibility for
current inputs; their first recording captures the current reference revision.
If references change or expire while semantic retrieval is awaiting embeddings,
the call returns empty context with degraded diagnostics instead of injecting
the obsolete snapshot. The next call reads the current references.

`before_task` filters by scope, expiration, and validation before ranking. By
default it uses local query-term coverage (excluding common English stop words)
and makes no embedding or LLM calls. Large reference records supply an excerpt
around a compact passage containing several distinct matching query terms. An
explicit `EmbeddingBackend` adds semantic ranking. Embedding failure or timeout
falls back to lexical ranking with degraded diagnostics. Default limits are four
hits and a minimum score of `0.2`; `top_k`, `min_score`, and `embedding_timeout`
are configurable in `AgentMemory.open`. Cold embeddings are batched, and semantic
retrieval shares one `embedding_timeout` deadline across its query and corpus
calls before falling back to lexical ranking.

The entire rendered `context.text`, including preamble and provenance, fits
`token_budget` according to the supplied `token_counter(text) -> int`. Use your
host model's tokenizer for its actual token budget. Without a counter, the
library conservatively counts UTF-8 bytes; that is not a model tokenizer.
`token_budget=0` or `top_k=0` produces empty context. Individual records may be
truncated or omitted to fit. `context.hits` describes only injected material and
`context.handle.record_ids` captures those IDs for the completed attempt.

## Persistence and receipts

Open a new disk-backed SQLite file with `await AgentMemory.open(path)` and always
close it, or use `async with await AgentMemory.open(path) as memory`. The store
uses versioned record envelopes and scoped keys. It supports concurrent tasks
sharing one open instance. A macOS/Linux advisory lock prevents separate owners
from opening the same database at once; this is not a distributed service or a
Windows storage implementation. Keep its persistent `.lock` sidecar in place.

Use these operations to manage records:

```python
records = await memory.inspect(scope="reporting-team")
eligible = await memory.inspect(scope="reporting-team", eligible_only=True)
record = await memory.get(record_id, scope="reporting-team")
deleted = await memory.delete(record_id, scope="reporting-team")
count = await memory.clear(scope="reporting-team")
indexed = await memory.rebuild_index(scope="reporting-team")
```

Scopes are trusted application partition labels. The caller must authorize the
scope before invoking the API; scope strings are not authentication or access
control. Records are stored as plaintext. The application must select permitted
data and manage database access and retention.

A `RecordingReceipt` distinguishes `stored`, `duplicate`, `promoted`, `rejected`,
`error`, and `deleted`. Check `durable`, `indexed`, `eligible`, `error`, and
`retryable` separately. A durable write can exist while semantic indexing is
pending; the index is derived data and can be rebuilt. An error receipt does not
prove that nothing was written. Ingest returns an `IngestReport` containing one
receipt per input record.

Within a scope, reusing a run ID with identical normalized experience is
idempotent. Reusing it with changed experience is rejected. Unknown experience
can be promoted by submitting the same experience with passing validation. This
deduplicates **recording**; calling the adapter again still invokes the host.
Knowledge IDs identify updatable source records in a scope; versions and
expiration are supplied by the application.

## Legacy data

Existing `Agent`, `AgentConfig`, `Case`, memory packs, and their persistence files
keep their existing interfaces. Use a separate SQLite path for this API; it does
not transparently reinterpret an old agent's storage or derived learning files.

The bundled verifier intentionally corrects an earlier false-positive behavior:
disabled or unavailable verification now maps to `PARTIAL` (or preserves an
existing execution failure). `Agent.run` retains the answer and diagnostic
verification metadata, assigns a neutral reward, and does not store an unknown
outcome. Explicit failed and successful judgments remain distinct. This avoids
teaching future tasks from an answer that could not be checked.

An explicit importer accepts a legacy JSON list pack, JSON case store, or legacy
SQLite case store (recognized by its file header):

```python
report = await memory.import_legacy("old-memory-pack.json", scope="reporting-team")
# A legacy .db store is opened read-only; use a separate destination database.
report = await memory.import_legacy("old-agent-memory.db", scope="reporting-team")
```

Import reads the source without modifying it and never executes imported code.
The resulting experience is **unknown**, even if the old file claims success or
contains a reward. It is excluded from retrieval until the application supplies
current validation for the same experience/run ID. Imports are idempotent;
inspect imported records for their run IDs and approved payload before validating
them. New IDs identify the imported records; an original case ID is retained in
metadata. Auxiliary hints/failures/skills files are not accepted by this importer.

## Integration acceptance

Before measuring task benefits, check one real host's context entry point,
normalizer, and independent artifact validator. Verify result identity and one
host invocation, cold empty retrieval, validated warm retrieval, unknown/failed
exclusion, scope separation, deletion after reopening, repeated recording, and
retrieval/recording/validator failure behavior. The repository's adapter tests
exercise those runtime boundaries without a model or paid calls.

For a later utility evaluation, compare the same host without memory, with a
fresh empty store, and with a frozen store learned from separate earlier tasks.
Use held-out inputs and independently grade the resulting artifacts. Count all
retrieval, embedding, validation, and recording overhead. The SDK and offline
example make no claim of a demonstrated enterprise success-rate improvement.
