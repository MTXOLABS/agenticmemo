---
name: benchmark
description: Run AgenticMemo's benchmark suites correctly and put results in the right place. Use whenever the user asks to benchmark, measure performance, compare standalone vs memory-augmented agents, reproduce the paper/README numbers, or generate results for launch material — even if they just say "run the numbers" or "how much does memory help".
---

# AgenticMemo Benchmarks

The harness compares a standalone LLM against the same LLM + AgenticMemo memory.
Entry point: `python -m benchmarks.run_benchmark` (always via `.venv/bin/python`).

## Golden rules

1. **Results go in `benchmarks/results/`** — it's gitignored (raw data stays local).
   Always pass `--output benchmarks/results/<name>.json`.
2. **Numbers are only publishable if produced with outcome verification ON**
   (`LearningConfig.enable_verification=True`, the default since v2.1). Results from
   before the verifier existed counted "produced any text" as success and are inflated —
   never quote them in README/paper/pitch material.
3. Mock provider is free and deterministic; real providers cost API money — confirm
   with the user before launching a real-provider run.

## Common invocations

```bash
# Free, deterministic smoke run (mock LLM)
.venv/bin/python -m benchmarks.run_benchmark --provider mock \
  --suite core --output benchmarks/results/mock_core.json

# The headline research comparison (Standalone vs +AgenticMemo) — real API, costs money
.venv/bin/python -m benchmarks.run_benchmark --research \
  --output benchmarks/results/research_$(date +%Y%m%d).json

# Domain suites used for launch numbers
.venv/bin/python -m benchmarks.run_benchmark --suite domain \
  --output benchmarks/results/domain_$(date +%Y%m%d).json
```

Suites: `all, hard, domain, core, memory, reflexion, efficiency, retrieval,
hard_core, hard_transfer, hard_reflexion, domain_finance, domain_real_estate, domain_transfer`.
API keys come from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` env vars (or `--anthropic-key`/`--openai-key`).

Real-provider runs are long — launch with `run_in_background` and check progress
periodically rather than blocking.

## Reporting results

When summarizing a run for the user, always report: success rate for both arms
(standalone vs +memory), the delta in percentage points, average steps, and token
usage — and say which provider/model and suite produced them. Date-stamp the output
file so old and new results never get confused.
