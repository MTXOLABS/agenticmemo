---
name: qa
description: Run the full Escape quality gate — tests, lint, security scan, and import check. Use this after ANY code change to the agenticmemo package, before every commit, and whenever the user says "run tests", "check the code", "is everything green", "verify", or finishes a feature/fix. Also use it when a session starts with uncommitted changes, to establish a known-good baseline.
---

# Escape Quality Gate

Run all four checks. A change is "done" only when every one passes.
Always use the project venv at `.venv/` — the system Python does not have the deps.

## The checks (run from repo root)

```bash
# 1. Tests — 70 tests, fake LLMs + local embeddings, no network/API keys needed (~35s)
.venv/bin/python -m pytest tests/ -q

# 2. Lint — config lives in pyproject.toml (E,F,I,N,UP; line length 100)
.venv/bin/ruff check agenticmemo tests benchmarks examples

# 3. Security scan — expected result: 0 medium/high (9 known-accepted LOW findings)
.venv/bin/bandit -r agenticmemo -q

# 4. Import sanity — catches broken __init__ exports that tests can miss
.venv/bin/python -c "import agenticmemo; print('OK', agenticmemo.__version__)"
```

Steps 1–4 are independent — run them in parallel.

## Interpreting results

- **Test failures**: read the failing assertion before touching code — the suite uses
  scripted fake LLMs, so failures usually mean a behavior contract changed, not flakiness.
- **Ruff**: `ruff check --fix` is safe for import sorting/unused imports. `benchmarks/`
  has per-file ignores (long mock-prompt strings) — don't "fix" those by editing config.
- **Bandit**: the 9 accepted LOW findings (asserts, fail-soft `try/except/pass`, `random`
  for GRPO sampling) are documented in `docs/launch/SECURITY_AUDIT.md` §A7. Anything
  MEDIUM+ or any new finding is a regression — stop and investigate.
- **New MD5 uses** must set `usedforsecurity=False` (they're fingerprints, not crypto).

## When adding code, preserve these invariants

- All I/O paths are `async`; wrap sync libraries in `asyncio.to_thread`.
- Learning steps (reflection, mining, verification) fail soft — never crash a run.
- Disk persistence writes to `.tmp` then `Path.replace()` (atomic).
- End-to-end tests use a scripted fake `LLMBackend` — see `tests/` for the pattern.

## Dependency audit (weekly / before release, not every change)

```bash
.venv/bin/pip-audit --skip-editable
```

Known-open: torch CVE-2025-3000 has no upstream fix yet — ignore it, flag anything else.
