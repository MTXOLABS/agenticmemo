# AgenticMemo — project instructions

Python framework for self-improving LLM agents (memory + retrieval + learning, no
fine-tuning). Package code in `agenticmemo/`; always use the venv at `.venv/`.

## Skills — use them

- **qa** — after any code change / before commit: tests, ruff, bandit, import check.
- **benchmark** — any performance/comparison run; results → `benchmarks/results/`.
- **release** — version bump, build, PyPI publish, tag.

## Conventions (enforced in review)

- Async-first: all I/O paths are `async`; wrap sync libs in `asyncio.to_thread`.
- Learning steps (reflection, mining, verification, hint extraction) fail soft —
  catch exceptions, return neutral, never crash `Agent.run`.
- Persistence writes are atomic: write `.tmp`, then `Path.replace()`.
- New MD5 uses are fingerprints only → `usedforsecurity=False`.
- Pydantic models for anything crossing module boundaries.
- Version lives in TWO places: `pyproject.toml` + `agenticmemo/version.py` — keep in sync.

## Repo layout notes

- `docs/launch/`, `paper/`, `benchmarks/results/` are **gitignored, internal** —
  business plans, paper sources, raw results. Never surface their contents in
  public files (README, docstrings).
- `docs/launch/CHECKLIST.md` is the master launch tracker — tick items as they land.
- `docs/launch/SECURITY_AUDIT.md` documents accepted bandit findings and the
  performance backlog — check it before "fixing" a scanner warning.
- Public benchmark numbers must come from post-verification runs only (v2.1+);
  older numbers counted any non-empty answer as success and are inflated.

## Tests

`tests/` uses scripted fake LLMs and local embeddings — no network or API keys.
End-to-end tests drive `Agent.run` with a fake `LLMBackend`; follow that pattern
for new learning components.
