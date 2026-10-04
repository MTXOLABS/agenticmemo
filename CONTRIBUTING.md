# Contributing to Escape

Thanks for your interest in Escape! This guide covers everything you need to
set up a development environment, make changes, and submit them.

Escape retains the `agenticmemo` distribution and Python imports for compatibility.

## Project layout

```
agenticmemo/          The framework package
├── core/             Agent, Planner, Executor (the main loop)
├── plugin/           AgentMemory hooks, adapter, scoped SQLite persistence
├── memory/           Case model, temporal graph, hierarchical index, shared pool
├── retrieval/        Ensemble retriever, BM25, embeddings
├── learning/         GRPO policy, Reflexion, verifier, filters, hints, skills (ESMC), failure mining (CFM)
├── llm/              Anthropic / OpenAI backends + LLMBackend interface
└── tools/            Tool base class, @tool decorator, registry, built-in tools
tests/                Pytest suite (async, no network required)
benchmarks/           Benchmark harness with mock LLM (results/ stays local)
examples/             Runnable end-to-end examples
```

## Development setup

Requires Python 3.10+.

```bash
git clone https://github.com/MTXOLABS/agenticmemo
cd agenticmemo
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

The `AgentMemory` disk store currently supports macOS and Linux.

## Running tests

```bash
pytest tests/ -v
```

The test suite uses fake LLMs and local embeddings. It needs no API keys; embedding
tests need their model cached locally to run without network access.
All tests must pass before a PR is merged.

## Linting

The project uses ruff with the config in `pyproject.toml` (`E, F, I, N, UP`, line length 100):

```bash
ruff check agenticmemo tests benchmarks examples
ruff check --fix .        # auto-fix imports/formatting issues
```

CI treats lint warnings as failures — run it locally before pushing.

## Code style

- **Async-first**: all I/O paths (`LLMBackend.complete`, `Tool.execute`, memory ops) are
  `async`. Never call blocking I/O inside them — wrap sync libraries in `asyncio.to_thread`.
- **Fail-soft learning**: LLM-based learning steps (reflection, hint extraction, mining,
  verification) must degrade gracefully — catch exceptions and return a neutral result,
  never crash the agent run.
- **Pydantic models** for all data that crosses module boundaries (`Case`, `Trajectory`,
  `Message`, configs).
- **Atomic persistence**: any writer that persists to disk must write to a `.tmp` file
  and `replace()` it, so a crash can't corrupt state.
- Type hints on all public functions; docstrings on all public classes and methods.

## Making changes

1. Fork and create a feature branch (`git checkout -b feat/my-change`).
2. Make the change, with tests. New learning components need both unit tests and,
   where feasible, an end-to-end test through `Agent.run` with a scripted fake LLM.
3. Run `pytest tests/ -q` and `ruff check .` — both must be clean.
4. Open a PR describing **what** changed and **why**. Link the issue if one exists.

For significant changes (new memory backends, retrieval signals, learning phases),
please open an issue first so the design can be discussed before you invest time.

## Running the benchmarks

```bash
python -m benchmarks.run_benchmark --help
```

Benchmarks run against a deterministic mock LLM by default, so they're reproducible
and free. Result JSONs are written locally and are not committed.

## Releasing (maintainers)

1. Bump `version` in `pyproject.toml` **and** `agenticmemo/version.py` (keep in sync).
2. Update the changelog section in the GitHub release notes.
3. Build and publish:
   ```bash
   pip install build twine
   python -m build
   twine upload dist/*
   ```
4. Tag the release: `git tag vX.Y.Z && git push --tags`.

## Reporting security issues

Please do **not** open public issues for security vulnerabilities —
see [SECURITY.md](SECURITY.md) for the private reporting process.
