---
name: release
description: Cut and publish an AgenticMemo release to PyPI the safe way — preflight checks, synchronized version bump, build, fresh-venv install test, upload, tag. Use whenever the user wants to release, publish, ship a version, push to PyPI, bump the version, or "make v2.x live".
---

# AgenticMemo Release Process

Releases are irreversible (PyPI versions can't be re-uploaded), so the order below
matters. Get explicit user confirmation before the actual `twine upload` step.

## 1. Preflight — all must be green

- Run the `qa` skill (tests + lint + bandit + import). No exceptions.
- `.venv/bin/pip-audit --skip-editable` — no new fixable CVEs.
- Working tree committed — never build a release from a dirty tree.
- Benchmarks in README are post-verification numbers (see `benchmark` skill rule 2).
- Check `docs/launch/CHECKLIST.md` for open P0 items and surface them to the user.

## 2. Version bump — TWO files must stay in sync

```
pyproject.toml            → version = "X.Y.Z"
agenticmemo/version.py    → __version__ = "X.Y.Z"
```

Semver: behavior changes to the learning loop = minor bump; bug fixes = patch.

## 3. Build and verify the artifact

```bash
.venv/bin/pip install -q build twine
rm -rf dist/                      # dist/ is gitignored, never commit it
.venv/bin/python -m build

# Install test in a THROWAWAY venv — catches packaging bugs the repo venv hides
python3 -m venv /tmp/relcheck && /tmp/relcheck/bin/pip install -q dist/*.whl \
  && /tmp/relcheck/bin/python -c "import agenticmemo; print(agenticmemo.__version__)" \
  && rm -rf /tmp/relcheck
```

The printed version must match the bump. If it prints the old version, the two
version files were out of sync — fix and rebuild.

## 4. Publish (ask the user before this step)

```bash
.venv/bin/twine upload dist/*     # needs PyPI token (~/.pypirc or TWINE_* env vars)
```

## 5. Tag and record

```bash
git tag vX.Y.Z && git push --tags
```

Then: create the GitHub release with notes, tick the release items in
`docs/launch/CHECKLIST.md`, and verify the new version appears on
`https://pypi.org/project/agenticmemo/` (can take a minute).
