# Contributing

Thank you — but read this first, because MagWave has an unusually strict bar.

## The one rule

**Bring a benchmark, not an opinion.** A change that alters a number must show which validation rung
proves it (see [docs/CHECKLISTS/validation.md](docs/CHECKLISTS/validation.md)). "It looks right" is not
acceptance, and a plausible field plot is the most dangerous artefact this project can produce.

## Getting set up

```bash
git clone https://github.com/jblane1000/magwave.git
cd magwave
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python tools/check_licence_boundary.py
ruff check . && ruff format --check .
pytest -q
```

GUI and CAD work needs the heavier extras: `uv pip install -e ".[dev,gui,cad]"`.

## Ground rules

- **Licence boundary:** never `import gmsh` or Elmer. They are external processes (ADR 0002). CI fails the
  build if you do.
- **Offline by default:** no network calls at runtime (NFR-6); tests run with no network.
- **Units:** SI internally; declare units at boundaries; a number without a unit is a bug (NFR-2).
- **No silent drops:** any body, material or boundary that cannot be honoured is reported (NFR-8).
- **Stable IDs:** PRD requirement IDs and problem codes are never renumbered.
- **Determinism:** seeded, reproducible; a change in results needs a changelog line explaining why.

## Pull requests

- One logical change per PR; reference the requirement ID it satisfies (e.g. `FR-011`).
- New feature ⇒ new test ⇒ (for numerics) new benchmark case with source and tolerance.
- Update `CHANGELOG.md` under "Unreleased".
- If you touched an engine's maths, attach the before/after benchmark table in the PR description.

## Commit style

Short imperative subject, body explains *why*. Example: `engine: reject permeable bodies in analytic tier
(FR-041)`.
