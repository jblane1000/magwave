# Agent instructions

This file is for AI coding agents working in this repository. Read `CONTRIBUTING.md` first — it is the
human-facing contract, and this is its operational form. Where they differ, `CONTRIBUTING.md` wins.

## The command set you must run before you say "done"

```bash
pytest -q
python -m magwave.cli bench
ruff check . && ruff format --check .
python tools/check_licence_boundary.py
python tools/adr_check_map.py
```

Paste the real output, including the summary lines. "Tests pass" without a count is not evidence, and a
change is not finished because the code compiles.

## Hard rules

- **Never weaken a gate to pass it.** Not a tolerance, not a benchmark expectation, not an assertion, not
  a skipped test. A failing gate is a finding to report, with the numbers.
- **New feature ⇒ new test**, and for anything numeric a benchmark case with its source and tolerance.
- **A decision someone could question later ⇒ ADR** in `docs/adr/`. Every constraint in that ADR should
  name the check that enforces it, or be labelled "prose only" — a rule nothing enforces is a comment.
- **Changing engine semantics ⇒ bump `engine_version`** and say so in the changelog. The version string
  must never describe older code than the tree it ships from.
- **Refusals are part of the interface.** A boundary that can fail gets one named code (`E_*`), one line
  on stderr, exit 1, no traceback. Convert once in the CLI, not with a `try` at every call site.
- **Validate before you spend.** Check output paths, mesh budgets and case schema before solving, not
  after. A refusal must arrive before the allocation it is refusing.
- **Respect the licence boundary** (ADR 0002): never `import gmsh` or Elmer, and never import `PySide6` or
  `OCP` outside the `gui`/`cad` extras. `tests/test_lint_fence.py` enforces both.
- **Nothing internal in tracked files.** No absolute home paths, account names, quota figures, brief
  text, or infrastructure details. `docs/briefs/` is local and gitignored; it never gets committed.

## The default path stays bit-identical

Unless the change is meant to alter results, the numeric output must not move. `python -m magwave.cli
bench` must stay 27/27 and `bench --json` must stay byte-identical to the previous release. When a change
*does* move a number, say so explicitly and explain why in the PR body.

## Write tests that can fail

- Prefer a real tolerance with a stated basis over `==` on floats.
- An absolute tolerance looser than the value under test makes every assertion pass. Compare magnitudes.
- Where a check is meant to catch a regression, perturb the value by hand once and confirm the test fails.

## Where things live

| Path | What |
|---|---|
| `src/magwave/cli.py` | command surface: `solve`, `report`, `bench`, `converge`, `import` |
| `src/magwave/cases.py` | case schema, including the optional `mesh` block |
| `src/magwave/converge.py` | mesh ladders and Richardson estimates |
| `src/magwave/fem2d/` | Tier 1 axisymmetric solver: `engine`, `mesh`, `force` |
| `tools/` | licence, ADR and engine guards, plus the Elmer oracles |
| `docs/adr/` | decision records with their enforcing checks |
| `docs/ROADMAP.md` | milestones and their gates — the definition of "next" |
| `docs/CHECKLISTS/validation.md` | which validation rung proves which claim |

## Commit and PR style

Short imperative subject, body explains *why*, reference the requirement ID (`FR-###`) it satisfies. One
logical change per pull request. Add a `CHANGELOG.md` line under "Unreleased". If you touched an engine's
maths, attach the before/after benchmark table.
