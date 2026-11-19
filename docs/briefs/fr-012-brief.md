# Brief — FR-012: booleans + defeaturing (completes M4)

**Owner of implementation:** Cody. **Engine:** you drive `agy` (Antigravity) for the code.
You write this brief's expansion if needed, you verify, you commit, you open the PR.
Max (coordinator) re-runs the acceptance commands afterwards and does not write implementation code.

## Repository state (verified by Max, 2026-11-19 18:2x)

- Repo: `/home/jblane1000/projects/magwave`
- Branch `main`, clean, HEAD `695f747` (M4 CAD import merged, PR #12).
- Baseline is green: **169 tests / 0 failures / 0 errors / 0 skipped**, bench **27/27 PASS**,
  `ruff check --no-cache .` clean, `tools/check_licence_boundary.py` → "licence boundary OK".

## Environment trap — read this first

Your shell inherits a Python 3.14 `PYTHONPATH` from the Hermes install environment, which shadows the
project venv and produces `No module named 'numpy._core._multiarray_umath'`. **Every python command must
be `env -u PYTHONPATH .venv/bin/python ...`.** Do not use bare `python` or `python3`.

## Requirement (PRD §6.2, verbatim)

> **FR-012** Boolean union/subtract and defeaturing for small features below a user threshold.
> **AC:** A part with 40 decorative fillets defeats to a clean solid and the volume change is reported.

## What to build

1. **New module `src/magwave/cad/defeature.py`**, following the existing lazy-OCP pattern in
   `src/magwave/cad/step.py` (`_load_ocp()` — OCP is imported only when a CAD path is used, never at
   package import):
   - `boolean_union(shapes: Sequence[Any]) -> BooleanResult` — `BRepAlgoAPI_Fuse`, sequentially, robust
     to >2 inputs; returns the result shape plus `volume_before` (sum of inputs), `volume_after`, and
     `volume_delta`.
   - `boolean_subtract(base: Any, tools: Sequence[Any]) -> BooleanResult` — `BRepAlgoAPI_Cut`, same
     reporting.
   - `defeature(shape: Any, *, max_feature_size_mm: float) -> DefeatureResult` — remove features whose
     characteristic size is below the threshold using `BRepAlgoAPI_Defeaturing` (OCCT ≥ 7.7, available in
     `cadquery-ocp 8.0.1.1.0`). Classify candidate faces by characteristic size (e.g. face bbox diagonal
     or surface area equivalent), skip non-feature faces, and record **what was removed**:
     `features_removed: int`, per-face descriptors (surface type + size), `volume_before`,
     `volume_after`, `volume_delta`, `threshold_mm`.
   - Failures must be **surfaced, not swallowed**: a feature that cannot be removed becomes an entry in
     the existing `Defect` dataclass from `cad/manifest.py` — the PRD line "no silent repair" applies.
2. **Manifest integration** — the import manifest reports the defeaturing outcome (threshold used,
   features removed, volume before/after/delta) so it survives into the report provenance. Extend
   `cad/manifest.py` with a small dataclass; do not change existing field names or required-ness.
3. **CLI** — extend the existing `magwave import` path in the CLI with
   `--defeature-below MM` (opt-in; absent = current behaviour, byte-identical manifests for existing
   callers). Keep `--json` output shape backward compatible: new keys may be added, none renamed.
4. **Tests** (in `tests/test_cad_import.py` or a new `tests/test_defeature.py`):
   - Fixture builder in `tests/fixtures/cad_builders.py`: `build_filleted_block()` — a block with
     **40 fillets** whose radius is below the test threshold, generated programmatically, no binary
     fixture committed.
   - Defeaturing that block yields a valid solid (one shell, `IsValid()`), `features_removed > 0`, and
     `volume_after` within 1 % of the analytic un-filleted block volume; the report states the delta.
   - Boolean union of two overlapping boxes: `volume_after` matches sum-minus-overlap analytically
     (tolerance 1e-6 relative); subtraction yields the analytic remainder.
   - Skip cleanly when `cadquery-ocp` is absent, matching the existing OCP-guard pattern.
5. **Docs** — update the FR-012 line in `docs/PRD.md` from open to satisfied with a one-line how;
   add the M4 note in `docs/ROADMAP.md`; if you need a new dependency, add an ADR — otherwise none.

## Hard fence

- Core imports stay Qt-free **and OCP-free** (lazy load only, as today).
- Licence boundary must stay clean: no GPL linkage, no new runtime dependency without an ADR.
- Do **not** touch the 27 frozen benchmark expected values or the analytic cases.
- Do not rename or remove existing public functions/CLI flags.
- Do not merge to `main`. Open a PR titled `M4: FR-012 booleans and defeaturing` against `main`.

## Acceptance commands (Max re-runs these; you must report each)

```bash
cd /home/jblane1000/projects/magwave
env -u PYTHONPATH .venv/bin/python -m pytest --junit-xml=/tmp/cody-fr012.xml -q
env -u PYTHONPATH .venv/bin/python -m magwave bench
.venv/bin/ruff check --no-cache .
env -u PYTHONPATH .venv/bin/python tools/check_licence_boundary.py
```

Pass criteria: no existing test regresses (baseline 169, plus your new tests), bench stays
`27/27 PASS`, ruff clean, licence gate OK.

## Report

Write your handover to `/home/jblane1000/projects/magwave/docs/briefs/fr-012-report.md`:
branch name, commit sha, PR URL, one line per acceptance command with its real output tail, what you
removed/defeated and the measured volume delta, and any deviation from this brief with the reason.
Reply to the dispatch with branch, sha, PR URL and one line per acceptance command.

State explicitly in the report **how many `agy` invocations produced the code** — if the count is zero,
say so plainly rather than omitting it.
