# Handover — FR-012: booleans + defeaturing (completes M4)

- **Branch:** `fr-012-booleans-defeature` (pushed to `origin`)
- **Implementation commit:** `2907ade437aa47197f623e94734c022c5aac10bb`
- **PR:** https://github.com/jblane1000/magwave/pull/13 — `M4: FR-012 booleans and defeaturing`, base `main`
- **Base:** `main` @ `695f747` (clean, baseline re-verified below before any change)
- **Not merged.** Max re-runs the acceptance commands.

## Engine

**Number of `agy` invocations that produced the code: 1.**

- Model: `gemini-pro-orchestrator-g1-pro` (Orchestrator/Argon) via `agy-work --model … --print-timeout 0 -p …`,
  run from the repo root on the branch.
- Work order handed to it: `docs/briefs/fr-012-agy-brief.md` (committed with this branch — it is the
  expanded brief, with the measured OCP API facts and the 40-fillet construction recipe).
- No implementation code was written by me. The only edits I made are: this report, the agy work
  order, the commit message, and the PR body. My pre-brief probe script (scratch, outside the repo)
  established the defeaturing recipe and the numbers below; it is not shipped.
- agy reported its own run as green and I re-ran everything independently; the numbers quoted here
  are from **my** runs, not from agy's summary.

## Baseline re-verified before work (on `main` @ `695f747`)

`169 tests / 0 failures / 0 errors / 0 skipped` (JUnit XML), `magwave bench` → `27/27 PASS`,
`ruff check --no-cache .` → clean, `tools/check_licence_boundary.py` → `licence boundary OK`.

## Acceptance commands — real results, re-run by me on the branch

Working directory: `/home/jblane1000/projects/magwave`.
`env -u PYTHONPATH .venv/bin/python` was used for every python command (the Hermes 3.14 `PYTHONPATH`
trap in the brief is real).

1. `env -u PYTHONPATH .venv/bin/python -m pytest --junit-xml=/tmp/cody-fr012.xml -q`
   → **exit 0**. JUnit XML `/tmp/cody-fr012.xml`: `tests=185 failures=0 errors=0 skipped=0`, time 8.884 s.
   Baseline 169 + 16 new tests. Note: pytest's own `N passed` summary line is not visible in the
   captured console text — OCCT's native stdout (`Step File Name : … Write Done`) is written by the
   C++ layer at interpreter teardown and lands after it in the stream, and piping through `grep`/`tail`
   does not find it. The XML is the authoritative count and is what I report.

2. `env -u PYTHONPATH .venv/bin/python -m magwave bench`
   → **exit 0**, tail:
   ```
   force_dipoles_oblique            |  5.240784322e-05 |  5.240784323e-05 |  4.99e-11 |    1e-09 | PASS
   cylinder_offaxis_Brho            |  4.596500281e-02 |  4.596499746e-02 |  1.16e-07 |    1e-06 | PASS
   cylinder_offaxis_Bz              |  7.369529239e-01 |  7.369529460e-01 |  3.00e-08 |    1e-06 | PASS
   27/27 PASS
   ```

3. `.venv/bin/ruff check --no-cache .`
   → **exit 0**, output: `All checks passed!`

4. `env -u PYTHONPATH .venv/bin/python tools/check_licence_boundary.py`
   → **exit 0**, output: `licence boundary OK (no GPL linkage)`

### Additional verification I ran beyond the four commands

- **Manifest byte-compatibility vs `main`.** `git worktree` at `695f747`, same STEP file, same roles,
  `magwave import` **without** `--defeature-below`: stdout `cmp` identical, JSON `cmp` identical,
  sha256 `8fb97844a35191bca256fe77f86778252940f31c935c8b98e54714f9bc481091` on both sides. Key set
  stays `['bodies','defects','format','path','units','warnings']` — `defeatures` is omitted when empty.
- **OCP-free import.** `import magwave, magwave.cad, magwave.cad.defeature` → `'OCP' in sys.modules`
  is `False`. Same via subprocess test in `tests/test_defeature.py`.
- **CLI end-to-end on the real part** (`filleted.step`, `--defeature-below 2.0 --role 0=permeable_solid`):
  exit 0, stdout `[0] threshold=2.0000 mm removed=40 volume 47863.8785 -> 47864.2832 mm^3 (delta +0.4047)`;
  JSON `defeatures[0]` = `{'body_index': 0, 'features_removed': 40, 'threshold_mm': 2.0,
  'volume_before_mm3': 47863.87847387614, 'volume_after_mm3': 47864.28319736492,
  'volume_delta_mm3': 0.4047234887839295}`, 40 `{'size_mm': 1.4012561310529488, 'surface_type': 'Torus'}`
  descriptors, body `volume_mm3` equals `volume_after_mm3`, delta internally consistent.
- **IGES path** (`box.igs`, 12×8×4): exit 0, `defeatures (1): [0] threshold=2.0000 mm removed=0
  volume 384.0000 -> 384.0000 mm^3`, and the pre-existing `imported_as_shell` defect still reported.
- **Flag fences:** mesh + `--defeature-below` → exit 1, `magwave import: --defeature-below applies to
  STEP and IGES only; mesh files are not defeatured`; `magwave bench --defeature-below 1.0` → exit 2,
  `bench: --defeature-below applies to import only`.
- **Scope:** `benchmarks/*.json`, `benchmarks/checks.py`, `analytic.py`, `pyproject.toml`, `docs/adr/`
  untouched (`git diff --stat` shows only the 8 files + new module + docs/briefs).

## What was removed / defeated, and the measured delta

Fixture: `tests/fixtures/cad_builders.py::build_filleted_block()` — 80 × 50 × 12 mm block, 40 Ø0.6 mm
through-holes in an 8 × 5 grid, and **40 R0.15 mm fillets on the top hole rims**, built at runtime (no
binary fixture committed).

`defeature(block, max_feature_size_mm=2.0)`, one OCCT `BRepAlgoAPI_Defeaturing` pass:

| Quantity | Measured |
|---|---|
| candidate / removed features | **40** — all `Torus`, bbox diagonal `1.4012561310529488 mm` |
| `volume_before` | `47863.87847387614 mm³` |
| `volume_after` | `47864.28319736492 mm³` |
| `volume_delta` | **`+0.4047234887839295 mm³`** |
| result | `BRepCheck_Analyzer.IsValid() == True`, exactly **1 solid** |
| analytic gross block | `48000.0 mm³` → `volume_after` is within **0.283 %** (brief allows 1 %) |

Reading of the numbers: the 40 fillets are *cuts* on the hole rims, so defeating them restores the
sharp edges and the volume goes **up** by 0.4047 mm³. The 0.283 % gap to 48000 mm³ is the 40 through-holes
(135.72 mm³ of material), which are not fillets and are correctly left in place — the hole walls are
12 mm tall (bbox diagonal ≈ 12 mm) and never enter the < 2 mm candidate set.

Booleans (analytic, `rel=1e-6`): two 10 mm cubes overlapping by 500 mm³ → union `1500.0`
(`volume_before` 2000, delta −500), subtract → `500.0` (`volume_before` 1000, delta −500);
three cubes at x = 0/5/10 → union `2000.0` (n > 2 robustness).

## Deviations from the brief, with reasons

1. **Sign of `volume_delta` in the 40-fillet test.** My brief contained a contradictory parenthetical
   ("removing material increases volume here … assert the sign you actually measure"). The code and test
   assert `volume_after > volume_before` (+0.4047 mm³), which is what physically happens and what was
   measured. The brief text was wrong, not the result.
2. **`cad/__init__.py` imports `.defeature` at module scope.** The fence I care about (OCP never loaded
   at import) holds and is verified. The consequence is a name shadowing: `magwave.cad.defeature` as an
   *attribute* of the package is now the `defeature` **function**, so `import magwave.cad.defeature as df`
   yields that function, not the module (verified: `AttributeError: 'function' object has no attribute
   '_load_ocp'`). `from magwave.cad import defeature` (the brief's API) and
   `from magwave.cad.defeature import defeature` both work. See finding F3.
3. **CLI test passes `--role 0=permeable_solid`.** `magwave import` returns 1 whenever a body is
   unassigned (pre-existing semantics, `cli.py:436-438`). Asserting exit 0 on a one-body STEP therefore
   needs the role; no CLI exit semantics were changed.
4. **Manifest compatibility tests are duplicated.** `tests/test_defeature.py` carries a module-level
   `pytest.importorskip("OCP")`, so the stdlib-only `Manifest`/`DefeatureRecord` round-trip test was
   also placed in `tests/test_cad_import.py` to keep it running when OCP is absent. Deliberate, cheap.
5. **Batch path does not re-run `BRepCheck`.** The brief's acceptance condition for the batch was
   `IsDone()` + non-null shape; validity is asserted on the result in the AC test instead. Not adding a
   validity gate inside `defeature()` was a deliberate scope call — a defeated-but-invalid solid is
   better reported by the caller than silently discarded — but it is a gap worth a decision (F4).

## Findings for the coordinator (reported, not silently fixed)

Quote-the-defect standard: `file:line`, what breaks, what triggers it.

- **F1 — per-face fallback removes ~1 face per pass.** `src/magwave/cad/defeature.py:264-286`: the
  fallback loop collects `pass_candidates` from `current`, then iterates that list while reassigning
  `current` on each success; the remaining `face` handles belong to the *previous* shape, so their
  removals fail. Measured by forcing the batch call to report not-done on the 40-fillet block:
  `features_removed: 3` (one per pass, `max_passes = 3`) plus one honest
  `defeature_failed: could not remove 37 candidate feature face(s) below 2.0000 mm`. The result is still
  valid/1 solid, and nothing is swallowed, so this is a degraded path rather than a wrong answer — but a
  correct fallback would re-classify after each single removal. Fix if you want it:
  take one candidate per iteration and restart the pass loop.
- **F2 — an OCCT exception escapes the CLI as a traceback.** `src/magwave/cli.py:416` catches
  `(MissingExtraError, FileNotFoundError, ValueError, OSError)`; `OCP.Standard.Standard_Failure`
  derives from `Exception` only (measured MRO: `Standard_Failure → Exception → BaseException`). So if
  `BRepAlgoAPI_Defeaturing.Build()` raises instead of returning `IsDone() == False`,
  `_run_import` does not catch it and the user gets a traceback rather than `magwave import: …` + exit 1.
  Not triggered by the fixture (the batch succeeds), so no test covers it.
- **F3 — module/function name shadowing.** `src/magwave/cad/__init__.py:11-18` re-exports the
  `defeature` function from the same-named module; `import magwave.cad.defeature as df` binds the
  function. Cosmetic for the documented API, confusing for anyone reaching for the module.
- **F4 — no validity gate on the batch result.** `src/magwave/cad/defeature.py:243-257` accepts any
  non-null `Shape()` from a done batch without `BRepCheck_Analyzer`. A defeaturing that "succeeds" into
  an invalid solid would be reported as clean geometry.
