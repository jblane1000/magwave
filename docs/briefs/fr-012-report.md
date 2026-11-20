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

---

# Fix round — 2026-11-19 (F1–F5, brief `docs/briefs/fr-012-fix-brief.md`)

Same branch `fr-012-booleans-defeature`, same PR #13, **not merged**. Work order handed to the engine:
`docs/briefs/fr-012-fix-agy-brief.md` (committed here — it carries the measured OCP facts the
coordinator's brief did not have).

**Engine: `agy` (Antigravity) — `gemini-pro-orchestrator-g1-pro` via
`agy-work --model … --print-timeout 0 -p '<pointer>'`. Number of `agy` invocations that produced the
code: 1** (11m 43s wall, exit 0). I wrote the work order, both briefs, this section, the commit
message and one repair to agy's own output (see F3 below); no implementation code is mine except that
repair, which is a 4-line deletion plus its test.

Every number in this section is from **my** runs after agy returned, not from agy's summary. Where I
quoted agy's own report, it is marked.

## F5 — the size metric (Max's finding)

**Before** (`HEAD` of this branch, bbox-diagonal metric), 80×50×12 block with the four vertical edges
filleted R0.5, `defeature(block, max_feature_size_mm=…)`:

| threshold | faces removed | reported `size_mm` | volume |
|---|---|---|---|
| 1.0 | 0 | — | 47997.4248 unchanged |
| **2.0** | **0** | — | 47997.4248 unchanged |
| 5.0 | 0 | — | 47997.4248 unchanged |
| 20.0 | 4 | 12.020815 (bbox diagonal) | 47997.4248 → 48000.0000 |

Reproduced independently this round; it matches Max's measurement (0 at 2 mm, 0 at 5 mm, 4 at 20 mm).

**After**, same shape, `max_feature_size_mm=2.0`:

| quantity | measured |
|---|---|
| faces removed | **4**, all `Cylinder`, every `size_mm == 0.5` |
| `BRepCheck_Analyzer(shape).IsValid()` | `True` |
| volume | `47997.424777960754 → 47999.999999999993 mm³` |
| `volume_delta` | **`+2.5752220392387 mm³`** |
| threshold 0.4 (just below R0.5) | 0 removed, delta 0 — the threshold now means what it says |
| threshold 20.0 | 4 removed — unchanged, no over-removal |

**What changed.** `_candidate_faces` now sizes a face by characteristic local size:
`Cylinder().Radius()`, `Torus().MinorRadius()`, `Sphere().Radius()`, bounding-box diagonal for other
non-planar surfaces; planes stay excluded. `FaceDescriptor.size_mm`, the module docstring, the
`defeature()` docstring and the PRD FR-012 measured paragraph all state the new meaning.

### F5b — one deviation from the brief's literal rule, forced by a measurement

The brief says "cylindrical face → radius", full stop. Implemented literally, that **deletes the AC
fixture's through-holes**: the 80×50×12 40-hole block has 40 Ø0.6 bores whose walls are `Cylinder`
faces with R0.3 < 2.0, so they join the candidate set. Measured on this branch with the literal rule:

```
thr 2.0 -> 80 candidates (40 Torus r=0.15 + 40 Cylinder R=0.3)
batch with ALL 80 -> IsDone=True, valid=True, volume 47863.8785 -> 48000.0000  (delta +136.1215 mm³)
```

OCCT happily defeats the bore walls and fills all 40 holes. That is exactly the "removes legitimate
geometry" failure FR-012 exists to prevent, and it breaks the AC test's `features_removed == 40` /
all-`Torus` assertions — which the brief's fence protects. So the fix carries one additional rule:

> A face that is a **complete revolution** is a bore or a boss wall, never a blend, and is not a
> candidate. Cylinder `u` span ≥ 2π−1e-6; torus `v` span ≥ 2π−1e-6; sphere `v` span ≥ π−1e-6.

Measured parametric spans that justify the discrimination (this OCP build):

| face | type | radius | u span | v span |
|---|---|---|---|---|
| bore wall in the AC fixture | Cylinder | 0.300 | `2.000π` | `3.772π` |
| rim fillet in the AC fixture | Torus | 0.150 (minor) | `2.000π` | `0.500π` |
| vertical edge fillet, corner block | Cylinder | 0.500 | `0.500π` | `3.820π` |

A blend along a straight edge can never be a full revolution about the cylinder axis, so the rule
cannot exclude a real fillet; a full revolution is a bore, a boss or a closed shell. With the rule in
place the AC fixture is unchanged: 40 candidates, 40 removed, all `Torus`, `size_mm == 0.15`, volume
`47863.8785 → 47864.2832` (`+0.4047`), valid — the same numbers as before the fix.

A tangency (blend-continuity) test was measured as an alternative discriminator and rejected: the
rim fillets are tangent along both boundary edges (0.0°) and the bore walls along one (tangent 0.0°,
sharp 90.0°), but a *vertical* edge fillet is tangent on its two side edges and sharp on its top and
bottom rims (94 % tangent boundary length), so a naive "all edges tangent" rule rejects the very
feature F5 is about. The parametric-span rule needs no adjacency map (`TopTools` data maps are not
exposed by this OCP build) and separates the cases cleanly.

### Test value that had to change

`test_defeature_threshold_too_small_removes_nothing` used `max_feature_size_mm=0.5` and asserted 0
removed. Under characteristic-radius semantics the AC fixture's R0.15 fillets are *below* 0.5, so
0.5 is no longer "too small". The threshold is now **0.1** (below the smallest feature, R0.15) and
**every assertion in that test is unchanged** — intent preserved, and it is a stronger test against
the new metric. The AC test (`test_defeature_forty_fillets`) was not touched.

## F4 — validity gate

**Chosen policy: return with a `Defect`, never raise.** An invalid result is never returned as a
success: `defeature()` returns the **original input shape** with `features_removed=0`,
`removed_features=[]`, `volume_after == volume_before`, `volume_delta == 0.0` and a
`Defect(kind="defeature_invalid", …)` naming how many removals were discarded.

- The batch result is gated on `BRepCheck_Analyzer(shape).IsValid()`; an invalid batch is discarded
  and the per-face fallback runs instead.
- The final result of the fallback path is gated too; if it fails, the input shape is returned with
  the defect.
- `"defeature_invalid"` appended to `DEFECT_KINDS` (`src/magwave/cad/manifest.py`), so it validates,
  round-trips through `to_dict`/`from_dict`, and serialises in a `Manifest`.

## F2 — OCP failures become a clean CLI error

`src/magwave/cli.py`: new `_is_ocp_failure(exc)` walks `type(exc).__mro__` and returns true when any
class comes from an `OCP`-prefixed module; `_run_import` gained `except Exception` that handles the
OCP case as `magwave import: <one line>` + **exit 1** and re-raises anything else. `cli.py` still
imports no OCP at module scope.

**Measured fact that invalidated the brief's suggested mechanism:** in this build OCP exceptions do
*not* derive from `Standard_Failure`. `OCP.gp.gp_Dir(0,0,0)` raises
`OCP.OCP.Standard.Standard_ConstructionError` with MRO
`[Standard_ConstructionError, Exception, BaseException, object]` — no `Standard_Failure` in it (its
own `__module__` is `OCP.OCP.Standard`). A `isinstance(exc, Standard_Failure)` or MRO-name check
would have missed every real OCP error, so the check is module-prefix based.

## F1 — fallback no longer uses stale face handles

`_defeature_shapes(ocp, shape, faces)` is now the single seam both paths use. The fallback re-fetches
the candidate list from `current` before every removal, removes one face, then restarts from a fresh
fetch — so no handle ever outlives the shape it came from. Failure to remove after a full sweep stops
the loop and reports the remainder as `defeature_failed`, as before.

Measured on the 40-fillet AC fixture with the batch forced to fail, **old vs new** (same harness,
`BRepAlgoAPI_Defeaturing` wrapped so the batch instance reports not-done):

| | removed | valid | volume delta | defects |
|---|---|---|---|---|
| old (`HEAD`) | **3 / 40** | True | `+0.0304` | `defeature_failed`: could not remove 37 … (1.4013..1.4013 mm) |
| new | **40 / 40** | True | `+0.4047` | none |

## F3 — module/function shadowing resolved

`magwave.cad` no longer re-exports a name called `defeature`. The package exports the function as
`defeature_shape`, and `magwave.cad.defeature` is the submodule again:

- canonical: `from magwave.cad.defeature import defeature`
- alias: `from magwave.cad import defeature_shape`
- `import magwave.cad.defeature as df; df.defeature(...)` now works — `df` is the module.

`from magwave.cad import defeature` now yields the **module** (a loud `TypeError` if called), which is
the standard Python outcome and is asserted in the test rather than hidden.

**Repair to agy's output (mine, 4 lines):** agy's first attempt ended `cad/__init__.py` with
`globals().pop("defeature", None)` to satisfy "the name no longer shadows the submodule". That deletes
the *submodule* attribute, so `import magwave.cad.defeature` followed by
`magwave.cad.defeature.defeature(...)` raised `AttributeError: module 'magwave.cad' has no attribute
'defeature'` (verified). The pop is unnecessary — not re-exporting the callable is enough — so I
removed it and strengthened `test_defeature_module_import_is_unambiguous` to assert
`isinstance(cad.defeature, types.ModuleType)` and identity with the submodule. agy's own test had
asserted `not hasattr(cad, "defeature")`, i.e. it had locked in the bug.

## New tests (6 — 185 → 191)

1. `tests/test_defeature.py::test_defeature_edge_fillet_uses_radius_not_bbox` — the F5 guard.
   **Verified to fail on pre-fix code**: with `HEAD`'s `defeature.py` restored, it fails at
   `assert 0 >= 4` (`features_removed=0`, volume unchanged `47997.424777960754`). It passes after.
2. `tests/test_defeature.py::test_defeature_bore_wall_is_not_a_candidate` — no `Cylinder` in
   `removed_features` on the AC fixture, `volume_after ≈ 47864.2832` (holes not filled).
3. `tests/test_defeature.py::test_defeature_invalid_result_is_reported_not_returned` — F4, via
   monkeypatched `_shape_is_valid`; asserts original shape returned, 0 removed, `defeature_invalid`
   present, and that the kind round-trips.
4. `tests/test_defeature.py::test_defeature_fallback_removes_all_candidates_when_batch_fails` — F1,
   batch forced to fail through `_defeature_shapes`; asserts ≥ 30 of 40 (measured 40).
5. `tests/test_defeature.py::test_defeature_module_import_is_unambiguous` — F3.
6. `tests/test_cad_import.py::test_cli_import_ocp_failure_is_clean_error` — F2, runtime-built
   `Standard_Failure` subclass with `__module__ = "OCP.OCP.Standard"`; exit 1, single line, no
   traceback. Also asserts `_is_ocp_failure(ValueError())` is False.

New fixture: `tests/fixtures/cad_builders.py::build_corner_filleted_block(radius=0.5)` — 80×50×12 box,
four vertical edges filleted, built at runtime, no binary fixture committed. Edge selection uses
`Bnd_Box` with **1e-6** tolerances: `BRepBndLib` inflates the box by the edge tolerance (~1e-7), so a
1e-9 test selects nothing and `MakeFillet.Build()` raises
`Standard_Failure: There are no suitable edges for chamfer or fillet`.

## Acceptance — my runs on the final tree

Working directory `/home/jblane1000/projects/magwave`, every python command with
`env -u PYTHONPATH`:

1. `env -u PYTHONPATH .venv/bin/python -m pytest --junit-xml=/tmp/cody-fr012-fix.xml -q`
   → **exit 0**; JUnit XML: `tests=191 failures=0 errors=0 skipped=0`, 16.47 s (baseline 185 + 6).
2. `env -u PYTHONPATH .venv/bin/python -m magwave bench` → **exit 0**, `27/27 PASS`. The frozen
   corpus (`benchmarks/*.json`, `benchmarks/checks.py`) is untouched by this change — `git diff
   --stat` contains no path under `benchmarks/` or `analytic.py` — so the 27 values are the same
   cases evaluated by unchanged code.
3. `.venv/bin/ruff check --no-cache .` → **exit 0**, `All checks passed!`
4. `env -u PYTHONPATH .venv/bin/python tools/check_licence_boundary.py` → **exit 0**,
   `licence boundary OK (no GPL linkage)`

Also checked: `import magwave, magwave.cad, magwave.cad.defeature, magwave.cli` leaves `'OCP' in
sys.modules` **False** (lazy-load fence intact); `benchmarks/*.json`, `benchmarks/checks.py`,
`analytic.py`, `pyproject.toml`, `docs/adr/` untouched (`git diff --stat` names no path under them;
the change is 9 tracked files + 2 new briefs).

