# agy brief — FR-012: booleans + defeaturing (completes M4)

Implementation work order. You write the code, the tests, and the two doc edits. You do **not** run
`git` (no commit, no branch, no push). Cody verifies and commits.

Repo root: `/home/jblane1000/projects/magwave` (branch `fr-012-booleans-defeature`, already checked out).

## Environment trap — read this first

Your shell may inherit a Python 3.14 `PYTHONPATH` that shadows the project venv and breaks numpy with
`No module named 'numpy._core._multiarray_umath'`. **Every python command must be**:

```bash
cd /home/jblane1000/projects/magwave
env -u PYTHONPATH .venv/bin/python ...
```

Never use bare `python` / `python3`. `cadquery-ocp` 8.0.1.1.0 and `trimesh` are installed in `.venv`;
`pytest` is installed in `.venv`.

## Requirement (PRD §6.2, verbatim)

> **FR-012** Boolean union/subtract and defeaturing for small features below a user threshold.
> **AC:** A part with 40 decorative fillets defeats to a clean solid and the volume change is reported.

## Baseline (do not regress)

169 tests, 0 failures / 0 errors / 0 skipped; `magwave bench` → `27/27 PASS`; `ruff check --no-cache .`
clean; `tools/check_licence_boundary.py` → `licence boundary OK`. The 27 frozen benchmark expected
values and the analytic cases are **out of bounds** — do not touch `benchmarks/*.json`,
`benchmarks/checks.py`, `analytic.py` or any expected value.

## What to build

### 1. New module `src/magwave/cad/defeature.py`

Follow the lazy-OCP pattern of `src/magwave/cad/step.py`: a module-level `_load_ocp() -> dict[str, Any]`
that imports OCP symbols **inside the function** and raises
`MissingExtraError("step", "cadquery-ocp")` (import it from `.` inside the function, as `step.py` does)
on `ImportError`. Nothing from `OCP` may be imported at module scope; `magwave.cad.defeature` must
import cleanly with OCP absent (`tests/test_package.py::test_every_module_imports` enforces this).

Verified OCP 8.0.1 API facts (measured on this host — do not substitute remembered pythonocc idioms):

- Downcasts are `TopoDS.Edge(shape)` and `TopoDS.Face(shape)`. **`TopoDS.Edge_s` / `TopoDS.Face_s` do
  not exist in this build** (`AttributeError`).
- `BRepAlgoAPI_Fuse(a, b).Shape()`, `BRepAlgoAPI_Cut(base, tool).Shape()`.
- `BRepAlgoAPI_Defeaturing()` → `.SetShape(shape)`, `.AddFaceToRemove(face)`,
  `.AddFacesToRemove(seq)`, `.SetRunParallel(False)`, `.Build()`, `.IsDone()`, `.Shape()`.
  `AddFacesToRemove` requires an OCCT list, **not** a Python list:
  `from OCP.OCP.collections import List_TopoDS_Shape` then `.Append(face)` each.
- Face bbox: `Bnd_Box()` + `BRepBndLib.Add_s(face, box)`; characteristic size =
  `box.CornerMin().Distance(box.CornerMax())`.
- Surface type: `BRepAdaptor_Surface(face).GetType()` (compare against `GeomAbs_Plane`,
  `GeomAbs_Cylinder`, `GeomAbs_Torus` from `OCP.GeomAbs`).
- Volume: `GProp_GProps()` + `BRepGProp.VolumeProperties_s(shape, props)`, `abs(float(props.Mass()))`.
- Validity / solid count: `BRepCheck_Analyzer(shape).IsValid()`; `TopExp_Explorer(shape, TopAbs_SOLID)`.

Public API (exact names and semantics):

```python
@dataclass
class BooleanResult:
    shape: Any
    volume_before: float      # sum of input volumes for union; base volume for subtract
    volume_after: float
    volume_delta: float       # volume_after - volume_before (signed)

@dataclass
class FaceDescriptor:
    surface_type: str         # GeomAbs_* name without the prefix, e.g. "Torus"
    size_mm: float            # bbox diagonal

@dataclass
class DefeatureResult:
    shape: Any
    threshold_mm: float
    features_removed: int
    removed_features: list[FaceDescriptor]
    volume_before: float
    volume_after: float
    volume_delta: float
    defects: list[Defect]     # from .manifest; body_index is None, caller sets it

def boolean_union(shapes: Sequence[Any]) -> BooleanResult
def boolean_subtract(base: Any, tools: Sequence[Any]) -> BooleanResult
def defeature(shape: Any, *, max_feature_size_mm: float) -> DefeatureResult
```

Behaviour:

- `boolean_union` fuses **sequentially** (`result = shapes[0]`, then fuse each subsequent), so 2 and
  n>2 inputs both work. `volume_before` = sum of the input volumes measured individually.
  `volume_delta` is negative for overlapping inputs — that is correct, do not clamp it.
- `boolean_subtract` cuts **sequentially** against the growing result; `volume_before` = base volume.
- An empty input sequence raises `ValueError` naming the argument. A boolean whose `Shape()` is null
  raises `ValueError` (never return a null shape as if it succeeded).
- `defeature` raises `ValueError` when `max_feature_size_mm` is not finite or `<= 0`.
- Candidate faces: bbox diagonal `< max_feature_size_mm` **and** surface type is not `GeomAbs_Plane`
  (a small planar face is not a feature; skipping it keeps this conservative). Record each removed
  face's `FaceDescriptor`.
- Removal strategy, in this order:
  1. Try all candidates in one `BRepAlgoAPI_Defeaturing` call (`AddFacesToRemove`). If `IsDone()` and
     the resulting `Shape()` is non-null, that is the result.
  2. Otherwise fall back to one face at a time on the current shape (`AddFaceToRemove`), accumulating
     the successful result and continuing classification on the rebuilt shape each pass. Bound the
     passes (e.g. 3) so a pathological shape cannot loop forever.
  3. Any candidate that still cannot be removed is **surfaced, not swallowed**: append one
     `Defect(kind="defeature_failed", detail=..., body_index=None)` summarising the count and the size
     range (one aggregated defect per pass — not 40 near-identical lines).
- `volume_delta == volume_after - volume_before` exactly as computed; never fabricate or zero it.
- Do not validate-and-throw away the shape silently: if the final shape is null, raise `ValueError`.

### 2. `src/magwave/cad/manifest.py` — report the defeaturing outcome

This module is **standard-library only** — keep it that way (`dataclasses`, `json`, `math`, `typing`).

- Add `"defeature_failed"` to `DEFECT_KINDS` (append; do not rename or reorder the existing kinds).
- Add two small dataclasses, validated in `__post_init__` in the style of `Body` / `Defect`
  (`_check_int`, `_check_float`; reject bool-as-int), each with `to_dict` / `from_dict`:
  - `RemovedFeature`: `surface_type: str`, `size_mm: float`.
  - `DefeatureRecord`: `body_index: int`, `threshold_mm: float`, `features_removed: int`,
    `volume_before_mm3: float`, `volume_after_mm3: float`, `volume_delta_mm3: float`,
    `removed_features: list[RemovedFeature] = field(default_factory=list)`.
- `Manifest` gains `defeatures: list[DefeatureRecord] = field(default_factory=list)`.
- **Backward compatibility is a hard requirement:**
  - `Manifest.to_dict()` must **omit** the `"defeatures"` key entirely when the list is empty, so a
    manifest produced by the current code path is byte-identical to today's (`to_json` sorts keys).
  - `Manifest.from_dict()` treats `"defeatures"` as **optional** and still rejects unknown keys.
  - Existing field names, order-independent key set, and required-ness for `path/format/units/bodies/
    defects/warnings` are unchanged.

### 3. Wiring in `src/magwave/cad/step.py`, `cad/__init__.py`, `cli.py`

- `import_step(path, *, check_interference=False, defeature_below: float | None = None)`. After the
  body loop, when `defeature_below is not None`: for each collected body shape run
  `defeature(shape, max_feature_size_mm=defeature_below)`; replace that body's `volume_mm3` with
  `volume_after` (rebuild the `Body`, do not just mutate the field), append a `DefeatureRecord`, and
  append each returned `Defect` with `body_index` set to that body's index. The `body_shapes` list is
  already parallel to `bodies` — use that.
- `import_iges(path, *, defeature_below: float | None = None)` — same treatment on the single healed
  shape (if it yielded 0 solids, still attempt, and let the existing `imported_as_shell` defect stand).
- `import_cad(path, *, format=None, check_interference=False, defeature_below=None)` — pass through for
  STEP and IGES. For mesh formats **raise** `ValueError("--defeature-below applies to STEP and IGES "
  "only; mesh files are not defeatured")` rather than ignoring the flag (no silent drops).
- `cad/__init__.py`: export `defeature`, `boolean_union`, `boolean_subtract`, `BooleanResult`,
  `DefeatureResult`, `FaceDescriptor`, `RemovedFeature`, `DefeatureRecord` (add to `__all__`). Keep the
  existing names and the lazy import behaviour — `import_cad` must not import `defeature` at module
  scope.
- `cli.py`: add `--defeature-below MM` (`metavar="MM"`, `type=float`, `default=None`) to the shared
  parser with help text `import: remove features smaller than MM (STEP/IGES only)`. Guard it like the
  existing flags: `if args.command != "import" and args.defeature_below is not None:
  parser.error(...)`. Thread it through `_run_import(cad_path, json_path, role_specs, defeature_below)`
  into `import_cad`. In `_format_manifest_summary`, when `manifest.defeatures` is non-empty print one
  line per record: body index, threshold, features removed, and
  `volume {before:.4f} -> {after:.4f} mm^3 (delta {delta:+.4f})`. `--json` output stays additively
  compatible: `defeatures` appears only when non-empty; no existing key is renamed or removed.

### 4. `tests/fixtures/cad_builders.py` — the 40-fillet fixture

Add, in the file's existing style (heavy imports inside the function, geometry built at runtime, **no
binary file committed**):

- `build_filleted_block()` → an OCP `TopoDS_Shape`: an 80 × 50 × 12 mm block with 40 decorative fillets.
- `build_filleted_step(out_path: Path)` → the same part written to STEP via the existing
  `_write_step_parts(out_path, [("filleted_block", shape)], as_assembly=False)`.

This construction is **measured to work on this host** — reproduce it, then verify:

1. Block `BRepPrimAPI_MakeBox(gp_Pnt(0,0,0), 80, 50, 12)`; volume 48000.0.
2. 40 through-holes of Ø0.6 (radius 0.3, height 14 starting at z=-1): 8 columns `x = 10.0 + i*8.6`
   (i=0..7), 5 rows `y = 9.0 + j*8.0` (j=0..4). Fuse the 40 cylinders first, then **one**
   `BRepAlgoAPI_Cut(block, fused)` → measured 47864.283197365 mm³.
3. Collect the top rims: iterate `TopExp_Explorer(cut, TopAbs_EDGE)`, `TopoDS.Edge(exp.Current())`,
   keep edges where `BRepAdaptor_Curve(e).GetType() == GeomAbs_Circle`, circle radius ≈ 0.3 and
   `circle.Location().Z() ≈ 12.0` (tolerance 1e-6). Measured: 80 edges found.
4. `BRepFilletAPI_MakeFillet(cut)`, `Add(0.15, edge)` for each, `Build()`, `IsDone()` →
   measured True, 86 faces, volume 47863.87847387621 mm³.
5. The 40 fillet faces are `GeomAbs_Torus` with bbox diagonal ≈ 1.40126 mm — i.e. a **2.0 mm threshold
   selects exactly those 40 faces** and nothing else.

### 5. `tests/test_defeature.py` — new test module

Module-level `pytest.importorskip("OCP")` (matching `tests/test_cad_import.py`). Tests:

- **`test_defeature_forty_fillets`** — the FR-012 AC. `defeature(build_filleted_block(),
  max_feature_size_mm=2.0)`:
  `features_removed == 40`; `len(removed_features) == 40` and every descriptor's `surface_type` is
  `"Torus"`; `BRepCheck_Analyzer(result.shape).IsValid()` is true; exactly **1** solid;
  `volume_after` within **1 %** of the analytic un-filleted block volume 48000.0 mm³;
  `features_removed > 0`; `volume_delta == pytest.approx(volume_after - volume_before)`;
  and the delta is reported, not zero — assert `abs(volume_delta) > 1e-9` and
  `volume_before > volume_after` (removing material increases volume here: fillets were cuts, so the
  defeated shape regains `~0.4047 mm³`; assert the sign you actually measure and comment why).
- **`test_boolean_union_overlap`** — two 10 mm cubes, second at `gp_Pnt(5,0,0)`:
  `volume_after == approx(1500.0, rel=1e-6)`, `volume_before == approx(2000.0)`,
  `volume_delta == approx(-500.0, rel=1e-6)`.
- **`test_boolean_subtract_overlap`** — base 10 mm cube minus the same tool:
  `volume_after == approx(500.0, rel=1e-6)`, `volume_before == approx(1000.0)`.
- **`test_boolean_union_three_shapes`** — three cubes (0,0,0), (5,0,0), (10,0,0): analytic union
  `1000*3 - 500*2 = 2000.0` (`rel=1e-6`). Proves n>2 robustness.
- **`test_defeature_threshold_too_small_removes_nothing`** — `max_feature_size_mm=0.5` (below the
  1.401 mm fillet faces): `features_removed == 0`, `volume_delta ≈ 0`, shape still valid.
- **`test_defeature_rejects_bad_threshold`** — `0`, `-1.0`, `float("nan")`, `float("inf")` each raise
  `ValueError`.
- **`test_defeature_failed_kind_is_representable`** — `"defeature_failed"` is in `DEFECT_KINDS` and a
  `Defect(kind="defeature_failed", detail="…")` round-trips through `to_dict`/`from_dict`.
- **Manifest compatibility** (no OCP needed, put in the same file above the importorskip-dependent
  tests only if that does not skip them — otherwise add them to `tests/test_cad_import.py`):
  a `Manifest` with no defeaturing serialises **without** a `"defeatures"` key; a manifest with one
  `DefeatureRecord` round-trips through `to_json`/`from_json`; `Manifest.from_dict` of a legacy dict
  (no `"defeatures"`) still loads.
- **CLI** (`tests/test_cad_import.py`, following the existing STEP-CLI test style):
  `main(["import", str(step), "--json", str(out), "--defeature-below", "2.0"]) == 0`; the JSON has a
  non-empty `"defeatures"` list with `features_removed >= 30` (STEP round-trip may split faces — assert
  the floor, do not fake 40) and `volume_after_mm3` within 1 % of 48000; the body's `volume_mm3`
  equals `volume_after_mm3`. And: the same import **without** the flag produces JSON with **no**
  `"defeatures"` key (byte-identical behaviour for existing callers).
- **Lazy-OCP guard** — subprocess check in the style of `tests/test_package.py`:
  `env -u PYTHONPATH .venv/bin/python -c "import sys, magwave.cad.defeature; assert 'OCP' not in
  sys.modules"` exits 0.

### 6. Docs

- `docs/PRD.md`, FR-012: keep the requirement text, append a measured line in the style already used by
  FR-010/FR-011, e.g. `**Measured 2026-11-19:** …` stating satisfied and how: `cad/defeature.py`
  (`boolean_union`, `boolean_subtract`, `defeature`), the manifest `DefeatureRecord`, and the real
  numbers your test run produced (features removed, volume before → after, delta, validity). **Use the
  numbers the test actually measured** — do not copy a number you did not observe.
- `docs/ROADMAP.md`: M4 row already lists defeaturing. Add one short note under the table (one or two
  lines) recording that FR-012 closed with `cad/defeature.py` + the 40-fillet regression test, dated
  2026-11-19. Do not restructure the table.

## Hard fence

- Core imports stay Qt-free **and OCP-free**: lazy load only. `manifest.py` stays standard-library only.
- No new runtime dependency, no `pyproject.toml` change, no ADR (OCP is already an ADR-0006 extra).
- Do not touch the 27 frozen benchmark expected values, `benchmarks/*.json`, or the analytic cases.
- Do not rename or remove existing public functions, dataclasses or CLI flags; no existing manifest key
  changes required-ness or disappears.
- Do not run `git` in any form. Do not create files outside the paths named above.
- Leave no debug prints, no `TODO`, no commented-out code in `src/`. Keep `ruff check` clean
  (`line-length 100`, rules `E,F,I,UP,B,SIM`, `E501` ignored).

## Acceptance — you must run these and paste the real tails

```bash
cd /home/jblane1000/projects/magwave
env -u PYTHONPATH .venv/bin/python -m pytest -q                      # 169 baseline + your new tests, 0 failures
env -u PYTHONPATH .venv/bin/python -m magwave bench                  # must stay 27/27 PASS
.venv/bin/ruff check --no-cache .
env -u PYTHONPATH .venv/bin/python tools/check_licence_boundary.py   # "licence boundary OK"
```

Iterate until all four are green. If a test you wrote is red, fix the code — do not weaken the
assertion, skip the test, or relax a tolerance to make it pass. If an expectation in this brief is
contradicted by what OCCT actually does, keep the honest assertion and report the measured number.

## Required final output from you

Reply with:

1. The list of files created/modified.
2. The real output tail of each of the four commands above (exit status included).
3. The measured numbers from `test_defeature_forty_fillets`: `features_removed`, `volume_before`,
   `volume_after`, `volume_delta`, valid/invalid, solid count.
4. The measured union/subtract volumes.
5. Any deviation from this brief and why — especially anything OCCT forced you to change.
