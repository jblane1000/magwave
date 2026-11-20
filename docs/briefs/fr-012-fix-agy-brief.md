# agy work order — FR-012 fix round (PR #13): feature-size metric + F1–F4

You are working in `/home/jblane1000/projects/magwave`, branch `fr-012-booleans-defeature`.
Read `docs/briefs/fr-012-fix-brief.md` for the coordinator's framing, but **this file is the
authoritative work order** — it contains measured OCP facts that override anything in that brief.

## Environment — mandatory

Every python command **must** be run as:

```bash
cd /home/jblane1000/projects/magwave
env -u PYTHONPATH .venv/bin/python -m pytest -q ...
env -u PYTHONPATH .venv/bin/python -m magwave bench
.venv/bin/ruff check --no-cache .
env -u PYTHONPATH .venv/bin/python tools/check_licence_boundary.py
```

A Python 3.14 `PYTHONPATH` from an unrelated environment shadows the 3.12 venv; `env -u PYTHONPATH`
is not optional. Python 3.12, OCP (cadquery-ocp) installed in the venv, pytest installed.

Baseline on this branch before your changes: **185 tests, 0 failures, 0 errors, 0 skipped**;
`magwave bench` 27/27 PASS; ruff clean; licence boundary OK. That is the floor. After your changes
every one of those must still pass, plus the new tests.

---

## The work, five findings

### F5 — the feature-size metric must be the LOCAL feature size, not the face extent

`src/magwave/cad/defeature.py::_candidate_faces` currently sizes a face by its bounding-box
diagonal. That makes a 0.5 mm fillet running the full 12 mm height of a block report 12.02 mm, so a
2 mm threshold removes nothing. Measured on this branch on an 80×50×12 block with the four vertical
edges filleted R0.5: 4 `GeomAbs_Cylinder` faces, bbox diagonal `12.020815496462795 mm`,
`Cylinder().Radius() == 0.5`; `defeature(block, max_feature_size_mm=2.0)` removes **0**;
`max_feature_size_mm=20.0` removes 4.

**Required replacement rule** (implement exactly this):

1. Size, by surface type (`BRepAdaptor_Surface.GetType()`):
   - `GeomAbs_Cylinder` → `Cylinder().Radius()`
   - `GeomAbs_Torus` → `Torus().MinorRadius()`
   - `GeomAbs_Sphere` → `Sphere().Radius()`
   - anything else non-planar → bounding-box diagonal (today's fallback)
   - `GeomAbs_Plane` stays excluded, as today.
2. **A face that is a complete revolution is a bore or a boss wall, never a blend, and must not be a
   candidate.** This is not optional and it is what keeps the 40-fillet acceptance test honest — see
   the measurement below. Discriminate with the parametric span of the face:

   ```python
   du = surface.LastUParameter() - surface.FirstUParameter()
   dv = surface.LastVParameter() - surface.FirstVParameter()
   # Cylinder: du >= 2*pi - 1e-6  -> full-revolution wall
   # Torus:    dv >= 2*pi - 1e-6  -> closed doughnut
   # Sphere:   dv >= pi   - 1e-6  -> whole sphere
   ```
   Use a module constant for the tolerance (`_SPAN_TOL = 1e-6`), not a bare literal at three sites.

   Measured parametric spans on this branch (this is why the rule is needed):

   | fixture face | type | radius | u span | v span |
   |---|---|---|---|---|
   | 40 hole walls in `build_filleted_block` | Cylinder | 0.300 | `2.000π` | `3.772π` |
   | 40 rim fillets in `build_filleted_block` | Torus | 0.150 (minor) | `2.000π` | `0.500π` |
   | 4 vertical edge fillets, 80×50×12 block | Cylinder | 0.500 | `0.500π` | `3.820π` |

   Without rule 2, a radius metric makes the 40 Ø0.6 through-hole walls (R0.3 < 2.0) candidates, and
   OCCT's batch `BRepAlgoAPI_Defeaturing` **succeeds at filling all 40 holes**: measured
   `IsDone() == True`, 80 features removed, volume `47863.8785 → 48000.0000` (+136.1215 mm³),
   result valid. That silently deletes functional geometry and breaks the AC test
   (`features_removed == 40`, all `Torus`). Rule 2 is what prevents it.

   With the rule, measured on this branch's fixtures: 40 candidates / 40 removed / all `Torus` at
   `size_mm == 0.15` on the AC fixture (volume `47863.8785 → 47864.2832`, delta `+0.4047`, valid),
   and 4 candidates / 4 removed / all `Cylinder` at `size_mm == 0.5` on the R0.5 corner-fillet block
   (volume `47997.4248 → 48000.0000`, delta `+2.5752`, valid).

3. Document the new semantics so the threshold reads as "fillet radius / small-feature size":
   - `src/magwave/cad/defeature.py` module docstring: state the size rule and the
     complete-revolution exclusion, and say a threshold of N mm means "remove blends whose radius
     (or other local size) is below N mm".
   - `FaceDescriptor.size_mm` docstring/comment: characteristic local size, not face extent.
   - `defeature()` docstring: currently says "Remove non-planar feature faces with bbox diagonal
     below …" — it must say characteristic size.
   - `docs/PRD.md`, the FR-012 measured line (currently `docs/PRD.md:91-96`, mentions
     "bbox diagonal `1.401256 mm`"): rewrite that measured paragraph to describe the characteristic
     size (cylinder radius / torus minor radius / sphere radius / bbox diagonal fallback), the
     complete-revolution exclusion, and the new numbers from the table above. Keep the existing
     measured-line style and keep it factual.

**New tests for F5** (in `tests/test_defeature.py`):

- `test_defeature_edge_fillet_uses_radius_not_bbox` — build the 80×50×12 block with the four
  vertical edges filleted at **R0.5**, `defeature(..., max_feature_size_mm=2.0)` must remove
  **≥ 4** faces, all `Cylinder` surface type, every `size_mm == pytest.approx(0.5)`, result
  `BRepCheck_Analyzer(...).IsValid()`, and `volume_delta` reported as `volume_after - volume_before`
  with `abs(volume_delta) > 1e-9`. On today's code this removes 0 and fails — that is the point.
- `test_defeature_bore_wall_is_not_a_candidate` — on `build_filleted_block()` at
  `max_feature_size_mm=2.0`, assert no `Cylinder` face is ever reported in `removed_features`
  (the 40 R0.3 bore walls stay) and `volume_after` stays `pytest.approx(47864.2832, rel=1e-6)`,
  i.e. the holes were not filled.

**Existing test value that must change, with the reason recorded in the code comment:**
`test_defeature_threshold_too_small_removes_nothing` uses `max_feature_size_mm=0.5` and asserts
0 removed. Under radius semantics the AC fixture's R0.15 torus fillets (size 0.15) are *below* 0.5,
so that threshold is no longer "too small". Change the threshold to **0.1** (below the smallest
feature, R0.15) and keep every assertion as-is. Do not change the AC test
(`test_defeature_forty_fillets`) at all.

**New fixture** in `tests/fixtures/cad_builders.py`:

```python
def build_corner_filleted_block(radius: float = 0.5) -> object:
    """80x50x12 mm block with the four vertical edges filleted at ``radius``."""
```

Construction facts (all measured on this host — follow them, the naive version fails):
- Build the box with `BRepPrimAPI_MakeBox(80.0, 50.0, 12.0)`.
- Select the four vertical edges with a `TopExp_Explorer(shape, TopAbs_EDGE)` pass and a
  `Bnd_Box` per edge: vertical means `abs(mx.Z()-mn.Z()) > 11.0`,
  `abs(mx.X()-mn.X()) < 1e-6`, `abs(mx.Y()-mn.Y()) < 1e-6`.
  **The 1e-6 tolerances matter** — `BRepBndLib.Add_s` inflates the box by the edge tolerance
  (measured ~1e-7), so a 1e-9 test matches nothing and `BRepFilletAPI_MakeFillet.Build()` then
  raises `Standard_Failure: There are no suitable edges for chamfer or fillet`.
- Add each edge **once**: skip an edge if `any(edge.IsSame(seen) for seen in chosen)` (the explorer
  visits each edge once per adjacent face).
- `fillet.Build()`; raise `RuntimeError("BRepFilletAPI_MakeFillet failed for build_corner_filleted_block")`
  if `not fillet.IsDone()`.
- Verified working shape: 4 `GeomAbs_Cylinder` faces, radius 0.5, volume `47997.4248 mm³`.

### F4 — no validity gate on the accepted result

`defeature()` currently accepts the batch result on `IsDone()` + non-null. Add a
`BRepCheck_Analyzer(shape).IsValid()` gate. **Chosen policy (implement this one):** an invalid result
is never returned as a success. Return the **original input shape** with `features_removed=0`,
`removed_features=[]`, `volume_after == volume_before`, `volume_delta == 0.0`, and a
`Defect(kind="defeature_invalid", detail=..., body_index=None)` whose detail names how many
candidate removals were discarded.

Implementation:
- Add `BRepCheck_Analyzer` to the `_load_ocp()` dict and to its import block.
- Add a small helper `_shape_is_valid(ocp, shape) -> bool` (a single place the tests can monkeypatch).
- Gate the batch result: if `IsDone()` and non-null but invalid, do **not** accept it — fall through
  to the per-face fallback (below). If the fallback then produces an invalid shape, the final gate
  fires.
- Gate the final result of the fallback path too.
- Add `"defeature_invalid"` to `DEFECT_KINDS` in `src/magwave/cad/manifest.py` (append after
  `"defeature_failed"`). This is a schema tuple used for round-trip validation.

**New test** `test_defeature_invalid_result_is_reported_not_returned`: monkeypatch
`magwave.cad.defeature._shape_is_valid` to return `False` (do not try to fabricate an invalid solid),
call `defeature(build_filleted_block(), max_feature_size_mm=2.0)`, and assert
`features_removed == 0`, `removed_features == []`, `volume_delta == 0.0`,
`any(d.kind == "defeature_invalid" for d in result.defects)`, and that the returned shape is the
input shape (`result.shape.IsSame(original)`). Also assert `"defeature_invalid" in DEFECT_KINDS`
and that a `Defect(kind="defeature_invalid", ...)` round-trips through `to_dict`/`from_dict`.

### F2 — OCP failures escape the CLI as a traceback

`src/magwave/cli.py::_run_import` catches `(MissingExtraError, FileNotFoundError, ValueError, OSError)`
around `import_cad`. OCP exceptions escape as a traceback.

**Measured fact that matters:** in this OCP build the exceptions do *not* derive from
`OCP.Standard.Standard_Failure`. Measured MRO of a real raise:

```
OCP.gp.gp_Dir(0.0, 0.0, 0.0) -> OCP.OCP.Standard.Standard_ConstructionError
mro: ['OCP.OCP.Standard.Standard_ConstructionError', 'builtins.Exception',
      'builtins.BaseException', 'builtins.object']
Standard_Failure.__module__ == 'OCP.OCP.Standard'
```

So detect OCP failures by **module name over the MRO**, with no OCP import in `cli.py` (core must
stay OCP-free):

```python
def _is_ocp_failure(exc: BaseException) -> bool:
    """True when *exc* originates in OpenCASCADE (Standard_Failure and its siblings)."""
    for cls in type(exc).__mro__:
        module = getattr(cls, "__module__", "") or ""
        if module == "OCP" or module.startswith("OCP."):
            return True
    return False
```

In `_run_import`, keep the existing catch clause and error text style, and add a second clause:

```python
    except Exception as exc:
        if _is_ocp_failure(exc):
            print(f"magwave import: {_one_line(exc)}", file=sys.stderr)
            return 1
        raise
```

Single line, exit 1, no traceback, same `magwave import: ` prefix.

**New test** `test_cli_import_ocp_failure_is_clean_error` in `tests/test_cad_import.py`: monkeypatch
`magwave.cad.import_cad` to raise an OCP-shaped exception built at runtime

```python
fake = type("Standard_Failure", (Exception,), {"__module__": "OCP.OCP.Standard"})
raise fake("boom from OCCT")
```

call `main(["import", str(tmp_path / "x.step")])`, assert the return code is `1`, stderr contains
`magwave import: boom from OCCT`, and `"Traceback"` is not in stderr. Do not require a genuinely
failing build. Also assert `_is_ocp_failure(ValueError())` is `False`, so the helper does not swallow
ordinary errors.

### F1 — per-face fallback uses stale face handles

`defeature.py:264-286`: the fallback collects `pass_candidates` from `current`, then reassigns
`current` inside the loop, so every face after the first belongs to the pre-removal shape; measured
3 removed out of 40 and the rest reported as `defeature_failed`.

Rewrite the fallback so **every face handle comes from the current shape**. Required structure:

1. Add a single testable seam and use it for BOTH paths:

   ```python
   def _defeature_shapes(ocp, shape, faces: list[Any]) -> Any | None:
       """Run one BRepAlgoAPI_Defeaturing pass over *faces*; return the new shape or None."""
   ```
   It builds the `List_TopoDS_Shape`, calls `SetShape`/`AddFacesToRemove`/`SetRunParallel(False)`/
   `Build()`, and returns the shape only when `IsDone()` and non-null; otherwise `None`. It does NOT
   apply the validity gate (F4's gate lives in `defeature`).
2. Batch path: one call with all candidate faces.
3. Fallback path: loop that re-fetches candidates from `current` and removes **one face at a time**:

   ```python
   current = shape
   removed: list[FaceDescriptor] = []
   attempts_left = max(2 * len(candidates), 8)
   while attempts_left > 0:
       fresh = _candidate_faces(ocp, current, threshold)   # always from *current*
       if not fresh:
           break
       progressed = False
       for idx in range(len(fresh)):
           if attempts_left <= 0:
               break
           fresh = _candidate_faces(ocp, current, threshold)
           if idx >= len(fresh):
               break
           face, desc = fresh[idx]
           attempts_left -= 1
           nxt = _defeature_shapes(ocp, current, [face])
           if nxt is not None:
               current = nxt
               removed.append(desc)
               progressed = True
               break        # re-fetch everything after any accepted removal
       if not progressed:
           break
   ```
   After the loop, re-fetch `remaining = _candidate_faces(ocp, current, threshold)` and, if any,
   append `_make_defeature_defect([desc for _, desc in remaining], threshold)`. Keep the existing
   `max_passes`-free structure; the stale-handle bug must be gone, and the "no progress → report the
   rest as `defeature_failed`" behaviour must be kept.

**New test** `test_defeature_fallback_removes_all_candidates_when_batch_fails`: monkeypatch
`magwave.cad.defeature._defeature_shapes` with a wrapper that delegates to the real function for a
single face but returns `None` when `len(faces) > 1` (forcing the batch to fail), then
`defeature(build_filleted_block(), max_feature_size_mm=2.0)` must remove **≥ 30** of 40
(`features_removed >= 30`), the result must be `BRepCheck_Analyzer(...).IsValid()`, and the volume
delta must equal `volume_after - volume_before`. Today's code removes 3 and fails.

### F3 — module/function name shadowing

`src/magwave/cad/__init__.py:11-18` re-exports the `defeature` **function**, so the package attribute
`magwave.cad.defeature` is the function and `import magwave.cad.defeature as df` binds the function
(`df._load_ocp` → `AttributeError`).

**Chosen resolution (implement this one):** stop re-exporting the name `defeature` from the package;
export the function as `defeature_shape` and document the module path.

- In `cad/__init__.py`: `from .defeature import ... defeature as defeature_shape`, add
  `"defeature_shape"` to `__all__`, remove `"defeature"` from `__all__`.
- Package docstring: document the two supported import paths —
  `from magwave.cad.defeature import defeature` (canonical) and
  `from magwave.cad import defeature_shape` (alias) — and say the submodule attribute is the module.
- Do **not** add a `defeature = ...` attribute back to the package.
- Update `tests/test_defeature.py`: drop `defeature` from the `from magwave.cad import (...)` block
  and import it from `magwave.cad.defeature`. Internal callers `src/magwave/cad/step.py:379,512`
  already do `from .defeature import defeature` — leave them.
- Any other reference to `magwave.cad.defeature` in tests must keep working: after this change
  `import magwave.cad.defeature as df` yields the **module**.

**New test** `test_defeature_module_import_is_unambiguous`:

```python
import magwave.cad as cad
import magwave.cad.defeature as df

assert not hasattr(cad, "defeature")          # the name no longer shadows the submodule
assert df is sys.modules["magwave.cad.defeature"]
assert callable(df.defeature)
assert cad.defeature_shape is df.defeature
```

Keep `test_lazy_ocp_guard_for_defeature_module` passing unchanged — importing the module must still
not load OCP.

---

## Fence — do not cross

- Do not weaken or delete existing assertions. `test_defeature_forty_fillets` stays exactly as-is
  and keeps passing (40 removed, all `Torus`, `volume_after ≈ 48000` `rel=0.01`).
  The only permitted edit to an existing test is the `0.5 → 0.1` threshold in
  `test_defeature_threshold_too_small_removes_nothing`, with a comment saying why.
- Do not touch `benchmarks/*.json`, `benchmarks/checks.py`, `analytic.py`, `pyproject.toml`,
  `docs/adr/`. The 27 frozen benchmark values and analytic cases must be unchanged.
- Core stays Qt-free and OCP-free: no module-scope OCP import in `src/magwave/cli.py` or
  `src/magwave/cad/__init__.py`; OCP only inside `_load_ocp()` and the `step.py` lazy paths.
- No new runtime dependency. No merges, no pushes, no commits — leave the tree dirty for review.
- Do not reformat unrelated code, do not rename public dataclasses, do not change the
  `DefeatureResult` field set.

## Definition of done — run these and paste real output in your final message

```bash
cd /home/jblane1000/projects/magwave
env -u PYTHONPATH .venv/bin/python -m pytest --junit-xml=/tmp/agy-fr012-fix.xml -q
env -u PYTHONPATH .venv/bin/python -m magwave bench
.venv/bin/ruff check --no-cache .
env -u PYTHONPATH .venv/bin/python tools/check_licence_boundary.py
```

Expected: 185 + the new tests, all passing; 27/27 PASS; `All checks passed!`; `licence boundary OK`.
Also report, from a real run, the F5 numbers: on `build_corner_filleted_block()`,
`defeature(..., max_feature_size_mm=2.0)` → features removed, IsValid, volume before/after/delta.

If a step is blocked by a missing API, stop and say exactly which call is missing rather than
inventing one. Do not report success you did not observe.
