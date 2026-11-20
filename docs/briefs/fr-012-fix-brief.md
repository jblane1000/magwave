# Fix round — FR-012 (PR #13): feature-size metric + four findings

**Owner:** Cody. **Engine:** implementation code from `agy` (`agy-work --model gemini-pro-orchestrator-g1-pro --print-timeout 0 -p '<pointer>'`). You write this brief, verify, commit, push to the same branch. Max re-runs acceptance.

Branch: `fr-012-booleans-defeature` (PR #13, open, not merged). Repo `/home/jblane1000/projects/magwave`.
Environment: **always** `env -u PYTHONPATH .venv/bin/python …` — a Python 3.14 `PYTHONPATH` from the
Hermes install env shadows the 3.12 venv.

## Verified baseline on your branch (Max ran these, they are the floor, not the target)

```
185 tests, 0 failures, 0 errors, 0 skipped
bench 27/27 PASS
ruff check --no-cache . → clean
tools/check_licence_boundary.py → OK
```

## F5 (Max's finding) — the size metric is wrong for the common case

`_candidate_faces` sizes a face by its **bounding-box diagonal**. The AC fixture passes only because
its fillets sit on Ø0.6 hole rims (diagonal 1.40 mm). Measured by Max on an ordinary part:

```
4 corner fillets R0.5 on an 80×50×12 block — reported face size = 12.0 mm
  threshold  2.0 mm → 0 removed
  threshold  5.0 mm → 0 removed
  threshold 20.0 mm → 4 removed
```

A 0.5 mm feature needing a 20 mm threshold means any threshold big enough to catch real edge fillets
also removes legitimate geometry — the opposite of FR-012's purpose. The AC as written cannot see this.

**Fix:** characteristic size must be the *local* feature size, not the extent of the face:
- Cylindrical face (`BRepAdaptor_Surface.GetType() == GeomAbs_Cylinder`): size = **radius**
  (`Cylinder().Radius()`); a fillet along a long edge is small however long it runs.
- Toroidal face (`GeomAbs_Torus`): size = **minor radius** (`Torus().MinorRadius()`).
- Spherical (`GeomAbs_Sphere`): radius.
- Anything else non-planar: fall back to the bbox diagonal, as today.
- Planes stay excluded.
Update `FaceDescriptor.size_mm` semantics accordingly and document it (module docstring + the PRD
FR-012 line) so the threshold means "fillet radius / small-feature size", not "face extent".

**New test that must fail on today's code and pass after the fix** —
`test_defeature_edge_fillet_uses_radius_not_bbox`: an 80×50×12 box with the four vertical edges
filleted at **R0.5** (build with `BRepFilletAPI_MakeFillet`, check `IsDone()`, add each edge once —
calling `Add` on arbitrary adjacent edges fails) must defeature to ≥ 4 Cylinder faces removed at
`max_feature_size_mm=2.0`, remain `BRepCheck_Analyzer(...).IsValid()`, and report the volume delta.
Confirm before the fix that it removes 0, so the test is a real guard, and say so in the report.

## F4 — no validity gate on the accepted result

`defeature()` accepts the batch result on `IsDone()` + non-null. Add a `BRepCheck_Analyzer(result.shape).IsValid()`
gate: an invalid result must not be returned as a success. Either raise or return with a `Defect`
(`kind="defeature_invalid"`) and `features_removed=0` — choose one, state which in the report, and test it.

## F2 — OCP failures escape as tracebacks

`defeature()` calls can raise OCP `Standard_Failure`, which derives from `Exception` alone, so
`cli.py`'s catch list (`MissingExtraError/FileNotFoundError/ValueError/OSError`) lets it escape as a raw
traceback instead of `magwave import: …` + exit 1. Broaden the CLI handling so an OCP failure becomes a
clean single-line error. Keep the existing error text style. Test with a monkeypatched raiser — do not
require a genuinely failing build.

## F1 — per-face fallback uses stale face handles

`defeature.py:264-286` reuses `FaceDescriptor`/face handles from the pre-removal shape, so a failed batch
degrades to ~1 face removed per pass. Re-fetch faces from the **current** shape each pass (or drive
`BRepAlgoAPI_Defeaturing` once per candidate against the evolving shape), so the fallback removes all
candidates it can. Max measured 3/40 removed on the forced path. Add a test that asserts the fallback
removes ≥ 30 of 40 when the batch path is forced to fail.

## F3 — module/function name shadowing

`import magwave.cad.defeature as df` binds the *function*, not the module, because `cad/__init__.py`
re-exports `defeature`. `df.defeature(...)` then raises `AttributeError`. Make the namespace
unambiguous: either re-export under a distinct alias (e.g. `defeature_shape`) or stop re-exporting and
document `from magwave.cad.defeature import defeature`. Whichever you choose, keep one documented import
path working and test the other does not silently mislead.

## Fence

- Do not weaken or delete existing assertions (the 40-fillet AC test stays as-is and keeps passing).
- Do not touch the 27 frozen benchmark values or the analytic cases.
- Core stays Qt-free and OCP-free (lazy load only) — `test_lazy_ocp_guard_for_defeature_module` must still pass.
- No new runtime dependency. No merges to `main`.

## Acceptance (Max re-runs, you report each line's real output)

```bash
cd /home/jblane1000/projects/magwave
env -u PYTHONPATH .venv/bin/python -m pytest --junit-xml=/tmp/cody-fr012-fix.xml -q
env -u PYTHONPATH .venv/bin/python -m magwave bench
.venv/bin/ruff check --no-cache .
env -u PYTHONPATH .venv/bin/python tools/check_licence_boundary.py
```

Also report the raw numbers for the F5 case: threshold 2.0 on the R0.5 corner-fillet block →
faces removed, valid, volume before/after/delta.

Append to `/home/jblane1000/projects/magwave/docs/briefs/fr-012-report.md` a "Fix round" section:
what changed per finding (F1–F5), the before/after numbers for the F5 case, the new test names, and how
many `agy` invocations produced the code. Reply with branch, new commit sha, and one line per acceptance
command.
