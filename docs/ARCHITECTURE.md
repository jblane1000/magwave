# Architecture

Status: Phase 0. Nothing here is code yet; this is the shape the code must take.

## 1. Principles

1. **The case file is the product.** Everything else is a way to produce or read one.
2. **Engines are replaceable.** The UI and the data model never import a solver.
3. **Licence boundaries are physical.** A GPL tool is a subprocess, never an import.
4. **Units are typed at the boundary.** SI inside, declared units at CAD and display.
5. **Fail loudly.** A skipped body or unreported non-convergence is a defect, not a convenience.
6. **Validation runs in CI from the first commit.**

## 2. Layers

```
            +---------------------------+
            |  ui/  (PySide6 + PyVista) |   optional, never imported by core
            +-------------+-------------+
                          |
+-------------------------v--------------------------+
|  cli/  (argparse)      api/  (public Python API)   |
+-------------------------+--------------------------+
                          |
+-------------------------v--------------------------+
|  core/                                             |
|    case      data model + (de)serialisation        |
|    units     dimensioned quantities                |
|    materials linear / B-H / PM demag + provenance   |
|    geometry  bodies, roles, transforms              |
|    probes    points, lines, planes, volumes         |
|    results   fields, forces, provenance, export     |
+-------------------------+--------------------------+
                          |
+-------------------------v--------------------------+
|  engines/                                          |
|    base.py    Engine protocol + capabilities       |
|    analytic/  Magpylib-backed (Tier 0, core dep)    |
|    fem2d/     axisymmetric/planar (Tier 1)          |
|    fem3d/     Elmer/NGSolve adapter (Tier 2)        |
+-------------------------+--------------------------+
                          |
+-------------------------v--------------------------+
|  io/                                               |
|    cad/   OCP (STEP/IGES), trimesh (mesh formats)   |
|    mesh/  gmsh subprocess (GPL boundary)            |
|    report/ HTML/PDF export                          |
+----------------------------------------------------+
```

## 3. The engine protocol

```python
class Engine(Protocol):
    name: str
    version: str

    def capabilities(self) -> Capabilities: ...  # materials, sources, dims, forces
    def validate(self, case: Case) -> list[Problem]: ...  # blocking problems, empty = ok
    def solve(self, case: Case, progress=None) -> Result: ...
```

- `capabilities()` is what makes FR-041 possible: a steel body submitted to the analytic engine is
  rejected with "requires permeable materials (Tier 1)" before any number is produced.
- `validate()` returns *problems*, each with a code, a severity and a human message. `Problem` codes are
  stable and documented, because the CLI, the GUI and the tests all read them.
- `Result` carries fields, forces, provenance (engine name/version, case hash, timestamps) and warnings.

Adding an engine must never require touching `core/` or `ui/`. That is the test of the boundary.

## 4. Data model (draft)

- `Case`: units, frame, bodies[], materials refs, sources[], study, probes[], engine selection, notes.
- `Body`: id, role (permeable | permanent-magnet | coil | air | symmetry), material ref, geometry ref
  (CAD path + shape index + hash, or analytic shape parameters), transform.
- `Material`: id, kind, parameters (μr | B-H table | Br + demag curve + temp coefficients), provenance.
- `Source`: kind (coil | current density | magnetization), geometry ref, values, units.
- `Study`: engine name, options (mesh size, air-box extents, tolerance, symmetry).
- `Result`: fields on requested grids, scalars (force, torque, inductance), provenance, warnings.
- `Problem`: code, severity (error | warning | info), message, offending object id.

## 5. Case file format

JSON, versioned by `schema_version`. Human-editable, diff-friendly, no binary blobs; CAD referenced by
relative path + SHA-256. See `core/case.py` in M1 and `docs/PRD.md` FR-001..FR-004.

Rationale: a `.sif`-style expert input file is exactly the barrier this project exists to remove, and a
binary project file cannot be reviewed or merged.

## 6. Units and frames

- Internally SI (metres, tesla, ampere, newton). Conversion only at import (CAD) and display.
- Every case declares `units: {length: mm|in, ...}` and `frame: {axis: Z, datum: <face id>, note}` —
  mirroring the CAD house rule that the functional underside is the datum, not the lowest point.
- A quantity without a unit is a defect (NFR-2).

## 7. Licensing boundary, concretely

- `import` of gmsh or Elmer is **forbidden** anywhere in the package. `io/mesh/gmsh_runner.py` writes a
  `.geo`, invokes the binary with `subprocess`, and reads the mesh back. Same for Elmer: write `.sif`,
  run `ElmerSolver`, parse output.
- CI enforces it: a test imports every module of the package with a hook that fails if `gmsh` or `Elmer`
  appear in `sys.modules`.
- PySide6 (LGPL) is linked dynamically; the packaged app ships Qt licence notices.

## 8. Performance posture

Tier 0 is vectorised numpy over Magpylib's closed forms; the target is 10⁴ points in ~100 ms, which
requires batching rather than per-point Python calls. Tier 1 assembles a sparse system and uses
SciPy/UMFPACK-class solvers; the axisymmetric case is 2D so the system stays small. Tier 2 delegates the
heavy lifting to Elmer/NGSolve as processes and streams their logs into the UI.

## 9. Failure and reporting model

Three severities, and only three: **error** blocks the run, **warning** annotates the result (and appears
in the exported report), **info** is logging. No body, material or boundary is ever dropped silently
(FR-011, NFR-8). Non-convergence is an error, not a footnote (FR-042).

## 10. Testing strategy

- Unit: units, case round-trip, material interpolation, geometry role assignment.
- Contract: a dummy engine exercises the protocol (FR-040).
- Benchmark: analytic rungs, tolerance-tagged, in CI (FR-070/071).
- Regression: stored case + expected numbers per engine version; a change of numbers requires a
  documented reason in the changelog.
- Licence: import-hook check (NFR-3). Offline: socket guard in tests (NFR-6).
