# agy brief — M5 Stage 1: axisymmetric Tier 1 magnetostatics solver (linear media)

Implementation work order. You write the solver, the tests and ADR-0007. You do **not** run `git`
(no branch, no commit, no push) — the branch `m5.1-tier1-fem2d` is already checked out and Cody
commits.

Repo root: `/home/jblane1000/projects/magwave`, branch `m5.1-tier1-fem2d`, base `f4e7b36`.

Read first, in this order, before writing anything:
`src/magwave/engines/base.py`, `src/magwave/engines/magpylib_engine.py`, `src/magwave/cases.py`,
`src/magwave/analytic.py`, `src/magwave/benchmarks/harness.py`, `src/magwave/cli.py`,
`src/magwave/cad/roles.py`, `src/magwave/cad/manifest.py`, `tests/test_engine_protocol.py`.

## Environment trap — read this first

A Python 3.14 `PYTHONPATH` from the surrounding agent environment shadows the project venv and breaks
numpy. **Every python command must be**:

```bash
cd /home/jblane1000/projects/magwave
env -u PYTHONPATH .venv/bin/python ...
```

Never bare `python`/`python3`. The venv is Python 3.12.3 with numpy 2.5.5, scipy 1.18.1, pytest 9.2.0
installed, and `.venv/bin/ruff` 0.17.5. `env -u PYTHONPATH .venv/bin/python -m pytest` is the runner
(that exact form). `docs/` is excluded from ruff (`extend-exclude = ["docs"]`).

## Requirement

M5 is Tier 1 FEM. Stage 1 = the solver skeleton, proved on problems with exact answers, in **linear
media only**: no steel, no B–H curve, no saturation, no demagnetisation, no forces. The gate is
measured agreement, not code volume: solenoid and magnetised-cylinder on-axis fields within **1 %** of
closed form, plus a two-level mesh-independence study with the errors printed.

## Fence (hard)

- `src/magwave/fem2d/` must not import `gmsh`, `OCP`, `Elmer`, `trimesh` — no external mesher. The
  licence gate `tools/check_licence_boundary.py` and `tests/test_package.py` both enforce variants of
  this; both must stay green.
- Do **not** edit: `src/magwave/benchmarks/cases.json` (27 frozen expected values),
  `src/magwave/benchmarks/checks.py`, `src/magwave/analytic.py`, `src/magwave/cases.py`,
  `src/magwave/units.py`, or any existing test file. Adding to files is fine where this brief says
  so; changing an existing behaviour is not.
- No new GPL dependency. `scipy` is BSD-3 and is the only new dependency.
- `ruff check --no-cache .` and `ruff format --check .` must both be clean over the whole repo.

## 1. Formulation — fixed, do not substitute

Axisymmetric magnetostatics with the azimuthal magnetic vector potential `A = A_phi(r,z) * e_phi`,
`nu = 1/mu_0`, `J = J_phi(r,z) * e_phi`:

```
curl( nu curl A ) = J_phi   in (r, z),  measure r dr dz
B_r = -dA/dz                B_z = (1/r) d(r A)/dr = A/r + dA/dr
```

Weak form with test function `v` (v = 0 on the Dirichlet boundary):

```
integral_Omega  nu * [ r*(dA/dr)(dv/dr) + r*(dA/dz)(dv/dz) + (dA/dr)*v + A*(dv/dr) + A*v/r ] dr dz
   =  integral_Omega  J_phi * v * r dr dz  +  integral_sheet  K_phi * v * r dl
```

This is the same bilinear form as `integral nu (curl A).(curl v) r dr dz`; derive the element matrix
from the expression above, not from the reduced form, because the `A*v/r` term is what makes the axis
well behaved.

**Element matrix, linear triangle, one-point (centroid) quadrature.** State this choice in the module
docstring, the ADR and the report — the brief allows centroid weighting or exact per-element
integration, and centroid is the one specified here. With vertices `i = 0,1,2`, area `A_e`,
centroid radius `r_c = (r_0+r_1+r_2)/3`, gradients `b_i = dN_i/dr`, `c_i = dN_i/dz`,
`N_i(centroid) = 1/3`:

```
K^e_ij = nu_e * A_e * [ r_c*(b_i*b_j + c_i*c_j) + (r_c/3)*(b_i + b_j) + 1/9 ]
```

The matrix is symmetric by construction. `nu_e = 1/MU0` everywhere in this stage (`MU0` from
`magwave.units`).

**Load vector, volume current, exact (no quadrature needed):**

```
f^e_i = J_e * A_e * (2*r_i + r_j + r_k) / 12      # integral N_i * r dA over the triangle
```

**Load vector, surface current sheet (the PM source), exact.** A sheet at `r = R` carrying
`K_phi` A/m over `z in [z1, z2]`, discretised along mesh edges on the line `r = R`. For one segment of
length `l` between nodes `a` and `b`:

```
f_a += K_phi * R * l / 2        f_b += K_phi * R * l / 2
```

**Boundary conditions.** Dirichlet `A = 0` on (a) every node with `r == 0` — the regularity condition
on the axis, which is exactly why this formulation needs no gauge and no `1/r` on the axis — and
(b) every node on the outer domain boundary. Eliminate constrained DOFs symmetrically (zero row and
column, `K_ii = 1`, `f_i = 0`). Natural Neumann applies on nothing else in this stage.

**Field recovery.** Inside an element: `A = sum N_i A_i`, `dA/dr`, `dA/dz` element-constant. So
`B_r = -sum c_i A_i` and `B_z = A/r + sum b_i A_i`. At a sample point with `r < 1e-12` use the limit
`B_z = 2 * sum b_i A_i` (in the axis-adjacent element `A = alpha*r` exactly, because both axis nodes
are pinned to 0, so `A/r -> alpha = sum b_i A_i`), and `B_r = 0` at `r == 0` by symmetry. This must
return a **finite** number at `r == 0` — a divide-by-zero or NaN is a test failure, not a warning.

## 2. Sources in this stage

- **Coil** (`Coil(kind="solenoid")`): rectangular winding cross-section in `(r,z)`, uniform
  `J_phi = turns*current / A_window` with `A_window = (outer_radius - inner_radius) * length`.
  `inner_radius == outer_radius` is a **current sheet** at `r = inner_radius` with
  `K_phi = turns*current/length` (use the sheet load vector, not a zero-area volume). The case schema
  has no fill-factor field and no schema change is permitted; the analytic path in `checks.py` uses
  fill factor 1.0 (`J = N*I/((R2-R1)*L)`), so use 1.0 and say so in the report/docstring.
- **Permanent magnet** (`Magnet(kind="cylinder")`, `br = (0, 0, Br)`): uniform axial magnetisation
  `M = Br/MU0`. Uniform M has no volume current; the equivalent source is the bound surface current
  `K = M x n`, which for an axially magnetised cylinder is `K_phi = M` on the cylindrical face
  (`r = radius`, `z` from `z1` to `z2`) and zero on the end faces. Implement it as exactly the sheet
  load above. Equivalently it is a thin solenoid of the same radius and length with `nI = M`; that
  identity is checked by `analytic.cylinder_Bz_axis` vs `analytic.thin_solenoid_Bz` and is why the
  PM test and the thin-solenoid test share a code path.
- Anything else (loop coil, cuboid/Halbach magnet, dipole, force) is rejected in `validate()`, never
  approximated.

## 3. Mesher — `src/magwave/fem2d/mesh.py`, no external mesher

Axis-aligned **tensor-product** grid over `(r, z)`, each rectangle split into two linear triangles
(`(i,j),(i+1,j),(i+1,j+1)` and `(i,j),(i+1,j+1),(i,j+1)`, or your equivalent — just be consistent
with `locate`). Build the coordinate lists like this:

1. The r- and z-axis coordinate lists are assembled from **breakpoints** (forced node lines: every
   region boundary and every sheet radius — so no element is split by a region edge and every source
   boundary is exactly a mesh line) plus subdivision of each interval.
2. Intervals inside the **refinement box** (which must contain every source) are subdivided into
   `ceil(length/h)` equal steps. The refinement box boundary itself is a breakpoint.
3. Intervals outside the refinement box grow geometrically away from it with ratio `growth` (default
   1.2), starting at `h`, until the domain boundary — this keeps a 5x air domain affordable. Do not
   let the growth create an aspect ratio worse than `growth`.

API (exact names; tests are written against these):

```python
@dataclass(frozen=True)
class MeshStats:
    nodes: int
    elements: int
    min_edge: float          # metres
    max_edge: float          # metres
    h_requested: float       # metres

@dataclass(frozen=True)
class Mesh:
    nodes: np.ndarray        # (n, 2) float64, columns (r, z), metres
    elements: np.ndarray     # (m, 3) int64, node indices, counter-clockwise
    r_axis: np.ndarray       # (nr,) unique r coordinates of the tensor grid, ascending
    z_axis: np.ndarray       # (nz,) unique z coordinates, ascending
    h_requested: float

    def stats(self) -> MeshStats: ...
    def locate(self, points: np.ndarray) -> np.ndarray:
        """Element index for each (r, z) point. ValueError outside the domain or r < 0.
        Deterministic, vectorised, points exactly on a shared edge or node must resolve
        (pick a documented convention), r == 0 must resolve."""

def rectangular_mesh(
    r_max: float,
    z_min: float,
    z_max: float,
    h: float,
    *,
    refinement_box: tuple[float, float, float, float] | None = None,  # (r0, r1, z0, z1)
    breakpoints_r: Sequence[float] = (),
    breakpoints_z: Sequence[float] = (),
    growth: float = 1.2,
    max_nodes: int = 400_000,
) -> Mesh:
    """Raise ValueError if the grid would exceed max_nodes (never silently coarsen)."""
```

`breakpoints_r`/`breakpoints_z` values outside `[r_max]`/`[z_min, z_max]` or duplicated within 1e-12
are a `ValueError`. `h <= 0`, `r_max <= 0`, `z_max <= z_min` are `ValueError`s.

## 4. Assembly and solve

`src/magwave/fem2d/assembly.py`:

```python
def stiffness(mesh, nu=1.0/MU0) -> scipy.sparse.csr_matrix      # (n_nodes, n_nodes)
def load_current_density(mesh, region: tuple[float, float, float, float], j_phi: float) -> np.ndarray
    """region = (r0, r1, z0, z1); elements whose centroid lies inside get the volume load."""
def load_sheet(mesh, radius: float, z_range: tuple[float, float], k_phi: float) -> np.ndarray
    """Raise ValueError if no mesh edge lies on the sheet line (the caller must put a
    breakpoint at the sheet radius). Both directions of the segment must be handled."""
```

`src/magwave/fem2d/solve.py`:

```python
@dataclass(frozen=True)
class SolveReport:
    unknowns: int
    constrained: int
    iterations: int            # 1 for this linear, direct solve
    relative_residual: float   # ||f - K A|| / ||f||, computed, not assumed
    wall_time_s: float
    mesh: MeshStats
    solver: str                # e.g. "scipy.sparse.linalg.spsolve (SuperLU)"

def solve_axisymmetric(mesh, load, nu=1.0/MU0, constrained=None) -> tuple[np.ndarray, SolveReport]
    """Dirichlet set defaults to {nodes with r == 0} | {nodes on the domain boundary}.
    Returns A_phi per node (m*T, i.e. Wb/m) and the report of FR-042."""
```

Also expose `dirichlet_nodes(mesh) -> np.ndarray[int]` (the default constrained set) so the engine and
tests agree on it. Use `spsolve`; it is deterministic (SuperLU, single-threaded, same input -> same
bits). Do not add an iterative fallback in this stage.

`src/magwave/fem2d/field.py`:

```python
def sample_field(mesh, a_phi, points_rz: np.ndarray) -> np.ndarray
    """(n,2) -> (n,2) of (B_r, B_z) in tesla. Finite at r == 0, per the limit above."""
def element_field(mesh, a_phi) -> np.ndarray     # (m, 2) element-centred (B_r, B_z)
```

`src/magwave/fem2d/__init__.py` must re-export the public names and import nothing heavy beyond
scipy/numpy.

## 5. Engine — `src/magwave/fem2d/engine.py`

`class Fem2dEngine` implementing the `Engine` protocol (`capabilities`, `validate`, `solve`) —
look at `MagpylibEngine` for the shape, including `raise_on_errors(ENGINE_ID, self.validate(case))`
at the top of `solve`.

```python
ENGINE_ID = "fem2d"
RUNG = "Tier 1"

Fem2dEngine(element_size: float | None = None,      # metres; None -> largest source extent / 40
            far_field_factor: float = 5.0,          # brief's minimum; boundary >= this x L_max away
            growth: float = 1.2,
            certified_accuracy: float = 1e-2,       # the 1 % this stage actually proves
            domain: tuple[float, float, float] | None = None,          # (r_max, z_min, z_max)
            refinement_box: tuple[float, float, float, float] | None = None)
```

`capabilities()` -> `Capabilities(engine_id="fem2d", rung="Tier 1", source_types=frozenset({"magnet",
"coil"}), magnet_kinds=frozenset({"cylinder"}), coil_kinds=frozenset({"solenoid"}),
quantities=frozenset({"B"}), force_methods=(), engine_version=<"<magwave __version__>+m5.1">)`.

`validate(case)` — every rejection is a `Problem` with a stable code and a sentence naming the case and
what is missing; `solve()` never returns a number for a case it cannot model:

| code | severity | when |
|---|---|---|
| `E_UNSUPPORTED_SOURCE_TYPE` | error | `Dipole`, or a component type the engine cannot express |
| `E_UNSUPPORTED_GEOMETRY` | error | magnet kind != `cylinder`, coil kind != `solenoid` |
| `E_UNSUPPORTED_MAGNETISATION` | error | `br[0] != 0` or `br[1] != 0` (only axial M is axisymmetric) |
| `E_NOT_AXISYMMETRIC` | error | `center[0] != 0` or `center[1] != 0` (source off the axis) |
| `E_UNSUPPORTED_QUANTITY` | error | probe quantity `F` |
| `E_UNSUPPORTED_MATERIAL` | error | **the steel gate**: any `case.manifest` body with role `permeable_solid` — message must name the missing capability and the tier that answers it, e.g. "requires a nonlinear B–H material law; this solver is linear-mu only. Permeable materials arrive in Tier 1 stage 2 (M5.2). Nothing was solved." |
| `E_UNSUPPORTED_MANIFEST` | error | any other manifest role except `air` / `symmetry_plane`: CAD geometry is not meshed by this solver yet |
| `E_ACCURACY_UNREACHABLE` | error | the case is otherwise solvable but `case.tolerance < certified_accuracy`: state the case's research-owned tolerance, the 1 % this stage certifies, and that the number is not claimed |
| `W_DEMAG_IGNORED` | warning | `magnet.demag_curve is not None` — rigidly polarised, matching `MagpylibEngine`'s convention |

Order: check components first, then manifest/material, then accuracy last (so a steel case reports the
material, not the tolerance).

Mesh construction inside `solve()`: from the components, build the source bounding box in `(r, z)`
(all sources are on the axis; `r in [0, max outer radius]`, `z in [min z1, max z2]`). `L_max` = the
largest single source extent (max over sources of `max(radial extent, axial extent)`). Then
`r_max = max_outer_radius + far_field_factor*L_max`, `z_min = min_z1 - pad`, `z_max = max_z2 + pad`
with the same pad — so the boundary is at least `far_field_factor * L_max` from the source in every
direction (the brief's "≥ 5x the largest dimension"). Breakpoints: every source's `r` boundaries
(inner/outer radius, sheet radius) and `z` ends; refinement box = the source box expanded by `2*h`.
`domain`/`refinement_box` constructor overrides exist so a test can force an identical mesh for two
different source sets — that is how the superposition test is made exact.

Probe mapping: `(x, y, z) -> (r = hypot(x, y), z)`, then `B = (B_r*x/r, B_r*y/r, B_z)` (and
`(0, 0, B_z)` when `r == 0`). `Sample.point` must be the original 3-D probe point, `component` the
probe's component, `quantity="B"`, `unit="T"`. A probe outside the mesh domain is an `EngineError`
with an actionable message (increase `far_field_factor`), not a clamp and not a NaN.

`method` string on the `Result`: name the formulation, the element type, the quadrature and the solve
method, e.g. `"axisymmetric A_phi FEM, linear triangles, centroid quadrature, Dirichlet A=0 on axis
and at 5x far field, scipy.sparse.linalg.spsolve"`.

`src/magwave/engines/__init__.py`: add `Fem2dEngine` to `__all__` and to the lazy `__getattr__`, same
pattern as `MagpylibEngine` (do not import scipy at module scope there).

## 6. Register in the runner and the CLI (additive only)

- `cli.py`: add `"fem2d"` to `ENGINES` and a branch in `_make_engine`. Nothing else about `bench`
  changes by default.
- `harness.py`: add, **without touching any existing function or dataclass**:
  ```python
  @dataclass(frozen=True)
  class AccuracyRow:      # name, check, computed, expected, rel_err, tolerance, certified_accuracy, within
  def measure_engine_accuracy(cases, engine) -> list[AccuracyRow]
      """For each case whose validate() fails ONLY with E_ACCURACY_UNREACHABLE, solve it anyway and
      record the measured relative error. Cases the engine cannot model at all are omitted."""
  def format_accuracy_table(rows) -> str
  ```
  export both from `magwave.benchmarks.__init__`.
- `cli.py`: `--report-accuracy` flag, valid only together with `bench --engine`. With it, print the
  ordinary engine table, then the accuracy table, then a line
  `measured: N case(s), worst rel.err X.XXe-XX, certified 1.00e-02` and exit 0 iff every measured row
  is within the engine's certified accuracy. Without the flag, behaviour is byte-identical to today.

The reason this exists: every axisymmetric case in the 27-case frozen corpus carries a research
tolerance between `1e-12` and `1e-4` — 10^2 to 10^10 times tighter than the 1 % this stage certifies.
So the honest answer set is empty and plain `bench --engine fem2d` is 27 SKIPs (exit 1 by the M2 rule
that an all-skipped run is not a pass). `--report-accuracy` is how the real, measured numbers appear
in the bench output without widening a single research-owned tolerance. Say all of that in ADR-0007
and the report.

## 7. Dependency, NOTICE

- `pyproject.toml`: add `"scipy>=1.11"` to `[project].dependencies` (after numpy). Nothing else.
- `NOTICE`: change the scipy line to `- scipy — BSD-3-Clause (core dependency from M5: sparse linear
  algebra for the Tier 1 FEM solver)`.
- `tools/check_licence_boundary.py` must still pass unchanged.

## 8. ADR — `docs/adr/0007-tier1-formulation.md`

Follow the shape of `docs/adr/0006-cad-import-dependency.md` (Status/Date/Deciders/Amends/Supersedes
header, Context, Decision, Consequences, measured numbers in tables, `file:line` or command references).
It must decide and justify, in this order:

1. **Why `A = A_phi e_phi`**: one scalar unknown, no gauge, currents enter as a right-hand side,
   `A_phi -> 0` on the axis is the regularity condition (and is why samples on the axis are finite).
2. **Why Dirichlet at 5x**: the truncation error budget — the far field of these sources decays as
   `1/r^3` in `B`, and the measured effect at 5x is below the 1 % gate (put the number you measured
   in the table). Note that a Kelvin/open-boundary treatment is deferred to Tier 2.
3. **Why `scipy`**: BSD-3, wheels for cp311–cp313 on all three platforms, `spsolve`/SuperLU direct,
   deterministic, no GPL linkage; measured version and the licence-gate result.
4. **Why a built-in mesher now**: gmsh is GPL and subprocess-only (ADR-0002), the stage-1 geometry is
   axis-aligned rectangles, and a tensor mesh with breakpoints gives exact region boundaries and exact
   sheet lines. It is not a mesher strategy for M6 — state that explicitly.
5. **Rejected alternative: reduced/total scalar potential (`Omega`-formulation, `H = H_s - grad
   Omega`)**. Rejected *for this stage*, deferred rather than dismissed: it needs a source field
   `H_s` and a cut for multiply-connected current paths, and it does not represent currents as a
   right-hand side; but it is the better form once `mu = mu(|B|)` arrives (stage 2), because it avoids
   the `1/mu` gradient term in the A-formulation. Say what would trigger adopting it.
6. Also record: centroid quadrature choice and its convergence consequence; the element matrix as
   implemented; the accuracy certification (`1 %`) and that it is *not* a research-owned tolerance.

## 9. Tests — this is the stage's real deliverable

Two new files, no edits to existing tests. Total added runtime under ~60 s.

`tests/test_fem2d.py` — the machinery, no closed-form comparison:

1. `test_mesh_stats_and_breakpoints` — a small mesh has nodes exactly at every requested breakpoint;
   `stats()` min/max edge are consistent with the node coordinates; `max_nodes` raises.
2. `test_stiffness_is_symmetric_and_singular_before_constraints` — `K` symmetric to 1e-12 relative
   in the assembled CSR; row sums of an unconstrained matrix are `~0` (a constant potential is in the
   null space only for elements off the axis — if that is not exactly true for this form, assert the
   weaker property you can actually justify and say in a comment why).
3. `test_axis_sample_is_finite` — a solved solenoid, sample at exactly `r = 0`: finite, `B_r == 0`,
   and `|B_z|` within 1 % of the neighbouring `r = h/100` sample.
4. `test_determinism` — same case, two independent engine instances: `np.array_equal` on the B
   values (bit identical, FR-043).
5. `test_superposition_on_a_fixed_mesh` — two coaxial solenoids (same radii, different z ranges) with
   `domain`/`refinement_box`/`element_size` forced; solve A, B and A+B on the identical mesh; the
   combined field equals the sum of the separate fields to `<= 1e-10` relative (state in the docstring
   that this is exact by linearity on one mesh, and that a cross-mesh comparison is deliberately not
   what is being tested).
6. `test_steel_case_is_rejected` — build a `Case` whose `manifest` has a body with role
   `permeable_solid` (use `magwave.cad.manifest` + `magwave.cad.roles.assign` or the constructors
   directly); `validate()` returns exactly one error with code `E_UNSUPPORTED_MATERIAL`; the message
   names "stage 2"; `solve()` raises `EngineError` and no field is produced.
7. `test_capability_gate_rejects_and_blocks` — a `Dipole` case, a `cuboid` magnet case, a `loop` coil
   case, an off-axis source, and a probe asking for `F`: each produces its own error code, and
   `solve()` raises for each.
8. `test_accuracy_gate_is_honest` — an otherwise-solvable axisymmetric case whose tolerance is 1e-4
   (i.e. `thick_solenoid_axis_Bz` from the corpus) validates with exactly `E_ACCURACY_UNREACHABLE`,
   and the message contains both numbers.
9. `test_engine_protocol_and_registration` — `isinstance(Fem2dEngine(), Engine)`; `capabilities()`
   fields as specified; `magwave.cli.main(["bench", "--engine", "fem2d", "--report-accuracy"]) == 0`
   and the accuracy table contains a measured row for a solenoid case.

`tests/test_fem2d_validation.py` — the gate. Use the closed forms from `magwave.analytic`
(`thick_solenoid_Bz`, `thin_solenoid_Bz`, `cylinder_Bz_axis`); do not re-derive anything.

- **Solenoid (FR-030's acceptance, re-proved through the FEM path).** `Coil(kind="solenoid",
  dims={"inner_radius": 0.010, "outer_radius": 0.020, "length": 0.100}, current=1.0, turns=1000)`.
  11 on-axis stations `z = linspace(-0.06, 0.06, 11)` (three of them outside the winding — that is
  deliberate). Closed form `thick_solenoid_Bz(0.010, 0.020, -0.05, 0.05, J, z)` with
  `J = turns*current/((0.020-0.010)*0.100)`. **Assert max relative error <= 1e-2 at both mesh levels**
  (level 1 `element_size=1e-3`, level 2 `1e-3/2`), and assert the level-2 error is smaller than level-1.
- **Uniformly magnetised cylinder.** `Magnet(kind="cylinder", dims={"radius": 0.0125, "length": 0.05},
  br=(0, 0, 1.2))`, 11 on-axis stations `z = linspace(-0.04, 0.04, 11)` against
  `cylinder_Bz_axis(0.0125, -0.025, 0.025, 1.2/MU0, z)`. Same two levels, same 1 % assertion, same
  monotonicity assertion.
- **Mesh-independence study, one test, numbers printed.** Both cases at `h` and `h/2`: print one
  aligned table (one row per case per level) with `case | h | nodes | elements | min edge | max rel
  err | solve wall time | PASS/FAIL`, via `print()` (the acceptance run writes
  `--junit-xml=/tmp/cody-m51.xml`, so captured stdout lands in the XML as `<system-out>` — that is
  where the report's numbers come from). Assert: error decreases with refinement for both cases, and
  the fine-level error is `<= 1e-2`.
- If a level-2 error exceeds `1e-2`, do **not** loosen an assertion: refine (another `h/2`), fix the
  formulation, or report the failure in your summary. The gate is the deliverable.

## 10. Run these yourself, and iterate until all of them are green

```bash
cd /home/jblane1000/projects/magwave
.venv/bin/ruff format src/magwave/fem2d tests/test_fem2d.py tests/test_fem2d_validation.py \
    src/magwave/cli.py src/magwave/benchmarks/harness.py src/magwave/benchmarks/__init__.py \
    src/magwave/engines/__init__.py
.venv/bin/ruff check --no-cache .
.venv/bin/ruff format --check .
env -u PYTHONPATH .venv/bin/python -m pytest -q
env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_fem2d_validation.py -s   # see the table
env -u PYTHONPATH .venv/bin/python -m magwave bench
env -u PYTHONPATH .venv/bin/python -m magwave bench --engine fem2d --report-accuracy
env -u PYTHONPATH .venv/bin/python tools/check_licence_boundary.py
```

Baseline that must not regress: the suite currently collects and passes **191 tests** (0 failures); `magwave bench` is
`27/27 PASS`; both ruff commands are clean; the licence gate prints
`licence boundary OK (no GPL linkage)`.

## 11. Output you must return

A short report, not a narrative: the files you created/changed, the exact commands above with their
real exit codes and output tails, the printed validation table (per-point errors at both levels), the
mesh statistics and wall times, and anything you could not do. If a gate is not met, say which number
and by how much — do not paper over it.
