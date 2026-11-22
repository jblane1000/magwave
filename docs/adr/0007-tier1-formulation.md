# ADR 0007 — Tier 1 axisymmetric FEM formulation (`A_phi`), built-in mesher, and SciPy direct solve

- Status: accepted
- Date: 2026-11-19
- Deciders: Jason (owner)
- Amends: ADR-0002 (licence strategy) — promotes `scipy` from optional to core dependency; ADR-0005 (validation first) — adds `bench --engine ... --report-accuracy` to measure Tier 1 error without weakening research tolerances
- Supersedes nothing

## Context

M5 Stage 1 (`magwave.fem2d`) introduces the Tier 1 finite-element solver skeleton for linear
axisymmetric magnetostatics (`mu = mu_0` everywhere; no steel, no B–H nonlinearity, no
demagnetisation, no forces). The acceptance gate is quantitative agreement within **1 %** (`1e-2`
relative) against the closed forms in `src/magwave/analytic.py` on an axisymmetric volume-winding
solenoid and a uniformly magnetised cylinder across 11 on-axis stations, plus a two-level
mesh-independence study (`h` and `h/2`) showing decreasing error with refinement.

Every axisymmetric case in `src/magwave/benchmarks/cases.json` carries a research-owned tolerance
between `1e-12` and `1e-4` — 100 to 10^10 times tighter than the `1e-2` accuracy this stage
certifies. Under `Engine.validate()`, all 9 otherwise-solvable axisymmetric corpus cases are
therefore honestly rejected with `E_ACCURACY_UNREACHABLE`, causing plain
`python -m magwave bench --engine fem2d` to report 27 SKIPs (exit 1, since an all-skipped run is
not a pass). The `--report-accuracy` flag on `bench --engine` exposes the real measured relative
errors of those 9 cases against the `1e-2` certified gate without touching a single research-owned
tolerance.

All figures below were measured on 2026-11-19 in a clean checkout of this repository using
`env -u PYTHONPATH .venv/bin/python` (Python 3.12.3, NumPy 2.5.5, SciPy 1.18.1).

## Decision

### 1. Formulation: azimuthal magnetic vector potential `A = A_phi(r, z) e_phi`

Solve `curl((1/mu_0) curl A) = J_phi` in cylindrical `(r, z)` coordinates with measure `r dr dz`
for the single scalar unknown `A_phi(r, z)`:

```
B_r = -dA/dz        B_z = (1/r) d(r A)/dr = A/r + dA/dr
```

Weak form with test function `v` (`v = 0` on the Dirichlet boundary):

```
integral_Omega nu * [ r*(dA/dr)(dv/dr) + r*(dA/dz)(dv/dz) + (dA/dr)*v + A*(dv/dr) + A*v/r ] dr dz
    = integral_Omega J_phi * v * r dr dz  +  integral_sheet K_phi * v * r dl
```

Why `A = A_phi e_phi`:
- **One scalar unknown and no gauge ambiguity:** in axisymmetry `div(A_phi e_phi) = 0`
  identically.
- **Currents enter directly on the right-hand side:** volume windings (`J_phi = N*I / ((R2 - R1)*L)`,
  fill factor `1.0` matching `src/magwave/benchmarks/checks.py:110` because `magwave.case/1` has
  no fill-factor field) use exact linear-triangle nodal integration; thin-sheet solenoids
  (`inner_radius == outer_radius`) and axially magnetised cylinders (`K_phi = M = Br_z / MU0`)
  share the exact surface-current sheet path (`src/magwave/fem2d/assembly.py:load_sheet`).
- **Axis regularity condition:** pinning `A_phi = 0` on `r = 0` enforces the physical regularity
  `A_phi = alpha * r + O(r^3)` near the axis, eliminating any `1/r` singularity in field recovery
  via the exact limit `B_r = 0`, `B_z = 2 * dA/dr` at `r < 1e-12`
  (`src/magwave/fem2d/field.py:sample_field`).

### 2. Element quadrature, matrix, and convergence

Linear P1 triangles use **one-point (centroid) quadrature** (`r = r_c = (r_0 + r_1 + r_2)/3 > 0`,
`N_i(centroid) = 1/3`, gradients `b_i = dN_i/dr`, `c_i = dN_i/dz`). From the `r dr dz` weak form
above, the implemented symmetric element stiffness matrix (`src/magwave/fem2d/assembly.py:stiffness`)
is:

```
K^e_ij = nu_e * A_e * [ r_c*(b_i*b_j + c_i*c_j) + (b_i + b_j)/3 + 1/(9*r_c) ]
```

Note on dimensional consistency: the weak-form cross terms `(dA/dr)*v + A*(dv/dr)` carry no extra
factor of `r` in the integrand, so centroid evaluation gives `b_i*(1/3) + (1/3)*b_j = (b_i + b_j)/3`,
and `A*v/r` gives `(1/3)*(1/3)/r_c = 1/(9*r_c)` (without `r_c` in the numerator of the cross term
or missing `1/r_c` in the mass-like term, the operator would fail dimensional homogeneity). Because
`B` is recovered from element gradients of piecewise-linear `A_phi`, `B` converges at `O(h)` while
`A_phi` converges at `O(h^2)` away from corners; alternating the cell diagonal along `z` (`j % 2`)
in `src/magwave/fem2d/mesh.py` eliminates one-sided axial drift, and averaging incident cells on
shared interior grid lines removes one-sided half-cell bias in `sample_field`.

### 3. Far-field Dirichlet truncation at `>= 5x`

Outer boundaries apply homogeneous Dirichlet `A_phi = 0` at a pad of `far_field_factor * L_max`
(`default = 5.0`, where `L_max` is the largest source extent). Because the dipole far field of these
sources decays as `1/r^3` in `B` (`A_phi ~ 1/r^2`), truncating at `5x` introduces sub-percent
boundary bias compared to an open domain, whereas `2x` fails the `1 %` gate on the magnetised
cylinder:

| Case (`h = 5e-4` m) | `far_field_factor` | Nodes | Max rel. error (11 pts) | Centre `z=0` rel. error | Solve wall time |
|---|---:|---:|---:|---:|---:|
| Solenoid (`L_max=0.10` m) | `2.0` | 19 924 | `4.3056e-03` | `1.0231e-03` | `0.076 s` |
| Solenoid (`L_max=0.10` m) | `5.0` | 22 119 | `6.0207e-04` | `1.9253e-04` | `0.097 s` |
| Solenoid (`L_max=0.10` m) | `10.0` | 23 636 | `3.2843e-04` | `1.5283e-04` | `0.104 s` |
| Cylinder (`L_max=0.05` m) | `2.0` | 10 045 | `1.8016e-02` (> 1 %) | `3.0918e-03` | `0.028 s` |
| Cylinder (`L_max=0.05` m) | `5.0` | 11 610 | `1.5396e-03` | `8.6870e-04` | `0.033 s` |
| Cylinder (`L_max=0.05` m) | `10.0` | 12 934 | `9.7648e-04` | `6.9598e-04` | `0.046 s` |

A Kelvin transformation / ballooning open-boundary treatment is deferred to Tier 2.

### 4. Core dependency: `scipy` (`scipy>=1.11`, BSD-3-Clause)

`scipy` is promoted to `[project].dependencies` in `pyproject.toml` and updated in `NOTICE`.
Measured runtime environment: `scipy 1.18.1`, BSD-3-Clause licence, pre-built wheels for
CPython 3.11–3.13 across Linux, Windows, and macOS. `solve_axisymmetric` uses
`scipy.sparse.linalg.spsolve` (SuperLU direct LU, single-threaded, deterministic bit-identical
output across independent solves, FR-043) with symmetric elimination of Dirichlet DOFs.
`env -u PYTHONPATH .venv/bin/python tools/check_licence_boundary.py` passes with
`licence boundary OK (no GPL linkage)`.

### 5. Built-in graded tensor mesher (`src/magwave/fem2d/mesh.py`)

`rectangular_mesh` builds an axis-aligned tensor-product grid with forced breakpoints at source
radii/sheet lines and axial ends, uniform subdivision (`ceil(L/h)`) inside `refinement_box`, and
geometric growth (`1.2` default) to the `5x` air boundary.

Why built-in now: `gmsh` is GPL and restricted to external subprocesses (ADR-0002), while Stage 1
sources are axis-aligned rectangles and cylindrical sheets where exact breakpoints guarantee zero
split elements and exact sheet edge placement without external dependencies. **This is explicitly
not a general mesher strategy for M6 or arbitrary CAD geometry**, which will route through external
meshing (e.g., containerised gmsh) once non-axis-aligned / curved permeable solids enter.

### 6. Rejected alternative (deferred): reduced/total scalar potential (`Omega`-formulation)

Formulating `H = H_s - grad Omega` is **rejected for Stage 1 and deferred rather than dismissed**:
- In free space with distributed and thin-sheet currents, scalar potential formulations require
  computing the Biot–Savart source field `H_s` everywhere and introducing topological cuts for
  multiply-connected current loops, rather than placing `J_phi` and `K_phi` directly on the RHS.
- However, once nonlinear steel (`mu = mu(|B|)`) arrives in Stage 2 (M5.2) or full 3-D magnetostatics
  in Tier 2, `Omega` becomes preferable in steel/air assemblies because it uses a single scalar
  unknown in 3-D and avoids differentiating discontinuous `1/mu` across steel interfaces.
- **Trigger for adopting `Omega`:** introducing 3-D permeable steel assemblies (where 3-D vector
  `A` requires 3 edge DOFs + gauging) or strong `mu`-jump interface accuracy issues in M5.2+.

### 7. Accuracy certification vs research-owned tolerances

`Fem2dEngine` certifies `certified_accuracy = 1e-2` (1 %). This is an engine capability ceiling,
**not** a research-owned benchmark tolerance. The validation study (`tests/test_fem2d_validation.py`)
and corpus accuracy report (`magwave bench --engine fem2d --report-accuracy`) both pass well below
`1e-2`:

- **Validation gate (`tests/test_fem2d_validation.py`):**
  - Solenoid: `3.1142e-03` at `h = 1e-3` (8 673 nodes, `0.0393 s`) -> `6.0207e-04` at `h = 5e-4` (22 119 nodes, `0.0839 s`).
  - Cylinder: `4.3501e-03` at `h = 1e-3` (4 902 nodes, `0.0125 s`) -> `1.5396e-03` at `h = 5e-4` (11 610 nodes, `0.0376 s`).
- **Frozen corpus (`magwave bench --engine fem2d --report-accuracy`):**
  - 27/27 skipped under research tolerances (`1e-12` to `1e-4`), 9/9 axisymmetric cases measured
    within `1.00e-02` certified accuracy with worst relative error `8.23e-03` (`cylinder_axis_outside_Bz`).

## Consequences

- `magwave.fem2d` (`src/magwave/fem2d/`) delivers a self-contained, deterministic, BSD/Apache-compatible
  Tier 1 axisymmetric solver integrated behind `Fem2dEngine` (`src/magwave/engines/__init__.py`) and
  registered in `src/magwave/cli.py`.
- `case.manifest` bodies with role `permeable_solid` are explicitly rejected with
  `E_UNSUPPORTED_MATERIAL` pointing to Tier 1 stage 2 (M5.2), never silently approximated as air.
- `tests/test_package.py` and `tools/check_licence_boundary.py` continue to enforce zero GPL imports
  across `src/`.

## Appendix A — M5.2 stage A: linear permeable media

- Date: 2026-11-21
- Scope: linear soft-magnetic media (`mu_r` scalar constant per body) and uniform applied field
  `H_0 z_hat` in `magwave.fem2d` (`Fem2dEngine` version suffix `+m5.2a`), versioned material
  library `magwave.materials` (`v1`), additive `magwave.case/1` component schemas (`permeable`,
  `uniform_field`), and the Tier 0 FR-041 refusal gate (`E_UNSUPPORTED_PERMEABLE`).

All figures below were measured on 2026-11-21 in this checkout using
`env -u PYTHONPATH .venv/bin/python` (Python 3.12.3, NumPy 2.5.5, SciPy 1.18.1).

### A.1 Element-wise reluctivity and centroid assignment on the tensor mesh

`stiffness(mesh, nu)` accepts either a scalar reluctivity (preserving Stage 1 byte-identical) or a
`(m,)` array of positive per-element reluctivities `nu_e = 1 / (MU0 * mu_r)` multiplied directly into
each element's `k_elem` row without Python loops (`src/magwave/fem2d/assembly.py`).

Because `rectangular_mesh` builds an axis-aligned tensor-product `(r, z)` grid where every node is
shared across all incident triangles, there are no hanging nodes. Material regions for
axisymmetric `sphere` and `cylinder` `permeable` components are assigned **per element by triangle
centroid** (`(r_c, z_c)` inside the body) rather than by snapping mesh edges to curved boundaries.
Non-axisymmetric `block` bodies are rejected in `validate()` with `E_UNSUPPORTED_GEOMETRY`, and
overlapping `permeable` bodies are rejected with `E_UNSUPPORTED_MATERIAL`.

**Staircase-interface limitation (honest statement).** Axis-aligned `(r, z)` breakpoints place exact
edges on bounding planes (`r = r_c +- R`, `z = z_c +- R`), **not** along curved spherical surfaces.
A sphere's surface is therefore represented by an *element-wise staircase* approximation on the
tensor grid, not a node-conforming interface curve.

### A.2 Measured convergence on the analytic permeable sphere (`tests/test_fem2d_permeable.py`)

Benchmark case (`examples/steel-sphere-uniform-field.json`): sphere of radius `R = 0.010` m at the
origin, material `steel-team20` (shipped `mu_r = 294.7` from initial-segment secant of the TEAM 20
table) with per-case override `mu_r = 100.0`, uniform applied field `H_0 = 1.0e4` A/m along `+z`,
analytic interior flux density `B_in = MU0 * mu_r * 3 / (mu_r + 2) * H_0 = 3.695991e-02` T
(Jackson §5.10 / Landau & Lifshitz Electrodynamics §8).

Because the curved surface is staircased, the 1 % (`1e-2`) gate is not met at `R/8` or `R/16`; it
is met monotonically at `R/32` and `R/64` (reported passing pair):

| Level | `h` (m) | Nodes | Elements | Interface nodes | Computed `B_z` (T) | Analytic `B_z` (T) | Rel. err | Gate (`<= 1e-2`) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `R/8` | `1.2500e-03` | 1 431 | 2 704 | 31 | `4.032305e-02` | `3.695991e-02` | `9.0994e-02` | FAIL |
| `R/16` | `6.2500e-04` | 3 003 | 5 776 | 63 | `3.814140e-02` | `3.695991e-02` | `3.1967e-02` | FAIL |
| `R/32` (reported 1) | `3.1250e-04` | 6 670 | 12 996 | 117 | `3.732261e-02` | `3.695991e-02` | `9.8133e-03` | PASS |
| `R/64` (reported 2) | `1.5625e-04` | 17 578 | 34 596 | 247 | `3.717376e-02` | `3.695991e-02` | `5.7858e-03` | PASS |

At the default engine setting (`Fem2dEngine()`, `h = R/40 = 2.5000e-04` m, 9 180 nodes, 17 956
elements), measured error is `7.6651e-03` with relative linear-solve residual `4.508e-14` in
`0.0438 s`. Pole normal-`B` continuity across the top pole (`(0, 0, R - h/4)` vs `(0, 0, R + h/4)`
at `h = 1.5625e-04` m) measures `3.48865900e-02` T on both sides (relative difference `1.9890e-16`).

### A.3 Applied-field Dirichlet condition and truncation estimate

A `uniform_field` component (`H_0 z_hat`) is imposed without adding a volume RHS load by setting
non-homogeneous Dirichlet values `A_phi = MU0 * H_0 * r / 2` on the outer domain boundaries and
keeping `A_phi = 0` on the symmetry axis `r = 0` via symmetric elimination in
`solve_axisymmetric(..., values=...)` (`src/magwave/fem2d/solve.py`).

- **Exactness in free space / air (`mu_r = 1.0`):** Because `A_phi = MU0 * H_0 * r / 2` lies in the
  P1 finite-element space, the air-only sphere (`mu_r = 1.0` at `h = 6.25e-4` m) reproduces
  `MU0 * H_0 = 1.256637061436e-02` T to `7.413e-14` relative and matches the no-permeable solve
  to `7.620e-14` relative.
- **Far-field truncation estimate:** The induced dipole perturbation of a sphere of radius `R`
  decays as `(R / r)^3`. With `l_max = 2R` and default `far_field_factor = 5.0`, the outer boundary
  sits at `11 R`, so `(R / 11R)^3 ~ 7.5e-4` (< 0.1 %). Measured sensitivity at `h = 1.5625e-4` m:

| `far_field_factor` | Nodes | Elements | Computed `B_z` (T) | Rel. err vs analytic | Solve wall time |
|---:|---:|---:|---:|---:|---:|
| `2.0` (`5 R`) | 15 753 | 30 976 | `3.66390978e-02` | `8.6801e-03` | `0.0669 s` |
| `5.0` (`11 R`, default) | 17 578 | 34 596 | `3.71737551e-02` | `5.7858e-03` | `0.0849 s` |
| `10.0` (`21 R`) | 19 110 | 37 636 | `3.72123431e-02` | `6.8298e-03` | `0.0942 s` |

The boundary shift from `5x` to `10x` is `1.04e-3` relative (~0.10 %), confirming that `5x`
truncation is well below the `5.79e-3` staircase discretisation floor at `R/64`.

### A.4 FR-041 gate and Stage B deferrals

- **FR-041 Tier 0 gate:** `MagpylibEngine` supports `uniform_field` by analytic superposition — enforced-by: tests/test_fr041_gate.py::test_tier0_refuses_a_permeable_manifest_body
  (`MU0 * h0` added to `B_z`, without applying `mu0_ratio`) and refuses any case containing a
  `permeable` component or a `permeable_solid` manifest body with `E_UNSUPPORTED_PERMEABLE`,
  pointing to `Tier 1 stage 2 (M5.2)` (`fem2d`) and producing zero numeric samples.
- **Deferred to Stage B (not implemented here):** nonlinear `B(H)` curve evaluation and
  differentiation, fixed-point / Newton-Raphson iteration, CAD body meshing for `permeable_solid`
  manifests, and external Elmer cross-comparison.

## Appendix B — M5.2 stage B: nonlinear B–H and Newton

- Date: 2026-11-21
- Scope: monotone `BhCurve` (`PchipInterpolator`) material response (`src/magwave/materials/nonlinear.py`),
  consistent tangent Newton–Raphson solver with Armijo backtracking (`src/magwave/fem2d/nonlinear.py`),
  FR-042 `ConvergenceInfo` reporting in `Result`, `magwave report`, and `--max-iterations`, plus the
  external Elmer `MagnetoDynamics2D` nonlinear cross-comparison (`tools/elmer_permeable.py`). CAD body
  meshing for `permeable_solid` manifests remains deferred to M6.

All figures below were measured on 2026-11-21 in this checkout using
`env -u PYTHONPATH .venv/bin/python` (Python 3.12.3, NumPy 2.5.5, SciPy 1.18.1, Elmer 9.0 via
Docker `scimulate/elmer`).

### B.1 Nonlinear B–H law and saturation extrapolation (`BhCurve`)

`BhCurve` (`src/magwave/materials/nonlinear.py`) inverts `|B| -> H` once per `Material` (cached)
using `scipy.interpolate.PchipInterpolator` across the strictly monotone tabulated points (rejecting
non-monotone `H` or `B` tables loudly without sorting or dropping points):

- **Below the first non-zero table point (`0 <= |B| < B_1`):** linear from the origin at shipped
  `mu_r`, `H(|B|) = |B| / (MU0 * mu_r)` on `[0, 0.9 * B_1]` (`0.009 T` for `steel-team20`), joined
  via a monotone `C^1` Hermite cubic bridge on `(0.9 * B_1, B_1)` to `(B_1, H_1, H'(B_1))` when
  rounded `mu_r` differs slightly from `B_1 / (MU0 * H_1)`.
- **Above the last table point (`|B| > B_last = 2.30 T`):** requires `saturation_B` (`Ms = 2.16 T`
  for `steel-team20`; `ValueError` naming material and field if `None`). Rather than extending the last
  table segment's finite permeability slope (which for TEAM 20 is `~1.59 * MU0` and would delay
  physical saturation indefinitely), the curve approaches the physical saturation asymptote
  `B = MU0 * H + Ms` (`H = (|B| - Ms) / MU0`, `dB/dH = MU0`) via a monotone `C^1` Hermite bridge on
  `(2.30, 2.388938) T` (bridging the `-23.6 kA/m` offset jump cleanly so Newton line search never
  stalls across `2.30 T`), and evaluates `H = (|B| - Ms) / MU0` exactly for `|B| >= 2.388938 T`.

### B.2 Consistent tangent Newton–Raphson and Armijo backtracking

Using the same weak-form element matrix `G^e` from `assembly.element_geometric_stiffness(mesh)` and
centroid flux density `|B_e| = sqrt(B_r^2 + B_z^2)` (`B_r = -sum_k c_k A_k`,
`B_z = sum_k (b_k + 1/(3 r_c)) A_k`), the discrete residual is:

```
R(A) = K(nu(|B|)) A - f          (constrained rows: A_i - g_i)
```

Differentiating `nu(|B_e|) G^e A_e` yields the **consistent symmetric tangent Jacobian** assembled
into a single CSR matrix (`src/magwave/fem2d/nonlinear.py`):

```
J = K(nu) + sum_e outer_e( (G^e A_e), (dnu/db)_e * (db/dA)_e )
(db/dA_k)_e = ( B_r * (-c_k) + B_z * (b_k + 1/(3 r_c)) ) / |B_e|
(dnu/db)_e  = ( H'(|B_e|)*|B_e| - H(|B_e|) ) / |B_e|^2
```

Elements with `|B_e| <= 1e-15 T` return `nu(0) = 1/(MU0*mu_r)`, `dnu/db = 0`, and `db/dA = 0` directly
without `0 / 0`. Central-difference verification on a coarse TEAM 20 cylinder+solenoid mesh across 5
random perturbations (`tests/test_fem2d_nonlinear.py`) measured relative errors `1.383e-11`,
`1.053e-11`, `1.010e-11`, `8.830e-12`, and `1.082e-11` (all `<< 1e-6`).

**Why the tangent stays symmetric** (verified, not assumed): for the `A_phi` element,
`(G^e A^e)_i = area_e r_c |B_e| (d|B|/dA_i)_e` with `A^e` the element nodal vector — the residual's
gradient operator *is* the flux operator — so the rank-one `dnu/d|B|` term collapses to
`dnu * area_e r_c |B_e| (db/dA) ⊗ (db/dA)`, which is symmetric by construction. Measured
`max |J - J^T| / max |J| = 1.6e-22` at a converged saturating solution, so no unsymmetric solver is
needed.

**Line search & convergence criteria (as implemented):**

- Initial iterate: `A^(0) = g` (`0` on free DOFs, Dirichlet `g` on constrained DOFs), so iteration 1
  performs the initial-permeability linear solve.
- Armijo backtracking on `||R||_2`: `ARMIJO_ALPHA0 = 1.0`, `ARMIJO_RHO = 0.5`, `ARMIJO_C = 1e-4`,
  `ARMIJO_MAX_HALVINGS = 12`, accepting `||R(A + alpha dA)||_2 <= (1 - c * alpha) ||R(A)||_2`.
- Convergence gate: `||R||_2 / ||f||_2 <= tol` (`DEFAULT_TOL = 1e-8`, `DEFAULT_MAX_ITERATIONS = 50`).
- Failure policy: hitting `max_iterations` or failing Armijo backtracking raises `EngineError` with
  `E_NOT_CONVERGED` naming the case, iteration count, cap, and final relative residual; no `Result` or
  samples are returned, and `magwave report --engine fem2d --max-iterations 1 <case>` exits `1` with
  empty stdout. Linear runs (`material_model="linear"` or linear-only materials) continue to use
  direct `spsolve` unchanged and report `iterations = 1`.

### B.3 Measured convergence, saturation sweep, and Elmer comparison

Saturating cylinder benchmark (`steel-team20` cylinder `R = 10 mm`, `L = 60 mm` inside coaxial
solenoid `R1 = 15 mm`, `R2 = 25 mm`, `L = 100 mm`, `N = 1000`, default `Fem2dEngine(material_model="nonlinear")`,
`2 958` nodes, `5 676` elements, `33` interface nodes, `I = 1.0 A`):

| Iteration | Relative residual `\|\|R\|\| / \|\|f\|\|` |
|---:|---:|
| 1 | `1.338776e-02` |
| 2 | `1.092056e-05` |
| 3 | `1.697520e-09` |

Saturation current sweep at probe `(0, 0, 0)` (`B_z`):

| Current (A) | Probe `B_z` (T) | `dB/dH` (T*m/A) | `(dB/dH) / MU0` | Newton iters | Final rel. residual |
|---:|---:|---:|---:|---:|---:|
| `0.5` | `0.070655` | `9.058758e-04` | `720.8731` | 3 | `6.972e-11` |
| `1.0` | `0.140492` | `1.832705e-03` | `1458.4205` | 3 | `1.698e-09` |
| `2.0` | `0.278204` | `3.929663e-03` | `3127.1261` | 4 | `6.272e-13` |
| `5.0` | `0.724099` | `2.167427e-03` | `1724.7837` | 4 | `1.290e-12` |
| `10.0` | `1.462570` | `1.313233e-04` | `104.5038` | 4 | `1.642e-10` |
| `20.0` | `2.206893` | `2.004515e-06` | `1.5951` | 7 | `1.703e-12` |
| `40.0` | `2.527636` | `1.256637e-06` | `1.0000` | 8 | `4.677e-10` |

TEAM 20 exhibits the physical Rayleigh permeability increase from the initial toe (`mu_r ~ 295`) to its
steepest slope around `~0.28 T` (`I = 2.0 A`), then `dB/dH` drops strictly monotonically through the
knee and saturates to `(dB/dH)/MU0 = 1.0000` (`B_z = 2.5276 T > 2.30 T`) at `40.0 A`.

External Elmer 9.0 second-code comparison (`tools/elmer_permeable.py`, `I = 1.0 A`, `1.0 m` air box,
`Real Cubic Monotone` `H-B Curve` fed from `magwave.materials` JSON):

| Code | Mesh | Nonlinear iters | Wall time | `B_z(0, 0, 0)` (T) | Ratio | Rel. diff | Gate (`<= 3 %`) |
|---|---|---:|---:|---:|---:|---:|---|
| Elmer `MagnetoDynamics2D` | `22 366` nodes / `22 050` quads | 5 | `1.93 s` | `0.143836` | — | — | — |
| MagWave `Fem2dEngine` (`h = 1e-3` m) | `9 291` nodes / `18 144` tris | 4 | `0.198 s` | `0.143932` | `1.00066` | `0.0665 %` | PASS |

### B.4 Review of §6 (`Omega`-formulation judgement)

Stage B does **not** change §6's decision to keep axisymmetric `A_phi` in Tier 1 and defer scalar
potential `Omega` to Tier 2 (3-D). In 2-D axisymmetry, `A_phi` already requires only one scalar unknown
per node, embeds thick/sheet coil currents directly on the RHS without topological cuts or Biot–Savart
pre-passes, yields a symmetric positive-definite consistent Newton tangent matrix, converges in 3–8
iterations across deep saturation, and agrees with Elmer `MagnetoDynamics2D` to `0.0665 %` on the
nonlinear steel cylinder. The trigger for `Omega` remains unchanged: full 3-D permeable steel
assemblies in Tier 2 (where 3-D vector `A` introduces three edge DOFs per edge plus gauging) or curved
3-D material interface accuracy limits in later milestones.

### B.5 Known limitation — default-mesh under-resolution of the shipped nonlinear example

The shipped example `examples/steel-cylinder-solenoid-nonlinear.json` solved through the CLI at the
**default** mesh (`magwave report --engine fem2d`, `2 958` nodes / `5 676` elements) reports
`B_z(0, 0, 0) = 0.140492 T`, which is **2.32 % below Elmer (`0.143836 T`)** and **2.39 % below our own
refined-mesh run (`0.143932 T`, `h = 1e-3` m, `9 291` nodes, `tools/elmer_permeable.py`)**. This is
**mesh convergence, not disagreement in the physics**: refine the same case and the two codes agree to
`0.0665 %` (B.3). The gap is a property of the automatically chosen default element size for this
geometry, and it is larger than the `3 %` Elmer gate only because that gate is applied at the refined
`h = 1e-3` m discretisation.

**Follow-up (not done in M5.2):** expose mesh density control in the case schema and CLI (e.g. a
per-case `mesh.element_size` / refinement level) so the shipped examples solve at a documented,
reproducible resolution instead of an implicit default. Resolved in M5.5 by [ADR 0011](0011-mesh-density-control.md).
