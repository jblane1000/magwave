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

All figures below were measured on 2026-11-19 in `/home/jblane1000/projects/magwave` using
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
