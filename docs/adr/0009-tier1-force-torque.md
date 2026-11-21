# ADR 0009 — Tier 1 axisymmetric force, honest torque refusal, and Tier 0 permeable force gate (`M5.4`)

- Status: accepted
- Date: 2026-11-21
- Deciders: Jason (owner)
- Amends: ADR-0007 (cross-reference only; does not modify ADR-0007) — adds Tier 1 force evaluation by two independent methods (`maxwell_stress_tensor` and `coenergy_virtual_work`), honest torque refusal (`E_TORQUE_NOT_MEANINGFUL`), and Tier 0 permeable-force refusal (`E_UNSUPPORTED_FORCE_PERMEABLE`).
- Supersedes nothing

## Context

Force is the engineering quantity users ask for most ("how hard will it pull?") and the easiest quantity to get quietly wrong: surface-stress integrals depend sensitively on mesh discretisation near corners and material discontinuities, while energy differentiation has sign and potential traps (free-current coenergy vs zero-current magnetic energy) that can agree by luck on one linear test while failing on steel or permanent magnets. M5.4 adds force to `magwave.fem2d` (`Fem2dEngine` suffix `+m5.4`) using **two independent arithmetic paths** verified against each other, against closed-form Neumann and Nagaoka references, and against an external Elmer virtual-work solve, together with an explicit mesh-sensitivity table.

All figures below were measured on 2026-11-21 in this checkout using `env -u PYTHONPATH .venv/bin/python` (Python 3.12.3, NumPy, SciPy, Elmer 9.0 via Docker `scimulate/elmer:latest`).

## Decision

### 1. Maxwell stress tensor (`src/magwave/fem2d/force.py:maxwell_force`)

In axisymmetric cylindrical coordinates `(r, phi, z)` with `T_ij = (1/mu_0)(B_i B_j - 0.5 delta_ij |B|^2)` and surface area element `dS = 2*pi*r dl`, the axial force on the body enclosed by a closed meridional contour in air is:

```
F_z = closed_integral [ (1/mu_0) B_r B_z n_r + (1/(2 mu_0)) (B_z^2 - B_r^2) n_z ] 2*pi*r dl
```

Radial and azimuthal components `F_r` and `F_theta` vanish identically by axisymmetry (`B_phi = 0`); `maxwell_force` integrates the radial traction over 8 symmetric azimuthal sectors and asserts `|F_x|, |F_y| <= 1e-13 * max(|F_z|, 1.0)`.

- **Closed contour loop:** rectangle-with-axis `(0, z_lo) -> (r_c, z_lo) -> (r_c, z_hi) -> (0, z_hi) -> (0, z_lo)` with outward normal `n` (`(0, -1)` bottom, `(+1, 0)` outer wall, `(0, +1)` top, `(-1, 0)` axis). The symmetry axis segment (`r = 0`, `dS = 0`) is included explicitly for bookkeeping and contributes `0.0`.
- **Quadrature:** each contour segment is partitioned at intersected `mesh.r_axis` / `mesh.z_axis` lines and integrated with 2-point Gauss–Legendre quadrature via `magwave.fem2d.field.sample_field`.
- **Contour placement rule & why never through steel or a magnet face:** — enforced-by: tests/test_m54_forces.py::test_torque_refusal_and_force_error_contracts
  1. The contour must enclose **only** the target active `Magnet` or `Coil` (`r_c > r_target + element_size`, `z_lo < z0_target`, `z_hi > z1_target`), checked against component geometry (not element masks). Passive `Permeable` bodies are never valid standalone targets (`E_FORCE_TARGET_UNKNOWN`).
  2. Every air segment (`bottom`, `outer`, `top`) must lie strictly in air outside all `Magnet`, `Coil` (including thin current sheets), and `Permeable` geometries (`E_FORCE_CONTOUR_IN_MATERIAL`), and must not enclose any other component (`E_FORCE_CONTOUR_DOES_NOT_ENCLOSE`).
  3. **Why never through steel or a magnet face:** across a steel interface (`mu_r != 1`) or a magnet face (`K_phi = M` / normal `H` jump), tangential `B` and/or normal `H` are discontinuous, so `T_ij` evaluated with vacuum `1/mu_0` jumps across the surface and piecewise-linear element-averaged `B` suffers an `O(1)` / first-order interface error that destroys contour independence.
- **Grid snapping & single-element air-gap rule:** requested `(r_c, z_lo, z_hi)` values are snapped to the nearest `mesh.r_axis` / `mesh.z_axis` lines and both the snapped coordinates and snap deltas are reported in `ForceResult.contour` and `ForceResult.notes`. If snapping moves a contour onto or inside any material body while air grid lines exist in the gap, `maxwell_force` raises `E_FORCE_CONTOUR_IN_MATERIAL` rather than silently integrating on an interface. Only when a requested coordinate falls within `15%` of the midpoint of a single-element air gap (`axis[k]` and `axis[k+1]` both material walls with zero interior grid lines, e.g., the 1 mm gap at `h = 1.0e-3` m in Gate 4) is the segment placed on the cell midline `0.5 * (axis[k] + axis[k+1])` and recorded explicitly in `ForceResult.notes`. — enforced-by: tests/test_m54_forces.py::test_torque_refusal_and_force_error_contracts
- **Fixed physical force refinement box:** `_force_engine_for_case` pads the source bounding box by a mesh-independent physical margin `max(0.5 * r_target, 2 * h)` instead of `Fem2dEngine`'s probe-default `2 * h` so that exterior air around sheets and gaps refines consistently under `h -> h/2` (without modifying `Fem2dEngine.solve()` / `solve_with_report()`).

### 2. Coenergy / virtual work (`src/magwave/fem2d/force.py:coenergy_force`)

Central finite difference over two independent `Fem2dEngine._solve_core` runs with the target body and its mesh breakpoints shifted by `+delta` and `-delta` in `z` (default `delta = element_size`, clamped to `0.25 * axial_gap` if `h >= 0.9 * axial_gap` and noted in `ForceResult.notes`):

- **Variational potential & sign convention (no hand-waving):**
  - **Cases with fixed-current `Coil` sources:** the stationary variational functional is magnetic coenergy `W' = integral(J . A) dV - integral(w(B)) dV = 2*pi * f_coil^T A - U`, which equals `sum_e 0.5 * nu_0 * |B_e|^2 * 2*pi*r_c,e*A_e` in linear air (`1e-16` agreement verified). At fixed currents, axial force is `F_z = +dW'/dz = +(W'(+delta) - W'(-delta)) / (2*delta)` — validated against the analytic Neumann force on the coaxial-loop pair (`-7.282368e-7 N`, attractive).
  - **Pure permanent-magnet + permeable/air cases (`J = 0`):** when there are no free currents, `integral(B . H) dV = 0` identically across the whole domain by integration by parts (`curl H = 0`, `div B = 0`). Differentiating `0.5 * integral(B . H) dV` would yield zero. Instead, the variational potential at fixed magnet polarisation/recoil law is the **total stored magnetic energy** `U = integral w(B) dV` with `F_z = -dU/dz = -(U(+delta) - U(-delta)) / (2*delta)`, where:
    - Air: `w = 0.5 * (1/mu_0) * |B|^2`
    - Linear steel: `w = 0.5 * nu_e * |B|^2`
    - Nonlinear steel: `w(|B|) = integral_0^|B| H(B') dB'` evaluated analytically via `BhCurve.magnetic_energy_density` (`src/magwave/materials/nonlinear.py`)
    - Uniform-`B_r` magnet: `w = (0.5/mu_0) * (B_r^2 + (B_z - B_rz)^2)`
    - Linear-recoil `DemagCurve`: `w = 0.5 * s * (B_r^2 + B_z^2) + sign * c * B_z + c^2 / (2 * s)`
    - Nonlinear second-quadrant `DemagCurve`: 16-point Gauss–Legendre ray integral from `(0, sign * B_r)` to `(B_r, B_z)` of `H(B(u)) . dB`.
- **Finite-difference step-size study & discretisation limitations (`coenergy (finite-difference step study)`):**
  On the Gate 3 magnet + TEAM 20 steel plate case at fixed `h = 5.0e-4 m` (`1 mm` air gap, `F_maxwell = 102.28330 N`), sweeping the virtual-work central-difference step `delta` across `h`, `h/2`, and `h/4` gives:

  | Rung | Step `delta` | `delta` (m) | Shifted mesh | `F_coenergy` (N) | Diff vs `delta = h` | vs Maxwell |
  |---|---|---:|---|---:|---:|---:|
  | coenergy (finite-difference step study) | `h` | `5.00e-04` | `12 126n / 23 808e` | `102.81233` | `+0.000%` | `0.515%` |
  | coenergy (finite-difference step study) | `h/2` | `2.50e-04` | `12 220n / 23 994e` | `101.98703` | `-0.803%` | `0.290%` |
  | coenergy (finite-difference step study) | `h/4` | `1.25e-04` | `12 220n / 23 994e` | `101.59921` | `-1.180%` | `0.669%` |

  - Peak-to-peak spread across `delta in {h, h/2, h/4}`: `1.21312 N` (`1.180%` of `F_coenergy(delta = h)`).
  - **Step-size sensitivity & agreement-gate limitation:** At `h = 5.0e-4 m`, the coenergy number carries a finite-difference step-size band of about `1%` (`0.803%` between `delta = h` and `h/2`, and `1.180%` peak-to-peak across `h`, `h/2`, `h/4`). This band combines the `O(delta^2)` curvature of `F(z)` across the `1 mm` gap with sub-`h` remeshing differences when shifted magnet endpoints land between uniform `h` grid lines (`12 126 -> 12 220` nodes). Therefore, the `2%` Maxwell-vs-coenergy agreement gate at `h = 5.0e-4 m` explicitly includes this finite-difference step error (alongside the `~1–1.5%` spatial discretisation error shown in Gate 4).
  - **Reported `delta` choice and rationale:** The default and Gate 3 / Gate 6 reported coenergy value uses `delta = h = 5.0e-4 m` (`102.81233 N`) because shifting by one element width aligns shifted magnet boundaries with existing uniform refinement-box grid lines (keeping the `12 126`-node / `23 808`-element mesh uniform without inserting sub-`h` slivers or amplifying finite-difference cancellation at `delta << h`, while automatically clamping to `0.25 * axial_gap` when `h >= 0.9 * axial_gap`). For Gate 4 (`h = 1.0e-3, 5.0e-4, 2.5e-4 m`), `delta` is fixed at `2.5e-4 m` (`h_min = 0.25 * gap`) so the mesh-refinement table isolates `h`-convergence at constant `delta` (and because `delta = h` at `h = 1.0e-3 m` equals the `1 mm` gap). *(See the M5.4c amendment below: silent non-aligned clamping and sub-cell `delta` sweeps are retired and refused with `E_FORCE_DELTA_NOT_ALIGNED`, and Gate 4's coenergy column is re-based to cell-aligned `delta = h`.)*

### 3. Honest torque refusal (`torque` -> `E_TORQUE_NOT_MEANINGFUL`)

`torque(case, engine, target=...)` and `Fem2dEngine.torque(...)` always raise `ForceError` with `code = "E_TORQUE_NOT_MEANINGFUL"`, explicitly stating that the axisymmetric `A_phi` formulation cannot represent a tangential stress component, torque about the symmetry axis is identically zero by construction (not a computed physical zero), and a 3D Tier 1 formulation would be required.

### 4. Tier 0 permeable force refusal (`MagpylibEngine`)

When any probe requests `"F"` AND the case contains a `Permeable` component or a `permeable_solid` manifest body, `MagpylibEngine.validate` emits `E_UNSUPPORTED_FORCE_PERMEABLE` **before** checking generic `E_UNSUPPORTED_PERMEABLE`, naming the missing capability (`force in the presence of permeable material` requiring Maxwell stress / virtual work on the FEM mesh) and Tier 1 engine `'fem2d'` (`Fem2dEngine.force`), ending with `"nothing was solved."` Free-space `F`-probe cases keep `E_UNSUPPORTED_QUANTITY` and name the tiers that answer force. `Fem2dEngine.validate` keeps `E_UNSUPPORTED_QUANTITY` for `F` probes and directs callers to `Fem2dEngine.force(case, method=...)`.

## Measured Verification Numbers

### Gate 1 — Two coaxial thin-sheet loops (`R = 10 mm`, `L = 2 mm`, `d = 10 mm`, `N = 1`, `I = 1 A`)

Neumann reference self-check at `L/a = 1e-3` vs closed-form loop `dM/dd`: relative difference `3.386e-07` (`<= 1e-4`, 48-pt Gauss–Legendre per sheet; `closed-form (Neumann)`).

| Rungs | `h` (m) | `F_ref` (N) | `F_maxwell` (N) | Maxwell err | `F_coenergy` (N) | Coenergy err | Falls? |
|---|---|---:|---:|---:|---:|---:|---|
| closed-form (Neumann), Maxwell stress, coenergy | `5.00e-04` | `-7.282368e-07` | `-7.314244e-07` | `0.438%` | `-7.310881e-07` | `0.392%` | baseline |
| closed-form (Neumann), Maxwell stress, coenergy | `2.50e-04` | `-7.282368e-07` | `-7.288135e-07` | `0.079%` | `-7.287392e-07` | `0.069%` | YES |

### Gate 2 — Solenoid inductance from coenergy (`R = 10 mm`, `L = 200 mm`, `N = 400`, `I = 1 A`, `h = 5e-4 m`)

Definition: `L_fem = 2 * W' / I^2 = 2 * W' * N^2 / (N * I)^2` (`coenergy`, `31 168` nodes / `61 236` elements). Because `l / D = 10` (`l / R = 20`) has a `-4.1193%` finite-length end effect, raw deviation from `L_ideal = mu_0 N^2 pi R^2 / l` exceeds `0.5%`; applying the exact Nagaoka/Lorenz thin-sheet factor `K_N = 0.95880712` (`closed-form (Neumann)`) reduces the deviation to `-0.0327%` (`<= 0.5%` PASS):

| Rung | `L_fem` (H) | `L_ideal` (H) | Raw dev | End-effect factor + basis | Corrected dev |
|---|---:|---:|---:|---|---:|
| coenergy | `3.027186e-04` | `3.158273e-04` | `-4.151%` | Exact Nagaoka `K_N = 0.95880712` (`l/D = 10`, `l/R = 20`) | `-0.0327%` |

### Gate 3 — Magnet on steel plate (`N42-20C-LINEAR-RECOIL` + `steel-team20`, `h = 5e-4 m`, `1 mm` gap)

- Mesh: `12 126` nodes / `23 808` elements
- Maxwell stress (`contour A = (0.012, -0.011, 0.0105)` m, snap delta `dz_hi = -1.735e-18 m`): `F_maxwell = 102.28330 N`
- Coenergy (`delta = 5.0e-4 m`, `U(+d) = 0.85175456 J`, `U(-d) = 0.95456689 J`, `-dU/dz`): `F_coenergy = 102.81233 N`
- Method agreement: `0.5146%` (`<= 2%` PASS)
- Magnet operating point (`ConvergenceInfo.magnet_operating_points[0]`): `H = -206321.3 A/m`, `B = 0.999520 T`

### Gate 4 — Force vs mesh sensitivity (required deliverable, `delta = 2.5e-4 m`, Contour A)

| Rungs | `h` (m) | Nodes | Elements | `F_maxwell` (N) | `d%` | `F_coenergy` (N) | `d%` | Richardson `p` | Est. error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Maxwell stress, coenergy | `1.00e-03` | 4 400 | 8 532 | `104.21125` | `---` | `102.57859` | `---` | `---` | `---` |
| Maxwell stress, coenergy | `5.00e-04` | 12 126 | 23 808 | `102.28330` | `-1.850%` | `101.98703` | `-0.577%` | `---` | `---` |
| Maxwell stress, coenergy | `2.50e-04` | 36 792 | 72 812 | `101.48761` | `-0.778%` | `101.63452` | `-0.346%` | `1.28 / 0.75` | `0.55% / 0.51%` |

- Richardson extrapolation:
  - Maxwell stress: `p = 1.277`, `F_extrap = 100.92843 N`, estimated discretisation error at `h = 2.5e-4 m` is `0.554%` (and `1.34%` at `h = 5.0e-4 m`).
  - Coenergy (`delta = 2.5e-4 m`): `p = 0.747`, `F_extrap = 101.11467 N`, estimated discretisation error at `h = 2.5e-4 m` is `0.514%` (and `0.86%` at `h = 5.0e-4 m`).
- **Convergence statement:** Both methods converge monotonically toward `~101.0 N` (`~0.55%` / `~0.51%` estimated discretisation error at `h = 2.5e-4 m`); no single mesh value is claimed exact, and `h = 5.0e-4 m` carries `~1–1.5%` discretisation error relative to the extrapolated limit. — enforced-by: tests/test_m54_forces.py::test_delta_ladder_analysis_exact_order_and_edge_cases

### Gate 5 — Contour independence (`h = 5e-4 m`, same mesh)

| Rung | Contour A (snapped, m) | Contour B (snapped, m) | `F_A` (N) | `F_B` (N) | Difference |
|---|---|---|---:|---:|---:|
| Maxwell stress | `(0.012, -0.011, 0.0105)` | `(0.015, -0.012, 0.0105)` | `102.28330` | `102.33411` | `0.0497%` |

### Gate 6 — Elmer second-code virtual work (`tools/elmer_force.py`, `delta = 2.5e-4 m`)

ElmerGrid `tools/elmer/magnet_plate.grd` (`18 315` nodes / `18 040` Q1 quads, `0.5 m` box) with identical `N42-20C-LINEAR-RECOIL` and TEAM 20 H-B curve, integrated with 2x2 Gauss–Legendre over Elmer's grid (`F_z = -(W(+d) - W(-d)) / (2d)`):

| Rung | Elmer mesh | `W(+d)` (J) | `W(-d)` (J) | `F_elmer` (N) | vs Maxwell (`h=5e-4`) | vs Coenergy (`h=5e-4`) | Gate (`<= 5%`) |
|---|---|---:|---:|---:|---:|---:|---|
| second code (Elmer) | `18315n/18040e` | `0.879490` | `0.930355` | `101.72932` | `0.545%` | `1.065%` | PASS |

### Gate 7 — Determinism (`h = 5e-4 m`, Gate 3 steel case)

- Maxwell stress SHA-256 (`repr`): `ee9c7c9901119dc5de153c3b59a91de2567fe7096d61e96fb36a4b6680075ccd` (equal: `True`)
- Coenergy SHA-256 (`repr`): `102431491605033de38933502893b25f3e94e052f7ca080f2dfdfb76c812cdcd` (equal: `True`)

## Consequences

- `magwave.fem2d.force` provides deterministic, Qt-free/OCP-free axisymmetric force evaluation (`maxwell_force`, `coenergy_force`, `magnetic_coenergy`, `ForceResult`, `ForceError`) and honest torque refusal (`torque`), integrated behind `Fem2dEngine.force` / `Fem2dEngine.torque`.
- `MagpylibEngine` rejects permeable force requests cleanly with `E_UNSUPPORTED_FORCE_PERMEABLE`, preventing silent free-space approximations when steel is present.
- All 27 frozen benchmarks and existing non-force `Fem2dEngine.solve()` results remain unchanged.

## Amendment — M5.4c (cell-aligned virtual-work shift)

### Hypothesis and Measured Test Outcome

- **Hypothesis (tested in M5.4c):** `delta = h` in M5.4 was not a lucky point. On a uniform axial refinement grid, shifting the active target by an exact integer multiple `k * h` (`k >= 1`) leaves all shifted boundaries on existing uniform grid lines, producing a discrete mesh identical to the unshifted mesh so the central difference evaluates one smooth discrete energy function with `O(delta^2)` convergence. Conversely, the `1.18%` spread seen when sweeping sub-cell steps (`delta < h`) in the rejected M5.4b attempt was an artifact of remeshing rather than physics.
- **Test outcome — CONFIRMED:** Running the cell-aligned ladder `(h, 2h, 3h)` at `h = 2.50e-04 m` (where all three multiples fit strictly inside the `1.0e-3 m` axial gap of the Gate 3 magnet + TEAM 20 steel plate case) preserves exact node/element count invariance across every rung (`36 792` nodes / `72 812` elements on both `+delta` and `-delta`, matching the unshifted mesh; `gap_cells_plus + gap_cells_minus == 8 = 2 * gap / h`), exhibits monotonic asymptotic `O(delta^2)` convergence (`p_per_pair = (+2.15022)`, `p_fit = +2.15022`, `monotonic = True`, `asymptotic = YES`), and extrapolates to `F_extrapolated = 101.37869 N +/- 0.25583 N` (`+/- 0.252%`). That extrapolant agrees within `0.8844%` with Gate 3 `F_maxwell(h = 5e-4) = 102.28330 N` (PASS `<= 2%`), within `0.107%` with `F_maxwell(h = 2.5e-4) = 101.48761 N`, and within `0.345%` with independent Elmer (`101.72932 N`). — enforced-by: tests/test_m54_forces.py::test_gate3_aligned_ladder_h2_5e4

### Superseded History — Rejected M5.4b Sub-Cell Sweep

The unmerged M5.4b branch (`feat/m5-4b-force-refine`, NOT merged) swept sub-cell steps `delta = h/8, h/4, h/2, h` at `h = 5.0e-4 m`:

| `delta` (m) | `F_coenergy` (N) | Nodes | Elements | Status |
|---:|---:|---:|---:|---|
| `6.25e-05` (`h/8`) | `101.18890` | 12 220 | 23 994 | SUPERSEDED HISTORY |
| `1.25e-04` (`h/4`) | `101.59921` | 12 220 | 23 994 | SUPERSEDED HISTORY |
| `2.50e-04` (`h/2`) | `101.98703` | 12 220 | 23 994 | SUPERSEDED HISTORY |
| `5.00e-04` (`h`) | `102.81233` | 12 126 | 23 808 | Cell-aligned `k=1` |

- `p_per_pair = (-0.08134, +1.08954)`, `p_fit = +0.50410`, `monotonic = True`, `asymptotic = NO`.
- `F_extrapolated = 100.20786 N +/- 1.04757 N`, agreement vs `F_maxwell = 102.28330 N` was `2.0291%` -> **FAIL** (`> 2%`).
- **Reason superseded:** At every `delta < h`, the shifted magnet faces landed between uniform `h` grid lines, inserting an extra `z`-line (`129 -> 130`), changing mesh counts (`12 126 -> 12 220` nodes, `23 808 -> 23 994` elements), and splitting the `1 mm` gap asymmetrically (`2` cells at `+delta` vs `3` at `-delta`). The `1.18%` band was a remeshing artifact rather than physics, and Richardson extrapolation on a discontinuous step function produced a confident wrong answer (`100.20786 N`, `1.5%` below Elmer). M5.4c replaces sub-cell shifting with strict alignment enforcement (`E_FORCE_DELTA_NOT_ALIGNED` raised before clearance checking or solving) and removes the silent `0.25 * max_shift` clamp when `delta is None`.

### Cell-Aligned Ladders & Mesh-Count Invariance (`coenergy_delta_study`)

#### Study A — `h = 5.00e-04 m` (Gate 3 `h`, base `12 126` nodes / `23 808` elements, `1 mm` axial gap)

| `delta` (m) | `delta/h` | `F_coenergy` (N) | Change vs prev % | Nodes (`+d / -d`) | Elements (`+d / -d`) | Gap cells (`+d / -d`) | Order `p` |
|---:|---:|---:|---:|---|---|---|---:|
| `5.00e-04` | `1.00` | `102.81233` | `---` | `12126 / 12126` | `23808 / 23808` | `1 / 3` | `---` |

- Refused rungs (verbatim, not solved):
  - `delta = 1.00e-03 m` (`2h`): `E_FORCE_UNSUPPORTED_CASE: gate3_magnet_steel_plate: virtual displacement delta=0.001 m (2*h with h=0.0005 m) meets or exceeds axial clearance 0.001 m to adjacent body; nothing was solved.`
  - `delta = 1.50e-03 m` (`3h`): `E_FORCE_UNSUPPORTED_CASE: gate3_magnet_steel_plate: virtual displacement delta=0.0015 m (3*h with h=0.0005 m) meets or exceeds axial clearance 0.001 m to adjacent body; nothing was solved.`
- Analysis: `p_per_pair = ()`, `p_fit = None`, `monotonic = True`, `asymptotic = NO`, `f_extrapolated_n = None`.
- Invariant check: **HOLDS** (`12126n / 23808e` everywhere matching base; `gap_cells_plus + gap_cells_minus = 1 + 3 = 4 = 2 * (1e-3 / 5e-4)`).

#### Study B — `h = 2.50e-04 m` (base `36 792` nodes / `72 812` elements, `1 mm` axial gap)

| `delta` (m) | `delta/h` | `F_coenergy` (N) | Change vs prev % | Nodes (`+d / -d`) | Elements (`+d / -d`) | Gap cells (`+d / -d`) | Order `p` |
|---:|---:|---:|---:|---|---|---|---:|
| `2.50e-04` | `1.00` | `101.63452` | `---` | `36792 / 36792` | `72812 / 72812` | `3 / 5` | `---` |
| `5.00e-04` | `2.00` | `102.51429` | `+0.866%` | `36792 / 36792` | `72812 / 72812` | `2 / 6` | `---` |
| `7.50e-04` | `3.00` | `104.09425` | `+1.541%` | `36792 / 36792` | `72812 / 72812` | `1 / 7` | `+2.150` |

- Refused rungs: none (`0` refused).
- Analysis (spacing-aware least-squares power-law fit across `(h, 2h, 3h)`):
  - `p_per_pair = (+2.15022)`, `p_fit = +2.15022`, `monotonic = True`, **`asymptotic = YES`** (`1.5 <= p_fit <= 2.5` and diff ratio `1.57996 / 0.87977 = 1.796 > 1`).
  - `F_extrapolated = 101.37869 N +/- 0.25583 N` (`+/- 0.252%`; `peak_to_peak = 2.45973 N` / `2.426%`).
  - **Caveat on the reported order:** three rungs determine the three-parameter model `F(delta) = F_0 + C * delta^p` exactly, so `p = 2.15022` is an interpolated order consistent with the nominal central-difference order `2`, not an over-determined convergence proof. The independent evidence that the aligned ladder is a smooth function of `delta` rather than a step function is the monotonic growth of the successive differences (`0.87977 -> 1.57996 N`, ratio `1.796 > 1`) together with mesh counts unchanged at every rung; a fourth aligned rung (`4h = 1.0e-3 m`) is geometrically impossible here because it equals the air gap.
- Invariant check: **HOLDS** (`36792n / 72812e` on both `+delta` and `-delta` across all 3 rungs, identical to unshifted base; `gap_cells_plus + gap_cells_minus = 3+5 = 2+6 = 1+7 = 8 = 2 * (1e-3 / 2.5e-4)`).

### Gate 3 Recompute (Unchanged `<= 2%` Gate), §4 Re-Basing, and Elmer Triangulation

- **Gate 3 (`F_maxwell` fixed at `h = 5.0e-4 m`: `102.28330 N`, `<= 2%` gate unchanged):**
  - **Basis (i) — aligned single step (`delta = h = 5.00e-04 m` at `h = 5.00e-04 m`):** `F_coenergy = 102.81233 N` vs `102.28330 N` -> **`0.5146%`** (**PASS**).
  - **Basis (ii) — aligned ladder extrapolation (`(h, 2h, 3h)` at `h = 2.50e-04 m`):** `F_extrapolated = 101.37869 +/- 0.25583 N` vs `102.28330 N` -> **`0.8844%`** (**PASS**).
  - Both bases agree cleanly on **PASS** (unambiguous).
- **Gate 4 (§4 Force vs Mesh) re-basing to cell-aligned `delta = h`:**
  - `h = 1.00e-03 m` (`4400n / 8532e`): `F_maxwell = 104.21125 N`; coenergy **REFUSED** (`E_FORCE_UNSUPPORTED_CASE: gate3_magnet_steel_plate: target 'magnet' has axial clearance 0.001 m to an adjacent body, which cannot accommodate even k=1 cell-aligned shift h=0.001 m (1.000e-03 m; no integer k*h fits); nothing was solved.`) — no fabricated value printed.
  - `h = 5.00e-04 m` (`12126n / 23808e`): `F_maxwell = 102.28330 N` (`-1.850%`), `F_coenergy(delta = 5.00e-04 m) = 102.81233 N`.
  - `h = 2.50e-04 m` (`36792n / 72812e`): `F_maxwell = 101.48761 N` (`-0.778%`; Richardson `p = 1.277`, `F_extrap = 100.92843 N`, `est_error = 0.554%`), `F_coenergy(delta = 2.50e-04 m) = 101.63452 N` (`-1.146%`).
  - Coenergy Richardson `p`: `n/a (two aligned rungs; three are required)`. M5.4's fixed non-aligned `delta = 2.50e-04 m` values at `h = 1.0e-3 / 5.0e-4 m` (`102.57859 / 101.98703 N`) are retired as remeshed-geometry values.
- **Elmer triangulation (`F_elmer = 101.72932 N`, live Gate 6 run `18315n / 18040e`):**
  - vs Maxwell (`h = 5.0e-4 m`, `102.28330 N`): `0.545%`
  - vs Coenergy (`h = 5.0e-4 m, delta = h`, `102.81233 N`): `1.065%`
  - vs Coenergy (`h = 2.5e-4 m, delta = h`, `101.63452 N`): `0.093%`
  - vs Coenergy (`h = 2.5e-4 m, aligned extrap`, `101.37869 N`): `0.345%` (whereas retired M5.4b extrapolation `100.20786 N` sat `1.50%` below Elmer)

### Governance, Version, and Exact Scope of Changes

- **No gate was relaxed:** `2%` Maxwell-vs-coenergy, `1%` loop force, `0.5%` Gate 2 Nagaoka inductance, and `5%` Elmer gates remain unchanged.
- **`engine_version` bump:** moved from `+m5.4` to **`+m5.4c`** in `src/magwave/fem2d/engine.py` (and `tests/test_fem2d.py`).
- **Moved / added symbols & files:**
  - `src/magwave/fem2d/force.py` & `src/magwave/fem2d/__init__.py`: added `E_FORCE_DELTA_NOT_ALIGNED`, `DeltaStep`, `DeltaStudyResult`, `coenergy_delta_study`, cell-alignment check `_aligned_cell_multiple` (runs before axial clearance in `coenergy_force`), gap-cell split invariant check, and deleted the `0.25 * max_shift` non-aligned clamp.
  - `tools/force_report.py`: replaced §3 sub-cell sweep with Study A (`h = 5.0e-4`) and Study B (`h = 2.5e-4`), added node/element proof table, two-basis Gate 3 verdicts, Elmer triangulation, and §4 aligned `delta = h` column with refusal at `h = 1.0e-3 m`.
  - `tests/test_m54_forces.py`: replaced step-sensitivity test with aligned-shift refusal, Study A/B invariant, Gate 3 agreement, and synthetic ladder order tests.
  - `docs/adr/0009-tier1-force-torque.md` & `CHANGELOG.md`: documented M5.4c findings and retired M5.4b.
