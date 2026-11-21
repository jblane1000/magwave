# ADR 0008 — Tier 1 permanent-magnet second-quadrant demagnetisation curves (`FR-020`)

- Status: accepted
- Date: 2026-11-21
- Deciders: Jason (owner)
- Amends: ADR-0007 (cross-reference only; does not modify ADR-0007) — extends the Tier 1 nonlinear Newton–Raphson solver (`magwave.fem2d`) with second-quadrant permanent-magnet demagnetisation curves and removes `W_DEMAG_IGNORED` in Tier 1 while replacing Tier 0's warning with an honest refusal (`E_UNSUPPORTED_DEMAG_CURVE`).
- Supersedes nothing

## Context

In M5 Stages 1–2 (ADR-0007), Tier 1 modelled every permanent magnet as uniformly magnetised with fixed remanence `Br` (`K_phi = Br_z / MU0` via `load_sheet`, interior reluctivity `1 / MU0`), regardless of external demagnetising fields, steel keepers, or operating temperature. Real permanent magnets operate on a nonlinear second-quadrant normal curve `B(H)` (or `H(B)`) whose remanence and coercivity shift with temperature. This stage (M5.3, PRD FR-020) implements versioned permanent-magnet demagnetisation curves, case-level operating temperature, self-consistent Newton operating-point location in Tier 1, and honest Tier 0 refusal.

All measured figures below were recorded on 2026-11-21 using `env -u PYTHONPATH .venv/bin/python` (Python 3.12.3, NumPy 2.5.5, SciPy 1.18.1, Elmer 9.0 via Docker `scimulate/elmer`).

## Decision

### 1. Sibling versioned library (`magwave.magnets/1`)

To preserve `LIBRARY_VERSION == 1` and bit-identical Tier 1 `method` strings on existing soft-magnetic benchmarks, permanent-magnet data is shipped as a sibling dataset (`src/magwave/materials/magnet_library.json`, `schema = "magwave.magnets/1"`, `version = 1`) loaded by `src/magwave/materials/magnets.py`:

- **Strict validation without repair:** `validate_demag_curve` requires at least 4 `(H, B)` points in ascending `H` ending at `H = 0` with strictly increasing `B`, raising `ValueError` naming the grade and offending point indices without sorting or deduplicating.
- **Monotone PCHIP interpolation:** `DemagCurve` fits shape-preserving `scipy.interpolate.PchipInterpolator` on `(H, B)` for `b_of_h(h)` and on `(B, H)` for `h_of_b(b)`, forbidding extrapolation beyond the tabulated domain.
- **Shipped grades:** `N42-20C` (sintered NdFeB, Arnold catalog data from `docs/research/materials-bh-data.md` §4.1) and `Y30H-1-20C` (hard ferrite C8 / Feroba3, Eclipse ferrite TDS from §4.1, including the positive ferrite temperature coefficient `alpha_hcj_per_c = +0.0027 / °C`).

### 2. Second-quadrant affine linearisation and Newton operating-point location

At each magnet element centroid flux density `b = |B_e|`, `DemagCurve.slope_and_offset(b)` evaluates the exact affine tangent to `H(B)`:

```
H(B) = s(B) * B + c(B),      s = dH/dB > 0,      c(B) = H(B) - s(B) * B  (c < 0)
```

with exact chain-rule derivatives `ds/dB = d²H/dB²` and `dc/dB = -B * ds/dB`. Inside `solve_axisymmetric_nonlinear` (`src/magwave/fem2d/nonlinear.py`), magnet elements reuse the exact same Newton–Raphson + Armijo backtracking loop as nonlinear steel:

- Element stiffness carries `s_e` as element reluctivity `nu_e`.
- The consistent tangent Jacobian (`_assemble_jacobian`, CSR matrix, zero 3-index tensors) accumulates both `d/dA [K(s) A]` and `d/dA [f_offset(c0 - c)]` via `dc/dB = -B * ds/dB`.
- At convergence, `Fem2dEngine.solve_with_report` samples `B` at each magnet body centroid and evaluates `H = DemagCurve.h_of_b(B)`, populating `ConvergenceInfo.magnet_operating_points` (`MagnetOperatingPoint`) and recording the grade, applied `temperature_c`, `Br(T)`, and centroid `(H, B)` in both `Result.method` and `ConvergenceInfo.material_model` (FR-042).
- When `magnet_regions=None` (or `demag_curve=None` on every magnet), both `solve_axisymmetric_nonlinear` and `Fem2dEngine.solve_with_report` execute the unchanged M5.2 code paths bit-identically.

### 3. Source split (`load_sheet` reference + volume offset)

Rather than replacing the surface-current sheet source, the formulation keeps the exact uniform-`Br(T)` sheet load as the reference and adds only the element-wise intercept deviation:

```
f = load_sheet(mesh, r_out, (z1, z2), sign * Br(T) / MU0) + f_offset(sign * (c0 - c_e))
c0 = -Br(T) / MU0
```

where `f_offset` is assembled by `load_magnetization_offset(mesh, mask, offsets)` (`src/magwave/fem2d/assembly.py`) with centroid quadrature weight `w_e = A_e * r_c` matching `element_geometric_stiffness` (`nodal_i = offset_e * w_e * (b_i + 1/(3 r_c))`). By discrete Stokes equivalence on the forced-breakpoint mesh, when a curve is exactly linear with slope `s = 1 / MU0` and intercept `c = -Br / MU0`, `c0 - c` vanishes identically (`correction == 0.0` everywhere) and the nonlinear solve reduces exactly to the uniform-`Br` answer.

### 4. Case-level temperature (`Case.temperature_c`)

`Case.temperature_c: float | None = None` (`None` ≡ 20 °C; omitted from `Case.to_dict()` when `None` or `20.0` for lossless round-trip of existing files; finite and `>= -273.15 °C`) scales the demag curve reversibly via `DemagCurve.at_temperature(T)`:

```
(h_i, b_i) -> (k_hcj * h_i, k_br * b_i),
k_br  = 1 + alpha_br_per_c  * (T - t_ref_c),
k_hcj = 1 + alpha_hcj_per_c * (T - t_ref_c).
```

### 5. Honest capability boundary (Tier 0 vs Tier 1)

- **Tier 1 (`fem2d`):** adds `"permanent_magnet_demag_curve"` to `Capabilities.materials`, resolves `Magnet.demag_curve` at `case.temperature_c`, removes `W_DEMAG_IGNORED`, and rejects unknown grade names with `E_UNSUPPORTED_MATERIAL` naming the requested grade and shipped grades.
- **Tier 0 (`magpylib`):** replaces `W_DEMAG_IGNORED` with `E_UNSUPPORTED_DEMAG_CURVE` error whenever any `Magnet` carries `demag_curve != None`, explicitly stating that Tier 0 would treat it as rigidly polarised with no demagnetising field and no operating point, and directing the user to `Tier 1 (fem2d)`. Uniform-`Br` magnets (`demag_curve=None`) remain unaffected.

## Known Limitations

1. **Piecewise-cubic smoothness at knots:** `PchipInterpolator` is `C^1`, so its second derivative `ds/dB` (and hence the Newton tangent Jacobian) has jump discontinuities at tabulated curve knots; near a knot or knee, Newton may require an extra Armijo backtracking halving, and trial steps exceeding `[b_min, b_max]` are rejected cleanly by Armijo halving.
2. **Reversible temperature scaling keeps `HcB / HcJ` constant:** scaling `H` uniformly by `k_hcj` and `B` by `k_br` tracks `Br(T)` and `HcJ(T)` directly and keeps `HcB / HcJ` constant (appropriate for square-loop grades), rather than shifting `HcB(T)` independently via `alpha_Br` along the recoil slope.
3. **Reconstructed interior points:** interior second-quadrant `(H, B)` points of `N42-20C` and `Y30H-1-20C` are reconstructed from the four published datasheet anchors (`Br`, `HcB`, `HcJ`, `(BH)max`) assuming a square intrinsic loop `B = J + MU0 * H`, not digitised from printed vendor curve plots.
4. **Recoil permeability `mu_rec != 1`:** real `N42-20C` has `mu_rec = Br / (MU0 * HcB) ≈ 1.2168 != 1`, so its field legitimately differs from the `mu_rec = 1` uniformly magnetised cylinder closed form (`2.517 %` at `z = 0.5 L` on the `R = 10 mm, L = 20 mm` cylinder).

## Measured Verification Numbers

### Gate 2 — Datasheet fidelity (`tests/test_m5_3_pm_demag.py`)

| Grade | Quantity | Datasheet | Computed | Rel. err | Gate (`<= 2 %`) |
|---|---|---:|---:|---:|---|
| `N42-20C` | `Br` (T) | `1.315` | `1.315` | `0.0000 %` | PASS |
| `N42-20C` | `HcB` (A/m) | `860000` | `860000` | `0.0000 %` | PASS |
| `N42-20C` | `(BH)max` (J/m³) | `334000` | `334105` | `0.0315 %` | PASS |
| `Y30H-1-20C` | `Br` (T) | `0.39` | `0.39` | `0.0000 %` | PASS |
| `Y30H-1-20C` | `HcB` (A/m) | `252000` | `252000` | `0.0000 %` | PASS |
| `Y30H-1-20C` | `(BH)max` (J/m³) | `28000` | `28016.1` | `0.0576 %` | PASS |

### Gate 3 — Analytic cylinder (`mu_rec = 1`) and uniform-`Br` reduction

**Validation of record:** the Tier-1 permanent-magnet validation is `tests/test_m5_3_pm_demag.py::test_g3_analytic_cylinder_and_exact_uniform_br_reduction` — a synthetic linear `1/MU0` curve against the analytic cylinder closed form within 1 %, reducing to the uniform-`Br` path within 1e-9; the shipped `examples/magnet-n42-*.json` values are regression snapshots of this engine at the stated mesh, not physical references. — enforced-by: tests/test_m5_3_pm_demag.py::test_g3_analytic_cylinder_and_exact_uniform_br_reduction

Cylinder `R = 10 mm`, `L = 20 mm`, `Br = 1.315 T`, synthetic linear `1/MU0` curve, `far_field_factor = 25`:

| `h` (m) | `z / L` | `z` (m) | Computed `Bz` (T) | Analytic `Bz` (T) | Rel. err (`<= 1 %`) |
|---:|---:|---:|---:|---:|---:|
| `1.0e-3` | `0.5` | `0.0100` | `5.876857e-01` | `5.880859e-01` | `0.0680 %` |
| `1.0e-3` | `1.0` | `0.0200` | `1.580633e-01` | `1.588366e-01` | `0.4868 %` |
| `1.0e-3` | `2.0` | `0.0400` | `2.103509e-02` | `2.097253e-02` | `0.2983 %` |
| `1.0e-3` | `5.0` | `0.1000` | `1.326803e-03` | `1.321228e-03` | `0.4220 %` |
| `5.0e-4` | `0.5` | `0.0100` | `5.876698e-01` | `5.880859e-01` | `0.0707 %` |
| `5.0e-4` | `1.0` | `0.0200` | `1.586566e-01` | `1.588366e-01` | `0.1133 %` |
| `5.0e-4` | `2.0` | `0.0400` | `2.105269e-02` | `2.097253e-02` | `0.3822 %` |
| `5.0e-4` | `5.0` | `0.1000` | `1.326627e-03` | `1.321228e-03` | `0.4086 %` |

- Offset vector check: `np.all(correction == 0.0)` is `True`.
- Uniform-`Br` reduction (`demag` vs `demag_curve=None` at `h = 5e-4 m`): max relative difference `0.000e+00` (`<= 1e-9`).

### Gate 4 — Operating point motion (`N42-20C`, `h = 5e-4 m`, probe `(0, 0, 0)`)

| State | `H` (A/m) | `B` (T) | Notes |
|---|---:|---:|---|
| Air | `-305484.2` | `0.924922` | Baseline free-space operating point |
| Dual `steel-team20` keeper | `-196209.0` | `1.065557` | Keeper raises load line (`+0.140636 T`) |
| Opposing solenoid (`Hz ≈ -3.0e5 A/m`) | `-592834.2` | `0.527908` | Uniform-`Br` gives `0.553495 T` (`4.623 %` away) |

### Gate 5 — Temperature scaling (`20 °C -> 100 °C`)

- `N42-20C`: `Br(20) = 1.31500 T -> Br(100) = 1.18876 T` (arithmetic `1.18876 T`, err `0.000e+00`); air centre `Bz` drops `0.924922 -> 0.692040 T`.
- `Y30H-1-20C`: `Br` drops `0.39000 -> 0.32760 T`, while `HcJ` rises `262000 -> 318592 A/m` (positive `alpha_hcj_per_c` sign flip); air centre `Bz` drops `0.271396 -> 0.249895 T`.

### Gate 6 — Elmer second-code cross-check (`tools/elmer_permeable.py --case magnet-keeper`)

Linear-vs-linear on the magnet (`Br = 1.315 T`, `HcB = 860000 A/m`, `mu_rec = 1.216795`, Elmer `Magnetization 2 = 860000 A/m`) with nonlinear `steel-team20` keepers in both codes (`1.0 m` air box):

- Elmer `MagnetoDynamics2D`: `22725` nodes / `22400` quads, `5` iters, `Bz(0,0,0) = 1.027957 T` (`1.80 s`).
- MagWave `Fem2dEngine` (`h = 5e-4 m`): `7560` nodes / `14740` tris, `4` iters, `Bz(0,0,0) = 1.027020 T` (`0.187 s`).
- Relative difference: **`0.0912 %`** (`<= 3.00 %` gate, PASS).

## Consequences

- `magwave.materials` adds `magnet_library.json` (`magwave.magnets/1`) and `magnets.py` without touching `library.json` or `LIBRARY_VERSION`, keeping all 27 frozen corpus cases and pre-M5.3 examples bit-identical.
- `Fem2dEngine` solves second-quadrant demagnetisation self-consistently in the shared Newton loop and reports `(H, B)` operating points per magnet body; `MagpylibEngine` refuses demag-curve magnets cleanly with `E_UNSUPPORTED_DEMAG_CURVE`.
