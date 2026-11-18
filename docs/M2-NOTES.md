# M2 notes — Tier 0 Magpylib engine

Engine: `magpylib` (`magwave.engines.magpylib_engine.MagpylibEngine`), engine version **magpylib
6.3.0** (read from `magpylib.__version__` at runtime and reported in every result, CSV header and
JSON payload), rung **`second-code`**: an independent published implementation of the same
closed-form source fields, used to cross-check `magwave.analytic` and as the CLI's field sampler.

## Cross-check: `python -m magwave bench --engine magpylib`

Same corpus (`src/magwave/benchmarks/cases.json`), same 15 cases, same expected values and the
same research-owned tolerances as `magwave bench`; only the producer of the number changes
(Magpylib instead of `benchmarks/checks.py`). Result: **10 cross-checked, 5 skipped, 10/10 PASS.**

| case | Magpylib source | rel. err (measured) | tolerance |
|---|---|---|---|
| dipole_axis_Bz | `misc.Dipole` | 2.65e-16 | 1e-9 |
| dipole_offaxis_Bx | `misc.Dipole` | 3.68e-16 | 1e-9 |
| dipole_offaxis_Bz | `misc.Dipole` | 4.32e-16 | 1e-9 |
| loop_axis_Bz | `current.Circle` | 1.69e-16 | 1e-10 |
| loop_offaxis_Bz | `current.Circle` | 2.07e-16 | 1e-10 |
| loop_offaxis_Brho | `current.Circle` | 1.23e-15 | 1e-10 |
| cuboid_axis_Bz | `magnet.Cuboid` | 1.88e-07 | 1e-6 |
| cuboid_transverse_Bx | `magnet.Cuboid` | 4.10e-07 | 1e-6 |
| cuboid_interior_Bz | `magnet.Cuboid` | 1.59e-07 | 1e-5 |
| cylinder_axis_Bz | `magnet.Cylinder` | 1.56e-16 | 1e-12 |

Each expected value is compared against the research value in `cases.json` (the same number the
analytic path is judged against), with the harness's single pass rule
`|computed - expected| <= tol * |expected|`. The cuboid residuals (1.6e-7…4.1e-7) are the same
size as the analytic path's against the same research values; they are the research reference's
own precision, not a Magpylib defect, and they sit inside the research tolerance.

### Skipped, and why (produced by `MagpylibEngine.validate()`, not by a list in the harness)

| case | code | reason |
|---|---|---|
| halbach_bore_B | `E_UNSUPPORTED_GEOMETRY` | The case is an ideal, infinitely long, continuously rotating Halbach cylinder. Magpylib has no exact source for it; a segmented finite approximation would be a different physical model, so no like-for-like check exists. |
| thin_solenoid_axis_Bz | `E_UNSUPPORTED_COIL` | Magpylib has no current-sheet source. Approximating it with a stack of `Circle` loops would be a discretised model of the analytic formula, not an independent check. |
| thick_solenoid_axis_Bz | `E_UNSUPPORTED_COIL` | Magpylib has no volume-current source; same argument as above (a loop lattice would be a fake of the formula under test). |
| force_dipole_coaxial_far | `E_UNSUPPORTED_QUANTITY` | The probe asks for force F. This engine samples B only: the Magpylib core exposes no closed-form force, only mesh-integrated `getFT` (again a discretised model), and Tier 0 force is FR-052's analytic path, not a field sample. |
| force_pointpole_g5L | `E_UNSUPPORTED_QUANTITY` | Same as above. |

Every skip still prints a row (`SKIP: <CODE>: <reason>`); the harness raises if the row count ever
differs from the number of expected values, and the CLI refuses to summarise if it differs from the
corpus length. A run with zero cross-checked cases exits 1, so an engine that skips everything
cannot report success.

## Magnet convention

`Magnet.br` is the polarisation `J = mu0*M` in tesla. It is passed straight to Magpylib's
`polarization=` argument (also J in tesla); `M = br/MU0` is derived, never required. No magnet data
is routed through `magnetization=`, which Magpylib would multiply by its own `mu_0` and so
reintroduce a 1.3e-10 relative error (fatal for the 1e-12 cylinder case). Measured cylinder error
without any correction: 1.56e-16.

## `mu_0` constant reconciliation (not a fit)

Magpylib 6 uses CODATA `magpy.mu_0 = 1.25663706127e-06` H/m. This repository fixes
`MU0 = 4e-7*pi` (`units.py`) and its tolerances were written against it. The ratio
`MU0 / magpy.mu_0 - 1 = 1.320328e-10` (read at runtime, not hard-coded) is a constant-convention
difference. After sampling, the engine multiplies the field of every source whose field is
`mu_0` × an ampere-based quantity by `MU0 / magpy.mu_0`: current loops (current in A) **and point
dipoles** (moment in A m², `B = mu_0/(4 pi) (3(m·r)r/r^5 - m/r^3)`). Magnets are not rescaled: their
field is evaluated from polarisation in tesla and involves no `mu_0`.

Measured through the engine path (`evaluate_with_engine`) with the ratio forced to 1 (before) and
as shipped (after):

| case | tolerance | before | after |
|---|---|---|---|
| loop_axis_Bz | 1e-10 | 1.320e-10 **FAIL** | 1.685e-16 PASS |
| loop_offaxis_Bz | 1e-10 | 1.320e-10 **FAIL** | 2.073e-16 PASS |
| loop_offaxis_Brho | 1e-10 | 1.320e-10 **FAIL** | 1.230e-15 PASS |
| dipole_axis_Bz | 1e-9 | 1.320e-10 PASS | 2.647e-16 PASS |
| dipole_offaxis_Bx | 1e-9 | 1.320e-10 PASS | 3.676e-16 PASS |
| dipole_offaxis_Bz | 1e-9 | 1.320e-10 PASS | 4.316e-16 PASS |

The "before" error is the same 1.320e-10 on every case and at every point — exactly the constant
ratio — and the "after" error is at machine precision. That is the signature of a unit-convention
difference, not of a fitted factor. `tests/test_magpylib_engine.py::test_mu0_reconciliation_is_the_exact_constant_ratio`
asserts the engine output equals raw Magpylib × `MU0/magpy.mu_0` bit-for-bit for a loop and a
dipole, and equals raw Magpylib for a magnet.

> Deviation from the M2 brief: the brief said dipoles take no correction because they "involve no
> `mu_0`". Magpylib's `Dipole` does use `mu_0` (moment is in A m²), as the measured 1.320e-10 shows,
> so the dipole is grouped with the current sources. The dipole cases pass either way (tolerance
> 1e-9); with the correction they agree to 3e-16.

## Sampling and export (FR-050)

`MagpylibEngine.sample(case, points) -> (n, 3)` builds one Magpylib `Collection` per case (two
sub-collections: `mu_0`-scaled sources and polarisation sources) and makes one vectorised `getB`
call for the whole point set. `python -m magwave solve CASE --format csv|json --grid ...|--points ...`
prints the samples; the CSV header and JSON payload name the units (m, T), the engine id and
version, and the rung. Values are printed with `repr(float)` (shortest round-trip), and the tests
assert the parsed export equals `engine.sample()` exactly.

**Performance:** 10,000 points on the `cuboid_axis_Bz` geometry, warm call, `time.perf_counter()`:
**14–16 ms** measured on this box (budget 100 ms;
`tests/test_magpylib_engine.py::test_ten_thousand_points_within_100_ms` prints the number on every run).

## What this Tier 0 engine cannot do

- **Force and torque**: no force model (`force_methods == ()`); F probes are rejected. Tier 0 force is
  the analytic path of FR-052 (`analytic.F_dipole_coaxial`, `F_pointpole_cylinders`), not yet wired
  into an engine.
- **Volume or sheet currents** (solenoids, FR-030/031 distributed windings): rejected, see above.
- **Halbach cylinders**: rejected, see above.
- **Permeable materials / steel**: free-space superposition only; there is no material model at all,
  so any future steel component must be rejected by `validate()` (FR-041) and sent to a FEM tier.
- **Demagnetisation / magnet interaction**: magnets are rigidly polarised; a non-null `demag_curve`
  is accepted with warning `W_DEMAG_IGNORED`, and magnets do not affect each other's polarisation.
- **Magnet shapes beyond cuboid and solid cylinder**, and **coil kinds beyond a single circular loop**.
- **Rotated sources**: the case schema has no orientation field yet, so every source is axis-aligned.
- **Line/plane/volume sampling helpers** beyond a rectilinear `--grid` and an explicit `--points` list.
