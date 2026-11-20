# M5 Stage 1 report — axisymmetric Tier 1 solver (linear media)

- Branch: `m5.1-tier1-fem2d`, from `main` @ `f4e7b36`
- PR: `M5.1: axisymmetric Tier 1 solver (linear media)` — open, **not merged**
- ADR: `docs/adr/0007-tier1-formulation.md`
- Brief: `docs/briefs/m5-stage1-brief.md` (owner's), expanded work order `docs/briefs/m5-stage1-agy-brief.md`

**Engine that produced the implementation code: 1 (one) `agy` invocation** —
`agy-work --model gemini-pro-orchestrator-g1-pro --print-timeout 0 -p '<pointer to the work order>'`,
wall clock 22 min, agy 1.5.13. The brief, the verification, the acceptance runs and this report are
Cody's. Cody wrote exactly one piece of code in the branch: the 20-line change to
`tests/test_fem2d_validation.py::test_mesh_independence_study` that records the table as a JUnit
property (see *Deviations*, item 2).

## 1. Formulation as implemented

Axisymmetric magnetostatics with the azimuthal vector potential `A = A_phi(r,z) e_phi`,
`nu = 1/mu_0` everywhere (linear media only):

```
curl( nu curl A ) = J_phi            measure r dr dz
B_r = -dA/dz    B_z = (1/r) d(rA)/dr = A/r + dA/dr
```

Weak form, test function `v` with `v = 0` on the Dirichlet boundary:

```
integral_Omega nu [ r A_r v_r + r A_z v_z + A_r v + A v_r + A v/r ] dr dz
    = integral_Omega J_phi v r dr dz + integral_sheet K_phi v r dl
```

This is the same bilinear form as `integral nu (curl A).(curl v) r dr dz`, written out because the
`A v/r` term is what makes the axis well behaved.

**Element matrix (P1 triangles, one-point centroid quadrature — the choice the brief allowed):**
with area `A_e`, centroid radius `r_c`, `b_i = dN_i/dr`, `c_i = dN_i/dz`, `N_i(r_c) = 1/3`:

```
K^e_ij = nu_e * A_e * [ r_c (b_i b_j + c_i c_j) + (b_i + b_j)/3 + 1/(9 r_c) ]
```

`src/magwave/fem2d/assembly.py:72-76`. **The owner's brief stated this formula wrongly** as
`r_c(b_i b_j + c_i c_j) + (r_c/3)(b_i + b_j) + 1/9`; the cross term carries no `r` in the integrand,
and the `A v/r` term is `(1/3)(1/3)/r_c`. The engine implemented the dimensionally consistent form and
recorded the correction in ADR-0007 §2. Cody re-derived it independently and checked it against an
exact algebraic identity that the wrong coefficients cannot satisfy:

| mesh | `1^T K 1` | `sum_e nu A_e / r_c` (exact for `A = 1`, since `sum_i b_i = 0`) | rel. |
|---|---|---|---|
| h = 0.02 | 6.104669265116090e+05 | 6.104669265116092e+05 | 3.8e-16 |
| h = 0.01 | 7.206530363225910e+05 | 7.206530363225909e+05 | 1.6e-16 |
| h = 0.005 | 8.309377386800904e+05 | 8.309377386800890e+05 | 1.7e-15 |

**Loads, integrated exactly:**
- volume current, `f^e_i = J_e A_e (2 r_i + r_j + r_k)/12` (`assembly.py:110`);
- surface current sheet on a mesh line at `r = R`, `f_a += K R l / 2`, `f_b += K R l / 2` per segment
  (`assembly.py:159`).

**Sources.** Coil: uniform `J_phi = turns*current/((R2-R1)*L)`, fill factor **1.0** — `magwave.case/1`
has no fill-factor field, no schema change was made, and `benchmarks/checks.py:110` uses 1.0, so the
comparison is like-for-like. `inner == outer` is a sheet with `K_phi = turns*current/L`. Magnet:
uniform axial `M` has no volume current, and its bound surface current `K = M x n` is `K_phi = M` on
the cylindrical face only (zero on the end discs) — exactly the sheet path, which is why
`cylinder_Bz_axis` and `thin_solenoid_Bz` agree analytically and share a code path.

**Axis handling.** `A_phi = 0` is imposed on every node at `r = 0` (regularity: `A = alpha r + O(r^3)`),
so no gauge fixing is needed and no `1/r` is ever divided by zero. Field recovery uses the exact limit
`B_r = 0`, `B_z = 2 dA/dr` for `r < 1e-12` (`field.py:42-44`, `field.py:91-95`); a sample point on the
axis therefore returns a finite value, and the 11-point on-axis gate below exercises that branch.

## 2. Mesh

Built-in graded tensor-product mesher, no external mesher (`fem2d/mesh.py`):

- forced **breakpoints** at every source radius, sheet line and axial end, so no element is split by a
  region boundary and every sheet sits exactly on a mesh line;
- uniform subdivision `ceil(L/h)` inside a refinement box containing all sources (plus the probe
  stations within `L_max` of them), geometric growth `1.2` outside it out to the air boundary;
- alternating cell diagonal along `z` to keep axial mirror symmetry;
- `max_nodes = 400_000` guard that raises rather than silently coarsening;
- statistics per solve (nodes, elements, min/max edge, `h`) reported in `SolveReport` (FR-042).

Domain: `pad = far_field_factor * L_max` (default 5.0, the brief's minimum) beyond the source in every
direction; Dirichlet `A = 0` on that boundary. Solve: `scipy.sparse.linalg.spsolve` (SuperLU) with
symmetric elimination of constrained DOFs, `iterations = 1`, and a **computed** relative residual.

## 3. The stage gate — measured agreement

Validation geometry: thick solenoid `R 10–20 mm`, `L 100 mm`, 1000 turns × 1 A
(`J = 1.0e6 A/m^2`); magnetised cylinder `R 12.5 mm`, `L 50 mm`, `Br = 1.2 T`. Eleven on-axis stations
each; three solenoid stations lie outside the winding on purpose.

Mesh-independence study (two levels, errors against closed form). This is the block recorded as a
JUnit property in `/tmp/cody-m51.xml` and printed by `pytest -s`:

```
case             |        h |   nodes | elements |   min edge | max rel err | solve wall time | result
------------------------------------------------------------------------------------------------------
solenoid         | 1.0000e-03 |    8673 |    16896 | 1.0000e-03 |  3.1142e-03 |         0.0401s | PASS
solenoid         | 5.0000e-04 |   22119 |    43488 | 5.0000e-04 |  6.0207e-04 |         0.0834s | PASS
cylinder         | 1.0000e-03 |    4902 |     9472 | 9.6154e-04 |  4.3501e-03 |         0.0145s | PASS
cylinder         | 5.0000e-04 |   11610 |    22684 | 5.0000e-04 |  1.5396e-03 |         0.0362s | PASS
```

| case | max rel. error, h = 1 mm | max rel. error, h = 0.5 mm | gate | ratio | observed order |
|---|---|---|---|---|---|
| solenoid | 3.1142e-03 | 6.0207e-04 | ≤ 1e-02 | 5.17 | 2.37 |
| cylinder | 4.3501e-03 | 1.5396e-03 | ≤ 1e-02 | 2.83 | 1.50 |

**1 % gate met at both levels, for both cases, with margin; the error decreases under refinement for
both.** The cylinder's order is 1.5 rather than ~2 because the sheet terminates in a right-angled
corner at the rim, where `B` is singular — the error there is dominated by the corner, not by the
element order.

Per-point relative errors, solenoid (z in metres):

| z | h = 1e-3 | h = 5e-4 |
|---|---|---|
| -0.0600 | 3.114174e-03 | 6.020736e-04 |
| -0.0480 | 1.344002e-03 | 4.695119e-05 |
| -0.0360 | 1.016768e-03 | 4.086907e-05 |
| -0.0240 | 5.269303e-04 | 1.738684e-04 |
| -0.0120 | 4.303810e-04 | 1.920395e-04 |
| ±0.0000 | 4.144362e-04 | 1.925347e-04 |
| +0.0120 | 4.303810e-04 | 1.920395e-04 |
| +0.0240 | 5.269303e-04 | 1.738684e-04 |
| +0.0360 | 1.016768e-03 | 4.086907e-05 |
| +0.0480 | 1.344002e-03 | 4.695119e-05 |
| +0.0600 | 3.114174e-03 | 6.020736e-04 |

Per-point relative errors, magnetised cylinder:

| z | h = 1e-3 | h = 5e-4 |
|---|---|---|
| -0.0400 | 2.837708e-03 | 4.066748e-04 |
| -0.0320 | 4.350070e-03 | 1.539557e-03 |
| -0.0240 | 3.335657e-04 | 6.925339e-04 |
| -0.0160 | 4.196925e-05 | 6.725722e-04 |
| -0.0080 | 7.302503e-04 | 8.465833e-04 |
| ±0.0000 | 8.495805e-04 | 8.687040e-04 |
| +0.0080 | 7.302503e-04 | 8.465833e-04 |
| +0.0160 | 4.196925e-05 | 6.725722e-04 |
| +0.0240 | 3.335657e-04 | 6.925339e-04 |
| +0.0320 | 4.350070e-03 | 1.539557e-03 |
| +0.0400 | 2.837708e-03 | 4.066748e-04 |

Some individual cylinder stations are *worse* at the finer level (e.g. `z = ±0.016`: 4.2e-05 → 6.7e-04)
by cancellation at the coarse level; the quantity the gate uses, and the one that must decrease, is the
maximum over the 11 stations, and it does.

Absolute values at the centre station (h = 0.5 mm), for sanity rather than agreement:

| case | FEM | closed form | rel. |
|---|---|---|---|
| solenoid `z = 0` | 1.202025932e-02 T | 1.202257408e-02 T | 1.93e-04 |
| cylinder `z = 0` | 1.072380238e+00 T | 1.073312629e+00 T | 8.69e-04 |

The cylinder centre value is `0.8937 * Br`, i.e. the expected demagnetising factor for `L = 4R`.

## 4. Independent checks Cody ran (outside the test suite)

1. **Second discretisation of the same physics.** The winding and the magnet were re-modelled as
   stacks of rings (solenoid: 40 radial × 200 axial slices; magnet: 400 axial slices with
   `I = M dz`), summed on-axis from `mu_0 I a^2 / (2 (a^2 + z^2)^{3/2})` in a throwaway script. Against
   the closed forms the ring stacks agree to **7.7e-05** (solenoid) and **7.5e-06** (magnet), so the
   reference values the gate uses are sound, and the FEM's 3.1e-03 / 4.4e-03 is FEM error, not
   reference error.
2. **Cross-process determinism (FR-043).** Two separate interpreter processes solving the same case
   produced **bit-identical** `B_z` (raw `float.hex()` per station diffed clean) and identical
   residuals. The in-suite test only proves it within one process; this rules out BLAS/LU
   nondeterminism.
3. **The ADR's far-field table.** Re-ran `far_field_factor` 2 / 5 / 10 at h = 0.5 mm; every row of
   ADR-0007 §3 reproduced to the digit (solenoid 4.3056e-03 → 6.0207e-04 → 3.2843e-04; cylinder
   1.8016e-02 → 1.5396e-03 → 9.7648e-04). The 2× column confirms the 5× choice is load-bearing: at 2×
   the cylinder **fails** the 1 % gate (1.80e-02).
4. **The element matrix identity** in §1 above.
5. **Residuals are real numbers, not constants:** 1.7e-13, 4.1e-14, 2.2e-14 on the corpus default
   meshes, 2.3e-13 on the solenoid validation mesh.

## 5. Frozen corpus: registered, honestly skipped, and measured

`fem2d` is registered as an engine (`capabilities()` → `engine_id="fem2d"`, `rung="Tier 1"`,
cylinder magnets + solenoid coils + `B` only, version `<pkg>+m5.1`) and in `ENGINES`/`_make_engine`.

The harness already expressed "a second engine on the same expected values" (M2's
`run_engine_benchmarks`), so no change was needed there. What *was* needed is honesty about accuracy:
every axisymmetric case in the 27-case corpus carries a research-owned tolerance from `1e-12` to
`1e-4`, i.e. 100× to 10^10× tighter than the 1 % this stage certifies. `validate()` therefore rejects
those cases with `E_ACCURACY_UNREACHABLE`, quoting both numbers, so the **honest answer set on the
frozen corpus is empty**:

```
env -u PYTHONPATH .venv/bin/python -m magwave bench --engine fem2d
0/0 PASS
0 cross-checked, 27 skipped        (exit 1: an all-skipped run is never a pass)
```

To put the real numbers in the bench output anyway, without touching one research-owned tolerance,
`bench --engine` gained an additive `--report-accuracy` flag (`harness.py:measure_engine_accuracy`,
`format_accuracy_table`; no existing function or dataclass changed). It solves the nine cases the engine
*can* model, judges them against the engine's certified 1 %, and prints both tolerances:

| case | computed | expected | rel.err | case tol | certified | within |
|---|---|---|---|---|---|---|
| thin_solenoid_axis_Bz | 1.122933480e-04 | 1.123970357e-04 | 9.23e-04 | 1e-12 | 1e-02 | PASS |
| thick_solenoid_axis_Bz | 8.966147012e-01 | 8.966736445e-01 | 6.57e-05 | 1e-04 | 1e-02 | PASS |
| cylinder_axis_Bz | 7.097791407e-01 | 7.108612701e-01 | 1.52e-03 | 1e-12 | 1e-02 | PASS |
| thin_solenoid_offcentre_Bz | 9.920391637e-05 | 9.922249326e-05 | 1.87e-04 | 1e-08 | 1e-02 | PASS |
| thin_solenoid_outside_Bz | 1.711300447e-05 | 1.718287150e-05 | 4.07e-03 | 1e-08 | 1e-02 | PASS |
| thick_solenoid_outside_Bz | 2.482285132e-01 | 2.494577437e-01 | 4.93e-03 | 1e-04 | 1e-02 | PASS |
| cylinder_axis_outside_Bz | 6.454492043e-02 | 6.508055644e-02 | 8.23e-03 | 1e-08 | 1e-02 | PASS |
| cylinder_offaxis_Brho | 4.590685536e-02 | 4.596499746e-02 | 1.26e-03 | 1e-06 | 1e-02 | PASS |
| cylinder_offaxis_Bz | 7.351547768e-01 | 7.369529460e-01 | 2.44e-03 | 1e-06 | 1e-02 | PASS |

`measured: 9 case(s), worst rel.err 8.23e-03, certified 1.00e-02` (exit 0).

The other 18 are skipped for a real capability gap — `E_UNSUPPORTED_SOURCE_TYPE` (dipoles),
`E_UNSUPPORTED_GEOMETRY` (cuboid/Halbach magnets, loop coils), `E_UNSUPPORTED_QUANTITY` (force).
`magwave bench --engine magpylib` is unchanged: `17/17 PASS`, 17 cross-checked, 10 skipped, exit 0.

## 6. Capability gate (FR-041)

A case whose `manifest` has a body with role `permeable_solid` is refused before anything is meshed:

```
PROBLEM: E_UNSUPPORTED_MATERIAL | error | solenoid_with_steel_yoke: body #0 (yoke) has role
'permeable_solid' and requires a nonlinear B–H material law; this solver is linear-mu only.
Permeable materials arrive in Tier 1 stage 2 (M5.2). Nothing was solved.
solve raised EngineError with codes: ['E_UNSUPPORTED_MATERIAL']
```

Any other non-`air`/`symmetry_plane` role is refused too (`E_UNSUPPORTED_MANIFEST`): CAD geometry is
not meshed by this solver yet, and silently ignoring a body would be a silent drop.

## 7. Acceptance commands — real output

| # | command | exit | output tail |
|---|---|---|---|
| 1 | `env -u PYTHONPATH .venv/bin/python -m pytest --junit-xml=/tmp/cody-m51.xml -q` | 0 | `203 tests, 0 failures, 0 errors, 0 skipped` in 17.9 s (baseline was 191; +12 new) |
| 2 | `env -u PYTHONPATH .venv/bin/python -m magwave bench` | 0 | `27/27 PASS` |
| 3 | `.venv/bin/ruff check --no-cache .` | 0 | `All checks passed!` |
| 4 | `.venv/bin/ruff format --check .` | 0 | `50 files already formatted` |
| 5 | `env -u PYTHONPATH .venv/bin/python tools/check_licence_boundary.py` | 0 | `licence boundary OK (no GPL linkage)` |

Supplementary (not in the list, run because they are this stage's point):

| command | exit | output |
|---|---|---|
| `pytest -q tests/test_fem2d_validation.py -s` | 0 | table in §3 |
| `magwave bench --engine fem2d` | 1 | `0/0 PASS`, `0 cross-checked, 27 skipped` (by design) |
| `magwave bench --engine fem2d --report-accuracy` | 0 | table in §5, worst 8.23e-03 |
| `magwave bench --engine magpylib` | 0 | `17/17 PASS` |

## 8. Deviations from the brief, each with a reason

1. **Element matrix as implemented differs from the brief's formula** (§1). The brief's coefficients
   were wrong; the engine used the consistent ones and documented the correction. Verified
   independently — this is a correction, not a deviation that needs accepting on trust.
2. **The numbers do not land in `/tmp/cody-m51.xml` through captured stdout.** The brief asserted that
   `--junit-xml` records each test's `print()` output; pytest only writes `<system-out>` for failing
   tests. `test_mesh_independence_study` now also calls `record_property("mesh_independence_table", …)`,
   which *is* written for passing tests — the table is inside the acceptance artifact (1823 chars,
   verified by parsing the XML). This is the one code edit Cody made in the branch (20 lines, test
   only), because it is a repair of the brief's own mistake rather than implementation.
3. **Bench registration: no additive harness capability was required.** The engine path already
   evaluated a corpus through any `Engine`. The *accuracy* gap, though, had no expression, so
   `--report-accuracy` was added additively (new flag, new dataclass, new two functions; no existing
   signature or behaviour changed, no research-owned tolerance touched).
4. **Fill factor 1.0**, since `magwave.case/1` has no fill-factor field and the schema was not to be
   changed; `checks.py` uses 1.0, so the two paths are comparable. Stated in the engine docstring.
5. **`far_field_factor` default 5.0** is the brief's minimum, not a larger pad: ADR-0007 §3 shows 5×
   already gives 6e-04 (solenoid) and 1.5e-03 (cylinder), comfortably inside the gate, and 10× buys
   only ~2× on the solver's own residual for +7 % nodes.
6. **`solve()` also raises `E_PROBE_OUTSIDE_DOMAIN`** from mesh construction when a probe falls outside
   the domain (validate() cannot know the domain without meshing). Same contract — `EngineError` with
   a `Problem` and an actionable message ("increase far_field_factor").

## 9. Defects found during verification — reported, not fixed

**D1. A raw `ValueError` escapes `Engine.solve()` when the mesh-size guard trips.**
`src/magwave/fem2d/engine.py:346` calls `rectangular_mesh(...)`, whose `max_nodes` guard raises
`ValueError` at `src/magwave/fem2d/mesh.py:296`. Nothing catches it, so it leaves `solve()` through
`_build_mesh_and_load` → `solve_with_report` → `solve`. `engines/base.py:10-13` states the contract:
`solve()` raises `EngineError` carrying `Problem`s for a case it cannot solve. Reproduction
(metre-scale corpus geometry, `R 0.5–1.5 m`, `L 2 m`, refined to 2 mm):

```
Fem2dEngine(element_size=2e-3).solve(thick_solenoid_axis_Bz case)
  builtins.ValueError -> mesh would require 855071 nodes (791 x 1081), exceeding max_nodes=400000
```

No case name, no remedy, wrong exception type for the protocol boundary. The same class of failure is
already handled properly two lines away for the probe-outside-domain case, so the fix is presumably
`try/except ValueError -> EngineError(Problem("E_MESH_TOO_LARGE", …))` naming the case, the node count
and the remedy. Left unfixed: it is a design decision (raise vs auto-coarsen) and Jason decides.

**Observation (not a defect).** At the default element size (`L_max/40`) the worst frozen-corpus case
lands at **8.23e-03 against a certified 1.00e-02** — only ~1.2× headroom. The certification is honest
per stage, but a user who reads "1 %" and runs a different geometry at the default mesh may exceed it.
The mitigation is explicit (`element_size`), and the number is on the record here rather than buried.

## 10. Files

Added: `src/magwave/fem2d/{__init__,mesh,assembly,solve,field,engine}.py`,
`tests/test_fem2d.py` (9 tests), `tests/test_fem2d_validation.py` (3 tests),
`docs/adr/0007-tier1-formulation.md`, `docs/briefs/m5-stage1-agy-brief.md`, this report.

Changed: `pyproject.toml` (`scipy>=1.11` core), `NOTICE` (scipy → core dependency),
`src/magwave/engines/__init__.py` (lazy `Fem2dEngine`), `src/magwave/cli.py` (`fem2d` engine,
`--report-accuracy`), `src/magwave/benchmarks/harness.py` + `__init__.py` (additive accuracy report).

Not touched, by fence: `benchmarks/cases.json` (27 frozen values), `benchmarks/checks.py`,
`analytic.py`, `cases.py`, `units.py`, and every pre-existing test file — verified with
`git diff --name-only`.
