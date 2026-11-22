# ADR 0011 — Per-case mesh density control (`mesh.element_size`), `--element-size`, and `converge` (`M5.5`)

- Status: accepted
- Date: 2026-11-22
- Deciders: Jason (owner)
- Amends: ADR-0007 §B.5 — resolves the default-mesh under-resolution follow-up without changing the engine default mesh formula or `engine_version`
- Supersedes nothing

## Context

`Fem2dEngine(element_size=...)` (`src/magwave/fem2d/engine.py`) already validated and honoured an explicit element size `h`, yet neither `magwave.case/1` (`src/magwave/cases.py`) nor the CLI (`src/magwave/cli.py`) exposed a way to set it. Every CLI run therefore fell back to `h = l_max / 40` (`min(..., min_sphere_radius / 40)` for spheres).

As documented in [ADR 0007](0007-tier1-formulation.md) §B.5, evaluating `examples/steel-cylinder-solenoid-nonlinear.json` at that implicit default (`h = 2.5e-3` m, `2 958` nodes / `5 676` elements) gave `Bz(0, 0, 0) = 0.14049244777644335 T` — `2.32 %` below Elmer `MagnetoDynamics2D` (`0.143836 T`). Refining the same geometry resolved the discrepancy to well under `0.1 %`, proving the gap was mesh density rather than physics.

Note on scope: `examples/magnet-n42-air.json` is an N42 permanent-magnet cylinder whose expected values (`Bz(0, 0, 0) = 0.9250100940070316 T`) are regression snapshots captured at the default mesh. Its `mesh` block stays absent so those regression expectations and the bit-identical default path remain intact.

All measured numbers below were recorded on 2026-11-22 in this checkout using `env -u PYTHONPATH .venv/bin/python` (Python 3.12.3, NumPy, SciPy).

## Decision

### 1. Case schema: `MeshSpec` and optional `case.mesh`

`src/magwave/cases.py` adds a frozen `MeshSpec(element_size: float)` dataclass and appends `mesh: MeshSpec | None = None` as the last field on `Case` (after `temperature_c`, leaving positional construction unchanged).

- **Lossless optional schema block.** `Case.to_dict()` emits `"mesh": {"element_size": <float>}` only when `case.mesh` is not `None`, so cases without `"mesh"` round-trip without adding any keys. -- enforced-by: tests/test_m55_mesh_density.py::test_mesh_spec_parse_roundtrip_and_absent_omission
- **Strict schema and positive-finite validation.** `MeshSpec.from_dict(doc, where)` reuses `_mapping` and `_keys(doc, where, ("element_size",))` so unknown keys are rejected by name, and rejects `bool`, non-numeric values, non-finite values (`NaN`, `inf`), and `<= 0` with `CaseError` naming the field and offending value; nothing is ever clamped, snapped, or defaulted. -- enforced-by: tests/test_m55_mesh_density.py::test_mesh_spec_refuses_invalid_values_and_unknown_keys

### 2. Why density belongs in the case rather than in an engine default

Changing `Fem2dEngine`'s implicit default from `l_max / 40` (for example to `l_max / 100`) was rejected:

1. One global ratio cannot be right for all geometries: a coarse thick coil in air converges well at `l_max / 40`, whereas a saturating steel cylinder inside a winding needs `1 mm` elements and curved spheres or thin air gaps need problem-specific resolution.
2. Raising the global default silently changes the meshes and probe values of every case that omits `mesh`, breaking the regression snapshots in `examples/magnet-n42-air.json` and `examples/magnet-n42-keeper.json` and mutating existing `fem2d` outputs.

Keeping `l_max / 40` untouched when `case.mesh is None` and `--element-size` is omitted preserves bit-identical behaviour on all existing benchmarks and regression cases, while putting `"mesh": {"element_size": ...}` in the case document makes the chosen discretisation explicit, version-controlled, and reproducible.

### 3. CLI surface (`--element-size`, `solve --engine fem2d`, `converge`) and precedence

- **Precedence (`--element-size` > `case.mesh` > `None`).** The CLI flag overrides `case.mesh.element_size`, which in turn overrides `Fem2dEngine`'s implicit `l_max / 40` fallback (`Fem2dEngine._build_mesh_and_load` honours `case.mesh.element_size` whenever `self._element_size is None`). -- enforced-by: tests/test_m55_mesh_density.py::test_cli_element_size_flag_beats_case_block
- **Case mesh block honoured without flag.** Running `magwave report --engine fem2d` or `magwave solve --engine fem2d` without `--element-size` applies `case.mesh.element_size` automatically. -- enforced-by: tests/test_m55_mesh_density.py::test_case_mesh_block_honoured_without_flag_in_report
- **Value validation before command/engine coupling.** `main()` parses and validates `--element-size` (`float`, finite, `> 0`, rejecting `bool`) before any command or engine check, printing a single named line to stderr and returning `1` with no traceback or `SystemExit`. -- enforced-by: tests/test_m55_mesh_density.py::test_cli_element_size_bad_values_exit_one_before_coupling
- **Coupling refusals for `--element-size`.** After value validation succeeds, `--element-size` is refused with `parser.error(...)` (exit `2`) whenever used with `--engine magpylib`, `bench`, `gui`, `import`, or `solve`/`report` without `--engine fem2d`. -- enforced-by: tests/test_m55_mesh_density.py::test_cli_element_size_validation_order_and_misuse_refusals
- **Direct probe evaluation in `solve --engine fem2d`.** `solve` accepts `--engine fem2d` when neither `--grid` nor `--points` (nor `--format`) is given, printing one `repr(float)` line per case probe (`probe #k (x, y, z) m  Q  comp = <repr> unit`), and refuses `--grid` or `--points` with exit `2` because `Fem2dEngine` has no point sampler. -- enforced-by: tests/test_m55_mesh_density.py::test_solve_fem2d_probe_output_and_sampling_refusal

### 4. Convergence subcommand (`magwave converge`) and honest edge cases

`magwave converge CASE.json [--engine fem2d] [--sizes H1,H2,...]` solves a strictly decreasing ladder of element sizes using the shared nonlinear material-model auto-detection helper (`_detect_fem2d_material_model`, also used by `report --engine fem2d` and `solve --engine fem2d`), prints full `repr` probe precision, computes Richardson observed order `p = log(|d_{n-1}| / |d_n|) / log(r)` and error `|d_n| / (r^p - 1)` when the last two step ratios agree to `1e-9`, and evaluates `converged: YES` only when estimated relative error `<= 1.000 %` and `|delta|` strictly falls over the last two intervals.

- **Three-rung ladder and CLI input contracts.** `converge` defaults `--engine` to `fem2d`, refuses `--engine magpylib` (exit `2`), validates strictly decreasing `--sizes` with at least two entries (exit `1`), defaults missing `--sizes` to `(4h, 2h, h)` anchored at `--element-size` or `case.mesh.element_size`, and reports `observed order p`, `estimated discretisation error`, `criterion`, and `converged`. -- enforced-by: tests/test_m55_mesh_density.py::test_converge_linear_case_three_rungs_and_case_mesh_default
- **Two-rung and non-uniform ladders never fabricate an order.** When given only two rungs (`two rungs`) or a non-uniform final step ratio (`non-uniform ladder`), `converge` reports `observed order: not estimable (<reason>)`, `estimated discretisation error: not reported (<reason>)`, and `converged: NO`. -- enforced-by: tests/test_m55_mesh_density.py::test_converge_two_rungs_and_non_uniform_not_estimable
- **Non-falling deltas emit a warning and force `converged: NO`.** Whenever `|delta|` does not strictly decrease over the last two intervals, `converge` prints `WARNING: |delta| does not fall under refinement on this ladder; this table is not convergence evidence` and forces `converged: NO`. -- enforced-by: tests/test_m55_mesh_density.py::test_converge_non_falling_deltas_warns_and_forces_no

## Measured Convergence on `examples/steel-cylinder-solenoid-nonlinear.json`

### Four-rung exploration (`--sizes 0.004,0.002,0.001,0.0005`)

Verbatim output of `env -u PYTHONPATH .venv/bin/python -m magwave.cli converge examples/steel-cylinder-solenoid-nonlinear.json --engine fem2d --sizes 0.004,0.002,0.001,0.0005`:

```text
|  h (m) | nodes | elements |     probe #0 Bz (T) | delta vs prev | rel vs finest |
| -----: | ----: | -------: | ------------------: | ------------: | ------------: |
|  0.004 |  1914 |     3640 | 0.13849749362463146 |           --- |       4.2274% |
|  0.002 |  3762 |     7252 | 0.14231387307209176 | +3.816379e-03 |       1.5883% |
|  0.001 |  8478 |    16536 | 0.14391693899251826 | +1.603066e-03 |       0.4798% |
| 0.0005 | 21995 |    43296 |  0.1446107969268211 | +6.938579e-04 |       0.0000% |

observed order p = 1.2081 (from the last three rungs, uniform ratio 2)
estimated discretisation error at h = 0.0005 m: 5.295145e-04 T (0.3662 % of the value)
criterion: estimated relative error <= 1.000 % and |delta| strictly falling over the last two intervals
converged: YES
```

Comparison of each rung against the Elmer `MagnetoDynamics2D` v9.0 reference (`0.143836 T`, `22 366` nodes):

| `h` (m) | Nodes | Elements | Computed `Bz(0,0,0)` (T) | Signed diff vs Elmer (`0.143836 T`) | Within 1 %? |
|---:|---:|---:|---:|---:|---|
| `0.004` | 1 914 | 3 640 | `0.13849749362463146` | `-3.7115 %` | NO |
| `0.002` | 3 762 | 7 252 | `0.14231387307209176` | `-1.0582 %` | NO |
| `0.001` | 8 478 | 16 536 | `0.14391693899251826` | `+0.0563 %` | YES (coarsest) |
| `0.0005` | 21 995 | 43 296 | `0.1446107969268211` | `+0.5387 %` | YES |

### Three-rung verification at the chosen size (`--sizes 0.004,0.002,0.001`)

`h = 0.001 m` is the coarsest rung within `1 %` of Elmer (`0.0563 %`; `h = 0.002 m` misses at `1.0582 %` low). Updating `examples/steel-cylinder-solenoid-nonlinear.json` to `"mesh": {"element_size": 0.001}` is therefore the honest minimum mesh density, not an over-refined target and not an under-resolved approximation.

Verbatim output of `env -u PYTHONPATH .venv/bin/python -m magwave.cli converge examples/steel-cylinder-solenoid-nonlinear.json --engine fem2d --sizes 0.004,0.002,0.001`:

```text
| h (m) | nodes | elements |     probe #0 Bz (T) | delta vs prev | rel vs finest |
| ----: | ----: | -------: | ------------------: | ------------: | ------------: |
| 0.004 |  1914 |     3640 | 0.13849749362463146 |           --- |       3.7657% |
| 0.002 |  3762 |     7252 | 0.14231387307209176 | +3.816379e-03 |       1.1139% |
| 0.001 |  8478 |    16536 | 0.14391693899251826 | +1.603066e-03 |       0.0000% |

observed order p = 1.2514 (from the last three rungs, uniform ratio 2)
estimated discretisation error at h = 0.001 m: 1.161074e-03 T (0.8068 % of the value)
criterion: estimated relative error <= 1.000 % and |delta| strictly falling over the last two intervals
converged: YES
```

At `h = 0.001 m`, the 3-rung Richardson discretisation error estimate reported by `converge` is `1.161074e-03 T` (`0.8068 % <= 1.000 %`, `observed order p = 1.2514`, `converged: YES`).

## Bit-Identical Default Path

- **Absent `mesh` preserves default outputs byte-for-byte.** With no `"mesh"` block and no `--element-size` flag, `Fem2dEngine` retains `self._version` (`+m5.4c`) and computes the exact `b0b8511` results (`Bz(0, 0, 0) = 0.14049244777644335 T`, `2 958` nodes / `5 676` elements on the un-meshed steel case), `examples/magnet-n42-air.json` keeps `"mesh"` absent, and every un-meshed example round-trips cleanly. -- enforced-by: tests/test_m55_mesh_density.py::test_default_path_bit_identical_and_examples_roundtrip

## Consequences

- Cases can carry explicit, reproducible FEM mesh resolution via `"mesh": {"element_size": <metres>}`, without modifying the global `Fem2dEngine` default or existing benchmark tolerances.
- Users and CI workflows can inspect `converge` tables directly from the CLI with full `repr` precision, observed order `p`, Richardson error estimates, and explicit warnings whenever refinement deltas fail to decrease.
- `examples/steel-cylinder-solenoid-nonlinear.json` now resolves to `0.14391693899251826 T` (`0.0563 %` from Elmer's `0.143836 T`) via `solve --engine fem2d` and `report --engine fem2d` while leaving all 27 frozen benchmarks and all other examples untouched.
