# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **M5.5 — Per-case mesh density control (`mesh.element_size`), `--element-size`, and `converge`**:
  added frozen `MeshSpec(element_size)` and optional trailing `Case.mesh` to `magwave.case/1`
  (`src/magwave/cases.py`, omitted from `Case.to_dict()` when `None`, strict `_keys` and
  positive-finite validation refusing `bool`, non-finite, `<= 0`, and unknown keys without
  clamping); added global `--element-size METRES` (validated before command/engine coupling with
  exit `1`, precedence `--element-size` > `case.mesh.element_size` > engine default),
  `solve --engine fem2d` direct case-probe output (refusing `--grid`/`--points` with exit `2`),
  and `magwave converge` (`src/magwave/converge.py`, Richardson order `p` and discretisation error
  on uniform ladders, honest `not estimable` on two-rung/non-uniform ladders, and `WARNING` on
  non-falling deltas); set `"mesh": {"element_size": 0.001}` on
  `examples/steel-cylinder-solenoid-nonlinear.json` as the coarsest rung within `1 %` of Elmer
  (`Bz(0, 0, 0) = 0.14391693899251826 T`, `8 478` nodes / `16 536` elements, `0.0563 %` vs Elmer
  `0.143836 T`, 3-rung Richardson error `0.8068 %` at `p = 1.2514`), leaving
  `examples/magnet-n42-air.json` and the default solver path bit-identical (`docs/adr/0011-mesh-density-control.md`).
- **M5.4c — Cell-aligned virtual-work shift (`+m5.4c`)**:
  enforced exact cell-aligned virtual displacements (`delta = k * h`, integer `k >= 1`) in
  `magwave.fem2d.force.coenergy_force`, refusing sub-cell/non-aligned steps with
  `E_FORCE_DELTA_NOT_ALIGNED` before clearance checks or solving and removing the silent
  `0.25 * max_shift` clamp; added `DeltaStep`, `DeltaStudyResult`, and `coenergy_delta_study`
  (default `(h, 2h, 3h)`) with separate `+delta`/`-delta` mesh-count and gap-cell split
  invariance verification; confirmed asymptotic `O(delta^2)` convergence on the Gate 3 magnet +
  steel plate case at `h = 2.5e-4 m` (`p_per_pair = +2.15022`, `p_fit = +2.15022`,
  `asymptotic = YES`, `F_extrap = 101.37869 +/- 0.25583 N`, `0.8844%` vs Maxwell `102.28330 N`,
  `0.345%` vs Elmer `101.72932 N`) alongside Gate 3 Basis (i) `delta = h = 5.0e-4 m`
  (`102.81233 N`, `0.5146%` vs Maxwell), both PASS on the unchanged `2%` gate; re-based
  `tools/force_report.py` §4 coenergy column to cell-aligned `delta = h` (explicit refusal at
  `h = 1.0e-3 m`); documented findings in `docs/adr/0009-tier1-force-torque.md`.
- **M5.4 — Tier 1 forces (Maxwell stress + coenergy), torque refusal, Tier 0 permeable force gate**:
  `magwave.fem2d.force` (`maxwell_force`, `coenergy_force`, `magnetic_coenergy`, `torque`
  refusing with `E_TORQUE_NOT_MEANINGFUL`) wired into `Fem2dEngine.force` (`+m5.4`), exact
  `BhCurve.magnetic_energy_density`, Tier 0 (`magpylib`) `E_UNSUPPORTED_FORCE_PERMEABLE` refusal,
  Elmer virtual-work cross-check (`tools/elmer_force.py`, `0.545 %` vs Maxwell / `1.065 %` vs
  coenergy), and 7-gate report (`tools/force_report.py`).
- **M5.2 Phase A — Linear permeable steel & FR-041 gate**: versioned material library
  (`magwave.materials` v1 with TEAM 20 `steel-team20` and per-case `mu_r` overrides), additive
  `permeable` and `uniform_field` components in `magwave.case/1`, Tier 0 (`magpylib`) FR-041
  refusal (`E_UNSUPPORTED_PERMEABLE`), and Tier 1 (`fem2d` `+m5.2a`) element-wise linear `nu` and
  Dirichlet applied-field solves validated against the analytic permeable sphere within 1 %.
- **M5.2 Phase B — Nonlinear B–H, consistent Newton, FR-042 convergence reporting, Elmer cross-check**:
  monotone `BhCurve` (`PchipInterpolator` + `MU0`-sloped `Ms` saturation asymptote), consistent
  tangent Newton–Raphson with Armijo backtracking (`magwave.fem2d.nonlinear`), `ConvergenceInfo`
  attached to `Result` and surfaced in `magwave report` and `--max-iterations` (`E_NOT_CONVERGED`),
  plus external Elmer `MagnetoDynamics2D` nonlinear cross-comparison (`tools/elmer_permeable.py`,
  `0.0665 %` agreement at `1.0 A`).

## [0.0.1] — 2026-11-17

### Added

- Phase 0 specification set: research dossier, PRD with requirement IDs and acceptance criteria,
  architecture with the engine protocol, five decision records, four checklists, roadmap, glossary.
- Repository scaffold: `src/magwave` package with version and CLI stub, contract tests, CI
  (lint, format, tests, licence boundary), Apache-2.0 licence, contributing and security policies.

### Notes

- No solver code yet. Nothing in this release computes a field; every capability is specified,
  not implemented. First code lands with milestone M1.
