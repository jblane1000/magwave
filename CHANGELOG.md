# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **M5.4b — Coenergy delta-convergence study & Gate 2 framing (`+m5.4b`)**:
  added `coenergy_delta_study`, `DeltaStep`, and `DeltaStudyResult` in `magwave.fem2d.force`
  (exported from `magwave.fem2d`) plus read-only `Fem2dEngine.element_size`; bumped
  `Fem2dEngine` version suffix to `+m5.4b`. Wired `tools/force_report.py` Gate 3 to report the
  observed-order delta-extrapolated coenergy force (`100.20786 N`, `step_uncertainty = 1.04757 N`
  / `1.045%`, combined band RSS `1.355%` / linear sum `1.908%`, non-asymptotic `p_fit = +0.50410`
  driven by sub-`h` remeshing `12 126 -> 12 220` nodes), recomputing Maxwell-vs-coenergy agreement
  at `2.0291%` (`FAIL` against the unchanged `2%` gate) while retaining single-step `delta = h`
  (`102.81233 N`, `0.5146%`) as superseded history. Reordered Gate 2 output, ADR 0009, and test
  documentation so the finite-length end-effect explanation precedes the table (`raw dev% = -4.151%`
  ideal-formula error vs `corrected dev% = -0.0327%` FEM-vs-Nagaoka/Lorenz reference `K_N = 0.95880712`).

### Fixed

- **Gate 3 step-convergence transparency & Gate 2 misleading lead**: prevented reporting a
  single hand-picked finite-difference step (`delta = h`) or a non-asymptotic remeshing ladder as
  a converged limit, and fixed Gate 2 presentation so the elementary formula's end-effect error is
  not misread as solver error.

### Added

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
