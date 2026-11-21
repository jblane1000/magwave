# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
