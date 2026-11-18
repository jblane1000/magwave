# ADR 0003 — Engines behind a protocol; capability-first validation

- Status: accepted
- Date: 2026-11-17

## Context

We will have at least three engines with different physics: analytic (Magpylib), 2D FEM (own kernel),
3D FEM (Elmer/NGSolve). The UI must not know which one ran, and the user must not be handed a number from
an engine that cannot model his problem (e.g. steel submitted to an analytic solver).

## Decision

- A single `Engine` protocol: `capabilities()`, `validate(case) -> list[Problem]`, `solve(case) -> Result`.
- **Capability-driven dispatch**: the engine declares what it can model (materials, source types,
  dimensionality, force methods). The core selects or rejects before solving (FR-041).
- Problem codes (e.g. `E_NO_PERMEABLE_MATERIAL`, `E_NON_CONVERGED`, `W_COARSE_GAP`) are stable, documented
  and asserted in tests.

## Consequences

- Adding an engine, or swapping Elmer for NGSolve, touches only `engines/`.
- The GUI's "why can't I run this?" answer is generated from the same code path the tests use.
- Slight overhead: every engine must implement `validate()` honestly; a lying engine is a defect class we
  test for with a deliberately wrong dummy engine.
