# Roadmap

Milestones are gates, not dates. A milestone closes only when its exit criterion is demonstrated.

| # | Milestone | Contents | Exit criterion |
|---|---|---|---|
| **M0** | Specifications | this repository: research, PRD, architecture, ADRs, checklists | owner review of Phase 0 |
| **M1** | Skeleton + harness | package layout, case model, units, CLI stub, **benchmark harness**, CI | 12 analytic cases green in CI; case round-trip test |
| **M2** | Tier 0 solver | Magpylib engine, magnet/coil sources, sampling, forces, CSV/JSON export | FR-030/031/032/050/052 satisfied; 10⁴ points ≤ 100 ms |
| **M3** | Desktop UI v1 | open case, assign roles, run, slice/glyph/streamline, report export | a new user completes the README walkthrough unassisted |
| **M4** | CAD import | STEP/IGES/mesh import, manifest, defeaturing, role assignment | PRD §6.2 satisfied; broken-assembly regression test |
| **M5** | Tier 1 FEM | axisymmetric/planar solver, nonlinear steel, PM demag, forces, mesh density control | FEMM/analytic agreement; mesh-independence checks |
| **M6** | v0.1 release | packaging (Linux/Windows), docs, first public release | PRD §9 acceptance gate |
| **M7** | Tier 2 3D | Elmer/NGSolve adapter, TEAM 20, open boundary | TEAM 20 within published scatter |

FR-012 closed 2026-11-19 with `src/magwave/cad/defeature.py` (`boolean_union`, `boolean_subtract`, `defeature`), manifest `DefeatureRecord` reporting, `--defeature-below` CLI wiring, and the 40-fillet regression test in `tests/test_defeature.py` (completing M4).

### M5 stage detail

Briefs and PRs number M5 sub-stages; this is the authoritative list, so a stage number never drifts.

| stage | contents | state |
|---|---|---|
| M5.1 | axisymmetric Tier 1 solver, linear media | delivered (PR #14) |
| M5.2 | permeable steel, linear + nonlinear B–H (FR-041) | delivered (PR #17) |
| M5.3 | permanent magnet with demagnetisation curve (FR-020) | delivered (PR #18) |
| M5.4 | force and torque: Maxwell stress + coenergy/virtual work | delivered (PR #19) |
| M5.4b | delta-extrapolation attempt | not adopted — sub-cell shifts remeshed, ladder invalid; recorded as superseded history in ADR 0009 (PR #20 closed) |
| M5.4c | cell-aligned virtual-work shift (`E_FORCE_DELTA_NOT_ALIGNED`) | delivered (PR #21) |
| M5.5 | mesh density control: per-case `mesh.element_size`, CLI override, convergence table | pending |

**Open boundary is not an M5 stage.** Infinite-domain handling belongs to **M7** with TEAM 20 and the
Elmer/NGSolve adapter. The analytic sphere references for it (arXiv:2611.15453) are captured in
`docs/briefs/lit-note-open-boundary-and-newton.md` and land with M7. Planar geometry remains M5 work.

Deliberately after v0.1: parametric sweeps UI, coil editor, phasor/eddy (Tier 3), scripting recipes.
