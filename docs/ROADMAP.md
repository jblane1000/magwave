# Roadmap

Milestones are gates, not dates. A milestone closes only when its exit criterion is demonstrated.

| # | Milestone | Contents | Exit criterion |
|---|---|---|---|
| **M0** | Specifications | this repository: research, PRD, architecture, ADRs, checklists | owner review of Phase 0 |
| **M1** | Skeleton + harness | package layout, case model, units, CLI stub, **benchmark harness**, CI | 12 analytic cases green in CI; case round-trip test |
| **M2** | Tier 0 solver | Magpylib engine, magnet/coil sources, sampling, forces, CSV/JSON export | FR-030/031/032/050/052 satisfied; 10⁴ points ≤ 100 ms |
| **M3** | Desktop UI v1 | open case, assign roles, run, slice/glyph/streamline, report export | a new user completes the README walkthrough unassisted |
| **M4** | CAD import | STEP/IGES/mesh import, manifest, defeaturing, role assignment | PRD §6.2 satisfied; broken-assembly regression test |
| **M5** | Tier 1 FEM | axisymmetric/planar solver, nonlinear steel, PM demag, forces | FEMM/analytic agreement; mesh-independence checks |
| **M6** | v0.1 release | packaging (Linux/Windows), docs, first public release | PRD §9 acceptance gate |
| **M7** | Tier 2 3D | Elmer/NGSolve adapter, TEAM 20, open boundary | TEAM 20 within published scatter |

FR-012 closed 2026-11-19 with `src/magwave/cad/defeature.py` (`boolean_union`, `boolean_subtract`, `defeature`), manifest `DefeatureRecord` reporting, `--defeature-below` CLI wiring, and the 40-fillet regression test in `tests/test_defeature.py` (completing M4).

Deliberately after v0.1: parametric sweeps UI, coil editor, phasor/eddy (Tier 3), scripting recipes.
