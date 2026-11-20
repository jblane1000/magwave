# Brief — M5 Stage 1: axisymmetric Tier 1 solver (linear media), validated against closed form

**Owner:** Cody. **Engine:** `agy` (`agy-work --model gemini-pro-orchestrator-g1-pro --print-timeout 0 -p '<pointer>'`).
You expand this brief, run agy, verify, commit, open a PR. Max re-runs acceptance.

Repo `/home/jblane1000/projects/magwave`, branch from `main` @ `f4e7b36`.
Environment: **always** `env -u PYTHONPATH .venv/bin/python …` (a 3.14 PYTHONPATH from the Hermes env shadows the 3.12 venv).

## Why this stage exists

M5 is Tier 1 FEM: axisymmetric/planar solver, nonlinear steel, PM demagnetization, forces. Its exit gate is
"FEMM/analytic agreement; mesh-independence checks". Nothing in the repo solves a PDE today — Tier 0 is
Magpylib free-space superposition. Stage 1 builds the solver skeleton and proves it on problems with exact
answers, before any nonlinear material is introduced. **No steel, no B–H, no saturation in this stage.**

## Formulation (fixed — do not substitute a different one without an ADR)

Axisymmetric magnetostatics with the **magnetic vector potential** `A = A_phi(r, z) · ê_phi`:

```
∇ × ( (1/μ) ∇ × A ) = J_phi        in 2D (r, z)
```

- Single scalar unknown per node; no gauge fixing required.
- Dirichlet `A_phi = 0` on the far-field boundary (place it ≥ 5× the largest dimension).
- Weak form carries the axisymmetric measure `2π r`: the stiffness contribution of a triangle includes
  the `r` weighting of its centroid (or exact per-element integration for linear triangles — state which
  you used).
- `B = ∇ × A` gives `B_r = -∂A_phi/∂z`, `B_z = (1/r) ∂(r A_phi)/∂r`. Watch the `1/r` on the axis: document
  how you handle `r → 0` (nodes on the axis are fine in the A_phi formulation because A_phi → 0 there; state it).

Sources for this stage:
- **Coil**: rectangular cross-section in the (r, z) plane carrying `J_phi = N·I / A_window` (turns × current
  over winding cross-section), uniform over the region. FR-030's fill factor applies.
- **Permanent magnet, uniform magnetization**: represent as an equivalent surface current
  `K = M × n` on the magnet's boundary (uniform M has no volume current). For an axially magnetized
  cylinder this is a surface current on the cylindrical face.

## Deliverables

1. `src/magwave/fem2d/` package, OCP/gmsh/Elmer-free:
   - `mesh.py` — mesh generator for axis-aligned rectangular regions (a body's (r,z) footprint), producing
     linear triangles with a requested element size `h`, refinement levels, and mesh statistics
     (nodes, elements, min edge, h). No external mesher: gmsh stays a subprocess-only, later-stage concern.
   - `assembly.py` — element stiffness/mass-type matrices for linear triangles with the `2πr` weighting.
   - `solve.py` — sparse assembly and solve (`scipy.sparse.linalg.spsolve`), returning `A_phi` plus
     convergence/convergence statistics: iterations (1 for linear), relative residual, mesh stats (FR-042).
   - `field.py` — element-wise `B` evaluation and interpolation to arbitrary (r, z) sample points;
     return in the existing `Sample`/`Result` shapes.
   - `engine.py` — an `Engine` implementation (`engine_id="fem2d"`, `rung="Tier 1"`) so the CLI and GUI can
     dispatch to it exactly like the Magpylib engine. It must advertise honestly in `Capabilities`:
     axisymmetric linear media + coil + uniformly magnetized PM only; **no** permeable material yet, and
     `validate()` must reject a case asking for steel with an actionable message naming Tier 1 stage 2.
2. **New core dependency `scipy`** in `pyproject.toml` (BSD-3; licence boundary must stay clean — verify
   `tools/check_licence_boundary.py` still passes and note the addition in NOTICE/PRD if the project's
   convention requires it). No new GPL dependency.
3. **ADR** `docs/adr/0007-tier1-formulation.md`: why A_phi axisymmetric, why Dirichlet-at-5×, why scipy,
   why a built-in mesher now. Include the rejected alternative (reduced scalar potential / Ω-formulation)
   and the reason it is deferred rather than dismissed.
4. **Validation tests** (this is the stage's real deliverable — the gate is agreement, not code volume):
   - **Solenoid**: finite-length solenoid (N, I, winding rectangle) on-axis `B_z` vs the closed-form
     elliptic-integral solution at ≥ 10 points, within **1 %**. This is FR-030's acceptance criterion
     re-proved through the FEM path.
   - **Uniformly magnetized cylinder**: on-axis field vs closed form at ≥ 10 points within **1 %**.
   - **Superposition**: two sources solved together equal the sum of separate solves (tolerance stated).
   - **Mesh independence**: the same case at two refinement levels (h and h/2); report the relative error
     against the analytic value for each and assert the error *decreases* and stays under 1 % at the finer
     level. Print/record the numbers so they land in the test output.
   - **Determinism** (FR-043): the same case solved twice gives bit-identical `B` values.
   - **Capability gate** (FR-041): a steel case on `fem2d` is rejected with a message naming the missing
     capability, and nothing is solved.
   - **Axis behaviour**: a sample point exactly on the axis returns a finite `B` (no divide-by-zero, no NaN).
5. **Benchmark integration**: register the FEM engine in the benchmark runner for the analytic cases it can
   honestly answer, without altering the 27 frozen expected values. If the harness cannot yet express a
   "second engine on the same expected values", add that capability *additively* and say so in the report.

## Fence

- No gmsh/Elmer/OCP import anywhere in `fem2d/`; CI's import hook must stay green.
- Do not change existing case schema semantics; if a field is added, older cases must still load.
- Do not weaken or edit the 27 frozen benchmark values, or any existing test.
- Do not merge to `main`. PR title: `M5.1: axisymmetric Tier 1 solver (linear media)`.

## Acceptance (Max re-runs these; report each line's real output)

```bash
cd /home/jblane1000/projects/magwave
env -u PYTHONPATH .venv/bin/python -m pytest --junit-xml=/tmp/cody-m51.xml -q
env -u PYTHONPATH .venv/bin/python -m magwave bench
.venv/bin/ruff check --no-cache .
.venv/bin/ruff format --check .          # CI gates on this; the FR-012 round missed it
env -u PYTHONPATH .venv/bin/python tools/check_licence_boundary.py
```

Plus the numbers that matter: per-test analytic-vs-FEM relative errors at both mesh levels, the mesh
statistics, and the solve wall time for each validation case.

## Report

`docs/briefs/m5-stage1-report.md`: formulation as implemented, element/mesh details, every validation
number in a table, the ADR path, any deviation with its reason, and the count of `agy` invocations that
produced the code (state zero plainly if zero).
