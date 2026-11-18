# Product Requirements — MagWave

Document version: **0.1.0** (Phase 0) · Status: **draft for review** · Owner: Jason (jblane1000)
Requirement IDs are stable and never renumbered. A requirement is satisfied only when its acceptance
criterion is demonstrated by a test, a benchmark or a recorded measurement — never by inspection.

Related: [RESEARCH.md](RESEARCH.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [ROADMAP.md](ROADMAP.md) ·
[CHECKLISTS/validation.md](CHECKLISTS/validation.md)

---

## 1. Problem

A mechanical/CAD engineer with a magnet or coil problem has three options today:

1. **A 2D tool** that cannot represent the part he actually has, with an interface from another decade.
2. **A five-figure commercial package** with annual maintenance, aimed at specialists.
3. **Hand calculation**, which stops working the moment there is iron in the circuit.

And from direct experience: a vendor demo that produced one picture, no numbers and no re-runnable case —
so when the same question came back four years later, nothing from that trial could be reused.

## 2. Users

- **P1 — Shop engineer (primary).** 25 years of CAD/CAM and machining; runs the design of a magnetizing
  fixture, a pot magnet, an actuator or a sensor housing. Works in millimetres and inches, thinks in field
  at a point, pull force and saturation margin. Does not want to author solver input files.
- **P2 — Student / hobbyist.** Wants to see the field from an arrangement of magnets and coils, correctly,
  for free, without a licence server.
- **P3 — Maintainer / contributor.** Needs a clean data model, a documented engine boundary and a test
  suite that catches regressions. This person is also the author, next month.

## 3. Goals

- **G1** Turn a CAD assembly into a field/force answer in minutes, on a desktop, with no licence.
- **G2** Produce results that can be re-run and cited: every run is a file on disk.
- **G3** Be provably right on the benchmarks it claims, with published tolerances.
- **G4** Keep the solver boundary clean enough that engines can be replaced without touching the UI.

## 4. Non-goals (explicit)

- **NG1** Parity with a commercial BEM suite across the board. We claim a subset, measured.
- **NG2** Motors, transformers, induction heating, EMC, high frequency. Out of scope for v1.
- **NG3** Optimisation / topology synthesis in v1. Parametric sweeps only.
- **NG4** Multi-user, cloud, licence management. Single-user desktop.
- **NG5** A material library scraped from a vendor. Public datasheets and user-entered curves only.

## 5. Scope by tier

| Tier | In scope | Acceptance gate |
|---|---|---|
| **T0 analytic** | Block / cylinder / ring permanent magnets (uniform magnetization); straight wire, loop, finite solenoid; superposition; B, H, |B|; force and torque between magnets; point/line/plane/volume field sampling; field-line display | Matches closed-form solutions to stated tolerance; ≤100 ms for 10⁴ sample points on a laptop |
| **T1 2D axisym/planar FEM** | Linear and nonlinear (B–H curve) steel; permanent magnets with demagnetization curve; coil sources with N turns and current; force via Maxwell stress tensor and coenergy; air-box or Kelvin boundary | Matches FEMM and analytic references to stated tolerance; 10³-node typical case < 60 s |
| **T2 3D FEM** | Arbitrary assemblies, nonlinear steel, PM + coils, force/torque, field export | TEAM 20 static force within published scatter; second-code agreement |
| **T3 transient/eddy** | Pulse magnetisation, eddy currents, motion | Research; no commitment beyond a documented experiment |

## 6. Functional requirements

### 6.1 Case files and reproducibility

- **FR-001** A case is a single human-readable text file (JSON) containing geometry references, materials,
  sources, study settings, probes, units and app version.
  **AC:** A case can be saved, reloaded and re-solved to bit-identical probe values (NFR-1) on the same machine.
- **FR-002** Every solve writes a results artefact (JSON + CSV) beside the case, including app version,
  engine name and version, and a hash of the case inputs.
  **AC:** `magwave run case.mw.json` produces results carrying all four fields; deleting the case does not
  orphan the results (they embed their own inputs hash).
- **FR-003** Case files declare their unit system and frame (axis, datum face) in the file itself.
  **AC:** Loading a case with mixed units fails with a clear error naming the offending field.
- **FR-004** Cases can reference CAD files by relative path with a stated hash.
  **AC:** A changed CAD file is detected and reported before solving (not silently used).

### 6.2 Geometry and CAD import

- **FR-010** Import STEP (AP203/214/242) and IGES through OpenCASCADE; STL/3MF/OBJ through trimesh.
  **AC:** A three-body STEP assembly imports with correct per-body volumes within 0.1 % of the source CAD.
- **FR-011** Report an import manifest: bodies found, units detected, open shells, non-manifold edges,
  interference pairs, and bodies skipped with the reason.
  **AC:** A deliberately broken assembly produces a manifest listing every defect; no body is dropped silently.
- **FR-012** Boolean union/subtract and defeaturing for small features below a user threshold.
  **AC:** A part with 40 decorative fillets defeats to a clean solid and the volume change is reported.
- **FR-013** Assign bodies to roles: permeable solid, permanent magnet, coil volume, air, symmetry plane.
  **AC:** Unassigned bodies block the solve and are listed.

### 6.3 Materials

- **FR-020** Material model: linear (μr), nonlinear (B–H table with monotone interpolation), permanent
  magnet (remanence + demagnetization curve in the second quadrant + temperature coefficient).
  **AC:** An NdFeB grade entered from a public datasheet reproduces the datasheet's Br, Hcb and BHmax within 2 %.
- **FR-021** Materials live in a versioned, plain-text library shipped with the app; user overrides are
  stored with the case, not silently global.
  **AC:** A user-edited material appears in the results provenance; deleting the override restores shipped values.
- **FR-022** No vendor-proprietary material data is shipped. Provenance field is mandatory per material.
  **AC:** CI check rejects a material without a provenance string.

### 6.4 Sources

- **FR-030** Coil source: N turns, current, conductor cross-section, fill factor; derived current density.
  **AC:** A solenoid's on-axis field matches the elliptic-integral solution within 1 % at 10 points.
- **FR-031** Impressed current density in a volume; impressed magnetization direction for PM bodies.
  **AC:** A uniformly magnetized cuboid reproduces the closed-form field at 20 sample points within 1 %.
- **FR-032** Multiple sources superpose; the UI lists each source's contribution on request.
  **AC:** Two sources summed give the same field as a single combined case.

### 6.5 Solving

- **FR-040** Engine abstraction: any engine implements `capabilities()`, `validate(case)`, `solve(case) -> Result`.
  **AC:** A dummy engine implementing the protocol runs end-to-end in tests without the UI.
- **FR-041** Pre-solve validation: capability gaps (e.g. steel submitted to the analytic engine) block the
  run with a message naming the missing capability and the tier that would answer it.
  **AC:** Steel case on Tier 0 exits non-zero with an actionable message; nothing is solved.
- **FR-042** Convergence reporting for FEM tiers: iteration count, residual, mesh statistics.
  **AC:** A deliberately non-converged case reports failure rather than a number.
- **FR-043** Determinism: same case + same engine version = same numbers (NFR-1).

### 6.6 Results and post-processing

- **FR-050** Field sampling at points, along lines, on planes and in volumes; export CSV/JSON.
  **AC:** Exported values match in-app values exactly.
- **FR-051** Visualisation: slice planes with |B| colour map, vector glyphs, streamlines, iso-surfaces of |B|.
  **AC:** A slice through a magnet shows the expected symmetry; the colour scale is reported with units.
- **FR-052** Force and torque: Maxwell stress tensor (FEM) and dipole/analytic (T0); the method used is
  stated in the result.
  **AC:** Force between two coaxial magnets matches the analytic result within 2 % outside near-contact.
- **FR-053** A report export (PDF/HTML) containing case summary, figures, numbers with units, benchmark
  provenance and engine versions.
  **AC:** Report contains everything needed to re-create the run without reading the app source.

### 6.7 Interfaces

- **FR-060** CLI: `magwave run`, `magwave validate`, `magwave export`, `magwave bench`.
  **AC:** Every command works headless in CI with no display.
- **FR-061** Desktop GUI (PySide6 + PyVista): open case, assign roles, run, view, export.
  **AC:** A new user completes "import STEP → assign → run → slice" from the README alone.
- **FR-062** Python API: the same objects the CLI uses, importable without Qt.
  **AC:** `import magwave` succeeds on a machine with no Qt installed.

### 6.8 Validation harness

- **FR-070** A benchmark suite runs in CI on every push; each case declares tolerance and source.
  **AC:** Suite green on `main`; a 1 % artificial perturbation to any expected value fails it.
- **FR-071** Published results table generated by CI, stamped with commit and versions.
  **AC:** `docs/BENCHMARKS.md` is regenerated by CI, not hand-edited.
- **FR-072** TEAM problems are implemented as data-only cases where the definition is public.
  **AC:** TEAM 20 case file exists and runs on the Tier-2 engine when available.

## 7. Non-functional requirements

- **NFR-1 Determinism.** Same inputs and versions produce identical numbers; summation order and mesh
  generation are seeded. Tolerance: bit-identical on the same platform.
- **NFR-2 Units.** SI internally; mm/in at CAD and UI boundaries; every displayed number carries a unit.
  **AC:** A unit round-trip test covers every quantity in the data model.
- **NFR-3 Licence boundary.** No GPL code linked into the app. gmsh/Elmer invoked as subprocesses only.
  **AC:** A CI licence check inspects the dependency tree and fails on a GPL runtime import.
- **NFR-4 Platforms.** Linux (primary) and Windows 10/11 desktop; Python ≥ 3.11.
- **NFR-5 Performance.** T0: ≥ 10⁴ sample points in ≤ 100 ms. T1: typical axisymmetric case ≤ 60 s, ≤ 4 GB.
- **NFR-6 Offline by default.** No network calls at runtime. An update check, if ever added, is opt-in.
- **NFR-7 Accessibility of data.** Every numeric result is exportable without the GUI.
- **NFR-8 No silent failures.** Any body, material or boundary that cannot be honoured is reported and
  blocks or annotates the result. A wrong number that looks plausible is the worst outcome.

## 8. Parity matrix against the priced reference (AMPERES)

Captured from the vendor's public product description, not from a licence or a binary.

| Capability | AMPERES | MagWave v1 target | Notes |
|---|---|---|---|
| 3D magnetostatics, static | yes | Tier 2 (later) | our first 3D milestone |
| 2D / axisymmetric | yes (MAGNETO) | Tier 1 (second milestone) | |
| Analytic fast path | no | **yes (Tier 0)** | our differentiator for shop questions |
| Nonlinear ferromagnetic materials | yes | Tier 1 / 2 | B–H curves |
| Permanent magnets with demag curve | yes | Tier 1 / 2 | second-quadrant model |
| Force / torque / inductance | yes | Tier 1 / 2 | method stated per result |
| Phasor (AC) | yes | **not in v1** | Tier 3 candidate |
| Transient eddy current | yes | **not in v1** | Tier 3 |
| Coils and windings editor | yes | simplified coil objects | v1 scope |
| Parametric sweeps | yes | yes (CLI-first) | |
| CAD healing | yes | import manifest + defeaturing | no silent repair |
| Batch / API | yes | CLI + Python API | our strength |
| BEM open-boundary solver | yes | no | FEM + air box/Kelvin in v1 |
| GUI polish, vendor support | yes | no | honest gap |

## 9. Acceptance gate for v0.1 (Tier 0 release)

All must hold, evidenced in CI output:

1. FR-001, FR-002, FR-003, FR-010, FR-011, FR-020, FR-030, FR-040, FR-041, FR-050, FR-060, FR-061,
   FR-062, FR-070 satisfied.
2. Benchmark suite: ≥ 12 analytic cases green; solenoid within 1 %; cuboid within 1 %; two-magnet force
   within 2 %.
3. `docs/BENCHMARKS.md` regenerated by CI and committed by the release workflow.
4. Licence check clean (NFR-3); no network calls at runtime (NFR-6).
5. A first-run walkthrough exists and was executed by a second person (or a fresh VM) from the README alone.

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| CAD assembly cleaning consumes the project | schedule | import manifest first, defeaturing minimal, fail loudly |
| Tier 1 accuracy disputes (forces) | credibility | publish tolerances and method per result; TEAM/FEMM reference |
| Scope creep toward a Maxwell clone | never ships | non-goals NG1–NG5, tier gates |
| GPL contamination via gmsh | licence | subprocess boundary + CI licence check |
| Silent wrong numbers | worst case | NFR-8, benchmark gate, no-silent-drop rule |

## 11. Requirement conventions

`FR-<module><nnn>` functional, `NFR-<n>` non-functional, `AC` acceptance criterion (testable statement).
A requirement is never renumbered; a withdrawn requirement is marked **Withdrawn** with its date.
