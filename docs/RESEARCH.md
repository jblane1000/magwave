# Research dossier — magnetostatics tooling and the gap MagWave fills

Status: Phase 0, compiled 2026-11-17. Versions and licences were read from the package indices on that
date; re-verify before quoting them. This file records the shape of the field, not a snapshot of the web.

## 1. The problem, as it actually occurred

A shop needed a 3D magnetostatics answer on a CAD assembly. The process available was:

1. Send the assembly STEP file to a commercial vendor.
2. Sit a video demo.
3. Receive a picture.

No input case, no solver output file, no reference numbers — and no purchase. Four years on, the trial
cannot be re-run, cannot be cited, and cannot be used to validate anything. **This is the requirement,
stated as a failure: the number must outlive the meeting.**

## 2. Open source, by layer

| Tool | Layer | Method | 3D? | Notes |
|---|---|---|---|---|
| **Magpylib** | library | closed-form analytic | n/a | Fields of block / cylinder / ring magnets, loops, lines, solenoids. BSD-3. No permeable material; bodies superpose but do not interact magnetically. **Tier-0 engine.** |
| **FEMM** | engine + GUI | 2D / axisymmetric FEM | **no** | Planar and axisymmetric only — its own FAQ states there is no 3D version and points at Radia for intrinsic 3D. Free, scriptable, dated UI. The target Tier 1 improves on. |
| **Elmer FEM** | engine | 3D FEM (+ magnetodynamics) | yes | Nonlinear, magnets + coils + forces, mature, **GPL**. Expert CLI: geometry via gmsh, solver input files. Call as a subprocess. |
| **GetDP / ONELAB** | engine | 3D FEM | yes | Same shape as Elmer; more general, equally expert-facing. |
| **NGSolve** | engine / library | 3D FEM | yes | Scriptable Python FEM; the numerical foundation under the rebuilt Radia. |
| **Radia** | engine + workflow | integral / BEM on NGSolve | yes | Accelerator heritage (undulators). Republished on PyPI with a Python API and an MCP interface; open-boundary operators. Geometry from blocks/arcs, meshed in (often commercial) Cubit — not a STEP-assembly front end. **Proof the BEM lane has open-source legs.** |
| **OpenMagnetics** | front end / library | component models | n/a | MIT. Cores, windings, losses, SPICE export. Designs magnetic *components*; does not solve a 3D field on your geometry. |
| **FreeCAD EM Workbench** | front end | FastHenry (quasi-static) | yes | Inductance / resistance extraction only. Not magnetostatics. |

Commercial: Ansys Maxwell 3D and COMSOL Multiphysics cover all of it at per-seat prices in the tens of
thousands with annual maintenance. The package a shop is actually quoted for magnet and coil work is
INTEGRATED Engineering Software (below).

**Lesson that generalises:** long-lived scientific codes get rewritten in place while the old homepage
stays up. Radia looked abandoned on a 1990s project page and turned out to be alive on the package
index. Survey PyPI and GitHub releases, not the vendor's marketing site.

## 3. The priced reference: INTEGRATED Engineering Software

Winnipeg, founded 1984; identity is **BEM** (with FEM where it helps).

- **AMPERES** — 3D magnetostatics: static, phasor and transient modes; nonlinear ferromagnetic and
  permanent-magnet materials; lossy magnetics; force, torque, flux linkage, inductance; coils-and-windings
  editor; periodic/symmetry reduction; parametric sweeps; adaptive meshing; CAD healing utilities; batch
  mode; an API.
- **FARADAY** — 3D eddy current, time-harmonic and transient, small skin depths, motion.
- **COULOMB** — 3D electrostatics. **MAGNETO / ELECTRO** — 2D and axisymmetric. Plus KELVIN, LORENTZ,
  CABLES, CASPOC, INDUCTO.
- Pricing is quote-based with annual maintenance. Ask for the maintenance line explicitly.
- Their published technical papers — including an analysis of **TEAM problem 20**, the 3D static-force
  benchmark — are free to read, which makes them a legal source of benchmark definitions and reference
  values.

Trademark family to avoid in our name: MAGNETO, AMPERES, FARADAY, COULOMB.

## 4. The gap we are actually filling

Not "a better kernel". Writing a competitive 3D nonlinear magnetostatics kernel is a decade of work by
several people, and Elmer, GetDP and NGSolve already exist.

The gap is the **workflow**: CAD assembly in → body/material/source assignment → mesh → solve →
field, force, report out — in one place, with sane defaults, scriptable, free, and with results that are
written down and re-runnable. Nobody open-source does that well today.

## 5. Verified licence stack (read from PyPI JSON on 2026-11-17)

| Package | Version | Licence | Consequence |
|---|---|---|---|
| PySide6 | 6.12.0 | LGPL-3.0 / GPL (dual) | Dynamic linking; ship Qt licence notices. Fine for a desktop app. |
| PyVista | 0.49.2 | MIT | Free. |
| Magpylib | 6.3.0 | BSD-2-Clause (repository LICENSE; PyPI metadata still lists BSD-3) | Free. |
| trimesh | 5.2.1 | MIT | Free. |
| numpy | 2.5.5 | BSD-3 / 0BSD / MIT / Zlib | Free. |
| scipy | 1.18.1 | BSD-3 | Free. |
| gmsh | 4.15.2 | **GPLv2+** | **Subprocess only.** Never link, never bundle. |
| Elmer | — | **GPL** | Same: external process. |
| OpenCASCADE (OCP / pythonocc) | — | LGPL-2.1 with exception | STEP/IGES import and boolean repair. Fine. |

## 6. Benchmark material available without a licence

- Closed-form: dipole, current loop, finite solenoid (elliptic integrals), uniformly magnetized cuboid
  and cylinder, Halbach array on-axis field. Magpylib's own test suite is a cross-check.
- **Compumag TEAM problems** — problem definitions and cross-code reference results published in the
  open literature. TEAM 20 is 3D static force. TEAM 13 (3D nonlinear), TEAM 24 (rotating machine),
  TEAM 25 (eddy current) are the natural next rungs for Tier 2/3.
- A real shop part measured with a calibrated Hall probe.

## 7. What we deliberately will not do

- Decompile, disassemble or reverse-engineer any commercial binary. Clean-room, capability-list driven.
- Copy a vendor's material library; use public datasheets or our own measurements.
- Claim parity with AMPERES as a whole. Claim parity on a defined subset, measured.

## Sources

- https://www.integratedsoft.com/products/amperes
- https://www.integratedsoft.com/products/faraday
- https://www.integratedsoft.com/products/products
- https://www.femm.info/doku/doku.php?id=faq
- https://pypi.org/project/radia/ ; https://magpylib.readthedocs.io/ ; https://openmagnetics.com/
- https://wiki.freecad.org/EM_Workbench
