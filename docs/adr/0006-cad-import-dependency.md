# ADR 0006 — CAD import: OCP for STEP/IGES as a dependency, trimesh for meshes

- Status: accepted
- Date: 2026-11-19
- Deciders: Jason (owner)
- Amends: ADR-0002 (licence strategy) — adds the measured OCCT linkage facts
- Supersedes nothing

## Context

M4 requires STEP/IGES/mesh import (FR-010), an import manifest that drops no body silently
(FR-011), defeaturing (FR-012) and per-body role assignment (FR-013). ADR-0002 already named
OpenCASCADE via OCP and trimesh for mesh formats, but the licensing and packaging questions
were never measured, and the PRD treats IGES as equivalent to STEP. Neither was verified.

All figures below were measured on 2026-11-19 in throwaway venvs
(`~/.hermes/cache/scratch/m4probe/`), Python 3.12 on the mainserver. Nothing is quoted from
documentation.

### Packaging

| Candidate | Result |
|---|---|
| `pythonocc-core` 0.16 | **sdist only — zero wheels for any Python tag.** Unusable for a published tool. |
| `cadquery-ocp` 8.0.1.1.0 | wheels for cp311–cp313 on `manylinux_2_28` (x86_64, aarch64), **`win_amd64` cp311–cp314**, macOS 11+ (arm64, x86_64). `requires-python >=3.11,<3.15`. |
| `trimesh` 5.2.1 | pure Python + numpy, MIT, ~15 MB. |

`cadquery-ocp` hard-depends on `vtk==9.6.2`, so an OCP-capable environment costs **~1.2 GB**
installed (285 MB of that is OCCT). This is the only real downside of the decision and it must
be surfaced in the install docs.

### Licence (the actual reason this ADR exists)

- OCCT is **LGPL-2.1 with the Open CASCADE Exception v1.0**. It is not GPL, so the process
  boundary in ADR-0002 was never aimed at it.
- Measured: the OCP wheel loads OCCT as **52 separate shared objects** (`libTK*.so.8.0.1` in
  `cadquery_ocp.libs/`, vendored rpath). That is dynamic linking with the library replaceable,
  which is the case LGPL-2.1 §6 permits for a program under any licence.
- The wheel's own `LICENSE` is Apache-2.0 **and carries no OCCT text or exception** — so the
  obligation to name OCCT falls on us.

### Capability (measured against the FRs)

- **FR-010 AC:** three-body STEP round-trip returned per-body volumes matching closed-form
  arithmetic to `2.2e-14`, `0`, `2.3e-16` relative — against a `1e-3` gate.
- **FR-011, the trap:** a naive `STEPControl_Reader` + `OneShape` read of that three-body file
  returned **one** solid of `1948.761 mm³` — the sum. Three parts merged silently. Reading
  through XCAF and recursing the product tree returns the correct three bodies. The regression
  test for FR-011 must be exactly this merged assembly.
- **FR-011, defect detection:** `ShapeAnalysis_Shell` reports `HasFreeEdges` true for a
  5-of-6-face open shell and false for the closed solid. Usable, real data for the manifest.
- **FR-013, names:** 2 of 3 body names survived an XCAF round-trip; one returned as `SOLID`.
  Name fidelity is writer-dependent and must not be assumed.
- **IGES is materially weaker than STEP:** written fine, but read back as **0 solids**, even
  after `ShapeFix`, while the healed volume was exactly right (384.0000 mm³) — the geometry
  arrives as faces and needs sewing plus solid reconstruction.
- **gmsh as a STEP reader is rejected:** the pip wheel needs `libXft.so.2` (absent on the
  mainserver: 0 matches), and ADR-0002 already forbids importing gmsh in-process. It stays an
  external, containerised tool.

## Decision

1. **`magwave[step]` extra = `cadquery-ocp`, pinned.** Dependency only. STEP is the primary
   import format; IGES is **best-effort** and the manifest must state whether it was healed to
   a solid.
2. **`magwave[mesh]` extra = `trimesh`.** STL/OBJ/PLY/3MF/GLB path with no OCCT at all, so a
   user with only STL pays ~15 MB, not 1.2 GB.
3. **Core stays `numpy` + `magpylib`.** Importing `magwave` must not load OCP, trimesh or Qt.
   Both CAD paths are lazy, mirroring the GUI rule.
4. **The import manifest is the interface.** Both paths emit the same body list (name,
   volume, bbox, solids count, defects, role). M4's tests are written against the manifest, not
   against OCP — so OCP can later be moved behind a `magwave-cad` sidecar process without
   rewriting the milestone.
5. **STEP reads must recurse the product tree.** A single-shape read is a defect, not a
   shortcut; the merged-assembly case is a regression test.
6. **NOTICE must name Open CASCADE Technology, its LGPL-2.1 + exception, and ship the OCCT
   licence text.** The wheel does not carry it.
7. **We depend, we do not distribute.** `cadquery_ocp.libs/` must never be vendored into a
   wheel, sdist, Docker image or one-click installer that we publish. If a bundled installer is
   ever built, LGPL §6 relinking obligations become ours and this ADR must be revisited first.

## Consequences

- STEP/IGES works on Linux, Windows and macOS from wheels, with no build step for the user.
- 1.2 GB is added to any environment that takes the `step` extra; the docs must say so, and the
  `mesh` extra exists for users who cannot pay it.
- The licence gate (`tools/check_licence_boundary.py`) keeps forbidding gmsh/Elmer in `src/`;
  OCP is Apache-2.0 at the metadata level and does not trip it, but the NOTICE obligation is
  manual and must be checked at release.
- IGES users get a worse experience than STEP users, stated plainly rather than hidden.
