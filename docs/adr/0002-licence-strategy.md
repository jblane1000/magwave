# ADR 0002 — Apache-2.0 core; GPL tools behind a process boundary

- Status: accepted
- Date: 2026-11-17
- Deciders: Jason (owner)

## Context

The strongest open-source solvers for our problem (gmsh, Elmer) are **GPL**. Linking them would force the
whole product to GPL. The stated goal is an open-source tool with the widest possible adoption, including
commercial and academic reuse.

## Decision

- **Apache-2.0** for this project (patent grant + permissive reuse).
- **gmsh and Elmer are external processes only**: write input files, run the binary, read output. No
  `import`, no bundling, no vendoring.
- PySide6 (LGPL-3.0/GPL dual) is dynamically linked, with the Qt licence notices shipped.
- OpenCASCADE via OCP (LGPL-2.1 with exception) for STEP/IGES; trimesh (MIT) for mesh formats.
- CI enforces the boundary with an import hook that fails if `gmsh` or `Elmer` appear in `sys.modules`
  during the test run.

## Consequences

- Permissive core, commercial-friendly, still able to use the best GPL engines — at the cost of
  subprocess plumbing (file-based I/O, error parsing, progress streaming).
- A GPL-linked solver of our own is allowed later (we are the copyright holder of our own code).
- Users on machines without gmsh/Elmer simply get Tier 0/1 and a clear capability message (FR-041).
