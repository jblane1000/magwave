# ADR 0001 — Python 3.11+ core, desktop shell in Qt

- Status: accepted
- Date: 2026-11-17
- Deciders: Jason (owner), Max (author)

## Context

The solver work is geometry, sparse linear algebra, and file I/O. The audience is a shop engineer on a
laptop, not a cluster. Contributions should be approachable by students and by the author in a year.

## Decision

- **Python ≥ 3.11** for everything. numpy/scipy for numerics, Magpylib for the analytic tier.
- **PySide6 (Qt 6) + PyVista** for the desktop UI; the core must import without Qt (FR-062).
- **Packaging** with `pyproject.toml` (PEP 621), `uv` for local workflows, GitHub Actions for CI.

## Consequences

- Fast iteration, readable geometry code, huge ecosystem (OCP, trimesh, meshio, scipy).
- Numerically slower than C++/Fortran: acceptable at Tier 0/1 sizes; Tier 2 delegates to external solvers.
- PySide6 is LGPL — dynamic linking, ship notices (see ADR 0002).
- Distributing a desktop app needs care (PyInstaller/`briefcase` deferred to M5).
