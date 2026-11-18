#!/usr/bin/env python3
"""Licence-boundary gate (NFR-3).

Fails if any package source imports gmsh or Elmer, or if pyproject.toml declares them as
dependencies. They are permitted only as external processes (see docs/adr/0002-licence-strategy.md).
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_MODULES = ("gmsh", "Elmer", "ElmerSolver")
FORBIDDEN_DISTS = ("gmsh", "pyelmer", "elmer")


def main() -> int:
    problems: list[str] = []

    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in FORBIDDEN_MODULES:
            if f"import {name}" in text or f"from {name}" in text:
                problems.append(f"{path.relative_to(ROOT)} imports {name}")

    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    declared: list[str] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)
    for spec in declared:
        dist = spec.split("==")[0].split(">=")[0].split("<")[0].split("[")[0].strip().lower()
        if dist in FORBIDDEN_DISTS:
            problems.append(f"pyproject declares dependency {spec!r}")

    if problems:
        print("licence boundary violated:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("licence boundary OK (no GPL linkage)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
