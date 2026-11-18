#!/usr/bin/env python3
"""Tier 1 oracle harness — Elmer FEMM-equivalent magnetostatics cross-check.

Runs a 2D axisymmetric magnetostatic solve in Elmer (via Docker, GPL tool behind a
process boundary per ADR 0002) and compares the extracted on-axis Bz against
magwave's closed-form ``thick_solenoid_Bz``.

Why Elmer and not FEMM: FEMM's Lua scripting interface does not execute under wine
(verified 2026-11-18 — a three-line script that writes one file times out with no
output). Elmer runs natively and is already the planned M7 engine.

Usage:
    python tools/elmer_oracle.py            # run, print table, exit 0/1
    python tools/elmer_oracle.py --tol 0.02 # override the 1% gate

Exit codes: 0 = all probes within tolerance, 1 = a probe exceeded it or the
solve failed. This is a gate, so it speaks: it prints the numbers either way.

Requires: docker with the ``scimulate/elmer`` image (``docker pull scimulate/elmer``).
The mesh and solver input are written to a temp dir; nothing outside it is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magwave.analytic import thick_solenoid_Bz  # noqa: E402

IMAGE = "scimulate/elmer"

# Geometry of the reference case: thick solenoid, J = N I / area.
R1, R2, Z1, Z2 = 0.02, 0.03, -0.05, 0.05
TURNS, CURRENT = 100, 1.0
J = TURNS * CURRENT / ((R2 - R1) * (Z2 - Z1))

# Air box half-extent (metres). 1.0 m was measured to give the tightest agreement;
# a 3 m box with the same element budget is worse because the near field coarsens.
BOX = 1.0
# Mesh grading: (limit, divisions) per axis. Fine through the coil, graded outward.
X_SUBCELLS = [0.0, R1, R2, BOX]
Y_SUBCELLS = [-BOX, Z1, Z2, BOX]
X_DIVS = [20, 10, 120]
Y_DIVS = [120, 10, 120]
COIL_ROW, COIL_COL = 1, 1  # 0-indexed subcell holding the coil cross-section

PROBES = [0.0, 0.03, -0.03, 0.08]


def _grid_file() -> str:
    n = len(X_DIVS)
    rows = []
    for r in range(n):
        rows.append(" ".join("2" if (r == COIL_ROW and c == COIL_COL) else "1" for c in range(n)))
    return (
        "***** ElmerGrid input file for structured grid generation *****\n"
        "Version = 210903\n"
        "Coordinate System = Cartesian 2D\n"
        f"Subcell Divisions in 2D = {len(X_DIVS)} {len(Y_DIVS)}\n"
        f"Subcell Limits 1 = {' '.join(str(v) for v in X_SUBCELLS)}\n"
        f"Subcell Limits 2 = {' '.join(str(v) for v in Y_SUBCELLS)}\n"
        "Material Structure in 2D\n  " + "\n  ".join(rows) + "\n"
        f"Element Divisions 1 = {' '.join(str(v) for v in X_DIVS)}\n"
        f"Element Divisions 2 = {' '.join(str(v) for v in Y_DIVS)}\n"
    )


def _sif_file() -> str:
    return f"""Header
  Mesh DB "." "mesh"
End

Simulation
  Coordinate System = Axi Symmetric
  Simulation Type = Steady State
  Steady State Max Iterations = 1
End

Constants
  Permeability of Vacuum = 1.25663706127e-6
End

Body 1
  Target Bodies(1) = 1
  Name = "Air"
  Equation = 1
  Material = 1
End

Body 2
  Target Bodies(1) = 2
  Name = "Coil"
  Equation = 1
  Material = 2
  Body Force = 1
End

Equation 1
  Active Solvers(1) = 1
End

Material 1
  Relative Permeability = 1.0
End

Material 2
  Relative Permeability = 1.0
End

Body Force 1
  Current Density = {J!r}
End

Solver 1
  Equation = "Mag"
  Procedure = "MagnetoDynamics2D" "MagnetoDynamics2D"
  Variable = -dofs 1 "Az"
  Exec Solver = Always
  Steady State Convergence Tolerance = 1.0e-8
  Linear System Solver = Direct
  Linear System Direct Method = UMFPACK
  Nonlinear System Max Iterations = 1
End

Solver 2
  Exec Solver = After All
  Equation = "ResultOutput"
  Procedure = "ResultOutputSolve" "ResultOutputSolver"
  Output File Name = "result"
  Vtu Format = Logical True
End
"""


def run_elmer(workdir: Path) -> tuple[dict[str, float], str]:
    """Mesh, solve, and return on-axis Bz per requested z."""
    (workdir / "grid.grd").write_text(_grid_file(), encoding="utf-8")
    sif = _sif_file().replace("Active Solvers(1) = 1", "Active Solvers(2) = 1 2")
    (workdir / "case.sif").write_text(sif, encoding="utf-8")

    script = (
        "cd /w && ElmerGrid 1 2 grid.grd -out mesh > grid.log 2>&1 && "
        "ElmerSolver case.sif > solve.log 2>&1 && "
        "cp mesh/mesh.nodes /w/mesh.nodes.txt && "
        'for f in mesh/*.vtu; do cp "$f" /w/result.vtu; done && '
        "chmod 644 /w/result.vtu /w/mesh.nodes.txt /w/solve.log 2>/dev/null; echo SOLVED"
    )
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{workdir}:/w",
            "-w",
            "/w",
            IMAGE,
            "bash",
            "-lc",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if "SOLVED" not in proc.stdout:
        tail = workdir / "solve.log"
        msg = (
            tail.read_text(encoding="utf-8", errors="replace")[-800:]
            if tail.exists()
            else proc.stderr[-800:]
        )
        raise RuntimeError(f"Elmer did not finish:\n{msg}")

    return _extract(workdir)


def _extract(workdir: Path) -> tuple[dict[str, float], str]:
    vtus = sorted(workdir.glob("result*.vtu")) or sorted((workdir / "mesh").glob("*.vtu"))
    if not vtus:
        solve = workdir / "solve.log"
        detail = (
            solve.read_text(encoding="utf-8", errors="replace")[-1000:]
            if solve.exists()
            else "(no solve.log)"
        )
        raise RuntimeError("no .vtu produced:\n" + detail)
    vtu = vtus[0]
    raw = vtu.read_bytes()
    marker = raw.find(b"<AppendedData")
    start = raw.find(b"_", marker) + 1
    blob = raw[start:]
    n_bytes = struct.unpack("<I", blob[0:4])[0]
    count = n_bytes // 8
    az = struct.unpack(f"<{count}d", blob[4 : 4 + n_bytes])

    coords: dict[int, tuple[float, float]] = {}
    nodes_file = workdir / "mesh.nodes.txt"
    if not nodes_file.exists():
        nodes_file = workdir / "mesh" / "mesh.nodes"
    for line in nodes_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            coords[int(parts[0])] = (float(parts[2]), float(parts[3]))

    xs = sorted({c[0] for c in coords.values()})
    ys = sorted({c[1] for c in coords.values()})
    nx, ny = len(xs), len(ys)
    dr = xs[1] - xs[0]

    def az_at(col: int, z: float) -> float:
        for k in range(ny - 1):
            if ys[k] <= z <= ys[k + 1]:
                t = (z - ys[k]) / (ys[k + 1] - ys[k])
                return (1 - t) * az[k * nx + col] + t * az[(k + 1) * nx + col]
        j = min(range(ny), key=lambda m: abs(ys[m] - z))
        return az[j * nx + col]

    # On axis Az ~ c1 r + c3 r^3, so Bz(0) = 2 c1 with c1 = (4 a1 - a2) / (2 dr).
    results = {z: (4 * az_at(1, z) - az_at(2, z)) / dr for z in PROBES}
    return results, f"nx={nx} ny={ny} dr={dr:.2e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tol", type=float, default=0.01, help="relative tolerance (default 1%%)")
    parser.add_argument("--json", metavar="PATH", default=None, help="write results as JSON")
    args = parser.parse_args()

    if shutil.which("docker") is None:
        print("docker not found", file=sys.stderr)
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="magwave-elmer-"))
    os.chmod(workdir, 0o777)  # container runs as a different uid
    try:
        values, meta = run_elmer(workdir)
    except Exception as exc:  # noqa: BLE001 - report, do not mask
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"J = {J:.6e} A/m^2   {meta}")
    print(f"{'z (m)':>8} {'Elmer Bz (T)':>17} {'closed form (T)':>17} {'ratio':>8} {'rel err':>9}")
    worst = 0.0
    rows = []
    for z in PROBES:
        got = values[z]
        want = thick_solenoid_Bz(R1, R2, Z1, Z2, J, z)
        rel = abs(got - want) / abs(want)
        worst = max(worst, rel)
        flag = "" if rel <= args.tol else "  FAIL"
        print(f"{z:>8.3f} {got:>17.9e} {want:>17.9e} {got / want:>8.4f} {rel:>8.2%}{flag}")
        rows.append({"z": z, "elmer": got, "closed_form": want, "rel_err": rel})

    print(f"\nworst relative error: {worst:.2%}   gate: {args.tol:.2%}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"J": J, "tol": args.tol, "worst_rel_err": worst, "probes": rows},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    ok = worst <= args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
