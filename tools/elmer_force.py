#!/usr/bin/env python3
"""Elmer second-code virtual-work force cross-check on the Gate 3 magnet + steel plate (M5.4).

Geometry (identical in Elmer and MagWave ``Fem2dEngine``):
    * Magnet: cylinder ``R = 0.010 m``, ``L = 0.020 m``, centred at ``(0, 0, z_mag)``,
      linear recoil ``N42-20C-LINEAR-RECOIL`` (``Br = 1.315 T``, ``HcB = 860000 A/m``,
      ``mu_rec = Br / (mu0 * HcB)``; Elmer ``Magnetization 2 = HcB``,
      ``Relative Permeability = mu_rec``).
    * Steel plate: ``steel-team20`` cylinder ``R = 0.030 m``, ``L = 0.010 m``, centred at
      ``(0, 0, 0.016)`` (1 mm air gap between magnet top ``+0.010 m`` and plate bottom
      ``+0.011 m``).
    * Far boundary: ``r = 0.5 m``, ``z in [-0.5, 0.5] m`` with ``Az = 0``.

Because ``scimulate/elmer`` has no ``MagForce`` solver, we regenerate the identical-topology
ElmerGrid mesh at ``z_mag = +delta`` and ``z_mag = -delta`` (default ``delta = 2.5e-4 m``),
read Elmer's solved nodal ``Az`` and mesh elements, integrate total stored magnetic energy
``W = U = sum_e int w(B) 2*pi*r dr dz`` across Elmer's grid with 2x2 Gauss–Legendre
quadrature using the shared material laws, and evaluate ``F_z = -(W(+delta) - W(-delta)) / (2*delta)``.

Exit codes:
    * ``0``: agreement against MagWave Maxwell stress & coenergy within ``--tol`` (default 5 %).
    * ``1``: disagreement > ``--tol``, wrong force sign, or Elmer failure.
    * ``3``: Docker / Elmer unavailable -> prints exactly one line ``SKIP: <verbatim reason>``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.elmer_permeable import (  # noqa: E402
    IMAGE,
    LINEAR_RECOIL_GRADE,
    ElmerUnavailableError,
    _ensure_docker_image,
    _linear_recoil_params,
    _one_line,
    _write_team20_hb_table,
)

from magwave.cases import Case, ExpectedValue, Magnet, Permeable, Probe  # noqa: E402
from magwave.fem2d.engine import Fem2dEngine  # noqa: E402
from magwave.fem2d.force import coenergy_force, maxwell_force  # noqa: E402
from magwave.materials import load_library  # noqa: E402
from magwave.units import MU0  # noqa: E402

DECK_DIR = Path(__file__).resolve().parent / "elmer"
PLATE_GRID_FILE = DECK_DIR / "magnet_plate.grd"
PLATE_SIF_FILE = DECK_DIR / "magnet_plate.sif"

MAGNET_RADIUS = 0.010
MAGNET_LENGTH = 0.020
PLATE_RADIUS = 0.030
PLATE_LENGTH = 0.010
PLATE_Z_CENTER = 0.016
DEFAULT_DELTA = 2.5e-4
MAGWAVE_H = 5.0e-4
GATE3_CONTOUR = (0.012, -0.011, 0.0105)

_GAUSS2 = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))


@dataclass(frozen=True)
class ElmerEnergySolve:
    z_magnet: float
    energy_j: float
    nodes: int
    elements: int
    iterations: int
    wall_time_s: float


def build_magnet_plate_case(z_magnet: float = 0.0) -> Case:
    """Build the Gate 3 N42 linear-recoil magnet + TEAM 20 steel plate geometry."""
    br, _, _, _ = _linear_recoil_params()
    return Case(
        name="gate3_magnet_steel_plate",
        description=(
            "N42 linear-recoil cylinder magnet (R=10 mm, L=20 mm) 1 mm below a TEAM 20 steel "
            "plate (R=30 mm, L=10 mm at z=16 mm)"
        ),
        sources=(
            "magwave.magnets v1 N42-20C linear recoil + magwave.materials v1 steel-team20",
            "Elmer MagnetoDynamics2D virtual-work cross-comparison (tools/elmer_force.py)",
        ),
        components=(
            Magnet(
                kind="cylinder",
                dims={"radius": MAGNET_RADIUS, "length": MAGNET_LENGTH},
                br=(0.0, 0.0, br),
                center=(0.0, 0.0, z_magnet),
                demag_curve=LINEAR_RECOIL_GRADE,
                name="magnet",
            ),
            Permeable(
                kind="cylinder",
                dims={"radius": PLATE_RADIUS, "length": PLATE_LENGTH},
                center=(0.0, 0.0, PLATE_Z_CENTER),
                material="steel-team20",
                name="steel_plate",
            ),
        ),
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
        expected=(
            ExpectedValue(
                check="magnet_plate_axial_force",
                quantity="Fz",
                value=102.0,
                unit="N",
                tolerance=0.05,
                source="Elmer MagnetoDynamics2D virtual work",
            ),
        ),
    )


def _extract_elmer_energy(workdir: Path, z_magnet: float, wall_time_s: float) -> ElmerEnergySolve:
    vtu_path = workdir / "result.vtu"
    nodes_path = workdir / "mesh.nodes.txt"
    elements_path = workdir / "mesh.elements.txt"
    header_path = workdir / "mesh.header.txt"
    solve_log = workdir / "solve.log"

    if not vtu_path.exists() or not nodes_path.exists() or not elements_path.exists():
        detail = (
            solve_log.read_text(encoding="utf-8", errors="replace")[-1200:]
            if solve_log.exists()
            else "(no solve.log)"
        )
        raise RuntimeError(f"Elmer did not produce VTU/mesh outputs:\n{detail}")

    raw = vtu_path.read_bytes()
    marker = raw.find(b"<AppendedData")
    start = raw.find(b"_", marker) + 1
    blob = raw[start:]
    n_bytes = struct.unpack("<I", blob[0:4])[0]
    az = np.array(struct.unpack(f"<{n_bytes // 8}d", blob[4 : 4 + n_bytes]), dtype=np.float64)

    node_lines = nodes_path.read_text(encoding="utf-8").splitlines()
    coords = np.zeros((len(node_lines), 2), dtype=np.float64)
    for line in node_lines:
        parts = line.split()
        if len(parts) >= 5:
            coords[int(parts[0]) - 1] = (float(parts[2]), float(parts[3]))

    elem_lines = elements_path.read_text(encoding="utf-8").splitlines()
    body_ids = np.empty(len(elem_lines), dtype=np.int64)
    quads = np.empty((len(elem_lines), 4), dtype=np.int64)
    for row_idx, line in enumerate(elem_lines):
        parts = line.split()
        body_ids[row_idx] = int(parts[1])
        quads[row_idx] = [
            int(parts[3]) - 1,
            int(parts[4]) - 1,
            int(parts[5]) - 1,
            int(parts[6]) - 1,
        ]

    nodes = len(node_lines)
    elements = len(elem_lines)
    if header_path.exists():
        first = header_path.read_text(encoding="utf-8").splitlines()[0].split()
        if len(first) >= 2:
            nodes = int(first[0])
            elements = int(first[1])

    q_pts = coords[quads]
    q_az = az[quads]
    r0 = q_pts[:, 0, 0]
    r1 = q_pts[:, 1, 0]
    z0 = q_pts[:, 0, 1]
    z1 = q_pts[:, 3, 1]
    dr = r1 - r0
    dz = z1 - z0
    if np.any(dr <= 0.0) or np.any(dz <= 0.0):
        raise RuntimeError("Elmer quad element orientation check failed (non-positive dr or dz)")

    _, hcb, mu_rec, _ = _linear_recoil_params()
    nu_rec = 1.0 / (MU0 * mu_rec)
    bh_curve = load_library().get("steel-team20").bh_curve()

    m_air = body_ids == 1
    m_mag = body_ids == 2
    m_steel = body_ids == 3

    energy_total = 0.0
    for xi in _GAUSS2:
        u = 0.5 * (1.0 + xi)
        r = r0 + u * dr
        for eta in _GAUSS2:
            v = 0.5 * (1.0 + eta)
            a_val = (
                (1.0 - u) * (1.0 - v) * q_az[:, 0]
                + u * (1.0 - v) * q_az[:, 1]
                + u * v * q_az[:, 2]
                + (1.0 - u) * v * q_az[:, 3]
            )
            dadr = (
                -(1.0 - v) * q_az[:, 0] + (1.0 - v) * q_az[:, 1] + v * q_az[:, 2] - v * q_az[:, 3]
            ) / dr
            dadz = (
                -(1.0 - u) * q_az[:, 0] - u * q_az[:, 1] + u * q_az[:, 2] + (1.0 - u) * q_az[:, 3]
            ) / dz
            b_r = -dadz
            b_z = a_val / r + dadr
            b_sq = b_r * b_r + b_z * b_z
            b_mag = np.sqrt(b_sq)
            dv = 0.25 * dr * dz * 2.0 * math.pi * r

            w_air = (0.5 / MU0) * b_sq[m_air]
            w_mag = 0.5 * nu_rec * b_sq[m_mag] - hcb * b_z[m_mag] + 0.5 * MU0 * mu_rec * (hcb * hcb)
            w_steel = bh_curve.magnetic_energy_density(b_mag[m_steel])
            energy_total += float(
                np.sum(w_air * dv[m_air])
                + np.sum(w_mag * dv[m_mag])
                + np.sum(w_steel * dv[m_steel])
            )

    log_text = solve_log.read_text(encoding="utf-8", errors="replace") if solve_log.exists() else ""
    iterations = sum(1 for line in log_text.splitlines() if "ComputeChange: NS" in line)
    return ElmerEnergySolve(
        z_magnet=z_magnet,
        energy_j=energy_total,
        nodes=nodes,
        elements=elements,
        iterations=iterations,
        wall_time_s=wall_time_s,
    )


def _run_elmer_plate_at_z(image: str, z_magnet: float) -> ElmerEnergySolve:
    _, hcb, mu_rec, _ = _linear_recoil_params()
    grd_text = (
        PLATE_GRID_FILE.read_text(encoding="utf-8")
        .replace("__Z_MAG_LO__", f"{z_magnet - 0.5 * MAGNET_LENGTH:.12g}")
        .replace("__Z_MAG_HI__", f"{z_magnet + 0.5 * MAGNET_LENGTH:.12g}")
    )
    sif_text = (
        PLATE_SIF_FILE.read_text(encoding="utf-8")
        .replace("__MU_REC__", f"{mu_rec:.16g}")
        .replace("__MAGNETIZATION_Z__", f"{hcb:.16g}")
    )

    with tempfile.TemporaryDirectory(prefix="magwave-elmer-force-") as tmp:
        workdir = Path(tmp)
        os.chmod(workdir, 0o777)  # noqa: S103 - required: this host runs userns-remapped Docker, and a 0700 dir cannot even be chdir-ed into by the container (verified for default and --user). Scope is one TemporaryDirectory per run.
        (workdir / "grid.grd").write_text(grd_text, encoding="utf-8")
        (workdir / "case.sif").write_text(sif_text, encoding="utf-8")
        _write_team20_hb_table(workdir / "team20_hb.dat")

        script = (
            "trap 'rm -rf /w/mesh; chmod 666 /w/* 2>/dev/null' EXIT && "
            "cd /w && "
            "ElmerGrid 1 2 grid.grd -out mesh > grid.log 2>&1 && "
            "ElmerSolver case.sif > solve.log 2>&1 && "
            "cp mesh/mesh.header /w/mesh.header.txt && "
            "cp mesh/mesh.nodes /w/mesh.nodes.txt && "
            "cp mesh/mesh.elements /w/mesh.elements.txt && "
            'for f in mesh/*.vtu; do cp "$f" /w/result.vtu; done && '
            "echo SOLVED"
        )
        t0 = time.perf_counter()
        proc = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{workdir}:/w",
                "-w",
                "/w",
                image,
                "bash",
                "-lc",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        wall_time_s = time.perf_counter() - t0

        has_elmer_logs = (workdir / "solve.log").exists() or (workdir / "grid.log").exists()
        if proc.returncode != 0 and not has_elmer_logs:
            raise ElmerUnavailableError(
                f"docker container execution failed: {_one_line(proc.stderr or proc.stdout)}"
            )
        if "SOLVED" not in proc.stdout:
            solve_log = workdir / "solve.log"
            grid_log = workdir / "grid.log"
            if solve_log.exists():
                tail = solve_log.read_text(encoding="utf-8", errors="replace")[-1200:]
            elif grid_log.exists():
                tail = grid_log.read_text(encoding="utf-8", errors="replace")[-1200:]
            else:
                tail = proc.stderr[-1200:]
            raise RuntimeError(f"Elmer solve failed:\n{tail}")

        return _extract_elmer_energy(workdir, z_magnet, wall_time_s)


def run_elmer_force_comparison(
    *,
    image: str = IMAGE,
    delta: float = DEFAULT_DELTA,
    tol: float = 0.05,
    allow_pull: bool = False,
) -> dict[str, object]:
    """Run Elmer virtual-work and compare against MagWave Maxwell stress & coenergy."""
    _ensure_docker_image(image, allow_pull=allow_pull)
    res_plus = _run_elmer_plate_at_z(image, +delta)
    res_minus = _run_elmer_plate_at_z(image, -delta)

    # Pure permanent magnet + steel (no free currents): F_z = -(U(+d) - U(-d)) / (2*d)
    f_elmer = -(res_plus.energy_j - res_minus.energy_j) / (2.0 * delta)

    _, _, _, linear_curve = _linear_recoil_params()
    case = build_magnet_plate_case(0.0)
    engine = Fem2dEngine(
        element_size=MAGWAVE_H,
        material_model="nonlinear",
        magnet_curves={LINEAR_RECOIL_GRADE: linear_curve},
    )
    mw_maxwell = maxwell_force(case, engine, target="magnet", contour=GATE3_CONTOUR)
    mw_coenergy = coenergy_force(case, engine, target="magnet", delta=MAGWAVE_H)

    f_mw_maxwell = mw_maxwell.force_n[2]
    f_mw_coenergy = mw_coenergy.force_n[2]

    rel_vs_maxwell = abs(f_mw_maxwell - f_elmer) / abs(f_elmer)
    rel_vs_coenergy = abs(f_mw_coenergy - f_elmer) / abs(f_elmer)
    attractive = f_elmer > 0.0 and f_mw_maxwell > 0.0 and f_mw_coenergy > 0.0
    passed = attractive and (rel_vs_maxwell <= tol) and (rel_vs_coenergy <= tol)

    return {
        "rung": "second code (Elmer)",
        "delta_m": delta,
        "tol": tol,
        "attractive": attractive,
        "passed": passed,
        "elmer": {
            "nodes": res_plus.nodes,
            "elements": res_plus.elements,
            "energy_plus_j": res_plus.energy_j,
            "energy_minus_j": res_minus.energy_j,
            "iterations_plus": res_plus.iterations,
            "iterations_minus": res_minus.iterations,
            "wall_time_s": res_plus.wall_time_s + res_minus.wall_time_s,
            "force_z_n": f_elmer,
            "sign_convention": "F_z = -(U(+delta) - U(-delta)) / (2*delta) for J=0 magnet+steel",
        },
        "magwave": {
            "element_size_m": MAGWAVE_H,
            "nodes": mw_maxwell.mesh_nodes,
            "elements": mw_maxwell.mesh_elements,
            "contour_m": list(mw_maxwell.contour) if mw_maxwell.contour is not None else None,
            "maxwell_force_z_n": f_mw_maxwell,
            "coenergy_force_z_n": f_mw_coenergy,
            "coenergy_delta_m": mw_coenergy.delta_m,
        },
        "rel_diff_vs_maxwell": rel_vs_maxwell,
        "rel_diff_vs_coenergy": rel_vs_coenergy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tol", type=float, default=0.05, help="relative agreement gate (default 0.05 = 5%%)"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_DELTA,
        help=f"virtual displacement delta in m (default {DEFAULT_DELTA:g})",
    )
    parser.add_argument("--image", default=IMAGE, help=f"Docker image (default {IMAGE})")
    parser.add_argument(
        "--allow-pull",
        action="store_true",
        help="allow pulling scimulate/elmer if not present locally",
    )
    parser.add_argument("--json", metavar="PATH", default=None, help="write comparison JSON")
    args = parser.parse_args()

    try:
        payload = run_elmer_force_comparison(
            image=args.image,
            delta=args.delta,
            tol=args.tol,
            allow_pull=args.allow_pull,
        )
    except ElmerUnavailableError as exc:
        print(f"SKIP: {exc}")
        return 3
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    elmer = payload["elmer"]
    mw = payload["magwave"]
    rel_m = float(payload["rel_diff_vs_maxwell"])
    rel_c = float(payload["rel_diff_vs_coenergy"])
    status = "PASS" if payload["passed"] else "FAIL"

    print(
        "Gate 6 — Elmer second-code virtual work (Gate 3 N42 linear-recoil magnet + TEAM 20 steel plate)"
    )
    print(f"Sign convention: {elmer['sign_convention']} (attractive +z: {payload['attractive']})")
    header = (
        f"{'Rung':<20} | {'Elmer mesh':<16} | {'W(+d) (J)':>12} | {'W(-d) (J)':>12} | "
        f"{'F_elmer (N)':>11} | {'vs Maxwell':>10} | {'vs Coenergy':>11} | Gate"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{payload['rung']:<20} | {elmer['nodes']:>6d}n/{elmer['elements']:>6d}e | "
        f"{elmer['energy_plus_j']:>12.6f} | {elmer['energy_minus_j']:>12.6f} | "
        f"{elmer['force_z_n']:>11.6f} | {rel_m:>9.3%} | {rel_c:>10.3%} | {status}"
    )
    print(
        f"MagWave (h={mw['element_size_m']:g} m, {mw['nodes']}n/{mw['elements']}e): "
        f"Maxwell={mw['maxwell_force_z_n']:.6f} N (contour={tuple(mw['contour_m'])}), "
        f"Coenergy={mw['coenergy_force_z_n']:.6f} N (delta={mw['coenergy_delta_m']:g} m); "
        f"tol={args.tol:.1%}"
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
