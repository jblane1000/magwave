#!/usr/bin/env python3
"""Nonlinear Elmer second-code cross-check for a saturating TEAM 20 steel cylinder (M5.2 Stage B).

Geometry (identical in Elmer and ``Fem2dEngine``):
    * Steel cylinder: ``radius = 10 mm``, ``length = 60 mm``, coaxial at ``(0, 0, 0)``,
      material ``steel-team20`` (38-point TEAM 20 B–H curve written verbatim from
      ``magwave.materials`` JSON so both codes consume one shared table).
    * Solenoid: ``inner_radius = 15 mm``, ``outer_radius = 25 mm``, ``length = 100 mm``,
      coaxial at ``(0, 0, 0)``, ``turns = 1000``, ``current = 1.0 A``
      (``J = 1.0e6 A/m^2``).
    * Air box: ``1.0 m`` half-extent (``r in [0, 1.0]``, ``z in [-1.0, 1.0]``), reusing the
      ``1.0 m`` domain measured in :mod:`tools.elmer_oracle` where far-field truncation is
      negligible relative to near-field discretisation error.
    * Probe: on axis at the centre ``(0, 0, 0)``, axial flux density ``B_z`` in tesla.

Exit codes:
    * ``0``: relative difference on ``B_z(0, 0, 0)`` is within ``--tol`` (default ``3 %``).
    * ``1``: relative difference exceeded ``--tol`` or Elmer failed to solve/converge.
    * ``3``: Elmer unavailable (no ``docker``, image missing without ``--allow-pull``, or
      container startup failed); prints ``SKIP: <verbatim reason>``.
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
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magwave.cases import Case, Coil, ExpectedValue, Magnet, Permeable, Probe  # noqa: E402
from magwave.fem2d.engine import Fem2dEngine  # noqa: E402
from magwave.materials import DemagCurve, load_library, load_magnet_library  # noqa: E402
from magwave.units import MU0  # noqa: E402

IMAGE = "scimulate/elmer"
DECK_DIR = Path(__file__).resolve().parent / "elmer"
GRID_FILE = DECK_DIR / "permeable_cylinder.grd"
SIF_FILE = DECK_DIR / "permeable_cylinder.sif"
KEEPER_GRID_FILE = DECK_DIR / "magnet_keeper.grd"
KEEPER_SIF_FILE = DECK_DIR / "magnet_keeper.sif"

STEEL_RADIUS = 0.010
STEEL_LENGTH = 0.060
COIL_R_IN = 0.015
COIL_R_OUT = 0.025
COIL_LENGTH = 0.100
COIL_TURNS = 1000
COIL_CURRENT = 1.0
BOX_EXTENT = 1.0
MAGWAVE_H = 1.0e-3

MAGNET_RADIUS = 0.010
MAGNET_LENGTH = 0.020
KEEPER_RADIUS = 0.010
KEEPER_LENGTH = 0.006
KEEPER_Z_OFFSET = 0.013
KEEPER_MAGWAVE_H = 5.0e-4
LINEAR_RECOIL_GRADE = "N42-20C-LINEAR-RECOIL"


class ElmerUnavailableError(RuntimeError):
    """Raised when Docker or the Elmer image cannot be launched (mapped to exit code 3)."""


@dataclass(frozen=True)
class ElmerRunResult:
    bz_center: float
    nodes: int
    elements: int
    iterations: int
    wall_time_s: float
    dr: float


def build_cylinder_case(*, current: float = COIL_CURRENT, expected_bz: float = 0.144) -> Case:
    """Build the shared B.4.1 geometry as a MagWave :class:`Case`."""
    return Case(
        name="nonlinear_team20_cylinder_solenoid",
        description=(
            "TEAM 20 steel cylinder (R=10 mm, L=60 mm) inside a coaxial thick solenoid "
            "(R1=15 mm, R2=25 mm, L=100 mm, N=1000)"
        ),
        sources=(
            "Compumag TEAM 20 B-H table (magwave.materials v1 steel-team20)",
            "Elmer MagnetoDynamics2D cross-comparison (tools/elmer_permeable.py)",
        ),
        components=(
            Permeable(
                kind="cylinder",
                dims={"radius": STEEL_RADIUS, "length": STEEL_LENGTH},
                center=(0.0, 0.0, 0.0),
                material="steel-team20",
                name="steel_core",
            ),
            Coil(
                kind="solenoid",
                dims={
                    "inner_radius": COIL_R_IN,
                    "outer_radius": COIL_R_OUT,
                    "length": COIL_LENGTH,
                },
                current=current,
                turns=COIL_TURNS,
                center=(0.0, 0.0, 0.0),
                name="excitation_coil",
            ),
        ),
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
        expected=(
            ExpectedValue(
                check="nonlinear_fem_reference",
                quantity="Bz",
                value=expected_bz,
                unit="T",
                tolerance=0.03,
                source="Elmer MagnetoDynamics2D cross-comparison",
            ),
        ),
    )


def build_magnet_keeper_case(*, demag_curve: str = LINEAR_RECOIL_GRADE) -> Case:
    """Build the N42 magnet + dual TEAM 20 pole-face keeper geometry matching examples/."""
    grade = load_magnet_library().get("N42-20C")
    return Case(
        name="n42_magnet_dual_keeper",
        description=(
            "N42 cylinder (R=10 mm, L=20 mm) with two TEAM 20 steel pole-face keepers "
            "(R=10 mm, L=6 mm at z=+-13 mm)"
        ),
        sources=(
            "magwave.magnets v1 N42-20C + magwave.materials v1 steel-team20",
            "Elmer MagnetoDynamics2D cross-comparison (tools/elmer_permeable.py)",
        ),
        components=(
            Magnet(
                kind="cylinder",
                dims={"radius": MAGNET_RADIUS, "length": MAGNET_LENGTH},
                br=(0.0, 0.0, grade.br_ref),
                center=(0.0, 0.0, 0.0),
                demag_curve=demag_curve,
                name="n42_magnet",
            ),
            Permeable(
                kind="cylinder",
                dims={"radius": KEEPER_RADIUS, "length": KEEPER_LENGTH},
                center=(0.0, 0.0, KEEPER_Z_OFFSET),
                material="steel-team20",
                name="keeper_top",
            ),
            Permeable(
                kind="cylinder",
                dims={"radius": KEEPER_RADIUS, "length": KEEPER_LENGTH},
                center=(0.0, 0.0, -KEEPER_Z_OFFSET),
                material="steel-team20",
                name="keeper_bottom",
            ),
        ),
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
        expected=(
            ExpectedValue(
                check="nonlinear_pm_demag_fem",
                quantity="Bz",
                value=1.028,
                unit="T",
                tolerance=0.03,
                source="Elmer MagnetoDynamics2D linear-recoil cross-comparison",
            ),
        ),
    )


def _linear_recoil_params() -> tuple[float, float, float, DemagCurve]:
    """Return ``(br, hcb, mu_rec, linear_DemagCurve)`` from the shipped ``N42-20C`` grade."""
    grade = load_magnet_library().get("N42-20C")
    br = float(grade.br_ref)
    hcb = float(grade.hcb)
    mu_rec = br / (MU0 * hcb)
    h_pts = (-1.15 * hcb, -0.8 * hcb, -0.4 * hcb, 0.0)
    curve_pts = [[h_val, br + MU0 * mu_rec * h_val] for h_val in h_pts]
    curve = DemagCurve(
        curve_pts,
        name=LINEAR_RECOIL_GRADE,
        br_ref=br,
        hcb=hcb,
        hcj=grade.hcj,
        t_ref_c=grade.t_ref_c,
    )
    return br, hcb, mu_rec, curve


def _one_line(text: str) -> str:
    return " ".join(text.strip().splitlines()) or "unknown error"


def _ensure_docker_image(image: str, *, allow_pull: bool) -> None:
    if shutil.which("docker") is None:
        raise ElmerUnavailableError("docker binary not found on PATH")

    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode == 0:
        return

    if not allow_pull:
        reason = _one_line(inspect.stderr)
        raise ElmerUnavailableError(
            f"docker image {image!r} is not present locally ({reason}); "
            "run with --allow-pull to fetch it"
        )

    pull = subprocess.run(
        ["docker", "pull", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if pull.returncode != 0:
        raise ElmerUnavailableError(f"docker pull {image!r} failed: {_one_line(pull.stderr)}")


def _write_team20_hb_table(target: Path) -> None:
    """Emit Elmer ``B H`` table pairs verbatim from ``magwave.materials`` JSON."""
    mat = load_library().get("steel-team20")
    if mat.bh_table is None:  # pragma: no cover - guarded by library schema
        raise RuntimeError("shipped steel-team20 material has no bh_table")
    lines = [f"      {b:.10g} {h:.10g}" for h, b in mat.bh_table]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_elmer_result(workdir: Path, wall_time_s: float) -> ElmerRunResult:
    vtu_path = workdir / "result.vtu"
    nodes_path = workdir / "mesh.nodes.txt"
    header_path = workdir / "mesh.header.txt"
    solve_log = workdir / "solve.log"

    if not vtu_path.exists() or not nodes_path.exists():
        detail = (
            solve_log.read_text(encoding="utf-8", errors="replace")[-1200:]
            if solve_log.exists()
            else "(no solve.log)"
        )
        raise RuntimeError(f"Elmer did not produce result.vtu:\n{detail}")

    raw = vtu_path.read_bytes()
    marker = raw.find(b"<AppendedData")
    start = raw.find(b"_", marker) + 1
    blob = raw[start:]
    n_bytes = struct.unpack("<I", blob[0:4])[0]
    count = n_bytes // 8
    az = struct.unpack(f"<{count}d", blob[4 : 4 + n_bytes])

    coords: dict[int, tuple[float, float]] = {}
    for line in nodes_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            coords[int(parts[0])] = (float(parts[2]), float(parts[3]))

    xs = sorted({pt[0] for pt in coords.values()})
    ys = sorted({pt[1] for pt in coords.values()})
    nx, ny = len(xs), len(ys)
    dr = xs[1] - xs[0]

    def az_at(col: int, z_val: float) -> float:
        for k in range(ny - 1):
            if ys[k] <= z_val <= ys[k + 1]:
                t = (z_val - ys[k]) / (ys[k + 1] - ys[k])
                return (1.0 - t) * az[k * nx + col] + t * az[(k + 1) * nx + col]
        j = min(range(ny), key=lambda m: abs(ys[m] - z_val))
        return az[j * nx + col]

    # On axis Az ~ c1 * r + c3 * r^3, so Bz(0, 0) = 2 * c1 with c1 = (4*a1 - a2) / (2*dr).
    bz_center = (4.0 * az_at(1, 0.0) - az_at(2, 0.0)) / dr

    nodes = nx * ny
    elements = (nx - 1) * (ny - 1)
    if header_path.exists():
        first_line = header_path.read_text(encoding="utf-8").splitlines()[0].split()
        if len(first_line) >= 2:
            nodes = int(first_line[0])
            elements = int(first_line[1])

    log_text = solve_log.read_text(encoding="utf-8", errors="replace") if solve_log.exists() else ""
    iterations = sum(1 for line in log_text.splitlines() if "ComputeChange: NS" in line)
    return ElmerRunResult(
        bz_center=float(bz_center),
        nodes=nodes,
        elements=elements,
        iterations=iterations,
        wall_time_s=wall_time_s,
        dr=dr,
    )


def _run_elmer_deck(
    image: str, grid_file: Path, sif_text: str, *, tmp_prefix: str
) -> ElmerRunResult:
    """Execute one Elmer deck via Docker in an isolated temporary directory."""
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmp:
        workdir = Path(tmp)
        os.chmod(workdir, 0o777)  # noqa: S103 - required: this host runs userns-remapped Docker, and a 0700 dir cannot even be chdir-ed into by the container (verified for default and --user). Scope is one TemporaryDirectory per run.
        shutil.copy2(grid_file, workdir / "grid.grd")
        (workdir / "case.sif").write_text(sif_text, encoding="utf-8")
        _write_team20_hb_table(workdir / "team20_hb.dat")

        script = (
            "trap 'rm -rf /w/mesh; chmod 666 /w/* 2>/dev/null' EXIT && "
            "cd /w && "
            "ElmerGrid 1 2 grid.grd -out mesh > grid.log 2>&1 && "
            "ElmerSolver case.sif > solve.log 2>&1 && "
            "cp mesh/mesh.header /w/mesh.header.txt && "
            "cp mesh/mesh.nodes /w/mesh.nodes.txt && "
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

        return _extract_elmer_result(workdir, wall_time_s)


def run_elmer(image: str) -> ElmerRunResult:
    """Run Elmer via Docker in an isolated temporary directory and extract ``Bz(0, 0, 0)``."""
    return _run_elmer_deck(
        image,
        GRID_FILE,
        SIF_FILE.read_text(encoding="utf-8"),
        tmp_prefix="magwave-elmer-nl-",
    )


def run_elmer_magnet_keeper(image: str) -> ElmerRunResult:
    """Run Elmer on the magnet-plus-keeper geometry with linear recoil ``mu_rec``."""
    _, hcb, mu_rec, _ = _linear_recoil_params()
    sif_text = (
        KEEPER_SIF_FILE.read_text(encoding="utf-8")
        .replace("__MU_REC__", f"{mu_rec:.16g}")
        .replace("__MAGNETIZATION_Z__", f"{hcb:.16g}")
    )
    return _run_elmer_deck(
        image,
        KEEPER_GRID_FILE,
        sif_text,
        tmp_prefix="magwave-elmer-mk-",
    )


def run_magwave() -> tuple[float, int, int, int, float, float]:
    """Solve the identical geometry with ``Fem2dEngine(material_model='nonlinear')``."""
    case = build_cylinder_case()
    engine = Fem2dEngine(
        element_size=MAGWAVE_H,
        domain=(BOX_EXTENT, -BOX_EXTENT, BOX_EXTENT),
        material_model="nonlinear",
    )
    t0 = time.perf_counter()
    result, _ = engine.solve_with_report(case)
    wall_time_s = time.perf_counter() - t0
    assert result.convergence is not None
    return (
        float(result.values()[0]),
        result.convergence.nodes,
        result.convergence.elements,
        result.convergence.iterations,
        float(result.convergence.final_relative_residual),
        wall_time_s,
    )


def run_magwave_magnet_keeper() -> tuple[float, int, int, int, float, float]:
    """Solve the magnet+keeper case with the matched linear recoil curve."""
    _, _, _, linear_curve = _linear_recoil_params()
    case = build_magnet_keeper_case(demag_curve=LINEAR_RECOIL_GRADE)
    engine = Fem2dEngine(
        element_size=KEEPER_MAGWAVE_H,
        domain=(BOX_EXTENT, -BOX_EXTENT, BOX_EXTENT),
        material_model="nonlinear",
        magnet_curves={LINEAR_RECOIL_GRADE: linear_curve},
    )
    t0 = time.perf_counter()
    result, _ = engine.solve_with_report(case)
    wall_time_s = time.perf_counter() - t0
    assert result.convergence is not None
    return (
        float(result.values()[0]),
        result.convergence.nodes,
        result.convergence.elements,
        result.convergence.iterations,
        float(result.convergence.final_relative_residual),
        wall_time_s,
    )


def _format_comparison(
    title: str,
    elmer: ElmerRunResult,
    mw_tuple: tuple[float, int, int, int, float, float],
    tol: float,
    *,
    note: str | None = None,
) -> tuple[dict[str, object], bool]:
    mw_bz, mw_nodes, mw_elems, mw_iters, mw_res, mw_time = mw_tuple
    ratio = mw_bz / elmer.bz_center
    rel_diff = abs(mw_bz - elmer.bz_center) / abs(elmer.bz_center)
    passed = rel_diff <= tol
    status = "PASS" if passed else "FAIL"

    print(title)
    if note:
        print(note)
    header = (
        f"{'Elmer Bz (T)':>14} | {'magwave Bz (T)':>14} | {'ratio':>8} | {'rel diff':>9} | "
        f"{'Elmer mesh':>16} | {'mw mesh':>16} | {'Elmer t':>8} | {'mw t':>8} | gate"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{elmer.bz_center:>14.6f} | {mw_bz:>14.6f} | {ratio:>8.5f} | {rel_diff:>8.4%} | "
        f"{elmer.nodes:>6d}n/{elmer.elements:>6d}e | {mw_nodes:>6d}n/{mw_elems:>6d}e | "
        f"{elmer.wall_time_s:>7.2f}s | {mw_time:>7.3f}s | {status}"
    )
    print(
        f"iterations: Elmer={elmer.iterations}, magwave={mw_iters} "
        f"(rel_res={mw_res:.3e}); tol={tol:.2%}"
    )

    payload: dict[str, object] = {
        "elmer": {
            "bz_T": elmer.bz_center,
            "nodes": elmer.nodes,
            "elements": elmer.elements,
            "iterations": elmer.iterations,
            "wall_time_s": elmer.wall_time_s,
        },
        "magwave": {
            "bz_T": mw_bz,
            "nodes": mw_nodes,
            "elements": mw_elems,
            "iterations": mw_iters,
            "final_relative_residual": mw_res,
            "wall_time_s": mw_time,
        },
        "ratio": ratio,
        "rel_diff": rel_diff,
        "tol": tol,
        "passed": passed,
    }
    return payload, passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tol", type=float, default=0.03, help="relative difference gate (default 0.03 = 3%%)"
    )
    parser.add_argument(
        "--case",
        choices=("all", "cylinder", "magnet-keeper"),
        default="all",
        help="comparison case to run (default: all)",
    )
    parser.add_argument(
        "--allow-pull",
        action="store_true",
        help="allow pulling scimulate/elmer if not present locally",
    )
    parser.add_argument("--image", default=IMAGE, help=f"Docker image (default {IMAGE})")
    parser.add_argument("--json", metavar="PATH", default=None, help="write comparison JSON")
    args = parser.parse_args()

    run_cyl = args.case in ("all", "cylinder")
    run_mk = args.case in ("all", "magnet-keeper")

    try:
        _ensure_docker_image(args.image, allow_pull=args.allow_pull)
        elmer_cyl = run_elmer(args.image) if run_cyl else None
        elmer_mk = run_elmer_magnet_keeper(args.image) if run_mk else None
    except ElmerUnavailableError as exc:
        print(f"SKIP: {exc}")
        return 3
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    all_passed = True
    case_payloads: dict[str, dict[str, object]] = {}

    if elmer_cyl is not None:
        cyl_payload, cyl_passed = _format_comparison(
            f"TEAM 20 nonlinear cylinder + solenoid (I={COIL_CURRENT:g} A, N={COIL_TURNS}, "
            f"probe=(0,0,0) Bz, box={BOX_EXTENT:g} m)",
            elmer_cyl,
            run_magwave(),
            args.tol,
        )
        case_payloads["cylinder"] = cyl_payload
        all_passed = all_passed and cyl_passed

    if elmer_mk is not None:
        if elmer_cyl is not None:
            print()
        br, hcb, mu_rec, _ = _linear_recoil_params()
        mk_payload, mk_passed = _format_comparison(
            "N42 magnet + dual TEAM 20 steel keepers (R=10 mm, L=20 mm + 2x6 mm keepers, "
            f"probe=(0,0,0) Bz, box={BOX_EXTENT:g} m)",
            elmer_mk,
            run_magwave_magnet_keeper(),
            args.tol,
            note=(
                f"Comparison mode: linear-vs-linear on the PM (Br={br:g} T, HcB={hcb:g} A/m, "
                f"mu_rec={mu_rec:.6f}, Magnetization={hcb:g} A/m in Elmer) because Elmer's native "
                "permanent-magnet model is linear recoil; the nonlinear second-quadrant curve "
                "is magwave-only (keepers remain nonlinear TEAM 20 in both codes)."
            ),
        )
        case_payloads["magnet-keeper"] = mk_payload
        all_passed = all_passed and mk_passed

    if args.json:
        if len(case_payloads) == 1:
            json_out: dict[str, object] = next(iter(case_payloads.values()))
        else:
            json_out = {"cases": case_payloads, "passed": all_passed, "tol": args.tol}
        Path(args.json).write_text(
            json.dumps(json_out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
