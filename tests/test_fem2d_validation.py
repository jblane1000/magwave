"""M5 Stage 1 validation gate: 1 % closed-form agreement & mesh-independence study."""

from __future__ import annotations

import numpy as np

from magwave.analytic import cylinder_Bz_axis, thick_solenoid_Bz
from magwave.cases import Case, Coil, ExpectedValue, Magnet, Probe
from magwave.fem2d.engine import Fem2dEngine
from magwave.units import MU0


def _solenoid_case() -> tuple[Case, np.ndarray, np.ndarray]:
    z_pts = np.linspace(-0.06, 0.06, 11)
    r1, r2, length = 0.010, 0.020, 0.100
    turns, current = 1000, 1.0
    j_phi = turns * current / ((r2 - r1) * length)
    exact = np.array([thick_solenoid_Bz(r1, r2, -0.05, 0.05, j_phi, float(z)) for z in z_pts])
    case = Case(
        name="validation_thick_solenoid",
        description="M5.1 validation solenoid across 11 on-axis stations",
        sources=("magwave.analytic.thick_solenoid_Bz",),
        components=(
            Coil(
                kind="solenoid",
                dims={"inner_radius": r1, "outer_radius": r2, "length": length},
                current=current,
                turns=turns,
                name="sol1",
            ),
        ),
        probes=tuple(Probe(point=(0.0, 0.0, float(z)), quantity="B", component="z") for z in z_pts),
        expected=(
            ExpectedValue(
                check="thick_solenoid_Bz",
                quantity="Bz",
                value=float(exact[5]),
                unit="T",
                tolerance=1e-2,
                source="magwave.analytic.thick_solenoid_Bz",
            ),
        ),
    )
    return case, z_pts, exact


def _cylinder_case() -> tuple[Case, np.ndarray, np.ndarray]:
    z_pts = np.linspace(-0.04, 0.04, 11)
    radius, length, br_z = 0.0125, 0.05, 1.2
    exact = np.array(
        [cylinder_Bz_axis(radius, -0.5 * length, 0.5 * length, br_z / MU0, float(z)) for z in z_pts]
    )
    case = Case(
        name="validation_cylinder_magnet",
        description="M5.1 validation uniformly magnetised cylinder across 11 on-axis stations",
        sources=("magwave.analytic.cylinder_Bz_axis",),
        components=(
            Magnet(
                kind="cylinder",
                dims={"radius": radius, "length": length},
                br=(0.0, 0.0, br_z),
                name="cyl1",
            ),
        ),
        probes=tuple(Probe(point=(0.0, 0.0, float(z)), quantity="B", component="z") for z in z_pts),
        expected=(
            ExpectedValue(
                check="cylinder_Bz_axis",
                quantity="Bz",
                value=float(exact[5]),
                unit="T",
                tolerance=1e-2,
                source="magwave.analytic.cylinder_Bz_axis",
            ),
        ),
    )
    return case, z_pts, exact


def _run_case_levels(case: Case, exact: np.ndarray, h_levels: tuple[float, ...]):
    rows = []
    for h in h_levels:
        engine = Fem2dEngine(element_size=h)
        result, report = engine.solve_with_report(case)
        computed = np.array(result.values(), dtype=np.float64)
        rel_errs = np.abs(computed - exact) / np.abs(exact)
        rows.append((h, report, rel_errs, float(np.max(rel_errs))))
    return rows


def test_solenoid_on_axis_1pct_gate() -> None:
    case, _, exact = _solenoid_case()
    levels = _run_case_levels(case, exact, (1e-3, 5e-4))
    max_err_1 = levels[0][3]
    max_err_2 = levels[1][3]
    assert max_err_1 <= 1e-2
    assert max_err_2 <= 1e-2
    assert max_err_2 < max_err_1


def test_cylinder_on_axis_1pct_gate() -> None:
    case, _, exact = _cylinder_case()
    levels = _run_case_levels(case, exact, (1e-3, 5e-4))
    max_err_1 = levels[0][3]
    max_err_2 = levels[1][3]
    assert max_err_1 <= 1e-2
    assert max_err_2 <= 1e-2
    assert max_err_2 < max_err_1


def test_mesh_independence_study(record_property) -> None:
    sol_case, sol_z, sol_exact = _solenoid_case()
    cyl_case, cyl_z, cyl_exact = _cylinder_case()

    sol_levels = _run_case_levels(sol_case, sol_exact, (1e-3, 5e-4))
    cyl_levels = _run_case_levels(cyl_case, cyl_exact, (1e-3, 5e-4))

    header = (
        f"{'case':<16} | {'h':>8} | {'nodes':>7} | {'elements':>8} | {'min edge':>10} | "
        f"{'max rel err':>11} | {'solve wall time':>15} | result"
    )
    lines: list[str] = ["", header, "-" * len(header)]
    for label, levels in (("solenoid", sol_levels), ("cylinder", cyl_levels)):
        for h, report, _, max_err in levels:
            status = "PASS" if max_err <= 1e-2 else "FAIL"
            lines.append(
                f"{label:<16} | {h:>8.4e} | {report.mesh.nodes:>7d} | {report.mesh.elements:>8d} | "
                f"{report.mesh.min_edge:>10.4e} | {max_err:>11.4e} | "
                f"{report.wall_time_s:>14.4f}s | {status}"
            )

    lines.append("")
    for label, z_pts, levels in (
        ("solenoid", sol_z, sol_levels),
        ("cylinder", cyl_z, cyl_levels),
    ):
        lines.append(f"Per-point relative errors ({label}):")
        lines.append("z [m]    | rel.err (h=1e-3) | rel.err (h=5e-4)")
        for z_val, e1, e2 in zip(z_pts, levels[0][2], levels[1][2], strict=True):
            lines.append(f"{z_val:+8.4f} | {e1:16.6e} | {e2:16.6e}")

    table = "\n".join(lines)
    print(table)
    # Land the numbers in the acceptance artifact itself: `pytest --junit-xml=...` records
    # properties even for passing tests, while captured stdout is only emitted on failure.
    record_property("mesh_independence_table", table)

    assert sol_levels[1][3] < sol_levels[0][3]
    assert cyl_levels[1][3] < cyl_levels[0][3]
    assert sol_levels[1][3] <= 1e-2
    assert cyl_levels[1][3] <= 1e-2
