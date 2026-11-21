"""M5.3 FR-020 permanent-magnet demagnetisation gate tests (Gates 1 through 8)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from magwave.analytic import cylinder_Bz_axis, thick_solenoid_Bz
from magwave.cases import Case, Coil, ExpectedValue, Magnet, Permeable, Probe
from magwave.engines import EngineError, Fem2dEngine, MagpylibEngine, Result
from magwave.fem2d import (
    E_NOT_CONVERGED,
    assemble_nonlinear_system,
    load_magnetization_offset,
)
from magwave.fem2d.assembly import element_geometry
from magwave.fem2d.engine import E_UNSUPPORTED_MATERIAL
from magwave.fem2d.solve import dirichlet_nodes
from magwave.materials import DemagCurve, demag_curve_for, load_magnet_library
from magwave.report import build_report_payload
from magwave.units import MU0

ROOT = Path(__file__).resolve().parents[1]
R_MAG = 0.010
L_MAG = 0.020
BR_N42 = 1.315


def _n42_magnet(*, demag_curve: str | None = "N42-20C", br_z: float = BR_N42) -> Magnet:
    return Magnet(
        kind="cylinder",
        dims={"radius": R_MAG, "length": L_MAG},
        br=(0.0, 0.0, br_z),
        center=(0.0, 0.0, 0.0),
        demag_curve=demag_curve,
        name="n42_magnet",
    )


def _n42_air_case(
    *,
    demag_curve: str | None = "N42-20C",
    temperature_c: float | None = None,
    probes: tuple[Probe, ...] | None = None,
) -> Case:
    probe_list = probes or (
        Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),
        Probe(point=(0.0, 0.0, 0.030), quantity="B", component="z"),
    )
    return Case(
        name="m53_n42_air",
        description="N42 cylinder magnet in air",
        sources=("M5.3 FR-020 gate test",),
        components=(_n42_magnet(demag_curve=demag_curve),),
        probes=probe_list,
        expected=tuple(
            ExpectedValue(
                check="cylinder_Bz_axis",
                quantity="Bz",
                value=1.0,
                unit="T",
                tolerance=0.05,
                source="M5.3 test",
            )
            for _ in probe_list
        ),
        temperature_c=temperature_c,
    )


def _n42_keeper_case(*, demag_curve: str | None = "N42-20C") -> Case:
    return Case(
        name="m53_n42_keeper",
        description="N42 cylinder magnet with dual TEAM 20 steel pole-face keepers",
        sources=("M5.3 FR-020 gate test",),
        components=(
            _n42_magnet(demag_curve=demag_curve),
            Permeable(
                kind="cylinder",
                dims={"radius": R_MAG, "length": 0.006},
                center=(0.0, 0.0, 0.013),
                material="steel-team20",
                name="keeper_top",
            ),
            Permeable(
                kind="cylinder",
                dims={"radius": R_MAG, "length": 0.006},
                center=(0.0, 0.0, -0.013),
                material="steel-team20",
                name="keeper_bottom",
            ),
        ),
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
        expected=(
            ExpectedValue(
                check="nonlinear_pm_demag_fem",
                quantity="Bz",
                value=1.06,
                unit="T",
                tolerance=0.05,
                source="M5.3 test",
            ),
        ),
    )


def _linear_vacuum_demag_curve(br: float = BR_N42) -> DemagCurve:
    """Synthetic linear second-quadrant curve with exact slope ``1/MU0`` (test-only)."""
    h_min = -br / MU0
    h_pts = np.linspace(1.1 * h_min, 0.0, 9)
    pts = [[float(h), float(br + MU0 * h)] for h in h_pts]
    pts[-1] = [0.0, float(br)]
    return DemagCurve(pts, name="SYNTH-LINEAR-MU0", br_ref=br)


def test_uniform_br_path_unchanged() -> None:
    """G1: a magnet with demag_curve=None uses the exact existing uniform-Br path."""
    z_pts = (0.0, 0.5 * L_MAG, L_MAG, 2.0 * L_MAG)
    probes = tuple(Probe(point=(0.0, 0.0, z), quantity="B", component="z") for z in z_pts)
    case = _n42_air_case(demag_curve=None, probes=probes)

    h = 5.0e-4
    engine = Fem2dEngine(
        element_size=h,
        far_field_factor=25.0,
        refinement_box=(0.0, R_MAG + 2 * h, -0.5 * L_MAG - 2 * h, 2.0 * L_MAG + 2 * h),
    )
    res1 = engine.solve(case)
    res2 = engine.solve(case)

    assert res1.method == engine.method
    assert res1.problems == ()
    assert res1.convergence is not None
    assert res1.convergence.iterations == 1
    assert res1.convergence.magnet_operating_points == ()

    # Compare the NUMERIC payload, never the serialised JSON text: engine_version is
    # provenance, so a version bump must not be able to fail this physics test. Probe
    # values are compared with exact float equality and every convergence number
    # (counts, residual, residual history) with the same strictness as before.
    def _numeric_payload(result: Result) -> dict[str, object]:
        payload = build_report_payload(case, result, case_file="tests/test_m5_3_pm_demag.py")
        convergence = payload["convergence"]
        return {
            "probes": [
                {"point": row["point"], "computed": row["computed"]} for row in payload["probes"]
            ],
            "convergence": {
                key: convergence[key]
                for key in (
                    "elements",
                    "final_relative_residual",
                    "interface_nodes",
                    "iterations",
                    "nodes",
                    "residual_history",
                )
            },
        }

    assert _numeric_payload(res1) == _numeric_payload(res2)

    m_unif = BR_N42 / MU0
    for z, computed in zip(z_pts, res1.values(), strict=True):
        exact = cylinder_Bz_axis(R_MAG, -0.5 * L_MAG, 0.5 * L_MAG, m_unif, z)
        assert abs(computed - exact) / abs(exact) <= 1e-2

    # Ensure report payload omits magnet_operating_points for non-demag cases
    dipole_case = Case.from_json(
        (ROOT / "examples" / "solve-dipole.json").read_text(encoding="utf-8")
    )
    dipole_res = MagpylibEngine().solve(dipole_case)
    dipole_payload = build_report_payload(
        dipole_case, dipole_res, case_file="examples/solve-dipole.json"
    )
    assert "convergence" not in dipole_payload

    steel_case = Case.from_json(
        (ROOT / "examples" / "steel-cylinder-solenoid-nonlinear.json").read_text(encoding="utf-8")
    )
    steel_res = Fem2dEngine(material_model="nonlinear").solve(steel_case)
    steel_payload = build_report_payload(
        steel_case, steel_res, case_file="examples/steel-cylinder-solenoid-nonlinear.json"
    )
    assert "magnet_operating_points" not in steel_payload["convergence"]


def test_g2_datasheet_fidelity() -> None:
    """G2: shipped N42-20C and Y30H-1-20C reproduce datasheet Br, HcB, and (BH)max within 2 %."""
    lib = load_magnet_library()
    specs = (
        ("N42-20C", 1.315, 860_000.0, 334_000.0),
        ("Y30H-1-20C", 0.390, 252_000.0, 28_000.0),
    )

    print("\nGate 2 — Shipped permanent-magnet datasheet fidelity:")
    header = (
        f"{'Grade':<12} | {'Quantity':<10} | {'Datasheet':>14} | {'Computed':>14} | "
        f"{'Rel err':>9} | {'Gate':>6}"
    )
    print(header)
    print("-" * len(header))

    for name, ds_br, ds_hcb, ds_bhmax in specs:
        curve = lib.get(name).demag_curve()
        comp_br = curve.br()
        comp_hcb = abs(curve.hcb())
        comp_bhmax = curve.bh_product_max()
        rows = (
            ("Br (T)", ds_br, comp_br),
            ("HcB (A/m)", ds_hcb, comp_hcb),
            ("(BH)max", ds_bhmax, comp_bhmax),
        )
        for qty, ds_val, comp_val in rows:
            rel_err = abs(comp_val - ds_val) / ds_val
            status = "PASS" if rel_err <= 0.02 else "FAIL"
            print(
                f"{name:<12} | {qty:<10} | {ds_val:>14.6g} | {comp_val:>14.6g} | "
                f"{rel_err:>8.4%} | {status:>6}"
            )
            assert rel_err <= 0.02

    n42_curve = lib.get("N42-20C").demag_curve()
    b_at_hcj = float(n42_curve.b_of_h(-955_000.0))
    assert abs(b_at_hcj - MU0 * (-955_000.0)) <= 1e-3


def test_g3_analytic_cylinder_and_exact_uniform_br_reduction() -> None:
    """G3a & G3b: synthetic 1/mu0 curve matches analytic cylinder <= 1 % and uniform-Br <= 1e-9."""
    synth = _linear_vacuum_demag_curve(BR_N42)
    z_offsets = (0.5 * L_MAG, L_MAG, 2.0 * L_MAG, 5.0 * L_MAG)
    probes = tuple(Probe(point=(0.0, 0.0, z), quantity="B", component="z") for z in z_offsets)
    demag_case = _n42_air_case(demag_curve="SYNTH-LINEAR-MU0", probes=probes)
    unif_case = _n42_air_case(demag_curve=None, probes=probes)

    m_unif = BR_N42 / MU0
    exact_vals = [cylinder_Bz_axis(R_MAG, -0.5 * L_MAG, 0.5 * L_MAG, m_unif, z) for z in z_offsets]

    print("\nGate 3a — Analytic cylinder vs synthetic 1/mu0 demag curve (R=10 mm, L=20 mm):")
    header = (
        f"{'h (m)':>9} | {'z/L':>5} | {'z (m)':>8} | {'Computed (T)':>13} | "
        f"{'Analytic (T)':>13} | {'Rel err':>9}"
    )
    print(header)
    print("-" * len(header))

    max_errs: list[float] = []
    for h in (1.0e-3, 5.0e-4):
        eng = Fem2dEngine(
            element_size=h,
            far_field_factor=25.0,
            refinement_box=(0.0, R_MAG + 2 * h, -0.5 * L_MAG - 2 * h, 5.0 * L_MAG + 2 * h),
            magnet_curves={"SYNTH-LINEAR-MU0": synth},
        )
        res_demag = eng.solve(demag_case)
        errs = []
        for z, comp, exact in zip(z_offsets, res_demag.values(), exact_vals, strict=True):
            rel_err = abs(comp - exact) / abs(exact)
            errs.append(rel_err)
            print(
                f"{h:>9.1e} | {z / L_MAG:>5.1f} | {z:>8.4f} | {comp:>13.6e} | "
                f"{exact:>13.6e} | {rel_err:>8.4%}"
            )
            assert rel_err <= 1e-2
        max_errs.append(max(errs))

    assert max_errs[1] < max_errs[0]

    # Gate 3b: exact offset == 0.0 and <= 1e-9 agreement with demag_curve=None
    h_ref = 5.0e-4
    eng_fine = Fem2dEngine(
        element_size=h_ref,
        far_field_factor=25.0,
        refinement_box=(0.0, R_MAG + 2 * h_ref, -0.5 * L_MAG - 2 * h_ref, 5.0 * L_MAG + 2 * h_ref),
        magnet_curves={"SYNTH-LINEAR-MU0": synth},
    )
    mesh, _, _ = eng_fine._build_mesh_and_load(demag_case)
    _, r_c, z_c, _, _ = element_geometry(mesh)
    mag_mask = (r_c <= R_MAG) & (np.abs(z_c) <= 0.5 * L_MAG)
    b_sample = np.linspace(0.1, 1.2, int(np.count_nonzero(mag_mask)))
    _, _, c_vals, _ = synth.slope_and_offset(b_sample)
    c0 = -synth.br() / MU0
    correction = load_magnetization_offset(mesh, mag_mask, c0 - c_vals)
    assert np.all(correction == 0.0)

    res_demag_fine = eng_fine.solve(demag_case)
    res_unif_fine = eng_fine.solve(unif_case)
    rel_diffs = [
        abs(d - u) / abs(u)
        for d, u in zip(res_demag_fine.values(), res_unif_fine.values(), strict=True)
    ]
    print(
        f"Gate 3b — max |demag - uniform_Br| / |uniform_Br| = {max(rel_diffs):.3e} (gate <= 1e-9)"
    )
    assert max(rel_diffs) <= 1e-9

    # Informational note (not gated): real N42 has mu_rec ≈ 1.217 != 1
    n42_grade = load_magnet_library().get("N42-20C")
    mu_rec_n42 = n42_grade.br_ref / (MU0 * n42_grade.hcb)
    real_n42_case = _n42_air_case(demag_curve="N42-20C", probes=probes)
    real_n42_res = eng_fine.solve(real_n42_case)
    real_diff_0 = abs(real_n42_res.values()[0] - exact_vals[0]) / abs(exact_vals[0])
    print(
        f"Note (informational): real N42-20C has mu_rec = {mu_rec_n42:.4f} != 1; "
        f"Bz(0.5L) = {real_n42_res.values()[0]:.6f} T vs closed-form (mu_rec=1) "
        f"{exact_vals[0]:.6f} T (rel diff {real_diff_0:.3%})."
    )


def test_g4_operating_point_moves() -> None:
    """G4: operating point (H, B) rises with a keeper and drops under an opposing coil."""
    curve = demag_curve_for("N42-20C", 20.0)
    engine = Fem2dEngine(element_size=5.0e-4, material_model="nonlinear")

    # (a) Alone in air
    air_case = _n42_air_case(
        demag_curve="N42-20C",
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
    )
    res_air = engine.solve(air_case)
    b_air = float(res_air.values()[0])
    h_air = float(curve.h_of_b(b_air))

    # (b) Beside dual TEAM 20 keepers
    keeper_case = _n42_keeper_case(demag_curve="N42-20C")
    res_keeper = engine.solve(keeper_case)
    b_keeper = float(res_keeper.values()[0])
    h_keeper = float(curve.h_of_b(b_keeper))

    # (c) Driven by an opposing coaxial thick solenoid giving Hz = -3.0e5 A/m at origin
    r1, r2, l_coil, turns = 0.015, 0.025, 0.060, 1000
    bz_per_j = thick_solenoid_Bz(r1, r2, -0.5 * l_coil, 0.5 * l_coil, 1.0, 0.0)
    j_opp = -3.0e5 * MU0 / bz_per_j
    current_opp = j_opp * ((r2 - r1) * l_coil) / turns
    hz_analytic = thick_solenoid_Bz(r1, r2, -0.5 * l_coil, 0.5 * l_coil, j_opp, 0.0) / MU0
    assert hz_analytic == pytest.approx(-3.0e5, rel=1e-12)

    opp_coil = Coil(
        kind="solenoid",
        dims={"inner_radius": r1, "outer_radius": r2, "length": l_coil},
        current=current_opp,
        turns=turns,
        center=(0.0, 0.0, 0.0),
        name="opposing_coil",
    )
    opp_case_demag = replace(
        air_case,
        name="m53_n42_opposing_coil",
        components=(_n42_magnet(demag_curve="N42-20C"), opp_coil),
    )
    opp_case_unif = replace(
        opp_case_demag,
        components=(_n42_magnet(demag_curve=None), opp_coil),
    )

    res_opp_demag = engine.solve(opp_case_demag)
    res_opp_unif = engine.solve(opp_case_unif)
    b_opp = float(res_opp_demag.values()[0])
    h_opp = float(curve.h_of_b(b_opp))
    b_opp_unif = float(res_opp_unif.values()[0])

    print("\nGate 4 — Operating point motion on N42-20C at (0, 0, 0):")
    print(f"  (a) Air:           H = {h_air:11.1f} A/m, B = {b_air:.6f} T")
    print(
        f"  (b) Keeper:        H = {h_keeper:11.1f} A/m, B = {b_keeper:.6f} T "
        f"(corroboration: keeper raises load line, dB = {b_keeper - b_air:+.6f} T)"
    )
    print(
        f"  (c) Opposing coil: H = {h_opp:11.1f} A/m, B = {b_opp:.6f} T "
        f"(uniform-Br = {b_opp_unif:.6f} T, rel diff = {abs(b_opp - b_opp_unif) / abs(b_opp_unif):.3%})"
    )

    assert b_keeper > b_air
    assert b_opp < b_air
    assert abs(b_opp - b_opp_unif) / abs(b_opp_unif) > 0.02
    assert abs(float(curve.b_of_h(h_opp)) - b_opp) <= 1e-3 * curve.br()

    assert res_opp_demag.convergence is not None
    op = res_opp_demag.convergence.magnet_operating_points[0]
    assert op.b_t == pytest.approx(b_opp, rel=1e-12)
    assert op.h_a_per_m == pytest.approx(h_opp, rel=1e-12)


def test_g5_temperature_and_ferrite_sign_flip() -> None:
    """G5: 100 C scales N42 Br by alpha_Br within 0.5 % and flips ferrite HcJ sign upward."""
    n42_20 = demag_curve_for("N42-20C", 20.0)
    n42_100 = demag_curve_for("N42-20C", 100.0)
    expected_br_100 = 1.315 * (1.0 - 0.0012 * 80.0)
    rel_br_err = abs(n42_100.br() - expected_br_100) / expected_br_100
    assert rel_br_err <= 5e-3

    engine = Fem2dEngine(element_size=5.0e-4)
    case_20 = _n42_air_case(
        demag_curve="N42-20C",
        temperature_c=20.0,
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
    )
    case_100 = replace(case_20, temperature_c=100.0)
    res_20 = engine.solve(case_20)
    res_100 = engine.solve(case_100)

    b_20 = float(res_20.values()[0])
    b_100 = float(res_100.values()[0])
    assert b_100 < b_20
    assert res_100.convergence is not None
    assert "100" in res_100.method
    assert "100" in res_100.convergence.material_model
    assert f"{n42_100.br():.6g}" in res_100.method

    ferrite_20 = demag_curve_for("Y30H-1-20C", 20.0)
    ferrite_100 = demag_curve_for("Y30H-1-20C", 100.0)
    assert ferrite_100.br() < ferrite_20.br()
    assert ferrite_100.hcj > ferrite_20.hcj  # positive alpha_hcj_per_c sign flip
    assert abs(ferrite_100.hcb()) > abs(ferrite_20.hcb())

    ferrite_case_20 = replace(
        case_20,
        components=(
            Magnet(
                kind="cylinder",
                dims={"radius": R_MAG, "length": L_MAG},
                br=(0.0, 0.0, 0.390),
                demag_curve="Y30H-1-20C",
                name="ferrite_magnet",
            ),
        ),
    )
    ferrite_case_100 = replace(ferrite_case_20, temperature_c=100.0)
    b_fer_20 = float(engine.solve(ferrite_case_20).values()[0])
    b_fer_100 = float(engine.solve(ferrite_case_100).values()[0])
    assert b_fer_100 < b_fer_20

    print("\nGate 5 — Temperature scaling (20 C -> 100 C):")
    print(
        f"  N42-20C:    Br(20)={n42_20.br():.5f} T -> Br(100)={n42_100.br():.5f} T "
        f"(expected {expected_br_100:.5f} T, err={rel_br_err:.3e}); B_center: {b_20:.6f} -> {b_100:.6f} T"
    )
    print(
        f"  Y30H-1-20C: Br: {ferrite_20.br():.5f} -> {ferrite_100.br():.5f} T; "
        f"HcJ: {ferrite_20.hcj:.0f} -> {ferrite_100.hcj:.0f} A/m (sign flip: +); "
        f"B_center: {b_fer_20:.6f} -> {b_fer_100:.6f} T"
    )


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker binary not found on PATH")
def test_g6_elmer_magnet_keeper() -> None:
    """G6: Elmer MagnetoDynamics2D linear-recoil cross-check on magnet+keeper within 3 %."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "elmer_permeable.py"), "--case", "magnet-keeper"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if proc.returncode == 3:
        skip_line = next(
            (line for line in proc.stdout.splitlines() if line.startswith("SKIP:")),
            proc.stdout.strip() or proc.stderr.strip() or "Elmer unavailable",
        )
        pytest.skip(skip_line)

    if proc.returncode != 0:
        pytest.fail(
            f"tools/elmer_permeable.py --case magnet-keeper failed (exit {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    assert "linear-vs-linear" in proc.stdout
    assert "PASS" in proc.stdout
    match = re.search(r"([0-9]+\.[0-9]+)%\s*\|.*\|\s*PASS\b", proc.stdout)
    assert match is not None, f"could not parse relative diff from output:\n{proc.stdout}"
    assert float(match.group(1)) <= 3.0


def test_g7_nonconvergence_honesty_and_cli_no_output_file(tmp_path: Path) -> None:
    """G7: non-converged solve raises E_NOT_CONVERGED, emits no Result, and creates no --out file."""
    case = _n42_air_case(demag_curve="N42-20C")
    with pytest.raises(EngineError) as excinfo:
        Fem2dEngine(max_iterations=1).solve(case)
    assert [p.code for p in excinfo.value.problems] == [E_NOT_CONVERGED]

    case_path = tmp_path / "n42_air.json"
    out_path = tmp_path / "should_not_exist.md"
    case_path.write_text(case.to_json(), encoding="utf-8")

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "magwave",
            "report",
            str(case_path),
            "--engine",
            "fem2d",
            "--max-iterations",
            "1",
            "--out",
            str(out_path),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert proc.stdout == ""
    assert E_NOT_CONVERGED in proc.stderr
    assert not out_path.exists()


def test_demag_jacobian_matches_central_differences() -> None:
    """Verify exact Newton Jacobian (both ds/dB and dc/dB terms) with N42 + TEAM 20 steel."""
    case = _n42_keeper_case(demag_curve="N42-20C")
    engine = Fem2dEngine(element_size=2.0e-3, material_model="nonlinear")
    mesh, load, _ = engine._build_mesh_and_load(case)
    assignments, region_ids = engine._permeable_assignments(mesh, case)
    demag_assignments = engine._magnet_demag_assignments(mesh, case, region_ids)
    constrained = dirichlet_nodes(mesh)
    element_curves = [(mask, mat.bh_curve()) for _, mask, mat, _ in assignments]
    magnet_regions = [(mask, curve) for _, _, mask, curve, _, _ in demag_assignments]

    res = engine.solve(case)
    assert res.convergence is not None
    # Solve raw nodal vector around operating point and perturb it
    from magwave.fem2d.nonlinear import solve_axisymmetric_nonlinear

    a_sol, _, _, _ = solve_axisymmetric_nonlinear(
        mesh,
        load,
        element_curves,
        constrained=constrained,
        magnet_regions=magnet_regions,
    )
    n_nodes = mesh.nodes.shape[0]
    is_c = np.zeros(n_nodes, dtype=bool)
    is_c[constrained] = True
    free_idx = np.flatnonzero(~is_c)

    rng = np.random.default_rng(7)
    a_pert = a_sol.copy()
    a_pert[free_idx] += 0.02 * float(np.max(np.abs(a_sol))) * rng.standard_normal(free_idx.size)
    _, j_mat, _, _ = assemble_nonlinear_system(
        mesh,
        a_pert,
        load,
        element_curves,
        constrained=constrained,
        magnet_regions=magnet_regions,
    )

    eps = 1e-7 * max(float(np.linalg.norm(a_pert)), 1.0)
    for _ in range(3):
        v = np.zeros(n_nodes, dtype=np.float64)
        v[free_idx] = rng.standard_normal(free_idx.size)
        v /= float(np.linalg.norm(v))
        jv = j_mat @ v
        r_plus, _, _, _ = assemble_nonlinear_system(
            mesh,
            a_pert + eps * v,
            load,
            element_curves,
            constrained=constrained,
            magnet_regions=magnet_regions,
        )
        r_minus, _, _, _ = assemble_nonlinear_system(
            mesh,
            a_pert - eps * v,
            load,
            element_curves,
            constrained=constrained,
            magnet_regions=magnet_regions,
        )
        jv_fd = (r_plus - r_minus) / (2.0 * eps)
        rel_err = float(np.linalg.norm(jv - jv_fd) / np.linalg.norm(jv_fd))
        assert rel_err <= 1e-6


def test_unknown_demag_grade_and_examples_roundtrip_and_tier0_refusal(tmp_path: Path) -> None:
    """Unknown grade is E_UNSUPPORTED_MATERIAL; shipped examples round-trip, solve, and refuse Tier 0."""
    bad_case = _n42_air_case(demag_curve="UNOBTAINIUM-99")
    engine = Fem2dEngine()
    errors = [p for p in engine.validate(bad_case) if p.is_error]
    assert len(errors) == 1
    assert errors[0].code == E_UNSUPPORTED_MATERIAL
    assert "UNOBTAINIUM-99" in errors[0].message
    assert "N42-20C" in errors[0].message
    assert "Y30H-1-20C" in errors[0].message
    assert "permanent_magnet_demag_curve" in engine.capabilities().materials

    tier0 = MagpylibEngine()
    for fname in ("magnet-n42-air.json", "magnet-n42-keeper.json"):
        path = ROOT / "examples" / fname
        raw = path.read_text(encoding="utf-8")
        loaded = Case.from_json(raw)
        assert json.loads(loaded.to_json()) == json.loads(raw)

        # Tier 0 refuses demag cases honestly
        with pytest.raises(EngineError) as t0_exc:
            tier0.solve(loaded)
        assert any(p.code == "E_UNSUPPORTED_DEMAG_CURVE" for p in t0_exc.value.problems)

        # Tier 1 solves via CLI report and emits magnet_operating_points in Markdown + JSON
        out_md = tmp_path / f"{fname}.md"
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "magwave",
                "report",
                str(path),
                "--engine",
                "fem2d",
                "--out",
                str(out_md),
            ],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        text = out_md.read_text(encoding="utf-8")
        assert "magnet_operating_point[n42_magnet]:" in text
        match = re.search(r"```json\n(.*?)\n```\s*\Z", text, flags=re.DOTALL)
        assert match is not None
        payload = json.loads(match.group(1))
        assert len(payload["convergence"]["magnet_operating_points"]) == 1
