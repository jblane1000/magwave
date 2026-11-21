"""M5.4 force, torque refusal, and Tier 0 permeable force refusal gate tests."""

from __future__ import annotations

import os
import re
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tools.elmer_force import (
    GATE3_CONTOUR,
    build_magnet_plate_case,
)
from tools.elmer_permeable import LINEAR_RECOIL_GRADE, _linear_recoil_params
from tools.force_report import (
    CONTOUR_B,
    build_coaxial_loops_case,
    build_solenoid_case,
    d_mutual_inductance_dd,
    mutual_inductance_loops,
    nagaoka_factor,
    neumann_coaxial_sheet_force,
)

from magwave.cad.manifest import Body, Manifest
from magwave.cases import Case, ExpectedValue, Magnet, Probe
from magwave.cli import main
from magwave.engines import EngineError, Fem2dEngine, MagpylibEngine
from magwave.fem2d import (
    E_FORCE_CONTOUR_DOES_NOT_ENCLOSE,
    E_FORCE_CONTOUR_IN_MATERIAL,
    E_FORCE_DELTA_NOT_ALIGNED,
    E_FORCE_TARGET_AMBIGUOUS,
    E_FORCE_TARGET_UNKNOWN,
    E_FORCE_UNSUPPORTED_CASE,
    E_TORQUE_NOT_MEANINGFUL,
    DeltaStep,
    DeltaStudyResult,
    ForceError,
    coenergy_delta_study,
    coenergy_force,
    magnetic_coenergy,
    maxwell_force,
    torque,
)
from magwave.fem2d.force import _analyze_delta_ladder
from magwave.materials import load_library
from magwave.units import MU0

ROOT = Path(__file__).resolve().parents[1]


def _steel_plate_engine(h: float = 5.0e-4) -> Fem2dEngine:
    _, _, _, linear_curve = _linear_recoil_params()
    return Fem2dEngine(
        element_size=h,
        material_model="nonlinear",
        magnet_curves={LINEAR_RECOIL_GRADE: linear_curve},
    )


@pytest.fixture(scope="module")
def _gate3_steel_results_h5e4():
    steel_case = build_magnet_plate_case(0.0)
    eng = _steel_plate_engine(5.0e-4)
    res_m = maxwell_force(steel_case, eng, target="magnet", contour=GATE3_CONTOUR)
    res_c = coenergy_force(steel_case, eng, target="magnet")
    return res_m, res_c


@pytest.fixture(scope="module")
def _gate3_aligned_study_h2_5e4() -> DeltaStudyResult:
    steel_case = build_magnet_plate_case(0.0)
    eng = _steel_plate_engine(2.5e-4)
    return coenergy_delta_study(steel_case, eng, target="magnet")


def test_bh_curve_magnetic_energy_density_matches_derivative() -> None:
    curve = load_library().get("steel-team20").bh_curve()
    b_vals = np.array([0.0, 0.005, 0.0095, 0.1, 0.8, 1.8, 2.3, 2.34, 2.5], dtype=np.float64)
    w_vals = curve.magnetic_energy_density(b_vals)
    c_vals = curve.coenergy_density(b_vals)
    assert w_vals[0] == 0.0
    assert np.all(np.diff(w_vals) > 0.0)
    # Verify d w / dB == H(B) across non-zero points using central difference
    eps = 1e-7
    dw_db = (
        curve.magnetic_energy_density(b_vals[1:] + eps)
        - curve.magnetic_energy_density(b_vals[1:] - eps)
    ) / (2.0 * eps)
    np.testing.assert_allclose(dw_db, curve.field_strength(b_vals[1:]), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        w_vals + c_vals, b_vals * curve.field_strength(b_vals), rtol=1e-12, atol=1e-12
    )


def test_gate1_neumann_reference_and_two_coaxial_loops() -> None:
    """Gate 1: Neumann self-check + <= 1% at h=5e-4 & 2.5e-4, with error falling."""
    a = 0.010
    d = 0.010
    h_fd = 1e-6
    dm_fd = (
        -mutual_inductance_loops(a, a, d + 2 * h_fd)
        + 8.0 * mutual_inductance_loops(a, a, d + h_fd)
        - 8.0 * mutual_inductance_loops(a, a, d - h_fd)
        + mutual_inductance_loops(a, a, d - 2 * h_fd)
    ) / (12.0 * h_fd)
    dm_closed = float(d_mutual_inductance_dd(a, a, d))
    assert abs(dm_closed - dm_fd) / abs(dm_closed) <= 1e-10

    f_thin = neumann_coaxial_sheet_force(a, a, 1e-5, 1e-5, d, 1.0, 1.0, n_gauss=48)
    assert abs(f_thin - dm_closed) / abs(dm_closed) <= 1e-4

    f_ref = neumann_coaxial_sheet_force(a, a, 0.002, 0.002, d, 1.0, 1.0, n_gauss=48)
    assert f_ref < 0.0

    case = build_coaxial_loops_case()
    snapshot = case.to_json()
    errs_m: list[float] = []
    errs_c: list[float] = []
    for h in (5.0e-4, 2.5e-4):
        eng = Fem2dEngine(element_size=h)
        rm = maxwell_force(case, eng, target="coil_top")
        rc = coenergy_force(case, eng, target="coil_top")
        assert rm.force_n[0] == 0.0 and rm.force_n[1] == 0.0
        assert rc.force_n[0] == 0.0 and rc.force_n[1] == 0.0
        assert rm.force_n[2] < 0.0
        assert rc.force_n[2] < 0.0
        err_m = abs(rm.force_n[2] - f_ref) / abs(f_ref)
        err_c = abs(rc.force_n[2] - f_ref) / abs(f_ref)
        assert err_m <= 0.01
        assert err_c <= 0.01
        errs_m.append(err_m)
        errs_c.append(err_c)

    assert errs_m[1] < errs_m[0]
    assert errs_c[1] < errs_c[0]
    assert case.to_json() == snapshot


def test_gate2_solenoid_inductance_from_coenergy() -> None:
    """Gate 2: L_fem = 2 W' / I^2 agrees with Nagaoka-corrected L_ideal within 0.5%."""
    case = build_solenoid_case()
    eng = Fem2dEngine(element_size=5.0e-4)
    w_co, _, _ = magnetic_coenergy(case, eng)

    r_s, l_s, n_s, i_s = 0.010, 0.200, 400, 1.0
    l_fem = 2.0 * w_co / (i_s * i_s)
    l_ideal = MU0 * (n_s**2) * (np.pi * r_s * r_s) / l_s
    k_n = nagaoka_factor(r_s, l_s)
    l_exact = k_n * l_ideal

    raw_dev = abs(l_fem - l_ideal) / l_ideal
    corr_dev = abs(l_fem - l_exact) / l_exact
    assert raw_dev > 0.005  # confirms end-effect correction is physically required at l/D=10
    assert corr_dev <= 0.005


def test_gate3_and_gate5_and_gate7_steel_case() -> None:
    """Gates 3, 5, 7: magnet+steel agreement <= 2%, contour independence <= 2%, determinism."""
    case = build_magnet_plate_case(0.0)
    snapshot = case.to_json()
    eng = _steel_plate_engine(5.0e-4)

    res_a = eng.force(case, method="maxwell", target="magnet", contour=GATE3_CONTOUR)
    res_c = eng.force(case, method="coenergy", target="magnet")
    assert res_a.method == "maxwell_stress_tensor"
    assert res_c.method == "coenergy_virtual_work"
    assert res_a.force_n[2] > 0.0
    assert res_c.force_n[2] > 0.0
    assert res_a.convergence is not None
    assert len(res_a.convergence.magnet_operating_points) == 1
    assert res_a.contour is not None
    assert any("snapped contour=" in note for note in res_a.notes)

    agree = abs(res_a.force_n[2] - res_c.force_n[2]) / max(
        abs(res_a.force_n[2]), abs(res_c.force_n[2])
    )
    assert agree <= 0.02

    # Gate 5: contour B independence on same mesh
    res_b = maxwell_force(case, eng, target="magnet", contour=CONTOUR_B)
    assert res_b.mesh_nodes == res_a.mesh_nodes
    assert res_b.mesh_elements == res_a.mesh_elements
    diff_ab = abs(res_a.force_n[2] - res_b.force_n[2]) / max(
        abs(res_a.force_n[2]), abs(res_b.force_n[2])
    )
    assert diff_ab <= 0.02

    # Gate 7: bit-identical determinism
    res_a2 = maxwell_force(
        case, _steel_plate_engine(5.0e-4), target="magnet", contour=GATE3_CONTOUR
    )
    res_c2 = coenergy_force(case, _steel_plate_engine(5.0e-4), target="magnet")
    assert repr(res_a) == repr(res_a2)
    assert repr(res_c) == repr(res_c2)
    assert struct.pack("!3d", *res_a.force_n) == struct.pack("!3d", *res_a2.force_n)
    assert struct.pack("!3d", *res_c.force_n) == struct.pack("!3d", *res_c2.force_n)
    assert case.to_json() == snapshot


def test_torque_refusal_and_force_error_contracts() -> None:
    """Torque refuses cleanly; target/contour/direction validation raises exact ForceError codes."""
    steel_case = build_magnet_plate_case(0.0)
    loops_case = build_coaxial_loops_case()
    eng = _steel_plate_engine(5.0e-4)

    with pytest.raises(ForceError) as t_exc:
        torque(steel_case, eng, target="magnet")
    assert t_exc.value.code == E_TORQUE_NOT_MEANINGFUL
    t_msg = t_exc.value.message.lower()
    assert "tangential stress" in t_msg
    assert "identically zero by construction" in t_msg
    assert "not a computed zero" in t_msg
    assert "3d" in t_msg

    with pytest.raises(ForceError) as t_exc2:
        eng.torque(steel_case)
    assert t_exc2.value.code == E_TORQUE_NOT_MEANINGFUL

    # Ambiguous omitted target on 2-coil case
    with pytest.raises(ForceError) as amb_exc:
        maxwell_force(loops_case, Fem2dEngine(element_size=1e-3))
    assert amb_exc.value.code == E_FORCE_TARGET_AMBIGUOUS
    assert "coil_bottom" in amb_exc.value.message and "coil_top" in amb_exc.value.message

    # Passive permeable target refused
    with pytest.raises(ForceError) as perm_exc:
        maxwell_force(steel_case, eng, target="steel_plate")
    assert perm_exc.value.code == E_FORCE_TARGET_UNKNOWN
    assert "Permeable" in perm_exc.value.message

    # Unknown target refused
    with pytest.raises(ForceError) as unk_exc:
        coenergy_force(steel_case, eng, target="does_not_exist")
    assert unk_exc.value.code == E_FORCE_TARGET_UNKNOWN

    # Contour through steel plate refused
    with pytest.raises(ForceError) as mat_steel_exc:
        maxwell_force(steel_case, eng, target="magnet", contour=(0.012, -0.011, 0.015))
    assert mat_steel_exc.value.code == E_FORCE_CONTOUR_IN_MATERIAL

    # Contour snapping onto magnet face at z=0.010 refused (never silently integrates on interface)
    with pytest.raises(ForceError) as snap_exc:
        maxwell_force(steel_case, eng, target="magnet", contour=(0.012, -0.011, 0.0101))
    assert snap_exc.value.code == E_FORCE_CONTOUR_IN_MATERIAL
    assert "snapped" in snap_exc.value.message.lower()

    # Contour with insufficient radial margin (<= R + h) refused
    with pytest.raises(ForceError) as rad_exc:
        maxwell_force(steel_case, eng, target="magnet", contour=(0.0104, -0.011, 0.0105))
    assert rad_exc.value.code == E_FORCE_CONTOUR_DOES_NOT_ENCLOSE

    # Contour enclosing both magnet and steel plate refused
    with pytest.raises(ForceError) as both_exc:
        maxwell_force(steel_case, eng, target="magnet", contour=(0.035, -0.012, 0.025))
    assert both_exc.value.code == E_FORCE_CONTOUR_DOES_NOT_ENCLOSE

    # Non-axial virtual displacement refused
    with pytest.raises(ForceError) as dir_exc:
        coenergy_force(steel_case, eng, target="magnet", direction="x")
    assert dir_exc.value.code == E_FORCE_UNSUPPORTED_CASE

    # Unsupported geometry (cuboid magnet) refused
    cuboid_case = Case(
        name="cuboid_force",
        description="non-axisymmetric cuboid",
        sources=("unit test",),
        components=(
            Magnet(
                kind="cuboid",
                dims={"size_x": 0.01, "size_y": 0.01, "size_z": 0.02},
                br=(0.0, 0.0, 1.0),
                name="cuboid_mag",
            ),
        ),
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
        expected=(
            ExpectedValue(
                check="test", quantity="Bz", value=1.0, unit="T", tolerance=0.1, source="test"
            ),
        ),
    )
    with pytest.raises(ForceError) as geom_exc:
        maxwell_force(cuboid_case, eng)
    assert geom_exc.value.code == E_FORCE_UNSUPPORTED_CASE

    # F-probe refusal in Fem2dEngine.validate points to Fem2dEngine.force
    f_probe_case = replace(
        steel_case,
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="F", component="z"),),
    )
    errs = [p for p in eng.validate(f_probe_case) if p.is_error]
    assert len(errs) == 1
    assert errs[0].code == "E_UNSUPPORTED_QUANTITY"
    assert "Fem2dEngine.force" in errs[0].message
    assert "nothing was solved" in errs[0].message.lower()


def test_tier0_permeable_force_refusal_gate(tmp_path: Path) -> None:
    """Tier 0 emits E_UNSUPPORTED_FORCE_PERMEABLE when F probe + permeable material co-occur."""
    base_steel = build_magnet_plate_case(0.0)
    force_steel = replace(
        base_steel,
        name="tier0_force_steel",
        components=(
            replace(base_steel.components[0], demag_curve=None),
            base_steel.components[1],
        ),
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="F", component="z"),),
    )

    tier0 = MagpylibEngine()
    errors = [p for p in tier0.validate(force_steel) if p.is_error]
    assert [p.code for p in errors] == ["E_UNSUPPORTED_FORCE_PERMEABLE"]
    msg = errors[0].message
    assert "force" in msg.lower()
    assert "permeable" in msg.lower()
    assert "tier 1" in msg.lower() and "fem2d" in msg.lower()
    assert msg.endswith("nothing was solved.")

    with pytest.raises(EngineError) as excinfo:
        tier0.solve(force_steel)
    assert [p.code for p in excinfo.value.problems] == ["E_UNSUPPORTED_FORCE_PERMEABLE"]

    # Also test permeable_solid manifest body + F probe
    manifest = Manifest(
        path="yoke.step",
        format="step",
        units="mm",
        bodies=[
            Body(
                index=0,
                name="yoke",
                role="permeable_solid",
                volume_mm3=1000.0,
                solids=1,
                bbox=(-10.0, -10.0, -5.0, 10.0, 10.0, 5.0),
                source="step",
            )
        ],
    )
    manifest_force_case = replace(
        force_steel,
        name="tier0_force_manifest_steel",
        components=(replace(base_steel.components[0], demag_curve=None),),
        manifest=manifest,
    )
    m_errors = [p for p in tier0.validate(manifest_force_case) if p.is_error]
    assert [p.code for p in m_errors] == ["E_UNSUPPORTED_FORCE_PERMEABLE"]

    # Free-space F probe stays E_UNSUPPORTED_QUANTITY and names tier
    freespace_force = replace(
        force_steel,
        name="tier0_freespace_force",
        components=(replace(base_steel.components[0], demag_curve=None),),
    )
    fs_errors = [p for p in tier0.validate(freespace_force) if p.is_error]
    assert [p.code for p in fs_errors] == ["E_UNSUPPORTED_QUANTITY"]
    assert "Tier" in fs_errors[0].message

    # CLI solve prints no numbers on stdout
    case_path = tmp_path / "force_steel.json"
    case_path.write_text(force_steel.to_json(), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    proc = subprocess.run(
        [sys.executable, "-m", "magwave", "solve", str(case_path), "--points", "0,0,0"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert proc.stdout.strip() == ""
    assert not any(re.match(r"^\s*[-+0-9.]", ln) for ln in proc.stdout.splitlines())
    assert "E_UNSUPPORTED_FORCE_PERMEABLE" in proc.stderr
    assert main(["solve", str(case_path), "--points", "0,0,0"]) != 0


def test_force_refuses_tighter_tolerance_without_weakening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case tolerance < certified_accuracy refuses cleanly without mutating or solving."""
    steel_case = build_magnet_plate_case(0.0)
    eng = _steel_plate_engine(5.0e-4)

    tight_case = replace(
        steel_case,
        expected=tuple(replace(item, tolerance=1e-3) for item in steel_case.expected),
        tolerance=1e-3,
    )

    def _forbid_solve(self: Fem2dEngine, case: Case) -> object:
        raise AssertionError("solver must not run when tolerance is tighter than certified")

    monkeypatch.setattr(Fem2dEngine, "_solve_core", _forbid_solve)

    for call in (
        lambda: maxwell_force(tight_case, eng, target="magnet", contour=GATE3_CONTOUR),
        lambda: coenergy_force(tight_case, eng, target="magnet"),
    ):
        with pytest.raises(ForceError) as excinfo:
            call()
        assert excinfo.value.code == E_FORCE_UNSUPPORTED_CASE
        msg = excinfo.value.message
        msg_lower = msg.lower()
        assert tight_case.name in msg
        assert "0.1%" in msg and "1.00e-03" in msg
        assert "1%" in msg and "1.00e-02" in msg
        assert "tier 1 force is not claimed" in msg_lower
        assert "relax" in msg_lower and "tighter b tolerance" in msg_lower
        assert "nothing was solved." in msg_lower

    assert tight_case.tolerance == 1e-3
    assert tight_case.expected[0].tolerance == 1e-3

    monkeypatch.undo()

    boundary_case = replace(
        steel_case,
        expected=tuple(replace(item, tolerance=1e-2) for item in steel_case.expected),
        tolerance=1e-2,
    )
    res_m = maxwell_force(boundary_case, eng, target="magnet", contour=GATE3_CONTOUR)
    res_c = coenergy_force(boundary_case, eng, target="magnet")
    assert res_m.force_n[2] > 0.0
    assert res_c.force_n[2] > 0.0
    agree = abs(res_m.force_n[2] - res_c.force_n[2]) / max(
        abs(res_m.force_n[2]), abs(res_c.force_n[2])
    )
    assert agree <= 0.02
    assert boundary_case.tolerance == 1e-2


def test_coenergy_refuses_non_aligned_delta_and_insufficient_clearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sub-cell/non-aligned delta raises E_FORCE_DELTA_NOT_ALIGNED before solving or clearance."""
    steel_case = build_magnet_plate_case(0.0)
    h = 5.0e-4
    eng = _steel_plate_engine(h)

    def _forbid_solve(self: Fem2dEngine, case: Case) -> object:
        raise AssertionError("solver must not run when delta is refused")

    monkeypatch.setattr(Fem2dEngine, "_solve_core", _forbid_solve)

    # Non-aligned deltas (including 2.5*h which also exceeds the 1 mm clearance) must raise
    # E_FORCE_DELTA_NOT_ALIGNED before the axial-clearance check runs.
    for bad_delta in (0.5 * h, 1.5 * h, 2.5 * h):
        with pytest.raises(ForceError) as excinfo:
            coenergy_force(steel_case, eng, target="magnet", delta=bad_delta)
        assert excinfo.value.code == E_FORCE_DELTA_NOT_ALIGNED
        msg = excinfo.value.message
        assert steel_case.name in msg
        assert f"{h:g}" in msg or f"{h:.3e}" in msg
        assert "integer multiple" in msg.lower()
        assert msg.endswith("nothing was solved.")

    # When h equals the 1 mm gap, no aligned k*h fits (default and explicit h refuse cleanly).
    eng_coarse = _steel_plate_engine(1.0e-3)
    for delta_arg in (None, 1.0e-3):
        with pytest.raises(ForceError) as clr_exc:
            coenergy_force(steel_case, eng_coarse, target="magnet", delta=delta_arg)
        assert clr_exc.value.code == E_FORCE_UNSUPPORTED_CASE
        clr_msg = clr_exc.value.message
        assert "1.00e-03" in clr_msg or "0.001" in clr_msg
        assert clr_msg.endswith("nothing was solved.")


def test_gate3_aligned_study_h5e4_refuses_2h_3h() -> None:
    """At h = 5e-4 m (1 mm gap), Study A accepts k=1 and refuses 2h and 3h cleanly."""
    steel_case = build_magnet_plate_case(0.0)
    eng = _steel_plate_engine(5.0e-4)
    study = coenergy_delta_study(steel_case, eng, target="magnet")

    assert len(study.steps) == 1
    assert len(study.refused) == 2
    step = study.steps[0]
    assert step.label == "h"
    assert step.delta_m == pytest.approx(5.0e-4, rel=1e-12)
    assert step.delta_over_h == pytest.approx(1.0, rel=1e-12)
    assert step.mesh_nodes_plus == study.base_mesh_nodes
    assert step.mesh_elements_plus == study.base_mesh_elements
    assert step.mesh_nodes_minus == study.base_mesh_nodes
    assert step.mesh_elements_minus == study.base_mesh_elements
    assert step.gap_cells_plus == 1
    assert step.gap_cells_minus == 3
    assert step.gap_cells_plus + step.gap_cells_minus == study.expected_gap_cell_sum == 4
    assert study.mesh_counts_invariant is True
    assert study.gap_cell_sum_invariant is True
    assert study.alignment_invariant_holds is True
    assert study.f_extrapolated_n is None
    for _, reason in study.refused:
        assert E_FORCE_UNSUPPORTED_CASE in reason


def test_gate3_aligned_ladder_h2_5e4(_gate3_aligned_study_h2_5e4: DeltaStudyResult) -> None:
    """At h = 2.5e-4 m, aligned (h, 2h, 3h) runs 3 steps with identical meshes and exact gap sum."""
    study = _gate3_aligned_study_h2_5e4

    assert len(study.steps) == 3
    assert len(study.refused) == 0
    assert [s.label for s in study.steps] == ["h", "2h", "3h"]
    assert study.expected_gap_cell_sum == 8

    expected_splits = [(3, 5), (2, 6), (1, 7)]
    for step, (exp_plus, exp_minus) in zip(study.steps, expected_splits, strict=True):
        assert step.mesh_nodes_plus == study.base_mesh_nodes
        assert step.mesh_elements_plus == study.base_mesh_elements
        assert step.mesh_nodes_minus == study.base_mesh_nodes
        assert step.mesh_elements_minus == study.base_mesh_elements
        assert step.gap_cells_plus == exp_plus
        assert step.gap_cells_minus == exp_minus
        assert (step.gap_cells_plus + step.gap_cells_minus) == study.expected_gap_cell_sum

    assert study.mesh_counts_invariant is True
    assert study.gap_cell_sum_invariant is True
    assert study.alignment_invariant_holds is True
    assert any("cell-alignment invariant HOLDS" in note for note in study.notes)


def test_gate3_aligned_delta_h_agrees_with_maxwell(_gate3_steel_results_h5e4) -> None:
    """Gate 3: aligned delta = h at h = 5e-4 m records alignment and agrees with Maxwell <= 2%."""
    res_m, res_c = _gate3_steel_results_h5e4
    f_m = float(res_m.force_n[2])
    f_c = float(res_c.force_n[2])
    assert res_c.delta_m == pytest.approx(5.0e-4, rel=1e-12)
    assert any("cell-aligned; no remeshing" in note for note in res_c.notes)
    assert f_m > 0.0 and f_c > 0.0
    agree = abs(f_m - f_c) / max(abs(f_m), abs(f_c))
    assert agree <= 0.02


def test_delta_ladder_analysis_exact_order_and_edge_cases() -> None:
    """Synthetic ladder checks: exact p=2.0 on arithmetic & geometric grids; None for <3 or non-monotonic."""
    h = 2.5e-4
    f0 = 101.5
    coeff = 1.2e6

    # Arithmetic (h, 2h, 3h) quadratic convergence F(d) = f0 + coeff * d^2
    arith_steps = tuple(
        DeltaStep(
            label="h" if k == 1 else f"{k}h",
            delta_m=k * h,
            delta_over_h=float(k),
            force_n=f0 + coeff * (k * h) ** 2,
            change_vs_prev_pct=None,
            mesh_nodes_plus=100,
            mesh_elements_plus=180,
            mesh_nodes_minus=100,
            mesh_elements_minus=180,
            gap_cells_plus=4 - k,
            gap_cells_minus=4 + k,
        )
        for k in (1, 2, 3)
    )
    arith_res = _analyze_delta_ladder(
        "magnet",
        h,
        arith_steps,
        (),
        base_mesh_counts=(100, 180),
        axial_gap_m=1.0e-3,
    )
    assert arith_res.monotonic is True
    assert arith_res.asymptotic is True
    assert arith_res.p_per_pair[0] == pytest.approx(2.0, rel=1e-9)
    assert arith_res.p_fit == pytest.approx(2.0, rel=1e-9)
    assert arith_res.f_extrapolated_n == pytest.approx(f0, rel=1e-12)
    assert arith_res.alignment_invariant_holds is True

    # Geometric (h, 2h, 4h, 8h) quadratic convergence
    geom_steps = tuple(
        DeltaStep(
            label="h" if k == 1 else f"{k}h",
            delta_m=k * h,
            delta_over_h=float(k),
            force_n=f0 + coeff * (k * h) ** 2,
            change_vs_prev_pct=None,
            mesh_nodes_plus=100,
            mesh_elements_plus=180,
            mesh_nodes_minus=100,
            mesh_elements_minus=180,
            gap_cells_plus=None,
            gap_cells_minus=None,
        )
        for k in (1, 2, 4, 8)
    )
    geom_res = _analyze_delta_ladder(
        "magnet",
        h,
        geom_steps,
        (),
        base_mesh_counts=(100, 180),
    )
    assert geom_res.asymptotic is True
    assert geom_res.p_fit == pytest.approx(2.0, rel=1e-9)
    assert geom_res.f_extrapolated_n == pytest.approx(f0, rel=1e-12)

    # Fewer than 3 rungs -> extrapolation unavailable
    two_res = _analyze_delta_ladder(
        "magnet",
        h,
        arith_steps[:2],
        (),
        base_mesh_counts=(100, 180),
        axial_gap_m=1.0e-3,
    )
    assert two_res.f_extrapolated_n is None
    assert two_res.asymptotic is False

    # Non-monotonic 3 rungs -> no positive observed order, extrapolation unavailable
    nonmono_steps = (
        arith_steps[0],
        DeltaStep(
            label="2h",
            delta_m=2 * h,
            delta_over_h=2.0,
            force_n=f0 + 0.5,
            change_vs_prev_pct=None,
            mesh_nodes_plus=100,
            mesh_elements_plus=180,
            mesh_nodes_minus=100,
            mesh_elements_minus=180,
            gap_cells_plus=2,
            gap_cells_minus=6,
        ),
        DeltaStep(
            label="3h",
            delta_m=3 * h,
            delta_over_h=3.0,
            force_n=f0 - 0.2,
            change_vs_prev_pct=None,
            mesh_nodes_plus=100,
            mesh_elements_plus=180,
            mesh_nodes_minus=100,
            mesh_elements_minus=180,
            gap_cells_plus=1,
            gap_cells_minus=7,
        ),
    )
    nonmono_res = _analyze_delta_ladder(
        "magnet",
        h,
        nonmono_steps,
        (),
        base_mesh_counts=(100, 180),
        axial_gap_m=1.0e-3,
    )
    assert nonmono_res.monotonic is False
    assert nonmono_res.asymptotic is False
    assert nonmono_res.f_extrapolated_n is None
