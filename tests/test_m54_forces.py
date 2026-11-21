"""M5.4 force, torque refusal, and Tier 0 permeable force refusal gate tests."""

from __future__ import annotations

import math
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
    """Gate 2: Nagaoka-corrected solenoid inductance agreement within 0.5%.

    Framing: ``raw_dev`` compares ``L_fem = 2 W' / I^2`` against the elementary ideal
    long-solenoid formula ``L_ideal = mu_0 * N^2 * pi * R^2 / l`` and therefore measures the
    error of the ideal formula (which omits finite-length end effects at ``l/D = 10``,
    ``l/R = 20``; the correction belongs to the reference, not the FEM). ``corr_dev`` compares
    ``L_fem`` against the exact Nagaoka/Lorenz thin-sheet reference ``L_ref = K_N * L_ideal``
    (basis: ``K_N(l/D=10) = 0.95880712``).
    """
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


def test_gate3_coenergy_finite_difference_step_sensitivity() -> None:
    """Gate 3 coenergy step sensitivity: the finite-difference step must not dominate the 2% method-agreement gate."""
    steel_case = build_magnet_plate_case(0.0)
    h = 5.0e-4
    eng = _steel_plate_engine(h)

    res_m = maxwell_force(steel_case, eng, target="magnet", contour=GATE3_CONTOUR)
    res_c_h = coenergy_force(steel_case, eng, target="magnet", delta=h)
    res_c_h2 = coenergy_force(steel_case, eng, target="magnet", delta=0.5 * h)

    f_m = res_m.force_n[2]
    f_h = res_c_h.force_n[2]
    f_h2 = res_c_h2.force_n[2]
    assert f_h > 0.0 and f_h2 > 0.0

    step_diff = abs(f_h - f_h2) / max(abs(f_h), abs(f_h2))
    assert step_diff <= 0.02
    assert abs(f_h - f_m) / max(abs(f_h), abs(f_m)) <= 0.02
    assert abs(f_h2 - f_m) / max(abs(f_h2), abs(f_m)) <= 0.02


@pytest.fixture(scope="module")
def _gate3_delta_study_and_maxwell() -> tuple[DeltaStudyResult, float, float, float]:
    """Run the Gate 3 steel delta-convergence study and Maxwell reference once per module."""
    h = 5.0e-4
    steel_case = build_magnet_plate_case(0.0)
    z_before = [float(comp.center[2]) for comp in steel_case.components]
    eng = _steel_plate_engine(h)

    res_m = maxwell_force(steel_case, eng, target="magnet", contour=GATE3_CONTOUR)
    study = coenergy_delta_study(steel_case, eng, target="magnet")

    z_after = [float(comp.center[2]) for comp in steel_case.components]
    assert z_after == z_before
    return study, float(res_m.force_n[2]), h, z_before[0]


def test_coenergy_delta_study_synthetic_order_and_asymptotic() -> None:
    """Synthetic O(delta^2) ladder verifies positive order p=2 and asymptotic=True; non-monotonic gives False."""
    h = 5.0e-4
    quad_steps = tuple(
        DeltaStep(
            label=label,
            delta_m=delta_m,
            delta_over_h=delta_m / h,
            force_n=100.0 + 10.0 * (delta_m / h) ** 2,
            mesh_nodes=1000,
            mesh_elements=1800,
        )
        for label, delta_m in (("h/8", h / 8.0), ("h/4", h / 4.0), ("h/2", h / 2.0), ("h", h))
    )
    quad = _analyze_delta_ladder("magnet", h, quad_steps, ())
    assert quad.monotonic is True
    assert quad.asymptotic is True
    assert quad.p_fit == pytest.approx(2.0, rel=1e-9, abs=1e-9)
    for p_val in quad.p_per_pair:
        assert p_val == pytest.approx(2.0, rel=1e-9, abs=1e-9)
    assert quad.f_extrapolated_n == pytest.approx(100.0, rel=1e-9, abs=1e-9)
    assert quad.step_uncertainty_n == pytest.approx(
        abs(quad_steps[0].force_n - 100.0), rel=1e-9, abs=1e-9
    )

    non_mono_steps = tuple(
        DeltaStep(
            label=label,
            delta_m=delta_m,
            delta_over_h=delta_m / h,
            force_n=force_n,
            mesh_nodes=1000,
            mesh_elements=1800,
        )
        for label, delta_m, force_n in (
            ("h/8", h / 8.0, 100.0),
            ("h/4", h / 4.0, 101.0),
            ("h/2", h / 2.0, 100.5),
            ("h", h, 102.0),
        )
    )
    non_mono = _analyze_delta_ladder("magnet", h, non_mono_steps, ())
    assert non_mono.monotonic is False
    assert non_mono.asymptotic is False
    assert None in non_mono.p_per_pair
    assert any("not in its asymptotic range" in note for note in non_mono.notes)


def test_coenergy_delta_study_gate3_contract_determinism_and_refusal(
    _gate3_delta_study_and_maxwell: tuple[DeltaStudyResult, float, float, float],
) -> None:
    """Gate 3 delta study satisfies ladder, uncertainty, mesh, determinism, and clearance refusal contracts."""
    study, _, h, _ = _gate3_delta_study_and_maxwell
    assert len(study.steps) >= 4
    deltas = [step.delta_m for step in study.steps]
    assert deltas == sorted(deltas)
    assert deltas[-1] / deltas[0] >= 8.0 - 1e-12

    forces = [step.force_n for step in study.steps]
    assert all(math.isfinite(f) and f > 0.0 for f in forces)
    assert all(step.mesh_nodes > 0 and step.mesh_elements > 0 for step in study.steps)
    assert len({step.mesh_nodes for step in study.steps}) >= 2

    assert isinstance(study.asymptotic, bool)
    if study.p_fit is None:
        assert len(study.notes) > 0
    else:
        assert math.isfinite(study.p_fit)
    for p_val in study.p_per_pair:
        if p_val is not None:
            assert math.isfinite(p_val)

    assert study.f_extrapolated_n is not None
    assert math.isfinite(study.f_extrapolated_n)
    assert 0.9 * min(forces) <= study.f_extrapolated_n <= 1.1 * max(forces)
    assert study.step_uncertainty_n > 0.0
    assert study.peak_to_peak_n > 0.0

    # Determinism across consecutive runs
    steel_case = build_magnet_plate_case(0.0)
    eng = _steel_plate_engine(h)
    study_repeat = coenergy_delta_study(steel_case, eng, target="magnet")
    assert repr(study_repeat) == repr(study)

    # Clearance refusal: delta = 2h = 1.0e-3 m equals the 1 mm axial gap
    with pytest.raises(ForceError) as ref_exc:
        coenergy_force(steel_case, eng, target="magnet", delta=2.0 * h)
    assert ref_exc.value.code == E_FORCE_UNSUPPORTED_CASE

    refused_study = coenergy_delta_study(
        steel_case,
        eng,
        target="magnet",
        deltas=(0.5 * h, 2.0 * h),
    )
    assert len(refused_study.steps) == 1
    assert math.isclose(refused_study.steps[0].delta_m, 0.5 * h, rel_tol=1e-12)
    assert len(refused_study.refused) == 1
    assert math.isclose(refused_study.refused[0][0], 2.0 * h, rel_tol=1e-12)
    assert E_FORCE_UNSUPPORTED_CASE in refused_study.refused[0][1]


def test_coenergy_delta_study_gate3_agreement_gate(
    _gate3_delta_study_and_maxwell: tuple[DeltaStudyResult, float, float, float],
) -> None:
    """Gate 3 unchanged 2% Maxwell-vs-coenergy agreement evaluated on delta-extrapolated coenergy."""
    study, f_maxwell, _, _ = _gate3_delta_study_and_maxwell
    assert study.f_extrapolated_n is not None
    f_extrap = study.f_extrapolated_n
    agree = abs(f_maxwell - f_extrap) / max(abs(f_maxwell), abs(f_extrap))
    assert agree <= 0.02, (
        f"Gate 3 Maxwell ({f_maxwell:.5f} N) vs delta-extrapolated coenergy "
        f"({f_extrap:.5f} N +/- {study.step_uncertainty_n:.5f} N step band) disagreement "
        f"is {100.0 * agree:.4f}% > 2.0000%"
    )
