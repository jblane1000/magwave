"""Nonlinear axisymmetric FEM solver, consistent Jacobian, and convergence tests (M5.2 Stage B)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import magwave.cases
import magwave.fem2d.engine
from magwave.cases import Case, Coil, ExpectedValue, Permeable, Probe
from magwave.engines import ConvergenceInfo, EngineError, Fem2dEngine
from magwave.fem2d import (
    E_NOT_CONVERGED,
    assemble_nonlinear_system,
    count_interface_nodes,
    element_field,
    solve_axisymmetric_nonlinear,
)
from magwave.fem2d.solve import dirichlet_nodes
from magwave.materials import MaterialsLibrary, load_library
from magwave.units import MU0


def _saturating_cylinder_case(*, current: float = 1.0) -> Case:
    return Case(
        name="nonlinear_team20_cylinder_solenoid",
        description="TEAM 20 steel cylinder inside coaxial thick solenoid",
        sources=("Compumag TEAM 20 B-H table (magwave.materials v1 steel-team20)",),
        components=(
            Permeable(
                kind="cylinder",
                dims={"radius": 0.010, "length": 0.060},
                center=(0.0, 0.0, 0.0),
                material="steel-team20",
                name="steel_core",
            ),
            Coil(
                kind="solenoid",
                dims={"inner_radius": 0.015, "outer_radius": 0.025, "length": 0.100},
                current=current,
                turns=1000,
                center=(0.0, 0.0, 0.0),
                name="excitation_coil",
            ),
        ),
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
        expected=(
            ExpectedValue(
                check="nonlinear_fem_reference",
                quantity="Bz",
                value=0.143836,
                unit="T",
                tolerance=0.03,
                source="Elmer MagnetoDynamics2D cross-check",
            ),
        ),
    )


def test_jacobian_matches_central_differences() -> None:
    case = _saturating_cylinder_case(current=10.0)
    engine = Fem2dEngine(element_size=0.005, material_model="nonlinear")
    mesh, load, _ = engine._build_mesh_and_load(case)
    assignments, _ = engine._permeable_assignments(mesh, case)
    constrained = dirichlet_nodes(mesh)
    element_curves = [(mask, mat.bh_curve()) for _, mask, mat, _ in assignments]

    n_nodes = mesh.nodes.shape[0]
    is_constrained = np.zeros(n_nodes, dtype=bool)
    is_constrained[constrained] = True
    free_idx = np.flatnonzero(~is_constrained)

    # First verify |B_e| == 0 guard: A = 0 must assemble a finite Jacobian with zero numpy warnings.
    with np.errstate(all="raise"):
        r0, j0, b0, nu0 = assemble_nonlinear_system(
            mesh, np.zeros(n_nodes, dtype=np.float64), load, element_curves, constrained=constrained
        )
    assert np.all(b0 == 0.0)
    assert np.all(np.isfinite(r0))
    assert np.all(np.isfinite(j0.data))
    assert np.all(nu0 > 0.0)

    # Perturb the converged saturating solution so elements sit across the B-H curve (~0.5..2.5 T)
    # without crossing negative B or zero-field kinks.
    a_sol, _, _, _ = solve_axisymmetric_nonlinear(
        mesh, load, element_curves, constrained=constrained
    )
    rng = np.random.default_rng(0)
    a_pert = a_sol.copy()
    a_pert[free_idx] += 0.05 * float(np.max(np.abs(a_sol))) * rng.standard_normal(free_idx.size)

    _, j_mat, b_mag, _ = assemble_nonlinear_system(
        mesh, a_pert, load, element_curves, constrained=constrained
    )
    steel_mask = assignments[0][1]
    assert float(np.max(b_mag[steel_mask])) > 1.0

    # Consistent co-energy Hessian is symmetric.
    asym = j_mat - j_mat.T
    rel_asym = float(np.linalg.norm(asym.data) / np.linalg.norm(j_mat.data))
    assert rel_asym <= 1e-12

    eps = 1e-7 * max(float(np.linalg.norm(a_pert)), 1.0)
    print("\nJacobian central-difference check (5 random directions):")
    rel_errors: list[float] = []
    for direction_idx in range(5):
        v = np.zeros(n_nodes, dtype=np.float64)
        v[free_idx] = rng.standard_normal(free_idx.size)
        v /= float(np.linalg.norm(v))

        jv_analytic = j_mat @ v
        r_plus, _, _, _ = assemble_nonlinear_system(
            mesh, a_pert + eps * v, load, element_curves, constrained=constrained
        )
        r_minus, _, _, _ = assemble_nonlinear_system(
            mesh, a_pert - eps * v, load, element_curves, constrained=constrained
        )
        jv_fd = (r_plus - r_minus) / (2.0 * eps)

        rel_err = float(np.linalg.norm(jv_analytic - jv_fd) / np.linalg.norm(jv_fd))
        rel_errors.append(rel_err)
        print(f"  direction {direction_idx + 1}: rel_err = {rel_err:.3e}")
        assert rel_err <= 1e-6


def test_saturating_cylinder_converges_and_reports() -> None:
    case = _saturating_cylinder_case(current=1.0)
    engine = Fem2dEngine(material_model="nonlinear")

    mesh, _, _ = engine._build_mesh_and_load(case)
    _, region_ids = engine._permeable_assignments(mesh, case)
    expected_iface = count_interface_nodes(mesh, region_ids)

    result = engine.solve(case)
    conv = result.convergence
    assert isinstance(conv, ConvergenceInfo)
    assert 3 <= conv.iterations <= 50
    assert conv.final_relative_residual <= 1e-8
    assert conv.nodes == mesh.nodes.shape[0]
    assert conv.elements == mesh.elements.shape[0]
    assert conv.interface_nodes == expected_iface
    assert conv.interface_nodes > 0
    assert len(conv.residual_history) == conv.iterations
    assert conv.residual_history[-1] == conv.final_relative_residual
    assert "Armijo" in conv.relaxation
    assert "Newton" in conv.solver
    assert "steel-team20" in conv.material_model

    print(
        f"\nSaturating cylinder convergence (I=1.0 A, nodes={conv.nodes}, "
        f"elements={conv.elements}, interface_nodes={conv.interface_nodes}):"
    )
    print(f"{'iter':>4} | {'rel_residual':>14}")
    print("-" * 21)
    for idx, res_val in enumerate(conv.residual_history, start=1):
        print(f"{idx:>4d} | {res_val:>14.6e}")

    hist = conv.residual_history
    for idx in range(1, len(hist) - 1):
        assert hist[idx + 1] < hist[idx], (
            f"residual did not decrease monotonically at iter {idx + 1} -> {idx + 2}: {hist}"
        )


def test_iteration_cap_fails_loudly() -> None:
    case = _saturating_cylinder_case(current=10.0)
    engine = Fem2dEngine(material_model="nonlinear", max_iterations=1)

    with pytest.raises(EngineError) as excinfo:
        engine.solve(case)

    problems = excinfo.value.problems
    assert len(problems) == 1
    assert problems[0].code == E_NOT_CONVERGED
    msg = problems[0].message
    assert case.name in msg
    assert "cap=1" in msg
    assert "final_relative_residual=" in msg


def test_solution_is_a_fixed_point_of_the_material_law() -> None:
    for current in (1.0, 10.0):
        case = _saturating_cylinder_case(current=current)
        engine = Fem2dEngine(material_model="nonlinear")
        mesh, load, _ = engine._build_mesh_and_load(case)
        assignments, _ = engine._permeable_assignments(mesh, case)
        constrained = dirichlet_nodes(mesh)
        element_curves = [(mask, mat.bh_curve()) for _, mask, mat, _ in assignments]

        a_phi, report, _, nu_final = solve_axisymmetric_nonlinear(
            mesh, load, element_curves, constrained=constrained
        )

        b_vec = element_field(mesh, a_phi)
        b_mag = np.hypot(b_vec[:, 0], b_vec[:, 1])
        nu_recomputed = np.full(mesh.elements.shape[0], 1.0 / MU0, dtype=np.float64)
        for mask, curve in element_curves:
            nu_recomputed[mask] = curve.reluctivity(b_mag[mask])

        max_rel_nu_dev = float(np.max(np.abs(nu_recomputed - nu_final) / nu_final))
        assert max_rel_nu_dev <= 1e-8
        assert report.relative_residual <= 1e-8

        r_check, _, _, _ = assemble_nonlinear_system(
            mesh, a_phi, load, element_curves, constrained=constrained
        )
        f_free = load.copy()
        f_free[constrained] = 0.0
        rel_res_check = float(np.linalg.norm(r_check) / np.linalg.norm(f_free))
        assert rel_res_check <= 1e-8


def test_current_sweep_saturates() -> None:
    currents = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
    curve = load_library().get("steel-team20").bh_curve()
    engine = Fem2dEngine(material_model="nonlinear")

    b_vals: list[float] = []
    mu_diff_vals: list[float] = []
    rel_mu_vals: list[float] = []

    print("\nCurrent sweep on TEAM 20 cylinder + solenoid:")
    header = (
        f"{'I (A)':>7} | {'probe Bz (T)':>12} | {'dB/dH (T*m/A)':>14} | "
        f"{'(dB/dH)/mu0':>11} | {'iters':>5} | {'rel_res':>10}"
    )
    print(header)
    print("-" * len(header))

    for current in currents:
        case = _saturating_cylinder_case(current=current)
        result = engine.solve(case)
        assert result.convergence is not None
        bz = float(result.values()[0])
        mu_diff = float(curve.differential_permeability(abs(bz)))
        rel_mu = mu_diff / MU0

        b_vals.append(bz)
        mu_diff_vals.append(mu_diff)
        rel_mu_vals.append(rel_mu)

        print(
            f"{current:>7.1f} | {bz:>12.6f} | {mu_diff:>14.6e} | "
            f"{rel_mu:>11.4f} | {result.convergence.iterations:>5d} | "
            f"{result.convergence.final_relative_residual:>10.3e}"
        )

    assert all(b2 > b1 for b1, b2 in zip(b_vals[:-1], b_vals[1:], strict=True))
    assert b_vals[-1] > curve.b_last  # exceeds 2.30 T -> exercises saturation extrapolation
    assert mu_diff_vals[-1] < mu_diff_vals[0]
    assert rel_mu_vals[-1] <= 3.0

    # TEAM 20 has an initial Rayleigh permeability rise on the low-field toe (< ~0.35 T)
    # before falling strictly monotonically past the knee into deep saturation.
    peak_idx = int(np.argmax(mu_diff_vals))
    assert b_vals[peak_idx] < 0.5
    for idx in range(peak_idx, len(mu_diff_vals) - 1):
        assert mu_diff_vals[idx + 1] < mu_diff_vals[idx]
    print(
        f"monotonicity verdict: final (dB/dH)/mu0 = {rel_mu_vals[-1]:.4f} < first "
        f"{rel_mu_vals[0]:.4f} (<= 3.0), probe Bmax={b_vals[-1]:.4f} T > 2.30 T, "
        f"strictly decreasing past Rayleigh peak at I={currents[peak_idx]:g} A "
        f"(Bz={b_vals[peak_idx]:.4f} T)."
    )


def test_nonlinear_rejects_a_material_without_bh_table(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _saturating_cylinder_case(current=1.0)
    test_lib = MaterialsLibrary.from_dict(
        {
            "schema": "magwave.materials/1",
            "version": 1,
            "materials": [
                {
                    "name": "linear-only-iron",
                    "mu_r": 120.0,
                    "bh_table": None,
                    "saturation_B": None,
                    "provenance": "unit test linear-only material without B-H table",
                }
            ],
        }
    )
    monkeypatch.setattr(magwave.materials, "load_library", lambda: test_lib)
    monkeypatch.setattr(magwave.fem2d.engine, "load_library", lambda: test_lib)

    linear_only_case = replace(
        base,
        name="linear_only_iron_case",
        components=(
            Permeable(
                kind="cylinder",
                dims={"radius": 0.010, "length": 0.060},
                center=(0.0, 0.0, 0.0),
                material="linear-only-iron",
                name="core",
            ),
            base.components[1],
        ),
    )

    nl_engine = Fem2dEngine(material_model="nonlinear")
    errors = [problem for problem in nl_engine.validate(linear_only_case) if problem.is_error]
    assert len(errors) == 1
    assert errors[0].code == "E_UNSUPPORTED_MATERIAL"
    assert "linear-only-iron" in errors[0].message
    assert "no B-H table" in errors[0].message

    with pytest.raises(EngineError) as excinfo:
        nl_engine.solve(linear_only_case)
    assert [p.code for p in excinfo.value.problems] == ["E_UNSUPPORTED_MATERIAL"]

    # Meanwhile the linear solver still handles linear-only materials in 1 direct step.
    lin_result = Fem2dEngine(material_model="linear").solve(linear_only_case)
    assert lin_result.convergence is not None
    assert lin_result.convergence.iterations == 1
