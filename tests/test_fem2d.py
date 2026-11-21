"""Machinery, boundary, capability, and protocol tests for :mod:`magwave.fem2d`."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import magwave.cli
from magwave.analytic import cylinder_Bz_axis
from magwave.benchmarks import load_corpus
from magwave.cad.manifest import Body, Manifest
from magwave.cad.roles import assign
from magwave.cases import Case, Coil, ExpectedValue, Probe
from magwave.engines import Engine, EngineError, Fem2dEngine
from magwave.fem2d import (
    ENGINE_ID,
    RUNG,
    dirichlet_nodes,
    load_current_density,
    rectangular_mesh,
    sample_field,
    solve_axisymmetric,
    stiffness,
)
from magwave.fem2d.assembly import element_geometry
from magwave.units import MU0

CORPUS = {case.name: case for case in load_corpus()}


def test_mesh_stats_and_breakpoints() -> None:
    bps_r = (0.01, 0.025)
    bps_z = (-0.04, 0.0, 0.03)
    mesh = rectangular_mesh(
        0.10,
        -0.10,
        0.10,
        0.005,
        refinement_box=(0.0, 0.03, -0.05, 0.05),
        breakpoints_r=bps_r,
        breakpoints_z=bps_z,
        growth=1.2,
    )
    for r_bp in bps_r:
        assert np.any(np.isclose(mesh.r_axis, r_bp, rtol=0.0, atol=1e-14))
    for z_bp in bps_z:
        assert np.any(np.isclose(mesh.z_axis, z_bp, rtol=0.0, atol=1e-14))

    stats = mesh.stats()
    assert stats.nodes == mesh.nodes.shape[0]
    assert stats.elements == mesh.elements.shape[0]
    assert stats.h_requested == 0.005

    tri = mesh.nodes[mesh.elements]
    edge_lengths = np.concatenate(
        (
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        )
    )
    assert stats.min_edge == pytest.approx(float(np.min(edge_lengths)), rel=1e-14)
    assert stats.max_edge == pytest.approx(float(np.max(edge_lengths)), rel=1e-14)

    with pytest.raises(ValueError, match="max_nodes"):
        rectangular_mesh(0.10, -0.10, 0.10, 0.001, max_nodes=100)


def test_stiffness_is_symmetric_and_singular_before_constraints() -> None:
    mesh = rectangular_mesh(0.10, -0.10, 0.10, 0.02)
    k_mat = stiffness(mesh)
    diff = k_mat - k_mat.T
    rel_asym = float(np.linalg.norm(diff.data) / np.linalg.norm(k_mat.data))
    assert rel_asym <= 1e-12

    # In the axisymmetric curl-curl form, A_phi = const is NOT in the null space because
    # B_z = (1/r) d(r A)/dr = c/r != 0. Instead, 1^T K 1 equals the exact integral of
    # nu * (1/r)^2 * r dr dz = sum_e nu * A_e / r_c, while the pure grad-grad part sums to 0.
    area, r_c, _, _, _ = element_geometry(mesh)
    ones = np.ones(mesh.nodes.shape[0], dtype=np.float64)
    quad_ones = float(ones @ (k_mat @ ones))
    expected_energy = float(np.sum((1.0 / MU0) * area / r_c))
    assert abs(quad_ones - expected_energy) / expected_energy <= 1e-12

    # Off the axis (in an annular mesh without r == 0), the curl-free gauge mode A_phi = 1/r
    # has nearly zero energy in the unconstrained bilinear form, demonstrating why B_z = 0
    # modes exist off-axis until r == 0 or outer Dirichlet conditions pin A_phi.
    r_vals = np.linspace(0.5, 1.0, 41)
    z_vals = np.linspace(-0.25, 0.25, 41)
    rr, zz = np.meshgrid(r_vals, z_vals, indexing="ij")
    annulus_nodes = np.column_stack((rr.ravel(), zz.ravel()))
    nr, nz = len(r_vals), len(z_vals)
    ii, jj = np.meshgrid(np.arange(nr - 1), np.arange(nz - 1), indexing="ij")
    n00 = (ii * nz + jj).ravel()
    n10 = ((ii + 1) * nz + jj).ravel()
    n11 = ((ii + 1) * nz + (jj + 1)).ravel()
    n01 = (ii * nz + (jj + 1)).ravel()
    annulus_elems = np.vstack((np.column_stack((n00, n10, n11)), np.column_stack((n00, n11, n01))))
    annulus = replace(
        rectangular_mesh(1.0, -0.25, 0.25, 0.0125),
        nodes=annulus_nodes,
        elements=annulus_elems,
        r_axis=r_vals,
        z_axis=z_vals,
    )
    k_ann = stiffness(annulus)
    curl_free = 1.0 / annulus.nodes[:, 0]
    rayleigh = float(curl_free @ (k_ann @ curl_free)) / float(
        curl_free @ (k_ann.diagonal() * curl_free)
    )
    assert rayleigh < 1e-4


def test_axis_sample_is_finite() -> None:
    h = 0.002
    mesh = rectangular_mesh(
        0.52,
        -0.55,
        0.55,
        h,
        refinement_box=(0.0, 0.025, -0.055, 0.055),
        breakpoints_r=(0.01, 0.02),
        breakpoints_z=(-0.05, 0.05),
    )
    j_phi = 1000.0 / ((0.02 - 0.01) * 0.10)
    load = load_current_density(mesh, (0.01, 0.02, -0.05, 0.05), j_phi)
    a_phi, report = solve_axisymmetric(mesh, load)
    assert report.relative_residual <= 1e-10
    assert np.all(a_phi[dirichlet_nodes(mesh)] == 0.0)

    b_axis = sample_field(mesh, a_phi, np.array([[0.0, 0.0]]))[0]
    b_near = sample_field(mesh, a_phi, np.array([[h / 100.0, 0.0]]))[0]
    assert np.all(np.isfinite(b_axis))
    assert b_axis[0] == 0.0
    assert abs(b_axis[1] - b_near[1]) / abs(b_near[1]) < 1e-2


def test_determinism() -> None:
    case = Case(
        name="det_solenoid",
        description="determinism check",
        sources=("unit test",),
        components=(
            Coil(
                kind="solenoid",
                dims={"inner_radius": 0.01, "outer_radius": 0.02, "length": 0.10},
                current=1.0,
                turns=1000,
                name="w1",
            ),
        ),
        probes=(
            Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),
            Probe(point=(0.005, 0.0, 0.03), quantity="B", component="magnitude"),
        ),
        expected=(
            ExpectedValue(
                check="thick_solenoid_Bz",
                quantity="Bz",
                value=0.012,
                unit="T",
                tolerance=0.05,
                source="unit test",
            ),
        ),
    )
    res1 = Fem2dEngine(element_size=0.002).solve(case)
    res2 = Fem2dEngine(element_size=0.002).solve(case)
    b1 = np.array([s.vector for s in res1.samples])
    b2 = np.array([s.vector for s in res2.samples])
    assert np.array_equal(b1, b2)


def test_superposition_on_a_fixed_mesh() -> None:
    """Superposition holds to <= 1e-10 relative by linearity on one shared mesh.

    Note: this deliberately tests linear superposition on an identical fixed mesh (forced via
    ``domain``, ``refinement_box``, and ``element_size``), not a cross-mesh comparison.
    """
    coil_a = Coil(
        kind="solenoid",
        dims={"inner_radius": 0.01, "outer_radius": 0.02, "length": 0.04},
        current=2.0,
        turns=200,
        center=(0.0, 0.0, -0.03),
        name="coil_a",
    )
    coil_b = Coil(
        kind="solenoid",
        dims={"inner_radius": 0.01, "outer_radius": 0.02, "length": 0.04},
        current=3.0,
        turns=300,
        center=(0.0, 0.0, 0.03),
        name="coil_b",
    )
    probes = (
        Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),
        Probe(point=(0.005, 0.0, 0.015), quantity="B", component="z"),
        Probe(point=(0.015, 0.0, -0.02), quantity="B", component="x"),
    )
    exp = (
        ExpectedValue(
            check="thick_solenoid_Bz",
            quantity="Bz",
            value=1.0,
            unit="T",
            tolerance=1.0,
            source="unit test",
        ),
    )
    case_a = Case(
        name="sup_a",
        description="A",
        sources=("test",),
        components=(coil_a,),
        probes=probes,
        expected=exp,
    )
    case_b = Case(
        name="sup_b",
        description="B",
        sources=("test",),
        components=(coil_b,),
        probes=probes,
        expected=exp,
    )
    case_ab = Case(
        name="sup_ab",
        description="A+B",
        sources=("test",),
        components=(coil_a, coil_b),
        probes=probes,
        expected=exp,
    )

    engine = Fem2dEngine(
        element_size=0.005,
        domain=(0.25, -0.25, 0.25),
        refinement_box=(0.0, 0.03, -0.06, 0.06),
    )
    mesh_a, _, _ = engine._build_mesh_and_load(case_a)
    mesh_ab, _, _ = engine._build_mesh_and_load(case_ab)
    assert np.allclose(mesh_a.nodes, mesh_ab.nodes, rtol=0.0, atol=1e-15)

    vec_a = np.array([s.vector for s in engine.solve(case_a).samples])
    vec_b = np.array([s.vector for s in engine.solve(case_b).samples])
    vec_ab = np.array([s.vector for s in engine.solve(case_ab).samples])

    diff = np.linalg.norm(vec_ab - (vec_a + vec_b))
    scale = np.linalg.norm(vec_ab)
    assert diff / scale <= 1e-10


def test_steel_case_is_rejected() -> None:
    manifest = Manifest(
        path="yoke.step",
        format="step",
        units="mm",
        bodies=[
            Body(
                index=0,
                name="yoke",
                role=None,
                volume_mm3=1000.0,
                solids=1,
                bbox=(-10.0, 10.0, -10.0, 10.0, -5.0, 5.0),
                source="step",
            )
        ],
    )
    assign(manifest, {0: "permeable_solid"})
    base = CORPUS["thick_solenoid_axis_Bz"]
    steel_case = replace(base, name="solenoid_with_steel_yoke", manifest=manifest)

    engine = Fem2dEngine()
    problems = engine.validate(steel_case)
    errors = [p for p in problems if p.is_error]
    assert len(errors) == 1
    assert errors[0].code == "E_UNSUPPORTED_MATERIAL"
    assert "stage 2" in errors[0].message.lower()

    with pytest.raises(EngineError) as excinfo:
        engine.solve(steel_case)
    assert [p.code for p in excinfo.value.problems] == ["E_UNSUPPORTED_MATERIAL"]


def test_capability_gate_rejects_and_blocks() -> None:
    engine = Fem2dEngine()
    base_cyl = CORPUS["cylinder_axis_Bz"]
    relaxed_exp = (replace(base_cyl.expected[0], tolerance=0.05),)

    off_axis_case = replace(
        base_cyl,
        name="off_axis_cylinder",
        components=(replace(base_cyl.components[0], center=(0.05, 0.0, 0.0)),),
        expected=relaxed_exp,
        tolerance=0.05,
    )
    tilted_m_case = replace(
        base_cyl,
        name="tilted_magnetisation",
        components=(replace(base_cyl.components[0], br=(0.5, 0.0, 1.0)),),
        expected=relaxed_exp,
        tolerance=0.05,
    )

    cases_and_codes = [
        (CORPUS["dipole_axis_Bz"], "E_UNSUPPORTED_SOURCE_TYPE"),
        (CORPUS["cuboid_axis_Bz"], "E_UNSUPPORTED_GEOMETRY"),
        (CORPUS["loop_axis_Bz"], "E_UNSUPPORTED_GEOMETRY"),
        (off_axis_case, "E_NOT_AXISYMMETRIC"),
        (tilted_m_case, "E_UNSUPPORTED_MAGNETISATION"),
        (CORPUS["force_pointpole_g5L"], "E_UNSUPPORTED_QUANTITY"),
    ]

    for case, expected_code in cases_and_codes:
        errors = [p for p in engine.validate(case) if p.is_error]
        assert any(p.code == expected_code for p in errors), (case.name, errors)
        with pytest.raises(EngineError):
            engine.solve(case)


def test_accuracy_gate_is_honest() -> None:
    engine = Fem2dEngine()
    case = CORPUS["thick_solenoid_axis_Bz"]
    assert case.tolerance == 1e-4

    errors = [p for p in engine.validate(case) if p.is_error]
    assert len(errors) == 1
    assert errors[0].code == "E_ACCURACY_UNREACHABLE"
    msg = errors[0].message
    assert "1.00e-04" in msg or "0.0001" in msg
    assert "1.00e-02" in msg or "0.01" in msg

    with pytest.raises(EngineError) as excinfo:
        engine.solve(case)
    assert excinfo.value.problems[0].code == "E_ACCURACY_UNREACHABLE"

    # Also verify W_DEMAG_IGNORED is emitted as a warning when solvable.
    cyl = CORPUS["cylinder_axis_Bz"]
    demag_case = replace(
        cyl,
        components=(replace(cyl.components[0], demag_curve="N42"),),
        expected=(
            replace(
                cyl.expected[0],
                value=cylinder_Bz_axis(1.0, -1.0, 1.0, cyl.components[0].br[2] / MU0, 0.0),
                tolerance=0.05,
            ),
        ),
        tolerance=0.05,
    )
    result = Fem2dEngine(element_size=0.05).solve(demag_case)
    assert [p.code for p in result.problems] == ["W_DEMAG_IGNORED"]


def test_engine_protocol_and_registration(capsys: pytest.CaptureFixture[str]) -> None:
    engine = Fem2dEngine()
    assert isinstance(engine, Engine)

    caps = engine.capabilities()
    assert caps.engine_id == ENGINE_ID == "fem2d"
    assert caps.rung == RUNG == "Tier 1"
    assert caps.source_types == frozenset({"magnet", "coil"})
    assert caps.magnet_kinds == frozenset({"cylinder"})
    assert caps.coil_kinds == frozenset({"solenoid"})
    assert caps.quantities == frozenset({"B"})
    assert caps.force_methods == ()
    assert caps.engine_version.endswith("+m5.2b")

    rc = magwave.cli.main(["bench", "--engine", "fem2d", "--report-accuracy"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "thick_solenoid_axis_Bz" in out
    assert "measured:" in out
