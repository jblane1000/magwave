"""Tier 1 linear permeable steel validation gate (M5.2 Phase A)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from magwave.cases import Case, ExpectedValue, Permeable, Probe, UniformField
from magwave.engines import EngineError, Fem2dEngine
from magwave.fem2d.assembly import element_geometry
from magwave.units import MU0

RADIUS = 0.010
H0 = 1.0e4
MU_R_SPHERE = 100.0


def _analytic_sphere_interior_bz(mu_r: float = MU_R_SPHERE, h0: float = H0) -> float:
    """Uniform interior B_z of a linear permeable sphere in uniform H0 z_hat.

    Reference: Jackson, Classical Electrodynamics, 3rd ed., §5.10;
    Landau & Lifshitz, Electrodynamics of Continuous Media, §8.
    """
    return MU0 * mu_r * (3.0 / (mu_r + 2.0)) * h0


def _sphere_case(
    *,
    mu_r: float | None = MU_R_SPHERE,
    tolerance: float = 1e-2,
    probes: tuple[Probe, ...] | None = None,
    include_permeable: bool = True,
) -> Case:
    effective_mu_r = 1.0 if not include_permeable else (mu_r if mu_r is not None else 294.7)
    analytic = _analytic_sphere_interior_bz(effective_mu_r, H0)
    components: tuple[Permeable | UniformField, ...]
    if include_permeable:
        components = (
            Permeable(
                kind="sphere",
                dims={"radius": RADIUS},
                center=(0.0, 0.0, 0.0),
                material="steel-team20",
                mu_r=mu_r,
                name="steel_sphere",
            ),
            UniformField(h0=H0, name="applied_field"),
        )
    else:
        components = (UniformField(h0=H0, name="applied_field"),)

    case_probes = probes or (Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),)
    return Case(
        name="permeable_sphere_uniform_field",
        description="Permeable sphere in uniform applied field along +z",
        sources=("Jackson §5.10 / Landau & Lifshitz Electrodynamics §8",),
        components=components,
        probes=case_probes,
        expected=(
            ExpectedValue(
                check="analytic_permeable_sphere_interior_Bz",
                quantity="Bz",
                value=analytic,
                unit="T",
                tolerance=tolerance,
                source="Jackson §5.10",
            ),
        ),
    )


def _interface_node_count(engine: Fem2dEngine, case: Case) -> int:
    mesh, _, _ = engine._build_mesh_and_load(case)
    _, r_c, z_c, _, _ = element_geometry(mesh)
    inside = (r_c**2 + z_c**2) <= RADIUS**2
    in_nodes = np.unique(mesh.elements[inside])
    out_nodes = np.unique(mesh.elements[~inside])
    return int(np.intersect1d(in_nodes, out_nodes).size)


def test_steel_sphere_matches_the_analytic_interior_field() -> None:
    case = _sphere_case(mu_r=MU_R_SPHERE, tolerance=1e-2)
    analytic = _analytic_sphere_interior_bz(MU_R_SPHERE, H0)

    levels = (RADIUS / 8.0, RADIUS / 16.0, RADIUS / 32.0, RADIUS / 64.0)
    rows: list[tuple[float, int, int, int, float, float, float, str]] = []

    header = (
        f"{'h':>10} | {'nodes':>7} | {'elements':>8} | {'interface nodes':>15} | "
        f"{'computed B_z':>14} | {'analytic B':>14} | {'rel err':>11} | result"
    )
    print("\n" + header)
    print("-" * len(header))

    last_result = None
    for h in levels:
        engine = Fem2dEngine(element_size=h)
        iface_nodes = _interface_node_count(engine, case)
        result, report = engine.solve_with_report(case)
        computed = result.values()[0]
        rel_err = abs(computed - analytic) / analytic
        status = "PASS" if rel_err <= 1e-2 else "FAIL"
        rows.append(
            (
                h,
                report.mesh.nodes,
                report.mesh.elements,
                iface_nodes,
                computed,
                analytic,
                rel_err,
                status,
            )
        )
        last_result = result
        print(
            f"{h:>10.4e} | {report.mesh.nodes:>7d} | {report.mesh.elements:>8d} | "
            f"{iface_nodes:>15d} | {computed:>14.6e} | {analytic:>14.6e} | "
            f"{rel_err:>11.4e} | {status}"
        )

    assert last_result is not None
    assert "magwave.materials v1" in last_result.method
    assert "steel-team20" in last_result.method
    assert "294.7" in last_result.method
    assert "100" in last_result.method
    assert "case override" in last_result.method

    passing = [row for row in rows if row[6] <= 1e-2]
    assert len(passing) >= 2, f"expected >= 2 passing refinement levels, got {rows}"
    reported_pair = passing[-2:]
    assert reported_pair[0][6] <= 1e-2
    assert reported_pair[1][6] <= 1e-2
    assert reported_pair[1][6] < reported_pair[0][6]


def test_air_only_sphere_reproduces_the_uniform_field() -> None:
    h = RADIUS / 16.0
    exact = MU0 * H0
    air_sphere = _sphere_case(mu_r=1.0)
    no_sphere = _sphere_case(include_permeable=False)

    engine = Fem2dEngine(element_size=h)
    bz_air_sphere = engine.solve(air_sphere).values()[0]
    bz_no_sphere = engine.solve(no_sphere).values()[0]

    rel_vs_exact = abs(bz_air_sphere - exact) / exact
    rel_vs_free = abs(bz_air_sphere - bz_no_sphere) / abs(bz_no_sphere)
    print(
        f"\nair-only sphere: Bz={bz_air_sphere:.12e} T, free-space Bz={bz_no_sphere:.12e} T, "
        f"exact={exact:.12e} T, rel_err_exact={rel_vs_exact:.3e}, rel_diff_free={rel_vs_free:.3e}"
    )

    assert rel_vs_exact <= 1e-10
    assert rel_vs_free <= 1e-12


def test_interface_is_conforming_and_the_exterior_field_is_right() -> None:
    """The steel/air interface is node-conforming, and the exterior field is physical.

    Two separate claims, because a pointwise "B is continuous across the boundary" check is
    *not* available here: the FE field is element-constant, so on either side of a staircase
    interface it jumps by O(h) no matter how correct the solution is. What can be asserted is
    (a) the mesh is conforming — every triangle edge is shared by exactly two triangles and
    every steel/air edge is shared by exactly one steel and one air element (no hanging
    nodes), and (b) the field *outside* the sphere matches the independent closed form for a
    permeable sphere in a uniform applied field, ``B_z = B0*(1 + 2k(R/z)^3)`` with
    ``k = (mu_r - 1)/(mu_r + 2)`` — a different formula from the interior gate above.
    """
    h = RADIUS / 32.0
    probe = Probe(point=(0.0, 0.0, 3.0 * RADIUS), quantity="B", component="z")
    case = _sphere_case(mu_r=MU_R_SPHERE, probes=(probe,))
    engine = Fem2dEngine(element_size=h)

    mesh, _, _ = engine._build_mesh_and_load(case)
    _, r_c, z_c, _, _ = element_geometry(mesh)
    inside = (r_c**2 + z_c**2) <= RADIUS**2
    elems = mesh.elements
    edges = np.concatenate(
        [
            np.sort(elems[:, (0, 1)], axis=1),
            np.sort(elems[:, (1, 2)], axis=1),
            np.sort(elems[:, (2, 0)], axis=1),
        ]
    )
    owners: dict[tuple[int, int], list[int]] = {}
    for edge, elem in zip(edges, np.tile(np.arange(elems.shape[0]), 3), strict=True):
        owners.setdefault((int(edge[0]), int(edge[1])), []).append(int(elem))
    counts = {edge: len(hits) for edge, hits in owners.items()}
    assert max(counts.values()) == 2, "an edge is shared by more than two triangles"
    interface = [edge for edge, hits in owners.items() if inside[hits[0]] != inside[hits[-1]]]
    assert interface, "expected steel/air interface edges"
    assert all(len(owners[edge]) == 2 for edge in interface)
    assert all(inside[owners[edge][0]] != inside[owners[edge][1]] for edge in interface)

    bz = engine.solve(case).values()[0]
    b0 = MU0 * H0
    k = (MU_R_SPHERE - 1.0) / (MU_R_SPHERE + 2.0)
    exact = b0 * (1.0 + 2.0 * k * (RADIUS / (3.0 * RADIUS)) ** 3)
    rel = abs(bz - exact) / exact
    print(
        f"\nconforming interface: {len(interface)} steel/air edges, all 2-sided; "
        f"exterior B_z(3R)={bz:.9e} T, closed form={exact:.9e} T, rel={rel:.3e} "
        f"(a uniform field would be {b0:.9e} T, i.e. {(bz - b0) / b0:+.3%} away)"
    )
    assert rel <= 1e-2


def test_element_normal_jump_reported_at_the_pole() -> None:
    """Diagnostic only: the O(h) normal-B jump at the pole, printed, not gated.

    Element-constant ``B`` cannot be pointwise continuous across a staircase interface; the
    number is reported so a regression that *inflates* it is visible, and the bound is loose
    on purpose (a tight bound here would be gating the discretisation, not the physics).
    """
    h = RADIUS / 32.0
    probes = (
        Probe(point=(0.0, 0.0, RADIUS - 2.5 * h), quantity="B", component="z"),
        Probe(point=(0.0, 0.0, RADIUS + 2.5 * h), quantity="B", component="z"),
    )
    result = Fem2dEngine(element_size=h).solve(_sphere_case(mu_r=MU_R_SPHERE, probes=probes))
    b_in, b_out = result.values()
    rel_diff = abs(b_in - b_out) / max(abs(b_in), abs(b_out))
    print(
        f"\npole normal-B jump (h=R/32, points 2.5h either side): "
        f"Bz(inside)={b_in:.6e} T, Bz(outside)={b_out:.6e} T, rel jump={rel_diff:.3e}"
    )
    assert rel_diff <= 2.5e-1


def test_determinism_two_engine_instances() -> None:
    case = _sphere_case(mu_r=MU_R_SPHERE)
    res1 = Fem2dEngine(element_size=RADIUS / 32.0).solve(case)
    res2 = Fem2dEngine(element_size=RADIUS / 32.0).solve(case)
    b1 = np.array([sample.vector for sample in res1.samples])
    b2 = np.array([sample.vector for sample in res2.samples])
    assert np.array_equal(b1, b2)


def test_permeable_case_respects_the_accuracy_gate() -> None:
    tight_case = _sphere_case(mu_r=MU_R_SPHERE, tolerance=1e-4)
    engine = Fem2dEngine()
    errors = [problem for problem in engine.validate(tight_case) if problem.is_error]
    assert len(errors) == 1
    assert errors[0].code == "E_ACCURACY_UNREACHABLE"
    with pytest.raises(EngineError) as excinfo:
        engine.solve(tight_case)
    assert [p.code for p in excinfo.value.problems] == ["E_ACCURACY_UNREACHABLE"]


def test_overlap_and_block_are_rejected() -> None:
    base = _sphere_case(mu_r=MU_R_SPHERE)
    overlap_case = replace(
        base,
        name="overlapping_spheres",
        components=(
            Permeable(
                kind="sphere",
                dims={"radius": 0.01},
                center=(0.0, 0.0, 0.0),
                material="steel-team20",
                mu_r=100.0,
                name="s1",
            ),
            Permeable(
                kind="sphere",
                dims={"radius": 0.01},
                center=(0.0, 0.0, 0.015),
                material="steel-team20",
                mu_r=100.0,
                name="s2",
            ),
            UniformField(h0=H0, name="applied_field"),
        ),
    )
    engine = Fem2dEngine()
    overlap_errors = [p for p in engine.validate(overlap_case) if p.is_error]
    assert len(overlap_errors) == 1
    assert overlap_errors[0].code == "E_UNSUPPORTED_MATERIAL"
    assert "overlap" in overlap_errors[0].message.lower()

    block_case = replace(
        base,
        name="permeable_block",
        components=(
            Permeable(
                kind="block",
                dims={"size_x": 0.02, "size_y": 0.02, "size_z": 0.02},
                center=(0.0, 0.0, 0.0),
                material="steel-team20",
                mu_r=100.0,
                name="b1",
            ),
            UniformField(h0=H0, name="applied_field"),
        ),
    )
    block_errors = [p for p in engine.validate(block_case) if p.is_error]
    assert len(block_errors) == 1
    assert block_errors[0].code == "E_UNSUPPORTED_GEOMETRY"
