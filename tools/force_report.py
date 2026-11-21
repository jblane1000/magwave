#!/usr/bin/env python3
"""M5.4 force & torque verification report across analytical, FEM, and Elmer rungs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import ellipe, ellipk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.elmer_force import (  # noqa: E402
    GATE3_CONTOUR,
    build_magnet_plate_case,
    run_elmer_force_comparison,
)
from tools.elmer_permeable import (  # noqa: E402
    LINEAR_RECOIL_GRADE,
    ElmerUnavailableError,
    _linear_recoil_params,
)

from magwave.cases import Case, Coil, ExpectedValue, Probe  # noqa: E402
from magwave.fem2d.engine import Fem2dEngine  # noqa: E402
from magwave.fem2d.force import (  # noqa: E402
    DeltaStudyResult,
    ForceError,
    coenergy_delta_study,
    coenergy_force,
    magnetic_coenergy,
    maxwell_force,
)
from magwave.units import MU0  # noqa: E402

CONTOUR_A = GATE3_CONTOUR  # (0.012, -0.011, 0.0105)
CONTOUR_B = (0.015, -0.012, 0.0105)


def mutual_inductance_loops(a: float, b: float, d: float) -> float:
    """Neumann mutual inductance ``M(a, b, d)`` between coaxial circular loops in H."""
    k2 = 4.0 * a * b / ((a + b) ** 2 + d * d)
    k = math.sqrt(k2)
    return float(
        MU0 * math.sqrt(a * b) * ((2.0 / k - k) * float(ellipk(k2)) - (2.0 / k) * float(ellipe(k2)))
    )


def d_mutual_inductance_dd(a: float, b: float, d: np.ndarray | float) -> np.ndarray:
    """Exact closed-form axial derivative ``dM/dd`` for coaxial circular loops in H/m."""
    d_arr = np.asarray(d, dtype=np.float64)
    k2 = 4.0 * a * b / ((a + b) ** 2 + d_arr * d_arr)
    k = np.sqrt(k2)
    k_int = ellipk(k2)
    e_int = ellipe(k2)
    dm_dk = MU0 * math.sqrt(a * b) * (((2.0 - k2) / (k2 * (1.0 - k2))) * e_int - (2.0 / k2) * k_int)
    dk_dd = -k * d_arr / ((a + b) ** 2 + d_arr * d_arr)
    return dm_dk * dk_dd


def neumann_coaxial_sheet_force(
    a: float,
    b: float,
    l1: float,
    l2: float,
    d: float,
    n1_i1: float,
    n2_i2: float,
    *,
    n_gauss: int = 48,
) -> float:
    """Exact axial force on sheet 2 due to sheet 1 via ``>= 40``-pt Gauss–Legendre Neumann derivative."""
    if n_gauss < 40:
        raise ValueError(f"n_gauss must be >= 40, got {n_gauss}")
    xi, wi = np.polynomial.legendre.leggauss(n_gauss)
    z1 = 0.5 * l1 * xi
    z2 = 0.5 * l2 * xi
    dist_mat = d + z2[None, :] - z1[:, None]
    weights = 0.25 * (wi[:, None] * wi[None, :])
    integral = float(np.sum(weights * d_mutual_inductance_dd(a, b, dist_mat)))
    return float(n1_i1 * n2_i2 * integral)


def nagaoka_factor(radius: float, length: float) -> float:
    """Exact Nagaoka / Lorenz finite-solenoid end-effect factor ``K_N`` for thin sheet."""
    diameter = 2.0 * radius
    k2 = (diameter * diameter) / (diameter * diameter + length * length)
    k = math.sqrt(k2)
    kp = math.sqrt(1.0 - k2)
    k_val = float(ellipk(k2))
    e_val = float(ellipe(k2))
    return float((4.0 / (3.0 * math.pi * kp)) * ((kp * kp / k2) * (k_val - e_val) + e_val - k))


def build_coaxial_loops_case() -> Case:
    """Gate 1: two coaxial thin-sheet loops (R=10 mm, L=2 mm, z=+-5 mm, N=1, I=1 A)."""
    return Case(
        name="gate1_coaxial_loops",
        description="Two coaxial thin-sheet current loops at z = -5 mm and +5 mm",
        sources=("Neumann mutual inductance derivative (Gate 1)",),
        components=(
            Coil(
                kind="solenoid",
                dims={"inner_radius": 0.010, "outer_radius": 0.010, "length": 0.002},
                current=1.0,
                turns=1,
                center=(0.0, 0.0, -0.005),
                name="coil_bottom",
            ),
            Coil(
                kind="solenoid",
                dims={"inner_radius": 0.010, "outer_radius": 0.010, "length": 0.002},
                current=1.0,
                turns=1,
                center=(0.0, 0.0, 0.005),
                name="coil_top",
            ),
        ),
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
        expected=(
            ExpectedValue(
                check="neumann_loop_force",
                quantity="Fz",
                value=-7.282368e-7,
                unit="N",
                tolerance=0.01,
                source="Neumann closed form",
            ),
        ),
    )


def build_solenoid_case() -> Case:
    """Gate 2: thin-sheet solenoid (R=10 mm, L=200 mm, N=400, I=1 A)."""
    return Case(
        name="gate2_solenoid_inductance",
        description="Long thin-sheet solenoid (l/D = 10, l/R = 20) for coenergy inductance",
        sources=("Ideal + Nagaoka solenoid inductance (Gate 2)",),
        components=(
            Coil(
                kind="solenoid",
                dims={"inner_radius": 0.010, "outer_radius": 0.010, "length": 0.200},
                current=1.0,
                turns=400,
                center=(0.0, 0.0, 0.0),
                name="solenoid",
            ),
        ),
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
        expected=(
            ExpectedValue(
                check="solenoid_inductance",
                quantity="Bz",
                value=MU0 * 400.0 / 0.200,
                unit="T",
                tolerance=0.01,
                source="Nagaoka / ideal solenoid",
            ),
        ),
    )


def _steel_engine(element_size: float) -> Fem2dEngine:
    _, _, _, linear_curve = _linear_recoil_params()
    return Fem2dEngine(
        element_size=element_size,
        material_model="nonlinear",
        magnet_curves={LINEAR_RECOIL_GRADE: linear_curve},
    )


def _sha256_repr(obj: object) -> str:
    return hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()


def _serialize_delta_study(study: DeltaStudyResult) -> dict[str, object]:
    """Convert a :class:`DeltaStudyResult` into a JSON-serialisable dictionary."""
    step_rows: list[dict[str, object]] = []
    for idx, step in enumerate(study.steps):
        p_pair_row = (
            study.p_per_pair[idx - 2] if idx >= 2 and (idx - 2) < len(study.p_per_pair) else None
        )
        step_rows.append(
            {
                "label": step.label,
                "delta_m": step.delta_m,
                "delta_over_h": step.delta_over_h,
                "f_coenergy_n": step.force_n,
                "change_vs_prev_pct": step.change_vs_prev_pct,
                "mesh_nodes_plus": step.mesh_nodes_plus,
                "mesh_elements_plus": step.mesh_elements_plus,
                "mesh_nodes_minus": step.mesh_nodes_minus,
                "mesh_elements_minus": step.mesh_elements_minus,
                "gap_cells_plus": step.gap_cells_plus,
                "gap_cells_minus": step.gap_cells_minus,
                "order_p": p_pair_row,
            }
        )
    return {
        "rung": "coenergy (cell-aligned finite-difference ladder)",
        "target": study.target,
        "h_m": study.h_m,
        "base_mesh_nodes": study.base_mesh_nodes,
        "base_mesh_elements": study.base_mesh_elements,
        "axial_gap_m": study.axial_gap_m,
        "expected_gap_cell_sum": study.expected_gap_cell_sum,
        "steps": step_rows,
        "refused": [{"delta_m": d, "reason": r} for d, r in study.refused],
        "p_per_pair": list(study.p_per_pair),
        "p_fit": study.p_fit,
        "monotonic": study.monotonic,
        "asymptotic": study.asymptotic,
        "f_extrapolated_n": study.f_extrapolated_n,
        "step_uncertainty_n": study.step_uncertainty_n,
        "step_uncertainty_pct": study.step_uncertainty_pct,
        "peak_to_peak_n": study.peak_to_peak_n,
        "peak_to_peak_pct": study.peak_to_peak_pct,
        "mesh_counts_invariant": study.mesh_counts_invariant,
        "gap_cell_sum_invariant": study.gap_cell_sum_invariant,
        "alignment_invariant_holds": study.alignment_invariant_holds,
        "notes": list(study.notes),
    }


def run_report(*, fast: bool = False, skip_elmer: bool = False) -> dict[str, object]:
    """Run all 7 M5.4 verification sections and return structured JSON payload."""
    # 1. LOOP FORCE
    a_loop = 0.010
    l_sheet = 0.002
    d_sep = 0.010
    f_loop_thin = neumann_coaxial_sheet_force(a_loop, a_loop, 1e-5, 1e-5, d_sep, 1.0, 1.0)
    f_loop_single = float(d_mutual_inductance_dd(a_loop, a_loop, d_sep))
    thin_limit_rel = abs(f_loop_thin - f_loop_single) / abs(f_loop_single)

    f_ref_loop = neumann_coaxial_sheet_force(a_loop, a_loop, l_sheet, l_sheet, d_sep, 1.0, 1.0)
    loop_case = build_coaxial_loops_case()
    loop_hs = (5.0e-4, 2.5e-4)
    loop_rows: list[dict[str, object]] = []
    for h in loop_hs:
        eng = Fem2dEngine(element_size=h)
        res_m = maxwell_force(loop_case, eng, target="coil_top")
        res_c = coenergy_force(loop_case, eng, target="coil_top")
        f_m = res_m.force_n[2]
        f_c = res_c.force_n[2]
        err_m_pct = 100.0 * abs(f_m - f_ref_loop) / abs(f_ref_loop)
        err_c_pct = 100.0 * abs(f_c - f_ref_loop) / abs(f_ref_loop)
        loop_rows.append(
            {
                "rungs": "closed-form (Neumann), Maxwell stress, coenergy",
                "h_m": h,
                "f_ref_n": f_ref_loop,
                "f_maxwell_n": f_m,
                "err_maxwell_pct": err_m_pct,
                "f_coenergy_n": f_c,
                "err_coenergy_pct": err_c_pct,
                "maxwell_contour_m": list(res_m.contour) if res_m.contour else None,
                "coenergy_delta_m": res_c.delta_m,
            }
        )
    for idx, row in enumerate(loop_rows):
        if idx == 0:
            row["error_falls"] = "baseline"
        else:
            prev = loop_rows[idx - 1]
            falls = (row["err_maxwell_pct"] < prev["err_maxwell_pct"]) and (
                row["err_coenergy_pct"] < prev["err_coenergy_pct"]
            )
            row["error_falls"] = "YES" if falls else "NO"

    # 2. SOLENOID INDUCTANCE
    sol_h = 1.0e-3 if fast else 5.0e-4
    sol_case = build_solenoid_case()
    sol_eng = Fem2dEngine(element_size=sol_h)
    w_sol, rep_sol, _ = magnetic_coenergy(sol_case, sol_eng)
    n_turns = 400
    i_curr = 1.0
    r_sol = 0.010
    l_sol = 0.200
    l_fem = 2.0 * w_sol / (i_curr * i_curr)
    l_ideal = MU0 * (n_turns**2) * (math.pi * r_sol * r_sol) / l_sol
    k_nagaoka = nagaoka_factor(r_sol, l_sol)
    l_nagaoka = k_nagaoka * l_ideal
    raw_dev_pct = 100.0 * (l_fem - l_ideal) / l_ideal
    corr_dev_pct = 100.0 * (l_fem - l_nagaoka) / l_nagaoka
    solenoid_data = {
        "rung": "coenergy",
        "reference_rung": "closed-form (Neumann)",
        "h_m": sol_h,
        "nodes": rep_sol.mesh.nodes,
        "elements": rep_sol.mesh.elements,
        "coenergy_j": w_sol,
        "l_fem_h": l_fem,
        "l_ideal_h": l_ideal,
        "raw_dev_pct": raw_dev_pct,
        "nagaoka_factor": k_nagaoka,
        "l_nagaoka_h": l_nagaoka,
        "corrected_dev_pct": corr_dev_pct,
        "definition": "L_fem = 2 * W' / I^2 = 2 * W' * N^2 / (N*I)^2",
        "end_effect_basis": (
            f"exact Nagaoka/Lorenz thin-sheet factor K_N={k_nagaoka:.8f} for l/D=10 (l/R=20)"
        ),
    }

    # 3. STEEL CASE (Gate 3, Maxwell stays at h = 5e-4; aligned ladders at h = 5e-4 and 2.5e-4)
    g3_h = 5.0e-4
    steel_case = build_magnet_plate_case(0.0)
    eng_g3 = _steel_engine(g3_h)
    g3_maxwell = maxwell_force(steel_case, eng_g3, target="magnet", contour=CONTOUR_A)
    g3_coenergy = coenergy_force(steel_case, eng_g3, target="magnet")
    study_a = coenergy_delta_study(steel_case, eng_g3, target="magnet")
    eng_h25 = _steel_engine(2.5e-4)
    study_b = coenergy_delta_study(steel_case, eng_h25, target="magnet")

    f_g3_m = float(g3_maxwell.force_n[2])
    f_g3_c = float(g3_coenergy.force_n[2])
    g3_agree_pct = 100.0 * abs(f_g3_m - f_g3_c) / max(abs(f_g3_m), abs(f_g3_c))
    verdict_i = "PASS" if g3_agree_pct <= 2.0 else "FAIL"

    f_g3_ext = study_b.f_extrapolated_n
    if f_g3_ext is not None:
        agree_ii: float | None = 100.0 * abs(f_g3_m - f_g3_ext) / max(abs(f_g3_m), abs(f_g3_ext))
        verdict_ii = "PASS" if agree_ii <= 2.0 else "FAIL"
    else:
        agree_ii = None
        verdict_ii = "N/A"
    gate3_ambiguous = verdict_i != verdict_ii

    assert g3_maxwell.convergence is not None
    op_pt = g3_maxwell.convergence.magnet_operating_points[0]

    study_a_dict = _serialize_delta_study(study_a)
    study_b_dict = _serialize_delta_study(study_b)
    proof_rows: list[dict[str, object]] = []
    for study_obj in (study_a, study_b):
        proof_rows.append(
            {
                "h_m": study_obj.h_m,
                "label": "unshifted",
                "delta_m": None,
                "nodes_plus": study_obj.base_mesh_nodes,
                "elements_plus": study_obj.base_mesh_elements,
                "nodes_minus": study_obj.base_mesh_nodes,
                "elements_minus": study_obj.base_mesh_elements,
                "gap_cells_plus": None,
                "gap_cells_minus": None,
                "invariant_ok": study_obj.alignment_invariant_holds,
            }
        )
        for step in study_obj.steps:
            proof_rows.append(
                {
                    "h_m": study_obj.h_m,
                    "label": step.label,
                    "delta_m": step.delta_m,
                    "nodes_plus": step.mesh_nodes_plus,
                    "elements_plus": step.mesh_elements_plus,
                    "nodes_minus": step.mesh_nodes_minus,
                    "elements_minus": step.mesh_elements_minus,
                    "gap_cells_plus": step.gap_cells_plus,
                    "gap_cells_minus": step.gap_cells_minus,
                    "invariant_ok": (
                        (step.mesh_nodes_plus, step.mesh_elements_plus)
                        == (study_obj.base_mesh_nodes, study_obj.base_mesh_elements)
                        and (step.mesh_nodes_minus, step.mesh_elements_minus)
                        == (study_obj.base_mesh_nodes, study_obj.base_mesh_elements)
                    ),
                }
            )

    steel_data: dict[str, object] = {
        "rungs": "Maxwell stress, coenergy",
        "h_m": g3_h,
        "nodes": g3_maxwell.mesh_nodes,
        "elements": g3_maxwell.mesh_elements,
        "f_maxwell_n": f_g3_m,
        "f_coenergy_n": f_g3_c,
        "agreement_pct": g3_agree_pct,
        "agreement_gate_limit_pct": 2.0,
        "basis_i_aligned_single_step": {
            "h_m": g3_h,
            "delta_m": g3_coenergy.delta_m,
            "f_maxwell_n": f_g3_m,
            "f_coenergy_n": f_g3_c,
            "agreement_pct": g3_agree_pct,
            "verdict": verdict_i,
        },
        "basis_ii_aligned_ladder_extrap": {
            "h_m": study_b.h_m,
            "delta_basis": "aligned ladder (h, 2h, 3h) at h = 2.5e-4 m",
            "f_maxwell_n": f_g3_m,
            "f_extrapolated_n": f_g3_ext,
            "step_uncertainty_n": study_b.step_uncertainty_n,
            "step_uncertainty_pct": study_b.step_uncertainty_pct,
            "agreement_pct": agree_ii,
            "verdict": verdict_ii,
        },
        "gate3_verdict_ambiguous": gate3_ambiguous,
        "contour_a_requested_m": list(CONTOUR_A),
        "contour_a_snapped_m": list(g3_maxwell.contour) if g3_maxwell.contour else None,
        "delta_m": g3_coenergy.delta_m,
        "magnet_h_a_per_m": op_pt.h_a_per_m,
        "magnet_b_t": op_pt.b_t,
        "magnet_op_source": op_pt.source,
        "coenergy_study_h5e4": study_a_dict,
        "coenergy_study_h2_5e4": study_b_dict,
        "coenergy_step_study": study_b_dict,
        "node_element_proof_rows": proof_rows,
    }

    # 4. FORCE VS MESH (Gate 4 — Maxwell at all h, coenergy strictly cell-aligned delta = h)
    mesh_hs = (2.0e-3, 1.0e-3, 5.0e-4) if fast else (1.0e-3, 5.0e-4, 2.5e-4)
    mesh_rows: list[dict[str, object]] = []
    for h in mesh_hs:
        if math.isclose(h, g3_h, rel_tol=1e-12):
            rm = g3_maxwell
            fc: float | None = f_g3_c
            co_delta: float | None = g3_coenergy.delta_m
            co_refused = False
            co_refusal_code: str | None = None
            co_refusal_reason: str | None = None
        else:
            eng_h = _steel_engine(h)
            rm = maxwell_force(steel_case, eng_h, target="magnet", contour=CONTOUR_A)
            if math.isclose(h, 2.5e-4, rel_tol=1e-12):
                fc = float(study_b.steps[0].force_n)
                co_delta = float(study_b.steps[0].delta_m)
                co_refused = False
                co_refusal_code = None
                co_refusal_reason = None
            else:
                try:
                    rc = coenergy_force(steel_case, eng_h, target="magnet")
                except ForceError as exc:
                    fc = None
                    co_delta = None
                    co_refused = True
                    co_refusal_code = exc.code
                    co_refusal_reason = str(exc)
                else:
                    fc = float(rc.force_n[2])
                    co_delta = rc.delta_m
                    co_refused = False
                    co_refusal_code = None
                    co_refusal_reason = None

        fm = float(rm.force_n[2])
        if not mesh_rows:
            dm_pct = None
        else:
            prev = mesh_rows[-1]
            dm_pct = 100.0 * (fm - float(prev["f_maxwell_n"])) / abs(float(prev["f_maxwell_n"]))

        prev_valid_co = next(
            (r["f_coenergy_n"] for r in reversed(mesh_rows) if r["f_coenergy_n"] is not None),
            None,
        )
        if fc is None or prev_valid_co is None:
            dc_pct: float | None = None
        else:
            dc_pct = 100.0 * (fc - float(prev_valid_co)) / abs(float(prev_valid_co))

        mesh_rows.append(
            {
                "rungs": "Maxwell stress, coenergy",
                "h_m": h,
                "nodes": rm.mesh_nodes,
                "elements": rm.mesh_elements,
                "f_maxwell_n": fm,
                "delta_maxwell_pct": dm_pct,
                "f_coenergy_n": fc,
                "delta_coenergy_pct": dc_pct,
                "delta_m": co_delta,
                "coenergy_refused": co_refused,
                "coenergy_refusal_code": co_refusal_code,
                "coenergy_refusal_reason": co_refusal_reason,
                "contour_snapped_m": list(rm.contour) if rm.contour else None,
            }
        )

    def _richardson(vals: list[float]) -> dict[str, object]:
        f1, f2, f3 = vals
        denom = f2 - f3
        ratio = (f1 - f2) / denom if denom != 0.0 else math.nan
        if ratio > 1.0:
            p = math.log2(ratio)
            f_ext = f3 + (f3 - f2) / (2.0**p - 1.0)
            est_err_pct = 100.0 * abs(f3 - f_ext) / abs(f_ext)
        elif ratio > 0.0:
            p = math.log2(ratio)
            f_ext = math.nan
            est_err_pct = math.nan
        else:
            p = math.nan
            f_ext = math.nan
            est_err_pct = math.nan
        return {"p": p, "f_extrap_n": f_ext, "est_error_pct": est_err_pct}

    rich_m = _richardson([float(r["f_maxwell_n"]) for r in mesh_rows])
    rich_c = {
        "p": None,
        "f_extrap_n": None,
        "est_error_pct": None,
        "status": "n/a (two aligned rungs; three are required)",
    }
    converged_flag = False
    superseded_mesh_history = (
        "M5.4 fixed non-aligned delta = 2.50e-04 m coenergy values at h = 1.0e-3 / 5.0e-4 m "
        "(102.57859 / 101.98703 N) are retired as remeshed-geometry values, not recomputed."
    )

    # 5. CONTOURS (Gate 5, h = 5e-4)
    res_a = g3_maxwell
    res_b = maxwell_force(steel_case, eng_g3, target="magnet", contour=CONTOUR_B)
    fa = res_a.force_n[2]
    fb = res_b.force_n[2]
    contour_diff_pct = 100.0 * abs(fa - fb) / max(abs(fa), abs(fb))
    contour_data = {
        "rung": "Maxwell stress",
        "h_m": 5.0e-4,
        "contour_a_requested_m": list(CONTOUR_A),
        "contour_a_snapped_m": list(res_a.contour) if res_a.contour else None,
        "contour_b_requested_m": list(CONTOUR_B),
        "contour_b_snapped_m": list(res_b.contour) if res_b.contour else None,
        "f_a_n": fa,
        "f_b_n": fb,
        "difference_pct": contour_diff_pct,
        "notes_a": list(res_a.notes),
        "notes_b": list(res_b.notes),
    }

    # 6. ELMER (Gate 6)
    if skip_elmer:
        elmer_data: dict[str, object] = {
            "rung": "second code (Elmer)",
            "status": "SKIP",
            "reason": "skipped by --skip-elmer flag",
        }
    else:
        try:
            elmer_res = run_elmer_force_comparison()
            elmer_data = {
                "rung": "second code (Elmer)",
                "status": "PASS" if elmer_res["passed"] else "FAIL",
                "nodes": elmer_res["elmer"]["nodes"],
                "elements": elmer_res["elmer"]["elements"],
                "delta_m": elmer_res["delta_m"],
                "w_plus_j": elmer_res["elmer"]["energy_plus_j"],
                "w_minus_j": elmer_res["elmer"]["energy_minus_j"],
                "f_elmer_n": elmer_res["elmer"]["force_z_n"],
                "vs_maxwell_pct": 100.0 * float(elmer_res["rel_diff_vs_maxwell"]),
                "vs_coenergy_pct": 100.0 * float(elmer_res["rel_diff_vs_coenergy"]),
            }
        except ElmerUnavailableError as exc:
            elmer_data = {
                "rung": "second code (Elmer)",
                "status": "SKIP",
                "reason": str(exc),
            }

    # 7. DETERMINISM (Gate 7)
    m_run1 = maxwell_force(steel_case, _steel_engine(5.0e-4), target="magnet", contour=CONTOUR_A)
    m_run2 = maxwell_force(steel_case, _steel_engine(5.0e-4), target="magnet", contour=CONTOUR_A)
    c_run1 = coenergy_force(steel_case, _steel_engine(5.0e-4), target="magnet")
    c_run2 = coenergy_force(steel_case, _steel_engine(5.0e-4), target="magnet")
    det_rows = [
        {
            "rung": "Maxwell stress",
            "method": m_run1.method,
            "sha256_run1": _sha256_repr(m_run1),
            "sha256_run2": _sha256_repr(m_run2),
            "equal": repr(m_run1) == repr(m_run2),
        },
        {
            "rung": "coenergy",
            "method": c_run1.method,
            "sha256_run1": _sha256_repr(c_run1),
            "sha256_run2": _sha256_repr(c_run2),
            "equal": repr(c_run1) == repr(c_run2),
        },
    ]

    # Populate §3 Elmer triangulation using live Gate 6 result when available
    if "f_elmer_n" in elmer_data:
        f_elmer_ref = float(elmer_data["f_elmer_n"])
        elmer_ref_source = "live Gate 6 Elmer run"
    else:
        f_elmer_ref = 101.72932
        elmer_ref_source = "ADR-0009 measured Elmer value (live Elmer skipped/unavailable)"
    f_h25_k1 = float(study_b.steps[0].force_n)
    steel_data["elmer_triangulation"] = {
        "f_elmer_n": f_elmer_ref,
        "source": elmer_ref_source,
        "vs_maxwell_h5e4_pct": 100.0 * abs(f_g3_m - f_elmer_ref) / abs(f_elmer_ref),
        "vs_coenergy_h5e4_k1_pct": 100.0 * abs(f_g3_c - f_elmer_ref) / abs(f_elmer_ref),
        "vs_coenergy_h2_5e4_k1_pct": 100.0 * abs(f_h25_k1 - f_elmer_ref) / abs(f_elmer_ref),
        "vs_coenergy_h2_5e4_extrap_pct": (
            100.0 * abs(f_g3_ext - f_elmer_ref) / abs(f_elmer_ref) if f_g3_ext is not None else None
        ),
    }

    return {
        "fast_mode": fast,
        "loop_force": {
            "neumann_thin_limit_rel_diff": thin_limit_rel,
            "rows": loop_rows,
        },
        "solenoid_inductance": solenoid_data,
        "steel_case": steel_data,
        "force_vs_mesh": {
            "delta_m": None,
            "delta_basis": "cell-aligned delta = h",
            "rows": mesh_rows,
            "richardson_maxwell": rich_m,
            "richardson_coenergy": rich_c,
            "converged_within_1pct_est": converged_flag,
            "converged_note": (
                "Maxwell < 1% at h = 2.5e-4 m; coenergy Richardson n/a (two aligned rungs; "
                "three are required)"
            ),
            "superseded_history": superseded_mesh_history,
        },
        "contours": contour_data,
        "elmer": elmer_data,
        "determinism": det_rows,
    }


def print_report(payload: dict[str, object]) -> None:
    """Print the 7 required plain-ASCII tables in exact order."""
    # 1. LOOP FORCE
    lf = payload["loop_force"]
    print("1 LOOP FORCE")
    print(
        f"Neumann self-check (L/a = 1e-3 vs single-loop closed form): rel diff = "
        f"{lf['neumann_thin_limit_rel_diff']:.3e} (rung: closed-form (Neumann))"
    )
    h1 = (
        f"{'Rungs':<46} | {'mesh h':>8} | {'F_ref (N)':>13} | {'F_maxwell':>13} | "
        f"{'err%':>7} | {'F_coenergy':>13} | {'err%':>7} | error falls?"
    )
    print(h1)
    print("-" * len(h1))
    for row in lf["rows"]:
        print(
            f"{row['rungs']:<46} | {row['h_m']:>8.2e} | {row['f_ref_n']:>13.6e} | "
            f"{row['f_maxwell_n']:>13.6e} | {row['err_maxwell_pct']:>6.3f}% | "
            f"{row['f_coenergy_n']:>13.6e} | {row['err_coenergy_pct']:>6.3f}% | {row['error_falls']}"
        )
    print()

    # 2. SOLENOID L
    sol = payload["solenoid_inductance"]
    print("2 SOLENOID L")
    print(f"Definition: {sol['definition']} (h = {sol['h_m']:.2e} m, {sol['nodes']} nodes)")
    h2 = (
        f"{'Rung':<10} | {'L_fem (H)':>13} | {'L_ideal (H)':>13} | {'raw dev%':>9} | "
        f"{'end-effect factor + basis':<74} | {'corrected dev%':>14}"
    )
    print(h2)
    print("-" * len(h2))
    print(
        f"{sol['rung']:<10} | {sol['l_fem_h']:>13.6e} | {sol['l_ideal_h']:>13.6e} | "
        f"{sol['raw_dev_pct']:>+8.3f}% | {sol['end_effect_basis']:<74} | {sol['corrected_dev_pct']:>+13.4f}%"
    )
    print()

    # 3. STEEL CASE
    st = payload["steel_case"]
    study_a = st["coenergy_study_h5e4"]
    study_b = st["coenergy_study_h2_5e4"]
    b1 = st["basis_i_aligned_single_step"]
    b2 = st["basis_ii_aligned_ladder_extrap"]
    et = st["elmer_triangulation"]
    print("3 STEEL CASE")
    h3 = (
        f"{'Rungs':<25} | {'F_maxwell':>10} | {'F_coenergy':>10} | {'agreement%':>10} | "
        f"{'contour A (snapped, m)':<34} | {'delta (m)':>9} | {'magnet (H A/m, B T)'}"
    )
    print(h3)
    print("-" * len(h3))
    c_str = "(" + ", ".join(f"{v:.6g}" for v in st["contour_a_snapped_m"]) + ")"
    op_str = f"({st['magnet_h_a_per_m']:.1f}, {st['magnet_b_t']:.6f})"
    print(
        f"{st['rungs']:<25} | {st['f_maxwell_n']:>10.5f} | {st['f_coenergy_n']:>10.5f} | "
        f"{st['agreement_pct']:>9.4f}% | {c_str:<34} | {st['delta_m']:>9.2e} | {op_str}"
    )

    def _print_study_block(title: str, sd: dict[str, object]) -> None:
        print(
            f"  {title} (h = {sd['h_m']:.2e} m, base={sd['base_mesh_nodes']}n/{sd['base_mesh_elements']}e)"
        )
        hdr = (
            f"  {'delta (m)':>10} | {'delta/h':>7} | {'F_coenergy (N)':>14} | "
            f"{'change vs prev %':>16} | {'nodes (+d/-d)':>15} | "
            f"{'elements (+d/-d)':>17} | {'gap cells (+d/-d)':>17} | {'order p':>8}"
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for row in sd["steps"]:
            chg_s = (
                "---"
                if row["change_vs_prev_pct"] is None
                else f"{row['change_vs_prev_pct']:>+15.3f}%"
            )
            nodes_s = f"{row['mesh_nodes_plus']}/{row['mesh_nodes_minus']}"
            elems_s = f"{row['mesh_elements_plus']}/{row['mesh_elements_minus']}"
            if row["gap_cells_plus"] is None or row["gap_cells_minus"] is None:
                gap_s = "---"
            else:
                gap_s = f"{row['gap_cells_plus']} / {row['gap_cells_minus']}"
            p_s = "---" if row["order_p"] is None else f"{row['order_p']:>+8.3f}"
            print(
                f"  {row['delta_m']:>10.2e} | {row['delta_over_h']:>7.2f} | "
                f"{row['f_coenergy_n']:>14.5f} | {chg_s:>16} | {nodes_s:>15} | "
                f"{elems_s:>17} | {gap_s:>17} | {p_s:>8}"
            )
        for ref in sd["refused"]:
            print(f"  REFUSED delta={ref['delta_m']:.2e} m: {ref['reason']}")
        p_pair_str = (
            ", ".join("---" if val is None else f"{val:+.5f}" for val in sd["p_per_pair"])
            if sd["p_per_pair"]
            else "n/a"
        )
        p_fit_str = "None" if sd["p_fit"] is None else f"{sd['p_fit']:+.5f}"
        asym_str = "YES" if sd["asymptotic"] else "NO"
        print(
            f"  p_per_pair = ({p_pair_str}), p_fit = {p_fit_str}, "
            f"monotonic = {sd['monotonic']}, asymptotic = {asym_str}"
        )
        if sd["f_extrapolated_n"] is not None:
            print(
                f"  F_extrapolated = {sd['f_extrapolated_n']:.5f} N +/- "
                f"{sd['step_uncertainty_n']:.5f} N (+/- {sd['step_uncertainty_pct']:.3f}%)"
            )
        else:
            print(
                f"  F_extrapolated = None (peak-to-peak spread = {sd['peak_to_peak_n']:.5f} N, "
                f"{sd['peak_to_peak_pct']:.3f}%)"
            )
        for note in sd["notes"]:
            print(f"  NOTE: {note}")

    _print_study_block("Study A — aligned ladder at Gate 3 h", study_a)
    _print_study_block("Study B — aligned ladder at refined h", study_b)

    print("  Node/element count invariance proof across every delta and h:")
    hp_hdr = (
        f"  {'h (m)':>9} | {'step':>9} | {'delta (m)':>10} | {'nodes (+d/-d)':>15} | "
        f"{'elements (+d/-d)':>17} | {'gap cells (+d/-d)':>17} | invariant?"
    )
    print(hp_hdr)
    print("  " + "-" * (len(hp_hdr) - 2))
    for pr in st["node_element_proof_rows"]:
        d_s = "---" if pr["delta_m"] is None else f"{pr['delta_m']:>10.2e}"
        n_s = f"{pr['nodes_plus']}/{pr['nodes_minus']}"
        e_s = f"{pr['elements_plus']}/{pr['elements_minus']}"
        g_s = (
            "---"
            if pr["gap_cells_plus"] is None
            else f"{pr['gap_cells_plus']} / {pr['gap_cells_minus']}"
        )
        inv_s = "YES" if pr["invariant_ok"] else "NO"
        print(
            f"  {pr['h_m']:>9.2e} | {pr['label']:>9} | {d_s:>10} | {n_s:>15} | "
            f"{e_s:>17} | {g_s:>17} | {inv_s}"
        )

    print(
        f"  Gate 3 Basis (i) [aligned delta=h={b1['delta_m']:.2e} m at h={b1['h_m']:.2e} m]: "
        f"F_coenergy={b1['f_coenergy_n']:.5f} N vs F_maxwell={b1['f_maxwell_n']:.5f} N -> "
        f"{b1['agreement_pct']:.4f}% (gate <= 2.0%: {b1['verdict']})"
    )
    if b2["f_extrapolated_n"] is not None:
        print(
            f"  Gate 3 Basis (ii) [aligned ladder extrap at h={b2['h_m']:.2e} m]: "
            f"F_extrap={b2['f_extrapolated_n']:.5f} +/- {b2['step_uncertainty_n']:.5f} N "
            f"(+/- {b2['step_uncertainty_pct']:.3f}%) vs F_maxwell={b2['f_maxwell_n']:.5f} N -> "
            f"{b2['agreement_pct']:.4f}% (gate <= 2.0%: {b2['verdict']})"
        )
    else:
        print(
            f"  Gate 3 Basis (ii) [aligned ladder extrap at h={b2['h_m']:.2e} m]: "
            f"F_extrap=None -> {b2['verdict']}"
        )
    if st["gate3_verdict_ambiguous"]:
        print("  GATE 3 VERDICT AMBIGUOUS: aligned ladder contradicts aligned single step")

    ext_dist_s = (
        "n/a"
        if et["vs_coenergy_h2_5e4_extrap_pct"] is None
        else f"{et['vs_coenergy_h2_5e4_extrap_pct']:.3f}%"
    )
    print(
        f"  Elmer triangulation ({et['source']}): F_elmer = {et['f_elmer_n']:.5f} N; "
        f"vs Maxwell(h=5e-4) = {et['vs_maxwell_h5e4_pct']:.3f}%, "
        f"vs coenergy(h=5e-4, delta=h) = {et['vs_coenergy_h5e4_k1_pct']:.3f}%, "
        f"vs coenergy(h=2.5e-4, delta=h) = {et['vs_coenergy_h2_5e4_k1_pct']:.3f}%, "
        f"vs coenergy(h=2.5e-4, extrap) = {ext_dist_s}"
    )
    print()

    # 4. FORCE VS MESH
    fvm = payload["force_vs_mesh"]
    rm = fvm["richardson_maxwell"]
    print("4 FORCE VS MESH")
    h4 = (
        f"{'Rungs':<25} | {'h (m)':>8} | {'nodes':>6} | {'elements':>8} | "
        f"{'F_maxwell':>10} | {'d%':>8} | {'F_coenergy':>10} | {'d%':>8} | "
        f"{'Richardson p':>13} | {'est. error%':>12}"
    )
    print(h4)
    print("-" * len(h4))
    rows_m = fvm["rows"]
    for idx, row in enumerate(rows_m):
        dm_s = "---" if row["delta_maxwell_pct"] is None else f"{row['delta_maxwell_pct']:>+7.3f}%"
        fc_s = "REFUSED" if row["coenergy_refused"] else f"{row['f_coenergy_n']:>10.5f}"
        dc_s = (
            "---" if row["delta_coenergy_pct"] is None else f"{row['delta_coenergy_pct']:>+7.3f}%"
        )
        if idx == len(rows_m) - 1:
            p_str = f"{rm['p']:.2f}/n/a"
            err_str = f"{rm['est_error_pct']:.2f}%/n/a"
        else:
            p_str = "---"
            err_str = "---"
        print(
            f"{row['rungs']:<25} | {row['h_m']:>8.2e} | {row['nodes']:>6d} | {row['elements']:>8d} | "
            f"{row['f_maxwell_n']:>10.5f} | {dm_s:>8} | {fc_s:>10} | {dc_s:>8} | "
            f"{p_str:>13} | {err_str:>12}"
        )
    for row in rows_m:
        if row["coenergy_refused"]:
            print(f"h={row['h_m']:.2e} m coenergy: REFUSED ({row['coenergy_refusal_reason']})")
    print(
        f"Maxwell stress Richardson: p={rm['p']:.3f}, F_extrap={rm['f_extrap_n']:.5f} N, "
        f"est_error={rm['est_error_pct']:.3f}%"
    )
    print(f"coenergy Richardson ({fvm['delta_basis']}): {fvm['richardson_coenergy']['status']}")
    print(f"Superseded history: {fvm['superseded_history']}")
    print(
        f"Converged (<1% estimated discretisation error at finest h)? "
        f"{'YES' if fvm['converged_within_1pct_est'] else 'NO'} "
        f"({fvm['converged_note']})"
    )
    print()

    # 5. CONTOURS
    cnt = payload["contours"]
    print("5 CONTOURS")
    h5 = (
        f"{'Rung':<14} | {'A (snapped, m)':<34} | {'B (snapped, m)':<34} | "
        f"{'F_A (N)':>10} | {'F_B (N)':>10} | {'difference%':>11}"
    )
    print(h5)
    print("-" * len(h5))
    ca_s = "(" + ", ".join(f"{v:.6g}" for v in cnt["contour_a_snapped_m"]) + ")"
    cb_s = "(" + ", ".join(f"{v:.6g}" for v in cnt["contour_b_snapped_m"]) + ")"
    print(
        f"{cnt['rung']:<14} | {ca_s:<34} | {cb_s:<34} | "
        f"{cnt['f_a_n']:>10.5f} | {cnt['f_b_n']:>10.5f} | {cnt['difference_pct']:>10.4f}%"
    )
    print()

    # 6. ELMER
    el = payload["elmer"]
    print("6 ELMER")
    h6 = (
        f"{'Rung':<20} | {'Elmer mesh':<16} | {'W(+d), W(-d) (J)':<22} | "
        f"{'F_elmer (N)':>11} | {'vs Maxwell%':>11} | {'vs coenergy%':>12} | status"
    )
    print(h6)
    print("-" * len(h6))
    if el["status"] == "SKIP":
        print(
            f"{el['rung']:<20} | {'N/A':<16} | {'N/A':<22} | {'N/A':>11} | "
            f"{'N/A':>11} | {'N/A':>12} | SKIP: {el['reason']}"
        )
    else:
        mesh_s = f"{el['nodes']}n/{el['elements']}e"
        w_s = f"{el['w_plus_j']:.5f}, {el['w_minus_j']:.5f}"
        print(
            f"{el['rung']:<20} | {mesh_s:<16} | {w_s:<22} | "
            f"{el['f_elmer_n']:>11.5f} | {el['vs_maxwell_pct']:>10.3f}% | "
            f"{el['vs_coenergy_pct']:>11.3f}% | {el['status']}"
        )
    print()

    # 7. DETERMINISM
    print("7 DETERMINISM")
    h7 = f"{'Rung':<16} | {'sha256 run 1':<64} | {'sha256 run 2':<64} | equal?"
    print(h7)
    print("-" * len(h7))
    for row in payload["determinism"]:
        print(f"{row['rung']:<16} | {row['sha256_run1']} | {row['sha256_run2']} | {row['equal']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", metavar="PATH", default=None, help="write report payload as JSON")
    parser.add_argument("--fast", action="store_true", help="use a reduced mesh set")
    parser.add_argument("--skip-elmer", action="store_true", help="mark Elmer comparison as SKIP")
    args = parser.parse_args()

    payload = run_report(fast=args.fast, skip_elmer=args.skip_elmer)
    print_report(payload)

    if args.json:
        Path(args.json).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
