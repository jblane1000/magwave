"""Nonlinear Newton–Raphson solver for axisymmetric ``A_phi`` magnetostatics (M5.2 Stage B).

Residual & consistent tangent Jacobian
--------------------------------------
With element geometric stiffness matrix ``G^e`` from
:func:`magwave.fem2d.assembly.element_geometric_stiffness` and element reluctivity
``nu_e = nu(|B_e|) = H(|B_e|) / |B_e|`` evaluated at the centroid flux density
``|B_e| = sqrt(B_r^2 + B_z^2)`` (``B_r = -sum_k c_k A_k``,
``B_z = sum_k (b_k + 1/(3 r_c)) A_k``):

.. math::

    R(A) = K(\\nu(|B|)) A - f, \\qquad R_i = A_i - g_i \\text{ on constrained rows.}

The consistent tangent Jacobian differentiates ``nu_e(|B_e|) * G^e A_e`` with respect to
the element nodal vector ``A_e``:

.. math::

    J = K(\\nu) + \\sum_e (G^e A_e) \\otimes \\left( \\frac{d\\nu}{d|B|} \\frac{d|B|}{dA_e} \\right),

where ``(d|B|/dA_k)_e = (B_r (-c_k) + B_z (b_k + 1/(3 r_c))) / |B_e|`` and
``(dnu/d|B|)_e = (H'(|B_e|)|B_e| - H(|B_e|)) / |B_e|^2``. When ``|B_e| <= 1e-15`` T,
:meth:`BhCurve.reluctivity_and_derivative` returns ``nu(0) = 1 / (MU0 * mu_r)`` and
``dnu/d|B| = 0`` directly and ``d|B|/dA`` is set to zero, eliminating ``0 / 0`` division.
``J`` is assembled as one CSR matrix without allocating any 3-index tensor.

Line search
-----------
Backtracking Armijo line search on ``||R||_2``:

* ``ARMIJO_ALPHA0 = 1.0``: always tries the full Newton step first so quadratic asymptotic
  convergence is preserved near the solution.
* ``ARMIJO_RHO = 0.5``: halves the trial step on rejection.
* ``ARMIJO_C = 1e-4``: Nocedal & Wright sufficient-decrease slope parameter
  ``||R(A + alpha dA)||_2 <= (1 - c * alpha) * ||R(A)||_2``.
* ``ARMIJO_MAX_HALVINGS = 12``: caps step reduction at ``2^-12 = 2.44e-4``; if no step is
  accepted, the iteration fails loudly via :class:`NonlinearConvergenceError` rather than
  taking an unaccepted step or switching solvers silently.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..materials.magnets import DemagCurve
from ..materials.nonlinear import BhCurve
from ..units import MU0
from .assembly import element_geometric_stiffness, element_geometry
from .mesh import Mesh
from .solve import SolveReport, _dirichlet_values_vector, dirichlet_nodes

ARMIJO_ALPHA0: float = 1.0
ARMIJO_RHO: float = 0.5
ARMIJO_C: float = 1e-4
ARMIJO_MAX_HALVINGS: int = 12

DEFAULT_TOL: float = 1e-8
DEFAULT_MAX_ITERATIONS: int = 50
_B_ZERO_TOL: float = 1e-15


class NonlinearConvergenceError(RuntimeError):
    """Raised when Newton iteration or Armijo backtracking fails to converge."""

    def __init__(
        self,
        message: str,
        *,
        iterations: int,
        max_iterations: int,
        final_relative_residual: float,
        residual_history: tuple[float, ...],
        reason: str,
    ) -> None:
        super().__init__(message)
        self.iterations = iterations
        self.max_iterations = max_iterations
        self.final_relative_residual = final_relative_residual
        self.residual_history = residual_history
        self.reason = reason


@dataclass(frozen=True)
class _NonlinearWorkspace:
    n_nodes: int
    m_elems: int
    elems: np.ndarray
    g_elem: np.ndarray
    br_coef: np.ndarray
    bz_coef: np.ndarray
    w_elem: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    is_constrained: np.ndarray
    keep_diag: sp.csr_matrix
    pin_diag: sp.csr_matrix
    g_vec: np.ndarray
    f_vec: np.ndarray
    base_nu: np.ndarray
    element_curves: tuple[tuple[np.ndarray, BhCurve], ...]
    magnet_regions: tuple[tuple[np.ndarray, DemagCurve, float], ...]


def count_interface_nodes(mesh: Mesh, region_ids: np.ndarray) -> int:
    """Count mesh nodes shared by elements belonging to different material regions."""
    ids = np.asarray(region_ids, dtype=np.int64)
    if ids.shape != (mesh.elements.shape[0],):
        raise ValueError(f"region_ids must have shape ({mesh.elements.shape[0]},), got {ids.shape}")
    unique_regions = np.unique(ids)
    if unique_regions.size <= 1:
        return 0
    n_nodes = int(mesh.nodes.shape[0])
    touch_count = np.zeros(n_nodes, dtype=np.int64)
    for rid in unique_regions:
        nodes_in_region = np.unique(mesh.elements[ids == rid])
        touch_count[nodes_in_region] += 1
    return int(np.count_nonzero(touch_count >= 2))


def _build_workspace(
    mesh: Mesh,
    load: np.ndarray,
    element_curves: Sequence[tuple[np.ndarray, BhCurve]],
    *,
    base_nu: np.ndarray | None = None,
    constrained: np.ndarray | Sequence[int] | None = None,
    values: Mapping[int, float] | np.ndarray | Sequence[float] | None = None,
    magnet_regions: Sequence[tuple[np.ndarray, DemagCurve] | tuple[np.ndarray, DemagCurve, float]]
    | None = None,
) -> _NonlinearWorkspace:
    n_nodes = int(mesh.nodes.shape[0])
    m_elems = int(mesh.elements.shape[0])
    f_vec = np.asarray(load, dtype=np.float64).copy()
    if f_vec.shape != (n_nodes,):
        raise ValueError(f"load must have shape ({n_nodes},), got {f_vec.shape}")
    if not np.all(np.isfinite(f_vec)):
        raise ValueError("load vector must be finite")

    if base_nu is None:
        base_nu_arr = np.full(m_elems, 1.0 / MU0, dtype=np.float64)
    else:
        base_nu_arr = np.asarray(base_nu, dtype=np.float64).copy()
        if base_nu_arr.shape != (m_elems,):
            raise ValueError(f"base_nu must have shape ({m_elems},), got {base_nu_arr.shape}")
        if not np.all(np.isfinite(base_nu_arr)) or np.any(base_nu_arr <= 0.0):
            raise ValueError("base_nu entries must be positive and finite")

    parsed_curves: list[tuple[np.ndarray, BhCurve]] = []
    occupied_mask = np.zeros(m_elems, dtype=bool)
    for raw_mask, curve in element_curves:
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (m_elems,):
            raise ValueError(f"element mask must have shape ({m_elems},), got {mask.shape}")
        parsed_curves.append((mask, curve))
        occupied_mask |= mask

    parsed_magnets: list[tuple[np.ndarray, DemagCurve, float]] = []
    if magnet_regions:
        for item in magnet_regions:
            if len(item) == 2:
                raw_mask, mag_curve = item
                sign_f = 1.0
            elif len(item) == 3:
                raw_mask, mag_curve, raw_sign = item
                sign_f = float(raw_sign)
                if sign_f not in (-1.0, 1.0):
                    raise ValueError(f"magnet region sign must be +1 or -1, got {raw_sign!r}")
            else:
                raise ValueError(
                    "magnet_regions entries must be (mask, DemagCurve) or (mask, DemagCurve, sign)"
                )
            mask = np.asarray(raw_mask, dtype=bool)
            if mask.shape != (m_elems,):
                raise ValueError(f"magnet mask must have shape ({m_elems},), got {mask.shape}")
            if np.any(occupied_mask & mask):
                raise ValueError("magnet_regions mask overlaps an existing nonlinear region")
            occupied_mask |= mask
            parsed_magnets.append((mask, mag_curve, sign_f))

    if constrained is None:
        raw_constrained = dirichlet_nodes(mesh)
        constrained_idx = raw_constrained
    else:
        raw_constrained = np.asarray(constrained, dtype=np.int64)
        if raw_constrained.ndim != 1:
            raise ValueError(f"constrained must be 1-D, got shape {raw_constrained.shape}")
        if np.any(raw_constrained < 0) or np.any(raw_constrained >= n_nodes):
            raise ValueError("constrained node indices out of range")
        constrained_idx = np.unique(raw_constrained)

    is_constrained = np.zeros(n_nodes, dtype=bool)
    is_constrained[constrained_idx] = True
    free_mask = ~is_constrained

    if values is None:
        g_vec = np.zeros(n_nodes, dtype=np.float64)
    else:
        g_vec = _dirichlet_values_vector(
            n_nodes, raw_constrained, constrained_idx, is_constrained, values
        )

    area, r_c, _, b_grad, c_grad = element_geometry(mesh)
    g_elem = element_geometric_stiffness(mesh)
    br_coef = -c_grad
    bz_coef = b_grad + 1.0 / (3.0 * r_c[:, None])
    w_elem = area * r_c

    rows = np.repeat(mesh.elements, 3, axis=1).ravel()
    cols = np.tile(mesh.elements, (1, 3)).ravel()
    keep_diag = sp.diags(free_mask.astype(np.float64), format="csr")
    pin_diag = sp.diags(is_constrained.astype(np.float64), format="csr")

    return _NonlinearWorkspace(
        n_nodes=n_nodes,
        m_elems=m_elems,
        elems=mesh.elements,
        g_elem=g_elem,
        br_coef=br_coef,
        bz_coef=bz_coef,
        w_elem=w_elem,
        rows=rows,
        cols=cols,
        is_constrained=is_constrained,
        keep_diag=keep_diag,
        pin_diag=pin_diag,
        g_vec=g_vec,
        f_vec=f_vec,
        base_nu=base_nu_arr,
        element_curves=tuple(parsed_curves),
        magnet_regions=tuple(parsed_magnets),
    )


def _evaluate_residual(
    ws: _NonlinearWorkspace, a_phi: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
]:
    ae = a_phi[ws.elems]
    b_r = np.sum(ws.br_coef * ae, axis=1)
    b_z = np.sum(ws.bz_coef * ae, axis=1)
    b_mag = np.hypot(b_r, b_z)

    nu = ws.base_nu.copy()
    dnu = np.zeros(ws.m_elems, dtype=np.float64)
    for mask, curve in ws.element_curves:
        if np.any(mask):
            nu_s, dnu_s = curve.reluctivity_and_derivative(b_mag[mask])
            nu[mask] = nu_s
            dnu[mask] = dnu_s

    if not ws.magnet_regions:
        ga = np.einsum("eij,ej->ei", ws.g_elem, ae)
        r_vec = -ws.f_vec.copy()
        np.add.at(r_vec, ws.elems, nu[:, None] * ga)
        r_vec[ws.is_constrained] = a_phi[ws.is_constrained] - ws.g_vec[ws.is_constrained]
        return r_vec, b_mag, b_r, b_z, nu, dnu, None, ga

    dc = np.zeros(ws.m_elems, dtype=np.float64)
    f_off_total = np.zeros(ws.n_nodes, dtype=np.float64)
    has_nonzero_offset = False
    for mask, mag_curve, sign in ws.magnet_regions:
        if np.any(mask):
            s_m, ds_m, c_m, dc_m = mag_curve.slope_and_offset(b_mag[mask])
            nu[mask] = s_m
            dnu[mask] = ds_m
            dc[mask] = sign * dc_m
            c0 = -mag_curve.br() / MU0
            offsets = sign * (c0 - c_m)
            if not np.all(offsets == 0.0):
                has_nonzero_offset = True
                f_elem = (offsets * ws.w_elem[mask])[:, None] * ws.bz_coef[mask]
                np.add.at(f_off_total, ws.elems[mask], f_elem)

    ga = np.einsum("eij,ej->ei", ws.g_elem, ae)
    r_vec = -ws.f_vec.copy()
    if has_nonzero_offset:
        r_vec -= f_off_total
    np.add.at(r_vec, ws.elems, nu[:, None] * ga)
    r_vec[ws.is_constrained] = a_phi[ws.is_constrained] - ws.g_vec[ws.is_constrained]
    return r_vec, b_mag, b_r, b_z, nu, dnu, dc, ga


def _assemble_jacobian(
    ws: _NonlinearWorkspace,
    b_mag: np.ndarray,
    b_r: np.ndarray,
    b_z: np.ndarray,
    nu: np.ndarray,
    dnu: np.ndarray,
    ga: np.ndarray,
    dc: np.ndarray | None = None,
) -> sp.csr_matrix:
    nonzero = b_mag > _B_ZERO_TOL
    safe_b = np.where(nonzero, b_mag, 1.0)
    db_da = np.where(
        nonzero[:, None],
        (b_r[:, None] * ws.br_coef + b_z[:, None] * ws.bz_coef) / safe_b[:, None],
        0.0,
    )
    j_elem = nu[:, None, None] * ws.g_elem + (
        dnu[:, None, None] * ga[:, :, None] * db_da[:, None, :]
    )
    if dc is not None:
        for mask, _, _ in ws.magnet_regions:
            if np.any(mask):
                j_elem[mask] += (
                    (ws.w_elem[mask] * dc[mask])[:, None, None]
                    * ws.bz_coef[mask, :, None]
                    * db_da[mask, None, :]
                )
    j_mat = sp.csr_matrix((j_elem.ravel(), (ws.rows, ws.cols)), shape=(ws.n_nodes, ws.n_nodes))
    j_bc = (ws.keep_diag @ j_mat @ ws.keep_diag) + ws.pin_diag
    j_bc.eliminate_zeros()
    return j_bc


def assemble_nonlinear_system(
    mesh: Mesh,
    a_phi: np.ndarray,
    load: np.ndarray,
    element_curves: Sequence[tuple[np.ndarray, BhCurve]],
    *,
    base_nu: np.ndarray | None = None,
    constrained: np.ndarray | Sequence[int] | None = None,
    values: Mapping[int, float] | np.ndarray | Sequence[float] | None = None,
    magnet_regions: Sequence[tuple[np.ndarray, DemagCurve] | tuple[np.ndarray, DemagCurve, float]]
    | None = None,
) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray, np.ndarray]:
    """Assemble ``(residual, jacobian_bc, b_mag, nu_elem)`` at nodal potential ``a_phi``."""
    a_vec = np.asarray(a_phi, dtype=np.float64)
    n_nodes = int(mesh.nodes.shape[0])
    if a_vec.shape != (n_nodes,):
        raise ValueError(f"a_phi must have shape ({n_nodes},), got {a_vec.shape}")
    ws = _build_workspace(
        mesh,
        load,
        element_curves,
        base_nu=base_nu,
        constrained=constrained,
        values=values,
        magnet_regions=magnet_regions,
    )
    r_vec, b_mag, b_r, b_z, nu, dnu, dc, ga = _evaluate_residual(ws, a_vec)
    j_bc = _assemble_jacobian(ws, b_mag, b_r, b_z, nu, dnu, ga, dc=dc)
    return r_vec, j_bc, b_mag, nu


def solve_axisymmetric_nonlinear(
    mesh: Mesh,
    load: np.ndarray,
    element_curves: Sequence[tuple[np.ndarray, BhCurve]],
    *,
    base_nu: np.ndarray | None = None,
    constrained: np.ndarray | Sequence[int] | None = None,
    values: Mapping[int, float] | np.ndarray | Sequence[float] | None = None,
    tol: float = DEFAULT_TOL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    magnet_regions: Sequence[tuple[np.ndarray, DemagCurve] | tuple[np.ndarray, DemagCurve, float]]
    | None = None,
) -> tuple[np.ndarray, SolveReport, tuple[float, ...], np.ndarray]:
    """Solve nonlinear axisymmetric ``K(nu(|B|)) A = f`` with Newton + Armijo backtracking.

    Returns ``(a_phi, solve_report, residual_history, nu_final)``.
    Raises :class:`NonlinearConvergenceError` if Armijo backtracking fails or the iteration
    cap is exceeded.
    """
    if (
        isinstance(tol, bool)
        or not isinstance(tol, int | float)
        or not math.isfinite(tol)
        or tol <= 0.0
    ):
        raise ValueError(f"tol must be positive and finite, got {tol!r}")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError(f"max_iterations must be a positive integer, got {max_iterations!r}")
    tol_f = float(tol)

    ws = _build_workspace(
        mesh,
        load,
        element_curves,
        base_nu=base_nu,
        constrained=constrained,
        values=values,
        magnet_regions=magnet_regions,
    )
    a_phi = ws.g_vec.copy()

    t0 = time.perf_counter()
    r_curr, b_mag, b_r, b_z, nu_curr, dnu_curr, dc_curr, ga_curr = _evaluate_residual(ws, a_phi)
    r_curr_norm = float(np.linalg.norm(r_curr))

    f_free = ws.f_vec.copy()
    f_free[ws.is_constrained] = 0.0
    f_free_norm = float(np.linalg.norm(f_free))
    ref_norm = f_free_norm if f_free_norm > 0.0 else r_curr_norm

    n_constrained = int(np.count_nonzero(ws.is_constrained))
    n_unknowns = ws.n_nodes - n_constrained

    if ref_norm == 0.0:
        report = SolveReport(
            unknowns=n_unknowns,
            constrained=n_constrained,
            iterations=1,
            relative_residual=0.0,
            wall_time_s=time.perf_counter() - t0,
            mesh=mesh.stats(),
            solver="Newton + Armijo backtracking, scipy.sparse.linalg.spsolve",
        )
        return a_phi, report, (0.0,), nu_curr

    history: list[float] = []
    for iteration in range(1, max_iterations + 1):
        j_bc = _assemble_jacobian(ws, b_mag, b_r, b_z, nu_curr, dnu_curr, ga_curr, dc=dc_curr)
        d_a = np.asarray(spla.spsolve(j_bc.tocsc(), -r_curr), dtype=np.float64)
        d_a[ws.is_constrained] = 0.0
        if not np.all(np.isfinite(d_a)):
            rel_fail = history[-1] if history else (r_curr_norm / ref_norm)
            raise NonlinearConvergenceError(
                f"Newton linear solve produced non-finite step at iteration {iteration}",
                iterations=iteration,
                max_iterations=max_iterations,
                final_relative_residual=rel_fail,
                residual_history=tuple(history),
                reason="non-finite Newton step",
            )

        alpha = ARMIJO_ALPHA0
        accepted = False
        for _ in range(ARMIJO_MAX_HALVINGS + 1):
            a_trial = a_phi + alpha * d_a
            a_trial[ws.is_constrained] = ws.g_vec[ws.is_constrained]
            if ws.magnet_regions:
                try:
                    r_t, bm_t, br_t, bz_t, nu_t, dnu_t, dc_t, ga_t = _evaluate_residual(ws, a_trial)
                except ValueError:
                    alpha *= ARMIJO_RHO
                    continue
            else:
                r_t, bm_t, br_t, bz_t, nu_t, dnu_t, dc_t, ga_t = _evaluate_residual(ws, a_trial)
            r_t_norm = float(np.linalg.norm(r_t))
            if r_t_norm <= (1.0 - ARMIJO_C * alpha) * r_curr_norm:
                a_phi = a_trial
                r_curr, b_mag, b_r, b_z, nu_curr, dnu_curr, dc_curr, ga_curr = (
                    r_t,
                    bm_t,
                    br_t,
                    bz_t,
                    nu_t,
                    dnu_t,
                    dc_t,
                    ga_t,
                )
                r_curr_norm = r_t_norm
                accepted = True
                break
            alpha *= ARMIJO_RHO

        if not accepted:
            rel_fail = history[-1] if history else (r_curr_norm / ref_norm)
            raise NonlinearConvergenceError(
                f"Armijo backtracking failed at iteration {iteration} after "
                f"{ARMIJO_MAX_HALVINGS} halvings (rel_res={rel_fail:.6e})",
                iterations=iteration,
                max_iterations=max_iterations,
                final_relative_residual=rel_fail,
                residual_history=tuple(history),
                reason=f"Armijo backtracking failed after {ARMIJO_MAX_HALVINGS} halvings",
            )

        rel_res = r_curr_norm / ref_norm
        history.append(rel_res)
        if rel_res <= tol_f:
            wall_time_s = time.perf_counter() - t0
            report = SolveReport(
                unknowns=n_unknowns,
                constrained=n_constrained,
                iterations=iteration,
                relative_residual=rel_res,
                wall_time_s=wall_time_s,
                mesh=mesh.stats(),
                solver="Newton + Armijo backtracking, scipy.sparse.linalg.spsolve",
            )
            return a_phi, report, tuple(history), nu_curr

    final_rel = history[-1]
    raise NonlinearConvergenceError(
        f"Newton solver reached cap max_iterations={max_iterations} "
        f"(final_relative_residual={final_rel:.6e} > tol={tol_f:g})",
        iterations=max_iterations,
        max_iterations=max_iterations,
        final_relative_residual=final_rel,
        residual_history=tuple(history),
        reason=f"reached max_iterations={max_iterations} without meeting tol={tol_f:g}",
    )
