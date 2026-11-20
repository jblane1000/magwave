"""Linear direct solve and Dirichlet boundary conditions for axisymmetric FEM."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..units import MU0
from .assembly import stiffness
from .mesh import Mesh, MeshStats

_TOL = 1e-12


@dataclass(frozen=True)
class SolveReport:
    """Linear solve metadata and verification residual (PRD FR-042)."""

    unknowns: int
    constrained: int
    iterations: int
    relative_residual: float
    wall_time_s: float
    mesh: MeshStats
    solver: str


def dirichlet_nodes(mesh: Mesh) -> np.ndarray:
    """Return sorted node indices on the symmetry axis (``r == 0``) or outer boundary."""
    r = mesh.nodes[:, 0]
    z = mesh.nodes[:, 1]
    r_max = float(mesh.r_axis[-1])
    z_min = float(mesh.z_axis[0])
    z_max = float(mesh.z_axis[-1])
    mask = (r <= _TOL) | (r >= r_max - _TOL) | (z <= z_min + _TOL) | (z >= z_max - _TOL)
    return np.flatnonzero(mask).astype(np.int64)


def solve_axisymmetric(
    mesh: Mesh,
    load: np.ndarray,
    nu: float = 1.0 / MU0,
    constrained: np.ndarray | Sequence[int] | None = None,
) -> tuple[np.ndarray, SolveReport]:
    """Assemble and solve ``K A = f`` with symmetric Dirichlet elimination (``A = 0``).

    Returns ``(a_phi, SolveReport)`` where ``a_phi`` is the nodal azimuthal vector potential
    in Wb/m (T*m).
    """
    n_nodes = int(mesh.nodes.shape[0])
    f_vec = np.asarray(load, dtype=np.float64).copy()
    if f_vec.shape != (n_nodes,):
        raise ValueError(f"load must have shape ({n_nodes},), got {f_vec.shape}")

    if constrained is None:
        constrained_idx = dirichlet_nodes(mesh)
    else:
        constrained_idx = np.unique(np.asarray(constrained, dtype=np.int64))
        if np.any(constrained_idx < 0) or np.any(constrained_idx >= n_nodes):
            raise ValueError("constrained node indices out of range")

    k_mat = stiffness(mesh, nu=nu)

    is_constrained = np.zeros(n_nodes, dtype=bool)
    is_constrained[constrained_idx] = True
    free_mask = ~is_constrained

    # Symmetric elimination: zero constrained rows and columns, set diagonal to 1 and f_i = 0.
    keep_diag = sp.diags(free_mask.astype(np.float64), format="csr")
    pin_diag = sp.diags(is_constrained.astype(np.float64), format="csr")
    k_bc = (keep_diag @ k_mat @ keep_diag) + pin_diag
    k_bc.eliminate_zeros()

    f_bc = f_vec
    f_bc[is_constrained] = 0.0

    t0 = time.perf_counter()
    a_phi = np.asarray(spla.spsolve(k_bc.tocsc(), f_bc), dtype=np.float64)
    wall_time_s = time.perf_counter() - t0
    a_phi[is_constrained] = 0.0

    f_norm = float(np.linalg.norm(f_bc))
    if f_norm == 0.0:
        rel_res = 0.0
    else:
        residual = f_bc - (k_bc @ a_phi)
        rel_res = float(np.linalg.norm(residual) / f_norm)

    report = SolveReport(
        unknowns=int(np.count_nonzero(free_mask)),
        constrained=int(constrained_idx.size),
        iterations=1,
        relative_residual=rel_res,
        wall_time_s=wall_time_s,
        mesh=mesh.stats(),
        solver="scipy.sparse.linalg.spsolve (SuperLU)",
    )
    return a_phi, report
