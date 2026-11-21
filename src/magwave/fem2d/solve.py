"""Linear direct solve and Dirichlet boundary conditions for axisymmetric FEM."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
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


def _dirichlet_values_vector(
    n_nodes: int,
    raw_constrained: np.ndarray,
    constrained_idx: np.ndarray,
    is_constrained: np.ndarray,
    values: Mapping[int, float] | np.ndarray | Sequence[float],
) -> np.ndarray:
    g_vec = np.zeros(n_nodes, dtype=np.float64)
    if isinstance(values, Mapping):
        for raw_idx, raw_val in values.items():
            if isinstance(raw_idx, bool) or not isinstance(raw_idx, int | np.integer):
                raise ValueError(f"Dirichlet node index must be int, got {raw_idx!r}")
            idx = int(raw_idx)
            if idx < 0 or idx >= n_nodes or not is_constrained[idx]:
                raise ValueError(f"Dirichlet value key {idx} is not a constrained node")
            if isinstance(raw_val, bool) or not isinstance(raw_val, int | float | np.number):
                raise ValueError(f"Dirichlet value for node {idx} must be finite float")
            val = float(raw_val)
            if not math.isfinite(val):
                raise ValueError(f"Dirichlet value for node {idx} must be finite, got {raw_val!r}")
            g_vec[idx] = val
        return g_vec

    val_arr = np.asarray(values, dtype=np.float64)
    if val_arr.ndim != 1:
        raise ValueError(f"Dirichlet values must be 1-D, got shape {val_arr.shape}")
    if not np.all(np.isfinite(val_arr)):
        raise ValueError("Dirichlet values must be finite")

    if val_arr.shape == (raw_constrained.shape[0],):
        if raw_constrained.shape[0] != constrained_idx.shape[0]:
            seen: dict[int, float] = {}
            for idx, val in zip(raw_constrained.tolist(), val_arr.tolist(), strict=True):
                if idx in seen and seen[idx] != val:
                    raise ValueError(f"conflicting Dirichlet values for duplicate node {idx}")
                seen[idx] = val
                g_vec[idx] = val
        else:
            g_vec[raw_constrained] = val_arr
    elif val_arr.shape == (constrained_idx.shape[0],):
        g_vec[constrained_idx] = val_arr
    elif val_arr.shape == (n_nodes,):
        g_vec[constrained_idx] = val_arr[constrained_idx]
    else:
        raise ValueError(
            f"Dirichlet values length {val_arr.shape[0]} matches neither constrained nodes "
            f"({constrained_idx.shape[0]}) nor mesh nodes ({n_nodes})"
        )
    return g_vec


def solve_axisymmetric(
    mesh: Mesh,
    load: np.ndarray,
    nu: float | np.ndarray = 1.0 / MU0,
    constrained: np.ndarray | Sequence[int] | None = None,
    values: Mapping[int, float] | np.ndarray | Sequence[float] | None = None,
) -> tuple[np.ndarray, SolveReport]:
    """Assemble and solve ``K A = f`` with symmetric Dirichlet elimination.

    ``nu`` may be a positive scalar or a ``(m,)`` element reluctivity array. ``values``
    prescribes Dirichlet ``A_phi`` at ``constrained`` nodes (default ``0`` everywhere).
    Returns ``(a_phi, SolveReport)`` where ``a_phi`` is the nodal azimuthal vector potential
    in Wb/m (T*m).
    """
    n_nodes = int(mesh.nodes.shape[0])
    f_vec = np.asarray(load, dtype=np.float64).copy()
    if f_vec.shape != (n_nodes,):
        raise ValueError(f"load must have shape ({n_nodes},), got {f_vec.shape}")

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

    k_mat = stiffness(mesh, nu=nu)

    is_constrained = np.zeros(n_nodes, dtype=bool)
    is_constrained[constrained_idx] = True
    free_mask = ~is_constrained

    g_vec: np.ndarray | None = None
    if values is not None:
        g_vec = _dirichlet_values_vector(
            n_nodes, raw_constrained, constrained_idx, is_constrained, values
        )

    # Symmetric elimination: zero constrained rows and columns, set diagonal to 1 and adjust RHS.
    keep_diag = sp.diags(free_mask.astype(np.float64), format="csr")
    pin_diag = sp.diags(is_constrained.astype(np.float64), format="csr")
    k_bc = (keep_diag @ k_mat @ keep_diag) + pin_diag
    k_bc.eliminate_zeros()

    if g_vec is None:
        f_bc = f_vec
        f_bc[is_constrained] = 0.0
    else:
        f_bc = f_vec - (k_mat @ g_vec)
        f_bc[is_constrained] = g_vec[is_constrained]

    t0 = time.perf_counter()
    a_phi = np.asarray(spla.spsolve(k_bc.tocsc(), f_bc), dtype=np.float64)
    wall_time_s = time.perf_counter() - t0
    if g_vec is None:
        a_phi[is_constrained] = 0.0
    else:
        a_phi[is_constrained] = g_vec[is_constrained]

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
