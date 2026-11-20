"""Magnetic flux density recovery from axisymmetric nodal ``A_phi``."""

from __future__ import annotations

import numpy as np

from .assembly import element_geometry
from .mesh import Mesh

_TOL = 1e-12


def element_field(mesh: Mesh, a_phi: np.ndarray) -> np.ndarray:
    """Return ``(m, 2)`` array of element-centroid ``(B_r, B_z)`` in tesla."""
    a_vec = np.asarray(a_phi, dtype=np.float64)
    if a_vec.shape != (mesh.nodes.shape[0],):
        raise ValueError(f"a_phi must have shape ({mesh.nodes.shape[0]},), got {a_vec.shape}")

    _, r_c, _, b, c = element_geometry(mesh)
    ae = a_vec[mesh.elements]  # (m, 3)
    dadr = np.sum(b * ae, axis=1)
    dadz = np.sum(c * ae, axis=1)
    a_c = np.mean(ae, axis=1)
    b_r = -dadz
    b_z = a_c / r_c + dadr
    return np.column_stack((b_r, b_z))


def _eval_elements(
    mesh: Mesh,
    a_vec: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    elem_indices: np.ndarray,
    r_val: float,
    z_val: float,
) -> tuple[float, float]:
    """Evaluate average ``(B_r, B_z)`` over ``elem_indices`` at ``(r_val, z_val)``."""
    ae = a_vec[mesh.elements[elem_indices]]
    dadr = np.sum(b[elem_indices] * ae, axis=1)
    dadz = np.sum(c[elem_indices] * ae, axis=1)
    if r_val < _TOL:
        # On the symmetry axis A = alpha*r so A/r -> dA/dr = alpha and B_z = 2 * dA/dr, B_r = 0.
        return 0.0, float(np.mean(2.0 * dadr))

    r0 = mesh.nodes[mesh.elements[elem_indices, 0], 0]
    z0 = mesh.nodes[mesh.elements[elem_indices, 0], 1]
    a_val = ae[:, 0] + dadr * (r_val - r0) + dadz * (z_val - z0)
    b_r = -dadz
    b_z = a_val / r_val + dadr
    return float(np.mean(b_r)), float(np.mean(b_z))


def sample_field(mesh: Mesh, a_phi: np.ndarray, points_rz: np.ndarray) -> np.ndarray:
    """Sample ``(B_r, B_z)`` in tesla at ``points_rz`` of shape ``(n, 2)`` or ``(2,)``.

    On the axis ``r < 1e-12``, uses the exact regularity limit ``B_r = 0``,
    ``B_z = 2 * sum b_i A_i``. At interior grid edges/nodes off-axis, averages across
    incident cell triangles so evaluations along mesh lines have no one-sided bias.
    """
    a_vec = np.asarray(a_phi, dtype=np.float64)
    if a_vec.shape != (mesh.nodes.shape[0],):
        raise ValueError(f"a_phi must have shape ({mesh.nodes.shape[0]},), got {a_vec.shape}")

    pts = np.asarray(points_rz, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_rz must have shape (n, 2) or (2,), got {np.shape(points_rz)}")
    if pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    located = mesh.locate(pts)
    _, _, _, b, c = element_geometry(mesh)

    nr = int(mesh.r_axis.shape[0])
    nz = int(mesh.z_axis.shape[0])
    n_cells = (nr - 1) * (nz - 1)

    pr = pts[:, 0]
    pz = pts[:, 1]
    ir0 = np.clip(np.searchsorted(mesh.r_axis, pr - _TOL, side="right") - 1, 0, nr - 2)
    ir1 = np.clip(np.searchsorted(mesh.r_axis, pr + _TOL, side="right") - 1, 0, nr - 2)
    iz0 = np.clip(np.searchsorted(mesh.z_axis, pz - _TOL, side="right") - 1, 0, nz - 2)
    iz1 = np.clip(np.searchsorted(mesh.z_axis, pz + _TOL, side="right") - 1, 0, nz - 2)

    out = np.empty((pts.shape[0], 2), dtype=np.float64)
    for k in range(pts.shape[0]):
        r_val = float(pr[k])
        z_val = float(pz[k])
        if r_val < _TOL:
            e_idx = int(located[k])
            dadr = float(np.dot(b[e_idx], a_vec[mesh.elements[e_idx]]))
            out[k, 0] = 0.0
            out[k, 1] = 2.0 * dadr
            continue

        if ir0[k] == ir1[k] and iz0[k] == iz1[k]:
            ir = int(ir0[k])
            iz = int(iz0[k])
            u = (r_val - float(mesh.r_axis[ir])) / float(mesh.r_axis[ir + 1] - mesh.r_axis[ir])
            v = (z_val - float(mesh.z_axis[iz])) / float(mesh.z_axis[iz + 1] - mesh.z_axis[iz])
            cell = ir * (nz - 1) + iz
            on_diag = abs((u + v) - 1.0) <= _TOL if (iz % 2) == 1 else abs(u - v) <= _TOL
            if on_diag:
                elems_k = np.array([cell, n_cells + cell], dtype=np.int64)
            else:
                elems_k = np.array([int(located[k])], dtype=np.int64)
        else:
            elems_list: list[int] = []
            for ic in range(int(ir0[k]), int(ir1[k]) + 1):
                for jc in range(int(iz0[k]), int(iz1[k]) + 1):
                    cell = ic * (nz - 1) + jc
                    elems_list.extend((cell, n_cells + cell))
            elems_k = np.array(elems_list, dtype=np.int64)

        out[k, 0], out[k, 1] = _eval_elements(mesh, a_vec, b, c, elems_k, r_val, z_val)

    return out
