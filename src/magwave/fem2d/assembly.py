"""Finite-element assembly for linear axisymmetric magnetostatics (A_phi formulation).

Weak form with azimuthal magnetic vector potential ``A = A_phi(r, z) e_phi``, test function
``v`` (vanishing on the Dirichlet boundary), element-wise reluctivity ``nu_e = 1 / (mu_0 * mu_r)``,
volume current density ``J_phi``, and surface current sheet ``K_phi``:

.. math::

    \\int_\\Omega \\nu \\left[ r \\left(\\frac{\\partial A}{\\partial r}\\frac{\\partial v}{\\partial r}
    + \\frac{\\partial A}{\\partial z}\\frac{\\partial v}{\\partial z}\\right)
    + \\frac{\\partial A}{\\partial r} v + A \\frac{\\partial v}{\\partial r} + \\frac{A v}{r} \\right] dr\\,dz
    = \\int_\\Omega J_\\phi v r\\,dr\\,dz + \\int_{\\Gamma_s} K_\\phi v r\\,dl

Element stiffness matrices use linear triangles with **one-point (centroid) quadrature**
(``r = r_c = (r_0 + r_1 + r_2)/3``, ``N_i(centroid) = 1/3``, gradients ``b_i = dN_i/dr``,
``c_i = dN_i/dz``):

.. math::

    K^e_{ij} = \\nu_e A_e \\left[ r_c (b_i b_j + c_i c_j) + \\frac{1}{3}(b_i + b_j) + \\frac{1}{9 r_c} \\right]

Volume and sheet current load vectors are integrated exactly. On a tensor-product ``(r, z)``
grid, every node is shared by all incident elements (no hanging nodes), so element-wise ``nu_e``
produces a staircase approximation of curved material interfaces rather than a node-conforming
interface curve.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from ..units import MU0
from .mesh import Mesh

_TOL = 1e-12


def element_geometry(
    mesh: Mesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(area, r_c, z_c, b, c)`` for all ``m`` triangles in ``mesh``.

    ``b`` and ``c`` have shape ``(m, 3)`` with ``b_i = dN_i/dr`` and ``c_i = dN_i/dz``.
    """
    pts = mesh.nodes[mesh.elements]  # (m, 3, 2)
    r0 = pts[:, 0, 0]
    z0 = pts[:, 0, 1]
    r1 = pts[:, 1, 0]
    z1 = pts[:, 1, 1]
    r2 = pts[:, 2, 0]
    z2 = pts[:, 2, 1]

    det = (r1 - r0) * (z2 - z0) - (r2 - r0) * (z1 - z0)
    if np.any(det <= 0.0):
        raise ValueError("mesh contains degenerate or clockwise elements (non-positive det)")

    area = 0.5 * det
    r_c = (r0 + r1 + r2) / 3.0
    z_c = (z0 + z1 + z2) / 3.0
    b = np.column_stack((z1 - z2, z2 - z0, z0 - z1)) / det[:, None]
    c = np.column_stack((r2 - r1, r0 - r2, r1 - r0)) / det[:, None]
    return area, r_c, z_c, b, c


def element_geometric_stiffness(mesh: Mesh) -> np.ndarray:
    """Return the ``(m, 3, 3)`` reluctivity-free geometric element matrices ``G^e``."""
    area, r_c, _, b, c = element_geometry(mesh)
    grad_dot = b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]
    b_sum = b[:, :, None] + b[:, None, :]
    return area[:, None, None] * (
        r_c[:, None, None] * grad_dot + b_sum / 3.0 + 1.0 / (9.0 * r_c[:, None, None])
    )


def stiffness(mesh: Mesh, nu: float | np.ndarray = 1.0 / MU0) -> sp.csr_matrix:
    """Assemble the symmetric global stiffness matrix ``K`` in CSR format.

    ``nu`` may be a positive scalar reluctivity (applied uniformly) or a ``(m,)`` array of
    positive per-element reluctivities where ``m = mesh.elements.shape[0]``.
    """
    m = int(mesh.elements.shape[0])
    if np.ndim(nu) == 0:
        if isinstance(nu, bool):
            raise ValueError(f"reluctivity nu must be a positive number, got {nu!r}")
        nu_f = float(nu)
        if not math.isfinite(nu_f) or nu_f <= 0.0:
            raise ValueError(f"reluctivity nu must be positive and finite, got {nu!r}")
        nu_factor: float | np.ndarray = nu_f
    else:
        nu_arr = np.asarray(nu, dtype=np.float64)
        if nu_arr.shape != (m,):
            raise ValueError(
                f"element reluctivity array must have shape ({m},), got {nu_arr.shape}"
            )
        if not np.all(np.isfinite(nu_arr)) or np.any(nu_arr <= 0.0):
            raise ValueError("element reluctivity entries must be positive and finite")
        nu_factor = nu_arr[:, None, None]

    k_elem = nu_factor * element_geometric_stiffness(mesh)

    n_nodes = int(mesh.nodes.shape[0])
    rows = np.repeat(mesh.elements, 3, axis=1).ravel()
    cols = np.tile(mesh.elements, (1, 3)).ravel()
    return sp.csr_matrix((k_elem.ravel(), (rows, cols)), shape=(n_nodes, n_nodes))


def load_current_density(
    mesh: Mesh,
    region: tuple[float, float, float, float],
    j_phi: float,
) -> np.ndarray:
    """Assemble the exact volume-current load vector over ``region = (r0, r1, z0, z1)``.

    Elements whose centroid lies inside ``region`` receive the exact linear-triangle load
    ``f^e_i = j_phi * A_e * (2 * r_i + r_j + r_k) / 12``.
    """
    r0, r1, z0, z1 = (float(v) for v in region)
    if not (0.0 <= r0 < r1) or not (z0 < z1):
        raise ValueError(f"invalid volume current region {(r0, r1, z0, z1)!r}")

    area, r_c, z_c, _, _ = element_geometry(mesh)
    mask = (r_c >= r0) & (r_c <= r1) & (z_c >= z0) & (z_c <= z1)

    n_nodes = int(mesh.nodes.shape[0])
    load = np.zeros(n_nodes, dtype=np.float64)
    if not np.any(mask):
        return load

    sel_elems = mesh.elements[mask]
    sel_area = area[mask]
    r_nodes = mesh.nodes[sel_elems, 0]  # (m_sel, 3)
    r_sum = np.sum(r_nodes, axis=1, keepdims=True)
    f_elem = float(j_phi) * sel_area[:, None] * (r_nodes + r_sum) / 12.0
    np.add.at(load, sel_elems, f_elem)
    return load


def load_sheet(
    mesh: Mesh,
    radius: float,
    z_range: tuple[float, float],
    k_phi: float,
) -> np.ndarray:
    """Assemble the exact surface-current sheet load along ``r = radius``, ``z in z_range``.

    Raises ``ValueError`` if ``radius <= 0``, ``z_range[1] <= z_range[0]``, or no mesh
    edge lies on the sheet line (callers must include ``radius`` in ``breakpoints_r``).
    """
    r_sheet = float(radius)
    z0, z1 = float(z_range[0]), float(z_range[1])
    if r_sheet <= 0.0:
        raise ValueError(f"sheet radius must be positive, got {radius!r}")
    if z1 <= z0:
        raise ValueError(f"sheet z_range must satisfy z1 > z0, got {z_range!r}")

    elems = mesh.elements
    e01 = np.sort(elems[:, (0, 1)], axis=1)
    e12 = np.sort(elems[:, (1, 2)], axis=1)
    e20 = np.sort(elems[:, (2, 0)], axis=1)
    all_edges = np.vstack((e01, e12, e20))
    unique_edges = np.unique(all_edges, axis=0)

    p_a = mesh.nodes[unique_edges[:, 0]]
    p_b = mesh.nodes[unique_edges[:, 1]]
    on_radius = (np.abs(p_a[:, 0] - r_sheet) <= _TOL) & (np.abs(p_b[:, 0] - r_sheet) <= _TOL)
    z_mid = 0.5 * (p_a[:, 1] + p_b[:, 1])
    in_z = (
        (z_mid >= z0)
        & (z_mid <= z1)
        & (np.minimum(p_a[:, 1], p_b[:, 1]) >= z0 - _TOL)
        & (np.maximum(p_a[:, 1], p_b[:, 1]) <= z1 + _TOL)
    )
    sheet_mask = on_radius & in_z
    if not np.any(sheet_mask):
        raise ValueError(
            f"no mesh edge found on sheet line r={r_sheet}, z in [{z0}, {z1}]; "
            "ensure radius and z endpoints are mesh breakpoints"
        )

    matched = unique_edges[sheet_mask]
    lengths = np.abs(p_b[sheet_mask, 1] - p_a[sheet_mask, 1])
    nodal = 0.5 * float(k_phi) * r_sheet * lengths

    load = np.zeros(int(mesh.nodes.shape[0]), dtype=np.float64)
    np.add.at(load, matched[:, 0], nodal)
    np.add.at(load, matched[:, 1], nodal)
    return load


def load_magnetization_offset(
    mesh: Mesh,
    mask: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Assemble the volume offset load ``f_offset(offsets)`` for magnet elements in ``mask``.

    For each masked element ``e`` with centroid radius ``r_c`` and area ``A_e``, the
    centroid quadrature weight ``w_e = A_e * r_c`` matches :func:`element_geometric_stiffness`,
    giving nodal load ``offsets_e * w_e * (b_i + 1 / (3 * r_c))``. ``offsets`` may have
    shape ``(m_elems,)`` or ``(count_nonzero(mask),)``.
    """
    m_elems = int(mesh.elements.shape[0])
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != (m_elems,):
        raise ValueError(f"mask must have shape ({m_elems},), got {mask_arr.shape}")

    n_masked = int(np.count_nonzero(mask_arr))
    offsets_arr = np.asarray(offsets, dtype=np.float64)
    if offsets_arr.shape == (m_elems,):
        sel_offsets = offsets_arr[mask_arr]
    elif offsets_arr.shape == (n_masked,):
        sel_offsets = offsets_arr
    else:
        raise ValueError(
            f"offsets must have shape ({m_elems},) or ({n_masked},), got {offsets_arr.shape}"
        )
    if not np.all(np.isfinite(sel_offsets)):
        raise ValueError("offsets entries must be finite")

    load = np.zeros(int(mesh.nodes.shape[0]), dtype=np.float64)
    if n_masked == 0 or np.all(sel_offsets == 0.0):
        return load

    area, r_c, _, b, _ = element_geometry(mesh)
    w_e = area[mask_arr] * r_c[mask_arr]
    bz_coef = b[mask_arr] + 1.0 / (3.0 * r_c[mask_arr, None])
    f_elem = (sel_offsets * w_e)[:, None] * bz_coef
    np.add.at(load, mesh.elements[mask_arr], f_elem)
    return load
