"""Element-wise array reluctivity tests for :func:`magwave.fem2d.assembly.stiffness`."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from magwave.fem2d.assembly import element_geometry, stiffness
from magwave.fem2d.mesh import rectangular_mesh
from magwave.units import MU0


def test_stiffness_scalar_and_array_agree() -> None:
    mesh = rectangular_mesh(0.02, -0.01, 0.01, 0.01)
    m = mesh.elements.shape[0]
    assert m == 8

    nu_const = 1.0 / (MU0 * 250.0)
    k_scalar = stiffness(mesh, nu=nu_const)
    k_const_arr = stiffness(mesh, nu=np.full(m, nu_const))
    diff_const = np.linalg.norm((k_scalar - k_const_arr).toarray())
    norm_const = np.linalg.norm(k_scalar.toarray())
    assert diff_const / norm_const <= 1e-14

    # Two-region 2x2 mesh: assign nu_1 for z_c < 0, nu_2 for z_c > 0 and compare against
    # hand-built element-by-element sum using scalar element stiffness formulas.
    _, _, z_c, _, _ = element_geometry(mesh)
    nu_1 = 1.0 / (MU0 * 100.0)
    nu_2 = 1.0 / MU0
    nu_mixed = np.where(z_c < 0.0, nu_1, nu_2)
    k_mixed = stiffness(mesh, nu=nu_mixed)

    area, r_c, _, b, c = element_geometry(mesh)
    n_nodes = mesh.nodes.shape[0]
    k_hand = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for e_idx in range(m):
        elem = mesh.elements[e_idx]
        nu_e = nu_1 if z_c[e_idx] < 0.0 else nu_2
        for i in range(3):
            for j in range(3):
                k_e = (
                    nu_e
                    * area[e_idx]
                    * (
                        r_c[e_idx] * (b[e_idx, i] * b[e_idx, j] + c[e_idx, i] * c[e_idx, j])
                        + (b[e_idx, i] + b[e_idx, j]) / 3.0
                        + 1.0 / (9.0 * r_c[e_idx])
                    )
                )
                k_hand[elem[i], elem[j]] += k_e

    k_hand_csr = sp.csr_matrix(k_hand)
    diff_mixed = np.linalg.norm((k_mixed - k_hand_csr).toarray())
    norm_mixed = np.linalg.norm(k_hand_csr.toarray())
    assert diff_mixed / norm_mixed <= 1e-14

    with pytest.raises(ValueError, match="shape"):
        stiffness(mesh, nu=np.ones(m - 1))
    with pytest.raises(ValueError, match="positive"):
        stiffness(mesh, nu=np.zeros(m))
    bad_neg = np.ones(m)
    bad_neg[0] = -1.0
    with pytest.raises(ValueError, match="positive"):
        stiffness(mesh, nu=bad_neg)
    bad_nan = np.ones(m)
    bad_nan[0] = np.nan
    with pytest.raises(ValueError, match="positive"):
        stiffness(mesh, nu=bad_nan)
