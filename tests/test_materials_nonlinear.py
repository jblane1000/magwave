"""Nonlinear B–H material curve tests (M5.2 Stage B)."""

from __future__ import annotations

import numpy as np
import pytest

from magwave.materials import BhCurve, MaterialsLibrary, load_library
from magwave.units import MU0


def test_table_points_are_inverted_exactly() -> None:
    mat = load_library().get("steel-team20")
    assert mat.bh_table is not None
    curve = mat.bh_curve()
    assert isinstance(curve, BhCurve)
    assert mat.bh_curve() is curve  # built once and cached per Material

    h_tab = np.array([pt[0] for pt in mat.bh_table], dtype=np.float64)
    b_tab = np.array([pt[1] for pt in mat.bh_table], dtype=np.float64)
    h_eval = curve.field_strength(b_tab)

    assert h_eval.shape == b_tab.shape
    assert h_eval[0] == 0.0
    np.testing.assert_allclose(h_eval[1:], h_tab[1:], rtol=1e-9, atol=0.0)


def test_monotone_no_overshoot() -> None:
    mat = load_library().get("steel-team20")
    assert mat.bh_table is not None
    curve = mat.bh_curve()

    for idx in range(len(mat.bh_table) - 1):
        h_lo, b_lo = mat.bh_table[idx]
        h_hi, b_hi = mat.bh_table[idx + 1]
        b_interior = np.linspace(b_lo, b_hi, 33)[1:-1]
        h_interior = curve.field_strength(b_interior)
        assert np.all(h_interior > h_lo)
        assert np.all(h_interior < h_hi)
        assert np.all(np.diff(h_interior) > 0.0)

    b_sweep = np.linspace(0.0, 6.0, 2001)
    h_sweep = curve.field_strength(b_sweep)
    assert np.all(np.diff(h_sweep) > 0.0)


def test_above_last_point_slope_tends_to_mu0() -> None:
    mat = load_library().get("steel-team20")
    curve = mat.bh_curve()

    pts = np.array([2.4, 2.6, 3.0], dtype=np.float64)
    mu_diff = curve.differential_permeability(pts)
    np.testing.assert_allclose(mu_diff, MU0, rtol=1e-9, atol=0.0)

    expected_h_24 = (2.4 - 2.16) / MU0
    assert float(curve.field_strength(2.4)) == pytest.approx(expected_h_24, rel=1e-12, abs=0.0)


def test_below_first_point_is_linear_from_origin() -> None:
    mat = load_library().get("steel-team20")
    curve = mat.bh_curve()

    b_low = 0.005
    expected_h = b_low / (MU0 * mat.mu_r)
    assert float(curve.field_strength(b_low)) == pytest.approx(expected_h, rel=1e-14, abs=0.0)
    assert float(curve.field_strength(0.0)) == 0.0
    assert float(curve.reluctivity(0.0)) == pytest.approx(1.0 / (MU0 * mat.mu_r), rel=1e-14)
    nu_zero, dnu_zero = curve.reluctivity_and_derivative(0.0)
    assert float(nu_zero) == pytest.approx(1.0 / (MU0 * mat.mu_r), rel=1e-14)
    assert float(dnu_zero) == 0.0


def test_nonmonotone_table_is_rejected() -> None:
    mat = load_library().get("steel-team20")
    assert mat.bh_table is not None

    # Repeat H at points 12 and 13 ("1720 -> 1720" in a synthetic test table).
    broken_table = [[float(h), float(b)] for h, b in mat.bh_table]
    broken_table[12][0] = 1720.0
    broken_table[13][0] = 1720.0

    with pytest.raises(ValueError) as excinfo:
        BhCurve(broken_table, mu_r=mat.mu_r, saturation_B=mat.saturation_B, name="broken-steel")
    msg = str(excinfo.value)
    assert "broken-steel" in msg
    assert "12" in msg
    assert "13" in msg
    assert "1720" in msg

    # Same check through the MaterialsLibrary loader using a test-only library dict.
    bad_lib = {
        "schema": "magwave.materials/1",
        "version": 1,
        "materials": [
            {
                "name": "test-nonmonotone-steel",
                "mu_r": 250.0,
                "bh_table": broken_table,
                "saturation_B": 2.1,
                "provenance": "unit test non-monotone B-H fixture",
            }
        ],
    }
    with pytest.raises(ValueError) as lib_exc:
        MaterialsLibrary.from_dict(bad_lib)
    lib_msg = str(lib_exc.value)
    assert "test-nonmonotone-steel" in lib_msg
    assert "12" in lib_msg
    assert "13" in lib_msg

    # Non-monotone B (unsorted/falling) must also be rejected, never silently sorted.
    falling_b = [[0.0, 0.0], [100.0, 0.5], [200.0, 0.4], [500.0, 1.0]]
    with pytest.raises(ValueError, match="falling-b-steel") as fall_exc:
        BhCurve(falling_b, mu_r=100.0, saturation_B=2.0, name="falling-b-steel")
    assert "1" in str(fall_exc.value)
    assert "2" in str(fall_exc.value)


def test_no_saturation_value_is_a_hard_error() -> None:
    table = ((0.0, 0.0), (100.0, 0.5), (500.0, 1.2), (2000.0, 1.6))
    curve = BhCurve(table, mu_r=1000.0, saturation_B=None, name="no-sat-steel")

    assert np.isfinite(curve.field_strength(1.6))
    with pytest.raises(ValueError) as excinfo:
        curve.field_strength(1.8)
    msg = str(excinfo.value)
    assert "no-sat-steel" in msg
    assert "1.8" in msg


def test_reluctivity_and_differential_permeability_stay_positive() -> None:
    curve = load_library().get("steel-team20").bh_curve()
    b_dense = np.linspace(0.0, 6.0, 10001)

    with np.errstate(all="raise"):
        nu_vals, dnu_vals = curve.reluctivity_and_derivative(b_dense)
        nu_direct = curve.reluctivity(b_dense)
        mu_diff = curve.differential_permeability(b_dense)

    assert np.all(np.isfinite(nu_vals))
    assert np.all(np.isfinite(dnu_vals))
    assert np.all(np.isfinite(mu_diff))
    assert float(np.min(nu_vals)) > 0.0
    assert float(np.min(mu_diff)) > 0.0
    np.testing.assert_allclose(nu_vals, nu_direct, rtol=1e-14, atol=0.0)
