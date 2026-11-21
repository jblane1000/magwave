"""Permanent-magnet second-quadrant library and Case.temperature_c tests (M5.3 FR-020)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from magwave.cases import Case, CaseError, Dipole, ExpectedValue, Probe
from magwave.materials import (
    LIBRARY_VERSION,
    MAGNET_LIBRARY_VERSION,
    MAGNET_SCHEMA,
    DemagCurve,
    MagnetGrade,
    MagnetLibrary,
    demag_curve_for,
    load_library,
    load_magnet_library,
)


def test_magnet_library_loads_and_preserves_soft_library_version() -> None:
    soft_lib = load_library()
    assert LIBRARY_VERSION == soft_lib.version == 1

    mag_lib = load_magnet_library()
    assert isinstance(mag_lib, MagnetLibrary)
    assert MAGNET_LIBRARY_VERSION == mag_lib.version == 1
    assert mag_lib.schema == MAGNET_SCHEMA == f"magwave.magnets/{MAGNET_LIBRARY_VERSION}"
    assert mag_lib.names() == ("N42-20C", "Y30H-1-20C")

    n42 = mag_lib.get("N42-20C")
    ferrite = mag_lib.get("Y30H-1-20C")
    assert isinstance(n42, MagnetGrade)
    assert isinstance(ferrite, MagnetGrade)
    assert n42.family == "NdFeB"
    assert "Arnold N42" in n42.provenance
    assert "RECONSTRUCTED" in n42.provenance
    assert ferrite.family == "Ferrite"
    assert "Eclipse" in ferrite.provenance
    assert ferrite.alpha_hcj_per_c > 0.0


def test_unknown_magnet_grade_names_shipped_alternatives() -> None:
    lib = load_magnet_library()
    with pytest.raises(ValueError) as excinfo:
        lib.get("missing-grade")
    msg = str(excinfo.value)
    assert "missing-grade" in msg
    assert "N42-20C" in msg
    assert "Y30H-1-20C" in msg


def test_magnet_library_strict_schema_checks() -> None:
    base = {
        "schema": "magwave.magnets/1",
        "version": 1,
        "magnets": [
            {
                "name": "test-grade",
                "grade": "TG",
                "family": "NdFeB",
                "curve": [[-300.0, -0.3], [-200.0, 0.0], [-100.0, 0.2], [0.0, 0.4]],
                "br_ref": 0.4,
                "hcb": 200.0,
                "hcj": 300.0,
                "bhmax": 20.0,
                "t_ref_c": 20.0,
                "alpha_br_per_c": -0.001,
                "alpha_hcj_per_c": -0.005,
                "t_max_c": 80.0,
                "provenance": "unit test",
            }
        ],
    }
    assert MagnetLibrary.from_dict(base).names() == ("test-grade",)

    extra_top = dict(base, extra="nope")
    with pytest.raises(ValueError, match="unknown key"):
        MagnetLibrary.from_dict(extra_top)

    bad_entry = json.loads(json.dumps(base))
    bad_entry["magnets"][0]["extra_field"] = 1.0
    with pytest.raises(ValueError, match="unknown key"):
        MagnetLibrary.from_dict(bad_entry)

    missing_entry = json.loads(json.dumps(base))
    del missing_entry["magnets"][0]["provenance"]
    with pytest.raises(ValueError, match="missing required key"):
        MagnetLibrary.from_dict(missing_entry)


def test_non_monotone_and_malformed_demag_curves_are_rejected() -> None:
    base_curve = [
        [-955000.0, -1.2002],
        [-860000.0, 0.0],
        [-500000.0, 0.6677],
        [-200000.0, 1.0607],
        [0.0, 1.315],
    ]

    # Non-monotone H (points 1 and 2)
    bad_h = [list(pt) for pt in base_curve]
    bad_h[2][0] = -860000.0
    with pytest.raises(ValueError) as exc_h:
        DemagCurve(bad_h, name="bad-h-grade")
    msg_h = str(exc_h.value)
    assert "bad-h-grade" in msg_h
    assert "1" in msg_h
    assert "2" in msg_h

    # Non-monotone B (unsorted/falling at points 2 and 3)
    bad_b = [list(pt) for pt in base_curve]
    bad_b[3][1] = 0.5
    with pytest.raises(ValueError) as exc_b:
        DemagCurve(bad_b, name="bad-b-grade")
    msg_b = str(exc_b.value)
    assert "bad-b-grade" in msg_b
    assert "2" in msg_b
    assert "3" in msg_b

    # Reject through MagnetLibrary loader as well (never sort silently)
    lib_payload = {
        "schema": "magwave.magnets/1",
        "version": 1,
        "magnets": [
            {
                "name": "loader-nonmonotone",
                "grade": "X",
                "family": "NdFeB",
                "curve": bad_b,
                "br_ref": 1.315,
                "hcb": 860000.0,
                "hcj": 955000.0,
                "bhmax": 334000.0,
                "t_ref_c": 20.0,
                "alpha_br_per_c": -0.0012,
                "alpha_hcj_per_c": -0.00618,
                "t_max_c": 80.0,
                "provenance": "unit test non-monotone",
            }
        ],
    }
    with pytest.raises(ValueError) as exc_lib:
        MagnetLibrary.from_dict(lib_payload)
    assert "loader-nonmonotone" in str(exc_lib.value)
    assert "2" in str(exc_lib.value)
    assert "3" in str(exc_lib.value)

    # Fewer than 4 points
    with pytest.raises(ValueError, match="at least 4 points"):
        DemagCurve(base_curve[:3], name="too-short")

    # Last H != 0
    bad_end = [list(pt) for pt in base_curve]
    bad_end[-1][0] = -10.0
    with pytest.raises(ValueError, match="H=0"):
        DemagCurve(bad_end, name="nonzero-end")


def test_demag_interpolation_derivatives_and_no_extrapolation() -> None:
    curve = demag_curve_for("N42-20C", 20.0)
    assert demag_curve_for("N42-20C", 20.0) is curve

    h_tab = np.array([pt[0] for pt in curve.curve], dtype=np.float64)
    b_tab = np.array([pt[1] for pt in curve.curve], dtype=np.float64)

    np.testing.assert_allclose(curve.b_of_h(h_tab), b_tab, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(curve.h_of_b(b_tab), h_tab, rtol=1e-13, atol=1e-9)

    b_grid = np.linspace(curve.b_min + 1e-4, curve.b_max - 1e-4, 250)
    s, ds_db, c, dc_db = curve.slope_and_offset(b_grid)
    assert np.all(s > 0.0)
    assert np.all(c < 0.0)
    np.testing.assert_allclose(s * b_grid + c, curve.h_of_b(b_grid), rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(dc_db, -b_grid * ds_db, rtol=1e-12, atol=1e-6)

    # Verify ds_db against central finite differences on smooth interior points
    b_probe = np.array([0.45, 0.75, 1.05])
    eps = 1e-6
    _, ds_analytic, _, dc_analytic = curve.slope_and_offset(b_probe)
    s_plus, _, c_plus, _ = curve.slope_and_offset(b_probe + eps)
    s_minus, _, c_minus, _ = curve.slope_and_offset(b_probe - eps)
    np.testing.assert_allclose(ds_analytic, (s_plus - s_minus) / (2.0 * eps), rtol=1e-5)
    np.testing.assert_allclose(dc_analytic, (c_plus - c_minus) / (2.0 * eps), rtol=1e-5)

    # No extrapolation beyond the tabulated domain
    with pytest.raises(ValueError, match="extrapolation"):
        curve.b_of_h(curve.h_min - 1.0)
    with pytest.raises(ValueError, match="extrapolation"):
        curve.b_of_h(0.1)
    with pytest.raises(ValueError, match="extrapolation"):
        curve.h_of_b(curve.b_min - 1e-3)
    with pytest.raises(ValueError, match="extrapolation"):
        curve.slope_and_offset(curve.b_max + 1e-3)


def test_temperature_scaling_for_ndfeb_and_ferrite() -> None:
    n42_100 = demag_curve_for("N42-20C", 100.0)
    assert n42_100.br() == pytest.approx(1.315 * (1.0 - 0.0012 * 80.0), rel=1e-12)
    assert abs(n42_100.hcb()) == pytest.approx(860000.0 * (1.0 - 0.00618 * 80.0), rel=1e-12)
    assert n42_100.hcj == pytest.approx(955000.0 * (1.0 - 0.00618 * 80.0), rel=1e-12)

    # Idempotency / non-compounding when re-scaled
    n42_back = n42_100.at_temperature(20.0)
    assert n42_back.br() == pytest.approx(1.315, rel=1e-14)
    assert abs(n42_back.hcb()) == pytest.approx(860000.0, rel=1e-14)

    ferrite_20 = demag_curve_for("Y30H-1-20C", 20.0)
    ferrite_100 = demag_curve_for("Y30H-1-20C", 100.0)
    assert ferrite_100.br() < ferrite_20.br()
    assert ferrite_100.hcj > ferrite_20.hcj
    assert abs(ferrite_100.hcb()) > abs(ferrite_20.hcb())


def test_case_temperature_c_schema_roundtrip_and_validation() -> None:
    base_kwargs = {
        "name": "temp_roundtrip",
        "description": "temperature field contract",
        "sources": ("unit test",),
        "components": (Dipole(moment=(0.0, 0.0, 1.0), name="d1"),),
        "probes": (Probe(point=(0.0, 0.0, 0.5), quantity="B", component="z"),),
        "expected": (
            ExpectedValue(
                check="dipole_B",
                quantity="Bz",
                value=1.6e-6,
                unit="T",
                tolerance=1e-9,
                source="unit test",
            ),
        ),
    }

    # None and 20.0 must omit temperature_c from JSON
    c_none = Case(**base_kwargs)
    c_20 = Case(**base_kwargs, temperature_c=20.0)
    assert "temperature_c" not in c_none.to_dict()
    assert "temperature_c" not in c_20.to_dict()
    assert c_none.to_json() == c_20.to_json()

    # Non-20.0 values (including 0.0 and 100.0) round-trip losslessly
    for temp in (0.0, -40.0, 100.0):
        c_t = Case(**base_kwargs, temperature_c=temp)
        raw = c_t.to_json()
        assert json.loads(raw)["temperature_c"] == temp
        restored = Case.from_json(raw)
        assert restored == c_t
        assert restored.to_json() == raw

    for bad_temp in (-274.0, float("nan"), float("inf"), True, "100"):
        with pytest.raises(CaseError, match="temperature_c"):
            Case(**base_kwargs, temperature_c=bad_temp)  # type: ignore[arg-type]
