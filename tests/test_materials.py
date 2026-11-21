"""Material library contract tests (PRD FR-020 / FR-021)."""

from __future__ import annotations

import pytest

from magwave.materials import LIBRARY_VERSION, SCHEMA, MaterialsLibrary, load_library
from magwave.units import MU0


def test_library_loads_and_is_versioned() -> None:
    lib = load_library()
    assert isinstance(lib, MaterialsLibrary)
    assert LIBRARY_VERSION == lib.version == 1
    assert lib.schema == SCHEMA == f"magwave.materials/{LIBRARY_VERSION}"
    assert "steel-team20" in lib.names()

    mat = lib.get("steel-team20")
    assert mat.provenance
    assert "TEAM 20" in mat.provenance


def test_team20_table_is_verbatim_and_monotone() -> None:
    mat = load_library().get("steel-team20")
    assert mat.bh_table is not None
    assert len(mat.bh_table) == 38
    assert mat.bh_table[0] == (0.0, 0.0)
    assert mat.bh_table[1] == (27.0, 0.010)
    assert mat.bh_table[-1] == (135000.0, 2.30)

    h_vals = [pt[0] for pt in mat.bh_table]
    b_vals = [pt[1] for pt in mat.bh_table]
    assert all(h2 > h1 for h1, h2 in zip(h_vals[:-1], h_vals[1:], strict=True))
    assert all(b2 > b1 for b1, b2 in zip(b_vals[:-1], b_vals[1:], strict=True))
    assert mat.saturation_B == 2.16


def test_mu_r_is_the_first_segment_secant() -> None:
    mat = load_library().get("steel-team20")
    assert mat.bh_table is not None
    h1, b1 = mat.bh_table[1]
    assert (h1, b1) == (27.0, 0.010)
    secant = b1 / (MU0 * h1)
    assert abs(mat.mu_r - secant) / secant <= 1e-3
    assert mat.nu() == pytest.approx(1.0 / (MU0 * mat.mu_r), rel=1e-15)


def test_unknown_material_names_the_alternatives() -> None:
    lib = load_library()
    with pytest.raises(ValueError, match="steel-team20"):
        lib.get("nope")


def test_override_reports_both_values() -> None:
    lib = load_library()
    shipped = lib.get("steel-team20")

    overridden, prov_override = lib.with_override("steel-team20", 100.0)
    assert overridden.mu_r == 100.0
    assert "294" in prov_override
    assert "100" in prov_override

    restored, prov_shipped = lib.with_override("steel-team20", None)
    assert restored.mu_r == shipped.mu_r
    assert "294" in prov_shipped
    assert "overridden" not in prov_shipped.lower()
