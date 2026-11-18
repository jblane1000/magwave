"""Unit helpers (NFR-2): SI internally, mm/in at the boundaries, exact round-trips."""

from __future__ import annotations

import math

import pytest

from magwave.units import MU0, deg_to_rad, inch_to_m, m_to_mm, mm_to_m

SAMPLES = (0.0, 1.0, 25.4, 1e-06, 1234.5678)


@pytest.mark.parametrize("value", SAMPLES)
def test_mm_roundtrip(value: float) -> None:
    assert m_to_mm(mm_to_m(value)) == pytest.approx(value, rel=1e-15)


@pytest.mark.parametrize("value", SAMPLES)
def test_inch_to_metre_is_exact(value: float) -> None:
    assert inch_to_m(value) == value * 0.0254


def test_known_lengths() -> None:
    assert mm_to_m(1000.0) == 1.0
    assert inch_to_m(1.0) == 0.0254
    assert inch_to_m(0.75) == pytest.approx(0.01905, rel=1e-15)
    assert mm_to_m(25.4) == inch_to_m(1.0)


def test_degrees_to_radians() -> None:
    assert deg_to_rad(0.0) == 0.0
    assert deg_to_rad(90.0) == pytest.approx(math.pi / 2.0, rel=1e-15)
    assert deg_to_rad(180.0) == pytest.approx(math.pi, rel=1e-15)
    assert deg_to_rad(360.0) == pytest.approx(2.0 * math.pi, rel=1e-15)


def test_mu0_matches_the_research_reference() -> None:
    """MU0 must be the same constant the reference formulas were verified with."""
    assert 4e-7 * math.pi == MU0
    assert pytest.approx(1.2566370614359173e-06, rel=1e-15) == MU0
