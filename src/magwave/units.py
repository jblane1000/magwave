"""Unit helpers for MagWave's SI-internal data model.

MagWave stores every quantity in SI base units (metres, amperes, tesla) and converts only
at the boundaries: CAD import, the desktop UI and report output (PRD NFR-2). Every function
here is pure: it takes a plain number in the unit named in the docstring and returns a plain
number in the unit named in the return clause. No global state, no rounding, no unit objects.
"""

from __future__ import annotations

import math

#: Vacuum permeability, H/m (equivalently T m / A).
#:
#: ``4*pi*1e-7`` is exact under the pre-2019 SI definition of the ampere; the post-2019
#: CODATA value differs in the 10th significant digit, which is orders of magnitude below
#: the tolerance of every benchmark in this repository.
MU0: float = 4e-7 * math.pi

_MILLI = 1e-3
_INCH = 0.0254
_RAD_PER_DEG = math.pi / 180.0


def mm_to_m(value: float) -> float:
    """Convert millimetres to metres. Input: mm. Output: m."""
    return value * _MILLI


def m_to_mm(value: float) -> float:
    """Convert metres to millimetres. Input: m. Output: mm."""
    return value / _MILLI


def inch_to_m(value: float) -> float:
    """Convert inches to metres. Exact: 1 in = 25.4 mm by definition. Input: in. Output: m."""
    return value * _INCH


def deg_to_rad(value: float) -> float:
    """Convert degrees to radians. Input: degrees. Output: rad."""
    return value * _RAD_PER_DEG
