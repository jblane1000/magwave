"""Closed-form magnetostatic fields — the Tier 0 analytic path.

Every expression here is a port of the research-verified implementation in
``docs/research/reference_formulas.py``; provenance, verification runs and the tolerance
each case may claim are recorded in ``docs/research/validation-sources.md``. Do not
re-derive the physics here, and do not "improve" a formula to make a benchmark pass.

Conventions (identical to the research file):

* SI throughout — metres, amperes, tesla, magnetisation ``M`` in A/m, remanence
  ``Br = mu0*M`` in tesla.
* Complete elliptic integrals use the **parameter** convention ``m = k**2``.
* Field points are returned in the coordinate system they are documented in; the
  cylindrical forms return ``(B_rho, B_z)``.

**No numerical quadrature is used anywhere in this module.** The elliptic integrals come
from the arithmetic-geometric mean, the solenoid and cuboid forms are closed expressions,
and the discretised reference integrators live only in the research file, where they are
used to verify these forms and never to compute a result.
"""

from __future__ import annotations

import math

import numpy as np

from .units import MU0

__all__ = [
    "cuboid_B",
    "cylinder_Bz_axis",
    "dipole_B",
    "ellipk_m",
    "ellipe_m",
    "F_dipole_coaxial",
    "F_pointpole_cylinders",
    "halbach_bore_ideal",
    "loop_B",
    "thick_solenoid_Bz",
    "thin_solenoid_Bz",
]

#: AGM converges quadratically; 30 iterations take double precision far past its limit
#: for every parameter a field point can produce (the worst case is m -> 1, i.e. a field
#: point on the conductor, which is rejected rather than evaluated).
_AGM_ITERATIONS = 30


def _ellip_agm(m: float) -> tuple[float, float]:
    """Return ``(K(m), E(m))`` by the arithmetic-geometric mean (Abramowitz & Stegun 17.6).

    ``m`` is the parameter ``k**2`` and must satisfy ``0 <= m < 1``.
    """
    a = 1.0
    b = math.sqrt(1.0 - m)
    # n = 0 term of sum(2**(n-1) * c_n**2): c_0 = sqrt(m), so the term is m/2.
    total = 0.5 * m
    weight = 1.0
    for _ in range(_AGM_ITERATIONS):
        c = 0.5 * (a - b)
        total += weight * c * c
        weight *= 2.0
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    k = math.pi / (2.0 * a)
    return k, k * (1.0 - total)


def ellipk_m(m: float) -> float:
    """Complete elliptic integral of the first kind, parameter convention ``m = k**2``.

    Input: dimensionless parameter, ``0 <= m < 1``. Output: dimensionless ``K(m)``.
    """
    if not 0.0 <= float(m) < 1.0:
        raise ValueError("ellipk_m: parameter m must satisfy 0 <= m < 1")
    return _ellip_agm(float(m))[0]


def ellipe_m(m: float) -> float:
    """Complete elliptic integral of the second kind, parameter convention ``m = k**2``.

    Input: dimensionless parameter, ``0 <= m < 1``. Output: dimensionless ``E(m)``.
    """
    if not 0.0 <= float(m) < 1.0:
        raise ValueError("ellipe_m: parameter m must satisfy 0 <= m < 1")
    return _ellip_agm(float(m))[1]


def dipole_B(moment, point) -> np.ndarray:
    """Field of a point dipole at the origin.

    Inputs: ``moment`` in A m^2, ``point`` in metres relative to the dipole. Output: B in
    tesla as a 3-vector. Exact for a uniformly magnetised sphere; far field otherwise, and
    the distributional term at ``r = 0`` is deliberately absent.
    """
    r = np.asarray(point, dtype=float)
    m = np.asarray(moment, dtype=float)
    rabs = float(np.sqrt(r @ r))
    if rabs < 1e-300:
        raise ValueError("dipole_B: field is undefined at the dipole position (r = 0)")
    return MU0 / (4.0 * np.pi) * (3.0 * float(m @ r) * r / rabs**5 - m / rabs**3)


def loop_B(a: float, rho: float, z: float, current: float = 1.0) -> tuple[float, float]:
    """Field of a circular current loop in the ``z = 0`` plane, centred on the origin.

    Inputs: ``a`` loop radius [m], ``rho >= 0`` cylindrical radius of the field point [m],
    ``z`` axial offset [m], ``current`` in amperes (total ampere-turns). Output:
    ``(B_rho, B_z)`` in tesla. Exact off axis in complete elliptic integrals; the point
    ``(rho = a, z = 0)`` lies on the conductor and is rejected.
    """
    if not a > 0.0:
        raise ValueError("loop_B: loop radius must be positive")
    if rho < 0.0:
        raise ValueError("loop_B: rho must be non-negative")
    q2 = (a + rho) ** 2 + z * z
    p2 = (a - rho) ** 2 + z * z
    if rho < 1e-13:
        return 0.0, MU0 * current * a * a / (2.0 * (a * a + z * z) ** 1.5)
    if p2 == 0.0:
        raise ValueError("loop_B: field point lies on the conductor (rho = a, z = 0)")
    m = 4.0 * a * rho / q2
    if m >= 1.0:
        raise ValueError("loop_B: elliptic parameter reached 1; field point is on the conductor")
    k = ellipk_m(m)
    e = ellipe_m(m)
    c = MU0 * current / (2.0 * np.pi * np.sqrt(q2))
    b_rho = c * (z / rho) * (-k + (a * a + rho * rho + z * z) / p2 * e)
    b_z = c * (k + (a * a - rho * rho - z * z) / p2 * e)
    return float(b_rho), float(b_z)


def thin_solenoid_Bz(R: float, z1: float, z2: float, nI: float, z: float) -> float:
    """On-axis field of a thin-walled (current-sheet) finite solenoid.

    Inputs: ``R`` radius [m], winding ends ``z1 < z2`` [m], ``nI`` ampere-turn density
    ``n*I`` [A], ``z`` field point on the axis [m]. Output: ``B_z`` in tesla.
    """
    if not z2 > z1:
        raise ValueError("thin_solenoid_Bz: require z2 > z1")
    c1 = (z - z1) / np.sqrt(R * R + (z - z1) ** 2)
    c2 = (z - z2) / np.sqrt(R * R + (z - z2) ** 2)
    return float(0.5 * MU0 * nI * (c1 - c2))


def thick_solenoid_Bz(R1: float, R2: float, z1: float, z2: float, J: float, z: float) -> float:
    """On-axis field of a solenoid with a rectangular winding cross-section.

    Inputs: inner and outer radii ``R1 < R2`` [m], winding ends ``z1 < z2`` [m], current
    density ``J`` [A/m^2], ``z`` field point on the axis [m]. Output: ``B_z`` in tesla.

    A volume winding is not a magnet model: the uniformly magnetised cylinder is the
    thin-wall limit and uses :func:`cylinder_Bz_axis`.
    """
    if not R2 > R1 >= 0.0:
        raise ValueError("thick_solenoid_Bz: require R2 > R1 >= 0")
    if not z2 > z1:
        raise ValueError("thick_solenoid_Bz: require z2 > z1")

    def bracket(s: float) -> float:
        # C(a, s) = s*asinh(a/|s|) is the antiderivative of s/sqrt(a^2 + s^2) in a.
        # Without |s| the s < 0 case carries the wrong sign — this is the trap the
        # research notes name; the ring-slice benchmark case pins it.
        if s == 0.0:
            return 0.0
        return s * (np.arcsinh(R2 / abs(s)) - np.arcsinh(R1 / abs(s)))

    return float(0.5 * MU0 * J * (bracket(z - z1) - bracket(z - z2)))


def cuboid_B(M: float, box: tuple[float, float, float, float, float, float], point) -> np.ndarray:
    """Field of a uniformly magnetised rectangular cuboid, magnetisation along ``+z``.

    Inputs: ``M`` magnetisation [A/m], ``box = (x1, x2, y1, y2, z1, z2)`` [m], ``point``
    [m]. Output: B in tesla as a 3-vector.

    The eight-corner magnetic-charge form is exact inside and outside the body; the
    interior value already contains the ``+mu0*M`` jump, so it must not be added again.
    A point exactly on a face plane is nudged off it by 1e-12 m (tangential B is
    continuous; do not sample on a face).
    """
    x1, x2, y1, y2, z1, z2 = box
    x, y, z = (float(v) for v in point)

    def gx(u: float, v: float, t: float) -> float:
        if t == 0.0:
            t = 1e-12
        return float(np.log((np.sqrt(u * u + v * v + t * t) - v) / abs(t)))

    def gy(u: float, v: float, t: float) -> float:
        if t == 0.0:
            t = 1e-12
        return float(np.log((np.sqrt(u * u + v * v + t * t) - u) / abs(t)))

    def gz(u: float, v: float, t: float) -> float:
        return float(np.arctan2(u * v, t * np.sqrt(u * u + v * v + t * t)))

    corners = ((x1, y1, 1.0), (x2, y1, -1.0), (x1, y2, -1.0), (x2, y2, 1.0))

    def patch(g, t: float) -> float:
        return sum(sign * g(x - xc, y - yc, t) for (xc, yc, sign) in corners)

    t_top, t_bottom = z - z2, z - z1
    fac = MU0 * M / (4.0 * np.pi)
    return np.array(
        [
            fac * (patch(gx, t_top) - patch(gx, t_bottom)),
            fac * (patch(gy, t_top) - patch(gy, t_bottom)),
            fac * (patch(gz, t_top) - patch(gz, t_bottom)),
        ]
    )


def cylinder_Bz_axis(R: float, z1: float, z2: float, M: float, z: float) -> float:
    """On-axis field of a uniformly magnetised cylinder, magnetisation along ``+z``.

    Inputs: ``R`` radius [m], end planes ``z1 < z2`` [m], ``M`` magnetisation [A/m], ``z``
    field point on the axis [m]. Output: ``B_z`` in tesla. Exact (two charged end discs).
    """
    if not R > 0.0:
        raise ValueError("cylinder_Bz_axis: radius must be positive")
    if not z2 > z1:
        raise ValueError("cylinder_Bz_axis: require z2 > z1")
    return float(
        0.5
        * MU0
        * M
        * ((z - z1) / np.sqrt((z - z1) ** 2 + R * R) - (z - z2) / np.sqrt((z - z2) ** 2 + R * R))
    )


def halbach_bore_ideal(Br: float, Ri: float, Ro: float) -> float:
    """Bore flux density of an infinite ``k = 1`` Halbach cylinder.

    Inputs: ``Br`` remanence [T], inner and outer radii ``Ri < Ro`` [m]. Output: the
    uniform bore flux density magnitude in tesla, ``Br*ln(Ro/Ri)``, directed along the
    fixed transverse axis (``x`` for the verified magnetisation pattern).

    The 2-D idealisation is exactly uniform in the bore; a real finite-length cylinder is
    lower at the ends, and the research dossier leaves that correction unverified. This
    function models the ideal only.
    """
    if not Ro > Ri > 0.0:
        raise ValueError("halbach_bore_ideal: require Ro > Ri > 0")
    return float(Br * np.log(Ro / Ri))


def F_dipole_coaxial(Br1: float, Br2: float, V1: float, V2: float, d: float) -> float:
    """Attraction between two coaxial dipoles aligned along the axis.

    Inputs: remanences ``Br1``, ``Br2`` [T], volumes ``V1``, ``V2`` [m^3], centre
    separation ``d`` [m]. Output: force magnitude in newtons.

    Far-zone only: it always *under*-estimates, and reaches 2 % of the exact charge model
    only beyond ``d >= 10 L`` for the ``L = 4R`` magnet pair benchmarked here.
    """
    if not d > 0.0:
        raise ValueError("F_dipole_coaxial: separation must be positive")
    m1 = (Br1 / MU0) * V1
    m2 = (Br2 / MU0) * V2
    return float(3.0 * MU0 * m1 * m2 / (2.0 * np.pi * d**4))


def F_pointpole_cylinders(Br: float, R: float, L: float, g: float) -> float:
    """Point-pole attraction between two identical coaxial cylindrical magnets.

    Inputs: ``Br`` remanence [T], ``R`` radius [m], ``L`` length [m], ``g`` gap between the
    near faces [m]. Output: force magnitude in newtons.

    Each magnet becomes two point charges ``M*pi*R^2`` at its face centres. The near faces
    attract (``g``), the far faces attract (``g + 2L``) and the two cross pairs repel
    (``g + L`` each). Within 2 % of the exact charge model for ``g >= 3L``, and it
    *over*-estimates.
    """
    if not g > 0.0:
        raise ValueError("F_pointpole_cylinders: gap must be positive")
    if not (R > 0.0 and L > 0.0):
        raise ValueError("F_pointpole_cylinders: R and L must be positive")
    c = np.pi * Br**2 * R**4 / (4.0 * MU0)
    return float(c * (1.0 / g**2 - 2.0 / (g + L) ** 2 + 1.0 / (g + 2.0 * L) ** 2))
