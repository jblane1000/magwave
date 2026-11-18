"""One evaluator per benchmark check.

Each ``check`` id names the analytic entry point a benchmark case exercises; the evaluator
reads the case's sources and probes and returns the single number the case's expected value
refers to. Everything is pure: a case in, a float out, no state and no hidden inputs. A
case that cannot be evaluated (missing source, wrong geometry for the check) raises
:class:`~magwave.cases.CaseError` rather than returning a plausible number — NFR-8.

The ids are also the ``check`` field of every expected value in
``src/magwave/benchmarks/cases.json``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypeVar

import numpy as np

from ..analytic import (
    F_dipole_coaxial,
    F_dipoles_general,
    F_pointpole_cylinders,
    cuboid_B,
    cylinder_B,
    cylinder_Bz_axis,
    dipole_B,
    halbach_bore_ideal,
    loop_B,
    thick_solenoid_Bz,
    thin_solenoid_Bz,
)
from ..cases import Case, CaseError, Coil, Dipole, Magnet, Probe
from ..units import MU0

CheckFn = Callable[[Case], float]
ComponentT = TypeVar("ComponentT", Magnet, Coil, Dipole)

#: Registry of named checks. Populated by the ``@check`` decorator below.
CHECKS: dict[str, CheckFn] = {}


def check(name: str) -> Callable[[CheckFn], CheckFn]:
    """Register an evaluator under ``name``; a duplicate name is a programming error."""

    def register(function: CheckFn) -> CheckFn:
        if name in CHECKS:
            raise RuntimeError(f"duplicate benchmark check {name!r}")
        CHECKS[name] = function
        return function

    return register


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CaseError(message)


def _sole(case: Case, kind: type[ComponentT], check_name: str) -> ComponentT:
    """Return the case's only component of ``kind``, or fail naming the check."""
    matches = [component for component in case.components if isinstance(component, kind)]
    _require(
        len(matches) == 1,
        f"{case.name}: check {check_name!r} needs exactly one {kind.__name__} component, "
        f"found {len(matches)}",
    )
    return matches[0]


def _probe(case: Case, quantity: str) -> Probe:
    matches = [probe for probe in case.probes if probe.quantity == quantity]
    _require(len(matches) == 1, f"{case.name}: expected one probe with quantity {quantity!r}")
    return matches[0]


def _pick(vector, component: str, case: Case) -> float:
    if component == "x":
        return float(vector[0])
    if component == "y":
        return float(vector[1])
    if component == "z":
        return float(vector[2])
    _require(component == "magnitude", f"{case.name}: unsupported component {component!r}")
    return float(np.linalg.norm(vector))


def _solenoid_Bz(case: Case, check_name: str, *, thick: bool) -> float:
    coil = _sole(case, Coil, check_name)
    _require(coil.kind == "solenoid", f"{case.name}: check {check_name!r} needs a solenoid coil")
    probe = _probe(case, "B")
    _require(
        probe.component in ("z", "magnitude"),
        f"{case.name}: the on-axis solenoid form yields B_z only, got component "
        f"{probe.component!r}",
    )
    inner = coil.dims["inner_radius"]
    outer = coil.dims["outer_radius"]
    length = coil.dims["length"]
    z1 = coil.center[2] - 0.5 * length
    z2 = coil.center[2] + 0.5 * length
    z = probe.point[2]
    ampere_turns = coil.turns * coil.current
    if thick:
        _require(
            outer > inner,
            f"{case.name}: thick solenoid needs outer_radius > inner_radius",
        )
        j = ampere_turns / ((outer - inner) * length)
        return thick_solenoid_Bz(inner, outer, z1, z2, j, z)
    _require(
        outer == inner,
        f"{case.name}: thin solenoid needs inner_radius == outer_radius "
        "(a volume winding is the thick form, not a magnet)",
    )
    return thin_solenoid_Bz(inner, z1, z2, ampere_turns / length, z)


@check("dipole_B")
def _dipole_B(case: Case) -> float:
    source = _sole(case, Dipole, "dipole_B")
    probe = _probe(case, "B")
    offset = np.subtract(probe.point, source.center)
    return _pick(dipole_B(source.moment, offset), probe.component, case)


@check("loop_B")
def _loop_B(case: Case) -> float:
    coil = _sole(case, Coil, "loop_B")
    _require(coil.kind == "loop", f"{case.name}: check 'loop_B' needs a loop coil")
    probe = _probe(case, "B")
    dx = probe.point[0] - coil.center[0]
    dy = probe.point[1] - coil.center[1]
    dz = probe.point[2] - coil.center[2]
    rho = math.hypot(dx, dy)
    b_rho, b_z = loop_B(coil.dims["radius"], rho, dz, coil.turns * coil.current)
    if rho == 0.0:
        bx = by = 0.0
    else:
        bx, by = b_rho * dx / rho, b_rho * dy / rho
    return _pick(np.array([bx, by, b_z]), probe.component, case)


@check("thin_solenoid_Bz")
def _thin_solenoid_Bz(case: Case) -> float:
    return _solenoid_Bz(case, "thin_solenoid_Bz", thick=False)


@check("thick_solenoid_Bz")
def _thick_solenoid_Bz(case: Case) -> float:
    return _solenoid_Bz(case, "thick_solenoid_Bz", thick=True)


def _cuboid_box(magnet: Magnet) -> tuple[float, float, float, float, float, float]:
    half = {key: 0.5 * value for key, value in magnet.dims.items()}
    return (
        magnet.center[0] - half["size_x"],
        magnet.center[0] + half["size_x"],
        magnet.center[1] - half["size_y"],
        magnet.center[1] + half["size_y"],
        magnet.center[2] - half["size_z"],
        magnet.center[2] + half["size_z"],
    )


@check("cuboid_B")
def _cuboid_B(case: Case) -> float:
    magnet = _sole(case, Magnet, "cuboid_B")
    _require(magnet.kind == "cuboid", f"{case.name}: check 'cuboid_B' needs a cuboid magnet")
    bre_x, bre_y, bre_z = magnet.br
    _require(
        bre_x == 0.0 and bre_y == 0.0,
        f"{case.name}: the eight-corner form here is magnetisation along +z only (M1); "
        "an off-axis remanence arrives with the M2 engine",
    )
    _require(bre_z != 0.0, f"{case.name}: br has no +z component")
    probe = _probe(case, "B")
    return _pick(cuboid_B(bre_z / MU0, _cuboid_box(magnet), probe.point), probe.component, case)


@check("cuboid_B_general")
def _cuboid_B_general(case: Case) -> float:
    magnet = _sole(case, Magnet, "cuboid_B_general")
    _require(
        magnet.kind == "cuboid", f"{case.name}: check 'cuboid_B_general' needs a cuboid magnet"
    )
    _require(any(c != 0.0 for c in magnet.br), f"{case.name}: br is zero")
    m_vec = tuple(c / MU0 for c in magnet.br)
    probe = _probe(case, "B")
    return _pick(cuboid_B(m_vec, _cuboid_box(magnet), probe.point), probe.component, case)


@check("cylinder_Bz_axis")
def _cylinder_Bz_axis(case: Case) -> float:
    magnet = _sole(case, Magnet, "cylinder_Bz_axis")
    _require(
        magnet.kind == "cylinder",
        f"{case.name}: check 'cylinder_Bz_axis' needs a solid cylinder magnet",
    )
    _require(
        magnet.br[0] == 0.0 and magnet.br[1] == 0.0 and magnet.br[2] != 0.0,
        f"{case.name}: on-axis cylinder form is magnetisation along +z only",
    )
    probe = _probe(case, "B")
    _require(
        probe.component in ("z", "magnitude"),
        f"{case.name}: the on-axis cylinder form yields B_z only",
    )
    length = magnet.dims["length"]
    return cylinder_Bz_axis(
        magnet.dims["radius"],
        magnet.center[2] - 0.5 * length,
        magnet.center[2] + 0.5 * length,
        magnet.br[2] / MU0,
        probe.point[2],
    )


@check("cylinder_B")
def _cylinder_B(case: Case) -> float:
    magnet = _sole(case, Magnet, "cylinder_B")
    _require(
        magnet.kind == "cylinder",
        f"{case.name}: check 'cylinder_B' needs a solid cylinder magnet",
    )
    _require(
        magnet.br[0] == 0.0 and magnet.br[1] == 0.0 and magnet.br[2] != 0.0,
        f"{case.name}: cylinder_B form is magnetisation along +z only",
    )
    probe = _probe(case, "B")
    dx = probe.point[0] - magnet.center[0]
    dy = probe.point[1] - magnet.center[1]
    rho = math.hypot(dx, dy)
    length = magnet.dims["length"]
    b_rho, b_z = cylinder_B(
        magnet.dims["radius"],
        magnet.center[2] - 0.5 * length,
        magnet.center[2] + 0.5 * length,
        magnet.br[2] / MU0,
        rho,
        probe.point[2],
    )
    if rho == 0.0:
        bx = by = 0.0
    else:
        bx, by = b_rho * dx / rho, b_rho * dy / rho
    return _pick(np.array([bx, by, b_z]), probe.component, case)


@check("halbach_bore_ideal")
def _halbach_bore_ideal(case: Case) -> float:
    magnet = _sole(case, Magnet, "halbach_bore_ideal")
    _require(
        magnet.kind == "halbach_cylinder",
        f"{case.name}: check 'halbach_bore_ideal' needs a halbach_cylinder magnet",
    )
    probe = _probe(case, "B")
    _require(
        probe.component == "magnitude",
        f"{case.name}: the ideal bore field is uniform and transverse; ask for 'magnitude'",
    )
    return halbach_bore_ideal(
        magnet.br[2], magnet.dims["inner_radius"], magnet.dims["outer_radius"]
    )


def _coaxial_cylinders(case: Case, check_name: str) -> tuple[Magnet, Magnet, float]:
    magnets = [component for component in case.components if isinstance(component, Magnet)]
    _require(
        len(magnets) == 2,
        f"{case.name}: check {check_name!r} needs exactly two magnets, found {len(magnets)}",
    )
    for magnet in magnets:
        _require(
            magnet.kind == "cylinder",
            f"{case.name}: check {check_name!r} is defined for cylindrical magnets",
        )
    first, second = magnets
    separation = math.dist(first.center, second.center)
    _require(separation > 0.0, f"{case.name}: the two magnets share a centre")
    return first, second, separation


@check("F_dipole_coaxial")
def _F_dipole_coaxial(case: Case) -> float:
    first, second, separation = _coaxial_cylinders(case, "F_dipole_coaxial")
    probe = _probe(case, "F")
    _require(
        probe.component in ("z", "magnitude"),
        f"{case.name}: the coaxial dipole force is axial; ask for 'z' or 'magnitude'",
    )
    volumes = [
        math.pi * magnet.dims["radius"] ** 2 * magnet.dims["length"] for magnet in (first, second)
    ]
    return F_dipole_coaxial(abs(first.br[2]), abs(second.br[2]), volumes[0], volumes[1], separation)


@check("F_pointpole_cylinders")
def _F_pointpole_cylinders(case: Case) -> float:
    first, second, separation = _coaxial_cylinders(case, "F_pointpole_cylinders")
    probe = _probe(case, "F")
    _require(
        probe.component in ("z", "magnitude"),
        f"{case.name}: the coaxial point-pole force is axial; ask for 'z' or 'magnitude'",
    )
    radius = first.dims["radius"]
    length = first.dims["length"]
    _require(
        radius == second.dims["radius"] and length == second.dims["length"],
        f"{case.name}: the point-pole form is for two identical magnets",
    )
    gap = separation - length
    _require(gap > 0.0, f"{case.name}: the magnets overlap; the gap must be positive")
    return F_pointpole_cylinders(abs(first.br[2]), radius, length, gap)


@check("F_dipoles_general")
def _F_dipoles_general(case: Case) -> float:
    dipoles = [component for component in case.components if isinstance(component, Dipole)]
    _require(
        len(dipoles) == 2,
        f"{case.name}: check 'F_dipoles_general' needs exactly two Dipole components, "
        f"found {len(dipoles)}",
    )
    first, second = dipoles
    _require(
        math.dist(first.center, second.center) > 0.0,
        f"{case.name}: the two dipoles share a centre",
    )
    probe = _probe(case, "F")
    _require(
        probe.point == second.center,
        f"{case.name}: force probe must sit at the second dipole centre {second.center}",
    )
    force = F_dipoles_general(first.moment, second.moment, np.subtract(second.center, first.center))
    return _pick(force, probe.component, case)
