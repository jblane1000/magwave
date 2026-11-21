"""Tier 0 engine: closed-form source fields from Magpylib (rung ``second-code``).

Magpylib is an independent, published implementation of the analytic field of permanent
magnets and current loops. Running the benchmark corpus through it cross-checks
:mod:`magwave.analytic` against a second code; it also gives the CLI a vectorised field
sampler (FR-050). It models free-space sources only: no permeable material, no
demagnetisation, no force.
"""

from __future__ import annotations

import magpylib as magpy
import numpy as np

from ..cases import Case, Coil, Dipole, Magnet, Permeable, UniformField
from ..units import MU0
from .base import Capabilities, Problem, Result, Sample, raise_on_errors

ENGINE_ID = "magpylib"
RUNG = "second-code"

E_UNSUPPORTED_SOURCE_TYPE = "E_UNSUPPORTED_SOURCE_TYPE"
E_UNSUPPORTED_GEOMETRY = "E_UNSUPPORTED_GEOMETRY"
E_UNSUPPORTED_COIL = "E_UNSUPPORTED_COIL"
E_UNSUPPORTED_QUANTITY = "E_UNSUPPORTED_QUANTITY"
E_UNSUPPORTED_PERMEABLE = "E_UNSUPPORTED_PERMEABLE"
E_UNSUPPORTED_DEMAG_CURVE = "E_UNSUPPORTED_DEMAG_CURVE"


def _label(component: Magnet | Coil | Dipole | Permeable | UniformField, index: int) -> str:
    name = component.name or f"#{index}"
    return f"component {name}"


class MagpylibEngine:
    """Samples the magnetic flux density B of a case's sources with Magpylib.

    Magnet convention
    -----------------
    A case stores ``Magnet.br`` as the polarisation ``J = mu0*M`` in tesla, so the
    magnetisation ``M = br/MU0`` in A/m is *derived, never required*. The case vector is
    passed straight to Magpylib's ``polarization=`` argument, which also takes J in tesla.
    Magnet data is never routed through a ``magnetization=`` argument: Magpylib multiplies
    magnetisation by its own ``mu_0``, which would reintroduce a 1.3e-10 relative error
    and break the 1e-12 cylinder case. Magnet fields therefore involve no ``mu_0`` at all
    and take no correction.

    Source mapping (Magpylib 6.x)::

        Dipole(moment, center)            -> magpy.misc.Dipole(moment, position)
        Magnet cuboid {size_x,y,z}, br    -> magpy.magnet.Cuboid(polarization=br,
                                                 dimension=(size_x, size_y, size_z))
        Magnet cylinder {radius,length}   -> magpy.magnet.Cylinder(polarization=br,
                                                 dimension=(2*radius, length))
        Coil loop {radius}, current, N    -> magpy.current.Circle(current=N*current,
                                                 diameter=2*radius)
        Magnet halbach_cylinder           -> not modelled: validate() error
        Coil solenoid                     -> not modelled: validate() error

    ``mu_0`` constant reconciliation
    --------------------------------
    Magpylib 6 uses the CODATA value ``magpy.mu_0 = 1.25663706127e-06`` H/m; this
    repository fixes ``MU0 = 4e-7*pi`` (:mod:`magwave.units`) and its research-owned
    tolerances were written against that. The two differ by 1.32e-10 relative, which on
    its own exceeds the 1e-10 tolerance of the loop cases. After sampling, the field
    contributed by sources whose field is ``mu_0`` times an ampere-based quantity — the
    current loops (A) **and the point dipoles** (moment in A m^2, ``B = mu_0/(4 pi) ...``)
    — is multiplied by ``MU0 / magpy.mu_0``, read from Magpylib at runtime. This is a
    constant-convention reconciliation, not a fit: it is the same exact ratio for every
    case and every point, and it is not applied to magnets (evaluated from polarisation
    in tesla, no ``mu_0`` involved). Measured before/after errors are in docs/M2-NOTES.md.

    Sampling: :meth:`sample` builds one Magpylib ``Collection`` per case (two
    sub-collections: the ``mu_0``-scaled sources and the polarisation sources) and makes
    one vectorised ``getB`` call for the whole point set — no Python loop over points.
    """

    def __init__(self) -> None:
        self._version = str(magpy.__version__)

    # ----------------------------------------------------------------------------------
    # protocol
    # ----------------------------------------------------------------------------------
    def capabilities(self) -> Capabilities:
        return Capabilities(
            engine_id=ENGINE_ID,
            rung=RUNG,
            source_types=frozenset({"magnet", "coil", "dipole", "uniform_field"}),
            magnet_kinds=frozenset({"cuboid", "cylinder"}),
            coil_kinds=frozenset({"loop"}),
            quantities=frozenset({"B"}),
            force_methods=(),
            engine_version=self._version,
        )

    @property
    def mu0_ratio(self) -> float:
        """``MU0 / magpy.mu_0`` — the repository-to-Magpylib vacuum-permeability ratio."""
        return MU0 / float(magpy.mu_0)

    def validate(self, case: Case) -> list[Problem]:
        """Return every reason this engine cannot model ``case`` (``[]`` if it can)."""
        problems: list[Problem] = []
        for index, component in enumerate(case.components):
            label = _label(component, index)
            if isinstance(component, Dipole | UniformField | Permeable):
                continue
            if isinstance(component, Magnet):
                if component.kind == "halbach_cylinder":
                    problems.append(
                        Problem(
                            E_UNSUPPORTED_GEOMETRY,
                            f"{case.name}: {label} is an ideal infinite Halbach cylinder; "
                            "magpylib has no exact source for it",
                            "error",
                        )
                    )
                elif component.kind not in self.capabilities().magnet_kinds:
                    problems.append(
                        Problem(
                            E_UNSUPPORTED_GEOMETRY,
                            f"{case.name}: {label} is a {component.kind!r} magnet, which this "
                            "engine does not model",
                            "error",
                        )
                    )
                if component.demag_curve is not None:
                    problems.append(
                        Problem(
                            E_UNSUPPORTED_DEMAG_CURVE,
                            f"{case.name}: {label} names demag curve {component.demag_curve!r}; "
                            "Tier 0 (magpylib) would treat it as rigidly polarised, no "
                            "demagnetising field, no operating point, and cannot locate an "
                            "operating point on a second-quadrant curve; use Tier 1 (fem2d) "
                            "— nothing was solved.",
                            "error",
                        )
                    )
                continue
            if isinstance(component, Coil):
                if component.kind == "solenoid":
                    sheet = component.dims["inner_radius"] == component.dims["outer_radius"]
                    what = "a thin current sheet" if sheet else "a volume winding"
                    problems.append(
                        Problem(
                            E_UNSUPPORTED_COIL,
                            f"{case.name}: {label} is a solenoid ({what}); magpylib has no "
                            "current-sheet or volume-current source, and a stack of loops "
                            "would be a discretised model of the analytic formula, not an "
                            "independent check",
                            "error",
                        )
                    )
                elif component.kind not in self.capabilities().coil_kinds:
                    problems.append(
                        Problem(
                            E_UNSUPPORTED_COIL,
                            f"{case.name}: {label} is a {component.kind!r} coil, which this "
                            "engine does not model",
                            "error",
                        )
                    )
                continue
            problems.append(  # pragma: no cover - Case already rejects unknown components
                Problem(
                    E_UNSUPPORTED_SOURCE_TYPE,
                    f"{case.name}: {label} has unsupported type {type(component).__name__}",
                    "error",
                )
            )
        has_permeable_component = any(isinstance(c, Permeable) for c in case.components)
        has_permeable_manifest = case.manifest is not None and any(
            body.role == "permeable_solid" for body in case.manifest.bodies
        )
        if has_permeable_component or has_permeable_manifest:
            problems.append(
                Problem(
                    E_UNSUPPORTED_PERMEABLE,
                    f"{case.name}: case requires permeable_solid / soft-magnetic material "
                    "response, which Tier 0 (magpylib, free-space only) cannot model; use "
                    "Tier 1 stage 2 (M5.2), engine 'fem2d' — nothing was solved.",
                    "error",
                )
            )
        for index, probe in enumerate(case.probes):
            if probe.quantity == "F":
                problems.append(
                    Problem(
                        E_UNSUPPORTED_QUANTITY,
                        f"{case.name}: probe #{index} asks for force F; this engine samples B "
                        "only (magpylib core has no closed-form force, only mesh-integrated "
                        "getFT, and Tier 0 force is the FR-052 analytic path)",
                        "error",
                    )
                )
            elif probe.quantity not in self.capabilities().quantities:
                problems.append(  # pragma: no cover - Probe already rejects other quantities
                    Problem(
                        E_UNSUPPORTED_QUANTITY,
                        f"{case.name}: probe #{index} asks for {probe.quantity!r}; this engine "
                        "samples B only",
                        "error",
                    )
                )
        return problems

    def solve(self, case: Case) -> Result:
        """Validate, then sample B at every probe point; raise ``EngineError`` if unsolvable."""
        warnings = raise_on_errors(ENGINE_ID, self.validate(case))
        points = np.array([probe.point for probe in case.probes], dtype=float)
        field = self._sample(case, points)
        samples = tuple(
            Sample(
                point=probe.point,
                quantity="B",
                vector=(float(row[0]), float(row[1]), float(row[2])),
                unit="T",
                component=probe.component,
            )
            for probe, row in zip(case.probes, field, strict=True)
        )
        return Result(
            engine_id=ENGINE_ID,
            engine_version=self._version,
            rung=RUNG,
            method=self.method,
            samples=samples,
            problems=tuple(warnings),
        )

    # ----------------------------------------------------------------------------------
    # sampling
    # ----------------------------------------------------------------------------------
    @property
    def method(self) -> str:
        return (
            f"magpylib {self._version} closed-form free-space source fields "
            "(Cuboid/Cylinder from polarisation J in tesla, Circle current loop, point "
            "Dipole); loop and dipole contributions rescaled by MU0/magpy.mu_0"
        )

    def sample(self, case: Case, points) -> np.ndarray:
        """B in tesla at ``points`` (metres, shape ``(n, 3)`` or ``(3,)``) -> ``(n, 3)``.

        Raises :class:`~magwave.engines.base.EngineError` if the case has a component this
        engine cannot model, and ``ValueError`` for malformed points.
        """
        raise_on_errors(ENGINE_ID, self.validate(case))
        array = np.asarray(points, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"points must have shape (n, 3), got {np.shape(points)}")
        if not np.all(np.isfinite(array)):
            raise ValueError("points must be finite (metres)")
        return self._sample(case, array)

    def _sample(self, case: Case, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((0, 3), dtype=float)
        scaled, polarised = self._build(case)
        groups = [group for group in (scaled, polarised) if group.children]
        field = np.zeros((len(points), 3), dtype=float)
        if groups:
            # One Collection per case, one vectorised getB call; sumup is done here so the
            # mu_0 reconciliation can be applied to the scaled group alone.
            collection = magpy.Collection(*groups)
            per_group = magpy.getB(list(collection.children), points, squeeze=False)
            per_group = np.asarray(per_group, dtype=float).reshape(len(groups), len(points), 3)
            for group, contribution in zip(groups, per_group, strict=True):
                field += contribution * self.mu0_ratio if group is scaled else contribution
        for component in case.components:
            if isinstance(component, UniformField):
                field[:, 2] += MU0 * component.h0
        return field

    def _build(self, case: Case) -> tuple[magpy.Collection, magpy.Collection]:
        """Map case components to Magpylib sources: (mu_0-scaled group, polarisation group)."""
        scaled = magpy.Collection(style_label="mu0-scaled (loops, dipoles)")
        polarised = magpy.Collection(style_label="polarisation (magnets)")
        for component in case.components:
            if isinstance(component, Magnet):
                polarised.add(self._magnet(component))
            elif isinstance(component, Dipole):
                scaled.add(magpy.misc.Dipole(moment=component.moment, position=component.center))
            elif isinstance(component, Coil):
                scaled.add(
                    magpy.current.Circle(
                        current=component.turns * component.current,
                        diameter=2.0 * component.dims["radius"],
                        position=component.center,
                    )
                )
        return scaled, polarised

    @staticmethod
    def _magnet(magnet: Magnet):
        if magnet.kind == "cuboid":
            dims = magnet.dims
            return magpy.magnet.Cuboid(
                polarization=magnet.br,
                dimension=(dims["size_x"], dims["size_y"], dims["size_z"]),
                position=magnet.center,
            )
        if magnet.kind == "cylinder":
            return magpy.magnet.Cylinder(
                polarization=magnet.br,
                dimension=(2.0 * magnet.dims["radius"], magnet.dims["length"]),
                position=magnet.center,
            )
        raise AssertionError(f"unvalidated magnet kind {magnet.kind!r}")  # pragma: no cover


def field_magnitude(field: np.ndarray) -> np.ndarray:
    """Row-wise ``|B|`` of an ``(n, 3)`` array — the one definition the CLI exports."""
    array = np.asarray(field, dtype=float)
    return np.sqrt(np.sum(array * array, axis=1))
