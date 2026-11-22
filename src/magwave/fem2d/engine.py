"""Tier 1 axisymmetric FEM engine implementing the MagWave :class:`Engine` protocol."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .force import ForceResult

from .. import __version__
from ..cases import Case, Coil, Dipole, Magnet, Permeable, UniformField
from ..engines.base import (
    Capabilities,
    ConvergenceInfo,
    EngineError,
    MagnetOperatingPoint,
    Problem,
    Result,
    Sample,
    raise_on_errors,
)
from ..materials import (
    LIBRARY_VERSION,
    MAGNET_LIBRARY_VERSION,
    DemagCurve,
    Material,
    demag_curve_for,
    load_library,
)
from ..units import MU0
from .assembly import element_geometry, load_current_density, load_sheet
from .field import sample_field
from .mesh import Mesh, rectangular_mesh
from .nonlinear import (
    ARMIJO_ALPHA0,
    ARMIJO_C,
    ARMIJO_MAX_HALVINGS,
    ARMIJO_RHO,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOL,
    NonlinearConvergenceError,
    count_interface_nodes,
    solve_axisymmetric_nonlinear,
)
from .solve import SolveReport, dirichlet_nodes, solve_axisymmetric

ENGINE_ID = "fem2d"
RUNG = "Tier 1"
MATERIAL_MODELS = ("linear", "nonlinear")

E_UNSUPPORTED_SOURCE_TYPE = "E_UNSUPPORTED_SOURCE_TYPE"
E_UNSUPPORTED_GEOMETRY = "E_UNSUPPORTED_GEOMETRY"
E_UNSUPPORTED_MAGNETISATION = "E_UNSUPPORTED_MAGNETISATION"
E_NOT_AXISYMMETRIC = "E_NOT_AXISYMMETRIC"
E_UNSUPPORTED_QUANTITY = "E_UNSUPPORTED_QUANTITY"
E_UNSUPPORTED_MATERIAL = "E_UNSUPPORTED_MATERIAL"
E_UNSUPPORTED_MANIFEST = "E_UNSUPPORTED_MANIFEST"
E_ACCURACY_UNREACHABLE = "E_ACCURACY_UNREACHABLE"
E_PROBE_OUTSIDE_DOMAIN = "E_PROBE_OUTSIDE_DOMAIN"
E_NOT_CONVERGED = "E_NOT_CONVERGED"

_AXIS_TOL = 1e-12


def _effective_temperature_c(case: Case) -> float:
    return 20.0 if case.temperature_c is None else float(case.temperature_c)


def _case_geometry_bounds(
    case: Case,
) -> tuple[list[float], list[float], float, float, float, float, float]:
    """Return ``(r_breaks, z_breaks, max_outer_radius, min_z1, max_z2, l_max, min_sphere_radius)``."""
    r_breaks: list[float] = []
    z_breaks: list[float] = []
    max_outer_radius = 0.0
    min_z1 = math.inf
    max_z2 = -math.inf
    l_max = 0.0
    min_sphere_radius = math.inf

    for component in case.components:
        if isinstance(component, Coil):
            r_in = component.dims["inner_radius"]
            r_out = component.dims["outer_radius"]
            length = component.dims["length"]
            z1 = component.center[2] - 0.5 * length
            z2 = component.center[2] + 0.5 * length
            r_breaks.extend((r_in, r_out))
            z_breaks.extend((z1, z2))
            max_outer_radius = max(max_outer_radius, r_out)
            min_z1 = min(min_z1, z1)
            max_z2 = max(max_z2, z2)
            l_max = max(l_max, r_out, length)
        elif isinstance(component, Magnet):
            r_out = component.dims["radius"]
            length = component.dims["length"]
            z1 = component.center[2] - 0.5 * length
            z2 = component.center[2] + 0.5 * length
            r_breaks.append(r_out)
            z_breaks.extend((z1, z2))
            max_outer_radius = max(max_outer_radius, r_out)
            min_z1 = min(min_z1, z1)
            max_z2 = max(max_z2, z2)
            l_max = max(l_max, r_out, length)
        elif isinstance(component, Permeable):
            rc = math.hypot(component.center[0], component.center[1])
            zc = component.center[2]
            if component.kind == "sphere":
                rad = component.dims["radius"]
                r_breaks.extend((rc - rad, rc + rad))
                z_breaks.extend((zc - rad, zc + rad))
                max_outer_radius = max(max_outer_radius, rc + rad)
                min_z1 = min(min_z1, zc - rad)
                max_z2 = max(max_z2, zc + rad)
                l_max = max(l_max, rc + rad, 2.0 * rad)
                min_sphere_radius = min(min_sphere_radius, rad)
            elif component.kind == "cylinder":
                rad = component.dims["radius"]
                length = component.dims["length"]
                z1 = zc - 0.5 * length
                z2 = zc + 0.5 * length
                r_breaks.extend((rc - rad, rc + rad))
                z_breaks.extend((z1, z2))
                max_outer_radius = max(max_outer_radius, rc + rad)
                min_z1 = min(min_z1, z1)
                max_z2 = max(max_z2, z2)
                l_max = max(l_max, rc + rad, length)

    return r_breaks, z_breaks, max_outer_radius, min_z1, max_z2, l_max, min_sphere_radius


def default_element_size(case: Case) -> float:
    """Return the implicit default ``Fem2dEngine`` element size (metres) for ``case``."""
    _, _, _, _, _, l_max, min_sphere_radius = _case_geometry_bounds(case)
    if l_max == 0.0:
        probe_r = max((math.hypot(p.point[0], p.point[1]) for p in case.probes), default=0.0)
        probe_z = [p.point[2] for p in case.probes] or [0.0]
        l_max = max(probe_r, max(probe_z) - min(probe_z), 0.1)
    h = l_max / 40.0
    if math.isfinite(min_sphere_radius):
        h = min(h, min_sphere_radius / 40.0)
    return h


def _body_rz_box(component: Magnet | Permeable) -> tuple[float, float, float, float]:
    rc = math.hypot(component.center[0], component.center[1])
    zc = component.center[2]
    rad = component.dims["radius"]
    if component.kind == "sphere":
        return max(0.0, rc - rad), rc + rad, zc - rad, zc + rad
    half_len = 0.5 * component.dims["length"]
    return max(0.0, rc - rad), rc + rad, zc - half_len, zc + half_len


def _magnet_overlaps_body(mag: Magnet, other: Magnet | Permeable) -> bool:
    """Return ``True`` if an axisymmetric cylinder magnet overlaps ``other`` in ``(r, z)``."""
    ma0, ma1, za0, za1 = _body_rz_box(mag)
    if other.kind == "cylinder":
        ob0, ob1, zb0, zb1 = _body_rz_box(other)
        return (min(ma1, ob1) - max(ma0, ob0)) > _AXIS_TOL and (
            min(za1, zb1) - max(za0, zb0)
        ) > _AXIS_TOL
    rs = math.hypot(other.center[0], other.center[1])
    zs = other.center[2]
    dr = max(ma0 - rs, 0.0, rs - ma1)
    dz = max(za0 - zs, 0.0, zs - za1)
    return math.hypot(dr, dz) < (other.dims["radius"] - _AXIS_TOL)


def _label(component: Magnet | Coil | Dipole | Permeable | UniformField, index: int) -> str:
    name = component.name or f"#{index}"
    return f"component {name}"


def _permeable_overlap(a: Permeable, b: Permeable) -> bool:
    """Return ``True`` if two axisymmetric permeable bodies overlap in ``(r, z)``."""
    ra = math.hypot(a.center[0], a.center[1])
    za = a.center[2]
    rb = math.hypot(b.center[0], b.center[1])
    zb = b.center[2]

    if a.kind == "sphere" and b.kind == "sphere":
        return math.hypot(ra - rb, za - zb) < (a.dims["radius"] + b.dims["radius"] - _AXIS_TOL)

    if a.kind == "cylinder" and b.kind == "cylinder":
        ra0, ra1 = max(0.0, ra - a.dims["radius"]), ra + a.dims["radius"]
        rb0, rb1 = max(0.0, rb - b.dims["radius"]), rb + b.dims["radius"]
        za0, za1 = za - 0.5 * a.dims["length"], za + 0.5 * a.dims["length"]
        zb0, zb1 = zb - 0.5 * b.dims["length"], zb + 0.5 * b.dims["length"]
        return (min(ra1, rb1) - max(ra0, rb0)) > _AXIS_TOL and (
            min(za1, zb1) - max(za0, zb0)
        ) > _AXIS_TOL

    sph, cyl = (a, b) if a.kind == "sphere" else (b, a)
    rs, zs = (ra, za) if sph is a else (rb, zb)
    rc, zc = (rb, zb) if cyl is b else (ra, za)
    rc0, rc1 = max(0.0, rc - cyl.dims["radius"]), rc + cyl.dims["radius"]
    zc0, zc1 = zc - 0.5 * cyl.dims["length"], zc + 0.5 * cyl.dims["length"]
    dr = max(rc0 - rs, 0.0, rs - rc1)
    dz = max(zc0 - zs, 0.0, zs - zc1)
    return math.hypot(dr, dz) < (sph.dims["radius"] - _AXIS_TOL)


def _dedup_sorted(values: list[float], low: float, high: float) -> list[float]:
    filtered = sorted(v for v in values if (low + _AXIS_TOL) < v < (high - _AXIS_TOL))
    out: list[float] = []
    for val in filtered:
        if not out or abs(val - out[-1]) > _AXIS_TOL:
            out.append(val)
    return out


class _FieldState:
    """Internal solved field state retained for Tier 1 force/energy post-processing."""

    mesh: Mesh
    a_phi: np.ndarray
    load: np.ndarray
    nu_final: np.ndarray
    assignments: list[tuple[Permeable, np.ndarray, Material, str]]
    demag_assignments: list[tuple[Magnet, int, np.ndarray, DemagCurve, float, bool]]
    element_size_used: float


# Decorate _FieldState cleanly below
_FieldState = dataclass(frozen=True)(_FieldState)


class Fem2dEngine:
    """Linear axisymmetric ``A_phi`` finite-element magnetostatics solver (Tier 1, M5.2a).

    Solves ``curl( nu curl A ) = J_phi`` in cylindrical ``(r, z)`` coordinates on a graded
    tensor-product triangular mesh using linear triangles and one-point centroid quadrature,
    with Dirichlet ``A = 0`` on the symmetry axis (``r = 0``) and either ``A = 0`` or
    ``A = MU0 * H0 * r / 2`` on the far-field outer boundaries when a ``uniform_field`` is
    present.

    Conventions
    -----------
    * Volume-winding solenoids (``inner_radius < outer_radius``) use fill factor ``1.0``
      (``J_phi = turns * current / ((outer_radius - inner_radius) * length)``), matching
      :mod:`magwave.benchmarks.checks`.
    * Thin solenoids (``inner_radius == outer_radius``) and axially magnetised cylinders
      (``K_phi = M = Br_z / MU0``) share the exact surface-current sheet load path.
    * Linear permeable ``sphere`` and ``cylinder`` components assign element-wise
      ``nu_e = 1 / (MU0 * mu_r)`` by element centroid on the tensor-product mesh. Every node
      is shared across incident elements (no hanging nodes), so curved interfaces are a
      staircase approximation rather than a node-conforming boundary.
    * Nonlinear B–H laws and CAD-meshed ``permeable_solid`` manifest bodies remain deferred
      to later stages.
    """

    def __init__(
        self,
        element_size: float | None = None,
        far_field_factor: float = 5.0,
        growth: float = 1.2,
        certified_accuracy: float = 1e-2,
        domain: tuple[float, float, float] | None = None,
        refinement_box: tuple[float, float, float, float] | None = None,
        material_model: str = "linear",
        tol: float = DEFAULT_TOL,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        magnet_curves: Mapping[str, DemagCurve] | None = None,
    ) -> None:
        if element_size is not None and (
            not math.isfinite(element_size) or float(element_size) <= 0.0
        ):
            raise ValueError(f"element_size must be positive and finite, got {element_size!r}")
        if not math.isfinite(far_field_factor) or float(far_field_factor) <= 0.0:
            raise ValueError(f"far_field_factor must be positive, got {far_field_factor!r}")
        if not math.isfinite(growth) or float(growth) <= 1.0:
            raise ValueError(f"growth must be > 1.0, got {growth!r}")
        if not math.isfinite(certified_accuracy) or float(certified_accuracy) <= 0.0:
            raise ValueError(f"certified_accuracy must be positive, got {certified_accuracy!r}")
        if material_model not in MATERIAL_MODELS:
            raise ValueError(
                f"material_model must be one of {MATERIAL_MODELS}, got {material_model!r}"
            )
        if (
            isinstance(tol, bool)
            or not isinstance(tol, int | float)
            or not math.isfinite(tol)
            or float(tol) <= 0.0
        ):
            raise ValueError(f"tol must be positive and finite, got {tol!r}")
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError(f"max_iterations must be a positive integer, got {max_iterations!r}")

        self._element_size = None if element_size is None else float(element_size)
        self._far_field_factor = float(far_field_factor)
        self._growth = float(growth)
        self._certified_accuracy = float(certified_accuracy)
        self._domain = None if domain is None else tuple(float(v) for v in domain)
        self._refinement_box = (
            None if refinement_box is None else tuple(float(v) for v in refinement_box)
        )
        self._material_model = material_model
        self._tol = float(tol)
        self._max_iterations = int(max_iterations)
        self._magnet_curves: dict[str, DemagCurve] = (
            dict(magnet_curves) if magnet_curves is not None else {}
        )
        self._version = f"{__version__}+m5.4c"

    @property
    def certified_accuracy(self) -> float:
        """Relative accuracy certified by this stage (default ``1e-2``, i.e. 1 %)."""
        return self._certified_accuracy

    @property
    def material_model(self) -> str:
        """Selected permeable material law (``'linear'`` or ``'nonlinear'``)."""
        return self._material_model

    @property
    def tol(self) -> float:
        """Relative residual convergence tolerance for nonlinear solves."""
        return self._tol

    @property
    def max_iterations(self) -> int:
        """Maximum Newton iterations allowed for nonlinear solves."""
        return self._max_iterations

    def capabilities(self) -> Capabilities:
        return Capabilities(
            engine_id=ENGINE_ID,
            rung=RUNG,
            source_types=frozenset({"magnet", "coil"}),
            magnet_kinds=frozenset({"cylinder"}),
            coil_kinds=frozenset({"solenoid"}),
            quantities=frozenset({"B"}),
            force_methods=("maxwell_stress_tensor", "coenergy_virtual_work"),
            engine_version=self._version,
            materials=frozenset(
                {"air", "linear_permeable", "nonlinear_permeable", "permanent_magnet_demag_curve"}
            ),
        )

    @property
    def method(self) -> str:
        return (
            "axisymmetric A_phi FEM, linear triangles, centroid quadrature, "
            f"Dirichlet A=0 on axis and at {self._far_field_factor:g}x far field, "
            "scipy.sparse.linalg.spsolve"
        )

    def _resolve_demag_curve(
        self, grade_name: str, temperature_c: float
    ) -> tuple[DemagCurve, bool]:
        if grade_name in self._magnet_curves:
            return self._magnet_curves[grade_name].at_temperature(temperature_c), True
        return demag_curve_for(grade_name, temperature_c), False

    def validate(self, case: Case) -> list[Problem]:
        """Validate ``case`` against Tier 1 capabilities, materials and accuracy."""
        problems: list[Problem] = []
        temp_c = _effective_temperature_c(case)

        # 1. Components first.
        permeable_bodies: list[Permeable] = []
        demag_magnets: list[Magnet] = []
        for index, component in enumerate(case.components):
            label = _label(component, index)
            if isinstance(component, UniformField):
                continue

            if isinstance(component, Permeable):
                if component.kind not in ("sphere", "cylinder"):
                    problems.append(
                        Problem(
                            E_UNSUPPORTED_GEOMETRY,
                            f"{case.name}: {label} is a {component.kind!r} permeable body; "
                            "fem2d supports axisymmetric 'sphere' and 'cylinder' bodies only",
                            "error",
                        )
                    )
                    continue
                try:
                    mat, _ = load_library().with_override(component.material, component.mu_r)
                except ValueError as exc:
                    problems.append(
                        Problem(
                            E_UNSUPPORTED_MATERIAL,
                            f"{case.name}: {label} material error: {exc}",
                            "error",
                        )
                    )
                    continue
                if self._material_model == "nonlinear":
                    if mat.bh_table is None:
                        problems.append(
                            Problem(
                                E_UNSUPPORTED_MATERIAL,
                                f"{case.name}: {label} material {mat.name!r} has no B-H table data; "
                                "cannot solve with material_model='nonlinear'",
                                "error",
                            )
                        )
                        continue
                    if component.mu_r is not None:
                        problems.append(
                            Problem(
                                E_UNSUPPORTED_MATERIAL,
                                f"{case.name}: {label} specifies linear mu_r override "
                                f"({component.mu_r:g}), which cannot be combined with "
                                "material_model='nonlinear'",
                                "error",
                            )
                        )
                        continue
                elif not math.isfinite(mat.mu_r) or mat.mu_r <= 0.0:
                    problems.append(
                        Problem(
                            E_UNSUPPORTED_MATERIAL,
                            f"{case.name}: {label} requires a positive scalar linear mu_r",
                            "error",
                        )
                    )
                    continue
                permeable_bodies.append(component)
                continue

            if isinstance(component, Dipole):
                problems.append(
                    Problem(
                        E_UNSUPPORTED_SOURCE_TYPE,
                        f"{case.name}: {label} is a point Dipole; fem2d models axisymmetric "
                        "cylinders and solenoids only",
                        "error",
                    )
                )
                continue

            if isinstance(component, Magnet):
                if component.kind != "cylinder":
                    problems.append(
                        Problem(
                            E_UNSUPPORTED_GEOMETRY,
                            f"{case.name}: {label} is a {component.kind!r} magnet; fem2d "
                            "supports axisymmetric 'cylinder' magnets only",
                            "error",
                        )
                    )
                    continue
                if component.center[0] != 0.0 or component.center[1] != 0.0:
                    problems.append(
                        Problem(
                            E_NOT_AXISYMMETRIC,
                            f"{case.name}: {label} is centred at {component.center}, off the "
                            "z symmetry axis (x == 0, y == 0 required)",
                            "error",
                        )
                    )
                if component.br[0] != 0.0 or component.br[1] != 0.0:
                    problems.append(
                        Problem(
                            E_UNSUPPORTED_MAGNETISATION,
                            f"{case.name}: {label} has non-axial remanence br={component.br}; "
                            "only axial magnetisation (0, 0, Br_z) is axisymmetric",
                            "error",
                        )
                    )
                if component.demag_curve is not None:
                    try:
                        self._resolve_demag_curve(component.demag_curve, temp_c)
                    except ValueError as exc:
                        problems.append(
                            Problem(
                                E_UNSUPPORTED_MATERIAL,
                                f"{case.name}: {label} magnet demag curve error: {exc}",
                                "error",
                            )
                        )
                    else:
                        demag_magnets.append(component)
                continue

            if isinstance(component, Coil):
                if component.kind != "solenoid":
                    problems.append(
                        Problem(
                            E_UNSUPPORTED_GEOMETRY,
                            f"{case.name}: {label} is a {component.kind!r} coil; fem2d "
                            "supports axisymmetric 'solenoid' coils only",
                            "error",
                        )
                    )
                    continue
                if component.center[0] != 0.0 or component.center[1] != 0.0:
                    problems.append(
                        Problem(
                            E_NOT_AXISYMMETRIC,
                            f"{case.name}: {label} is centred at {component.center}, off the "
                            "z symmetry axis (x == 0, y == 0 required)",
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

        if len(permeable_bodies) > 1:
            overlaps: list[str] = []
            for idx_a, body_a in enumerate(permeable_bodies):
                for body_b in permeable_bodies[idx_a + 1 :]:
                    if _permeable_overlap(body_a, body_b):
                        name_a = body_a.name or f"#{case.components.index(body_a)}"
                        name_b = body_b.name or f"#{case.components.index(body_b)}"
                        overlaps.append(f"{name_a} and {name_b}")
            if overlaps:
                problems.append(
                    Problem(
                        E_UNSUPPORTED_MATERIAL,
                        f"{case.name}: permeable bodies overlap ({', '.join(overlaps)})",
                        "error",
                    )
                )

        if demag_magnets:
            mag_overlaps: list[str] = []
            for idx_a, mag_a in enumerate(demag_magnets):
                for mag_b in demag_magnets[idx_a + 1 :]:
                    if _magnet_overlaps_body(mag_a, mag_b):
                        name_a = mag_a.name or f"#{case.components.index(mag_a)}"
                        name_b = mag_b.name or f"#{case.components.index(mag_b)}"
                        mag_overlaps.append(f"{name_a} and {name_b}")
                for body in permeable_bodies:
                    if _magnet_overlaps_body(mag_a, body):
                        name_a = mag_a.name or f"#{case.components.index(mag_a)}"
                        name_b = body.name or f"#{case.components.index(body)}"
                        mag_overlaps.append(f"{name_a} and {name_b}")
            if mag_overlaps:
                problems.append(
                    Problem(
                        E_UNSUPPORTED_MATERIAL,
                        f"{case.name}: demag magnet overlaps another body ({', '.join(mag_overlaps)})",
                        "error",
                    )
                )

        for index, probe in enumerate(case.probes):
            if probe.quantity == "F":
                problems.append(
                    Problem(
                        E_UNSUPPORTED_QUANTITY,
                        f"{case.name}: probe #{index} asks for force 'F'; point probes cannot "
                        "name a force target or integration contour — call "
                        "Fem2dEngine.force(case, method=...) instead; nothing was solved.",
                        "error",
                    )
                )
            elif probe.quantity not in self.capabilities().quantities:
                problems.append(
                    Problem(
                        E_UNSUPPORTED_QUANTITY,
                        f"{case.name}: probe #{index} asks for {probe.quantity!r}; fem2d "
                        "solve() samples B only — call Fem2dEngine.force(case, method=...) for "
                        "forces; nothing was solved.",
                        "error",
                    )
                )

        # 2. Manifest / materials second.
        if case.manifest is not None:
            for body in case.manifest.bodies:
                if body.role == "permeable_solid":
                    problems.append(
                        Problem(
                            E_UNSUPPORTED_MATERIAL,
                            f"{case.name}: body #{body.index} ({body.display_name}) has role "
                            "'permeable_solid'; CAD bodies are not meshed yet in Tier 1 "
                            "stage 2 (M5.2). Express the body as a 'permeable' component with "
                            "explicit geometry, or wait for CAD meshing. Nothing was solved.",
                            "error",
                        )
                    )
                elif body.role not in ("air", "symmetry_plane"):
                    problems.append(
                        Problem(
                            E_UNSUPPORTED_MANIFEST,
                            f"{case.name}: body #{body.index} ({body.display_name}) has manifest "
                            f"role {body.role!r}; CAD geometry is not meshed by this solver yet",
                            "error",
                        )
                    )

        # 3. Accuracy gate last (only when the case is otherwise solvable).
        if (
            not any(problem.is_error for problem in problems)
            and case.tolerance < self._certified_accuracy
        ):
            problems.append(
                Problem(
                    E_ACCURACY_UNREACHABLE,
                    f"{case.name}: research-owned tolerance {case.tolerance:g} "
                    f"({case.tolerance:.2e}) is tighter than fem2d's certified accuracy "
                    f"{self._certified_accuracy:g} ({self._certified_accuracy:.2e}, 1 %); "
                    "the number is not claimed",
                    "error",
                )
            )

        return problems

    def _build_mesh_and_load(
        self, case: Case
    ) -> tuple[Mesh, np.ndarray, list[tuple[float, float]]]:
        """Construct the graded mesh and assemble volume/sheet loads for ``case``."""
        (
            r_breaks,
            z_breaks,
            max_outer_radius,
            min_z1,
            max_z2,
            l_max,
            min_sphere_radius,
        ) = _case_geometry_bounds(case)

        effective_h = (
            self._element_size
            if self._element_size is not None
            else (case.mesh.element_size if case.mesh is not None else None)
        )

        if l_max == 0.0:
            probe_r = max((math.hypot(p.point[0], p.point[1]) for p in case.probes), default=0.0)
            probe_z = [p.point[2] for p in case.probes] or [0.0]
            max_outer_radius = probe_r
            min_z1 = min(probe_z)
            max_z2 = max(probe_z)
            default_len = 10.0 * effective_h if effective_h is not None else 0.1
            l_max = max(probe_r, max_z2 - min_z1, default_len)

        if effective_h is not None:
            h = effective_h
        else:
            h = l_max / 40.0
            if math.isfinite(min_sphere_radius):
                h = min(h, min_sphere_radius / 40.0)
        if self._domain is not None:
            r_max, z_min, z_max = self._domain
        else:
            pad = self._far_field_factor * l_max
            r_max = max_outer_radius + pad
            z_min = min_z1 - pad
            z_max = max_z2 + pad

        probes_rz: list[tuple[float, float]] = []
        for index, probe in enumerate(case.probes):
            pr = math.hypot(probe.point[0], probe.point[1])
            pz = probe.point[2]
            if pr > r_max or pz < z_min or pz > z_max:
                raise EngineError(
                    ENGINE_ID,
                    [
                        Problem(
                            E_PROBE_OUTSIDE_DOMAIN,
                            f"{case.name}: probe #{index} at {probe.point} maps to "
                            f"(r={pr}, z={pz}), outside domain [0, {r_max}] x [{z_min}, {z_max}]; "
                            f"increase far_field_factor (currently {self._far_field_factor})",
                            "error",
                        )
                    ],
                )
            probes_rz.append((pr, pz))

        if self._refinement_box is not None:
            ref_box = self._refinement_box
        else:
            near_r = [max_outer_radius]
            near_z_min = min_z1
            near_z_max = max_z2
            for pr, pz in probes_rz:
                if pr <= max_outer_radius + l_max:
                    near_r.append(pr)
                if (min_z1 - l_max) <= pz <= (max_z2 + l_max):
                    near_z_min = min(near_z_min, pz)
                    near_z_max = max(near_z_max, pz)
            ref_r1 = min(r_max, max(near_r) + 2.0 * h)
            ref_z0 = max(z_min, near_z_min - 2.0 * h)
            ref_z1 = min(z_max, near_z_max + 2.0 * h)
            ref_box = (0.0, ref_r1, ref_z0, ref_z1)

        mesh = rectangular_mesh(
            r_max,
            z_min,
            z_max,
            h,
            refinement_box=ref_box,
            breakpoints_r=_dedup_sorted(r_breaks, 0.0, r_max),
            breakpoints_z=_dedup_sorted(z_breaks, z_min, z_max),
            growth=self._growth,
        )

        load = np.zeros(mesh.nodes.shape[0], dtype=np.float64)
        for component in case.components:
            if isinstance(component, Coil):
                r_in = component.dims["inner_radius"]
                r_out = component.dims["outer_radius"]
                length = component.dims["length"]
                z1 = component.center[2] - 0.5 * length
                z2 = component.center[2] + 0.5 * length
                ampere_turns = component.turns * component.current
                if r_out > r_in:
                    # Fill factor 1.0 per benchmarks/checks.py (Case schema has no fill factor).
                    j_phi = ampere_turns / ((r_out - r_in) * length)
                    load += load_current_density(mesh, (r_in, r_out, z1, z2), j_phi)
                else:
                    k_phi = ampere_turns / length
                    load += load_sheet(mesh, r_in, (z1, z2), k_phi)
            elif isinstance(component, Magnet):
                r_out = component.dims["radius"]
                length = component.dims["length"]
                z1 = component.center[2] - 0.5 * length
                z2 = component.center[2] + 0.5 * length
                if component.demag_curve is None:
                    k_phi = component.br[2] / MU0
                else:
                    curve, _ = self._resolve_demag_curve(
                        component.demag_curve, _effective_temperature_c(case)
                    )
                    sign = -1.0 if component.br[2] < 0.0 else 1.0
                    k_phi = sign * curve.br() / MU0
                load += load_sheet(mesh, r_out, (z1, z2), k_phi)

        return mesh, load, probes_rz

    def _permeable_assignments(
        self, mesh: Mesh, case: Case
    ) -> tuple[list[tuple[Permeable, np.ndarray, Material, str]], np.ndarray]:
        """Return per-body element masks/materials and per-element region IDs."""
        permeables = [c for c in case.components if isinstance(c, Permeable)]
        m_elems = int(mesh.elements.shape[0])
        region_ids = np.zeros(m_elems, dtype=np.int64)
        if not permeables:
            return [], region_ids

        _, r_c, z_c, _, _ = element_geometry(mesh)
        lib = load_library()
        assignments: list[tuple[Permeable, np.ndarray, Material, str]] = []
        for body_idx, component in enumerate(permeables, start=1):
            mat, prov_line = lib.with_override(component.material, component.mu_r)
            rc = math.hypot(component.center[0], component.center[1])
            zc = component.center[2]
            if component.kind == "sphere":
                rad = component.dims["radius"]
                mask = ((r_c - rc) ** 2 + (z_c - zc) ** 2) <= rad**2
            elif component.kind == "cylinder":
                rad = component.dims["radius"]
                half_len = 0.5 * component.dims["length"]
                mask = (np.abs(r_c - rc) <= rad) & (np.abs(z_c - zc) <= half_len)
            else:  # pragma: no cover - validate() rejects other kinds
                raise AssertionError(f"unvalidated permeable kind {component.kind!r}")
            assignments.append((component, mask, mat, prov_line))
            if self._material_model == "nonlinear" or mat.mu_r != 1.0:
                region_ids[mask] = body_idx
        return assignments, region_ids

    def _element_reluctivity(self, mesh: Mesh, case: Case) -> tuple[float | np.ndarray, list[str]]:
        """Return scalar or ``(m,)`` element reluctivity and provenance lines."""
        assignments, _ = self._permeable_assignments(mesh, case)
        if not assignments:
            return 1.0 / MU0, []

        nu_arr = np.full(mesh.elements.shape[0], 1.0 / MU0, dtype=np.float64)
        notes: list[str] = []
        for _, mask, mat, prov_line in assignments:
            nu_arr[mask] = mat.nu()
            notes.append(prov_line)
        return nu_arr, notes

    @staticmethod
    def _dirichlet_bc(
        mesh: Mesh, case: Case
    ) -> tuple[np.ndarray, np.ndarray | None, UniformField | None]:
        """Return ``(constrained_nodes, values_or_None, uniform_field_or_None)``."""
        constrained = dirichlet_nodes(mesh)
        uniform = next((c for c in case.components if isinstance(c, UniformField)), None)
        if uniform is None:
            return constrained, None, None
        r_vals = mesh.nodes[constrained, 0]
        vals = 0.5 * MU0 * uniform.h0 * r_vals
        vals[r_vals <= _AXIS_TOL] = 0.0
        return constrained, vals, uniform

    def _magnet_demag_assignments(
        self,
        mesh: Mesh,
        case: Case,
        region_ids: np.ndarray,
    ) -> list[tuple[Magnet, int, np.ndarray, DemagCurve, float, bool]]:
        """Return demag-magnet assignments and update ``region_ids`` for interface counting."""
        if not any(isinstance(c, Magnet) and c.demag_curve is not None for c in case.components):
            return []

        _, r_c, z_c, _, _ = element_geometry(mesh)
        temp_c = _effective_temperature_c(case)
        next_id = int(np.max(region_ids)) + 1 if region_ids.size else 1
        demag_assignments: list[tuple[Magnet, int, np.ndarray, DemagCurve, float, bool]] = []
        for comp_idx, component in enumerate(case.components):
            if isinstance(component, Magnet) and component.demag_curve is not None:
                curve, is_custom = self._resolve_demag_curve(component.demag_curve, temp_c)
                r_out = component.dims["radius"]
                half_len = 0.5 * component.dims["length"]
                zc = component.center[2]
                mask = (r_c <= r_out) & (np.abs(z_c - zc) <= half_len)
                sign = -1.0 if component.br[2] < 0.0 else 1.0
                region_ids[mask] = next_id
                next_id += 1
                demag_assignments.append((component, comp_idx, mask, curve, sign, is_custom))
        return demag_assignments

    def _solve_core(self, case: Case) -> tuple[Result, SolveReport, _FieldState]:
        """Validate and solve ``case``, returning ``(Result, SolveReport, _FieldState)``."""
        warnings = raise_on_errors(ENGINE_ID, self.validate(case))
        mesh, load, probes_rz = self._build_mesh_and_load(case)
        assignments, region_ids = self._permeable_assignments(mesh, case)
        demag_assignments = self._magnet_demag_assignments(mesh, case, region_ids)
        interface_nodes = count_interface_nodes(mesh, region_ids)
        constrained, values, uniform = self._dirichlet_bc(mesh, case)

        if demag_assignments:
            temp_c = _effective_temperature_c(case)
            if assignments and self._material_model == "nonlinear":
                element_curves = [(mask, mat.bh_curve()) for _, mask, mat, _ in assignments]
                base_nu: np.ndarray | None = None
                steel_notes = [
                    f"{mat.name} nonlinear B-H (PCHIP)"
                    + (f", Ms={mat.saturation_B:g} T" if mat.saturation_B is not None else "")
                    for _, _, mat, _ in assignments
                ]
            elif assignments:
                element_curves = []
                base_nu = np.full(mesh.elements.shape[0], 1.0 / MU0, dtype=np.float64)
                steel_notes = []
                for _, mask, mat, prov_line in assignments:
                    base_nu[mask] = mat.nu()
                    steel_notes.append(prov_line)
            else:
                element_curves = []
                base_nu = None
                steel_notes = []

            magnet_regions = [
                (mask, curve, sign) if sign < 0.0 else (mask, curve)
                for _, _, mask, curve, sign, _ in demag_assignments
            ]
            try:
                a_phi, report, history, nu_final = solve_axisymmetric_nonlinear(
                    mesh,
                    load,
                    element_curves,
                    base_nu=base_nu,
                    constrained=constrained,
                    values=values,
                    tol=self._tol,
                    max_iterations=self._max_iterations,
                    magnet_regions=magnet_regions,
                )
            except NonlinearConvergenceError as exc:
                raise EngineError(
                    ENGINE_ID,
                    [
                        Problem(
                            E_NOT_CONVERGED,
                            f"{case.name}: Newton solver did not converge after {exc.iterations} "
                            f"iteration(s) (cap={exc.max_iterations}, "
                            f"final_relative_residual={exc.final_relative_residual:.6e}, "
                            f"reason={exc.reason}); no result produced",
                            "error",
                        )
                    ],
                ) from exc

            op_points: list[MagnetOperatingPoint] = []
            mag_notes: list[str] = []
            for component, comp_idx, _, curve, sign, is_custom in demag_assignments:
                bz_centroid = float(
                    sample_field(
                        mesh, a_phi, np.array([[0.0, component.center[2]]], dtype=np.float64)
                    )[0, 1]
                )
                b_op = sign * bz_centroid
                h_op = float(curve.h_of_b(b_op))
                mag_label = component.name or f"#{comp_idx}"
                lib_tag = (
                    "custom curve" if is_custom else f"magwave.magnets v{MAGNET_LIBRARY_VERSION}"
                )
                op_source = f"{curve.name} @ {temp_c:g} C (Br={curve.br():.6g} T, {lib_tag})"
                op_points.append(
                    MagnetOperatingPoint(
                        name=mag_label,
                        h_a_per_m=h_op,
                        b_t=b_op,
                        source=op_source,
                    )
                )
                mag_notes.append(
                    f"{mag_label}: {curve.name} at T={temp_c:g} C, "
                    f"Br(T)={curve.br():.6g} T, op=(H={h_op:.1f} A/m, B={b_op:.6f} T)"
                )

            all_notes = mag_notes + steel_notes
            material_desc = "; ".join(all_notes)
            relaxation_desc = (
                f"Armijo backtracking, alpha0={ARMIJO_ALPHA0:g}, rho={ARMIJO_RHO:g}, "
                f"c={ARMIJO_C:g}, max {ARMIJO_MAX_HALVINGS} halvings"
            )
            solver_desc = "Newton + Armijo backtracking, scipy.sparse.linalg.spsolve"
            convergence = ConvergenceInfo(
                iterations=report.iterations,
                final_relative_residual=float(report.relative_residual),
                residual_history=tuple(float(val) for val in history),
                nodes=report.mesh.nodes,
                elements=report.mesh.elements,
                interface_nodes=interface_nodes,
                solver=solver_desc,
                relaxation=relaxation_desc,
                material_model=material_desc,
                magnet_operating_points=tuple(op_points),
            )
            bc_desc = (
                f"applied field H0 = {uniform.h0:g} A/m via Dirichlet A = MU0*H0*r/2 "
                f"at {self._far_field_factor:g}x far field (A=0 on axis)"
                if uniform is not None
                else f"Dirichlet A=0 on axis and at {self._far_field_factor:g}x far field"
            )
            lib_sources = [f"magwave.magnets v{MAGNET_LIBRARY_VERSION}"]
            if steel_notes:
                lib_sources.append(f"magwave.materials v{LIBRARY_VERSION}")
            method_str = ", ".join(
                [
                    "axisymmetric A_phi FEM, linear triangles, centroid quadrature",
                    f"second-quadrant PM demag ({'; '.join(lib_sources)}: {material_desc})",
                    bc_desc,
                    solver_desc,
                ]
            )
        elif self._material_model == "nonlinear" and bool(assignments):
            element_curves = [(mask, mat.bh_curve()) for _, mask, mat, _ in assignments]
            nl_notes = [
                f"{mat.name} nonlinear B-H (PCHIP)"
                + (f", Ms={mat.saturation_B:g} T" if mat.saturation_B is not None else "")
                for _, _, mat, _ in assignments
            ]
            try:
                a_phi, report, history, nu_final = solve_axisymmetric_nonlinear(
                    mesh,
                    load,
                    element_curves,
                    constrained=constrained,
                    values=values,
                    tol=self._tol,
                    max_iterations=self._max_iterations,
                )
            except NonlinearConvergenceError as exc:
                raise EngineError(
                    ENGINE_ID,
                    [
                        Problem(
                            E_NOT_CONVERGED,
                            f"{case.name}: Newton solver did not converge after {exc.iterations} "
                            f"iteration(s) (cap={exc.max_iterations}, "
                            f"final_relative_residual={exc.final_relative_residual:.6e}, "
                            f"reason={exc.reason}); no result produced",
                            "error",
                        )
                    ],
                ) from exc

            material_desc = "; ".join(nl_notes)
            relaxation_desc = (
                f"Armijo backtracking, alpha0={ARMIJO_ALPHA0:g}, rho={ARMIJO_RHO:g}, "
                f"c={ARMIJO_C:g}, max {ARMIJO_MAX_HALVINGS} halvings"
            )
            solver_desc = "Newton + Armijo backtracking, scipy.sparse.linalg.spsolve"
            convergence = ConvergenceInfo(
                iterations=report.iterations,
                final_relative_residual=float(report.relative_residual),
                residual_history=tuple(float(val) for val in history),
                nodes=report.mesh.nodes,
                elements=report.mesh.elements,
                interface_nodes=interface_nodes,
                solver=solver_desc,
                relaxation=relaxation_desc,
                material_model=material_desc,
            )
            bc_desc = (
                f"applied field H0 = {uniform.h0:g} A/m via Dirichlet A = MU0*H0*r/2 "
                f"at {self._far_field_factor:g}x far field (A=0 on axis)"
                if uniform is not None
                else f"Dirichlet A=0 on axis and at {self._far_field_factor:g}x far field"
            )
            method_str = ", ".join(
                [
                    "axisymmetric A_phi FEM, linear triangles, centroid quadrature",
                    f"element-wise nonlinear nu(|B|) from magwave.materials v{LIBRARY_VERSION} "
                    f"({material_desc})",
                    bc_desc,
                    solver_desc,
                ]
            )
        else:
            if not assignments:
                nu: float | np.ndarray = 1.0 / MU0
                material_notes: list[str] = []
                nu_final = np.full(mesh.elements.shape[0], 1.0 / MU0, dtype=np.float64)
            else:
                nu_arr = np.full(mesh.elements.shape[0], 1.0 / MU0, dtype=np.float64)
                material_notes = []
                for _, mask, mat, prov_line in assignments:
                    nu_arr[mask] = mat.nu()
                    material_notes.append(prov_line)
                nu = nu_arr
                nu_final = nu_arr

            a_phi, report = solve_axisymmetric(
                mesh, load, nu=nu, constrained=constrained, values=values
            )
            if material_notes or uniform is not None:
                method_parts = [
                    "axisymmetric A_phi FEM, linear triangles, centroid quadrature",
                ]
                if material_notes:
                    method_parts.append(
                        f"element-wise nu from magwave.materials v{LIBRARY_VERSION} "
                        f"({'; '.join(material_notes)})"
                    )
                if uniform is not None:
                    method_parts.append(
                        f"applied field H0 = {uniform.h0:g} A/m via Dirichlet A = MU0*H0*r/2 "
                        f"at {self._far_field_factor:g}x far field (A=0 on axis)"
                    )
                else:
                    method_parts.append(
                        f"Dirichlet A=0 on axis and at {self._far_field_factor:g}x far field"
                    )
                method_parts.append("scipy.sparse.linalg.spsolve")
                method_str = ", ".join(method_parts)
            else:
                method_str = self.method

            convergence = ConvergenceInfo(
                iterations=1,
                final_relative_residual=float(report.relative_residual),
                residual_history=(float(report.relative_residual),),
                nodes=report.mesh.nodes,
                elements=report.mesh.elements,
                interface_nodes=interface_nodes,
                solver="direct, scipy.sparse.linalg.spsolve",
                relaxation="none (direct linear solve)",
                material_model=(
                    "air (mu_r = 1)" if not material_notes else "; ".join(material_notes)
                ),
            )

        pts_rz = np.asarray(probes_rz, dtype=np.float64)
        br_bz = sample_field(mesh, a_phi, pts_rz)

        samples: list[Sample] = []
        for probe, (b_r, b_z) in zip(case.probes, br_bz, strict=True):
            x, y, _ = probe.point
            r = math.hypot(x, y)
            if r == 0.0:
                vector = (0.0, 0.0, float(b_z))
            else:
                vector = (float(b_r) * x / r, float(b_r) * y / r, float(b_z))
            samples.append(
                Sample(
                    point=probe.point,
                    quantity="B",
                    vector=vector,
                    unit="T",
                    component=probe.component,
                )
            )

        result = Result(
            engine_id=ENGINE_ID,
            engine_version=self._version,
            rung=RUNG,
            method=method_str,
            samples=tuple(samples),
            problems=tuple(warnings),
            convergence=convergence,
        )
        state = _FieldState(
            mesh=mesh,
            a_phi=a_phi,
            load=load,
            nu_final=nu_final,
            assignments=assignments,
            demag_assignments=demag_assignments,
            element_size_used=float(mesh.h_requested),
        )
        return result, report, state

    def solve_with_report(self, case: Case) -> tuple[Result, SolveReport]:
        """Validate and solve ``case``, returning ``(Result, SolveReport)``."""
        result, report, _ = self._solve_core(case)
        return result, report

    def solve(self, case: Case) -> Result:
        """Validate and solve ``case``; raise :class:`EngineError` if unsolvable."""
        result, _ = self.solve_with_report(case)
        return result

    def force(
        self,
        case: Case,
        *,
        method: str = "maxwell",
        target: str | None = None,
        contour: tuple[float, float, float] | None = None,
        delta: float | None = None,
    ) -> ForceResult:
        """Compute axial force on ``target`` using Maxwell stress tensor or virtual work."""
        from .force import coenergy_force, maxwell_force

        if method in ("maxwell", "maxwell_stress_tensor"):
            return maxwell_force(case, self, target=target, contour=contour)
        if method in ("coenergy", "coenergy_virtual_work"):
            return coenergy_force(case, self, target=target, delta=delta)
        raise ValueError(
            f"unsupported force method {method!r}; expected 'maxwell' or 'coenergy' "
            f"(or {self.capabilities().force_methods!r})"
        )

    def torque(self, case: Case, *, target: str | None = None) -> ForceResult:
        """Refuse torque honestly: axisymmetric A_phi has zero azimuthal stress by construction."""
        from .force import torque as _torque

        return _torque(case, self, target=target)

    def __repr__(self) -> str:
        return (
            f"Fem2dEngine(element_size={self._element_size!r}, "
            f"far_field_factor={self._far_field_factor!r}, growth={self._growth!r}, "
            f"certified_accuracy={self._certified_accuracy!r}, "
            f"material_model={self._material_model!r}, tol={self._tol!r}, "
            f"max_iterations={self._max_iterations!r})"
        )


def __getattr__(name: str) -> Any:  # pragma: no cover
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
