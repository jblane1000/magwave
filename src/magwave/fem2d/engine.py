"""Tier 1 axisymmetric FEM engine implementing the MagWave :class:`Engine` protocol."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import __version__
from ..cases import Case, Coil, Dipole, Magnet
from ..engines.base import Capabilities, EngineError, Problem, Result, Sample, raise_on_errors
from ..units import MU0
from .assembly import load_current_density, load_sheet
from .field import sample_field
from .mesh import Mesh, rectangular_mesh
from .solve import SolveReport, solve_axisymmetric

ENGINE_ID = "fem2d"
RUNG = "Tier 1"

E_UNSUPPORTED_SOURCE_TYPE = "E_UNSUPPORTED_SOURCE_TYPE"
E_UNSUPPORTED_GEOMETRY = "E_UNSUPPORTED_GEOMETRY"
E_UNSUPPORTED_MAGNETISATION = "E_UNSUPPORTED_MAGNETISATION"
E_NOT_AXISYMMETRIC = "E_NOT_AXISYMMETRIC"
E_UNSUPPORTED_QUANTITY = "E_UNSUPPORTED_QUANTITY"
E_UNSUPPORTED_MATERIAL = "E_UNSUPPORTED_MATERIAL"
E_UNSUPPORTED_MANIFEST = "E_UNSUPPORTED_MANIFEST"
E_ACCURACY_UNREACHABLE = "E_ACCURACY_UNREACHABLE"
E_PROBE_OUTSIDE_DOMAIN = "E_PROBE_OUTSIDE_DOMAIN"
W_DEMAG_IGNORED = "W_DEMAG_IGNORED"

_AXIS_TOL = 1e-12


def _label(component: Magnet | Coil | Dipole, index: int) -> str:
    name = component.name or f"#{index}"
    return f"component {name}"


def _dedup_sorted(values: list[float], low: float, high: float) -> list[float]:
    filtered = sorted(v for v in values if (low + _AXIS_TOL) < v < (high - _AXIS_TOL))
    out: list[float] = []
    for val in filtered:
        if not out or abs(val - out[-1]) > _AXIS_TOL:
            out.append(val)
    return out


class Fem2dEngine:
    """Linear axisymmetric ``A_phi`` finite-element magnetostatics solver (Tier 1, M5.1).

    Solves ``curl( (1/mu_0) curl A ) = J_phi`` in cylindrical ``(r, z)`` coordinates on a
    graded tensor-product triangular mesh using linear triangles and one-point centroid
    quadrature, with Dirichlet ``A = 0`` on the symmetry axis (``r = 0``) and far-field
    outer boundaries.

    Conventions
    -----------
    * Volume-winding solenoids (``inner_radius < outer_radius``) use fill factor ``1.0``
      (``J_phi = turns * current / ((outer_radius - inner_radius) * length)``), matching
      :mod:`magwave.benchmarks.checks`.
    * Thin solenoids (``inner_radius == outer_radius``) and axially magnetised cylinders
      (``K_phi = M = Br_z / MU0``) share the exact surface-current sheet load path.
    * Linear media only (``nu = 1 / MU0`` everywhere); ``permeable_solid`` manifest roles
      are rejected until Tier 1 stage 2 (M5.2).
    """

    def __init__(
        self,
        element_size: float | None = None,
        far_field_factor: float = 5.0,
        growth: float = 1.2,
        certified_accuracy: float = 1e-2,
        domain: tuple[float, float, float] | None = None,
        refinement_box: tuple[float, float, float, float] | None = None,
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

        self._element_size = None if element_size is None else float(element_size)
        self._far_field_factor = float(far_field_factor)
        self._growth = float(growth)
        self._certified_accuracy = float(certified_accuracy)
        self._domain = None if domain is None else tuple(float(v) for v in domain)
        self._refinement_box = (
            None if refinement_box is None else tuple(float(v) for v in refinement_box)
        )
        self._version = f"{__version__}+m5.1"

    @property
    def certified_accuracy(self) -> float:
        """Relative accuracy certified by this stage (default ``1e-2``, i.e. 1 %)."""
        return self._certified_accuracy

    def capabilities(self) -> Capabilities:
        return Capabilities(
            engine_id=ENGINE_ID,
            rung=RUNG,
            source_types=frozenset({"magnet", "coil"}),
            magnet_kinds=frozenset({"cylinder"}),
            coil_kinds=frozenset({"solenoid"}),
            quantities=frozenset({"B"}),
            force_methods=(),
            engine_version=self._version,
        )

    @property
    def method(self) -> str:
        return (
            "axisymmetric A_phi FEM, linear triangles, centroid quadrature, "
            f"Dirichlet A=0 on axis and at {self._far_field_factor:g}x far field, "
            "scipy.sparse.linalg.spsolve"
        )

    def validate(self, case: Case) -> list[Problem]:
        """Validate ``case`` against Tier 1 stage 1 capabilities, materials and accuracy."""
        problems: list[Problem] = []

        # 1. Components first.
        for index, component in enumerate(case.components):
            label = _label(component, index)
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
                    problems.append(
                        Problem(
                            W_DEMAG_IGNORED,
                            f"{case.name}: {label} names demag curve {component.demag_curve!r}; "
                            "fem2d stage 1 treats permanent magnets as rigidly polarised",
                            "warning",
                        )
                    )
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

        for index, probe in enumerate(case.probes):
            if probe.quantity not in self.capabilities().quantities:
                problems.append(
                    Problem(
                        E_UNSUPPORTED_QUANTITY,
                        f"{case.name}: probe #{index} asks for {probe.quantity!r}; fem2d "
                        "stage 1 samples B only (forces arrive later)",
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
                            "'permeable_solid' and requires a nonlinear B–H material law; this "
                            "solver is linear-mu only. Permeable materials arrive in Tier 1 "
                            "stage 2 (M5.2). Nothing was solved.",
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
        r_breaks: list[float] = []
        z_breaks: list[float] = []
        max_outer_radius = 0.0
        min_z1 = math.inf
        max_z2 = -math.inf
        l_max = 0.0

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

        h = self._element_size if self._element_size is not None else (l_max / 40.0)
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
                k_phi = component.br[2] / MU0
                load += load_sheet(mesh, r_out, (z1, z2), k_phi)

        return mesh, load, probes_rz

    def solve_with_report(self, case: Case) -> tuple[Result, SolveReport]:
        """Validate and solve ``case``, returning ``(Result, SolveReport)``."""
        warnings = raise_on_errors(ENGINE_ID, self.validate(case))
        mesh, load, probes_rz = self._build_mesh_and_load(case)
        a_phi, report = solve_axisymmetric(mesh, load, nu=1.0 / MU0)

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
            method=self.method,
            samples=tuple(samples),
            problems=tuple(warnings),
        )
        return result, report

    def solve(self, case: Case) -> Result:
        """Validate and solve ``case``; raise :class:`EngineError` if unsolvable."""
        result, _ = self.solve_with_report(case)
        return result

    def __repr__(self) -> str:
        return (
            f"Fem2dEngine(element_size={self._element_size!r}, "
            f"far_field_factor={self._far_field_factor!r}, growth={self._growth!r}, "
            f"certified_accuracy={self._certified_accuracy!r})"
        )


def __getattr__(name: str) -> Any:  # pragma: no cover
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
