"""Axisymmetric Tier 1 electromagnetic force evaluation and honest torque refusal (M5.4).

Two independent methods are implemented on top of :class:`~magwave.fem2d.engine.Fem2dEngine`:

1. **Maxwell stress tensor** (:func:`maxwell_force`) over a closed meridional contour in air
   enclosing only the target body:

   .. math::

       F_z = \\oint \\left[ \\frac{1}{\\mu_0} B_r B_z n_r
             + \\frac{1}{2 \\mu_0} (B_z^2 - B_r^2) n_z \\right] 2\\pi r \\, dl,

   integrated segment-by-segment with 2-point Gauss–Legendre quadrature via
   :func:`magwave.fem2d.field.sample_field`. Transverse Cartesian components (``F_x``, ``F_y``)
   and azimuthal force ``F_theta`` vanish identically by axisymmetric ``A_phi`` construction
   (``d/dphi = 0``, ``B_phi = 0``); no azimuthal sector check is performed.

2. **Coenergy / virtual work** (:func:`coenergy_force`) by central finite difference over two
   independent solves at axial displacements ``z +- delta``:

   * **With fixed-current coils:** differentiates magnetic coenergy
     :math:`W' = \\int \\mathbf{J}\\cdot\\mathbf{A}\\,dV - U` with :math:`F_z = +dW'/dz`
     (which reduces to :math:`\\frac{1}{2}\\int (B^2/\\mu_0)\\,dV` in linear air).
   * **Pure permanent magnet + permeable/air (no free currents):** differentiates total stored
     magnetic energy :math:`U = \\int w(\\mathbf{B})\\,dV` with :math:`F_z = -dU/dz`, since
     :math:`\\frac{1}{2}\\int \\mathbf{B}\\cdot\\mathbf{H}\\,dV \\equiv 0` when :math:`\\mathbf{J}=0`.

   Only **cell-aligned** virtual displacements (:math:`\\delta = k\\,h` for positive integer
   :math:`k`) are accepted. Shifting by an exact integer multiple of the axial refinement cell
   size :math:`h` keeps every shifted boundary on existing uniform grid lines, reproducing the
   identical mesh so the central difference evaluates one discrete energy function; sub-cell
   steps (:math:`\\delta \\neq k\\,h`) remesh and insert extra :math:`z`-lines, turning the
   finite-difference error into a step function of :math:`\\delta`. Sub-cell or non-aligned
   steps are refused explicitly with :data:`E_FORCE_DELTA_NOT_ALIGNED` without solving.

3. **Torque** (:func:`torque`) refuses explicitly with :data:`E_TORQUE_NOT_MEANINGFUL`: the
   axisymmetric ``A_phi`` formulation has no azimuthal asymmetry, so axial torque is
   identically zero by construction rather than a computed physical zero.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from ..cases import Case, Coil, Component, Magnet, Permeable, Probe
from ..engines.base import ConvergenceInfo, EngineError
from ..units import MU0
from .assembly import element_geometry, load_current_density, load_sheet
from .field import element_field, sample_field
from .solve import SolveReport

if TYPE_CHECKING:
    from .engine import Fem2dEngine, _FieldState

E_FORCE_TARGET_UNKNOWN = "E_FORCE_TARGET_UNKNOWN"
E_FORCE_TARGET_AMBIGUOUS = "E_FORCE_TARGET_AMBIGUOUS"
E_FORCE_CONTOUR_IN_MATERIAL = "E_FORCE_CONTOUR_IN_MATERIAL"
E_FORCE_CONTOUR_DOES_NOT_ENCLOSE = "E_FORCE_CONTOUR_DOES_NOT_ENCLOSE"
E_TORQUE_NOT_MEANINGFUL = "E_TORQUE_NOT_MEANINGFUL"
E_FORCE_UNSUPPORTED_CASE = "E_FORCE_UNSUPPORTED_CASE"
E_FORCE_DELTA_NOT_ALIGNED = "E_FORCE_DELTA_NOT_ALIGNED"

METHOD_MAXWELL = "maxwell_stress_tensor"
METHOD_COENERGY = "coenergy_virtual_work"

_AXIS_TOL = 1e-12
_GAUSS2_PTS = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
_RAY_GL_T, _RAY_GL_W = np.polynomial.legendre.leggauss(16)
_RAY_GL_U = 0.5 * (_RAY_GL_T + 1.0)
_RAY_GL_WU = 0.5 * _RAY_GL_W


class ForceError(ValueError):
    """Raised when a force or torque request cannot be meaningfully or validly evaluated."""

    def __init__(self, code: str, message: str) -> None:
        if not code or not isinstance(code, str):
            raise ValueError("ForceError.code must be a non-empty string")
        if not message or not isinstance(message, str):
            raise ValueError("ForceError.message must be a non-empty string")
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class ForceResult:
    """Result of a Tier 1 axisymmetric force calculation with full provenance."""

    target: str
    force_n: tuple[float, float, float]
    magnitude_n: float
    method: str
    method_detail: str
    contour: tuple[float, float, float] | None
    mesh_nodes: int
    mesh_elements: int
    element_size: float
    convergence: ConvergenceInfo | None
    coenergy_j: float | None = None
    delta_m: float | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeltaStep:
    """One accepted cell-aligned virtual-work solve in a delta convergence ladder."""

    label: str
    delta_m: float
    delta_over_h: float
    force_n: float
    change_vs_prev_pct: float | None
    mesh_nodes_plus: int
    mesh_elements_plus: int
    mesh_nodes_minus: int
    mesh_elements_minus: int
    gap_cells_plus: int | None
    gap_cells_minus: int | None


@dataclass(frozen=True)
class DeltaStudyResult:
    """Result of an aligned finite-difference virtual-work step convergence study."""

    target: str
    h_m: float
    base_mesh_nodes: int
    base_mesh_elements: int
    axial_gap_m: float | None
    expected_gap_cell_sum: int | None
    steps: tuple[DeltaStep, ...]
    refused: tuple[tuple[float, str], ...]
    p_per_pair: tuple[float | None, ...]
    p_fit: float | None
    monotonic: bool
    asymptotic: bool
    f_extrapolated_n: float | None
    step_uncertainty_n: float
    step_uncertainty_pct: float
    peak_to_peak_n: float
    peak_to_peak_pct: float
    mesh_counts_invariant: bool
    gap_cell_sum_invariant: bool | None
    alignment_invariant_holds: bool
    notes: tuple[str, ...]


def _component_label(component: Component, index: int) -> str:
    return component.name or f"#{index}"


def _resolve_target(case: Case, target: str | None) -> tuple[int, Magnet | Coil, str]:
    """Resolve the active ``Magnet`` or ``Coil`` force target without mutating ``case``."""
    active_candidates: list[tuple[int, Magnet | Coil, str]] = [
        (idx, comp, _component_label(comp, idx))
        for idx, comp in enumerate(case.components)
        if isinstance(comp, Magnet | Coil)
    ]

    if target is None:
        if len(active_candidates) == 1:
            return active_candidates[0]
        if not active_candidates:
            raise ForceError(
                E_FORCE_TARGET_UNKNOWN,
                f"{case.name}: case has no Magnet or Coil force target (a passive Permeable body "
                "is never a force target on its own; its force is the negative of the active "
                "source's force); nothing was solved.",
            )
        names = ", ".join(label for _, _, label in active_candidates)
        raise ForceError(
            E_FORCE_TARGET_AMBIGUOUS,
            f"{case.name}: target is omitted and case has {len(active_candidates)} candidate "
            f"Magnet/Coil sources ({names}); pass target=<name> explicitly.",
        )

    if not isinstance(target, str) or not target:
        raise ForceError(
            E_FORCE_TARGET_UNKNOWN,
            f"{case.name}: target must be a non-empty component name, got {target!r}.",
        )

    for idx, comp in enumerate(case.components):
        if comp.name == target:
            if isinstance(comp, Permeable):
                raise ForceError(
                    E_FORCE_TARGET_UNKNOWN,
                    f"{case.name}: target {target!r} is a passive Permeable body; a Permeable "
                    "body is never a force target on its own (the force on steel comes from the "
                    "field, and a passive body's force is the negative of the active magnet/coil "
                    "force); pass the Magnet or Coil name instead.",
                )
            if not isinstance(comp, Magnet | Coil):
                raise ForceError(
                    E_FORCE_TARGET_UNKNOWN,
                    f"{case.name}: target {target!r} has type {type(comp).__name__}; only active "
                    "Magnet or Coil components can be force targets.",
                )
            return idx, comp, comp.name

    available = ", ".join(_component_label(c, i) for i, c in enumerate(case.components))
    raise ForceError(
        E_FORCE_TARGET_UNKNOWN,
        f"{case.name}: unknown target {target!r}; components in case: {available}.",
    )


def _prepare_solve_case(case: Case, engine: Fem2dEngine) -> Case:
    """Return a non-mutating copy of ``case`` suitable for field solves underlying force."""
    if case.tolerance is not None and case.tolerance < engine.certified_accuracy:
        declared_pct = 100.0 * case.tolerance
        certified_pct = 100.0 * engine.certified_accuracy
        raise ForceError(
            E_FORCE_UNSUPPORTED_CASE,
            f"{case.name}: declared tolerance {declared_pct:g}% ({case.tolerance:.2e}) "
            f"is tighter than fem2d's certified accuracy {certified_pct:g}% "
            f"({engine.certified_accuracy:.2e}); Tier 1 force is not claimed at a tighter "
            "tolerance than that, so either relax the case tolerance or request the force "
            "through a case that does not declare a tighter B tolerance; nothing was solved.",
        )

    has_force_probe = any(probe.quantity == "F" for probe in case.probes)
    solve_case = (
        replace(
            case,
            probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
        )
        if has_force_probe
        else case
    )

    errors = [problem for problem in engine.validate(solve_case) if problem.is_error]
    if errors:
        detail = "; ".join(str(problem) for problem in errors)
        raise ForceError(
            E_FORCE_UNSUPPORTED_CASE,
            f"{case.name}: case is not solvable by Tier 1 axisymmetric FEM ({detail}); "
            "nothing was solved.",
        )
    return solve_case


def _target_outer_radius(comp: Magnet | Coil) -> float:
    if isinstance(comp, Coil):
        return float(comp.dims["outer_radius"])
    return float(comp.dims["radius"])


def _component_rz_extents(
    comp: Component,
) -> tuple[float, float, float, float, str] | None:
    """Return ``(r_in, r_out, z_lo, z_hi, shape_tag)`` in ``(r, z)`` or ``None``."""
    if isinstance(comp, Coil):
        if comp.kind != "solenoid":
            return None
        r_in = float(comp.dims["inner_radius"])
        r_out = float(comp.dims["outer_radius"])
        half_len = 0.5 * float(comp.dims["length"])
        zc = float(comp.center[2])
        tag = "sheet" if r_in == r_out else "rect"
        return r_in, r_out, zc - half_len, zc + half_len, tag
    if isinstance(comp, Magnet):
        r_out = float(comp.dims["radius"])
        half_len = 0.5 * float(comp.dims["length"])
        zc = float(comp.center[2])
        return 0.0, r_out, zc - half_len, zc + half_len, "rect"
    if isinstance(comp, Permeable):
        rc = math.hypot(comp.center[0], comp.center[1])
        zc = float(comp.center[2])
        rad = float(comp.dims["radius"])
        if comp.kind == "sphere":
            return max(0.0, rc - rad), rc + rad, zc - rad, zc + rad, "sphere"
        half_len = 0.5 * float(comp.dims["length"])
        return max(0.0, rc - rad), rc + rad, zc - half_len, zc + half_len, "rect"
    return None


def _force_engine_for_case(
    solve_case: Case,
    engine: Fem2dEngine,
    target_comp: Magnet | Coil,
) -> tuple[Fem2dEngine, float]:
    """Create a force-configured ``Fem2dEngine`` with fixed physical refinement bounds.

    Unlike ``Fem2dEngine``'s default probe-oriented ``2*h`` refinement padding (which shrinks
    with ``h`` and leaves exterior air unrefined under mesh refinement), force integration
    requires uniform refinement in a fixed physical neighbourhood around the sources and target
    across ``h``, contours, and virtual displacements ``+-delta``.
    """
    from .engine import Fem2dEngine as EngineCls

    max_outer_r = 0.0
    min_z1 = math.inf
    max_z2 = -math.inf
    l_max = 0.0
    min_sphere_radius = math.inf

    for component in solve_case.components:
        ext = _component_rz_extents(component)
        if ext is None:
            continue
        _, r_out, z1, z2, tag = ext
        max_outer_r = max(max_outer_r, r_out)
        min_z1 = min(min_z1, z1)
        max_z2 = max(max_z2, z2)
        if tag == "sphere":
            rad = float(component.dims["radius"])
            l_max = max(l_max, r_out, 2.0 * rad)
            min_sphere_radius = min(min_sphere_radius, rad)
        else:
            l_max = max(l_max, r_out, z2 - z1)

    if engine._element_size is not None:
        h_eff = float(engine._element_size)
    else:
        h_eff = l_max / 40.0
        if math.isfinite(min_sphere_radius):
            h_eff = min(h_eff, min_sphere_radius / 40.0)

    if engine._domain is not None:
        domain = engine._domain
    else:
        pad = engine._far_field_factor * l_max
        domain = (max_outer_r + pad, min_z1 - pad, max_z2 + pad)

    r_max, z_min, z_max = domain
    if engine._refinement_box is not None:
        ref_box = engine._refinement_box
    else:
        margin = max(0.5 * _target_outer_radius(target_comp), 2.0 * h_eff)
        ref_box = (
            0.0,
            min(r_max, max_outer_r + margin),
            max(z_min, min_z1 - margin),
            min(z_max, max_z2 + margin),
        )

    configured = EngineCls(
        element_size=h_eff,
        far_field_factor=engine._far_field_factor,
        growth=engine._growth,
        certified_accuracy=engine._certified_accuracy,
        domain=domain,
        refinement_box=ref_box,
        material_model=engine._material_model,
        tol=engine._tol,
        max_iterations=engine._max_iterations,
        magnet_curves=engine._magnet_curves,
    )
    return configured, h_eff


def _segment_intersects_component(
    comp: Component,
    *,
    is_horizontal: bool,
    fixed: float,
    start: float,
    end: float,
) -> bool:
    """Return ``True`` if a horizontal/vertical air segment intersects ``comp`` geometry."""
    ext = _component_rz_extents(comp)
    if ext is None:
        return False
    r0_c, r1_c, z0_c, z1_c, tag = ext

    if tag == "sphere":
        assert isinstance(comp, Permeable)
        rc = math.hypot(comp.center[0], comp.center[1])
        zc = float(comp.center[2])
        rad = float(comp.dims["radius"])
        if is_horizontal:
            dr = max(start - rc, 0.0, rc - end)
            dz = fixed - zc
        else:
            dr = fixed - rc
            dz = max(start - zc, 0.0, zc - end)
        return math.hypot(dr, dz) <= rad + _AXIS_TOL

    if is_horizontal:
        return (z0_c - _AXIS_TOL <= fixed <= z1_c + _AXIS_TOL) and (
            min(end, r1_c) >= max(start, r0_c) - _AXIS_TOL
        )
    return (r0_c - _AXIS_TOL <= fixed <= r1_c + _AXIS_TOL) and (
        min(end, z1_c) >= max(start, z0_c) - _AXIS_TOL
    )


def _component_enclosed_by_box(comp: Component, r_c: float, z_lo: float, z_hi: float) -> bool:
    """Return ``True`` if ``comp`` lies inside or overlaps ``[0, r_c] x [z_lo, z_hi]``."""
    ext = _component_rz_extents(comp)
    if ext is None:
        return False
    r0_c, r1_c, z0_c, z1_c, tag = ext
    if tag == "sphere":
        assert isinstance(comp, Permeable)
        rc = math.hypot(comp.center[0], comp.center[1])
        zc = float(comp.center[2])
        rad = float(comp.dims["radius"])
        dr = max(0.0 - rc, 0.0, rc - r_c)
        dz = max(z_lo - zc, 0.0, zc - z_hi)
        return math.hypot(dr, dz) <= rad + _AXIS_TOL
    return (min(r_c, r1_c) >= max(0.0, r0_c) - _AXIS_TOL) and (
        min(z_hi, z1_c) >= max(z_lo, z0_c) - _AXIS_TOL
    )


def _check_contour_against_geometry(
    case: Case,
    target_idx: int,
    target_comp: Magnet | Coil,
    contour: tuple[float, float, float],
    h_eff: float,
    domain: tuple[float, float, float],
    *,
    require_radial_margin: bool,
    stage: str,
) -> None:
    """Validate contour clearance, air placement, and single-target enclosure."""
    r_c, z_lo, z_hi = contour
    r_max, z_min, z_max = domain
    if not (math.isfinite(r_c) and math.isfinite(z_lo) and math.isfinite(z_hi)):
        raise ForceError(
            E_FORCE_CONTOUR_DOES_NOT_ENCLOSE,
            f"{case.name}: contour coordinates {contour!r} must be finite.",
        )
    if r_c <= 0.0 or z_hi <= z_lo:
        raise ForceError(
            E_FORCE_CONTOUR_DOES_NOT_ENCLOSE,
            f"{case.name}: contour {contour!r} must satisfy r_c > 0 and z_hi > z_lo.",
        )
    if r_c >= r_max - _AXIS_TOL or z_lo <= z_min + _AXIS_TOL or z_hi >= z_max - _AXIS_TOL:
        raise ForceError(
            E_FORCE_CONTOUR_DOES_NOT_ENCLOSE,
            f"{case.name}: {stage} contour {contour!r} extends outside interior of FEM domain "
            f"[0, {r_max:g}] x [{z_min:g}, {z_max:g}].",
        )

    # 1. Contour air segments (bottom, outer wall, top) must not touch or cross any body.
    segments = (
        ("bottom", True, z_lo, 0.0, r_c),
        ("outer", False, r_c, z_lo, z_hi),
        ("top", True, z_hi, 0.0, r_c),
    )
    for idx, comp in enumerate(case.components):
        label = _component_label(comp, idx)
        for seg_name, is_h, fixed, start, end in segments:
            if _segment_intersects_component(
                comp, is_horizontal=is_h, fixed=fixed, start=start, end=end
            ):
                raise ForceError(
                    E_FORCE_CONTOUR_IN_MATERIAL,
                    f"{case.name}: {stage} contour {contour!r} {seg_name} segment intersects "
                    f"component {label!r} ({type(comp).__name__}); Maxwell stress contour must "
                    "lie strictly in air outside every magnet, coil, and permeable body.",
                )

    # 2. Must strictly enclose target (and satisfy r_c > target_r + element_size pre-snap).
    ext = _component_rz_extents(target_comp)
    assert ext is not None
    _, target_r, target_z0, target_z1, _ = ext
    if z_lo >= target_z0 - _AXIS_TOL or z_hi <= target_z1 + _AXIS_TOL:
        raise ForceError(
            E_FORCE_CONTOUR_DOES_NOT_ENCLOSE,
            f"{case.name}: {stage} contour z-span [{z_lo:g}, {z_hi:g}] does not strictly "
            f"enclose target axial extent [{target_z0:g}, {target_z1:g}].",
        )
    if require_radial_margin:
        min_r = target_r + h_eff
        if r_c <= min_r + _AXIS_TOL:
            raise ForceError(
                E_FORCE_CONTOUR_DOES_NOT_ENCLOSE,
                f"{case.name}: contour radius r_c={r_c:g} m must be strictly greater than "
                f"target radius + one element ({target_r:g} + {h_eff:g} = {min_r:g} m).",
            )
    elif r_c <= target_r + _AXIS_TOL:
        raise ForceError(
            E_FORCE_CONTOUR_DOES_NOT_ENCLOSE,
            f"{case.name}: {stage} contour radius r_c={r_c:g} m does not enclose target "
            f"radius {target_r:g} m.",
        )

    # 3. Must enclose no other body besides target.
    for idx, comp in enumerate(case.components):
        if idx == target_idx:
            continue
        if _component_enclosed_by_box(comp, r_c, z_lo, z_hi):
            label = _component_label(comp, idx)
            raise ForceError(
                E_FORCE_CONTOUR_DOES_NOT_ENCLOSE,
                f"{case.name}: {stage} contour {contour!r} encloses non-target component "
                f"{label!r}; contour must enclose only the target body.",
            )


def _default_contour(
    case: Case,
    target_idx: int,
    target_comp: Magnet | Coil,
    h_eff: float,
) -> tuple[float, float, float]:
    """Place a default air-gap rectangular contour enclosing only ``target_comp``."""
    ext = _component_rz_extents(target_comp)
    assert ext is not None
    _, target_r, target_z0, target_z1, _ = ext
    default_pad = max(0.25 * target_r, 2.0 * h_eff)

    above_z0s: list[float] = []
    below_z1s: list[float] = []
    outer_r0s: list[float] = []

    for idx, comp in enumerate(case.components):
        if idx == target_idx:
            continue
        o_ext = _component_rz_extents(comp)
        if o_ext is None:
            continue
        o_r0, _, o_z0, o_z1, _ = o_ext
        if o_z0 > target_z1 + _AXIS_TOL and o_r0 <= target_r + default_pad:
            above_z0s.append(o_z0)
        if o_z1 < target_z0 - _AXIS_TOL and o_r0 <= target_r + default_pad:
            below_z1s.append(o_z1)

    z_hi = (
        0.5 * (target_z1 + min(above_z0s))
        if above_z0s and (min(above_z0s) - target_z1) < 2.0 * default_pad
        else target_z1 + default_pad
    )
    z_lo = (
        0.5 * (max(below_z1s) + target_z0)
        if below_z1s and (target_z0 - max(below_z1s)) < 2.0 * default_pad
        else target_z0 - default_pad
    )

    for idx, comp in enumerate(case.components):
        if idx == target_idx:
            continue
        o_ext = _component_rz_extents(comp)
        if o_ext is None:
            continue
        o_r0, _, o_z0, o_z1, _ = o_ext
        if min(z_hi, o_z1) >= max(z_lo, o_z0) - _AXIS_TOL and o_r0 > target_r + _AXIS_TOL:
            outer_r0s.append(o_r0)

    r_c = (
        0.5 * (target_r + min(outer_r0s))
        if outer_r0s and (min(outer_r0s) - target_r) < 2.0 * default_pad
        else target_r + default_pad
    )
    return float(r_c), float(z_lo), float(z_hi)


def _point_hits_any_material_on_radial_span(case: Case, z_val: float, r_max_seg: float) -> bool:
    return any(
        _segment_intersects_component(
            comp, is_horizontal=True, fixed=z_val, start=0.0, end=r_max_seg
        )
        for comp in case.components
    )


def _snap_axis_value(
    req: float,
    axis: np.ndarray,
    *,
    is_interior_boundary: Callable[[float], bool] | None = None,
) -> tuple[float, float, bool]:
    """Snap ``req`` to nearest ``axis`` line, or single-element gap midline if wall-bounded."""
    idx_near = int(np.argmin(np.abs(axis - req)))
    snapped = float(axis[idx_near])

    if is_interior_boundary is not None and len(axis) >= 1:
        pos = int(np.searchsorted(axis, req, side="right")) - 1
        if 0 <= pos < len(axis) - 1:
            a0 = float(axis[pos])
            a1 = float(axis[pos + 1])
            width = a1 - a0
            mid = 0.5 * (a0 + a1)
            if (
                width > _AXIS_TOL
                and abs(req - mid) <= 0.15 * width
                and is_interior_boundary(a0)
                and is_interior_boundary(a1)
            ):
                return mid, mid - req, True

    return snapped, snapped - req, False


def _snap_contour_to_mesh(
    case: Case,
    r_req: float,
    z_lo_req: float,
    z_hi_req: float,
    r_axis: np.ndarray,
    z_axis: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[str, ...]]:
    """Snap contour lines to nearest mesh axis lines and return provenance notes."""
    r_snap, dr_snap, _ = _snap_axis_value(r_req, r_axis)
    notes: list[str] = []

    def _z_hits_wall(z_val: float) -> bool:
        return _point_hits_any_material_on_radial_span(case, z_val, r_snap)

    z_lo_snap, dz_lo_snap, lo_mid = _snap_axis_value(
        z_lo_req, z_axis, is_interior_boundary=_z_hits_wall
    )
    z_hi_snap, dz_hi_snap, hi_mid = _snap_axis_value(
        z_hi_req, z_axis, is_interior_boundary=_z_hits_wall
    )

    notes.append(
        f"requested contour=({r_req:.6g}, {z_lo_req:.6g}, {z_hi_req:.6g}) m; "
        f"snapped contour=({r_snap:.6g}, {z_lo_snap:.6g}, {z_hi_snap:.6g}) m; "
        f"snap delta=(dr={dr_snap:+.3e}, dz_lo={dz_lo_snap:+.3e}, dz_hi={dz_hi_snap:+.3e}) m"
    )
    if lo_mid:
        notes.append(
            f"z_lo={z_lo_snap:.6g} m placed on single-element air-gap cell midline (no interior "
            "z grid line between adjacent material boundaries)"
        )
    if hi_mid:
        notes.append(
            f"z_hi={z_hi_snap:.6g} m placed on single-element air-gap cell midline (no interior "
            "z grid line between adjacent material boundaries)"
        )
    return (r_snap, z_lo_snap, z_hi_snap), tuple(notes)


def maxwell_force(
    case: Case,
    engine: Fem2dEngine,
    *,
    target: str | None = None,
    contour: tuple[float, float, float] | None = None,
) -> ForceResult:
    """Compute axial electromagnetic force on ``target`` via the Maxwell stress tensor.

    Integrates the axisymmetric Maxwell stress tensor over the closed meridional contour
    ``(0, z_lo) -> (r_c, z_lo) -> (r_c, z_hi) -> (0, z_hi) -> (0, z_lo)`` lying strictly
    in air and enclosing only ``target``:

    .. math::

        F_z = \\oint \\left[ \\frac{1}{\\mu_0} B_r B_z n_r
              + \\frac{1}{2\\mu_0} (B_z^2 - B_r^2) n_z \\right] 2\\pi r \\, dl
    """
    target_idx, target_comp, target_label = _resolve_target(case, target)
    solve_case = _prepare_solve_case(case, engine)
    force_engine, h_eff = _force_engine_for_case(solve_case, engine, target_comp)

    try:
        result, report, state = force_engine._solve_core(solve_case)
    except EngineError as exc:
        raise ForceError(E_FORCE_UNSUPPORTED_CASE, str(exc)) from exc

    mesh = state.mesh
    domain = (float(mesh.r_axis[-1]), float(mesh.z_axis[0]), float(mesh.z_axis[-1]))

    if contour is None:
        r_req, z_lo_req, z_hi_req = _default_contour(solve_case, target_idx, target_comp, h_eff)
    else:
        if len(contour) != 3:
            raise ForceError(
                E_FORCE_CONTOUR_DOES_NOT_ENCLOSE,
                f"{case.name}: contour must be (r_c, z_lo, z_hi), got {contour!r}.",
            )
        r_req, z_lo_req, z_hi_req = (float(contour[0]), float(contour[1]), float(contour[2]))

    _check_contour_against_geometry(
        solve_case,
        target_idx,
        target_comp,
        (r_req, z_lo_req, z_hi_req),
        h_eff,
        domain,
        require_radial_margin=True,
        stage="requested",
    )

    snapped, snap_notes = _snap_contour_to_mesh(
        solve_case, r_req, z_lo_req, z_hi_req, mesh.r_axis, mesh.z_axis
    )
    _check_contour_against_geometry(
        solve_case,
        target_idx,
        target_comp,
        snapped,
        h_eff,
        domain,
        require_radial_margin=False,
        stage="snapped",
    )

    r_c, z_lo, z_hi = snapped
    r_breaks = np.unique(
        np.concatenate(
            (
                np.array([0.0, r_c], dtype=np.float64),
                mesh.r_axis[(mesh.r_axis > 0.0) & (mesh.r_axis < r_c)],
            )
        )
    )
    z_breaks = np.unique(
        np.concatenate(
            (
                np.array([z_lo, z_hi], dtype=np.float64),
                mesh.z_axis[(mesh.z_axis > z_lo) & (mesh.z_axis < z_hi)],
            )
        )
    )

    # Closed meridional loop: bottom (+r), outer wall (+z), top (-r), axis (-z, dS = 0).
    loop_segments = (
        ("bottom", r_breaks, z_lo, True, 0.0, -1.0),
        ("outer", z_breaks, r_c, False, 1.0, 0.0),
        ("top", r_breaks[::-1], z_hi, True, 0.0, 1.0),
        ("axis", z_breaks[::-1], 0.0, False, -1.0, 0.0),
    )

    f_z_total = 0.0
    radial_force_2pi = 0.0
    seg_values: dict[str, float] = {}

    for seg_name, coords, fixed, is_r_varying, nr_n, nz_n in loop_segments:
        if not is_r_varying and fixed <= _AXIS_TOL:
            # On the symmetry axis r = 0, dS = 2*pi*r*dl = 0 identically; kept for bookkeeping.
            seg_values[seg_name] = 0.0
            continue

        pts_list: list[tuple[float, float]] = []
        weights_list: list[float] = []
        for u0, u1 in zip(coords[:-1], coords[1:], strict=False):
            length = abs(float(u1 - u0))
            if length <= _AXIS_TOL:
                continue
            mid = 0.5 * (float(u0) + float(u1))
            half = 0.5 * length
            sign_dir = 1.0 if u1 >= u0 else -1.0
            for xi in _GAUSS2_PTS:
                coord = mid + sign_dir * half * xi
                if is_r_varying:
                    pts_list.append((coord, fixed))
                    weights_list.append(half * 2.0 * math.pi * coord)
                else:
                    pts_list.append((fixed, coord))
                    weights_list.append(half * 2.0 * math.pi * fixed)

        pts_arr = np.asarray(pts_list, dtype=np.float64)
        w_arr = np.asarray(weights_list, dtype=np.float64)
        b_vals = sample_field(mesh, state.a_phi, pts_arr)
        b_r = b_vals[:, 0]
        b_z = b_vals[:, 1]

        t_z = ((b_r * b_z) / MU0) * nr_n + ((b_z * b_z - b_r * b_r) / (2.0 * MU0)) * nz_n
        t_r = ((b_r * b_r - b_z * b_z) / (2.0 * MU0)) * nr_n + ((b_r * b_z) / MU0) * nz_n

        seg_fz = float(np.sum(t_z * w_arr))
        seg_values[seg_name] = seg_fz
        f_z_total += seg_fz
        radial_force_2pi += float(np.sum(t_r * w_arr))

    # F_x = F_y = F_theta = 0 hold identically by axisymmetric A_phi construction
    # (d/dphi = 0, B_phi = 0). The meridional radial traction integral radial_force_2pi
    # is NOT asserted to vanish. It is not a Cartesian component of the closed-surface
    # flux: the x and y components of T.n carry cos(phi) and sin(phi), which integrate to
    # zero over the azimuth, which is precisely why F_x = F_y = 0 identically and why the
    # removed 8-sector mean(cos(phis)) check could never fire (mean = -5.6e-17). This
    # azimuthally summed radial traction is therefore free to be non-zero; measured
    # |F_r|/|F_z| on shipped force cases: coaxial_loops 0.828..1.055; magnet_plate
    # CONTOUR_A 0.461..0.468, CONTOUR_B 0.360. An empty vacuum box (r=0.02,
    # z=0.030..0.040), where the true force is zero and F_z=8.6e-05 N, returns
    # radial_force_2pi=-5.0e-02 N. A tolerance gate on this quantity would fire on
    # every valid case, so none is applied.
    notes = (
        *snap_notes,
        f"segment F_z (N): bottom={seg_values['bottom']:+.6e}, outer={seg_values['outer']:+.6e}, "
        f"top={seg_values['top']:+.6e}, axis={seg_values['axis']:.1f}",
        "transverse force: Fx=0 N, Fy=0 N, F_theta=0 N by axisymmetric A_phi construction "
        "(d/dphi=0, B_phi=0; no azimuthal sector check); meridional radial traction "
        f"integral={radial_force_2pi:+.6e} N",
    )
    method_detail = (
        "Maxwell stress tensor F_z = closed_integral[(B_r*B_z*n_r + 0.5*(B_z^2 - B_r^2)*n_z)/mu0 "
        "2*pi*r dl] (2-pt Gauss via sample_field); "
        f"snapped contour=(r_c={r_c:.6g}, z_lo={z_lo:.6g}, z_hi={z_hi:.6g}) m"
    )
    return ForceResult(
        target=target_label,
        force_n=(0.0, 0.0, float(f_z_total)),
        magnitude_n=abs(float(f_z_total)),
        method=METHOD_MAXWELL,
        method_detail=method_detail,
        contour=snapped,
        mesh_nodes=report.mesh.nodes,
        mesh_elements=report.mesh.elements,
        element_size=h_eff,
        convergence=result.convergence,
        notes=notes,
    )


def _pm_demag_energy_density(
    curve,
    b_r: np.ndarray,
    b_z: np.ndarray,
    sign: float,
) -> np.ndarray:
    """Stored magnetic energy density ``int_{B0}^{B} H . dB'`` inside a demag-curve magnet."""
    if curve._linear_slope is not None and curve._linear_intercept is not None:
        s = float(curve._linear_slope)
        c = float(curve._linear_intercept)
        return 0.5 * s * (b_r * b_r + b_z * b_z) + (sign * c) * b_z + (c * c) / (2.0 * s)

    b0_z = sign * curve.br()
    d_br = b_r
    d_bz = b_z - b0_z
    br_u = d_br[:, None] * _RAY_GL_U[None, :]
    bz_u = b0_z + d_bz[:, None] * _RAY_GL_U[None, :]
    bmag_u = np.clip(np.hypot(br_u, bz_u), curve.b_min, curve.b_max)
    s_u, _, c_u, _ = curve.slope_and_offset(bmag_u)
    integrand = (s_u * br_u) * d_br[:, None] + (s_u * bz_u + sign * c_u) * d_bz[:, None]
    return np.sum(integrand * _RAY_GL_WU[None, :], axis=1)


def _coil_only_load(mesh, case: Case) -> np.ndarray:
    """Assemble nodal RHS load from free-current ``Coil`` components only (for J . A)."""
    f_coil = np.zeros(int(mesh.nodes.shape[0]), dtype=np.float64)
    for comp in case.components:
        if isinstance(comp, Coil):
            r_in = float(comp.dims["inner_radius"])
            r_out = float(comp.dims["outer_radius"])
            length = float(comp.dims["length"])
            z1 = float(comp.center[2]) - 0.5 * length
            z2 = float(comp.center[2]) + 0.5 * length
            ampere_turns = float(comp.turns) * float(comp.current)
            if r_out > r_in:
                j_phi = ampere_turns / ((r_out - r_in) * length)
                f_coil += load_current_density(mesh, (r_in, r_out, z1, z2), j_phi)
            else:
                k_phi = ampere_turns / length
                f_coil += load_sheet(mesh, r_in, (z1, z2), k_phi)
    return f_coil


def _total_magnetic_energy_and_coenergy(
    solve_case: Case,
    engine: Fem2dEngine,
    state: _FieldState,
) -> tuple[float, float, bool]:
    """Return ``(U_total_J, W_coenergy_J, has_free_currents)`` from solved element fields."""
    mesh = state.mesh
    area, r_c, z_c, _, _ = element_geometry(mesh)
    vol = 2.0 * math.pi * r_c * area
    b_elem = element_field(mesh, state.a_phi)
    b_r = b_elem[:, 0]
    b_z = b_elem[:, 1]
    b_sq = b_r * b_r + b_z * b_z
    b_mag = np.sqrt(b_sq)

    used = np.zeros(int(mesh.elements.shape[0]), dtype=bool)
    u_total = 0.0

    for _, mask, mat, _ in state.assignments:
        if not np.any(mask):
            continue
        used |= mask
        if engine.material_model == "nonlinear":
            w_st = mat.bh_curve().magnetic_energy_density(b_mag[mask])
            u_total += float(np.sum(w_st * vol[mask]))
        else:
            u_total += float(np.sum(0.5 * mat.nu() * b_sq[mask] * vol[mask]))

    for _, _, mask, curve, sign, _ in state.demag_assignments:
        if not np.any(mask):
            continue
        used |= mask
        w_pm = _pm_demag_energy_density(curve, b_r[mask], b_z[mask], sign)
        u_total += float(np.sum(w_pm * vol[mask]))

    for comp in solve_case.components:
        if isinstance(comp, Magnet) and comp.demag_curve is None:
            r_out = float(comp.dims["radius"])
            half_len = 0.5 * float(comp.dims["length"])
            zc = float(comp.center[2])
            mask = (r_c <= r_out) & (np.abs(z_c - zc) <= half_len)
            if not np.any(mask):
                continue
            used |= mask
            br_z = float(comp.br[2])
            w_m = (0.5 / MU0) * (b_r[mask] ** 2 + (b_z[mask] - br_z) ** 2)
            u_total += float(np.sum(w_m * vol[mask]))

    air_mask = ~used
    u_total += float(np.sum((0.5 / MU0) * b_sq[air_mask] * vol[air_mask]))

    has_coils = any(isinstance(c, Coil) for c in solve_case.components)
    if not has_coils:
        return u_total, u_total, False

    f_coil = _coil_only_load(mesh, solve_case)
    j_dot_a = 2.0 * math.pi * float(np.dot(f_coil, state.a_phi))
    w_coenergy = j_dot_a - u_total
    return u_total, w_coenergy, True


def magnetic_coenergy(
    case: Case,
    engine: Fem2dEngine,
) -> tuple[float, SolveReport, ConvergenceInfo | None]:
    """Compute total magnetic coenergy (or stored energy if no coils) for ``case`` in joules."""
    solve_case = _prepare_solve_case(case, engine)
    active = [c for c in solve_case.components if isinstance(c, Magnet | Coil)]
    if active:
        force_engine, _ = _force_engine_for_case(solve_case, engine, active[0])
    else:  # pragma: no cover - Case requires at least one source
        force_engine = engine
    try:
        result, report, state = force_engine._solve_core(solve_case)
    except EngineError as exc:
        raise ForceError(E_FORCE_UNSUPPORTED_CASE, str(exc)) from exc
    u_total, w_co, has_coils = _total_magnetic_energy_and_coenergy(solve_case, force_engine, state)
    return (w_co if has_coils else u_total), report, result.convergence


def _nearest_axial_gap_face(
    case: Case,
    target_idx: int,
    target_comp: Magnet | Coil,
) -> tuple[float, str, float] | None:
    """Return ``(gap_m, side, other_face_z)`` to the nearest axially blocking body, or ``None``."""
    t_ext = _component_rz_extents(target_comp)
    assert t_ext is not None
    t_r0, t_r1, t_z0, t_z1, _ = t_ext
    best: tuple[float, str, float] | None = None

    for idx, comp in enumerate(case.components):
        if idx == target_idx:
            continue
        o_ext = _component_rz_extents(comp)
        if o_ext is None:
            continue
        o_r0, o_r1, o_z0, o_z1, tag = o_ext
        candidate: tuple[float, str, float] | None = None
        if tag == "sphere":
            assert isinstance(comp, Permeable)
            rc = math.hypot(comp.center[0], comp.center[1])
            zc = float(comp.center[2])
            rad = float(comp.dims["radius"])
            dr = max(t_r0 - rc, 0.0, rc - t_r1)
            if dr < rad - _AXIS_TOL:
                z_clearance = math.sqrt(max(0.0, rad * rad - dr * dr))
                if t_z0 >= zc + z_clearance - _AXIS_TOL:
                    candidate = (
                        max(0.0, t_z0 - (zc + z_clearance)),
                        "below",
                        zc + z_clearance,
                    )
                elif t_z1 <= zc - z_clearance + _AXIS_TOL:
                    candidate = (
                        max(0.0, (zc - z_clearance) - t_z1),
                        "above",
                        zc - z_clearance,
                    )
                else:
                    candidate = (0.0, "overlap", zc)
        elif min(t_r1, o_r1) - max(t_r0, o_r0) >= -_AXIS_TOL:
            if o_z0 >= t_z1 - _AXIS_TOL:
                candidate = (max(0.0, o_z0 - t_z1), "above", o_z0)
            elif t_z0 >= o_z1 - _AXIS_TOL:
                candidate = (max(0.0, t_z0 - o_z1), "below", o_z1)
            else:
                candidate = (0.0, "overlap", o_z0)

        if candidate is not None and (best is None or candidate[0] < best[0]):
            best = candidate
    return best


def _max_safe_axial_shift(case: Case, target_idx: int, target_comp: Magnet | Coil) -> float:
    """Return the largest axial shift ``|dz|`` before ``target_comp`` touches another body."""
    nearest = _nearest_axial_gap_face(case, target_idx, target_comp)
    return math.inf if nearest is None else nearest[0]


def _count_gap_cells(z_axis: np.ndarray, z_lo: float, z_hi: float) -> int:
    """Return the count of ``z_axis`` cells spanning the open axial gap ``(z_lo, z_hi)``."""
    return int(np.sum((z_axis > z_lo + _AXIS_TOL) & (z_axis < z_hi - _AXIS_TOL))) + 1


def _aligned_cell_multiple(delta: float, h: float) -> int | None:
    """Return integer ``k >= 1`` iff ``delta`` equals ``k * h`` within ``1e-9`` relative tol."""
    if not (math.isfinite(delta) and math.isfinite(h)) or delta <= 0.0 or h <= 0.0:
        return None
    k_float = delta / h
    k_int = round(k_float)
    if k_int >= 1 and abs(k_float - k_int) <= 1e-9 * max(1.0, abs(k_float)):
        return int(k_int)
    return None


def _resolve_aligned_delta(
    case: Case,
    target_label: str,
    h_eff: float,
    max_shift: float,
    delta: float | None,
) -> tuple[float, int]:
    """Validate and resolve a cell-aligned virtual displacement ``(delta_m, k_mult)``."""
    if delta is None:
        if math.isfinite(max_shift) and h_eff >= max_shift - _AXIS_TOL:
            raise ForceError(
                E_FORCE_UNSUPPORTED_CASE,
                f"{case.name}: target {target_label!r} has axial clearance {max_shift:g} m to an "
                f"adjacent body, which cannot accommodate even k=1 cell-aligned shift "
                f"h={h_eff:g} m ({h_eff:.3e} m; no integer k*h fits); nothing was solved.",
            )
        return float(h_eff), 1

    if isinstance(delta, bool) or not isinstance(delta, int | float):
        raise ForceError(
            E_FORCE_UNSUPPORTED_CASE,
            f"{case.name}: delta must be a positive finite number, got {delta!r}; nothing was solved.",
        )
    delta_used = float(delta)
    if not math.isfinite(delta_used) or delta_used <= 0.0:
        raise ForceError(
            E_FORCE_UNSUPPORTED_CASE,
            f"{case.name}: delta must be positive and finite, got {delta!r}; nothing was solved.",
        )

    # Alignment check runs BEFORE axial-clearance check so non-aligned steps always emit
    # E_FORCE_DELTA_NOT_ALIGNED cleanly without solving.
    k_mult = _aligned_cell_multiple(delta_used, h_eff)
    if k_mult is None:
        raise ForceError(
            E_FORCE_DELTA_NOT_ALIGNED,
            f"{case.name}: virtual displacement delta={delta_used:g} m ({delta_used:.3e} m) is "
            f"not aligned with h={h_eff:g} m ({h_eff:.3e} m, ratio delta/h={delta_used / h_eff:.6g}): "
            "virtual-work delta must be an exact integer multiple k*h of the local axial cell "
            "size h (a cell-aligned shift reproduces the identical mesh, so the central "
            "difference sees one discrete energy function; a sub-cell shift remeshes, which "
            "makes the finite-difference error a step function of delta); nothing was solved.",
        )

    if math.isfinite(max_shift) and delta_used >= max_shift - _AXIS_TOL:
        raise ForceError(
            E_FORCE_UNSUPPORTED_CASE,
            f"{case.name}: virtual displacement delta={delta_used:g} m ({k_mult}*h with "
            f"h={h_eff:g} m) meets or exceeds axial clearance {max_shift:g} m to adjacent body; "
            "nothing was solved.",
        )
    return delta_used, k_mult


def _shift_target(case: Case, target_idx: int, dz: float) -> Case:
    """Return a non-mutating copy of ``case`` with component ``target_idx`` shifted by ``dz``."""
    comp = case.components[target_idx]
    cx, cy, cz = comp.center
    shifted_comp = replace(comp, center=(cx, cy, cz + dz))
    new_components = tuple(
        shifted_comp if idx == target_idx else item for idx, item in enumerate(case.components)
    )
    return replace(case, components=new_components)


def _eval_coenergy_pair(
    solve_case: Case,
    force_engine: Fem2dEngine,
    target_idx: int,
    delta_used: float,
) -> tuple[float, float, float, bool, _FieldState, _FieldState]:
    """Solve ``z +- delta_used`` and return ``(f_z, pot_plus, pot_minus, has_coils, state_p, state_m)``."""
    case_plus = _shift_target(solve_case, target_idx, +delta_used)
    case_minus = _shift_target(solve_case, target_idx, -delta_used)
    try:
        _, _, state_plus = force_engine._solve_core(case_plus)
        _, _, state_minus = force_engine._solve_core(case_minus)
    except EngineError as exc:
        raise ForceError(E_FORCE_UNSUPPORTED_CASE, str(exc)) from exc

    u_p, w_p, has_coils = _total_magnetic_energy_and_coenergy(case_plus, force_engine, state_plus)
    u_m, w_m, _ = _total_magnetic_energy_and_coenergy(case_minus, force_engine, state_minus)

    if has_coils:
        pot_plus, pot_minus = w_p, w_m
        f_z = +(pot_plus - pot_minus) / (2.0 * delta_used)
    else:
        pot_plus, pot_minus = u_p, u_m
        f_z = -(pot_plus - pot_minus) / (2.0 * delta_used)
    return float(f_z), float(pot_plus), float(pot_minus), has_coils, state_plus, state_minus


def coenergy_force(
    case: Case,
    engine: Fem2dEngine,
    *,
    target: str | None = None,
    delta: float | None = None,
    direction: str = "z",
) -> ForceResult:
    """Compute axial force on ``target`` by cell-aligned central-difference virtual work.

    * With fixed-current ``Coil`` sources: evaluates coenergy
      :math:`W' = \\int \\mathbf{J}\\cdot\\mathbf{A}\\,dV - U` at ``z +- delta`` and returns
      :math:`F_z = +(W'(+\\delta) - W'(-\\delta)) / (2\\delta)`.
    * Without ``Coil`` sources (permanent magnets and permeable steel only): evaluates total
      magnetic energy :math:`U = \\int w(\\mathbf{B})\\,dV` at ``z +- delta`` and returns
      :math:`F_z = -(U(+\\delta) - U(-\\delta)) / (2\\delta)`.

    ``delta`` must be an exact integer multiple ``k * h`` of the axial cell size ``h``;
    non-aligned steps raise :data:`E_FORCE_DELTA_NOT_ALIGNED` without solving.
    """
    if direction != "z":
        raise ForceError(
            E_FORCE_UNSUPPORTED_CASE,
            f"{case.name}: direction={direction!r} is not axisymmetric; only axial 'z' virtual "
            "displacement is meaningful in Tier 1 axisymmetric FEM.",
        )

    target_idx, target_comp, target_label = _resolve_target(case, target)
    solve_case = _prepare_solve_case(case, engine)
    force_engine, h_eff = _force_engine_for_case(solve_case, engine, target_comp)
    max_shift = _max_safe_axial_shift(solve_case, target_idx, target_comp)
    delta_used, k_mult = _resolve_aligned_delta(solve_case, target_label, h_eff, max_shift, delta)

    notes: list[str] = [
        f"delta={delta_used:.3e} m = {k_mult}*h (cell-aligned; no remeshing)",
    ]

    try:
        res_0, rep_0, _ = force_engine._solve_core(solve_case)
    except EngineError as exc:
        raise ForceError(E_FORCE_UNSUPPORTED_CASE, str(exc)) from exc

    f_z, pot_plus, pot_minus, has_coils, _, _ = _eval_coenergy_pair(
        solve_case, force_engine, target_idx, delta_used
    )

    if has_coils:
        pot_label = "coenergy W' = int(J.A - w(B)) dV"
        sign_desc = "F_z = +(W'(+delta) - W'(-delta)) / (2*delta) at fixed currents"
        notes.append(f"W'(+delta)={pot_plus:.10e} J, W'(-delta)={pot_minus:.10e} J")
    else:
        pot_label = "total magnetic energy U = int w(B) dV (J = 0)"
        sign_desc = (
            "F_z = -(U(+delta) - U(-delta)) / (2*delta) for fixed magnet polarisation/recoil"
        )
        notes.append(f"U(+delta)={pot_plus:.10e} J, U(-delta)={pot_minus:.10e} J")

    pot_mid = 0.5 * (pot_plus + pot_minus)
    method_detail = (
        f"Central-difference virtual work over {pot_label}; {sign_desc}; "
        f"delta={delta_used:g} m = {k_mult}*h (h={h_eff:g} m)"
    )
    return ForceResult(
        target=target_label,
        force_n=(0.0, 0.0, float(f_z)),
        magnitude_n=abs(float(f_z)),
        method=METHOD_COENERGY,
        method_detail=method_detail,
        contour=None,
        mesh_nodes=rep_0.mesh.nodes,
        mesh_elements=rep_0.mesh.elements,
        element_size=h_eff,
        convergence=res_0.convergence,
        coenergy_j=pot_mid,
        delta_m=delta_used,
        notes=tuple(notes),
    )


def _format_delta_label(delta_m: float, h_m: float) -> str:
    k_mult = _aligned_cell_multiple(delta_m, h_m)
    if k_mult == 1:
        return "h"
    if k_mult is not None:
        return f"{k_mult}h"
    ratio = delta_m / h_m
    if 0.0 < ratio < 1.0:
        inv = round(1.0 / ratio)
        if 2 <= inv <= 1024 and math.isclose(ratio, 1.0 / inv, rel_tol=1e-9, abs_tol=1e-12):
            return f"h/{inv}"
    return f"delta/h={ratio:.4g}"


def _observed_order_for_triple(
    d0: float,
    d1: float,
    d2: float,
    diff_fine: float,
    diff_coarse: float,
) -> float | None:
    """Return observed order ``p`` for a triple ``d0 < d1 < d2`` given ``D0=F1-F0``, ``D1=F2-F1``."""
    if diff_fine == 0.0 or diff_coarse == 0.0:
        return None
    ratio_d = diff_coarse / diff_fine
    if not math.isfinite(ratio_d) or ratio_d <= 0.0:
        return None
    r0 = d1 / d0
    r1 = d2 / d1
    if math.isclose(r0, r1, rel_tol=1e-9, abs_tol=1e-12):
        return float(math.log(ratio_d) / math.log(r0))

    def _g(p_val: float) -> float:
        if abs(p_val) < 1e-12:
            return math.log(r1) / math.log(r0) - ratio_d
        return (r0**p_val) * ((r1**p_val) - 1.0) / ((r0**p_val) - 1.0) - ratio_d

    lo, hi = -20.0, 20.0
    g_lo, g_hi = _g(lo), _g(hi)
    if not (math.isfinite(g_lo) and math.isfinite(g_hi)) or g_lo * g_hi > 0.0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        g_mid = _g(mid)
        if g_mid == 0.0:
            return float(mid)
        if g_lo * g_mid <= 0.0:
            hi = mid
        else:
            lo = mid
            g_lo = g_mid
    return float(0.5 * (lo + hi))


def _fit_observed_order(
    steps: tuple[DeltaStep, ...],
    diffs: list[float],
    run: list[int],
) -> float | None:
    """Fit power-law order ``p`` across a same-sign difference run ``run``.

    Fits ``log|F(d_{i+1}) - F(d_i)| = alpha + log|(d_{i+1}^p - d_i^p) / p|`` so both
    geometric ladders (where it reduces identically to OLS slope vs ``log delta``) and
    arithmetic cell-aligned ladders ``(h, 2h, 3h, ...)`` recover exact ``p`` for ``F0 + C*d^p``.
    """
    if len(run) < 2:
        return None

    d_lo = np.array([steps[i].delta_m for i in run], dtype=np.float64)
    d_hi = np.array([steps[i + 1].delta_m for i in run], dtype=np.float64)
    y_fit = np.log(np.array([abs(diffs[i]) for i in run], dtype=np.float64))
    if not np.all(np.isfinite(y_fit)) or np.ptp(d_lo) <= 0.0:
        return None

    ratios = d_hi / d_lo
    if np.allclose(ratios, ratios[0], rtol=1e-9, atol=1e-12):
        slope = float(np.polyfit(np.log(d_lo), y_fit, 1)[0])
        return slope if math.isfinite(slope) else None

    if len(run) == 2 and run[1] == run[0] + 1:
        idx = run[0]
        return _observed_order_for_triple(
            steps[idx].delta_m,
            steps[idx + 1].delta_m,
            steps[idx + 2].delta_m,
            diffs[idx],
            diffs[idx + 1],
        )

    log_d_lo = np.log(d_lo)
    log_d_hi = np.log(d_hi)
    log_r = np.log(ratios)

    def _sse(p_val: float) -> float:
        if abs(p_val) < 1e-7:
            f_p = (
                np.log(log_r)
                + 0.5 * p_val * (log_d_lo + log_d_hi)
                + (p_val * p_val) * (log_r * log_r) / 24.0
            )
        else:
            x_term = p_val * log_r
            f_p = p_val * log_d_lo + np.log(np.abs(np.expm1(x_term)) / abs(p_val))
        resid = y_fit - f_p
        resid = resid - np.mean(resid)
        return float(np.dot(resid, resid))

    p_grid = np.linspace(-20.0, 20.0, 401)
    losses = np.array([_sse(float(p_val)) for p_val in p_grid], dtype=np.float64)
    best_idx = int(np.argmin(losses))
    a = float(p_grid[max(0, best_idx - 1)])
    b = float(p_grid[min(len(p_grid) - 1, best_idx + 1)])
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = _sse(c), _sse(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = _sse(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = _sse(d)
    p_best = 0.5 * (a + b)
    return float(p_best) if math.isfinite(p_best) else None


def _analyze_delta_ladder(
    target: str,
    h_m: float,
    steps: tuple[DeltaStep, ...],
    refused: tuple[tuple[float, str], ...],
    *,
    base_mesh_counts: tuple[int, int],
    axial_gap_m: float | None = None,
    extra_notes: tuple[str, ...] = (),
) -> DeltaStudyResult:
    """Compute observed order, asymptotic flag, extrapolation, uncertainty, and alignment checks."""
    n = len(steps)
    forces = [s.force_n for s in steps]
    peak_to_peak_n = float(max(forces) - min(forces)) if forces else 0.0
    diffs = [steps[i + 1].force_n - steps[i].force_n for i in range(n - 1)]

    if n <= 1:
        monotonic = True
    else:
        monotonic = (
            all(d > 0.0 for d in diffs)
            or all(d < 0.0 for d in diffs)
            or all(d == 0.0 for d in diffs)
        )

    p_per_pair = tuple(
        _observed_order_for_triple(
            steps[i].delta_m,
            steps[i + 1].delta_m,
            steps[i + 2].delta_m,
            diffs[i],
            diffs[i + 1],
        )
        for i in range(max(0, n - 2))
    )

    best_run: list[int] = []
    curr_run: list[int] = []
    curr_sign = 0
    for idx, diff in enumerate(diffs):
        if diff == 0.0:
            if len(curr_run) >= 2 and len(curr_run) > len(best_run):
                best_run = curr_run
            curr_run = []
            curr_sign = 0
            continue
        sign = 1 if diff > 0.0 else -1
        if sign == curr_sign:
            curr_run.append(idx)
        else:
            if len(curr_run) >= 2 and len(curr_run) > len(best_run):
                best_run = curr_run
            curr_run = [idx]
            curr_sign = sign
    if len(curr_run) >= 2 and len(curr_run) > len(best_run):
        best_run = curr_run

    p_fit = _fit_observed_order(steps, diffs, best_run)

    ratios_gt_one = n >= 3 and all(
        diffs[i] != 0.0 and (diffs[i + 1] / diffs[i]) > 1.0 for i in range(n - 2)
    )
    asymptotic = bool(
        n >= 3 and monotonic and ratios_gt_one and p_fit is not None and 1.5 <= p_fit <= 2.5
    )

    notes: list[str] = list(extra_notes)
    f_ext: float | None
    if n < 3:
        f_ext = None
        notes.append("fewer than 3 valid delta steps; extrapolation unavailable")
    elif all(d == 0.0 for d in diffs):
        f_ext = float(steps[0].force_n)
        notes.append("force is invariant across all tested delta steps")
    else:
        p_used = (
            p_fit
            if (p_fit is not None and math.isfinite(p_fit) and p_fit > 0.0)
            else next(
                (
                    p_val
                    for p_val in p_per_pair
                    if p_val is not None and math.isfinite(p_val) and p_val > 0.0
                ),
                None,
            )
        )
        if p_used is None:
            f_ext = None
            notes.append(
                "no positive observed convergence order exists across tested delta steps; "
                "extrapolation unavailable"
            )
        else:
            r0 = steps[1].delta_m / steps[0].delta_m
            f_ext = float(
                steps[0].force_n + (steps[0].force_n - steps[1].force_n) / (r0**p_used - 1.0)
            )

    triple_exts: list[tuple[int, float]] = []
    for idx, p_val in enumerate(p_per_pair):
        if p_val is not None and math.isfinite(p_val) and p_val > 0.0:
            r_triple = steps[idx + 1].delta_m / steps[idx].delta_m
            ext_val = float(
                steps[idx].force_n
                + (steps[idx].force_n - steps[idx + 1].force_n) / (r_triple**p_val - 1.0)
            )
            triple_exts.append((idx, ext_val))

    if f_ext is not None:
        correction = abs(f_ext - steps[0].force_n)
        candidates = [f_ext, *(val for _, val in triple_exts)]
        spread = float(max(candidates) - min(candidates)) if len(candidates) >= 2 else 0.0
        step_uncertainty_n = float(max(correction, spread))
    else:
        step_uncertainty_n = float(peak_to_peak_n)

    ref_scale = (
        abs(f_ext)
        if (f_ext is not None and f_ext != 0.0)
        else (max(abs(f) for f in forces) if forces else 0.0)
    )
    step_uncertainty_pct = float(100.0 * step_uncertainty_n / ref_scale) if ref_scale > 0.0 else 0.0
    peak_to_peak_pct = float(100.0 * peak_to_peak_n / ref_scale) if ref_scale > 0.0 else 0.0

    if n >= 3 and not asymptotic:
        notes.append(
            "the delta ladder is not in its asymptotic range (nominal central-difference order 2), "
            "so the extrapolation is a best estimate with a correspondingly wider band, not a "
            "demonstrated limit"
        )
    if triple_exts:
        triple_desc = ", ".join(
            f"({steps[idx].label}, {steps[idx + 1].label}, {steps[idx + 2].label}): {val:.5f} N"
            for idx, val in triple_exts
        )
        notes.append(f"triple extrapolations with positive order: {triple_desc}")

    mesh_counts_invariant = bool(steps) and all(
        (s.mesh_nodes_plus, s.mesh_elements_plus) == base_mesh_counts
        and (s.mesh_nodes_minus, s.mesh_elements_minus) == base_mesh_counts
        for s in steps
    )

    expected_gap_cell_sum: int | None = None
    gap_cell_sum_invariant: bool | None = None
    if axial_gap_m is not None and math.isfinite(axial_gap_m):
        g_over_h = axial_gap_m / h_m
        g_int = round(g_over_h)
        if abs(g_over_h - g_int) <= 1e-9 * max(1.0, abs(g_over_h)):
            expected_gap_cell_sum = int(2 * g_int)
            gap_cell_sum_invariant = bool(steps) and all(
                s.gap_cells_plus is not None
                and s.gap_cells_minus is not None
                and (s.gap_cells_plus + s.gap_cells_minus) == expected_gap_cell_sum
                for s in steps
            )
        else:
            gap_cell_sum_invariant = False

    alignment_invariant_holds = (
        bool(steps) and mesh_counts_invariant and (gap_cell_sum_invariant is not False)
    )

    if steps:
        gap_clause = (
            f"gap_cells_plus + gap_cells_minus == {expected_gap_cell_sum} (= 2*gap/h)"
            if expected_gap_cell_sum is not None
            else (
                "no adjacent axial body (gap cell sum n/a)"
                if axial_gap_m is None
                else f"axial gap {axial_gap_m:g} m is not an integer multiple of h={h_m:g} m"
            )
        )
        status = "HOLDS" if alignment_invariant_holds else "FAILED"
        notes.append(
            f"cell-alignment invariant {status} at h={h_m:.2e} m: "
            f"base={base_mesh_counts[0]}n/{base_mesh_counts[1]}e, "
            f"mesh_counts_invariant={mesh_counts_invariant}, {gap_clause}"
        )

    if refused:
        notes.append(f"{len(refused)} delta step(s) refused")

    return DeltaStudyResult(
        target=target,
        h_m=float(h_m),
        base_mesh_nodes=int(base_mesh_counts[0]),
        base_mesh_elements=int(base_mesh_counts[1]),
        axial_gap_m=axial_gap_m,
        expected_gap_cell_sum=expected_gap_cell_sum,
        steps=steps,
        refused=refused,
        p_per_pair=p_per_pair,
        p_fit=p_fit,
        monotonic=monotonic,
        asymptotic=asymptotic,
        f_extrapolated_n=f_ext,
        step_uncertainty_n=step_uncertainty_n,
        step_uncertainty_pct=step_uncertainty_pct,
        peak_to_peak_n=peak_to_peak_n,
        peak_to_peak_pct=peak_to_peak_pct,
        mesh_counts_invariant=mesh_counts_invariant,
        gap_cell_sum_invariant=gap_cell_sum_invariant,
        alignment_invariant_holds=alignment_invariant_holds,
        notes=tuple(notes),
    )


def coenergy_delta_study(
    case: Case,
    engine: Fem2dEngine,
    *,
    target: str | None = None,
    deltas: Iterable[float] | None = None,
    direction: str = "z",
) -> DeltaStudyResult:
    """Run a cell-aligned finite-difference virtual-work ladder ``(h, 2h, 3h)`` by default."""
    if direction != "z":
        raise ForceError(
            E_FORCE_UNSUPPORTED_CASE,
            f"{case.name}: direction={direction!r} is not axisymmetric; only axial 'z' virtual "
            "displacement is meaningful in Tier 1 axisymmetric FEM.",
        )

    target_idx, target_comp, target_label = _resolve_target(case, target)
    solve_case = _prepare_solve_case(case, engine)
    force_engine, h_eff = _force_engine_for_case(solve_case, engine, target_comp)
    base_mesh, _, _ = force_engine._build_mesh_and_load(solve_case)
    base_counts = (int(base_mesh.nodes.shape[0]), int(base_mesh.elements.shape[0]))
    gap_info = _nearest_axial_gap_face(solve_case, target_idx, target_comp)
    max_shift = math.inf if gap_info is None else gap_info[0]
    axial_gap_m = None if gap_info is None else float(gap_info[0])

    t_ext = _component_rz_extents(target_comp)
    assert t_ext is not None
    _, _, t_z0, t_z1, _ = t_ext

    if deltas is None:
        delta_list = [h_eff, 2.0 * h_eff, 3.0 * h_eff]
    else:
        raw_deltas: list[float] = []
        for item in deltas:
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise ForceError(
                    E_FORCE_UNSUPPORTED_CASE,
                    f"{case.name}: deltas entries must be positive finite numbers, got {item!r}.",
                )
            val = float(item)
            if not math.isfinite(val) or val <= 0.0:
                raise ForceError(
                    E_FORCE_UNSUPPORTED_CASE,
                    f"{case.name}: deltas entries must be positive and finite, got {item!r}.",
                )
            raw_deltas.append(val)
        if not raw_deltas:
            raise ForceError(
                E_FORCE_UNSUPPORTED_CASE,
                f"{case.name}: deltas must contain at least one positive value.",
            )
        raw_deltas.sort()
        delta_list = []
        for val in raw_deltas:
            if not delta_list or abs(val - delta_list[-1]) > 1e-15 * max(1.0, abs(val)):
                delta_list.append(val)

    steps: list[DeltaStep] = []
    refused: list[tuple[float, str]] = []
    extra_notes: list[str] = []

    for delta_m in delta_list:
        try:
            delta_used, _ = _resolve_aligned_delta(
                solve_case, target_label, h_eff, max_shift, delta_m
            )
            f_z, _, _, _, state_plus, state_minus = _eval_coenergy_pair(
                solve_case, force_engine, target_idx, delta_used
            )
        except ForceError as exc:
            refused.append((float(delta_m), str(exc)))
            continue

        n_plus = int(state_plus.mesh.nodes.shape[0])
        e_plus = int(state_plus.mesh.elements.shape[0])
        n_minus = int(state_minus.mesh.nodes.shape[0])
        e_minus = int(state_minus.mesh.elements.shape[0])

        gap_cells_plus: int | None = None
        gap_cells_minus: int | None = None
        if gap_info is not None:
            _, side, other_face_z = gap_info
            if side == "above":
                gap_cells_plus = _count_gap_cells(
                    state_plus.mesh.z_axis, t_z1 + delta_used, other_face_z
                )
                gap_cells_minus = _count_gap_cells(
                    state_minus.mesh.z_axis, t_z1 - delta_used, other_face_z
                )
            elif side == "below":
                gap_cells_plus = _count_gap_cells(
                    state_plus.mesh.z_axis, other_face_z, t_z0 + delta_used
                )
                gap_cells_minus = _count_gap_cells(
                    state_minus.mesh.z_axis, other_face_z, t_z0 - delta_used
                )

        if not steps or steps[-1].force_n == 0.0:
            change_pct: float | None = None
        else:
            prev_f = steps[-1].force_n
            change_pct = float(100.0 * (f_z - prev_f) / abs(prev_f))

        steps.append(
            DeltaStep(
                label=_format_delta_label(delta_used, h_eff),
                delta_m=delta_used,
                delta_over_h=float(delta_used / h_eff),
                force_n=f_z,
                change_vs_prev_pct=change_pct,
                mesh_nodes_plus=n_plus,
                mesh_elements_plus=e_plus,
                mesh_nodes_minus=n_minus,
                mesh_elements_minus=e_minus,
                gap_cells_plus=gap_cells_plus,
                gap_cells_minus=gap_cells_minus,
            )
        )

    return _analyze_delta_ladder(
        target_label,
        h_eff,
        tuple(steps),
        tuple(refused),
        base_mesh_counts=base_counts,
        axial_gap_m=axial_gap_m,
        extra_notes=tuple(extra_notes),
    )


def torque(
    case: Case,
    engine: Fem2dEngine,  # noqa: ARG001 - kept for API symmetry with maxwell_force/coenergy_force
    *,
    target: str | None = None,
) -> ForceResult:
    """Refuse torque honestly: axisymmetric ``A_phi`` has identically zero axial torque."""
    target_note = f" on target {target!r}" if target is not None else ""
    raise ForceError(
        E_TORQUE_NOT_MEANINGFUL,
        f"{case.name}: cannot compute torque{target_note}: the axisymmetric A_phi formulation "
        "cannot represent a tangential stress component, so torque about the symmetry axis is "
        "identically zero by construction — this is not a computed zero, and a 3D Tier 1 "
        "formulation would be required; nothing was solved.",
    )
