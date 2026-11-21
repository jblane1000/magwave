"""Axisymmetric Tier 1 electromagnetic force evaluation and honest torque refusal (M5.4).

Two independent methods are implemented on top of :class:`~magwave.fem2d.engine.Fem2dEngine`:

1. **Maxwell stress tensor** (:func:`maxwell_force`) over a closed meridional contour in air
   enclosing only the target body:

   .. math::

       F_z = \\oint \\left[ \\frac{1}{\\mu_0} B_r B_z n_r
             + \\frac{1}{2 \\mu_0} (B_z^2 - B_r^2) n_z \\right] 2\\pi r \\, dl,

   integrated segment-by-segment with 2-point Gauss–Legendre quadrature via
   :func:`magwave.fem2d.field.sample_field`. Radial and azimuthal components vanish by
   axisymmetry (verified numerically across 8 azimuthal sectors).

2. **Coenergy / virtual work** (:func:`coenergy_force`) by central finite difference over two
   independent solves at axial displacements ``z +- delta``:

   * **With fixed-current coils:** differentiates magnetic coenergy
     :math:`W' = \\int \\mathbf{J}\\cdot\\mathbf{A}\\,dV - U` with :math:`F_z = +dW'/dz`
     (which reduces to :math:`\\frac{1}{2}\\int (B^2/\\mu_0)\\,dV` in linear air).
   * **Pure permanent magnet + permeable/air (no free currents):** differentiates total stored
     magnetic energy :math:`U = \\int w(\\mathbf{B})\\,dV` with :math:`F_z = -dU/dz`, since
     :math:`\\frac{1}{2}\\int \\mathbf{B}\\cdot\\mathbf{H}\\,dV \\equiv 0` when :math:`\\mathbf{J}=0`.

3. **Torque** (:func:`torque`) refuses explicitly with :data:`E_TORQUE_NOT_MEANINGFUL`: the
   axisymmetric ``A_phi`` formulation has no azimuthal asymmetry, so axial torque is
   identically zero by construction rather than a computed physical zero.
"""

from __future__ import annotations

import math
from collections.abc import Callable
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

    # Assert F_x, F_y, F_theta = 0 numerically across 8 symmetric azimuthal sectors.
    phis = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
    fx_check = float(radial_force_2pi * np.mean(np.cos(phis)))
    fy_check = float(radial_force_2pi * np.mean(np.sin(phis)))
    sym_tol = 1e-13 * max(abs(f_z_total), abs(radial_force_2pi), 1.0)
    if abs(fx_check) > sym_tol or abs(fy_check) > sym_tol:
        raise AssertionError(
            f"axisymmetry numerical check failed: Fx={fx_check:.3e}, Fy={fy_check:.3e}"
        )

    notes = snap_notes + (
        f"segment F_z (N): bottom={seg_values['bottom']:+.6e}, outer={seg_values['outer']:+.6e}, "
        f"top={seg_values['top']:+.6e}, axis={seg_values['axis']:.1f}",
        f"axisymmetry check (8 sectors): Fx={fx_check:.3e} N, Fy={fy_check:.3e} N, F_theta=0 (B_phi=0)",
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


def _max_safe_axial_shift(case: Case, target_idx: int, target_comp: Magnet | Coil) -> float:
    """Return the largest axial shift ``|dz|`` before ``target_comp`` touches another body."""
    t_ext = _component_rz_extents(target_comp)
    assert t_ext is not None
    t_r0, t_r1, t_z0, t_z1, _ = t_ext
    max_shift = math.inf

    for idx, comp in enumerate(case.components):
        if idx == target_idx:
            continue
        o_ext = _component_rz_extents(comp)
        if o_ext is None:
            continue
        o_r0, o_r1, o_z0, o_z1, tag = o_ext
        if tag == "sphere":
            assert isinstance(comp, Permeable)
            rc = math.hypot(comp.center[0], comp.center[1])
            zc = float(comp.center[2])
            rad = float(comp.dims["radius"])
            dr = max(t_r0 - rc, 0.0, rc - t_r1)
            if dr < rad - _AXIS_TOL:
                z_clearance = math.sqrt(max(0.0, rad * rad - dr * dr))
                if t_z0 >= zc + z_clearance - _AXIS_TOL:
                    max_shift = min(max_shift, max(0.0, t_z0 - (zc + z_clearance)))
                elif t_z1 <= zc - z_clearance + _AXIS_TOL:
                    max_shift = min(max_shift, max(0.0, (zc - z_clearance) - t_z1))
                else:
                    max_shift = 0.0
        elif min(t_r1, o_r1) - max(t_r0, o_r0) >= -_AXIS_TOL:
            if o_z0 >= t_z1 - _AXIS_TOL:
                max_shift = min(max_shift, max(0.0, o_z0 - t_z1))
            elif t_z0 >= o_z1 - _AXIS_TOL:
                max_shift = min(max_shift, max(0.0, t_z0 - o_z1))
            else:
                max_shift = 0.0
    return max_shift


def _shift_target(case: Case, target_idx: int, dz: float) -> Case:
    """Return a non-mutating copy of ``case`` with component ``target_idx`` shifted by ``dz``."""
    comp = case.components[target_idx]
    cx, cy, cz = comp.center
    shifted_comp = replace(comp, center=(cx, cy, cz + dz))
    new_components = tuple(
        shifted_comp if idx == target_idx else item for idx, item in enumerate(case.components)
    )
    return replace(case, components=new_components)


def coenergy_force(
    case: Case,
    engine: Fem2dEngine,
    *,
    target: str | None = None,
    delta: float | None = None,
    direction: str = "z",
) -> ForceResult:
    """Compute axial force on ``target`` by central-difference virtual work.

    * With fixed-current ``Coil`` sources: evaluates coenergy
      :math:`W' = \\int \\mathbf{J}\\cdot\\mathbf{A}\\,dV - U` at ``z +- delta`` and returns
      :math:`F_z = +(W'(+\\delta) - W'(-\\delta)) / (2\\delta)`.
    * Without ``Coil`` sources (permanent magnets and permeable steel only): evaluates total
      magnetic energy :math:`U = \\int w(\\mathbf{B})\\,dV` at ``z +- delta`` and returns
      :math:`F_z = -(U(+\\delta) - U(-\\delta)) / (2\\delta)`.
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

    notes: list[str] = []
    if delta is None:
        delta_used = h_eff
        if math.isfinite(max_shift) and delta_used >= 0.9 * max_shift:
            if max_shift <= _AXIS_TOL:
                raise ForceError(
                    E_FORCE_UNSUPPORTED_CASE,
                    f"{case.name}: target {target_label!r} has zero axial clearance to an "
                    "adjacent body; virtual axial displacement is impossible.",
                )
            delta_used = 0.25 * max_shift
            notes.append(
                f"default delta={h_eff:g} m clamped to 0.25 * axial_gap = {delta_used:g} m "
                f"(adjacent body gap={max_shift:g} m)"
            )
    else:
        if isinstance(delta, bool) or not isinstance(delta, int | float):
            raise ForceError(
                E_FORCE_UNSUPPORTED_CASE,
                f"{case.name}: delta must be a positive finite number, got {delta!r}.",
            )
        delta_used = float(delta)
        if not math.isfinite(delta_used) or delta_used <= 0.0:
            raise ForceError(
                E_FORCE_UNSUPPORTED_CASE,
                f"{case.name}: delta must be positive and finite, got {delta!r}.",
            )
        if math.isfinite(max_shift) and delta_used >= max_shift - _AXIS_TOL:
            raise ForceError(
                E_FORCE_UNSUPPORTED_CASE,
                f"{case.name}: virtual displacement delta={delta_used:g} m exceeds axial "
                f"clearance {max_shift:g} m to adjacent body.",
            )

    case_plus = _shift_target(solve_case, target_idx, +delta_used)
    case_minus = _shift_target(solve_case, target_idx, -delta_used)

    try:
        res_0, rep_0, _ = force_engine._solve_core(solve_case)
        _, _, state_plus = force_engine._solve_core(case_plus)
        _, _, state_minus = force_engine._solve_core(case_minus)
    except EngineError as exc:
        raise ForceError(E_FORCE_UNSUPPORTED_CASE, str(exc)) from exc

    u_p, w_p, has_coils = _total_magnetic_energy_and_coenergy(case_plus, force_engine, state_plus)
    u_m, w_m, _ = _total_magnetic_energy_and_coenergy(case_minus, force_engine, state_minus)

    if has_coils:
        pot_plus, pot_minus = w_p, w_m
        f_z = +(pot_plus - pot_minus) / (2.0 * delta_used)
        pot_label = "coenergy W' = int(J.A - w(B)) dV"
        sign_desc = "F_z = +(W'(+delta) - W'(-delta)) / (2*delta) at fixed currents"
        notes.append(f"W'(+delta)={pot_plus:.10e} J, W'(-delta)={pot_minus:.10e} J")
    else:
        pot_plus, pot_minus = u_p, u_m
        f_z = -(pot_plus - pot_minus) / (2.0 * delta_used)
        pot_label = "total magnetic energy U = int w(B) dV (J = 0)"
        sign_desc = (
            "F_z = -(U(+delta) - U(-delta)) / (2*delta) for fixed magnet polarisation/recoil"
        )
        notes.append(f"U(+delta)={pot_plus:.10e} J, U(-delta)={pot_minus:.10e} J")

    pot_mid = 0.5 * (pot_plus + pot_minus)
    method_detail = (
        f"Central-difference virtual work over {pot_label}; {sign_desc}; "
        f"delta={delta_used:g} m (h={h_eff:g} m)"
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
