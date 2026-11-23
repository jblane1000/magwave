"""Axisymmetric tensor-product triangular mesh generator (no external mesher).

Generates an axis-aligned ``(r, z)`` tensor grid with forced breakpoints at source and
refinement boundaries, uniform subdivision inside the refinement box, and geometric growth
away from it. Each grid rectangle is split into two counter-clockwise linear triangles
whose diagonal direction alternates along ``z`` to preserve axial mirror symmetry.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_TOL = 1e-12
DEFAULT_MAX_NODES: int = 400_000


class MeshBudgetError(ValueError):
    """Raised when the requested mesh resolution exceeds the node budget."""

    def __init__(
        self,
        *,
        nodes: int,
        nr: int,
        nz: int,
        max_nodes: int,
        element_size: float,
    ) -> None:
        self.nodes = int(nodes)
        self.nr = int(nr)
        self.nz = int(nz)
        self.max_nodes = int(max_nodes)
        self.element_size = float(element_size)
        super().__init__(
            f"element_size={self.element_size:g} m requires {self.nodes} nodes "
            f"({self.nr} x {self.nz}), exceeding max_nodes={self.max_nodes}"
        )


@dataclass(frozen=True)
class MeshStats:
    """Summary statistics for a 2-D axisymmetric triangular mesh."""

    nodes: int
    elements: int
    min_edge: float
    max_edge: float
    h_requested: float


@dataclass(frozen=True)
class Mesh:
    """Axisymmetric ``(r, z)`` triangular mesh over ``[0, r_max] x [z_min, z_max]``."""

    nodes: np.ndarray
    elements: np.ndarray
    r_axis: np.ndarray
    z_axis: np.ndarray
    h_requested: float

    def stats(self) -> MeshStats:
        """Compute node/element counts and min/max triangle edge lengths in metres."""
        tri_pts = self.nodes[self.elements]  # (m, 3, 2)
        e01 = np.linalg.norm(tri_pts[:, 1] - tri_pts[:, 0], axis=1)
        e12 = np.linalg.norm(tri_pts[:, 2] - tri_pts[:, 1], axis=1)
        e20 = np.linalg.norm(tri_pts[:, 0] - tri_pts[:, 2], axis=1)
        edges = np.concatenate((e01, e12, e20))
        return MeshStats(
            nodes=int(self.nodes.shape[0]),
            elements=int(self.elements.shape[0]),
            min_edge=float(np.min(edges)),
            max_edge=float(np.max(edges)),
            h_requested=float(self.h_requested),
        )

    def locate(self, points: np.ndarray) -> np.ndarray:
        """Return the element index containing each ``(r, z)`` point.

        Points on shared grid edges or nodes resolve deterministically: cell lookup uses
        ``side="right"`` biased by ``1e-12``, and diagonal selection breaks ties in favour
        of the lower triangle ``T0`` on even ``z``-rows (``u >= v - 1e-12``) and the upper
        triangle ``T1`` on odd ``z``-rows (``u + v >= 1 - 1e-12``). Points with ``r < 0``
        or outside the domain raise ``ValueError``.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must have shape (n, 2) or (2,), got {np.shape(points)}")
        if pts.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        if not np.all(np.isfinite(pts)):
            raise ValueError("points must contain finite coordinates")

        r = pts[:, 0]
        z = pts[:, 1]
        r_max = float(self.r_axis[-1])
        z_min = float(self.z_axis[0])
        z_max = float(self.z_axis[-1])
        if np.any(r < 0.0) or np.any(r > r_max) or np.any(z < z_min) or np.any(z > z_max):
            raise ValueError(
                f"point(s) outside mesh domain r in [0, {r_max}], z in [{z_min}, {z_max}]"
            )

        nr = self.r_axis.shape[0]
        nz = self.z_axis.shape[0]
        ir = np.clip(np.searchsorted(self.r_axis, r + _TOL, side="right") - 1, 0, nr - 2)
        iz = np.clip(np.searchsorted(self.z_axis, z + _TOL, side="right") - 1, 0, nz - 2)

        u = (r - self.r_axis[ir]) / (self.r_axis[ir + 1] - self.r_axis[ir])
        v = (z - self.z_axis[iz]) / (self.z_axis[iz + 1] - self.z_axis[iz])
        cell_idx = ir * (nz - 1) + iz
        n_cells = (nr - 1) * (nz - 1)
        is_backslash = (iz % 2) == 1
        in_t0 = np.where(is_backslash, (u + v) < (1.0 - _TOL), u >= (v - _TOL))
        return np.where(in_t0, cell_idx, n_cells + cell_idx).astype(np.int64)


def _validate_breakpoints(
    values: Sequence[float],
    low: float,
    high: float,
    name: str,
) -> list[float]:
    parsed = [float(v) for v in values]
    for v in parsed:
        if not math.isfinite(v):
            raise ValueError(f"{name}: breakpoint {v!r} must be finite")
        if v < low or v > high:
            raise ValueError(f"{name}: breakpoint {v!r} outside [{low}, {high}]")
    ordered = sorted(parsed)
    for a, b in zip(ordered, ordered[1:], strict=False):
        if abs(b - a) <= _TOL:
            raise ValueError(f"{name}: duplicate or near-duplicate breakpoints {a!r} and {b!r}")
    return ordered


def _geometric_steps(length: float, s0: float, growth: float) -> np.ndarray:
    """Partition ``length`` into steps starting at ~``s0`` with growth ratio <= ``growth``."""
    if length <= s0 * growth:
        return np.array([length], dtype=np.float64)

    # Smallest k >= 1 such that s0 * sum_{m=0}^{k-1} growth^m >= length.
    total = 0.0
    step = s0
    k = 0
    while total < length - _TOL:
        total += step
        step *= growth
        k += 1

    if k * s0 > length:
        # Length lies between S(k-1, growth) and k*s0; use k equal sub-steps <= s0.
        return np.full(k, length / k, dtype=np.float64)

    # Solve s0 * sum_{m=0}^{k-1} g^m == length for g in [1.0, growth] by bisection.
    lo, hi = 1.0, growth
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s_mid = k * s0 if abs(mid - 1.0) < 1e-14 else s0 * (mid**k - 1.0) / (mid - 1.0)
        if s_mid < length:
            lo = mid
        else:
            hi = mid
    g_eff = 0.5 * (lo + hi)
    steps = s0 * (g_eff ** np.arange(k, dtype=np.float64))
    steps *= length / float(np.sum(steps))
    return steps


def _merge_axis_points(
    a_min: float,
    a_max: float,
    ref0: float,
    ref1: float,
    breakpoints: list[float],
) -> np.ndarray:
    """Merge domain endpoints, refinement bounds and breakpoints keeping exact breakpoints."""
    candidates: list[tuple[float, int]] = [
        (a_min, 2),
        (a_max, 2),
        (ref0, 0),
        (ref1, 0),
        *((bp, 1) for bp in breakpoints),
    ]
    candidates.sort(key=lambda item: (item[0], -item[1]))
    merged: list[tuple[float, int]] = []
    for val, prio in candidates:
        if not merged or abs(val - merged[-1][0]) > _TOL:
            merged.append((val, prio))
        elif prio > merged[-1][1]:
            merged[-1] = (val, prio)
    return np.array([val for val, _ in merged], dtype=np.float64)


def _classify_axis_intervals(
    pts: np.ndarray, ref0: float, ref1: float
) -> tuple[list[int], list[int], list[int]]:
    """Partition interval indices into uniform interior, left exterior, and right exterior."""
    inside_idx: list[int] = []
    left_idx: list[int] = []
    right_idx: list[int] = []
    for idx in range(len(pts) - 1):
        x0 = float(pts[idx])
        x1 = float(pts[idx + 1])
        if x0 >= ref0 - _TOL and x1 <= ref1 + _TOL:
            inside_idx.append(idx)
        elif x1 <= ref0 + _TOL:
            left_idx.append(idx)
        else:
            right_idx.append(idx)
    return inside_idx, left_idx, right_idx


def _count_axis_nodes(
    a_min: float,
    a_max: float,
    ref0: float,
    ref1: float,
    breakpoints: list[float],
    h: float,
    growth: float,
) -> int:
    """Count 1-D axis nodes without allocating interior coordinate arrays."""
    pts = _merge_axis_points(a_min, a_max, ref0, ref1, breakpoints)
    inside_idx, left_idx, right_idx = _classify_axis_intervals(pts, ref0, ref1)

    total_steps = 0
    for idx in inside_idx:
        length = float(pts[idx + 1]) - float(pts[idx])
        total_steps += max(1, math.ceil(length / h - _TOL))

    s_curr = h
    for idx in reversed(left_idx):
        steps = _geometric_steps(float(pts[idx + 1]) - float(pts[idx]), s_curr, growth)
        s_curr = float(steps[-1]) * growth
        total_steps += int(steps.shape[0])

    s_curr = h
    for idx in right_idx:
        steps = _geometric_steps(float(pts[idx + 1]) - float(pts[idx]), s_curr, growth)
        s_curr = float(steps[-1]) * growth
        total_steps += int(steps.shape[0])

    return total_steps + 1


def _build_axis(
    a_min: float,
    a_max: float,
    ref0: float,
    ref1: float,
    breakpoints: list[float],
    h: float,
    growth: float,
) -> np.ndarray:
    """Build 1-D ascending coordinate array with uniform interior and graded exterior."""
    pts = _merge_axis_points(a_min, a_max, ref0, ref1, breakpoints)
    segments: list[np.ndarray | None] = [None] * (len(pts) - 1)
    inside_idx, left_idx, right_idx = _classify_axis_intervals(pts, ref0, ref1)

    for idx in inside_idx:
        x0 = float(pts[idx])
        x1 = float(pts[idx + 1])
        length = x1 - x0
        n_steps = max(1, math.ceil(length / h - _TOL))
        segments[idx] = np.linspace(x0, x1, n_steps + 1, dtype=np.float64)

    # Grow outward to the left from ref0 toward a_min.
    s_curr = h
    for idx in reversed(left_idx):
        x0 = float(pts[idx])
        x1 = float(pts[idx + 1])
        steps = _geometric_steps(x1 - x0, s_curr, growth)
        s_curr = float(steps[-1]) * growth
        coords = x1 - np.concatenate(([0.0], np.cumsum(steps)))
        coords = coords[::-1]
        coords[0] = x0
        coords[-1] = x1
        segments[idx] = coords

    # Grow outward to the right from ref1 toward a_max.
    s_curr = h
    for idx in right_idx:
        x0 = float(pts[idx])
        x1 = float(pts[idx + 1])
        steps = _geometric_steps(x1 - x0, s_curr, growth)
        s_curr = float(steps[-1]) * growth
        coords = x0 + np.concatenate(([0.0], np.cumsum(steps)))
        coords[0] = x0
        coords[-1] = x1
        segments[idx] = coords

    pieces = [seg[:-1] for seg in segments if seg is not None]
    pieces.append(np.array([a_max], dtype=np.float64))
    return np.concatenate(pieces)


def _validate_mesh_inputs(
    r_max: float,
    z_min: float,
    z_max: float,
    h: float,
    *,
    refinement_box: tuple[float, float, float, float] | None,
    breakpoints_r: Sequence[float],
    breakpoints_z: Sequence[float],
    growth: float,
    max_nodes: int,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    list[float],
    list[float],
]:
    r_max_f = float(r_max)
    z_min_f = float(z_min)
    z_max_f = float(z_max)
    h_f = float(h)
    growth_f = float(growth)

    if not math.isfinite(r_max_f) or r_max_f <= 0.0:
        raise ValueError(f"r_max must be positive and finite, got {r_max!r}")
    if not (math.isfinite(z_min_f) and math.isfinite(z_max_f)) or z_max_f <= z_min_f:
        raise ValueError(f"require finite z_max > z_min, got z_min={z_min!r}, z_max={z_max!r}")
    if not math.isfinite(h_f) or h_f <= 0.0:
        raise ValueError(f"h must be positive and finite, got {h!r}")
    if not math.isfinite(growth_f) or growth_f <= 1.0:
        raise ValueError(f"growth must be > 1.0 and finite, got {growth!r}")
    if max_nodes < 4:
        raise ValueError(f"max_nodes must be >= 4, got {max_nodes!r}")

    bps_r = _validate_breakpoints(breakpoints_r, 0.0, r_max_f, "breakpoints_r")
    bps_z = _validate_breakpoints(breakpoints_z, z_min_f, z_max_f, "breakpoints_z")

    if refinement_box is None:
        ref_r0, ref_r1, ref_z0, ref_z1 = 0.0, r_max_f, z_min_f, z_max_f
    else:
        ref_r0, ref_r1, ref_z0, ref_z1 = (float(v) for v in refinement_box)
        if not (0.0 <= ref_r0 < ref_r1 <= r_max_f):
            raise ValueError(
                f"refinement_box radial bounds ({ref_r0}, {ref_r1}) invalid for [0, {r_max_f}]"
            )
        if not (z_min_f <= ref_z0 < ref_z1 <= z_max_f):
            raise ValueError(
                f"refinement_box axial bounds ({ref_z0}, {ref_z1}) invalid for [{z_min_f}, {z_max_f}]"
            )

    return r_max_f, z_min_f, z_max_f, h_f, growth_f, ref_r0, ref_r1, ref_z0, ref_z1, bps_r, bps_z


def _prepare_rectangular_mesh(
    r_max: float,
    z_min: float,
    z_max: float,
    h: float,
    *,
    refinement_box: tuple[float, float, float, float] | None,
    breakpoints_r: Sequence[float],
    breakpoints_z: Sequence[float],
    growth: float,
    max_nodes: int,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    list[float],
    list[float],
    int,
    int,
]:
    (
        r_max_f,
        z_min_f,
        z_max_f,
        h_f,
        growth_f,
        ref_r0,
        ref_r1,
        ref_z0,
        ref_z1,
        bps_r,
        bps_z,
    ) = _validate_mesh_inputs(
        r_max,
        z_min,
        z_max,
        h,
        refinement_box=refinement_box,
        breakpoints_r=breakpoints_r,
        breakpoints_z=breakpoints_z,
        growth=growth,
        max_nodes=max_nodes,
    )
    nr = _count_axis_nodes(0.0, r_max_f, ref_r0, ref_r1, bps_r, h_f, growth_f)
    nz = _count_axis_nodes(z_min_f, z_max_f, ref_z0, ref_z1, bps_z, h_f, growth_f)
    total_nodes = nr * nz
    if total_nodes > max_nodes:
        raise MeshBudgetError(
            nodes=total_nodes,
            nr=nr,
            nz=nz,
            max_nodes=max_nodes,
            element_size=h_f,
        )
    return (
        r_max_f,
        z_min_f,
        z_max_f,
        h_f,
        growth_f,
        ref_r0,
        ref_r1,
        ref_z0,
        ref_z1,
        bps_r,
        bps_z,
        nr,
        nz,
    )


def check_rectangular_mesh_budget(
    r_max: float,
    z_min: float,
    z_max: float,
    h: float,
    *,
    refinement_box: tuple[float, float, float, float] | None = None,
    breakpoints_r: Sequence[float] = (),
    breakpoints_z: Sequence[float] = (),
    growth: float = 1.2,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> tuple[int, int]:
    """Return ``(nr, nz)`` if within ``max_nodes``, else raise :class:`MeshBudgetError`."""
    *_, nr, nz = _prepare_rectangular_mesh(
        r_max,
        z_min,
        z_max,
        h,
        refinement_box=refinement_box,
        breakpoints_r=breakpoints_r,
        breakpoints_z=breakpoints_z,
        growth=growth,
        max_nodes=max_nodes,
    )
    return nr, nz


def rectangular_mesh(
    r_max: float,
    z_min: float,
    z_max: float,
    h: float,
    *,
    refinement_box: tuple[float, float, float, float] | None = None,
    breakpoints_r: Sequence[float] = (),
    breakpoints_z: Sequence[float] = (),
    growth: float = 1.2,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> Mesh:
    """Build a graded axisymmetric tensor-product triangular mesh over ``[0, r_max] x [z_min, z_max]``.

    Raises ``MeshBudgetError`` (a ``ValueError`` subclass) if ``max_nodes`` is exceeded, and
    ``ValueError`` if bounds, step size, growth, or breakpoints are invalid (never silently coarsens).
    """
    (
        r_max_f,
        z_min_f,
        z_max_f,
        h_f,
        growth_f,
        ref_r0,
        ref_r1,
        ref_z0,
        ref_z1,
        bps_r,
        bps_z,
        nr,
        nz,
    ) = _prepare_rectangular_mesh(
        r_max,
        z_min,
        z_max,
        h,
        refinement_box=refinement_box,
        breakpoints_r=breakpoints_r,
        breakpoints_z=breakpoints_z,
        growth=growth,
        max_nodes=max_nodes,
    )

    r_axis = _build_axis(0.0, r_max_f, ref_r0, ref_r1, bps_r, h_f, growth_f)
    z_axis = _build_axis(z_min_f, z_max_f, ref_z0, ref_z1, bps_z, h_f, growth_f)
    assert int(r_axis.shape[0]) == nr and int(z_axis.shape[0]) == nz

    rr, zz = np.meshgrid(r_axis, z_axis, indexing="ij")
    nodes = np.column_stack((rr.ravel(), zz.ravel())).astype(np.float64)

    ii, jj = np.meshgrid(np.arange(nr - 1), np.arange(nz - 1), indexing="ij")
    n00 = (ii * nz + jj).ravel()
    n10 = ((ii + 1) * nz + jj).ravel()
    n11 = ((ii + 1) * nz + (jj + 1)).ravel()
    n01 = (ii * nz + (jj + 1)).ravel()

    backslash = (jj.ravel() % 2) == 1
    t0 = np.where(
        backslash[:, None],
        np.column_stack((n00, n10, n01)),
        np.column_stack((n00, n10, n11)),
    )
    t1 = np.where(
        backslash[:, None],
        np.column_stack((n10, n11, n01)),
        np.column_stack((n00, n11, n01)),
    )
    elements = np.vstack((t0, t1)).astype(np.int64)

    return Mesh(
        nodes=nodes,
        elements=elements,
        r_axis=r_axis,
        z_axis=z_axis,
        h_requested=h_f,
    )
