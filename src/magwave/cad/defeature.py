"""Lazy OpenCASCADE (OCP) boolean operations and defeaturing (PRD FR-012, ADR-0006).

All ``OCP`` imports are deferred until a geometry function is called so that importing
:mod:`magwave.cad` or :mod:`magwave.cad.defeature` never loads OCCT when ``[step]`` is
absent or unused.

Candidate feature faces are sized by characteristic local size rather than face extent:

* ``GeomAbs_Cylinder`` -> ``Cylinder().Radius()``
* ``GeomAbs_Torus`` -> ``Torus().MinorRadius()``
* ``GeomAbs_Sphere`` -> ``Sphere().Radius()``
* other non-planar surfaces -> bounding-box diagonal fallback
* ``GeomAbs_Plane`` -> excluded

Complete-revolution faces (cylinder ``u`` span ``>= 2*pi - _SPAN_TOL``, torus ``v`` span
``>= 2*pi - _SPAN_TOL``, sphere ``v`` span ``>= pi - _SPAN_TOL``) are bores, bosses, or
closed shells rather than blends and are excluded from candidate removal. Thus a threshold
of ``N`` mm removes blends whose radius (or other local characteristic size) is below ``N`` mm.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .manifest import Defect

_SPAN_TOL = 1e-6


def _load_ocp() -> dict[str, Any]:
    """Import required OCP symbols lazily or raise :class:`MissingExtraError`."""
    from . import MissingExtraError

    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepAlgoAPI import (
            BRepAlgoAPI_Cut,
            BRepAlgoAPI_Defeaturing,
            BRepAlgoAPI_Fuse,
        )
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepGProp import BRepGProp
        from OCP.GeomAbs import (
            GeomAbs_Cylinder,
            GeomAbs_Plane,
            GeomAbs_Sphere,
            GeomAbs_Torus,
        )
        from OCP.GProp import GProp_GProps
        from OCP.OCP.collections import List_TopoDS_Shape
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
    except ImportError as exc:
        raise MissingExtraError("step", "cadquery-ocp") from exc

    return {
        "Bnd_Box": Bnd_Box,
        "BRepAdaptor_Surface": BRepAdaptor_Surface,
        "BRepAlgoAPI_Cut": BRepAlgoAPI_Cut,
        "BRepAlgoAPI_Defeaturing": BRepAlgoAPI_Defeaturing,
        "BRepAlgoAPI_Fuse": BRepAlgoAPI_Fuse,
        "BRepBndLib": BRepBndLib,
        "BRepCheck_Analyzer": BRepCheck_Analyzer,
        "BRepGProp": BRepGProp,
        "GeomAbs_Cylinder": GeomAbs_Cylinder,
        "GeomAbs_Plane": GeomAbs_Plane,
        "GeomAbs_Sphere": GeomAbs_Sphere,
        "GeomAbs_Torus": GeomAbs_Torus,
        "GProp_GProps": GProp_GProps,
        "List_TopoDS_Shape": List_TopoDS_Shape,
        "TopAbs_FACE": TopAbs_FACE,
        "TopExp_Explorer": TopExp_Explorer,
        "TopoDS": TopoDS,
    }


@dataclass
class BooleanResult:
    """Result of a sequential OCCT boolean union or subtract operation."""

    shape: Any
    volume_before: float
    volume_after: float
    volume_delta: float


@dataclass
class FaceDescriptor:
    """Summary of a candidate or removed CAD face.

    ``size_mm`` is the characteristic local feature size in millimetres (radius or minor
    radius for cylinders, tori, and spheres; bounding-box diagonal fallback otherwise), not
    the face extent.
    """

    surface_type: str
    size_mm: float


@dataclass
class DefeatureResult:
    """Result of OCCT defeaturing with reported volume delta and any defects."""

    shape: Any
    threshold_mm: float
    features_removed: int
    removed_features: list[FaceDescriptor]
    volume_before: float
    volume_after: float
    volume_delta: float
    defects: list[Defect] = field(default_factory=list)


def _ensure_non_null(shape: Any, context: str) -> None:
    if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
        raise ValueError(f"{context}: shape is null")


def _shape_volume(ocp: dict[str, Any], shape: Any) -> float:
    props = ocp["GProp_GProps"]()
    ocp["BRepGProp"].VolumeProperties_s(shape, props)
    return abs(float(props.Mass()))


def _shape_is_valid(ocp: dict[str, Any], shape: Any) -> bool:
    """Return True when OCCT ``BRepCheck_Analyzer`` reports *shape* valid."""
    return bool(ocp["BRepCheck_Analyzer"](shape).IsValid())


def _surface_type_name(stype: Any) -> str:
    raw = getattr(stype, "name", str(stype))
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1]
    return raw.removeprefix("GeomAbs_")


def _face_bbox_diagonal(ocp: dict[str, Any], face: Any) -> float | None:
    box = ocp["Bnd_Box"]()
    ocp["BRepBndLib"].Add_s(face, box)
    if box.IsVoid():
        return None
    return float(box.CornerMin().Distance(box.CornerMax()))


def _candidate_faces(
    ocp: dict[str, Any], shape: Any, max_feature_size_mm: float
) -> list[tuple[Any, FaceDescriptor]]:
    candidates: list[tuple[Any, FaceDescriptor]] = []
    exp = ocp["TopExp_Explorer"](shape, ocp["TopAbs_FACE"])
    while exp.More():
        face = ocp["TopoDS"].Face(exp.Current())
        surface = ocp["BRepAdaptor_Surface"](face)
        stype = surface.GetType()
        if stype != ocp["GeomAbs_Plane"]:
            du = float(surface.LastUParameter() - surface.FirstUParameter())
            dv = float(surface.LastVParameter() - surface.FirstVParameter())
            size: float | None
            if stype == ocp["GeomAbs_Cylinder"]:
                if du >= 2.0 * math.pi - _SPAN_TOL:
                    exp.Next()
                    continue
                size = float(surface.Cylinder().Radius())
            elif stype == ocp["GeomAbs_Torus"]:
                if dv >= 2.0 * math.pi - _SPAN_TOL:
                    exp.Next()
                    continue
                size = float(surface.Torus().MinorRadius())
            elif stype == ocp["GeomAbs_Sphere"]:
                if dv >= math.pi - _SPAN_TOL:
                    exp.Next()
                    continue
                size = float(surface.Sphere().Radius())
            else:
                size = _face_bbox_diagonal(ocp, face)
            if size is not None and size < max_feature_size_mm:
                candidates.append(
                    (
                        face,
                        FaceDescriptor(
                            surface_type=_surface_type_name(stype),
                            size_mm=size,
                        ),
                    )
                )
        exp.Next()
    return candidates


def _make_defeature_defect(failed: list[FaceDescriptor], threshold_mm: float) -> Defect:
    sizes = [item.size_mm for item in failed]
    return Defect(
        kind="defeature_failed",
        detail=(
            f"could not remove {len(failed)} candidate feature face(s) below "
            f"{threshold_mm:.4f} mm (size range {min(sizes):.4f}..{max(sizes):.4f} mm)"
        ),
        body_index=None,
    )


def _defeature_shapes(ocp: dict[str, Any], shape: Any, faces: list[Any]) -> Any | None:
    """Run one BRepAlgoAPI_Defeaturing pass over *faces*; return the new shape or None."""
    seq = ocp["List_TopoDS_Shape"]()
    for face in faces:
        seq.Append(face)
    algo = ocp["BRepAlgoAPI_Defeaturing"]()
    algo.SetShape(shape)
    algo.AddFacesToRemove(seq)
    algo.SetRunParallel(False)
    algo.Build()
    if algo.IsDone():
        out = algo.Shape()
        if out is not None and not out.IsNull():
            return out
    return None


def boolean_union(shapes: Sequence[Any]) -> BooleanResult:
    """Fuse ``shapes`` sequentially into one solid/compound and report volume delta."""
    if not shapes:
        raise ValueError("shapes: must contain at least one shape")

    ocp = _load_ocp()
    for idx, shp in enumerate(shapes):
        _ensure_non_null(shp, f"shapes[{idx}]")

    volume_before = sum(_shape_volume(ocp, shp) for shp in shapes)
    result = shapes[0]
    for idx, shp in enumerate(shapes[1:], start=1):
        fused = ocp["BRepAlgoAPI_Fuse"](result, shp).Shape()
        _ensure_non_null(fused, f"boolean_union result at shapes[{idx}]")
        result = fused

    volume_after = _shape_volume(ocp, result)
    return BooleanResult(
        shape=result,
        volume_before=volume_before,
        volume_after=volume_after,
        volume_delta=volume_after - volume_before,
    )


def boolean_subtract(base: Any, tools: Sequence[Any]) -> BooleanResult:
    """Subtract ``tools`` sequentially from ``base`` and report volume delta."""
    _ensure_non_null(base, "base")
    if not tools:
        raise ValueError("tools: must contain at least one tool shape")

    ocp = _load_ocp()
    for idx, tool in enumerate(tools):
        _ensure_non_null(tool, f"tools[{idx}]")

    volume_before = _shape_volume(ocp, base)
    result = base
    for idx, tool in enumerate(tools):
        cut = ocp["BRepAlgoAPI_Cut"](result, tool).Shape()
        _ensure_non_null(cut, f"boolean_subtract result at tools[{idx}]")
        result = cut

    volume_after = _shape_volume(ocp, result)
    return BooleanResult(
        shape=result,
        volume_before=volume_before,
        volume_after=volume_after,
        volume_delta=volume_after - volume_before,
    )


def defeature(shape: Any, *, max_feature_size_mm: float) -> DefeatureResult:
    """Remove non-planar feature faces with characteristic local size below ``max_feature_size_mm``."""
    if isinstance(max_feature_size_mm, bool) or not isinstance(
        max_feature_size_mm, int | float
    ):
        raise ValueError(
            f"max_feature_size_mm: expected float, got {type(max_feature_size_mm).__name__}"
        )
    threshold = float(max_feature_size_mm)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(
            f"max_feature_size_mm: must be finite and > 0, got {max_feature_size_mm!r}"
        )

    _ensure_non_null(shape, "shape")
    ocp = _load_ocp()

    volume_before = _shape_volume(ocp, shape)
    candidates = _candidate_faces(ocp, shape, threshold)
    if not candidates:
        return DefeatureResult(
            shape=shape,
            threshold_mm=threshold,
            features_removed=0,
            removed_features=[],
            volume_before=volume_before,
            volume_after=volume_before,
            volume_delta=0.0,
            defects=[],
        )

    # Strategy 1: remove all candidate faces in one OCCT BRepAlgoAPI_Defeaturing pass.
    batch_shape = _defeature_shapes(ocp, shape, [face for face, _ in candidates])
    if batch_shape is not None and _shape_is_valid(ocp, batch_shape):
        volume_after = _shape_volume(ocp, batch_shape)
        removed_descriptors = [desc for _, desc in candidates]
        return DefeatureResult(
            shape=batch_shape,
            threshold_mm=threshold,
            features_removed=len(removed_descriptors),
            removed_features=removed_descriptors,
            volume_before=volume_before,
            volume_after=volume_after,
            volume_delta=volume_after - volume_before,
            defects=[],
        )

    # Strategy 2: fall back to per-face removal, re-fetching handles from *current* each step.
    current = shape
    removed: list[FaceDescriptor] = []
    defects: list[Defect] = []
    attempts_left = max(2 * len(candidates), 8)
    while attempts_left > 0:
        fresh = _candidate_faces(ocp, current, threshold)
        if not fresh:
            break
        progressed = False
        for idx in range(len(fresh)):
            if attempts_left <= 0:
                break
            fresh = _candidate_faces(ocp, current, threshold)
            if idx >= len(fresh):
                break
            face, desc = fresh[idx]
            attempts_left -= 1
            nxt = _defeature_shapes(ocp, current, [face])
            if nxt is not None:
                current = nxt
                removed.append(desc)
                progressed = True
                break
        if not progressed:
            break

    _ensure_non_null(current, "defeature result")
    if not _shape_is_valid(ocp, current):
        discarded = len(removed) if removed else len(candidates)
        return DefeatureResult(
            shape=shape,
            threshold_mm=threshold,
            features_removed=0,
            removed_features=[],
            volume_before=volume_before,
            volume_after=volume_before,
            volume_delta=0.0,
            defects=[
                Defect(
                    kind="defeature_invalid",
                    detail=(
                        f"defeatured shape failed BRepCheck_Analyzer validity check; "
                        f"discarded {discarded} candidate feature removal(s) below "
                        f"{threshold:.4f} mm"
                    ),
                    body_index=None,
                )
            ],
        )

    remaining = _candidate_faces(ocp, current, threshold)
    if remaining:
        defects.append(_make_defeature_defect([desc for _, desc in remaining], threshold))

    volume_after = _shape_volume(ocp, current)
    return DefeatureResult(
        shape=current,
        threshold_mm=threshold,
        features_removed=len(removed),
        removed_features=removed,
        volume_before=volume_before,
        volume_after=volume_after,
        volume_delta=volume_after - volume_before,
        defects=defects,
    )
