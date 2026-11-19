"""Lazy OpenCASCADE (OCP) boolean operations and defeaturing (PRD FR-012, ADR-0006).

All ``OCP`` imports are deferred until a geometry function is called so that importing
:mod:`magwave.cad` or :mod:`magwave.cad.defeature` never loads OCCT when ``[step]`` is
absent or unused.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .manifest import Defect


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
        from OCP.BRepGProp import BRepGProp
        from OCP.GeomAbs import GeomAbs_Plane
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
        "BRepGProp": BRepGProp,
        "GeomAbs_Plane": GeomAbs_Plane,
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
    """Summary of a candidate or removed CAD face."""

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
        stype = ocp["BRepAdaptor_Surface"](face).GetType()
        if stype != ocp["GeomAbs_Plane"]:
            diag = _face_bbox_diagonal(ocp, face)
            if diag is not None and diag < max_feature_size_mm:
                candidates.append(
                    (
                        face,
                        FaceDescriptor(
                            surface_type=_surface_type_name(stype),
                            size_mm=diag,
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
    """Remove non-planar feature faces with bbox diagonal below ``max_feature_size_mm``."""
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
    seq = ocp["List_TopoDS_Shape"]()
    for face, _ in candidates:
        seq.Append(face)
    batch = ocp["BRepAlgoAPI_Defeaturing"]()
    batch.SetShape(shape)
    batch.AddFacesToRemove(seq)
    batch.SetRunParallel(False)
    batch.Build()
    if batch.IsDone():
        batch_shape = batch.Shape()
        if batch_shape is not None and not batch_shape.IsNull():
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

    # Strategy 2: fall back to per-face removal across bounded passes.
    current = shape
    removed: list[FaceDescriptor] = []
    defects: list[Defect] = []
    max_passes = 3
    for _ in range(max_passes):
        pass_candidates = _candidate_faces(ocp, current, threshold)
        if not pass_candidates:
            break
        progressed = False
        failed_in_pass: list[FaceDescriptor] = []
        for face, desc in pass_candidates:
            single = ocp["BRepAlgoAPI_Defeaturing"]()
            single.SetShape(current)
            single.AddFaceToRemove(face)
            single.SetRunParallel(False)
            single.Build()
            if single.IsDone():
                next_shape = single.Shape()
                if next_shape is not None and not next_shape.IsNull():
                    current = next_shape
                    removed.append(desc)
                    progressed = True
                    continue
            failed_in_pass.append(desc)
        if failed_in_pass and not progressed:
            defects.append(_make_defeature_defect(failed_in_pass, threshold))
            break
    else:
        remaining = _candidate_faces(ocp, current, threshold)
        if remaining:
            defects.append(_make_defeature_defect([desc for _, desc in remaining], threshold))

    _ensure_non_null(current, "defeature result")
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
