"""Lazy OpenCASCADE (OCP) STEP and IGES importers (PRD FR-010, FR-011, ADR-0006).

All ``OCP`` imports are deferred until an import function is called so that importing
:mod:`magwave.cad` or :mod:`magwave.cad.step` never loads OCCT when the ``[step]`` extra
is absent or unused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .manifest import Body, DefeatureRecord, Defect, Manifest, RemovedFeature

_UNNAMED_SENTINELS = frozenset({"SOLID", "SHELL", "COMPOUND", "NONE"})
_OCCT_TRANSLATOR_PREFIX = "Open CASCADE STEP translator"


def _load_ocp() -> dict[str, Any]:
    """Import required OCP symbols lazily or raise :class:`MissingExtraError`."""
    from . import MissingExtraError

    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.IGESControl import IGESControl_Reader
        from OCP.Interface import Interface_Static
        from OCP.ShapeAnalysis import ShapeAnalysis_Shell
        from OCP.ShapeFix import ShapeFix_Shape
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDataStd import TDataStd_Name
        from OCP.TDF import TDF_Label
        from OCP.TDocStd import TDocStd_Document
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
        from OCP.TopExp import TopExp, TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.XCAFDoc import XCAFDoc_DocumentTool

        try:
            from OCP.TDF import TDF_LabelSequence
        except ImportError:
            from OCP.OCP.collections import Sequence_TDF_Label as TDF_LabelSequence

        try:
            from OCP.TColStd import TColStd_SequenceOfAsciiString
        except ImportError:
            from OCP.OCP.collections import (
                Sequence_TCollection_AsciiString as TColStd_SequenceOfAsciiString,
            )

        try:
            from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
        except ImportError:
            from OCP.OCP.collections import (
                IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as TopTools_IndexedDataMapOfShapeListOfShape,
            )
    except ImportError as exc:
        raise MissingExtraError("step", "cadquery-ocp") from exc

    return {
        "Bnd_Box": Bnd_Box,
        "BRepAlgoAPI_Common": BRepAlgoAPI_Common,
        "BRepBndLib": BRepBndLib,
        "BRepBuilderAPI_Sewing": BRepBuilderAPI_Sewing,
        "BRepGProp": BRepGProp,
        "GProp_GProps": GProp_GProps,
        "IFSelect_RetDone": IFSelect_RetDone,
        "IGESControl_Reader": IGESControl_Reader,
        "Interface_Static": Interface_Static,
        "ShapeAnalysis_Shell": ShapeAnalysis_Shell,
        "ShapeFix_Shape": ShapeFix_Shape,
        "STEPCAFControl_Reader": STEPCAFControl_Reader,
        "TCollection_ExtendedString": TCollection_ExtendedString,
        "TColStd_SequenceOfAsciiString": TColStd_SequenceOfAsciiString,
        "TDataStd_Name": TDataStd_Name,
        "TDF_Label": TDF_Label,
        "TDF_LabelSequence": TDF_LabelSequence,
        "TDocStd_Document": TDocStd_Document,
        "TopAbs_EDGE": TopAbs_EDGE,
        "TopAbs_FACE": TopAbs_FACE,
        "TopAbs_SOLID": TopAbs_SOLID,
        "TopExp": TopExp,
        "TopExp_Explorer": TopExp_Explorer,
        "TopLoc_Location": TopLoc_Location,
        "TopTools_IndexedDataMapOfShapeListOfShape": TopTools_IndexedDataMapOfShapeListOfShape,
        "XCAFDoc_DocumentTool": XCAFDoc_DocumentTool,
    }


def _label_name(ocp: dict[str, Any], lab: Any) -> str | None:
    attr = ocp["TDataStd_Name"]()
    if lab.FindAttribute(ocp["TDataStd_Name"].GetID_s(), attr):
        return str(attr.Get().ToExtString())
    return None


def _normalize_step_name(raw: str | None) -> tuple[str | None, str | None]:
    """Return ``(clean_name, unnamed_reason)`` for an XCAF label name."""
    if raw is None:
        return None, "missing body name in STEP metadata"
    stripped = raw.strip()
    if not stripped:
        return None, "empty body name in STEP metadata"
    if stripped.upper() in _UNNAMED_SENTINELS:
        return None, f"generic placeholder name {raw!r} in STEP metadata"
    if stripped.startswith(_OCCT_TRANSLATOR_PREFIX):
        return None, f"default translator name {raw!r} in STEP metadata"
    return stripped, None


def _count_solids(ocp: dict[str, Any], shape: Any) -> int:
    exp = ocp["TopExp_Explorer"](shape, ocp["TopAbs_SOLID"])
    count = 0
    while exp.More():
        count += 1
        exp.Next()
    return count


def _shape_volume(ocp: dict[str, Any], shape: Any) -> float:
    props = ocp["GProp_GProps"]()
    ocp["BRepGProp"].VolumeProperties_s(shape, props)
    return abs(float(props.Mass()))


def _shape_bbox(ocp: dict[str, Any], shape: Any) -> tuple[float, float, float, float, float, float]:
    box = ocp["Bnd_Box"]()
    ocp["BRepBndLib"].Add_s(shape, box)
    if box.IsVoid():
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pmin = box.CornerMin()
    pmax = box.CornerMax()
    return (
        float(pmin.X()),
        float(pmin.Y()),
        float(pmin.Z()),
        float(pmax.X()),
        float(pmax.Y()),
        float(pmax.Z()),
    )


def _check_shell_defects(ocp: dict[str, Any], shape: Any) -> tuple[bool, bool, int]:
    """Return ``(has_free_edges, has_bad_edges, non_manifold_edge_count)``."""
    analysis = ocp["ShapeAnalysis_Shell"]()
    analysis.LoadShells(shape)
    analysis.CheckOrientedShells(shape, True)
    has_free = bool(analysis.HasFreeEdges())
    has_bad = bool(analysis.HasBadEdges())

    edge_map = ocp["TopTools_IndexedDataMapOfShapeListOfShape"]()
    ocp["TopExp"].MapShapesAndAncestors_s(shape, ocp["TopAbs_EDGE"], ocp["TopAbs_FACE"], edge_map)
    nm_edges = 0
    for i in range(1, edge_map.Extent() + 1):
        if edge_map.FindFromIndex(i).Extent() > 2:
            nm_edges += 1
    return has_free, has_bad, nm_edges


def _bbox_overlaps(
    a: tuple[float, float, float, float, float, float],
    b: tuple[float, float, float, float, float, float],
    tol: float = 1e-6,
) -> bool:
    return (
        a[0] <= b[3] + tol
        and b[0] <= a[3] + tol
        and a[1] <= b[4] + tol
        and b[1] <= a[4] + tol
        and a[2] <= b[5] + tol
        and b[2] <= a[5] + tol
    )


def _step_file_units(ocp: dict[str, Any], reader: Any) -> tuple[str, list[str]]:
    """Inspect native STEP units while OCCT transfers into millimetres."""
    ocp["Interface_Static"].SetCVal_s("xstep.cascade.unit", "MM")
    lu = ocp["TColStd_SequenceOfAsciiString"]()
    au = ocp["TColStd_SequenceOfAsciiString"]()
    su = ocp["TColStd_SequenceOfAsciiString"]()
    reader.Reader().FileUnits(lu, au, su)
    names = [str(lu.Value(i).ToCString()).strip().lower() for i in range(1, lu.Length() + 1)]
    warnings: list[str] = []
    unique = sorted({n for n in names if n})
    if len(unique) > 1:
        warnings.append(
            f"STEP file declares multiple length units {unique}; converted to 'mm' on import."
        )
    elif len(unique) == 1 and unique[0] not in ("millimetre", "millimeter", "mm"):
        warnings.append(
            f"STEP file native length unit is {unique[0]!r}; converted to 'mm' on import."
        )
    return "mm", warnings


def import_step(
    path: str | Path,
    *,
    check_interference: bool = False,
    defeature_below: float | None = None,
) -> Manifest:
    """Import a STEP file via XCAF product-tree traversal (FR-010, FR-011, FR-012, ADR-0006)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"STEP file not found: {file_path}")

    ocp = _load_ocp()
    ocp["Interface_Static"].SetCVal_s("xstep.cascade.unit", "MM")

    reader = ocp["STEPCAFControl_Reader"]()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    status = reader.ReadFile(str(file_path))
    if status != ocp["IFSelect_RetDone"]:
        raise ValueError(f"failed to read STEP file {file_path}: status={status}")

    units, warnings = _step_file_units(ocp, reader)

    doc = ocp["TDocStd_Document"](ocp["TCollection_ExtendedString"]("XmlXCAF"))
    if not reader.Transfer(doc):
        raise ValueError(f"failed to transfer STEP document {file_path}")

    shape_tool = ocp["XCAFDoc_DocumentTool"].ShapeTool_s(doc.Main())
    tops = ocp["TDF_LabelSequence"]()
    shape_tool.GetFreeShapes(tops)

    bodies: list[Body] = []
    defects: list[Defect] = []
    body_shapes: list[Any] = []

    def walk(lab: Any, loc: Any) -> None:
        if shape_tool.IsAssembly_s(lab):
            comps = ocp["TDF_LabelSequence"]()
            shape_tool.GetComponents_s(lab, comps)
            if comps.Length() == 0:
                asm_name = _label_name(ocp, lab) or "unnamed assembly"
                defects.append(
                    Defect(
                        kind="skipped_body",
                        detail=f"assembly node {asm_name!r} has 0 components",
                        body_index=None,
                    )
                )
                return
            for i in range(1, comps.Length() + 1):
                comp_lab = comps.Value(i)
                comp_loc = shape_tool.GetLocation_s(comp_lab)
                next_loc = loc.Multiplied(comp_loc)
                referred = ocp["TDF_Label"]()
                if shape_tool.GetReferredShape_s(comp_lab, referred):
                    walk(referred, next_loc)
                else:
                    walk(comp_lab, next_loc)
            return

        raw_name = _label_name(ocp, lab)
        shp = shape_tool.GetShape_s(lab)
        if shp is None or shp.IsNull():
            defects.append(
                Defect(
                    kind="skipped_body",
                    detail=f"leaf {raw_name or 'unnamed'!r} has null geometry",
                    body_index=None,
                )
            )
            return

        if not loc.IsIdentity():
            shp = shp.Moved(loc)

        solids = _count_solids(ocp, shp)
        has_free, has_bad, nm_edges = _check_shell_defects(ocp, shp)

        if solids == 0:
            label_desc = raw_name or "unnamed"
            if has_free:
                defects.append(
                    Defect(
                        kind="open_shell",
                        detail=f"leaf {label_desc!r} has free edges (open shell)",
                        body_index=None,
                    )
                )
            if has_bad or nm_edges > 0:
                defects.append(
                    Defect(
                        kind="non_manifold_edge",
                        detail=f"leaf {label_desc!r} has {nm_edges} non-manifold/bad edges",
                        body_index=None,
                    )
                )
            defects.append(
                Defect(
                    kind="skipped_body",
                    detail=f"leaf {label_desc!r} contains 0 solids (open shell or surface geometry)",
                    body_index=None,
                )
            )
            return

        body_idx = len(bodies)
        clean_name, unnamed_reason = _normalize_step_name(raw_name)
        if unnamed_reason is not None:
            defects.append(
                Defect(
                    kind="unnamed_body",
                    detail=unnamed_reason,
                    body_index=body_idx,
                )
            )

        if has_free:
            defects.append(
                Defect(
                    kind="open_shell",
                    detail=f"body #{body_idx} ({clean_name or f'body_{body_idx}'}) has free edges",
                    body_index=body_idx,
                )
            )
        if has_bad or nm_edges > 0:
            defects.append(
                Defect(
                    kind="non_manifold_edge",
                    detail=(
                        f"body #{body_idx} ({clean_name or f'body_{body_idx}'}) has "
                        f"{nm_edges} non-manifold edge(s)"
                    ),
                    body_index=body_idx,
                )
            )
        if solids > 1:
            defects.append(
                Defect(
                    kind="merged_body",
                    detail=(
                        f"body #{body_idx} ({clean_name or f'body_{body_idx}'}) contains "
                        f"{solids} solids merged in one product"
                    ),
                    body_index=body_idx,
                )
            )

        volume = _shape_volume(ocp, shp)
        bbox = _shape_bbox(ocp, shp)
        bodies.append(
            Body(
                index=body_idx,
                name=clean_name,
                role=None,
                volume_mm3=volume,
                solids=solids,
                bbox=bbox,
                source="step",
            )
        )
        body_shapes.append(shp)

    identity = ocp["TopLoc_Location"]()
    if tops.Length() == 0:
        defects.append(
            Defect(
                kind="skipped_body",
                detail="STEP file contains no free shapes",
                body_index=None,
            )
        )
    else:
        for i in range(1, tops.Length() + 1):
            walk(tops.Value(i), identity)

    defeatures: list[DefeatureRecord] = []
    if defeature_below is not None:
        from .defeature import defeature

        for idx, shp in enumerate(body_shapes):
            res = defeature(shp, max_feature_size_mm=defeature_below)
            body_shapes[idx] = res.shape
            old_body = bodies[idx]
            bodies[idx] = Body(
                index=old_body.index,
                name=old_body.name,
                role=old_body.role,
                volume_mm3=res.volume_after,
                solids=old_body.solids,
                bbox=_shape_bbox(ocp, res.shape),
                source=old_body.source,
            )
            defeatures.append(
                DefeatureRecord(
                    body_index=old_body.index,
                    threshold_mm=res.threshold_mm,
                    features_removed=res.features_removed,
                    volume_before_mm3=res.volume_before,
                    volume_after_mm3=res.volume_after,
                    volume_delta_mm3=res.volume_delta,
                    removed_features=[
                        RemovedFeature(surface_type=item.surface_type, size_mm=item.size_mm)
                        for item in res.removed_features
                    ],
                )
            )
            for defect in res.defects:
                defects.append(
                    Defect(kind=defect.kind, detail=defect.detail, body_index=old_body.index)
                )

    if check_interference and len(body_shapes) > 1:
        for i in range(len(body_shapes)):
            for j in range(i + 1, len(body_shapes)):
                if not _bbox_overlaps(bodies[i].bbox, bodies[j].bbox):
                    continue
                common = ocp["BRepAlgoAPI_Common"](body_shapes[i], body_shapes[j]).Shape()
                if common.IsNull():
                    continue
                overlap = _shape_volume(ocp, common)
                if overlap > 1e-6:
                    defects.append(
                        Defect(
                            kind="interference",
                            detail=(
                                f"bodies #{i} ({bodies[i].display_name}) and "
                                f"#{j} ({bodies[j].display_name}) overlap by {overlap:.4f} mm^3"
                            ),
                            body_index=i,
                        )
                    )

    return Manifest(
        path=str(file_path),
        format="step",
        units=units,
        bodies=bodies,
        defects=defects,
        warnings=warnings,
        defeatures=defeatures,
    )


def import_iges(
    path: str | Path, *, defeature_below: float | None = None
) -> Manifest:
    """Best-effort IGES import with ShapeFix + sewing (FR-010, FR-012, ADR-0006)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"IGES file not found: {file_path}")

    ocp = _load_ocp()
    ocp["Interface_Static"].SetCVal_s("xstep.cascade.unit", "MM")

    reader = ocp["IGESControl_Reader"]()
    status = reader.ReadFile(str(file_path))
    if status != ocp["IFSelect_RetDone"]:
        raise ValueError(f"failed to read IGES file {file_path}: status={status}")
    reader.TransferRoots()
    raw = reader.OneShape()

    bodies: list[Body] = []
    defects: list[Defect] = []
    warnings: list[str] = []

    if raw is None or raw.IsNull():
        defects.append(
            Defect(
                kind="skipped_body",
                detail="IGES file yielded null geometry",
                body_index=None,
            )
        )
        return Manifest(
            path=str(file_path),
            format="iges",
            units="mm",
            bodies=bodies,
            defects=defects,
            warnings=warnings,
        )

    fixer = ocp["ShapeFix_Shape"](raw)
    fixer.Perform()
    fixed = fixer.Shape()

    sew = ocp["BRepBuilderAPI_Sewing"](1e-6)
    sew.Add(fixed)
    sew.Perform()
    healed = sew.SewedShape()

    solids = _count_solids(ocp, healed)
    volume = _shape_volume(ocp, healed)
    bbox = _shape_bbox(ocp, healed)
    has_free, has_bad, nm_edges = _check_shell_defects(ocp, healed)

    bodies.append(
        Body(
            index=0,
            name=None,
            role=None,
            volume_mm3=volume,
            solids=solids,
            bbox=bbox,
            source="iges",
        )
    )

    defeatures: list[DefeatureRecord] = []
    if defeature_below is not None:
        from .defeature import defeature

        res = defeature(healed, max_feature_size_mm=defeature_below)
        healed = res.shape
        old_body = bodies[0]
        bodies[0] = Body(
            index=old_body.index,
            name=old_body.name,
            role=old_body.role,
            volume_mm3=res.volume_after,
            solids=old_body.solids,
            bbox=_shape_bbox(ocp, healed),
            source=old_body.source,
        )
        defeatures.append(
            DefeatureRecord(
                body_index=0,
                threshold_mm=res.threshold_mm,
                features_removed=res.features_removed,
                volume_before_mm3=res.volume_before,
                volume_after_mm3=res.volume_after,
                volume_delta_mm3=res.volume_delta,
                removed_features=[
                    RemovedFeature(surface_type=item.surface_type, size_mm=item.size_mm)
                    for item in res.removed_features
                ],
            )
        )
        for defect in res.defects:
            defects.append(Defect(kind=defect.kind, detail=defect.detail, body_index=0))

    if solids == 0:
        defects.append(
            Defect(
                kind="imported_as_shell",
                detail="IGES geometry healed to shell with 0 solids; carrying healed volume",
                body_index=0,
            )
        )
        warnings.append("IGES geometry imported as shell (0 solids); no solid parity claimed.")

    if has_free:
        defects.append(
            Defect(
                kind="open_shell",
                detail="IGES healed shell has free edges",
                body_index=0,
            )
        )
    if has_bad or nm_edges > 0:
        defects.append(
            Defect(
                kind="non_manifold_edge",
                detail=f"IGES geometry has {nm_edges} non-manifold edge(s)",
                body_index=0,
            )
        )

    return Manifest(
        path=str(file_path),
        format="iges",
        units="mm",
        bodies=bodies,
        defects=defects,
        warnings=warnings,
        defeatures=defeatures,
    )
