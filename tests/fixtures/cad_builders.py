"""Runtime OCP and trimesh fixture generators for CAD import tests.

All generators build geometry at runtime inside ``tmp_path``; no binary CAD files are
committed to the repository, and heavy libraries are imported inside functions only.
"""

from __future__ import annotations

from pathlib import Path


def _ocp_xcaf_tools():
    try:
        from OCP.TDF import TDF_LabelSequence
    except ImportError:
        from OCP.OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    return {
        "IFSelect_RetDone": IFSelect_RetDone,
        "STEPCAFControl_Writer": STEPCAFControl_Writer,
        "TCollection_ExtendedString": TCollection_ExtendedString,
        "TDataStd_Name": TDataStd_Name,
        "TDF_LabelSequence": TDF_LabelSequence,
        "TDocStd_Document": TDocStd_Document,
        "TopLoc_Location": TopLoc_Location,
        "XCAFDoc_DocumentTool": XCAFDoc_DocumentTool,
    }


def _write_step_parts(
    out_path: Path, parts: list[tuple[str | None, object]], *, as_assembly: bool = True
) -> Path:
    ocp = _ocp_xcaf_tools()
    doc = ocp["TDocStd_Document"](ocp["TCollection_ExtendedString"]("XmlXCAF"))
    st = ocp["XCAFDoc_DocumentTool"].ShapeTool_s(doc.Main())

    if as_assembly:
        root = st.NewShape()
        ocp["TDataStd_Name"].Set_s(root, ocp["TCollection_ExtendedString"]("assembly"))
        for name, shape in parts:
            part_lab = st.AddShape(shape, False)
            if name is not None:
                ocp["TDataStd_Name"].Set_s(part_lab, ocp["TCollection_ExtendedString"](name))
            comp_lab = st.AddComponent(root, part_lab, ocp["TopLoc_Location"]())
            if name is not None:
                ocp["TDataStd_Name"].Set_s(comp_lab, ocp["TCollection_ExtendedString"](name))
        st.UpdateAssemblies()
    else:
        for name, shape in parts:
            lab = st.AddShape(shape, False)
            if name is not None:
                ocp["TDataStd_Name"].Set_s(lab, ocp["TCollection_ExtendedString"](name))

    writer = ocp["STEPCAFControl_Writer"]()
    writer.SetNameMode(True)
    writer.SetColorMode(True)
    if not writer.Transfer(doc):
        raise RuntimeError("STEPCAFControl_Writer.Transfer failed")
    status = writer.Write(str(out_path))
    if status != ocp["IFSelect_RetDone"]:
        raise RuntimeError(f"STEPCAFControl_Writer.Write failed: {status}")
    return out_path


def build_three_body_step(out_path: Path) -> Path:
    """Build the ADR-0006 3-body assembly STEP: box 20x10x5, washer OD16/ID6x2, cylinder D8x12."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    box = BRepPrimAPI_MakeBox(gp_Pnt(-10.0, -5.0, -2.5), 20.0, 10.0, 5.0).Shape()
    ax_w = gp_Ax2(gp_Pnt(0.0, 0.0, 2.5), gp_Dir(0.0, 0.0, 1.0))
    outer = BRepPrimAPI_MakeCylinder(ax_w, 8.0, 2.0).Shape()
    inner = BRepPrimAPI_MakeCylinder(ax_w, 3.0, 2.0).Shape()
    washer = BRepAlgoAPI_Cut(outer, inner).Shape()
    ax_b = gp_Ax2(gp_Pnt(0.0, 0.0, -6.0), gp_Dir(0.0, 0.0, 1.0))
    bobbin = BRepPrimAPI_MakeCylinder(ax_b, 4.0, 12.0).Shape()

    return _write_step_parts(
        out_path,
        [("magnet", box), ("washer", washer), ("bobbin", bobbin)],
        as_assembly=True,
    )


def build_open_shell_step(out_path: Path) -> Path:
    """Build a STEP file containing a 5-of-6-face open shell."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
    exp = TopExp_Explorer(box, TopAbs_FACE)
    faces = []
    while exp.More():
        faces.append(exp.Current())
        exp.Next()

    sew = BRepBuilderAPI_Sewing(1e-6)
    for face in faces[1:]:
        sew.Add(face)
    sew.Perform()
    open_shell = sew.SewedShape()
    return _write_step_parts(out_path, [("open_box", open_shell)], as_assembly=False)


def build_closed_box_step(out_path: Path, name: str | None = "closed_box") -> Path:
    """Build a single closed box STEP file with the given optional name."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
    return _write_step_parts(out_path, [(name, box)], as_assembly=False)


def build_non_manifold_step(out_path: Path) -> Path:
    """Build a STEP file with a non-manifold edge (3 faces meeting at one edge)."""
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_Sewing,
    )
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
    poly = BRepBuilderAPI_MakePolygon(
        gp_Pnt(0.0, 0.0, 0.0),
        gp_Pnt(10.0, 0.0, 0.0),
        gp_Pnt(10.0, -5.0, 0.0),
        gp_Pnt(0.0, -5.0, 0.0),
        True,
    )
    fin = BRepBuilderAPI_MakeFace(poly.Wire()).Face()

    sew = BRepBuilderAPI_Sewing(1e-6)
    sew.SetNonManifoldMode(True)
    exp = TopExp_Explorer(box, TopAbs_FACE)
    while exp.More():
        sew.Add(exp.Current())
        exp.Next()
    sew.Add(fin)
    sew.Perform()
    nm_shape = sew.SewedShape()
    return _write_step_parts(out_path, [("non_manifold_part", nm_shape)], as_assembly=False)


def build_merged_compound_step(out_path: Path) -> Path:
    """Build a STEP file where a single leaf product holds 2 merged solids."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Compound

    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 5.0, 5.0, 5.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(20.0, 0.0, 0.0), 5.0, 5.0, 5.0).Shape()
    comp = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(comp)
    builder.Add(comp, b1)
    builder.Add(comp, b2)
    return _write_step_parts(out_path, [("merged_pair", comp)], as_assembly=False)


def build_overlapping_step(out_path: Path) -> Path:
    """Build a 2-body STEP assembly whose boxes overlap by 500 mm^3."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(5.0, 0.0, 0.0), 10.0, 10.0, 10.0).Shape()
    return _write_step_parts(out_path, [("left", b1), ("right", b2)], as_assembly=True)


def build_box_iges(out_path: Path) -> Path:
    """Build a 12x8x4 mm box IGES file."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IGESControl import IGESControl_Writer

    box = BRepPrimAPI_MakeBox(12.0, 8.0, 4.0).Shape()
    writer = IGESControl_Writer()
    writer.AddShape(box)
    writer.ComputeModel()
    if not writer.Write(str(out_path)):
        raise RuntimeError(f"IGESControl_Writer.Write failed for {out_path}")
    return out_path


def build_box_stl(out_path: Path, extents: tuple[float, float, float] = (10.0, 8.0, 6.0)) -> Path:
    """Build a watertight box STL file via trimesh."""
    import trimesh

    mesh = trimesh.creation.box(extents=extents)
    mesh.export(str(out_path))
    return out_path


def build_two_body_glb(out_path: Path) -> Path:
    """Build a 2-geometry GLB scene with named parts via trimesh."""
    import trimesh

    scene = trimesh.Scene()
    m1 = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    m2 = trimesh.creation.box(extents=(4.0, 5.0, 6.0))
    m2.apply_translation([20.0, 0.0, 0.0])
    scene.add_geometry(m1, node_name="core", geom_name="core")
    scene.add_geometry(m2, node_name="yoke", geom_name="yoke")
    scene.export(str(out_path))
    return out_path


def build_filleted_block() -> object:
    """Build an 80x50x12 mm block with 40 through-holes and 40 R0.15 mm top-rim fillets."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.GeomAbs import GeomAbs_Circle
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    block = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 80.0, 50.0, 12.0).Shape()
    cylinders = []
    for i in range(8):
        for j in range(5):
            x = 10.0 + i * 8.6
            y = 9.0 + j * 8.0
            ax = gp_Ax2(gp_Pnt(x, y, -1.0), gp_Dir(0.0, 0.0, 1.0))
            cylinders.append(BRepPrimAPI_MakeCylinder(ax, 0.3, 14.0).Shape())

    fused = cylinders[0]
    for cyl in cylinders[1:]:
        fused = BRepAlgoAPI_Fuse(fused, cyl).Shape()
    cut = BRepAlgoAPI_Cut(block, fused).Shape()

    fillet = BRepFilletAPI_MakeFillet(cut)
    exp = TopExp_Explorer(cut, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge(exp.Current())
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() == GeomAbs_Circle:
            circ = curve.Circle()
            if abs(circ.Radius() - 0.3) < 1e-6 and abs(circ.Location().Z() - 12.0) < 1e-6:
                fillet.Add(0.15, edge)
        exp.Next()
    fillet.Build()
    if not fillet.IsDone():
        raise RuntimeError("BRepFilletAPI_MakeFillet failed for build_filleted_block")
    return fillet.Shape()


def build_filleted_step(out_path: Path) -> Path:
    """Write the 40-fillet block to a single-part STEP file."""
    return _write_step_parts(
        out_path, [("filleted_block", build_filleted_block())], as_assembly=False
    )


def build_corner_filleted_block(radius: float = 0.5) -> object:
    """80x50x12 mm block with the four vertical edges filleted at ``radius``."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    shape = BRepPrimAPI_MakeBox(80.0, 50.0, 12.0).Shape()
    fillet = BRepFilletAPI_MakeFillet(shape)
    chosen = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge(exp.Current())
        bnd = Bnd_Box()
        BRepBndLib.Add_s(edge, bnd)
        mn = bnd.CornerMin()
        mx = bnd.CornerMax()
        if (
            abs(mx.Z() - mn.Z()) > 11.0
            and abs(mx.X() - mn.X()) < 1e-6
            and abs(mx.Y() - mn.Y()) < 1e-6
            and not any(edge.IsSame(seen) for seen in chosen)
        ):
            chosen.append(edge)
            fillet.Add(radius, edge)
        exp.Next()
    fillet.Build()
    if not fillet.IsDone():
        raise RuntimeError("BRepFilletAPI_MakeFillet failed for build_corner_filleted_block")
    return fillet.Shape()
