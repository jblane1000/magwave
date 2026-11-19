"""Boolean operations and defeaturing tests (FR-012)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("OCP")

from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer

from magwave.cad import (
    DEFECT_KINDS,
    Body,
    DefeatureRecord,
    Defect,
    Manifest,
    RemovedFeature,
    boolean_subtract,
    boolean_union,
    defeature,
)
from tests.fixtures.cad_builders import build_filleted_block


def _count_solids(shape: object) -> int:
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    count = 0
    while exp.More():
        count += 1
        exp.Next()
    return count


def test_defeature_forty_fillets() -> None:
    """FR-012 AC: 40-fillet block defeats cleanly to 1 valid solid with volume delta reported."""
    result = defeature(build_filleted_block(), max_feature_size_mm=2.0)

    assert result.features_removed == 40
    assert len(result.removed_features) == 40
    assert all(item.surface_type == "Torus" for item in result.removed_features)
    assert BRepCheck_Analyzer(result.shape).IsValid()
    assert _count_solids(result.shape) == 1
    assert result.volume_after == pytest.approx(48000.0, rel=0.01)
    assert result.features_removed > 0
    assert result.volume_delta == pytest.approx(result.volume_after - result.volume_before)
    assert abs(result.volume_delta) > 1e-9
    # Fillets cut material from the top hole rims, so removing them restores the sharp edges
    # and increases the solid volume from ~47863.8785 to ~47864.2832 mm^3 (+0.4047 mm^3).
    assert result.volume_after > result.volume_before


def test_boolean_union_overlap() -> None:
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(5.0, 0.0, 0.0), 10.0, 10.0, 10.0).Shape()

    res = boolean_union([b1, b2])
    assert res.volume_before == pytest.approx(2000.0)
    assert res.volume_after == pytest.approx(1500.0, rel=1e-6)
    assert res.volume_delta == pytest.approx(-500.0, rel=1e-6)


def test_boolean_subtract_overlap() -> None:
    base = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).Shape()
    tool = BRepPrimAPI_MakeBox(gp_Pnt(5.0, 0.0, 0.0), 10.0, 10.0, 10.0).Shape()

    res = boolean_subtract(base, [tool])
    assert res.volume_before == pytest.approx(1000.0)
    assert res.volume_after == pytest.approx(500.0, rel=1e-6)
    assert res.volume_delta == pytest.approx(-500.0, rel=1e-6)


def test_boolean_union_three_shapes() -> None:
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(5.0, 0.0, 0.0), 10.0, 10.0, 10.0).Shape()
    b3 = BRepPrimAPI_MakeBox(gp_Pnt(10.0, 0.0, 0.0), 10.0, 10.0, 10.0).Shape()

    res = boolean_union([b1, b2, b3])
    assert res.volume_before == pytest.approx(3000.0)
    assert res.volume_after == pytest.approx(2000.0, rel=1e-6)
    assert res.volume_delta == pytest.approx(-1000.0, rel=1e-6)


def test_boolean_empty_inputs_raise_value_error() -> None:
    box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
    with pytest.raises(ValueError, match="shapes"):
        boolean_union([])
    with pytest.raises(ValueError, match="tools"):
        boolean_subtract(box, [])


def test_defeature_threshold_too_small_removes_nothing() -> None:
    res = defeature(build_filleted_block(), max_feature_size_mm=0.5)
    assert res.features_removed == 0
    assert res.removed_features == []
    assert res.volume_delta == pytest.approx(0.0, abs=1e-9)
    assert BRepCheck_Analyzer(res.shape).IsValid()


@pytest.mark.parametrize("bad_threshold", [0, -1.0, float("nan"), float("inf")])
def test_defeature_rejects_bad_threshold(bad_threshold: float) -> None:
    box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
    with pytest.raises(ValueError):
        defeature(box, max_feature_size_mm=bad_threshold)


def test_defeature_failed_kind_is_representable() -> None:
    assert "defeature_failed" in DEFECT_KINDS
    defect = Defect(kind="defeature_failed", detail="could not remove 2 faces", body_index=0)
    restored = Defect.from_dict(defect.to_dict())
    assert restored == defect


def test_manifest_defeature_backward_compatibility() -> None:
    plain = Manifest(
        path="part.step",
        format="step",
        units="mm",
        bodies=[
            Body(
                index=0,
                name="part",
                role=None,
                volume_mm3=1000.0,
                solids=1,
                bbox=(0.0, 0.0, 0.0, 10.0, 10.0, 10.0),
                source="step",
            )
        ],
    )
    assert "defeatures" not in plain.to_dict()

    legacy_dict = {
        "path": "part.step",
        "format": "step",
        "units": "mm",
        "bodies": [b.to_dict() for b in plain.bodies],
        "defects": [],
        "warnings": [],
    }
    assert Manifest.from_dict(legacy_dict) == plain

    with_defeature = Manifest(
        path="part.step",
        format="step",
        units="mm",
        bodies=plain.bodies,
        defeatures=[
            DefeatureRecord(
                body_index=0,
                threshold_mm=2.0,
                features_removed=1,
                volume_before_mm3=999.5,
                volume_after_mm3=1000.0,
                volume_delta_mm3=0.5,
                removed_features=[RemovedFeature(surface_type="Torus", size_mm=1.4)],
            )
        ],
    )
    roundtripped = Manifest.from_json(with_defeature.to_json())
    assert roundtripped == with_defeature


def test_lazy_ocp_guard_for_defeature_module() -> None:
    code = "import sys, magwave.cad.defeature; assert 'OCP' not in sys.modules"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert completed.returncode == 0, completed.stderr
