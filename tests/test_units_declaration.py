"""FR-003 unit system and frame declaration tests for ``magwave.case/2``."""

from __future__ import annotations

import copy
import json
import struct
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from magwave.analytic import cuboid_B
from magwave.benchmarks import dump_corpus, load_corpus
from magwave.benchmarks.checks import CHECKS
from magwave.cases import (
    SCHEMA,
    Case,
    CaseError,
    Coil,
    Dipole,
    ExpectedValue,
    FrameSpec,
    Magnet,
    Probe,
    UniformField,
    UnitSpec,
)
from magwave.units import MU0

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"


def _base_case_doc() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "name": "units_frame_base",
        "description": "Minimal valid magwave.case/2 document for unit/frame contract tests",
        "sources": ["unit test"],
        "tolerance": 1e-6,
        "units": {"length": "m", "field": "T", "current": "A", "force": "N"},
        "frame": {"axis": "z"},
        "components": [
            {
                "type": "magnet",
                "name": "cyl",
                "kind": "cylinder",
                "dims": {"radius": 0.01, "length": 0.02},
                "br": [0.0, 0.0, 1.315],
                "center": [0.0, 0.0, 0.0],
                "demag_curve": None,
            }
        ],
        "probes": [{"point": [0.0, 0.0, 0.03], "quantity": "B", "component": "z"}],
        "expected": [
            {
                "check": "cylinder_Bz_axis",
                "quantity": "Bz",
                "value": 0.048,
                "unit": "T",
                "tolerance": 1e-6,
                "source": "unit test",
            }
        ],
    }


def test_examples_and_corpus_roundtrip_byte_identically() -> None:
    example_paths = sorted(EXAMPLES_DIR.glob("*.json"))
    assert len(example_paths) == 5
    for path in example_paths:
        raw_bytes = path.read_bytes()
        raw_text = raw_bytes.decode("utf-8")
        loaded = Case.from_json(raw_text)
        reserialised = loaded.to_json()
        assert reserialised == raw_text
        assert reserialised.encode("utf-8") == raw_bytes
        assert loaded.units == UnitSpec(length="m", field="T", current="A", force="N")
        assert loaded.frame == FrameSpec(axis="z", datum_face="")

    corpus_path = ROOT / "src" / "magwave" / "benchmarks" / "cases.json"
    corpus_text = corpus_path.read_text(encoding="utf-8")
    cases = load_corpus(corpus_path)
    assert len(cases) == 27
    assert dump_corpus(cases) == corpus_text
    for case in cases:
        assert case.units == UnitSpec(length="m", field="T", current="A", force="N")
        assert case.frame == FrameSpec(axis="z", datum_face="")


def test_mm_mt_mn_seeded_property_roundtrip_is_byte_identical() -> None:
    """Seeded corpus in mm/mT/mN round-trips byte-identically through SI conversion."""
    rng = np.random.default_rng(20261125)

    def _pos_dec() -> float:
        return round(float(rng.uniform(0.05, 5000.0)), int(rng.integers(0, 7)))

    def _signed_dec() -> float:
        val = round(float(rng.uniform(-5000.0, 5000.0)), int(rng.integers(0, 7)))
        return 0.0 if val == 0.0 else val

    def _sig15() -> float:
        return float(f"{float(rng.uniform(1e-3, 5e3)):.15g}")

    edge_positives = (1015.82, 3.33, 0.1, 12.345, 0.001, 25.4, 999.999999)
    edge_signed = (-1015.82, 1015.82, -3.33, 3.33, 0.1, -0.2, 0.0)

    for idx in range(200):
        if idx < len(edge_positives):
            rad = edge_positives[idx]
            length = edge_positives[(idx + 1) % len(edge_positives)]
            cz = edge_signed[idx % len(edge_signed)]
            bz = edge_positives[(idx + 2) % len(edge_positives)]
            pz = edge_signed[(idx + 3) % len(edge_signed)]
            elem = edge_positives[(idx + 4) % len(edge_positives)]
            exp_b = edge_positives[(idx + 5) % len(edge_positives)]
            exp_f = edge_signed[(idx + 6) % len(edge_signed)]
        elif idx % 2 == 0:
            rad, length = _pos_dec(), _pos_dec()
            cz = _signed_dec()
            bz = _signed_dec()
            pz = _signed_dec()
            elem = _pos_dec()
            exp_b = _signed_dec()
            exp_f = _signed_dec()
        else:
            rad, length = _sig15(), _sig15()
            cz = _sig15() * (-1.0 if int(rng.integers(0, 2)) else 1.0)
            bz = _sig15() * (-1.0 if int(rng.integers(0, 2)) else 1.0)
            pz = _sig15() * (-1.0 if int(rng.integers(0, 2)) else 1.0)
            elem = _sig15()
            exp_b = _sig15()
            exp_f = _sig15()

        raw_doc: dict[str, Any] = {
            "schema": SCHEMA,
            "name": f"mm_prop_{idx}",
            "description": "Seeded mm/mT/mN round-trip property sample",
            "sources": ["property test"],
            "tolerance": 1e-6,
            "units": {"length": "mm", "field": "mT", "current": "A", "force": "mN"},
            "frame": {"axis": "z"},
            "mesh": {"element_size": elem},
            "components": [
                {
                    "type": "magnet",
                    "name": "m1",
                    "kind": "cylinder",
                    "dims": {"radius": rad, "length": length},
                    "br": [0.0, 0.0, bz],
                    "center": [0.0, 0.0, cz],
                    "demag_curve": None,
                }
            ],
            "probes": [{"point": [0.0, 0.0, pz], "quantity": "B", "component": "z"}],
            "expected": [
                {
                    "check": "cylinder_Bz_axis",
                    "quantity": "Bz",
                    "value": exp_b,
                    "unit": "mT",
                    "tolerance": 1e-6,
                    "source": "property test",
                },
                {
                    "check": "F_pointpole_cylinders",
                    "quantity": "Fz",
                    "value": exp_f,
                    "unit": "mN",
                    "tolerance": 1e-4,
                    "source": "property test",
                },
            ],
        }
        if idx % 3 == 0:
            raw_doc["frame"]["datum_face"] = f"face_{idx}"

        raw_text = json.dumps(raw_doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        loaded = Case.from_json(raw_text)

        mag = loaded.components[0]
        assert isinstance(mag, Magnet)
        assert mag.dims["radius"] == float(Decimal(str(rad)).scaleb(-3))
        assert mag.dims["length"] == float(Decimal(str(length)).scaleb(-3))
        assert mag.br[2] == float(Decimal(str(bz)).scaleb(-3))
        assert mag.center[2] == float(Decimal(str(cz)).scaleb(-3))
        assert loaded.probes[0].point[2] == float(Decimal(str(pz)).scaleb(-3))
        assert loaded.mesh is not None
        assert loaded.mesh.element_size == float(Decimal(str(elem)).scaleb(-3))
        assert loaded.expected[0].value == float(Decimal(str(exp_b)).scaleb(-3))
        assert loaded.expected[1].value == float(Decimal(str(exp_f)).scaleb(-3))

        assert loaded.to_json() == raw_text
        assert Case.from_json(loaded.to_json()) == loaded


def test_units_declaration_is_load_bearing_bit_identical_field() -> None:
    """Identical physical magnets in m/T, mm/T, and mm/mT produce bit-identical sampled B."""
    doc_m = _base_case_doc()
    doc_m["components"][0]["dims"] = {"radius": 0.01, "length": 0.02}
    doc_m["components"][0]["center"] = [0.0, 0.0, 0.005]
    doc_m["components"][0]["br"] = [0.0, 0.0, 1.315]
    doc_m["probes"][0]["point"] = [0.0, 0.0, 0.03]

    doc_mm = copy.deepcopy(doc_m)
    doc_mm["units"]["length"] = "mm"
    doc_mm["components"][0]["dims"] = {"radius": 10.0, "length": 20.0}
    doc_mm["components"][0]["center"] = [0.0, 0.0, 5.0]
    doc_mm["probes"][0]["point"] = [0.0, 0.0, 30.0]

    doc_mm_mt = copy.deepcopy(doc_mm)
    doc_mm_mt["units"]["field"] = "mT"
    doc_mm_mt["components"][0]["br"] = [0.0, 0.0, 1315.0]
    doc_mm_mt["expected"][0]["unit"] = "mT"
    doc_mm_mt["expected"][0]["value"] = 48.0

    case_m = Case.from_dict(doc_m)
    case_mm = Case.from_dict(doc_mm)
    case_mm_mt = Case.from_dict(doc_mm_mt)

    bz_m = CHECKS["cylinder_Bz_axis"](case_m)
    bz_mm = CHECKS["cylinder_Bz_axis"](case_mm)
    bz_mm_mt = CHECKS["cylinder_Bz_axis"](case_mm_mt)

    assert bz_m == bz_mm == bz_mm_mt
    assert struct.pack("!d", bz_m) == struct.pack("!d", bz_mm) == struct.pack("!d", bz_mm_mt)

    # A cosmetic implementation that ignores units.field and treats 1315 mT as 1315 T diverges 1000x.
    doc_cosmetic_field = copy.deepcopy(doc_mm_mt)
    doc_cosmetic_field["units"]["field"] = "T"
    doc_cosmetic_field["expected"][0]["unit"] = "T"
    assert CHECKS["cylinder_Bz_axis"](Case.from_dict(doc_cosmetic_field)) != bz_m

    # Loop coil (B ~ I / R, not scale-invariant): m vs mm produces bit-identical B,
    # whereas ignoring units.length diverges 1000x.
    loop_m = {
        "schema": SCHEMA,
        "name": "loop_scale_check",
        "description": "Coil loop m vs mm bit-identity check",
        "sources": ["unit test"],
        "tolerance": 1e-10,
        "units": {"length": "m", "field": "T", "current": "A", "force": "N"},
        "frame": {"axis": "z"},
        "components": [
            {
                "type": "coil",
                "name": "c1",
                "kind": "loop",
                "dims": {"radius": 0.02},
                "current": 2.0,
                "turns": 5,
                "center": [0.0, 0.0, 0.005],
            }
        ],
        "probes": [{"point": [0.004, 0.0, 0.015], "quantity": "B", "component": "z"}],
        "expected": [
            {
                "check": "loop_B",
                "quantity": "Bz",
                "value": 1e-4,
                "unit": "T",
                "tolerance": 1e-10,
                "source": "unit test",
            }
        ],
    }
    loop_mm = copy.deepcopy(loop_m)
    loop_mm["units"]["length"] = "mm"
    loop_mm["components"][0]["dims"]["radius"] = 20.0
    loop_mm["components"][0]["center"] = [0.0, 0.0, 5.0]
    loop_mm["probes"][0]["point"] = [4.0, 0.0, 15.0]

    loop_bz_m = CHECKS["loop_B"](Case.from_dict(loop_m))
    loop_bz_mm = CHECKS["loop_B"](Case.from_dict(loop_mm))
    assert loop_bz_m == loop_bz_mm
    assert struct.pack("!d", loop_bz_m) == struct.pack("!d", loop_bz_mm)

    loop_cosmetic_len = copy.deepcopy(loop_mm)
    loop_cosmetic_len["units"]["length"] = "m"
    assert CHECKS["loop_B"](Case.from_dict(loop_cosmetic_len)) != loop_bz_m

    # Cuboid magnet with non-power-of-two decimal dimensions: mm+mT vs m+T.
    cub_m = _base_case_doc()
    cub_m["components"] = [
        {
            "type": "magnet",
            "name": "cub",
            "kind": "cuboid",
            "dims": {"size_x": 0.0101582, "size_y": 0.02, "size_z": 0.00333},
            "br": [0.0, 0.0, 1.2],
            "center": [0.0, 0.0, 0.002],
            "demag_curve": None,
        }
    ]
    cub_m["probes"] = [{"point": [0.004, 0.005, 0.015], "quantity": "B", "component": "magnitude"}]

    cub_mm_mt = copy.deepcopy(cub_m)
    cub_mm_mt["units"] = {"length": "mm", "field": "mT", "current": "A", "force": "N"}
    cub_mm_mt["components"][0]["dims"] = {"size_x": 10.1582, "size_y": 20.0, "size_z": 3.33}
    cub_mm_mt["components"][0]["br"] = [0.0, 0.0, 1200.0]
    cub_mm_mt["components"][0]["center"] = [0.0, 0.0, 2.0]
    cub_mm_mt["probes"][0]["point"] = [4.0, 5.0, 15.0]
    cub_mm_mt["expected"][0]["unit"] = "mT"
    cub_mm_mt["expected"][0]["value"] = 48.0

    c_m = Case.from_dict(cub_m)
    c_mm = Case.from_dict(cub_mm_mt)
    mag_m = c_m.components[0]
    mag_mm = c_mm.components[0]
    assert isinstance(mag_m, Magnet) and isinstance(mag_mm, Magnet)

    def _cuboid_box(mag: Magnet) -> tuple[float, float, float, float, float, float]:
        hx = 0.5 * mag.dims["size_x"]
        hy = 0.5 * mag.dims["size_y"]
        hz = 0.5 * mag.dims["size_z"]
        return (
            mag.center[0] - hx,
            mag.center[0] + hx,
            mag.center[1] - hy,
            mag.center[1] + hy,
            mag.center[2] - hz,
            mag.center[2] + hz,
        )

    b_vec_m = cuboid_B(mag_m.br[2] / MU0, _cuboid_box(mag_m), c_m.probes[0].point)
    b_vec_mm = cuboid_B(mag_mm.br[2] / MU0, _cuboid_box(mag_mm), c_mm.probes[0].point)
    assert [struct.pack("!d", float(v)) for v in b_vec_m] == [
        struct.pack("!d", float(v)) for v in b_vec_mm
    ]


def test_mixed_units_rejected_with_exact_path_and_both_units() -> None:
    # Field unit contradiction at expected[0].
    doc = _base_case_doc()
    doc["expected"][0]["unit"] = "mT"
    with pytest.raises(CaseError) as excinfo:
        Case.from_dict(doc)
    assert str(excinfo.value) == "case.expected[0].unit: 'mT' contradicts case.units.field = 'T'"

    # Reverse field contradiction.
    doc_rev = _base_case_doc()
    doc_rev["units"]["field"] = "mT"
    doc_rev["expected"][0]["unit"] = "T"
    with pytest.raises(CaseError) as excinfo_rev:
        Case.from_dict(doc_rev)
    assert (
        str(excinfo_rev.value) == "case.expected[0].unit: 'T' contradicts case.units.field = 'mT'"
    )

    # Force unit contradiction at expected[1].
    doc_force = _base_case_doc()
    doc_force["expected"].append(
        {
            "check": "F_pointpole_cylinders",
            "quantity": "Fz",
            "value": 10.0,
            "unit": "mN",
            "tolerance": 1e-4,
            "source": "unit test",
        }
    )
    with pytest.raises(CaseError) as excinfo_f:
        Case.from_dict(doc_force)
    assert str(excinfo_f.value) == "case.expected[1].unit: 'mN' contradicts case.units.force = 'N'"

    # Quantity vs unit mismatch (field quantity given a force unit).
    doc_cross = _base_case_doc()
    doc_cross["expected"][0]["unit"] = "N"
    with pytest.raises(CaseError) as excinfo_cross:
        Case.from_dict(doc_cross)
    assert (
        str(excinfo_cross.value) == "case.expected[0].unit: 'N' contradicts case.units.field = 'T'"
    )

    # Direct Python Case construction also catches mixed units.
    with pytest.raises(CaseError) as excinfo_py:
        Case(
            name="py_mixed",
            description="direct python construction",
            sources=("unit test",),
            components=(Dipole(moment=(0.0, 0.0, 1.0), name="d1"),),
            probes=(Probe(point=(0.0, 0.0, 0.5), quantity="B", component="z"),),
            expected=(
                ExpectedValue(
                    check="dipole_B",
                    quantity="Bz",
                    value=1.0,
                    unit="mT",
                    tolerance=1e-6,
                    source="unit test",
                ),
            ),
            units=UnitSpec(length="m", field="T", current="A", force="N"),
        )
    assert str(excinfo_py.value) == "case.expected[0].unit: 'mT' contradicts case.units.field = 'T'"

    # Non-finite / non-numeric / non-positive geometry in mm names exact JSON path.
    for bad_val in (float("nan"), float("inf"), "10", True, -5.0):
        bad_dim_doc = _base_case_doc()
        bad_dim_doc["units"]["length"] = "mm"
        bad_dim_doc["components"][0]["dims"]["radius"] = bad_val
        with pytest.raises(CaseError) as bad_exc:
            Case.from_dict(bad_dim_doc)
        assert "case.components[0].dims.radius" in str(bad_exc.value)

    for bad_coord in (float("nan"), float("inf"), "0.5", False):
        bad_probe_doc = _base_case_doc()
        bad_probe_doc["units"]["length"] = "mm"
        bad_probe_doc["probes"][0]["point"][2] = bad_coord
        with pytest.raises(CaseError) as bad_p_exc:
            Case.from_dict(bad_probe_doc)
        assert "case.probes[0].point[2]" in str(bad_p_exc.value)


def test_missing_units_and_missing_frame_rejected_with_path() -> None:
    no_units = _base_case_doc()
    del no_units["units"]
    with pytest.raises(CaseError, match=r"case\.units"):
        Case.from_dict(no_units)

    no_frame = _base_case_doc()
    del no_frame["frame"]
    with pytest.raises(CaseError, match=r"case\.frame"):
        Case.from_dict(no_frame)

    for key in ("length", "field", "current", "force"):
        missing_sub = _base_case_doc()
        del missing_sub["units"][key]
        with pytest.raises(CaseError, match=r"case\.units"):
            Case.from_dict(missing_sub)

    missing_axis = _base_case_doc()
    del missing_axis["frame"]["axis"]
    with pytest.raises(CaseError, match=r"case\.frame"):
        Case.from_dict(missing_axis)


def test_unsupported_units_refused_with_path_and_supported_list() -> None:
    # Documented reason inches are out of scope: non-power-of-ten fails float round-trip.
    assert (3.33 * 0.0254) / 0.0254 == 3.3300000000000005

    for unit_key, bad_unit, expected_supported in (
        ("length", "in", ("m", "mm")),
        ("length", "cm", ("m", "mm")),
        ("field", "G", ("T", "mT")),
        ("current", "mA", ("A",)),
        ("force", "lbf", ("N", "mN")),
    ):
        doc = _base_case_doc()
        doc["units"][unit_key] = bad_unit
        with pytest.raises(CaseError) as excinfo:
            Case.from_dict(doc)
        msg = str(excinfo.value)
        assert f"case.units.{unit_key}" in msg
        assert f"'{bad_unit}'" in msg
        for item in expected_supported:
            assert item in msg

    bad_exp = _base_case_doc()
    bad_exp["expected"][0]["unit"] = "G"
    with pytest.raises(CaseError) as exp_exc:
        Case.from_dict(bad_exp)
    assert "case.expected[0].unit" in str(exp_exc.value)
    assert "'G'" in str(exp_exc.value)


def test_frame_axis_validation_and_datum_face_roundtrip() -> None:
    for bad_axis in ("x", "y"):
        doc = _base_case_doc()
        doc["frame"]["axis"] = bad_axis
        with pytest.raises(CaseError) as excinfo:
            Case.from_dict(doc)
        msg = str(excinfo.value)
        assert "case.frame.axis" in msg
        assert bad_axis in msg
        assert "only 'z' is supported at Tier 1" in msg

    unknown_axis = _base_case_doc()
    unknown_axis["frame"]["axis"] = "w"
    with pytest.raises(CaseError, match=r"case\.frame\.axis"):
        Case.from_dict(unknown_axis)

    doc_no_datum = _base_case_doc()
    loaded_no_datum = Case.from_dict(doc_no_datum)
    assert loaded_no_datum.frame == FrameSpec(axis="z", datum_face="")
    assert loaded_no_datum.to_dict()["frame"] == {"axis": "z"}

    doc_with_datum = _base_case_doc()
    doc_with_datum["frame"] = {"axis": "z", "datum_face": "bottom pole face"}
    raw = json.dumps(doc_with_datum, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    loaded_datum = Case.from_json(raw)
    assert loaded_datum.frame == FrameSpec(axis="z", datum_face="bottom pole face")
    assert loaded_datum.to_json() == raw

    bad_datum = _base_case_doc()
    bad_datum["frame"]["datum_face"] = 123
    with pytest.raises(CaseError, match=r"case\.frame\.datum_face"):
        Case.from_dict(bad_datum)


def test_schema_v1_document_refused_with_expected_schema_error() -> None:
    # Genuine magwave.case/1 document without units/frame must be refused on case.schema first.
    v1_doc = _base_case_doc()
    v1_doc["schema"] = "magwave.case/1"
    del v1_doc["units"]
    del v1_doc["frame"]
    with pytest.raises(CaseError) as excinfo:
        Case.from_dict(v1_doc)
    msg = str(excinfo.value)
    assert "case.schema" in msg
    assert "magwave.case/2" in msg
    assert "magwave.case/1" in msg

    v1_with_blocks = _base_case_doc()
    v1_with_blocks["schema"] = "magwave.case/1"
    with pytest.raises(CaseError) as excinfo2:
        Case.from_dict(v1_with_blocks)
    assert "magwave.case/2" in str(excinfo2.value)


def test_non_scaled_quantities_stay_si_under_mm_units() -> None:
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "name": "non_scaled_si",
        "description": "Verify coil.current, uniform_field.h0, and dipole.moment stay SI under mm",
        "sources": ["unit test"],
        "tolerance": 1e-6,
        "units": {"length": "mm", "field": "mT", "current": "A", "force": "N"},
        "frame": {"axis": "z"},
        "components": [
            {
                "type": "coil",
                "name": "c1",
                "kind": "loop",
                "dims": {"radius": 50.0},
                "current": 12.5,
                "turns": 10,
                "center": [0.0, 0.0, 10.0],
            },
            {
                "type": "dipole",
                "name": "d1",
                "moment": [0.0, 0.0, 2.5],
                "center": [0.0, 0.0, -10.0],
            },
            {
                "type": "uniform_field",
                "name": "u1",
                "h0": 10000.0,
            },
        ],
        "probes": [{"point": [0.0, 0.0, 20.0], "quantity": "B", "component": "z"}],
        "expected": [
            {
                "check": "loop_B",
                "quantity": "Bz",
                "value": 1.0,
                "unit": "mT",
                "tolerance": 1e-6,
                "source": "unit test",
            }
        ],
    }
    raw = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    loaded = Case.from_json(raw)

    coil, dipole, uniform = loaded.components
    assert isinstance(coil, Coil)
    assert isinstance(dipole, Dipole)
    assert isinstance(uniform, UniformField)
    assert coil.dims["radius"] == 0.05
    assert coil.center == (0.0, 0.0, 0.01)
    assert coil.current == 12.5
    assert dipole.moment == (0.0, 0.0, 2.5)
    assert dipole.center == (0.0, 0.0, -0.01)
    assert uniform.h0 == 10000.0
    assert loaded.probes[0].point == (0.0, 0.0, 0.02)
    assert loaded.expected[0].value == 0.001
    assert loaded.to_json() == raw
