"""Case JSON contract: lossless round-trip, and rejection of anything that would lie.

Three properties are load-bearing (PRD FR-001, NFR-8):

1. ``Case -> dict -> Case`` is lossless, floats included.
2. An unknown key is an error, not a silent drop — a dropped field produces a plausible
   wrong number.
3. A missing required field is an error, and a case may not state a tolerance looser than
   the expected values it contains (tolerances are research-owned).
"""

from __future__ import annotations

import json

import pytest

from magwave.benchmarks import dump_corpus, load_corpus
from magwave.cases import (
    SCHEMA,
    Case,
    CaseError,
    Coil,
    Dipole,
    ExpectedValue,
    Magnet,
    Probe,
)

CASES = load_corpus()


def _case_dict() -> dict:
    """A minimal valid case document, as JSON would deliver it."""
    return {
        "schema": SCHEMA,
        "name": "unit_case",
        "description": "hand-built case for the contract tests",
        "sources": ["unit test"],
        "tolerance": 1e-09,
        "components": [
            {
                "type": "dipole",
                "name": "m1",
                "moment": [0.0, 0.0, 1.0],
                "center": [0.0, 0.0, 0.0],
            }
        ],
        "probes": [{"point": [0.0, 0.0, 0.5], "quantity": "B", "component": "z"}],
        "expected": [
            {
                "check": "dipole_B",
                "quantity": "Bz",
                "value": 1.6e-06,
                "unit": "T",
                "tolerance": 1e-09,
                "source": "hand-derived",
            }
        ],
    }


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_corpus_case_roundtrip_is_lossless(case: Case) -> None:
    assert Case.from_dict(case.to_dict()) == case
    assert Case.from_json(case.to_json()) == case


def test_every_component_kind_roundtrips() -> None:
    components = (
        Magnet(
            kind="cuboid",
            dims={"size_x": 0.01, "size_y": 0.02, "size_z": 0.003},
            br=(0.0, 0.0, 1.2),
            center=(0.0, 0.0, 0.05),
            demag_curve="N42-20C-second-quadrant",
            name="block",
        ),
        Coil(
            kind="solenoid",
            dims={"inner_radius": 0.004, "outer_radius": 0.006, "length": 0.02},
            current=0.25,
            turns=700,
            center=(0.0, 0.0, 0.0),
            name="winding",
        ),
        Dipole(moment=(0.0, 0.0, 0.31), center=(0.0, 0.0, -0.02), name="far"),
    )
    case = Case(
        name="kinds",
        description="one of every component kind",
        sources=("unit test",),
        components=components,
        probes=(Probe(point=(0.0, 0.0, 0.1), quantity="B", component="magnitude"),),
        expected=(
            ExpectedValue(
                check="dipole_B",
                quantity="|B|",
                value=1.0,
                unit="T",
                tolerance=1e-06,
                source="unit test",
            ),
        ),
    )
    restored = Case.from_dict(case.to_dict())
    assert restored == case
    first = restored.components[0]
    assert isinstance(first, Magnet)
    assert first.demag_curve == "N42-20C-second-quadrant"


def test_awkward_floats_roundtrip_exactly() -> None:
    """repr-based JSON floats are lossless; this pins it rather than trusting it."""
    values = (0.1 + 0.2, 1e-17, 1.7976931348623157e308, 5e-324, 1.0 / 3.0, -0.0)
    for value in values:
        item = ExpectedValue(
            check="dipole_B",
            quantity="Bz",
            value=value,
            unit="T",
            tolerance=1e-09,
            source="unit test",
        )
        restored = ExpectedValue.from_dict(json.loads(json.dumps(item.to_dict())))
        assert restored == item
        assert repr(restored.value) == repr(item.value)


def test_to_dict_keys_are_sorted_at_every_level() -> None:
    case = Case.from_dict(_case_dict())
    document = case.to_dict()
    assert list(document) == sorted(document)
    for component in document["components"]:
        assert list(component) == sorted(component)
        if "dims" in component:
            assert list(component["dims"]) == sorted(component["dims"])
    for probe in document["probes"]:
        assert list(probe) == sorted(probe)
    for item in document["expected"]:
        assert list(item) == sorted(item)


def test_corpus_file_roundtrips_through_the_serialiser() -> None:
    payload = json.loads(dump_corpus(CASES))
    assert payload["schema"] == "magwave.benchmarks/1"
    assert len(payload["cases"]) == len(CASES)


def test_unknown_case_key_is_rejected() -> None:
    document = _case_dict()
    document["owner"] = "somebody"
    with pytest.raises(CaseError, match="unknown key"):
        Case.from_dict(document)


def test_unknown_component_key_is_rejected() -> None:
    document = _case_dict()
    document["components"][0]["colour"] = "blue"
    with pytest.raises(CaseError, match="unknown key"):
        Case.from_dict(document)


def test_unknown_probe_key_is_rejected() -> None:
    document = _case_dict()
    document["probes"][0]["weight"] = 1.0
    with pytest.raises(CaseError, match="unknown key"):
        Case.from_dict(document)


def test_unknown_expected_key_is_rejected() -> None:
    document = _case_dict()
    document["expected"][0]["confidence"] = 0.9
    with pytest.raises(CaseError, match="unknown key"):
        Case.from_dict(document)


@pytest.mark.parametrize("field", ["name", "description", "sources", "components", "probes"])
def test_missing_case_field_is_rejected(field: str) -> None:
    document = _case_dict()
    del document[field]
    with pytest.raises(CaseError, match="missing required field"):
        Case.from_dict(document)


def test_missing_component_field_is_rejected() -> None:
    document = _case_dict()
    del document["components"][0]["moment"]
    with pytest.raises(CaseError, match="missing required field"):
        Case.from_dict(document)


def test_missing_probe_field_is_rejected() -> None:
    document = _case_dict()
    del document["probes"][0]["quantity"]
    with pytest.raises(CaseError, match="missing required field"):
        Case.from_dict(document)


def test_missing_expected_field_is_rejected() -> None:
    document = _case_dict()
    del document["expected"][0]["tolerance"]
    with pytest.raises(CaseError, match="missing required field"):
        Case.from_dict(document)


def test_wrong_schema_is_rejected() -> None:
    document = _case_dict()
    document["schema"] = "magwave.case/99"
    with pytest.raises(CaseError, match="schema"):
        Case.from_dict(document)


def test_widened_case_tolerance_is_rejected() -> None:
    document = _case_dict()
    document["tolerance"] = 1e-03
    with pytest.raises(CaseError, match="may not be widened"):
        Case.from_dict(document)


def test_looser_expected_tolerance_than_case_tolerance_is_rejected() -> None:
    """The case gate is the tightest expected tolerance; stating a looser one is an error."""
    document = _case_dict()
    document["expected"][0]["tolerance"] = 1e-06
    with pytest.raises(CaseError, match="may not be widened"):
        Case.from_dict(document)


def test_unknown_magnet_kind_is_rejected() -> None:
    document = _case_dict()
    document["components"] = [
        {"type": "magnet", "kind": "sphere", "dims": {"radius": 0.01}, "br": [0, 0, 1.0]}
    ]
    with pytest.raises(CaseError, match="kind"):
        Case.from_dict(document)


def test_unknown_component_type_is_rejected() -> None:
    document = _case_dict()
    document["components"] = [{"type": "wire", "dims": {}}]
    with pytest.raises(CaseError, match="unknown component type"):
        Case.from_dict(document)


def test_extra_dimension_key_is_rejected() -> None:
    document = _case_dict()
    document["components"] = [
        {
            "type": "magnet",
            "kind": "cuboid",
            "dims": {"size_x": 0.01, "size_y": 0.01, "size_z": 0.01, "size_w": 0.01},
            "br": [0.0, 0.0, 1.0],
        }
    ]
    with pytest.raises(CaseError, match="unknown key"):
        Case.from_dict(document)


def test_non_positive_dimension_is_rejected() -> None:
    document = _case_dict()
    document["components"] = [
        {
            "type": "magnet",
            "kind": "cylinder",
            "dims": {"radius": 0.0, "length": 0.02},
            "br": [0.0, 0.0, 1.0],
        }
    ]
    with pytest.raises(CaseError, match="must be positive"):
        Case.from_dict(document)


def test_empty_case_lists_are_rejected() -> None:
    document = _case_dict()
    document["probes"] = []
    with pytest.raises(CaseError, match="at least one probe"):
        Case.from_dict(document)


def test_invalid_probe_component_is_rejected() -> None:
    document = _case_dict()
    document["probes"][0]["component"] = "theta"
    with pytest.raises(CaseError, match="component"):
        Case.from_dict(document)


def test_bad_json_is_rejected_with_a_case_error() -> None:
    with pytest.raises(CaseError, match="not valid JSON"):
        Case.from_json("{not json")
