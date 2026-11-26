"""Schema round-trip, backward compatibility, and validation tests for permeable cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from magwave.cases import (
    SCHEMA,
    Case,
    CaseError,
    ExpectedValue,
    Permeable,
    Probe,
    UniformField,
)

ROOT = Path(__file__).resolve().parents[1]


def _sample_permeable_case(*, mu_r: float | None = 100.0) -> Case:
    return Case(
        name="permeable_roundtrip_demo",
        description="Permeable sphere with uniform applied field",
        sources=("unit test",),
        components=(
            Permeable(
                kind="sphere",
                dims={"radius": 0.01},
                center=(0.0, 0.0, 0.0),
                material="steel-team20",
                mu_r=mu_r,
                name="steel_sphere",
            ),
            UniformField(h0=1.0e4, name="applied_field"),
        ),
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
        expected=(
            ExpectedValue(
                check="analytic_permeable_sphere_interior_Bz",
                quantity="Bz",
                value=0.0369599,
                unit="T",
                tolerance=1e-2,
                source="unit test",
            ),
        ),
    )


def test_permeable_component_round_trips() -> None:
    case = _sample_permeable_case(mu_r=100.0)
    first_json = case.to_json()
    restored = Case.from_json(first_json)
    assert restored == case
    assert restored.to_json() == first_json

    no_override = _sample_permeable_case(mu_r=None)
    raw_first = no_override.to_json()
    assert Case.from_json(raw_first).to_json() == raw_first
    assert "mu_r" not in json.loads(raw_first)["components"][0]


def test_schema_files_still_load_unchanged() -> None:
    example_paths = sorted((ROOT / "examples").glob("*.json"))
    assert example_paths
    for path in example_paths:
        raw_text = path.read_text(encoding="utf-8")
        raw_obj = json.loads(raw_text)
        assert raw_obj["schema"] == SCHEMA == "magwave.case/2"
        loaded = Case.from_json(raw_text)
        assert json.loads(loaded.to_json()) == raw_obj

    corpus_path = ROOT / "src" / "magwave" / "benchmarks" / "cases.json"
    corpus_obj = json.loads(corpus_path.read_text(encoding="utf-8"))
    for raw_case in corpus_obj["cases"]:
        assert raw_case["schema"] == SCHEMA == "magwave.case/2"
        loaded = Case.from_dict(raw_case)
        assert json.loads(loaded.to_json()) == raw_case


def test_bad_permeable_is_rejected() -> None:
    with pytest.raises(CaseError, match="material"):
        Permeable(kind="sphere", dims={"radius": 0.01}, material="unobtanium")

    with pytest.raises(CaseError, match="mu_r"):
        Permeable(kind="sphere", dims={"radius": 0.01}, material="steel-team20", mu_r=0.0)

    with pytest.raises(CaseError, match="mu_r"):
        Permeable(kind="sphere", dims={"radius": 0.01}, material="steel-team20", mu_r=-5.0)

    with pytest.raises(CaseError, match="kind"):
        Permeable(kind="torus", dims={"radius": 0.01}, material="steel-team20")

    with pytest.raises(CaseError, match="radius"):
        Permeable(kind="sphere", dims={}, material="steel-team20")

    with pytest.raises(CaseError, match="h0"):
        UniformField(h0=0.0)

    with pytest.raises(CaseError, match="uniform_field"):
        Case(
            name="two_uniform_fields",
            description="reject duplicate uniform fields",
            sources=("unit test",),
            components=(
                UniformField(h0=100.0, name="u1"),
                UniformField(h0=200.0, name="u2"),
            ),
            probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
            expected=(
                ExpectedValue(
                    check="uniform_B",
                    quantity="Bz",
                    value=1e-4,
                    unit="T",
                    tolerance=1e-2,
                    source="unit test",
                ),
            ),
        )
