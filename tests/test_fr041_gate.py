"""FR-041 gate tests: Tier 0 must refuse permeable media and still answer free-space fields."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from magwave.benchmarks import load_corpus
from magwave.cad.manifest import Body, Manifest
from magwave.cases import Case, ExpectedValue, Permeable, Probe, UniformField
from magwave.cli import main
from magwave.engines import EngineError, MagpylibEngine
from magwave.units import MU0


def _steel_sphere_case() -> Case:
    h0 = 1.0e4
    mu_r = 100.0
    analytic_bz = MU0 * mu_r * (3.0 / (mu_r + 2.0)) * h0
    return Case(
        name="fr041_steel_sphere",
        description="Permeable sphere in uniform applied field",
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
            UniformField(h0=h0, name="applied_field"),
        ),
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
        expected=(
            ExpectedValue(
                check="analytic_permeable_sphere_interior_Bz",
                quantity="Bz",
                value=analytic_bz,
                unit="T",
                tolerance=1e-2,
                source="Jackson §5.10",
            ),
        ),
    )


def test_tier0_refuses_a_permeable_component() -> None:
    engine = MagpylibEngine()
    case = _steel_sphere_case()
    errors = [problem for problem in engine.validate(case) if problem.is_error]
    assert len(errors) == 1
    assert errors[0].code == "E_UNSUPPORTED_PERMEABLE"
    msg = errors[0].message
    assert "Tier 1" in msg
    assert "stage 2" in msg
    assert "nothing was solved" in msg.lower()

    with pytest.raises(EngineError) as excinfo:
        engine.solve(case)
    assert [p.code for p in excinfo.value.problems] == ["E_UNSUPPORTED_PERMEABLE"]


def test_tier0_refuses_a_permeable_manifest_body() -> None:
    base = next(case for case in load_corpus() if case.name == "dipole_axis_Bz")
    manifest = Manifest(
        path="yoke.step",
        format="step",
        units="mm",
        bodies=[
            Body(
                index=0,
                name="yoke",
                role="permeable_solid",
                volume_mm3=1000.0,
                solids=1,
                bbox=(-10.0, -10.0, -5.0, 10.0, 10.0, 5.0),
                source="step",
            )
        ],
    )
    case = replace(base, name="dipole_with_manifest_steel", manifest=manifest)

    engine = MagpylibEngine()
    errors = [problem for problem in engine.validate(case) if problem.is_error]
    assert len(errors) == 1
    assert errors[0].code == "E_UNSUPPORTED_PERMEABLE"
    assert "Tier 1" in errors[0].message
    assert "stage 2" in errors[0].message
    assert "nothing was solved" in errors[0].message.lower()

    with pytest.raises(EngineError) as excinfo:
        engine.solve(case)
    assert [p.code for p in excinfo.value.problems] == ["E_UNSUPPORTED_PERMEABLE"]


def test_cli_solve_exits_nonzero_and_prints_no_numbers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_path = tmp_path / "steel_sphere.json"
    case_path.write_text(_steel_sphere_case().to_json(), encoding="utf-8")

    rc = main(["solve", str(case_path), "--points", "0,0,0"])
    captured = capsys.readouterr()

    assert rc == 2
    assert captured.out.strip() == ""
    assert not any(re.match(r"^\s*[-+0-9.]", line) for line in captured.out.splitlines())
    assert "E_UNSUPPORTED_PERMEABLE" in captured.err
    assert "permeable_solid" in captured.err
    assert "stage 2" in captured.err


def test_tier0_still_answers_uniform_field_without_steel() -> None:
    steel_case = _steel_sphere_case()
    h0 = 1.0e4
    air_case = replace(
        steel_case,
        name="fr041_uniform_field_only",
        components=tuple(c for c in steel_case.components if isinstance(c, UniformField)),
        expected=(replace(steel_case.expected[0], value=MU0 * h0),),
    )

    engine = MagpylibEngine()
    assert engine.validate(air_case) == []
    result = engine.solve(air_case)
    assert len(result.samples) == 1
    np.testing.assert_allclose(result.samples[0].vector, (0.0, 0.0, MU0 * h0), rtol=1e-12, atol=0.0)
    assert result.values()[0] == pytest.approx(MU0 * h0, rel=1e-12)
