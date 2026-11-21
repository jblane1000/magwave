"""Tier 0 Magpylib engine: cross-check against the corpus, skips, conventions, speed.

The engine path evaluates the same 27 cases, expected values and research-owned
tolerances as ``magwave bench``; cases the engine cannot model are skipped by its own
``validate()`` with a stated reason. The anti-vacuity test perturbs one expected value by
1 % and proves the engine path reports FAIL, in the style of ``test_benchmarks.py``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import magpylib as magpy
import numpy as np
import pytest

from magwave.benchmarks import (
    engine_all_passed,
    engine_summary,
    evaluate_with_engine,
    load_corpus,
    run_engine_benchmarks,
)
from magwave.cases import Case, Coil, Dipole, Magnet, Probe
from magwave.cli import main
from magwave.engines import Engine, EngineError, MagpylibEngine
from magwave.units import MU0

CASES = load_corpus()
BY_NAME = {case.name: case for case in CASES}
EXPECTED_SKIPS = {
    "halbach_bore_B": "E_UNSUPPORTED_GEOMETRY",
    "thin_solenoid_axis_Bz": "E_UNSUPPORTED_COIL",
    "thick_solenoid_axis_Bz": "E_UNSUPPORTED_COIL",
    "thin_solenoid_offcentre_Bz": "E_UNSUPPORTED_COIL",
    "thin_solenoid_outside_Bz": "E_UNSUPPORTED_COIL",
    "thick_solenoid_outside_Bz": "E_UNSUPPORTED_COIL",
    "force_dipole_coaxial_far": "E_UNSUPPORTED_QUANTITY",
    "force_pointpole_g5L": "E_UNSUPPORTED_QUANTITY",
    "force_dipoles_transverse": "E_UNSUPPORTED_QUANTITY",
    "force_dipoles_oblique": "E_UNSUPPORTED_QUANTITY",
}
ENGINE = MagpylibEngine()


def test_engine_satisfies_protocol_and_declares_capabilities() -> None:
    assert isinstance(ENGINE, Engine)
    caps = ENGINE.capabilities()
    assert caps.engine_id == "magpylib"
    assert caps.rung == "second-code"
    assert caps.engine_version == magpy.__version__
    assert caps.source_types == {"magnet", "coil", "dipole", "uniform_field"}
    assert caps.magnet_kinds == {"cuboid", "cylinder"}
    assert caps.coil_kinds == {"loop"}
    assert caps.quantities == {"B"}
    assert caps.force_methods == ()


def test_engine_bench_cross_checks_seventeen_and_skips_exactly_ten() -> None:
    rows = run_engine_benchmarks(CASES, ENGINE)
    assert len(rows) == len(CASES)
    skipped = {row.name: row.skip.code for row in rows if row.skipped}
    assert skipped == EXPECTED_SKIPS
    failures = [
        f"{r.name} ({r.rel_err:.2e} > {r.tolerance:.0e})" for r in rows if r.status == "FAIL"
    ]
    assert not failures, f"engine cross-check failures: {failures}"
    assert engine_summary(rows) == {
        "cross_checked": 17,
        "passed": 17,
        "failed": 0,
        "skipped": 10,
        "total": 27,
    }
    assert engine_all_passed(rows)


def test_engine_path_can_fail_a_perturbed_case() -> None:
    """A 1 % wrong expected value must fail through the engine, tolerance untouched."""
    case = BY_NAME["cuboid_axis_Bz"]
    item = case.expected[0]
    perturbed = replace(case, expected=(replace(item, value=item.value * 1.01),))
    assert perturbed.tolerance == case.tolerance
    rows = evaluate_with_engine(perturbed, ENGINE)
    assert [row.status for row in rows] == ["FAIL"]
    assert not engine_all_passed(rows)


def test_cli_engine_bench_summary_and_exit(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "--engine", "magpylib"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "17 cross-checked, 10 skipped"
    assert lines[-2] == "17/17 PASS"
    table = lines[2:-2]
    assert len(table) == len(CASES)
    for line, case in zip(table, CASES, strict=True):
        assert line.startswith(case.name)
        result = line.rsplit(" | ", 1)[1]
        if case.name in EXPECTED_SKIPS:
            assert result.startswith(f"SKIP: {EXPECTED_SKIPS[case.name]}: {case.name}: ")
        else:
            assert result == "PASS"


def test_cli_engine_bench_json_carries_engine_provenance(tmp_path: Path) -> None:
    target = tmp_path / "engine.json"
    assert main(["bench", "--engine", "magpylib", "--json", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["engine"] == {
        "id": "magpylib",
        "version": magpy.__version__,
        "rung": "second-code",
    }
    assert payload["summary"]["skipped"] == 10
    assert len(payload["results"]) == len(CASES)


def test_cli_engine_bench_exits_one_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import magwave.benchmarks as benchmarks

    case = BY_NAME["loop_axis_Bz"]
    item = case.expected[0]
    broken = [
        replace(c, expected=(replace(item, value=item.value * 1.01),)) if c is case else c
        for c in CASES
    ]
    monkeypatch.setattr(benchmarks, "load_corpus", lambda: broken)
    assert main(["bench", "--engine", "magpylib"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2] == "16/17 PASS"


def test_solve_raises_engine_error_for_unmodelled_case() -> None:
    with pytest.raises(EngineError) as excinfo:
        ENGINE.solve(BY_NAME["halbach_bore_B"])
    assert [p.code for p in excinfo.value.problems] == ["E_UNSUPPORTED_GEOMETRY"]
    with pytest.raises(EngineError):
        ENGINE.sample(BY_NAME["thin_solenoid_axis_Bz"], [(0.0, 0.0, 0.0)])


def test_validate_never_raises_for_out_of_remit_cases() -> None:
    for case in CASES:
        problems = ENGINE.validate(case)
        codes = {p.code for p in problems if p.severity == "error"}
        assert codes == ({EXPECTED_SKIPS[case.name]} if case.name in EXPECTED_SKIPS else set())


def test_demag_curve_is_refused_with_error() -> None:
    case = BY_NAME["cylinder_axis_Bz"]
    magnet = replace(case.components[0], demag_curve="N42-20C")
    demag_case = replace(case, components=(magnet,))
    errors = [p for p in ENGINE.validate(demag_case) if p.is_error]
    assert [p.code for p in errors] == ["E_UNSUPPORTED_DEMAG_CURVE"]
    msg = errors[0].message.lower()
    assert "rigidly polarised" in msg
    assert "second-quadrant" in msg
    assert "tier 1" in msg
    assert "fem2d" in msg

    with pytest.raises(EngineError) as excinfo:
        ENGINE.solve(demag_case)
    assert [p.code for p in excinfo.value.problems] == ["E_UNSUPPORTED_DEMAG_CURVE"]

    with pytest.raises(EngineError) as sample_exc:
        ENGINE.sample(demag_case, [(0.0, 0.0, 0.0)])
    assert [p.code for p in sample_exc.value.problems] == ["E_UNSUPPORTED_DEMAG_CURVE"]


def test_mu0_reconciliation_is_the_exact_constant_ratio() -> None:
    """Loops and dipoles are rescaled by exactly MU0/magpy.mu_0; magnets are untouched."""
    assert ENGINE.mu0_ratio == MU0 / magpy.mu_0
    point = np.array([[0.3, 0.0, 0.4]])
    loop = Coil(kind="loop", dims={"radius": 1.0}, current=2.5, turns=3, name="l1")
    raw = magpy.current.Circle(current=7.5, diameter=2.0).getB(point).reshape(1, 3)
    np.testing.assert_array_equal(ENGINE.sample(_case(loop), point), raw * (MU0 / magpy.mu_0))
    magnet = Magnet(
        kind="cuboid", dims={"size_x": 1, "size_y": 0.6, "size_z": 0.8}, br=(0, 0, 1), name="m1"
    )
    raw = (
        magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 0.6, 0.8))
        .getB(point)
        .reshape(1, 3)
    )
    np.testing.assert_array_equal(ENGINE.sample(_case(magnet), point), raw)
    dipole = Dipole(moment=(0.0, 0.2, 1.0), name="d1")
    raw = magpy.misc.Dipole(moment=(0.0, 0.2, 1.0)).getB(point).reshape(1, 3)
    np.testing.assert_array_equal(ENGINE.sample(_case(dipole), point), raw * (MU0 / magpy.mu_0))


def test_sources_superpose() -> None:
    """FR-032 AC: two sources summed equal the single combined case."""
    dipole = Dipole(moment=(0.1, 0.0, 1.0), center=(0.0, 0.2, -0.5), name="d1")
    magnet = Magnet(
        kind="cylinder",
        dims={"radius": 0.1, "length": 0.3},
        br=(0.2, 0.0, 1.1),
        center=(0, 0, 1),
        name="m1",
    )
    loop = Coil(
        kind="loop", dims={"radius": 0.4}, current=3.0, turns=5, center=(0.1, 0, 0), name="l1"
    )
    points = np.random.default_rng(7).uniform(-2.0, 2.0, size=(50, 3))
    parts = sum(ENGINE.sample(_case(c), points) for c in (dipole, magnet, loop))
    combined = ENGINE.sample(_case(dipole, magnet, loop), points)
    np.testing.assert_allclose(combined, parts, rtol=1e-13, atol=1e-18)


def test_sample_rejects_malformed_points() -> None:
    case = BY_NAME["dipole_axis_Bz"]
    with pytest.raises(ValueError):
        ENGINE.sample(case, [[0.0, 1.0]])
    with pytest.raises(ValueError):
        ENGINE.sample(case, [[0.0, math.nan, 1.0]])
    assert ENGINE.sample(case, (0.0, 0.0, 0.5)).shape == (1, 3)


def test_ten_thousand_points_within_100_ms(capsys: pytest.CaptureFixture[str]) -> None:
    """ROADMAP M2: 10^4 points <= 100 ms, on the cuboid_axis_Bz geometry."""
    case = BY_NAME["cuboid_axis_Bz"]
    rng = np.random.default_rng(2026)
    points = rng.uniform((-2.0, -2.0, -1.0), (2.0, 2.0, 4.0), size=(10_000, 3))
    ENGINE.sample(case, points)  # warm-up
    start = time.perf_counter()
    field = ENGINE.sample(case, points)
    elapsed_ms = (time.perf_counter() - start) * 1e3
    with capsys.disabled():  # shown even under plain `pytest -q`
        print(f"\n10000 points in {elapsed_ms:.1f} ms")
    assert field.shape == (10_000, 3)
    assert np.all(np.isfinite(field))
    assert elapsed_ms <= 100.0, f"10000 points took {elapsed_ms:.1f} ms (> 100 ms budget)"


def _case(*components) -> Case:
    return Case(
        name="adhoc",
        description="ad-hoc engine test case",
        sources=("test",),
        components=components,
        probes=(Probe(point=(0.0, 0.0, 5.0), quantity="B", component="z"),),
        expected=(BY_NAME["dipole_axis_Bz"].expected[0],),
    )
