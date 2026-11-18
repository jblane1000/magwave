"""The benchmark suite must be green, and it must be able to fail.

A suite that cannot fail is theatre: these tests run the harness, prove every case passes,
and then perturb one expected value by 1 % to prove the harness reports FAIL and the CLI
exits 1 (FR-070 anti-vacuity, PRD §9).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from magwave.analytic import F_dipoles_general, cel, cuboid_B, cylinder_B
from magwave.benchmarks import (
    all_passed,
    corpus_path,
    dump_corpus,
    evaluate,
    format_table,
    load_corpus,
    results_payload,
    run_benchmarks,
)
from magwave.benchmarks.checks import CHECKS
from magwave.cases import CaseError
from magwave.cli import main

CASES = load_corpus()

EXPECTED_NAMES = (
    "dipole_axis_Bz",
    "dipole_offaxis_Bx",
    "dipole_offaxis_Bz",
    "loop_axis_Bz",
    "loop_offaxis_Bz",
    "loop_offaxis_Brho",
    "thin_solenoid_axis_Bz",
    "thick_solenoid_axis_Bz",
    "cuboid_axis_Bz",
    "cuboid_transverse_Bx",
    "cuboid_interior_Bz",
    "cylinder_axis_Bz",
    "halbach_bore_B",
    "force_dipole_coaxial_far",
    "force_pointpole_g5L",
    "dipole_tilted_Bx",
    "loop_offaxis_negz_By",
    "loop_near_wire_Bz",
    "thin_solenoid_offcentre_Bz",
    "thin_solenoid_outside_Bz",
    "thick_solenoid_outside_Bz",
    "cylinder_axis_outside_Bz",
    "cuboid_transverse_Bz",
    "force_dipoles_transverse",
    "force_dipoles_oblique",
    "cylinder_offaxis_Brho",
    "cylinder_offaxis_Bz",
)

FROZEN_BASELINE = (
    ("dipole_axis_Bz", 1.6000000000000001e-06),
    ("dipole_offaxis_Bx", 1.1520000000000002e-06),
    ("dipole_offaxis_Bz", 7.3600000000000034e-07),
    ("loop_axis_Bz", 6.2831853071795864e-07),
    ("loop_offaxis_Bz", 5.1067993442893440e-07),
    ("loop_offaxis_Brho", 8.6113352302538164e-08),
    ("thin_solenoid_axis_Bz", 0.00011239703569665162),
    ("thick_solenoid_axis_Bz", 0.89667512471228072),
    ("cuboid_axis_Bz", 0.018378605228966245),
    ("cuboid_transverse_Bx", 0.021946882352138448),
    ("cuboid_interior_Bz", 0.70050112298543643),
    ("cylinder_axis_Bz", 0.71086127010533862),
    ("halbach_bore_B", 1.0),
    ("force_dipole_coaxial_far", 4.3384700819103159e-05),
    ("force_pointpole_g5L", 0.0068239795918367506),
)


def test_corpus_has_at_least_twelve_cases() -> None:
    assert len(CASES) >= 12


def test_corpus_contains_wave_a_and_stretch_cases() -> None:
    assert len(CASES) == 27
    assert tuple(case.name for case in CASES) == EXPECTED_NAMES


def test_existing_fifteen_cases_reproduce_frozen_baseline() -> None:
    results = run_benchmarks(CASES[:15])
    for result, (name, baseline) in zip(results, FROZEN_BASELINE, strict=True):
        assert result.name == name
        assert abs(result.computed - baseline) / abs(baseline) <= 1e-12


def test_case_names_are_unique() -> None:
    names = [case.name for case in CASES]
    assert len(names) == len(set(names))


def test_every_expected_value_names_a_registered_check() -> None:
    for case in CASES:
        for item in case.expected:
            assert item.check in CHECKS, f"{case.name}: unknown check {item.check!r}"


def test_one_row_per_case() -> None:
    results = run_benchmarks(CASES)
    assert len(results) == len(CASES)


def test_all_cases_pass() -> None:
    results = run_benchmarks(CASES)
    failures = [f"{r.name} ({r.rel_err:.2e} > {r.tolerance:.0e})" for r in results if not r.passed]
    assert not failures, f"failing benchmark cases: {failures}"
    assert all_passed(results)


def test_harness_can_fail_a_perturbed_case() -> None:
    """A 1 % wrong expected value must fail, with the tolerance left untouched (FR-070)."""
    case = next(case for case in CASES if case.name == "dipole_axis_Bz")
    item = case.expected[0]
    perturbed = replace(case, expected=(replace(item, value=item.value * 1.01),))
    results = evaluate(perturbed)
    assert not results[0].passed
    assert not all_passed(results)


def test_table_reports_one_aligned_row_per_case() -> None:
    results = run_benchmarks(CASES)
    lines = format_table(results).splitlines()
    assert len(lines) == len(results) + 2  # header + rule + one row per case
    assert lines[0].startswith("case")
    for line, result in zip(lines[2:], results, strict=True):
        assert line.startswith(result.name)
        assert line.endswith("PASS")


def test_cli_bench_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench"]) == 0
    out = capsys.readouterr().out
    assert f"{len(CASES)}/{len(CASES)} PASS" in out


def test_cli_bench_writes_json(tmp_path: Path) -> None:
    target = tmp_path / "bench.json"
    assert main(["bench", "--json", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == "magwave.bench/1"
    assert payload["summary"] == {"passed": len(CASES), "failed": 0, "total": len(CASES)}
    assert len(payload["results"]) == len(CASES)
    for row in payload["results"]:
        assert row["passed"] is True
        assert {"name", "check", "computed", "expected", "rel_err", "tolerance"} <= set(row)


def test_results_payload_shape() -> None:
    results = run_benchmarks(CASES)
    payload = results_payload(results)
    assert len(payload["results"]) == len(results)


def test_committed_corpus_is_byte_stable() -> None:
    """Serialising the loaded corpus must reproduce the committed file exactly."""
    text = corpus_path().read_text(encoding="utf-8")
    assert dump_corpus(CASES) == text


def test_new_case_perturbation_fails_bench(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A 1 % perturbation of a new expected value must make `magwave bench` fail."""
    import magwave.benchmarks as benchmarks

    target = next(case for case in CASES if case.name == "cuboid_transverse_Bz")
    item = target.expected[0]
    perturbed = replace(target, expected=(replace(item, value=item.value * 1.01),))
    assert not evaluate(perturbed)[0].passed

    broken = [perturbed if case is target else case for case in CASES]
    monkeypatch.setattr(benchmarks, "load_corpus", lambda: broken)
    assert main(["bench"]) == 1
    out = capsys.readouterr().out
    assert f"{len(CASES) - 1}/{len(CASES)} PASS" in out
    assert "FAIL" in out


def test_analytic_general_cuboid_cylinder_and_dipole_force_edges() -> None:
    box = (-0.5, 0.5, -0.3, 0.3, 1.0, 1.8)
    pt = (0.6, 0.2, 2.5)
    np.testing.assert_array_equal(cuboid_B(8e5, box, pt), cuboid_B((0.0, 0.0, 8e5), box, pt))
    # Superposition over x, y, z magnetisation components
    total = cuboid_B((3e5, -4e5, 5e5), box, pt)
    parts = (
        cuboid_B((3e5, 0.0, 0.0), box, pt)
        + cuboid_B((0.0, -4e5, 0.0), box, pt)
        + cuboid_B((0.0, 0.0, 5e5), box, pt)
    )
    np.testing.assert_allclose(total, parts, rtol=1e-14, atol=1e-16)
    with pytest.raises(ValueError, match="scalar"):
        cuboid_B((1.0, 2.0), box, pt)

    # cylinder_B on-axis limit matches cylinder_Bz_axis; rim singularities rejected
    b_rho_axis, b_z_axis = cylinder_B(1.0, -1.0, 1.0, 8e5, 0.0, 2.5)
    assert b_rho_axis == 0.0
    assert b_z_axis == pytest.approx(0.06508055648801153, rel=1e-12)
    with pytest.raises(ValueError, match="rim singularity"):
        cylinder_B(1.0, -1.0, 1.0, 8e5, 1.0, 1.0)
    with pytest.raises(ValueError, match="non-zero"):
        cel(0.0, 1.0, 1.0, -1.0)

    # F_dipoles_general reduces to F_dipole_coaxial on axis and rejects zero separation
    f_ax = F_dipoles_general((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.5))
    assert f_ax[0] == 0.0 and f_ax[1] == 0.0
    assert f_ax[2] < 0.0  # attractive along -z for dipole 2 at +z
    with pytest.raises(ValueError, match="share a position"):
        F_dipoles_general((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))

    # checks.py rejects malformed geometry for new checks
    bad_force_case = replace(
        next(c for c in CASES if c.name == "force_dipoles_transverse"),
        probes=(
            replace(
                next(c for c in CASES if c.name == "force_dipoles_transverse").probes[0],
                point=(0.0, 0.0, 0.0),
            ),
        ),
    )
    with pytest.raises(CaseError, match="second dipole centre"):
        CHECKS["F_dipoles_general"](bad_force_case)
