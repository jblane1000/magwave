"""The benchmark suite must be green, and it must be able to fail.

A suite that cannot fail is theatre: these tests run the harness, prove every case passes,
and then perturb one expected value by 1 % to prove the harness reports FAIL and the CLI
exits 1 (FR-070 anti-vacuity, PRD §9).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

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
from magwave.cli import main

CASES = load_corpus()


def test_corpus_has_at_least_twelve_cases() -> None:
    assert len(CASES) >= 12


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
