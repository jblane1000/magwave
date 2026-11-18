"""The engine protocol works without any solver, and a lying engine is caught (ADR-0003).

FR-040 AC: a dummy engine implementing the protocol runs end-to-end in tests. ADR-0003
consequence: "a lying engine is a defect class we test for with a deliberately wrong dummy
engine". Neither engine here touches Magpylib; the second one validates clean and returns
a wrong vector, and must be failed by the same harness path the Magpylib cross-check uses
(:func:`magwave.benchmarks.evaluate_with_engine`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from magwave.analytic import dipole_B
from magwave.benchmarks import (
    engine_all_passed,
    engine_summary,
    evaluate_with_engine,
    load_corpus,
    run_engine_benchmarks,
)
from magwave.cases import Case, Dipole
from magwave.engines.base import (
    Capabilities,
    Engine,
    EngineError,
    Problem,
    Result,
    Sample,
    errors_in,
    pick,
    raise_on_errors,
)

CASES = load_corpus()
DIPOLE_CASES = [case for case in CASES if case.name.startswith("dipole_")]


class DipoleOnlyEngine:
    """A minimal honest engine: closed-form point dipoles, B only, nothing else."""

    engine_id = "dummy-dipole"
    scale = 1.0

    def capabilities(self) -> Capabilities:
        return Capabilities(
            engine_id=self.engine_id,
            rung="test-double",
            source_types=frozenset({"dipole"}),
            magnet_kinds=frozenset(),
            coil_kinds=frozenset(),
            quantities=frozenset({"B"}),
            force_methods=(),
            engine_version="0",
        )

    def validate(self, case: Case) -> list[Problem]:
        problems = [
            Problem("E_UNSUPPORTED_SOURCE_TYPE", f"{case.name}: {type(c).__name__}", "error")
            for c in case.components
            if not isinstance(c, Dipole)
        ]
        problems += [
            Problem("E_UNSUPPORTED_QUANTITY", f"{case.name}: probe asks for {p.quantity}", "error")
            for p in case.probes
            if p.quantity != "B"
        ]
        return problems

    def solve(self, case: Case) -> Result:
        warnings = raise_on_errors(self.engine_id, self.validate(case))
        samples = []
        for probe in case.probes:
            total = np.zeros(3)
            for dipole in case.components:
                total += dipole_B(dipole.moment, np.subtract(probe.point, dipole.center))
            vector = tuple(float(v) * self.scale for v in total)
            samples.append(Sample(probe.point, "B", vector, "T", probe.component))
        return Result(
            engine_id=self.engine_id,
            engine_version="0",
            rung="test-double",
            method="closed-form point dipole (test double)",
            samples=tuple(samples),
            problems=tuple(warnings),
        )


class LyingEngine(DipoleOnlyEngine):
    """Validates clean, then returns a vector 1 % too large: the ADR-0003 defect class."""

    engine_id = "liar"
    scale = 1.01


class OffByOnePointEngine(DipoleOnlyEngine):
    """Returns a correct-looking field for the wrong point."""

    engine_id = "wrong-point"

    def solve(self, case: Case) -> Result:
        result = super().solve(case)
        moved = tuple(
            replace(s, point=(s.point[0], s.point[1], s.point[2] + 1.0)) for s in result.samples
        )
        return replace(result, samples=moved)


def test_dummy_engine_satisfies_the_protocol() -> None:
    assert isinstance(DipoleOnlyEngine(), Engine)


def test_dummy_engine_runs_end_to_end_over_the_corpus() -> None:
    engine = DipoleOnlyEngine()
    rows = run_engine_benchmarks(CASES, engine)
    assert len(rows) == len(CASES)
    passed = {row.name for row in rows if row.status == "PASS"}
    skipped = {row.name for row in rows if row.status == "SKIP"}
    assert passed == {case.name for case in DIPOLE_CASES}
    assert skipped == {case.name for case in CASES} - passed
    assert engine_all_passed(rows)
    for row in rows:
        if row.skipped:
            assert row.result_text.startswith(f"SKIP: {row.skip.code}: ")
            assert row.computed is None


def test_solve_refuses_a_case_validate_rejects() -> None:
    engine = DipoleOnlyEngine()
    case = next(case for case in CASES if case.name == "cylinder_axis_Bz")
    assert errors_in(engine.validate(case))
    with pytest.raises(EngineError) as excinfo:
        engine.solve(case)
    assert isinstance(excinfo.value, ValueError)
    assert [p.code for p in excinfo.value.problems] == ["E_UNSUPPORTED_SOURCE_TYPE"]


def test_lying_engine_is_failed_by_the_cross_check_path() -> None:
    liar = LyingEngine()
    case = DIPOLE_CASES[0]
    assert liar.validate(case) == []  # it claims it can model the case ...
    rows = evaluate_with_engine(case, liar)  # ... and the cross-check path catches it
    assert [row.status for row in rows] == ["FAIL"]
    assert rows[0].rel_err == pytest.approx(0.01, rel=1e-9)
    all_rows = run_engine_benchmarks(CASES, liar)
    assert not engine_all_passed(all_rows)
    assert engine_summary(all_rows)["failed"] == len(DIPOLE_CASES)


def test_engine_answering_the_wrong_point_is_rejected() -> None:
    with pytest.raises(EngineError) as excinfo:
        evaluate_with_engine(DIPOLE_CASES[0], OffByOnePointEngine())
    assert excinfo.value.problems[0].code == "E_RESULT_MISMATCH"


def test_an_all_skipped_run_is_not_a_pass() -> None:
    non_dipole = [case for case in CASES if case not in DIPOLE_CASES]
    rows = run_engine_benchmarks(non_dipole, DipoleOnlyEngine())
    assert engine_summary(rows)["cross_checked"] == 0
    assert not engine_all_passed(rows)


def test_result_rejects_error_problems_and_reports_components() -> None:
    sample = Sample((0.0, 0.0, 1.0), "B", (3.0, 4.0, 12.0), "T", "y")
    result = Result("e", "1", "r", "m", (sample,), (Problem("W_X", "a warning", "warning"),))
    assert result.values() == (4.0,)
    assert result.value(0, "x") == 3.0
    assert result.value(0, "magnitude") == 13.0
    with pytest.raises(ValueError):
        Result("e", "1", "r", "m", (sample,), (Problem("E_X", "an error", "error"),))
    with pytest.raises(ValueError):
        Problem("E_X", "bad severity", "fatal")
    with pytest.raises(ValueError):
        pick((1.0, 2.0, 3.0), "w")


def test_protocol_and_harness_import_without_magpylib() -> None:
    """The protocol is solver-free: importing it must not pull in Magpylib."""
    code = (
        "import sys, magwave.engines, magwave.engines.base, magwave.benchmarks, magwave.cli; "
        "assert 'magpylib' not in sys.modules, 'magpylib imported'"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert completed.returncode == 0, completed.stderr
