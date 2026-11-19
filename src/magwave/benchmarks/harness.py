"""Benchmark harness: evaluate the analytic corpus and report one aligned row per case.

``python -m magwave bench`` prints the table below and exits 1 if any case fails, so CI
fails on a wrong number rather than on a written claim (FR-070, NFR-8).

    case                               |         computed |         expected |    rel.err |      tol | result

The pass rule is the research file's: ``|computed - expected| <= tolerance * |expected|``,
with the tolerance taken from the case — research-owned, never adjusted here.

``python -m magwave bench --engine magpylib`` evaluates the same corpus, expected values and
tolerances through an :class:`~magwave.engines.base.Engine` instead of ``checks.py``
(:func:`run_engine_benchmarks`). A case the engine's ``validate()`` rejects is reported as
``SKIP: <CODE>: <reason>`` — one row per expected value either way, never dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..cases import Case, CaseError
from ..engines.base import Engine, EngineError, Problem, errors_in
from .checks import CHECKS

CORPUS_SCHEMA = "magwave.benchmarks/1"
RESULTS_SCHEMA = "magwave.bench/1"

#: Guards against a zero expected value turning the relative error into 0/0.
_RELATIVE_FLOOR = 1e-30


@dataclass(frozen=True)
class CaseResult:
    """The outcome of one expected value in one case."""

    name: str
    check: str
    quantity: str
    computed: float
    expected: float
    tolerance: float
    rel_err: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in sorted(
                (
                    "name",
                    "check",
                    "quantity",
                    "computed",
                    "expected",
                    "tolerance",
                    "rel_err",
                    "passed",
                )
            )
        }


def _compare(computed: float, expected: float, tolerance: float) -> tuple[float, bool]:
    """The research file's pass rule: return ``(rel_err, passed)``.

    Shared by the analytic path and the engine path so both judge a number identically.
    """
    difference = abs(computed - expected)
    scale = max(abs(expected), _RELATIVE_FLOOR)
    rel_err = difference / abs(expected) if expected else difference
    return rel_err, difference <= tolerance * scale


def evaluate(case: Case) -> list[CaseResult]:
    """Evaluate every expected value in ``case``; raise on an unknown check id."""
    results: list[CaseResult] = []
    for item in case.expected:
        function = CHECKS.get(item.check)
        if function is None:
            raise CaseError(
                f"{case.name}: unknown check {item.check!r}; known checks are {sorted(CHECKS)}"
            )
        computed = float(function(case))
        rel_err, passed = _compare(computed, item.value, item.tolerance)
        results.append(
            CaseResult(
                name=case.name,
                check=item.check,
                quantity=item.quantity,
                computed=computed,
                expected=item.value,
                tolerance=item.tolerance,
                rel_err=rel_err,
                passed=passed,
            )
        )
    return results


def run_benchmarks(cases: list[Case]) -> list[CaseResult]:
    """Evaluate a corpus in order, one result per expected value."""
    return [result for case in cases for result in evaluate(case)]


def all_passed(results: list[CaseResult]) -> bool:
    return all(result.passed for result in results)


def _table_header() -> list[str]:
    header = (
        f"{'case':<32} | {'computed':>16} | {'expected':>16} | {'rel.err':>9} | {'tol':>8} | result"
    )
    return [header, "-" * len(header)]


def format_table(results: list[CaseResult]) -> str:
    """Render the aligned table: name | computed | expected | rel.err | tol | PASS/FAIL."""
    lines = _table_header()
    for result in results:
        lines.append(
            f"{result.name:<32} | {result.computed:>16.9e} | {result.expected:>16.9e} | "
            f"{result.rel_err:>9.2e} | {result.tolerance:>8.0e} | "
            f"{'PASS' if result.passed else 'FAIL'}"
        )
    return "\n".join(lines)


def corpus_path() -> Path:
    """The committed corpus that ships inside the package."""
    return Path(__file__).with_name("cases.json")


def load_corpus(path: str | Path | None = None) -> list[Case]:
    """Load and validate the benchmark corpus.

    Rejects a wrong corpus schema, a non-list payload and duplicate case names: two cases
    with one name would make a results table ambiguous.
    """
    source = Path(path) if path is not None else corpus_path()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CaseError(f"{source}: not valid JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise CaseError(f"{source}: expected a JSON object with 'schema' and 'cases'")
    unknown = sorted(set(payload) - {"schema", "cases"})
    if unknown:
        raise CaseError(f"{source}: unknown key(s) {', '.join(map(repr, unknown))}")
    if payload.get("schema") != CORPUS_SCHEMA:
        raise CaseError(
            f"{source}: expected schema {CORPUS_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise CaseError(f"{source}: 'cases' must be a list")
    cases = [Case.from_dict(item) for item in raw_cases]
    names = [case.name for case in cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise CaseError(f"{source}: duplicate case name(s) {', '.join(map(repr, duplicates))}")
    return cases


def dump_corpus(cases: list[Case]) -> str:
    """Serialise a corpus in the committed on-disk form (sorted keys, two-space indent)."""
    payload = {"schema": CORPUS_SCHEMA, "cases": [case.to_dict() for case in cases]}
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def results_payload(results: list[CaseResult]) -> dict[str, Any]:
    """The ``--json`` document: results plus a summary that a CI step can read."""
    passed = sum(1 for result in results if result.passed)
    return {
        "schema": RESULTS_SCHEMA,
        "results": [result.to_dict() for result in results],
        "summary": {"passed": passed, "failed": len(results) - passed, "total": len(results)},
    }


# --------------------------------------------------------------------------------------
# engine path: the same corpus, expected values and tolerances, through an Engine
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EngineCaseResult:
    """One expected value evaluated through an engine — PASS, FAIL or SKIP.

    A SKIP is produced only by the engine's own ``validate()`` returning an error problem
    (``skip``); it carries that problem's code and reason so no case is dropped silently.
    """

    name: str
    check: str
    quantity: str
    expected: float
    tolerance: float
    computed: float | None
    rel_err: float | None
    passed: bool | None
    skip: Problem | None = None

    @property
    def skipped(self) -> bool:
        return self.skip is not None

    @property
    def status(self) -> str:
        if self.skip is not None:
            return "SKIP"
        return "PASS" if self.passed else "FAIL"

    @property
    def result_text(self) -> str:
        if self.skip is not None:
            return f"SKIP: {self.skip.code}: {self.skip.message}"
        return self.status

    def to_dict(self) -> dict[str, Any]:
        skip = None if self.skip is None else {"code": self.skip.code, "reason": self.skip.message}
        data = {
            "name": self.name,
            "check": self.check,
            "quantity": self.quantity,
            "computed": self.computed,
            "expected": self.expected,
            "tolerance": self.tolerance,
            "rel_err": self.rel_err,
            "passed": self.passed,
            "status": self.status,
            "skip": skip,
        }
        return {key: data[key] for key in sorted(data)}


def _probe_index(case: Case, position: int) -> int:
    """Which probe expected value ``position`` refers to: the sole probe, or by index."""
    if len(case.probes) == 1:
        return 0
    if len(case.probes) == len(case.expected):
        return position
    raise CaseError(
        f"{case.name}: cannot pair {len(case.expected)} expected value(s) with "
        f"{len(case.probes)} probes; use one probe, or one probe per expected value"
    )


def evaluate_with_engine(case: Case, engine: Engine) -> list[EngineCaseResult]:
    """Evaluate every expected value of ``case`` through ``engine``.

    The engine's ``validate()`` decides SKIP; otherwise ``solve()`` produces the number and
    the research file's pass rule (:func:`_compare`) judges it against the case's own
    tolerance. The result is checked against the probes it claims to answer — sample count,
    point and quantity — so an engine cannot pass by answering a different question.
    """
    if getattr(case, "manifest", None) is not None:
        from ..cad.roles import require_all_assigned

        require_all_assigned(case.manifest)
    errors = errors_in(engine.validate(case))
    if errors:
        reason = errors[0]
        if len(errors) > 1:
            extra = "; ".join(f"{p.code}: {p.message}" for p in errors[1:])
            reason = Problem(reason.code, f"{reason.message} (also {extra})", "error")
        return [
            EngineCaseResult(
                name=case.name,
                check=item.check,
                quantity=item.quantity,
                expected=item.value,
                tolerance=item.tolerance,
                computed=None,
                rel_err=None,
                passed=None,
                skip=reason,
            )
            for item in case.expected
        ]

    engine_id = engine.capabilities().engine_id
    result = engine.solve(case)
    if len(result.samples) != len(case.probes):
        raise EngineError(
            engine_id,
            [
                Problem(
                    "E_RESULT_MISMATCH",
                    f"{case.name}: engine returned {len(result.samples)} sample(s) for "
                    f"{len(case.probes)} probe(s)",
                    "error",
                )
            ],
        )
    for index, (probe, sample) in enumerate(zip(case.probes, result.samples, strict=True)):
        if tuple(sample.point) != probe.point or sample.quantity != probe.quantity:
            raise EngineError(
                engine_id,
                [
                    Problem(
                        "E_RESULT_MISMATCH",
                        f"{case.name}: sample #{index} answers {sample.quantity} at "
                        f"{sample.point}, but probe #{index} asks for {probe.quantity} at "
                        f"{probe.point}",
                        "error",
                    )
                ],
            )

    rows: list[EngineCaseResult] = []
    for position, item in enumerate(case.expected):
        index = _probe_index(case, position)
        sample = result.samples[index]
        if sample.unit != item.unit:
            raise CaseError(
                f"{case.name}: expected value is in {item.unit!r} but the engine sample is in "
                f"{sample.unit!r}"
            )
        computed = result.value(index, case.probes[index].component)
        rel_err, passed = _compare(computed, item.value, item.tolerance)
        rows.append(
            EngineCaseResult(
                name=case.name,
                check=item.check,
                quantity=item.quantity,
                expected=item.value,
                tolerance=item.tolerance,
                computed=computed,
                rel_err=rel_err,
                passed=passed,
            )
        )
    return rows


def run_engine_benchmarks(cases: list[Case], engine: Engine) -> list[EngineCaseResult]:
    """Evaluate a corpus through ``engine``: exactly one row per expected value, in order.

    Fails loudly if the row count or the set of case names ever disagrees with the corpus —
    a case may be skipped (with a reason) but never dropped.
    """
    rows = [row for case in cases for row in evaluate_with_engine(case, engine)]
    wanted = sum(len(case.expected) for case in cases)
    if len(rows) != wanted or {row.name for row in rows} != {case.name for case in cases}:
        raise RuntimeError(
            f"engine benchmark produced {len(rows)} row(s) for {len(cases)} case(s) "
            f"({wanted} expected value(s)); a case was dropped"
        )
    return rows


def format_engine_table(rows: list[EngineCaseResult]) -> str:
    """The analytic table's columns; ``result`` holds PASS / FAIL / SKIP: <CODE>: <reason>."""
    lines = _table_header()
    for row in rows:
        computed = "-" if row.computed is None else f"{row.computed:.9e}"
        rel_err = "-" if row.rel_err is None else f"{row.rel_err:.2e}"
        lines.append(
            f"{row.name:<32} | {computed:>16} | {row.expected:>16.9e} | "
            f"{rel_err:>9} | {row.tolerance:>8.0e} | {row.result_text}"
        )
    return "\n".join(lines)


def engine_summary(rows: list[EngineCaseResult]) -> dict[str, int]:
    """Counts for the summary lines and the JSON payload."""
    skipped = sum(1 for row in rows if row.skipped)
    passed = sum(1 for row in rows if row.status == "PASS")
    checked = len(rows) - skipped
    return {
        "cross_checked": checked,
        "passed": passed,
        "failed": checked - passed,
        "skipped": skipped,
        "total": len(rows),
    }


def engine_all_passed(rows: list[EngineCaseResult]) -> bool:
    """True when every non-skipped row passed and at least one row was cross-checked."""
    summary = engine_summary(rows)
    return summary["cross_checked"] > 0 and summary["failed"] == 0


def engine_results_payload(rows: list[EngineCaseResult], engine: Engine) -> dict[str, Any]:
    """The ``--json`` document for the engine path: results, summary and engine provenance."""
    capabilities = engine.capabilities()
    return {
        "schema": RESULTS_SCHEMA,
        "engine": {
            "id": capabilities.engine_id,
            "version": capabilities.engine_version,
            "rung": capabilities.rung,
        },
        "results": [row.to_dict() for row in rows],
        "summary": engine_summary(rows),
    }
