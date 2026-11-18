"""Benchmark harness: evaluate the analytic corpus and report one aligned row per case.

``python -m magwave bench`` prints the table below and exits 1 if any case fails, so CI
fails on a wrong number rather than on a written claim (FR-070, NFR-8).

    case                               |         computed |         expected |    rel.err |      tol | result

The pass rule is the research file's: ``|computed - expected| <= tolerance * |expected|``,
with the tolerance taken from the case — research-owned, never adjusted here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..cases import Case, CaseError
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
        difference = abs(computed - item.value)
        scale = max(abs(item.value), _RELATIVE_FLOOR)
        rel_err = difference / abs(item.value) if item.value else difference
        results.append(
            CaseResult(
                name=case.name,
                check=item.check,
                quantity=item.quantity,
                computed=computed,
                expected=item.value,
                tolerance=item.tolerance,
                rel_err=rel_err,
                passed=difference <= item.tolerance * scale,
            )
        )
    return results


def run_benchmarks(cases: list[Case]) -> list[CaseResult]:
    """Evaluate a corpus in order, one result per expected value."""
    return [result for case in cases for result in evaluate(case)]


def all_passed(results: list[CaseResult]) -> bool:
    return all(result.passed for result in results)


def format_table(results: list[CaseResult]) -> str:
    """Render the aligned table: name | computed | expected | rel.err | tol | PASS/FAIL."""
    header = (
        f"{'case':<32} | {'computed':>16} | {'expected':>16} | {'rel.err':>9} | {'tol':>8} | result"
    )
    lines = [header, "-" * len(header)]
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
