"""Deterministic Markdown report presenter for solved cases (PRD FR-053).

Turns a :class:`~magwave.cases.Case` evaluated by an :class:`~magwave.engines.base.Engine`
into a reproducible Markdown report with provenance, a probe table (augmented with
expected value, tolerance and PASS/FAIL for the 15 corpus cases), and a final fenced
JSON block carrying the same facts.

No solver is imported at module import time; values come strictly from the supplied engine.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .benchmarks.harness import _compare, load_corpus
from .cases import Case, CaseError, ExpectedValue
from .engines.base import Engine, EngineError, Problem, Result

REPORT_SCHEMA = "magwave.report/1"


def display_case_file(case_path: str | Path) -> str:
    """Return a deterministic, non-absolute, cwd-independent display path for ``case_path``."""
    path = Path(case_path)
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve()
    for parent in resolved.parents:
        if (parent / "pyproject.toml").is_file():
            return resolved.relative_to(parent).as_posix()
    return resolved.name


def _units_summary(result: Result) -> str:
    """Summarise position and sample units in deterministic order of appearance."""
    parts = ["position m"]
    seen: set[tuple[str, str]] = set()
    for sample in result.samples:
        pair = (sample.quantity, sample.unit)
        if pair not in seen:
            seen.add(pair)
            parts.append(f"{sample.quantity} {sample.unit}")
    return ", ".join(parts)


def _probes_match(case: Case, corpus_case: Case) -> bool:
    """Return ``True`` when ``case`` and ``corpus_case`` share identical probes in order."""
    return len(case.probes) == len(corpus_case.probes) and all(
        a.point == b.point and a.quantity == b.quantity and a.component == b.component
        for a, b in zip(case.probes, corpus_case.probes, strict=True)
    )


def _corpus_expected(
    case: Case, corpus_by_name: Mapping[str, Case]
) -> tuple[bool, bool, list[ExpectedValue | None]]:
    """Return ``(is_corpus, name_mismatch, expected_per_probe)`` using the corpus authority."""
    corpus_case = corpus_by_name.get(case.name)
    if corpus_case is None:
        return False, False, [None] * len(case.probes)
    if not _probes_match(case, corpus_case):
        return False, True, [None] * len(case.probes)
    expected_items = [
        corpus_case.expected[i] if i < len(corpus_case.expected) else None
        for i in range(len(case.probes))
    ]
    return True, False, expected_items


def _validate_result_against_case(case: Case, result: Result) -> None:
    """Ensure ``result.samples`` aligns 1-to-1 with ``case.probes`` in point and quantity."""
    if len(result.samples) != len(case.probes):
        raise EngineError(
            result.engine_id,
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
                result.engine_id,
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


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|")


def build_report_payload(
    case: Case,
    result: Result,
    *,
    case_file: str | Path,
    corpus: Sequence[Case] | None = None,
) -> dict[str, Any]:
    """Build the machine-readable report dictionary from ``case`` and ``result``."""
    _validate_result_against_case(case, result)
    corpus_cases = load_corpus() if corpus is None else corpus
    corpus_by_name = {item.name: item for item in corpus_cases}
    is_corpus, name_mismatch, expected_items = _corpus_expected(case, corpus_by_name)

    probe_rows: list[dict[str, Any]] = []
    for index, (probe, sample, expected_item) in enumerate(
        zip(case.probes, result.samples, expected_items, strict=True)
    ):
        computed = float(result.value(index, probe.component))
        row: dict[str, Any] = {
            "label": f"probe #{index}",
            "point": [float(coord) for coord in probe.point],
            "quantity": probe.quantity,
            "component": probe.component,
            "computed": computed,
            "unit": sample.unit,
        }
        if is_corpus and expected_item is not None:
            if sample.unit != expected_item.unit:
                raise CaseError(
                    f"{case.name}: expected value is in {expected_item.unit!r} but the engine "
                    f"sample is in {sample.unit!r}"
                )
            rel_err, passed = _compare(computed, expected_item.value, expected_item.tolerance)
            row["expected"] = float(expected_item.value)
            row["tolerance"] = float(expected_item.tolerance)
            row["rel_err"] = float(rel_err)
            row["passed"] = bool(passed)
            row["result"] = "PASS" if passed else "FAIL"
        probe_rows.append(row)

    payload: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "name": case.name,
        "description": case.description,
        "engine": result.engine_id,
        "engine_version": result.engine_version,
        "rung": result.rung,
        "method": result.method,
        "units": _units_summary(result),
        "case_file": display_case_file(case_file),
        "corpus_case": is_corpus,
        "probes": probe_rows,
    }
    if result.convergence is not None:
        conv = result.convergence
        conv_payload: dict[str, Any] = {
            "elements": int(conv.elements),
            "final_relative_residual": float(conv.final_relative_residual),
            "interface_nodes": int(conv.interface_nodes),
            "iterations": int(conv.iterations),
            "material_model": conv.material_model,
            "nodes": int(conv.nodes),
            "relaxation": conv.relaxation,
            "residual_history": [float(val) for val in conv.residual_history],
            "solver": conv.solver,
        }
        if conv.magnet_operating_points:
            conv_payload["magnet_operating_points"] = [
                {
                    "b_t": float(op.b_t),
                    "h_a_per_m": float(op.h_a_per_m),
                    "name": op.name,
                    "source": op.source,
                }
                for op in conv.magnet_operating_points
            ]
        payload["convergence"] = conv_payload
    if name_mismatch:
        payload["corpus_name_mismatch"] = True
    if is_corpus:
        checked = [row["passed"] for row in probe_rows if "passed" in row]
        all_passed = bool(checked) and all(checked)
        payload["passed"] = all_passed
        payload["result"] = "PASS" if all_passed else "FAIL"
    return payload


def format_report_markdown(payload: Mapping[str, Any]) -> str:
    """Render a report payload dictionary as deterministic Markdown."""
    title_desc = " ".join(str(payload["description"]).splitlines())
    lines = [
        f"# {payload['name']} — {title_desc}",
        "",
        f"engine: {payload['engine']}",
        f"engine_version: {payload['engine_version']}",
        f"rung: {payload['rung']}",
        f"units: {payload['units']}",
        f"case_file: {payload['case_file']}",
        f"method: {payload['method']}",
    ]
    conv = payload.get("convergence")
    if isinstance(conv, Mapping):
        lines.extend(
            [
                f"iterations: {conv['iterations']}",
                f"final_relative_residual: {float(conv['final_relative_residual']):.6e}",
                f"nodes: {conv['nodes']}",
                f"elements: {conv['elements']}",
                f"interface_nodes: {conv['interface_nodes']}",
                f"solver: {conv['solver']}",
                f"relaxation: {conv['relaxation']}",
                f"material_model: {conv['material_model']}",
            ]
        )
        ops = conv.get("magnet_operating_points")
        if isinstance(ops, Sequence) and not isinstance(ops, str | bytes):
            for op in ops:
                lines.append(
                    f"magnet_operating_point[{op['name']}]: "
                    f"H={float(op['h_a_per_m']):.1f} A/m, B={float(op['b_t']):.6f} T "
                    f"({op['source']})"
                )
    lines.append("")
    is_corpus = bool(payload["corpus_case"])
    if is_corpus:
        lines.extend(
            [
                "| probe | quantity | component | computed | unit | expected | tolerance | result |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| probe | quantity | component | computed | unit |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
    for row in payload["probes"]:
        point_text = ", ".join(repr(float(coord)) for coord in row["point"])
        probe_label = _escape_cell(f"{row['label']} ({point_text}) m")
        cells = [
            probe_label,
            _escape_cell(str(row["quantity"])),
            _escape_cell(str(row["component"])),
            repr(float(row["computed"])),
            _escape_cell(str(row["unit"])),
        ]
        if is_corpus:
            if "expected" in row:
                cells.extend(
                    [
                        repr(float(row["expected"])),
                        repr(float(row["tolerance"])),
                        str(row["result"]),
                    ]
                )
            else:
                cells.extend(["-", "-", "-"])
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "```json",
            json.dumps(payload, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def render_report(
    case: Case,
    engine: Engine,
    *,
    case_file: str | Path,
    corpus: Sequence[Case] | None = None,
) -> str:
    """Solve ``case`` with ``engine`` and format the deterministic Markdown report."""
    result = engine.solve(case)
    payload = build_report_payload(case, result, case_file=case_file, corpus=corpus)
    return format_report_markdown(payload)
