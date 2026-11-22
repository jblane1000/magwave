"""Mesh-convergence ladder analysis and deterministic table formatter."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .cases import Probe

#: Relative Richardson error ceiling (%) for ``converged: YES``.
CONVERGENCE_REL_LIMIT_PCT = 1.0
_RATIO_UNIFORM_TOL = 1e-9


@dataclass(frozen=True)
class ConvergenceRung:
    """Measured solve output at one mesh element size."""

    element_size: float
    nodes: int
    elements: int
    probe_values: tuple[float, ...]
    probe_units: tuple[str, ...]


@dataclass(frozen=True)
class ProbeConvergence:
    """Richardson convergence summary for one probe."""

    probe_index: int
    unit: str
    delta_falling: bool | None
    ratio_uniform: bool
    ratio: float | None
    observed_order: float | None
    order_reason: str | None
    est_error_abs: float | None
    est_error_rel_pct: float | None
    error_reason: str | None
    converged: bool


@dataclass(frozen=True)
class ConvergenceStudy:
    """Full mesh-convergence ladder result across all case probes."""

    case_name: str
    probe_headers: tuple[str, ...]
    rungs: tuple[ConvergenceRung, ...]
    summary: ProbeConvergence
    per_probe: tuple[ProbeConvergence, ...]
    delta_falling: bool | None
    converged: bool


def probe_column_header(index: int, probe: Probe, unit: str) -> str:
    """Return the table column header for probe ``index`` (e.g. ``probe #0 Bz (T)``)."""
    if probe.component == "magnitude":
        qty_label = f"|{probe.quantity}|"
    else:
        qty_label = f"{probe.quantity}{probe.component}"
    return f"probe #{index} {qty_label} ({unit})"


def _analyse_probe(
    probe_index: int,
    rungs: Sequence[ConvergenceRung],
    ratio_uniform: bool,
    ratio: float | None,
) -> ProbeConvergence:
    unit = rungs[0].probe_units[probe_index]
    if len(rungs) == 2:
        return ProbeConvergence(
            probe_index=probe_index,
            unit=unit,
            delta_falling=None,
            ratio_uniform=False,
            ratio=None,
            observed_order=None,
            order_reason="two rungs",
            est_error_abs=None,
            est_error_rel_pct=None,
            error_reason="two rungs",
            converged=False,
        )

    v_a = rungs[-3].probe_values[probe_index]
    v_b = rungs[-2].probe_values[probe_index]
    v_c = rungs[-1].probe_values[probe_index]
    d_prev = v_b - v_a
    d_last = v_c - v_b
    delta_falling = abs(d_last) < abs(d_prev)

    if not ratio_uniform or ratio is None:
        return ProbeConvergence(
            probe_index=probe_index,
            unit=unit,
            delta_falling=delta_falling,
            ratio_uniform=False,
            ratio=None,
            observed_order=None,
            order_reason="non-uniform ladder",
            est_error_abs=None,
            est_error_rel_pct=None,
            error_reason="non-uniform ladder",
            converged=False,
        )

    if abs(d_prev) == 0.0 or abs(d_last) == 0.0:
        reason = "zero final delta" if abs(d_prev) > 0.0 else "zero delta"
        return ProbeConvergence(
            probe_index=probe_index,
            unit=unit,
            delta_falling=delta_falling,
            ratio_uniform=True,
            ratio=ratio,
            observed_order=None,
            order_reason=reason,
            est_error_abs=None,
            est_error_rel_pct=None,
            error_reason=reason,
            converged=False,
        )

    order = math.log(abs(d_prev) / abs(d_last)) / math.log(ratio)
    if order <= 0.0 or not delta_falling:
        return ProbeConvergence(
            probe_index=probe_index,
            unit=unit,
            delta_falling=delta_falling,
            ratio_uniform=True,
            ratio=ratio,
            observed_order=order,
            order_reason=None,
            est_error_abs=None,
            est_error_rel_pct=None,
            error_reason="non-positive observed order",
            converged=False,
        )

    denom = ratio**order - 1.0
    if denom <= 0.0:  # pragma: no cover - guarded by order > 0 and ratio > 1
        return ProbeConvergence(
            probe_index=probe_index,
            unit=unit,
            delta_falling=delta_falling,
            ratio_uniform=True,
            ratio=ratio,
            observed_order=order,
            order_reason=None,
            est_error_abs=None,
            est_error_rel_pct=None,
            error_reason="non-positive observed order",
            converged=False,
        )

    err_abs = abs(d_last) / denom
    if abs(v_c) == 0.0:
        return ProbeConvergence(
            probe_index=probe_index,
            unit=unit,
            delta_falling=delta_falling,
            ratio_uniform=True,
            ratio=ratio,
            observed_order=order,
            order_reason=None,
            est_error_abs=err_abs,
            est_error_rel_pct=None,
            error_reason="finest value is zero",
            converged=False,
        )

    err_rel_pct = (err_abs / abs(v_c)) * 100.0
    converged = err_rel_pct <= CONVERGENCE_REL_LIMIT_PCT and delta_falling
    return ProbeConvergence(
        probe_index=probe_index,
        unit=unit,
        delta_falling=delta_falling,
        ratio_uniform=True,
        ratio=ratio,
        observed_order=order,
        order_reason=None,
        est_error_abs=err_abs,
        est_error_rel_pct=err_rel_pct,
        error_reason=None,
        converged=converged,
    )


def _select_summary_probe(per_probe: Sequence[ProbeConvergence]) -> ProbeConvergence:
    if len(per_probe) == 1:
        return per_probe[0]
    for item in per_probe:
        if item.delta_falling is False:
            return item
    for item in per_probe:
        if item.est_error_rel_pct is None:
            return item
    return max(
        per_probe,
        key=lambda item: item.est_error_rel_pct if item.est_error_rel_pct is not None else 0.0,
    )


def analyse_ladder(
    case_name: str,
    probe_headers: Sequence[str],
    rungs: Sequence[ConvergenceRung],
) -> ConvergenceStudy:
    """Compute Richardson convergence metrics for a strictly decreasing element-size ladder."""
    rung_tuple = tuple(rungs)
    header_tuple = tuple(probe_headers)
    if len(rung_tuple) < 2:
        raise ValueError(f"at least 2 rungs are required, got {len(rung_tuple)}")
    for idx, rung in enumerate(rung_tuple):
        if not math.isfinite(rung.element_size) or rung.element_size <= 0.0:
            raise ValueError(
                f"rung #{idx} element_size must be finite > 0, got {rung.element_size!r}"
            )
        if idx > 0 and rung.element_size >= rung_tuple[idx - 1].element_size:
            raise ValueError(
                "element sizes must be strictly decreasing: "
                f"{rung_tuple[idx - 1].element_size!r} followed by {rung.element_size!r}"
            )
        if len(rung.probe_values) != len(header_tuple):
            raise ValueError(
                f"rung #{idx} returned {len(rung.probe_values)} probe value(s), "
                f"expected {len(header_tuple)}"
            )

    if len(rung_tuple) >= 3:
        h_a = rung_tuple[-3].element_size
        h_b = rung_tuple[-2].element_size
        h_c = rung_tuple[-1].element_size
        r_prev = h_a / h_b
        r_last = h_b / h_c
        ratio_uniform = abs(r_prev - r_last) <= _RATIO_UNIFORM_TOL
        ratio = r_last if ratio_uniform else None
    else:
        ratio_uniform = False
        ratio = None

    per_probe = tuple(
        _analyse_probe(idx, rung_tuple, ratio_uniform, ratio) for idx in range(len(header_tuple))
    )
    summary = _select_summary_probe(per_probe)
    if len(rung_tuple) == 2:
        delta_falling: bool | None = None
        converged = False
    else:
        delta_falling = all(bool(item.delta_falling) for item in per_probe)
        converged = delta_falling and all(item.converged for item in per_probe)

    return ConvergenceStudy(
        case_name=case_name,
        probe_headers=header_tuple,
        rungs=rung_tuple,
        summary=summary,
        per_probe=per_probe,
        delta_falling=delta_falling,
        converged=converged,
    )


def _format_delta_cell(prev_rung: ConvergenceRung, curr_rung: ConvergenceRung) -> str:
    n_probes = len(curr_rung.probe_values)
    if n_probes == 1:
        diff = curr_rung.probe_values[0] - prev_rung.probe_values[0]
        return f"{diff:+.6e}"
    max_diff = max(
        abs(a - b) for a, b in zip(curr_rung.probe_values, prev_rung.probe_values, strict=True)
    )
    return f"{max_diff:.6e}"


def _format_rel_finest_cell(rung: ConvergenceRung, finest: ConvergenceRung) -> str:
    rels: list[float] = []
    for val, ref in zip(rung.probe_values, finest.probe_values, strict=True):
        if ref == 0.0:
            return "n/a"
        rels.append(abs(val - ref) / abs(ref) * 100.0)
    return f"{max(rels):.4f}%"


def format_convergence_table(study: ConvergenceStudy) -> str:
    """Format ``study`` as a deterministic Markdown table followed by Richardson summary lines."""
    headers = [
        "h (m)",
        "nodes",
        "elements",
        *study.probe_headers,
        "delta vs prev",
        "rel vs finest",
    ]
    finest = study.rungs[-1]
    rows: list[list[str]] = []
    for idx, rung in enumerate(study.rungs):
        delta_cell = "---" if idx == 0 else _format_delta_cell(study.rungs[idx - 1], rung)
        rel_cell = _format_rel_finest_cell(rung, finest)
        rows.append(
            [
                repr(float(rung.element_size)),
                str(rung.nodes),
                str(rung.elements),
                *(repr(float(val)) for val in rung.probe_values),
                delta_cell,
                rel_cell,
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) for col in range(len(headers))
    ]

    def _join_row(cells: Sequence[str]) -> str:
        padded = [cell.rjust(widths[col]) for col, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    lines = [
        _join_row(headers),
        "| "
        + " | ".join("-" * max(3, widths[col] - 1) + ":" for col in range(len(headers)))
        + " |",
        *(_join_row(row) for row in rows),
        "",
    ]

    summary = study.summary
    if summary.observed_order is not None and summary.ratio is not None:
        lines.append(
            f"observed order p = {summary.observed_order:.4f} "
            f"(from the last three rungs, uniform ratio {summary.ratio:g})"
        )
    else:
        lines.append(f"observed order: not estimable ({summary.order_reason})")

    if summary.est_error_abs is not None and summary.est_error_rel_pct is not None:
        lines.append(
            f"estimated discretisation error at h = {finest.element_size:g} m: "
            f"{summary.est_error_abs:.6e} {summary.unit} "
            f"({summary.est_error_rel_pct:.4f} % of the value)"
        )
    else:
        lines.append(f"estimated discretisation error: not reported ({summary.error_reason})")

    if study.delta_falling is False:
        lines.append(
            "WARNING: |delta| does not fall under refinement on this ladder; "
            "this table is not convergence evidence"
        )

    lines.append(
        "criterion: estimated relative error <= 1.000 % and "
        "|delta| strictly falling over the last two intervals"
    )
    lines.append(f"converged: {'YES' if study.converged else 'NO'}")
    return "\n".join(lines) + "\n"


def convergence_payload(study: ConvergenceStudy) -> dict[str, Any]:
    """Return a JSON-serialisable dictionary for ``study``."""
    return {
        "case": study.case_name,
        "probe_headers": list(study.probe_headers),
        "rungs": [
            {
                "element_size": rung.element_size,
                "nodes": rung.nodes,
                "elements": rung.elements,
                "probe_values": list(rung.probe_values),
                "probe_units": list(rung.probe_units),
            }
            for rung in study.rungs
        ],
        "observed_order": study.summary.observed_order,
        "order_reason": study.summary.order_reason,
        "ratio": study.summary.ratio,
        "est_error_abs": study.summary.est_error_abs,
        "est_error_rel_pct": study.summary.est_error_rel_pct,
        "error_reason": study.summary.error_reason,
        "delta_falling": study.delta_falling,
        "converged": study.converged,
    }
