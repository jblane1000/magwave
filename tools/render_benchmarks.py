#!/usr/bin/env python3
"""Render docs/BENCHMARKS.md from the benchmark suite JSON and check for drift.

Why this exists
---------------
``FR-071`` calls for a published results table in ``docs/BENCHMARKS.md``, stamped with the
engine version. A hand-written table drifts silently whenever a case, tolerance or solver
changes. Generating the document directly from ``python -m magwave bench --json`` keeps the
published numbers and engine version coupled to what the code actually computes.

A check nobody has seen fail is decoration, so ``--check`` re-renders into a temporary file
and fails if the committed copy differs by even a single byte (or is missing). The output
contains no timestamp, git hash, hostname or absolute path so consecutive runs on unchanged
code are byte-identical.

Usage
-----
    render_benchmarks.py                # render docs/BENCHMARKS.md
    render_benchmarks.py --out PATH     # render to PATH (relative to cwd)
    render_benchmarks.py --check        # verify docs/BENCHMARKS.md is up to date

Exit codes
----------
    0  success, or target file is identical to a fresh render under --check
    1  target file is stale/missing under --check, or benchmark JSON is invalid
       (if ``magwave bench`` itself fails, its non-zero exit code is propagated)
    2  usage error
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from magwave import __version__

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "BENCHMARKS.md"
DEFAULT_DISPLAY = "docs/BENCHMARKS.md"

REQUIRED_TOP_KEYS = ("schema", "summary", "results")
REQUIRED_SUMMARY_KEYS = ("passed", "failed", "total")
REQUIRED_RESULT_KEYS = (
    "name",
    "quantity",
    "check",
    "computed",
    "expected",
    "rel_err",
    "tolerance",
    "passed",
)


def _fmt_num(value: float | int | None) -> str:
    """Format a benchmark number in fixed scientific notation, or ``n/a`` if null."""
    if value is None:
        return "n/a"
    return f"{value:.6e}"


def _escape_cell(value: str) -> str:
    """Escape pipe characters so quantities such as ``|B|`` stay inside one GFM column."""
    return value.replace("|", r"\|")


def _validate_payload(payload: Any) -> str | None:
    """Return a single-line error message if ``payload`` lacks required fields, else None."""
    if not isinstance(payload, dict):
        return "render_benchmarks: bench JSON root must be an object"
    for key in REQUIRED_TOP_KEYS:
        if key not in payload:
            return f"render_benchmarks: bench JSON is missing key {key!r}"
    if not isinstance(payload["schema"], str):
        return "render_benchmarks: bench JSON 'schema' must be a string"
    summary = payload["summary"]
    if not isinstance(summary, dict):
        return "render_benchmarks: bench JSON 'summary' must be an object"
    for key in REQUIRED_SUMMARY_KEYS:
        if key not in summary:
            return f"render_benchmarks: bench JSON summary is missing key {key!r}"
    results = payload["results"]
    if not isinstance(results, list):
        return "render_benchmarks: bench JSON 'results' must be a list"
    if not results:
        return "render_benchmarks: bench JSON has empty 'results'"
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            return f"render_benchmarks: bench JSON results[{index}] must be an object"
        for key in REQUIRED_RESULT_KEYS:
            if key not in row:
                return f"render_benchmarks: bench JSON results[{index}] is missing key {key!r}"
    return None


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the deterministic ``docs/BENCHMARKS.md`` document from a bench JSON payload."""
    schema = payload["schema"]
    summary = payload["summary"]
    results: list[dict[str, Any]] = list(payload["results"])

    if summary["failed"] == 0:
        counts_line = f"{summary['passed']} of {summary['total']} cases pass"
    else:
        counts_line = (
            f"{summary['passed']} of {summary['total']} cases pass ({summary['failed']} failing)"
        )

    candidates = [row for row in results if row["rel_err"] is not None]
    if candidates:
        worst = min(
            candidates,
            key=lambda row: (
                -float(row["rel_err"]),
                str(row["name"]),
                str(row["quantity"]),
                str(row["check"]),
            ),
        )
        worst_line = (
            f"Worst case by relative error: `{worst['name']}` "
            f"(`{worst['quantity']}`, check `{worst['check']}`), "
            f"rel err {_fmt_num(worst['rel_err'])} against tolerance {_fmt_num(worst['tolerance'])}"
        )
    else:
        worst_line = "Worst case by relative error: n/a"

    sorted_rows = sorted(
        results,
        key=lambda row: (str(row["name"]), str(row["quantity"]), str(row["check"])),
    )

    table_lines = [
        "| case | quantity | check | computed | expected | rel err | tolerance | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted_rows:
        status = "PASS" if row["passed"] else "FAIL"
        table_lines.append(
            f"| {_escape_cell(str(row['name']))} "
            f"| {_escape_cell(str(row['quantity']))} "
            f"| {_escape_cell(str(row['check']))} "
            f"| {_fmt_num(row['computed'])} "
            f"| {_fmt_num(row['expected'])} "
            f"| {_fmt_num(row['rel_err'])} "
            f"| {_fmt_num(row['tolerance'])} "
            f"| {status} |"
        )

    blocks = [
        "# Benchmarks",
        (
            "Generated by `tools/render_benchmarks.py` from `python -m magwave bench --json`. "
            "Do not edit this file by hand; rerun the tool and commit the result. "
            "The committed copy carries the engine version; the release workflow adds the "
            "release commit stamp."
        ),
        f"Engine: magwave {__version__} · bench schema `{schema}`",
        counts_line,
        worst_line,
        "\n".join(table_lines),
    ]
    return "\n\n".join(blocks) + "\n"


def _run_bench(tmp_json: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "magwave", "bench", "--json", str(tmp_json)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render docs/BENCHMARKS.md from `python -m magwave bench --json`."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output path (default: docs/BENCHMARKS.md under the repository root)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the target file is missing or differs from a fresh render",
    )
    args = parser.parse_args(argv)

    if args.out is None:
        target = DEFAULT_OUT
        display = DEFAULT_DISPLAY
    else:
        target = Path(args.out)
        display = args.out

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        tmp_json = tmp_path / "bench.json"
        proc = _run_bench(tmp_json)
        if proc.returncode != 0:
            print(
                f"render_benchmarks: benchmark suite failed (propagating exit code {proc.returncode})",
                file=sys.stderr,
            )
            if proc.stderr:
                sys.stderr.write(proc.stderr if proc.stderr.endswith("\n") else proc.stderr + "\n")
            return proc.returncode

        try:
            payload = json.loads(tmp_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"render_benchmarks: failed to read bench JSON ({exc})", file=sys.stderr)
            return 1

        error = _validate_payload(payload)
        if error is not None:
            print(error, file=sys.stderr)
            return 1

        rendered_bytes = render_markdown(payload).encode("utf-8")

        if args.check:
            tmp_md = tmp_path / "BENCHMARKS.md"
            tmp_md.write_bytes(rendered_bytes)
            fresh_bytes = tmp_md.read_bytes()
            try:
                existing_bytes = target.read_bytes()
            except OSError:
                existing_bytes = None
            if existing_bytes == fresh_bytes:
                print(f"{display}: up to date")
                return 0
            print(
                f'{display} is stale: rerun "python tools/render_benchmarks.py" and commit the regenerated file.',
                file=sys.stderr,
            )
            return 1

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(rendered_bytes)
        print(f"{display}: written")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
