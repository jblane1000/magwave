"""Command-line entry point (PRD FR-060).

Implemented: ``magwave bench`` — runs the analytic benchmark suite headless and exits 1 if
any case fails. Still specified, still unimplemented: ``run`` / ``validate`` / ``export``,
which arrive with the Tier 0 engine in M2 (see docs/ROADMAP.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="magwave", description="MagWave magnetic field simulator")
    parser.add_argument("--version", action="version", version=f"magwave {__version__}")
    parser.add_argument(
        "command",
        nargs="?",
        help="command to run (bench; run/validate/export arrive with M2)",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="bench: also write the results as JSON to PATH",
    )
    return parser


def _run_bench(json_path: str | None) -> int:
    from .benchmarks import all_passed, format_table, load_corpus, results_payload, run_benchmarks

    results = run_benchmarks(load_corpus())
    print(format_table(results))
    passed = sum(1 for result in results if result.passed)
    print(f"{passed}/{len(results)} PASS")
    if json_path:
        payload = results_payload(results)
        Path(json_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if all_passed(results) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        print(f"magwave {__version__} — M1 skeleton. See docs/PRD.md.")
        return 0
    if args.command == "bench":
        return _run_bench(args.json)
    print(
        f"magwave: command {args.command!r} is not implemented yet (see docs/ROADMAP.md).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
