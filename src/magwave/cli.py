"""Command-line entry point.

Phase 0 stub: reports the version and the project state. Real commands (run / validate /
export / bench) are specified in docs/PRD.md FR-060 and land with M1.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="magwave", description="MagWave magnetic field simulator")
    parser.add_argument("--version", action="version", version=f"magwave {__version__}")
    parser.add_argument(
        "command",
        nargs="?",
        help="command to run (not implemented in Phase 0: run, validate, export, bench)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command:
        print(
            f"magwave: command {args.command!r} is not implemented yet "
            "(Phase 0 — see docs/ROADMAP.md).",
            file=sys.stderr,
        )
        return 2
    print(f"magwave {__version__} — Phase 0 (specification only). See docs/PRD.md.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
