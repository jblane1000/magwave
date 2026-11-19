"""Command-line entry point (PRD FR-060).

Implemented:

* ``magwave bench [--engine magpylib] [--json PATH]`` — runs the benchmark corpus headless
  and exits 1 if any case fails. Without ``--engine`` the closed-form checks in
  ``benchmarks/checks.py`` produce the numbers; with ``--engine magpylib`` the same cases,
  expected values and tolerances are evaluated through the Tier 0 engine, and cases the
  engine cannot model are reported as ``SKIP: <CODE>: <reason>``.
* ``magwave solve CASE.json --format csv|json (--grid SPEC | --points SPEC)`` — samples B
  with the Tier 0 engine and prints the samples with units, engine id, version and rung
  (FR-050). Exit 2 on a malformed case or one the engine cannot model.
* ``magwave report CASE.json [--engine magpylib] [--out PATH|-]`` — evaluates a case and
  writes a deterministic Markdown report with provenance, probe table, corpus comparison,
  and a final JSON block (FR-053). Exit 1 if the case fails to load or the engine refuses
  it.
* ``magwave gui [CASE.json]`` — opens the optional PySide6 + PyVista desktop UI with an
  optional case JSON file (FR-061). Exit 1 with install instructions if the ``gui`` extra
  is not installed.
* ``magwave import PATH [--json MANIFEST.json] [--role INDEX=ROLE ...]`` — imports a STEP,
  IGES, or mesh file into a manifest, optionally assigns body roles, and exits 1 if any
  body remains unassigned or if any ``skipped_body`` defect occurred (FR-010/011/013).

Still specified, still unimplemented: ``run`` / ``validate`` / ``export`` (docs/ROADMAP.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import __version__

ENGINES = ("magpylib",)
FORMATS = ("csv", "json")
SAMPLES_SCHEMA = "magwave.samples/1"
#: Grid end points are inclusive when within this distance (metres) of the last step.
GRID_INCLUSIVE_TOL = 1e-12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="magwave", description="MagWave magnetic field simulator")
    parser.add_argument("--version", action="version", version=f"magwave {__version__}")
    parser.add_argument(
        "command",
        nargs="?",
        help="command to run (bench, solve, report, gui, import; run/validate/export are not implemented yet)",
    )
    parser.add_argument(
        "case",
        nargs="?",
        default=None,
        help="solve/report/gui/import: path to a case JSON or CAD file",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="bench/import: write the results or manifest as JSON to PATH",
    )
    parser.add_argument(
        "--role",
        action="append",
        metavar="INDEX=ROLE",
        default=[],
        help="import: assign body role by index or name (repeatable)",
    )
    parser.add_argument(
        "--defeature-below",
        metavar="MM",
        type=float,
        default=None,
        help="import: remove features smaller than MM (STEP/IGES only)",
    )
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default=None,
        help="bench/report: evaluate with this engine (default for report: magpylib)",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="report: write Markdown report to PATH or '-' for stdout (default '-')",
    )
    parser.add_argument(
        "--format",
        choices=FORMATS,
        default=None,
        help="solve: output format (default csv)",
    )
    parser.add_argument(
        "--grid",
        metavar="SPEC",
        default=None,
        help='solve: "xmin,xmax,xstep:ymin,ymax,ystep:zmin,zmax,zstep" in metres, max inclusive',
    )
    parser.add_argument(
        "--points",
        metavar="SPEC",
        default=None,
        help='solve: explicit points "x,y,z; x,y,z; ..." in metres',
    )
    return parser


# --------------------------------------------------------------------------------------
# bench
# --------------------------------------------------------------------------------------
def _write_json(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_bench(json_path: str | None) -> int:
    from .benchmarks import all_passed, format_table, load_corpus, results_payload, run_benchmarks

    results = run_benchmarks(load_corpus())
    print(format_table(results))
    passed = sum(1 for result in results if result.passed)
    print(f"{passed}/{len(results)} PASS")
    if json_path:
        payload = results_payload(results)
        _write_json(json_path, payload)
    return 0 if all_passed(results) else 1


def _run_engine_bench(engine_name: str, json_path: str | None) -> int:
    from .benchmarks import (
        engine_all_passed,
        engine_results_payload,
        engine_summary,
        format_engine_table,
        load_corpus,
        run_engine_benchmarks,
    )

    engine = _make_engine(engine_name)
    cases = load_corpus()
    rows = run_engine_benchmarks(cases, engine)
    print(format_engine_table(rows))
    if len(rows) != len(cases):
        # The corpus carries one expected value per case; anything else is a dropped or
        # duplicated row and must not be summarised as if it were a clean run.
        print(
            f"magwave: engine bench produced {len(rows)} row(s) for {len(cases)} case(s)",
            file=sys.stderr,
        )
        return 1
    summary = engine_summary(rows)
    print(f"{summary['passed']}/{summary['cross_checked']} PASS")
    print(f"{summary['cross_checked']} cross-checked, {summary['skipped']} skipped")
    if json_path:
        _write_json(json_path, engine_results_payload(rows, engine))
    return 0 if engine_all_passed(rows) else 1


def _make_engine(name: str):
    if name == "magpylib":
        from .engines.magpylib_engine import MagpylibEngine

        return MagpylibEngine()
    raise ValueError(f"unknown engine {name!r}")  # pragma: no cover - argparse choices


# --------------------------------------------------------------------------------------
# solve
# --------------------------------------------------------------------------------------
def _floats(text: str, count: int, what: str) -> list[float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != count:
        raise ValueError(f"{what}: expected {count} comma-separated numbers, got {text!r}")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{what}: not a number in {text!r}") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{what}: values must be finite, got {text!r}")
    return values


def parse_grid(spec: str) -> np.ndarray:
    """``"xmin,xmax,xstep:ymin,ymax,ystep:zmin,zmax,zstep"`` -> ``(n, 3)`` points, metres.

    Each axis runs ``min, min+step, ...`` up to and including ``max`` (within 1e-12 m).
    Points are ordered with x slowest and z fastest.
    """
    axes_text = spec.split(":")
    if len(axes_text) != 3:
        raise ValueError(f"--grid: expected three ':'-separated axes, got {spec!r}")
    axes = []
    for label, text in zip("xyz", axes_text, strict=True):
        start, stop, step = _floats(text, 3, f"--grid {label} axis")
        if step <= 0.0:
            raise ValueError(f"--grid {label} axis: step must be positive, got {step!r}")
        if stop < start:
            raise ValueError(f"--grid {label} axis: max {stop!r} is below min {start!r}")
        count = int(math.floor((stop - start + GRID_INCLUSIVE_TOL) / step)) + 1
        axes.append(start + step * np.arange(count, dtype=float))
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([axis.ravel() for axis in mesh], axis=1)


def parse_points(spec: str) -> np.ndarray:
    """``"x,y,z; x,y,z; ..."`` -> ``(n, 3)`` points, metres."""
    chunks = [chunk.strip() for chunk in spec.split(";")]
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        raise ValueError("--points: no points given")
    return np.array(
        [_floats(chunk, 3, f"--points #{i}") for i, chunk in enumerate(chunks)], dtype=float
    )


def _format_csv(points: np.ndarray, field: np.ndarray, magnitude: np.ndarray, engine) -> str:
    capabilities = engine.capabilities()
    lines = [
        "# magwave field samples",
        f"# engine: {capabilities.engine_id} {capabilities.engine_version}",
        f"# rung: {capabilities.rung}",
        "# units: position m, B T",
        "x_m,y_m,z_m,Bx_T,By_T,Bz_T,Bmag_T",
    ]
    for point, vector, size in zip(points, field, magnitude, strict=True):
        values = [*point.tolist(), *vector.tolist(), float(size)]
        lines.append(",".join(repr(float(value)) for value in values))
    return "\n".join(lines) + "\n"


def _format_json(points: np.ndarray, field: np.ndarray, magnitude: np.ndarray, engine) -> str:
    capabilities = engine.capabilities()
    payload = {
        "schema": SAMPLES_SCHEMA,
        "engine": {
            "id": capabilities.engine_id,
            "version": capabilities.engine_version,
            "rung": capabilities.rung,
        },
        "units": {"position": "m", "B": "T"},
        "samples": [
            {"point": point.tolist(), "B": vector.tolist(), "Bmag": float(size)}
            for point, vector, size in zip(points, field, magnitude, strict=True)
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _run_solve(case_path: str, points: np.ndarray, output_format: str) -> int:
    from .cases import Case, CaseError
    from .engines.base import errors_in
    from .engines.magpylib_engine import MagpylibEngine, field_magnitude

    try:
        case = Case.from_json(Path(case_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, CaseError) as exc:
        print(f"magwave solve: {case_path}: {exc}", file=sys.stderr)
        return 2
    if case.manifest is not None:
        from .cad.roles import UnassignedBodiesError, require_all_assigned

        try:
            require_all_assigned(case.manifest)
        except UnassignedBodiesError as exc:
            print(f"magwave solve: {exc}", file=sys.stderr)
            return 2
    engine = MagpylibEngine()
    problems = engine.validate(case)
    errors = errors_in(problems)
    if errors:
        capabilities = engine.capabilities()
        print(
            f"magwave solve: engine {capabilities.engine_id} {capabilities.engine_version} "
            f"cannot solve case {case.name!r}:",
            file=sys.stderr,
        )
        for problem in errors:
            print(f"  {problem}", file=sys.stderr)
        return 2
    for problem in problems:
        print(f"magwave solve: {problem}", file=sys.stderr)
    field = engine.sample(case, points)
    magnitude = field_magnitude(field)
    formatter = _format_csv if output_format == "csv" else _format_json
    sys.stdout.write(formatter(points, field, magnitude, engine))
    return 0


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def _one_line(text: object) -> str:
    return " ".join(str(text).splitlines())


def _run_report(case_path: str, engine_name: str, out_path: str | None) -> int:
    from .cases import Case, CaseError
    from .engines.base import EngineError
    from .report import render_report

    try:
        case = Case.from_json(Path(case_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, CaseError) as exc:
        print(f"magwave report: {case_path}: {_one_line(exc)}", file=sys.stderr)
        return 1
    if case.manifest is not None:
        from .cad.roles import UnassignedBodiesError, require_all_assigned

        try:
            require_all_assigned(case.manifest)
        except UnassignedBodiesError as exc:
            print(f"magwave report: {_one_line(exc)}", file=sys.stderr)
            return 1
    engine = _make_engine(engine_name)
    try:
        markdown = render_report(case, engine, case_file=case_path)
    except (EngineError, CaseError) as exc:
        print(f"magwave report: {_one_line(exc)}", file=sys.stderr)
        return 1
    if out_path is None or out_path == "-":
        sys.stdout.write(markdown)
        return 0
    try:
        with Path(out_path).open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(markdown)
    except OSError as exc:
        print(f"magwave report: {out_path}: {_one_line(exc)}", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------------------
# gui
# --------------------------------------------------------------------------------------
def _run_gui(case_path: str | None) -> int:
    from .gui import MISSING_GUI_MESSAGE, MissingGuiError, launch_gui, require_gui

    try:
        require_gui()
    except MissingGuiError:
        print(MISSING_GUI_MESSAGE, file=sys.stderr)
        return 1
    return launch_gui(case_path)


# --------------------------------------------------------------------------------------
# import
# --------------------------------------------------------------------------------------
def _parse_role_specs(specs: list[str]) -> dict[int | str, str]:
    mapping: dict[int | str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--role: expected INDEX=ROLE, got {spec!r}")
        raw_key, raw_role = spec.split("=", 1)
        key_text = raw_key.strip()
        role_text = raw_role.strip()
        if not key_text or not role_text:
            raise ValueError(f"--role: expected INDEX=ROLE, got {spec!r}")
        key: int | str = int(key_text) if key_text.isdigit() else key_text
        mapping[key] = role_text
    return mapping


def _format_manifest_summary(manifest) -> str:
    lines = [
        f"path: {manifest.path}",
        f"format: {manifest.format}  units: {manifest.units}  bodies: {len(manifest.bodies)}",
    ]
    for body in manifest.bodies:
        role_text = body.role if body.role is not None else "<unassigned>"
        lines.append(
            f"  [{body.index}] {body.display_name}  role={role_text}  "
            f"solids={body.solids}  volume={body.volume_mm3:.4f} mm^3  source={body.source}"
        )
    if manifest.defeatures:
        lines.append(f"defeatures ({len(manifest.defeatures)}):")
        for record in manifest.defeatures:
            lines.append(
                f"  [{record.body_index}] threshold={record.threshold_mm:.4f} mm  "
                f"removed={record.features_removed}  "
                f"volume {record.volume_before_mm3:.4f} -> {record.volume_after_mm3:.4f} mm^3 "
                f"(delta {record.volume_delta_mm3:+.4f})"
            )
    if manifest.defects:
        lines.append(f"defects ({len(manifest.defects)}):")
        for defect in manifest.defects:
            target = f"body #{defect.body_index}" if defect.body_index is not None else "global"
            lines.append(f"  - {defect.kind} ({target}): {defect.detail}")
    if manifest.warnings:
        lines.append(f"warnings ({len(manifest.warnings)}):")
        for warning in manifest.warnings:
            lines.append(f"  - {warning}")
    return "\n".join(lines)


def _run_import(
    cad_path: str,
    json_path: str | None,
    role_specs: list[str],
    defeature_below: float | None = None,
) -> int:
    from .cad import MissingExtraError, assign, import_cad, unassigned

    try:
        mapping = _parse_role_specs(role_specs)
    except ValueError as exc:
        print(f"magwave import: {_one_line(exc)}", file=sys.stderr)
        return 2

    try:
        manifest = import_cad(cad_path, defeature_below=defeature_below)
    except (MissingExtraError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"magwave import: {_one_line(exc)}", file=sys.stderr)
        return 1

    if mapping:
        try:
            assign(manifest, mapping)
        except (ValueError, TypeError) as exc:
            print(f"magwave import: {_one_line(exc)}", file=sys.stderr)
            return 1

    print(_format_manifest_summary(manifest))

    if json_path is not None:
        try:
            Path(json_path).write_text(manifest.to_json() + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"magwave import: {json_path}: {_one_line(exc)}", file=sys.stderr)
            return 1

    has_skipped = any(defect.kind == "skipped_body" for defect in manifest.defects)
    if unassigned(manifest) or has_skipped:
        return 1
    return 0


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        print(f"magwave {__version__} — M1 skeleton. See docs/PRD.md.")
        return 0
    if args.out is not None and args.command != "report":
        parser.error(f"{args.command}: --out applies to report only")
    if args.command not in ("solve", "report", "gui", "import") and args.case is not None:
        parser.error(f"{args.command}: case path applies to solve, report, gui and import")
    if args.command != "solve" and (args.grid or args.points or args.format):
        parser.error(f"{args.command}: --format, --grid and --points apply to solve")
    if args.command != "import" and args.role:
        parser.error(f"{args.command}: --role applies to import only")
    if args.command != "import" and args.defeature_below is not None:
        parser.error(f"{args.command}: --defeature-below applies to import only")
    if args.command == "bench":
        if args.engine is None:
            return _run_bench(args.json)
        return _run_engine_bench(args.engine, args.json)
    if args.command == "solve":
        if args.case is None:
            parser.error("solve: a case JSON path is required")
        if args.json or args.engine:
            parser.error("solve: --json and --engine apply to bench only")
        if (args.grid is None) == (args.points is None):
            parser.error("solve: exactly one of --grid or --points is required")
        try:
            points = parse_grid(args.grid) if args.grid is not None else parse_points(args.points)
        except ValueError as exc:
            parser.error(str(exc))
        return _run_solve(args.case, points, args.format or "csv")
    if args.command == "report":
        if args.json or args.format or args.grid or args.points:
            parser.error("report: --json, --format, --grid and --points do not apply to report")
        if args.case is None:
            print("magwave report: a case JSON path is required", file=sys.stderr)
            return 1
        return _run_report(args.case, args.engine or "magpylib", args.out)
    if args.command == "gui":
        if args.json or args.engine:
            parser.error("gui: --json and --engine do not apply to gui")
        return _run_gui(args.case)
    if args.command == "import":
        if args.engine:
            parser.error("import: --engine does not apply to import")
        if args.case is None:
            parser.error("import: a CAD file path is required")
        return _run_import(args.case, args.json, args.role, args.defeature_below)
    print(
        f"magwave: command {args.command!r} is not implemented yet (see docs/ROADMAP.md).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
