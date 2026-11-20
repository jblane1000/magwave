#!/usr/bin/env python3
"""Offscreen desktop GUI walkthrough gate (PRD FR-061).

Drives the real MagWave ``MainWindow`` under ``QT_QPA_PLATFORM=offscreen``, clicks the real
``Run`` button through Qt, captures step screenshots for human inspection, and bases its
PASS/FAIL verdict strictly on live widget state (``run_button``, visible ``QDialog`` widgets,
and ``probes_table`` column 4), never on screenshot pixels.

Usage::

    python tools/gui_walkthrough.py --case examples/solve-dipole.json --shots /tmp/gw
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Must be set before PySide6 or magwave.gui is imported anywhere in this process.
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def _pump(app: Any, rounds: int = 12) -> None:
    """Process pending Qt events for ``rounds`` iterations so widgets paint and settle."""
    for _ in range(rounds):
        app.processEvents()


def _capture_widget(widget: Any, path: Path) -> Path:
    """Grab ``widget`` into a PNG file at ``path``, print its absolute path/size, and return it."""
    resolved = path.resolve()
    saved = widget.grab().save(str(resolved), "PNG")
    if not saved or not resolved.is_file():
        raise RuntimeError(f"failed to save screenshot to {resolved}")
    size = resolved.stat().st_size
    print(f"shot: {resolved} ({size} bytes)")
    return resolved


def _find_run_button(window: Any, qt_widgets: Any) -> Any | None:
    """Locate the Run button via ``window.run_button`` or widget-tree scan."""
    direct = getattr(window, "run_button", None)
    if isinstance(direct, qt_widgets.QAbstractButton):
        return direct
    for button in window.findChildren(qt_widgets.QAbstractButton):
        if "run" in (button.text() or "").lower():
            return button
    return None


def _visible_dialogs(qt_widgets: Any) -> list[Any]:
    """Return all currently visible top-level ``QDialog`` instances."""
    return [
        widget
        for widget in qt_widgets.QApplication.topLevelWidgets()
        if isinstance(widget, qt_widgets.QDialog) and widget.isVisible()
    ]


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _dialog_text(dialog: Any, qt_widgets: Any) -> str:
    """Extract deduplicated, whitespace-collapsed text from a dialog and its child labels."""
    parts: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        cleaned = _collapse_ws(raw)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            parts.append(cleaned)

    boxes: list[Any] = []
    if isinstance(dialog, qt_widgets.QMessageBox):
        boxes.append(dialog)
    boxes.extend(dialog.findChildren(qt_widgets.QMessageBox))
    for box in boxes:
        _add(box.text())
        _add(box.informativeText())
        _add(box.detailedText())

    for label in dialog.findChildren(qt_widgets.QLabel):
        _add(label.text())

    return " | ".join(parts)


def _find_probes_table(window: Any, qt_widgets: Any) -> Any | None:
    """Return ``window.probes_table`` or fall back to a table with a 'Solved' header column."""
    direct = getattr(window, "probes_table", None)
    if isinstance(direct, qt_widgets.QTableWidget):
        return direct
    for table in window.findChildren(qt_widgets.QTableWidget):
        if table.columnCount() > 4:
            header = table.horizontalHeaderItem(4)
            if header is not None and "solved" in header.text().lower():
                return table
    return None


def _cell_text(table: Any, row: int, col: int) -> str:
    item = table.item(row, col)
    return item.text().strip() if item is not None else ""


def _read_probe_rows(window: Any, qt_widgets: Any) -> list[tuple[str, str, float | None, str]]:
    """Read ``(quantity, component, solved_value, raw_text)`` tuples from ``probes_table``."""
    table = _find_probes_table(window, qt_widgets)
    if table is None:
        return []
    rows: list[tuple[str, str, float | None, str]] = []
    for row in range(table.rowCount()):
        qty = _cell_text(table, row, 2)
        comp = _cell_text(table, row, 3)
        raw = _cell_text(table, row, 4)
        parsed: float | None = None
        if raw and raw != "\u2014":
            try:
                value = float(raw)
            except ValueError:
                value = math.nan
            if math.isfinite(value):
                parsed = value
        rows.append((qty, comp, parsed, raw))
    return rows


def _read_problem_lines(window: Any) -> list[str]:
    """Read any visible lines from ``window.problems_list`` for diagnostics."""
    problems_list = getattr(window, "problems_list", None)
    if problems_list is None:
        return []
    lines: list[str] = []
    for row in range(problems_list.count()):
        item = problems_list.item(row)
        if item is not None:
            text = _collapse_ws(item.text())
            if text:
                lines.append(text)
    return lines


def _match_expected(
    probe_qty: str,
    probe_comp: str,
    expected: Sequence[Any],
    fallback_index: int = 0,
) -> Any:
    """Match a probe's ``(quantity, component)`` against ``Case.expected`` entries."""
    if probe_comp == "magnitude":
        candidates = (f"|{probe_qty}|", probe_qty, f"{probe_qty}_mag")
    else:
        candidates = (f"{probe_qty}{probe_comp}", f"{probe_qty}_{probe_comp}")

    for item in expected:
        if item.quantity in candidates:
            return item

    lowered = {candidate.lower() for candidate in candidates}
    for item in expected:
        if item.quantity.lower() in lowered:
            return item

    if 0 <= fallback_index < len(expected):
        return expected[fallback_index]
    return expected[0]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offscreen MagWave desktop GUI walkthrough on a case JSON.",
    )
    parser.add_argument(
        "--case",
        type=Path,
        required=True,
        help="Path to a MagWave case JSON file (for example, examples/solve-dipole.json).",
    )
    parser.add_argument(
        "--shots",
        type=Path,
        default=Path("/tmp/gw"),
        help="Directory where PNG screenshots are written (default: /tmp/gw).",
    )
    return parser


def run_walkthrough(case_path: Path, shots_dir: Path) -> int:
    """Execute the offscreen GUI walkthrough on ``case_path`` and return exit status."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    resolved_shots = shots_dir.resolve()
    resolved_shots.mkdir(parents=True, exist_ok=True)

    if not case_path.is_file():
        print(f"FAIL: case file not found: {case_path}")
        return 1

    import magwave.gui as gui
    from magwave.cases import Case

    try:
        case = Case.from_json(case_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"FAIL: could not load case {case_path}: {_collapse_ws(str(exc))}")
        return 1

    try:
        _, _, _, QtWidgets, _ = gui.require_gui()
    except gui.MissingGuiError as exc:
        print(f"FAIL: GUI dependency unavailable: {exc}")
        return 1

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = gui._build_qt_classes()["MainWindow"](case_path)

    try:
        window.resize(1600, 1000)
        window.show()
        _pump(app, rounds=12)

        _capture_widget(window, resolved_shots / "01-loaded.png")

        run_button = _find_run_button(window, QtWidgets)
        if run_button is None or not run_button.isEnabled():
            state = "missing" if run_button is None else "disabled"
            print(f"FAIL: Run button is {state} after loading {case_path}")
            return 1

        before_dialogs = _visible_dialogs(QtWidgets)
        run_button.click()
        _pump(app, rounds=40)

        new_dialogs = [
            dialog
            for dialog in _visible_dialogs(QtWidgets)
            if not any(dialog is prev for prev in before_dialogs)
        ]
        for idx, dialog in enumerate(new_dialogs, start=1):
            shot_name = "02-dialog.png" if idx == 1 else f"02-dialog-{idx}.png"
            _capture_widget(dialog, resolved_shots / shot_name)
            title = _collapse_ws(dialog.windowTitle())
            body = _dialog_text(dialog, QtWidgets)
            print(f"dialog: title={title!r} text={body!r}")
            dialog.close()
            _pump(app, rounds=12)

        _capture_widget(window, resolved_shots / "03-after-run.png")

        probe_rows = _read_probe_rows(window, QtWidgets)
        if not probe_rows or any(parsed is None for _, _, parsed, _ in probe_rows):
            for problem in _read_problem_lines(window):
                print(f"problem: {problem}")
            print("FAIL: no solved probe value in probes_table after clicking Run")
            return 1

        checks: list[tuple[str, float, float, float, float, bool]] = []
        for idx, (qty, comp, solved, _raw) in enumerate(probe_rows):
            assert solved is not None
            expected_item = _match_expected(qty, comp, case.expected, fallback_index=idx)
            expected_val = float(expected_item.value)
            tol = float(expected_item.tolerance)
            abs_err = abs(solved - expected_val)
            passed = abs_err <= tol
            checks.append((expected_item.quantity, solved, expected_val, abs_err, tol, passed))

        overall_passed = all(item[5] for item in checks)
        if overall_passed:
            decisive = max(checks, key=lambda item: item[3])
        else:
            decisive = next(item for item in checks if not item[5])

        label, solved, expected_val, abs_err, tol, _ = decisive
        verdict = "PASS" if overall_passed else "FAIL"
        print(
            f"{verdict}: {label} solved {solved!r} vs expected {expected_val!r} "
            f"(abs err {abs_err!r}, tol {tol!r})"
        )
        return 0 if overall_passed else 1
    finally:
        window.close()
        _pump(app, rounds=6)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the offscreen desktop GUI walkthrough."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    return run_walkthrough(args.case, args.shots)


if __name__ == "__main__":
    raise SystemExit(main())
