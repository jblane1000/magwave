"""Offscreen desktop UI tests (PRD FR-061, ADR-0004).

All widget tests run under ``QT_QPA_PLATFORM=offscreen`` and skip at collection time
when the optional ``gui`` extra (``PySide6``) is not installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import magpylib as magpy
import numpy as np
import pytest

if sys.platform.startswith("linux") and not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyvista")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from magwave.benchmarks import load_corpus  # noqa: E402
from magwave.cases import Case, Coil, Dipole, ExpectedValue, Magnet, Probe  # noqa: E402
from magwave.cli import main  # noqa: E402
from magwave.engines import MagpylibEngine  # noqa: E402
from magwave.engines.base import Capabilities, EngineError, Problem, Result  # noqa: E402
from magwave.gui import (  # noqa: E402
    MISSING_GUI_MESSAGE,
    MainWindow,
    Steel,
    _build_qt_classes,
    launch_gui,
    with_component,
)
from magwave.report import render_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "solve-dipole.json"
CORPUS = {case.name: case for case in load_corpus()}


def test_window_constructs_with_case_and_displays_metadata(qtbot) -> None:
    window = MainWindow(EXAMPLE)
    qtbot.addWidget(window)

    assert window.case is not None
    assert window.case_name_label.text() == "dipole_axis_Bz"
    assert window.case_file_label.text() == "examples/solve-dipole.json"

    assert window.components_table.rowCount() == 1
    assert window.components_table.item(0, 0).text() == "m1"
    assert window.components_table.item(0, 1).text() == "dipole"

    assert window.probes_table.rowCount() == 1
    assert window.expected_table.rowCount() == 1
    assert window.expected_table.item(0, 0).text() == "dipole_B"

    # Provenance (engine id, engine version, rung, method) is always visible
    caps = MagpylibEngine().capabilities()
    assert window.engine_id_label.text() == caps.engine_id
    assert window.engine_version_label.text() == caps.engine_version
    assert window.rung_label.text() == caps.rung
    assert window.method_label.text()

    # In offscreen CI, PyVista datasets are built without faking a live render context
    assert window.viewport.probe_mesh is not None
    assert len(window.viewport.component_meshes) == 1
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        assert window.viewport.is_rendering_live is False


def test_window_displays_multiple_component_names(qtbot) -> None:
    case = Case(
        name="multi_source_demo",
        description="Cuboid magnet plus circular loop",
        sources=("unit test",),
        components=(
            Magnet(
                kind="cuboid",
                dims={"size_x": 0.02, "size_y": 0.02, "size_z": 0.01},
                br=(0.0, 0.0, 1.2),
                name="drive_magnet",
            ),
            Coil(
                kind="loop",
                dims={"radius": 0.05},
                current=2.0,
                turns=10,
                center=(0.0, 0.0, 0.03),
                name="sense_coil",
            ),
        ),
        probes=(Probe(point=(0.0, 0.0, 0.06), quantity="B", component="z"),),
        expected=(
            ExpectedValue(
                check="dipole_B",
                quantity="Bz",
                value=1.0e-3,
                unit="T",
                tolerance=1e-3,
                source="unit test",
            ),
        ),
    )
    window = MainWindow(case)
    qtbot.addWidget(window)

    names = [window.components_table.item(r, 0).text() for r in range(2)]
    assert names == ["drive_magnet", "sense_coil"]
    assert len(window.viewport.component_meshes) == 2


def test_role_assignment_permitted_and_forbidden(qtbot) -> None:
    window = MainWindow(CORPUS["cuboid_axis_Bz"])
    qtbot.addWidget(window)

    # Solving first populates results; changing a permitted kind clears stale results
    assert window.run_case() is not None
    assert len(window.shown_probe_values()) == 1

    assert window.assign_role(0, "cylinder") is True
    assert isinstance(window.case.components[0], Magnet)
    assert window.case.components[0].kind == "cylinder"
    assert window.components_table.item(0, 2).text() == "cylinder"
    assert window.result is None
    assert window.shown_probe_values() == ()

    # Forbidden: assigning permeable steel to a Magnet is rejected loudly, not ignored
    assert window.assign_role(0, "steel:permeable") is False
    assert window.case.components[0].kind == "cylinder"
    assert any("steel" in msg.lower() for msg in window.problem_messages())

    # Forbidden: changing a Magnet to a Coil kind is rejected loudly
    assert window.assign_role(0, "loop") is False
    assert window.case.components[0].kind == "cylinder"
    assert any("forbids" in msg.lower() for msg in window.problem_messages())

    # Coil permitted switch: loop <-> solenoid
    coil_window = MainWindow(CORPUS["loop_axis_Bz"])
    qtbot.addWidget(coil_window)
    assert coil_window.assign_role(0, "solenoid") is True
    assert isinstance(coil_window.case.components[0], Coil)
    assert coil_window.case.components[0].kind == "solenoid"
    assert coil_window.assign_role(0, "coil:loop") is True
    assert coil_window.case.components[0].kind == "loop"

    # Dipole forbidden change
    dipole_window = MainWindow(EXAMPLE)
    qtbot.addWidget(dipole_window)
    assert dipole_window.assign_role(0, "cuboid") is False
    assert isinstance(dipole_window.case.components[0], Dipole)
    assert any("forbids" in msg.lower() for msg in dipole_window.problem_messages())


def test_solve_runs_real_engine_and_matches_result_values(qtbot) -> None:
    window = MainWindow(CORPUS["cuboid_axis_Bz"])
    qtbot.addWidget(window)

    qtbot.mouseClick(window.run_button, QtCore.Qt.MouseButton.LeftButton)

    expected = MagpylibEngine().solve(window.case)
    assert window.result is not None
    assert window.shown_probe_values() == expected.values()
    assert window.engine_id_label.text() == "magpylib"
    assert window.engine_version_label.text() == magpy.__version__
    assert window.rung_label.text() == "second-code"
    assert window.method_label.text() == expected.method

    assert window.viewport.probe_mesh is not None
    np.testing.assert_array_equal(
        window.viewport.probe_mesh.point_data["value"],
        np.asarray(expected.values(), dtype=float),
    )
    assert window.viewport.glyph_mesh is not None
    assert window.viewport.glyph_mesh.n_points > 0
    assert window.viewport.scalar_bar_title == "B (T)"


def test_capability_gap_steel_shows_problem_and_produces_no_result(qtbot) -> None:
    steel_case = with_component(
        CORPUS["cuboid_axis_Bz"],
        Steel(kind="cuboid", name="pole_steel"),
    )
    window = MainWindow(steel_case)
    qtbot.addWidget(window)

    assert window.run_case() is None
    assert window.result is None
    assert window.shown_probe_values() == ()
    messages = window.problem_messages()
    assert any("E_UNSUPPORTED_SOURCE_TYPE" in msg and "Steel" in msg for msg in messages)


@pytest.mark.parametrize(
    ("case_name", "expected_code"),
    [
        ("halbach_bore_B", "E_UNSUPPORTED_GEOMETRY"),
        ("thin_solenoid_axis_Bz", "E_UNSUPPORTED_COIL"),
        ("force_dipole_coaxial_far", "E_UNSUPPORTED_QUANTITY"),
    ],
)
def test_corpus_capability_gaps_block_solve_without_result(
    qtbot, case_name: str, expected_code: str
) -> None:
    window = MainWindow(CORPUS[case_name])
    qtbot.addWidget(window)

    assert window.run_case() is None
    assert window.result is None
    assert window.shown_probe_values() == ()
    assert any(expected_code in msg for msg in window.problem_messages())


def test_solve_catches_engine_error_and_fabricates_no_result(qtbot) -> None:
    class _FailingAfterValidateEngine:
        def capabilities(self) -> Capabilities:
            return Capabilities(
                engine_id="failing-test-engine",
                rung="test-double",
                source_types=frozenset({"dipole"}),
                magnet_kinds=frozenset(),
                coil_kinds=frozenset(),
                quantities=frozenset({"B"}),
                force_methods=(),
                engine_version="0.1",
            )

        def validate(self, case: Case) -> list[Problem]:
            return []

        def solve(self, case: Case) -> Result:
            raise EngineError(
                "failing-test-engine",
                [Problem("E_SOLVE_ABORTED", f"{case.name}: solver refused at runtime", "error")],
            )

    window = MainWindow(EXAMPLE, engine=_FailingAfterValidateEngine())
    qtbot.addWidget(window)

    assert window.run_case() is None
    assert window.result is None
    assert window.shown_probe_values() == ()
    assert any("E_SOLVE_ABORTED" in msg for msg in window.problem_messages())


def test_export_report_matches_render_report_exactly(qtbot, tmp_path: Path) -> None:
    window = MainWindow(EXAMPLE)
    qtbot.addWidget(window)

    window.run_case()
    out_file = tmp_path / "dipole_axis_Bz-report.md"
    written = window.export_report(out_file)

    assert written == out_file
    expected = render_report(window.case, window.engine, case_file=EXAMPLE)
    assert out_file.read_bytes() == expected.encode("utf-8")
    assert str(tmp_path) not in out_file.read_text(encoding="utf-8")
    assert str(ROOT) not in out_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("blocked_module", ["PySide6", "pyvista"])
def test_magwave_gui_without_extra_exits_nonzero_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    blocked_module: str,
) -> None:
    monkeypatch.setitem(sys.modules, blocked_module, None)
    code = main(["gui", str(EXAMPLE)])
    captured = capsys.readouterr()
    assert code != 0
    assert captured.out == ""
    assert captured.err.strip() == MISSING_GUI_MESSAGE

    code_no_arg = main(["gui"])
    captured_no_arg = capsys.readouterr()
    assert code_no_arg != 0
    assert captured_no_arg.out == ""
    assert captured_no_arg.err.strip() == MISSING_GUI_MESSAGE


def test_cli_imports_without_pyside6_in_fresh_subprocess() -> None:
    code = (
        "import sys; sys.modules['PySide6'] = None; sys.modules['pyvista'] = None; "
        "import magwave, magwave.cli, magwave.gui; "
        "rc = magwave.cli.main(['gui']); "
        "assert rc == 1, rc"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr.strip() == MISSING_GUI_MESSAGE


def test_gui_cli_rejects_bench_flags_and_launches_cleanly(
    monkeypatch: pytest.MonkeyPatch, qapp: QtWidgets.QApplication
) -> None:
    for argv in (["gui", "--json", "out.json"], ["gui", "--engine", "magpylib"]):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2

    monkeypatch.setattr(qapp, "exec", lambda: 0)
    assert launch_gui(EXAMPLE) == 0
    for widget in qapp.topLevelWidgets():
        if isinstance(widget, MainWindow):
            widget.close()


def test_embedded_plotter_does_not_let_qt_shadow_pyvista_render() -> None:
    """The embedded plotter must not inherit Qt's ``render``/``update``/``close``.

    ``QVTKRenderWindowInteractor`` is a ``QWidget``, so with the naive MRO Qt's
    ``QWidget.render`` shadows ``pyvista.BasePlotter.render``. PyVista internals
    call ``self.parent.render()`` during ``add_mesh``; ``QWidget.render`` requires
    a ``QPaintDevice`` argument and raises ``TypeError: not enough arguments``.
    That only fires with a live display, so the offscreen suite never saw it.
    This asserts the class rebinds the three colliding names, with no display needed.
    """
    from pyvista.plotting.plotter import BasePlotter

    plotter_cls = _build_qt_classes()["EmbeddedPlotter"]

    for name in ("render", "update", "close"):
        assert getattr(plotter_cls, name) is getattr(BasePlotter, name), (
            f"Qt's {name!r} shadows BasePlotter.{name}; the embedded plotter will "
            "raise TypeError during add_mesh on a real display"
        )


def test_unassigned_manifest_blocks_gui_run_case(qtbot) -> None:
    from dataclasses import replace

    from magwave.cad import Body, Manifest

    base_case = CORPUS["dipole_axis_Bz"]
    manifest = Manifest(
        path="assembly.step",
        format="step",
        units="mm",
        bodies=[
            Body(
                index=0,
                name="magnet",
                role=None,
                volume_mm3=1000.0,
                solids=1,
                bbox=(-10.0, -5.0, -2.5, 10.0, 5.0, 2.5),
                source="step",
            )
        ],
    )
    blocked_case = replace(base_case, manifest=manifest)
    window = MainWindow(blocked_case)
    qtbot.addWidget(window)

    assert window.run_case() is None
    assert window.result is None
    assert window.shown_probe_values() == ()
    assert any("E_UNASSIGNED_BODIES" in msg for msg in window.problem_messages())
