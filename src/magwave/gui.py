"""Optional PySide6 + PyVista desktop UI for MagWave (PRD FR-061, ADR-0004).

All Qt and PyVista imports are deferred until :func:`require_gui` (or a GUI class/runner)
is actually invoked, so importing :mod:`magwave.gui` or :mod:`magwave.cli` stays strictly
Qt-free and succeeds without the ``gui`` extra installed.
"""

from __future__ import annotations

import copy
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .cases import (
    COIL_DIMS,
    MAGNET_DIMS,
    QUANTITY_UNITS,
    Case,
    Coil,
    Dipole,
    Magnet,
)
from .engines.base import Engine, EngineError, Problem, Result, errors_in
from .report import display_case_file, render_report

MISSING_GUI_MESSAGE = 'magwave gui needs the GUI extra: pip install "magwave[gui]"'

_ROLE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("magnet: cuboid", "magnet:cuboid"),
    ("magnet: cylinder", "magnet:cylinder"),
    ("magnet: halbach_cylinder", "magnet:halbach_cylinder"),
    ("coil: loop", "coil:loop"),
    ("coil: solenoid", "coil:solenoid"),
    ("dipole", "dipole"),
    ("steel (permeable)", "steel:permeable"),
)

_QT_CLASSES: dict[str, Any] = {}


class MissingGuiError(RuntimeError):
    """Raised when the desktop UI is requested without the ``gui`` extra installed."""


class RoleAssignmentError(ValueError):
    """Raised when a requested component role/kind change is forbidden by the case model."""


@dataclass(frozen=True)
class Steel:
    """Placeholder permeable-material body used to exercise capability-gap validation (FR-041).

    The Tier 0 analytic engine cannot model permeable material and rejects any case
    containing a :class:`Steel` component via :meth:`Engine.validate`.
    """

    kind: str = "cuboid"
    dims: Mapping[str, float] = None  # type: ignore[assignment]
    mu_r: float = 1000.0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = ""

    def __post_init__(self) -> None:
        if self.dims is None:
            object.__setattr__(self, "dims", {"size_x": 0.02, "size_y": 0.02, "size_z": 0.02})


def with_component(case: Case, component: Any, index: int = 0) -> Case:
    """Return a shallow-copied :class:`Case` with ``components[index]`` replaced.

    Intended for constructing capability-gap cases (e.g. injecting :class:`Steel`)
    that bypass ``Case`` schema construction so an engine's ``validate()`` rejects them.
    """
    components = list(case.components)
    components[index] = component
    clone = copy.copy(case)
    object.__setattr__(clone, "components", tuple(components))
    return clone


def require_gui() -> tuple[Any, Any, Any, Any, Any]:
    """Import and return ``(PySide6, QtCore, QtGui, QtWidgets, pyvista)`` or raise :class:`MissingGuiError`."""
    try:
        import PySide6
        import pyvista as pv
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise MissingGuiError(MISSING_GUI_MESSAGE) from exc
    return PySide6, QtCore, QtGui, QtWidgets, pv


def has_live_display() -> bool:
    """Return ``True`` only when Qt is not offscreen and a graphical display is available."""
    platform = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
    if platform == "offscreen":
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _default_engine() -> Engine:
    from .engines.magpylib_engine import MagpylibEngine

    return MagpylibEngine()


def _component_role(component: Any) -> str:
    if isinstance(component, Magnet):
        return "magnet"
    if isinstance(component, Coil):
        return "coil"
    if isinstance(component, Dipole):
        return "dipole"
    if isinstance(component, Steel):
        return "steel"
    return type(component).__name__.lower()


def _component_kind(component: Any) -> str:
    if isinstance(component, Magnet | Coil | Steel):
        return component.kind
    if isinstance(component, Dipole):
        return "point_dipole"
    return "unknown"


def _component_option_key(component: Any) -> str:
    if isinstance(component, Magnet):
        return f"magnet:{component.kind}"
    if isinstance(component, Coil):
        return f"coil:{component.kind}"
    if isinstance(component, Dipole):
        return "dipole"
    if isinstance(component, Steel):
        return "steel:permeable"
    return _component_role(component)


def _component_params_text(component: Any) -> str:
    if isinstance(component, Magnet):
        dims = ", ".join(f"{k}={v:g} m" for k, v in component.dims.items())
        demag = f", demag={component.demag_curve}" if component.demag_curve else ""
        return f"Br={component.br} T, center={component.center} m, {dims}{demag}"
    if isinstance(component, Coil):
        dims = ", ".join(f"{k}={v:g} m" for k, v in component.dims.items())
        return f"I={component.current:g} A, turns={component.turns}, center={component.center} m, {dims}"
    if isinstance(component, Dipole):
        return f"moment={component.moment} A m^2, center={component.center} m"
    if isinstance(component, Steel):
        dims = ", ".join(f"{k}={v:g} m" for k, v in component.dims.items())
        return f"mu_r={component.mu_r:g}, center={component.center} m, {dims}"
    return repr(component)


def _adapt_magnet_dims(component: Magnet, target_kind: str) -> dict[str, float]:
    dims = component.dims
    if target_kind == component.kind:
        return dict(dims)
    if component.kind == "cuboid":
        sx, sy, sz = dims["size_x"], dims["size_y"], dims["size_z"]
        radius = 0.5 * min(sx, sy)
        inner = 0.5 * radius
        length = sz
    elif component.kind == "cylinder":
        radius = dims["radius"]
        inner = 0.5 * radius
        length = dims["length"]
        sx = sy = 2.0 * radius
        sz = length
    else:
        radius = dims["outer_radius"]
        inner = dims["inner_radius"]
        length = 2.0 * radius
        sx = sy = sz = 2.0 * radius

    if target_kind == "cuboid":
        return {"size_x": sx, "size_y": sy, "size_z": sz}
    if target_kind == "cylinder":
        return {"radius": radius, "length": length}
    return {"inner_radius": inner, "outer_radius": radius}


def _adapt_coil_dims(component: Coil, target_kind: str) -> dict[str, float]:
    dims = component.dims
    if target_kind == component.kind:
        return dict(dims)
    if target_kind == "loop":
        return {"radius": dims["outer_radius"]}
    radius = dims["radius"]
    return {"inner_radius": radius, "outer_radius": radius, "length": radius}


def convert_component_role(component: Any, target: str) -> Any:
    """Convert ``component`` to ``target`` role/kind where permitted by :mod:`magwave.cases`.

    Raises :class:`RoleAssignmentError` with an explanation when the model forbids the edit.
    """
    raw = target.strip().lower()
    label = getattr(component, "name", "") or _component_role(component)

    if raw in ("steel", "permeable", "steel:permeable"):
        raise RoleAssignmentError(
            f"Cannot assign {label!r} to permeable steel: case schema magwave.case/1 models "
            "free-space sources (magnet, coil, dipole) only; permeable material requires Tier 1 FEM."
        )

    if isinstance(component, Magnet):
        if raw == "magnet":
            return component
        kind = raw.split(":", 1)[1] if raw.startswith("magnet:") else raw
        if kind in MAGNET_DIMS:
            return replace(component, kind=kind, dims=_adapt_magnet_dims(component, kind))
        raise RoleAssignmentError(
            f"Model forbids changing magnet {label!r} to {target!r}: a Magnet carries "
            "remanence Br and can only take magnet kinds (cuboid, cylinder, halbach_cylinder)."
        )

    if isinstance(component, Coil):
        if raw == "coil":
            return component
        kind = raw.split(":", 1)[1] if raw.startswith("coil:") else raw
        if kind in COIL_DIMS:
            return replace(component, kind=kind, dims=_adapt_coil_dims(component, kind))
        raise RoleAssignmentError(
            f"Model forbids changing coil {label!r} to {target!r}: a Coil carries current and "
            "turns and can only take coil kinds (loop, solenoid)."
        )

    if isinstance(component, Dipole):
        if raw in ("dipole", "dipole:dipole"):
            return component
        raise RoleAssignmentError(
            f"Model forbids changing dipole {label!r} to {target!r}: a Dipole is an ideal "
            "point moment with no geometric dimensions."
        )

    raise RoleAssignmentError(
        f"Model forbids reassigning component {label!r} of type {type(component).__name__}."
    )


def _component_mesh(pv: Any, component: Any) -> Any:
    cx, cy, cz = component.center
    if isinstance(component, Magnet):
        dims = component.dims
        if component.kind == "cuboid":
            hx = 0.5 * dims["size_x"]
            hy = 0.5 * dims["size_y"]
            hz = 0.5 * dims["size_z"]
            return pv.Box(bounds=(cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz))
        if component.kind == "cylinder":
            return pv.Cylinder(
                center=component.center,
                direction=(0.0, 0.0, 1.0),
                radius=dims["radius"],
                height=dims["length"],
            )
        height = max(2.0 * dims["outer_radius"], 0.01)
        outer = pv.Cylinder(
            center=component.center,
            direction=(0.0, 0.0, 1.0),
            radius=dims["outer_radius"],
            height=height,
            capping=False,
        )
        inner = pv.Cylinder(
            center=component.center,
            direction=(0.0, 0.0, 1.0),
            radius=dims["inner_radius"],
            height=height,
            capping=False,
        )
        return outer + inner
    if isinstance(component, Coil):
        dims = component.dims
        if component.kind == "loop":
            loop = pv.Circle(radius=dims["radius"])
            loop.translate(component.center, inplace=True)
            return loop
        return pv.Cylinder(
            center=component.center,
            direction=(0.0, 0.0, 1.0),
            radius=dims["outer_radius"],
            height=dims["length"],
            capping=False,
        )
    if isinstance(component, Steel):
        dims = component.dims
        if component.kind == "cylinder" and "radius" in dims and "length" in dims:
            return pv.Cylinder(
                center=component.center,
                direction=(0.0, 0.0, 1.0),
                radius=dims["radius"],
                height=dims["length"],
            )
        hx = 0.5 * dims.get("size_x", 0.02)
        hy = 0.5 * dims.get("size_y", 0.02)
        hz = 0.5 * dims.get("size_z", 0.02)
        return pv.Box(bounds=(cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz))
    return pv.Sphere(radius=0.03, center=component.center)


def _scalar_bar_title(result: Result | None) -> str:
    if result is None or not result.samples:
        return ""
    units = [f"{sample.quantity} ({sample.unit})" for sample in result.samples]
    seen: list[str] = []
    for item in units:
        if item not in seen:
            seen.append(item)
    return ", ".join(seen)


def _scene_extent(component_meshes: list[tuple[str, Any]], points: np.ndarray) -> float:
    coords: list[np.ndarray] = []
    if len(points):
        coords.append(points)
    for _, mesh in component_meshes:
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        coords.append(np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]], dtype=float))
    if not coords:
        return 1.0
    stacked = np.vstack(coords)
    span = np.max(stacked, axis=0) - np.min(stacked, axis=0)
    diag = float(np.linalg.norm(span))
    return diag if diag > 0.0 else 1.0


def _build_qt_classes() -> dict[str, Any]:
    if _QT_CLASSES:
        return _QT_CLASSES

    _, QtCore, _, QtWidgets, pv = require_gui()

    from pyvista.plotting.plotter import BasePlotter
    from pyvista.plotting.render_window_interactor import RenderWindowInteractor
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    class _QtPlotter(QVTKRenderWindowInteractor, BasePlotter):
        # ``QVTKRenderWindowInteractor`` derives from ``QWidget``, so in this MRO
        # Qt's own ``render``/``update``/``close`` shadow ``BasePlotter``'s.
        # PyVista internals call ``self.parent.render()`` during ``add_mesh``, and
        # ``QWidget.render`` requires a QPaintDevice argument, which raises
        # ``TypeError: not enough arguments``. Rebind the three colliding names to
        # their BasePlotter implementations so the object behaves as a plotter
        # rather than as a bare widget. See the regression test in test_gui.py.
        render = BasePlotter.render
        update = BasePlotter.update
        close = BasePlotter.close

        def __init__(self, widget_parent: Any = None) -> None:
            QVTKRenderWindowInteractor.__init__(self, widget_parent)
            BasePlotter.__init__(self)
            self._initialized = False
            self.ren_win = self.GetRenderWindow()
            for renderer in self.renderers:
                self.ren_win.AddRenderer(renderer)
            self.iren = RenderWindowInteractor(
                self, light_follow_camera=False, interactor=self._Iren
            )
            self.iren.set_render_window(self.ren_win)
            self.reset_key_events()
            self.iren.enable_interactor_style()
            self.set_background(self._theme.background)
            self._Iren.Initialize()
            self._initialized = True

        def closeEvent(self, event: Any) -> None:  # noqa: N802
            BasePlotter.close(self)
            QVTKRenderWindowInteractor.closeEvent(self, event)

    def _create_embedded_plotter(parent: Any) -> Any:
        return _QtPlotter(parent)

    class FieldViewport(QtWidgets.QWidget):
        """3D PyVista viewport showing component geometry, probes, and vector glyphs.

        In headless or ``QT_QPA_PLATFORM=offscreen`` sessions without a display context,
        it constructs the genuine PyVista ``PolyData`` datasets in memory without claiming
        a live OpenGL render took place.
        """

        def __init__(self, parent: Any = None) -> None:
            super().__init__(parent)
            self._pv = pv
            self.component_meshes: list[tuple[str, Any]] = []
            self.probe_mesh: Any | None = None
            self.glyph_mesh: Any | None = None
            self.scalar_bar_title: str = ""
            self.is_rendering_live: bool = False
            self.plotter: Any | None = None

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)

            self.status_label = QtWidgets.QLabel(
                "No case loaded. Open a case to build geometry and probe datasets."
            )
            self.status_label.setWordWrap(True)
            self.status_label.setObjectName("viewport_status_label")

            if has_live_display():
                try:
                    self.plotter = _create_embedded_plotter(self)
                    self.is_rendering_live = True
                    layout.addWidget(self.plotter, stretch=1)
                except Exception as exc:  # pragma: no cover - hardware/driver dependent
                    self.plotter = None
                    self.is_rendering_live = False
                    self.status_label.setText(
                        f"PyVista live 3D render unavailable ({exc}); datasets built in memory."
                    )
            layout.addWidget(self.status_label)

        def update_scene(self, case: Case | None, result: Result | None = None) -> None:
            self.component_meshes = []
            self.probe_mesh = None
            self.glyph_mesh = None
            self.scalar_bar_title = ""

            if case is None:
                if self.plotter is not None:
                    self.plotter.clear()
                self.status_label.setText("No case loaded.")
                return

            for idx, component in enumerate(case.components):
                label = getattr(component, "name", "") or f"#{idx}"
                self.component_meshes.append((label, _component_mesh(self._pv, component)))

            points = np.array([probe.point for probe in case.probes], dtype=float)
            self.probe_mesh = self._pv.PolyData(points)
            if result is not None and len(result.samples) == len(case.probes):
                self.probe_mesh.point_data["value"] = np.asarray(result.values(), dtype=float)
                self.probe_mesh.point_data["vector"] = np.asarray(
                    [sample.vector for sample in result.samples], dtype=float
                )
                extent = _scene_extent(self.component_meshes, points)
                factor = max(0.05, 0.15 * extent)
                self.glyph_mesh = self.probe_mesh.glyph(orient="vector", scale=False, factor=factor)
                self.scalar_bar_title = _scalar_bar_title(result)

            self._render_if_live()
            self._update_status_text(case, result)

        def _render_if_live(self) -> None:
            if not self.is_rendering_live or self.plotter is None:
                return
            self.plotter.clear()
            for _, mesh in self.component_meshes:
                self.plotter.add_mesh(mesh, color="steelblue", opacity=0.45, show_edges=True)
            if self.probe_mesh is not None:
                if "value" in self.probe_mesh.point_data:
                    self.plotter.add_mesh(
                        self.probe_mesh,
                        scalars="value",
                        render_points_as_spheres=True,
                        point_size=14,
                        scalar_bar_args={"title": self.scalar_bar_title},
                    )
                else:
                    self.plotter.add_mesh(
                        self.probe_mesh,
                        color="crimson",
                        render_points_as_spheres=True,
                        point_size=12,
                    )
            if self.glyph_mesh is not None and self.glyph_mesh.n_points > 0:
                self.plotter.add_mesh(self.glyph_mesh, color="darkorange")
            self.plotter.add_axes()
            self.plotter.reset_camera()
            self.plotter.render()

        def _update_status_text(self, case: Case, result: Result | None) -> None:
            mode = (
                "Live 3D PyVista viewport"
                if self.is_rendering_live
                else "Headless mode (no live OpenGL render context; PyVista datasets built in memory)"
            )
            if result is None:
                self.status_label.setText(
                    f"{mode} — showing {len(self.component_meshes)} component(s) and "
                    f"{len(case.probes)} unsolved probe point(s). Slices/iso-surfaces/streamlines "
                    "are not implemented in v1."
                )
            else:
                self.status_label.setText(
                    f"{mode} — showing {len(self.component_meshes)} component(s), "
                    f"{len(case.probes)} probe(s) coloured by {self.scalar_bar_title}, and "
                    "B-vector glyphs. Slices/iso-surfaces/streamlines are not implemented in v1."
                )

        def closeEvent(self, event: Any) -> None:  # noqa: N802
            if self.plotter is not None:
                self.plotter.close()
                self.plotter = None
            super().closeEvent(event)

    class MainWindow(QtWidgets.QMainWindow):
        """MagWave v1 desktop window: inspect case, assign roles, run, view, export report."""

        def __init__(
            self,
            case: Case | str | Path | None = None,
            *,
            engine: Engine | None = None,
            case_file: str | Path | None = None,
            parent: Any = None,
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("MagWave — Desktop UI v1")
            self.resize(1160, 760)

            self.engine: Engine = engine if engine is not None else _default_engine()
            self.case: Case | None = None
            self.case_file: Path | str | None = case_file
            self.result: Result | None = None
            self.problems: tuple[Problem, ...] = ()

            self._build_actions_and_menus()
            self._build_layout()
            self._refresh_provenance()

            if case is not None:
                self.load_case(case, case_file=case_file)
            else:
                self._refresh_all()

        def _build_actions_and_menus(self) -> None:
            file_menu = self.menuBar().addMenu("&File")
            sim_menu = self.menuBar().addMenu("&Simulation")

            self.open_action = file_menu.addAction("&Open Case...")
            self.open_action.setShortcut("Ctrl+O")
            self.open_action.triggered.connect(self._on_open_clicked)

            self.export_action = file_menu.addAction("&Export Report...")
            self.export_action.setShortcut("Ctrl+E")
            self.export_action.triggered.connect(self._on_export_clicked)

            file_menu.addSeparator()
            self.quit_action = file_menu.addAction("&Quit")
            self.quit_action.setShortcut("Ctrl+Q")
            self.quit_action.triggered.connect(self.close)

            self.run_action = sim_menu.addAction("&Run")
            self.run_action.setShortcut("F5")
            self.run_action.triggered.connect(self.run_case)

        def _build_layout(self) -> None:
            central = QtWidgets.QWidget(self)
            self.setCentralWidget(central)
            root = QtWidgets.QVBoxLayout(central)

            # Top action bar + case summary + provenance
            toolbar_row = QtWidgets.QHBoxLayout()
            self.open_button = QtWidgets.QPushButton("Open Case...")
            self.open_button.clicked.connect(self._on_open_clicked)
            self.run_button = QtWidgets.QPushButton("Run")
            self.run_button.clicked.connect(self.run_case)
            self.export_button = QtWidgets.QPushButton("Export Report...")
            self.export_button.clicked.connect(self._on_export_clicked)
            toolbar_row.addWidget(self.open_button)
            toolbar_row.addWidget(self.run_button)
            toolbar_row.addWidget(self.export_button)
            toolbar_row.addStretch(1)
            root.addLayout(toolbar_row)

            header_row = QtWidgets.QHBoxLayout()

            case_box = QtWidgets.QGroupBox("Case")
            case_form = QtWidgets.QFormLayout(case_box)
            self.case_name_label = QtWidgets.QLabel("—")
            self.case_file_label = QtWidgets.QLabel("—")
            self.case_description_label = QtWidgets.QLabel("—")
            self.case_description_label.setWordWrap(True)
            self.case_sources_label = QtWidgets.QLabel("—")
            self.case_sources_label.setWordWrap(True)
            case_form.addRow("Name:", self.case_name_label)
            case_form.addRow("File:", self.case_file_label)
            case_form.addRow("Description:", self.case_description_label)
            case_form.addRow("Sources:", self.case_sources_label)
            header_row.addWidget(case_box, stretch=1)

            prov_box = QtWidgets.QGroupBox("Provenance (Engine & Rung)")
            prov_form = QtWidgets.QFormLayout(prov_box)
            self.engine_id_label = QtWidgets.QLabel("—")
            self.engine_version_label = QtWidgets.QLabel("—")
            self.rung_label = QtWidgets.QLabel("—")
            self.method_label = QtWidgets.QLabel("—")
            self.method_label.setWordWrap(True)
            prov_form.addRow("Engine ID:", self.engine_id_label)
            prov_form.addRow("Engine Version:", self.engine_version_label)
            prov_form.addRow("Rung:", self.rung_label)
            prov_form.addRow("Method:", self.method_label)
            header_row.addWidget(prov_box, stretch=1)

            root.addLayout(header_row)

            # Main splitter: left data tables & problems, right 3D PyVista viewport
            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

            left_panel = QtWidgets.QWidget()
            left_layout = QtWidgets.QVBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)

            comp_group = QtWidgets.QGroupBox("Components / Role Assignment")
            comp_layout = QtWidgets.QVBoxLayout(comp_group)
            self.components_table = QtWidgets.QTableWidget(0, 5)
            self.components_table.setHorizontalHeaderLabels(
                ["Name", "Role", "Kind", "Assign Role / Kind", "Parameters (SI)"]
            )
            self.components_table.horizontalHeader().setStretchLastSection(True)
            comp_layout.addWidget(self.components_table)
            left_layout.addWidget(comp_group, stretch=2)

            probe_group = QtWidgets.QGroupBox("Probes")
            probe_layout = QtWidgets.QVBoxLayout(probe_group)
            self.probes_table = QtWidgets.QTableWidget(0, 6)
            self.probes_table.setHorizontalHeaderLabels(
                ["Probe", "Point (m)", "Quantity", "Component", "Solved Value", "Unit"]
            )
            self.probes_table.horizontalHeader().setStretchLastSection(True)
            probe_layout.addWidget(self.probes_table)
            left_layout.addWidget(probe_group, stretch=2)

            exp_group = QtWidgets.QGroupBox("Expected Values")
            exp_layout = QtWidgets.QVBoxLayout(exp_group)
            self.expected_table = QtWidgets.QTableWidget(0, 6)
            self.expected_table.setHorizontalHeaderLabels(
                ["Check", "Quantity", "Expected", "Unit", "Tolerance", "Source"]
            )
            self.expected_table.horizontalHeader().setStretchLastSection(True)
            exp_layout.addWidget(self.expected_table)
            left_layout.addWidget(exp_group, stretch=1)

            prob_group = QtWidgets.QGroupBox("Validation & Solver Problems")
            prob_layout = QtWidgets.QVBoxLayout(prob_group)
            self.problems_list = QtWidgets.QListWidget()
            prob_layout.addWidget(self.problems_list)
            left_layout.addWidget(prob_group, stretch=1)

            view_group = QtWidgets.QGroupBox("3D Field View (PyVista)")
            view_layout = QtWidgets.QVBoxLayout(view_group)
            self.viewport = FieldViewport(self)
            view_layout.addWidget(self.viewport)

            splitter.addWidget(left_panel)
            splitter.addWidget(view_group)
            splitter.setSizes([640, 520])
            root.addWidget(splitter, stretch=1)

        def load_case(
            self,
            source: Case | str | Path,
            *,
            case_file: str | Path | None = None,
        ) -> Case:
            """Load a :class:`Case` object or JSON case path into the window."""
            if isinstance(source, Case):
                loaded = source
                resolved_file: str | Path = (
                    case_file if case_file is not None else f"{loaded.name}.json"
                )
            else:
                path = Path(source)
                loaded = Case.from_json(path.read_text(encoding="utf-8"))
                resolved_file = case_file if case_file is not None else path

            self.case = loaded
            self.case_file = resolved_file
            self.result = None
            self.problems = ()
            self._refresh_all()
            self.statusBar().showMessage(f"Loaded case {loaded.name}")
            return loaded

        def assign_role(self, index: int, role_or_kind: str) -> bool:
            """Attempt to change ``case.components[index]`` role/kind; report if forbidden."""
            if self.case is None:
                return False
            if index < 0 or index >= len(self.case.components):
                raise IndexError(f"component index {index} out of range")

            current = self.case.components[index]
            try:
                updated = convert_component_role(current, role_or_kind)
            except RoleAssignmentError as exc:
                problem = Problem("E_ROLE_FORBIDDEN", str(exc), "error")
                self.problems = (problem,)
                self._refresh_problems()
                self._sync_component_combo(index, current)
                self.statusBar().showMessage(str(exc))
                return False

            components = list(self.case.components)
            components[index] = updated
            if all(isinstance(c, Magnet | Coil | Dipole) for c in components):
                self.case = replace(self.case, components=tuple(components))
            else:
                clone = copy.copy(self.case)
                object.__setattr__(clone, "components", tuple(components))
                self.case = clone
            self.result = None
            self.problems = ()
            self._refresh_all()
            self.statusBar().showMessage(
                f"Updated component #{index} to {_component_role(updated)} ({_component_kind(updated)})"
            )
            return True

        def run_case(self) -> Result | None:
            """Validate with ``engine.validate()`` first; solve only when no errors exist."""
            if self.case is None:
                return None

            validation = self.engine.validate(self.case)
            if errors_in(validation):
                self.result = None
                self.problems = tuple(validation)
                self._refresh_probes()
                self._refresh_problems()
                self._refresh_provenance()
                self.viewport.update_scene(self.case, result=None)
                self.statusBar().showMessage(
                    f"Solve blocked: {len(errors_in(validation))} validation error(s)."
                )
                return None

            try:
                result = self.engine.solve(self.case)
            except EngineError as exc:
                self.result = None
                self.problems = exc.problems
                self._refresh_probes()
                self._refresh_problems()
                self._refresh_provenance()
                self.viewport.update_scene(self.case, result=None)
                self.statusBar().showMessage(f"Engine error: {exc}")
                return None

            self.result = result
            self.problems = result.problems
            self._refresh_probes()
            self._refresh_problems()
            self._refresh_provenance()
            self.viewport.update_scene(self.case, result=result)
            self.statusBar().showMessage(
                f"Solved {self.case.name} ({result.engine_id} {result.engine_version}, "
                f"rung {result.rung})"
            )
            return result

        def export_report(self, out_path: str | Path) -> Path:
            """Render the deterministic Markdown report via :func:`render_report` and write it."""
            if self.case is None:
                raise RuntimeError("No case loaded to export.")
            case_ref = self.case_file if self.case_file is not None else f"{self.case.name}.json"
            markdown = render_report(self.case, self.engine, case_file=case_ref)
            target = Path(out_path)
            with target.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(markdown)
            self.statusBar().showMessage(f"Exported report to {target.name}")
            return target

        def shown_probe_values(self) -> tuple[float, ...]:
            """Return the solved scalar values currently displayed in the probes table."""
            values: list[float] = []
            for row in range(self.probes_table.rowCount()):
                item = self.probes_table.item(row, 4)
                if item is None:
                    continue
                text = item.text().strip()
                if text and text != "—":
                    values.append(float(text))
            return tuple(values)

        def problem_messages(self) -> list[str]:
            """Return the problem lines currently displayed in the problems list widget."""
            return [
                self.problems_list.item(row).text() for row in range(self.problems_list.count())
            ]

        def _refresh_all(self) -> None:
            has_case = self.case is not None
            self.run_button.setEnabled(has_case)
            self.run_action.setEnabled(has_case)
            self.export_button.setEnabled(has_case)
            self.export_action.setEnabled(has_case)
            self._refresh_header()
            self._refresh_components()
            self._refresh_probes()
            self._refresh_expected()
            self._refresh_problems()
            self._refresh_provenance()
            self.viewport.update_scene(self.case, result=self.result)

        def _refresh_header(self) -> None:
            if self.case is None:
                self.case_name_label.setText("—")
                self.case_file_label.setText("—")
                self.case_description_label.setText("—")
                self.case_sources_label.setText("—")
                return
            self.case_name_label.setText(self.case.name)
            case_ref = self.case_file if self.case_file is not None else f"{self.case.name}.json"
            self.case_file_label.setText(display_case_file(case_ref))
            self.case_description_label.setText(self.case.description)
            self.case_sources_label.setText("; ".join(self.case.sources))

        def _refresh_provenance(self) -> None:
            if self.result is not None:
                self.engine_id_label.setText(self.result.engine_id)
                self.engine_version_label.setText(self.result.engine_version)
                self.rung_label.setText(self.result.rung)
                self.method_label.setText(self.result.method)
                return
            caps = self.engine.capabilities()
            self.engine_id_label.setText(caps.engine_id)
            self.engine_version_label.setText(caps.engine_version)
            self.rung_label.setText(caps.rung)
            default_method = getattr(self.engine, "method", "pending solve")
            self.method_label.setText(str(default_method))

        def _refresh_components(self) -> None:
            self.components_table.setRowCount(0)
            if self.case is None:
                return
            self.components_table.setRowCount(len(self.case.components))
            for row, component in enumerate(self.case.components):
                name = getattr(component, "name", "") or f"#{row}"
                self.components_table.setItem(row, 0, self._ro_item(name))
                self.components_table.setItem(row, 1, self._ro_item(_component_role(component)))
                self.components_table.setItem(row, 2, self._ro_item(_component_kind(component)))

                combo = QtWidgets.QComboBox()
                for label, key in _ROLE_OPTIONS:
                    combo.addItem(label, key)
                self.components_table.setCellWidget(row, 3, combo)
                self._sync_component_combo(row, component)
                combo.activated.connect(
                    lambda _idx, r=row, box=combo: self.assign_role(r, str(box.currentData()))
                )

                self.components_table.setItem(
                    row, 4, self._ro_item(_component_params_text(component))
                )

        def _sync_component_combo(self, row: int, component: Any) -> None:
            combo = self.components_table.cellWidget(row, 3)
            if not isinstance(combo, QtWidgets.QComboBox):
                return
            key = _component_option_key(component)
            idx = combo.findData(key)
            combo.blockSignals(True)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        def _refresh_probes(self) -> None:
            self.probes_table.setRowCount(0)
            if self.case is None:
                return
            solved = self.result.values() if self.result is not None else ()
            self.probes_table.setRowCount(len(self.case.probes))
            for row, probe in enumerate(self.case.probes):
                point_text = ", ".join(repr(float(c)) for c in probe.point)
                if row < len(solved):
                    val_text = repr(float(solved[row]))
                    unit_text = self.result.samples[row].unit if self.result else ""
                else:
                    val_text = "—"
                    unit_text = QUANTITY_UNITS.get(probe.quantity, "")
                self.probes_table.setItem(row, 0, self._ro_item(f"probe #{row}"))
                self.probes_table.setItem(row, 1, self._ro_item(f"({point_text})"))
                self.probes_table.setItem(row, 2, self._ro_item(probe.quantity))
                self.probes_table.setItem(row, 3, self._ro_item(probe.component))
                self.probes_table.setItem(row, 4, self._ro_item(val_text))
                self.probes_table.setItem(row, 5, self._ro_item(unit_text))

        def _refresh_expected(self) -> None:
            self.expected_table.setRowCount(0)
            if self.case is None:
                return
            self.expected_table.setRowCount(len(self.case.expected))
            for row, item in enumerate(self.case.expected):
                self.expected_table.setItem(row, 0, self._ro_item(item.check))
                self.expected_table.setItem(row, 1, self._ro_item(item.quantity))
                self.expected_table.setItem(row, 2, self._ro_item(repr(float(item.value))))
                self.expected_table.setItem(row, 3, self._ro_item(item.unit))
                self.expected_table.setItem(row, 4, self._ro_item(repr(float(item.tolerance))))
                self.expected_table.setItem(row, 5, self._ro_item(item.source))

        def _refresh_problems(self) -> None:
            self.problems_list.clear()
            for problem in self.problems:
                self.problems_list.addItem(str(problem))

        @staticmethod
        def _ro_item(text: str) -> Any:
            item = QtWidgets.QTableWidgetItem(text)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            return item

        def _on_open_clicked(self) -> None:
            start_dir = str(Path(self.case_file).parent) if self.case_file is not None else "."
            chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open MagWave Case", start_dir, "MagWave Cases (*.json);;All Files (*)"
            )
            if not chosen:
                return
            try:
                self.load_case(chosen)
            except Exception as exc:
                self.problems = (Problem("E_CASE_LOAD", str(exc), "error"),)
                self._refresh_problems()
                self.statusBar().showMessage(f"Failed to load case: {exc}")

        def _on_export_clicked(self) -> None:
            if self.case is None:
                return
            default_name = f"{self.case.name}-report.md"
            chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export MagWave Report",
                default_name,
                "Markdown Reports (*.md);;All Files (*)",
            )
            if not chosen:
                return
            try:
                self.export_report(chosen)
            except (EngineError, OSError) as exc:
                if isinstance(exc, EngineError):
                    self.problems = exc.problems
                else:
                    self.problems = (Problem("E_REPORT_EXPORT", str(exc), "error"),)
                self._refresh_problems()
                self.statusBar().showMessage(f"Export failed: {exc}")

        def closeEvent(self, event: Any) -> None:  # noqa: N802
            self.viewport.close()
            super().closeEvent(event)

    _QT_CLASSES["EmbeddedPlotter"] = _QtPlotter
    _QT_CLASSES["FieldViewport"] = FieldViewport
    _QT_CLASSES["MainWindow"] = MainWindow
    return _QT_CLASSES


def __getattr__(name: str) -> Any:
    if name in ("FieldViewport", "MainWindow"):
        return _build_qt_classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def launch_gui(case_path: str | Path | None = None, *, engine: Engine | None = None) -> int:
    """Launch the desktop window for ``case_path`` (or an empty window if ``None``)."""
    if not has_live_display():
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _, _, _, QtWidgets, _ = require_gui()
    window_cls = _build_qt_classes()["MainWindow"]
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    try:
        window = window_cls(case_path, engine=engine)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"magwave gui: {case_path}: {' '.join(str(exc).splitlines())}", file=sys.stderr)
        return 1
    window.show()
    return int(app.exec())
