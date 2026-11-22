"""The architecture fence is a lint rule, and this proves it fires.

The core must stay Qt-free and OCP-free so the library installs and imports without the
`gui` or CAD extras. That used to be enforced only by installation — CI built core alone
and the import failed there, minutes later and in a different job from the line that
caused it. `flake8-tidy-imports` now refuses the import where it is written, with a
message that names the boundary.

A rule nobody has seen fire is a hope, so these tests drive the real linter over real
snippets: the banned import must be rejected in a core module and accepted in the module
declared to own it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUFF = shutil.which("ruff") or str(REPO / ".venv" / "bin" / "ruff")

pytestmark = pytest.mark.skipif(
    not Path(RUFF).exists(), reason="ruff is not installed in this environment"
)


def _ruff_on(source: str, filename: str) -> subprocess.CompletedProcess:
    """Lint `source` as if it lived at `filename`, using this repository's config."""
    return subprocess.run(
        [RUFF, "check", "--no-cache", "--select", "TID251", "--stdin-filename", filename, "-"],
        input=source,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
    )


def test_qt_import_is_refused_in_a_core_module():
    proc = _ruff_on("from PySide6.QtWidgets import QApplication\n", "src/magwave/_probe.py")
    assert proc.returncode == 1, "the fence did not fire on a Qt import in core"
    assert "GUI boundary" in proc.stdout + proc.stderr


def test_qt_import_is_allowed_in_the_declared_boundary_module():
    proc = _ruff_on("from PySide6.QtWidgets import QApplication\n", "src/magwave/gui.py")
    assert proc.returncode == 0, f"gui.py must be allowed to import Qt: {proc.stdout}"


def test_ocp_import_is_refused_outside_the_cad_package():
    proc = _ruff_on("from OCP.BRep import BRep_Tool\n", "src/magwave/fem2d/_probe.py")
    assert proc.returncode == 1, "the fence did not fire on an OCP import outside cad/"
    assert "CAD boundary" in proc.stdout + proc.stderr


def test_ocp_import_is_allowed_inside_the_cad_package():
    proc = _ruff_on("from OCP.BRep import BRep_Tool\n", "src/magwave/cad/step.py")
    assert proc.returncode == 0, f"cad/ must be allowed to import OCP: {proc.stdout}"


def test_the_repository_itself_is_lint_clean():
    """The gate every engine round must satisfy: ruff check and ruff format both pass."""
    check = subprocess.run(
        [RUFF, "check", "--no-cache", "."], capture_output=True, text=True, timeout=300, cwd=REPO
    )
    assert check.returncode == 0, check.stdout[-2000:]
    fmt = subprocess.run(
        [RUFF, "format", "--check", "."],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO,
    )
    assert fmt.returncode == 0, fmt.stdout[-2000:]
