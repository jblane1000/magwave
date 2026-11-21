"""Subprocess acceptance test for the offscreen GUI walkthrough CLI (PRD FR-061)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from magwave.cases import Case
from magwave.engines import MagpylibEngine

if sys.platform.startswith("linux") and not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyvista")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "gui_walkthrough.py"
EXAMPLE = ROOT / "examples" / "solve-dipole.json"


def test_gui_walkthrough_solve_dipole(tmp_path: Path) -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["QT_QPA_PLATFORM"] = "offscreen"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--case",
            str(EXAMPLE),
            "--shots",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert completed.returncode == 0, (
        f"gui_walkthrough exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )

    expected_solved = float(
        MagpylibEngine().solve(Case.from_json(EXAMPLE.read_text(encoding="utf-8"))).values()[0]
    )

    assert "PASS" in completed.stdout
    pass_lines = [line for line in completed.stdout.splitlines() if line.startswith("PASS:")]
    assert len(pass_lines) == 1, completed.stdout
    verdict_line = pass_lines[0]
    assert repr(expected_solved) in verdict_line
    assert "1.6e-06" in verdict_line

    for name in ("01-loaded.png", "03-after-run.png"):
        shot_path = (tmp_path / name).resolve()
        assert shot_path.is_file(), f"missing screenshot {shot_path}"
        assert shot_path.stat().st_size > 0, f"empty screenshot {shot_path}"
        assert shot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert str(shot_path) in completed.stdout
