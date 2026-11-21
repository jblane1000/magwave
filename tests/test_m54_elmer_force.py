"""Elmer second-code virtual-work force cross-comparison gate test (M5.4 Gate 6)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "elmer_force.py"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker binary not found on PATH")
def test_elmer_agrees_on_magnet_plate_axial_force() -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    proc = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if proc.returncode == 3:
        skip_line = next(
            (line for line in proc.stdout.splitlines() if line.startswith("SKIP:")),
            proc.stdout.strip() or proc.stderr.strip() or "Elmer unavailable",
        )
        pytest.skip(skip_line)

    if proc.returncode != 0:
        pytest.fail(
            f"tools/elmer_force.py failed (exit {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    assert "PASS" in proc.stdout
    match = re.search(
        r"([0-9]+\.[0-9]+)%\s*\|\s*([0-9]+\.[0-9]+)%\s*\|\s*PASS\b",
        proc.stdout,
    )
    assert match is not None, f"could not parse relative diffs from output:\n{proc.stdout}"
    assert float(match.group(1)) <= 5.0
    assert float(match.group(2)) <= 5.0
