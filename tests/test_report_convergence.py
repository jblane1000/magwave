"""CLI and Markdown convergence reporting tests for ``magwave report`` (FR-042)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from magwave.cli import main

ROOT = Path(__file__).resolve().parents[1]
NONLINEAR_EXAMPLE = ROOT / "examples" / "steel-cylinder-solenoid-nonlinear.json"
LINEAR_EXAMPLE = ROOT / "examples" / "solve-dipole.json"
LINEAR_STEEL_EXAMPLE = ROOT / "examples" / "steel-sphere-uniform-field.json"


def _extract_final_json(markdown: str) -> dict:
    match = re.search(r"```json\n(.*?)\n```\s*\Z", markdown, flags=re.DOTALL)
    assert match is not None, "report must end with a fenced ```json block"
    return json.loads(match.group(1))


def test_report_prints_the_convergence_block(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["report", "--engine", "fem2d", str(NONLINEAR_EXAMPLE)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert captured.err == ""

    text = captured.out
    assert "iterations:" in text
    assert "final_relative_residual:" in text
    assert "nodes:" in text
    assert "elements:" in text
    assert "interface_nodes:" in text
    assert "relaxation:" in text
    assert "material_model:" in text

    payload = _extract_final_json(text)
    conv = payload["convergence"]
    assert 3 <= conv["iterations"] <= 50
    assert conv["final_relative_residual"] <= 1e-8
    assert conv["nodes"] > 0
    assert conv["elements"] > 0
    assert conv["interface_nodes"] > 0
    assert len(conv["residual_history"]) == conv["iterations"]
    assert "Armijo" in conv["relaxation"]
    assert "steel-team20" in conv["material_model"]

    # Deterministic repeat run (and reversed flag order) produces identical bytes.
    rc2 = main(["report", str(NONLINEAR_EXAMPLE), "--engine", "fem2d"])
    assert rc2 == 0
    assert capsys.readouterr().out == text

    # Linear fem2d solve also populates convergence metadata with iterations == 1.
    rc_lin = main(["report", "--engine", "fem2d", str(LINEAR_STEEL_EXAMPLE)])
    assert rc_lin == 0
    lin_payload = _extract_final_json(capsys.readouterr().out)
    assert lin_payload["convergence"]["iterations"] == 1
    assert lin_payload["convergence"]["interface_nodes"] > 0


def test_report_with_iteration_cap_exits_nonzero_and_prints_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["report", "--engine", "fem2d", "--max-iterations", "1", str(NONLINEAR_EXAMPLE)])
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == ""
    err_lines = captured.err.splitlines()
    assert len(err_lines) == 1
    assert "E_NOT_CONVERGED" in err_lines[0]
    assert "cap=1" in err_lines[0]


def test_max_iterations_cli_validation() -> None:
    for argv in (
        ["report", str(LINEAR_EXAMPLE), "--max-iterations", "5"],
        ["report", str(LINEAR_EXAMPLE), "--engine", "magpylib", "--max-iterations", "5"],
        ["bench", "--max-iterations", "5"],
        ["bench", "--engine", "magpylib", "--max-iterations", "5"],
        ["report", str(NONLINEAR_EXAMPLE), "--engine", "fem2d", "--max-iterations", "0"],
        ["report", str(NONLINEAR_EXAMPLE), "--engine", "fem2d", "--max-iterations", "-2"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2
