"""``magwave solve``: field sampling and CSV/JSON export (FR-050).

FR-050 AC: exported values match in-app values exactly. The header names units, engine id,
version and rung — the acceptance hook that makes an exported file self-describing.
"""

from __future__ import annotations

import json
from pathlib import Path

import magpylib as magpy
import numpy as np
import pytest

from magwave.benchmarks import load_corpus
from magwave.cases import Case
from magwave.cli import main, parse_grid, parse_points
from magwave.engines import MagpylibEngine
from magwave.engines.magpylib_engine import field_magnitude

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "solve-dipole.json"
GRID = "0,0.2,0.1:-0.2,0.2,0.2:0.3,0.5,0.1"
CORPUS = {case.name: case for case in load_corpus()}


def _write_case(tmp_path: Path, name: str) -> Path:
    target = tmp_path / f"{name}.json"
    target.write_text(CORPUS[name].to_json(), encoding="utf-8")
    return target


def test_example_case_is_a_valid_case_document() -> None:
    case = Case.from_json(EXAMPLE.read_text(encoding="utf-8"))
    assert case.to_dict() == CORPUS["dipole_axis_Bz"].to_dict()


def test_grid_is_inclusive_of_max_and_ordered_x_slowest() -> None:
    points = parse_grid(GRID)
    assert points.shape == (3 * 3 * 3, 3)
    np.testing.assert_allclose(np.unique(points[:, 0]), [0.0, 0.1, 0.2])
    np.testing.assert_allclose(np.unique(points[:, 1]), [-0.2, 0.0, 0.2])
    np.testing.assert_allclose(np.unique(points[:, 2]), [0.3, 0.4, 0.5])
    assert points[0].tolist() == [0.0, -0.2, 0.3]
    assert points[1, 2] > points[0, 2]  # z fastest
    assert parse_grid("0,0,1:0,0,1:1,1,1").shape == (1, 3)


def test_csv_export_matches_engine_sample_exactly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_path = _write_case(tmp_path, "cuboid_axis_Bz")
    assert main(["solve", str(case_path), "--format", "csv", "--grid", GRID]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:5] == [
        "# magwave field samples",
        f"# engine: magpylib {magpy.__version__}",
        "# rung: second-code",
        "# units: position m, B T",
        "x_m,y_m,z_m,Bx_T,By_T,Bz_T,Bmag_T",
    ]
    rows = lines[5:]
    points = parse_grid(GRID)
    assert len(rows) == len(points) == 27
    exported = np.array([[float(value) for value in row.split(",")] for row in rows])
    engine = MagpylibEngine()
    field = engine.sample(CORPUS["cuboid_axis_Bz"], points)
    assert np.array_equal(exported[:, :3], points)
    assert np.array_equal(exported[:, 3:6], field)
    assert np.array_equal(exported[:, 6], field_magnitude(field))


def test_json_export_matches_engine_sample_exactly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = "0,0,0.5; 0.3,0,0.4; -0.1,0.2,0.7"
    assert main(["solve", str(EXAMPLE), "--format", "json", "--points", spec]) == 0
    text = capsys.readouterr().out
    payload = json.loads(text)
    assert text == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert payload["schema"] == "magwave.samples/1"
    assert payload["engine"] == {
        "id": "magpylib",
        "version": magpy.__version__,
        "rung": "second-code",
    }
    assert payload["units"] == {"position": "m", "B": "T"}
    points = parse_points(spec)
    field = MagpylibEngine().sample(CORPUS["dipole_axis_Bz"], points)
    assert [s["point"] for s in payload["samples"]] == points.tolist()
    assert [s["B"] for s in payload["samples"]] == field.tolist()
    assert [s["Bmag"] for s in payload["samples"]] == field_magnitude(field).tolist()
    # the probe point of the example reproduces the benchmark value
    assert payload["samples"][0]["B"][2] == pytest.approx(1.6e-6, rel=1e-9)


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--grid", GRID, "--points", "0,0,1"],
        ["--grid", "0,1,0.5:0,1"],
        ["--grid", "0,1,0:0,1,1:0,1,1"],
        ["--points", "0,0"],
        ["--points", "a,b,c"],
    ],
)
def test_point_selection_errors_are_argparse_errors(extra: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["solve", str(EXAMPLE), *extra])
    assert excinfo.value.code == 2


def test_case_the_engine_cannot_model_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_path = _write_case(tmp_path, "halbach_bore_B")
    assert main(["solve", str(case_path), "--points", "0,0,0"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "E_UNSUPPORTED_GEOMETRY" in captured.err


def test_malformed_case_file_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    broken["surprise"] = 1
    target = tmp_path / "broken.json"
    target.write_text(json.dumps(broken), encoding="utf-8")
    assert main(["solve", str(target), "--points", "0,0,1"]) == 2
    assert "unknown key" in capsys.readouterr().err
    assert main(["solve", str(tmp_path / "missing.json"), "--points", "0,0,1"]) == 2
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert main(["solve", str(tmp_path / "bad.json"), "--points", "0,0,1"]) == 2
