"""``magwave solve`` results artefact with provenance (PRD FR-002)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import magpylib as magpy
import numpy as np
import pytest

import magwave
from magwave.benchmarks import load_corpus
from magwave.cases import Case
from magwave.cli import E_OUTPUT_PATH, main, parse_points
from magwave.engines import Fem2dEngine, MagpylibEngine
from magwave.engines.magpylib_engine import field_magnitude

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "solve-dipole.json"
STEEL_EXAMPLE = ROOT / "examples" / "steel-cylinder-solenoid-nonlinear.json"
CORPUS = {case.name: case for case in load_corpus()}


def _copy_example(src: Path, dest: Path) -> Path:
    dest.write_bytes(src.read_bytes())
    return dest


def test_solve_writes_provenance_artefacts_and_is_deterministic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_path = _copy_example(EXAMPLE, tmp_path / "dipole.mw.json")
    case_bytes = case_path.read_bytes()
    expected_sha = hashlib.sha256(case_bytes).hexdigest()
    case = Case.from_json(case_bytes.decode("utf-8"))
    spec = "0,0,0.5; 0.3,0,0.4"

    assert main(["solve", str(case_path), "--format", "csv", "--points", spec]) == 0
    cap1 = capsys.readouterr()

    json_path = tmp_path / "dipole.mw.results.json"
    csv_path = tmp_path / "dipole.mw.results.csv"
    assert json_path.is_file()
    assert csv_path.is_file()
    assert cap1.err.splitlines() == [str(json_path), str(csv_path)]

    json_bytes_1 = json_path.read_bytes()
    csv_bytes_1 = csv_path.read_bytes()

    # Re-run: both artefacts must be byte-identical and stdout stays unchanged.
    assert main(["solve", str(case_path), "--format", "csv", "--points", spec]) == 0
    cap2 = capsys.readouterr()
    assert cap2.out == cap1.out
    assert json_path.read_bytes() == json_bytes_1
    assert csv_path.read_bytes() == csv_bytes_1

    # No absolute paths leak into either artefact.
    json_text = json_bytes_1.decode("utf-8")
    csv_text = csv_bytes_1.decode("utf-8")
    assert str(tmp_path) not in json_text
    assert str(tmp_path) not in csv_text
    assert str(ROOT) not in json_text
    assert str(ROOT) not in csv_text

    # JSON schema, sorted keys, provenance, units, and exact samples.
    payload = json.loads(json_text)
    assert json_text == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert payload["schema"] == "magwave.results/1"
    assert payload["app_version"] == magwave.__version__
    assert payload["engine"] == {"id": "magpylib", "version": magpy.__version__}
    assert payload["case"] == {
        "file": "dipole.mw.json",
        "name": case.name,
        "sha256": expected_sha,
    }
    assert payload["units"] == "position m, B T"

    points = parse_points(spec)
    field = MagpylibEngine().sample(case, points)
    magnitude = field_magnitude(field)
    assert [s["point"] for s in payload["samples"]] == points.tolist()
    assert [s["B"] for s in payload["samples"]] == field.tolist()
    assert [s["Bmag"] for s in payload["samples"]] == magnitude.tolist()

    # CSV provenance header lines, column header, and exact sample rows.
    csv_lines = csv_text.splitlines()
    assert csv_lines[:6] == [
        f"# app_version: {magwave.__version__}",
        "# engine: magpylib",
        f"# engine_version: {magpy.__version__}",
        f"# case: {case.name}",
        f"# case_sha256: {expected_sha}",
        "x_m,y_m,z_m,Bx_T,By_T,Bz_T,Bmag_T",
    ]
    rows = np.array([[float(val) for val in line.split(",")] for line in csv_lines[6:]])
    assert np.array_equal(rows[:, :3], points)
    assert np.array_equal(rows[:, 3:6], field)
    assert np.array_equal(rows[:, 6], magnitude)


def test_solve_json_stdout_stays_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_path = _copy_example(EXAMPLE, tmp_path / "dipole.json")
    spec = "0,0,0.5"
    assert main(["solve", str(case_path), "--format", "json", "--points", spec]) == 0
    cap = capsys.readouterr()
    stdout_payload = json.loads(cap.out)
    assert stdout_payload["schema"] == "magwave.samples/1"
    assert cap.out == json.dumps(stdout_payload, indent=2, sort_keys=True) + "\n"
    assert (tmp_path / "dipole.results.json").is_file()
    assert (tmp_path / "dipole.results.csv").is_file()


def test_deleting_case_file_does_not_orphan_artefacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_path = _copy_example(EXAMPLE, tmp_path / "ephemeral.mw.json")
    expected_sha = hashlib.sha256(case_path.read_bytes()).hexdigest()
    assert main(["solve", str(case_path), "--points", "0,0,0.5"]) == 0
    _ = capsys.readouterr()

    json_path = tmp_path / "ephemeral.mw.results.json"
    csv_path = tmp_path / "ephemeral.mw.results.csv"
    case_path.unlink()
    assert not case_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["case"]["name"] == "dipole_axis_Bz"
    assert payload["case"]["file"] == "ephemeral.mw.json"
    assert payload["case"]["sha256"] == expected_sha
    assert payload["app_version"] == magwave.__version__
    assert payload["engine"]["id"] == "magpylib"
    assert payload["engine"]["version"] == magpy.__version__
    assert f"# case_sha256: {expected_sha}" in csv_path.read_text(encoding="utf-8")


def test_results_dir_overrides_output_location(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_dir = tmp_path / "cases"
    out_dir = tmp_path / "artefacts"
    case_dir.mkdir()
    out_dir.mkdir()
    case_path = _copy_example(EXAMPLE, case_dir / "case.mw.json")

    assert (
        main(["solve", str(case_path), "--points", "0,0,0.5", "--results-dir", str(out_dir)]) == 0
    )
    cap = capsys.readouterr()

    json_path = out_dir / "case.mw.results.json"
    csv_path = out_dir / "case.mw.results.csv"
    assert json_path.is_file()
    assert csv_path.is_file()
    assert not (case_dir / "case.mw.results.json").exists()
    assert not (case_dir / "case.mw.results.csv").exists()
    assert cap.err.splitlines() == [str(json_path), str(csv_path)]


def test_one_byte_case_change_updates_case_sha256(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_path = _copy_example(EXAMPLE, tmp_path / "mutable.json")
    assert main(["solve", str(case_path), "--points", "0,0,0.5"]) == 0
    _ = capsys.readouterr()
    json_path = tmp_path / "mutable.results.json"
    sha_before = json.loads(json_path.read_text(encoding="utf-8"))["case"]["sha256"]

    original = case_path.read_bytes()
    modified = original + b" "
    assert len(modified) == len(original) + 1
    case_path.write_bytes(modified)

    assert main(["solve", str(case_path), "--points", "0,0,0.5"]) == 0
    _ = capsys.readouterr()
    sha_after = json.loads(json_path.read_text(encoding="utf-8"))["case"]["sha256"]
    assert sha_after != sha_before
    assert sha_after == hashlib.sha256(modified).hexdigest()


def test_solve_fem2d_writes_provenance_artefacts_and_honours_results_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_dir = tmp_path / "fem_case"
    out_dir = tmp_path / "fem_out"
    case_dir.mkdir()
    out_dir.mkdir()
    case_path = _copy_example(STEEL_EXAMPLE, case_dir / "steel.mw.json")
    expected_sha = hashlib.sha256(case_path.read_bytes()).hexdigest()
    case = Case.from_json(case_path.read_text(encoding="utf-8"))
    caps = Fem2dEngine(element_size=0.001, material_model="nonlinear").capabilities()

    assert main(["solve", str(case_path), "--engine", "fem2d"]) == 0
    cap1 = capsys.readouterr()
    json_path = case_dir / "steel.mw.results.json"
    csv_path = case_dir / "steel.mw.results.csv"
    assert json_path.is_file()
    assert csv_path.is_file()
    assert cap1.err.splitlines() == [str(json_path), str(csv_path)]
    assert cap1.out.startswith("probe #0 (0.0, 0.0, 0.0) m  B  z = ")

    json_bytes_1 = json_path.read_bytes()
    csv_bytes_1 = csv_path.read_bytes()

    payload = json.loads(json_bytes_1.decode("utf-8"))
    assert payload["schema"] == "magwave.results/1"
    assert payload["app_version"] == magwave.__version__
    assert payload["engine"] == {"id": caps.engine_id, "version": caps.engine_version}
    assert payload["case"] == {
        "file": "steel.mw.json",
        "name": case.name,
        "sha256": expected_sha,
    }
    assert payload["units"] == "position m, B T"
    assert len(payload["samples"]) == len(case.probes)
    expected_bz = float(cap1.out.split("= ", 1)[1].split(" T", 1)[0])
    assert payload["samples"][0]["B"][2] == expected_bz

    # Re-run into --results-dir must be byte-identical to the beside-case files.
    assert main(["solve", str(case_path), "--engine", "fem2d", "--results-dir", str(out_dir)]) == 0
    cap2 = capsys.readouterr()
    assert cap2.out == cap1.out
    assert (out_dir / "steel.mw.results.json").read_bytes() == json_bytes_1
    assert (out_dir / "steel.mw.results.csv").read_bytes() == csv_bytes_1


@pytest.mark.parametrize(
    "argv",
    [
        ["bench", "--results-dir", "out_dir"],
        ["report", str(EXAMPLE), "--results-dir", "out_dir"],
        ["converge", str(STEEL_EXAMPLE), "--results-dir", "out_dir"],
        ["gui", "--results-dir", "out_dir"],
        ["import", "unused.step", "--results-dir", "out_dir"],
    ],
)
def test_results_dir_refused_on_non_solve_commands(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


def test_invalid_results_dir_refuses_before_solving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_path = _copy_example(EXAMPLE, tmp_path / "dipole.json")
    missing_dir = tmp_path / "missing-dir"

    def _forbid(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("solver ran despite unwritable --results-dir")

    monkeypatch.setattr(MagpylibEngine, "sample", _forbid)

    rc = main(["solve", str(case_path), "--points", "0,0,0.5", "--results-dir", str(missing_dir)])
    cap = capsys.readouterr()
    assert rc == 1
    assert cap.out == ""
    err_lines = cap.err.splitlines()
    assert len(err_lines) == 1
    assert f"magwave solve: {E_OUTPUT_PATH}:" in err_lines[0]
    assert "missing directory" in err_lines[0]


def test_results_module_imports_without_magpylib() -> None:
    code = (
        "import sys, magwave.results, magwave.cli; "
        "assert 'magpylib' not in sys.modules, 'magpylib imported'"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert completed.returncode == 0, completed.stderr
