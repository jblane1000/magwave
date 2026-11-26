"""Tests for ``magwave validate``, ``magwave run``, and ``magwave export`` (FR-060)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import magwave
from magwave.benchmarks import load_corpus
from magwave.cad.manifest import Body, Manifest
from magwave.cli import E_OUTPUT_PATH, main
from magwave.engines import Fem2dEngine, MagpylibEngine

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
DIPOLE_EXAMPLE = EXAMPLES_DIR / "solve-dipole.json"
STEEL_EXAMPLE = EXAMPLES_DIR / "steel-cylinder-solenoid-nonlinear.json"
STEEL_SPHERE_EXAMPLE = EXAMPLES_DIR / "steel-sphere-uniform-field.json"
CORPUS = {case.name: case for case in load_corpus()}


def _copy_case(src: Path, dest: Path) -> Path:
    dest.write_bytes(src.read_bytes())
    return dest


def _headless_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM", "PYTHONPATH")
    }
    return env


# --------------------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "example_path",
    sorted(EXAMPLES_DIR.glob("*.json")),
    ids=lambda path: path.name,
)
def test_validate_shipped_examples_exit_zero_and_emit_deterministic_json(
    example_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    assert main(["validate", str(example_path), "--json", str(report_path)]) == 0
    cap = capsys.readouterr()
    assert cap.err == ""
    assert (
        cap.out.strip()
        == "valid: true (schema: magwave.case/2; structural only — no engine checked)"
    )

    text_first = report_path.read_text(encoding="utf-8")
    payload = json.loads(text_first)
    assert payload == {"errors": [], "schema": "magwave.case/2", "valid": True}
    assert text_first == json.dumps(payload, indent=2, sort_keys=True) + "\n"

    report_path_2 = tmp_path / "report2.json"
    assert main(["validate", str(example_path), "--json", str(report_path_2)]) == 0
    _ = capsys.readouterr()
    assert report_path_2.read_text(encoding="utf-8") == text_first


def test_validate_reports_all_defects_not_just_first(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doc = json.loads(DIPOLE_EXAMPLE.read_text(encoding="utf-8"))
    doc["units"]["length"] = "in"
    doc["frame"]["axis"] = "x"
    doc["tolerance"] = 0.5
    doc["unexpected_top_key"] = 42
    doc["probes"][0]["component"] = "radial"
    doc["expected"][0]["unit"] = "mT"
    doc["mesh"] = {"element_size": -0.01}

    case_path = tmp_path / "multi_defect.json"
    case_path.write_text(json.dumps(doc), encoding="utf-8")
    report_path = tmp_path / "validate.json"

    rc = main(["validate", str(case_path), "--json", str(report_path)])
    cap = capsys.readouterr()

    assert rc == 1
    assert cap.out == ""
    err_lines = cap.err.splitlines()
    for expected_path in (
        "case.unexpected_top_key",
        "case.units.length",
        "case.frame.axis",
        "case.probes[0].component",
        "case.expected[0].unit",
        "case.tolerance",
        "case.mesh.element_size",
    ):
        assert any(line.startswith(f"{expected_path}: ") for line in err_lines), (
            f"missing {expected_path!r} in {err_lines}"
        )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["valid"] is False
    assert payload["schema"] == "magwave.case/2"
    reported_paths = {item["path"] for item in payload["errors"]}
    assert {
        "case.unexpected_top_key",
        "case.units.length",
        "case.frame.axis",
        "case.probes[0].component",
        "case.expected[0].unit",
        "case.tolerance",
        "case.mesh.element_size",
    }.issubset(reported_paths)


def test_validate_rejects_case_v1_naming_schema_and_checks_other_defects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doc = json.loads(DIPOLE_EXAMPLE.read_text(encoding="utf-8"))
    doc["schema"] = "magwave.case/1"
    case_path = tmp_path / "v1.json"
    case_path.write_text(json.dumps(doc), encoding="utf-8")
    json_out = tmp_path / "v1_report.json"

    rc = main(["validate", str(case_path), "--json", str(json_out)])
    cap = capsys.readouterr()

    assert rc == 1
    assert cap.out == ""
    assert "case.schema:" in cap.err
    assert "magwave.case/2" in cap.err
    assert "magwave.case/1" in cap.err

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload == {
        "errors": [
            {
                "message": "expected 'magwave.case/2', got 'magwave.case/1'",
                "path": "case.schema",
            }
        ],
        "schema": "magwave.case/1",
        "valid": False,
    }


def test_validate_checks_component_dims_and_manifest_roles_without_solving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbid(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("validate must never invoke an engine")

    monkeypatch.setattr(MagpylibEngine, "solve", _forbid)
    monkeypatch.setattr(MagpylibEngine, "sample", _forbid)
    monkeypatch.setattr(Fem2dEngine, "solve", _forbid)

    doc = json.loads((EXAMPLES_DIR / "magnet-n42-air.json").read_text(encoding="utf-8"))
    doc["components"][0]["dims"]["radius"] = -0.01
    manifest = Manifest(
        path="assy.step",
        format="step",
        units="mm",
        bodies=[
            Body(
                index=0,
                name="magnet_body",
                role=None,
                volume_mm3=100.0,
                solids=1,
                bbox=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                source="step",
            )
        ],
    )
    doc["manifest"] = manifest.to_dict()
    case_path = tmp_path / "bad_dims_and_manifest.json"
    case_path.write_text(json.dumps(doc), encoding="utf-8")

    rc = main(["validate", str(case_path)])
    cap = capsys.readouterr()
    assert rc == 1
    assert "case.components[0].dims.radius:" in cap.err
    assert "case.manifest.bodies[0].role:" in cap.err

    # Assigning the role and fixing the dimension passes validation cleanly.
    doc["components"][0]["dims"]["radius"] = 0.01
    doc["manifest"]["bodies"][0]["role"] = "permanent_magnet"
    good_path = tmp_path / "good_manifest.json"
    good_path.write_text(json.dumps(doc), encoding="utf-8")
    assert main(["validate", str(good_path)]) == 0
    cap_good = capsys.readouterr()
    assert "valid: true" in cap_good.out


def test_validate_refuses_bad_json_missing_file_and_unwritable_json_out(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    assert main(["validate", str(bad_json)]) == 1
    assert "case: not valid JSON" in capsys.readouterr().err

    missing = tmp_path / "missing.json"
    assert main(["validate", str(missing)]) == 1
    assert "case:" in capsys.readouterr().err

    unwritable = tmp_path / "missing_dir" / "out.json"
    assert main(["validate", str(DIPOLE_EXAMPLE), "--json", str(unwritable)]) == 1
    err = capsys.readouterr().err
    assert E_OUTPUT_PATH in err
    assert str(unwritable) in err
    assert "Traceback" not in err


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------


def test_run_dipole_writes_provenance_artefacts_and_is_deterministic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_path = _copy_case(DIPOLE_EXAMPLE, tmp_path / "dipole.mw.json")
    expected_sha = hashlib.sha256(case_path.read_bytes()).hexdigest()

    assert main(["run", str(case_path)]) == 0
    cap1 = capsys.readouterr()

    json_path = tmp_path / "dipole.mw.results.json"
    csv_path = tmp_path / "dipole.mw.results.csv"
    assert json_path.is_file()
    assert csv_path.is_file()
    assert cap1.err.splitlines() == [str(json_path), str(csv_path)]
    assert "dipole_axis_Bz" in cap1.out
    assert "probe #0" in cap1.out

    json_bytes_1 = json_path.read_bytes()
    csv_bytes_1 = csv_path.read_bytes()
    payload = json.loads(json_bytes_1.decode("utf-8"))
    assert payload["schema"] == "magwave.results/1"
    assert payload["app_version"] == magwave.__version__
    assert payload["case"]["sha256"] == expected_sha
    assert payload["samples"][0]["B"][2] == pytest.approx(1.6e-6, rel=1e-9)

    assert main(["run", str(case_path)]) == 0
    cap2 = capsys.readouterr()
    assert cap2.out == cap1.out
    assert cap2.err == cap1.err
    assert json_path.read_bytes() == json_bytes_1
    assert csv_path.read_bytes() == csv_bytes_1


def test_run_results_dir_and_fem2d_engine(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    steel_copy = _copy_case(STEEL_EXAMPLE, tmp_path / "steel.json")

    assert (
        main(
            [
                "run",
                str(steel_copy),
                "--engine",
                "fem2d",
                "--element-size",
                "0.002",
                "--results-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    cap = capsys.readouterr()
    json_path = out_dir / "steel.results.json"
    csv_path = out_dir / "steel.results.csv"
    assert json_path.is_file()
    assert csv_path.is_file()
    assert not (tmp_path / "steel.results.json").exists()
    assert cap.err.splitlines() == [str(json_path), str(csv_path)]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["engine"]["id"] == "fem2d"
    assert payload["case"]["sha256"] == hashlib.sha256(steel_copy.read_bytes()).hexdigest()


def test_run_exits_two_on_engine_refusal_and_one_on_broken_case(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    steel_copy = _copy_case(STEEL_EXAMPLE, tmp_path / "steel.json")
    assert main(["run", str(steel_copy), "--engine", "magpylib"]) == 2
    cap_refuse = capsys.readouterr()
    assert cap_refuse.out == ""
    assert "E_UNSUPPORTED_PERMEABLE" in cap_refuse.err
    assert not (tmp_path / "steel.results.json").exists()
    assert not (tmp_path / "steel.results.csv").exists()

    broken = json.loads(DIPOLE_EXAMPLE.read_text(encoding="utf-8"))
    broken["surprise"] = 1
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    assert main(["run", str(broken_path)]) == 1
    cap_broken = capsys.readouterr()
    assert cap_broken.out == ""
    assert "unknown key" in cap_broken.err
    assert not (tmp_path / "broken.results.json").exists()

    assert main(["run", str(tmp_path / "nonexistent.json")]) == 1
    assert capsys.readouterr().out == ""


def test_run_refuses_solve_sampling_flags_and_unwritable_results_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_path = _copy_case(DIPOLE_EXAMPLE, tmp_path / "dipole.json")

    for bad_flag in (
        ["--points", "0,0,0.5"],
        ["--grid", "0,1,1:0,1,1:0,1,1"],
        ["--format", "json"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(["run", str(case_path), *bad_flag])
        assert excinfo.value.code == 2
        err_msg = capsys.readouterr().err
        assert "do not apply" in err_msg

    missing_dir = tmp_path / "missing_results_dir"
    assert main(["run", str(case_path), "--results-dir", str(missing_dir)]) == 1
    err = capsys.readouterr().err
    assert E_OUTPUT_PATH in err


def test_cli_help_lists_new_commands_and_run_exit_codes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "not implemented yet" not in help_text
    for command_name in ("validate", "run", "export"):
        assert command_name in help_text
    assert "0 success, 1 case fails to load or validate, 2 engine refuses the case" in help_text


def test_run_dry_run_refuses_bad_probe_without_solving_or_writing_artefacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbid(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dry-run must not build a mesh or solve")

    monkeypatch.setattr(Fem2dEngine, "solve", _forbid)
    monkeypatch.setattr(Fem2dEngine, "_build_mesh_and_load", _forbid)

    doc = json.loads(STEEL_SPHERE_EXAMPLE.read_text(encoding="utf-8"))
    doc["probes"][0]["point"] = [50.0, 0.0, 0.0]
    bad_case = tmp_path / "bad_sphere.json"
    bad_case.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    assert main(["run", str(bad_case), "--engine", "fem2d", "--dry-run"]) == 2
    cap = capsys.readouterr()
    assert cap.out == ""
    assert "E_PROBE_OUTSIDE_DOMAIN" in cap.err
    assert not (tmp_path / "bad_sphere.results.json").exists()
    assert not (tmp_path / "bad_sphere.results.csv").exists()


def test_run_dry_run_accepts_good_case_without_solving_or_writing_artefacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbid(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dry-run must not build a mesh or solve")

    monkeypatch.setattr(Fem2dEngine, "solve", _forbid)
    monkeypatch.setattr(Fem2dEngine, "_build_mesh_and_load", _forbid)

    good_case = _copy_case(STEEL_SPHERE_EXAMPLE, tmp_path / "good_sphere.json")
    missing_dir = tmp_path / "nonexistent_results_dir"

    assert (
        main(
            [
                "run",
                str(good_case),
                "--engine",
                "fem2d",
                "--dry-run",
                "--results-dir",
                str(missing_dir),
            ]
        )
        == 0
    )
    cap = capsys.readouterr()
    assert cap.err == ""
    assert "fem2d" in cap.out
    assert "steel_sphere_uniform_field" in cap.out
    assert "dry run, nothing was solved" in cap.out
    assert not (tmp_path / "good_sphere.results.json").exists()
    assert not (tmp_path / "good_sphere.results.csv").exists()


def test_run_without_dry_run_still_writes_both_artefacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    good_case = _copy_case(STEEL_SPHERE_EXAMPLE, tmp_path / "good_sphere.json")

    assert main(["run", str(good_case), "--engine", "fem2d"]) == 0
    cap = capsys.readouterr()
    json_path = tmp_path / "good_sphere.results.json"
    csv_path = tmp_path / "good_sphere.results.csv"
    assert json_path.is_file()
    assert csv_path.is_file()
    assert cap.err.splitlines() == [str(json_path), str(csv_path)]
    assert "steel_sphere_uniform_field" in cap.out


def test_validate_reports_structural_only_and_points_engine_flag_to_run_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    good_case = _copy_case(STEEL_SPHERE_EXAMPLE, tmp_path / "good_sphere.json")

    assert main(["validate", str(good_case)]) == 0
    cap_ok = capsys.readouterr()
    assert cap_ok.err == ""
    assert (
        cap_ok.out.strip()
        == "valid: true (schema: magwave.case/2; structural only — no engine checked)"
    )

    with pytest.raises(SystemExit) as excinfo:
        main(["validate", str(good_case), "--engine", "fem2d"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "schema only" in err
    assert "magwave run CASE --engine ID --dry-run" in err

    for subcommand, extra in (
        ("validate", [str(good_case)]),
        ("solve", [str(good_case), "--points", "0,0,0"]),
        ("bench", []),
    ):
        with pytest.raises(SystemExit) as bad_dry:
            main([subcommand, *extra, "--dry-run"])
        assert bad_dry.value.code == 2
        assert "--dry-run applies to run only" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------------------


def test_export_csv_to_file_matches_solve_csv_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_path = _copy_case(DIPOLE_EXAMPLE, tmp_path / "dipole.json")
    spec = "0,0,0.5; 0.3,0,0.4; -0.1,0.2,0.7"
    out_csv = tmp_path / "exported.csv"

    assert main(["solve", str(case_path), "--format", "csv", "--points", spec]) == 0
    solve_cap = capsys.readouterr()

    # Clean up solve artefacts to verify export does not create .results.* files.
    (tmp_path / "dipole.results.json").unlink()
    (tmp_path / "dipole.results.csv").unlink()

    assert (
        main(["export", str(case_path), "--points", spec, "--format", "csv", "--out", str(out_csv)])
        == 0
    )
    export_cap = capsys.readouterr()
    assert export_cap.out == ""
    assert export_cap.err == ""
    assert not (tmp_path / "dipole.results.json").exists()
    assert not (tmp_path / "dipole.results.csv").exists()

    exported_text = out_csv.read_text(encoding="utf-8")
    assert exported_text == solve_cap.out

    solve_rows = np.array(
        [[float(val) for val in line.split(",")] for line in solve_cap.out.splitlines()[5:]]
    )
    export_rows = np.array(
        [[float(val) for val in line.split(",")] for line in exported_text.splitlines()[5:]]
    )
    assert np.array_equal(export_rows, solve_rows)


def test_export_json_stdout_is_byte_identical_to_solve_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_path = _copy_case(DIPOLE_EXAMPLE, tmp_path / "dipole.json")
    spec = "0,0,0.5; 0.1,0.2,0.3"
    results_dir = tmp_path / "solve_artefacts"
    results_dir.mkdir()

    assert (
        main(
            [
                "solve",
                str(case_path),
                "--format",
                "json",
                "--points",
                spec,
                "--results-dir",
                str(results_dir),
            ]
        )
        == 0
    )
    solve_out = capsys.readouterr().out

    assert main(["export", str(case_path), "--points", spec, "--format", "json", "--out", "-"]) == 0
    export_dash = capsys.readouterr()
    assert export_dash.err == ""
    assert export_dash.out == solve_out

    # Default omitted --out also writes to stdout identically.
    assert main(["export", str(case_path), "--points", spec, "--format", "json"]) == 0
    export_default = capsys.readouterr()
    assert export_default.err == ""
    assert export_default.out == solve_out


def test_export_refuses_unwritable_out_fem2d_and_unsolvable_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbid(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("export must not sample when --out is unwritable")

    monkeypatch.setattr(MagpylibEngine, "sample", _forbid)

    bad_out = tmp_path / "no_such_dir" / "out.csv"
    assert main(["export", str(DIPOLE_EXAMPLE), "--points", "0,0,0.5", "--out", str(bad_out)]) == 1
    cap_out = capsys.readouterr()
    assert cap_out.out == ""
    assert E_OUTPUT_PATH in cap_out.err
    assert str(bad_out) in cap_out.err
    assert "Traceback" not in cap_out.err

    assert main(["export", str(DIPOLE_EXAMPLE), "--points", "0,0,0.5", "--engine", "fem2d"]) == 2
    cap_fem = capsys.readouterr()
    assert "Fem2dEngine has no point sampler" in cap_fem.err

    halbach_path = tmp_path / "halbach.json"
    halbach_path.write_text(CORPUS["halbach_bore_B"].to_json(), encoding="utf-8")
    assert main(["export", str(halbach_path), "--points", "0,0,0"]) == 2
    assert "E_UNSUPPORTED_GEOMETRY" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# headless CI and no-Qt import guard
# --------------------------------------------------------------------------------------


def test_validate_run_export_work_headless_with_display_removed(tmp_path: Path) -> None:
    env = _headless_env()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    out_csv = tmp_path / "dipole.csv"

    p_validate = subprocess.run(
        [sys.executable, "-m", "magwave", "validate", str(DIPOLE_EXAMPLE)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_validate.returncode == 0, p_validate.stderr
    assert "valid: true" in p_validate.stdout

    p_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "magwave",
            "run",
            str(DIPOLE_EXAMPLE),
            "--results-dir",
            str(results_dir),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_run.returncode == 0, p_run.stderr
    assert (results_dir / "solve-dipole.results.json").is_file()
    assert (results_dir / "solve-dipole.results.csv").is_file()

    p_export = subprocess.run(
        [
            sys.executable,
            "-m",
            "magwave",
            "export",
            str(DIPOLE_EXAMPLE),
            "--points",
            "0,0,0.5",
            "--out",
            str(out_csv),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_export.returncode == 0, p_export.stderr
    assert out_csv.is_file()


def test_validate_run_export_never_import_gui_or_plotting_packages_headless(
    tmp_path: Path,
) -> None:
    env = _headless_env()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    code = (
        "import sys; "
        "from magwave.cli import main; "
        f"assert main(['validate', {str(DIPOLE_EXAMPLE)!r}]) == 0; "
        "assert 'magpylib' not in sys.modules, 'validate imported magpylib'; "
        f"assert main(['run', {str(DIPOLE_EXAMPLE)!r}, '--results-dir', {str(results_dir)!r}]) == 0; "
        f"assert main(['export', {str(DIPOLE_EXAMPLE)!r}, '--points', '0,0,0.5', '--out', '-']) == 0; "
        "banned = {'PySide6', 'pyvista', 'matplotlib'}; "
        "leaked = sorted(m for m in banned if any(k == m or k.startswith(m + '.') for k in sys.modules)); "
        "assert not leaked, f'GUI/plot modules imported: {leaked}'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
