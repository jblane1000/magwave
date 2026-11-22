"""M5.5 mesh density control tests: schema, CLI precedence, ``solve --engine fem2d``, and ``converge``."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from magwave.cases import (
    Case,
    CaseError,
    Coil,
    ExpectedValue,
    MeshSpec,
    Probe,
)
from magwave.cli import main
from magwave.converge import ConvergenceRung, analyse_ladder, format_convergence_table
from magwave.engines import Fem2dEngine, Result

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
STEEL_EXAMPLE = EXAMPLES_DIR / "steel-cylinder-solenoid-nonlinear.json"
N42_AIR_EXAMPLE = EXAMPLES_DIR / "magnet-n42-air.json"
ELMER_STEEL_BZ = 0.143836


def _extract_final_json(markdown: str) -> dict:
    match = re.search(r"```json\n(.*?)\n```\s*\Z", markdown, flags=re.DOTALL)
    assert match is not None, "report must end with a fenced ```json block"
    return json.loads(match.group(1))


def _linear_solenoid_case(*, mesh: MeshSpec | None = None) -> Case:
    return Case(
        name="m55_linear_solenoid_converge",
        description="Cheap linear solenoid for mesh convergence tests",
        sources=("unit test",),
        components=(
            Coil(
                kind="solenoid",
                dims={"inner_radius": 0.01, "outer_radius": 0.02, "length": 0.04},
                current=1.0,
                turns=100,
                center=(0.0, 0.0, 0.0),
                name="coil",
            ),
        ),
        probes=(Probe(point=(0.0, 0.0, 0.0), quantity="B", component="z"),),
        expected=(
            ExpectedValue(
                check="solenoid_axis_Bz",
                quantity="Bz",
                value=0.0028,
                unit="T",
                tolerance=0.02,
                source="unit test",
            ),
        ),
        mesh=mesh,
    )


def _parse_table_rows(output: str) -> list[list[str]]:
    table_lines = [line for line in output.splitlines() if line.startswith("|")]
    assert len(table_lines) >= 4
    rows: list[list[str]] = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


@pytest.fixture(scope="module")
def steel_direct_1mm() -> Result:
    case = Case.from_json(STEEL_EXAMPLE.read_text(encoding="utf-8"))
    return Fem2dEngine(element_size=1e-3, material_model="nonlinear").solve(case)


def test_mesh_spec_parse_roundtrip_and_absent_omission() -> None:
    spec = MeshSpec(element_size=0.001)
    assert spec.to_dict() == {"element_size": 0.001}
    assert MeshSpec.from_dict({"element_size": 0.001}, "case.mesh") == spec

    case_with_mesh = _linear_solenoid_case(mesh=spec)
    as_dict = case_with_mesh.to_dict()
    assert as_dict["mesh"] == {"element_size": 0.001}
    assert Case.from_dict(as_dict) == case_with_mesh
    assert Case.from_json(case_with_mesh.to_json()) == case_with_mesh

    case_without_mesh = _linear_solenoid_case(mesh=None)
    assert "mesh" not in case_without_mesh.to_dict()
    assert Case.from_dict(case_without_mesh.to_dict()) == case_without_mesh

    # Positional construction without mesh remains unchanged.
    positional = Case(
        "pos_case",
        "positional construction",
        ("unit test",),
        case_without_mesh.components,
        case_without_mesh.probes,
        case_without_mesh.expected,
    )
    assert positional.mesh is None
    assert "mesh" not in positional.to_dict()


@pytest.mark.parametrize(
    ("bad_value", "expected_fragment"),
    [
        (-1, "-1"),
        (0, "0"),
        (float("nan"), "nan"),
        (float("inf"), "inf"),
        ("0.001", "'0.001'"),
        (True, "True"),
    ],
)
def test_mesh_spec_refuses_invalid_values_and_unknown_keys(
    bad_value: object, expected_fragment: str
) -> None:
    base_doc = _linear_solenoid_case().to_dict()
    bad_doc = dict(base_doc)
    bad_doc["mesh"] = {"element_size": bad_value}
    with pytest.raises(CaseError) as excinfo:
        Case.from_dict(bad_doc)
    msg = str(excinfo.value)
    assert "case.mesh.element_size" in msg
    assert expected_fragment in msg

    unknown_doc = dict(base_doc)
    unknown_doc["mesh"] = {"element_size": 0.001, "refinement": 2}
    with pytest.raises(CaseError, match="refinement") as unk_exc:
        Case.from_dict(unknown_doc)
    assert "case.mesh" in str(unk_exc.value)

    missing_doc = dict(base_doc)
    missing_doc["mesh"] = {}
    with pytest.raises(CaseError, match="element_size"):
        Case.from_dict(missing_doc)

    non_obj_doc = dict(base_doc)
    non_obj_doc["mesh"] = 0.001
    with pytest.raises(CaseError, match="case.mesh"):
        Case.from_dict(non_obj_doc)

    with pytest.raises(CaseError, match="mesh.element_size"):
        MeshSpec(element_size=-1.0)


def test_cli_element_size_flag_beats_case_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], steel_direct_1mm: Result
) -> None:
    doc = json.loads(STEEL_EXAMPLE.read_text(encoding="utf-8"))
    doc["mesh"] = {"element_size": 0.004}
    coarse_case_path = tmp_path / "steel_coarse.json"
    coarse_case_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")

    assert main(["report", "--engine", "fem2d", str(coarse_case_path)]) == 0
    coarse_payload = _extract_final_json(capsys.readouterr().out)
    coarse_bz = float(coarse_payload["probes"][0]["computed"])

    assert (
        main(
            [
                "report",
                "--engine",
                "fem2d",
                "--element-size",
                "0.001",
                str(coarse_case_path),
            ]
        )
        == 0
    )
    flag_payload = _extract_final_json(capsys.readouterr().out)
    flag_bz = float(flag_payload["probes"][0]["computed"])

    assert coarse_bz != flag_bz
    assert abs(coarse_bz - ELMER_STEEL_BZ) > abs(flag_bz - ELMER_STEEL_BZ)
    assert flag_bz == steel_direct_1mm.values()[0]
    assert steel_direct_1mm.convergence is not None
    assert flag_payload["convergence"]["nodes"] == steel_direct_1mm.convergence.nodes


def test_case_mesh_block_honoured_without_flag_in_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], steel_direct_1mm: Result
) -> None:
    doc = json.loads(STEEL_EXAMPLE.read_text(encoding="utf-8"))
    doc["mesh"] = {"element_size": 0.001}
    case_path = tmp_path / "steel_1mm.json"
    case_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")

    assert main(["report", "--engine", "fem2d", str(case_path)]) == 0
    payload = _extract_final_json(capsys.readouterr().out)
    assert payload["probes"][0]["computed"] == steel_direct_1mm.values()[0]
    assert steel_direct_1mm.convergence is not None
    assert payload["convergence"]["nodes"] == steel_direct_1mm.convergence.nodes


@pytest.mark.parametrize(
    "bad_size",
    ["-1", "0", "nan", "inf", "-inf", "true", "abc", "-1e-3"],
)
def test_cli_element_size_bad_values_exit_one_before_coupling(
    bad_size: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # Even when paired with --engine magpylib, value validation fires first with exit 1.
    for argv in (
        ["solve", str(STEEL_EXAMPLE), "--element-size", bad_size],
        ["report", str(STEEL_EXAMPLE), "--engine", "magpylib", "--element-size", bad_size],
    ):
        rc = main(argv)
        captured = capsys.readouterr()
        assert rc == 1
        assert captured.out == ""
        err_lines = captured.err.splitlines()
        assert len(err_lines) == 1
        assert "--element-size" in err_lines[0]
        assert bad_size in err_lines[0]
        assert "Traceback" not in captured.err


def test_cli_element_size_validation_order_and_misuse_refusals(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["solve", str(STEEL_EXAMPLE), "--element-size", "-1"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert captured.err.strip() == "magwave: --element-size: must be a finite number > 0, got '-1'"
    assert "Traceback" not in captured.err

    for argv in (
        [
            "solve",
            str(STEEL_EXAMPLE),
            "--engine",
            "magpylib",
            "--points",
            "0,0,0",
            "--element-size",
            "0.001",
        ],
        ["report", str(STEEL_EXAMPLE), "--engine", "magpylib", "--element-size", "0.001"],
        ["converge", str(STEEL_EXAMPLE), "--engine", "magpylib", "--element-size", "0.001"],
        ["bench", "--element-size", "0.001"],
        ["bench", "--engine", "fem2d", "--element-size", "0.001"],
        ["gui", "--element-size", "0.001"],
        ["import", "x.step", "--element-size", "0.001"],
        ["solve", str(STEEL_EXAMPLE), "--points", "0,0,0", "--element-size", "0.001"],
        ["report", str(STEEL_EXAMPLE), "--element-size", "0.001"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "solve" in err and "report" in err and "converge" in err and "fem2d" in err


def test_converge_linear_case_three_rungs_and_case_mesh_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _linear_solenoid_case(mesh=MeshSpec(element_size=0.002))
    case_path = tmp_path / "lin_solenoid.json"
    case_path.write_text(case.to_json(), encoding="utf-8")

    rc = main(["converge", str(case_path), "--engine", "fem2d", "--sizes", "0.008,0.004,0.002"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert captured.err == ""

    rows = _parse_table_rows(captured.out)
    assert len(rows) == 3
    assert [float(row[0]) for row in rows] == [0.008, 0.004, 0.002]
    nodes = [int(row[1]) for row in rows]
    assert nodes[0] < nodes[1] < nodes[2]
    deltas = [abs(float(row[4])) for row in rows[1:]]
    assert deltas[1] < deltas[0]
    for row in rows:
        # Full IEEE-754 repr precision preserved (more than 6 significant digits).
        assert repr(float(row[3])) == row[3]
        assert len(row[3]) > 10
    assert "observed order p =" in captured.out
    assert "estimated discretisation error at h = 0.002 m:" in captured.out
    assert "criterion:" in captured.out
    assert "converged:" in captured.out

    # Omitting --sizes honours case.mesh.element_size as finest rung (4h, 2h, h).
    rc_default = main(["converge", str(case_path)])
    out_default = capsys.readouterr().out
    assert rc_default == 0
    assert [float(row[0]) for row in _parse_table_rows(out_default)] == [0.008, 0.004, 0.002]

    # --element-size without --sizes overrides case.mesh.element_size.
    rc_flag = main(["converge", str(case_path), "--element-size", "0.0015"])
    out_flag = capsys.readouterr().out
    assert rc_flag == 0
    assert [float(row[0]) for row in _parse_table_rows(out_flag)] == [0.006, 0.003, 0.0015]

    # Invalid --sizes exit 1 without traceback; --engine magpylib exits 2.
    for bad_sizes in (
        "0.001",
        "0.001,0.002",
        "0.002,0.002",
        "0.002,0",
        "0.002,-0.001",
        "0.002,nan",
        "0.002,abc",
    ):
        rc_bad = main(["converge", str(case_path), "--sizes", bad_sizes])
        bad_cap = capsys.readouterr()
        assert rc_bad == 1
        assert bad_cap.out == ""
        assert len(bad_cap.err.splitlines()) == 1
        assert "--sizes" in bad_cap.err
        assert "Traceback" not in bad_cap.err

    with pytest.raises(SystemExit) as excinfo:
        main(["converge", str(case_path), "--engine", "magpylib"])
    assert excinfo.value.code == 2
    assert "magpylib" in capsys.readouterr().err


def test_converge_two_rungs_and_non_uniform_not_estimable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_path = tmp_path / "lin_solenoid.json"
    case_path.write_text(_linear_solenoid_case().to_json(), encoding="utf-8")

    rc2 = main(["converge", str(case_path), "--sizes", "0.008,0.004"])
    cap2 = capsys.readouterr()
    assert rc2 == 0, cap2.err
    assert "observed order: not estimable (two rungs)" in cap2.out
    assert "estimated discretisation error: not reported (two rungs)" in cap2.out
    assert "observed order p =" not in cap2.out
    assert "WARNING:" not in cap2.out
    assert "converged: NO" in cap2.out

    rc3 = main(["converge", str(case_path), "--sizes", "0.008,0.005,0.002"])
    cap3 = capsys.readouterr()
    assert rc3 == 0, cap3.err
    assert "observed order: not estimable (non-uniform ladder)" in cap3.out
    assert "estimated discretisation error: not reported (non-uniform ladder)" in cap3.out
    assert "converged: NO" in cap3.out


def test_converge_non_falling_deltas_warns_and_forces_no(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_path = tmp_path / "lin_solenoid.json"
    case_path.write_text(_linear_solenoid_case().to_json(), encoding="utf-8")

    rc = main(["converge", str(case_path), "--sizes", "0.02,0.019,0.001"])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    assert (
        "WARNING: |delta| does not fall under refinement on this ladder; "
        "this table is not convergence evidence"
    ) in cap.out
    assert "converged: NO" in cap.out

    # Uniform-ratio synthetic non-falling deltas must also warn and refuse convergence.
    study = analyse_ladder(
        "synthetic_non_falling",
        ("probe #0 Bz (T)",),
        (
            ConvergenceRung(0.004, 100, 180, (1.0,), ("T",)),
            ConvergenceRung(0.002, 400, 720, (1.01,), ("T",)),
            ConvergenceRung(0.001, 1600, 2880, (1.05,), ("T",)),
        ),
    )
    assert study.delta_falling is False
    assert study.converged is False
    formatted = format_convergence_table(study)
    assert (
        "WARNING: |delta| does not fall under refinement on this ladder; "
        "this table is not convergence evidence"
    ) in formatted
    assert "estimated discretisation error: not reported (non-positive observed order)" in formatted
    assert "converged: NO" in formatted


def test_solve_fem2d_probe_output_and_sampling_refusal(
    capsys: pytest.CaptureFixture[str], steel_direct_1mm: Result
) -> None:
    rc = main(["solve", str(STEEL_EXAMPLE), "--engine", "fem2d"])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    assert cap.err == ""
    assert cap.out == f"probe #0 (0.0, 0.0, 0.0) m  B  z = {steel_direct_1mm.values()[0]!r} T\n"

    for extra in (["--points", "0,0,0"], ["--grid", "0,0.01,0.01:0,0,1:0,0,1"]):
        rc_bad = main(["solve", str(STEEL_EXAMPLE), "--engine", "fem2d", *extra])
        bad_cap = capsys.readouterr()
        assert rc_bad == 2
        assert bad_cap.out == ""
        assert len(bad_cap.err.splitlines()) == 1
        assert "sampler" in bad_cap.err or "fem2d" in bad_cap.err


def test_default_path_bit_identical_and_examples_roundtrip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    steel_doc = json.loads(STEEL_EXAMPLE.read_text(encoding="utf-8"))
    assert steel_doc["mesh"] == {"element_size": 0.001}
    steel_doc.pop("mesh")
    no_mesh_path = tmp_path / "steel_no_mesh.json"
    no_mesh_path.write_text(json.dumps(steel_doc, indent=2, sort_keys=True), encoding="utf-8")

    rc_report = main(["report", str(no_mesh_path), "--engine", "fem2d"])
    cap_report = capsys.readouterr()
    assert rc_report == 0, cap_report.err
    payload = _extract_final_json(cap_report.out)
    assert payload["probes"][0]["computed"] == 0.14049244777644335
    assert payload["convergence"]["nodes"] == 2958
    assert payload["convergence"]["elements"] == 5676

    n42_case = Case.from_json(N42_AIR_EXAMPLE.read_text(encoding="utf-8"))
    assert n42_case.mesh is None
    assert "mesh" not in json.loads(N42_AIR_EXAMPLE.read_text(encoding="utf-8"))

    for example_path in sorted(EXAMPLES_DIR.glob("*.json")):
        raw_text = example_path.read_text(encoding="utf-8")
        raw_obj = json.loads(raw_text)
        loaded = Case.from_json(raw_text)
        if "mesh" not in raw_obj:
            assert loaded.mesh is None
            assert "mesh" not in loaded.to_dict()
            assert (
                json.dumps(loaded.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                == raw_text
            )
        assert Case.from_json(loaded.to_json()).to_json() == loaded.to_json()
