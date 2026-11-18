"""``magwave report``: deterministic Markdown report generation (FR-053)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import magpylib as magpy
import numpy as np
import pytest

from magwave.analytic import dipole_B
from magwave.benchmarks import load_corpus
from magwave.cases import Case, Dipole, ExpectedValue, Probe
from magwave.cli import main
from magwave.engines import MagpylibEngine
from magwave.engines.base import Capabilities, Engine, Problem, Result, Sample, raise_on_errors
from magwave.report import display_case_file, render_report

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "solve-dipole.json"
CORPUS = {case.name: case for case in load_corpus()}
PROVENANCE_KEYS = ("engine", "engine_version", "rung", "units", "case_file")


def _extract_final_json(markdown: str) -> dict:
    match = re.search(r"```json\n(.*?)\n```\s*\Z", markdown, flags=re.DOTALL)
    assert match is not None, "report must end with a fenced ```json block"
    return json.loads(match.group(1))


def _write_case(target: Path, case: Case) -> Path:
    target.write_text(case.to_json(), encoding="utf-8")
    return target


class _CustomDummyEngine:
    """Minimal dummy engine with custom provenance to verify no hard-coding."""

    def capabilities(self) -> Capabilities:
        return Capabilities(
            engine_id="custom-solver",
            rung="custom-rung",
            source_types=frozenset({"dipole"}),
            magnet_kinds=frozenset(),
            coil_kinds=frozenset(),
            quantities=frozenset({"B"}),
            force_methods=(),
            engine_version="9.8.7",
        )

    def validate(self, case: Case) -> list[Problem]:
        return [
            Problem("E_UNSUPPORTED_SOURCE_TYPE", f"{case.name}: {type(c).__name__}", "error")
            for c in case.components
            if not isinstance(c, Dipole)
        ]

    def solve(self, case: Case) -> Result:
        warnings = raise_on_errors("custom-solver", self.validate(case))
        samples = []
        for probe in case.probes:
            total = np.zeros(3)
            for dipole in case.components:
                total += dipole_B(dipole.moment, np.subtract(probe.point, dipole.center))
            vector = (float(total[0]), float(total[1]), float(total[2]))
            samples.append(Sample(probe.point, "B", vector, "T", probe.component))
        return Result(
            engine_id="custom-solver",
            engine_version="9.8.7",
            rung="custom-rung",
            method="custom dummy dipole",
            samples=tuple(samples),
            problems=tuple(warnings),
        )


def test_report_contains_provenance_sections_and_every_probe_value(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "report.md"
    assert main(["report", str(EXAMPLE), "--engine", "magpylib", "--out", str(out_path)]) == 0
    text = out_path.read_text(encoding="utf-8")

    case = Case.from_json(EXAMPLE.read_text(encoding="utf-8"))
    result = MagpylibEngine().solve(case)

    lines = text.splitlines()
    # 1. Title line with case name and description
    assert lines[0].startswith("# ")
    assert case.name in lines[0]
    assert case.description in lines[0]

    # 2. Provenance block in exact order
    provenance_lines = lines[2:7]
    assert [line.split(": ", 1)[0] for line in provenance_lines] == list(PROVENANCE_KEYS)
    assert "engine: magpylib" in provenance_lines
    assert f"engine_version: {magpy.__version__}" in provenance_lines
    assert "rung: second-code" in provenance_lines
    assert "units: position m, B T" in provenance_lines

    # 3 & 4. Probe table with expected, tolerance, PASS/FAIL for corpus case
    table_lines = [line for line in lines if line.startswith("|")]
    assert len(table_lines) == 2 + len(case.probes)
    assert "expected" in table_lines[0]
    assert "tolerance" in table_lines[0]
    assert "result" in table_lines[0]
    for value, row_line in zip(result.values(), table_lines[2:], strict=True):
        assert repr(float(value)) in row_line
        assert repr(float(case.expected[0].value)) in row_line
        assert repr(float(case.expected[0].tolerance)) in row_line
        assert "PASS" in row_line

    # 5. Final fenced json block carrying the same facts
    payload = _extract_final_json(text)
    assert payload["schema"] == "magwave.report/1"
    assert payload["name"] == case.name
    assert payload["description"] == case.description
    assert payload["engine"] == "magpylib"
    assert payload["engine_version"] == magpy.__version__
    assert payload["rung"] == "second-code"
    assert payload["units"] == "position m, B T"
    assert payload["corpus_case"] is True
    assert payload["passed"] is True
    assert payload["result"] == "PASS"
    assert len(payload["probes"]) == len(case.probes)
    assert payload["probes"][0]["computed"] == result.values()[0]
    assert payload["probes"][0]["expected"] == case.expected[0].value
    assert payload["probes"][0]["tolerance"] == case.expected[0].tolerance
    assert payload["probes"][0]["result"] == "PASS"

    # Section ordering: title < provenance < table < json
    idx_title = text.index(f"# {case.name}")
    idx_prov = text.index("engine: magpylib")
    idx_table = text.index("| probe |")
    idx_json = text.index("```json")
    assert idx_title < idx_prov < idx_table < idx_json


def test_report_two_runs_produce_identical_bytes_and_no_absolute_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    r1 = tmp_path / "r1.md"
    r2 = tmp_path / "r2.md"
    assert main(["report", str(EXAMPLE), "--out", str(r1)]) == 0
    assert main(["report", str(EXAMPLE), "--out", str(r2)]) == 0
    bytes1 = r1.read_bytes()
    bytes2 = r2.read_bytes()
    assert bytes1 == bytes2

    assert main(["report", str(EXAMPLE), "--out", "-"]) == 0
    stdout_bytes = capsys.readouterr().out.encode("utf-8")
    assert stdout_bytes == bytes1

    # Even when the input case path is an absolute path outside cwd, no absolute path leaks
    local_case = _write_case(tmp_path / "custom_abs_case.json", CORPUS["dipole_axis_Bz"])
    out_abs = tmp_path / "abs_report.md"
    assert main(["report", str(local_case), "--out", str(out_abs)]) == 0
    report_text = out_abs.read_text(encoding="utf-8")
    assert str(tmp_path) not in report_text
    assert str(ROOT) not in report_text
    payload = _extract_final_json(report_text)
    assert payload["case_file"] == "custom_abs_case.json"
    assert not Path(payload["case_file"]).is_absolute()
    assert display_case_file("examples/solve-dipole.json") == "examples/solve-dipole.json"


def test_non_corpus_multi_probe_case_omits_corpus_columns_and_reports_all_probes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = Case(
        name="custom_two_probe_dipole",
        description="Non-corpus case with two probes",
        sources=("unit test",),
        components=(Dipole(moment=(0.0, 0.0, 1.0), name="d1"),),
        probes=(
            Probe(point=(0.0, 0.0, 0.5), quantity="B", component="z"),
            Probe(point=(0.3, 0.0, 0.4), quantity="B", component="magnitude"),
        ),
        expected=(
            ExpectedValue(
                check="dipole_B",
                quantity="Bz",
                value=1.6e-6,
                unit="T",
                tolerance=1e-9,
                source="unit test",
            ),
        ),
    )
    case_path = _write_case(tmp_path / "non_corpus.json", case)
    assert main(["report", str(case_path)]) == 0
    text = capsys.readouterr().out
    result = MagpylibEngine().solve(case)

    table_lines = [line for line in text.splitlines() if line.startswith("|")]
    assert len(table_lines) == 2 + len(case.probes)
    assert "expected" not in table_lines[0]
    assert "tolerance" not in table_lines[0]
    for value, row_line in zip(result.values(), table_lines[2:], strict=True):
        assert repr(float(value)) in row_line

    payload = _extract_final_json(text)
    assert payload["corpus_case"] is False
    assert "passed" not in payload
    assert "result" not in payload
    assert [row["computed"] for row in payload["probes"]] == list(result.values())
    assert all("expected" not in row for row in payload["probes"])


def test_corpus_case_reports_fail_when_perturbed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = CORPUS["cuboid_axis_Bz"]
    item = case.expected[0]
    perturbed = replace(case, expected=(replace(item, value=item.value * 1.01),))
    case_path = _write_case(tmp_path / "perturbed.json", perturbed)
    assert main(["report", str(case_path)]) == 0
    text = capsys.readouterr().out
    table_lines = [line for line in text.splitlines() if line.startswith("|")]
    assert "FAIL" in table_lines[2]
    payload = _extract_final_json(text)
    assert payload["corpus_case"] is True
    assert payload["passed"] is False
    assert payload["result"] == "FAIL"
    assert payload["probes"][0]["passed"] is False
    assert payload["probes"][0]["result"] == "FAIL"


def test_report_reads_engine_provenance_dynamically() -> None:
    engine = _CustomDummyEngine()
    assert isinstance(engine, Engine)
    case = CORPUS["dipole_axis_Bz"]
    text = render_report(case, engine, case_file="examples/solve-dipole.json", corpus=[case])
    assert "engine: custom-solver\n" in text
    assert "engine_version: 9.8.7\n" in text
    assert "rung: custom-rung\n" in text
    payload = _extract_final_json(text)
    assert payload["engine"] == "custom-solver"
    assert payload["engine_version"] == "9.8.7"
    assert payload["rung"] == "custom-rung"


@pytest.mark.parametrize(
    ("case_name", "expected_code"),
    [
        ("halbach_bore_B", "E_UNSUPPORTED_GEOMETRY"),
        ("thin_solenoid_axis_Bz", "E_UNSUPPORTED_COIL"),
        ("force_dipole_coaxial_far", "E_UNSUPPORTED_QUANTITY"),
    ],
)
def test_unmodelled_case_exits_one_with_single_line_stderr(
    case_name: str, expected_code: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_path = _write_case(tmp_path / f"{case_name}.json", CORPUS[case_name])
    assert main(["report", str(case_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    err_lines = captured.err.splitlines()
    assert len(err_lines) == 1
    assert "magwave report:" in err_lines[0]
    assert expected_code in err_lines[0]


def test_load_and_output_errors_exit_one_with_single_line_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Missing case path argument
    assert main(["report"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1

    # Missing file
    assert main(["report", str(tmp_path / "missing.json")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1

    # Invalid JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    assert main(["report", str(bad_json)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1

    # Malformed case schema
    broken = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    broken["unexpected"] = True
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    assert main(["report", str(broken_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1

    # Unwritable output path
    assert main(["report", str(EXAMPLE), "--out", str(tmp_path / "no_dir" / "out.md")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1


def test_report_module_imports_without_magpylib() -> None:
    code = (
        "import sys, magwave.report, magwave.cli; "
        "assert 'magpylib' not in sys.modules, 'magpylib imported'"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_invalid_flag_combinations_raise_argparse_error() -> None:
    for argv in (
        ["report", str(EXAMPLE), "--grid", "0,1,1:0,1,1:0,1,1"],
        ["report", str(EXAMPLE), "--json", "out.json"],
        ["bench", "--out", "out.md"],
        ["solve", str(EXAMPLE), "--points", "0,0,1", "--out", "out.md"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2
