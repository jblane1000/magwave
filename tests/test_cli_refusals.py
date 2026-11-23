"""CLI boundary refusal tests for mesh node budget (E_MESH_NODE_BUDGET) and --json paths (E_OUTPUT_PATH)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from magwave.cases import Case
from magwave.cli import E_OUTPUT_PATH, main
from magwave.engines import EngineError, Fem2dEngine
from magwave.fem2d import (
    DEFAULT_MAX_NODES,
    E_MESH_NODE_BUDGET,
    MeshBudgetError,
    check_rectangular_mesh_budget,
    rectangular_mesh,
)

ROOT = Path(__file__).resolve().parents[1]
STEEL_EXAMPLE = ROOT / "examples" / "steel-cylinder-solenoid-nonlinear.json"


def _expected_budget_failure(case: Case, element_size: float) -> tuple[int, int, int]:
    """Compute (nodes, nr, nz) via Fem2dEngine parameters without allocating the mesh."""
    engine = Fem2dEngine(element_size=element_size, material_model="nonlinear")
    r_max, z_min, z_max, h, ref_box, bps_r, bps_z, _, probe_problems = engine._mesh_parameters(case)
    assert not probe_problems
    with pytest.raises(MeshBudgetError) as excinfo:
        check_rectangular_mesh_budget(
            r_max,
            z_min,
            z_max,
            h,
            refinement_box=ref_box,
            breakpoints_r=bps_r,
            breakpoints_z=bps_z,
            growth=engine._growth,
        )
    err = excinfo.value
    assert err.max_nodes == DEFAULT_MAX_NODES
    return err.nodes, err.nr, err.nz


def _assert_single_line_refusal(
    stderr: str,
    stdout: str,
    command: str,
    code: str,
) -> str:
    assert stdout == ""
    assert "Traceback" not in stderr
    lines = stderr.splitlines()
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith(f"magwave {command}: ")
    assert code in line
    return line


# --------------------------------------------------------------------------------------
# Defect 1: mesh node budget refusal (E_MESH_NODE_BUDGET)
# --------------------------------------------------------------------------------------


def test_rectangular_mesh_raises_structured_mesh_budget_error_before_allocation() -> None:
    with pytest.raises(MeshBudgetError) as excinfo:
        rectangular_mesh(0.10, -0.10, 0.10, 0.001, max_nodes=100)
    err = excinfo.value
    assert isinstance(err, ValueError)
    assert err.max_nodes == 100
    assert err.element_size == 0.001
    assert err.nodes == err.nr * err.nz
    assert err.nodes > 100

    # Extremely fine h must fail cleanly in budget calculation without allocating gigabytes.
    with pytest.raises(MeshBudgetError) as tiny_exc:
        rectangular_mesh(0.10, -0.10, 0.10, 1e-10)
    assert tiny_exc.value.nodes > DEFAULT_MAX_NODES


def test_fem2d_engine_validate_solve_and_build_refuse_with_e_mesh_node_budget() -> None:
    case = Case.from_json(STEEL_EXAMPLE.read_text(encoding="utf-8"))
    nodes, nr, nz = _expected_budget_failure(case, 1e-5)
    engine = Fem2dEngine(element_size=1e-5, material_model="nonlinear")

    problems = engine.validate(case)
    assert len(problems) == 1
    problem = problems[0]
    assert problem.code == E_MESH_NODE_BUDGET
    assert problem.is_error
    assert "1e-05" in problem.message
    assert f"{nodes} nodes ({nr} x {nz})" in problem.message
    assert f"max_nodes={DEFAULT_MAX_NODES}" in problem.message

    with pytest.raises(EngineError) as solve_exc:
        engine.solve(case)
    assert [p.code for p in solve_exc.value.problems] == [E_MESH_NODE_BUDGET]

    with pytest.raises(EngineError) as build_exc:
        engine._build_mesh_and_load(case)
    assert [p.code for p in build_exc.value.problems] == [E_MESH_NODE_BUDGET]


@pytest.mark.parametrize(
    ("command", "argv_tail", "failing_h"),
    [
        ("solve", ["--engine", "fem2d", "--element-size", "1e-5"], 1e-5),
        ("report", ["--engine", "fem2d", "--element-size", "1e-5"], 1e-5),
        ("converge", ["--engine", "fem2d", "--sizes", "1e-4,1e-5"], 1e-5),
        ("converge", ["--engine", "fem2d", "--element-size", "1e-5"], 4e-5),
    ],
)
def test_cli_fem2d_commands_refuse_over_budget_mesh_uniformly(
    command: str,
    argv_tail: list[str],
    failing_h: float,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case = Case.from_json(STEEL_EXAMPLE.read_text(encoding="utf-8"))
    nodes, nr, nz = _expected_budget_failure(case, failing_h)

    def _forbid_build_or_solve(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("mesh builder / solver must not run when mesh budget is exceeded")

    monkeypatch.setattr(Fem2dEngine, "_build_mesh_and_load", _forbid_build_or_solve)
    if command == "converge":
        monkeypatch.setattr(Fem2dEngine, "_solve_core", _forbid_build_or_solve)

    rc = main([command, str(STEEL_EXAMPLE), *argv_tail])
    captured = capsys.readouterr()

    assert rc == 1
    line = _assert_single_line_refusal(
        captured.err, captured.out, command, f"error: {E_MESH_NODE_BUDGET}:"
    )
    assert f"element_size={failing_h:g} m" in line
    assert f"{nodes} nodes ({nr} x {nz})" in line
    assert f"max_nodes={DEFAULT_MAX_NODES}" in line


def test_solve_report_converge_emit_identical_refusal_detail_for_same_request(
    capsys: pytest.CaptureFixture[str],
) -> None:
    details: list[str] = []
    invocations = [
        ("solve", ["solve", str(STEEL_EXAMPLE), "--engine", "fem2d", "--element-size", "1e-5"]),
        ("report", ["report", str(STEEL_EXAMPLE), "--engine", "fem2d", "--element-size", "1e-5"]),
        (
            "converge",
            ["converge", str(STEEL_EXAMPLE), "--engine", "fem2d", "--sizes", "1e-4,1e-5"],
        ),
    ]
    for command, argv in invocations:
        rc = main(argv)
        captured = capsys.readouterr()
        assert rc == 1
        line = _assert_single_line_refusal(
            captured.err, captured.out, command, f"error: {E_MESH_NODE_BUDGET}:"
        )
        prefix = f"magwave {command}: "
        assert line.startswith(prefix)
        details.append(line.removeprefix(prefix))

    assert details[0] == details[1] == details[2]


def test_case_mesh_block_over_budget_refuses_without_cli_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doc = json.loads(STEEL_EXAMPLE.read_text(encoding="utf-8"))
    doc["mesh"] = {"element_size": 1e-5}
    case_path = tmp_path / "over_dense.json"
    case_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    case = Case.from_json(case_path.read_text(encoding="utf-8"))

    for command, extra, failing_h in (
        ("solve", ["--engine", "fem2d"], 1e-5),
        ("report", ["--engine", "fem2d"], 1e-5),
        ("converge", [], 4e-5),
    ):
        nodes, nr, nz = _expected_budget_failure(case, failing_h)
        rc = main([command, str(case_path), *extra])
        captured = capsys.readouterr()
        assert rc == 1
        line = _assert_single_line_refusal(
            captured.err, captured.out, command, f"error: {E_MESH_NODE_BUDGET}:"
        )
        assert f"element_size={failing_h:g} m" in line
        assert f"{nodes} nodes ({nr} x {nz})" in line
        assert f"max_nodes={DEFAULT_MAX_NODES}" in line


def test_cli_subprocess_mesh_refusal_exits_one_single_line() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "magwave.cli",
            "solve",
            str(STEEL_EXAMPLE),
            "--engine",
            "fem2d",
            "--element-size",
            "1e-5",
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    _assert_single_line_refusal(proc.stderr, proc.stdout, "solve", f"error: {E_MESH_NODE_BUDGET}:")


# --------------------------------------------------------------------------------------
# Defect 2: --json output path validated before work (E_OUTPUT_PATH)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "base_argv"),
    [
        ("bench", ["bench"]),
        ("bench", ["bench", "--engine", "magpylib"]),
        ("bench", ["bench", "--engine", "fem2d"]),
        ("converge", ["converge", str(STEEL_EXAMPLE), "--sizes", "0.004,0.002"]),
        ("import", ["import", "unused.step"]),
    ],
)
def test_json_missing_directory_refuses_before_doing_work(
    command: str,
    base_argv: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import magwave.benchmarks as benchmarks
    import magwave.cad as cad

    def _forbid(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("command performed work despite unwritable --json path")

    monkeypatch.setattr(benchmarks, "run_benchmarks", _forbid)
    monkeypatch.setattr(benchmarks, "run_engine_benchmarks", _forbid)
    monkeypatch.setattr(Fem2dEngine, "solve", _forbid)
    monkeypatch.setattr(cad, "import_cad", _forbid)

    bad_path = tmp_path / "nonexistent-dir" / "out.json"
    rc = main([*base_argv, "--json", str(bad_path)])
    captured = capsys.readouterr()

    assert rc == 1
    line = _assert_single_line_refusal(
        captured.err, captured.out, command, f"{E_OUTPUT_PATH}: {bad_path}: "
    )
    assert "missing directory" in line


@pytest.mark.parametrize(
    ("command", "base_argv"),
    [
        ("bench", ["bench"]),
        ("bench", ["bench", "--engine", "magpylib"]),
        ("bench", ["bench", "--engine", "fem2d"]),
        ("converge", ["converge", str(STEEL_EXAMPLE), "--sizes", "0.004,0.002"]),
        ("import", ["import", "unused.step"]),
    ],
)
def test_json_unwritable_path_variants_refuse_before_doing_work(
    command: str,
    base_argv: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import magwave.benchmarks as benchmarks
    import magwave.cad as cad

    def _forbid(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("command performed work despite invalid --json path")

    monkeypatch.setattr(benchmarks, "run_benchmarks", _forbid)
    monkeypatch.setattr(benchmarks, "run_engine_benchmarks", _forbid)
    monkeypatch.setattr(Fem2dEngine, "solve", _forbid)
    monkeypatch.setattr(cad, "import_cad", _forbid)

    existing_file = tmp_path / "regular.txt"
    existing_file.write_text("keep", encoding="utf-8")

    cases: list[tuple[str, str]] = [
        (str(tmp_path), "not writable"),
        (f"{tmp_path / 'missing'}{os.sep}", "not writable"),
        ("", "not writable"),
        (str(existing_file / "out.json"), "not a directory"),
    ]
    for bad_path, reason_fragment in cases:
        rc = main([*base_argv, "--json", bad_path])
        captured = capsys.readouterr()
        assert rc == 1
        line = _assert_single_line_refusal(
            captured.err, captured.out, command, f"{E_OUTPUT_PATH}: {bad_path}: "
        )
        assert reason_fragment in line


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses DAC file permissions")
@pytest.mark.parametrize(
    ("command", "base_argv"),
    [
        ("bench", ["bench"]),
        ("converge", ["converge", str(STEEL_EXAMPLE), "--sizes", "0.004,0.002"]),
        ("import", ["import", "unused.step"]),
    ],
)
def test_json_readonly_dir_and_readonly_file_refuse_before_doing_work(
    command: str,
    base_argv: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import magwave.benchmarks as benchmarks
    import magwave.cad as cad

    def _forbid(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("command performed work despite read-only --json target")

    monkeypatch.setattr(benchmarks, "run_benchmarks", _forbid)
    monkeypatch.setattr(Fem2dEngine, "solve", _forbid)
    monkeypatch.setattr(cad, "import_cad", _forbid)

    ro_dir = tmp_path / "ro_dir"
    ro_dir.mkdir()
    ro_dir.chmod(0o555)
    try:
        rc_dir = main([*base_argv, "--json", str(ro_dir / "out.json")])
        cap_dir = capsys.readouterr()
        assert rc_dir == 1
        line_dir = _assert_single_line_refusal(
            cap_dir.err, cap_dir.out, command, f"{E_OUTPUT_PATH}: "
        )
        assert "not writable" in line_dir
    finally:
        ro_dir.chmod(0o755)

    ro_file = tmp_path / "ro_file.json"
    ro_file.write_text('{"preserve": true}\n', encoding="utf-8")
    ro_file.chmod(0o444)
    try:
        rc_file = main([*base_argv, "--json", str(ro_file)])
        cap_file = capsys.readouterr()
        assert rc_file == 1
        line_file = _assert_single_line_refusal(
            cap_file.err, cap_file.out, command, f"{E_OUTPUT_PATH}: {ro_file}: "
        )
        assert "not writable" in line_file
        assert ro_file.read_text(encoding="utf-8") == '{"preserve": true}\n'
    finally:
        ro_file.chmod(0o644)


def test_writable_json_output_is_byte_identical_and_preserved_on_later_refusal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["bench"]) == 0
    plain_cap = capsys.readouterr()
    assert plain_cap.err == ""

    bench_json = tmp_path / "bench.json"
    assert main(["bench", "--json", str(bench_json)]) == 0
    json_cap = capsys.readouterr()
    assert json_cap.err == ""
    assert json_cap.out == plain_cap.out
    assert json.loads(bench_json.read_text(encoding="utf-8"))["summary"]["total"] == 27

    # Pre-flight validation must not truncate an existing writable file when converge fails.
    existing_conv = tmp_path / "existing_converge.json"
    existing_conv.write_text('{"sentinel": 123}\n', encoding="utf-8")
    rc = main(
        [
            "converge",
            str(STEEL_EXAMPLE),
            "--sizes",
            "1e-4,1e-5",
            "--json",
            str(existing_conv),
        ]
    )
    conv_cap = capsys.readouterr()
    assert rc == 1
    _assert_single_line_refusal(
        conv_cap.err, conv_cap.out, "converge", f"error: {E_MESH_NODE_BUDGET}:"
    )
    assert existing_conv.read_text(encoding="utf-8") == '{"sentinel": 123}\n'


def test_json_flag_on_unsupported_commands_stays_exit_two() -> None:
    for argv in (
        ["solve", str(STEEL_EXAMPLE), "--points", "0,0,0", "--json", "/nonexistent-dir/out.json"],
        ["report", str(STEEL_EXAMPLE), "--json", "/nonexistent-dir/out.json"],
        ["gui", "--json", "/nonexistent-dir/out.json"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2
