"""Tests for tools/engine_guard.py and tools/adr_check_map.py.

These two tools are the deterministic replacements for prose rules, so they are
tested the way a rule should be: with a positive case that proves they fire and a
negative case that proves they stay quiet. A guard that never blocks is worse than
no guard, because it creates the impression of enforcement.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / "tools" / "engine_guard.py"
MAPPER = REPO / "tools" / "adr_check_map.py"


def _run(args, cwd=None, timeout=60):
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


# --------------------------------------------------------------------------- guard


def test_guard_is_clear_on_a_quiet_repository(tmp_path):
    (tmp_path / ".git").mkdir()
    proc = _run([str(GUARD), "--repo", str(tmp_path), "--json"])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["clear"] is True
    assert payload["engines"] == []


def test_guard_blocks_when_an_engine_already_owns_the_repo(tmp_path):
    """Positive control: a process named `agy` running in the tree must block."""
    (tmp_path / ".git").mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    engine = fake_bin / "agy"
    engine.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n", encoding="utf-8")
    engine.chmod(0o755)

    child = subprocess.Popen([str(engine)], cwd=tmp_path)
    try:
        proc = _run([str(GUARD), "--repo", str(tmp_path), "--json"])
        payload = json.loads(proc.stdout)
        assert proc.returncode == 3, f"expected block, got {proc.returncode}: {proc.stderr}"
        assert payload["blocked"] is True
        assert any(row["pid"] == child.pid for row in payload["engines"])
        assert payload["engines"][0]["matched_by"] in {"cwd", "cmdline"}
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_guard_ignores_an_engine_in_a_different_repository(tmp_path):
    """Negative control: the bake-off runs two engines on two clones, which is legal."""
    other = tmp_path / "other"
    target = tmp_path / "target"
    for path in (other, target):
        (path / ".git").mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    engine = fake_bin / "agy"
    engine.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n", encoding="utf-8")
    engine.chmod(0o755)

    child = subprocess.Popen([str(engine)], cwd=other)
    try:
        proc = _run([str(GUARD), "--repo", str(target), "--json"])
        assert proc.returncode == 0, f"cross-repo engine must not block: {proc.stderr}"
        assert json.loads(proc.stdout)["clear"] is True
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_guard_detects_the_repo_path_inside_a_prompt(tmp_path):
    """Real incident shape: the engine's cwd was the tree but the prompt named it too."""
    (tmp_path / ".git").mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    engine = fake_bin / "agy"
    engine.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n", encoding="utf-8")
    engine.chmod(0o755)

    child = subprocess.Popen([str(engine), "--print", f"work in {tmp_path} please"], cwd=tmp_path)
    try:
        payload = json.loads(_run([str(GUARD), "--repo", str(tmp_path), "--json"]).stdout)
        assert payload["blocked"] is True
        row = next(r for r in payload["engines"] if r["pid"] == child.pid)
        assert row["matched_by"] in {"cwd", "cmdline"}
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_guard_force_overrides_with_a_warning(tmp_path):
    (tmp_path / ".git").mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    engine = fake_bin / "agy"
    engine.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n", encoding="utf-8")
    engine.chmod(0o755)

    child = subprocess.Popen([str(engine)], cwd=tmp_path)
    try:
        proc = _run([str(GUARD), "--repo", str(tmp_path), "--force"])
        assert proc.returncode == 0
        assert "WARNING" in proc.stderr
    finally:
        child.terminate()
        child.wait(timeout=10)


# -------------------------------------------------------------------------- mapper


def _write_adr(root: Path, name: str, body: str) -> None:
    adr_dir = root / "docs" / "adr"
    adr_dir.mkdir(parents=True, exist_ok=True)
    (adr_dir / name).write_text(body, encoding="utf-8")


def test_mapper_accepts_a_resolvable_link(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_guard_case():\n    assert True\n", encoding="utf-8"
    )
    _write_adr(
        tmp_path,
        "0001-demo.md",
        "# Demo\n\n- **Rule one.** Do the thing. — enforced-by: tests/test_x.py::test_guard_case\n",
    )
    proc = _run([str(MAPPER), "--root", str(tmp_path), "--json"])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["enforced"] == 1
    assert payload["dangling"] == []


def test_mapper_fails_on_a_dangling_link(tmp_path):
    """A constraint claiming a test that does not exist is a false claim, so it fails."""
    _write_adr(
        tmp_path,
        "0002-demo.md",
        "# Demo\n\n- **Rule two.** Do the thing. — enforced-by: tests/test_nope.py::test_missing\n",
    )
    proc = _run([str(MAPPER), "--root", str(tmp_path)])
    assert proc.returncode == 1
    assert "DANGLING" in proc.stdout
    assert "tests/test_nope.py" in proc.stdout


def test_mapper_fails_on_a_symbol_that_is_not_defined(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_present():\n    pass\n", encoding="utf-8"
    )
    _write_adr(
        tmp_path,
        "0003-demo.md",
        "# Demo\n\n- **Rule three.** — enforced-by: tests/test_x.py::test_absent\n",
    )
    proc = _run([str(MAPPER), "--root", str(tmp_path)])
    assert proc.returncode == 1
    assert "test_absent" in proc.stdout


def test_mapper_reports_untagged_constraints_and_strict_fails(tmp_path):
    _write_adr(
        tmp_path,
        "0004-demo.md",
        "- **Untagged rule.** Nobody checks this.\n- **Tagged rule.** Checked. — enforced-by: docs/adr/0004-demo.md\n",
    )
    lenient = _run([str(MAPPER), "--root", str(tmp_path)])
    assert lenient.returncode == 0
    assert "untagged 1" in lenient.stdout

    strict = _run([str(MAPPER), "--root", str(tmp_path), "--strict"])
    assert strict.returncode == 1


def test_mapper_on_this_repository_has_no_dangling_links():
    """The real repository must not claim an enforcement that does not exist."""
    proc = _run([str(MAPPER), "--root", str(REPO), "--json"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["dangling"] == []
    assert payload["constraints"] > 0
