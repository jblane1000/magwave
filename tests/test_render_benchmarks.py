"""Tests for tools/render_benchmarks.py and the docs/BENCHMARKS.md staleness gate (FR-071)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RENDERER = REPO / "tools" / "render_benchmarks.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RENDERER), *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_render_is_byte_identical_across_runs(tmp_path: Path) -> None:
    run1 = tmp_path / "run1.md"
    run2 = tmp_path / "run2.md"

    proc1 = _run("--out", str(run1))
    assert proc1.returncode == 0, proc1.stderr
    proc2 = _run("--out", str(run2))
    assert proc2.returncode == 0, proc2.stderr

    bytes1 = run1.read_bytes()
    bytes2 = run2.read_bytes()
    assert bytes1 == bytes2

    text = bytes1.decode("utf-8")
    assert str(Path.home()) not in text
    assert "/home/" not in text
    assert "/Users/" not in text
    assert str(REPO) not in text
    assert re.search(r"\d{4}-\d{2}-\d{2}", text) is None
    assert re.search(r"(?<![.\w])[0-9a-f]{7,}(?![\w]|[+-]\d)", text) is None


def test_check_accepts_a_fresh_render(tmp_path: Path) -> None:
    target = tmp_path / "BENCHMARKS.md"

    rendered = _run("--out", str(target))
    assert rendered.returncode == 0, rendered.stderr

    checked = _run("--check", "--out", str(target))
    assert checked.returncode == 0, checked.stderr


def test_check_rejects_a_perturbed_render(tmp_path: Path) -> None:
    target = tmp_path / "BENCHMARKS.md"

    rendered = _run("--out", str(target))
    assert rendered.returncode == 0, rendered.stderr

    original = target.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    sep_index = next(i for i, line in enumerate(lines) if line.startswith("|---|"))
    data_index = sep_index + 1
    parts = re.split(r"(?<!\\)\|", lines[data_index])
    computed_cell = parts[4]
    mantissa, sep, exponent = computed_cell.partition("e")
    assert sep == "e", f"unexpected computed cell format: {computed_cell!r}"
    replacement_digit = "1" if mantissa[-1] != "1" else "2"
    parts[4] = mantissa[:-1] + replacement_digit + sep + exponent
    lines[data_index] = "|".join(parts)
    perturbed = "".join(lines)
    assert perturbed != original
    target.write_text(perturbed, encoding="utf-8")

    checked = _run("--check", "--out", str(target))
    assert checked.returncode == 1
    assert target.name in checked.stderr
    assert "stale" in checked.stderr
