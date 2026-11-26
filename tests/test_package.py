"""Package-level contract tests.

These are deliberately thin: Phase 0 ships structure, not physics. What they do prove is that the
scaffolding is honest — version agreement, a headless CLI, and the licence boundary (NFR-3).
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import tomllib
from pathlib import Path

import pytest

import magwave
from magwave.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_version_matches_pyproject() -> None:
    assert magwave.__version__ == _pyproject()["project"]["version"]


def test_cli_runs_headless() -> None:
    assert main([]) == 0


def test_cli_version_flag_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_unknown_command_fails_loudly() -> None:
    assert main(["not-a-command"]) == 2


def test_every_module_imports() -> None:
    for info in pkgutil.walk_packages(magwave.__path__, prefix="magwave."):
        importlib.import_module(info.name)


def test_licence_boundary_no_gpl_imports() -> None:
    """NFR-3: gmsh and Elmer must never be imported — they are subprocess-only tools."""
    forbidden = {"gmsh", "Elmer", "elmer"}
    assert forbidden.isdisjoint(sys.modules), f"GPL module imported: {forbidden & set(sys.modules)}"
    sources = (ROOT / "src").rglob("*.py")
    offenders = [
        path
        for path in sources
        if any(
            f"import {name}" in path.read_text() or f"from {name}" in path.read_text()
            for name in forbidden
        )
    ]
    assert not offenders, f"GPL import found in: {offenders}"
