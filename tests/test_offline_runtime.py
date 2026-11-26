"""Offline runtime guard tests (NFR-6: no outbound network calls at runtime).

These tests execute the CLI entry point ``magwave.cli.main([...])`` in-process under a
``monkeypatch`` on ``socket`` so any outbound socket creation or DNS lookup immediately fails with
an explicit network-guard error. Running in-process rather than in a subprocess is essential here: a
subprocess spawns a fresh Python interpreter that would escape the ``monkeypatch`` and prove
nothing.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import NoReturn

import pytest

from magwave.cli import main

ROOT = Path(__file__).resolve().parents[1]
DIPOLE_EXAMPLE = ROOT / "examples" / "solve-dipole.json"
NETWORK_ERROR_MESSAGE = "network call attempted"


class _NetworkCallBlockedError(AssertionError):
    """Raised by the NFR-6 test guard if any runtime path touches sockets or DNS."""


def _forbid_network(*_args: object, **_kwargs: object) -> NoReturn:
    raise _NetworkCallBlockedError(NETWORK_ERROR_MESSAGE)


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "socket", _forbid_network)
    monkeypatch.setattr(socket, "create_connection", _forbid_network)
    monkeypatch.setattr(socket, "getaddrinfo", _forbid_network)


def test_network_guard_trips_on_socket_and_dns_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_network(monkeypatch)

    with pytest.raises(_NetworkCallBlockedError, match=f"^{NETWORK_ERROR_MESSAGE}$"):
        socket.socket()

    with pytest.raises(_NetworkCallBlockedError, match=f"^{NETWORK_ERROR_MESSAGE}$"):
        socket.create_connection(("example.com", 80))

    with pytest.raises(_NetworkCallBlockedError, match=f"^{NETWORK_ERROR_MESSAGE}$"):
        socket.getaddrinfo("example.com", 80)


def test_cli_runtime_paths_make_no_network_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run_results"
    run_dir.mkdir()
    solve_dir = tmp_path / "solve_results"
    solve_dir.mkdir()

    _block_network(monkeypatch)

    rc_validate = main(["validate", str(DIPOLE_EXAMPLE)])
    assert rc_validate == 0, f"validate returned {rc_validate}"
    validate_cap = capsys.readouterr()
    assert "valid: true" in validate_cap.out

    rc_run = main(["run", str(DIPOLE_EXAMPLE), "--results-dir", str(run_dir)])
    assert rc_run == 0, f"run returned {rc_run}"
    _ = capsys.readouterr()
    assert (run_dir / "solve-dipole.results.json").is_file()
    assert (run_dir / "solve-dipole.results.csv").is_file()

    rc_solve = main(
        [
            "solve",
            str(DIPOLE_EXAMPLE),
            "--points",
            "0,0,0.5",
            "--format",
            "json",
            "--results-dir",
            str(solve_dir),
        ]
    )
    assert rc_solve == 0, f"solve returned {rc_solve}"
    solve_cap = capsys.readouterr()
    payload = json.loads(solve_cap.out)
    assert len(payload["samples"]) == 1
    assert (solve_dir / "solve-dipole.results.json").is_file()
