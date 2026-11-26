"""Deterministic results artefact writer with provenance (PRD FR-002).

Writes ``<case-stem>.results.json`` (schema ``magwave.results/1``) and
``<case-stem>.results.csv`` with provenance metadata (app version, engine id and
version, case name, file basename, and SHA-256 hex digest of the case bytes).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .cases import Case
from .engines.base import Capabilities, Result, Sample
from .report import _units_summary

RESULTS_SCHEMA = "magwave.results/1"


def results_paths(
    case_path: str | Path,
    results_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return ``(json_path, csv_path)`` for ``case_path`` in ``results_dir`` or beside it."""
    source = Path(case_path)
    target_dir = source.parent if results_dir is None else Path(results_dir)
    return (
        target_dir / f"{source.stem}.results.json",
        target_dir / f"{source.stem}.results.csv",
    )


def write_results(
    case_path: str | Path,
    case: Case,
    capabilities: Capabilities,
    samples: Sequence[Sample],
    *,
    results_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write deterministic JSON and CSV results artefacts with provenance.

    Returns ``(json_path, csv_path)``.
    """
    source = Path(case_path)
    json_path, csv_path = results_paths(source, results_dir)
    case_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    sample_tuple = tuple(samples)

    units = _units_summary(
        Result(
            engine_id=capabilities.engine_id,
            engine_version=capabilities.engine_version,
            rung=capabilities.rung,
            method="",
            samples=sample_tuple,
        )
    )

    payload: dict[str, Any] = {
        "app_version": __version__,
        "case": {
            "file": source.name,
            "name": case.name,
            "sha256": case_sha256,
        },
        "engine": {
            "id": capabilities.engine_id,
            "version": capabilities.engine_version,
        },
        "samples": [
            {
                "B": [float(coord) for coord in sample.vector],
                "Bmag": float(sample.value("magnitude")),
                "point": [float(coord) for coord in sample.point],
            }
            for sample in sample_tuple
        ],
        "schema": RESULTS_SCHEMA,
        "units": units,
    }
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    csv_lines = [
        f"# app_version: {__version__}",
        f"# engine: {capabilities.engine_id}",
        f"# engine_version: {capabilities.engine_version}",
        f"# case: {case.name}",
        f"# case_sha256: {case_sha256}",
        "x_m,y_m,z_m,Bx_T,By_T,Bz_T,Bmag_T",
    ]
    for sample in sample_tuple:
        values = [*sample.point, *sample.vector, sample.value("magnitude")]
        csv_lines.append(",".join(repr(float(value)) for value in values))
    csv_text = "\n".join(csv_lines) + "\n"

    json_path.write_text(json_text, encoding="utf-8", newline="\n")
    csv_path.write_text(csv_text, encoding="utf-8", newline="\n")
    return json_path, csv_path
