"""Versioned, provenance-tracked material library (PRD FR-020 / FR-021)."""

from __future__ import annotations

import functools
import importlib.resources
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..units import MU0
from .nonlinear import BhCurve, bh_curve_for, validate_bh_table

SCHEMA_PREFIX = "magwave.materials/"
SCHEMA = f"{SCHEMA_PREFIX}1"

_ENTRY_KEYS = frozenset({"name", "mu_r", "bh_table", "saturation_B", "provenance"})


def _positive_finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}: expected a number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{where}: must be finite and > 0, got {value!r}")
    return out


def _non_negative_finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}: expected a number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{where}: must be finite and >= 0, got {value!r}")
    return out


def _non_empty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: expected a non-empty string")
    return value


def _parse_bh_table(
    raw: Any, where: str, *, name: str | None = None
) -> tuple[tuple[float, float], ...] | None:
    if raw is None:
        return None
    return validate_bh_table(raw, name if name is not None else where)


@dataclass(frozen=True)
class Material:
    """One soft-magnetic material record with linear ``mu_r`` and optional B–H curve."""

    name: str
    mu_r: float
    bh_table: tuple[tuple[float, float], ...] | None
    saturation_B: float | None
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_str(self.name, "material.name"))
        object.__setattr__(self, "mu_r", _positive_finite(self.mu_r, "material.mu_r"))
        object.__setattr__(
            self,
            "bh_table",
            _parse_bh_table(self.bh_table, f"material({self.name}).bh_table", name=self.name),
        )
        if self.saturation_B is not None:
            object.__setattr__(
                self,
                "saturation_B",
                _positive_finite(self.saturation_B, f"material({self.name}).saturation_B"),
            )
        object.__setattr__(
            self,
            "provenance",
            _non_empty_str(self.provenance, f"material({self.name}).provenance"),
        )

    def nu(self) -> float:
        """Linear reluctivity ``1 / (MU0 * mu_r)`` in A*m / (T*m) = m/H."""
        return 1.0 / (MU0 * self.mu_r)

    def bh_curve(self) -> BhCurve:
        """Return the cached monotone :class:`BhCurve` for this material."""
        return bh_curve_for(self)


class MaterialsLibrary:
    """In-memory view over the validated shipped materials library."""

    def __init__(self, version: int, schema: str, materials: Sequence[Material]) -> None:
        self.version = version
        self.schema = schema
        by_name: dict[str, Material] = {}
        for mat in materials:
            if mat.name in by_name:
                raise ValueError(f"duplicate material name {mat.name!r}")
            by_name[mat.name] = mat
        self._materials = by_name

    @classmethod
    def from_dict(cls, data: Any) -> MaterialsLibrary:
        """Parse and validate a materials library mapping using the shipped schema rules."""
        return _parse_library(data)

    def names(self) -> tuple[str, ...]:
        """Sorted tuple of available material names."""
        return tuple(sorted(self._materials))

    def get(self, name: str) -> Material:
        """Look up a material by name, listing available names on failure."""
        if name not in self._materials:
            available = ", ".join(self.names()) or "<none>"
            raise ValueError(f"unknown material {name!r}; shipped materials: {available}")
        return self._materials[name]

    def with_override(self, name: str, mu_r: float | None) -> tuple[Material, str]:
        """Return ``(material, provenance_line)`` applying an optional per-case ``mu_r`` override."""
        base = self.get(name)
        if mu_r is None:
            return base, f"{base.name}: mu_r {base.mu_r:g} (shipped)"
        mu_r_val = _positive_finite(mu_r, f"with_override({name!r}).mu_r")
        overridden = replace(base, mu_r=mu_r_val)
        provenance_line = (
            f"{base.name}: mu_r {base.mu_r:g} (shipped) "
            f"overridden to {overridden.mu_r:g} (case override)"
        )
        return overridden, provenance_line


def _parse_library(data: Any) -> MaterialsLibrary:
    if not isinstance(data, Mapping):
        raise ValueError(f"materials library: expected JSON object, got {type(data).__name__}")
    expected_top = {"schema", "version", "materials"}
    missing = sorted(expected_top - set(data))
    unknown = sorted(set(data) - expected_top)
    if missing:
        raise ValueError(f"materials library: missing required key(s) {missing}")
    if unknown:
        raise ValueError(f"materials library: unknown key(s) {unknown}")

    schema = _non_empty_str(data["schema"], "materials library.schema")
    if not schema.startswith(SCHEMA_PREFIX):
        raise ValueError(
            f"materials library.schema: expected prefix {SCHEMA_PREFIX!r}, got {schema!r}"
        )
    raw_schema_ver = schema[len(SCHEMA_PREFIX) :]
    if not raw_schema_ver.isdigit():
        raise ValueError(f"materials library.schema: invalid version suffix in {schema!r}")
    schema_version = int(raw_schema_ver)

    version = data["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"materials library.version: expected positive integer, got {version!r}")
    if version != schema_version:
        raise ValueError(
            f"materials library version mismatch: version={version} != schema version {schema_version}"
        )

    raw_materials = data["materials"]
    if (
        isinstance(raw_materials, str | bytes)
        or not isinstance(raw_materials, Sequence)
        or not raw_materials
    ):
        raise ValueError("materials library.materials: expected a non-empty list")

    parsed: list[Material] = []
    for idx, entry in enumerate(raw_materials):
        where = f"materials[{idx}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where}: expected JSON object, got {type(entry).__name__}")
        missing_e = sorted(_ENTRY_KEYS - set(entry))
        unknown_e = sorted(set(entry) - _ENTRY_KEYS)
        if missing_e:
            raise ValueError(f"{where}: missing required key(s) {missing_e}")
        if unknown_e:
            raise ValueError(f"{where}: unknown key(s) {unknown_e}")
        parsed.append(
            Material(
                name=entry["name"],
                mu_r=entry["mu_r"],
                bh_table=entry["bh_table"],
                saturation_B=entry["saturation_B"],
                provenance=entry["provenance"],
            )
        )

    return MaterialsLibrary(version=version, schema=schema, materials=parsed)


@functools.lru_cache(maxsize=1)
def load_library() -> MaterialsLibrary:
    """Load, validate, and cache the packaged materials library."""
    raw_text = (
        importlib.resources.files("magwave.materials")
        .joinpath("library.json")
        .read_text(encoding="utf-8")
    )
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"materials library: invalid JSON ({exc})") from exc
    return _parse_library(payload)


LIBRARY_VERSION: int = load_library().version

__all__ = [
    "LIBRARY_VERSION",
    "SCHEMA",
    "BhCurve",
    "Material",
    "MaterialsLibrary",
    "load_library",
]
