"""Import manifest data structures (PRD FR-011, ADR-0006).

Pure standard-library dataclasses: no third-party imports are permitted in this module.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SOURCES: tuple[str, ...] = ("step", "iges", "mesh")
DEFECT_KINDS: tuple[str, ...] = (
    "open_shell",
    "non_manifold_edge",
    "skipped_body",
    "merged_body",
    "unnamed_body",
    "imported_as_shell",
    "interference",
)


def _check_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}: expected int, got {type(value).__name__}")
    return value


def _check_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name}: expected float, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name}: must be finite, got {value!r}")
    return out


def _check_bbox(value: Any, name: str) -> tuple[float, float, float, float, float, float]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence) or len(value) != 6:
        raise ValueError(f"{name}: expected a 6-element sequence (x0, y0, z0, x1, y1, z1)")
    coords = tuple(_check_float(item, f"{name}[{i}]") for i, item in enumerate(value))
    return coords  # type: ignore[return-value]


@dataclass
class Body:
    """One imported CAD or mesh body in the manifest."""

    index: int
    name: str | None
    role: str | None
    volume_mm3: float
    solids: int
    bbox: tuple[float, float, float, float, float, float]
    source: str

    def __post_init__(self) -> None:
        self.index = _check_int(self.index, "body.index")
        if self.index < 0:
            raise ValueError(f"body.index: must be non-negative, got {self.index}")
        if self.name is not None:
            if not isinstance(self.name, str):
                raise ValueError(f"body.name: expected str or None, got {type(self.name).__name__}")
            if self.name == "":
                self.name = None
        if self.role is not None and (not isinstance(self.role, str) or not self.role):
            raise ValueError("body.role: expected non-empty str or None")
        self.volume_mm3 = _check_float(self.volume_mm3, "body.volume_mm3")
        self.solids = _check_int(self.solids, "body.solids")
        if self.solids < 0:
            raise ValueError(f"body.solids: must be non-negative, got {self.solids}")
        self.bbox = _check_bbox(self.bbox, "body.bbox")
        if self.source not in SOURCES:
            raise ValueError(f"body.source: expected one of {SOURCES}, got {self.source!r}")

    @property
    def display_name(self) -> str:
        """Presentation fallback name when ``name`` is absent."""
        return self.name if self.name is not None else f"body_{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [float(coord) for coord in self.bbox],
            "index": self.index,
            "name": self.name,
            "role": self.role,
            "solids": self.solids,
            "source": self.source,
            "volume_mm3": float(self.volume_mm3),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Body:
        if not isinstance(data, Mapping):
            raise ValueError(f"body: expected mapping, got {type(data).__name__}")
        required = {"index", "name", "role", "volume_mm3", "solids", "bbox", "source"}
        missing = sorted(required - set(data))
        unknown = sorted(set(data) - required)
        if missing:
            raise ValueError(f"body: missing required field(s): {missing}")
        if unknown:
            raise ValueError(f"body: unknown field(s): {unknown}")
        return cls(
            index=data["index"],
            name=data["name"],
            role=data["role"],
            volume_mm3=data["volume_mm3"],
            solids=data["solids"],
            bbox=data["bbox"],
            source=data["source"],
        )


@dataclass
class Defect:
    """One geometry, naming, or import defect surfaced in the manifest (FR-011)."""

    kind: str
    detail: str
    body_index: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in DEFECT_KINDS:
            raise ValueError(f"defect.kind: expected one of {DEFECT_KINDS}, got {self.kind!r}")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("defect.detail: expected a non-empty string")
        if self.body_index is not None:
            self.body_index = _check_int(self.body_index, "defect.body_index")

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_index": self.body_index,
            "detail": self.detail,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Defect:
        if not isinstance(data, Mapping):
            raise ValueError(f"defect: expected mapping, got {type(data).__name__}")
        required = {"kind", "detail"}
        optional = {"body_index"}
        missing = sorted(required - set(data))
        unknown = sorted(set(data) - required - optional)
        if missing:
            raise ValueError(f"defect: missing required field(s): {missing}")
        if unknown:
            raise ValueError(f"defect: unknown field(s): {unknown}")
        return cls(
            kind=data["kind"],
            detail=data["detail"],
            body_index=data.get("body_index"),
        )


@dataclass
class Manifest:
    """Complete import manifest for a CAD or mesh file (FR-010, FR-011, FR-013)."""

    path: str
    format: str
    units: str
    bodies: list[Body] = field(default_factory=list)
    defects: list[Defect] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("manifest.path: expected a non-empty string")
        if not isinstance(self.format, str) or not self.format:
            raise ValueError("manifest.format: expected a non-empty string")
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("manifest.units: expected a non-empty string")
        self.bodies = list(self.bodies)
        self.defects = list(self.defects)
        self.warnings = [str(item) for item in self.warnings]
        for body in self.bodies:
            if not isinstance(body, Body):
                raise ValueError(f"manifest.bodies: expected Body, got {type(body).__name__}")
        for defect in self.defects:
            if not isinstance(defect, Defect):
                raise ValueError(f"manifest.defects: expected Defect, got {type(defect).__name__}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bodies": [body.to_dict() for body in self.bodies],
            "defects": [defect.to_dict() for defect in self.defects],
            "format": self.format,
            "path": self.path,
            "units": self.units,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        if not isinstance(data, Mapping):
            raise ValueError(f"manifest: expected mapping, got {type(data).__name__}")
        required = {"path", "format", "units", "bodies", "defects", "warnings"}
        missing = sorted(required - set(data))
        unknown = sorted(set(data) - required)
        if missing:
            raise ValueError(f"manifest: missing required field(s): {missing}")
        if unknown:
            raise ValueError(f"manifest: unknown field(s): {unknown}")
        raw_bodies = data["bodies"]
        raw_defects = data["defects"]
        raw_warnings = data["warnings"]
        if not isinstance(raw_bodies, Sequence) or isinstance(raw_bodies, str | bytes):
            raise ValueError("manifest.bodies: expected a sequence")
        if not isinstance(raw_defects, Sequence) or isinstance(raw_defects, str | bytes):
            raise ValueError("manifest.defects: expected a sequence")
        if not isinstance(raw_warnings, Sequence) or isinstance(raw_warnings, str | bytes):
            raise ValueError("manifest.warnings: expected a sequence")
        for i, item in enumerate(raw_warnings):
            if not isinstance(item, str):
                raise ValueError(f"manifest.warnings[{i}]: expected str, got {type(item).__name__}")
        return cls(
            path=data["path"],
            format=data["format"],
            units=data["units"],
            bodies=[Body.from_dict(item) for item in raw_bodies],
            defects=[Defect.from_dict(item) for item in raw_defects],
            warnings=list(raw_warnings),
        )

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise the manifest deterministically as JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        """Deserialise a manifest from JSON text."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest: not valid JSON ({exc})") from exc
        return cls.from_dict(data)
