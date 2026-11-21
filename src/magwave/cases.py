"""Case model — the unit of reproducibility for MagWave.

A case carries the sources, the probe locations, the expected numbers and the tolerance
each number is allowed, in one document that round-trips through JSON without loss
(PRD FR-001/FR-002). Geometry is metres and amperes throughout (NFR-2); convert at the
boundary with :mod:`magwave.units`.

Why the validation is strict: a case file that silently drops or ignores a field produces
a plausible-looking wrong number, which PRD NFR-8 names as the worst possible outcome. So
unknown keys, missing required fields and widened tolerances are all errors, never
defaults.

JSON schema ``magwave.case/1``
------------------------------
Keys are emitted in sorted order at every level, so a re-serialisation of a loaded case is
byte-identical. Unknown keys are rejected at every level.

    {
      "components": [<component>, ...],      # at least one source
      "description": "<why this case exists>",
      "expected": [<expected>, ...],         # at least one expected value
      "name": "<unique case name>",
      "probes": [<probe>, ...],              # at least one probe
      "schema": "magwave.case/1",
      "sources": ["<provenance>", ...],
      "tolerance": <relative>                # must equal the tightest expected tolerance
      [, "temperature_c": <deg C>]           # optional operating temperature (omitted when null or 20.0)
    }

    <component> — one of five, discriminated by "type":
      magnet:        {"type": "magnet", "kind": "cuboid" | "cylinder" | "halbach_cylinder",
                      "dims": {...metres...}, "br": [bx, by, bz]  # tesla
                      [, "center": [x, y, z], "demag_curve": null | "ref", "name": ""]}
      coil:          {"type": "coil", "kind": "loop" | "solenoid",
                      "dims": {...metres...}, "current": <A>, "turns": <int>
                      [, "center": [x, y, z], "name": ""]}
      dipole:        {"type": "dipole", "moment": [mx, my, mz]  # A m^2
                      [, "center": [x, y, z], "name": ""]}
      permeable:     {"type": "permeable", "kind": "sphere" | "cylinder" | "block",
                      "dims": {...metres...}, "material": "<library name>"
                      [, "center": [x, y, z], "mu_r": <float>, "name": ""]}
      uniform_field: {"type": "uniform_field", "h0": <A/m> [, "name": ""]}

    dims by kind (metres):
      cuboid           {"size_x", "size_y", "size_z"}
      cylinder         {"radius", "length"}
      halbach_cylinder {"inner_radius", "outer_radius"}
      loop             {"radius"}
      solenoid         {"inner_radius", "outer_radius", "length"}
                       # inner == outer is a thin current sheet; inner < outer is a
                       # volume winding with J = turns*current / ((outer-inner)*length)
      sphere           {"radius"}
      block            {"size_x", "size_y", "size_z"}

    <probe>    = {"point": [x, y, z], "quantity": "B" | "F",
                  "component": "x" | "y" | "z" | "magnitude"}
    <expected> = {"check": "<analytic check id>", "quantity": "<label>",
                  "unit": "T" | "N", "value": <number>,
                  "tolerance": <relative>, "source": "<provenance>"}
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from .cad.manifest import Manifest

SCHEMA = "magwave.case/1"

MAGNET_DIMS: dict[str, tuple[str, ...]] = {
    "cuboid": ("size_x", "size_y", "size_z"),
    "cylinder": ("radius", "length"),
    "halbach_cylinder": ("inner_radius", "outer_radius"),
}
COIL_DIMS: dict[str, tuple[str, ...]] = {
    "loop": ("radius",),
    "solenoid": ("inner_radius", "outer_radius", "length"),
}
PERMEABLE_DIMS: dict[str, tuple[str, ...]] = {
    "sphere": ("radius",),
    "cylinder": ("radius", "length"),
    "block": ("size_x", "size_y", "size_z"),
}
PROBE_QUANTITIES = ("B", "F")
PROBE_COMPONENTS = ("x", "y", "z", "magnitude")
QUANTITY_UNITS = {"B": "T", "F": "N"}


class CaseError(ValueError):
    """A case document is malformed, incomplete or has been weakened."""


def _fail(where: str, message: str) -> NoReturn:
    raise CaseError(f"{where}: {message}")


def _sorted(data: Mapping[str, Any]) -> dict[str, Any]:
    return {key: data[key] for key in sorted(data)}


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(where, f"expected a JSON object, got {type(value).__name__}")
    return value


def _keys(
    data: Mapping[str, Any],
    where: str,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> None:
    present = set(data)
    missing = sorted(set(required) - present)
    unknown = sorted(present - set(required) - set(optional))
    if missing:
        _fail(where, f"missing required field(s) {', '.join(map(repr, missing))}")
    if unknown:
        allowed = sorted(set(required) | set(optional))
        _fail(
            where,
            f"unknown key(s) {', '.join(map(repr, unknown))}; allowed keys are {allowed}",
        )


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(where, "expected a non-empty string")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(where, f"expected a number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        _fail(where, "must be finite")
    return out


def _triple(value: Any, where: str) -> tuple[float, float, float]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence) or len(value) != 3:
        _fail(where, "expected a list of three numbers")
    return (
        _number(value[0], f"{where}[0]"),
        _number(value[1], f"{where}[1]"),
        _number(value[2], f"{where}[2]"),
    )


def _dims(value: Any, where: str, required: Sequence[str]) -> dict[str, float]:
    data = _mapping(value, where)
    _keys(data, where, required)
    out: dict[str, float] = {}
    for key in required:
        size = _number(data[key], f"{where}.{key}")
        if size <= 0.0:
            _fail(f"{where}.{key}", "must be positive (dimensions are metres)")
        out[key] = size
    return _sorted(out)


def _name(value: Any, where: str) -> str:
    if value is None:
        return ""
    return _text(value, where)


def _optional_name(value: Any, where: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        _fail(where, f"expected a string, got {type(value).__name__}")
    return value


# --------------------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Magnet:
    """A permanent magnet.

    ``kind`` selects the shape (``cuboid``, ``cylinder`` or ``halbach_cylinder``); ``dims``
    are metres and must match that shape's key set. ``br`` is the remanence vector in
    tesla — magnetisation is ``M = Br/mu0`` in A/m. ``demag_curve`` is an optional
    reference to second-quadrant data (Tier 1), unused by the analytic path.
    """

    kind: str
    dims: Mapping[str, float]
    br: tuple[float, float, float]
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    demag_curve: str | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in MAGNET_DIMS:
            _fail("magnet.kind", f"expected one of {sorted(MAGNET_DIMS)}, got {self.kind!r}")
        dims = _dims(self.dims, "magnet.dims", MAGNET_DIMS[self.kind])
        if self.kind == "halbach_cylinder" and not dims["outer_radius"] > dims["inner_radius"]:
            _fail("magnet.dims", "halbach_cylinder requires outer_radius > inner_radius")
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "br", _triple(self.br, "magnet.br"))
        object.__setattr__(self, "center", _triple(self.center, "magnet.center"))
        if self.demag_curve is not None:
            object.__setattr__(self, "demag_curve", _text(self.demag_curve, "magnet.demag_curve"))
        object.__setattr__(self, "name", _name(self.name, "magnet.name"))

    def to_dict(self) -> dict[str, Any]:
        return _sorted(
            {
                "type": "magnet",
                "name": self.name,
                "kind": self.kind,
                "dims": dict(self.dims),
                "br": list(self.br),
                "center": list(self.center),
                "demag_curve": self.demag_curve,
            }
        )

    @classmethod
    def from_dict(cls, data: Any, where: str = "magnet") -> Magnet:
        doc = _mapping(data, where)
        _keys(
            doc,
            where,
            ("type", "kind", "dims", "br"),
            ("center", "demag_curve", "name"),
        )
        if doc["type"] != "magnet":
            _fail(where, f"expected type 'magnet', got {doc['type']!r}")
        return cls(
            kind=_text(doc["kind"], f"{where}.kind"),
            dims=doc["dims"],
            br=doc["br"],
            center=tuple(doc.get("center", (0.0, 0.0, 0.0))),
            demag_curve=doc.get("demag_curve"),
            name=doc.get("name", ""),
        )


@dataclass(frozen=True)
class Coil:
    """A coil: a single circular loop or a solenoid winding.

    ``dims`` are metres and must match ``kind``. ``current`` is amperes per turn and
    ``turns`` the turn count, so the ampere-turns are ``turns*current``. A solenoid with
    ``inner_radius == outer_radius`` is a thin current sheet; with ``inner < outer`` it is
    a volume winding of current density ``J = turns*current/((outer-inner)*length)``.
    """

    kind: str
    dims: Mapping[str, float]
    current: float
    turns: int = 1
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in COIL_DIMS:
            _fail("coil.kind", f"expected one of {sorted(COIL_DIMS)}, got {self.kind!r}")
        dims = _dims(self.dims, "coil.dims", COIL_DIMS[self.kind])
        if self.kind == "solenoid" and not dims["outer_radius"] >= dims["inner_radius"]:
            _fail("coil.dims", "solenoid requires outer_radius >= inner_radius")
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "current", _number(self.current, "coil.current"))
        if isinstance(self.turns, bool) or not isinstance(self.turns, int):
            _fail("coil.turns", f"expected an integer, got {type(self.turns).__name__}")
        if self.turns < 1:
            _fail("coil.turns", "must be at least 1")
        object.__setattr__(self, "center", _triple(self.center, "coil.center"))
        object.__setattr__(self, "name", _name(self.name, "coil.name"))

    def to_dict(self) -> dict[str, Any]:
        return _sorted(
            {
                "type": "coil",
                "name": self.name,
                "kind": self.kind,
                "dims": dict(self.dims),
                "current": self.current,
                "turns": self.turns,
                "center": list(self.center),
            }
        )

    @classmethod
    def from_dict(cls, data: Any, where: str = "coil") -> Coil:
        doc = _mapping(data, where)
        _keys(doc, where, ("type", "kind", "dims", "current", "turns"), ("center", "name"))
        if doc["type"] != "coil":
            _fail(where, f"expected type 'coil', got {doc['type']!r}")
        return cls(
            kind=_text(doc["kind"], f"{where}.kind"),
            dims=doc["dims"],
            current=doc["current"],
            turns=doc["turns"],
            center=tuple(doc.get("center", (0.0, 0.0, 0.0))),
            name=doc.get("name", ""),
        )


@dataclass(frozen=True)
class Dipole:
    """An ideal point dipole, the far-field model of a uniformly magnetised body.

    ``moment`` is A m^2, ``center`` metres.
    """

    moment: tuple[float, float, float]
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "moment", _triple(self.moment, "dipole.moment"))
        object.__setattr__(self, "center", _triple(self.center, "dipole.center"))
        object.__setattr__(self, "name", _name(self.name, "dipole.name"))

    def to_dict(self) -> dict[str, Any]:
        return _sorted(
            {
                "type": "dipole",
                "name": self.name,
                "moment": list(self.moment),
                "center": list(self.center),
            }
        )

    @classmethod
    def from_dict(cls, data: Any, where: str = "dipole") -> Dipole:
        doc = _mapping(data, where)
        _keys(doc, where, ("type", "moment"), ("center", "name"))
        if doc["type"] != "dipole":
            _fail(where, f"expected type 'dipole', got {doc['type']!r}")
        return cls(
            moment=tuple(doc["moment"]),
            center=tuple(doc.get("center", (0.0, 0.0, 0.0))),
            name=doc.get("name", ""),
        )


@dataclass(frozen=True)
class Permeable:
    """A soft-magnetic permeable body referencing a shipped material and optional ``mu_r``."""

    kind: str
    dims: Mapping[str, float]
    material: str
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mu_r: float | None = None
    name: str = ""

    def __post_init__(self) -> None:
        from .materials import load_library

        if self.kind not in PERMEABLE_DIMS:
            _fail(
                "permeable.kind",
                f"expected one of {sorted(PERMEABLE_DIMS)}, got {self.kind!r}",
            )
        dims = _dims(self.dims, "permeable.dims", PERMEABLE_DIMS[self.kind])
        material = _text(self.material, "permeable.material")
        try:
            load_library().get(material)
        except ValueError as exc:
            _fail("permeable.material", str(exc))
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "center", _triple(self.center, "permeable.center"))
        if self.mu_r is not None:
            mu_r_val = _number(self.mu_r, "permeable.mu_r")
            if mu_r_val <= 0.0:
                _fail("permeable.mu_r", "must be positive (> 0)")
            object.__setattr__(self, "mu_r", mu_r_val)
        object.__setattr__(self, "name", _optional_name(self.name, "permeable.name"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "permeable",
            "name": self.name,
            "kind": self.kind,
            "dims": dict(self.dims),
            "center": list(self.center),
            "material": self.material,
        }
        if self.mu_r is not None:
            payload["mu_r"] = self.mu_r
        return _sorted(payload)

    @classmethod
    def from_dict(cls, data: Any, where: str = "permeable") -> Permeable:
        doc = _mapping(data, where)
        _keys(
            doc,
            where,
            ("type", "kind", "dims", "material"),
            ("center", "mu_r", "name"),
        )
        if doc["type"] != "permeable":
            _fail(where, f"expected type 'permeable', got {doc['type']!r}")
        raw_center = doc.get("center", (0.0, 0.0, 0.0))
        center = tuple(raw_center) if isinstance(raw_center, list) else raw_center
        return cls(
            kind=_text(doc["kind"], f"{where}.kind"),
            dims=doc["dims"],
            material=doc["material"],
            center=center,
            mu_r=doc.get("mu_r"),
            name=doc.get("name", ""),
        )


@dataclass(frozen=True)
class UniformField:
    """A uniform applied magnetic field ``H0 * z_hat`` in free space (``h0`` in A/m)."""

    h0: float
    name: str = ""

    def __post_init__(self) -> None:
        h0_val = _number(self.h0, "uniform_field.h0")
        if h0_val == 0.0:
            _fail("uniform_field.h0", "must be non-zero (A/m)")
        object.__setattr__(self, "h0", h0_val)
        object.__setattr__(self, "name", _optional_name(self.name, "uniform_field.name"))

    def to_dict(self) -> dict[str, Any]:
        return _sorted({"type": "uniform_field", "name": self.name, "h0": self.h0})

    @classmethod
    def from_dict(cls, data: Any, where: str = "uniform_field") -> UniformField:
        doc = _mapping(data, where)
        _keys(doc, where, ("type", "h0"), ("name",))
        if doc["type"] != "uniform_field":
            _fail(where, f"expected type 'uniform_field', got {doc['type']!r}")
        return cls(h0=doc["h0"], name=doc.get("name", ""))


Component = Magnet | Coil | Dipole | Permeable | UniformField

COMPONENT_TYPES: dict[str, Any] = {
    "magnet": Magnet,
    "coil": Coil,
    "dipole": Dipole,
    "permeable": Permeable,
    "uniform_field": UniformField,
}


def component_from_dict(data: Any, where: str = "component") -> Component:
    """Build a component from its JSON object, rejecting an unknown ``type``."""
    doc = _mapping(data, where)
    kind = doc.get("type")
    if kind not in COMPONENT_TYPES:
        _fail(where, f"unknown component type {kind!r}; expected one of {sorted(COMPONENT_TYPES)}")
    return COMPONENT_TYPES[kind].from_dict(doc, where)


# --------------------------------------------------------------------------------------
# probes and expected values
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Probe:
    """Where a quantity is evaluated.

    ``point`` is metres. ``quantity`` is ``B`` (tesla) or ``F`` (newton); ``component``
    selects the Cartesian component or the vector magnitude.
    """

    point: tuple[float, float, float]
    quantity: str
    component: str = "magnitude"

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _triple(self.point, "probe.point"))
        if self.quantity not in PROBE_QUANTITIES:
            _fail(
                "probe.quantity", f"expected one of {list(PROBE_QUANTITIES)}, got {self.quantity!r}"
            )
        if self.component not in PROBE_COMPONENTS:
            _fail(
                "probe.component",
                f"expected one of {list(PROBE_COMPONENTS)}, got {self.component!r}",
            )

    def to_dict(self) -> dict[str, Any]:
        return _sorted(
            {
                "point": list(self.point),
                "quantity": self.quantity,
                "component": self.component,
            }
        )

    @classmethod
    def from_dict(cls, data: Any, where: str = "probe") -> Probe:
        doc = _mapping(data, where)
        _keys(doc, where, ("point", "quantity"), ("component",))
        return cls(
            point=tuple(doc["point"]),
            quantity=_text(doc["quantity"], f"{where}.quantity"),
            component=doc.get("component", "magnitude"),
        )


@dataclass(frozen=True)
class ExpectedValue:
    """One expected number, with the check that produces it and the tolerance it earns.

    ``tolerance`` is **research-owned** (docs/research/validation-sources.md): it encodes
    the verified quality of the formula, not the convenience of the implementation. It may
    be tightened, never widened.
    """

    check: str
    quantity: str
    value: float
    unit: str
    tolerance: float
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "check", _text(self.check, "expected.check"))
        object.__setattr__(self, "quantity", _text(self.quantity, "expected.quantity"))
        object.__setattr__(self, "value", _number(self.value, "expected.value"))
        unit = _text(self.unit, "expected.unit")
        if unit not in QUANTITY_UNITS.values():
            _fail(
                "expected.unit",
                f"expected one of {sorted(set(QUANTITY_UNITS.values()))}, got {unit!r}",
            )
        object.__setattr__(self, "unit", unit)
        tolerance = _number(self.tolerance, "expected.tolerance")
        if tolerance <= 0.0:
            _fail("expected.tolerance", "must be positive (relative tolerances only)")
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "source", _text(self.source, "expected.source"))

    def to_dict(self) -> dict[str, Any]:
        return _sorted(
            {
                "check": self.check,
                "quantity": self.quantity,
                "value": self.value,
                "unit": self.unit,
                "tolerance": self.tolerance,
                "source": self.source,
            }
        )

    @classmethod
    def from_dict(cls, data: Any, where: str = "expected") -> ExpectedValue:
        doc = _mapping(data, where)
        _keys(doc, where, ("check", "quantity", "value", "unit", "tolerance", "source"))
        return cls(
            check=doc["check"],
            quantity=doc["quantity"],
            value=doc["value"],
            unit=doc["unit"],
            tolerance=doc["tolerance"],
            source=doc["source"],
        )


# --------------------------------------------------------------------------------------
# the case
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    """A complete, reproducible benchmark or user case.

    ``tolerance`` is the case gate. It is derived from the expected values and may not be
    stated looser than the tightest one: a case cannot widen a research-owned tolerance to
    make itself pass.
    """

    name: str
    description: str
    sources: tuple[str, ...]
    components: tuple[Component, ...]
    probes: tuple[Probe, ...]
    expected: tuple[ExpectedValue, ...]
    tolerance: float | None = None
    manifest: Manifest | None = None
    temperature_c: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "case.name"))
        object.__setattr__(self, "description", _text(self.description, "case.description"))

        if isinstance(self.sources, str) or not isinstance(self.sources, Sequence):
            _fail("case.sources", "expected a list of strings")
        sources = tuple(_text(item, f"case.sources[{i}]") for i, item in enumerate(self.sources))
        if not sources:
            _fail("case.sources", "at least one provenance string is required")
        object.__setattr__(self, "sources", sources)

        for field_name in ("components", "probes", "expected"):
            if isinstance(getattr(self, field_name), str | bytes) or not isinstance(
                getattr(self, field_name), Sequence
            ):
                _fail(f"case.{field_name}", "expected a list")
        components = tuple(self.components)
        probes = tuple(self.probes)
        expected = tuple(self.expected)
        if not components:
            _fail("case.components", "at least one source is required")
        if not probes:
            _fail("case.probes", "at least one probe is required")
        if not expected:
            _fail("case.expected", "at least one expected value is required")
        for component in components:
            if not isinstance(component, Magnet | Coil | Dipole | Permeable | UniformField):
                _fail("case.components", f"unsupported component {type(component).__name__}")
        if sum(isinstance(component, UniformField) for component in components) > 1:
            _fail("case.components", "at most one uniform_field component is allowed")
        for probe in probes:
            if not isinstance(probe, Probe):
                _fail("case.probes", f"unsupported probe {type(probe).__name__}")
        for item in expected:
            if not isinstance(item, ExpectedValue):
                _fail("case.expected", f"unsupported expected value {type(item).__name__}")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "probes", probes)
        object.__setattr__(self, "expected", expected)

        tightest = min(item.tolerance for item in expected)
        if self.tolerance is None:
            object.__setattr__(self, "tolerance", tightest)
        else:
            stated = _number(self.tolerance, "case.tolerance")
            if stated != tightest:
                _fail(
                    "case.tolerance",
                    f"{stated!r} must equal the tightest expected-value tolerance {tightest!r}; "
                    "tolerances are research-owned and may not be widened",
                )
            object.__setattr__(self, "tolerance", stated)

        names = [component.name for component in components if component.name]
        if len(names) != len(set(names)):
            _fail("case.components", "component names must be unique within a case")

        if self.manifest is not None and not isinstance(self.manifest, Manifest):
            _fail(
                "case.manifest",
                f"expected Manifest or None, got {type(self.manifest).__name__}",
            )

        if self.temperature_c is not None:
            temp = _number(self.temperature_c, "case.temperature_c")
            if temp < -273.15:
                _fail(
                    "case.temperature_c",
                    f"must be >= -273.15 deg C (absolute zero), got {temp!r}",
                )
            object.__setattr__(self, "temperature_c", temp)

    def to_dict(self) -> dict[str, Any]:
        """Return the case as a JSON-ready dict with keys sorted at every level."""
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "name": self.name,
            "description": self.description,
            "sources": list(self.sources),
            "tolerance": self.tolerance,
            "components": [component.to_dict() for component in self.components],
            "probes": [probe.to_dict() for probe in self.probes],
            "expected": [item.to_dict() for item in self.expected],
        }
        if self.manifest is not None:
            payload["manifest"] = self.manifest.to_dict()
        if self.temperature_c is not None and self.temperature_c != 20.0:
            payload["temperature_c"] = self.temperature_c
        return _sorted(payload)

    @classmethod
    def from_dict(cls, data: Any) -> Case:
        """Build a case from its JSON object; reject unknown keys and missing fields."""
        doc = _mapping(data, "case")
        _keys(
            doc,
            "case",
            ("schema", "name", "description", "sources", "components", "probes", "expected"),
            ("tolerance", "manifest", "temperature_c"),
        )
        schema = _text(doc["schema"], "case.schema")
        if schema != SCHEMA:
            _fail("case.schema", f"expected {SCHEMA!r}, got {schema!r}")
        components = doc["components"]
        probes = doc["probes"]
        expected = doc["expected"]
        if not isinstance(components, Sequence) or isinstance(components, str | bytes):
            _fail("case.components", "expected a list")
        if not isinstance(probes, Sequence) or isinstance(probes, str | bytes):
            _fail("case.probes", "expected a list")
        if not isinstance(expected, Sequence) or isinstance(expected, str | bytes):
            _fail("case.expected", "expected a list")
        raw_manifest = doc.get("manifest")
        manifest: Manifest | None = None
        if raw_manifest is not None:
            try:
                manifest = Manifest.from_dict(raw_manifest)
            except ValueError as exc:
                _fail("case.manifest", str(exc))
        return cls(
            name=doc["name"],
            description=doc["description"],
            sources=tuple(doc["sources"]),
            components=tuple(
                component_from_dict(item, f"case.components[{i}]")
                for i, item in enumerate(components)
            ),
            probes=tuple(
                Probe.from_dict(item, f"case.probes[{i}]") for i, item in enumerate(probes)
            ),
            expected=tuple(
                ExpectedValue.from_dict(item, f"case.expected[{i}]")
                for i, item in enumerate(expected)
            ),
            tolerance=doc.get("tolerance"),
            manifest=manifest,
            temperature_c=doc.get("temperature_c"),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialise with sorted keys — the on-disk form is deterministic."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Case:
        """Parse a case from JSON text, failing loudly on anything unexpected."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - message shaping only
            raise CaseError(f"case: not valid JSON ({exc})") from exc
        return cls.from_dict(data)
