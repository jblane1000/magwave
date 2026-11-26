"""Case model — the unit of reproducibility for MagWave.

A case carries the sources, the probe locations, the expected numbers, the tolerance
each number is allowed, the declared unit system, and the reference frame in one
document that round-trips through JSON without loss (PRD FR-001/FR-002/FR-003).
Internally, the model stores SI units throughout (metres, amperes, tesla, newtons;
NFR-2) and converts at the JSON boundary.

Why the validation is strict: a case file that silently drops, ignores, or guesses a
unit or axis produces a plausible-looking wrong number, which PRD NFR-8 names as the
worst possible outcome. So unknown keys, missing required fields, mixed units,
unsupported units, unsupported symmetry axes, and widened tolerances are all errors,
never defaults.

JSON schema ``magwave.case/2``
------------------------------
Keys are emitted in sorted order at every level, so a re-serialisation of a loaded case is
byte-identical. Unknown keys are rejected at every level.

    {
      "components": [<component>, ...],      # at least one source
      "description": "<why this case exists>",
      "expected": [<expected>, ...],         # at least one expected value
      "frame": {"axis": "z"[, "datum_face": "<optional text>"]},
      "name": "<unique case name>",
      "probes": [<probe>, ...],              # at least one probe
      "schema": "magwave.case/2",
      "sources": ["<provenance>", ...],
      "tolerance": <relative>,               # must equal the tightest expected tolerance
      "units": {"current": "A", "field": "T" | "mT", "force": "N" | "mN", "length": "m" | "mm"}
      [, "mesh": {"element_size": <units.length>}] # optional FEM element size
      [, "temperature_c": <deg C>]           # optional operating temperature (omitted when null or 20.0)
    }

Boundary scaling (``units``) applies to:
  * ``units.length`` (``m``, ``mm``): component ``dims`` and ``center``, probe ``point``,
    and ``mesh.element_size``.
  * ``units.field`` (``T``, ``mT``): magnet ``br`` and ``expected[].value`` for ``B``
    quantities.
  * ``units.force`` (``N``, ``mN``): ``expected[].value`` for ``F`` quantities.
  * ``units.current`` (``A`` only): ``coil.current`` (in A).
  * Not scaled (always SI, no declared unit key): ``uniform_field.h0`` (A/m) and
    ``dipole.moment`` (A m^2).

    <component> — one of five, discriminated by "type":
      magnet:        {"type": "magnet", "kind": "cuboid" | "cylinder" | "halbach_cylinder",
                      "dims": {...units.length...}, "br": [bx, by, bz]  # units.field
                      [, "center": [x, y, z], "demag_curve": null | "ref", "name": ""]}
      coil:          {"type": "coil", "kind": "loop" | "solenoid",
                      "dims": {...units.length...}, "current": <A>, "turns": <int>
                      [, "center": [x, y, z], "name": ""]}
      dipole:        {"type": "dipole", "moment": [mx, my, mz]  # A m^2
                      [, "center": [x, y, z], "name": ""]}
      permeable:     {"type": "permeable", "kind": "sphere" | "cylinder" | "block",
                      "dims": {...units.length...}, "material": "<library name>"
                      [, "center": [x, y, z], "mu_r": <float>, "name": ""]}
      uniform_field: {"type": "uniform_field", "h0": <A/m> [, "name": ""]}

    dims by kind (in ``units.length``):
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
                  "unit": "T" | "mT" | "N" | "mN", "value": <number>,
                  "tolerance": <relative>, "source": "<provenance>"}
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, NoReturn

from .cad.manifest import Manifest

SCHEMA = "magwave.case/2"

SUPPORTED_LENGTH_UNITS: dict[str, int] = {"m": 0, "mm": -3}
SUPPORTED_FIELD_UNITS: dict[str, int] = {"T": 0, "mT": -3}
SUPPORTED_CURRENT_UNITS: dict[str, int] = {"A": 0}
SUPPORTED_FORCE_UNITS: dict[str, int] = {"N": 0, "mN": -3}
SUPPORTED_EXPECTED_UNITS: tuple[str, ...] = ("N", "T", "mN", "mT")

FRAME_AXES: tuple[str, ...] = ("x", "y", "z")
SUPPORTED_FRAME_AXES: tuple[str, ...] = ("z",)

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


def _positive_finite_size(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        _fail(where, f"must be a finite number > 0, got {value!r}")
    return float(value)


def _triple(value: Any, where: str) -> tuple[float, float, float]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence) or len(value) != 3:
        _fail(where, "expected a list of three numbers")
    return (
        _number(value[0], f"{where}[0]"),
        _number(value[1], f"{where}[1]"),
        _number(value[2], f"{where}[2]"),
    )


def _scale_power10(value: float, exp10: int) -> float:
    if exp10 == 0:
        return value
    return float(Decimal(str(value)).scaleb(exp10))


def _scale_triple(triple: tuple[float, float, float], exp10: int) -> tuple[float, float, float]:
    if exp10 == 0:
        return triple
    return (
        _scale_power10(triple[0], exp10),
        _scale_power10(triple[1], exp10),
        _scale_power10(triple[2], exp10),
    )


def _scale_dims(dims: Mapping[str, float], exp10: int) -> dict[str, float]:
    if exp10 == 0:
        return dict(dims)
    return {key: _scale_power10(value, exp10) for key, value in dims.items()}


def _dims(
    value: Any,
    where: str,
    required: Sequence[str],
    *,
    unit_label: str = "metres",
) -> dict[str, float]:
    data = _mapping(value, where)
    _keys(data, where, required)
    out: dict[str, float] = {}
    for key in required:
        size = _number(data[key], f"{where}.{key}")
        if size <= 0.0:
            _fail(f"{where}.{key}", f"must be positive (dimensions are {unit_label})")
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


def _validate_unit(value: Any, where: str, supported: Sequence[str] | Mapping[str, int]) -> str:
    unit = _text(value, where)
    if unit not in supported:
        _fail(where, f"unsupported unit {unit!r} (supported: {', '.join(supported)})")
    return unit


def _validate_frame_axis(value: Any, where: str) -> str:
    axis = _text(value, where)
    if axis not in FRAME_AXES:
        _fail(where, f"expected one of {list(FRAME_AXES)}, got {axis!r}")
    if axis not in SUPPORTED_FRAME_AXES:
        _fail(
            where,
            f"axis {axis!r} is not supported; only 'z' is supported at Tier 1 "
            "(rotation is not implemented)",
        )
    return axis


# --------------------------------------------------------------------------------------
# units and frame declarations
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class UnitSpec:
    """Declared boundary unit system for a case document."""

    length: str = "m"
    field: str = "T"
    current: str = "A"
    force: str = "N"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "length", _validate_unit(self.length, "case.units.length", SUPPORTED_LENGTH_UNITS)
        )
        object.__setattr__(
            self, "field", _validate_unit(self.field, "case.units.field", SUPPORTED_FIELD_UNITS)
        )
        object.__setattr__(
            self,
            "current",
            _validate_unit(self.current, "case.units.current", SUPPORTED_CURRENT_UNITS),
        )
        object.__setattr__(
            self, "force", _validate_unit(self.force, "case.units.force", SUPPORTED_FORCE_UNITS)
        )

    def to_dict(self) -> dict[str, Any]:
        return _sorted(
            {
                "length": self.length,
                "field": self.field,
                "current": self.current,
                "force": self.force,
            }
        )

    @classmethod
    def from_dict(cls, data: Any, where: str = "case.units") -> UnitSpec:
        doc = _mapping(data, where)
        _keys(doc, where, ("length", "field", "current", "force"))
        return cls(
            length=_validate_unit(doc["length"], f"{where}.length", SUPPORTED_LENGTH_UNITS),
            field=_validate_unit(doc["field"], f"{where}.field", SUPPORTED_FIELD_UNITS),
            current=_validate_unit(doc["current"], f"{where}.current", SUPPORTED_CURRENT_UNITS),
            force=_validate_unit(doc["force"], f"{where}.force", SUPPORTED_FORCE_UNITS),
        )


@dataclass(frozen=True)
class FrameSpec:
    """Declared coordinate frame and rotational symmetry axis for a case document."""

    axis: str = "z"
    datum_face: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", _validate_frame_axis(self.axis, "case.frame.axis"))
        object.__setattr__(
            self, "datum_face", _optional_name(self.datum_face, "case.frame.datum_face")
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"axis": self.axis}
        if self.datum_face:
            payload["datum_face"] = self.datum_face
        return _sorted(payload)

    @classmethod
    def from_dict(cls, data: Any, where: str = "case.frame") -> FrameSpec:
        doc = _mapping(data, where)
        _keys(doc, where, ("axis",), ("datum_face",))
        return cls(
            axis=_validate_frame_axis(doc["axis"], f"{where}.axis"),
            datum_face=_optional_name(doc.get("datum_face", ""), f"{where}.datum_face"),
        )


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

    def to_dict(self, *, length_exp10: int = 0, field_exp10: int = 0) -> dict[str, Any]:
        return _sorted(
            {
                "type": "magnet",
                "name": self.name,
                "kind": self.kind,
                "dims": _scale_dims(self.dims, length_exp10),
                "br": list(_scale_triple(self.br, field_exp10)),
                "center": list(_scale_triple(self.center, length_exp10)),
                "demag_curve": self.demag_curve,
            }
        )

    @classmethod
    def from_dict(
        cls,
        data: Any,
        where: str = "magnet",
        *,
        length_exp10: int = 0,
        field_exp10: int = 0,
        length_unit: str = "metres",
    ) -> Magnet:
        doc = _mapping(data, where)
        _keys(
            doc,
            where,
            ("type", "kind", "dims", "br"),
            ("center", "demag_curve", "name"),
        )
        if doc["type"] != "magnet":
            _fail(where, f"expected type 'magnet', got {doc['type']!r}")
        kind = _text(doc["kind"], f"{where}.kind")
        if kind not in MAGNET_DIMS:
            _fail(f"{where}.kind", f"expected one of {sorted(MAGNET_DIMS)}, got {kind!r}")
        dims = _dims(doc["dims"], f"{where}.dims", MAGNET_DIMS[kind], unit_label=length_unit)
        br = _triple(doc["br"], f"{where}.br")
        center = _triple(doc.get("center", (0.0, 0.0, 0.0)), f"{where}.center")
        return cls(
            kind=kind,
            dims=_scale_dims(dims, length_exp10),
            br=_scale_triple(br, field_exp10),
            center=_scale_triple(center, length_exp10),
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

    def to_dict(self, *, length_exp10: int = 0) -> dict[str, Any]:
        return _sorted(
            {
                "type": "coil",
                "name": self.name,
                "kind": self.kind,
                "dims": _scale_dims(self.dims, length_exp10),
                "current": self.current,
                "turns": self.turns,
                "center": list(_scale_triple(self.center, length_exp10)),
            }
        )

    @classmethod
    def from_dict(
        cls,
        data: Any,
        where: str = "coil",
        *,
        length_exp10: int = 0,
        length_unit: str = "metres",
    ) -> Coil:
        doc = _mapping(data, where)
        _keys(doc, where, ("type", "kind", "dims", "current", "turns"), ("center", "name"))
        if doc["type"] != "coil":
            _fail(where, f"expected type 'coil', got {doc['type']!r}")
        kind = _text(doc["kind"], f"{where}.kind")
        if kind not in COIL_DIMS:
            _fail(f"{where}.kind", f"expected one of {sorted(COIL_DIMS)}, got {kind!r}")
        dims = _dims(doc["dims"], f"{where}.dims", COIL_DIMS[kind], unit_label=length_unit)
        center = _triple(doc.get("center", (0.0, 0.0, 0.0)), f"{where}.center")
        return cls(
            kind=kind,
            dims=_scale_dims(dims, length_exp10),
            current=doc["current"],
            turns=doc["turns"],
            center=_scale_triple(center, length_exp10),
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

    def to_dict(self, *, length_exp10: int = 0) -> dict[str, Any]:
        return _sorted(
            {
                "type": "dipole",
                "name": self.name,
                "moment": list(self.moment),
                "center": list(_scale_triple(self.center, length_exp10)),
            }
        )

    @classmethod
    def from_dict(
        cls,
        data: Any,
        where: str = "dipole",
        *,
        length_exp10: int = 0,
    ) -> Dipole:
        doc = _mapping(data, where)
        _keys(doc, where, ("type", "moment"), ("center", "name"))
        if doc["type"] != "dipole":
            _fail(where, f"expected type 'dipole', got {doc['type']!r}")
        moment = _triple(doc["moment"], f"{where}.moment")
        center = _triple(doc.get("center", (0.0, 0.0, 0.0)), f"{where}.center")
        return cls(
            moment=moment,
            center=_scale_triple(center, length_exp10),
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

    def to_dict(self, *, length_exp10: int = 0) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "permeable",
            "name": self.name,
            "kind": self.kind,
            "dims": _scale_dims(self.dims, length_exp10),
            "center": list(_scale_triple(self.center, length_exp10)),
            "material": self.material,
        }
        if self.mu_r is not None:
            payload["mu_r"] = self.mu_r
        return _sorted(payload)

    @classmethod
    def from_dict(
        cls,
        data: Any,
        where: str = "permeable",
        *,
        length_exp10: int = 0,
        length_unit: str = "metres",
    ) -> Permeable:
        doc = _mapping(data, where)
        _keys(
            doc,
            where,
            ("type", "kind", "dims", "material"),
            ("center", "mu_r", "name"),
        )
        if doc["type"] != "permeable":
            _fail(where, f"expected type 'permeable', got {doc['type']!r}")
        kind = _text(doc["kind"], f"{where}.kind")
        if kind not in PERMEABLE_DIMS:
            _fail(
                f"{where}.kind",
                f"expected one of {sorted(PERMEABLE_DIMS)}, got {kind!r}",
            )
        dims = _dims(doc["dims"], f"{where}.dims", PERMEABLE_DIMS[kind], unit_label=length_unit)
        center = _triple(doc.get("center", (0.0, 0.0, 0.0)), f"{where}.center")
        return cls(
            kind=kind,
            dims=_scale_dims(dims, length_exp10),
            material=doc["material"],
            center=_scale_triple(center, length_exp10),
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


def component_from_dict(
    data: Any,
    where: str = "component",
    *,
    length_exp10: int = 0,
    field_exp10: int = 0,
    length_unit: str = "metres",
) -> Component:
    """Build a component from its JSON object, rejecting an unknown ``type``."""
    doc = _mapping(data, where)
    kind = doc.get("type")
    if kind not in COMPONENT_TYPES:
        _fail(where, f"unknown component type {kind!r}; expected one of {sorted(COMPONENT_TYPES)}")
    if kind == "magnet":
        return Magnet.from_dict(
            doc,
            where,
            length_exp10=length_exp10,
            field_exp10=field_exp10,
            length_unit=length_unit,
        )
    if kind == "coil":
        return Coil.from_dict(doc, where, length_exp10=length_exp10, length_unit=length_unit)
    if kind == "permeable":
        return Permeable.from_dict(doc, where, length_exp10=length_exp10, length_unit=length_unit)
    if kind == "dipole":
        return Dipole.from_dict(doc, where, length_exp10=length_exp10)
    return UniformField.from_dict(doc, where)


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

    def to_dict(self, *, length_exp10: int = 0) -> dict[str, Any]:
        return _sorted(
            {
                "point": list(_scale_triple(self.point, length_exp10)),
                "quantity": self.quantity,
                "component": self.component,
            }
        )

    @classmethod
    def from_dict(
        cls,
        data: Any,
        where: str = "probe",
        *,
        length_exp10: int = 0,
    ) -> Probe:
        doc = _mapping(data, where)
        _keys(doc, where, ("point", "quantity"), ("component",))
        point = _triple(doc["point"], f"{where}.point")
        return cls(
            point=_scale_triple(point, length_exp10),
            quantity=_text(doc["quantity"], f"{where}.quantity"),
            component=doc.get("component", "magnitude"),
        )


def _expected_quantity_kind(quantity: str) -> str | None:
    stripped = quantity.strip("|")
    if stripped.startswith("B"):
        return "field"
    if stripped.startswith("F"):
        return "force"
    return None


def _validate_expected_unit(item: ExpectedValue, index: int, units: UnitSpec) -> str:
    kind = _expected_quantity_kind(item.quantity)
    if kind is None:
        kind = "field" if item.unit in SUPPORTED_FIELD_UNITS else "force"
    if kind == "field":
        if item.unit != units.field:
            _fail(
                f"case.expected[{index}].unit",
                f"{item.unit!r} contradicts case.units.field = {units.field!r}",
            )
        return "field"
    if item.unit != units.force:
        _fail(
            f"case.expected[{index}].unit",
            f"{item.unit!r} contradicts case.units.force = {units.force!r}",
        )
    return "force"


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
        unit = _validate_unit(self.unit, "expected.unit", SUPPORTED_EXPECTED_UNITS)
        object.__setattr__(self, "unit", unit)
        tolerance = _number(self.tolerance, "expected.tolerance")
        if tolerance <= 0.0:
            _fail("expected.tolerance", "must be positive (relative tolerances only)")
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "source", _text(self.source, "expected.source"))

    def to_dict(self, *, value_exp10: int = 0) -> dict[str, Any]:
        return _sorted(
            {
                "check": self.check,
                "quantity": self.quantity,
                "value": _scale_power10(self.value, value_exp10),
                "unit": self.unit,
                "tolerance": self.tolerance,
                "source": self.source,
            }
        )

    @classmethod
    def from_dict(
        cls,
        data: Any,
        where: str = "expected",
        *,
        value_exp10: int = 0,
    ) -> ExpectedValue:
        doc = _mapping(data, where)
        _keys(doc, where, ("check", "quantity", "value", "unit", "tolerance", "source"))
        unit = _validate_unit(doc["unit"], f"{where}.unit", SUPPORTED_EXPECTED_UNITS)
        value = _scale_power10(_number(doc["value"], f"{where}.value"), value_exp10)
        return cls(
            check=doc["check"],
            quantity=doc["quantity"],
            value=value,
            unit=unit,
            tolerance=doc["tolerance"],
            source=doc["source"],
        )


# --------------------------------------------------------------------------------------
# mesh control and the case
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MeshSpec:
    """Per-case FEM mesh density control (metres)."""

    element_size: float

    def __post_init__(self) -> None:
        size = _positive_finite_size(self.element_size, "mesh.element_size")
        object.__setattr__(self, "element_size", size)

    def to_dict(self, *, length_exp10: int = 0) -> dict[str, Any]:
        return {"element_size": _scale_power10(self.element_size, length_exp10)}

    @classmethod
    def from_dict(
        cls,
        data: Any,
        where: str = "mesh",
        *,
        length_exp10: int = 0,
    ) -> MeshSpec:
        doc = _mapping(data, where)
        _keys(doc, where, ("element_size",))
        size = _positive_finite_size(doc["element_size"], f"{where}.element_size")
        return cls(element_size=_scale_power10(size, length_exp10))


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
    mesh: MeshSpec | None = None
    units: UnitSpec = field(default_factory=UnitSpec)
    frame: FrameSpec = field(default_factory=FrameSpec)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "case.name"))
        object.__setattr__(self, "description", _text(self.description, "case.description"))

        if not isinstance(self.units, UnitSpec):
            _fail("case.units", f"expected UnitSpec, got {type(self.units).__name__}")
        if not isinstance(self.frame, FrameSpec):
            _fail("case.frame", f"expected FrameSpec, got {type(self.frame).__name__}")

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
        for i, item in enumerate(expected):
            if not isinstance(item, ExpectedValue):
                _fail("case.expected", f"unsupported expected value {type(item).__name__}")
            _validate_expected_unit(item, i, self.units)
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

        if self.mesh is not None and not isinstance(self.mesh, MeshSpec):
            _fail("case.mesh", f"expected MeshSpec or None, got {type(self.mesh).__name__}")

    def _component_to_dict(
        self, component: Component, length_exp: int, field_exp: int
    ) -> dict[str, Any]:
        if isinstance(component, Magnet):
            return component.to_dict(length_exp10=-length_exp, field_exp10=-field_exp)
        if isinstance(component, Coil | Permeable | Dipole):
            return component.to_dict(length_exp10=-length_exp)
        return component.to_dict()

    def to_dict(self) -> dict[str, Any]:
        """Return the case as a JSON-ready dict with keys sorted at every level."""
        length_exp = SUPPORTED_LENGTH_UNITS[self.units.length]
        field_exp = SUPPORTED_FIELD_UNITS[self.units.field]
        force_exp = SUPPORTED_FORCE_UNITS[self.units.force]
        expected_out = []
        for i, item in enumerate(self.expected):
            kind = _validate_expected_unit(item, i, self.units)
            val_exp = -(field_exp if kind == "field" else force_exp)
            expected_out.append(item.to_dict(value_exp10=val_exp))
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "name": self.name,
            "description": self.description,
            "sources": list(self.sources),
            "tolerance": self.tolerance,
            "units": self.units.to_dict(),
            "frame": self.frame.to_dict(),
            "components": [
                self._component_to_dict(component, length_exp, field_exp)
                for component in self.components
            ],
            "probes": [probe.to_dict(length_exp10=-length_exp) for probe in self.probes],
            "expected": expected_out,
        }
        if self.manifest is not None:
            payload["manifest"] = self.manifest.to_dict()
        if self.mesh is not None:
            payload["mesh"] = self.mesh.to_dict(length_exp10=-length_exp)
        if self.temperature_c is not None and self.temperature_c != 20.0:
            payload["temperature_c"] = self.temperature_c
        return _sorted(payload)

    @classmethod
    def from_dict(cls, data: Any) -> Case:
        """Build a case from its JSON object; reject unknown keys and missing fields."""
        doc = _mapping(data, "case")
        if "schema" in doc:
            schema = _text(doc["schema"], "case.schema")
            if schema != SCHEMA:
                _fail("case.schema", f"expected {SCHEMA!r}, got {schema!r}")
        if "units" not in doc:
            _fail("case.units", "missing required field 'units'")
        if "frame" not in doc:
            _fail("case.frame", "missing required field 'frame'")
        _keys(
            doc,
            "case",
            (
                "schema",
                "name",
                "description",
                "sources",
                "components",
                "probes",
                "expected",
                "units",
                "frame",
            ),
            ("tolerance", "manifest", "temperature_c", "mesh"),
        )
        units = UnitSpec.from_dict(doc["units"], "case.units")
        frame = FrameSpec.from_dict(doc["frame"], "case.frame")
        length_exp = SUPPORTED_LENGTH_UNITS[units.length]
        field_exp = SUPPORTED_FIELD_UNITS[units.field]
        force_exp = SUPPORTED_FORCE_UNITS[units.force]

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
        raw_mesh = doc.get("mesh")
        mesh: MeshSpec | None = None
        if raw_mesh is not None:
            mesh = MeshSpec.from_dict(raw_mesh, "case.mesh", length_exp10=length_exp)

        parsed_expected: list[ExpectedValue] = []
        for i, item in enumerate(expected):
            raw_expected = ExpectedValue.from_dict(item, f"case.expected[{i}]")
            kind = _validate_expected_unit(raw_expected, i, units)
            val_exp = field_exp if kind == "field" else force_exp
            if val_exp != 0:
                raw_expected = ExpectedValue(
                    check=raw_expected.check,
                    quantity=raw_expected.quantity,
                    value=_scale_power10(raw_expected.value, val_exp),
                    unit=raw_expected.unit,
                    tolerance=raw_expected.tolerance,
                    source=raw_expected.source,
                )
            parsed_expected.append(raw_expected)

        return cls(
            name=doc["name"],
            description=doc["description"],
            sources=tuple(doc["sources"]),
            components=tuple(
                component_from_dict(
                    item,
                    f"case.components[{i}]",
                    length_exp10=length_exp,
                    field_exp10=field_exp,
                    length_unit=units.length,
                )
                for i, item in enumerate(components)
            ),
            probes=tuple(
                Probe.from_dict(item, f"case.probes[{i}]", length_exp10=length_exp)
                for i, item in enumerate(probes)
            ),
            expected=tuple(parsed_expected),
            tolerance=doc.get("tolerance"),
            manifest=manifest,
            temperature_c=doc.get("temperature_c"),
            mesh=mesh,
            units=units,
            frame=frame,
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialise with sorted keys — the on-disk form is deterministic."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Case:
        """Parse a case from JSON text, failing loudly on anything unexpected."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - message shaping only
            raise CaseError(f"case: not valid JSON ({exc})") from exc
        return cls.from_dict(data)
