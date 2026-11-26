"""Multi-error structural validation for ``magwave.case/2`` documents (PRD FR-060).

Validates JSON syntax, schema version, required and optional keys, units and frame
declarations, component dimensions and materials, probes, expected values and unit
consistency, tolerance gate, mesh spec, operating temperature, and CAD manifest roles —
collecting *every* defect rather than failing on the first one.

This module never solves and never imports GUI, CAD, or solver-runtime dependencies.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cad.manifest import Manifest
from .cad.roles import ROLES, unassigned
from .cases import (
    COIL_DIMS,
    COMPONENT_TYPES,
    FRAME_AXES,
    MAGNET_DIMS,
    PERMEABLE_DIMS,
    PROBE_COMPONENTS,
    PROBE_QUANTITIES,
    SCHEMA,
    SUPPORTED_CURRENT_UNITS,
    SUPPORTED_EXPECTED_UNITS,
    SUPPORTED_FIELD_UNITS,
    SUPPORTED_FORCE_UNITS,
    SUPPORTED_FRAME_AXES,
    SUPPORTED_LENGTH_UNITS,
    Case,
    CaseError,
    _expected_quantity_kind,
)

_TOP_REQUIRED = (
    "schema",
    "name",
    "description",
    "sources",
    "components",
    "probes",
    "expected",
    "units",
    "frame",
)
_TOP_OPTIONAL = ("tolerance", "manifest", "temperature_c", "mesh")

_UNIT_FIELDS: tuple[tuple[str, Mapping[str, int]], ...] = (
    ("length", SUPPORTED_LENGTH_UNITS),
    ("field", SUPPORTED_FIELD_UNITS),
    ("current", SUPPORTED_CURRENT_UNITS),
    ("force", SUPPORTED_FORCE_UNITS),
)


@dataclass(frozen=True)
class ValidationError:
    """One structural or schema defect located at a specific JSON path."""

    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"message": self.message, "path": self.path}


@dataclass(frozen=True)
class ValidationReport:
    """Full validation outcome for a case document."""

    valid: bool
    schema: str
    errors: tuple[ValidationError, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "errors": [error.to_dict() for error in self.errors],
            "schema": self.schema,
            "valid": self.valid,
        }


def _add(errors: list[ValidationError], path: str, message: str) -> None:
    errors.append(ValidationError(path=path, message=message))


def _check_keys(
    data: Mapping[str, Any],
    where: str,
    required: Sequence[str],
    optional: Sequence[str],
    errors: list[ValidationError],
) -> None:
    present = set(data)
    missing = sorted(set(required) - present)
    unknown = sorted(present - set(required) - set(optional))
    for key in missing:
        _add(errors, f"{where}.{key}", f"missing required field {key!r}")
    if unknown:
        allowed = sorted(set(required) | set(optional))
        for key in unknown:
            _add(errors, f"{where}.{key}", f"unknown key {key!r}; allowed keys are {allowed}")


def _check_text(value: Any, where: str, errors: list[ValidationError]) -> str | None:
    if not isinstance(value, str) or not value:
        _add(errors, where, "expected a non-empty string")
        return None
    return value


def _check_number(value: Any, where: str, errors: list[ValidationError]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _add(errors, where, f"expected a number, got {type(value).__name__}")
        return None
    out = float(value)
    if not math.isfinite(out):
        _add(errors, where, "must be finite")
        return None
    return out


def _check_positive_size(value: Any, where: str, errors: list[ValidationError]) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        _add(errors, where, f"must be a finite number > 0, got {value!r}")
        return None
    return float(value)


def _check_triple(value: Any, where: str, errors: list[ValidationError]) -> None:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence) or len(value) != 3:
        _add(errors, where, "expected a list of three numbers")
        return
    for idx in range(3):
        _check_number(value[idx], f"{where}[{idx}]", errors)


def _check_dims(
    value: Any,
    where: str,
    required: Sequence[str],
    errors: list[ValidationError],
) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        _add(errors, where, f"expected a JSON object, got {type(value).__name__}")
        return None
    _check_keys(value, where, required, (), errors)
    out: dict[str, float] = {}
    for key in required:
        if key not in value:
            continue
        size = _check_number(value[key], f"{where}.{key}", errors)
        if size is not None:
            if size <= 0.0:
                _add(errors, f"{where}.{key}", "must be positive")
            else:
                out[key] = size
    return out


def _check_unit(
    value: Any,
    where: str,
    supported: Sequence[str] | Mapping[str, int],
    errors: list[ValidationError],
) -> str | None:
    unit = _check_text(value, where, errors)
    if unit is None:
        return None
    if unit not in supported:
        _add(errors, where, f"unsupported unit {unit!r} (supported: {', '.join(supported)})")
        return None
    return unit


def _validate_units(
    data: Mapping[str, Any], errors: list[ValidationError]
) -> tuple[str | None, str | None]:
    if "units" not in data:
        return None, None
    raw = data["units"]
    where = "case.units"
    if not isinstance(raw, Mapping):
        _add(errors, where, f"expected a JSON object, got {type(raw).__name__}")
        return None, None
    _check_keys(raw, where, ("length", "field", "current", "force"), (), errors)
    valid_units: dict[str, str | None] = {}
    for key, supported in _UNIT_FIELDS:
        if key in raw:
            valid_units[key] = _check_unit(raw[key], f"{where}.{key}", supported, errors)
        else:
            valid_units[key] = None
    return valid_units.get("field"), valid_units.get("force")


def _validate_frame(data: Mapping[str, Any], errors: list[ValidationError]) -> None:
    if "frame" not in data:
        return
    raw = data["frame"]
    where = "case.frame"
    if not isinstance(raw, Mapping):
        _add(errors, where, f"expected a JSON object, got {type(raw).__name__}")
        return
    _check_keys(raw, where, ("axis",), ("datum_face",), errors)
    if "axis" in raw:
        axis = _check_text(raw["axis"], f"{where}.axis", errors)
        if axis is not None:
            if axis not in FRAME_AXES:
                _add(errors, f"{where}.axis", f"expected one of {list(FRAME_AXES)}, got {axis!r}")
            elif axis not in SUPPORTED_FRAME_AXES:
                _add(
                    errors,
                    f"{where}.axis",
                    f"axis {axis!r} is not supported; only 'z' is supported at Tier 1 "
                    "(rotation is not implemented)",
                )
    if (
        "datum_face" in raw
        and raw["datum_face"] is not None
        and not isinstance(raw["datum_face"], str)
    ):
        _add(
            errors,
            f"{where}.datum_face",
            f"expected a string, got {type(raw['datum_face']).__name__}",
        )


def _check_required_name(doc: Mapping[str, Any], where: str, errors: list[ValidationError]) -> str:
    if "name" not in doc:
        _add(errors, f"{where}.name", "expected a non-empty string")
        return ""
    raw_name = doc["name"]
    if raw_name is None:
        return ""
    name = _check_text(raw_name, f"{where}.name", errors)
    return name or ""


def _check_optional_name(doc: Mapping[str, Any], where: str, errors: list[ValidationError]) -> str:
    if "name" not in doc or doc["name"] is None:
        return ""
    raw_name = doc["name"]
    if not isinstance(raw_name, str):
        _add(errors, f"{where}.name", f"expected a string, got {type(raw_name).__name__}")
        return ""
    return raw_name


def _validate_component(
    item: Any,
    where: str,
    errors: list[ValidationError],
) -> tuple[str | None, str]:
    if not isinstance(item, Mapping):
        _add(errors, where, f"expected a JSON object, got {type(item).__name__}")
        return None, ""
    if "type" not in item:
        _add(errors, f"{where}.type", "missing required field 'type'")
        return None, ""
    comp_type = item["type"]
    if comp_type not in COMPONENT_TYPES:
        _add(
            errors,
            f"{where}.type",
            f"unknown component type {comp_type!r}; expected one of {sorted(COMPONENT_TYPES)}",
        )
        return None, ""

    if comp_type == "magnet":
        _check_keys(
            item, where, ("type", "kind", "dims", "br"), ("center", "demag_curve", "name"), errors
        )
        kind = _check_text(item["kind"], f"{where}.kind", errors) if "kind" in item else None
        if kind is not None and kind not in MAGNET_DIMS:
            _add(errors, f"{where}.kind", f"expected one of {sorted(MAGNET_DIMS)}, got {kind!r}")
            kind = None
        if "dims" in item:
            if kind is not None:
                dims = _check_dims(item["dims"], f"{where}.dims", MAGNET_DIMS[kind], errors)
                if (
                    kind == "halbach_cylinder"
                    and dims is not None
                    and "inner_radius" in dims
                    and "outer_radius" in dims
                    and not dims["outer_radius"] > dims["inner_radius"]
                ):
                    _add(
                        errors,
                        f"{where}.dims",
                        "halbach_cylinder requires outer_radius > inner_radius",
                    )
            elif not isinstance(item["dims"], Mapping):
                _add(
                    errors,
                    f"{where}.dims",
                    f"expected a JSON object, got {type(item['dims']).__name__}",
                )
        if "br" in item:
            _check_triple(item["br"], f"{where}.br", errors)
        if "center" in item:
            _check_triple(item["center"], f"{where}.center", errors)
        if "demag_curve" in item and item["demag_curve"] is not None:
            _check_text(item["demag_curve"], f"{where}.demag_curve", errors)
        return comp_type, _check_required_name(item, where, errors)

    if comp_type == "coil":
        _check_keys(
            item, where, ("type", "kind", "dims", "current", "turns"), ("center", "name"), errors
        )
        kind = _check_text(item["kind"], f"{where}.kind", errors) if "kind" in item else None
        if kind is not None and kind not in COIL_DIMS:
            _add(errors, f"{where}.kind", f"expected one of {sorted(COIL_DIMS)}, got {kind!r}")
            kind = None
        if "dims" in item:
            if kind is not None:
                dims = _check_dims(item["dims"], f"{where}.dims", COIL_DIMS[kind], errors)
                if (
                    kind == "solenoid"
                    and dims is not None
                    and "inner_radius" in dims
                    and "outer_radius" in dims
                    and not dims["outer_radius"] >= dims["inner_radius"]
                ):
                    _add(errors, f"{where}.dims", "solenoid requires outer_radius >= inner_radius")
            elif not isinstance(item["dims"], Mapping):
                _add(
                    errors,
                    f"{where}.dims",
                    f"expected a JSON object, got {type(item['dims']).__name__}",
                )
        if "current" in item:
            _check_number(item["current"], f"{where}.current", errors)
        if "turns" in item:
            turns = item["turns"]
            if isinstance(turns, bool) or not isinstance(turns, int):
                _add(errors, f"{where}.turns", f"expected an integer, got {type(turns).__name__}")
            elif turns < 1:
                _add(errors, f"{where}.turns", "must be at least 1")
        if "center" in item:
            _check_triple(item["center"], f"{where}.center", errors)
        return comp_type, _check_required_name(item, where, errors)

    if comp_type == "dipole":
        _check_keys(item, where, ("type", "moment"), ("center", "name"), errors)
        if "moment" in item:
            _check_triple(item["moment"], f"{where}.moment", errors)
        if "center" in item:
            _check_triple(item["center"], f"{where}.center", errors)
        return comp_type, _check_required_name(item, where, errors)

    if comp_type == "permeable":
        _check_keys(
            item,
            where,
            ("type", "kind", "dims", "material"),
            ("center", "mu_r", "name"),
            errors,
        )
        kind = _check_text(item["kind"], f"{where}.kind", errors) if "kind" in item else None
        if kind is not None and kind not in PERMEABLE_DIMS:
            _add(
                errors,
                f"{where}.kind",
                f"expected one of {sorted(PERMEABLE_DIMS)}, got {kind!r}",
            )
            kind = None
        if "dims" in item:
            if kind is not None:
                _check_dims(item["dims"], f"{where}.dims", PERMEABLE_DIMS[kind], errors)
            elif not isinstance(item["dims"], Mapping):
                _add(
                    errors,
                    f"{where}.dims",
                    f"expected a JSON object, got {type(item['dims']).__name__}",
                )
        if "material" in item:
            material = _check_text(item["material"], f"{where}.material", errors)
            if material is not None:
                from .materials import load_library

                try:
                    load_library().get(material)
                except ValueError as exc:
                    _add(errors, f"{where}.material", str(exc))
        if "center" in item:
            _check_triple(item["center"], f"{where}.center", errors)
        if "mu_r" in item and item["mu_r"] is not None:
            mu_r = _check_number(item["mu_r"], f"{where}.mu_r", errors)
            if mu_r is not None and mu_r <= 0.0:
                _add(errors, f"{where}.mu_r", "must be positive (> 0)")
        return comp_type, _check_optional_name(item, where, errors)

    # uniform_field
    _check_keys(item, where, ("type", "h0"), ("name",), errors)
    if "h0" in item:
        h0 = _check_number(item["h0"], f"{where}.h0", errors)
        if h0 is not None and h0 == 0.0:
            _add(errors, f"{where}.h0", "must be non-zero (A/m)")
    return comp_type, _check_optional_name(item, where, errors)


def _validate_components(data: Mapping[str, Any], errors: list[ValidationError]) -> None:
    if "components" not in data:
        return
    raw = data["components"]
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        _add(errors, "case.components", "expected a list")
        return
    if not raw:
        _add(errors, "case.components", "at least one source is required")
        return
    seen_names: set[str] = set()
    duplicate_reported = False
    uniform_count = 0
    for idx, item in enumerate(raw):
        comp_type, name = _validate_component(item, f"case.components[{idx}]", errors)
        if comp_type == "uniform_field":
            uniform_count += 1
        if name:
            if name in seen_names and not duplicate_reported:
                _add(errors, "case.components", "component names must be unique within a case")
                duplicate_reported = True
            seen_names.add(name)
    if uniform_count > 1:
        _add(errors, "case.components", "at most one uniform_field component is allowed")


def _validate_probes(data: Mapping[str, Any], errors: list[ValidationError]) -> None:
    if "probes" not in data:
        return
    raw = data["probes"]
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        _add(errors, "case.probes", "expected a list")
        return
    if not raw:
        _add(errors, "case.probes", "at least one probe is required")
        return
    for idx, item in enumerate(raw):
        where = f"case.probes[{idx}]"
        if not isinstance(item, Mapping):
            _add(errors, where, f"expected a JSON object, got {type(item).__name__}")
            continue
        _check_keys(item, where, ("point", "quantity"), ("component",), errors)
        if "point" in item:
            _check_triple(item["point"], f"{where}.point", errors)
        if "quantity" in item:
            qty = _check_text(item["quantity"], f"{where}.quantity", errors)
            if qty is not None and qty not in PROBE_QUANTITIES:
                _add(
                    errors,
                    f"{where}.quantity",
                    f"expected one of {list(PROBE_QUANTITIES)}, got {qty!r}",
                )
        if "component" in item:
            comp = item["component"]
            if not isinstance(comp, str) or comp not in PROBE_COMPONENTS:
                _add(
                    errors,
                    f"{where}.component",
                    f"expected one of {list(PROBE_COMPONENTS)}, got {comp!r}",
                )


def _validate_expected(
    data: Mapping[str, Any],
    declared_field: str | None,
    declared_force: str | None,
    errors: list[ValidationError],
) -> tuple[list[float], bool]:
    if "expected" not in data:
        return [], False
    raw = data["expected"]
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        _add(errors, "case.expected", "expected a list")
        return [], False
    if not raw:
        _add(errors, "case.expected", "at least one expected value is required")
        return [], False

    tolerances: list[float] = []
    all_tols_valid = True
    for idx, item in enumerate(raw):
        where = f"case.expected[{idx}]"
        if not isinstance(item, Mapping):
            _add(errors, where, f"expected a JSON object, got {type(item).__name__}")
            all_tols_valid = False
            continue
        _check_keys(
            item,
            where,
            ("check", "quantity", "value", "unit", "tolerance", "source"),
            (),
            errors,
        )
        if "check" in item:
            _check_text(item["check"], f"{where}.check", errors)
        qty: str | None = None
        if "quantity" in item:
            qty = _check_text(item["quantity"], f"{where}.quantity", errors)
        if "value" in item:
            _check_number(item["value"], f"{where}.value", errors)
        unit: str | None = None
        if "unit" in item:
            unit = _check_unit(item["unit"], f"{where}.unit", SUPPORTED_EXPECTED_UNITS, errors)
        if "tolerance" in item:
            tol = _check_number(item["tolerance"], f"{where}.tolerance", errors)
            if tol is None:
                all_tols_valid = False
            elif tol <= 0.0:
                _add(
                    errors,
                    f"{where}.tolerance",
                    "must be positive (relative tolerances only)",
                )
                all_tols_valid = False
            else:
                tolerances.append(tol)
        else:
            all_tols_valid = False
        if "source" in item:
            _check_text(item["source"], f"{where}.source", errors)

        if unit is not None and qty is not None:
            kind = _expected_quantity_kind(qty)
            if kind is None:
                kind = "field" if unit in SUPPORTED_FIELD_UNITS else "force"
            if kind == "field" and declared_field is not None and unit != declared_field:
                _add(
                    errors,
                    f"{where}.unit",
                    f"{unit!r} contradicts case.units.field = {declared_field!r}",
                )
            elif kind == "force" and declared_force is not None and unit != declared_force:
                _add(
                    errors,
                    f"{where}.unit",
                    f"{unit!r} contradicts case.units.force = {declared_force!r}",
                )

    return tolerances, all_tols_valid


def _validate_manifest(data: Mapping[str, Any], errors: list[ValidationError]) -> None:
    raw_manifest = data.get("manifest")
    if raw_manifest is None:
        return
    try:
        manifest = Manifest.from_dict(raw_manifest)
    except ValueError as exc:
        _add(errors, "case.manifest", str(exc))
        return
    for pos, body in enumerate(manifest.bodies):
        if body in unassigned(manifest):
            _add(
                errors,
                f"case.manifest.bodies[{pos}].role",
                f"body #{body.index} ({body.display_name}) has unassigned or invalid role "
                f"{body.role!r}; expected one of {ROLES}",
            )


def _validate_mesh(data: Mapping[str, Any], errors: list[ValidationError]) -> None:
    raw_mesh = data.get("mesh")
    if raw_mesh is None:
        return
    where = "case.mesh"
    if not isinstance(raw_mesh, Mapping):
        _add(errors, where, f"expected a JSON object, got {type(raw_mesh).__name__}")
        return
    _check_keys(raw_mesh, where, ("element_size",), (), errors)
    if "element_size" in raw_mesh:
        _check_positive_size(raw_mesh["element_size"], f"{where}.element_size", errors)


def validate_case_data(data: Any) -> ValidationReport:
    """Validate an already-parsed JSON value against ``magwave.case/2``."""
    if not isinstance(data, Mapping):
        return ValidationReport(
            valid=False,
            schema="",
            errors=(
                ValidationError(
                    path="case",
                    message=f"expected a JSON object, got {type(data).__name__}",
                ),
            ),
        )

    raw_schema = data.get("schema")
    schema_seen = raw_schema if isinstance(raw_schema, str) else ""
    errors: list[ValidationError] = []

    _check_keys(data, "case", _TOP_REQUIRED, _TOP_OPTIONAL, errors)

    if "schema" in data:
        parsed_schema = _check_text(data["schema"], "case.schema", errors)
        if parsed_schema is not None and parsed_schema != SCHEMA:
            _add(errors, "case.schema", f"expected {SCHEMA!r}, got {parsed_schema!r}")

    if "name" in data:
        _check_text(data["name"], "case.name", errors)
    if "description" in data:
        _check_text(data["description"], "case.description", errors)

    if "sources" in data:
        sources = data["sources"]
        if isinstance(sources, str) or not isinstance(sources, Sequence):
            _add(errors, "case.sources", "expected a list of strings")
        elif not sources:
            _add(errors, "case.sources", "at least one provenance string is required")
        else:
            for idx, item in enumerate(sources):
                _check_text(item, f"case.sources[{idx}]", errors)

    declared_field, declared_force = _validate_units(data, errors)
    _validate_frame(data, errors)
    _validate_components(data, errors)
    _validate_probes(data, errors)
    expected_tols, all_tols_valid = _validate_expected(data, declared_field, declared_force, errors)

    if data.get("tolerance") is not None:
        stated = _check_number(data["tolerance"], "case.tolerance", errors)
        if stated is not None and all_tols_valid and expected_tols:
            tightest = min(expected_tols)
            if stated != tightest:
                _add(
                    errors,
                    "case.tolerance",
                    f"{stated!r} must equal the tightest expected-value tolerance {tightest!r}; "
                    "tolerances are research-owned and may not be widened",
                )

    if data.get("temperature_c") is not None:
        temp = _check_number(data["temperature_c"], "case.temperature_c", errors)
        if temp is not None and temp < -273.15:
            _add(
                errors,
                "case.temperature_c",
                f"must be >= -273.15 deg C (absolute zero), got {temp!r}",
            )

    _validate_manifest(data, errors)
    _validate_mesh(data, errors)

    if not errors:
        # Parity safety net against Case.from_dict so validate never accepts a case Case.from_dict rejects.
        try:
            Case.from_dict(data)
        except CaseError as exc:  # pragma: no cover - defensive parity guard
            text = str(exc)
            path, sep, message = text.partition(": ")
            if sep:
                _add(errors, path, message)
            else:
                _add(errors, "case", text)

    return ValidationReport(valid=not errors, schema=schema_seen, errors=tuple(errors))


def validate_case_text(text: str) -> ValidationReport:
    """Parse JSON text and validate it against ``magwave.case/2``."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return ValidationReport(
            valid=False,
            schema="",
            errors=(ValidationError(path="case", message=f"not valid JSON ({exc})"),),
        )
    return validate_case_data(data)


def validate_case_file(path: str | Path) -> ValidationReport:
    """Read and validate a case JSON file against ``magwave.case/2``."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        detail = " ".join(str(exc).splitlines())
        return ValidationReport(
            valid=False,
            schema="",
            errors=(ValidationError(path="case", message=f"{path}: {detail}"),),
        )
    return validate_case_text(text)
