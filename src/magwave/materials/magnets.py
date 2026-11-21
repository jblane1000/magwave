"""Versioned permanent-magnet second-quadrant demagnetisation library (FR-020 / M5.3)."""

from __future__ import annotations

import functools
import importlib.resources
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..units import MU0

MAGNET_SCHEMA_PREFIX = "magwave.magnets/"
MAGNET_SCHEMA = f"{MAGNET_SCHEMA_PREFIX}1"

_MAGNET_ENTRY_KEYS = frozenset(
    {
        "name",
        "grade",
        "family",
        "curve",
        "br_ref",
        "hcb",
        "hcj",
        "bhmax",
        "t_ref_c",
        "alpha_br_per_c",
        "alpha_hcj_per_c",
        "t_max_c",
        "provenance",
    }
)

_ABSOLUTE_ZERO_C = -273.15


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}: expected a number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{where}: must be finite, got {value!r}")
    return out


def _positive_finite(value: Any, where: str) -> float:
    out = _finite_number(value, where)
    if out <= 0.0:
        raise ValueError(f"{where}: must be finite and > 0, got {value!r}")
    return out


def _temperature_finite(value: Any, where: str) -> float:
    out = _finite_number(value, where)
    if out < _ABSOLUTE_ZERO_C:
        raise ValueError(f"{where}: must be >= {_ABSOLUTE_ZERO_C} C, got {value!r}")
    return out


def _non_empty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: expected a non-empty string")
    return value


def validate_demag_curve(raw: Any, name: str) -> tuple[tuple[float, float], ...]:
    """Validate a second-quadrant ``[[H, B], ...]`` curve without sorting or dropping points.

    Requires at least 4 points, strictly increasing ``H`` ending at ``H[-1] == 0``, and
    strictly increasing ``B``. Raises ``ValueError`` naming ``name`` and the offending
    point indices on any defect.
    """
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise ValueError(f"magnet {name!r}: curve expected a list of [H, B] pairs")
    if len(raw) < 4:
        raise ValueError(f"magnet {name!r}: curve must contain at least 4 points, got {len(raw)}")

    points: list[tuple[float, float]] = []
    prev_h = -math.inf
    prev_b = -math.inf
    for idx, pair in enumerate(raw):
        if isinstance(pair, str | bytes) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"magnet {name!r}: curve[{idx}] expected a 2-element [H, B] pair")
        h_val = _finite_number(pair[0], f"magnet {name!r}: curve[{idx}][0]")
        b_val = _finite_number(pair[1], f"magnet {name!r}: curve[{idx}][1]")
        if idx > 0 and h_val <= prev_h:
            raise ValueError(
                f"magnet {name!r}: curve H is not strictly increasing at points "
                f"{idx - 1} and {idx} (H {prev_h:g} -> {h_val:g})"
            )
        if idx > 0 and b_val <= prev_b:
            raise ValueError(
                f"magnet {name!r}: curve B is not strictly increasing at points "
                f"{idx - 1} and {idx} (B {prev_b:g} -> {b_val:g})"
            )
        prev_h = h_val
        prev_b = b_val
        points.append((h_val, b_val))

    last_idx = len(points) - 1
    if points[-1][0] != 0.0:
        raise ValueError(
            f"magnet {name!r}: curve must end at H=0 (got H={points[-1][0]:g} at point {last_idx})"
        )
    return tuple(points)


class DemagCurve:
    """Monotone second-quadrant demagnetisation curve ``B(H)`` / ``H(B)`` with affine slopes."""

    def __init__(
        self,
        curve: Sequence[Sequence[float]],
        *,
        name: str = "custom",
        br_ref: float | None = None,
        hcb: float | None = None,
        hcj: float | None = None,
        bhmax: float | None = None,
        t_ref_c: float = 20.0,
        alpha_br_per_c: float = 0.0,
        alpha_hcj_per_c: float = 0.0,
        temperature_c: float | None = None,
        _ref_curve: tuple[tuple[float, float], ...] | None = None,
        _ref_br: float | None = None,
        _ref_hcb: float | None = None,
        _ref_hcj: float | None = None,
        _ref_bhmax: float | None = None,
    ) -> None:
        self._name = _non_empty_str(name, "DemagCurve.name")
        validated = validate_demag_curve(curve, self._name)
        self._curve = validated
        self._t_ref_c = _temperature_finite(t_ref_c, f"magnet {self._name!r}: t_ref_c")
        self._temperature_c = (
            self._t_ref_c
            if temperature_c is None
            else _temperature_finite(temperature_c, f"magnet {self._name!r}: temperature_c")
        )
        self._alpha_br_per_c = _finite_number(
            alpha_br_per_c, f"magnet {self._name!r}: alpha_br_per_c"
        )
        self._alpha_hcj_per_c = _finite_number(
            alpha_hcj_per_c, f"magnet {self._name!r}: alpha_hcj_per_c"
        )
        self._br_ref = (
            float(validated[-1][1])
            if br_ref is None
            else _positive_finite(br_ref, f"magnet {self._name!r}: br_ref")
        )
        self._hcb_ref = (
            None if hcb is None else _positive_finite(hcb, f"magnet {self._name!r}: hcb")
        )
        self._hcj = (
            abs(float(validated[0][0]))
            if hcj is None
            else _positive_finite(hcj, f"magnet {self._name!r}: hcj")
        )
        self._bhmax = (
            None if bhmax is None else _positive_finite(bhmax, f"magnet {self._name!r}: bhmax")
        )

        self._ref_curve = validated if _ref_curve is None else _ref_curve
        self._ref_br = self._br_ref if _ref_br is None else _ref_br
        self._ref_hcb = self._hcb_ref if _ref_hcb is None else _ref_hcb
        self._ref_hcj = self._hcj if _ref_hcj is None else _ref_hcj
        self._ref_bhmax = self._bhmax if _ref_bhmax is None else _ref_bhmax

        self._h_tab = np.array([pt[0] for pt in validated], dtype=np.float64)
        self._b_tab = np.array([pt[1] for pt in validated], dtype=np.float64)
        self._b_of_h_interp = PchipInterpolator(self._h_tab, self._b_tab, extrapolate=False)
        self._h_of_b_interp = PchipInterpolator(self._b_tab, self._h_tab, extrapolate=False)
        self._dh_db = self._h_of_b_interp.derivative()
        self._d2h_db2 = self._dh_db.derivative()

        # Detect collinear curves (including the synthetic uniform-Br linear 1/MU0 curve) so
        # slope_and_offset avoids PCHIP spline floating-point wobble and produces exact zeros
        # for c0 - c and ds_db / dc_db.
        br_end = float(self._b_tab[-1])
        s_chord = float((self._h_tab[-1] - self._h_tab[0]) / (self._b_tab[-1] - self._b_tab[0]))
        s_vac = 1.0 / MU0
        s_cand = s_vac if abs(s_chord - s_vac) <= 1e-12 * s_vac else s_chord
        c_cand = -s_cand * br_end
        max_dev = float(np.max(np.abs(self._h_tab - (s_cand * self._b_tab + c_cand))))
        if max_dev <= 1e-11 * max(abs(float(self._h_tab[0])), 1.0):
            self._linear_slope: float | None = s_cand
            self._linear_intercept: float | None = c_cand
        else:
            self._linear_slope = None
            self._linear_intercept = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def curve(self) -> tuple[tuple[float, float], ...]:
        return self._curve

    @property
    def temperature_c(self) -> float:
        return self._temperature_c

    @property
    def t_ref_c(self) -> float:
        return self._t_ref_c

    @property
    def alpha_br_per_c(self) -> float:
        return self._alpha_br_per_c

    @property
    def alpha_hcj_per_c(self) -> float:
        return self._alpha_hcj_per_c

    @property
    def br_ref(self) -> float:
        return self._br_ref

    @property
    def hcb_ref(self) -> float | None:
        return self._hcb_ref

    @property
    def hcj(self) -> float:
        return self._hcj

    @property
    def bhmax(self) -> float | None:
        return self._bhmax

    @property
    def h_min(self) -> float:
        return float(self._h_tab[0])

    @property
    def h_max(self) -> float:
        return float(self._h_tab[-1])

    @property
    def b_min(self) -> float:
        return float(self._b_tab[0])

    @property
    def b_max(self) -> float:
        return float(self._b_tab[-1])

    def _validate_h(self, h: Any) -> tuple[np.ndarray, tuple[int, ...]]:
        h_arr = np.asarray(h, dtype=np.float64)
        if not np.all(np.isfinite(h_arr)):
            raise ValueError(f"magnet {self._name!r}: H must be finite")
        h_flat = np.atleast_1d(h_arr).ravel()
        h_lo = float(self._h_tab[0])
        h_hi = float(self._h_tab[-1])
        if np.any(h_flat < h_lo) or np.any(h_flat > h_hi):
            raise ValueError(
                f"magnet {self._name!r}: field H outside tabulated domain "
                f"[{h_lo:g}, {h_hi:g}] A/m (got range [{float(np.min(h_flat)):g}, "
                f"{float(np.max(h_flat)):g}] A/m; extrapolation is forbidden)"
            )
        return h_flat, h_arr.shape

    def _validate_b(self, b: Any) -> tuple[np.ndarray, tuple[int, ...]]:
        b_arr = np.asarray(b, dtype=np.float64)
        if not np.all(np.isfinite(b_arr)):
            raise ValueError(f"magnet {self._name!r}: B must be finite")
        b_flat = np.atleast_1d(b_arr).ravel()
        b_lo = float(self._b_tab[0])
        b_hi = float(self._b_tab[-1])
        if np.any(b_flat < b_lo) or np.any(b_flat > b_hi):
            raise ValueError(
                f"magnet {self._name!r}: flux density B outside tabulated domain "
                f"[{b_lo:g}, {b_hi:g}] T (got range [{float(np.min(b_flat)):g}, "
                f"{float(np.max(b_flat)):g}] T; extrapolation is forbidden)"
            )
        return b_flat, b_arr.shape

    def b_of_h(self, h: Any) -> np.ndarray:
        """Evaluate monotone ``B(H)`` in tesla across the tabulated domain (no extrapolation)."""
        h_flat, shape = self._validate_h(h)
        out = np.asarray(self._b_of_h_interp(h_flat), dtype=np.float64)
        return out.reshape(shape)

    def h_of_b(self, b: Any) -> np.ndarray:
        """Evaluate monotone ``H(B)`` in A/m across the tabulated domain (no extrapolation)."""
        b_flat, shape = self._validate_b(b)
        if self._linear_slope is not None and self._linear_intercept is not None:
            out = self._linear_slope * b_flat + self._linear_intercept
            return out.reshape(shape)
        out = np.asarray(self._h_of_b_interp(b_flat), dtype=np.float64)
        return out.reshape(shape)

    def slope_and_offset(self, b: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(s, ds_db, c, dc_db)`` for ``H(B) = s*B + c`` with ``s = dH/dB``."""
        b_flat, shape = self._validate_b(b)
        if self._linear_slope is not None and self._linear_intercept is not None:
            s_out = np.full_like(b_flat, self._linear_slope)
            ds_out = np.zeros_like(b_flat)
            c_out = np.full_like(b_flat, self._linear_intercept)
            dc_out = np.zeros_like(b_flat)
        else:
            h_vals = np.asarray(self._h_of_b_interp(b_flat), dtype=np.float64)
            s_out = np.asarray(self._dh_db(b_flat), dtype=np.float64)
            ds_out = np.asarray(self._d2h_db2(b_flat), dtype=np.float64)
            c_out = h_vals - s_out * b_flat
            dc_out = -b_flat * ds_out
        return (
            s_out.reshape(shape),
            ds_out.reshape(shape),
            c_out.reshape(shape),
            dc_out.reshape(shape),
        )

    def br(self) -> float:
        """Remanence ``B(0)`` in tesla."""
        return float(self.b_of_h(0.0))

    def hcb(self) -> float:
        """Normal coercivity ``H(B=0)`` in A/m (negative in the second quadrant)."""
        return float(self.h_of_b(0.0))

    def bh_product_max(self, samples: int = 4001) -> float:
        """Maximum energy product ``max(-H * B)`` in J/m^3 over ``samples >= 2001`` points."""
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2001:
            raise ValueError(f"samples must be an integer >= 2001, got {samples!r}")
        h_grid = np.linspace(float(self._h_tab[0]), float(self._h_tab[-1]), samples)
        b_grid = self.b_of_h(h_grid)
        return float(np.max(-h_grid * b_grid))

    def at_temperature(self, t_c: float) -> DemagCurve:
        """Return a :class:`DemagCurve` scaled reversibly to temperature ``t_c`` in deg C."""
        t_val = _temperature_finite(t_c, f"magnet {self._name!r}: temperature_c")
        if t_val == self._temperature_c:
            return self
        dt = t_val - self._t_ref_c
        k_br = 1.0 + self._alpha_br_per_c * dt
        k_hcj = 1.0 + self._alpha_hcj_per_c * dt
        if k_br <= 0.0 or k_hcj <= 0.0:
            raise ValueError(
                f"magnet {self._name!r}: temperature {t_val:g} C drives scaling factor "
                f"non-positive (k_br={k_br:g}, k_hcj={k_hcj:g})"
            )
        scaled_curve = tuple(
            (0.0 if h_i == 0.0 else k_hcj * h_i, 0.0 if b_i == 0.0 else k_br * b_i)
            for h_i, b_i in self._ref_curve
        )
        return DemagCurve(
            scaled_curve,
            name=self._name,
            br_ref=self._ref_br * k_br,
            hcb=None if self._ref_hcb is None else self._ref_hcb * k_hcj,
            hcj=self._ref_hcj * k_hcj,
            bhmax=None if self._ref_bhmax is None else self._ref_bhmax * k_br * k_hcj,
            t_ref_c=self._t_ref_c,
            alpha_br_per_c=self._alpha_br_per_c,
            alpha_hcj_per_c=self._alpha_hcj_per_c,
            temperature_c=t_val,
            _ref_curve=self._ref_curve,
            _ref_br=self._ref_br,
            _ref_hcb=self._ref_hcb,
            _ref_hcj=self._ref_hcj,
            _ref_bhmax=self._ref_bhmax,
        )


@dataclass(frozen=True)
class MagnetGrade:
    """One permanent-magnet grade record from ``magnet_library.json``."""

    name: str
    grade: str
    family: str
    curve: tuple[tuple[float, float], ...]
    br_ref: float
    hcb: float
    hcj: float
    bhmax: float
    t_ref_c: float
    alpha_br_per_c: float
    alpha_hcj_per_c: float
    t_max_c: float
    provenance: str

    def __post_init__(self) -> None:
        name = _non_empty_str(self.name, "magnet.name")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "grade", _non_empty_str(self.grade, f"magnet({name}).grade"))
        object.__setattr__(self, "family", _non_empty_str(self.family, f"magnet({name}).family"))
        object.__setattr__(self, "curve", validate_demag_curve(self.curve, name))
        object.__setattr__(self, "br_ref", _positive_finite(self.br_ref, f"magnet({name}).br_ref"))
        object.__setattr__(self, "hcb", _positive_finite(self.hcb, f"magnet({name}).hcb"))
        object.__setattr__(self, "hcj", _positive_finite(self.hcj, f"magnet({name}).hcj"))
        object.__setattr__(self, "bhmax", _positive_finite(self.bhmax, f"magnet({name}).bhmax"))
        object.__setattr__(
            self, "t_ref_c", _temperature_finite(self.t_ref_c, f"magnet({name}).t_ref_c")
        )
        object.__setattr__(
            self,
            "alpha_br_per_c",
            _finite_number(self.alpha_br_per_c, f"magnet({name}).alpha_br_per_c"),
        )
        object.__setattr__(
            self,
            "alpha_hcj_per_c",
            _finite_number(self.alpha_hcj_per_c, f"magnet({name}).alpha_hcj_per_c"),
        )
        object.__setattr__(
            self, "t_max_c", _temperature_finite(self.t_max_c, f"magnet({name}).t_max_c")
        )
        object.__setattr__(
            self, "provenance", _non_empty_str(self.provenance, f"magnet({name}).provenance")
        )

    def demag_curve(self, temperature_c: float | None = None) -> DemagCurve:
        """Build a :class:`DemagCurve` for this grade at ``temperature_c`` (default ``t_ref_c``)."""
        base = DemagCurve(
            self.curve,
            name=self.name,
            br_ref=self.br_ref,
            hcb=self.hcb,
            hcj=self.hcj,
            bhmax=self.bhmax,
            t_ref_c=self.t_ref_c,
            alpha_br_per_c=self.alpha_br_per_c,
            alpha_hcj_per_c=self.alpha_hcj_per_c,
            temperature_c=self.t_ref_c,
        )
        if temperature_c is not None:
            return base.at_temperature(temperature_c)
        return base


class MagnetLibrary:
    """In-memory view over the validated shipped permanent-magnet library."""

    def __init__(self, version: int, schema: str, magnets: Sequence[MagnetGrade]) -> None:
        self.version = version
        self.schema = schema
        by_name: dict[str, MagnetGrade] = {}
        for grade in magnets:
            if grade.name in by_name:
                raise ValueError(f"duplicate magnet grade name {grade.name!r}")
            by_name[grade.name] = grade
        self._magnets = by_name

    @classmethod
    def from_dict(cls, data: Any) -> MagnetLibrary:
        """Parse and validate a permanent-magnet library mapping using strict schema rules."""
        return _parse_magnet_library(data)

    def names(self) -> tuple[str, ...]:
        """Sorted tuple of available permanent-magnet grade names."""
        return tuple(sorted(self._magnets))

    def get(self, name: str) -> MagnetGrade:
        """Look up a magnet grade by name, listing shipped grades on failure."""
        if name not in self._magnets:
            available = ", ".join(self.names()) or "<none>"
            raise ValueError(f"unknown magnet grade {name!r}; shipped grades: {available}")
        return self._magnets[name]


def _parse_magnet_library(data: Any) -> MagnetLibrary:
    if not isinstance(data, Mapping):
        raise ValueError(f"magnet library: expected JSON object, got {type(data).__name__}")
    expected_top = {"schema", "version", "magnets"}
    missing = sorted(expected_top - set(data))
    unknown = sorted(set(data) - expected_top)
    if missing:
        raise ValueError(f"magnet library: missing required key(s) {missing}")
    if unknown:
        raise ValueError(f"magnet library: unknown key(s) {unknown}")

    schema = _non_empty_str(data["schema"], "magnet library.schema")
    if not schema.startswith(MAGNET_SCHEMA_PREFIX):
        raise ValueError(
            f"magnet library.schema: expected prefix {MAGNET_SCHEMA_PREFIX!r}, got {schema!r}"
        )
    raw_schema_ver = schema[len(MAGNET_SCHEMA_PREFIX) :]
    if not raw_schema_ver.isdigit():
        raise ValueError(f"magnet library.schema: invalid version suffix in {schema!r}")
    schema_version = int(raw_schema_ver)

    version = data["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"magnet library.version: expected positive integer, got {version!r}")
    if version != schema_version:
        raise ValueError(
            f"magnet library version mismatch: version={version} != schema version {schema_version}"
        )

    raw_magnets = data["magnets"]
    if (
        isinstance(raw_magnets, str | bytes)
        or not isinstance(raw_magnets, Sequence)
        or not raw_magnets
    ):
        raise ValueError("magnet library.magnets: expected a non-empty list")

    parsed: list[MagnetGrade] = []
    for idx, entry in enumerate(raw_magnets):
        where = f"magnets[{idx}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where}: expected JSON object, got {type(entry).__name__}")
        missing_e = sorted(_MAGNET_ENTRY_KEYS - set(entry))
        unknown_e = sorted(set(entry) - _MAGNET_ENTRY_KEYS)
        if missing_e:
            raise ValueError(f"{where}: missing required key(s) {missing_e}")
        if unknown_e:
            raise ValueError(f"{where}: unknown key(s) {unknown_e}")
        parsed.append(
            MagnetGrade(
                name=entry["name"],
                grade=entry["grade"],
                family=entry["family"],
                curve=entry["curve"],
                br_ref=entry["br_ref"],
                hcb=entry["hcb"],
                hcj=entry["hcj"],
                bhmax=entry["bhmax"],
                t_ref_c=entry["t_ref_c"],
                alpha_br_per_c=entry["alpha_br_per_c"],
                alpha_hcj_per_c=entry["alpha_hcj_per_c"],
                t_max_c=entry["t_max_c"],
                provenance=entry["provenance"],
            )
        )

    return MagnetLibrary(version=version, schema=schema, magnets=parsed)


@functools.lru_cache(maxsize=1)
def load_magnet_library() -> MagnetLibrary:
    """Load, validate, and cache the packaged permanent-magnet library."""
    raw_text = (
        importlib.resources.files("magwave.materials")
        .joinpath("magnet_library.json")
        .read_text(encoding="utf-8")
    )
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"magnet library: invalid JSON ({exc})") from exc
    return _parse_magnet_library(payload)


MAGNET_LIBRARY_VERSION: int = load_magnet_library().version


@functools.lru_cache(maxsize=64)
def demag_curve_for(name: str, temperature_c: float = 20.0) -> DemagCurve:
    """Return the cached :class:`DemagCurve` for grade ``name`` scaled to ``temperature_c``."""
    return load_magnet_library().get(name).demag_curve(temperature_c)


__all__ = [
    "MAGNET_LIBRARY_VERSION",
    "MAGNET_SCHEMA",
    "MAGNET_SCHEMA_PREFIX",
    "DemagCurve",
    "MagnetGrade",
    "MagnetLibrary",
    "demag_curve_for",
    "load_magnet_library",
    "validate_demag_curve",
]
