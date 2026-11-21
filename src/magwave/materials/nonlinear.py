"""Nonlinear soft-magnetic B–H material response (M5.2 Stage B).

Models tabulated ``B(|B|) -> H`` curves using monotone shape-preserving piecewise cubic
Hermite interpolation (:class:`scipy.interpolate.PchipInterpolator`) across the measured
table, linear behaviour ``H = |B| / (MU0 * mu_r)`` below the first non-zero table point,
and the ``MU0``-sloped saturation asymptote ``B = MU0 * H + Ms`` (i.e. ``H = (|B| - Ms) /
MU0``) above the table endpoint.

Smoothness across the low- and high-field endpoints
---------------------------------------------------
Tabulated datasheets often round ``mu_r`` slightly relative to ``B_1 / (MU0 * H_1)`` (for
``steel-team20``, shipped ``mu_r = 294.7`` vs initial secant ``294.7314``, a ``0.01 %``
difference) and terminate at a measured ``(H_last, B_last)`` whose offset ``B_last -
MU0 * H_last`` differs slightly from the stated high-field asymptote ``Ms`` (for
``steel-team20``, ``2.30 - MU0 * 135000 = 2.1304 T`` vs ``Ms = 2.16 T``, a ``23.6 kA/m``
jump in ``H`` at ``B = 2.30 T`` if switched abruptly).

To preserve both exact table inversion at every tabulated point, exact initial linear law
at low fields, and exact ``H = (|B| - Ms) / MU0`` saturation behaviour above the knee
without introducing a step jump in ``H(|B|)`` that would stall Newton backtracking:

* Below ``B_1``: pure linear ``H = |B| / (MU0 * mu_r)`` on ``[0, 0.9 * B_1]`` (or all of
  ``[0, B_1)`` if ``mu_r`` matches the first secant to machine precision), followed by a
  monotone :math:`C^1` cubic Hermite transition on ``(0.9 * B_1, B_1)`` into the PCHIP
  interpolant.
* Above ``B_last``: a monotone :math:`C^1` cubic Hermite transition on
  ``(B_last, B_join)`` (for ``steel-team20``, ``B_join = 2.3889 T < 2.40 T``) from
  ``(B_last, H_last, H'(B_last))`` into the exact saturation asymptote ``H = (|B| - Ms) /
  MU0`` with slope ``1 / MU0`` for all ``|B| >= B_join``.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..units import MU0

if TYPE_CHECKING:
    from . import Material


def _check_finite_number(value: Any, where: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}: expected a number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{where}: must be finite, got {value!r}")
    if allow_zero:
        if out < 0.0:
            raise ValueError(f"{where}: must be >= 0, got {value!r}")
    elif out <= 0.0:
        raise ValueError(f"{where}: must be > 0, got {value!r}")
    return out


def validate_bh_table(raw: Any, name: str) -> tuple[tuple[float, float], ...]:
    """Validate a raw ``[[H, B], ...]`` sequence as strictly increasing in both ``H`` and ``B``.

    Raises ``ValueError`` naming ``name`` and the offending point indices without sorting or
    dropping points.
    """
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise ValueError(f"material {name!r}: bh_table expected a list of [H, B] pairs")
    if len(raw) < 2:
        raise ValueError(f"material {name!r}: bh_table must contain at least 2 points")

    points: list[tuple[float, float]] = []
    prev_h = -math.inf
    prev_b = -math.inf
    for idx, pair in enumerate(raw):
        if isinstance(pair, str | bytes) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"material {name!r}: bh_table[{idx}] expected a 2-element [H, B] pair")
        h_val = _check_finite_number(
            pair[0], f"material {name!r}: bh_table[{idx}][0]", allow_zero=True
        )
        b_val = _check_finite_number(
            pair[1], f"material {name!r}: bh_table[{idx}][1]", allow_zero=True
        )
        if idx > 0 and h_val <= prev_h:
            raise ValueError(
                f"material {name!r}: bh_table H is not strictly increasing at points "
                f"{idx - 1} and {idx} (H {prev_h:g} -> {h_val:g})"
            )
        if idx > 0 and b_val <= prev_b:
            raise ValueError(
                f"material {name!r}: bh_table B is not strictly increasing at points "
                f"{idx - 1} and {idx} (B {prev_b:g} -> {b_val:g})"
            )
        prev_h = h_val
        prev_b = b_val
        points.append((h_val, b_val))
    return tuple(points)


def _hermite_cubic_eval(
    x: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    m0: float,
    m1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate cubic Hermite interpolant ``(y, dy/dx)`` on ``[x0, x1]``."""
    length = x1 - x0
    t = (x - x0) / length
    t2 = t * t
    t3 = t2 * t
    y = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * (length * m0)
        + (-2.0 * t3 + 3.0 * t2) * y1
        + (t3 - t2) * (length * m1)
    )
    dy = (
        (6.0 * t2 - 6.0 * t) * ((y0 - y1) / length)
        + (3.0 * t2 - 4.0 * t + 1.0) * m0
        + (3.0 * t2 - 2.0 * t) * m1
    )
    return y, dy


def _hermite_cubic_integral(
    x: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    m0: float,
    m1: float,
) -> np.ndarray:
    """Evaluate exact integral ``int_{x0}^{x} y(xi) dxi`` for cubic Hermite on ``[x0, x1]``."""
    length = x1 - x0
    t = (x - x0) / length
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    return length * (
        (0.5 * t4 - t3 + t) * y0
        + (0.25 * t4 - (2.0 / 3.0) * t3 + 0.5 * t2) * (length * m0)
        + (-0.5 * t4 + t3) * y1
        + (0.25 * t4 - (1.0 / 3.0) * t3) * (length * m1)
    )


def _hermite_min_deriv(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    m0: float,
    m1: float,
) -> float:
    """Exact minimum of ``dy/dx`` for a cubic Hermite polynomial on ``[x0, x1]``."""
    length = x1 - x0
    secant = (y1 - y0) / length
    a = 3.0 * (m0 + m1 - 2.0 * secant)
    b = 6.0 * secant - 4.0 * m0 - 2.0 * m1
    c = m0
    candidates = [m0, m1]
    if abs(a) > 0.0:
        t_star = -b / (2.0 * a)
        if 0.0 < t_star < 1.0:
            candidates.append(a * t_star * t_star + b * t_star + c)
    return min(candidates)


class BhCurve:
    """Monotone B(|B|) -> H inversion of a tabulated B-H curve."""

    def __init__(
        self,
        table: Sequence[Sequence[float]],
        mu_r: float,
        saturation_B: float | None,
        name: str,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("BhCurve name must be a non-empty string")
        self._name = name
        self._mu_r = _check_finite_number(mu_r, f"material {name!r}: mu_r", allow_zero=False)
        self._saturation_B = (
            None
            if saturation_B is None
            else _check_finite_number(
                saturation_B, f"material {name!r}: saturation_B", allow_zero=False
            )
        )
        validated = validate_bh_table(table, name)
        self._table = validated
        self._h_tab = np.array([pt[0] for pt in validated], dtype=np.float64)
        self._b_tab = np.array([pt[1] for pt in validated], dtype=np.float64)

        self._pchip = PchipInterpolator(self._b_tab, self._h_tab, extrapolate=False)
        self._dpchip = self._pchip.derivative()
        self._pchip_anti = self._pchip.antiderivative()

        first_idx = 1 if self._b_tab[0] == 0.0 else 0
        self._b1 = float(self._b_tab[first_idx])
        self._h1 = float(self._h_tab[first_idx])
        self._b_last = float(self._b_tab[-1])
        self._h_last = float(self._h_tab[-1])

        # Low-field configuration: linear H = b / (MU0 * mu_r) up to _b_low_blend, then C^1 bridge
        # to (B1, H1) when mu_r differs slightly from B1 / (MU0 * H1).
        self._s_lin = 1.0 / (MU0 * self._mu_r)
        self._s_b1 = float(self._dpchip(self._b1))
        self._b_low_blend = self._b1
        if abs(self._b1 * self._s_lin - self._h1) > 1e-12 * max(1.0, self._h1):
            cand = 0.9 * self._b1
            for _ in range(20):
                y_cand = cand * self._s_lin
                if (
                    y_cand < self._h1
                    and _hermite_min_deriv(
                        cand, self._b1, y_cand, self._h1, self._s_lin, self._s_b1
                    )
                    > 0.0
                ):
                    self._b_low_blend = cand
                    break
                cand *= 0.5

        # High-field configuration above B_last: C^1 Hermite transition to (b - Ms)/MU0.
        self._s_last = float(self._dpchip(self._b_last))
        self._s_sat = 1.0 / MU0
        self._b_join = self._b_last
        if self._saturation_B is not None:
            h_sat_last = (self._b_last - self._saturation_B) / MU0
            gap = self._h_last - h_sat_last
            if abs(gap) > 1e-12 * max(1.0, self._h_last):
                length = max(3.0 * abs(gap) / self._s_sat, 1e-6)
                min_allowed = 0.25 * min(self._s_last, self._s_sat)
                for _ in range(40):
                    b_j = self._b_last + length
                    h_j = (b_j - self._saturation_B) / MU0
                    if (
                        h_j > self._h_last
                        and _hermite_min_deriv(
                            self._b_last, b_j, self._h_last, h_j, self._s_last, self._s_sat
                        )
                        >= min_allowed
                    ):
                        self._b_join = b_j
                        break
                    length *= 2.0

        # Precompute exact cumulative integrals int_0^b H(B') dB' at piecewise-smooth knots.
        self._w_low_blend = 0.5 * self._s_lin * (self._b_low_blend**2)
        if self._b_low_blend < self._b1:
            bridge_low_int = float(
                _hermite_cubic_integral(
                    np.array([self._b1], dtype=np.float64),
                    self._b_low_blend,
                    self._b1,
                    self._b_low_blend * self._s_lin,
                    self._h1,
                    self._s_lin,
                    self._s_b1,
                )[0]
            )
            self._w_b1 = self._w_low_blend + bridge_low_int
        else:
            self._w_b1 = self._w_low_blend
        self._pchip_anti_b1 = float(self._pchip_anti(self._b1))
        self._w_b_last = self._w_b1 + (float(self._pchip_anti(self._b_last)) - self._pchip_anti_b1)
        if self._saturation_B is not None and self._b_join > self._b_last:
            h_join = (self._b_join - self._saturation_B) / MU0
            bridge_high_int = float(
                _hermite_cubic_integral(
                    np.array([self._b_join], dtype=np.float64),
                    self._b_last,
                    self._b_join,
                    self._h_last,
                    h_join,
                    self._s_last,
                    self._s_sat,
                )[0]
            )
            self._w_b_join = self._w_b_last + bridge_high_int
        else:
            self._w_b_join = self._w_b_last

    @property
    def name(self) -> str:
        return self._name

    @property
    def mu_r(self) -> float:
        return self._mu_r

    @property
    def saturation_B(self) -> float | None:
        return self._saturation_B

    @property
    def b_first_nonzero(self) -> float:
        return self._b1

    @property
    def b_last(self) -> float:
        return self._b_last

    def _validate_query(self, b: Any) -> tuple[np.ndarray, tuple[int, ...]]:
        b_arr = np.asarray(b, dtype=np.float64)
        if not np.all(np.isfinite(b_arr)) or np.any(b_arr < 0.0):
            raise ValueError(f"material {self._name!r}: |B| must be finite and >= 0")
        return np.atleast_1d(b_arr).ravel(), b_arr.shape

    def _eval_h_and_dh(self, b_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h_out = np.empty_like(b_flat)
        dh_out = np.empty_like(b_flat)

        high = b_flat > self._b_last
        if np.any(high) and self._saturation_B is None:
            raise ValueError(
                f"material {self._name!r}: field |B| = {float(np.max(b_flat[high])):.6g} T "
                f"exceeds last B-H table point {self._b_last:g} T and no saturation_B is set"
            )

        low_lin = b_flat <= self._b_low_blend
        low_bridge = (b_flat > self._b_low_blend) & (b_flat < self._b1)
        mid = (b_flat >= self._b1) & (b_flat <= self._b_last)
        high_bridge = high & (b_flat < self._b_join)
        high_sat = high & (b_flat >= self._b_join)

        if np.any(low_lin):
            h_out[low_lin] = b_flat[low_lin] * self._s_lin
            dh_out[low_lin] = self._s_lin

        if np.any(low_bridge):
            y_b, dy_b = _hermite_cubic_eval(
                b_flat[low_bridge],
                self._b_low_blend,
                self._b1,
                self._b_low_blend * self._s_lin,
                self._h1,
                self._s_lin,
                self._s_b1,
            )
            h_out[low_bridge] = y_b
            dh_out[low_bridge] = dy_b

        if np.any(mid):
            b_mid = b_flat[mid]
            h_out[mid] = self._pchip(b_mid)
            dh_out[mid] = self._dpchip(b_mid)

        if np.any(high_bridge):
            assert self._saturation_B is not None
            h_join = (self._b_join - self._saturation_B) / MU0
            y_h, dy_h = _hermite_cubic_eval(
                b_flat[high_bridge],
                self._b_last,
                self._b_join,
                self._h_last,
                h_join,
                self._s_last,
                self._s_sat,
            )
            h_out[high_bridge] = y_h
            dh_out[high_bridge] = dy_h

        if np.any(high_sat):
            assert self._saturation_B is not None
            h_out[high_sat] = (b_flat[high_sat] - self._saturation_B) * self._s_sat
            dh_out[high_sat] = self._s_sat

        return h_out, dh_out

    def field_strength(self, b: Any) -> np.ndarray:
        """Return magnetic field strength ``H(|B|)`` in A/m (vectorised, ``b >= 0``)."""
        b_flat, shape = self._validate_query(b)
        h_out, _ = self._eval_h_and_dh(b_flat)
        return h_out.reshape(shape)

    def dH_dB(self, b: Any) -> np.ndarray:
        """Return ``dH/d|B|`` in A/(m*T) (vectorised, ``b >= 0``)."""
        b_flat, shape = self._validate_query(b)
        _, dh_out = self._eval_h_and_dh(b_flat)
        return dh_out.reshape(shape)

    def reluctivity(self, b: Any) -> np.ndarray:
        """Return secant reluctivity ``nu(|B|) = H(|B|) / |B|`` in A/(m*T)."""
        b_flat, shape = self._validate_query(b)
        h_out, _ = self._eval_h_and_dh(b_flat)
        nu_out = np.empty_like(b_flat)
        low_lin = b_flat <= self._b_low_blend
        nu_out[low_lin] = self._s_lin
        nonlin = ~low_lin
        if np.any(nonlin):
            nu_out[nonlin] = h_out[nonlin] / b_flat[nonlin]
        return nu_out.reshape(shape)

    def reluctivity_and_derivative(self, b: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(nu(|B|), dnu/d|B|)`` with exact cancellation-free limit as ``|B| -> 0``."""
        b_flat, shape = self._validate_query(b)
        h_out, dh_out = self._eval_h_and_dh(b_flat)
        nu_out = np.empty_like(b_flat)
        dnu_out = np.empty_like(b_flat)
        low_lin = b_flat <= self._b_low_blend
        nu_out[low_lin] = self._s_lin
        dnu_out[low_lin] = 0.0
        nonlin = ~low_lin
        if np.any(nonlin):
            b_nz = b_flat[nonlin]
            h_nz = h_out[nonlin]
            dh_nz = dh_out[nonlin]
            nu_out[nonlin] = h_nz / b_nz
            dnu_out[nonlin] = (dh_nz * b_nz - h_nz) / (b_nz * b_nz)
        return nu_out.reshape(shape), dnu_out.reshape(shape)

    def differential_permeability(self, b: Any) -> np.ndarray:
        """Return differential permeability ``dB/dH = 1 / (dH/dB)`` in T*m/A."""
        b_flat, shape = self._validate_query(b)
        _, dh_out = self._eval_h_and_dh(b_flat)
        return (1.0 / dh_out).reshape(shape)

    def magnetic_energy_density(self, b: Any) -> np.ndarray:
        """Return exact stored magnetic energy density ``int_0^|B| H(B') dB'`` in J/m^3."""
        b_flat, shape = self._validate_query(b)
        high = b_flat > self._b_last
        if np.any(high) and self._saturation_B is None:
            raise ValueError(
                f"material {self._name!r}: field |B| = {float(np.max(b_flat[high])):.6g} T "
                f"exceeds last B-H table point {self._b_last:g} T and no saturation_B is set"
            )

        w_out = np.empty_like(b_flat)
        low_lin = b_flat <= self._b_low_blend
        low_bridge = (b_flat > self._b_low_blend) & (b_flat < self._b1)
        mid = (b_flat >= self._b1) & (b_flat <= self._b_last)
        high_bridge = high & (b_flat < self._b_join)
        high_sat = high & (b_flat >= self._b_join)

        if np.any(low_lin):
            w_out[low_lin] = 0.5 * self._s_lin * (b_flat[low_lin] ** 2)

        if np.any(low_bridge):
            w_out[low_bridge] = self._w_low_blend + _hermite_cubic_integral(
                b_flat[low_bridge],
                self._b_low_blend,
                self._b1,
                self._b_low_blend * self._s_lin,
                self._h1,
                self._s_lin,
                self._s_b1,
            )

        if np.any(mid):
            w_out[mid] = self._w_b1 + (
                np.asarray(self._pchip_anti(b_flat[mid]), dtype=np.float64) - self._pchip_anti_b1
            )

        if np.any(high_bridge):
            assert self._saturation_B is not None
            h_join = (self._b_join - self._saturation_B) / MU0
            w_out[high_bridge] = self._w_b_last + _hermite_cubic_integral(
                b_flat[high_bridge],
                self._b_last,
                self._b_join,
                self._h_last,
                h_join,
                self._s_last,
                self._s_sat,
            )

        if np.any(high_sat):
            assert self._saturation_B is not None
            b_s = b_flat[high_sat]
            w_out[high_sat] = self._w_b_join + 0.5 * self._s_sat * (
                (b_s - self._saturation_B) ** 2 - (self._b_join - self._saturation_B) ** 2
            )

        return w_out.reshape(shape)

    def coenergy_density(self, b: Any) -> np.ndarray:
        """Return magnetic coenergy density ``|B|*H(|B|) - w(|B|)`` in J/m^3."""
        b_flat, shape = self._validate_query(b)
        h_out, _ = self._eval_h_and_dh(b_flat)
        w_out = self.magnetic_energy_density(b_flat)
        return (b_flat * h_out - w_out).reshape(shape)


@functools.lru_cache(maxsize=64)
def bh_curve_for(material: Material) -> BhCurve:
    """Build and cache a :class:`BhCurve` for a :class:`Material`."""
    if material.bh_table is None:
        raise ValueError(f"material {material.name!r} has no B-H table data")
    return BhCurve(
        material.bh_table,
        mu_r=material.mu_r,
        saturation_B=material.saturation_B,
        name=material.name,
    )
