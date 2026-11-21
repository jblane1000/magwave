"""The engine protocol and its value types (ADR-0003, PRD FR-040/FR-041).

Every engine — the Tier 0 Magpylib engine today, the FEM tiers later — implements one
protocol: :meth:`Engine.capabilities`, :meth:`Engine.validate` and :meth:`Engine.solve`.
The contract that makes the protocol worth having is *capability-first validation*:

* ``validate(case)`` returns ``[]`` when the engine can model every component and probe of
  the case, and a list of :class:`Problem` objects otherwise. A case that is merely outside
  the engine's remit is an answer (``severity="error"`` problems), never an exception.
* ``solve(case)`` calls ``validate(case)`` first and raises :class:`EngineError` carrying
  the problems if any of them is an error. An engine that returns a number for a case it
  cannot model hands the user a plausible wrong answer — the defect class ADR-0003 exists
  to prevent (NFR-8).

This module imports nothing but the standard library and :mod:`magwave.cases`, so the
protocol can be exercised (and a lying engine caught) without any solver installed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..cases import PROBE_COMPONENTS, Case

#: Problem severities. An ``error`` means the case cannot be solved by this engine; a
#: ``warning`` is attached to a result that was produced anyway.
SEVERITIES = ("error", "warning")

_AXES = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class Problem:
    """One reason an engine cannot (``error``) or can only approximately (``warning``) solve.

    ``code`` is stable and asserted in tests (e.g. ``E_UNSUPPORTED_GEOMETRY``); ``message``
    is a human sentence naming the case/component and the reason.
    """

    code: str
    message: str
    severity: str

    def __post_init__(self) -> None:
        if not self.code or not isinstance(self.code, str):
            raise ValueError("problem.code must be a non-empty string")
        if not self.message or not isinstance(self.message, str):
            raise ValueError("problem.message must be a non-empty string")
        if self.severity not in SEVERITIES:
            raise ValueError(f"problem.severity must be one of {SEVERITIES}, got {self.severity!r}")

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def __str__(self) -> str:
        return f"{self.severity}: {self.code}: {self.message}"


@dataclass(frozen=True)
class Capabilities:
    """What an engine can model — the basis of capability-driven dispatch (FR-041)."""

    engine_id: str
    rung: str
    source_types: frozenset[str]
    magnet_kinds: frozenset[str]
    coil_kinds: frozenset[str]
    quantities: frozenset[str]
    force_methods: tuple[str, ...]
    #: Version string of the code that produces the numbers (provenance, FR-043/FR-053).
    engine_version: str = "unknown"
    #: Material models this engine supports (e.g. ``"air"``, ``"linear_permeable"``).
    materials: frozenset[str] = frozenset({"air"})


@dataclass(frozen=True)
class Sample:
    """One evaluated vector: ``point`` in metres, ``vector`` in ``unit``.

    ``component`` records which scalar the request asked for at this point (``x``, ``y``,
    ``z`` or ``magnitude``); it selects what :meth:`Result.values` reports.
    """

    point: tuple[float, float, float]
    quantity: str
    vector: tuple[float, float, float]
    unit: str
    component: str = "magnitude"

    def __post_init__(self) -> None:
        if self.component not in PROBE_COMPONENTS:
            raise ValueError(
                f"sample.component must be one of {PROBE_COMPONENTS}, got {self.component!r}"
            )

    def value(self, component: str | None = None) -> float:
        """The scalar ``component`` of :attr:`vector` (default: :attr:`component`)."""
        return pick(self.vector, self.component if component is None else component)


@dataclass(frozen=True)
class ConvergenceInfo:
    """Solver convergence metadata attached to a :class:`Result` (PRD FR-042)."""

    iterations: int
    final_relative_residual: float
    residual_history: tuple[float, ...]
    nodes: int
    elements: int
    interface_nodes: int
    solver: str
    relaxation: str
    material_model: str


@dataclass(frozen=True)
class Result:
    """What :meth:`Engine.solve` returns: samples plus the provenance needed to trust them."""

    engine_id: str
    engine_version: str
    rung: str
    method: str
    samples: tuple[Sample, ...]
    problems: tuple[Problem, ...] = ()
    convergence: ConvergenceInfo | None = None

    def __post_init__(self) -> None:
        errors = [problem for problem in self.problems if problem.is_error]
        if errors:
            raise ValueError(
                "a Result cannot carry error problems; the engine must raise EngineError: "
                + "; ".join(map(str, errors))
            )

    def values(self) -> tuple[float, ...]:
        """One scalar per sample: each sample's requested component."""
        return tuple(sample.value() for sample in self.samples)

    def value(self, index: int, component: str) -> float:
        """Component ``x``/``y``/``z``/``magnitude`` of sample ``index``."""
        return self.samples[index].value(component)


class EngineError(ValueError):
    """``solve()`` was asked for a case the engine has declared it cannot model."""

    def __init__(self, engine_id: str, problems: Iterable[Problem]) -> None:
        self.engine_id = engine_id
        self.problems: tuple[Problem, ...] = tuple(problems)
        detail = "; ".join(str(problem) for problem in self.problems) or "no problems given"
        super().__init__(f"engine {engine_id!r} cannot solve this case: {detail}")


@runtime_checkable
class Engine(Protocol):
    """The engine protocol (ADR-0003). See the module docstring for the contract."""

    def capabilities(self) -> Capabilities: ...

    def validate(self, case: Case) -> list[Problem]: ...

    def solve(self, case: Case) -> Result: ...


def errors_in(problems: Iterable[Problem]) -> list[Problem]:
    """The ``severity="error"`` subset of ``problems``, order preserved."""
    return [problem for problem in problems if problem.is_error]


def raise_on_errors(engine_id: str, problems: Iterable[Problem]) -> list[Problem]:
    """Raise :class:`EngineError` if any problem is an error; otherwise return the warnings."""
    problems = list(problems)
    errors = errors_in(problems)
    if errors:
        raise EngineError(engine_id, errors)
    return problems


def pick(vector: Iterable[float], component: str) -> float:
    """Select ``x``/``y``/``z`` or the Euclidean ``magnitude`` of a 3-vector."""
    values = tuple(float(item) for item in vector)
    if len(values) != 3:
        raise ValueError(f"expected a 3-vector, got {len(values)} components")
    if component in _AXES:
        return values[_AXES[component]]
    if component == "magnitude":
        return math.sqrt(values[0] ** 2 + values[1] ** 2 + values[2] ** 2)
    raise ValueError(f"unsupported component {component!r}; expected one of {PROBE_COMPONENTS}")
