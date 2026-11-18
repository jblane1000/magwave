"""Engines behind one protocol (ADR-0003, PRD FR-040/FR-041).

:mod:`magwave.engines.base` defines the protocol and its value types and imports no
solver. Concrete engines are imported lazily, so ``import magwave.engines`` never pulls
in Magpylib (or, later, a FEM backend) until an engine is actually asked for.
"""

from __future__ import annotations

from typing import Any

from .base import (
    Capabilities,
    Engine,
    EngineError,
    Problem,
    Result,
    Sample,
    errors_in,
    pick,
    raise_on_errors,
)

__all__ = [
    "Capabilities",
    "Engine",
    "EngineError",
    "MagpylibEngine",
    "Problem",
    "Result",
    "Sample",
    "errors_in",
    "pick",
    "raise_on_errors",
]


def __getattr__(name: str) -> Any:
    if name == "MagpylibEngine":
        from .magpylib_engine import MagpylibEngine

        return MagpylibEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
