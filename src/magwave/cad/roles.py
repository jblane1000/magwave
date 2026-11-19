"""Body role assignment and pre-solve validation (PRD FR-013, ADR-0006)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .manifest import Body, Defect, Manifest

ROLES: tuple[str, ...] = (
    "permeable_solid",
    "permanent_magnet",
    "coil_volume",
    "air",
    "symmetry_plane",
)


class UnassignedBodiesError(ValueError):
    """Raised when solving is attempted while CAD manifest bodies lack roles (FR-013)."""

    def __init__(self, bodies: Iterable[Body]) -> None:
        self.bodies: tuple[Body, ...] = tuple(bodies)
        listing = ", ".join(f"#{body.index} ({body.display_name})" for body in self.bodies)
        super().__init__(f"unassigned bodies block solve: {listing}")


def assign(manifest: Manifest, mapping: Mapping[int | str, str]) -> Manifest:
    """Assign roles to bodies in ``manifest`` by index or name (FR-013).

    Validates every role value against :data:`ROLES` (raising ``ValueError`` on an
    unknown role). Unmatched body keys are recorded as ``skipped_body`` defects rather
    than silently ignored.
    """
    if not isinstance(mapping, Mapping):
        raise TypeError(f"mapping: expected a mapping, got {type(mapping).__name__}")

    for key, role in mapping.items():
        if isinstance(key, bool) or not isinstance(key, int | str):
            raise TypeError(f"mapping key must be int or str, got {type(key).__name__}")
        if not isinstance(role, str) or role not in ROLES:
            raise ValueError(f"unknown role {role!r} for key {key!r}; expected one of {ROLES}")

    by_index = {body.index: body for body in manifest.bodies}

    for key, role in mapping.items():
        if isinstance(key, int):
            matched = [by_index[key]] if key in by_index else []
            fallback_index: int | None = key
        else:
            matched = [body for body in manifest.bodies if body.name == key]
            if not matched and key.isdigit():
                idx = int(key)
                if idx in by_index:
                    matched = [by_index[idx]]
                fallback_index = idx
            else:
                fallback_index = None

        if not matched:
            manifest.defects.append(
                Defect(
                    kind="skipped_body",
                    detail=f"unknown body key {key!r} in role assignment",
                    body_index=fallback_index,
                )
            )
            continue

        for body in matched:
            body.role = role

    return manifest


def unassigned(manifest: Manifest) -> list[Body]:
    """Return all bodies in ``manifest`` without a valid assigned role."""
    return [body for body in manifest.bodies if body.role is None or body.role not in ROLES]


def require_all_assigned(manifest_or_case: Any) -> None:
    """Raise :class:`UnassignedBodiesError` if any body in ``manifest_or_case`` lacks a role."""
    if isinstance(manifest_or_case, Manifest):
        manifest = manifest_or_case
    elif hasattr(manifest_or_case, "manifest"):
        manifest = manifest_or_case.manifest
        if manifest is None:
            return
        if not isinstance(manifest, Manifest):
            raise TypeError(
                f"expected Manifest or None on .manifest, got {type(manifest).__name__}"
            )
    else:
        raise TypeError(
            f"expected Manifest or object with .manifest, got {type(manifest_or_case).__name__}"
        )

    missing = unassigned(manifest)
    if missing:
        raise UnassignedBodiesError(missing)
