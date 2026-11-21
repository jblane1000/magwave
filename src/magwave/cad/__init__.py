"""CAD and mesh import, manifest, and body role assignment (FR-010, FR-011, FR-012, FR-013).

No heavy optional backend (``OCP`` or ``trimesh``) is imported at module import time;
backends are loaded lazily inside :func:`import_cad` according to file format.

Supported import paths for defeaturing:

* ``from magwave.cad.defeature import defeature`` (canonical; ``magwave.cad.defeature`` is the
  submodule, not shadowed by the function)
* ``from magwave.cad import defeature_shape`` (package-level alias)
"""

from __future__ import annotations

from pathlib import Path

from .defeature import (
    BooleanResult,
    DefeatureResult,
    FaceDescriptor,
    boolean_subtract,
    boolean_union,
)
from .defeature import defeature as defeature_shape
from .manifest import (
    DEFECT_KINDS,
    SOURCES,
    Body,
    DefeatureRecord,
    Defect,
    Manifest,
    RemovedFeature,
)
from .roles import ROLES, UnassignedBodiesError, assign, require_all_assigned, unassigned

STEP_FORMATS = frozenset({"step", "stp"})
IGES_FORMATS = frozenset({"iges", "igs"})
MESH_FORMATS = frozenset({"stl", "obj", "ply", "3mf", "glb", "gltf"})
SUPPORTED_FORMATS = STEP_FORMATS | IGES_FORMATS | MESH_FORMATS


class MissingExtraError(ImportError):
    """Raised when CAD import requires an optional extra that is not installed."""

    def __init__(self, extra: str, package: str) -> None:
        self.extra = extra
        self.package = package
        super().__init__(
            f"CAD import requires optional package {package!r}; "
            f'install with: pip install "magwave[{extra}]"'
        )


def import_cad(
    path: str | Path,
    *,
    format: str | None = None,
    check_interference: bool = False,
    defeature_below: float | None = None,
) -> Manifest:
    """Import a STEP, IGES, or mesh file into a :class:`Manifest` (FR-010, FR-011, FR-012).

    Raises :class:`MissingExtraError` naming ``magwave[step]`` or ``magwave[mesh]`` when
    the corresponding optional dependency is absent.
    """
    file_path = Path(path)
    resolved = (format if format is not None else file_path.suffix.lstrip(".")).strip().lower()
    if not resolved:
        raise ValueError(f"cannot determine CAD format for {file_path!s}; pass format= explicitly")

    if resolved in STEP_FORMATS:
        from .step import import_step

        return import_step(
            file_path,
            check_interference=check_interference,
            defeature_below=defeature_below,
        )
    if resolved in IGES_FORMATS:
        from .step import import_iges

        return import_iges(file_path, defeature_below=defeature_below)
    if resolved in MESH_FORMATS:
        if defeature_below is not None:
            raise ValueError(
                "--defeature-below applies to STEP and IGES only; mesh files are not defeatured"
            )
        from .mesh_io import import_mesh

        return import_mesh(file_path, fmt=resolved)

    raise ValueError(
        f"unsupported CAD format {resolved!r}; expected one of {sorted(SUPPORTED_FORMATS)}"
    )


__all__ = [
    "DEFECT_KINDS",
    "IGES_FORMATS",
    "MESH_FORMATS",
    "ROLES",
    "SOURCES",
    "STEP_FORMATS",
    "SUPPORTED_FORMATS",
    "Body",
    "BooleanResult",
    "DefeatureRecord",
    "DefeatureResult",
    "Defect",
    "FaceDescriptor",
    "Manifest",
    "MissingExtraError",
    "RemovedFeature",
    "UnassignedBodiesError",
    "assign",
    "boolean_subtract",
    "boolean_union",
    "defeature_shape",
    "import_cad",
    "require_all_assigned",
    "unassigned",
]
