"""Lazy trimesh importer for STL/OBJ/PLY/3MF/GLB/GLTF meshes (PRD FR-010, ADR-0006).

All ``trimesh`` imports are deferred until :func:`import_mesh` is called so that
importing :mod:`magwave.cad` or :mod:`magwave.cad.mesh_io` succeeds without the
``[mesh]`` extra installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .manifest import Body, Defect, Manifest


def _load_trimesh() -> Any:
    from . import MissingExtraError

    try:
        import trimesh
    except ImportError as exc:
        raise MissingExtraError("mesh", "trimesh") from exc
    return trimesh


def _mesh_bbox(mesh: Any) -> tuple[float, float, float, float, float, float]:
    bounds = getattr(mesh, "bounds", None)
    if bounds is None or len(bounds) != 2:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    lo, hi = bounds[0], bounds[1]
    return (
        float(lo[0]),
        float(lo[1]),
        float(lo[2]),
        float(hi[0]),
        float(hi[1]),
        float(hi[2]),
    )


def _clean_mesh_name(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    stripped = raw.strip()
    return stripped if stripped else None


def import_mesh(path: str | Path, *, fmt: str | None = None) -> Manifest:
    """Import a mesh file (STL/OBJ/PLY/3MF/GLB/GLTF) via ``trimesh`` (FR-010)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"mesh file not found: {file_path}")

    trimesh = _load_trimesh()
    resolved_fmt = (fmt or file_path.suffix.lstrip(".") or "mesh").lower()

    load_kwargs: dict[str, Any] = {}
    if fmt is not None:
        load_kwargs["file_type"] = fmt
    loaded = trimesh.load(str(file_path), **load_kwargs)

    raw_units = getattr(loaded, "units", None)
    warnings: list[str] = []
    if isinstance(raw_units, str) and raw_units.strip():
        units = raw_units.strip()
    else:
        units = "unknown"
        warnings.append("Mesh file carries no unit metadata; units set to 'unknown'.")

    bodies: list[Body] = []
    defects: list[Defect] = []

    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.items())
        if not geometries:
            defects.append(
                Defect(
                    kind="skipped_body",
                    detail="mesh scene contains no geometries",
                    body_index=None,
                )
            )
    else:
        meta_name = None
        metadata = getattr(loaded, "metadata", None)
        if isinstance(metadata, dict):
            meta_name = metadata.get("name")
        geometries = [(meta_name, loaded)]

    for raw_name, geom in geometries:
        if not hasattr(geom, "volume"):
            defects.append(
                Defect(
                    kind="skipped_body",
                    detail=f"geometry {raw_name or 'unnamed'!r} is not a surface/volume mesh",
                    body_index=None,
                )
            )
            continue

        body_idx = len(bodies)
        name = _clean_mesh_name(raw_name)
        is_watertight = bool(getattr(geom, "is_watertight", False))
        solids = 1 if is_watertight else 0
        volume = abs(float(geom.volume))
        bbox = _mesh_bbox(geom)

        bodies.append(
            Body(
                index=body_idx,
                name=name,
                role=None,
                volume_mm3=volume,
                solids=solids,
                bbox=bbox,
                source="mesh",
            )
        )
        if not is_watertight:
            defects.append(
                Defect(
                    kind="open_shell",
                    detail=f"mesh body #{body_idx} ({name or f'body_{body_idx}'}) is not watertight",
                    body_index=body_idx,
                )
            )

    return Manifest(
        path=str(file_path),
        format=resolved_fmt,
        units=units,
        bodies=bodies,
        defects=defects,
        warnings=warnings,
    )
