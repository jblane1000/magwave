"""Tier 1 axisymmetric 2-D FEM solver for linear magnetostatics."""

from __future__ import annotations

from .assembly import load_current_density, load_sheet, stiffness
from .engine import ENGINE_ID, RUNG, Fem2dEngine
from .field import element_field, sample_field
from .mesh import Mesh, MeshStats, rectangular_mesh
from .solve import SolveReport, dirichlet_nodes, solve_axisymmetric

__all__ = [
    "ENGINE_ID",
    "RUNG",
    "Fem2dEngine",
    "Mesh",
    "MeshStats",
    "SolveReport",
    "dirichlet_nodes",
    "element_field",
    "load_current_density",
    "load_sheet",
    "rectangular_mesh",
    "sample_field",
    "solve_axisymmetric",
    "stiffness",
]
