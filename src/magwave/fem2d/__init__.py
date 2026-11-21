"""Tier 1 axisymmetric 2-D FEM solver for linear and nonlinear magnetostatics."""

from __future__ import annotations

from .assembly import load_current_density, load_magnetization_offset, load_sheet, stiffness
from .engine import E_NOT_CONVERGED, ENGINE_ID, RUNG, Fem2dEngine
from .field import element_field, sample_field
from .mesh import Mesh, MeshStats, rectangular_mesh
from .nonlinear import (
    NonlinearConvergenceError,
    assemble_nonlinear_system,
    count_interface_nodes,
    solve_axisymmetric_nonlinear,
)
from .solve import SolveReport, dirichlet_nodes, solve_axisymmetric

__all__ = [
    "ENGINE_ID",
    "E_NOT_CONVERGED",
    "RUNG",
    "Fem2dEngine",
    "Mesh",
    "MeshStats",
    "NonlinearConvergenceError",
    "SolveReport",
    "assemble_nonlinear_system",
    "count_interface_nodes",
    "dirichlet_nodes",
    "element_field",
    "load_current_density",
    "load_magnetization_offset",
    "load_sheet",
    "rectangular_mesh",
    "sample_field",
    "solve_axisymmetric",
    "solve_axisymmetric_nonlinear",
    "stiffness",
]
