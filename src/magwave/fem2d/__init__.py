"""Tier 1 axisymmetric 2-D FEM solver for linear and nonlinear magnetostatics."""

from __future__ import annotations

from .assembly import load_current_density, load_magnetization_offset, load_sheet, stiffness
from .engine import E_MESH_NODE_BUDGET, E_NOT_CONVERGED, ENGINE_ID, RUNG, Fem2dEngine
from .field import element_field, sample_field
from .force import (
    E_FORCE_CONTOUR_DOES_NOT_ENCLOSE,
    E_FORCE_CONTOUR_IN_MATERIAL,
    E_FORCE_DELTA_NOT_ALIGNED,
    E_FORCE_TARGET_AMBIGUOUS,
    E_FORCE_TARGET_UNKNOWN,
    E_FORCE_UNSUPPORTED_CASE,
    E_TORQUE_NOT_MEANINGFUL,
    DeltaStep,
    DeltaStudyResult,
    ForceError,
    ForceResult,
    coenergy_delta_study,
    coenergy_force,
    magnetic_coenergy,
    maxwell_force,
    torque,
)
from .mesh import (
    DEFAULT_MAX_NODES,
    Mesh,
    MeshBudgetError,
    MeshStats,
    check_rectangular_mesh_budget,
    rectangular_mesh,
)
from .nonlinear import (
    NonlinearConvergenceError,
    assemble_nonlinear_system,
    count_interface_nodes,
    solve_axisymmetric_nonlinear,
)
from .solve import SolveReport, dirichlet_nodes, solve_axisymmetric

__all__ = [
    "DEFAULT_MAX_NODES",
    "ENGINE_ID",
    "E_FORCE_CONTOUR_DOES_NOT_ENCLOSE",
    "E_FORCE_CONTOUR_IN_MATERIAL",
    "E_FORCE_DELTA_NOT_ALIGNED",
    "E_FORCE_TARGET_AMBIGUOUS",
    "E_FORCE_TARGET_UNKNOWN",
    "E_FORCE_UNSUPPORTED_CASE",
    "E_MESH_NODE_BUDGET",
    "E_NOT_CONVERGED",
    "E_TORQUE_NOT_MEANINGFUL",
    "RUNG",
    "DeltaStep",
    "DeltaStudyResult",
    "Fem2dEngine",
    "ForceError",
    "ForceResult",
    "Mesh",
    "MeshBudgetError",
    "MeshStats",
    "NonlinearConvergenceError",
    "SolveReport",
    "assemble_nonlinear_system",
    "check_rectangular_mesh_budget",
    "coenergy_delta_study",
    "coenergy_force",
    "count_interface_nodes",
    "dirichlet_nodes",
    "element_field",
    "load_current_density",
    "load_magnetization_offset",
    "load_sheet",
    "magnetic_coenergy",
    "maxwell_force",
    "rectangular_mesh",
    "sample_field",
    "solve_axisymmetric",
    "solve_axisymmetric_nonlinear",
    "stiffness",
    "torque",
]
