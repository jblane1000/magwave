# Checklist — mesh quality (Tier 1 and above)

The mesh is where a plausible wrong answer is born. Check these before believing any force number.

## Geometry before meshing
- [ ] Import manifest reviewed: units, bodies found, open shells, interferences (FR-011).
- [ ] Defeaturing applied only below the declared threshold; volume change reported.
- [ ] Every body has a role; unassigned bodies block the solve (FR-013).
- [ ] Symmetry planes declared and physically true (not assumed for convenience).

## Mesh construction
- [ ] Air region: box or Kelvin layer; extents ≥ 5× the largest device dimension, or justified otherwise.
- [ ] Element size in every air gap ≥ 3 elements across the gap; the thinnest gap dictates the mesh cost
      and that cost is stated before the run.
- [ ] Surface mesh on force surfaces (pole faces, magnet faces) refined: ≥ 4 elements across the face.
- [ ] Aspect ratio and skewness reported; worst elements identified by location.
- [ ] Mesh statistics archived with the result (node count, element count, quality metrics).

## Convergence
- [ ] Nonlinear: iteration count and final residual reported; residual below the declared tolerance.
- [ ] Mesh independence: a second run at 1.5× element density changes the answer < 2 %.
      If it does not converge with refinement, the model is wrong before the solver is.
- [ ] Force: recompute with a slightly shifted integration surface; change < 2 %.

## Red flags that invalidate a result
- [ ] Any element thinner than 1/10 of the smallest feature (likely a CAD artefact).
- [ ] Non-convergence reported as success.
- [ ] A force that matches "what we expected" — that is a reason to check harder, not to ship.
