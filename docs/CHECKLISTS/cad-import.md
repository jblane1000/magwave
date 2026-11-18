# Checklist — CAD import

Assemblies are hostile. This is the dominant failure mode in commercial tools too; treat it as a
first-class feature, not an error message.

## Before import
- [ ] Units confirmed at source (mm vs in), not assumed. A 25.4× error is silent.
- [ ] Assembly exported with bodies, not a single fused solid, if per-body assignment is needed.
- [ ] STEP AP203/214/242 preferred; IGES only when STEP is unavailable; STL for visual checks only.

## Import manifest (must be produced every time)
- [ ] Body count and per-body volume, compared against CAD's own mass properties (0.1 % agreement).
- [ ] Units detected and declared; mismatch with the case file reported.
- [ ] Open shells / non-manifold edges listed with body ids.
- [ ] Interference pairs listed with overlap volume.
- [ ] Bodies skipped and the reason for each — never silent (FR-011).

## Cleaning
- [ ] Small-feature threshold chosen and recorded (e.g. all fillets < 0.5 mm).
- [ ] Booleans performed on the repaired copy; original CAD untouched.
- [ ] Coil and magnet bodies verified as closed solids, correct winding direction / magnetization axis.
- [ ] Datum/frame recorded in the case: axis, datum face, note (mirrors the CAD house rule).

## After import
- [ ] Visual check: assembly orientation matches CAD (Z-up by default).
- [ ] A deliberately broken assembly produces the expected manifest defects (regression test).
- [ ] Case file references CAD by relative path + SHA-256 (FR-004); a changed file is detected.
