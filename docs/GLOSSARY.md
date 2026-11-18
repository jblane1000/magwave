# Glossary

- **B** — magnetic flux density (tesla). **H** — magnetic field strength (A/m). **M** — magnetization (A/m).
- **μ0** — permeability of free space, 4π×10⁻⁷ H/m. **μr** — relative permeability.
- **Br** — remanence: flux density remaining when the applied field is removed.
- **Hcb** — coercivity of the B–H curve (where B = 0). **Hcj** — intrinsic coercivity (where M = 0).
- **BHmax** — maximum energy product, the usual figure of merit for a permanent magnet grade.
- **Demagnetization curve** — the second-quadrant part of the B–H loop where a permanent magnet operates.
- **Air gap** — the non-magnetic clearance a field must cross; the single biggest driver of mesh cost.
- **Maxwell stress tensor** — surface-integral method for force from a computed field.
- **Coenergy / energy derivative** — force from the change in stored energy with position; used for
  nonlinear materials where the stress tensor is less reliable.
- **Axisymmetric** — a 2D reduction for bodies of revolution; the physics of solenoids and fixtures.
- **Kelvin transformation / balloon boundary** — ways to represent an unbounded region in FEM without a
  large air box.
- **TEAM problems** — Compumag's Testing Electromagnetic Analysis Methods benchmark set with published
  cross-code results. TEAM 20 = 3D static force.
- **BEM / integral method** — solves for sources on boundaries; open regions come free, no air mesh.
- **FEM** — solves the field through a volume mesh; handles nonlinear steel naturally, needs a bounded domain.
- **Provenance** — the record of where a number came from: engine, version, case hash, tolerance rung.
