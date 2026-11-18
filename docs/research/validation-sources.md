# Validation sources — rung 1 (closed-form analytics) and rung 2 (TEAM benchmarks)

Status: compiled 2026-11-17. Every formula below was verified numerically before publication against an
independent discretised Biot–Savart/charge integration; the verified implementations and their self-checks
live in [reference_formulas.py](reference_formulas.py) (run `python3 reference_formulas.py`, exit 0 = all pass).
Anything that could not be confirmed from a source is marked **[UNVERIFIED]** with a note on what is missing.

Conventions used throughout:

* SI units: `mu0 = 4*pi*1e-7`, fields in tesla, magnetisation `M` in A/m, remanence `Br = mu0*M` in tesla.
* Elliptic integrals use the **parameter convention** `m = k^2` — the `scipy.special.ellipk(m)` /
  `scipy.special.ellipe(m)` convention (Abramowitz & Stegun; DLMF §19). Implementations that take the
  *modulus* `k` (mpmath `ellipk(k)`, Mathematica `EllipticK[m]` uses m too — check yours) must be converted.
* Verified max relative errors quoted per formula were measured with double-precision numpy in this
  repository's verification runs (see appendix).

---

## (a) Magnetic dipole field

```python
# B(r) of a point dipole m at the origin
B = mu0/(4*pi*r**3) * (3*rhat*dot(m, rhat) - m)      # vector form
# or, per component: B_i = mu0/(4*pi) * (3*r_i*(m dot r)/r**5 - m_i/r**3)
```

`m` = dipole moment vector [A m^2] (= `M*V` for a uniformly magnetised body of volume `V`,
`= N*I*A_hat` for a loop); `r` = field-point vector from the dipole [m]; `r = norm(r)`;
`rhat = r/r`. The distributional term `+ (2*mu0/3)*m*delta(r)` is irrelevant for sampling.

Exact for a uniformly magnetised **sphere**; far-field of any body. Far-zone validity for the two-magnet
force is quantified in (g); for *fields*, use `r >~ 5`--`10` times the largest body dimension for <1 %
agreement (rule of thumb — exact crossover depends on aspect ratio and is not tabulated here).

Citations: Jackson, *Classical Electrodynamics*, 3rd ed., §5.5 (book, no URL); Wikipedia
<https://en.wikipedia.org/wiki/Magnetic_dipole>.

---

## (b) Circular current loop — on-axis and off-axis (complete elliptic integrals)

```python
# loop: radius a, carries current I, lies in the z = 0 plane, centred on the origin
# field point: cylindrical (rho, z);  B = B_rho*rho_hat + B_z*z_hat

# On axis (rho = 0) — exact, elementary:
Bz = mu0 * I * a**2 / (2*(a**2 + z**2)**1.5)

# Off axis (rho > 0) — exact:
Q2  = (a + rho)**2 + z**2          # squared distance to the far side of the wire
p2  = (a - rho)**2 + z**2          # squared distance to the near side
m   = 4*a*rho / Q2                 # elliptic parameter (m = k^2 convention)
K, E = ellipk(m), ellipe(m)        # COMPLETE elliptic integrals, parameter m
# NOTE: 1/Q2, not 1/Q2**1.5 — the common wrong variant; verified here
B_rho = mu0*I*z/(2*pi*rho*sqrt(Q2)) * ( -K + (a**2 + rho**2 + z**2)/p2 * E )
B_z   = mu0*I/(2*pi*sqrt(Q2))       * (  K + (a**2 - rho**2 - z**2)/p2 * E )
```

`a`, `I`, `rho` as above; `z` = axial offset of the field point; `m`, `K(m)`, `E(m)` as in the
conventions block. Limits: `rho -> 0` reproduces the on-axis form analytically (verified);
`|r| >> a` reproduces the dipole of (a). At the wire itself (`rho = a, z = 0`) the field is singular and the
elliptic-integral form diverges logarithmically — do not sample exactly on the wire.

**Verified here:** max relative error **5.7e-14** vs a 60 000-segment Biot–Savart reference at 10
points including `rho = a, z = 0.005` and near-axis points (`reference_formulas.py`, section (b)).

Citations: J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, "Simple Analytic Expressions for the
Magnetic Field of a Circular Current Loop", NASA/TM-2013-217919,
<https://ntrs.nasa.gov/citations/20140002333> (PDF: `.../downloads/20140002333.pdf`) — same expressions in
Cartesian/spherical/cylindrical form; R. A. Schill, "General relation for the vector magnetic field of a
circular current loop", IEEE Trans. Magn. 39(2), 961 (2003); M. Ortner, P. Leitner, F. Slanovc,
"Numerically stable and computationally efficient expression for the magnetic field of a current loop",
*Magnetism* 3(1), 11–31 (2022) — numerically stable implementation notes; NIST DLMF §19,
<https://dlmf.nist.gov/19> for the integral definitions.

---

## (c) Finite solenoid, on-axis

Two distinct objects must not be confused (a volume winding is **not** a magnet model — see (e)):

**Thin-walled solenoid** (current sheet, radius `R`, ampere-turn density `nI`):

```python
# winding occupies z1 < z' < z2;  field point on the axis at z;  nI = n*I [A]
c1 = (z - z1)/sqrt(R**2 + (z - z1)**2)     # cos of the half-angle to the near end (z1)
c2 = (z - z2)/sqrt(R**2 + (z - z2)**2)
Bz = mu0*nI/2 * (c1 - c2)                  # exact for the sheet; elementary
```

**Thick solenoid** (rectangular winding cross-section, `R1 <= a <= R2`, current density `J` [A/m^2]):

```python
# on-axis field; elementary (no elliptic integrals on axis!)
s1, s2 = z - z1, z - z2
def bracket(s):                            # antiderivative of s/sqrt(a^2+s^2) in a
    return 0 if s == 0 else s*(arcsinh(R2/abs(s)) - arcsinh(R1/abs(s)))
Bz = mu0*J/2 * (bracket(s1) - bracket(s2))
# NOTE: asinh(R/|s|) — without |s| the s<0 case has the wrong sign (verified)
```

`z1, z2` = winding end planes; `s1, s2` = signed axial offsets from the field point.
Elliptic-integral machinery (Garrett's `F(p,u)` functions, or the `cel` form of Derby–Olbert below) is
needed only **off axis**; on axis the expressions above are exact elementary forms.

**Verified here:** thin form vs a 3000-loop sum < 1.5e-7 (quadrature of the reference; exact form is
machine-precision); thick form vs an independent 60-slice ring sum at 1.6e-6 relative
(`reference_formulas.py`, section (c)). A commonly derived variant with an `a**3` integrand and a
`sqrt(R^2+s^2)` bracket is **wrong** — it disagrees with the loop-counting derivation and with the ring
sum; it appeared in an earlier draft of this file and was removed.

Citations: M. W. Garrett, "Axially Symmetric Systems for Generating and Measuring Magnetic Fields,
Part I", J. Appl. Phys. 22(9), 1091 (1951), <https://pubs.aip.org/aip/jap/article/22/9/1091/160000/> —
the classical thick-solenoid treatment (elliptic-integral `F(p,u)` off axis); N. Derby, S. Olbert,
"Cylindrical magnets and ideal solenoids", Am. J. Phys. 78(3), 229–235 (2010), DOI 10.1119/1.3256157,
<https://arxiv.org/abs/0909.3880> — exact on-axis form (their eq. 2) and exact off-axis solution via one
compactly defined function `C(k,c,p,s)` with published pseudo-code.

---

## (d) Uniformly magnetised rectangular cuboid — the 8-corner formula

```python
# cuboid: x in [x1,x2], y in [y1,y2], z in [z1,z2]; uniform M along +z [A/m]
# equivalent bound "charges": sigma = -M on the z1 face, +M on the z2 face
corners = [(x1,y1,+1), (x2,y1,-1), (x1,y2,-1), (x2,y2,+1)]   # alternating signs

def Gx(u,v,t): return log((sqrt(u*u+v*v+t*t) - v)/abs(t))    # t = 0: use t = +-1e-12 (tangential B continuous)
def Gy(u,v,t): return log((sqrt(u*u+v*v+t*t) - u)/abs(t))
def Gz(u,v,t): return atan2(u*v, t*sqrt(u*u+v*v+t*t))        # atan2 -> correct branch both sides

def patch(G, t):    # alternating corner sum for one charged face
    return sum(s*G(x - Xc, y - Yc, t) for (Xc,Yc,s) in corners)

Bx = mu0*M/(4*pi) * (patch(Gx, z - z2) - patch(Gx, z - z1))   # +M face at z2, -M face at z1
By = mu0*M/(4*pi) * (patch(Gy, z - z2) - patch(Gy, z - z1))
Bz = mu0*M/(4*pi) * (patch(Gz, z - z2) - patch(Gz, z - z1))
```

`(Xc,Yc,s)` = corner coordinates and alternating sign; `u,v,t` = field point minus corner, per pair
(`u = x-Xc`, `v = y-Yc`, `t = z-z_i`); `t = 0` means the point lies in the face plane — the `Gz` term
evaluates correctly through the `atan2` branch, `Gx/Gy` need a tiny `t` nudge (do not sample exactly on
a face). **The formula returns the true B outside *and* inside the magnet** (the inside value already
contains the `+mu0*M` jump — do not add it again; verified). For magnetisation along `x` or `y`,
permute the coordinate roles.

**Verified here:** max relative error **5.8e-7** vs a 800x800-point charged-face quadrature at 7 external
points (`|B|` up to ~0.026 T), plus an interior-point check where the corner value matched
`mu0*(H_charge + M)` to 1.0e-6 (`reference_formulas.py`, section (d)).

Citations: R. Engel-Herbert, T. Hesjedal, "Calculation of the magnetic stray field of a uniaxial magnetic
domain", J. Appl. Phys. 97, 074504 (2005); J. M. Camacho, V. Sosa, "Alternative method to calculate the
magnetic field of permanent magnets with azimuthal symmetry", Rev. Mex. Fis. E 59(1), 8–17 (2013);
J. Yang et al., Supercond. Sci. Technol. 3, 591 (1990); Magpylib's reference list for cuboid fields
<https://magpylib.readthedocs.io/en/latest/_pages/user_guide/guide_resources_01_physics.html>
(cites exactly these).

---

## (e) Uniformly magnetised cylinder — on-axis and general point

```python
# cylinder: radius R, z in [z1,z2], uniform M along +z  (Br = mu0*M)

# On axis — exact, elementary (two charged end discs):
Bz = mu0*M/2 * ( (z - z1)/sqrt((z - z1)**2 + R**2) - (z - z2)/sqrt((z - z2)**2 + R**2) )

# Any point — EXACT via the Amperean equivalence: the field of the cylinder is
# identical to that of an ideal (surface-current) solenoid with n*I = M:
B(p) = integral over z' in [z1, z2] of  loop_B(R, rho, p_z - z', dI)  with  dI = M*dz'
# i.e. superpose the (b) loop formula with dI = M*dz'; numerically exact, and
# Derby–Olbert / Caciagli et al. give the integrated closed form ("cel"/Q functions)
```

`rho`, `p_z` = cylindrical coordinates of the field point. On-axis limit of the superposition reproduces
the elementary form exactly (the two charged-disc statement of Derby–Olbert is the same result).
A volume-wound thick solenoid with the same outer dimensions gives a *different* field — the magnet
equivalence is the **thin surface sheet** `K_phi = M` on the side wall only.

**Verified here:** on-axis elementary form matched the loop superposition to <2e-8; off-axis
superposition (`nt=2000` slices) vs an independent Biot–Savart sheet integration (6000x300) max
relative error **3.9e-6** over 6 points (`reference_formulas.py`, section (e)).

Citations: Derby & Olbert (as above; also gives `Q(r,z)`/`C(k,c,p,s)` closed forms and a compact
algorithm); A. Caciagli, R. J. Baars, A. P. Philipse, B. W. M. Kuipers, "Exact expression for the magnetic
field of a finite cylinder with arbitrary uniform magnetization", J. Magn. Magn. Mater. 456, 423–432
(2018); F. Slanovc, M. Ortner, M. Moridi, C. Abert, D. Suess, "Full analytical solution for the magnetic
field of uniformly magnetized cylinder tiles", J. Magn. Magn. Mater. 559, 169482 (2022).

---

## (f) Halbach cylinder (k = 1) — bore field and uniformity

```python
# magnetisation pattern (verified numerically here — see below):
M(phi) = (Br/mu0) * ( cos(phi)*rhat + sin(phi)*phihat )
# which in Cartesian components is  M = (Br/mu0) * ( cos(2*phi)*xhat + sin(2*phi)*yhat )

# Infinite cylinder, 2-D exact: bore field is UNIFORM, transverse, magnitude
B_bore = Br * ln(Ro/Ri)          # Ro/Ri = e  ->  B_bore = Br  (classic design check)
```

`Ro`, `Ri` = outer/inner radii; `phi` = azimuth in the plane perpendicular to the cylinder axis;
`Br` = remanent flux density. The field direction is the fixed axis `xhat` for the pattern above.
Perfect uniformity (zero gradient in the bore) is an exact property of the 2-D idealisation, not an
approximation.

**Verified here:** the 2-D bound-current system of the pattern above
(volume `Jz = 2*M0*sin(phi)/r`, side sheets `Kz = -/+M0*sin(phi)` at `Ri,Ro`, `M0 = Br/mu0`) was integrated
over a 220x540 polar grid and reproduced `|B| = Br*ln(Ro/Ri)` at 5 bore sample points with spread
< 1e-6 at `Ro/Ri = e`; the alternative patterns (`cos(2 phi) rhat + sin(2 phi) phihat`, radial-only)
give non-uniform or zero bore field and are ruled out (`reference_formulas.py`, section (f)).

**Finite length:** end effects reduce the bore field; there is **no simple published closed-form
correction** — [UNVERIFIED] here because two equivalent 3-D current/charge representations of the same
magnetisation disagreed in this dossier's 3-D verification runs and the discrepancy is not yet resolved;
do **not** use those runs as reference values. For expected finite-length behaviour and segmented
(16-segment) cylinder fields, use R. Bjørk, C. R. H. Bahl, A. Smith, N. Pryds, "Optimization and
improvement of Halbach cylinder design", J. Appl. Phys. 104, 013910 (2008), DOI 10.1063/1.2952537
(PDF: <https://backend.orbit.dtu.dk/ws/files/4814667/bjork.pdf>) and R. Bjørk, "The ideal dimensions of a
Halbach cylinder of finite length", arXiv:1410.0496 (2014). Validation strategy for MagWave Tier 0:
build the Halbach from cuboid segments (the engine already implements (d)), check the segment
superposition against itself exactly, and check the long-cylinder centre field against `Br*ln(Ro/Ri)`
within the tolerance in the table — the length at which that holds must be taken from Bjørk's published
curves, not from this file. Practical build: N discrete segments, each spanning `360/N` degrees and
magnetised along `M(phi_mid)` evaluated at the segment's mid-angle (the pattern above rotates through
two full turns per revolution, so the segment direction is `M` at its mid-angle, implemented as a
rotation of a reference cuboid by `2*phi_mid` about the cylinder axis).

Citations: K. Halbach, "Design of permanent multipole magnets with oriented rare earth cobalt material",
Nucl. Instrum. Methods 169, 1–10 (1980), <https://www.sciencedirect.com/science/article/pii/0029554X80900944>;
Wikipedia "Halbach array" (uniform-field cylinder section), <https://en.wikipedia.org/wiki/Halbach_array>;
Bjørk 2008 / Bjørk 2014 as above.

---

## (g) Force between two coaxial magnets

Configuration: two magnets on a common axis, magnetised along it, centres separated by `d`
(gap between nearest faces `g`, each magnet length `L`).

```python
# (1) Dipole-dipole (coaxial) — attraction magnitude
m1 = M1*V1 ; m2 = M2*V2                  # moments [A m^2] = Br/mu0 * volume
F = 3*mu0*m1*m2/(2*pi*d**4)              # valid only in the far zone (see below)

# (2) Point-pole model, two identical cylinders (radius R, length L, gap g):
#     each magnet replaced by point charges q = M*pi*R^2 at its two face centres
q2 = mu0*M**2*(pi*R**2)**2/(4*pi)        # = pi*Br**2*R**4/(4*mu0)
F = q2 * (1/g**2 - 2/(g+L)**2 + 1/(g+2*L)**2)     # attraction magnitude
# NOTE the pole pairs: near faces attract (g), far faces attract (g+2L),
# the two cross pairs repel (g+L each) -> -2/(g+L)^2

# (3) EXACT for uniformly magnetised bodies with mu_r ~ 1 (charge model), any pole-face shape:
F_z = sum over the 2x2 face pairs of
      (mu0*s1*s2/(4*pi)) * IntInt (z2_face - z1_face)/r**3 dA1 dA2
# s1, s2 = face charge densities (+-M); for coaxial circular faces this double
# integral is cheap numerically (Gauss-Legendre radial x midpoint angular) and
# reduces to the dipole formula in the far limit (verified, 0.31 % at g = 20 L)
```

`z1_face`, `z2_face` = axial positions of the interacting faces; `s` in A/m; `d = g + L` for identical
cylinders. For **cuboid** magnets the exact closed form is Akoun & Yonnet (below); for coaxial
**cylinders** the exact closed form is elliptic-integral (Vokoun et al., below).

**Validity of the dipole approximation** (computed here for two identical cylinders with `L = 4R`,
`Br = 1.2 T`, `R = 5 mm`, `L = 20 mm`; errors of (1) and (2) relative to the exact (3)):

| gap g | exact F (N) | point-pole error | dipole error |
|---|---|---|---|
| 0.3 L | 7.17 | +98.4 % | -58.8 % |
| 0.5 L | 3.23 | +42.5 % | -48.4 % |
| 1 L | 0.757 | +13.6 % | -30.3 % |
| 1.5 L | 0.271 | +7.0 % | -20.3 % |
| 2 L | 0.122 | +4.3 % | -14.4 % |
| 3 L | 0.0359 | +2.2 % | -8.3 % |
| 5 L | 0.00676 | +0.92 % | -3.7 % |
| 8 L | 1.31e-3 | +0.40 % | -1.7 % |
| 12 L | 2.98e-4 | +0.19 % | -0.80 % |
| 20 L | 4.35e-5 | +0.07 % | -0.31 % |

**Bottom line (computed):** the dipole approximation is within 2 % only for centre distance
`d = g + L >~ 9...10 L` (here `d >~ 40 R`), and it always **under**-estimates attraction; the point-pole
formula is within 2 % for `g >~ 3 L` and **over**-estimates. MagWave's PRD acceptance ("within 2 %
outside near-contact", FR-052) should reference the exact charge-model result, not the dipole.

**Verified here:** the exact (3) implementation reproduces the dipole far limit to 0.31 % at `g = 20 L`
and matches the point-pole formula to 0.92 % at `g = 5 L`; full table above from this session's runs
(`reference_formulas.py`, section (g)).

Citations: dipole force: Wikipedia "Force between magnets" (dipole-dipole section),
<https://en.wikipedia.org/wiki/Force_between_magnets>; G. Lehner, *Electromagnetic Field Theory for
Engineers and Physicists*, p. 309 (2008). Point-pole/cylindrical formulas: same Wikipedia article;
E. P. Furlani, *Permanent Magnet and Electromechanical Devices*, Academic Press (2001) (book). Exact
closed forms: G. Akoun, J.-P. Yonnet, "3D analytical calculation of the forces exerted between two
cuboidal magnets", IEEE Trans. Magn. MAG-20(5), 1962–1964 (1984), IEEE doc 1063554; D. Vokoun,
M. Beleggia, L. Heller, P. Sittner, "Magnetostatic interactions and forces between cylindrical permanent
magnets", J. Magn. Magn. Mater. 321(22), 3758–3763 (2009), DOI 10.1016/j.jmmm.2009.07.030.

---

## TEAM benchmarks — what a magnetostatics solver can actually use

From the official Compumag TEAM problem index <https://www.compumag.org/wp/team/> (problem statements as
linked PDFs). Static magnetostatics-relevant problems on that list:

| Problem | Title | Use |
|---|---|---|
| **TEAM 13** | 3-D Non-Linear Magnetostatic Model (`problem13.pdf`) | Tier-2 3D nonlinear fields (coil between steel channels + plate; 1000/3000 AT) |
| **TEAM 20** | 3-D Static Force Problem (`problem20.pdf`) | **primary force benchmark** — full definition below |
| **TEAM 23** | Force Calculations in Permanent Magnets (`problem23.pdf`) | permanent-magnet force benchmark; natural second rung for MagWave (definition not yet read — [UNVERIFIED] details) |
| **TEAM 25** | Optimization of Die Press Model (`problem25.pdf`) | magnetostatic forward problem inside an optimization benchmark; later |
| **TEAM 33a** | Experimental Validation of Magnetic Local Force Formulations (`problem-33a_V3.pdf`) | validates *how* a code computes force (Maxwell stress vs other formulations); pairs with rung-4 |

Everything else on the electronic list (1b, 2, 3–9, 15, 17, 18, 21 family, 22, 24, 27–31, 33b, 34–36) is
eddy-current, transient, waveguide, induction-motor, SMES-optimisation or hysteresis material — not
static magnetostatics (TEAM 21 is a stray-field **loss** model: Tier 3, out of v1 scope). Older numbering
(TEAM 10, 11, 12, 14, 16, ...) exists historically but only the problems hosted on the index above carry
an authoritative public statement.

## TEAM 20 — full problem definition

Sources: official statement <https://www.compumag.org/wp/wp-content/uploads/2018/06/problem20.pdf>
(scanned; text recovered here page-by-page); canonical results paper N. Takahashi, T. Nakata, H. Morishige,
"Summary of results for problem 20 (3-D static force problem)", COMPEL 14(2/3), 57–75 (1995) — free
2-page reprint <https://www.integratedsoft.com/Documents/tech_3ax.pdf>, full text (paywalled)
<https://www.emerald.com/compel/article/14/2-3/57/78434/>; a complete third-party model implementing the
definition: LS-DYNA example <https://lsdyna.ansys.com/team20/> (mesh recovered here from
<https://lsdyna.ansys.com/wp-content/uploads/2022/12/EM_example_TEAM20.zip>); LS-DYNA Verification Manual
2025 R1, case VM-LSDYNA-EMAG-008, <https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/pdf/Ansys_LS-DYNA_Verification_Manual.pdf>.

**Description.** A "pot-magnet" dc solenoid: a rectangular centre pole inside a rectangular yoke frame,
excited by one circular-section dc coil. The centre pole and yoke are steel; the coil is copper (the
statement only specifies ampere-turns). The model was proposed by T. Nakata, N. Takahashi, H. Morishige,
"Proposal of a model for verification of software for 3-D static force calculation", in *Verification of
Software for 3-D Electromagnetic Field Analysis* (Z. Cheng, K. Jiang, N. Takahashi, eds.), pp. 139–147,
International Academic Publishers (1992). **The coil has 381 turns** (Takahashi 1995 reprint, §II) and the
drive cases are **1000 / 3000 / 4500 / 5000 ampere-turns (dc)** — i.e. 2.625 / 7.874 / 11.811 / 13.123 A
at 381 turns — chosen to saturate the steel. A quarter model may be used with symmetry planes.

**Geometry (mm).** As recovered from the official figures and cross-checked against the LS-DYNA example
mesh (node-coordinate bounding boxes per part, which agree with the statement's Table 3 ranges to 0.1 mm);
the statement's figures are scans, so treat individual digits below as "consistent across both sources":

* Centre pole: **25 x 10 mm** cross-section (`-12.5 <= x <= 12.5`, `-5 <= y <= 5`), **98.5 mm long**,
  occupying `z ~ [26.6, 125.1]` with the top face at the inner face of the top yoke plate
  (`z = 125`) and ~1.6 mm clearance above the bottom plate.
* Yoke frame: total height **150 mm** (`z = 0..150`), footprint **127 x 25 mm** (`-63.5 <= x <= 63.5`,
  `-12.5 <= y <= 12.5`); side legs **25 mm thick** (`38.5 <= |x| <= 63.5`); top and bottom plates
  **25 mm thick** (`z in [125,150]` and `[0,25]`).
* Coil ring: plan-view outer boundary **75 x 75 mm** with **~23 mm corner radii**, inner boundary
  **39 x 39 mm** with **~5 mm corner radii** (radial thickness 18 mm), axial extent
  `z in [26.7, 123.3]` (**96.6 mm long**); air gaps: **7 mm** to the pole face along x,
  **1 mm** to the yoke legs, **1.7 mm** to both plates.

**Material.** Steel (no grade name in the statement; the B-H curve *is* the definition — third parties
refer to it as "MA2 steel" [UNVERIFIED as a grade name]). B-H curve from the statement's Fig. 3/Table 1:

```
B(T)  : 0.00  0.010 0.025 0.05  0.10  0.15  0.20  0.30  0.40  0.50
H(A/m): 0    27    58    100   153   185   205   233   255   285
B     : 0.60  0.70  0.80  0.90  1.00  1.10  1.20  1.30  1.35  1.40
H     : 320   355   405   470   555   673   836   1065  1220  1420
B     : 1.45  1.50  1.55  1.60  1.65  1.70  1.75  1.80  1.85  1.90
H     : 1720  2130  2670  3480  4500  5950  7650  10100 13000 15900
B     : 1.95  2.00  2.05  2.10  2.15  2.20  2.25  2.30
H     : 21100 26300 32900 42700 61700 84300 110000 135000
```
Above 2.3 T: `B = mu0*H + Ms` with **Ms = 2.16 T** (statement eq. (1); the curve was measured with a
permeameter up to 2.3 T). This table matches the LS-DYNA example's `BfuncH.k` curve exactly (verified).

**Quantities to compute** (statement Tables 2–5):

1. `Bz` at `P1 = (0, 0, 25.75)` and `P2 = (12.5, 5.0, 25.75)` (mm), for each of the four AT cases.
2. Average `Bz` along the pole centre-line path (alpha-beta): `x in [-12.5, 12.5]`, `y in [-5, 5]`,
   `z = 75`; and in the yoke (gamma-delta): `x in [38.5, 63.5]`, `y in [-12.5, 12.5]`, `z = 75`.
3. `Bx` distributions: line a-b: `x = 13.0, y = 4.0, z = 120.0..130.0` (11 points); line c-d:
   `x = 13.0, z = 125.0, y = 0.0..13.0` (14 points). (5000 AT is the case plotted in the summary paper.)
4. `Fz` — the electromagnetic force on the **centre pole**, for all four AT cases.

**Published reference values (verified, with sources):**

| Quantity | Value | Provenance |
|---|---|---|
| Measured `Fz` at 3000 AT | **54.4 N** | measurement reported by T. Nakata, N. Takahashi, M. Nakano, H. Morishige, K. Matsubara, "Improvement of measurement of 3-D static force problem (Problem 20)", Proc. Miami TEAM Workshop (Nov 1993); quoted by the EMWorks application note <https://www.emworks.com/application/team-20-dc-solenoid-static-force-benchmark-with-emworks> |
| Measured mean `Bz` in pole (alpha-beta) at 3000 AT | **-1.75 T** | T. Nakata, N. Takahashi, H. Morishige, J. L. Coulomb, J. C. Sabonnadiere, "Analysis of 3-D static force problem", Proc. TEAM Workshop on Computation of Applied Electromagnetics in Materials, pp. 73-79 (1993); quoted by the same EMWorks note |
| Computed `Fz` ( EMS/EMWorks, own solve) | 54.68 N | EMWorks note (above) |
| Computed mean pole `Bz` (EMS) | -1.71 T | EMWorks note (above) |
| Cross-code spread (21 solutions, 13 codes, 11 groups) | **[UNVERIFIED]** | the numeric per-code tables (Tables 4-6 of the COMPEL paper) are not reproduced in any openly accessible copy found; the free reprint carries the analysis (second-order elements agree better; Maxwell-stress results good at fine meshes) but not the tables. Obtain the COMPEL paper before quoting a scatter number. |
| Modern code check | LS-DYNA TEAM 20 results within **4.5 %** of the measurements for all outputs except `Bz` at `P1` at 1000 AT (documented as a measurement-difficulty point) | LS-DYNA Verification Manual 2025 R1, VM-LSDYNA-EMAG-008 (URL above) |
| Measurement apparatus | Hall sensor (accuracy 2 %, active area 1x2 mm), search coils, load cell on a non-magnetic rod, pole positioning restrained within 10 um | Takahashi 1995 reprint, section IV |

The canonical summary paper is the citation of record for this benchmark in CI runs; the 3000 AT
measured pair above is the only *numeric* measured value pair that is freely re-derivable today.

---

## Tolerance rationale table

Recommended CI tolerances for FR-070 ("each case declares tolerance and source"). "Formula precision" =
verified numerical quality of the closed form itself; "engine test" = the gate for a Tier-0 engine's
result against the formula.

| Case | Suite tolerance | Formula precision (verified) | Rationale / dominant error source |
|---|---|---|---|
| (a) dipole | 1e-9 rel (field points, far zone) | machine precision | formula is exact; errors only from sampling in the near zone — always pair the tolerance with a minimum-distance guard |
| (b) loop on/off axis | 1e-10 rel | 5.7e-14 vs Biot-Savart | double-precision elliptic integrals (scipy ~1e-15); the checklist's "<=1 %" is a loose outer bound — do not weaken below 1e-8, which would hide a wrong prefactor |
| (c) solenoid on-axis (thin + thick) | 1e-8 rel | < 1.6e-6 vs independent sums (reference quadrature-limited) | elementary forms are machine-precision; the traps are the `cos`-convention and the wrong-sign `asinh` variant — the 1e-8 gate kills both |
| (d) cuboid 8-corner | 1e-6 rel | 5.8e-7 vs fine quadrature | exact formula; residual is the verification quadrature. Near-face/edge sample points amplify floating-point cancellation — keep sample points >= 0.01x body size away, or nudge off face planes |
| (e) cylinder on-axis / general | on-axis 1e-8; off-axis 1e-3 (loop-superposition route) or 1e-8 with the Derby-Olbert closed form | 3.9e-6 (superposition vs independent sheet integration) | superposition error is purely quadrature (converges as 1/nt); prefer implementing the `cel` closed form for the tight gate |
| (f) Halbach bore field | 2 % vs `Br*ln(Ro/Ri)` at the validated test geometry; uniformity figure 0 in the 2-D ideal, gate the engine with 2 % | 2-D formula verified to ~1e-6 | errors are physical modelling, not numerics: segmentation of the rotating M, finite-length end effects, and material mu_r > 1 — all bounded by published data (Bjørk 2008) [finite-length factor: UNVERIFIED here] |
| (g) two-magnet force | exact-charge reference: 1e-3 (quad) or 1e-6 (Akoun-Yonnet closed form); dipole reference: 2 % only for `d >=~ 10 L` | far-limit 0.31 %, mid-range table above | the dipole is systematically low and fails fast near contact (−30 % at d = 2L); the checklist's "<= 2 % outside near-contact" must define *near-contact* (recommend: dipole gate only for `d >= 10L`; otherwise compare against the exact charge model) |
| TEAM 20 | `Fz` <= 10 % vs 54.4 N at 3000 AT (deviation recorded, not hidden); `Bz` at P1/P2 and line values <= 5 % except `P1` at 1000 AT (documented outlier); mean pole `Bz` <= 3 % vs -1.75 T | n/a (rung 2) | stacked uncertainties: 2 % Hall probe, 2.3 T-clamped B-H extrapolation (`B = mu0*H + Ms`), pole/yoke positioning ~10 um, and the unresolved cross-code scatter (21 solutions, 13 codes — numeric spread [UNVERIFIED]). A modern production solver landed <= 4.5 % (LS-DYNA VM), so 10 % leaves room for coarser meshes while still catching real defects; the 1 % anti-vacuity perturbation check (PRD FR-070) is far tighter and catches sign/scale errors |

If the COMPEL Table-6 spread becomes available, tighten the TEAM 20 force gate from 10 % toward the
published inter-code scatter and record the position within it, per CHECKLISTS/validation.md.

---

## Appendix — how the numbers in this file were verified

* All formulas implemented in `reference_formulas.py` (numpy only, no packages installed) with embedded
  self-checks; `python3 reference_formulas.py` exits 0. Reference integrators: straight-chord Biot–Savart
  sums (>= 60 000 segments), mid-point charged-face quadrature (>= 800x800), and ring-slice sums; elliptic
  integrals computed by composite Simpson quadrature (validated against `K(0.5) = 1.8540746773013719`,
  `E(0.5) = 1.3506438810476755`).
* TEAM 20 statement text was recovered from the official scanned PDF (pages 2-4 were image-only; LZW
  content streams + figure OCR used). Geometry was cross-checked three ways: the statement's Table 2/3/4
  coordinates, the LS-DYNA example mesh node bounding boxes, and OCR of the scanned figures; agreement
  within 0.1 mm except the figure-only corner radii (flagged with "~").
* Known limitations: (i) the finite-length Halbach correction is unresolved here (3-D current- vs
  charge-model verification runs disagreed; marked [UNVERIFIED]); (ii) the numeric per-code spread table
  of TEAM 20 is behind the COMPEL paywall; (iii) dipole far-zone field-validity distances are given as a
  rule of thumb, not a computed table (the force analogue is computed).
