# Materials & data dossier — B–H curves, demagnetization data, and the Tier-0 engine

Status: Phase 0 research · compiled 2026-11-17 · supports PRD FR-020/021/022 and non-goal NG5.
Every number below was read from the cited public page/PDF on that date. Licences and datasheet
revisions drift: re-verify a URL before quoting it in a shipped material record. NG5/clean-room rule
applies throughout: no vendor-proprietary library may be copied; only public datasheets, open
datasets, benchmarks and the owner's own measurements [PRD §4 NG5, FR-022].

---

## 1) Where a developer may legally obtain B–H and demagnetization data

Two legal classes, treated differently: **manufacturer published datasheets** (© manufacturer,
"reference only" data; quoting parameters with attribution is standard engineering practice, but there
is no open-data licence — the provenance string must carry the exact source URL, revision and access
date) and **genuinely open sources** (explicit licence attached). Standards define *measurement
methods*, not data, and are paywalled.

| Source | Provides | Licence / terms | URL |
|---|---|---|---|
| Arnold Magnetic Technologies catalogs & per-grade PDFs | NdFeB grade summary (N30–N55, M/H/SH… variants) with Br, HcB, HcJ, (BH)max, α(Br), α(HcJ), Tmax; Sm₂Co₁₇/SmCo₅ (Recoma) grade tables + demag curves at 20–350 °C; cast Alnico typical values | Published datasheet, © Arnold, "general reference only… should not be relied upon" | https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Catalog-151021.pdf ; https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Recoma-Combined-160301.pdf ; https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Cast-Alnico-Permanent-Magnet-Brochure-101117-1.pdf |
| Shin-Etsu Chemical per-grade sheets | N52 (NMF-series) full property sheet: Br, HcB, HcJ, (BH)max, μ_recoil, α(Br), α(HcJ), Curie, density; demag curves to 140 °C | Published datasheet, "subject to change without notice" | https://www.shinetsu.co.jp/serem/e/download/N52sheet.pdf |
| Eclipse Magnetics technical data sheets | NdFeB: standard 55-grade range (min values @20 °C), temperature-coefficient table, IEC 60404-8-1 R7 codes, intrinsic-vs-normal curve explainer. Ferrite: Y-grades (Chinese std), C-grades (US std), HF-grades (IEC 60404-8-1), temp coefficients | Published datasheet, © Eclipse | https://www.eclipsemagnetics.com/site/assets/files/22177/eclipsemagnetics_ndfeb_datasheet.pdf ; https://www.eclipsemagnetics.com/site/assets/files/19602/ferrite_ceramic_datasheet.pdf |
| Cleveland-Cliffs (ex-AK Steel) product data bulletins | DI-MAX non-oriented electrical steels M-15…M-47 (core-loss limits, DC properties table); ARMCO pure iron bulletin (magnetization curves as bands, saturation, resistivity) | Published datasheet, © 2026 Cleveland-Cliffs, "for the purpose of information only" | https://www.clevelandcliffs.com/_assets/_835429d6b3dea882921737e43e6dea75/clevelandcliffs/db/1190/10494/file/CLF_ProductData_DiMaxM15M47_042026.pdf ; https://www.aksteel.nl/files/downloads/clf_productdata__armco_pure_iron_pdb_euro_final_072022_92.pdf |
| Vacuum Schmelze VACODYM/VACOMAX product information | NdFeB + SmCo grade tables (public brochure; also mirrored on CERN Indico) | Published brochure, © VAC [terms page not reviewed — UNVERIFIED] | https://vacuumschmelze.com/03_Documents/Brochures/Product%20Information%20VACODYM-VACOMAX%20en.pdf ; https://indico.cern.ch/event/682107/attachments/1560598/2465492/VACODYM-VACOMAX-PD002_2015_en.pdf |
| TDK ferrite magnet datasheets | FB-series hard ferrite sheets & curves per grade (e.g. FB9B) | Published datasheet, © TDK [numbers not extracted here; URL verified] | https://product.tdk.com/en/search/magnet/magnet/ferrite/info?part_no=FB9B |
| K&J Magnetics grade chart | Retailer-published Br/BHmax tables for common grades | Page fetch returned HTTP 429 during this review; terms [UNVERIFIED] | https://www.kjmagnetics.com/neodymium-magnet-specifications.asp |
| FEMM (Finite Element Method Magnetics) | Built-in materials library incl. soft steels and hard magnets; the hard-magnet entries are stated to be "representative values based on a survey of many vendor materials" | Program under Aladdin Free Public License v8 (free to use/redistribute, **not** OSI-approved, GPL-incompatible); data terms not separately stated [UNVERIFIED per-file] | https://www.femm.info/doku/doku.php?id=license ; https://www.femm.info/doku/doku.php?id=hardmagneticmaterials |
| Compumag TEAM benchmark problems | Problem definitions incl. measured steel B–H (e.g. TEAM 20, needed above 2.3 T, permeameter-measured) and published reference results | Public benchmark definitions from the open literature (papers © their publishers; the Compumag PDF is freely downloadable) | https://www.compumag.org/wp/wp-content/uploads/2018/06/problem20.pdf ; summary: https://www.integratedsoft.com/documents/tech_3ax.pdf |
| upb-lea/materialdatabase (Paderborn University LEA) | Measured complex permeability/permittivity CSVs for soft ferrites (N95-type, HF power magnetics; sources: datasheets, TDK MDT, Princeton MagNet, own lab) | **GPL-3.0** — usable as data, but treat with the NFR-3 lens: no GPL code linkage; re-derive from its cited primary sources if in doubt | https://github.com/upb-lea/materialdatabase |
| OpenMagnetics (PyOpenMagnetics) | Open library of magnetic-component materials and models (ferrites etc.), machine-readable | MIT | https://github.com/OpenMagnetics |
| MagWeb (SMAG handbook / BHmag) | ~commercial B–H database for thousands of materials | **Commercial, paid** — named here only so nobody mistake it for open. Not a permitted source under NG5 | https://www.magweb.us/ |
| IEC 60404-5:2015 | Measurement method for PM magnetic properties (B, polarization J) — cite as *method standard*, not data | Paywalled standard, © IEC | https://webstore.iec.ch/en/publication/22142 |

Notes. (a) The Cliffs DI-MAX and ARMCO bulletins define their test conventions as ASTM A343 / A677 /
A683 / A34 — paywalled ASTM standards; reference them as method context only. (b) Carpenter Hiperco 50
(Fe-Co) is documented behind a gated portal (form required); a distributor PDF exists at
https://www.efineametals.com/wp-content/uploads/2022/11/Hiperco-50-Data-Sheet-11_3_2022.pdf [contents
not verified here]. (c) Suggested FR-022 provenance string:
`provenance = "Arnold Magnetic Technologies NdFeB grade summary N42, Catalog-151021.pdf, read
2026-11-17, © Arnold, published datasheet (reference data, not licensed for redistribution)"`.

## 2) Curve data formats worth supporting, and safe interpolation

**Canonical shipped format (FR-021, plain text, versioned):** an ordered table of `(H_i, B_i)` pairs in
SI (H: A/m, B: T), strictly increasing H, with mandatory metadata: temperature of measurement (or "RT"),
magnetization direction/state (initial magnetization curve vs major loop), anneal condition, units,
source URL, revision and provenance string. 20–50 well-spaced points capture a soft-material curve;
monotone resampling handles the rest. A `type: "linear"` variant carries only `mur`.

**Analytic approximations** (fit *to* tabulated data, never as invented physics):
- **Linear μr** — the Tier-1 cold-start default for weakly saturated problems.
- **Saturating rational fit (Fröhlich–Kennelly type)**, e.g. `B = μ0·H + Js·(H/(a + H))` with `(Js, a)`
  fit by least squares — cheap, monotone by construction, good inside its fitted range.
- **Tabulated + monotone interpolation — the default and the source of truth.** Approximations are
  documented conveniences; the table wins any conflict.

**Safe interpolation (this is a correctness item, not style):**
- Use a **monotone shape-preserving cubic Hermite** (PCHIP / Fritsch–Carlson; `scipy.interpolate.
  PchipInterpolator`, SciPy 1.18.1 BSD-3 is already in the stack). Steffen's method is an equivalent
  C¹ alternative. These never overshoot because they clamp the derivative.
- **Do not use natural cubic splines or Akima as defaults**: both oscillate near strong curvature
  (the knee of a B–H curve), producing non-physical negative differential permeability — Newton
  iterations then diverge or, worse, converge to a nonsense answer (NFR-8).
- Check `dB/dH > 0` everywhere with a finite-difference pass and reject a table whose slope is
  non-positive (that is a data-entry defect — FR-021/AC path, fail loudly).
- Piecewise-linear is always available as a fallback (C⁰, exactly monotone), acceptable for coarse
  tables but visible in kinked field solutions.

**Extrapolation rules (deliberately strict):**
1. Inside the table: interpolate (PCHIP).
2. `0 ≤ H < H_first`: extend linearly from the origin through the first segment direction (µr of the
   first segment), never negative B.
3. `H > H_last`: extend linearly with the **last-segment slope**, but cap so
   `B ≤ μ0·H + Js_max` (saturation asymptote), with `Js_max` taken from the provenance-recorded
   saturation induction; warn (annotate the result), don't fail.
4. Beyond ~+20 % of the last tabulated H, or if the cap would bind: hard error naming the material
   and the offending field — better a stopped run than a plausible wrong number (NFR-8, FR-042 spirit).

**Scope note:** v1 ships the *single-valued DC curve* (initial or normal curve from the datasheet).
Hysteresis loops, loss and eddy effects are Tier-3 territory (NG2); say so in the material record
rather than pretending a major loop is the initial curve.

## 3) Permanent-magnet operating point: the second quadrant in practice

**Two curves live on every PM datasheet.** The *intrinsic* (polarization) curve is `J(H)`; the *normal*
(or flux-density) curve is `B(H) = μ0·H + J(H)`. `HcB` (coercivity of B) is where the normal curve
crosses zero; `HcJ` (intrinsic coercivity) is where the intrinsic curve crosses zero, and it is always
larger/more negative in H. `BHmax` is read off the **normal** curve. For NdFeB at room temperature the
normal curve is nearly linear at slope μ_rec ≈ 1.05 (Shin-Etsu N52 sheet: "Recoil Permeability μr
1.05") from `Br` down to `HcB`, then falls more steeply near the knee; the intrinsic curve stays flat
near `J ≈ Br` until approaching `−HcJ`, where it collapses — that collapse is the **knee**. Eclipse's
datasheet states the design rule exactly: keep the *intrinsic* working point above the knee, ideally
at the BHmax working point, or the loss becomes permanent and irreversible
(https://www.eclipsemagnetics.com/site/assets/files/22177/eclipsemagnetics_ndfeb_datasheet.pdf).

**Load line.** A magnet in a circuit sits on a `Pc = B/H` line set by its geometry (self-demag +
external circuit). `Pc ≈ 2` on the normal curve corresponds to `Pci = Pc + 1 ≈ 3` on the intrinsic
curve (the +1 is the μ0·H term). High demag fields or short magnets lower Pc and drag the working
point toward the knee — which is why the same grade "handles" different temperatures in different
shapes (Eclipse ferrite and NdFeB sheets both carry this caveat in equivalent words).

**Temperature application (reversible model, the v1 model).** With measured-or-datasheet coefficients
around the 20 °C reference:
`Br(T) = Br20·(1 + α_Br/100 · (T − 20))` and `HcJ(T) = HcJ20·(1 + α_HcJ/100 · (T − 20))`.

Verified coefficient examples, all with their α ranges printed on the sheets:
- **NdFeB:** α(Br) −0.11 to −0.12 %/K, α(HcJ) −0.49…−0.70 %/K depending on variant; N52/SH-type
  variants push α(HcJ) to −0.618 %/°C (Arnold N42: α(Br) −0.12, α(HcJ) −0.618, Tw 80 °C; Eclipse
  family table: N −0.120/−0.70 … VH/AH −0.090/−0.49, rated to 230 °C).
- **SmCo:** α(Br) −0.035 %/K (Sm₂Co₁₇ Recoma 26–35E) / −0.045…−0.05 %/K (SmCo₅ Recoma 18–25);
  Recoma per-grade sheets state α(HcJ) −0.19 %/°C for SmCo₅; max operating temp 250–350 °C.
- **Alnico:** "lowest temperature coefficient of any commercial magnet material (.02 per cent per
  degree centigrade)" (Arnold Cast Alnico brochure intro).
- **Hard ferrite — sign flip:** α(Br) ≈ **−0.2 %/°C** but α(Hci) ≈ **+0.27 %/°C**; the intrinsic
  coercivity *rises* with temperature (Eclipse ferrite sheet temperature-coefficients row). Ferrite is
  safe against self-demag but its Br drops ~10 % from 20→80 °C; curves from −40 to +100 °C are printed
  on the datasheet.

**Consequences MagWave must implement (FR-020):**
1. Scale `Br` and `HcJ` separately — the knee position changes with T, so re-plot the whole curve, not
   just the endpoint.
2. Carry the demag curve in the second quadrant as data (or as the linear recoil segment plus a knee
   point) — a magnet whose intrinsic working point crosses the knee has permanently lost magnetization;
   flag as a warning-level annotation on the result (it is a state change of the part, not a solver bug).
3. Record which curve the datasheet labels — vendors plot both `B` and `J` in the same figure, and
   misreading one for the other is the classic first error.
4. Units: Br in T, H in kA/m or kOe (state conversions: 1 kA/m = 12.566 Oe), α in %/K or %/°C — they
   are the same number, but *write which*.

## 4) Practical starter-library numbers (each with source URL + terms)

All values below are transcribed from the cited public datasheet on 2026-11-17. "min"/"typ" flags mean
what the sheet says. Do not silently average two vendors' grades — same letter, different melt.

### 4.1 Magnet grades — grade / Br / HcB / HcJ / (BH)max

| Grade (family) | Br | HcB | HcJ | (BH)max | α(Br), α(HcJ) %/°C | Tmax | Source (URL + terms) |
|---|---|---|---|---|---|---|---|
| N42 NdFeB, min @20 °C (Eclipse) | 1.28 T | 923 kA/m | 955 kA/m | 318 kJ/m³ (40 MGOe) | see temp table | 80 °C | [Eclipse NdFeB TDS](https://www.eclipsemagnetics.com/site/assets/files/22177/eclipsemagnetics_ndfeb_datasheet.pdf), published datasheet |
| N42 NdFeB, typical (Arnold) | 1.315 T typ | 860 kA/m min | 955 kA/m min | 334 kJ/m³ typ | −0.12 / −0.618 | 80 °C | [Arnold catalog](https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Catalog-151021.pdf), published datasheet |
| N52 NdFeB (Arnold, typ/min) | 1.45 T typ | 836 kA/m min | 875 kA/m min | 406 kJ/m³ typ | −0.12 / −0.618 | **60 °C** | [Arnold catalog](https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Catalog-151021.pdf) |
| N52 NdFeB (Shin-Etsu, range) | 1.40–1.47 T | 820 kA/m min | 875 kA/m min | 378–422 kJ/m³ | −0.11 / −0.61 (20–140 °C) | — ; μrec 1.05, Curie 330 °C, 7.55 g/cm³ | [Shin-Etsu N52 sheet](https://www.shinetsu.co.jp/serem/e/download/N52sheet.pdf) |
| N35 NdFeB, min (Eclipse) | 1.17 T | 867 kA/m | 955 kA/m | 263 kJ/m³ (33 MGOe) | per family table | 80 °C | [Eclipse NdFeB TDS](https://www.eclipsemagnetics.com/site/assets/files/22177/eclipsemagnetics_ndfeb_datasheet.pdf) |
| Recoma 28 Sm₂Co₁₇ (Arnold, typ/min) | 1.10 / 1.04 T | 800 / 700 kA/m | 2000 / 1200 kA/m | 225 / 195 kJ/m³ | −0.035 (Br, 20–150 °C) | 350 °C | [Recoma brochure](https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Recoma-Combined-160301.pdf) |
| Recoma 30S Sm₂Co₁₇ (Arnold, typ/min) | 1.12 / 1.09 T | 845 / 820 kA/m | 2150 / 1750 kA/m | 235 / 225 kJ/m³ | −0.035 | 350 °C | [Recoma brochure](https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Recoma-Combined-160301.pdf) |
| Recoma 20 SmCo₅ (Arnold, typ/min) | 0.90 / 0.85 T | 700 / 640 kA/m | 2400 / 2000 kA/m | 160 / 140 kJ/m³ | −0.045 / −0.19 (HcJ) | 250 °C | [Recoma brochure](https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Recoma-Combined-160301.pdf) |
| Alnico 5 cast (Arnold, typical) | 12.5 kG (1.25 T) | 640 Oe (≈51 kA/m) | — (no HcJ printed in this brochure) | 5.50 MGOe (43.8 kJ/m³) | ≈−0.02 (family, brochure text) | high (Tc ≈ 800–900 °C class) [exact value per grade — UNVERIFIED here] | [Cast Alnico brochure](https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Cast-Alnico-Permanent-Magnet-Brochure-101117-1.pdf) |
| C5 = Feroba2 = Y30 hard ferrite (Eclipse) | 370–400 mT | 175–210 kA/m | 180–220 kA/m | 26.0–30.0 kJ/m³ | −0.2 / *(+0.27)* | 250–300 °C; density 4.9–5.1 g/cm³ | [Eclipse ferrite TDS](https://www.eclipsemagnetics.com/site/assets/files/19602/ferrite_ceramic_datasheet.pdf) |
| C8 = Feroba3 = Y30H-1 hard ferrite (Eclipse) | 380–400 mT | 230–275 kA/m | 235–290 kA/m | 27.0–32.0 kJ/m³ | same family values | 250–300 °C | [Eclipse ferrite TDS](https://www.eclipsemagnetics.com/site/assets/files/19602/ferrite_ceramic_datasheet.pdf) |

FR-020 AC cross-check for CI: N42 entered as Br 1.315 T, HcB 860 kA/m, HcJ 955 kA/m (Arnold typ/min)
must reproduce those three numbers and (BH)max ≈ 334 kJ/m³ within 2 % from the material model.

### 4.2 Soft materials — only what the sources actually print

- **ARMCO Pure Iron** (soft iron benchmark): density 7.86 g/cm³; electrical resistivity 10.7 μΩ·cm
  (RT); "maximum intrinsic induction (B−μ0H), or a saturation induction value of **2.15 T**". The
  bulletin gives magnetization *curves as bands* (Figs 7/8/10: low/medium induction band, high-
  induction curve 0.5–2.15 T vs 500–5·10⁵ A/m, μr bands after SRA vs normalization anneal) — no
  numeric B–H table. Digitize from the PDF or measure; record anneal state in provenance (the sheet
  shows cold work alone can halve low-field permeability and raise Hc).
  Source: https://www.aksteel.nl/files/downloads/clf_productdata__armco_pure_iron_pdb_euro_final_072022_92.pdf — © 2026 Cleveland-Cliffs, published datasheet.
- **DI-MAX M-19 class electrical steel** (fully processed, ASTM A677): core-loss limit 1.55 W/lb at
  1.5 T / 60 Hz / 29 gauge; the current Cliffs brochure's DC-properties table (coercive force 0.41–
  0.57 Oe, μmax 6500–8000 FP, saturation 20.1–20.8 kG) lists **M-15/M-27/M-36/M-43/M-45/M-47, not
  M-19** — do not attribute family numbers to M-19. For M-19's B–H points, request the vendor's
  per-grade sheet or use the FEMM library entry as a cross-check (Aladdin FPL terms noted in §1).
  Source: https://www.clevelandcliffs.com/_assets/_835429d6b3dea882921737e43e6dea75/clevelandcliffs/db/1190/10494/file/CLF_ProductData_DiMaxM15M47_042026.pdf — © 2026 Cleveland-Cliffs.
- **TEAM 20 benchmark steel** (validation material, FR-072): the problem definition publishes the
  measured B–H curve of the benchmark steel (given to 2.3 T and beyond, measured by permeameter) plus
  the coil (381 turns, 1000/3000/4500/5000 A·t) and published force/flux reference values. Ship the
  curve as a benchmark material, provenance = Compumag TEAM 20.
  Sources: https://www.compumag.org/wp/wp-content/uploads/2018/06/problem20.pdf ;
  https://www.integratedsoft.com/documents/tech_3ax.pdf (Okayama Univ. summary).
- **Low-carbon steel 1018/1020:** no openly licensed tabulated B–H dataset was located in this review
  (searched GitHub, university repositories, published-datasheet routes; MagWeb's BHmag is commercial
  and Scribd copies are user uploads of unknown origin). Ship nothing for it until the owner measures
  a curve (FR-021 override path) or a licensed open source is found; an [UNVERIFIED] placeholder in
  the library would violate FR-022's spirit.
- **General soft-material fills:** powder/soft-ferrite component materials can come from the open
  MIT-licensed OpenMagnetics library or the GPL-3.0 Paderborn database (HF complex permeability —
  neither is a DC B–H source; note that distinction in the record).

## 5) Magpylib 6.3 as the Tier-0 analytic engine

**Identity.** Magpylib 6.3.0 on PyPI (verified 2026-11-17), "Python package for computation of
magnetic fields of magnets, currents and moments"
(https://pypi.org/project/magpylib/). Repo licence file reads "Copyright (c) 2022, Silicon Austria
Labs, Magpylib Developers" with a two-condition redistribution text — **BSD-2-Clause per
https://github.com/magpylib/magpylib/blob/main/LICENSE** (earlier RESEARCH.md recorded BSD-3-Clause
from the index; the LICENSE file in the repo is the binding text — verify the tag you pin against,
the SPDX on GitHub also says BSD-2-Clause). Runtime deps: numpy, scipy, matplotlib, plotly.

**Sources and shapes (v6 object classes, from the docs API page and user guide):**
- Magnets: `Cuboid`, `Cylinder`, `CylinderSegment`, `Sphere`, `Tetrahedron`, `TriangularMesh`
  (complex shapes via `from_ConvexHull` and surface meshes).
- Currents: `Circle` (loop), `Polyline` (line segments), `TriangleSheet`, `TriangleStrip`.
- Misc: `Dipole` (point moment), `CustomSource`; observers: `Sensor`; grouping: `Collection`
  ("An operation applied to a Collection is applied to every object in it. The Collection itself
  behaves like a single source object").
- Top-level: `getB/getH/getJ/getM` (fields), `getFT` (force/torque via a `meshing` numerical scheme),
  `show()` with matplotlib/plotly/pyvista backends, paths (motion sequences, path-varying
  properties), animation, and a loop-free functional interface (`magpylib.func.cuboid_field`).
  Docs: https://magpylib.readthedocs.io/en/stable/_pages/user_guide/guide_start_02_fundamentals.html
  and https://magpylib.readthedocs.io/en/stable/_autogen/magpylib.html

**Superposition and performance.** Bodies carry a uniform polarization (SI by default since v5);
field answers are computed by superposition, and Collections/vectorised `getB` evaluate every object
at every path position in one numpy call — the user guide is explicit: "hand over all field
computation instances … at the same time … Avoid Python loops at all costs !!!" This is exactly the
MagWave NFR-5 posture (10⁴ points ≤ 100 ms via batching, per ARCHITECTURE §8).

**What it explicitly cannot do (and what the docs say about the gap):**
- **No permeable/soft material in the core solver**: magnetization is prescribed, not solved; there is
  no μr and no B–H iteration. The docs' physics page states the residual error honestly: material
  response creates inhomogeneous magnetization, but "modern high-grade permanent magnets (NdFeB, SmCo,
  Ferrite) have very weak material responses (remanent permeability) as low as μr ≈ 1.05"
  (https://magpylib.readthedocs.io/en/stable/_pages/user_guide/guide_resources_01_physics.html).
- **No self-consistent interaction** between sources (bodies do not magnetize each other). Advanced
  workarounds exist *in the docs* — demagnetization-factor corrections, a "material response
  extension" that "enables modeling of self-interactions and inhomogeneous magnetization … supports
  soft-magnetic materials with large permeabilities (μr ~ 1000)", and the method of images for
  magnets near soft plates [extension availability/maturity — UNVERIFIED: described on the physics
  page, no independent release verified].
- **Idealization limits stated by the project itself**: solutions are exact for the idealized inputs,
  but real magnets deviate "a few percent in magnetization amplitude and several degrees in
  magnetization direction", and current distributions are idealized too (physics page, "Accuracy of
  the Solutions").
- Therefore, per FR-041, the analytic engine's `capabilities()` declares no permeable-material
  support; a steel body on Tier 0 must be rejected with a pointer to Tier 1/2, before any number is
  produced.

**Division of labour.** Tier 0 (Magpylib) answers magnet-in-air questions instantly and reproduces
closed forms for CI; Tier 1/2 (FEM with B–H and PM demag curves per §§2–3) answers everything with
iron inside. The materials in this dossier are the *same* records in both tiers — FR-020's data model
must not care which engine consumes them.

---

Compiled from the sources listed per item; fetch-and-verify log for this dossier (all read on
2026-11-17): Arnold Catalog/Recoma/Alnico PDFs, Shin-Etsu N52 PDF, Eclipse NdFeB+ferrite PDFs,
Cleveland-Cliffs DI-MAX + ARMCO PDFs, TEAM 20 definition + INTEGRATED summary PDF, FEMM licence +
hard-mag wiki pages, Magpylib PyPI JSON + GitHub LICENSE + readthedocs (fundamentals, physics, API)
pages, upb-lea/materialdatabase and OpenMagnetics repo metadata, IEC 60404-5 webstore page. Anything
labelled [UNVERIFIED] above was not directly read and must be checked before it enters a material
record or a release note.
