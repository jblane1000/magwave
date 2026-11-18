# MagWave

Open-source 3D magnetic field simulator for **permanent magnets, coils and — later — steel**.

MagWave is built for the case the existing tools get wrong: you have a **CAD assembly**, you need to know
the field, and you do not want to spend five figures or a weekend on solver input files.

> **Status: Phase 0 — specification.** This repository currently contains the research dossier, the PRD,
> the architecture, the decision records and the repository scaffolding. There is no solver code yet.
> The first code lands with milestone M1 (see [docs/ROADMAP.md](docs/ROADMAP.md)).

## Why this exists

In roughly 2022 the author's employer evaluated a commercial 3D magnetostatics package over a video
demo, sending one assembly STEP file as the test piece. The demo produced a picture and no recoverable
number; the tool was never purchased. Four years later there is still nothing to re-run.

That is the failure MagWave is designed against: a result you cannot re-run is not a result. Every case
is a versioned file, every number is written to disk, and every claim in the docs names the benchmark it
rests on.

## Scope, honestly

| Tier | Answers | Cannot answer | Status |
|---|---|---|---|
| **0 — analytic** | field of block/cylinder/ring magnets and coils anywhere, instantly; forces between magnets | anything containing permeable (steel) material | first milestone |
| **1 — 2D / axisymmetric FEM** | steel, saturation, forces on round or constant-section parts: solenoids, pot magnets, magnetizing fixtures, speakers | arbitrary 3D assemblies | second |
| **2 — 3D FEM** | arbitrary assemblies with iron, saturation, forces | speed of a 40-year commercial BEM kernel | later target |
| **3 — transient / eddy** | capacitor-discharge pulses, eddy currents, motion | — | research |

Tier 0 cannot answer a magnetizing fixture, because a fixture has a steel body. Tier 0 exists to prove
the whole pipeline — CAD view, materials, results, report, tests — cheaply and exactly, so that the
first real solver drops into a tested frame.

## Method is chosen by the physics, not by taste

- **Analytic** — canonical shapes only. Exact, instant, no mesh, blind to permeable material.
- **FEM** (volume mesh) — nonlinear steel, arbitrary geometry. Needs a bounded air region.
- **BEM / integral** — open regions and exact boundaries come free; nonlinear iteration is the hard part.
  This is why commercial magnetostatics is BEM.

## Validation is the product

No feature is "done" because it looks right. Each claim is tied to one of four rungs:

1. **Closed-form analytics** — dipole, loop, finite solenoid (elliptic integrals), magnetized cuboid and
   cylinder.
2. **Published benchmarks** — Compumag **TEAM** problems with cross-code results in the open literature
   (**TEAM 20** is the 3D static-force problem).
3. **An independent second code** agreeing on the same geometry to a stated tolerance.
4. **A calibrated Hall probe** on a real part.

See [docs/CHECKLISTS/validation.md](docs/CHECKLISTS/validation.md).

## Licence

**Apache-2.0.** Note the deliberate boundary: `gmsh` and Elmer are **GPL** and are only ever invoked as
external processes — never linked, never bundled. Qt (PySide6) is LGPL and is dynamically linked.
See [docs/adr/0002-licence-strategy.md](docs/adr/0002-licence-strategy.md).

## Documentation

- [docs/RESEARCH.md](docs/RESEARCH.md) — the tool landscape, the priced reference, verified licences.
- [docs/PRD.md](docs/PRD.md) — requirements with IDs and acceptance criteria; the parity matrix.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — layers, the engine protocol, the case file.
- [docs/ROADMAP.md](docs/ROADMAP.md) — milestones M0–M7.
- [docs/adr/](docs/adr/) — decision records (stack, licence, engines, UI, validation).
- [docs/CHECKLISTS/](docs/CHECKLISTS/) — release, validation, mesh quality, CAD import.

## Quickstart

This walkthrough uses the shipped point-dipole example (`examples/solve-dipole.json`) and runs offline from a fresh checkout.

1. **Create a virtual environment and install MagWave with the `gui` extra:**

   ```bash
   git clone https://github.com/jblane1000/magwave.git
   cd magwave
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[gui]"
   ```

2. **Open the shipped example case in the desktop window:**

   ```bash
   magwave gui examples/solve-dipole.json
   ```

   In the window you will see:
   - **Case**: `dipole_axis_Bz` (`examples/solve-dipole.json`)
   - **Provenance (Engine & Rung)**: `Engine ID: magpylib`, the installed `Engine Version`, `Rung: second-code`, and the solver `Method`
   - **Components / Role Assignment**: `m1` (`dipole`, `moment=(0.0, 0.0, 1.0) A m^2`) — switching a component's kind works where `magwave.case/1` permits it (e.g. magnet `cuboid` ↔ `cylinder`, coil `loop` ↔ `solenoid`); unsupported assignments such as `steel (permeable)` report a blocking explanation in **Validation & Solver Problems**
   - **Probes**: `probe #0` at `(0.0, 0.0, 0.5)` m requesting `B`, component `z` (`T`)
   - **Expected Values**: check `dipole_B`, `Bz = 1.6e-06 T`, tolerance `1e-09`

3. **Run the solve:** click **Run** (or press `F5`). The engine runs `validate()` first and then solves `probe #0`, showing `1.6000000000000004e-06` `T` under `rung: second-code`.

4. **Read the 3D Field View (PyVista):**
   - **What it shows**: 3D component geometry (a sphere for a dipole, box/cylinder for magnets, circle/cylinder for coils), probe points coloured by the solved scalar (`B (T)`), and 3D `B`-vector arrow glyphs at each solved probe point.
   - **What it does not show**: slice planes, iso-surfaces, and field streamlines are not implemented in v1. In headless environments (`QT_QPA_PLATFORM=offscreen` or no display server), PyVista constructs the scene datasets in memory without claiming a live 3D render canvas.

5. **Export the Markdown report:** click **Export Report...** (or press `Ctrl+E`) and save to the default filename `dipole_axis_Bz-report.md` (headless CLI equivalent: `magwave report examples/solve-dipole.json --out dipole_axis_Bz-report.md`).

   Open `dipole_axis_Bz-report.md` (for example with `cat dipole_axis_Bz-report.md`). You should see the provenance header (`engine: magpylib`, `rung: second-code`), the fenced JSON summary at the bottom, and this probe table line:

   ```text
   | probe #0 (0.0, 0.0, 0.5) m | B | z | 1.6000000000000004e-06 | T | 1.6e-06 | 1e-09 | PASS |
   ```

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
ruff check . && ruff format --check .
```

The GUI and CAD extras are optional and heavier:

```bash
uv pip install -e ".[dev,gui,cad]"
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bring a benchmark, not an opinion.
