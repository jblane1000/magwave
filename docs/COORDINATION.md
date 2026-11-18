# Coordination — who does what, and what the gates are

MagWave is run as a small project team. This file is the standing arrangement: it says who owns which
work, what each work package must hand over, and what has to be true before anything moves.

**The rule that overrides everything else:** an agent's own summary is not evidence. The coordinator
re-runs every acceptance command and pastes the raw output. A number quoted without a rung
(closed-form · TEAM · second code · measurement) is an opinion, not a result.

## Roster

| Member | Domain | Owns | Does not do |
|---|---|---|---|
| **Max** (coordinator) | architecture, specification | PRD, architecture records, integration, verification, release gate | does not silently rewrite another member's deliverable |
| **Reese** | research | validation sources, TEAM problem data, materials/B–H provenance, landscape refresh | does not write solver or UI code |
| **Cody** | implementation | package skeleton, benchmark harness, Tier 0 engine, Tier 1 kernel | does not merge its own work; the coordinator verifies |
| **Manny** | CAD | import test corpus (clean + deliberately broken STEP assemblies), import-manifest review, datum/frame discipline | does not choose physics methods |
| **Crea** | documents & design | PDF editions, README/docs polish, diagrams, figure styling | does not summarise or reorder source content |
| **Sammy** | delivery | email to inbox/Kindle when asked, receipts | does not compose technical content |

Helen (health) and Vex (trading) are not on this project.

## Work packages

| ID | Package | Owner | Exit criterion |
|---|---|---|---|
| WP-01 | Validation sources dossier: exact closed-form formulas, TEAM problem definitions and published reference values, tolerance rationale | Reese | `docs/research/validation-sources.md` with a citation per formula and per reference value |
| WP-02 | Materials and magnet data: public B–H and demagnetization curve sources, temperature coefficients, licence per source | Reese | `docs/research/materials-bh-data.md`; every dataset carries provenance and licence |
| WP-03 | M1 skeleton + benchmark harness: case model, units, CLI, harness that fails on a 1 % perturbation | Cody | CI green; harness demonstrated by deliberately breaking one expected value |
| WP-04 | Tier 0 engine: magnets, coils, sampling, forces, export | Cody | FR-030/031/032/050/052 satisfied against WP-01 numbers |
| WP-05 | Desktop UI v1: open case, assign roles, run, slice/glyph/streamline, report export | Cody | a new user completes the README walkthrough unassisted |
| WP-06 | CAD import corpus + manifest: three clean assemblies, three defective ones (open shell, interference, mixed units) | Manny | manifest reports every defect; no body dropped silently |
| WP-07 | Document editions: A4 and Kindle/A5 PDFs from one source | Crea | both editions render from one source with identical figures |
| WP-08 | Delivery: Kindle and inbox sends with hash-verified attachments | Sammy | recipient, message id, downloaded size and sha256 reported |

## Gates

- **G0 — Phase 0 review (Jason).** Passed when he has read the PRD and said to proceed. Only Phase 0
  documents exist until then.
- **G1 — harness green.** Benchmark suite runs in CI, tolerance-tagged, anti-vacuity proven.
- **G2 — Tier 0 acceptance.** PRD §9 satisfied with CI evidence.
- **G3 — UI walkthrough.** FR-061 demonstrated by someone who did not write it.
- **G4 — CAD manifest.** PRD §6.2 satisfied; the broken-assembly regression test exists.
- **G5 — v0.1 release.** Release checklist completed; licence gate clean; offline test passes.

## Working rules

1. **One item at a time.** Each change is verified before the next begins; a fix is not "done" because
   it was written.
2. **Every number names its rung.** Closed-form, TEAM, second code, or measurement — no exceptions.
3. **Nothing merges on the strength of a summary.** The coordinator re-runs the acceptance command.
4. **No silent drops.** A body, material or boundary that cannot be honoured is reported, never skipped.
5. **Licence boundary is physical.** `gmsh` and Elmer are subprocesses; the CI gate enforces it.
6. **Specs are the source of truth.** Requirement IDs (FR-###, NFR-#) are never renumbered; a withdrawn
   requirement is marked *Withdrawn* with its date.
7. **Every implementation brief names its engine — and its tier.** Cody's implementation work is
   written by `agy`, not by Cody's own model. A brief to Cody says so explicitly — "implement via an
   `agy-work` brief; you write the brief, run it, and verify against the acceptance command" — because
   an advisory rule that is not in the dispatch does not fire: on 2026-11-18 an M1 job ran 66 tool
   calls and 184k tokens of Cody's own authorship with zero `agy` calls. Cody may still patch directly
   for a one-line repair or to fix `agy`'s own output, and says which engine produced the code.
8. **Tier policy (set 2026-11-18).** `agy` is free to pick the tier, but the default is fixed by job
   class: **implementation-class work → `gemini-pro-orchestrator-g1-pro`** (Argon plans and reviews,
   Hydrogen implements; the default for `agy-work` since this date); **small well-specified jobs →
   `gemini-4-hydrogen-high`** (`xhigh` when gnarly); **checks and reviews → Claude**
   (`claude-sonnet-5-5-high` quick, `claude-opus-5-5-high` deep), the scarce pool, spent on
   verification rather than authoring; **hard design or debugging → `gemini-4-argon-high`** or
   Orchestrator. Measured cost of the change: one M3-class feature took 8m43s and **2 Argon quota
   points** under Orchestrator versus **47 Claude weekly points** under Opus.

## Cadence

- Research and design: asynchronously, one dossier at a time, reviewed before anything depends on it.
- Implementation: small, verifiable increments; CI must be green at each push.
- Reporting: the coordinator reports to Jason with evidence, not with activity.
