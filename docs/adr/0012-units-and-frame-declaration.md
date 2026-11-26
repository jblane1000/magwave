# ADR 0012 — Case unit system and frame declaration (`magwave.case/2`, `FR-003`)

- Status: accepted
- Date: 2026-11-25
- Deciders: Jason (owner)
- Amends: ADR-0005 (`Case` validation-first contract) — promotes boundary units and symmetry axis from implicit assumptions to required, validated case fields
- Supersedes nothing

## Context

`magwave.case/1` (`src/magwave/cases.py`) carried no unit declaration and no frame declaration: geometry was implicitly metres, magnetic flux density was implicitly tesla, forces were implicitly newtons, and rotational symmetry was implicitly about the `z` axis. Worse, `ExpectedValue.__post_init__` checked `expected[].unit` against a global SI set (`{"B": "T", "F": "N"}`), so a document could not declare millimetre or millitesla inputs and nothing cross-checked expected units against a case-level unit block.

PRD FR-003 requires that every case file declare its unit system and reference frame explicitly in the file itself, and that mixed or unsupported units fail with a clear error naming the offending JSON field. PRD NFR-2 remains unchanged: all solvers and internal dataclasses store SI units (`m`, `T`, `A`, `N`), converting strictly at the JSON boundary.

All measured float round-trip numbers below were recorded on 2026-11-25 in this checkout using `env -u PYTHONPATH .venv/bin/python` (Python 3.12.3).

## Decision

### 1. Schema bump to `magwave.case/2` (no dual reader)

`SCHEMA` in `src/magwave/cases.py` is bumped from `"magwave.case/1"` to `"magwave.case/2"`. There is no dual reader: all five files under `examples/` and all 27 cases in `src/magwave/benchmarks/cases.json` (under top-level `"schema": "magwave.benchmarks/1"`) are migrated with `"units": {"current": "A", "field": "T", "force": "N", "length": "m"}` and `"frame": {"axis": "z"}`, leaving every numeric value unchanged.

- **Refuse `magwave.case/1` on `case.schema`.** `Case.from_dict` validates `case.schema` before checking missing `units` or `frame` keys, so a `magwave.case/1` document is refused with `case.schema: expected 'magwave.case/2', got 'magwave.case/1'`. -- enforced-by: tests/test_units_declaration.py::test_schema_v1_document_refused_with_expected_schema_error
- **Shipped examples and benchmark corpus migrate without numeric drift.** All five `examples/*.json` files and all 27 cases in `src/magwave/benchmarks/cases.json` declare `magwave.case/2` in SI (`m`, `T`, `A`, `N`, `axis: z`) and round-trip byte-identically. -- enforced-by: tests/test_units_declaration.py::test_examples_and_corpus_roundtrip_byte_identically

### 2. Required `units` block (`UnitSpec`) and why inches are refused

`src/magwave/cases.py` adds a frozen `UnitSpec(length="m", field="T", current="A", force="N")` dataclass. Supported unit sets are intentionally restricted to exact power-of-ten factors:

- `units.length`: `"m"` (`10^0`), `"mm"` (`10^-3`)
- `units.field`: `"T"` (`10^0`), `"mT"` (`10^-3`)
- `units.current`: `"A"` (`10^0`)
- `units.force`: `"N"` (`10^0`), `"mN"` (`10^-3`)

Inches (`"in"`) and all other non-power-of-ten units are deliberately out of scope and refused. Measured reason: in IEEE-754 float64, `(3.33 * 0.0254) / 0.0254 == 3.3300000000000005`, so an inch-declared case cannot round-trip byte-identically through SI conversion. If exactness cannot be guaranteed for a unit, the unit is refused rather than silently converted.

- **Missing `units` and unsupported unit names fail with exact JSON paths.** Omitting `units` raises `CaseError` naming `case.units`; any unknown or unsupported unit name (such as `length: "in"`) fails naming `case.units.<key>` (or `case.expected[i].unit`) and listing supported units. -- enforced-by: tests/test_units_declaration.py::test_unsupported_units_refused_with_path_and_supported_list
- **Missing `case.units` and `case.frame` keys are refused on documents.** `Case.from_dict` requires both `units` and `frame` on every JSON document, naming `case.units` or `case.frame` when absent. -- enforced-by: tests/test_units_declaration.py::test_missing_units_and_missing_frame_rejected_with_path

### 3. Exact power-of-ten boundary conversion (`Decimal.scaleb`) and scaling scope

Measured across 50,000 pseudo-random rounded decimal values (`round(uniform, 0..6 dp)`) and 50,000 15-significant-digit floats: raw binary float scaling `(v * 1e-3) / 1e-3` fails round-trip equality on `923` of `50,000` rounded values and `591` of `50,000` 15-digit values (for example `(-1015.82 * 1e-3) / 1e-3 == -1015.8200000000002`, and `1015.82 * 1e-3 == 1.0158200000000002 != 1.01582`), whereas `float(Decimal(str(v)).scaleb(exp10))` produces `0` mismatches across both 50,000-sample sets and maps `1015.82 mm` to the exact float64 `1.01582 m`. When `exp10 == 0` (`m`, `T`, `A`, `N`), values pass through untouched without decimal conversion.

Boundary scaling applies exclusively to quantities whose dimension matches a declared `units` key:

- Scaled by `units.length`: component `dims` and `center`, probe `point`, and `mesh.element_size`.
- Scaled by `units.field`: magnet `br` and `expected[].value` for field (`B`) quantities.
- Scaled by `units.force`: `expected[].value` for force (`F`) quantities.
- Not scaled (always SI): `coil.current` (always `A`), `uniform_field.h0` (`A/m` — no field-strength unit key exists), and `dipole.moment` (`A m^2` — not a length).

- **Byte-identical round-trip for declared `mm`/`mT`/`mN` cases.** A seeded corpus of cases declared in `mm`/`mT`/`mN` converts to SI on load and re-serialises byte-identically via `Case.from_json(text).to_json() == text`. -- enforced-by: tests/test_units_declaration.py::test_mm_mt_mn_seeded_property_roundtrip_is_byte_identical
- **Load-bearing SI boundary conversion.** Identical physical magnets declared in `m`/`T`, `mm`/`T`, and `mm`/`mT` produce bit-identical sampled magnetic fields (`struct.pack("!d", ...)` equality). -- enforced-by: tests/test_units_declaration.py::test_units_declaration_is_load_bearing_bit_identical_field
- **Non-scaled SI fields remain untouched under `mm`.** `coil.current` (A), `uniform_field.h0` (A/m), and `dipole.moment` (A m^2) are never scaled by `units.length` and round-trip unchanged. -- enforced-by: tests/test_units_declaration.py::test_non_scaled_quantities_stay_si_under_mm_units

### 4. Mixed-unit cross-check moved from `ExpectedValue` to `Case`

`ExpectedValue.__post_init__` retains only a structural check against `SUPPORTED_EXPECTED_UNITS` (`N`, `T`, `mN`, `mT`). `Case.__post_init__` and `Case.from_dict` cross-check every `expected[i]` against `case.units.field` (for `B` quantities) and `case.units.force` (for `F` quantities), reporting `case.expected[i].unit: '<got>' contradicts case.units.<field|force> = '<declared>'`.

- **Mixed units fail naming the exact JSON path and both unit strings.** Contradicting `case.units.field` or `case.units.force` in `expected[i].unit`, or supplying a non-finite/non-numeric/non-positive geometry value, raises `CaseError` naming the exact JSON path. -- enforced-by: tests/test_units_declaration.py::test_mixed_units_rejected_with_exact_path_and_both_units

### 5. Required `frame` block (`FrameSpec`) without silent axis rotation

`src/magwave/cases.py` adds a frozen `FrameSpec(axis="z", datum_face="")` dataclass. `axis` must be one of `("x", "y", "z")` and only `"z"` is accepted by Tier 1; `"x"` and `"y"` are refused with `CaseError` naming `case.frame.axis` rather than silently rotating or ignoring the declaration (NFR-8). `datum_face` is optional free text omitted from `FrameSpec.to_dict()` when empty (matching `Case.temperature_c`).

`Case` keeps `units: UnitSpec = field(default_factory=UnitSpec)` and `frame: FrameSpec = field(default_factory=FrameSpec)` as trailing defaults for programmatic Python callers, while `Case.from_dict` strictly requires both blocks on all JSON documents.

- **Only `axis: "z"` is accepted; `x`/`y` are refused without rotation and empty `datum_face` is omitted.** Declaring `x` or `y` raises `CaseError` naming `case.frame.axis`; `axis: "z"` omits empty `datum_face` on serialisation and preserves non-empty `datum_face` losslessly. -- enforced-by: tests/test_units_declaration.py::test_frame_axis_validation_and_datum_face_roundtrip

## Consequences

- Every case file on disk is self-describing in units (`m`/`mm`, `T`/`mT`, `A`, `N`/`mN`) and symmetry frame (`axis: "z"`), closing the silent-units and silent-axis NFR-8 gap.
- Adding a new length/field/force prefix (for example `um`) later only requires adding its exact power-of-ten exponent to the `SUPPORTED_*_UNITS` table without changing internal solver code.
- Supporting `axis: "x"` or `"y"` in a future solver tier will require explicit coordinate transforms in that solver and updating `SUPPORTED_FRAME_AXES` rather than touching `magwave.case/2` documents.
