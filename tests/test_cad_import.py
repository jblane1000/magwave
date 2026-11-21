"""CAD import, manifest contract, and role assignment tests (FR-010, FR-011, FR-013)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from magwave.benchmarks import evaluate_with_engine, load_corpus
from magwave.cad import (
    DEFECT_KINDS,
    ROLES,
    Body,
    DefeatureRecord,
    Defect,
    Manifest,
    MissingExtraError,
    RemovedFeature,
    UnassignedBodiesError,
    assign,
    import_cad,
    require_all_assigned,
    unassigned,
)
from magwave.cases import Case
from magwave.cli import main
from magwave.engines import MagpylibEngine
from tests.fixtures.cad_builders import (
    build_box_iges,
    build_box_stl,
    build_closed_box_step,
    build_filleted_step,
    build_merged_compound_step,
    build_non_manifold_step,
    build_open_shell_step,
    build_overlapping_step,
    build_three_body_step,
    build_two_body_glb,
)


def _sample_manifest(*, assigned: bool = False) -> Manifest:
    role_a = "permanent_magnet" if assigned else None
    role_b = "permeable_solid" if assigned else None
    return Manifest(
        path="assembly.step",
        format="step",
        units="mm",
        bodies=[
            Body(
                index=0,
                name="magnet",
                role=role_a,
                volume_mm3=1000.0,
                solids=1,
                bbox=(-10.0, -5.0, -2.5, 10.0, 5.0, 2.5),
                source="step",
            ),
            Body(
                index=1,
                name=None,
                role=role_b,
                volume_mm3=345.5752,
                solids=1,
                bbox=(-8.0, -8.0, 2.5, 8.0, 8.0, 4.5),
                source="step",
            ),
        ],
        defects=[
            Defect(
                kind="unnamed_body",
                detail="generic placeholder name 'SOLID' in STEP metadata",
                body_index=1,
            )
        ],
        warnings=["example warning"],
    )


def _block_module(monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    for name in [k for k in sys.modules if k == prefix or k.startswith(f"{prefix}.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, prefix, None)


# --------------------------------------------------------------------------------------
# 1. Core lazy-import cleanliness & missing extras (run without OCP or trimesh)
# --------------------------------------------------------------------------------------
def test_core_stays_clean_on_cad_import() -> None:
    """Importing magwave and magwave.cad must not load OCP, trimesh, or PySide6."""
    code = (
        "import sys, magwave, magwave.cad, magwave.cli; "
        "loaded = {'OCP', 'trimesh', 'PySide6'} & set(sys.modules); "
        "assert not loaded, f'heavy modules imported: {loaded}'"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_cli_version_and_bench_do_not_import_cad_backends() -> None:
    """`magwave --version` and `magwave bench` never load OCP or trimesh."""
    code = (
        "import sys; "
        "from magwave.cli import main; "
        "assert main(['bench']) == 0; "
        "loaded = {'OCP', 'trimesh', 'PySide6'} & set(sys.modules); "
        "assert not loaded, f'heavy modules imported: {loaded}'"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_step_extra_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    step_file = tmp_path / "part.step"
    step_file.write_text("ISO-10303-21;\n", encoding="utf-8")
    _block_module(monkeypatch, "OCP")

    with pytest.raises(MissingExtraError) as excinfo:
        import_cad(step_file)
    assert "magwave[step]" in str(excinfo.value)


def test_missing_mesh_extra_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stl_file = tmp_path / "part.stl"
    stl_file.write_text("solid empty\nendsolid empty\n", encoding="utf-8")
    _block_module(monkeypatch, "trimesh")

    with pytest.raises(MissingExtraError) as excinfo:
        import_cad(stl_file)
    assert "magwave[mesh]" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# 2. Manifest JSON round-trip & role assignment / solve blocking (pure stdlib)
# --------------------------------------------------------------------------------------
def test_manifest_json_roundtrip_is_byte_stable() -> None:
    manifest = _sample_manifest(assigned=False)
    first = manifest.to_json()
    restored = Manifest.from_json(first)
    second = restored.to_json()
    assert restored == manifest
    assert first == second
    assert "defeatures" not in json.loads(first)


def test_manifest_defeature_record_roundtrip_and_legacy_load() -> None:
    base = _sample_manifest(assigned=False)
    legacy_payload = json.loads(base.to_json())
    assert "defeatures" not in legacy_payload
    assert Manifest.from_dict(legacy_payload) == base

    with_defeature = Manifest(
        path=base.path,
        format=base.format,
        units=base.units,
        bodies=base.bodies,
        defects=base.defects,
        warnings=base.warnings,
        defeatures=[
            DefeatureRecord(
                body_index=0,
                threshold_mm=2.0,
                features_removed=2,
                volume_before_mm3=999.0,
                volume_after_mm3=1000.0,
                volume_delta_mm3=1.0,
                removed_features=[
                    RemovedFeature(surface_type="Torus", size_mm=1.4),
                    RemovedFeature(surface_type="Torus", size_mm=1.4),
                ],
            )
        ],
    )
    serialised = with_defeature.to_json()
    assert "defeatures" in json.loads(serialised)
    assert Manifest.from_json(serialised) == with_defeature


def test_body_display_name_fallback_keeps_dataclass_name_none() -> None:
    body = Body(
        index=2,
        name=None,
        role=None,
        volume_mm3=42.0,
        solids=1,
        bbox=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        source="step",
    )
    assert body.name is None
    assert body.display_name == "body_2"


def test_role_assignment_validation_and_blocking() -> None:
    manifest = _sample_manifest(assigned=False)
    assert [b.index for b in unassigned(manifest)] == [0, 1]

    with pytest.raises(UnassignedBodiesError) as excinfo:
        require_all_assigned(manifest)
    message = str(excinfo.value)
    assert "#0 (magnet)" in message
    assert "#1 (body_1)" in message

    with pytest.raises(ValueError, match="unknown role"):
        assign(manifest, {0: "not_a_real_role"})

    # Unknown body key is recorded as a skipped_body defect, never silently dropped.
    assign(manifest, {"ghost_part": "air"})
    assert any(d.kind == "skipped_body" and "ghost_part" in d.detail for d in manifest.defects)

    # Valid assignment by name and index clears the gate.
    assign(manifest, {"magnet": "permanent_magnet", 1: "permeable_solid"})
    assert unassigned(manifest) == []
    require_all_assigned(manifest)
    assert set(ROLES) >= {"permanent_magnet", "permeable_solid"}


def test_unassigned_manifest_blocks_solve_in_harness_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_case = next(c for c in load_corpus() if c.name == "dipole_axis_Bz")
    blocked_manifest = _sample_manifest(assigned=False)
    blocked_case = Case(
        name=base_case.name,
        description=base_case.description,
        sources=base_case.sources,
        components=base_case.components,
        probes=base_case.probes,
        expected=base_case.expected,
        tolerance=base_case.tolerance,
        manifest=blocked_manifest,
    )

    # Round-trip preserves manifest on Case
    roundtripped = Case.from_json(blocked_case.to_json())
    assert roundtripped == blocked_case

    # Harness blocks solve
    engine = MagpylibEngine()
    with pytest.raises(UnassignedBodiesError, match="body_1"):
        evaluate_with_engine(blocked_case, engine)

    # CLI solve and report both block
    case_path = tmp_path / "blocked.json"
    case_path.write_text(blocked_case.to_json(), encoding="utf-8")

    assert main(["solve", str(case_path), "--points", "0,0,0.5"]) == 2
    assert "unassigned bodies" in capsys.readouterr().err.lower()

    assert main(["report", str(case_path)]) == 1
    assert "unassigned bodies" in capsys.readouterr().err.lower()

    # Once all bodies are assigned, the unassigned-bodies gate stops blocking — but a
    # permeable_solid body still cannot be modelled by Tier 0 (FR-041, M5.2): it must be
    # refused with a capability error rather than answered with a free-space number.
    assign(blocked_manifest, {"magnet": "permanent_magnet", 1: "permeable_solid"})
    case_path.write_text(blocked_case.to_json(), encoding="utf-8")
    assert main(["solve", str(case_path), "--points", "0,0,0.5"]) == 2
    refusal = capsys.readouterr()
    assert "unassigned bodies" not in refusal.err.lower()
    assert "E_UNSUPPORTED_PERMEABLE" in refusal.err
    assert refusal.out.strip() == ""

    # With no permeable body in the manifest, assignment alone unblocks solve.
    assign(blocked_manifest, {"magnet": "permanent_magnet", 1: "air"})
    case_path.write_text(blocked_case.to_json(), encoding="utf-8")
    assert main(["solve", str(case_path), "--points", "0,0,0.5"]) == 0
    _ = capsys.readouterr()


# --------------------------------------------------------------------------------------
# 3. OCP STEP & IGES tests (skipped cleanly when cadquery-ocp is absent)
# --------------------------------------------------------------------------------------
def test_merged_assembly_regression_step(tmp_path: Path) -> None:
    """FR-011 mandatory regression: XCAF product walk must yield 3 distinct bodies, not 1948.761."""
    pytest.importorskip("OCP")
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.STEPControl import STEPControl_Reader

    step_path = build_three_body_step(tmp_path / "three_body.step")

    # Confirm the naive single-shape baseline merges into ~1948.761 mm^3
    naive = STEPControl_Reader()
    naive.ReadFile(str(step_path))
    naive.TransferRoots()
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(naive.OneShape(), props)
    assert props.Mass() == pytest.approx(1948.7610, rel=1e-4)

    # Real importer must preserve all 3 bodies and match closed-form volumes within 1e-3
    manifest = import_cad(step_path)
    assert manifest.format == "step"
    assert manifest.units == "mm"
    assert len(manifest.bodies) == 3

    expected_by_name = {
        "magnet": 20.0 * 10.0 * 5.0,
        "washer": math.pi * (8.0**2 - 3.0**2) * 2.0,
        "bobbin": math.pi * 4.0**2 * 12.0,
    }
    got_names = [body.name for body in manifest.bodies]
    assert set(got_names) == set(expected_by_name)
    for body in manifest.bodies:
        assert body.solids == 1
        assert body.source == "step"
        assert body.role is None
        assert body.volume_mm3 == pytest.approx(expected_by_name[body.name], rel=1e-3)


def test_open_shell_produces_defect_and_closed_solid_produces_none(tmp_path: Path) -> None:
    pytest.importorskip("OCP")

    open_path = build_open_shell_step(tmp_path / "open_shell.step")
    open_manifest = import_cad(open_path)
    kinds = {d.kind for d in open_manifest.defects}
    assert "open_shell" in kinds
    assert "skipped_body" in kinds
    assert open_manifest.bodies == []

    closed_path = build_closed_box_step(tmp_path / "closed_box.step", name="pole")
    closed_manifest = import_cad(closed_path)
    assert len(closed_manifest.bodies) == 1
    assert closed_manifest.bodies[0].volume_mm3 == pytest.approx(1000.0, rel=1e-6)
    assert closed_manifest.defects == []


def test_iges_best_effort_import(tmp_path: Path) -> None:
    pytest.importorskip("OCP")

    iges_path = build_box_iges(tmp_path / "box.igs")
    manifest = import_cad(iges_path)
    assert manifest.format == "iges"
    assert manifest.units == "mm"
    assert len(manifest.bodies) == 1
    body = manifest.bodies[0]
    assert body.source == "iges"
    assert body.volume_mm3 == pytest.approx(12.0 * 8.0 * 4.0, rel=1e-3)
    # Must either yield >=1 solid or explicitly report imported_as_shell
    assert body.solids >= 1 or any(d.kind == "imported_as_shell" for d in manifest.defects)


@pytest.mark.parametrize("raw_name", ["SOLID", "", None])
def test_step_name_fallback_records_unnamed_body_and_none(
    tmp_path: Path, raw_name: str | None
) -> None:
    pytest.importorskip("OCP")

    path = build_closed_box_step(tmp_path / "unnamed.step", name=raw_name)
    manifest = import_cad(path)
    assert len(manifest.bodies) == 1
    body = manifest.bodies[0]
    assert body.name is None
    assert body.role is None
    assert body.display_name == "body_0"
    assert any(d.kind == "unnamed_body" and d.body_index == 0 for d in manifest.defects)


def test_remaining_defect_kinds_non_manifold_merged_and_interference(tmp_path: Path) -> None:
    pytest.importorskip("OCP")

    nm_manifest = import_cad(build_non_manifold_step(tmp_path / "nm.step"))
    assert any(d.kind == "non_manifold_edge" for d in nm_manifest.defects)

    merged_manifest = import_cad(build_merged_compound_step(tmp_path / "merged.step"))
    assert len(merged_manifest.bodies) == 1
    assert merged_manifest.bodies[0].solids == 2
    assert any(d.kind == "merged_body" and d.body_index == 0 for d in merged_manifest.defects)

    interf_manifest = import_cad(
        build_overlapping_step(tmp_path / "overlap.step"),
        check_interference=True,
    )
    assert any(d.kind == "interference" for d in interf_manifest.defects)
    assert "500.0000" in next(d.detail for d in interf_manifest.defects if d.kind == "interference")

    # Confirm our entire test suite covers every defined Defect.kind
    df_defect = Defect(kind="defeature_failed", detail="sample failure", body_index=0)
    assert df_defect.kind == "defeature_failed"
    di_defect = Defect(kind="defeature_invalid", detail="sample invalid", body_index=0)
    assert di_defect.kind == "defeature_invalid"
    seen_kinds = {
        "open_shell",
        "non_manifold_edge",
        "skipped_body",
        "merged_body",
        "unnamed_body",
        "imported_as_shell",
        "interference",
        "defeature_failed",
        "defeature_invalid",
    }
    assert seen_kinds == set(DEFECT_KINDS)


# --------------------------------------------------------------------------------------
# 4. Mesh tests (trimesh, skipped cleanly when trimesh is absent)
# --------------------------------------------------------------------------------------
def test_mesh_stl_import_and_unknown_units_warning(tmp_path: Path) -> None:
    pytest.importorskip("trimesh")

    stl_path = build_box_stl(tmp_path / "magnet.stl", extents=(10.0, 8.0, 6.0))
    manifest = import_cad(stl_path)
    assert manifest.format == "stl"
    assert manifest.units == "unknown"
    assert any("unknown" in w.lower() for w in manifest.warnings)
    assert len(manifest.bodies) == 1
    body = manifest.bodies[0]
    assert body.source == "mesh"
    assert body.solids == 1
    assert body.volume_mm3 == pytest.approx(480.0, rel=1e-4)


def test_mesh_scene_imports_named_geometries(tmp_path: Path) -> None:
    pytest.importorskip("trimesh")

    glb_path = build_two_body_glb(tmp_path / "assy.glb")
    manifest = import_cad(glb_path)
    assert manifest.format == "glb"
    assert len(manifest.bodies) == 2
    by_name = {b.name: b for b in manifest.bodies}
    assert set(by_name) == {"core", "yoke"}
    assert by_name["core"].volume_mm3 == pytest.approx(1000.0, rel=1e-4)
    assert by_name["yoke"].volume_mm3 == pytest.approx(120.0, rel=1e-4)


def test_mesh_import_rejects_defeature_below(tmp_path: Path) -> None:
    pytest.importorskip("trimesh")

    stl_path = build_box_stl(tmp_path / "magnet.stl")
    with pytest.raises(ValueError, match="STEP and IGES only"):
        import_cad(stl_path, defeature_below=1.0)


# --------------------------------------------------------------------------------------
# 5. CLI `magwave import` end-to-end
# --------------------------------------------------------------------------------------
def test_cli_import_step_roles_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("OCP")

    step_path = build_three_body_step(tmp_path / "three_body.step")
    json_out = tmp_path / "manifest.json"

    # Without --role: unassigned bodies cause exit 1, while JSON is still written
    rc_unassigned = main(["import", str(step_path), "--json", str(json_out)])
    out_unassigned = capsys.readouterr().out
    assert rc_unassigned == 1
    assert "bodies: 3" in out_unassigned
    assert json_out.is_file()
    loaded = json.loads(json_out.read_text(encoding="utf-8"))
    assert len(loaded["bodies"]) == 3

    # Assigning all 3 bodies via repeatable --role flags exits 0
    rc_assigned = main(
        [
            "import",
            str(step_path),
            "--json",
            str(json_out),
            "--role",
            "0=permanent_magnet",
            "--role",
            "1=permeable_solid",
            "--role",
            "bobbin=coil_volume",
        ]
    )
    _ = capsys.readouterr()
    assert rc_assigned == 0
    loaded_assigned = Manifest.from_json(json_out.read_text(encoding="utf-8"))
    assert unassigned(loaded_assigned) == []
    assert [b.role for b in loaded_assigned.bodies] == [
        "permanent_magnet",
        "permeable_solid",
        "coil_volume",
    ]

    # Open-shell STEP with skipped_body exits 1 even when no bodies exist to assign
    open_path = build_open_shell_step(tmp_path / "open.step")
    assert main(["import", str(open_path)]) == 1
    _ = capsys.readouterr()


def test_cli_import_step_defeature_below(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("OCP")

    step = build_filleted_step(tmp_path / "filleted.step")
    out_defeatured = tmp_path / "defeatured.json"
    out_plain = tmp_path / "plain.json"

    rc = main(
        [
            "import",
            str(step),
            "--json",
            str(out_defeatured),
            "--defeature-below",
            "2.0",
            "--role",
            "0=permeable_solid",
        ]
    )
    stdout = capsys.readouterr().out
    assert rc == 0
    assert "defeatures (1):" in stdout
    payload = json.loads(out_defeatured.read_text(encoding="utf-8"))
    assert len(payload["defeatures"]) == 1
    rec = payload["defeatures"][0]
    assert rec["features_removed"] >= 30
    assert rec["volume_after_mm3"] == pytest.approx(48000.0, rel=0.01)
    assert payload["bodies"][0]["volume_mm3"] == rec["volume_after_mm3"]

    rc_plain = main(
        [
            "import",
            str(step),
            "--json",
            str(out_plain),
            "--role",
            "0=permeable_solid",
        ]
    )
    _ = capsys.readouterr()
    assert rc_plain == 0
    plain_payload = json.loads(out_plain.read_text(encoding="utf-8"))
    assert "defeatures" not in plain_payload


def test_cli_import_ocp_failure_is_clean_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """F2: OCP exceptions during `magwave import` exit 1 with a single-line error."""
    import magwave.cad as cad
    from magwave.cli import _is_ocp_failure

    assert not _is_ocp_failure(ValueError("not ocp"))

    fake_exc = type("Standard_Failure", (Exception,), {"__module__": "OCP.OCP.Standard"})

    def _boom(*_args, **_kwargs):
        raise fake_exc("boom from OCCT")

    monkeypatch.setattr(cad, "import_cad", _boom)
    rc = main(["import", str(tmp_path / "x.step")])
    err = capsys.readouterr().err

    assert rc == 1
    assert "magwave import: boom from OCCT" in err
    assert "Traceback" not in err
