from __future__ import annotations

import json
from pathlib import Path

import pytest

from generation_relation_core.canonical import canonical_bytes

from experiments.source_map_projection.scripts.run_all import (
    CONTRACTS,
    EXPERIMENT_ROOT,
    FIXTURES,
    OFFICIAL_ROOT,
    RUN_ROOT,
    ambiguity_report,
    collect_core,
    dependency_hashes,
    normalized_records,
    run_multistage_four_modes,
    strict_projection_report,
)
from experiments.source_map_projection.src.canonical_source_map import run_official_non_indexed_tests
from experiments.source_map_projection.src.core_collector import snapshot_document
from experiments.source_map_projection.src.core_to_source_map import project_stage
from experiments.source_map_projection.src.deterministic_transformer import adversarial_transform
from experiments.source_map_projection.src.independent_oracle import adversarial_oracle
from experiments.source_map_projection.src.node_bridge import capture_native_map
from experiments.source_map_projection.src.projection_validator import (
    compare_mapping_records,
    imported_leaf_modules,
    run_negative_controls,
)


@pytest.fixture(scope="module")
def hashes() -> dict[str, str]:
    try:
        return dependency_hashes()
    except Exception as exc:
        pytest.skip(f"exact Node dependency installation unavailable: {exc}")


def test_forward_receipts_core_projection_and_independent_native_map_are_exact(tmp_path: Path, hashes) -> None:
    result = adversarial_transform(FIXTURES, run_id="pytest-adversarial")
    collector, snapshot = collect_core(result, hashes)
    projection = project_stage(snapshot, collector.registry, result.stage_id)
    native = capture_native_map(result, tmp_path / "native.map")
    compare_mapping_records(normalized_records(native), projection["canonical_records"])
    assert native == projection["document"]
    assert result.output_bytes == adversarial_oracle(FIXTURES)["output_bytes"]
    assert len(snapshot.tables.explicit_dispositions) == 1
    assert len(snapshot.tables.generation_bindings) > projection["mapping_count"]


def test_output_only_mode_has_no_receipts_and_same_bytes() -> None:
    output_only = adversarial_transform(FIXTURES, run_id="same", record_receipts=False)
    contract = adversarial_transform(FIXTURES, run_id="same")
    assert output_only.receipts == ()
    assert output_only.output_bytes == contract.output_bytes
    assert "sourceMappingURL" not in output_only.output_bytes.decode("utf-8")


def test_three_strict_projection_counterexamples_and_two_result_ambiguities(tmp_path: Path, hashes) -> None:
    strict = strict_projection_report(hashes, tmp_path)
    ambiguity = ambiguity_report(hashes)
    assert strict["status"] == "PASS"
    assert strict["counterexample_count"] == 3
    assert strict["statement"] == "Source Map equality does not imply generation-fact equality."
    assert all(case["map_equal"] and not case["generation_facts_equal"] for case in strict["cases"])
    assert ambiguity["status"] == "PASS"
    assert ambiguity["ambiguity_case_count"] == 2


def test_generated_origin_composition_matches_independent_map_composition(tmp_path: Path, hashes) -> None:
    report, snapshot = run_multistage_four_modes(hashes, tmp_path, persist_maps=False)
    assert report["status"] == "PASS"
    assert report["native_core_composition_exact"]
    assert report["composed_mapping_count"] == 5
    assert report["generated_origin_count"] == 5
    assert report["direct_shortcut_count"] == 0
    assert report["broken_bridge_count"] == 0
    assert snapshot.record["snapshot_id"]


def test_all_30_frozen_negative_controls_return_exact_reason_codes(tmp_path: Path, hashes) -> None:
    single = adversarial_transform(FIXTURES, run_id="negative-single")
    collector, snapshot = collect_core(single, hashes)
    from experiments.source_map_projection.src.deterministic_transformer import multistage_transform

    stage1, stage2 = multistage_transform(FIXTURES, run_id="negative-multistage")
    _, multistage_snapshot = collect_core(stage1, hashes, stage2)
    baseline = capture_native_map(single, tmp_path / "baseline.map")
    result = run_negative_controls(
        normalized_records(baseline),
        project_stage(snapshot, collector.registry, single.stage_id)["document"],
        snapshot,
        multistage_snapshot,
    )
    frozen = json.loads((CONTRACTS / "negative_controls.json").read_text(encoding="utf-8"))["controls"]
    assert result["status"] == "PASS"
    assert result["passed"] == result["total"] == 30
    assert [row["actual_reason_code"] for row in result["controls"]] == [row["reason_code"] for row in frozen]


def test_authority_paths_remain_isolated() -> None:
    source = EXPERIMENT_ROOT / "src"
    assert not imported_leaf_modules(source / "core_collector.py") & {
        "canonical_source_map", "node_bridge", "independent_oracle",
    }
    assert not imported_leaf_modules(source / "core_to_source_map.py") & {
        "node_bridge", "independent_oracle", "transformation_dsl",
    }
    assert "independent_oracle" not in imported_leaf_modules(source / "deterministic_transformer.py")


def test_fixed_official_non_indexed_suite_when_downloaded() -> None:
    root = OFFICIAL_ROOT / "source-map-tests"
    if not root.is_dir():
        pytest.skip("official tests are deliberately stored under ignored data_private; run the full bootstrap first")
    report = run_official_non_indexed_tests(root)
    assert report["status"] == "PASS"
    assert report["applicable_passed"] == report["applicable_total"] == 80
    assert report["excluded_total"] == 19


def test_committed_formal_artifacts_report_two_equal_runs() -> None:
    artifacts = EXPERIMENT_ROOT / "artifacts"
    determinism = json.loads((artifacts / "determinism.json").read_text(encoding="utf-8"))
    mapping = json.loads((artifacts / "mapping_equivalence.json").read_text(encoding="utf-8"))
    assert determinism["status"] == "PASS"
    assert determinism["run_count"] == determinism["equal_run_count"] == 2
    assert mapping["status"] == "PASS"
    assert mapping["cases"]["medium"]["mapping_count"] == 660
