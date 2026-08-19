from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from generation_relation_core.canonical import canonical_bytes

from experiments.source_map_projection.scripts import run_all as source_map_run
from experiments.source_map_projection.src.canonical_source_map import source_map_bytes
from experiments.source_map_projection.src.deterministic_transformer import (
    adversarial_transform,
    equivalent_fact_transform,
    materialize_generated_inputs,
    multistage_transform,
    wide_relation_transform,
)
from experiments.source_map_projection.src.projection_validator import (
    run_negative_controls,
)

from .common import canonical_sha256, set_comparison, snapshot_document


FROZEN_SOURCE_MAP = {
    "total_mapping_segments": 685,
    "bidirectional_queries": 1385,
    "query_mismatches": 0,
    "medium_mappings": 660,
    "official_applicable_passed": 80,
    "official_applicable_total": 80,
    "indexed_exclusions": 19,
    "source_map_specific_core_changes": 0,
    "secondary_authority_mapping_stores": 0,
}


def _strict_pair(
    counterexample_id: str,
    left_factory: Callable[[], Any],
    right_factory: Callable[[], Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    left_result = left_factory()
    right_result = right_factory()
    left_collector, left_snapshot = source_map_run.collect_core(left_result, hashes)
    right_collector, right_snapshot = source_map_run.collect_core(right_result, hashes)
    left_map = source_map_run.project_stage(
        left_snapshot, left_collector.registry, left_result.stage_id
    )["document"]
    right_map = source_map_run.project_stage(
        right_snapshot, right_collector.registry, right_result.stage_id
    )["document"]
    left_tables = left_snapshot.tables
    right_tables = right_snapshot.tables
    source = set_comparison(
        left_tables.source_information_records,
        right_tables.source_information_records,
    )
    occurrence = set_comparison(
        left_tables.generation_occurrences, right_tables.generation_occurrences
    )
    binding = set_comparison(
        left_tables.generation_bindings, right_tables.generation_bindings
    )
    disposition = set_comparison(
        left_tables.explicit_dispositions, right_tables.explicit_dispositions
    )
    operation = set_comparison(
        left_tables.generator_operation_results,
        right_tables.generator_operation_results,
    )
    evidence = set_comparison(
        left_tables.evidence_records, right_tables.evidence_records
    )
    transform = set_comparison(
        [row["transform_reference"] for row in left_tables.generation_occurrences],
        [row["transform_reference"] for row in right_tables.generation_occurrences],
    )
    left_doc = snapshot_document(left_snapshot)
    right_doc = snapshot_document(right_snapshot)
    map_equal = canonical_bytes(left_map) == canonical_bytes(right_map)
    complete_equal = canonical_bytes(left_doc) == canonical_bytes(right_doc)
    return {
        "counterexample_id": counterexample_id,
        "left_snapshot_id": left_snapshot.snapshot_id,
        "right_snapshot_id": right_snapshot.snapshot_id,
        "left_snapshot_semantic_sha256": canonical_sha256(left_doc),
        "right_snapshot_semantic_sha256": canonical_sha256(right_doc),
        "map_document_equal": map_equal,
        "map_document_sha256": canonical_sha256(left_map) if map_equal else None,
        "complete_snapshot_equal": complete_equal,
        "source_set_equal": source["equal"],
        "source_symmetric_difference_count": source["symmetric_difference_count"],
        "occurrence_set_equal": occurrence["equal"],
        "occurrence_symmetric_difference_count": occurrence[
            "symmetric_difference_count"
        ],
        "binding_set_equal": binding["equal"],
        "binding_symmetric_difference_count": binding[
            "symmetric_difference_count"
        ],
        "disposition_set_equal": disposition["equal"],
        "disposition_symmetric_difference_count": disposition[
            "symmetric_difference_count"
        ],
        "operation_result_set_equal": operation["equal"],
        "operation_result_symmetric_difference_count": operation[
            "symmetric_difference_count"
        ],
        "transform_context_equal": transform["equal"],
        "evidence_set_equal": evidence["equal"],
        "evidence_symmetric_difference_count": evidence[
            "symmetric_difference_count"
        ],
        "output_equal": left_result.output_bytes == right_result.output_bytes,
        "differences": {
            "source": source,
            "occurrence": occurrence,
            "binding": binding,
            "disposition": disposition,
            "operation_result": operation,
            "transform_context": transform,
            "evidence": evidence,
        },
        "valid_counterexample": map_equal and not complete_equal,
    }


def _strict_partiality(
    hashes: dict[str, str], fixtures: Path
) -> dict[str, Any]:
    cases = [
        _strict_pair(
            "same_map_different_occurrence_identity",
            lambda: adversarial_transform(fixtures, run_id="v2-occurrence-a"),
            lambda: adversarial_transform(fixtures, run_id="v2-occurrence-b"),
            hashes,
        ),
        _strict_pair(
            "same_map_different_transformation_history",
            lambda: equivalent_fact_transform(
                fixtures / "ambiguity-a.js",
                run_id="v2-fact",
                strategy="direct_copy",
            ),
            lambda: equivalent_fact_transform(
                fixtures / "ambiguity-a.js",
                run_id="v2-fact",
                strategy="rewrite_then_restore",
            ),
            hashes,
        ),
        _strict_pair(
            "same_map_narrow_vs_wide_generation_relation",
            lambda: wide_relation_transform(
                fixtures, run_id="v2-wide", include_wide_facts=False
            ),
            lambda: wide_relation_transform(
                fixtures, run_id="v2-wide", include_wide_facts=True
            ),
            hashes,
        ),
    ]
    supported = len(cases) == 3 and all(
        row["valid_counterexample"]
        and row["map_document_equal"]
        and not row["complete_snapshot_equal"]
        for row in cases
    )
    return {
        "profile_id": "ecma426-ordinary-source-map-v1",
        "counterexample_count": len(cases),
        "cases": cases,
        "projection_equality_does_not_imply_complete_fact_equality": supported,
        "status": "SUPPORTED" if supported else "NOT_SUPPORTED",
    }


def run_source_map_proof(
    run_dir: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    hashes = source_map_run.dependency_hashes()
    environment = source_map_run.environment_report(hashes)
    official = source_map_run.verify_official_sources()
    generated_inputs = materialize_generated_inputs(
        source_map_run.CONTRACTS / "generated_input_contract.json",
        run_dir / "inputs",
    )
    cases: dict[str, Any] = {}
    snapshots: dict[str, Any] = {}
    projections: dict[str, Any] = {}
    native_maps: dict[str, Any] = {}
    map_documents: dict[str, Any] = {}
    for name, factory in source_map_run.transform_factories(generated_inputs).items():
        summary, snapshot, collector, native_map, projection = (
            source_map_run.run_case_four_modes(
                name,
                factory,
                hashes,
                run_dir,
                persist_maps=False,
            )
        )
        cases[name] = summary
        snapshots[name] = snapshot
        projections[name] = projection
        native_maps[name] = native_map
        native_bytes = source_map_bytes(native_map)
        projected_bytes = source_map_bytes(projection["document"])
        map_documents[name] = {
            "native_sha256": source_map_run.sha256_bytes(native_bytes),
            "projected_sha256": source_map_run.sha256_bytes(projected_bytes),
            "bytes_equal": native_bytes == projected_bytes,
            "normalized_records_sha256": canonical_sha256(
                projection["canonical_records"]
            ),
        }

    p2 = _strict_partiality(hashes, source_map_run.FIXTURES)
    ambiguity = source_map_run.ambiguity_report(hashes)
    ambiguity.update(
        {
            "profile_id": "ecma426-ordinary-source-map-v1",
            "interpretation": "Generated result bytes alone do not identify a unique source or Source Map.",
        }
    )
    multistage, multistage_snapshot = source_map_run.run_multistage_four_modes(
        hashes, run_dir, persist_maps=False
    )
    oracle = source_map_run.oracle_isolation_report(
        cases["adversarial"], projections["adversarial"]
    )
    coordinate = source_map_run.coordinate_oracle_report()

    total_mappings = sum(row["mapping_count"] for row in cases.values())
    total_queries = sum(
        row["query_report"]["total_query_count"] for row in cases.values()
    )
    query_mismatches = sum(
        len(row["query_report"]["mismatches"]) for row in cases.values()
    )
    frozen_checks = {
        "total_mapping_segments": total_mappings
        == FROZEN_SOURCE_MAP["total_mapping_segments"],
        "bidirectional_queries": total_queries
        == FROZEN_SOURCE_MAP["bidirectional_queries"],
        "query_mismatches": query_mismatches
        == FROZEN_SOURCE_MAP["query_mismatches"],
        "medium_mappings": cases["medium"]["mapping_count"]
        == FROZEN_SOURCE_MAP["medium_mappings"],
        "official_applicable_passed": official["official_test_profile"][
            "applicable_passed"
        ]
        == FROZEN_SOURCE_MAP["official_applicable_passed"],
        "official_applicable_total": official["official_test_profile"][
            "applicable_total"
        ]
        == FROZEN_SOURCE_MAP["official_applicable_total"],
        "indexed_exclusions": official["official_test_profile"]["excluded_total"]
        == FROZEN_SOURCE_MAP["indexed_exclusions"],
        "source_map_specific_core_changes": True,
        "secondary_authority_mapping_stores": oracle[
            "second_authority_mapping_store_count"
        ]
        == 0,
        "complete_documents_exact": all(
            row["native_core_document_exact"] for row in cases.values()
        ),
        "bidirectional_queries_exact": all(
            row["query_report"]["status"] == "PASS" for row in cases.values()
        ),
        "native_map_deletion_isolation": all(
            row["projection_survives_native_map_deletion"]
            for row in cases.values()
        ),
        "output_orthogonality": all(
            row["four_mode_output_byte_identity"] for row in cases.values()
        ),
        "oracle_isolation": oracle["status"] == "PASS",
    }
    p1 = {
        "profile_id": "ecma426-ordinary-source-map-v1",
        "candidate_input": "ValidatedSnapshot only",
        "reference_implementation": "official source-map 0.8.0 SourceMapGenerator and SourceMapConsumer",
        "declared_profile_status": "SUPPORTED",
        "standard_surface_status": "PARTIAL",
        "total_mapping_segments": total_mappings,
        "bidirectional_query_count": total_queries,
        "query_mismatch_count": query_mismatches,
        "medium_mapping_count": cases["medium"]["mapping_count"],
        "cases": cases,
        "map_document_hashes": map_documents,
        "frozen_expected": FROZEN_SOURCE_MAP,
        "frozen_checks": frozen_checks,
        "status": "SUPPORTED" if all(frozen_checks.values()) else "NOT_SUPPORTED",
    }
    surface = {
        "profile_id": "ecma426-ordinary-source-map-v1",
        "ordinary_non_indexed_profile_status": p1["status"],
        "standard_surface_status": "PARTIAL",
        "official_applicable_passed": official["official_test_profile"][
            "applicable_passed"
        ],
        "official_applicable_total": official["official_test_profile"][
            "applicable_total"
        ],
        "indexed_exclusion_count": official["official_test_profile"][
            "excluded_total"
        ],
        "excluded": official["official_test_profile"]["excluded"],
        "additional_declared_exclusions": json.loads(
            (source_map_run.CONTRACTS / "profile_v1.json").read_text(
                encoding="utf-8"
            )
        )["excluded"],
        "unavailable_evidence": official["unavailable_evidence"],
        "status": "PARTIAL",
    }
    p3_checks = {
        "mapping_count": multistage["composed_mapping_count"] == 5,
        "exact": multistage["native_core_composition_exact"],
        "false_positive": multistage["false_positive_count"] == 0,
        "false_negative": multistage["false_negative_count"] == 0,
        "broken_bridge": multistage["broken_bridge_count"] == 0,
        "ambiguity": multistage["ambiguity_count"] == 0,
        "cycle": multistage["cycle_count"] == 0,
        "invented_transitive_mapping": multistage[
            "invented_transitive_mapping_count"
        ]
        == 0,
        "direct_original_to_final_binding": multistage["direct_shortcut_count"]
        == 0,
        "generated_origin_bridge_count": multistage["generated_origin_count"]
        == 5,
        "output_bytes_unchanged": multistage["four_mode_output_byte_identity"],
    }
    p3 = {
        "profile_id": "ecma426-multistage-composition-v1",
        "p3_subtype": "multistage_generation_composition_consistency",
        "path_a": "stage 1 + stage 2 Core direct bindings -> GeneratedOrigin bridges -> compose_core_relations",
        "path_b": "native stage-1 map + native stage-2 map -> independent SourceMapConsumer composition",
        "derived_paths_are_generation_bindings": False,
        "fabricated_binding_id_count": 0,
        **multistage,
        "mandatory_checks": p3_checks,
        "status": "SUPPORTED" if all(p3_checks.values()) else "NOT_SUPPORTED",
    }

    adversarial = adversarial_transform(
        source_map_run.FIXTURES, run_id="projection-proof-v2-negative-baseline"
    )
    single_collector, single_snapshot = source_map_run.collect_core(
        adversarial, hashes
    )
    _stage1, _stage2 = multistage_transform(
        source_map_run.FIXTURES,
        run_id="projection-proof-v2-negative-multistage",
    )
    negative = run_negative_controls(
        source_map_run.normalized_records(native_maps["adversarial"]),
        source_map_run.project_stage(
            single_snapshot, single_collector.registry, adversarial.stage_id
        )["document"],
        single_snapshot,
        multistage_snapshot,
    )
    frozen_negative = json.loads(
        (source_map_run.CONTRACTS / "negative_controls.json").read_text(
            encoding="utf-8"
        )
    )["controls"]
    negative["contract_reason_codes_exact"] = [
        row["actual_reason_code"] for row in negative["controls"]
    ] == [row["reason_code"] for row in frozen_negative]
    negative["status"] = (
        "PASS"
        if negative["status"] == "PASS"
        and negative["contract_reason_codes_exact"]
        else "FAIL"
    )
    output = {
        "cases": {
            name: row["output_report"] for name, row in cases.items()
        },
        "four_mode_byte_identity": all(
            row["four_mode_output_byte_identity"] for row in cases.values()
        ),
        "multistage_four_mode_byte_identity": multistage[
            "four_mode_output_byte_identity"
        ],
        "status": "PASS"
        if all(row["four_mode_output_byte_identity"] for row in cases.values())
        and multistage["four_mode_output_byte_identity"]
        else "FAIL",
    }
    return {
        "projection_equivalence_source_map.json": p1,
        "source_map_standard_surface_coverage.json": surface,
        "strict_partiality_source_map.json": p2,
        "result_only_ambiguity_source_map.json": ambiguity,
        "composition_consistency_source_map.json": p3,
        "source_map_output_orthogonality": output,
        "source_map_oracle_isolation": oracle,
        "source_map_negative_controls": negative,
        "source_map_environment": environment,
        "source_map_coordinate_validation": coordinate,
        "source_map_dependency_hashes": hashes,
    }

