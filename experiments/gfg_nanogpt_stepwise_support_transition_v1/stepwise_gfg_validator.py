from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import (
    SupportGFG,
    validate_support_gfg,
)

from .stepwise_gfg import GRAPH_SCHEMA


INPUT_PHASE_OCCURRENCES = {
    "support_left_finite_difference_occurrence",
    "support_left_categorical_change_occurrence",
    "support_left_acceleration_finite_difference_occurrence",
}
TARGET_PHASE_OCCURRENCES = {
    "support_right_finite_difference_occurrence",
    "support_law_break_finite_difference_occurrence",
    "support_right_categorical_change_occurrence",
    "support_categorical_law_break_occurrence",
}


def validate_stepwise_gfg(
    *,
    database_path: Path,
    source_database_path: Path,
    tensor_directory: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    manifest_material = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(payload_sha256(manifest_material) == manifest["manifest_sha256"], "SST_GFG_MANIFEST_HASH_MISMATCH")
    require(file_sha256(database_path) == manifest["database_sha256"], "SST_GFG_DATABASE_HASH_MISMATCH")
    structural = validate_support_gfg(
        database_path,
        source_database_path=source_database_path,
        tensor_directory=tensor_directory,
        expected_schema=GRAPH_SCHEMA,
        require_checkpoint_grid=False,
    )
    require(structural["status"] == "PASS", "SST_GFG_STRUCTURAL_VALIDATION_FAILED")

    graph = SupportGFG(database_path)
    phase_occurrence_ids: set[str] = set()
    atomic_fact_block_count = 0
    phase_fact_source_count = 0
    input_fact_count = 0
    target_fact_count = 0
    disposition_fact_count = 0
    generated_origin_source_count = 0
    batch_selection_order_disposition_count = 0
    validation_target_recovery_fact_count = 0
    try:
        final_block_sha = None
        for row, block in graph.blocks():
            final_block_sha = row["block_sha256"]
            occurrences = {value["occurrence_id"]: value for value in block["occurrences"]}
            objects = {value["object_id"]: value for value in block["objects"]}
            for fact in block["fact_blocks"]:
                atomic_fact_block_count += 1
                occurrence = occurrences.get(fact["occurrence_id"])
                # Facts may refer to an occurrence created in an earlier block.
                if occurrence is None:
                    continue
                occurrence_type = str(occurrence["occurrence_type"])
                if occurrence_type in INPUT_PHASE_OCCURRENCES | TARGET_PHASE_OCCURRENCES:
                    phase_occurrence_ids.add(fact["occurrence_id"])
                    phase_fact_source_count += len(fact["sources"])
                    expected_temporal_role = (
                        "input_available_at_cut" if occurrence_type in INPUT_PHASE_OCCURRENCES else "target_only_after_cut"
                    )
                    require(
                        occurrence["payload"]["temporal_role"] == expected_temporal_role,
                        "SST_GFG_PHASE_OCCURRENCE_TEMPORAL_ROLE_MISMATCH",
                    )
                    outcome = objects.get(fact["outcome"]["object_id"])
                    require(outcome is not None, "SST_GFG_PHASE_OUTCOME_NOT_IN_CREATION_BLOCK")
                    require(
                        outcome["payload"]["temporal_role"] == expected_temporal_role,
                        "SST_GFG_PHASE_OUTCOME_TEMPORAL_ROLE_MISMATCH",
                    )
                    future_source_roles = [
                        source["relation_role"] for source in fact["sources"] if "future" in source["relation_role"]
                    ]
                    if expected_temporal_role == "input_available_at_cut":
                        require(not future_source_roles, "SST_GFG_PHASE_FUTURE_SOURCE_IN_PREDICTOR_INPUT")
                        input_fact_count += 1
                    else:
                        require(bool(future_source_roles), "SST_GFG_PHASE_TARGET_MISSING_FUTURE_SOURCE")
                        target_fact_count += 1
                    generated_origin_source_count += sum(
                        source["source_kind"] == "generated_origin" for source in fact["sources"]
                    )
                if occurrence_type in {
                    "categorical_acceleration_disposition_occurrence",
                    "right_difference_disposition_occurrence",
                }:
                    outcome = objects.get(fact["outcome"]["object_id"])
                    require(outcome is not None and outcome["object_kind"] == "ExplicitDisposition", "SST_GFG_PHASE_DISPOSITION_INVALID")
                    disposition_fact_count += 1
                if occurrence_type == "batch_selection_order_availability_occurrence":
                    outcome = objects.get(fact["outcome"]["object_id"])
                    require(outcome is not None and outcome["object_kind"] == "ExplicitDisposition", "SST_GFG_BATCH_ORDER_DISPOSITION_INVALID")
                    require(
                        outcome["payload"]["disposition"]
                        == "SOURCE_BATCH_SELECTION_ORDER_NOT_CAPTURED_IN_PARTICIPANT_GFG",
                        "SST_GFG_BATCH_ORDER_DISPOSITION_KIND_INVALID",
                    )
                    require(
                        outcome["payload"]["reconstruction_or_guess_used"] is False,
                        "SST_GFG_BATCH_ORDER_RECONSTRUCTION_USED",
                    )
                    source_roles = {source["relation_role"] for source in fact["sources"]}
                    require(
                        any(role.endswith("training_batch_inputs") for role in source_roles)
                        and any(role.endswith("training_batch_targets") for role in source_roles),
                        "SST_GFG_BATCH_ORDER_DISPOSITION_SOURCE_INCOMPLETE",
                    )
                    batch_selection_order_disposition_count += 1
                if occurrence_type == "validation_target_recovery_occurrence":
                    outcome = objects.get(fact["outcome"]["object_id"])
                    require(
                        outcome is not None
                        and outcome["object_kind"] == "content_addressed_tensor"
                        and outcome["role"] == "derived_validation_dataset_targets",
                        "SST_GFG_VALIDATION_TARGET_RECOVERY_OUTCOME_INVALID",
                    )
                    require(
                        occurrence["payload"]["future_training_fact_used"] is False,
                        "SST_GFG_VALIDATION_TARGET_RECOVERY_FUTURE_USED",
                    )
                    validation_target_recovery_fact_count += 1
        require(final_block_sha == manifest["final_block_sha256"], "SST_GFG_FINAL_BLOCK_HASH_MISMATCH")
    finally:
        graph.close()
    require(bool(phase_occurrence_ids), "SST_GFG_PHASE_LAYER_EMPTY")
    require(input_fact_count > 0 and target_fact_count > 0, "SST_GFG_PHASE_TEMPORAL_PARTITION_INCOMPLETE")
    require(disposition_fact_count > 0, "SST_GFG_PHASE_DISPOSITION_COVERAGE_EMPTY")
    require(generated_origin_source_count > 0, "SST_GFG_PHASE_GENERATED_ORIGIN_COVERAGE_EMPTY")
    material = {
        "schema": "nanogpt-stepwise-support-transition-gfg-validation-v1",
        "status": "PASS",
        "database_sha256": manifest["database_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "structural_validation_sha256": structural["validation_sha256"],
        "counts": structural["counts"],
        "atomic_fact_block_count": atomic_fact_block_count,
        "phase_occurrence_count": len(phase_occurrence_ids),
        "phase_fact_source_count": phase_fact_source_count,
        "phase_input_fact_count": input_fact_count,
        "phase_target_fact_count": target_fact_count,
        "phase_disposition_fact_count": disposition_fact_count,
        "phase_generated_origin_source_count": generated_origin_source_count,
        "batch_selection_order_disposition_count": batch_selection_order_disposition_count,
        "validation_target_recovery_fact_count": validation_target_recovery_fact_count,
        "future_leakage_audit": "PASS",
        "phase_origin_traceability": "PASS",
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(output_path, result)
    return result
