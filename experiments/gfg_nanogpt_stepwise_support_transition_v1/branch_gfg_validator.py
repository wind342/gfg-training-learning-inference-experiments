from __future__ import annotations

from collections import Counter
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

from .branch_gfg import BRANCHES, BRANCH_GRAPH_SCHEMA


def _checked_result(path: Path, schema: str) -> dict[str, Any]:
    value = read_json(path)
    require(value["schema"] == schema, f"SST_BRANCH_VALIDATION_SCHEMA_INVALID:{path}")
    material = {key: child for key, child in value.items() if key != "result_sha256"}
    require(payload_sha256(material) == value["result_sha256"], f"SST_BRANCH_VALIDATION_RESULT_HASH_INVALID:{path}")
    require(value["status"] == "PASS", f"SST_BRANCH_VALIDATION_RESULT_NOT_PASS:{path}")
    return value


def _main_objects(database_path: Path, object_ids: set[str]) -> dict[str, dict[str, Any]]:
    graph = SupportGFG(database_path)
    found: dict[str, dict[str, Any]] = {}
    try:
        for _row, block in graph.blocks():
            for value in block["objects"]:
                if value["object_id"] in object_ids:
                    found[value["object_id"]] = value
            if set(found) == object_ids:
                break
    finally:
        graph.close()
    require(set(found) == object_ids, "SST_BRANCH_VALIDATION_MAIN_OBJECT_COVERAGE_INCOMPLETE")
    return found


def validate_branch_gfg(
    *,
    database_path: Path,
    source_database_path: Path,
    tensor_directory: Path,
    manifest_path: Path,
    main_manifest_path: Path,
    branch_entry_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    manifest_material = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(payload_sha256(manifest_material) == manifest["manifest_sha256"], "SST_BRANCH_GFG_MANIFEST_HASH_MISMATCH")
    require(file_sha256(database_path) == manifest["database_sha256"], "SST_BRANCH_GFG_DATABASE_HASH_MISMATCH")

    main_manifest = read_json(main_manifest_path)
    main_manifest_material = {key: value for key, value in main_manifest.items() if key != "manifest_sha256"}
    require(
        payload_sha256(main_manifest_material) == main_manifest["manifest_sha256"],
        "SST_BRANCH_GFG_MAIN_MANIFEST_HASH_MISMATCH",
    )
    require(
        main_manifest["manifest_sha256"] == manifest["main_stepwise_gfg_manifest_sha256"],
        "SST_BRANCH_GFG_MAIN_MANIFEST_REFERENCE_MISMATCH",
    )
    main_database_path = main_manifest_path.parent / str(main_manifest["database"])
    require(
        file_sha256(main_database_path) == main_manifest["database_sha256"],
        "SST_BRANCH_GFG_MAIN_DATABASE_HASH_MISMATCH",
    )

    structural = validate_support_gfg(
        database_path,
        source_database_path=source_database_path,
        tensor_directory=tensor_directory,
        expected_schema=BRANCH_GRAPH_SCHEMA,
        require_checkpoint_grid=False,
    )
    require(structural["status"] == "PASS", "SST_BRANCH_GFG_STRUCTURAL_VALIDATION_FAILED")

    seed_roots = sorted((branch_entry_root / "branch-seeds").glob("branch-seed-*"))
    require(bool(seed_roots), "SST_BRANCH_GFG_VALIDATION_SEEDS_EMPTY")
    expected_horizons: dict[str, set[int]] = {}
    expected_continuations: dict[str, set[int]] = {}
    for seed_root in seed_roots:
        seed = _checked_result(seed_root / "seed_result.json", "nanogpt-stepwise-branch-seed-v1")
        receipt = _checked_result(seed_root / "branch_receipt.json", "nanogpt-stepwise-four-branch-receipt-v1")
        require(seed["seed_id"] == receipt["seed_id"], "SST_BRANCH_GFG_SEED_RECEIPT_ID_MISMATCH")
        seed_id = str(seed["seed_id"])
        expected_horizons[seed_id] = {int(value) for value in receipt["legal_horizons"]}
        expected_continuations[seed_id] = {
            int(value["physical_optimizer_step"]) for value in receipt["continuation_results"]
        }
        require(1 in expected_horizons[seed_id], f"SST_BRANCH_GFG_HORIZON_ONE_MISSING:{seed_id}")
        expected_catalog = {
            f"{horizon}:{branch}"
            for horizon in expected_horizons[seed_id]
            for branch in BRANCHES
        }
        require(
            set(manifest["seed_catalog"].get(seed_id, {})) == expected_catalog,
            f"SST_BRANCH_GFG_SEED_CATALOG_MISMATCH:{seed_id}",
        )
    require(set(manifest["seed_catalog"]) == set(expected_horizons), "SST_BRANCH_GFG_SEED_SET_MISMATCH")
    require(int(manifest["seed_count"]) == len(seed_roots), "SST_BRANCH_GFG_SEED_COUNT_MISMATCH")

    graph = SupportGFG(database_path)
    occurrences: dict[str, dict[str, Any]] = {}
    objects: dict[str, dict[str, Any]] = {}
    facts_by_occurrence: dict[str, list[dict[str, Any]]] = {}
    occurrence_counts: Counter[str] = Counter()
    main_origin_objects: list[dict[str, Any]] = []
    batch_disposition_count = 0
    intermediate_disposition_count = 0
    validation_target_recovery_count = 0
    causal_difference_count = 0
    final_block_sha: str | None = None
    try:
        for row, block in graph.blocks():
            final_block_sha = str(row["block_sha256"])
            for occurrence in block["occurrences"]:
                occurrences[occurrence["occurrence_id"]] = occurrence
                occurrence_counts[str(occurrence["occurrence_type"])] += 1
            for value in block["objects"]:
                objects[value["object_id"]] = value
                if value["object_kind"] == "GeneratedOrigin" and value["role"] == "generated_main_training_state_origin":
                    main_origin_objects.append(value)
            for fact in block["fact_blocks"]:
                facts_by_occurrence.setdefault(str(fact["occurrence_id"]), []).append(fact)
                occurrence = occurrences.get(str(fact["occurrence_id"]))
                require(occurrence is not None, "SST_BRANCH_GFG_FACT_OCCURRENCE_NOT_AVAILABLE")
                outcome = objects.get(str(fact["outcome"]["object_id"]))
                require(outcome is not None, "SST_BRANCH_GFG_FACT_OUTCOME_NOT_AVAILABLE")
                occurrence_type = str(occurrence["occurrence_type"])
                if occurrence_type == "batch_selection_order_availability_occurrence":
                    require(outcome["object_kind"] == "ExplicitDisposition", "SST_BRANCH_GFG_BATCH_ORDER_NOT_DISPOSITION")
                    require(
                        outcome["payload"]["disposition"]
                        == "SOURCE_BATCH_SELECTION_ORDER_NOT_CAPTURED_IN_PARTICIPANT_GFG",
                        "SST_BRANCH_GFG_BATCH_ORDER_DISPOSITION_INVALID",
                    )
                    require(
                        occurrence["payload"]["reconstruction_or_guess_used"] is False,
                        "SST_BRANCH_GFG_BATCH_ORDER_RECONSTRUCTION_USED",
                    )
                    source_roles = {source["relation_role"] for source in fact["sources"]}
                    require(
                        any(role.endswith("training_batch_inputs") for role in source_roles)
                        and any(role.endswith("training_batch_targets") for role in source_roles),
                        "SST_BRANCH_GFG_BATCH_ORDER_SOURCE_INCOMPLETE",
                    )
                    batch_disposition_count += 1
                elif occurrence_type == "actual_aligned_branch_training_step_occurrence" and outcome["object_kind"] == "ExplicitDisposition":
                    require(
                        outcome["payload"]["disposition"]
                        == "BRANCH_INTERMEDIATE_TENSOR_PAYLOAD_NOT_MATERIALIZED_UNDER_FROZEN_PROFILE",
                        "SST_BRANCH_GFG_INTERMEDIATE_DISPOSITION_INVALID",
                    )
                    intermediate_disposition_count += 1
                elif occurrence_type == "validation_target_recovery_occurrence":
                    require(
                        outcome["object_kind"] == "content_addressed_tensor"
                        and outcome["role"] == "derived_validation_dataset_targets",
                        "SST_BRANCH_GFG_VALIDATION_TARGET_RECOVERY_INVALID",
                    )
                    require(
                        occurrence["payload"]["future_training_fact_used"] is False,
                        "SST_BRANCH_GFG_VALIDATION_TARGET_RECOVERY_FUTURE_USED",
                    )
                    validation_target_recovery_count += 1
                elif occurrence_type == "matched_four_branch_causal_contrast_occurrence":
                    is_horizon_one = int(occurrence["payload"]["horizon"]) == 1
                    require(
                        occurrence["payload"]["trajectory_finite_difference_conflated"] is False,
                        "SST_BRANCH_GFG_CAUSAL_TRAJECTORY_DIFFERENCE_CONFLATED",
                    )
                    require(
                        bool(occurrence["payload"]["Phi_full_is_D_U_S_k"]) == is_horizon_one,
                        "SST_BRANCH_GFG_CAUSAL_DIFFERENCE_SEMANTICS_INVALID",
                    )
                    if str(outcome["role"]).startswith("actual_update_causal_difference_D_U_S_k"):
                        require(is_horizon_one, "SST_BRANCH_GFG_D_U_S_OUTSIDE_HORIZON_ONE")
                        causal_difference_count += 1
        require(final_block_sha == manifest["final_block_sha256"], "SST_BRANCH_GFG_FINAL_BLOCK_HASH_MISMATCH")
    finally:
        graph.close()

    require(len(main_origin_objects) == len(seed_roots), "SST_BRANCH_GFG_MAIN_ORIGIN_COUNT_MISMATCH")
    required_main_ids = {str(value["payload"]["source_object_id"]) for value in main_origin_objects}
    indexed_main = _main_objects(main_database_path, required_main_ids)
    for origin in main_origin_objects:
        payload = origin["payload"]
        source = indexed_main[str(payload["source_object_id"])]
        require(payload["source_graph_manifest_sha256"] == main_manifest["manifest_sha256"], "SST_BRANCH_GFG_MAIN_ORIGIN_MANIFEST_MISMATCH")
        require(payload["source_graph_database_sha256"] == main_manifest["database_sha256"], "SST_BRANCH_GFG_MAIN_ORIGIN_DATABASE_MISMATCH")
        require(payload["source_content_sha256"] == source["content_sha256"], "SST_BRANCH_GFG_MAIN_ORIGIN_CONTENT_MISMATCH")
        require(payload["source_role"] == source["role"], "SST_BRANCH_GFG_MAIN_ORIGIN_ROLE_MISMATCH")
        require(int(payload["source_optimizer_step"]) == int(source["optimizer_step"]), "SST_BRANCH_GFG_MAIN_ORIGIN_STEP_MISMATCH")

    actual_seed_branches: dict[str, set[str]] = {seed_id: set() for seed_id in expected_horizons}
    actual_horizons: dict[str, set[tuple[int, str]]] = {seed_id: set() for seed_id in expected_horizons}
    actual_continuations: dict[str, set[tuple[int, str]]] = {seed_id: set() for seed_id in expected_horizons}
    actual_contrasts: dict[str, set[int]] = {seed_id: set() for seed_id in expected_horizons}
    for occurrence_id, occurrence in occurrences.items():
        occurrence_type = str(occurrence["occurrence_type"])
        payload = occurrence["payload"]
        if occurrence_type in {
            "actual_full_training_step_occurrence",
            "actual_forward_backward_clip_without_optimizer_occurrence",
            "explicit_branch_state_composition_occurrence",
        }:
            branch = str(payload["branch"])
            seed_id = str(payload["seed_id"])
            require(seed_id in actual_seed_branches, "SST_BRANCH_GFG_SEED_BRANCH_ID_UNKNOWN")
            actual_seed_branches[seed_id].add(branch)
            source_roles = {
                source["relation_role"]
                for fact in facts_by_occurrence.get(occurrence_id, [])
                for source in fact["sources"]
            }
            require("immutable_prebranch_state" in source_roles, "SST_BRANCH_GFG_SEED_PRESTATE_SOURCE_MISSING")
            if branch in {"full_step", "skip_step"}:
                require(
                    any(role.endswith("training_batch_inputs") for role in source_roles)
                    and any(role.endswith("training_batch_targets") for role in source_roles),
                    "SST_BRANCH_GFG_SEED_BATCH_SOURCE_INCOMPLETE",
                )
        elif occurrence_type == "branch_horizon_state_materialization_occurrence":
            state_outcomes = [
                objects[fact["outcome"]["object_id"]]
                for fact in facts_by_occurrence[occurrence_id]
                if objects[fact["outcome"]["object_id"]]["object_kind"] == "restorable_branch_state"
            ]
            require(len(state_outcomes) == 1, "SST_BRANCH_GFG_HORIZON_STATE_OUTCOME_NOT_UNIQUE")
            actual_horizons[str(payload["seed_id"])].add(
                (int(payload["horizon"]), str(state_outcomes[0]["payload"]["branch"]))
            )
        elif occurrence_type == "actual_aligned_branch_training_step_occurrence":
            actual_continuations[str(payload["seed_id"])].add((int(occurrence["optimizer_step"]), str(payload["branch"])))
        elif occurrence_type == "matched_four_branch_causal_contrast_occurrence":
            actual_contrasts[str(payload["seed_id"])].add(int(payload["horizon"]))

    for seed_id, horizons in expected_horizons.items():
        require(actual_seed_branches[seed_id] == set(BRANCHES), f"SST_BRANCH_GFG_SEED_BRANCH_COVERAGE_INVALID:{seed_id}")
        require(
            actual_horizons[seed_id] == {(horizon, branch) for horizon in horizons for branch in BRANCHES},
            f"SST_BRANCH_GFG_HORIZON_COVERAGE_INVALID:{seed_id}",
        )
        require(
            actual_continuations[seed_id]
            == {(step, branch) for step in expected_continuations[seed_id] for branch in BRANCHES},
            f"SST_BRANCH_GFG_CONTINUATION_COVERAGE_INVALID:{seed_id}",
        )
        require(actual_contrasts[seed_id] == horizons, f"SST_BRANCH_GFG_CONTRAST_COVERAGE_INVALID:{seed_id}")
    expected_intermediate_dispositions = sum(len(steps) * len(BRANCHES) for steps in expected_continuations.values())
    require(
        intermediate_disposition_count == expected_intermediate_dispositions,
        "SST_BRANCH_GFG_INTERMEDIATE_DISPOSITION_COUNT_MISMATCH",
    )
    require(causal_difference_count > 0, "SST_BRANCH_GFG_D_U_S_K_EMPTY")

    material = {
        "schema": "nanogpt-stepwise-causal-branch-gfg-validation-v1",
        "status": "PASS",
        "database_sha256": manifest["database_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "main_stepwise_gfg_manifest_sha256": main_manifest["manifest_sha256"],
        "structural_validation_sha256": structural["validation_sha256"],
        "counts": structural["counts"],
        "seed_count": len(seed_roots),
        "horizon_observation_count": sum(len(values) for values in actual_horizons.values()),
        "aligned_continuation_count": sum(len(values) for values in actual_continuations.values()),
        "causal_contrast_count": sum(len(values) for values in actual_contrasts.values()),
        "actual_update_causal_difference_count": causal_difference_count,
        "batch_selection_order_disposition_count": batch_disposition_count,
        "intermediate_payload_disposition_count": intermediate_disposition_count,
        "validation_target_recovery_fact_count": validation_target_recovery_count,
        "main_generated_origin_count": len(main_origin_objects),
        "trajectory_causal_difference_separation": "PASS",
        "branch_coverage": "PASS",
        "main_gfg_origin_traceability": "PASS",
        "future_leakage_audit": "PASS",
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(output_path, result)
    return result
