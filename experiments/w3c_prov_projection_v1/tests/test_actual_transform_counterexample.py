from __future__ import annotations

import json
from pathlib import Path

from experiments.w3c_prov_projection_v1.src.events import GeneratorVariant
from experiments.w3c_prov_projection_v1.src.science_runs import (
    ACTUAL_TRANSFORM_REQUIRED_FALSE,
    ACTUAL_TRANSFORM_REQUIRED_TRUE,
    actual_transform_context_counterexample,
    run_full,
    run_transform_counterexample_negative_controls,
    strict_projection_counterexamples,
)


ROOT = Path(__file__).resolve().parents[1]


def test_actual_transform_contract_matches_executable_gate() -> None:
    contract = json.loads(
        (ROOT / "profiles" / "actual_transform_counterexample_contract_v1.json").read_text(encoding="utf-8")
    )
    assert tuple(contract["required_true_conditions"]) == ACTUAL_TRANSFORM_REQUIRED_TRUE
    assert tuple(contract["required_false_conditions"]) == ACTUAL_TRANSFORM_REQUIRED_FALSE


def test_two_actual_transform_branches_execute_and_project_identically() -> None:
    left = run_full(GeneratorVariant(transform_variant="left_associative"))
    right = run_full(GeneratorVariant(transform_variant="right_associative"))
    left_receipt = left.transform_receipts[0]
    right_receipt = right.transform_receipts[0]

    assert left_receipt.executed_function_or_code_path == "generator._execute_left_associative"
    assert right_receipt.executed_function_or_code_path == "generator._execute_right_associative"
    assert left_receipt.executed_branch_id != right_receipt.executed_branch_id
    assert left_receipt.intermediate_values != right_receipt.intermediate_values
    assert left_receipt.output_value == right_receipt.output_value == 11
    assert left.output == right.output
    assert left.snapshot.snapshot_id != right.snapshot.snapshot_id
    assert left.candidate_records == right.candidate_records
    assert left.candidate_provn == right.candidate_provn
    assert left.reference_from_provo == right.reference_from_provo


def test_actual_transform_counterexample_satisfies_every_machine_condition() -> None:
    group, artifact = actual_transform_context_counterexample()
    assert group["id"] == "actual_transform_context_difference"
    assert group["status"] == "SUPPORTED"
    assert group["actual_execution_difference"]
    assert group["transform_reference_differences"]
    assert group["occurrence_payload_differences"]
    assert artifact["status"] == "SUPPORTED"
    assert all(artifact["conditions"][name] for name in ACTUAL_TRANSFORM_REQUIRED_TRUE)
    assert not any(artifact["conditions"][name] for name in ACTUAL_TRANSFORM_REQUIRED_FALSE)


def test_strict_projection_requires_all_four_groups() -> None:
    result, reverse = strict_projection_counterexamples()
    assert result["requested_group_count"] == result["valid_group_count"] == 4
    assert [item["id"] for item in result["groups"]] == [
        "evidence_profile_external_difference",
        "environment_and_operation_result_difference",
        "generated_origin_bridge_difference",
        "actual_transform_context_difference",
    ]
    assert all(item["status"] == "SUPPORTED" for item in result["groups"])
    assert reverse["status"] == "SUPPORTED"


def test_all_transform_counterexample_mutations_fail_closed() -> None:
    result = run_transform_counterexample_negative_controls()
    assert result["status"] == "SUPPORTED"
    assert result["detected_count"] == result["negative_control_count"] == 10
    assert all(item["status"] == "FAIL_CLOSED" for item in result["controls"])
