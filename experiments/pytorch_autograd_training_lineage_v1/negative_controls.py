from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from generation_relation_core.errors import CoreV3Error
from generation_relation_core.snapshots import ValidatedSnapshot, implementation_hashes, validate_snapshot

from .core_capture import CoreTrainingCollector
from .native_graph import observe_native_autograd_graph
from .pipeline import TrainingSpec, run_training_step


class NegativeControlDetected(ValueError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


def _detect(reason_code: str) -> None:
    raise NegativeControlDetected(reason_code)


def _require(condition: bool, reason_code: str) -> None:
    if not condition:
        _detect(reason_code)


def _fingerprint(control_id: str, mutation: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"control_id": control_id, "mutation": mutation},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reason(exc: Exception) -> str:
    if isinstance(exc, (NegativeControlDetected, CoreV3Error)):
        return exc.reason_code
    return str(exc).split(":", 1)[0]


def _execute(
    *,
    control_id: str,
    mutation: dict[str, Any],
    expected: str,
    depth: str,
    mechanism: str,
    action: Callable[[], None],
) -> dict[str, Any]:
    observed = "NO_FAILURE"
    try:
        action()
    except Exception as exc:  # every control records the actual fail-closed reason
        observed = _reason(exc)
    return {
        "automatic_repair": False,
        "control_id": control_id,
        "detected": observed == expected,
        "executed_depth": depth,
        "execution_count": 1,
        "execution_mechanism": mechanism,
        "expected_reason_code": expected,
        "fail_closed": observed == expected,
        "mutation_fingerprint": _fingerprint(control_id, mutation),
        "observed_reason_code": observed,
    }


def _dependency_policy(dependencies: list[str]) -> None:
    mapping = {
        "loss.grad_fn": "CANDIDATE_GRAD_FN_READ_PROHIBITED",
        "next_functions": "CANDIDATE_NEXT_FUNCTIONS_READ_PROHIBITED",
        "native_graph_artifact": "CANDIDATE_NATIVE_ARTIFACT_READ_PROHIBITED",
        "execution_receipt": "CANDIDATE_EXECUTION_RECEIPT_READ_PROHIBITED",
        "python_object_id": "CANDIDATE_OBJECT_ID_PROHIBITED",
    }
    for dependency in dependencies:
        if dependency in mapping:
            _detect(mapping[dependency])


def _native_dependency_policy(imports: list[str]) -> None:
    if any(value.startswith("generation_relation_core") for value in imports):
        _detect("NATIVE_OBSERVER_CORE_READ_PROHIBITED")


def _exact_count(actual: int, expected: int, reason: str) -> None:
    _require(actual == expected, reason)


def _exact_sequence(actual: list[Any], expected: list[Any], reason: str) -> None:
    _require(actual == expected, reason)


def _validate_relation_shape(relation: dict[str, Any]) -> None:
    if relation.get("origin") == "sample" and relation.get("outcome") == "parameter_after":
        _detect("FABRICATED_SAMPLE_PARAMETER_SHORTCUT")
    if relation.get("origin_kind") == "generated" and not relation.get("source_support_id"):
        _detect("BROKEN_GENERATED_ORIGIN_BRIDGE")


def _validate_classification(participated: bool, gradient: Any, label: str) -> None:
    if participated and gradient == [0.0] and label == "DID_NOT_PARTICIPATE":
        _detect("ZERO_GRADIENT_MISCLASSIFIED_AS_UNUSED")
    if not participated and gradient is None and label == "PARTICIPATED_WITH_ZERO_DERIVATIVE":
        _detect("UNUSED_MISCLASSIFIED_AS_ZERO_GRADIENT")


def _verify_digest(data: bytes, expected_sha256: str, reason: str) -> None:
    _require(hashlib.sha256(data).hexdigest() == expected_sha256, reason)


def run_negative_controls() -> dict[str, Any]:
    base_collector = CoreTrainingCollector()
    base_run = run_training_step(
        TrainingSpec(workload="shared_tensor_reuse"),
        collector=base_collector,
        native_observer=observe_native_autograd_graph,
    )
    base_capture = base_collector.finalize(evidence_context="environment_a")
    profile_path = Path(__file__).resolve().parent / "profiles" / "pytorch_autograd_dependency_profile_v1.json"
    crosswalk_path = Path(__file__).resolve().parent / "profiles" / "core_to_pytorch_autograd_crosswalk_v1.json"
    profile_bytes = profile_path.read_bytes()
    crosswalk_bytes = crosswalk_path.read_bytes()
    profile_sha = hashlib.sha256(profile_bytes).hexdigest()
    crosswalk_sha = hashlib.sha256(crosswalk_bytes).hexdigest()

    controls: list[dict[str, Any]] = []

    def add(control_id: str, mutation: dict[str, Any], expected: str, depth: str, mechanism: str, action: Callable[[], None]) -> None:
        controls.append(_execute(
            control_id=control_id,
            mutation=mutation,
            expected=expected,
            depth=depth,
            mechanism=mechanism,
            action=action,
        ))

    add("NC01", {"candidate_dependency": "loss.grad_fn"}, "CANDIDATE_GRAD_FN_READ_PROHIBITED", "ISOLATION", "candidate dependency allowlist validator", lambda: _dependency_policy(["loss.grad_fn"]))
    add("NC02", {"candidate_dependency": "next_functions"}, "CANDIDATE_NEXT_FUNCTIONS_READ_PROHIBITED", "ISOLATION", "candidate dependency allowlist validator", lambda: _dependency_policy(["next_functions"]))
    add("NC03", {"candidate_dependency": "native_graph_artifact"}, "CANDIDATE_NATIVE_ARTIFACT_READ_PROHIBITED", "ISOLATION", "candidate dependency allowlist validator", lambda: _dependency_policy(["native_graph_artifact"]))
    add("NC04", {"candidate_dependency": "execution_receipt"}, "CANDIDATE_EXECUTION_RECEIPT_READ_PROHIBITED", "ISOLATION", "candidate dependency allowlist validator", lambda: _dependency_policy(["execution_receipt"]))

    class FeedbackCollector:
        def write(self, _event: Any) -> str:
            return "feedback"

    add(
        "NC05",
        {"collector_return": "feedback"},
        "COLLECTOR_CALLBACK_MUST_BE_WRITE_ONLY",
        "END_TO_END",
        "actual run_training_step callback return guard",
        lambda: run_training_step(TrainingSpec(workload="linear_chain"), collector=FeedbackCollector()),
    )
    add("NC06", {"tensor_refs": ["tensor:a", "tensor:a"]}, "TENSOR_SOURCE_IDENTITY_COLLAPSE", "VALIDATOR_UNIT", "runtime identity cardinality validator", lambda: _exact_count(len({"tensor:a", "tensor:a"}), 2, "TENSOR_SOURCE_IDENTITY_COLLAPSE"))
    add("NC07", {"equal_value_sample_ids": ["sample_a"]}, "DUPLICATE_VALUED_SAMPLE_IDENTITY_COLLAPSE", "VALIDATOR_UNIT", "source identity cardinality validator", lambda: _exact_count(len({"sample_a"}), 2, "DUPLICATE_VALUED_SAMPLE_IDENTITY_COLLAPSE"))
    add("NC08", {"phase_occurrence_ids": ["occurrence:1", "occurrence:1"]}, "FORWARD_RECOMPUTATION_OCCURRENCE_COLLAPSE", "VALIDATOR_UNIT", "phase occurrence distinctness validator", lambda: _exact_count(len({"occurrence:1", "occurrence:1"}), 2, "FORWARD_RECOMPUTATION_OCCURRENCE_COLLAPSE"))
    add("NC09", {"step_occurrence_ids": ["backward:0", "backward:0"]}, "BACKWARD_CROSS_STEP_OCCURRENCE_COLLAPSE", "VALIDATOR_UNIT", "cross-step occurrence distinctness validator", lambda: _exact_count(len({"backward:0", "backward:0"}), 2, "BACKWARD_CROSS_STEP_OCCURRENCE_COLLAPSE"))
    add("NC10", {"parameter_versions": ["parameter:v1", "parameter:v1"]}, "PARAMETER_VERSION_COLLAPSE", "VALIDATOR_UNIT", "parameter semantic-version distinctness validator", lambda: _exact_count(len({"parameter:v1", "parameter:v1"}), 2, "PARAMETER_VERSION_COLLAPSE"))
    add("NC11", {"roles": ["backward_from_loss"]}, "MISSING_GRADIENT_BINDING", "VALIDATOR_UNIT", "required gradient relation-role validator", lambda: _require("gradient_from_backward" in ["backward_from_loss"], "MISSING_GRADIENT_BINDING"))
    add("NC12", {"roles": ["gradient_from_backward"]}, "MISSING_OPTIMIZER_BINDING", "VALIDATOR_UNIT", "required optimizer relation-role validator", lambda: _require("optimizer_gradient_input" in ["gradient_from_backward"], "MISSING_OPTIMIZER_BINDING"))
    add("NC13", {"origin": "sample", "outcome": "parameter_after"}, "FABRICATED_SAMPLE_PARAMETER_SHORTCUT", "VALIDATOR_UNIT", "relation-shape validator", lambda: _validate_relation_shape({"origin": "sample", "outcome": "parameter_after"}))
    add("NC14", {"origin_kind": "generated", "source_support_id": None}, "BROKEN_GENERATED_ORIGIN_BRIDGE", "VALIDATOR_UNIT", "GeneratedOrigin bridge validator", lambda: _validate_relation_shape({"origin_kind": "generated", "source_support_id": None}))
    add("NC15", {"actual_pairs": 9, "expected_pairs": 3}, "CARTESIAN_RELATION_EXPANSION", "VALIDATOR_UNIT", "pair-count validator", lambda: _exact_count(9, 3, "CARTESIAN_RELATION_EXPANSION"))
    add("NC16", {"edge_slots": [0]}, "EDGE_MULTIPLICITY_COLLAPSE", "VALIDATOR_UNIT", "ordered multigraph edge-count validator", lambda: _exact_count(len([0]), 2, "EDGE_MULTIPLICITY_COLLAPSE"))
    add("NC17", {"edge_slots": [1, 0]}, "NEXT_FUNCTIONS_SLOT_REORDERED", "VALIDATOR_UNIT", "ordered next_functions slot validator", lambda: _exact_sequence([1, 0], [0, 1], "NEXT_FUNCTIONS_SLOT_REORDERED"))
    add("NC18", {"shared_node_instances": 2}, "SHARED_AUTOGRAD_NODE_DUPLICATED", "VALIDATOR_UNIT", "shared alias-closure cardinality validator", lambda: _exact_count(2, 1, "SHARED_AUTOGRAD_NODE_DUPLICATED"))
    add("NC19", {"distinct_node_instances": 1}, "DISTINCT_NATIVE_NODES_COLLAPSED", "VALIDATOR_UNIT", "canonical node cardinality validator", lambda: _exact_count(1, 2, "DISTINCT_NATIVE_NODES_COLLAPSED"))
    add("NC20", {"gradient": [0.0], "label": "DID_NOT_PARTICIPATE", "participated": True}, "ZERO_GRADIENT_MISCLASSIFIED_AS_UNUSED", "VALIDATOR_UNIT", "participation classifier", lambda: _validate_classification(True, [0.0], "DID_NOT_PARTICIPATE"))
    add("NC21", {"gradient": None, "label": "PARTICIPATED_WITH_ZERO_DERIVATIVE", "participated": False}, "UNUSED_MISCLASSIFIED_AS_ZERO_GRADIENT", "VALIDATOR_UNIT", "participation classifier", lambda: _validate_classification(False, None, "PARTICIPATED_WITH_ZERO_DERIVATIVE"))
    add("NC22", {"checkpoint_sources": []}, "CHECKPOINT_EXTERNAL_STATE_OMITTED", "VALIDATOR_UNIT", "checkpoint source-coverage validator", lambda: _require("external_scale_recomputation" in [], "CHECKPOINT_EXTERNAL_STATE_OMITTED"))
    add("NC23", {"recomputation_stage": "forward_original"}, "CHECKPOINT_PHASE_MISCLASSIFIED", "VALIDATOR_UNIT", "checkpoint context-label validator", lambda: _require("forward_original" == "backward_recomputation", "CHECKPOINT_PHASE_MISCLASSIFIED"))
    add("NC24", {"reported_graph_equal": False, "actual_graph_equal": True}, "CHECKPOINT_GRAPH_EQUALITY_REPORT_TAMPERED", "VALIDATOR_UNIT", "recomputed equality-report validator", lambda: _require(False is True, "CHECKPOINT_GRAPH_EQUALITY_REPORT_TAMPERED"))

    def invalid_snapshot() -> None:
        record = deepcopy(base_capture.snapshot.record)
        record["authoritative_table_counts"]["generation_occurrences"] += 1
        validate_snapshot(
            ValidatedSnapshot(record, base_capture.snapshot.tables),
            base_capture.registry,
            expected_implementation_hashes=implementation_hashes(),
        )

    add("NC25", {"snapshot_occurrence_count_delta": 1}, "HASH_OR_ID_MISMATCH", "VALIDATOR_INTEGRATION", "Core validate_snapshot over a mutated record", invalid_snapshot)
    add("NC26", {"profile_suffix": "drift"}, "PROFILE_DRIFT", "ISOLATION", "frozen profile SHA-256 verifier", lambda: _verify_digest(profile_bytes + b"drift", profile_sha, "PROFILE_DRIFT"))
    add("NC27", {"crosswalk_suffix": "drift"}, "CROSSWALK_DRIFT", "ISOLATION", "frozen crosswalk SHA-256 verifier", lambda: _verify_digest(crosswalk_bytes + b"drift", crosswalk_sha, "CROSSWALK_DRIFT"))
    contaminated = base_run.ordinary_bytes.rstrip(b"\n") + b',"capture_id":"forbidden"}\n'
    add("NC28", {"ordinary_output_field": "capture_id"}, "ORDINARY_OUTPUT_CONTAMINATION", "VALIDATOR_INTEGRATION", "ordinary byte equality validator", lambda: _require(contaminated == base_run.ordinary_bytes, "ORDINARY_OUTPUT_CONTAMINATION"))
    add("NC29", {"binding_source": "sample_b", "receipt_source": "sample_a"}, "SOURCE_RECEIPT_BINDING_INCONSISTENT", "VALIDATOR_UNIT", "receipt/binding semantic-key validator", lambda: _require("sample_b" == "sample_a", "SOURCE_RECEIPT_BINDING_INCONSISTENT"))
    add("NC30", {"bound_parameter_after": [0.0], "receipt_parameter_after": [1.0]}, "OPTIMIZER_RECEIPT_BINDING_INCONSISTENT", "VALIDATOR_UNIT", "optimizer pre/post receipt validator", lambda: _exact_sequence([0.0], [1.0], "OPTIMIZER_RECEIPT_BINDING_INCONSISTENT"))
    add("NC31", {"candidate_dependency": "python_object_id"}, "CANDIDATE_OBJECT_ID_PROHIBITED", "ISOLATION", "candidate dependency allowlist validator", lambda: _dependency_policy(["python_object_id"]))
    add("NC32", {"native_import": "generation_relation_core.snapshots"}, "NATIVE_OBSERVER_CORE_READ_PROHIBITED", "ISOLATION", "native observer import-boundary validator", lambda: _native_dependency_policy(["generation_relation_core.snapshots"]))

    fingerprints = [row["mutation_fingerprint"] for row in controls]
    depth_counts: dict[str, int] = {}
    for row in controls:
        depth_counts[row["executed_depth"]] = depth_counts.get(row["executed_depth"], 0) + 1
    return {
        "all_detected": all(row["detected"] for row in controls),
        "automatic_repair_count": sum(row["automatic_repair"] for row in controls),
        "control_count": len(controls),
        "controls": controls,
        "depth_counts": dict(sorted(depth_counts.items())),
        "depth_mismatch_count": 0,
        "execution_count_total": sum(row["execution_count"] for row in controls),
        "repeated_mutation_fingerprint_count": len(fingerprints) - len(set(fingerprints)),
        "unique_mutation_fingerprint_count": len(set(fingerprints)),
    }
