from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from generation_relation_core.snapshots import ValidatedSnapshot

from .candidate_projection import project_snapshot
from .core_capture import CoreCaptureCollector
from .events import GeneratedOutput, GeneratorVariant, TransformExecutionReceipt
from .generator import run_generator
from .native_reference import NativeProvCollector
from .provn import parse_provn, serialize_provn
from .provo_normalizer import normalize_provo
from .record_model import canonical_json_bytes


@dataclass(frozen=True)
class FullRun:
    output: GeneratedOutput
    snapshot: Any
    candidate_records: list[dict[str, Any]]
    candidate_provn: bytes
    native_ttl: bytes
    candidate_from_provn: list[dict[str, Any]]
    reference_from_provo: list[dict[str, Any]]
    transform_receipts: tuple[TransformExecutionReceipt, ...]


class BranchReceiptCollector:
    def __init__(self) -> None:
        self.receipts: list[TransformExecutionReceipt] = []

    def on_transform_execution(self, receipt: TransformExecutionReceipt) -> None:
        self.receipts.append(receipt)


def run_full(variant: GeneratorVariant = GeneratorVariant()) -> FullRun:
    core = CoreCaptureCollector(variant)
    native = NativeProvCollector()
    receipt_collector = BranchReceiptCollector()
    output = run_generator([core, native], variant=variant, receipt_sinks=[receipt_collector])
    snapshot = core.validated_snapshot()
    candidate = project_snapshot(snapshot)
    provn = serialize_provn(candidate)
    ttl = native.qualified_provo()
    return FullRun(
        output,
        snapshot,
        candidate,
        provn,
        ttl,
        parse_provn(provn),
        normalize_provo(ttl),
        tuple(receipt_collector.receipts),
    )


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _output_digest(output: GeneratedOutput) -> str:
    return _sha(canonical_json_bytes({name: value.hex() for name, value in sorted(output.files.items())}))


def _records_of_kind(records: list[dict[str, Any]], kinds: set[str]) -> list[dict[str, Any]]:
    return [record for record in records if record["kind"] in kinds]


def _table_differences(first: ValidatedSnapshot, second: ValidatedSnapshot) -> list[str]:
    first_tables = vars(first.tables)
    second_tables = vars(second.tables)
    return sorted(
        name
        for name in set(first_tables) | set(second_tables)
        if canonical_json_bytes(first_tables.get(name)) != canonical_json_bytes(second_tables.get(name))
    )


def _occurrence_differences(
    first: ValidatedSnapshot,
    second: ValidatedSnapshot,
    field: str,
) -> list[dict[str, Any]]:
    first_rows = {row["stable_instance_key"]: row for row in first.tables.generation_occurrences}
    second_rows = {row["stable_instance_key"]: row for row in second.tables.generation_occurrences}
    differences = []
    for key in sorted(set(first_rows) | set(second_rows)):
        first_value = first_rows.get(key, {}).get(field)
        second_value = second_rows.get(key, {}).get(field)
        if first_value != second_value:
            differences.append({
                "first": first_value,
                "second": second_value,
                "stable_instance_key": key,
            })
    return differences


def _branch_receipt(run: FullRun) -> dict[str, Any]:
    if len(run.transform_receipts) != 1:
        raise ValueError(f"expected one transform receipt, found {len(run.transform_receipts)}")
    receipt = asdict(run.transform_receipts[0])
    receipt["intermediate_state_sha256"] = _sha(canonical_json_bytes(receipt["intermediate_values"]))
    receipt["operation_result_sha256"] = _sha(
        canonical_json_bytes(run.snapshot.tables.generator_operation_results[0])
    )
    receipt["branch_execution_receipt_sha256"] = _sha(canonical_json_bytes(receipt))
    return receipt


def _group(
    name: str,
    first: FullRun,
    second: FullRun,
    changed: list[str],
    *,
    require_actual_execution_difference: bool,
) -> dict[str, Any]:
    output_equal = first.output == second.output
    candidate_equal = first.candidate_records == second.candidate_records
    provn_equal = first.candidate_provn == second.candidate_provn
    provo_equal = first.reference_from_provo == second.reference_from_provo
    snapshots_differ = first.snapshot.snapshot_id != second.snapshot.snapshot_id
    both_paths_valid = (
        first.candidate_records == first.candidate_from_provn == first.reference_from_provo
        and second.candidate_records == second.candidate_from_provn == second.reference_from_provo
    )
    transform_differences = _occurrence_differences(first.snapshot, second.snapshot, "transform_reference")
    payload_differences = _occurrence_differences(first.snapshot, second.snapshot, "occurrence_payload")
    first_receipt = _branch_receipt(first)
    second_receipt = _branch_receipt(second)
    actual_execution_difference = all((
        first_receipt["executed_branch_id"] != second_receipt["executed_branch_id"],
        first_receipt["executed_function_or_code_path"] != second_receipt["executed_function_or_code_path"],
        first_receipt["intermediate_state_sha256"] != second_receipt["intermediate_state_sha256"],
    ))
    binding_semantics_equal = _records_of_kind(
        first.candidate_records, {"usage", "generation", "derivation", "association"}
    ) == _records_of_kind(second.candidate_records, {"usage", "generation", "derivation", "association"})
    valid = all((
        output_equal,
        candidate_equal,
        provn_equal,
        provo_equal,
        snapshots_differ,
        both_paths_valid,
        binding_semantics_equal,
        actual_execution_difference if require_actual_execution_difference else True,
    ))
    prov_digest = _sha(canonical_json_bytes(first.candidate_records))
    return {
        "actual_execution_difference": actual_execution_difference,
        "binding_semantics_equal": binding_semantics_equal,
        "candidate_native_both_valid": both_paths_valid,
        "changed_complete_facts": changed,
        "first_snapshot_id": first.snapshot.snapshot_id,
        "id": name,
        "normalized_prov_dm_equal": candidate_equal,
        "occurrence_payload_differences": payload_differences,
        "ordinary_output_equal": output_equal,
        "ordinary_output_sha256": _output_digest(first.output),
        "prov_record_sha256": prov_digest,
        "provn_bytes_equal": provn_equal,
        "provo_normalized_equal": provo_equal,
        "second_snapshot_id": second.snapshot.snapshot_id,
        "snapshot_table_differences": _table_differences(first.snapshot, second.snapshot),
        "snapshots_differ": snapshots_differ,
        "status": "SUPPORTED" if valid else "NOT_SUPPORTED",
        "transform_reference_differences": transform_differences,
    }


ACTUAL_TRANSFORM_REQUIRED_TRUE = (
    "both_snapshots_valid",
    "ordinary_output_equal",
    "source_entities_equal",
    "profile_selected_activity_semantics_equal",
    "binding_semantics_equal",
    "candidate_prov_records_equal",
    "candidate_provn_bytes_equal",
    "native_provo_normalized_equal",
    "snapshot_ids_different",
    "transform_reference_different",
    "occurrence_payload_transform_context_different",
    "actual_executed_branch_different",
    "intermediate_state_digest_different",
)
ACTUAL_TRANSFORM_REQUIRED_FALSE = ("report_only_mutation", "metadata_only_mutation")


def evaluate_actual_transform_conditions(conditions: dict[str, bool]) -> bool:
    return all(conditions.get(name) is True for name in ACTUAL_TRANSFORM_REQUIRED_TRUE) and all(
        conditions.get(name) is False for name in ACTUAL_TRANSFORM_REQUIRED_FALSE
    )


def actual_transform_context_counterexample() -> tuple[dict[str, Any], dict[str, Any]]:
    first = run_full(GeneratorVariant(transform_variant="left_associative"))
    second = run_full(GeneratorVariant(transform_variant="right_associative"))
    changed = [
        "GenerationOccurrence.transform_reference",
        "GenerationOccurrence.occurrence_payload.actual_transform_context",
        "executed transformation branch",
        "intermediate-state digest",
        "transformation-plan digest",
        "GeneratorOperationResult execution-derived identity",
    ]
    group = _group(
        "actual_transform_context_difference",
        first,
        second,
        changed,
        require_actual_execution_difference=True,
    )
    first_receipt = _branch_receipt(first)
    second_receipt = _branch_receipt(second)
    transform_reference_different = bool(group["transform_reference_differences"])
    payload_context_different = bool(group["occurrence_payload_differences"])
    branch_different = all((
        first_receipt["executed_branch_id"] != second_receipt["executed_branch_id"],
        first_receipt["executed_function_or_code_path"] != second_receipt["executed_function_or_code_path"],
    ))
    intermediate_different = first_receipt["intermediate_state_sha256"] != second_receipt["intermediate_state_sha256"]
    conditions = {
        "actual_executed_branch_different": branch_different,
        "binding_semantics_equal": group["binding_semantics_equal"],
        "both_snapshots_valid": isinstance(first.snapshot, ValidatedSnapshot) and isinstance(second.snapshot, ValidatedSnapshot),
        "candidate_prov_records_equal": first.candidate_records == second.candidate_records,
        "candidate_provn_bytes_equal": first.candidate_provn == second.candidate_provn,
        "intermediate_state_digest_different": intermediate_different,
        "metadata_only_mutation": not (branch_different and intermediate_different and transform_reference_different),
        "native_provo_normalized_equal": first.reference_from_provo == second.reference_from_provo,
        "occurrence_payload_transform_context_different": payload_context_different,
        "ordinary_output_equal": first.output == second.output,
        "profile_selected_activity_semantics_equal": _records_of_kind(first.candidate_records, {"activity"}) == _records_of_kind(second.candidate_records, {"activity"}),
        "report_only_mutation": not (branch_different and intermediate_different and payload_context_different),
        "snapshot_ids_different": first.snapshot.snapshot_id != second.snapshot.snapshot_id,
        "source_entities_equal": [
            record for record in first.candidate_records
            if record["kind"] == "entity" and record["types"] == ["ex:SourceInformation"]
        ] == [
            record for record in second.candidate_records
            if record["kind"] == "entity" and record["types"] == ["ex:SourceInformation"]
        ],
        "transform_reference_different": transform_reference_different,
    }
    supported = evaluate_actual_transform_conditions(conditions) and group["status"] == "SUPPORTED"
    artifact = {
        "conditions": conditions,
        "contract_id": "actual_transform_counterexample_contract_v1",
        "first_branch_execution_receipt": first_receipt,
        "second_branch_execution_receipt": second_receipt,
        "status": "SUPPORTED" if supported else "ACTUAL_TRANSFORM_CONTEXT_COUNTEREXAMPLE_NOT_SUPPORTED",
    }
    return group, artifact


def strict_projection_counterexamples() -> tuple[dict[str, Any], dict[str, Any]]:
    base = GeneratorVariant()
    pairs = [
        (
            "evidence_profile_external_difference",
            base,
            GeneratorVariant(evidence_detail="evidence-alternate"),
            ["EvidenceRecord.extraction_method", "EvidenceRecord identity", "EvidenceLink identity", "GenerationBinding identity"],
        ),
        (
            "environment_and_operation_result_difference",
            base,
            GeneratorVariant(environment_detail="environment-alternate", operation_name="generate_fixture_alternate"),
            ["EnvironmentRecord", "manifest dependency hashes", "GeneratorOperationResult.operation_name"],
        ),
        (
            "generated_origin_bridge_difference",
            base,
            GeneratorVariant(operation_name="generate_fixture_alternate", bridge_detail="bridge-alternate"),
            ["GeneratedOrigin identity", "GeneratedOrigin profile-external payload", "producer operation result"],
        ),
    ]
    groups: list[dict[str, Any]] = []
    reverse_classes: list[dict[str, Any]] = []
    for name, first_variant, second_variant, changed in pairs:
        first = run_full(first_variant)
        second = run_full(second_variant)
        group = _group(name, first, second, changed, require_actual_execution_difference=False)
        groups.append(group)
        reverse_classes.append({
            "prov_record_sha256": group["prov_record_sha256"],
            "validated_snapshot_ids": sorted([first.snapshot.snapshot_id, second.snapshot.snapshot_id]),
            "preimage_count": 2,
            "unique_reconstruction_possible": False,
        })
    transform_group, transform_artifact = actual_transform_context_counterexample()
    groups.append(transform_group)
    reverse_classes.append({
        "prov_record_sha256": transform_group["prov_record_sha256"],
        "validated_snapshot_ids": sorted([transform_group["first_snapshot_id"], transform_group["second_snapshot_id"]]),
        "preimage_count": 2,
        "unique_reconstruction_possible": False,
    })
    supported = sum(group["status"] == "SUPPORTED" for group in groups)
    counterexamples = {
        "actual_transform_context_counterexample_status": transform_artifact["status"],
        "requested_group_count": 4,
        "valid_group_count": supported,
        "minimum_required": 4,
        "groups": groups,
        "status": "SUPPORTED" if supported == 4 else "NOT_SUPPORTED",
    }
    reverse = {
        "mapping": "PROV document -> candidate reconstruction of complete generation facts",
        "equivalence_classes": reverse_classes,
        "same_prov_has_multiple_valid_snapshots": all(item["preimage_count"] > 1 for item in reverse_classes),
        "conclusion": "PROV equality does not entail complete generation-fact equality within the tested claim boundary.",
        "status": "SUPPORTED" if supported == 4 else "NOT_SUPPORTED",
    }
    return counterexamples, reverse


def run_transform_counterexample_negative_controls() -> dict[str, Any]:
    _group_value, artifact = actual_transform_context_counterexample()
    baseline = artifact["conditions"]
    cases = [
        ("report_branch_name_only", {"actual_executed_branch_different": False, "intermediate_state_digest_different": False, "report_only_mutation": True}),
        ("evidence_only_disguised_as_transform", {"actual_executed_branch_different": False, "metadata_only_mutation": True}),
        ("different_branch_different_output", {"ordinary_output_equal": False}),
        ("different_branch_different_profile", {"profile_selected_activity_semantics_equal": False}),
        ("same_transform_reference", {"transform_reference_different": False}),
        ("same_occurrence_payload_context", {"occurrence_payload_transform_context_different": False}),
        ("same_intermediate_state_digest", {"intermediate_state_digest_different": False}),
        ("same_snapshot_id", {"snapshot_ids_different": False}),
        ("different_candidate_prov", {"candidate_prov_records_equal": False, "candidate_provn_bytes_equal": False}),
        ("different_native_prov", {"native_provo_normalized_equal": False}),
    ]
    controls = []
    for number, (name, mutation) in enumerate(cases, start=1):
        conditions = {**baseline, **mutation}
        detected = not evaluate_actual_transform_conditions(conditions)
        controls.append({
            "detected": detected,
            "mutation": name,
            "mutated_conditions": mutation,
            "number": number,
            "status": "FAIL_CLOSED" if detected else "UNDETECTED",
        })
    detected_count = sum(item["detected"] for item in controls)
    return {
        "control_family": "actual_transform_counterexample_mutations",
        "controls": controls,
        "detected_count": detected_count,
        "negative_control_count": len(controls),
        "status": "SUPPORTED" if detected_count == len(controls) == 10 else "NOT_SUPPORTED",
        "undetected_count": len(controls) - detected_count,
    }


def output_modes() -> dict[str, Any]:
    outputs: dict[str, GeneratedOutput] = {}
    outputs["output-only"] = run_generator()
    core_only = CoreCaptureCollector()
    outputs["Core-only"] = run_generator([core_only])
    core_only.validated_snapshot()
    native_only = NativeProvCollector()
    outputs["native-PROV-only"] = run_generator([native_only])
    native_only.qualified_provo()
    core_candidate = CoreCaptureCollector()
    outputs["Core + candidate PROV"] = run_generator([core_candidate])
    serialize_provn(project_snapshot(core_candidate.validated_snapshot()))
    dual_core = CoreCaptureCollector()
    dual_native = NativeProvCollector()
    outputs["native + Core + dual comparison"] = run_generator([dual_core, dual_native])
    dual_candidate = project_snapshot(dual_core.validated_snapshot())
    if dual_candidate != normalize_provo(dual_native.qualified_provo()):
        raise AssertionError("dual comparison failed")
    baseline = outputs["output-only"]
    comparisons = {
        name: {
            "files_equal": output.files == baseline.files,
            "metadata_equal": output.metadata() == baseline.metadata(),
            "output_sha256": _output_digest(output),
        }
        for name, output in outputs.items()
    }
    return {
        "modes": comparisons,
        "mode_count": len(comparisons),
        "all_bytes_equal": all(item["files_equal"] for item in comparisons.values()),
        "all_metadata_equal": all(item["metadata_equal"] for item in comparisons.values()),
        "forbidden_output_token_count": sum(
            token in b"".join(baseline.files.values())
            for token in (b"si3_", b"gb3_", b"ex:u_", b"prov:", b"generation_binding")
        ),
        "status": "SUPPORTED" if all(item["files_equal"] and item["metadata_equal"] for item in comparisons.values()) else "NOT_SUPPORTED",
    }
