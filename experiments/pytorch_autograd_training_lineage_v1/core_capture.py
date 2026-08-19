from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

from compat.v2.projections import derive_legacy_projections
from generation_relation_core.canonical import canonical_bytes, payload_sha256
from generation_relation_core.entities import (
    environment_record,
    evidence_link,
    explicit_disposition,
    generated_origin,
    generation_binding,
    generation_occurrence,
    generator_manifest,
    generator_operation_result,
    perceptual_support,
    predicate_profile,
    relation_evidence_for_material,
    relation_material,
    source_information,
    support_space,
)
from generation_relation_core.predicate_registry import PredicateRegistry, implementation_sha256
from generation_relation_core.snapshots import (
    CoreV3Tables,
    SnapshotValidation,
    ValidatedSnapshot,
    build_snapshot,
    implementation_hashes,
    validate_snapshot,
)

from .pipeline import RuntimeEvent


DOMAIN_SCOPE = "pytorch-autograd-training-lineage-v1"
TORCH_WHEEL_SHA256 = "a8b450c1e58e5800e5b4691dac412f8d2d65a1dc3298166f91596603a3531e6f"


def training_support_predicate(support_payload: dict, query_payload: dict, predicate: str) -> bool:
    return predicate == "membership" and support_payload["support_kind"] == query_payload["support_kind"]


@dataclass(frozen=True)
class CoreCaptureResult:
    snapshot: ValidatedSnapshot
    validation: SnapshotValidation
    registry: PredicateRegistry
    execution_receipts: list[dict[str, Any]]


class CoreTrainingCollector:
    """Synchronous write-only collector for actual training callbacks."""

    def __init__(self) -> None:
        self._receipts: list[dict[str, Any]] = []

    def write(self, event: RuntimeEvent) -> None:
        self._receipts.append({
            "kind": event.kind,
            "ordinal": event.ordinal,
            "payload": deepcopy(event.payload),
            "stage": event.stage,
            "step_key": event.step_key,
        })
        return None

    @property
    def receipts(self) -> list[dict[str, Any]]:
        return deepcopy(self._receipts)

    def finalize(self, *, evidence_context: str) -> CoreCaptureResult:
        return build_core_capture(self.receipts, evidence_context=evidence_context)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_hash() -> str:
    root = Path(__file__).resolve().parent
    rows = []
    for name in ("core_capture.py", "pipeline.py"):
        path = root / name
        rows.append({"name": name, "sha256": _file_sha256(path)})
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


def _profile_bundle() -> tuple[list[dict], list[dict], PredicateRegistry]:
    space = support_space(
        domain_scope_id=DOMAIN_SCOPE,
        support_space_name="training-generation-outcome-v1",
        support_payload_schema={
            "type": "object",
            "required": ["support_key", "support_kind"],
            "properties": {
                "support_key": {"type": "string"},
                "support_kind": {"type": "string"},
            },
            "additionalProperties": True,
        },
        query_payload_schema={
            "type": "object",
            "required": ["support_kind"],
            "properties": {"support_kind": {"type": "string"}},
            "additionalProperties": False,
        },
        normalization_rule="canonical JSON; tensor values retain float64 list order",
    )
    profile = predicate_profile(
        domain_scope_id=DOMAIN_SCOPE,
        support_space_id=space["support_space_id"],
        predicate_kind="exact_training_support_kind",
        supported_predicates=["membership"],
        predicate_authority="experiment-local-frozen-profile",
        authorized=True,
        implementation_module="experiments.pytorch_autograd_training_lineage_v1.core_capture",
        implementation_symbol="training_support_predicate",
        predicate_implementation_sha256=implementation_sha256(training_support_predicate),
        normalization_rule="exact UTF-8 support_kind equality",
        result_ordering_rule="support_id ascending",
    )
    registry = PredicateRegistry(
        [space],
        [profile],
        {profile["predicate_profile_id"]: training_support_predicate},
    )
    return [space], [profile], registry


def _origin_id(reference: dict[str, str]) -> str:
    return reference.get("source_information_id", reference.get("generated_origin_id", ""))


def _outcome_id(reference: dict[str, str]) -> str:
    return reference.get("support_id", reference.get("disposition_id", ""))


def _leaf_source_ref(leaf_ref: str) -> str:
    category, name = leaf_ref.split(":", 1)
    if category == "input":
        return f"source:sample:{name}"
    if category == "parameter":
        return f"source:parameter:{name}:before"
    raise ValueError(f"UNKNOWN_LEAF_REF:{leaf_ref}")


def _gradient_dependency_refs(
    operation_receipts: list[dict[str, Any]],
    loss_ref: str,
) -> dict[str, set[str]]:
    """Derive frozen local reverse-AD value dependencies from actual callbacks."""
    producers = {row["payload"]["output_ref"]: row for row in operation_receipts}
    original = [row for row in operation_receipts if row["stage"] == "forward_original"]
    recomputation = [
        row for row in operation_receipts if row["stage"] == "backward_recomputation"
    ]
    rewrite: dict[str, str] = {}
    if original and recomputation:
        original_terminal = max(original, key=lambda row: row["ordinal"])["payload"]["output_ref"]
        recomputation_terminal = max(
            recomputation, key=lambda row: row["ordinal"]
        )["payload"]["output_ref"]
        rewrite[original_terminal] = recomputation_terminal

    dependencies_by_ref: dict[str, set[str]] = {loss_ref: set()}
    pending = [loss_ref]
    leaf_dependencies: dict[str, set[str]] = {}
    while pending:
        output_ref = pending.pop(0)
        current = dependencies_by_ref[output_ref]
        receipt = producers.get(output_ref)
        if receipt is None:
            leaf_dependencies.setdefault(output_ref, set()).update(current)
            continue
        payload = receipt["payload"]
        operation = payload["operation_type"]
        inputs = payload["input_refs"]
        for slot, input_ref in enumerate(inputs):
            if not payload["input_tensors"][slot]["requires_grad"]:
                continue
            effective_input = rewrite.get(input_ref, input_ref)
            local: set[str] = set()
            if operation in {"tracked_matmul", "tracked_mul"}:
                local.update(rewrite.get(value, value) for index, value in enumerate(inputs) if index != slot)
            elif operation == "tracked_relu":
                local.add(rewrite.get(output_ref, output_ref))
            elif operation in {"tracked_pow", "tracked_sin"}:
                local.add(effective_input)
            elif operation not in {"tracked_add", "tracked_mean", "tracked_sum"}:
                raise ValueError(f"UNDECLARED_LOCAL_GRADIENT_RULE:{operation}")
            updated = dependencies_by_ref.get(effective_input, set()) | current | local
            if updated != dependencies_by_ref.get(effective_input):
                dependencies_by_ref[effective_input] = updated
                pending.append(effective_input)
    return leaf_dependencies


def build_core_capture(
    receipts: list[dict[str, Any]],
    *,
    evidence_context: str,
) -> CoreCaptureResult:
    support_spaces, profiles, registry = _profile_bundle()
    profile = profiles[0]
    space = support_spaces[0]
    code_hash = _code_hash()
    environment = environment_record(
        runtime_name="CPython+PyTorch",
        runtime_version=f"{platform.python_version()}+torch-2.13.0+cpu",
        operating_system=f"{platform.system()}-{platform.release()}",
        dependency_hashes={
            "collector_code": code_hash,
            "evidence_context": hashlib.sha256(evidence_context.encode("utf-8")).hexdigest(),
            "torch_wheel": TORCH_WHEEL_SHA256,
        },
    )
    manifest = generator_manifest(
        generator_name="PyTorch eager Autograd training step collector",
        generator_version="pytorch-autograd-training-lineage-v1",
        generator_code_hash=code_hash,
        supported_support_space_ids=[space["support_space_id"]],
        supported_predicate_profile_ids=[profile["predicate_profile_id"]],
        supported_operations=[
            "capture_backward",
            "capture_forward_operation",
            "capture_gradient",
            "capture_optimizer_update",
            "capture_recomputation_operation",
        ],
        authorized_evidence_authorities=["actual-write-only-pytorch-callback-v1"],
        dependency_hashes=[TORCH_WHEEL_SHA256],
    )

    sources: list[dict] = []
    occurrences: list[dict] = []
    generated_origins: list[dict] = []
    supports: list[dict] = []
    dispositions: list[dict] = []
    bindings: list[dict] = []
    evidence_records: list[dict] = []
    evidence_links: list[dict] = []
    operation_results: list[dict] = []

    source_by_ref: dict[str, dict] = {}
    support_by_ref: dict[str, dict] = {}
    occurrence_by_event: dict[tuple[str, str, int], dict] = {}
    generated_by_support: dict[str, dict] = {}

    def add_occurrence(receipt: dict[str, Any], occurrence_type: str, payload: dict[str, Any]) -> dict:
        key = (receipt["stage"], receipt["kind"], receipt["ordinal"])
        if key in occurrence_by_event:
            return occurrence_by_event[key]
        row = generation_occurrence(
            domain_scope_id=DOMAIN_SCOPE,
            generator_manifest_id=manifest["generator_manifest_id"],
            occurrence_stage=receipt["stage"],
            occurrence_type=occurrence_type,
            stable_instance_key=(
                f"{receipt['step_key']}:{receipt['stage']}:{receipt['kind']}:{receipt['ordinal']}"
            ),
            occurrence_index=receipt["ordinal"],
            transform_reference={
                "authority": "actual PyTorch public execution",
                "operation_type": occurrence_type,
            },
            occurrence_payload=payload,
        )
        occurrences.append(row)
        occurrence_by_event[key] = row
        return row

    def add_support(support_key: str, support_kind: str, payload: dict[str, Any]) -> dict:
        if support_key in support_by_ref:
            raise ValueError(f"DUPLICATE_SUPPORT_KEY:{support_key}")
        row = perceptual_support(
            domain_scope_id=DOMAIN_SCOPE,
            support_space_id=space["support_space_id"],
            support_payload={
                **payload,
                "support_key": support_key,
                "support_kind": support_kind,
            },
            predicate_profile_id=profile["predicate_profile_id"],
        )
        supports.append(row)
        support_by_ref[support_key] = row
        return row

    def generated_for(support: dict) -> dict:
        support_id = support["support_id"]
        if support_id not in generated_by_support:
            row = generated_origin(
                domain_scope_id=DOMAIN_SCOPE,
                generator_manifest_id=manifest["generator_manifest_id"],
                origin_type="prior_training_support",
                origin_payload={
                    "source_support_id": support_id,
                    "support_key": support["support_payload"]["support_key"],
                    "support_kind": support["support_payload"]["support_kind"],
                },
            )
            generated_origins.append(row)
            generated_by_support[support_id] = row
        return generated_by_support[support_id]

    def add_binding(origin_reference: dict, occurrence: dict, outcome_reference: dict, role: str) -> dict:
        material = relation_material(
            domain_scope_id=DOMAIN_SCOPE,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence["generation_occurrence_id"],
            outcome_reference=outcome_reference,
            relation_role=role,
        )
        related = [
            _origin_id(origin_reference),
            occurrence["generation_occurrence_id"],
            _outcome_id(outcome_reference),
        ]
        evidence = relation_evidence_for_material(
            material,
            artifact_locator=f"candidate://relation_materials.jsonl#sha256={payload_sha256(material)}",
            evidence_authority="actual-write-only-pytorch-callback-v1",
            extraction_method="synchronous post-operation callback or actual tensor/optimizer boundary hook",
            extraction_code_hash=code_hash,
            environment_hash=environment["environment_payload_sha256"],
            related_record_ids=related,
        )
        binding = generation_binding(
            domain_scope_id=DOMAIN_SCOPE,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence["generation_occurrence_id"],
            outcome_reference=outcome_reference,
            relation_role=role,
            evidence_ids=[evidence["evidence_id"]],
        )
        link = evidence_link(
            evidence_id=evidence["evidence_id"],
            subject_type="generation_binding",
            subject_id=binding["generation_binding_id"],
            evidence_role="primary_generation_relation",
        )
        operation = generator_operation_result(
            generator_manifest_id=manifest["generator_manifest_id"],
            operation_name="capture_generation_binding",
            produced_entity_ids=[binding["generation_binding_id"]],
            evidence_ids=[evidence["evidence_id"]],
        )
        evidence_records.append(evidence)
        bindings.append(binding)
        evidence_links.append(link)
        operation_results.append(operation)
        return binding

    for receipt in receipts:
        if receipt["kind"] not in {"source", "source_metadata"}:
            continue
        payload = receipt["payload"]
        source_ref = payload["source_ref"]
        if source_ref in source_by_ref:
            raise ValueError(f"DUPLICATE_SOURCE_REF:{source_ref}")
        source = source_information(
            domain_scope_id=DOMAIN_SCOPE,
            source_identity=(
                f"{receipt['step_key']}:{payload['source_identity']}:{payload['version']}"
            ),
            source_parent_id=None,
            source_granularity=payload["source_role"],
            source_payload={
                **payload,
                "evidence_context": evidence_context,
                "receipt_ordinal": receipt["ordinal"],
                "stage": receipt["stage"],
            },
        )
        sources.append(source)
        source_by_ref[source_ref] = source

    loss_ref = next(row["payload"]["loss_ref"] for row in receipts if row["kind"] == "backward_start")
    operation_receipts = [row for row in receipts if row["kind"] == "operation"]
    for receipt in operation_receipts:
        payload = receipt["payload"]
        output_ref = payload["output_ref"]
        kind = "loss" if output_ref == loss_ref else (
            "recomputed_activation" if receipt["stage"] == "backward_recomputation" else "activation"
        )
        occurrence = add_occurrence(receipt, payload["operation_type"], {
            **payload,
            "callback_kind": "actual_post_operation",
            "evidence_context": evidence_context,
            "stage": receipt["stage"],
        })
        support = add_support(output_ref, kind, {
            "operation_type": payload["operation_type"],
            "stage": receipt["stage"],
            "tensor": payload["output_tensor"],
        })
        for slot, input_ref in enumerate(payload["input_refs"]):
            if input_ref in source_by_ref:
                origin = {
                    "kind": "registered_source",
                    "source_information_id": source_by_ref[input_ref]["source_information_id"],
                }
            elif input_ref in support_by_ref:
                generated = generated_for(support_by_ref[input_ref])
                origin = {
                    "kind": "generated_origin",
                    "generated_origin_id": generated["generated_origin_id"],
                }
            else:
                raise ValueError(f"OPERATION_INPUT_ORIGIN_MISSING:{input_ref}")
            add_binding(
                origin,
                occurrence,
                {"kind": "support", "support_id": support["support_id"]},
                f"operation_input|slot={slot}|ordinal={slot}",
            )

    backward_receipt = next(row for row in receipts if row["kind"] == "backward_start")
    backward_occurrence = add_occurrence(
        backward_receipt,
        "torch.Tensor.backward",
        {"loss_ref": loss_ref, "stage": "backward"},
    )
    backward_support = add_support(
        f"{backward_receipt['step_key']}:backward:completion",
        "backward_completion",
        {"loss_support_id": support_by_ref[loss_ref]["support_id"]},
    )
    loss_origin = generated_for(support_by_ref[loss_ref])
    add_binding(
        {"kind": "generated_origin", "generated_origin_id": loss_origin["generated_origin_id"]},
        backward_occurrence,
        {"kind": "support", "support_id": backward_support["support_id"]},
        "backward_from_loss|ordinal=0",
    )

    gradient_by_parameter_name: dict[str, dict] = {}
    gradient_dependencies = _gradient_dependency_refs(operation_receipts, loss_ref)
    for receipt in [row for row in receipts if row["kind"] == "gradient"]:
        leaf_ref = receipt["payload"]["leaf_ref"]
        occurrence = add_occurrence(
            receipt,
            "tensor_gradient_hook",
            {
                **receipt["payload"],
                "backward_occurrence_id": backward_occurrence["generation_occurrence_id"],
                "backward_occurrence_key": backward_occurrence["stable_instance_key"],
                "stage": "gradient_production",
            },
        )
        support_key = f"{receipt['step_key']}:gradient:{leaf_ref}"
        support = add_support(support_key, "gradient", {
            "gradient": receipt["payload"]["gradient"],
            "leaf_ref": leaf_ref,
        })
        backward_origin = generated_for(backward_support)
        add_binding(
            {"kind": "generated_origin", "generated_origin_id": backward_origin["generated_origin_id"]},
            occurrence,
            {"kind": "support", "support_id": support["support_id"]},
            "backward_invocation_context|ordinal=0",
        )
        source_ref = _leaf_source_ref(leaf_ref)
        source = source_by_ref[source_ref]
        add_binding(
            {"kind": "registered_source", "source_information_id": source["source_information_id"]},
            occurrence,
            {"kind": "support", "support_id": support["support_id"]},
            "gradient_for_leaf|ordinal=1",
        )
        runtime_leaf_ref = _leaf_source_ref(leaf_ref)
        for dependency_ordinal, dependency_ref in enumerate(
            sorted(gradient_dependencies.get(runtime_leaf_ref, set())),
            start=2,
        ):
            if dependency_ref in source_by_ref:
                dependency_origin = {
                    "kind": "registered_source",
                    "source_information_id": source_by_ref[dependency_ref]["source_information_id"],
                }
            elif dependency_ref in support_by_ref:
                dependency_generated = generated_for(support_by_ref[dependency_ref])
                dependency_origin = {
                    "kind": "generated_origin",
                    "generated_origin_id": dependency_generated["generated_origin_id"],
                }
            else:
                raise ValueError(f"GRADIENT_VALUE_DEPENDENCY_MISSING:{dependency_ref}")
            add_binding(
                dependency_origin,
                occurrence,
                {"kind": "support", "support_id": support["support_id"]},
                f"gradient_value_dependency|ordinal={dependency_ordinal}",
            )
        if leaf_ref.startswith("parameter:"):
            gradient_by_parameter_name[leaf_ref.split(":", 1)[1]] = support

    for receipt in [row for row in receipts if row["kind"] == "gradient_absent"]:
        leaf_ref = receipt["payload"]["leaf_ref"]
        occurrence = add_occurrence(
            receipt,
            "gradient_not_produced",
            {**receipt["payload"], "stage": "gradient_production"},
        )
        disposition = explicit_disposition(
            domain_scope_id=DOMAIN_SCOPE,
            core_disposition_category="inactive",
            domain_reason_code=receipt["payload"]["reason_code"],
            disposition_payload={"leaf_ref": leaf_ref},
        )
        dispositions.append(disposition)
        source = source_by_ref[_leaf_source_ref(leaf_ref)]
        add_binding(
            {"kind": "registered_source", "source_information_id": source["source_information_id"]},
            occurrence,
            {"kind": "disposition", "disposition_id": disposition["disposition_id"]},
            "unused_in_loss|ordinal=0",
        )

    optimizer_before = next(row for row in receipts if row["kind"] == "optimizer_before")
    optimizer_after = next(row for row in receipts if row["kind"] == "optimizer_after")
    optimizer_occurrence = add_occurrence(
        optimizer_after,
        "torch.optim.SGD.step",
        {
            "after": optimizer_after["payload"],
            "before": optimizer_before["payload"],
            "stage": "optimizer_update",
        },
    )
    optimizer_state_source = source_by_ref["source:optimizer:state:before"]
    optimizer_after_support = add_support(
        f"{optimizer_after['step_key']}:optimizer_state:after",
        "optimizer_state_after_step",
        {"optimizer_state": optimizer_after["payload"]["optimizer_state"]},
    )
    add_binding(
        {"kind": "registered_source", "source_information_id": optimizer_state_source["source_information_id"]},
        optimizer_occurrence,
        {"kind": "support", "support_id": optimizer_after_support["support_id"]},
        "optimizer_state_transition|ordinal=0",
    )
    for parameter_name in sorted(optimizer_after["payload"]["parameter_values"]):
        gradient_support = gradient_by_parameter_name.get(parameter_name)
        if gradient_support is None:
            continue
        before_source = source_by_ref[f"source:parameter:{parameter_name}:before"]
        after_support = add_support(
            f"{optimizer_after['step_key']}:parameter:{parameter_name}:after",
            "parameter_after_step",
            {
                "parameter_name": parameter_name,
                "semantic_version": "after_step",
                "tensor": optimizer_after["payload"]["parameter_values"][parameter_name],
            },
        )
        outcome = {"kind": "support", "support_id": after_support["support_id"]}
        add_binding(
            {"kind": "registered_source", "source_information_id": before_source["source_information_id"]},
            optimizer_occurrence,
            outcome,
            "parameter_previous_version|ordinal=0",
        )
        gradient_origin = generated_for(gradient_support)
        add_binding(
            {"kind": "generated_origin", "generated_origin_id": gradient_origin["generated_origin_id"]},
            optimizer_occurrence,
            outcome,
            "optimizer_gradient_input|ordinal=1",
        )
        add_binding(
            {"kind": "generated_origin", "generated_origin_id": gradient_origin["generated_origin_id"]},
            optimizer_occurrence,
            {"kind": "support", "support_id": optimizer_after_support["support_id"]},
            f"optimizer_gradient_state_input|parameter={parameter_name}",
        )
        add_binding(
            {"kind": "registered_source", "source_information_id": optimizer_state_source["source_information_id"]},
            optimizer_occurrence,
            outcome,
            "optimizer_state_input|ordinal=2",
        )

    tables = CoreV3Tables(
        source_information_records=sources,
        generation_occurrences=occurrences,
        generated_origins=generated_origins,
        perceptual_support_records=supports,
        explicit_dispositions=dispositions,
        generation_bindings=bindings,
        support_space_records=support_spaces,
        predicate_profiles=profiles,
        evidence_records=evidence_records,
        evidence_links=evidence_links,
        generator_manifests=[manifest],
        generator_operation_results=operation_results,
        environment_records=[environment],
    )
    (
        tables.legacy_source_binding_projections,
        tables.legacy_occurrence_binding_projections,
    ) = derive_legacy_projections(
        tables.source_information_records,
        tables.generation_occurrences,
        tables.generation_bindings,
    )
    implementation = implementation_hashes()
    snapshot = build_snapshot(tables, registry, expected_implementation_hashes=implementation)
    validation = validate_snapshot(snapshot, registry, expected_implementation_hashes=implementation)
    return CoreCaptureResult(snapshot, validation, registry, deepcopy(receipts))
