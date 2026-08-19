from __future__ import annotations

from dataclasses import asdict
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
from generation_relation_core.snapshots import CoreV3Tables, ValidatedSnapshot, build_snapshot, validate_snapshot


def native_key_membership(support: dict, query: dict, predicate: str) -> bool:
    return predicate == "membership" and support.get("native_support_key") == query.get("native_support_key")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


class EvidenceBuilder:
    def __init__(self, *, seed: int, code_hash: str, dependencies: dict[str, str]) -> None:
        self.domain = f"gfg_rl_feedback_closure_v1:seed:{seed}"
        self.code_hash = code_hash
        self.space = support_space(
            domain_scope_id=self.domain,
            support_space_name="content_addressed_rl_execution_result",
            support_payload_schema={
                "type": "object",
                "required": ["native_support_key"],
                "properties": {"native_support_key": {"type": "string", "minLength": 1}},
            },
            query_payload_schema={
                "type": "object",
                "required": ["native_support_key"],
                "properties": {"native_support_key": {"type": "string", "minLength": 1}},
            },
            normalization_rule="Core canonical JSON; arrays preserve order",
        )
        self.profile = predicate_profile(
            domain_scope_id=self.domain,
            support_space_id=self.space["support_space_id"],
            predicate_kind="native_support_key_membership",
            supported_predicates=["membership"],
            predicate_authority="gfg_rl_feedback_closure_v1",
            authorized=True,
            implementation_module=native_key_membership.__module__,
            implementation_symbol=native_key_membership.__name__,
            predicate_implementation_sha256=implementation_sha256(native_key_membership),
            normalization_rule=self.space["normalization_rule"],
            result_ordering_rule="ascending support_id",
        )
        self.environment = environment_record(
            runtime_name="CPython+PyTorch",
            runtime_version=platform.python_version(),
            operating_system=platform.platform(),
            dependency_hashes=dependencies,
        )
        self.manifest = generator_manifest(
            generator_name="GFGRLFeedbackClosureRunner",
            generator_version="1.0.0",
            generator_code_hash=code_hash,
            supported_support_space_ids=[self.space["support_space_id"]],
            supported_predicate_profile_ids=[self.profile["predicate_profile_id"]],
            supported_operations=["execute_delayed_feedback_training_update"],
            authorized_evidence_authorities=["gfg_rl_feedback_closure_v1"],
            dependency_hashes=sorted(set(dependencies.values())),
        )
        self.tables = CoreV3Tables(
            support_space_records=[self.space],
            predicate_profiles=[self.profile],
            generator_manifests=[self.manifest],
            environment_records=[self.environment],
        )
        self.sources: dict[str, dict] = {}
        self.origins: dict[str, dict] = {}
        self.occurrences: dict[str, dict] = {}
        self.supports: dict[str, dict] = {}
        self.typed_edges: list[dict[str, Any]] = []

    def source(self, identity: str, granularity: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = source_information(
            domain_scope_id=self.domain,
            source_identity=identity,
            source_parent_id=None,
            source_granularity=granularity,
            source_payload=payload,
        )
        return self.sources.setdefault(row["source_information_id"], row)

    def occurrence(self, key: str, index: int, stage: str, transform: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        row = generation_occurrence(
            domain_scope_id=self.domain,
            generator_manifest_id=self.manifest["generator_manifest_id"],
            occurrence_stage=stage,
            occurrence_type="concrete_rl_execution",
            stable_instance_key=key,
            occurrence_index=index,
            transform_reference=transform,
            occurrence_payload=payload,
        )
        return self.occurrences.setdefault(row["generation_occurrence_id"], row)

    def support(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = perceptual_support(
            domain_scope_id=self.domain,
            support_space_id=self.space["support_space_id"],
            support_payload={"native_support_key": key, **payload},
            predicate_profile_id=self.profile["predicate_profile_id"],
        )
        return self.supports.setdefault(row["support_id"], row)

    def origin_from(self, support: dict[str, Any], producing_binding: dict[str, Any], origin_type: str) -> dict[str, Any]:
        row = generated_origin(
            domain_scope_id=self.domain,
            generator_manifest_id=self.manifest["generator_manifest_id"],
            origin_type=origin_type,
            origin_payload={
                "support_id": support["support_id"],
                "producing_generation_binding_id": producing_binding["generation_binding_id"],
                "support_payload_sha256": support["support_payload_sha256"],
            },
        )
        return self.origins.setdefault(row["generated_origin_id"], row)

    @staticmethod
    def source_ref(source: dict[str, Any]) -> dict[str, str]:
        return {"kind": "registered_source", "source_information_id": source["source_information_id"]}

    @staticmethod
    def origin_ref(origin: dict[str, Any]) -> dict[str, str]:
        return {"kind": "generated_origin", "generated_origin_id": origin["generated_origin_id"]}

    def bind(
        self,
        origin_reference: dict[str, str],
        occurrence: dict[str, Any],
        outcome: dict[str, Any],
        role: str,
        locator: str,
    ) -> dict[str, Any]:
        outcome_reference = {"kind": "support", "support_id": outcome["support_id"]}
        origin_id = origin_reference.get("source_information_id") or origin_reference.get("generated_origin_id")
        if origin_id is None:
            raise RuntimeError("RL_CORE_ORIGIN_ID_MISSING")
        material = relation_material(
            domain_scope_id=self.domain,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence["generation_occurrence_id"],
            outcome_reference=outcome_reference,
            relation_role=role,
        )
        evidence = relation_evidence_for_material(
            material,
            artifact_locator=f"candidate://relation_materials.jsonl#sha256={payload_sha256(material)}",
            evidence_authority="gfg_rl_feedback_closure_v1",
            extraction_method="synchronous minibatch update receipt under frozen protocol",
            extraction_code_hash=self.code_hash,
            environment_hash=self.environment["environment_payload_sha256"],
            related_record_ids=sorted([
                origin_id, occurrence["generation_occurrence_id"], outcome["support_id"],
            ]),
        )
        binding = generation_binding(
            domain_scope_id=self.domain,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence["generation_occurrence_id"],
            outcome_reference=outcome_reference,
            relation_role=role,
            evidence_ids=[evidence["evidence_id"]],
        )
        self.tables.evidence_records.append(evidence)
        self.tables.generation_bindings.append(binding)
        self.tables.evidence_links.append(evidence_link(
            evidence_id=evidence["evidence_id"],
            subject_type="generation_binding",
            subject_id=binding["generation_binding_id"],
            evidence_role="primary_generation_relation",
        ))
        return binding

    def finish(self) -> ValidatedSnapshot:
        self.tables.source_information_records = sorted(self.sources.values(), key=lambda row: row["source_information_id"])
        self.tables.generated_origins = sorted(self.origins.values(), key=lambda row: row["generated_origin_id"])
        self.tables.generation_occurrences = sorted(self.occurrences.values(), key=lambda row: row["generation_occurrence_id"])
        self.tables.perceptual_support_records = sorted(self.supports.values(), key=lambda row: row["support_id"])
        self.tables.generator_operation_results = [generator_operation_result(
            generator_manifest_id=self.manifest["generator_manifest_id"],
            operation_name="execute_delayed_feedback_training_update",
            produced_entity_ids=sorted(row["generation_binding_id"] for row in self.tables.generation_bindings),
            evidence_ids=sorted(row["evidence_id"] for row in self.tables.evidence_records),
        )]
        source_projection, occurrence_projection = derive_legacy_projections(
            self.tables.source_information_records,
            self.tables.generation_occurrences,
            self.tables.generation_bindings,
            validate_schema=False,
        )
        self.tables.legacy_source_binding_projections = source_projection
        self.tables.legacy_occurrence_binding_projections = occurrence_projection
        registry = PredicateRegistry(
            [self.space], [self.profile], {self.profile["predicate_profile_id"]: native_key_membership},
        )
        return build_snapshot(self.tables, registry)


def serialize_snapshot(snapshot: ValidatedSnapshot) -> dict[str, Any]:
    return {"record": snapshot.record, "tables": asdict(snapshot.tables)}


def validate_serialized_snapshot(payload: dict[str, Any]) -> str:
    tables = CoreV3Tables(**payload["tables"])
    profile = tables.predicate_profiles[0]
    registry = PredicateRegistry(
        tables.support_space_records,
        tables.predicate_profiles,
        {profile["predicate_profile_id"]: native_key_membership},
    )
    snapshot = ValidatedSnapshot(record=payload["record"], tables=tables)
    return validate_snapshot(snapshot, registry).snapshot_id


def graph_from_snapshot(builder: EvidenceBuilder, snapshot: ValidatedSnapshot) -> dict[str, Any]:
    occurrences = {row["generation_occurrence_id"]: row for row in snapshot.tables.generation_occurrences}
    nodes: list[dict[str, Any]] = []
    for row in snapshot.tables.source_information_records:
        nodes.append({"node_id": row["source_information_id"], "node_kind": "source", "identity": row["source_identity"]})
    for row in snapshot.tables.generated_origins:
        nodes.append({"node_id": row["generated_origin_id"], "node_kind": "generated_origin", "payload": row["origin_payload"]})
    for row in snapshot.tables.generation_occurrences:
        nodes.append({
            "node_id": row["generation_occurrence_id"],
            "node_kind": "occurrence",
            "stage": row["occurrence_stage"],
            "transform_reference": row["transform_reference"],
        })
    for row in snapshot.tables.perceptual_support_records:
        nodes.append({"node_id": row["support_id"], "node_kind": "outcome_support", "payload": row["support_payload"]})
    edges: list[dict[str, Any]] = []
    for binding in snapshot.tables.generation_bindings:
        fact_id = binding["generation_binding_id"]
        origin = binding["origin_reference"]
        origin_id = origin.get("source_information_id") or origin.get("generated_origin_id")
        support_id = binding["outcome_reference"]["support_id"]
        nodes.append({
            "node_id": fact_id,
            "node_kind": "atomic_generation_fact",
            "u": origin,
            "tau": occurrences[binding["generation_occurrence_id"]]["transform_reference"],
            "omega": binding["generation_occurrence_id"],
            "z": binding["outcome_reference"],
            "rho": binding["relation_role"],
        })
        edges.extend([
            {"edge_kind": "origin_of_fact", "source": origin_id, "target": fact_id},
            {"edge_kind": "realizes_fact", "source": binding["generation_occurrence_id"], "target": fact_id},
            {"edge_kind": "fact_outcome", "source": fact_id, "target": support_id},
        ])
    edges.extend(builder.typed_edges)
    nodes.sort(key=lambda row: row["node_id"])
    edges.sort(key=canonical_bytes)
    graph = {
        "schema": "gfg-rl-feedback-closure-graph-v1",
        "validated_snapshot_id": snapshot.snapshot_id,
        "nodes": nodes,
        "edges": edges,
    }
    graph["graph_sha256"] = payload_sha256(graph)
    validate_graph(graph)
    return graph


def validate_graph(graph: dict[str, Any]) -> dict[str, Any]:
    material = dict(graph)
    claimed = material.pop("graph_sha256")
    if payload_sha256(material) != claimed:
        raise RuntimeError("RL_GFG_HASH_MISMATCH")
    node_ids = [row["node_id"] for row in graph["nodes"]]
    if len(node_ids) != len(set(node_ids)):
        raise RuntimeError("RL_GFG_DUPLICATE_NODE")
    admitted = set(node_ids)
    facts = {row["node_id"] for row in graph["nodes"] if row["node_kind"] == "atomic_generation_fact"}
    incidence = {fact_id: 0 for fact_id in facts}
    typed = set()
    for edge in graph["edges"]:
        if edge["source"] not in admitted or edge["target"] not in admitted:
            raise RuntimeError("RL_GFG_ENDPOINT_MISSING")
        if edge["edge_kind"] == "realizes_fact" and edge["target"] in incidence:
            incidence[edge["target"]] += 1
        typed.add(edge["edge_kind"])
    if any(value != 1 for value in incidence.values()):
        raise RuntimeError("RL_GFG_INCIDENCE_INVALID")
    required = {"program_order", "produced_consequence", "credited_to_action", "parameter_version_flow"}
    if not required <= typed:
        raise RuntimeError(f"RL_GFG_REQUIRED_RELATION_MISSING:{sorted(required - typed)}")
    return {"status": "PASS", "nodes": len(node_ids), "edges": len(graph["edges"]), "facts": len(facts)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_seed_evidence(seed_result_path: Path, output_dir: Path) -> dict[str, Any]:
    seed_result = json.loads(seed_result_path.read_text(encoding="utf-8"))
    seed = int(seed_result["seed"])
    ledger_paths = {condition: Path(seed_result["conditions"][condition]["ledger"]) for condition in ("A", "B", "C")}
    dependencies = {f"ledger_{condition}": file_sha256(path) for condition, path in ledger_paths.items()}
    builder = EvidenceBuilder(seed=seed, code_hash=file_sha256(Path(__file__)), dependencies=dependencies)
    occurrence_index = 0
    for condition, ledger_path in ledger_paths.items():
        previous_occurrence: str | None = None
        for line_number, receipt in enumerate(_load_jsonl(ledger_path), start=1):
            update = int(receipt["update"])
            locator = f"{ledger_path.as_uri()}#line={line_number}"
            cue_source = builder.source(
                f"{condition}:{update}:cue:{receipt['cue_batch_sha256']}",
                "minibatch_cue_ledger",
                {"condition": condition, "update": update, "sha256": receipt["cue_batch_sha256"]},
            )
            pre_state = builder.source(
                f"{condition}:{update}:pre-state:{receipt['pre_state_sha256']}",
                "parameter_optimizer_receiving_state",
                {"condition": condition, "update": update, "sha256": receipt["pre_state_sha256"]},
            )
            infer = builder.occurrence(
                f"{condition}:{update}:frozen-inference", occurrence_index, "frozen_inference",
                {"operation": "two_stage_gru_policy_inference", "persistent_state_update": False},
                {"condition": condition, "update": update, "uniform_sha256": receipt["uniform_batch_sha256"]},
            )
            occurrence_index += 1
            action_support = builder.support(
                f"{condition}:{update}:actions:{receipt['action_ledger_sha256']}",
                {"kind": "two_stage_action_ledger", "sha256": receipt["action_ledger_sha256"]},
            )
            action_fact = builder.bind(builder.source_ref(cue_source), infer, action_support, "cue_conditions_frozen_action_inference", locator)
            action_origin = builder.origin_from(action_support, action_fact, "generated_action_ledger")

            consequence = builder.occurrence(
                f"{condition}:{update}:terminal-consequence", occurrence_index, "environment_consequence",
                {"operation": "terminal_two_component_consequence"}, {"condition": condition, "update": update},
            )
            occurrence_index += 1
            consequence_support = builder.support(
                f"{condition}:{update}:physical:{receipt['physical_consequence_sha256']}",
                {"kind": "physical_terminal_consequence", "sha256": receipt["physical_consequence_sha256"]},
            )
            consequence_fact = builder.bind(
                builder.origin_ref(action_origin), consequence, consequence_support,
                "actions_produce_terminal_consequence", locator,
            )
            consequence_origin = builder.origin_from(consequence_support, consequence_fact, "generated_physical_consequence")

            binding_occurrence = builder.occurrence(
                f"{condition}:{update}:binding", occurrence_index, "consequence_binding",
                {"operation": "condition_specific_episode_binding", "condition": condition},
                {"condition": condition, "update": update, "binding_source_sha256": receipt["binding_source_sha256"]},
            )
            occurrence_index += 1
            assigned_support = builder.support(
                f"{condition}:{update}:assigned:{receipt['assigned_consequence_sha256']}",
                {"kind": "assigned_consequence_ledger", "sha256": receipt["assigned_consequence_sha256"]},
            )
            assigned_fact = builder.bind(
                builder.origin_ref(consequence_origin), binding_occurrence, assigned_support,
                "episode_consequence_binding", locator,
            )
            assigned_origin = builder.origin_from(assigned_support, assigned_fact, "generated_assigned_consequence")

            credit = builder.occurrence(
                f"{condition}:{update}:credit", occurrence_index, "temporal_credit",
                {"operation": "condition_specific_temporal_credit", "condition": condition},
                {"condition": condition, "update": update, "credit_target_sha256": receipt["credit_target_sha256"]},
            )
            occurrence_index += 1
            credit_support = builder.support(
                f"{condition}:{update}:credit:{receipt['credit_target_sha256']}",
                {"kind": "credit_assignment_ledger", "sha256": receipt["credit_target_sha256"]},
            )
            credit_fact = builder.bind(
                builder.origin_ref(assigned_origin), credit, credit_support,
                "assigned_consequence_credited_to_action", locator,
            )
            credit_origin = builder.origin_from(credit_support, credit_fact, "generated_credit_assignment")

            training = builder.occurrence(
                f"{condition}:{update}:adamw", occurrence_index, "training_update",
                {"operation": "AdamW_policy_update"},
                {"condition": condition, "update": update, "gradient_sha256": receipt["gradient_sha256"]},
            )
            occurrence_index += 1
            update_support = builder.support(
                f"{condition}:{update}:update:{receipt['actual_update_sha256']}",
                {"kind": "actual_parameter_update", "sha256": receipt["actual_update_sha256"], "norm": receipt["actual_update_norm"]},
            )
            update_fact = builder.bind(
                builder.origin_ref(credit_origin), training, update_support,
                "credited_signal_forms_actual_update", locator,
            )
            builder.bind(
                builder.source_ref(pre_state), training, update_support,
                "receiving_state_conditions_actual_update", locator,
            )
            update_origin = builder.origin_from(update_support, update_fact, "generated_actual_update")
            post_support = builder.support(
                f"{condition}:{update}:post-state:{receipt['post_state_sha256']}",
                {"kind": "parameter_optimizer_post_state", "sha256": receipt["post_state_sha256"]},
            )
            post_fact = builder.bind(
                builder.origin_ref(update_origin), training, post_support,
                "actual_update_forms_new_parameter_optimizer_state", locator,
            )

            builder.typed_edges.extend([
                {"edge_kind": "produced_consequence", "source": action_fact["generation_binding_id"], "target": consequence_fact["generation_binding_id"]},
                {"edge_kind": "bound_to_episode", "source": consequence_fact["generation_binding_id"], "target": assigned_fact["generation_binding_id"], "condition": condition},
                {"edge_kind": "credited_to_action", "source": assigned_fact["generation_binding_id"], "target": credit_fact["generation_binding_id"], "condition": condition},
                {"edge_kind": "forms_training_update", "source": credit_fact["generation_binding_id"], "target": update_fact["generation_binding_id"]},
                {"edge_kind": "parameter_version_flow", "source": update_fact["generation_binding_id"], "target": post_fact["generation_binding_id"]},
            ])
            ordered = [infer, consequence, binding_occurrence, credit, training]
            if previous_occurrence is not None:
                builder.typed_edges.append({"edge_kind": "program_order", "source": previous_occurrence, "target": infer["generation_occurrence_id"]})
            for left, right in zip(ordered, ordered[1:]):
                builder.typed_edges.append({
                    "edge_kind": "program_order",
                    "source": left["generation_occurrence_id"],
                    "target": right["generation_occurrence_id"],
                })
            previous_occurrence = training["generation_occurrence_id"]
    snapshot = builder.finish()
    graph = graph_from_snapshot(builder, snapshot)
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot_path = output_dir / "CORE_V3_SNAPSHOT.json"
    graph_path = output_dir / "GFG.json"
    write_json(snapshot_path, serialize_snapshot(snapshot))
    write_json(graph_path, graph)
    snapshot_id = validate_serialized_snapshot(json.loads(snapshot_path.read_text(encoding="utf-8")))
    graph_result = validate_graph(json.loads(graph_path.read_text(encoding="utf-8")))
    result = {
        "schema": "rl-seed-evidence-result-v1",
        "status": "PASS",
        "seed": seed,
        "snapshot_id": snapshot_id,
        "snapshot": str(snapshot_path),
        "snapshot_sha256": file_sha256(snapshot_path),
        "graph": str(graph_path),
        "graph_file_sha256": file_sha256(graph_path),
        "graph_validation": graph_result,
        "ledger_hashes": dependencies,
    }
    write_json(output_dir / "EVIDENCE_RESULT.json", result)
    return result


def build_all(aggregate_path: Path, output_root: Path) -> dict[str, Any]:
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    entries = [
        build_seed_evidence(Path(seed_path), output_root / f"seed-{json.loads(Path(seed_path).read_text(encoding='utf-8'))['seed']}")
        for seed_path in aggregate["seed_result_paths"]
    ]
    result = {
        "schema": "rl-formal-evidence-manifest-v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in entries) else "FAIL",
        "aggregate_result": str(aggregate_path),
        "aggregate_result_sha256": file_sha256(aggregate_path),
        "entries": entries,
    }
    write_json(output_root / "EVIDENCE_MANIFEST.json", result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_all(args.aggregate.resolve(), args.output_root.resolve()), indent=2, sort_keys=True))
