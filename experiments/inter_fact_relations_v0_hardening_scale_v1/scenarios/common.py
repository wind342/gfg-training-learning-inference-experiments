from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..common import canonical_sha256, content_id
from ..src.relation_model import make_evidence, make_relation
from ..src.run_identity import make_fact, make_occurrence


class RuntimeScenarioBuilder:
    def __init__(self, *, label: str, execution_run_id: str | None = None) -> None:
        self.label = label
        self.run_id = execution_run_id or content_id(
            "run1_", {"scenario_label": label}
        )
        self.occurrences: list[dict[str, Any]] = []
        self.facts: list[dict[str, Any]] = []
        self.occurrence_index: dict[str, dict[str, Any]] = {}
        self.fact_index: dict[str, dict[str, Any]] = {}
        self.primitive_relations: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []
        self.program_order_receipts: list[dict[str, Any]] = []
        self.generated_origin_receipts: list[dict[str, Any]] = []
        self.message_receipts: list[dict[str, Any]] = []
        self.synchronization_receipts: list[dict[str, Any]] = []
        self.resource_access_receipts: list[dict[str, Any]] = []
        self.reads_from_receipts: list[dict[str, Any]] = []
        self.conflict_receipts: list[dict[str, Any]] = []
        self.unknown_edges: list[dict[str, Any]] = []
        self.unclassified_messages: list[dict[str, Any]] = []
        self.unclassified_operations: list[dict[str, Any]] = []
        self.external_communications: list[dict[str, Any]] = []
        self.unclassified_synchronization_operations: list[dict[str, Any]] = []
        self.unclassified_resource_accesses: list[dict[str, Any]] = []
        self.scope_contract_overrides: dict[str, dict[str, Any]] = {}

    def add_occurrence(
        self,
        *,
        actor_id: str,
        sequence_index: int,
        operation: str,
        semantic_slot: int,
        scope_id: str,
        fact_count: int = 3,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        occurrence = make_occurrence(
            execution_run_id=self.run_id,
            actor_id=actor_id,
            sequence_index=sequence_index,
            operation=operation,
            semantic_slot=semantic_slot,
            scope_id=scope_id,
        )
        occurrence_id = occurrence["concrete_occurrence_instance_id"]
        if occurrence_id in self.occurrence_index:
            raise ValueError("duplicate occurrence")
        self.occurrences.append(occurrence)
        self.occurrence_index[occurrence_id] = occurrence
        created_facts = [
            make_fact(
                occurrence=occurrence,
                fact_slot=fact_slot,
                value={
                    "scenario": self.label,
                    "actor": actor_id,
                    "sequence": sequence_index,
                    "slot": fact_slot,
                },
            )
            for fact_slot in range(fact_count)
        ]
        for fact in created_facts:
            self.facts.append(fact)
            self.fact_index[fact["fact_id"]] = fact
        return occurrence, created_facts

    def _record_relation(
        self,
        *,
        endpoint_level: str,
        relation_type: str,
        source_id: str,
        target_id: str,
        evidence_kind: str,
        establishment_source: str,
        authority_id: str,
        receipt_ref: str,
        occurrence_ids: list[str],
        fact_ids: list[str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        evidence = make_evidence(
            evidence_kind=evidence_kind,
            establishment_source=establishment_source,
            authority_id=authority_id,
            execution_run_id=self.run_id,
            receipt_ref=receipt_ref,
            occurrence_ids=occurrence_ids,
            fact_ids=fact_ids,
            payload=payload,
        )
        relation = make_relation(
            endpoint_level=endpoint_level,
            relation_type=relation_type,
            source_id=source_id,
            target_id=target_id,
            establishment_source=establishment_source,
            authority_id=authority_id,
            execution_run_id=self.run_id,
            evidence_refs=[evidence["evidence_id"]],
        )
        self.evidence.append(evidence)
        self.primitive_relations.append(relation)
        return relation

    def add_program_order(
        self, source_occurrence_id: str, target_occurrence_id: str
    ) -> dict[str, Any]:
        source = self.occurrence_index[source_occurrence_id]
        target = self.occurrence_index[target_occurrence_id]
        receipt = {
            "receipt_id": content_id(
                "poreceipt1_",
                {
                    "run": self.run_id,
                    "source": source_occurrence_id,
                    "target": target_occurrence_id,
                },
            ),
            "execution_run_id": self.run_id,
            "actor_id": source["actor_id"],
            "source_occurrence_id": source_occurrence_id,
            "target_occurrence_id": target_occurrence_id,
            "source_sequence_index": source["sequence_index"],
            "target_sequence_index": target["sequence_index"],
            "recorded_by": "executor_wrapper",
            "authority_id": "controlled-executor-wrapper-v1",
            "establishment_source": "wrapper_established",
        }
        self.program_order_receipts.append(receipt)
        return self._record_relation(
            endpoint_level="occurrence",
            relation_type="program_order",
            source_id=source_occurrence_id,
            target_id=target_occurrence_id,
            evidence_kind="program_order_log",
            establishment_source="wrapper_established",
            authority_id="controlled-executor-wrapper-v1",
            receipt_ref=receipt["receipt_id"],
            occurrence_ids=[source_occurrence_id, target_occurrence_id],
            fact_ids=[],
            payload={
                "actor_id": receipt["actor_id"],
                "source_sequence_index": receipt["source_sequence_index"],
                "target_sequence_index": receipt["target_sequence_index"],
                "recorded_by": receipt["recorded_by"],
            },
        )

    def add_all_program_order(self) -> None:
        by_actor: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for occurrence in self.occurrences:
            by_actor[occurrence["actor_id"]].append(occurrence)
        for rows in by_actor.values():
            rows.sort(key=lambda row: row["sequence_index"])
            for source, target in zip(rows, rows[1:]):
                self.add_program_order(
                    source["concrete_occurrence_instance_id"],
                    target["concrete_occurrence_instance_id"],
                )

    def _replace_fact_with_generated_origin(
        self, source_fact: dict[str, Any], target_fact: dict[str, Any]
    ) -> dict[str, Any]:
        old_fact_id = target_fact["fact_id"]
        generated_origin = {
            "generated_origin_id": content_id(
                "genorigin1_",
                {
                    "producer_fact_id": source_fact["fact_id"],
                    "prior_support_id": source_fact["support_id"],
                    "consumer_occurrence_id": target_fact["occurrence_id"],
                    "consumer_fact_slot": target_fact["fact_slot"],
                },
            ),
            "producer_fact_id": source_fact["fact_id"],
            "prior_support_id": source_fact["support_id"],
        }
        target_fact["generated_origin"] = generated_origin
        target_fact["semantic_projection"]["coordinates"]["u"] = {
            "kind": "generated_origin",
            **generated_origin,
        }
        target_fact["semantic_fact_id"] = content_id(
            "semfact1_", target_fact["semantic_projection"]
        )
        target_fact["core_fact_id"] = content_id(
            "corefact1_",
            {
                "core_content_occurrence_id": target_fact["semantic_projection"][
                    "coordinates"
                ]["omega_bar"]["core_content_occurrence_id"],
                "fact_slot": target_fact["fact_slot"],
                "support_id": target_fact["support_id"],
                "generated_origin_id": generated_origin["generated_origin_id"],
            },
        )
        target_fact["fact_id"] = content_id(
            "fact1_",
            {
                "occurrence_id": target_fact["occurrence_id"],
                "fact_slot": target_fact["fact_slot"],
                "support_id": target_fact["support_id"],
                "generated_origin_id": generated_origin["generated_origin_id"],
            },
        )
        del self.fact_index[old_fact_id]
        self.fact_index[target_fact["fact_id"]] = target_fact
        return target_fact

    def add_generated_origin(
        self, source_fact: dict[str, Any], target_fact: dict[str, Any]
    ) -> dict[str, Any]:
        target_fact = self._replace_fact_with_generated_origin(
            source_fact, target_fact
        )
        receipt = {
            "receipt_id": content_id(
                "goreceipt1_",
                {
                    "run": self.run_id,
                    "producer": source_fact["fact_id"],
                    "consumer": target_fact["fact_id"],
                },
            ),
            "execution_run_id": self.run_id,
            "producer_fact_id": source_fact["fact_id"],
            "consumer_fact_id": target_fact["fact_id"],
            "prior_support_id": source_fact["support_id"],
            "generated_origin_id": target_fact["generated_origin"][
                "generated_origin_id"
            ],
        }
        self.generated_origin_receipts.append(receipt)
        return self._record_relation(
            endpoint_level="fact",
            relation_type="generated_origin_dependency",
            source_id=source_fact["fact_id"],
            target_id=target_fact["fact_id"],
            evidence_kind="generated_origin_record",
            establishment_source="generator_established",
            authority_id="controlled-generation-wrapper-v1",
            receipt_ref=receipt["receipt_id"],
            occurrence_ids=[
                source_fact["occurrence_id"],
                target_fact["occurrence_id"],
            ],
            fact_ids=[source_fact["fact_id"], target_fact["fact_id"]],
            payload={
                "producer_fact_id": source_fact["fact_id"],
                "consumer_fact_id": target_fact["fact_id"],
                "prior_support_id": source_fact["support_id"],
                "generated_origin_id": receipt["generated_origin_id"],
            },
        )

    def add_message(
        self,
        source_occurrence_id: str,
        target_occurrence_id: str,
        *,
        channel_id: str,
        payload: Any,
        broadcast_allowed: bool = False,
    ) -> dict[str, Any]:
        payload_digest = canonical_sha256(payload)
        message_id = content_id(
            "message1_",
            {
                "run": self.run_id,
                "source": source_occurrence_id,
                "target": target_occurrence_id,
                "channel_id": channel_id,
                "payload_digest": payload_digest,
            },
        )
        receipt = {
            "receipt_id": content_id(
                "msgreceipt1_", {"message_id": message_id}
            ),
            "execution_run_id": self.run_id,
            "message_id": message_id,
            "send_occurrence_id": source_occurrence_id,
            "receive_occurrence_id": target_occurrence_id,
            "channel_id": channel_id,
            "payload_digest": payload_digest,
            "broadcast_allowed": broadcast_allowed,
        }
        self.message_receipts.append(receipt)
        return self._record_relation(
            endpoint_level="occurrence",
            relation_type="message_send_receive",
            source_id=source_occurrence_id,
            target_id=target_occurrence_id,
            evidence_kind="message_record",
            establishment_source="wrapper_established",
            authority_id="controlled-message-wrapper-v1",
            receipt_ref=receipt["receipt_id"],
            occurrence_ids=[source_occurrence_id, target_occurrence_id],
            fact_ids=[],
            payload={
                "message_id": message_id,
                "send_occurrence_id": source_occurrence_id,
                "receive_occurrence_id": target_occurrence_id,
                "channel_id": channel_id,
                "payload_digest": payload_digest,
            },
        )

    def add_synchronization(
        self,
        pre_occurrence_ids: list[str],
        release_occurrence_id: str,
        *,
        synchronization_kind: str = "barrier",
        generation: int = 0,
    ) -> list[dict[str, Any]]:
        participants = sorted(
            {
                self.occurrence_index[value]["actor_id"]
                for value in [*pre_occurrence_ids, release_occurrence_id]
            }
        )
        synchronization_id = content_id(
            "sync1_",
            {
                "run": self.run_id,
                "kind": synchronization_kind,
                "generation": generation,
                "pre": sorted(pre_occurrence_ids),
                "release": release_occurrence_id,
            },
        )
        receipt = {
            "receipt_id": content_id(
                "syncreceipt1_", {"synchronization_id": synchronization_id}
            ),
            "execution_run_id": self.run_id,
            "synchronization_id": synchronization_id,
            "synchronization_kind": synchronization_kind,
            "participant_actor_ids": participants,
            "generation": generation,
            "release_occurrence_id": release_occurrence_id,
            "pre_occurrence_ids": sorted(pre_occurrence_ids),
            "pre_phase": "before_release",
            "post_phase": "release",
        }
        self.synchronization_receipts.append(receipt)
        relations = []
        for source_occurrence_id in sorted(pre_occurrence_ids):
            relations.append(
                self._record_relation(
                    endpoint_level="occurrence",
                    relation_type="synchronizes_with",
                    source_id=source_occurrence_id,
                    target_id=release_occurrence_id,
                    evidence_kind="synchronization_record",
                    establishment_source="wrapper_established",
                    authority_id="controlled-synchronization-wrapper-v1",
                    receipt_ref=receipt["receipt_id"],
                    occurrence_ids=[
                        source_occurrence_id,
                        release_occurrence_id,
                    ],
                    fact_ids=[],
                    payload={
                        "synchronization_id": synchronization_id,
                        "synchronization_kind": synchronization_kind,
                        "participant_actor_ids": participants,
                        "generation": generation,
                        "release_occurrence_id": release_occurrence_id,
                        "pre_phase": "before_release",
                        "post_phase": "release",
                    },
                )
            )
        return relations

    def add_resource_access(
        self,
        fact: dict[str, Any],
        *,
        resource_id: str,
        version_id: str,
        access_mode: str,
        observed_version_id: str | None = None,
    ) -> dict[str, Any]:
        access = {
            "access_id": content_id(
                "access1_",
                {
                    "run": self.run_id,
                    "fact_id": fact["fact_id"],
                    "resource_id": resource_id,
                    "version_id": version_id,
                    "access_mode": access_mode,
                    "observed_version_id": observed_version_id,
                },
            ),
            "execution_run_id": self.run_id,
            "fact_id": fact["fact_id"],
            "occurrence_id": fact["occurrence_id"],
            "resource_id": resource_id,
            "version_id": version_id,
            "access_mode": access_mode,
            "observed_version_id": observed_version_id,
        }
        self.resource_access_receipts.append(access)
        return access

    def add_reads_from(
        self,
        source_fact: dict[str, Any],
        target_fact: dict[str, Any],
        *,
        resource_id: str,
        version_id: str,
    ) -> dict[str, Any]:
        write = self.add_resource_access(
            source_fact,
            resource_id=resource_id,
            version_id=version_id,
            access_mode="write",
        )
        read = self.add_resource_access(
            target_fact,
            resource_id=resource_id,
            version_id=version_id,
            access_mode="read",
            observed_version_id=version_id,
        )
        receipt = {
            "receipt_id": content_id(
                "rfreceipt1_",
                {
                    "run": self.run_id,
                    "write": write["access_id"],
                    "read": read["access_id"],
                },
            ),
            "execution_run_id": self.run_id,
            "resource_id": resource_id,
            "version_id": version_id,
            "producer_write_access_id": write["access_id"],
            "consumer_read_access_id": read["access_id"],
            "source_fact_id": source_fact["fact_id"],
            "target_fact_id": target_fact["fact_id"],
        }
        self.reads_from_receipts.append(receipt)
        return self._record_relation(
            endpoint_level="fact",
            relation_type="reads_from",
            source_id=source_fact["fact_id"],
            target_id=target_fact["fact_id"],
            evidence_kind="read_record",
            establishment_source="generator_established",
            authority_id="controlled-resource-version-wrapper-v1",
            receipt_ref=receipt["receipt_id"],
            occurrence_ids=[
                source_fact["occurrence_id"],
                target_fact["occurrence_id"],
            ],
            fact_ids=[source_fact["fact_id"], target_fact["fact_id"]],
            payload={
                "resource_id": resource_id,
                "version_id": version_id,
                "producer_write_access_id": write["access_id"],
                "consumer_read_access_id": read["access_id"],
                "source_fact_id": source_fact["fact_id"],
                "target_fact_id": target_fact["fact_id"],
            },
        )

    def add_conflict(
        self,
        left_fact: dict[str, Any],
        right_fact: dict[str, Any],
        *,
        resource_id: str,
        version_id: str,
        left_mode: str = "write",
        right_mode: str = "write",
    ) -> dict[str, Any]:
        left = self.add_resource_access(
            left_fact,
            resource_id=resource_id,
            version_id=version_id,
            access_mode=left_mode,
            observed_version_id=version_id if left_mode == "read" else None,
        )
        right = self.add_resource_access(
            right_fact,
            resource_id=resource_id,
            version_id=version_id,
            access_mode=right_mode,
            observed_version_id=version_id if right_mode == "read" else None,
        )
        receipt = {
            "receipt_id": content_id(
                "conflictreceipt1_",
                {
                    "run": self.run_id,
                    "left_access_id": left["access_id"],
                    "right_access_id": right["access_id"],
                },
            ),
            "execution_run_id": self.run_id,
            "left_access_id": left["access_id"],
            "right_access_id": right["access_id"],
            "left_fact_id": left_fact["fact_id"],
            "right_fact_id": right_fact["fact_id"],
        }
        self.conflict_receipts.append(receipt)
        return self._record_relation(
            endpoint_level="fact",
            relation_type="conflicts_with",
            source_id=left_fact["fact_id"],
            target_id=right_fact["fact_id"],
            evidence_kind="resource_access_record",
            establishment_source="wrapper_established",
            authority_id="controlled-resource-access-wrapper-v1",
            receipt_ref=receipt["receipt_id"],
            occurrence_ids=[
                left_fact["occurrence_id"],
                right_fact["occurrence_id"],
            ],
            fact_ids=[left_fact["fact_id"], right_fact["fact_id"]],
            payload={
                "resource_id": resource_id,
                "version_id": version_id,
                "left_access_id": left["access_id"],
                "right_access_id": right["access_id"],
                "left_occurrence_id": left_fact["occurrence_id"],
                "right_occurrence_id": right_fact["occurrence_id"],
            },
        )

    def add_unknown_edge(self, scope_id: str, label: str) -> None:
        self.unknown_edges.append(
            {
                "unknown_edge_id": content_id(
                    "unknownedge1_",
                    {"run": self.run_id, "scope_id": scope_id, "label": label},
                ),
                "scope_id": scope_id,
                "label": label,
            }
        )

    def runtime_receipts(self) -> dict[str, Any]:
        return {
            "execution_run_id": self.run_id,
            "occurrences": self.occurrences,
            "facts": self.facts,
            "program_order_receipts": self.program_order_receipts,
            "generated_origin_receipts": self.generated_origin_receipts,
            "message_receipts": self.message_receipts,
            "synchronization_receipts": self.synchronization_receipts,
            "resource_access_receipts": self.resource_access_receipts,
            "reads_from_receipts": self.reads_from_receipts,
            "conflict_receipts": self.conflict_receipts,
            "executor_coverage_receipts": [
                {
                    "occurrence_id": row["concrete_occurrence_instance_id"],
                    "recorded_by": "controlled-executor-wrapper-v1",
                }
                for row in self.occurrences
            ],
            "unknown_edges": self.unknown_edges,
            "unclassified_messages": self.unclassified_messages,
            "unclassified_operations": self.unclassified_operations,
            "external_communications": self.external_communications,
            "unclassified_synchronization_operations": (
                self.unclassified_synchronization_operations
            ),
            "unclassified_resource_accesses": self.unclassified_resource_accesses,
            "schema_version": "runtime-receipts-v1",
        }

    def primitive_store(self) -> dict[str, Any]:
        return {
            "execution_run_id": self.run_id,
            "primitive_relations": self.primitive_relations,
            "evidence": self.evidence,
            "schema_version": "unvalidated-primitive-store-v1",
        }

    def capture_contract(self) -> dict[str, Any]:
        by_scope: defaultdict[str, list[str]] = defaultdict(list)
        for occurrence in self.occurrences:
            by_scope[occurrence["scope_id"]].append(
                occurrence["concrete_occurrence_instance_id"]
            )
        scopes = []
        for scope_id, occurrence_ids in sorted(by_scope.items()):
            row = {
                "scope_id": scope_id,
                "covered_occurrence_ids": sorted(occurrence_ids),
                "planned_capture": {
                    "program_order": True,
                    "messages": True,
                    "synchronization": True,
                    "generated_origin": True,
                    "reads_from": True,
                    "resource_access": True,
                },
                "external_communication_absent": True,
                "unobserved_scheduler_relation_ruled_out": True,
            }
            row.update(self.scope_contract_overrides.get(scope_id, {}))
            scopes.append(row)
        material = {
            "execution_run_id": self.run_id,
            "scopes": scopes,
            "schema_version": "declared-capture-contract-v1",
        }
        return {"contract_id": content_id("capcontract1_", material), **material}
