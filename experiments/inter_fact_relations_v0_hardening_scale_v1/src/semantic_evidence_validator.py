from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable

from ..common import ExperimentError
from .relation_model import (
    FACT_PRIMITIVE_TYPES,
    OCCURRENCE_PRIMITIVE_TYPES,
    PRIMITIVE_RELATION_TYPES,
)


EXPECTED_EVIDENCE_KINDS = {
    "program_order": "program_order_log",
    "generated_origin_dependency": "generated_origin_record",
    "message_send_receive": "message_record",
    "synchronizes_with": "synchronization_record",
    "reads_from": "read_record",
    "conflicts_with": "resource_access_record",
}

RECEIPT_COLLECTIONS = {
    "program_order": "program_order_receipts",
    "generated_origin_dependency": "generated_origin_receipts",
    "message_send_receive": "message_receipts",
    "synchronizes_with": "synchronization_receipts",
    "reads_from": "reads_from_receipts",
    "conflicts_with": "conflict_receipts",
}


def _unique_index(
    rows: list[dict[str, Any]], key: str, duplicate_reason: str
) -> dict[str, dict[str, Any]]:
    values = [row[key] for row in rows]
    if len(values) != len(set(values)):
        raise ExperimentError(duplicate_reason)
    return {row[key]: row for row in rows}


class SemanticEvidenceValidator:
    def __init__(self, runtime_receipts: dict[str, Any]) -> None:
        self.receipts = runtime_receipts
        self.run_id = runtime_receipts["execution_run_id"]
        self.occurrences = _unique_index(
            runtime_receipts["occurrences"],
            "concrete_occurrence_instance_id",
            "DUPLICATE_RUNTIME_OCCURRENCE",
        )
        self.facts = _unique_index(
            runtime_receipts["facts"], "fact_id", "DUPLICATE_RUNTIME_FACT"
        )
        self.accesses = _unique_index(
            runtime_receipts.get("resource_access_receipts", []),
            "access_id",
            "DUPLICATE_RESOURCE_ACCESS_RECEIPT",
        )
        self.receipt_indexes = {
            relation_type: _unique_index(
                runtime_receipts.get(collection, []),
                "receipt_id",
                f"DUPLICATE_{relation_type.upper()}_RECEIPT",
            )
            for relation_type, collection in RECEIPT_COLLECTIONS.items()
        }
        self._validators: dict[
            str,
            Callable[
                [dict[str, Any], dict[str, Any], dict[str, Any]],
                None,
            ],
        ] = {
            "program_order": self._validate_program_order,
            "generated_origin_dependency": self._validate_generated_origin,
            "message_send_receive": self._validate_message,
            "synchronizes_with": self._validate_synchronization,
            "reads_from": self._validate_reads_from,
            "conflicts_with": self._validate_conflict,
        }

    def validate(self, primitive_store: dict[str, Any]) -> dict[str, Any]:
        if primitive_store.get("execution_run_id") != self.run_id:
            raise ExperimentError("PRIMITIVE_STORE_RUN_ID_MISMATCH")
        relations = primitive_store["primitive_relations"]
        evidence_rows = primitive_store["evidence"]
        relation_ids = [row["relation_id"] for row in relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ExperimentError("DUPLICATE_PRIMITIVE_RELATION")
        evidence_ids = [row["evidence_id"] for row in evidence_rows]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ExperimentError("DUPLICATE_EVIDENCE_ID")
        evidence_index = {row["evidence_id"]: row for row in evidence_rows}
        pairing: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
        type_counts: Counter[str] = Counter()

        for relation in relations:
            relation_type = relation["relation_type"]
            if relation_type not in PRIMITIVE_RELATION_TYPES:
                raise ExperimentError("PRIMITIVE_RELATION_TYPE_INVALID")
            expected_level = (
                "occurrence"
                if relation_type in OCCURRENCE_PRIMITIVE_TYPES
                else "fact"
            )
            if relation["endpoint_level"] != expected_level:
                raise ExperimentError("PRIMITIVE_ENDPOINT_LEVEL_INVALID")
            if relation["execution_run_id"] != self.run_id:
                raise ExperimentError("RELATION_RUN_ID_MISMATCH")
            if relation["establishment_source"] not in {
                "generator_established",
                "wrapper_established",
            }:
                raise ExperimentError("PRIMITIVE_ESTABLISHMENT_SOURCE_INVALID")
            refs = relation["evidence_refs"]
            if len(refs) != 1:
                raise ExperimentError("PRIMITIVE_EXACTLY_ONE_EVIDENCE_REQUIRED")
            if refs[0] not in evidence_index:
                raise ExperimentError("RELATION_EVIDENCE_MISSING")
            evidence = evidence_index[refs[0]]
            if evidence["execution_run_id"] != self.run_id:
                raise ExperimentError("EVIDENCE_RUN_ID_MISMATCH")
            if (
                evidence["evidence_kind"]
                != EXPECTED_EVIDENCE_KINDS[relation_type]
            ):
                raise ExperimentError("EVIDENCE_KIND_MISMATCH")
            receipt = self.receipt_indexes[relation_type].get(
                evidence["receipt_ref"]
            )
            if receipt is None:
                raise ExperimentError("AUTHORITATIVE_RECEIPT_MISSING")
            self._validate_common_binding(relation, evidence, receipt)
            self._validators[relation_type](relation, evidence, receipt)
            if relation_type == "message_send_receive":
                pairing[receipt["message_id"]].append(
                    (
                        receipt["send_occurrence_id"],
                        receipt["receive_occurrence_id"],
                    )
                )
            type_counts[relation_type] += 1

        for message_id, pairs in pairing.items():
            message_receipt = next(
                row
                for row in self.receipts["message_receipts"]
                if row["message_id"] == message_id
            )
            if not message_receipt.get("broadcast_allowed", False) and len(pairs) != 1:
                raise ExperimentError("DUPLICATE_MESSAGE_PAIRING")
            if len(pairs) != len(set(pairs)):
                raise ExperimentError("DUPLICATE_MESSAGE_PAIRING")

        return {
            "status": "PASS",
            "execution_run_id": self.run_id,
            "primitive_relation_count": len(relations),
            "evidence_count": len(evidence_rows),
            "relation_type_counts": {
                key: type_counts.get(key, 0)
                for key in sorted(PRIMITIVE_RELATION_TYPES)
            },
            "primitive_relations": sorted(
                relations, key=lambda row: row["relation_id"]
            ),
            "evidence": sorted(evidence_rows, key=lambda row: row["evidence_id"]),
            "occurrence_catalog": sorted(
                (
                    {
                        "occurrence_id": row["concrete_occurrence_instance_id"],
                        "scope_id": row["scope_id"],
                    }
                    for row in self.occurrences.values()
                ),
                key=lambda row: row["occurrence_id"],
            ),
            "fact_catalog": sorted(
                (
                    {
                        "fact_id": row["fact_id"],
                        "occurrence_id": row["occurrence_id"],
                    }
                    for row in self.facts.values()
                ),
                key=lambda row: row["fact_id"],
            ),
        }

    def _validate_common_binding(
        self,
        relation: dict[str, Any],
        evidence: dict[str, Any],
        receipt: dict[str, Any],
    ) -> None:
        if evidence["authority_id"] != relation["authority_id"]:
            raise ExperimentError("EVIDENCE_AUTHORITY_MISMATCH")
        if evidence["establishment_source"] != relation["establishment_source"]:
            raise ExperimentError("EVIDENCE_ESTABLISHMENT_SOURCE_MISMATCH")
        if receipt["execution_run_id"] != self.run_id:
            raise ExperimentError("RECEIPT_RUN_ID_MISMATCH")
        if relation["endpoint_level"] == "occurrence":
            expected_occurrences = sorted(
                [relation["source_id"], relation["target_id"]]
            )
            if sorted(evidence["occurrence_ids"]) != expected_occurrences:
                raise ExperimentError("EVIDENCE_ENDPOINT_BINDING_MISMATCH")
            if evidence["fact_ids"]:
                raise ExperimentError("EVIDENCE_ENDPOINT_BINDING_MISMATCH")
            if any(value not in self.occurrences for value in expected_occurrences):
                raise ExperimentError("RELATION_OCCURRENCE_ENDPOINT_UNKNOWN")
        else:
            expected_facts = sorted([relation["source_id"], relation["target_id"]])
            if sorted(evidence["fact_ids"]) != expected_facts:
                raise ExperimentError("EVIDENCE_ENDPOINT_BINDING_MISMATCH")
            if any(value not in self.facts for value in expected_facts):
                raise ExperimentError("RELATION_FACT_ENDPOINT_UNKNOWN")
            expected_occurrences = sorted(
                [
                    self.facts[relation["source_id"]]["occurrence_id"],
                    self.facts[relation["target_id"]]["occurrence_id"],
                ]
            )
            if sorted(evidence["occurrence_ids"]) != expected_occurrences:
                raise ExperimentError("EVIDENCE_ENDPOINT_BINDING_MISMATCH")

    def _validate_program_order(
        self,
        relation: dict[str, Any],
        evidence: dict[str, Any],
        receipt: dict[str, Any],
    ) -> None:
        source = self.occurrences[relation["source_id"]]
        target = self.occurrences[relation["target_id"]]
        if source["actor_id"] != target["actor_id"]:
            raise ExperimentError("PROGRAM_ORDER_ACTOR_MISMATCH")
        if receipt["actor_id"] != source["actor_id"]:
            raise ExperimentError("PROGRAM_ORDER_ACTOR_MISMATCH")
        if (
            receipt["source_occurrence_id"] != relation["source_id"]
            or receipt["target_occurrence_id"] != relation["target_id"]
        ):
            raise ExperimentError("PROGRAM_ORDER_ENDPOINT_MISMATCH")
        if not (
            source["sequence_index"]
            == receipt["source_sequence_index"]
            < receipt["target_sequence_index"]
            == target["sequence_index"]
        ):
            raise ExperimentError("PROGRAM_ORDER_SEQUENCE_INVALID")
        if receipt.get("recorded_by") not in {
            "executor_wrapper",
            "scheduler_wrapper",
        }:
            raise ExperimentError("PROGRAM_ORDER_RECEIPT_REQUIRED")
        if (
            receipt.get("authority_id") != relation["authority_id"]
            or receipt.get("authority_id") != evidence["authority_id"]
        ):
            raise ExperimentError("PROGRAM_ORDER_AUTHORITY_MISMATCH")
        if (
            receipt.get("establishment_source")
            != relation["establishment_source"]
            or receipt.get("establishment_source")
            != evidence["establishment_source"]
        ):
            raise ExperimentError(
                "PROGRAM_ORDER_ESTABLISHMENT_SOURCE_MISMATCH"
            )
        if evidence["payload"] != {
            "actor_id": receipt["actor_id"],
            "source_sequence_index": receipt["source_sequence_index"],
            "target_sequence_index": receipt["target_sequence_index"],
            "recorded_by": receipt["recorded_by"],
        }:
            raise ExperimentError("PROGRAM_ORDER_PAYLOAD_MISMATCH")

    def _validate_generated_origin(
        self,
        relation: dict[str, Any],
        evidence: dict[str, Any],
        receipt: dict[str, Any],
    ) -> None:
        producer = self.facts[relation["source_id"]]
        consumer = self.facts[relation["target_id"]]
        if (
            receipt["producer_fact_id"] != producer["fact_id"]
            or receipt["consumer_fact_id"] != consumer["fact_id"]
        ):
            raise ExperimentError("GENERATED_ORIGIN_ENDPOINT_MISMATCH")
        if producer["support_id"] != receipt["prior_support_id"]:
            raise ExperimentError("GENERATED_ORIGIN_PRIOR_SUPPORT_MISMATCH")
        generated_origin = consumer.get("generated_origin")
        if not generated_origin:
            raise ExperimentError("GENERATED_ORIGIN_ENTITY_MISSING")
        if (
            generated_origin["prior_support_id"] != producer["support_id"]
            or generated_origin["producer_fact_id"] != producer["fact_id"]
            or generated_origin["generated_origin_id"]
            != receipt["generated_origin_id"]
        ):
            raise ExperimentError("GENERATED_ORIGIN_PRIOR_SUPPORT_MISMATCH")
        if evidence["payload"] != {
            "producer_fact_id": producer["fact_id"],
            "consumer_fact_id": consumer["fact_id"],
            "prior_support_id": producer["support_id"],
            "generated_origin_id": receipt["generated_origin_id"],
        }:
            raise ExperimentError("GENERATED_ORIGIN_PAYLOAD_MISMATCH")

    def _validate_message(
        self,
        relation: dict[str, Any],
        evidence: dict[str, Any],
        receipt: dict[str, Any],
    ) -> None:
        if receipt["send_occurrence_id"] != relation["source_id"]:
            raise ExperimentError("MESSAGE_SEND_ENDPOINT_MISMATCH")
        if receipt["receive_occurrence_id"] != relation["target_id"]:
            raise ExperimentError("MESSAGE_RECEIVE_ENDPOINT_MISMATCH")
        required = {
            "message_id": receipt["message_id"],
            "send_occurrence_id": receipt["send_occurrence_id"],
            "receive_occurrence_id": receipt["receive_occurrence_id"],
            "channel_id": receipt["channel_id"],
            "payload_digest": receipt["payload_digest"],
        }
        if evidence["payload"] != required:
            raise ExperimentError("MESSAGE_PAYLOAD_MISMATCH")
        matching_ids = [
            row
            for row in self.receipts["message_receipts"]
            if row["message_id"] == receipt["message_id"]
        ]
        if len(matching_ids) != 1:
            raise ExperimentError("DUPLICATE_MESSAGE_PAIRING")

    def _validate_synchronization(
        self,
        relation: dict[str, Any],
        evidence: dict[str, Any],
        receipt: dict[str, Any],
    ) -> None:
        if relation["source_id"] not in receipt["pre_occurrence_ids"]:
            raise ExperimentError("SYNCHRONIZATION_PRE_PHASE_MISMATCH")
        if relation["target_id"] != receipt["release_occurrence_id"]:
            raise ExperimentError("SYNCHRONIZATION_RELEASE_MISMATCH")
        source_actor = self.occurrences[relation["source_id"]]["actor_id"]
        target_actor = self.occurrences[relation["target_id"]]["actor_id"]
        if source_actor not in receipt["participant_actor_ids"]:
            raise ExperimentError("BARRIER_PARTICIPANT_MISSING")
        if target_actor not in receipt["participant_actor_ids"]:
            raise ExperimentError("BARRIER_PARTICIPANT_MISSING")
        required = {
            "synchronization_id": receipt["synchronization_id"],
            "synchronization_kind": receipt["synchronization_kind"],
            "participant_actor_ids": sorted(receipt["participant_actor_ids"]),
            "generation": receipt["generation"],
            "release_occurrence_id": receipt["release_occurrence_id"],
            "pre_phase": receipt["pre_phase"],
            "post_phase": receipt["post_phase"],
        }
        if evidence["payload"] != required:
            raise ExperimentError("SYNCHRONIZATION_PAYLOAD_MISMATCH")

    def _validate_reads_from(
        self,
        relation: dict[str, Any],
        evidence: dict[str, Any],
        receipt: dict[str, Any],
    ) -> None:
        if (
            receipt["source_fact_id"] != relation["source_id"]
            or receipt["target_fact_id"] != relation["target_id"]
        ):
            raise ExperimentError("READS_FROM_ENDPOINT_MISMATCH")
        write = self.accesses.get(receipt["producer_write_access_id"])
        read = self.accesses.get(receipt["consumer_read_access_id"])
        if write is None or read is None:
            raise ExperimentError("READS_FROM_ACCESS_RECEIPT_MISSING")
        if write["access_mode"] != "write" or read["access_mode"] != "read":
            raise ExperimentError("READS_FROM_ACCESS_MODE_INVALID")
        if write["fact_id"] != relation["source_id"]:
            raise ExperimentError("READS_FROM_ENDPOINT_MISMATCH")
        if read["fact_id"] != relation["target_id"]:
            raise ExperimentError("READS_FROM_ENDPOINT_MISMATCH")
        if (
            write["resource_id"] != read["resource_id"]
            or write["resource_id"] != receipt["resource_id"]
        ):
            raise ExperimentError("READS_FROM_RESOURCE_MISMATCH")
        if (
            write["version_id"] != read["observed_version_id"]
            or write["version_id"] != receipt["version_id"]
        ):
            raise ExperimentError("READS_FROM_VERSION_MISMATCH")
        required = {
            "resource_id": receipt["resource_id"],
            "version_id": receipt["version_id"],
            "producer_write_access_id": write["access_id"],
            "consumer_read_access_id": read["access_id"],
            "source_fact_id": relation["source_id"],
            "target_fact_id": relation["target_id"],
        }
        if evidence["payload"] != required:
            raise ExperimentError("READS_FROM_PAYLOAD_MISMATCH")

    def _validate_conflict(
        self,
        relation: dict[str, Any],
        evidence: dict[str, Any],
        receipt: dict[str, Any],
    ) -> None:
        left = self.accesses.get(receipt["left_access_id"])
        right = self.accesses.get(receipt["right_access_id"])
        if left is None or right is None:
            raise ExperimentError("CONFLICT_ACCESS_RECEIPT_MISSING")
        if sorted([left["fact_id"], right["fact_id"]]) != sorted(
            [relation["source_id"], relation["target_id"]]
        ):
            raise ExperimentError("CONFLICT_ENDPOINT_MISMATCH")
        if left["resource_id"] != right["resource_id"]:
            raise ExperimentError("CONFLICT_RESOURCE_MISMATCH")
        if left["version_id"] != right["version_id"]:
            raise ExperimentError("CONFLICT_RESOURCE_VERSION_MISMATCH")
        if left["access_mode"] == right["access_mode"] == "read":
            raise ExperimentError("READ_READ_CONFLICT_INVALID")
        required = {
            "resource_id": left["resource_id"],
            "version_id": left["version_id"],
            "left_access_id": left["access_id"],
            "right_access_id": right["access_id"],
            "left_occurrence_id": left["occurrence_id"],
            "right_occurrence_id": right["occurrence_id"],
        }
        if evidence["payload"] != required:
            raise ExperimentError("CONFLICT_PAYLOAD_MISMATCH")


def validate_primitive_store(
    primitive_store: dict[str, Any], runtime_receipts: dict[str, Any]
) -> dict[str, Any]:
    return SemanticEvidenceValidator(runtime_receipts).validate(primitive_store)
