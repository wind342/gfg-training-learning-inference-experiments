"""Registered Snapshot-only query and signed-effect interpretation."""

from __future__ import annotations

from typing import Any

from generation_relation_core.entities import query_request
from generation_relation_core.query_engine import QueryEngine
from generation_relation_core.snapshots import (
    SnapshotValidation,
    ValidatedSnapshot,
)

from .algebra import SignedPair, signed_sum


def _outcome_payload(
    snapshot: ValidatedSnapshot, reference: dict[str, str]
) -> tuple[str, str, dict[str, Any]]:
    if reference["kind"] == "support":
        row = next(
            item
            for item in snapshot.tables.perceptual_support_records
            if item["support_id"] == reference["support_id"]
        )
        return "support", row["support_id"], row["support_payload"]
    if reference["kind"] == "disposition":
        row = next(
            item
            for item in snapshot.tables.explicit_dispositions
            if item["disposition_id"] == reference["disposition_id"]
        )
        return (
            "disposition",
            row["disposition_id"],
            row["disposition_payload"],
        )
    raise ValueError(f"unknown outcome reference: {reference!r}")


def _complete_fact(
    snapshot: ValidatedSnapshot, relation: dict[str, Any]
) -> dict[str, Any]:
    origin = relation["origin"]
    if origin["kind"] != "registered_source":
        raise ValueError("this frozen experiment expects registered sources")
    source = next(
        row
        for row in snapshot.tables.source_information_records
        if row["source_information_id"]
        == origin["source_information_id"]
    )
    occurrence = next(
        row
        for row in snapshot.tables.generation_occurrences
        if row["generation_occurrence_id"]
        == relation["generation_occurrence_id"]
    )
    outcome_kind, outcome_id, payload = _outcome_payload(
        snapshot, relation["outcome"]
    )
    multiplicity = payload.get("effect_multiplicity")
    effect_identity = payload.get("effect_identity")
    if (
        not isinstance(multiplicity, int)
        or isinstance(multiplicity, bool)
        or multiplicity <= 0
    ):
        raise ValueError("fact has no positive effect multiplicity")
    if (
        not isinstance(effect_identity, str)
        or not effect_identity.startswith("x_")
    ):
        raise ValueError("fact has no canonical effect identity")
    return {
        "binding_identity": relation["generation_binding_id"],
        "effect_identity": effect_identity,
        "multiplicity": multiplicity,
        "rho": relation["relation_role"],
        "scope": source["domain_scope_id"],
        "tau": occurrence["transform_reference"],
        "u": {
            "source_identity": source["source_identity"],
            "source_information_id": source[
                "source_information_id"
            ],
            "source_payload": source["source_payload"],
        },
        "z": {
            "outcome_id": outcome_id,
            "outcome_kind": outcome_kind,
            "outcome_payload": payload,
        },
        "omega": {
            "generation_occurrence_id": occurrence[
                "generation_occurrence_id"
            ],
            "occurrence_payload": occurrence["occurrence_payload"],
            "occurrence_type": occurrence["occurrence_type"],
            "stable_instance_key": occurrence["stable_instance_key"],
        },
    }


def _matching_rules(
    fact: dict[str, Any], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    transform = fact["tau"]
    return [
        rule
        for rule in contract["rules"]
        if rule["transform_operation"] == transform.get("operation")
        and rule["occurrence_type"]
        == fact["omega"]["occurrence_type"]
        and rule["relation_role"] == fact["rho"]
        and rule["outcome_kind"] == fact["z"]["outcome_kind"]
    ]


class RegisteredSignedEffectQuery:
    """Interpret only relations returned through the registered Core query path."""

    def __init__(
        self,
        snapshot: ValidatedSnapshot,
        validation: SnapshotValidation,
        predicate_registry: Any,
        signed_effect_contract: dict[str, Any],
        query_contract: dict[str, Any],
    ) -> None:
        if validation.snapshot_id != snapshot.snapshot_id:
            raise ValueError("validation token does not belong to snapshot")
        self.snapshot = snapshot
        self.validation = validation
        self.engine = QueryEngine(snapshot, predicate_registry)
        self.signed_effect_contract = signed_effect_contract
        self.query_contract = query_contract

    def complete_facts(self) -> list[dict[str, Any]]:
        request = query_request(
            domain_scope_id=self.query_contract["domain_scope_id"],
            support_space_id=self.snapshot.tables.support_space_records[
                0
            ]["support_space_id"],
            predicate_profile_id=self.snapshot.tables.predicate_profiles[
                0
            ]["predicate_profile_id"],
            predicate=self.query_contract["support_predicate"],
            query_payload={},
            requested_granularity="generation_relation",
        )
        support_result = self.engine.execute(request).result
        relations = [
            relation
            for hit in support_result["hits"]
            for relation in hit["generation_relations"]
        ]
        relations.extend(
            self.engine.disposition_relations(
                self.query_contract["domain_scope_id"]
            )
        )
        by_binding = {
            relation["generation_binding_id"]: relation
            for relation in relations
        }
        expected = {
            row["generation_binding_id"]
            for row in self.snapshot.tables.generation_bindings
        }
        if set(by_binding) != expected:
            raise ValueError(
                "registered queries did not recover every binding exactly"
            )
        return [
            _complete_fact(self.snapshot, by_binding[binding_id])
            for binding_id in sorted(by_binding)
        ]

    def interpret(self) -> dict[str, Any]:
        facts = self.complete_facts()
        contributions: list[dict[str, Any]] = []
        algebra_terms: list[SignedPair] = []
        neutral_facts: list[str] = []
        unmatched_facts: list[str] = []
        for fact in facts:
            matches = _matching_rules(
                fact, self.signed_effect_contract
            )
            if not matches:
                unmatched_facts.append(fact["binding_identity"])
                continue
            if len(matches) != 1:
                raise ValueError("signed-effect contract is ambiguous")
            sign = matches[0]["sign"]
            variable = fact["effect_identity"]
            multiplicity = fact["multiplicity"]
            if sign == "positive":
                algebra_terms.append(
                    SignedPair.positive_variable(
                        variable, multiplicity
                    )
                )
            elif sign == "negative":
                algebra_terms.append(
                    SignedPair.negative_variable(
                        variable, multiplicity
                    )
                )
            elif sign == "neutral_or_not_applicable":
                neutral_facts.append(fact["binding_identity"])
            else:
                raise ValueError(f"unknown algebraic sign: {sign}")
            contributions.append(
                {
                    "binding_identity": fact["binding_identity"],
                    "effect_identity": variable,
                    "multiplicity": multiplicity,
                    "occurrence_identity": fact["omega"][
                        "stable_instance_key"
                    ],
                    "relation_role": fact["rho"],
                    "sign": sign,
                }
            )
        signed_pair = signed_sum(algebra_terms)
        return {
            "algebraic_contributions": sorted(
                contributions,
                key=lambda row: row["binding_identity"],
            ),
            "complete_facts": facts,
            "explicit_disposition_count": len(
                self.snapshot.tables.explicit_dispositions
            ),
            "net_projection": signed_pair.net_projection().to_document(),
            "neutral_fact_ids": sorted(neutral_facts),
            "signed_pair": signed_pair.to_document(),
            "snapshot_id": self.snapshot.snapshot_id,
            "unmatched_fact_ids": sorted(unmatched_facts),
        }
