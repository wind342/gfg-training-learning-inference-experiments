from __future__ import annotations

from typing import Any

from experiments.signed_generation_algebra_v1.algebra import (
    SignedPair,
    signed_sum,
)

from ..graph_model import ValidatedGenerationFactGraphV2


def _matching_rules(
    node: Any,
    outcome_kind: str,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    occurrence = node.omega_bar["generation_occurrence"]
    return [
        rule
        for rule in contract["rules"]
        if rule["transform_operation"] == node.tau.get("operation")
        and rule["occurrence_type"]
        == occurrence["occurrence_type"]
        and rule["relation_role"] == node.rho
        and rule["outcome_kind"] == outcome_kind
    ]


def project_graph_to_signed_algebra(
    validated_graph: ValidatedGenerationFactGraphV2,
    contract: dict[str, Any],
) -> dict[str, Any]:
    contributions = []
    terms: list[SignedPair] = []
    neutral = []
    unmatched = []
    disposition_count = 0
    complete_facts = []
    for node in sorted(
        validated_graph.graph.fact_nodes,
        key=lambda row: row.generation_binding_id,
    ):
        reference = node.z["reference"]
        entity = node.z["entity"]
        if reference["kind"] == "support":
            outcome_kind = "support"
            outcome_id = reference["support_id"]
            payload = entity["support_payload"]
        elif reference["kind"] == "disposition":
            outcome_kind = "disposition"
            outcome_id = reference["disposition_id"]
            payload = entity["disposition_payload"]
            disposition_count += 1
        else:
            raise ValueError("SIGNED_PROJECTION_OUTCOME_KIND_UNKNOWN")
        multiplicity = payload.get("effect_multiplicity")
        effect_identity = payload.get("effect_identity")
        if (
            not isinstance(multiplicity, int)
            or isinstance(multiplicity, bool)
            or multiplicity <= 0
        ):
            raise ValueError("SIGNED_PROJECTION_MULTIPLICITY_INVALID")
        if (
            not isinstance(effect_identity, str)
            or not effect_identity.startswith("x_")
        ):
            raise ValueError(
                "SIGNED_PROJECTION_EFFECT_IDENTITY_INVALID"
            )
        source = node.u["entity"]
        occurrence = node.omega_bar["generation_occurrence"]
        complete_facts.append(
            {
                "binding_identity": node.generation_binding_id,
                "effect_identity": effect_identity,
                "multiplicity": multiplicity,
                "rho": node.rho,
                "scope": node.domain_scope_id,
                "tau": node.tau,
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
                    "generation_occurrence_id": (
                        node.generation_occurrence_id
                    ),
                    "occurrence_payload": occurrence[
                        "occurrence_payload"
                    ],
                    "occurrence_type": occurrence["occurrence_type"],
                    "stable_instance_key": occurrence[
                        "stable_instance_key"
                    ],
                },
            }
        )
        matches = _matching_rules(node, outcome_kind, contract)
        if not matches:
            unmatched.append(node.generation_binding_id)
            continue
        if len(matches) != 1:
            raise ValueError("SIGNED_EFFECT_CONTRACT_AMBIGUOUS")
        sign = matches[0]["sign"]
        if sign == "positive":
            terms.append(
                SignedPair.positive_variable(
                    effect_identity, multiplicity
                )
            )
        elif sign == "negative":
            terms.append(
                SignedPair.negative_variable(
                    effect_identity, multiplicity
                )
            )
        elif sign == "neutral_or_not_applicable":
            neutral.append(node.generation_binding_id)
        else:
            raise ValueError("SIGNED_EFFECT_SIGN_UNKNOWN")
        contributions.append(
            {
                "binding_identity": node.generation_binding_id,
                "effect_identity": effect_identity,
                "multiplicity": multiplicity,
                "occurrence_identity": occurrence[
                    "stable_instance_key"
                ],
                "relation_role": node.rho,
                "sign": sign,
            }
        )
    signed_pair = signed_sum(terms)
    return {
        "algebraic_contributions": contributions,
        "complete_facts": complete_facts,
        "explicit_disposition_count": disposition_count,
        "net_projection": signed_pair.net_projection().to_document(),
        "neutral_fact_ids": sorted(neutral),
        "signed_pair": signed_pair.to_document(),
        "snapshot_id": (
            validated_graph.graph.metadata.source_snapshot_ids[0]
        ),
        "unmatched_fact_ids": sorted(unmatched),
    }
