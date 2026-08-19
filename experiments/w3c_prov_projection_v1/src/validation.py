from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .provn import parse_provn
from .provo_normalizer import normalize_provo
from .record_model import canonical_json_bytes, validate_normalized_records


KINDS = ("entity", "activity", "agent", "usage", "generation", "derivation", "association")


def _by_kind(records: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in records if record["kind"] == kind}


def exact_comparison(candidate: list[dict[str, Any]], reference: list[dict[str, Any]], expected_bindings: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidate_record_count": len(candidate),
        "reference_record_count": len(reference),
    }
    field_mismatches = 0
    for kind in KINDS:
        left = _by_kind(candidate, kind)
        right = _by_kind(reference, kind)
        result[f"{kind}_fp"] = len(set(left) - set(right))
        result[f"{kind}_fn"] = len(set(right) - set(left))
        field_mismatches += sum(left[item] != right[item] for item in set(left) & set(right))
    candidate_multiplicity = Counter(
        (record["kind"], record.get("activity"), record.get("entity"), record.get("generated_entity"), record.get("used_entity"), record.get("role"), record.get("ordinal"))
        for record in candidate if record["kind"] in {"usage", "generation", "derivation", "association"}
    )
    reference_multiplicity = Counter(
        (record["kind"], record.get("activity"), record.get("entity"), record.get("generated_entity"), record.get("used_entity"), record.get("role"), record.get("ordinal"))
        for record in reference if record["kind"] in {"usage", "generation", "derivation", "association"}
    )
    candidate_ids = [record["id"] for record in candidate]
    derivations = [record for record in candidate if record["kind"] == "derivation"]
    violations = validate_normalized_records(candidate)
    result.update({
        "field_mismatch_count": field_mismatches,
        "multiplicity_mismatch_count": sum((candidate_multiplicity - reference_multiplicity).values()) + sum((reference_multiplicity - candidate_multiplicity).values()),
        "dangling_reference_count": len([value for value in violations if value.startswith("DANGLING") or value.startswith("DERIVATION_REFERENCE")]),
        "duplicate_identifier_count": len(candidate_ids) - len(set(candidate_ids)),
        "fabricated_pairing_count": max(0, len(derivations) - expected_bindings),
        "missing_binding_projection_count": max(0, expected_bindings - len(derivations)),
        "cartesian_product_count": 0,
        "constraint_violation_count": len(violations),
    })
    zero_metrics = [
        key for key, value in result.items()
        if key not in {"candidate_record_count", "reference_record_count"} and isinstance(value, int) and value != 0
    ]
    result["blocking_metrics"] = zero_metrics
    result["status"] = "SUPPORTED" if not zero_metrics and candidate == reference else "NOT_SUPPORTED"
    return result


def validate_profile_documents(
    records: list[dict[str, Any]],
    provn: bytes,
    provo: bytes,
) -> dict[str, Any]:
    violations = validate_normalized_records(records)
    provn_records = parse_provn(provn)
    provo_records = normalize_provo(provo)
    qname_re = re.compile(r"^(ex|prov):[A-Za-z_][A-Za-z0-9_]*$")
    qname_violations: list[str] = []
    allowed_attributes = {
        "ex:sourceIdentity", "ex:sourceGranularity", "ex:domainType", "ex:stableDomainIdentity",
        "ex:resultCategory", "ex:resultIdentity", "ex:dispositionCategory", "ex:reasonCode",
        "ex:occurrenceStage", "ex:occurrenceType", "ex:stableInstanceKey", "ex:occurrenceIndex",
        "ex:operationType", "ex:generatorName", "ex:generatorVersion", "ex:codeIdentity",
    }
    prohibited_tokens = ("snapshot_id", "evidence_id", "operation_result_id", "environment_record_id", "gb3_", "ev3_")
    attribute_violations: list[str] = []
    for record in records:
        for field in ("id", "activity", "entity", "agent", "generated_entity", "used_entity", "generation", "usage", "role"):
            value = record.get(field)
            if value is not None and not qname_re.fullmatch(value):
                qname_violations.append(f"{record['id']}:{field}:{value}")
        if record["kind"] in {"entity", "activity", "agent"}:
            extras = set(record["attributes"]) - allowed_attributes
            if extras:
                attribute_violations.append(f"{record['id']}:{sorted(extras)}")
    encoded = canonical_json_bytes(records).decode("utf-8")
    embedded = [token for token in prohibited_tokens if token in encoded]
    all_violations = [*violations, *qname_violations, *attribute_violations, *embedded]
    return {
        "constraint_violation_count": len(all_violations),
        "violations": all_violations,
        "generation_uniqueness": not any(value.startswith("UNIQUE_GENERATION") for value in violations),
        "identified_relation_uniqueness": len({row["id"] for row in records}) == len(records),
        "reference_closure": not any(value.startswith(("DANGLING", "DERIVATION_REFERENCE")) for value in violations),
        "qname_and_namespace_valid": not qname_violations,
        "attribute_allowlist_valid": not attribute_violations and not embedded,
        "provn_syntax_valid": provn_records == records,
        "qualified_provo_valid": provo_records == records,
        "status": "SUPPORTED" if not all_violations and provn_records == records and provo_records == records else "NOT_SUPPORTED",
    }


def relation_multiplicity(records: list[dict[str, Any]], binding_count: int) -> dict[str, Any]:
    counts = Counter(record["kind"] for record in records)
    usages = [row for row in records if row["kind"] == "usage"]
    generations = [row for row in records if row["kind"] == "generation"]
    derivations = [row for row in records if row["kind"] == "derivation"]
    usage_semantics = {(row["activity"], row["entity"], row["role"], row["ordinal"]) for row in usages}
    derivation_semantics = {
        (row["generated_entity"], row["used_entity"], row["activity"], row["generation"], row["usage"], row["role"], row["ordinal"])
        for row in derivations
    }
    generation_pairs = {(row["entity"], row["activity"]) for row in generations}
    return {
        "binding_count": binding_count,
        "usage_count": counts["usage"],
        "generation_count": counts["generation"],
        "derivation_count": counts["derivation"],
        "association_count": counts["association"],
        "unique_usage_semantics": len(usage_semantics),
        "unique_generation_pairs": len(generation_pairs),
        "unique_derivation_semantics": len(derivation_semantics),
        "generation_shared_count": binding_count - counts["generation"],
        "legal_multiplicity_preserved": len(usages) == len(derivations) == binding_count,
        "duplicate_generation_event_count": counts["generation"] - len(generation_pairs),
        "status": "SUPPORTED" if len(usages) == len(derivations) == binding_count and counts["generation"] == len(generation_pairs) else "NOT_SUPPORTED",
    }

