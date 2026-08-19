"""Independent adapter over the frozen hand-authored database Oracle."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from generation_relation_core.canonical import canonical_bytes

from experiments.database_lineage.src.synthetic_oracle import (
    BUSINESS_BACKWARD,
    BUSINESS_DIRECT_PAIRS,
    BUSINESS_DISPOSITIONS,
    BUSINESS_FORWARD,
    MANY_TO_MANY_DIRECT_PAIRS,
)

from .errors import ProjectionProofError
from .projection_profile import ProjectionProfile
from .projection_result import empty_result, normalize_result


def _role(input_tuple_id: str, output_tuple_id: str) -> str:
    if output_tuple_id.startswith("orders_selected:"):
        return "selection_input"
    if output_tuple_id.startswith("orders_customers:"):
        return (
            "join_right_input"
            if input_tuple_id.startswith("customers:")
            else "join_left_input"
        )
    if output_tuple_id.startswith("orders_customers_items:"):
        return (
            "join_right_input"
            if input_tuple_id.startswith("items:")
            else "join_left_input"
        )
    if output_tuple_id.startswith("orders_customers_items_products:"):
        return (
            "join_right_input"
            if input_tuple_id.startswith("products:")
            else "join_left_input"
        )
    if output_tuple_id.startswith("line_totals:"):
        return "projection_input"
    if output_tuple_id.startswith("customer_aggregates:"):
        return "aggregation_contributor"
    if output_tuple_id.startswith("customer_rank:"):
        return "sort_input"
    if output_tuple_id.startswith("customer_top_1:"):
        return "limit_retained"
    if output_tuple_id.startswith("many_to_many:"):
        return (
            "join_right_input"
            if input_tuple_id.startswith("products:")
            else "join_left_input"
        )
    raise ProjectionProofError("ORACLE_SCOPE_UNDECLARED", output_tuple_id)


def _support_records(
    workload_id: str, pairs: set[tuple[str, str]]
) -> list[dict[str, Any]]:
    return [
        {
            "workload_id": workload_id,
            "input_tuple_id": input_id,
            "output_tuple_id": output_id,
            "outcome_kind": "support",
            "relation_role": _role(input_id, output_id),
            "relation_ordinal": 0,
        }
        for input_id, output_id in sorted(pairs)
    ]


def _oracle_paths(
    *,
    final_output: str,
    pairs: set[tuple[str, str]],
    base_sources: set[str],
) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    reverse: dict[str, list[str]] = defaultdict(list)
    for input_id, output_id in pairs:
        reverse[output_id].append(input_id)

    def walk(
        current: str, visited: frozenset[str]
    ) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
        if current in visited:
            raise ProjectionProofError("HIERARCHY_CYCLE", current)
        if current in base_sources:
            return [(current, (current,), ())]
        result = []
        for input_id in sorted(reverse.get(current, [])):
            for source, nodes, roles in walk(input_id, visited | {current}):
                result.append(
                    (source, (*nodes, current), (*roles, _role(input_id, current)))
                )
        return result

    return sorted(
        walk(final_output, frozenset()),
        key=lambda item: canonical_bytes([item[0], list(item[1]), list(item[2])]),
    )


def business_oracle_result(profile: ProjectionProfile) -> dict[str, Any]:
    workload_id = "database-business-v1"
    direct = _support_records(workload_id, BUSINESS_DIRECT_PAIRS)
    disposition_rows = [
        {
            "workload_id": workload_id,
            "input_tuple_id": tuple_id,
            "output_tuple_id": tuple_id,
            "core_disposition_category": "suppressed",
            "reason_code": reason,
        }
        for tuple_id, reason in sorted(BUSINESS_DISPOSITIONS)
    ]
    direct.extend(
        {
            "workload_id": workload_id,
            "input_tuple_id": row["input_tuple_id"],
            "output_tuple_id": row["output_tuple_id"],
            "outcome_kind": "disposition",
            "relation_role": row["reason_code"],
            "relation_ordinal": 0,
        }
        for row in disposition_rows
    )
    base_sources = set(BUSINESS_FORWARD)
    path_rows = []
    backward_rows = []
    path_outputs_by_source: dict[str, list[str]] = defaultdict(list)
    for output_id in sorted(BUSINESS_BACKWARD):
        paths = _oracle_paths(
            final_output=output_id,
            pairs=BUSINESS_DIRECT_PAIRS,
            base_sources=base_sources,
        )
        if {source for source, _nodes, _roles in paths} != set(
            BUSINESS_BACKWARD[output_id]
        ):
            raise ProjectionProofError("ORACLE_INTERNAL_MISMATCH", "BACKWARD")
        ordinals: dict[str, int] = defaultdict(int)
        for source, nodes, roles in paths:
            path_rows.append(
                {
                    "workload_id": workload_id,
                    "output_tuple_id": output_id,
                    "source_tuple_id": source,
                    "path_ordinal": ordinals[source],
                    "tuple_path": list(nodes),
                    "relation_roles": list(roles),
                    "path_length": len(roles),
                }
            )
            ordinals[source] += 1
            path_outputs_by_source[source].append(output_id)
        backward_rows.append(
            {
                "workload_id": workload_id,
                "output_tuple_id": output_id,
                "source_tuple_ids": sorted(BUSINESS_BACKWARD[output_id]),
                "derivation_path_count": len(paths),
            }
        )
    forward_rows = []
    for source_id, expected_outputs in sorted(BUSINESS_FORWARD.items()):
        actual_outputs = set(path_outputs_by_source.get(source_id, []))
        if actual_outputs != set(expected_outputs):
            raise ProjectionProofError("ORACLE_INTERNAL_MISMATCH", "FORWARD")
        forward_rows.append(
            {
                "workload_id": workload_id,
                "source_tuple_id": source_id,
                "output_tuple_ids": sorted(expected_outputs),
                "derivation_path_count": len(path_outputs_by_source.get(source_id, [])),
            }
        )
    result = empty_result(profile)
    result["records"].update(
        {
            "direct_relations": direct,
            "backward_lineage": backward_rows,
            "forward_lineage": forward_rows,
            "derivation_paths": path_rows,
            "explicit_dispositions": disposition_rows,
            "multiplicity": [
                {
                    "workload_id": workload_id,
                    "total_relation_count": len(direct),
                    "support_relation_count": len(BUSINESS_DIRECT_PAIRS),
                    "disposition_relation_count": len(BUSINESS_DISPOSITIONS),
                    "derivation_path_count": len(path_rows),
                    "final_output_count": len(BUSINESS_BACKWARD),
                }
            ],
            "duplicate_identities": [],
        }
    )
    return normalize_result(result, profile)


def many_to_many_oracle_result(profile: ProjectionProfile) -> dict[str, Any]:
    workload_id = "database-many-to-many-v1"
    direct = _support_records(workload_id, MANY_TO_MANY_DIRECT_PAIRS)
    result = empty_result(profile)
    result["records"].update(
        {
            "direct_relations": direct,
            "backward_lineage": [],
            "forward_lineage": [],
            "derivation_paths": [],
            "explicit_dispositions": [],
            "multiplicity": [
                {
                    "workload_id": workload_id,
                    "total_relation_count": len(direct),
                    "support_relation_count": len(direct),
                    "disposition_relation_count": 0,
                    "derivation_path_count": 0,
                    "final_output_count": 0,
                }
            ],
            "duplicate_identities": [
                {
                    "workload_id": workload_id,
                    "case_id": "equal-valued-products-distinct-identities",
                    "source_tuple_ids": ["products:p1a", "products:p1b"],
                    "distinct_identity_count": 2,
                    "source_payload_equal": True,
                    "output_tuple_ids": [
                        f"many_to_many:{index:08d}" for index in range(4)
                    ],
                    "direct_relation_count": 4,
                }
            ],
        }
    )
    return normalize_result(result, profile)
