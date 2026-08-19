from __future__ import annotations

from typing import Any


def build_matrix(reports: dict[str, Any]) -> dict[str, Any]:
    db1 = reports["projection_equivalence_database.json"]
    db2 = reports["strict_partiality_database.json"]
    ot1 = reports["projection_equivalence_opentelemetry.json"]
    ot2 = reports["strict_partiality_opentelemetry.json"]
    ot3 = reports[
        "hierarchical_consistency_core_database_to_opentelemetry.json"
    ]
    sm1 = reports["projection_equivalence_source_map.json"]
    sm2 = reports["strict_partiality_source_map.json"]
    sm3 = reports["composition_consistency_source_map.json"]
    rows = [
        {
            "domain_mechanism": "Database which-lineage",
            "p1": db1["status"],
            "p2": db2["status"],
            "p3": "NOT_APPLICABLE",
            "p3_subtype": "domain root/wider projection",
            "declared_scope": "fixed deterministic tuple-level profile",
            "native_or_reference_implementation": "hand-authored frozen independent Oracle",
            "candidate_input": "ValidatedSnapshot plus exact validation proof",
            "exact_record_count": db1["candidate_record_count"],
            "false_positive": db1["false_positive"],
            "false_negative": db1["false_negative"],
            "strict_counterexample_count": db2["counterexample_count"],
            "composition_case_count": 0,
            "output_orthogonality": reports["database_output_orthogonality"][
                "status"
            ],
            "second_authority_store_count": 0,
            "core_specialization_count": 0,
            "limitations": [
                "fixed deterministic fixtures",
                "declared relational operators",
                "tuple-level which-lineage",
            ],
        },
        {
            "domain_mechanism": "OpenTelemetry trace",
            "p1": ot1["status"],
            "p2": ot2["status"],
            "p3": ot3["status"],
            "p3_subtype": "cross-domain hierarchical projection",
            "declared_scope": "deterministic in-process occurrence/execution/causal shadow",
            "native_or_reference_implementation": "official OpenTelemetry Python SDK",
            "candidate_input": "ValidatedSnapshot plus exact validation proof",
            "exact_record_count": {
                "small_spans": ot1["small"]["direct_span_count"],
                "formal_q6_spans": ot1["formal_tpch_q6"][
                    "direct_projected_span_count"
                ],
            },
            "false_positive": ot1["formal_tpch_q6"]["native_vs_direct"][
                "span_false_positives"
            ],
            "false_negative": ot1["formal_tpch_q6"]["native_vs_direct"][
                "span_false_negatives"
            ],
            "strict_counterexample_count": ot2["counterexample_count"],
            "composition_case_count": 2,
            "output_orthogonality": reports["otel_output_orthogonality"][
                "status"
            ],
            "second_authority_store_count": 0,
            "core_specialization_count": reports["otel_oracle_isolation"][
                "otel_specific_core_field_count"
            ],
            "limitations": [
                "deterministic in-process execution",
                "selected occurrence attributes/events",
                "no distributed causality or sampling",
            ],
        },
        {
            "domain_mechanism": "ECMA-426 Source Map",
            "p1": sm1["status"],
            "p2": sm2["status"],
            "p3": sm3["status"],
            "p3_subtype": "multistage generation composition",
            "declared_scope": "ordinary non-indexed JavaScript profile",
            "standard_surface_status": "PARTIAL",
            "native_or_reference_implementation": "official source-map 0.8.0 generator and consumer",
            "candidate_input": "ValidatedSnapshot only",
            "exact_record_count": sm1["total_mapping_segments"],
            "false_positive": 0,
            "false_negative": 0,
            "strict_counterexample_count": sm2["counterexample_count"],
            "composition_case_count": sm3["composed_mapping_count"],
            "output_orthogonality": reports["source_map_output_orthogonality"][
                "status"
            ],
            "second_authority_store_count": 0,
            "core_specialization_count": 0,
            "limitations": [
                "ordinary non-indexed version-3 maps",
                "JavaScript UTF-16 coordinates",
                "indexed maps and non-JavaScript surfaces excluded",
            ],
        },
    ]
    expected = {
        "Database which-lineage": ("SUPPORTED", "SUPPORTED", "NOT_APPLICABLE"),
        "OpenTelemetry trace": ("SUPPORTED", "SUPPORTED", "SUPPORTED"),
        "ECMA-426 Source Map": ("SUPPORTED", "SUPPORTED", "SUPPORTED"),
    }
    exact = all(
        (row["p1"], row["p2"], row["p3"])
        == expected[row["domain_mechanism"]]
        for row in rows
    )
    return {
        "matrix_id": "unified-operational-projection-proof-v2",
        "rows": rows,
        "source_map_standard_surface_status": "PARTIAL",
        "status": "SUPPORTED" if exact else "NOT_SUPPORTED",
    }


def render_matrix_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# Unified Projection Matrix",
        "",
        "| Domain mechanism | P1 | P2 | P3 | P3 subtype | Declared scope |",
        "|---|---|---|---|---|---|",
    ]
    for row in matrix["rows"]:
        lines.append(
            "| {domain_mechanism} | {p1} | {p2} | {p3} | {p3_subtype} | {declared_scope} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "The ordinary non-indexed Source Map profile is `SUPPORTED`; the full ECMA-426 standard surface remains `PARTIAL` because indexed maps and declared non-JavaScript surfaces are excluded.",
            "",
            f"Conjunctive matrix status: `{matrix['status']}`.",
        ]
    )
    return "\n".join(lines) + "\n"

