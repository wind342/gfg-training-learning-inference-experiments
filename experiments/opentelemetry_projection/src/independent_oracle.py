"""Frozen expectations. Production/projection modules must never import this file."""

SELECTION_TRACE_ORACLE = {
    "schema_version": "otel-projection-v1",
    "trace_semantic_key": "trace:query-run:oracle-selection-run",
    "spans": [
        {
            "span_semantic_key": "span:query-run:oracle-selection-run",
            "name": "query.execute",
            "parent_semantic_key": None,
            "linked_semantic_keys": [],
            "status": "OK",
            "attributes": {
                "logical.order": 0,
                "span.kind": "query_root",
                "execution.kind": "deterministic_relational_query",
                "execution.run_id": "oracle-selection-run",
            },
            "events": [
                {
                    "name": "query.execution",
                    "attributes": {"execution.run_id": "oracle-selection-run"},
                }
            ],
        },
        {
            "span_semantic_key": "span:occurrence:00000000:relational_selection_execution:oracle_selection:00000000",
            "name": "operator.selection",
            "parent_semantic_key": "span:query-run:oracle-selection-run",
            "linked_semantic_keys": [],
            "status": "OK",
            "attributes": {
                "logical.order": 1,
                "span.kind": "occurrence",
                "operation.type": "selection",
                "operation.stage": "oracle_selection",
                "occurrence.type": "relational_selection_execution",
                "occurrence.stable_instance_key": "oracle_selection:00000000",
                "occurrence.index": 0,
                "outcome.kind": "support",
                "transform.operator_type": "selection",
                "transform.stage": "oracle_selection",
                "occurrence.cardinality": 1,
            },
            "events": [
                {
                    "name": "generation.occurrence",
                    "attributes": {"occurrence.index": 0, "outcome.kind": "support"},
                }
            ],
        },
        {
            "span_semantic_key": "span:occurrence:00000001:relational_selection_execution:oracle_selection:disposition:oracle:b",
            "name": "operator.selection",
            "parent_semantic_key": "span:query-run:oracle-selection-run",
            "linked_semantic_keys": [],
            "status": "OK",
            "attributes": {
                "logical.order": 2,
                "span.kind": "occurrence",
                "operation.type": "selection",
                "operation.stage": "oracle_selection",
                "occurrence.type": "relational_selection_execution",
                "occurrence.stable_instance_key": "oracle_selection:disposition:oracle:b",
                "occurrence.index": 1,
                "outcome.kind": "disposition",
                "transform.operator_type": "selection",
                "transform.stage": "oracle_selection",
                "occurrence.cardinality": 1,
            },
            "events": [
                {
                    "name": "generation.occurrence",
                    "attributes": {
                        "occurrence.index": 1,
                        "outcome.kind": "disposition",
                    },
                }
            ],
        },
        {
            "span_semantic_key": "span:occurrence:00000002:relational_selection_execution:oracle_selection:00000001",
            "name": "operator.selection",
            "parent_semantic_key": "span:query-run:oracle-selection-run",
            "linked_semantic_keys": [],
            "status": "OK",
            "attributes": {
                "logical.order": 3,
                "span.kind": "occurrence",
                "operation.type": "selection",
                "operation.stage": "oracle_selection",
                "occurrence.type": "relational_selection_execution",
                "occurrence.stable_instance_key": "oracle_selection:00000001",
                "occurrence.index": 2,
                "outcome.kind": "support",
                "transform.operator_type": "selection",
                "transform.stage": "oracle_selection",
                "occurrence.cardinality": 1,
            },
            "events": [
                {
                    "name": "generation.occurrence",
                    "attributes": {"occurrence.index": 2, "outcome.kind": "support"},
                }
            ],
        },
    ],
}


BUSINESS_TOPOLOGY_ORACLE = {
    "root_span_count": 1,
    "occurrence_span_count": 36,
    "total_span_count": 37,
    "stage_order": [
        "orders_selected",
        "orders_customers",
        "orders_customers_items",
        "orders_customers_items_products",
        "line_totals",
        "customer_aggregates",
        "customer_rank",
        "customer_top_1",
    ],
}
