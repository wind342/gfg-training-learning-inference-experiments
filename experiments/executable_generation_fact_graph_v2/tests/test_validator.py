import pytest

from experiments.executable_generation_fact_graph_v2.graph_validator import (
    GraphValidationErrorV2,
    reject_acyclic_relation_family_cycles,
    reject_symmetric_double_write,
)


def test_order_graphs_are_validated(order_run):
    _, context = order_run
    assert all(
        row["validated_graph"].validation.status == "PASS"
        for row in context["contexts"]
    )


def test_cross_family_directed_cycle_is_allowed(order_run):
    _, context = order_run
    registry = context["contracts"]["relation_type_registry"][
        "relations"
    ]
    reject_acyclic_relation_family_cycles(
        [
            {
                "relation_type": "generated_origin_dependency",
                "source_node_id": "A",
                "target_node_id": "B",
            },
            {
                "relation_type": "reads_from",
                "source_node_id": "B",
                "target_node_id": "A",
            },
        ],
        registry,
    )


def test_declared_acyclic_family_cycle_is_rejected(order_run):
    _, context = order_run
    registry = context["contracts"]["relation_type_registry"][
        "relations"
    ]
    with pytest.raises(
        GraphValidationErrorV2,
        match="ACYCLIC_RELATION_FAMILY_CYCLE",
    ):
        reject_acyclic_relation_family_cycles(
            [
                {
                    "relation_type": "program_order",
                    "source_node_id": "A",
                    "target_node_id": "B",
                },
                {
                    "relation_type": "synchronizes_with",
                    "source_node_id": "B",
                    "target_node_id": "A",
                },
            ],
            registry,
        )


def test_independent_symmetric_parallel_edges_are_allowed():
    reject_symmetric_double_write(
        [
            {
                "relation_semantics": "symmetric",
                "relation_type": "conflicts_with",
                "original_relation_id": "relation-a",
                "source_node_id": "A",
                "target_node_id": "B",
            },
            {
                "relation_semantics": "symmetric",
                "relation_type": "conflicts_with",
                "original_relation_id": "relation-b",
                "source_node_id": "B",
                "target_node_id": "A",
            },
        ]
    )


def test_same_symmetric_relation_instance_is_rejected():
    with pytest.raises(
        GraphValidationErrorV2,
        match="SYMMETRIC_RELATION_INSTANCE_DOUBLE_WRITTEN",
    ):
        reject_symmetric_double_write(
            [
                {
                    "relation_semantics": "symmetric",
                    "relation_type": "conflicts_with",
                    "original_relation_id": "relation-a",
                    "source_node_id": "A",
                    "target_node_id": "B",
                },
                {
                    "relation_semantics": "symmetric",
                    "relation_type": "conflicts_with",
                    "original_relation_id": "relation-a",
                    "source_node_id": "B",
                    "target_node_id": "A",
                },
            ]
        )
