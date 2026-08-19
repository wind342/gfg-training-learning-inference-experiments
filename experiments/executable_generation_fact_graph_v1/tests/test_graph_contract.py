from __future__ import annotations

from experiments.executable_generation_fact_graph_v1.graph_compiler import (
    _relation_edges,
)
from experiments.executable_generation_fact_graph_v1.graph_model import (
    GraphFactNode,
)


def _node(binding: str, occurrence: str) -> GraphFactNode:
    return GraphFactNode(
        graph_node_id="node-" + binding,
        execution_run_id="run",
        snapshot_id="snapshot",
        generation_binding_id=binding,
        domain_scope_id="domain",
        source_reference={},
        realized_transformation={},
        concrete_occurrence={},
        outcome_reference={},
        relation_role="role",
        occurrence_identity=occurrence,
        outcome_identity="outcome-" + binding,
        evidence_refs=[],
        fact_content_hash="content-" + binding,
        node_instance_hash="instance-" + binding,
        native_fact_identity=None,
    )


def test_occurrence_without_fact_fails_closed() -> None:
    relation = {
        "relation_id": "relation",
        "execution_run_id": "run",
        "relation_type": "program_order",
        "endpoint_level": "occurrence",
        "source_id": "missing",
        "target_id": "occurrence",
        "establishment_source": "wrapper_established",
        "authority_id": "authority",
        "evidence_refs": [],
    }
    try:
        _relation_edges(
            [_node("binding", "occurrence")],
            {"relation_store_id": "store", "relations": [relation]},
            {
                "rules": {
                    "program_order": {
                        "rule_id": "rule",
                    }
                }
            },
            {
                "relations": {
                    "program_order": {
                        "semantics": "directed",
                        "kind": "primitive",
                    }
                }
            },
            execution_run_id="run",
            schema_version="v1",
        )
    except ValueError as exc:
        assert str(exc).startswith(
            "OCCURRENCE_ENDPOINT_WITHOUT_FACT_NODE"
        )
    else:
        raise AssertionError("missing occurrence endpoint was not rejected")


def test_multi_fact_occurrence_fails_before_cartesian_expansion() -> None:
    relation = {
        "relation_id": "relation",
        "execution_run_id": "run",
        "relation_type": "program_order",
        "endpoint_level": "occurrence",
        "source_id": "left",
        "target_id": "right",
        "establishment_source": "wrapper_established",
        "authority_id": "authority",
        "evidence_refs": [],
    }
    try:
        _relation_edges(
            [
                _node("left-a", "left"),
                _node("left-b", "left"),
                _node("right-a", "right"),
                _node("right-b", "right"),
            ],
            {"relation_store_id": "store", "relations": [relation]},
            {
                "rules": {
                    "program_order": {
                        "rule_id": "rule",
                    }
                }
            },
            {
                "relations": {
                    "program_order": {
                        "semantics": "directed",
                        "kind": "primitive",
                    }
                }
            },
            execution_run_id="run",
            schema_version="v1",
        )
    except ValueError as exc:
        assert str(exc).startswith(
            "AMBIGUOUS_OCCURRENCE_TO_FACT_LIFTING"
        )
    else:
        raise AssertionError("ambiguous occurrence lifting was not rejected")
