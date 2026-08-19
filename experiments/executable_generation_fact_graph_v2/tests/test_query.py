def test_order_query_examples_include_execution_structures(order_run):
    result, _ = order_run
    examples = result["graphs"][0]["subgraph_examples"]
    assert examples["reads_from_edges"]
    assert examples["queue_send_receive_edges"]
    assert examples["barrier_event_synchronization_edges"]
    assert "order_compensation_target" in examples["RefundCommitted"]


def test_common_query_engine_does_not_claim_compensation_policy(
    order_run,
):
    _, context = order_run
    query = context["contexts"][0]["query_engine"]
    assert not hasattr(query, "compensation_target")
    assert len(query.all_relation_edges()) == len(
        query.graph.relation_edges
    )
