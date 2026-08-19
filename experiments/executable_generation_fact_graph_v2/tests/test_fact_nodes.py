def test_one_fact_node_per_binding(order_run):
    _, context = order_run
    for row in context["contexts"]:
        graph = row["validated_graph"].graph
        expected = row["snapshot_input"][
            "snapshot"
        ].tables.generation_bindings
        assert len(graph.fact_nodes) == len(expected)
        assert len(
            {node.generation_binding_id for node in graph.fact_nodes}
        ) == len(expected)
