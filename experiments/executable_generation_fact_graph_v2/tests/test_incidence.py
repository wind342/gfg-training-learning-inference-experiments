def test_every_fact_has_exactly_one_incidence(order_run):
    _, context = order_run
    for row in context["contexts"]:
        graph = row["validated_graph"].graph
        assert len(graph.incidence_edges) == len(graph.fact_nodes)
        targets = [
            edge.target_fact_node_id
            for edge in graph.incidence_edges
        ]
        assert len(targets) == len(set(targets))
