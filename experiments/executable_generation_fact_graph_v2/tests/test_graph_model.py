from experiments.executable_generation_fact_graph_v2.graph_model import (
    ExecutableGenerationFactGraphV2,
)


def test_graph_round_trip(order_run):
    _, context = order_run
    graph = context["contexts"][0]["validated_graph"].graph
    assert (
        ExecutableGenerationFactGraphV2.from_dict(
            graph.to_dict()
        ).to_dict()
        == graph.to_dict()
    )
