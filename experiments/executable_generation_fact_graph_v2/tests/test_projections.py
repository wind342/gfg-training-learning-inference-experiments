def test_order_relation_projection_is_exact_and_fact_view_is_honest(
    order_run,
):
    result, _ = order_run
    assert all(
        row["relation_projection_exact"]
        for row in result["graphs"]
    )
    assert all(
        row["fact_only_omitted_count"] > 0
        for row in result["graphs"]
    )
