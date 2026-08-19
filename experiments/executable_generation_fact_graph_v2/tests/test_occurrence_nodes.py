def test_zero_fact_occurrences_are_preserved(order_run):
    result, context = order_run
    assert result["gates"]["zero_fact_occurrences_present"]
    for row in context["contexts"]:
        validation = row["validated_graph"].validation
        assert validation.counts["zero_fact_occurrences"] == 11
