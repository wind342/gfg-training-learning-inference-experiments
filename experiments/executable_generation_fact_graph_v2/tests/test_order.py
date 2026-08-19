def test_projected_order_candidate_matches_independent_reference(
    order_run,
):
    result, _ = order_run
    assert result["status"] == "PASS"
    assert result["projection_compatibility_query_count"] == 56
    assert result["projection_compatibility_mismatch_count"] == 0


def test_direct_graph_candidate_answers_all_order_queries(
    order_run,
):
    result, _ = order_run
    assert result["direct_graph_query_count"] == 56
    assert result["direct_graph_query_mismatch_count"] == 0
    assert result["false_positive_count"] == 0
    assert result["false_negative_count"] == 0
    assert result["compensation_query_count"] == 4
    assert (
        result["process_isolation"]["direct_graph"][
            "compensation_policy_id"
        ]
        == "order-compensation-target-policy-v1"
    )
    assert result["gates"]["order_compensation_queries_4_exact"]


def test_direct_graph_candidate_isolated_from_old_views(order_run):
    result, _ = order_run
    direct = result["process_isolation"]["direct_graph"]
    assert direct["candidate_source_audit"]["status"] == "PASS"
    assert direct["candidate_runtime_file_read_audit"] == {
        "input_file_only": True,
        "read_count": 1,
    }
    assert not direct["candidate_reads_projected_facts"]
    assert not direct["candidate_reads_projected_sidecar"]
    assert not direct["candidate_reads_raw_receipts"]
    assert not direct["candidate_reads_reference_output"]
