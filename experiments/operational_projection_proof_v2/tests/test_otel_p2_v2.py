def test_otel_p2_reports_all_required_differences(artifact) -> None:
    report = artifact("strict_partiality_opentelemetry.json")
    assert report["status"] == "SUPPORTED"
    assert report["counterexample_count"] == 2
    assert report["total_binding_symmetric_difference_count"] == 20
    for case in report["cases"]:
        assert case["ordinary_output_equal"] is True
        assert case["native_normalized_otel_equal"] is True
        assert case["direct_core_projection_equal"] is True
        assert case["complete_snapshot_equal"] is False
        assert case["source_set_equal"] is False
        assert case["binding_set_equal"] is False
        assert case["direct_relation_set_equal"] is False
        assert case["backward_lineage_equal"] is False
        assert case["normalized_trace_hash_equal"] is True
        assert case["valid_counterexample"] is True

