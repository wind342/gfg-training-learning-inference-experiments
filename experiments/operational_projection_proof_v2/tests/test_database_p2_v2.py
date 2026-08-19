def test_database_p2_contains_two_complete_snapshot_counterexamples(artifact) -> None:
    report = artifact("strict_partiality_database.json")
    assert report["status"] == "SUPPORTED"
    assert report["counterexample_count"] == 2
    required = {
        "left_snapshot_id",
        "right_snapshot_id",
        "complete_snapshot_equal",
        "projection_hash_equal",
        "binding_set_equal",
        "binding_symmetric_difference_count",
        "occurrence_set_equal",
        "occurrence_symmetric_difference_count",
        "transform_context_equal",
        "environment_context_equal",
        "output_equal",
        "valid_counterexample",
    }
    for case in report["cases"]:
        assert required <= set(case)
        assert case["complete_snapshot_equal"] is False
        assert case["projection_hash_equal"] is True
        assert case["output_equal"] is True
        assert case["valid_counterexample"] is True

