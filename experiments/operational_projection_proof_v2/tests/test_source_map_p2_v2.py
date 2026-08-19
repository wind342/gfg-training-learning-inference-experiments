def test_source_map_p2_compares_real_snapshot_tables(artifact) -> None:
    report = artifact("strict_partiality_source_map.json")
    assert report["status"] == "SUPPORTED"
    assert report["counterexample_count"] == 3
    required = {
        "left_snapshot_id",
        "right_snapshot_id",
        "left_snapshot_semantic_sha256",
        "right_snapshot_semantic_sha256",
        "map_document_equal",
        "map_document_sha256",
        "complete_snapshot_equal",
        "source_set_equal",
        "occurrence_set_equal",
        "occurrence_symmetric_difference_count",
        "binding_set_equal",
        "binding_symmetric_difference_count",
        "disposition_set_equal",
        "disposition_symmetric_difference_count",
        "operation_result_set_equal",
        "transform_context_equal",
        "evidence_set_equal",
        "valid_counterexample",
    }
    for case in report["cases"]:
        assert required <= set(case)
        assert case["map_document_equal"] is True
        assert case["complete_snapshot_equal"] is False
        assert case["valid_counterexample"] is True
    ambiguity = artifact("result_only_ambiguity_source_map.json")
    assert ambiguity["ambiguity_case_count"] == 2
    assert all(case["generated_output_equal"] and not case["source_maps_equal"] for case in ambiguity["cases"])

