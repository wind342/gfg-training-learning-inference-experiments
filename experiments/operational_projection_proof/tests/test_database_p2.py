def test_database_which_lineage_has_two_strict_partiality_counterexamples(
    proof_reports,
) -> None:
    report = proof_reports["strict_partiality_database.json"]
    assert report["status"] == "SUPPORTED"
    assert report["counterexample_count"] == 2
    assert report["projection_equal"] is True
    assert report["complete_snapshot_equal"] is False
    assert report["binding_set_equal"] is False
    assert report["transform_context_equal"] is False
    assert all(case["valid_counterexample"] for case in report["cases"])
    assert report["interpretation"] == (
        "Projection equality does not imply complete generation-fact equality."
    )
