def test_v1_tree_and_artifact_bytes_are_preserved(artifact) -> None:
    report = artifact("v1_preservation.json")
    assert report["status"] == "PASS"
    assert report["expected_tree_id"] == report["observed_tree_id"]
    assert report["expected_artifact_tree_id"] == report["observed_artifact_tree_id"]
    assert report["working_or_committed_differences"] == []
    assert report["not_evaluated_statuses_preserved"] is True

