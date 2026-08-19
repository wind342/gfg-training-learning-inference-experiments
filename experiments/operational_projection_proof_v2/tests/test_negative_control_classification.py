def test_negative_controls_are_classified_by_actual_depth(artifact) -> None:
    report = artifact("negative_control_classification.json")
    assert report["status"] == "PASS"
    assert report["source_map_control_count"] == 30
    assert set(report["category_counts"]) == {"END_TO_END", "ISOLATION", "VALIDATOR_UNIT"}
    for control in report["controls"]:
        assert control["category"] in {"END_TO_END", "ISOLATION", "VALIDATOR_UNIT"}
        assert control["expected_reason_code"] == control["actual_reason_code"]
        assert control["partial_output_count"] == 0
        assert control["automatic_repair_count"] == 0
        assert control["status"] == "FAIL_CLOSED"

