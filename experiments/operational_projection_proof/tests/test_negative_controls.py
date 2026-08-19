def test_all_fail_closed_negative_controls_return_exact_reason_codes(
    proof_reports,
) -> None:
    report = proof_reports["negative_controls.json"]
    assert report["status"] == "SUPPORTED"
    assert report["negative_control_count"] == 13
    assert report["passed_count"] == 13
    assert report["failed_count"] == 0
    assert all(
        row["actual_reason_code"] == row["expected_reason_code"]
        for row in report["controls"]
    )
