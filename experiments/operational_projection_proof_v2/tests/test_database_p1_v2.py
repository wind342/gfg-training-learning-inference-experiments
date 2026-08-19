def test_database_p1_is_exact_and_frozen(artifact) -> None:
    report = artifact("projection_equivalence_database.json")
    assert report["status"] == "SUPPORTED"
    assert report["rerun_on_integrated_branch"] is True
    assert report["candidate_record_count"] == 112
    assert report["reference_record_count"] == 112
    assert report["false_positive"] == 0
    assert report["false_negative"] == 0
    assert report["field_mismatch"] == 0
    assert report["multiplicity_mismatch"] == 0
    assert all(report["frozen_checks"].values())

