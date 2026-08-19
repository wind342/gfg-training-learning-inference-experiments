def test_no_second_authority_store_exists(artifact) -> None:
    report = artifact("second_authority_audit.json")
    assert report["status"] == "PASS"
    assert report["secondary_authority_store_count"] == 0
    assert report["candidate_answer_input_from_reference_count"] == 0
    assert report["candidate_fallback_relation_store_count"] == 0
    assert report["forbidden_persisted_secondary_relation_stores"] == []

