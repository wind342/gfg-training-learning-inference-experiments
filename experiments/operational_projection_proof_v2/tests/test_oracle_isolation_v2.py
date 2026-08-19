def test_all_candidate_reference_paths_are_isolated(artifact) -> None:
    report = artifact("oracle_isolation.json")
    assert report["status"] == "PASS"
    assert report["domains"]["database"]["oracle_leakage_count"] == 0
    assert report["domains"]["opentelemetry"]["oracle_leakage_count"] == 0
    assert report["domains"]["source_map"]["core_projection_native_or_receipt_import_count"] == 0

