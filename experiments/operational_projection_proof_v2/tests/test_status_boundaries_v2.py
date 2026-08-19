def test_only_conjunctive_supported_status_is_emitted(artifact) -> None:
    report = artifact("run_summary.json")
    assert report["status"] == "UNIFIED_OPERATIONAL_PROJECTION_PROOF_V2_SUPPORTED"
    assert report["blocking_reasons"] == []
    assert all(report["mandatory_checks"].values())
    assert set(report["domain_statuses"].values()) == {"SUPPORTED"}
    assert report["status"] not in {"MOSTLY_SUPPORTED", "SUBSTANTIALLY_SUPPORTED", "PASS_WITH_MINOR_ISSUES"}

