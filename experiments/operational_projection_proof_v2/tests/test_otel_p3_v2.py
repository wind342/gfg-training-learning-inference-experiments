def test_otel_p3_is_hierarchical_and_isolated(artifact) -> None:
    report = artifact("hierarchical_consistency_core_database_to_opentelemetry.json")
    assert report["status"] == "SUPPORTED"
    assert report["p3_subtype"] == "cross-domain hierarchical projection"
    assert report["small"]["direct_vs_hierarchical"]["exact"] is True
    assert report["small"]["canonical_bytes_equal"] is True
    assert report["formal_tpch_q6"]["direct_vs_hierarchical"]["exact"] is True
    assert report["module_isolation"]["status"] == "PASS"
    assert all(report["mandatory_checks"].values())

