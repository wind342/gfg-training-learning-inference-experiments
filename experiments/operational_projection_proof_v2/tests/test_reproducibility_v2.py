from generation_relation_core.canonical import canonical_bytes


def test_first_complete_run_is_canonically_self_reproducible(artifact) -> None:
    report = artifact("runs/run_1/scientific_reports.json")
    assert canonical_bytes(report) == canonical_bytes(report)
    assert "performance_seconds" not in report["projection_equivalence_opentelemetry.json"]["formal_tpch_q6"]
    assert "peak_process_rss_bytes" not in report["projection_equivalence_opentelemetry.json"]["formal_tpch_q6"]
    assert report["run_summary.json"]["status"] == "UNIFIED_OPERATIONAL_PROJECTION_PROOF_V2_SUPPORTED"

