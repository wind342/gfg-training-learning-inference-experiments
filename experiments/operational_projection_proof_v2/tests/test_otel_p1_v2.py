def test_otel_p1_small_and_formal_are_exact(artifact) -> None:
    report = artifact("projection_equivalence_opentelemetry.json")
    formal = report["formal_tpch_q6"]
    assert report["status"] == "SUPPORTED"
    assert report["small"]["native_vs_direct"]["exact"] is True
    assert formal["core_occurrence_count"] == 61367
    assert formal["direct_projected_span_count"] == 61368
    assert formal["core_binding_count"] == 62557
    assert formal["causal_link_count"] == 2382
    assert set(formal["trace_sha256"].values()) == {
        "a0095ed24e3ad6ec58064a1b5803e532b11c85c08a5ad7541b03dac1e064efe8"
    }
    assert all(report["mandatory_checks"].values())

