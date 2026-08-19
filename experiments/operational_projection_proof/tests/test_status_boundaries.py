def test_unfinished_domains_are_not_promoted(proof_reports) -> None:
    for name in (
        "projection_equivalence_opentelemetry.json",
        "projection_equivalence_source_map.json",
        "strict_partiality_opentelemetry.json",
        "strict_partiality_source_map.json",
        "hierarchical_consistency_core_database_to_opentelemetry.json",
        "hierarchical_consistency_source_map_composition.json",
    ):
        report = proof_reports[name]
        assert report["status"] == "NOT_EVALUATED"
        assert report.get("exact_equal") is not True
