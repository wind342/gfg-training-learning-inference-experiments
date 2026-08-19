def test_source_map_p1_profile_and_surface_statuses_are_separate(artifact) -> None:
    report = artifact("projection_equivalence_source_map.json")
    surface = artifact("source_map_standard_surface_coverage.json")
    assert report["status"] == "SUPPORTED"
    assert report["standard_surface_status"] == "PARTIAL"
    assert report["total_mapping_segments"] == 685
    assert report["bidirectional_query_count"] == 1385
    assert report["query_mismatch_count"] == 0
    assert report["medium_mapping_count"] == 660
    assert all(report["frozen_checks"].values())
    assert surface["ordinary_non_indexed_profile_status"] == "SUPPORTED"
    assert surface["status"] == "PARTIAL"
    assert surface["official_applicable_passed"] == 80
    assert surface["official_applicable_total"] == 80
    assert surface["indexed_exclusion_count"] == 19

