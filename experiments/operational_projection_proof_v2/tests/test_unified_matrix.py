def test_unified_matrix_uses_the_two_distinct_p3_subtypes(artifact) -> None:
    report = artifact("unified_projection_matrix.json")
    assert report["status"] == "SUPPORTED"
    rows = {row["domain_mechanism"]: row for row in report["rows"]}
    assert (rows["Database which-lineage"]["p1"], rows["Database which-lineage"]["p2"], rows["Database which-lineage"]["p3"]) == ("SUPPORTED", "SUPPORTED", "NOT_APPLICABLE")
    assert rows["OpenTelemetry trace"]["p3_subtype"] == "cross-domain hierarchical projection"
    assert rows["ECMA-426 Source Map"]["p3_subtype"] == "multistage generation composition"
    assert rows["ECMA-426 Source Map"]["standard_surface_status"] == "PARTIAL"

