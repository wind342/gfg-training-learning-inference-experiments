def test_core_change_lineage_distinguishes_inherited_and_new_changes(artifact) -> None:
    report = artifact("core_change_lineage.json")
    assert report["status"] == "PASS"
    assert report["inherited_core_changes_main_to_database"] == [
        "src/generation_relation_core/relation_evidence.py"
    ]
    assert report["inherited_core_schema_change_count"] == 0
    assert report["new_core_changes_database_to_unified"] == []
    assert report["new_core_schema_change_count"] == 0
    assert report["new_domain_specific_core_field_count"] == 0

