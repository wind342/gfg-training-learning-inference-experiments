def test_source_map_p3_is_multistage_composition_not_hierarchical_projection(artifact) -> None:
    report = artifact("composition_consistency_source_map.json")
    assert report["status"] == "SUPPORTED"
    assert report["p3_subtype"] == "multistage_generation_composition_consistency"
    assert report["composed_mapping_count"] == 5
    assert report["generated_origin_count"] == 5
    assert report["false_positive_count"] == 0
    assert report["false_negative_count"] == 0
    assert report["broken_bridge_count"] == 0
    assert report["ambiguity_count"] == 0
    assert report["cycle_count"] == 0
    assert report["invented_transitive_mapping_count"] == 0
    assert report["direct_shortcut_count"] == 0
    assert report["derived_paths_are_generation_bindings"] is False
    assert report["fabricated_binding_id_count"] == 0
    assert report["four_mode_output_byte_identity"] is True

