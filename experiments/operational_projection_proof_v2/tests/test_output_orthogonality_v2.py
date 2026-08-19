def test_all_normal_outputs_are_orthogonal(artifact) -> None:
    report = artifact("output_orthogonality.json")
    assert report["status"] == "PASS"
    assert report["domains"]["database"]["csv_byte_identical"] is True
    assert report["domains"]["database"]["json_byte_identical"] is True
    assert report["domains"]["opentelemetry"]["csv_byte_identical"] is True
    assert report["domains"]["opentelemetry"]["json_byte_identical"] is True
    assert report["domains"]["source_map"]["four_mode_byte_identity"] is True

