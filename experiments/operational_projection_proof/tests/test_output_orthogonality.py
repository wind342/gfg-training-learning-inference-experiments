def test_database_output_is_byte_identical_with_contract_on_and_off(
    proof_reports,
) -> None:
    report = proof_reports["output_orthogonality.json"]
    assert report["status"] == "SUPPORTED"
    assert report["csv_byte_identical"] is True
    assert report["json_byte_identical"] is True
    assert report["forbidden_fields"] == []
