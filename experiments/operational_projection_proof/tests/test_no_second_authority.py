from experiments.operational_projection_proof.src.isolation_checks import (
    detect_second_authority_source,
)


def test_core_and_candidate_have_no_second_authority_store(proof_reports) -> None:
    report = proof_reports["second_authority_audit.json"]
    assert report["status"] == "SUPPORTED"
    assert report["core_schema_unchanged"] is True
    assert report["second_authority_store_count"] == 0


def test_second_authority_declaration_is_detected() -> None:
    assert detect_second_authority_source("lineage_table = {}\n") == ["lineage_table"]
