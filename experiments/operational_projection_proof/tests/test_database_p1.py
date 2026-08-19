def test_database_exact_derivability_is_full_field_exact(proof_reports) -> None:
    report = proof_reports["projection_equivalence_database.json"]
    assert report["status"] == "SUPPORTED"
    assert report["candidate_record_count"] == report["reference_record_count"] == 112
    assert report["false_positive"] == 0
    assert report["false_negative"] == 0
    assert report["field_mismatch"] == 0
    assert report["multiplicity_mismatch"] == 0
    assert report["section_results"]["derivation_paths"]["candidate_count"] == 20
    assert report["section_results"]["explicit_dispositions"]["candidate_count"] == 7
    qualification = report["proof_qualification"]
    assert qualification["status"] == "SUPPORTED"
    assert qualification["validated_snapshot_count"] == 2
    assert (
        qualification["validated_binding_count"]
        == qualification["relation_evidence_closure_count"]
    )
    assert (
        qualification["successful_operation_closure_count"]
        == qualification["validated_binding_count"]
    )
    assert all(qualification["checks"].values())


def test_candidate_survives_reference_deletion_and_runtime_trap(proof_reports) -> None:
    report = proof_reports["projection_equivalence_database.json"]
    assert report["candidate_after_reference_deleted_equal"] is True
    assert report["oracle_runtime_trap_passed"] is True
