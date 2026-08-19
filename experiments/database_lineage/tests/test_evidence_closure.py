import copy

import pytest

from generation_relation_core.errors import CoreV3Error
from generation_relation_core.relation_evidence import RelationEvidenceResolver


def test_every_binding_has_one_primary_evidence_and_one_successful_operation(
    business_run,
) -> None:
    _adapter, _rows, snapshot, _reader = business_run
    resolved = RelationEvidenceResolver().resolve(snapshot.tables)
    assert len(resolved) == len(snapshot.tables.generation_bindings)
    assert all(
        item.evidence_link_id and item.operation_result_id for item in resolved.values()
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing_primary", "BINDING_HAS_NO_PRIMARY_EVIDENCE"),
        ("duplicate_primary", "BINDING_HAS_MULTIPLE_PRIMARY_EVIDENCE"),
        ("incomplete_related", "EVIDENCE_ENTITY_MISMATCH"),
        ("material_mismatch", "BINDING_EVIDENCE_MATERIAL_MISMATCH"),
        ("failed_operation", "OPERATION_BINDING_CLOSURE_FAILED"),
        ("duplicate_operation", "OPERATION_BINDING_CLOSURE_FAILED"),
    ],
)
def test_evidence_failure_modes_fail_closed(
    business_run, mutation: str, reason: str
) -> None:
    _adapter, _rows, snapshot, _reader = business_run
    tables = copy.deepcopy(snapshot.tables)
    binding = tables.generation_bindings[0]
    link = next(
        row
        for row in tables.evidence_links
        if row["subject_id"] == binding["generation_binding_id"]
    )
    evidence = next(
        row
        for row in tables.evidence_records
        if row["evidence_id"] == link["evidence_id"]
    )
    operation = next(
        row
        for row in tables.generator_operation_results
        if binding["generation_binding_id"] in row["produced_entity_ids"]
    )
    if mutation == "missing_primary":
        tables.evidence_links.remove(link)
    elif mutation == "duplicate_primary":
        tables.evidence_links.append(copy.deepcopy(link))
    elif mutation == "incomplete_related":
        evidence["related_record_ids"] = []
    elif mutation == "material_mismatch":
        evidence["artifact_sha256"] = "0" * 64
    elif mutation == "failed_operation":
        operation["produced_entity_ids"].remove(binding["generation_binding_id"])
    else:
        tables.generator_operation_results.append(copy.deepcopy(operation))
    with pytest.raises(CoreV3Error) as exc:
        RelationEvidenceResolver().resolve(tables, preverified=True)
    assert exc.value.reason_code == reason
