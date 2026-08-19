from __future__ import annotations

import copy

import pytest

from generation_relation_core.errors import CoreV3Error
from generation_relation_core.relation_evidence import RelationEvidenceResolver


class _NoLinearContains(list):
    def __contains__(self, item: object) -> bool:
        raise AssertionError(f"linear membership scan attempted for {item}")


def test_each_binding_has_exactly_one_primary_evidence_and_operation(core_fixture) -> None:
    resolver = RelationEvidenceResolver()
    resolved = resolver.resolve(core_fixture.snapshot.tables)
    assert set(resolved) == {
        row["generation_binding_id"] for row in core_fixture.snapshot.tables.generation_bindings
    }
    assert resolver.fallback_count == 0
    assert all(item.operation_result_id for item in resolved.values())


def test_missing_or_duplicate_primary_evidence_fails_closed(core_fixture) -> None:
    tables = copy.deepcopy(core_fixture.snapshot.tables)
    binding_id = tables.generation_bindings[0]["generation_binding_id"]
    primary = next(row for row in tables.evidence_links if row["subject_id"] == binding_id)
    tables.evidence_links.remove(primary)
    with pytest.raises(CoreV3Error) as exc:
        RelationEvidenceResolver().resolve(tables)
    assert exc.value.reason_code == "BINDING_HAS_NO_PRIMARY_EVIDENCE"

    tables = copy.deepcopy(core_fixture.snapshot.tables)
    tables.evidence_links.append(copy.deepcopy(primary))
    with pytest.raises(CoreV3Error) as exc:
        RelationEvidenceResolver().resolve(tables)
    assert exc.value.reason_code == "BINDING_HAS_MULTIPLE_PRIMARY_EVIDENCE"


def test_evidence_authority_and_operation_closure_fail_closed(core_fixture) -> None:
    tables = copy.deepcopy(core_fixture.snapshot.tables)
    binding = tables.generation_bindings[0]
    evidence = next(row for row in tables.evidence_records if row["evidence_id"] in binding["evidence_ids"])
    evidence["evidence_authority"] = "unauthorized"
    with pytest.raises(CoreV3Error) as exc:
        RelationEvidenceResolver().resolve(tables, preverified=True)
    assert exc.value.reason_code == "EVIDENCE_AUTHORITY_UNAUTHORIZED"

    tables = copy.deepcopy(core_fixture.snapshot.tables)
    operation = tables.generator_operation_results[0]
    operation["produced_entity_ids"].remove(binding["generation_binding_id"])
    with pytest.raises(CoreV3Error) as exc:
        RelationEvidenceResolver().resolve(tables, preverified=True)
    assert exc.value.reason_code == "OPERATION_BINDING_CLOSURE_FAILED"


def test_operation_closure_uses_prebuilt_membership_indexes(core_fixture) -> None:
    tables = copy.deepcopy(core_fixture.snapshot.tables)
    for operation in tables.generator_operation_results:
        operation["produced_entity_ids"] = _NoLinearContains(operation["produced_entity_ids"])
        operation["evidence_ids"] = _NoLinearContains(operation["evidence_ids"])
    resolved = RelationEvidenceResolver().resolve(tables, preverified=True)
    assert len(resolved) == len(tables.generation_bindings)
