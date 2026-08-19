from __future__ import annotations

import copy

import pytest

from experiments.database_lineage.src.resolver_reference import LegacyScanRelationEvidenceResolver
from generation_relation_core.errors import CoreV3Error
from generation_relation_core.relation_evidence import RelationEvidenceResolver


def _mutate(tables, mutation: str) -> None:
    binding = tables.generation_bindings[0]
    binding_id = binding["generation_binding_id"]
    link = next(row for row in tables.evidence_links if row["subject_id"] == binding_id)
    evidence = next(row for row in tables.evidence_records if row["evidence_id"] == link["evidence_id"])
    operation = next(row for row in tables.generator_operation_results if binding_id in row["produced_entity_ids"])
    if mutation == "missing_primary":
        tables.evidence_links.remove(link)
    elif mutation == "duplicate_primary":
        tables.evidence_links.append(copy.deepcopy(link))
    elif mutation == "bad_authority":
        evidence["evidence_authority"] = "not_authorized"
    elif mutation == "missing_related":
        evidence["related_record_ids"] = []
    elif mutation == "material_mismatch":
        evidence["artifact_sha256"] = "0" * 64
    elif mutation == "missing_operation":
        operation["produced_entity_ids"].remove(binding_id)
    elif mutation == "duplicate_operation":
        tables.generator_operation_results.append(copy.deepcopy(operation))
    else:
        raise ValueError(mutation)


def _outcome(resolver, tables):
    try:
        value = resolver.resolve(tables, preverified=True)
    except CoreV3Error as exc:
        return ("failure", exc.reason_code)
    return (
        "success",
        sorted(
            (key, item.evidence_link_id, item.evidence_id, item.operation_result_id)
            for key, item in value.items()
        ),
    )


def test_preindex_and_indexed_resolvers_have_identical_success(core_fixture) -> None:
    tables = copy.deepcopy(core_fixture.snapshot.tables)
    assert _outcome(LegacyScanRelationEvidenceResolver(), tables) == _outcome(RelationEvidenceResolver(), tables)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_primary", "duplicate_primary", "bad_authority", "missing_related",
        "material_mismatch", "missing_operation", "duplicate_operation",
    ],
)
def test_preindex_and_indexed_resolvers_have_identical_failures(core_fixture, mutation: str) -> None:
    legacy_tables = copy.deepcopy(core_fixture.snapshot.tables)
    indexed_tables = copy.deepcopy(core_fixture.snapshot.tables)
    _mutate(legacy_tables, mutation)
    _mutate(indexed_tables, mutation)
    legacy = _outcome(LegacyScanRelationEvidenceResolver(), legacy_tables)
    indexed = _outcome(RelationEvidenceResolver(), indexed_tables)
    assert legacy == indexed
    assert legacy[0] == "failure"

