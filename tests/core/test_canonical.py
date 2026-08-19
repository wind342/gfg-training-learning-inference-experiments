from __future__ import annotations

import json
from pathlib import Path

import pytest

from generation_relation_core.canonical import (
    canonical_bytes,
    finalize_entity,
    sha256_bytes,
    table_hash,
    verify_entity,
)
from generation_relation_core.entities import source_information
from generation_relation_core.errors import CoreV3Error


PROTOCOL_ROOT = Path(__file__).parents[2] / "protocol" / "core_v3"


def test_positive_canonical_vectors() -> None:
    vectors = json.loads(
        (PROTOCOL_ROOT / "test_vectors" / "canonical_positive.json").read_text(encoding="utf-8")
    )["vectors"]
    for vector in vectors:
        data = canonical_bytes(vector["input"])
        assert data.hex() == vector["canonical_utf8_hex"]
        assert sha256_bytes(data) == vector["sha256"]


def test_negative_vector_contract_is_covered_by_fail_closed_reasons(core_fixture) -> None:
    contract = json.loads(
        (PROTOCOL_ROOT / "test_vectors" / "canonical_negative.json").read_text(encoding="utf-8")
    )
    expected = {row["expected_failure"] for row in contract["vectors"] if "expected_failure" in row}
    covered = {
        "HASH_SELF_REFERENCE",
        "HASH_OR_ID_MISMATCH",
        "NON_FINITE_NUMBER",
        "DUPLICATE_SET_ELEMENT",
        "UNSUPPORTED_RUNTIME_TYPE",
        "LONE_SURROGATE",
        "BINDING_OUTCOME_CARDINALITY_INVALID",
        "NON_CANONICAL_SET_ORDER",
        "DUPLICATE_ENTITY_ID",
        "BINDING_ID_OR_HASH_MISMATCH",
        "EVIDENCE_ENTITY_MISMATCH",
    }
    assert expected == covered

    binding = core_fixture.snapshot.tables.generation_bindings[0]
    with pytest.raises(CoreV3Error) as exc:
        finalize_entity("GenerationBinding", dict(binding))
    assert "HASH_SELF_REFERENCE" in str(exc.value)

    source = dict(core_fixture.snapshot.tables.source_information_records[0])
    source["source_information_id"] = "si3_" + "0" * 64
    with pytest.raises(CoreV3Error) as exc:
        verify_entity("SourceInformationRecord", source)
    assert exc.value.reason_code == "HASH_OR_ID_MISMATCH"

    with pytest.raises(CoreV3Error) as exc:
        table_hash([binding, binding], "GenerationBinding")
    assert exc.value.reason_code == "DUPLICATE_ENTITY_ID"


def test_negative_zero_has_positive_zero_identity() -> None:
    left = source_information(
        domain_scope_id="d",
        source_identity="zero",
        source_parent_id=None,
        source_granularity="number",
        source_payload={"value": -0.0},
    )
    right = source_information(
        domain_scope_id="d",
        source_identity="zero",
        source_parent_id=None,
        source_granularity="number",
        source_payload={"value": 0},
    )
    assert left == right
