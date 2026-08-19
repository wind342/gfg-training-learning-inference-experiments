from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from experiments.inter_fact_relations_v0_hardening_scale_v1.scenarios.primitive_semantic_validation import (
    build,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.capture_auditor import (
    audit_capture,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.semantic_evidence_validator import (
    validate_primitive_store,
)


ROOT = Path(__file__).resolve().parents[1]


def _schema(name: str):
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def test_relation_evidence_and_capture_schemas() -> None:
    builder = build()
    receipts = builder.runtime_receipts()
    validated = validate_primitive_store(builder.primitive_store(), receipts)
    capture = audit_capture(builder.capture_contract(), receipts, validated)
    relation_validator = Draft202012Validator(_schema("relation.schema.json"))
    evidence_validator = Draft202012Validator(_schema("evidence.schema.json"))
    capture_validator = Draft202012Validator(_schema("capture_audit.schema.json"))
    for row in validated["primitive_relations"]:
        assert list(relation_validator.iter_errors(row)) == []
    for row in validated["evidence"]:
        assert list(evidence_validator.iter_errors(row)) == []
    assert list(capture_validator.iter_errors(capture)) == []
