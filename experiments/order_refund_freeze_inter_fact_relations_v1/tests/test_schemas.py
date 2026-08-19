from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from experiments.order_refund_freeze_inter_fact_relations_v1.src.generation_fact_collector import (
    collect_atomic_facts,
)
from experiments.order_refund_freeze_inter_fact_relations_v1.src.orchestrator import (
    run_workflow,
)
from experiments.order_refund_freeze_inter_fact_relations_v1.src.relation_sidecar_collector import (
    collect_relation_sidecar,
)


ROOT = Path(__file__).resolve().parents[1]


def _schema(name: str):
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def test_atomic_fact_and_relation_schemas() -> None:
    run = run_workflow(
        "CONCURRENT_REFUND_WINS", repeat_index=4, capture_enabled=True
    )
    atomic = collect_atomic_facts(run)
    sidecar = collect_relation_sidecar(run, atomic)
    fact_validator = Draft202012Validator(_schema("atomic_fact.schema.json"))
    relation_validator = Draft202012Validator(_schema("relation.schema.json"))
    assert all(
        not list(fact_validator.iter_errors(row)) for row in atomic["facts"]
    )
    assert all(
        not list(relation_validator.iter_errors(row))
        for row in sidecar["relations"]
    )
