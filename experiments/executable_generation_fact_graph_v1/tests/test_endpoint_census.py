from __future__ import annotations

import json
from pathlib import Path

from experiments.executable_generation_fact_graph_v1.endpoint_census import (
    census_primitive_relation_endpoints,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ORDER_ARTIFACTS = (
    REPOSITORY_ROOT
    / "experiments"
    / "order_refund_freeze_inter_fact_relations_v1"
    / "artifacts"
)


def _load(name: str):
    return json.loads((ORDER_ARTIFACTS / name).read_text(encoding="utf-8"))


def test_frozen_order_endpoint_census_reproduces_exact_failure() -> None:
    result = census_primitive_relation_endpoints(
        _load("atomic_generation_facts.json"),
        _load("primitive_relation_sidecar.json"),
    )
    assert result["primitive_relation_count"] == 83
    assert result["legally_mappable_relation_count"] == 28
    assert result["unmappable_primitive_relation_count"] == 55
    assert result["endpoint_type_combination_counts"] == {
        "fact->fact": 23,
        "occurrence->occurrence": 60,
        "fact->occurrence": 0,
        "occurrence->fact": 0,
    }
    assert not result["pure_fact_vertex_model_supported"]
    assert not any(result["prohibited_action_counts"].values())


def test_unmappable_rows_retain_exact_native_identities() -> None:
    result = census_primitive_relation_endpoints(
        _load("atomic_generation_facts.json"),
        _load("primitive_relation_sidecar.json"),
    )
    rows = result["unmappable_relations"]
    assert len(rows) == 55
    assert len({row["relation_id"] for row in rows}) == 55
    assert all(row["source_endpoint_id"] for row in rows)
    assert all(row["target_endpoint_id"] for row in rows)
    assert all(row["reason_codes"] for row in rows)
