import json
from pathlib import Path


ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "scale_result.json"
)


def test_materialized_scale_result():
    row = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert row["status"] == "PASS"
    assert row["occurrence_node_count"] == 10_000
    assert row["fact_node_count"] == 30_000
    assert row["primitive_relation_edge_count"] == 28_900
