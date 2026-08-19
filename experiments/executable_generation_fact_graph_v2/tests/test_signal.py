import json
from pathlib import Path


ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "signal_result.json"
)


def test_materialized_signal_result():
    row = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert row["status"] == "PASS"
    assert row["path_count"] == 2880
    assert row["raw_source_count"] == 197
