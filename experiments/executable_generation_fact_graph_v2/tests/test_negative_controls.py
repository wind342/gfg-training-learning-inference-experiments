import json
from pathlib import Path


ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "negative_controls.json"
)


def test_all_48_negative_controls_are_detected_once():
    row = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert row["status"] == "PASS"
    assert row["detected_count"] == 48
    assert all(
        control["execution_count"] == 1
        for control in row["controls"]
    )
