from __future__ import annotations

import json
from pathlib import Path

from .INDEPENDENT_CHECKER import PACKAGE, check


def test_frozen_formal_result_passes_independent_recomputation() -> None:
    report = check(PACKAGE / "FORMAL_RESULTS.json", PACKAGE / "FORMAL_GFG.json")
    assert report["status"] == "PASS"
    assert len(report["run_checks"]) == 6
    assert all(row["pass"] for row in report["run_checks"])


def test_tampered_verdict_is_rejected(tmp_path: Path) -> None:
    source = (PACKAGE / "FORMAL_RESULTS.json").read_text(encoding="utf-8")
    tampered = tmp_path / "FORMAL_RESULTS.json"
    tampered.write_text(
        source.replace(
            "CROSS_SYSTEM_FROZEN_PROJECTION_SUPPORTED",
            "CROSS_SYSTEM_FROZEN_PROJECTION_NOT_SUPPORTED",
            1,
        ),
        encoding="utf-8",
    )
    report = check(tampered, PACKAGE / "FORMAL_GFG.json")
    assert report["status"] == "FAIL"
    assert "FORMAL_VERDICT_MISMATCH" in report["errors"]


def test_tampered_mechanism_metric_is_rejected(tmp_path: Path) -> None:
    document = json.loads(
        (PACKAGE / "FORMAL_RESULTS.json").read_text(encoding="utf-8")
    )
    document["runs"][0]["support"]["maximum_query_profile_l1"] = 0.0
    tampered = tmp_path / "FORMAL_RESULTS.json"
    tampered.write_text(json.dumps(document), encoding="utf-8")
    report = check(tampered, PACKAGE / "FORMAL_GFG.json")
    assert report["status"] == "FAIL"
    assert "RUN_FAILED:resnet:20260820" in report["errors"]
