from __future__ import annotations

import json
from pathlib import Path


ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"


def _load_json(name: str):
    return json.loads(
        (ARTIFACT_ROOT / name).read_text(encoding="utf-8")
    )


def test_result_report_uses_final_machine_status() -> None:
    result = _load_json("v2_final_result.json")
    report = (ARTIFACT_ROOT / "RESULT.md").read_text(
        encoding="utf-8"
    )
    expected = f"Final status: **{result['final_status']}**"
    assert expected in report
    assert "\ufffd" not in report


def test_protected_audit_binds_containing_commit_trees() -> None:
    audit = _load_json("protected_path_audit.json")
    assert "head" not in audit
    assert audit["audit_subject"]["revision"] == "CONTAINING_COMMIT"
    assert (
        audit["audit_subject"]["binding"]
        == "PROTECTED_PATH_TREE_OBJECT_IDS"
    )
    assert audit["audit_subject"]["verification_ref"] == "HEAD"
