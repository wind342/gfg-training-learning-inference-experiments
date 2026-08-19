from __future__ import annotations

import json

from generation_relation_core.canonical import canonical_bytes

from ..materialize import (
    ARTIFACT_ROOT,
    _result_markdown,
    refresh_manifest,
)
from ..protected_audit import protected_path_audit
from ..runner import (
    FINAL_STATUS_NOT_SUPPORTED,
    FINAL_STATUS_SUPPORTED,
)


def _load(name: str):
    return json.loads(
        (ARTIFACT_ROOT / name).read_text(encoding="utf-8")
    )


def _write(name: str, value) -> None:
    (ARTIFACT_ROOT / name).write_bytes(
        canonical_bytes(value) + b"\n"
    )


def main() -> int:
    result = _load("v2_final_result.json")
    tests = _load("test_results.json")
    protected = protected_path_audit()
    by_label = {row["label"]: row for row in tests["runs"]}
    result["protected_path_audit"] = protected
    result["test_evidence"] = {
        "status": tests["status"],
        "v2_focused": by_label["v2_focused"]["status"],
        "core": by_label["core"]["status"],
        "full_repository": by_label["full_repository"]["status"],
        "excluded_from_scientific_sha256": True,
    }
    result["mandatory_gates"]["v2_focused_tests_passed"] = (
        by_label["v2_focused"]["status"] == "PASS"
    )
    result["mandatory_gates"]["core_tests_passed"] = (
        by_label["core"]["status"] == "PASS"
    )
    result["mandatory_gates"]["full_repository_tests_passed"] = (
        by_label["full_repository"]["status"] == "PASS"
    )
    result["failed_mandatory_gates"] = sorted(
        key
        for key, value in result["mandatory_gates"].items()
        if not value
    )
    result["final_status"] = (
        FINAL_STATUS_SUPPORTED
        if not result["failed_mandatory_gates"]
        else FINAL_STATUS_NOT_SUPPORTED
    )
    _write("protected_path_audit.json", protected)
    _write("v2_final_result.json", result)
    result_path = ARTIFACT_ROOT / "RESULT.md"
    markdown = _result_markdown(result).rstrip() + "\n"
    markdown += (
        "\n## Test finalization\n\n"
        f"- v2 focused: `{by_label['v2_focused']['status']}`\n"
        f"- frozen Core: `{by_label['core']['status']}`\n"
        f"- full repository: `{by_label['full_repository']['status']}`\n"
        f"- protected paths: `{protected['status']}`\n"
        f"- final status: `{result['final_status']}`\n"
    )
    result_path.write_text(
        markdown, encoding="utf-8", newline="\n"
    )
    manifest = refresh_manifest(
        ARTIFACT_ROOT,
        scientific_sha256=result["scientific_sha256"],
    )
    print(
        json.dumps(
            {
                "final_status": result["final_status"],
                "failed_mandatory_gates": result[
                    "failed_mandatory_gates"
                ],
                "test_status": tests["status"],
                "protected_status": protected["status"],
                "artifact_count": manifest["artifact_count"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return (
        0
        if result["final_status"] == FINAL_STATUS_SUPPORTED
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
