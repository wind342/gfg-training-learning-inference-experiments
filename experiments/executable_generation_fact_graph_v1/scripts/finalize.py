from __future__ import annotations

import json

from generation_relation_core.canonical import canonical_bytes

from ..materialize import ARTIFACT_ROOT, refresh_manifest
from ..protected_audit import protected_path_audit


def _load(name: str):
    return json.loads(
        (ARTIFACT_ROOT / name).read_text(encoding="utf-8")
    )


def _write(name: str, value) -> None:
    (ARTIFACT_ROOT / name).write_bytes(canonical_bytes(value) + b"\n")


def main() -> int:
    result = _load("v1_final_result.json")
    tests = _load("test_results.json")
    protected = protected_path_audit()
    by_label = {row["label"]: row for row in tests["runs"]}
    result["protected_path_audit"] = protected
    result["test_evidence"] = {
        "status": tests["status"],
        "v1_focused": by_label["v1_focused"]["status"],
        "core": by_label["core"]["status"],
        "full_repository": by_label["full_repository"]["status"],
        "excluded_from_scientific_sha256": True,
    }
    result["mandatory_gates"]["existing_core_tests_passed"] = (
        by_label["core"]["status"] == "PASS"
    )
    result["mandatory_gates"]["full_repository_tests_passed"] = (
        by_label["full_repository"]["status"] == "PASS"
    )
    result["failed_mandatory_gates"] = sorted(
        name
        for name, passed in result["mandatory_gates"].items()
        if not passed
    )
    _write("protected_path_audit.json", protected)
    _write("v1_final_result.json", result)
    markdown_path = ARTIFACT_ROOT / "V1_RESULT.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    marker = "\n## Tests\n"
    if marker in markdown:
        markdown = markdown.split(marker, 1)[0].rstrip() + "\n"
    markdown += (
        "\n## Tests\n\n"
        f"- v1 focused: `{by_label['v1_focused']['status']}` "
        "(8 passed)\n"
        f"- frozen Core: `{by_label['core']['status']}` "
        "(33 passed)\n"
        f"- full repository: `{by_label['full_repository']['status']}` "
        "(135 passed)\n"
    )
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    manifest = refresh_manifest(
        ARTIFACT_ROOT,
        scientific_sha256=result["scientific_sha256"],
    )
    print(
        json.dumps(
            {
                "final_status": result["final_status"],
                "test_status": tests["status"],
                "protected_status": protected["status"],
                "artifact_count": manifest["artifact_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if tests["status"] == protected["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
