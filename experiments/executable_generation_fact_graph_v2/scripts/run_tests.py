from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from generation_relation_core.canonical import canonical_bytes


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "test_results.json"
)


def _run(label: str, arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {
        "label": label,
        "command": [sys.executable, "-m", "pytest", *arguments],
        "exit_code": completed.returncode,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    runs = [
        _run(
            "v2_focused",
            [
                "-q",
                "experiments/executable_generation_fact_graph_v2/tests",
            ],
        ),
        _run("core", ["-q", "tests/core"]),
        _run("full_repository", ["-q"]),
    ]
    result = {
        "schema_version": (
            "executable-generation-fact-graph-tests-v2"
        ),
        "status": (
            "PASS"
            if all(row["status"] == "PASS" for row in runs)
            else "FAIL"
        ),
        "runs": runs,
    }
    ARTIFACT_PATH.write_bytes(canonical_bytes(result) + b"\n")
    for row in runs:
        print(f"{row['label']}: {row['status']}")
        print(row["stdout"])
        if row["stderr"]:
            print(row["stderr"], file=sys.stderr)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
