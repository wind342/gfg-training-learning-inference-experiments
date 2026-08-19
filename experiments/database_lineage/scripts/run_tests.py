from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

from experiments.database_lineage.src.metrics import write_json


REPO = Path(__file__).resolve().parents[3]
ARTIFACT = REPO / "experiments" / "database_lineage" / "artifacts" / "test_results.json"


def run(name: str, paths: list[str]) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *paths],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout + completed.stderr
    print(output, end="")
    passed = sum(int(value) for value in re.findall(r"(\d+) passed", output))
    failed = sum(int(value) for value in re.findall(r"(\d+) failed", output))
    errors = sum(int(value) for value in re.findall(r"(\d+) errors?", output))
    skipped = sum(int(value) for value in re.findall(r"(\d+) skipped", output))
    return {
        "name": name,
        "command": [sys.executable, "-m", "pytest", "-q", *paths],
        "returncode": completed.returncode,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "seconds": elapsed,
        "summary_output": output[-4000:],
    }


def main() -> int:
    suites = [
        run("existing_core", ["tests"]),
        run("database_lineage_experiment", ["experiments/database_lineage/tests"]),
    ]
    write_json(
        ARTIFACT,
        {
            "suites": suites,
            "all_passed": all(not item["returncode"] for item in suites),
        },
    )
    return 0 if all(not item["returncode"] for item in suites) else 1


if __name__ == "__main__":
    raise SystemExit(main())
