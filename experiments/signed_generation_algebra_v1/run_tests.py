"""Run the repository test suite and persist a compact machine result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from generation_relation_core.canonical import canonical_json_file_bytes

from .run_experiment import DEFAULT_REPORT_ROOT, REPOSITORY_ROOT


def _count(pattern: str, output: str) -> int:
    matches = re.findall(pattern, output)
    return sum(int(value) for value in matches)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_ROOT / "test_results.json",
    )
    args = parser.parse_args(argv)
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = completed.stdout + completed.stderr
    sys.stdout.write(output)
    result = {
        "command": "python -m pytest -q",
        "error_count": _count(r"(\d+) errors?", output),
        "exit_code": completed.returncode,
        "failed_count": _count(r"(\d+) failed", output),
        "passed_count": _count(r"(\d+) passed", output),
        "skipped_count": _count(r"(\d+) skipped", output),
        "status": (
            "passed" if completed.returncode == 0 else "failed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_file_bytes(result))
    print(json.dumps(result, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
