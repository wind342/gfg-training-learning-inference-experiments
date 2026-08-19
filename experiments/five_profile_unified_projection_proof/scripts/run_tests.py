from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ..src.common import write_json


TEST_SUITES = {
    "core": ["tests/core"],
    "database_which_lineage": ["experiments/database_lineage/tests"],
    "source_map": ["tests/experiments/source_map_projection"],
    "opentelemetry": ["experiments/opentelemetry_projection/tests"],
    "three_profile_unified": ["experiments/operational_projection_proof_v2/tests"],
    "w3c_prov_generation_profile": ["experiments/w3c_prov_projection_v1/tests"],
    "pytorch_autograd_dependency_profile": ["experiments/pytorch_autograd_training_lineage_v1/tests"],
    "five_profile_unified": ["tests/experiments/five_profile_unified_projection_proof"],
}


def _junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "tests": sum(int(suite.attrib.get("tests", "0")) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", "0")) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", "0")) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", "0")) for suite in suites),
    }


def run_test_suites(repo: Path, receipt_root: Path) -> dict[str, Any]:
    receipt_root.mkdir(parents=True, exist_ok=True)
    suites: dict[str, Any] = {}
    for name, targets in TEST_SUITES.items():
        junit = receipt_root / f"{name}.junit.xml"
        command = [sys.executable, "-m", "pytest", "-q", *targets, f"--junitxml={junit}"]
        process = subprocess.run(command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        counts = _junit_counts(junit) if junit.is_file() else {"tests": 0, "failures": 0, "errors": 1, "skipped": 0}
        passed = process.returncode == 0 and counts["tests"] > 0 and counts["failures"] == counts["errors"] == counts["skipped"] == 0
        suites[name] = {
            "command": ["python", "-m", "pytest", "-q", *targets, "--junitxml=<RUNTIME_RECEIPT>"],
            "exit_code": process.returncode,
            "counts": counts,
            "status": "PASS" if passed else "FAIL",
            "output_tail": process.stdout[-4000:].replace(str(repo), "<REPO>").replace(str(receipt_root), "<RUNTIME>"),
        }
    totals = {key: sum(row["counts"][key] for row in suites.values()) for key in ("tests", "failures", "errors", "skipped")}
    all_passed = all(row["status"] == "PASS" for row in suites.values())
    return {"status": "PASS" if all_passed else "FAIL", "all_passed": all_passed, "totals": totals, "suites": suites}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every test suite required by the unified proof.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--receipt-root", type=Path, default=Path("data_private/five_profile_unified_projection_proof/tests"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_test_suites(args.repo.resolve(), (args.repo / args.receipt_root).resolve() if not args.receipt_root.is_absolute() else args.receipt_root)
    if args.output:
        write_json(args.output.resolve(), result)
    print(json.dumps({"status": result["status"], "totals": result["totals"]}, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
