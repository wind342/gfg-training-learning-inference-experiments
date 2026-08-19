from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..common import ExperimentError, canonical_sha256, load_json, write_json


ALLOWED_COMPARE_KEYS = {"candidate", "reference", "query_manifest_sha256"}


def compare_outputs(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != ALLOWED_COMPARE_KEYS:
        raise ExperimentError("COMPARE_INPUT_SCOPE_INVALID")
    candidate = payload["candidate"]
    reference = payload["reference"]
    if candidate["status"] != "PASS" or reference["status"] != "PASS":
        raise ExperimentError("COMPARE_INPUT_PROCESS_FAILED")
    if candidate["execution_run_id"] != reference["execution_run_id"]:
        raise ExperimentError("COMPARE_RUN_ID_MISMATCH")
    candidate_rows = candidate["answers"]
    reference_rows = reference["answers"]
    if len(candidate_rows) != len(reference_rows):
        raise ExperimentError("COMPARE_QUERY_COUNT_MISMATCH")
    false_positive = 0
    false_negative = 0
    mismatches: list[dict[str, Any]] = []
    for candidate_row, reference_row in zip(candidate_rows, reference_rows):
        if candidate_row["query_id"] != reference_row["query_id"]:
            raise ExperimentError("COMPARE_QUERY_ORDER_MISMATCH")
        if candidate_row["query_type"] != reference_row["query_type"]:
            raise ExperimentError("COMPARE_QUERY_TYPE_MISMATCH")
        left = candidate_row["result"]
        right = reference_row["result"]
        if left != right:
            mismatches.append(
                {
                    "query_id": candidate_row["query_id"],
                    "candidate": left,
                    "reference": right,
                }
            )
            if left is True and right is False:
                false_positive += 1
            elif left is False and right is True:
                false_negative += 1
    result_material = {
        "execution_run_id": candidate["execution_run_id"],
        "query_manifest_sha256": payload["query_manifest_sha256"],
        "query_count": len(candidate_rows),
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    return {
        "status": "PASS" if not mismatches else "FAIL",
        **result_material,
        "comparison_sha256": canonical_sha256(result_material),
        "schema_version": "candidate-reference-comparison-v1",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output_path = Path(args.output)
    try:
        result = compare_outputs(load_json(Path(args.input)))
    except ExperimentError as error:
        write_json(
            output_path,
            {
                "status": "FAIL",
                "reason_code": str(error),
                "partial_success": False,
                "process_role": "compare",
            },
        )
        return 2
    write_json(output_path, result)
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
