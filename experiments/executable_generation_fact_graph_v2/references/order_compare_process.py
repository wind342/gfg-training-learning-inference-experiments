from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


COMPARED_FIELDS = (
    "answer",
    "status",
    "result_ids",
    "explicit_disposition_ids",
)


def compare_payload(payload: dict) -> dict:
    if set(payload) != {
        "candidate_answers",
        "reference_answers",
        "schema_version",
    }:
        raise ValueError("ORDER_V2_COMPARE_INPUT_SCOPE_INVALID")
    candidate = payload["candidate_answers"]
    reference = payload["reference_answers"]
    candidate_rows = candidate["answers"]
    reference_rows = reference["answers"]
    reference_by_key = {
        (row["scenario"], row["query_id"]): row
        for row in reference_rows
    }
    mismatches = []
    false_positive = 0
    false_negative = 0
    for row in candidate_rows:
        key = (row["scenario"], row["query_id"])
        expected = reference_by_key.get(key)
        equal = (
            expected is not None
            and all(
                row.get(field) == expected.get(field)
                for field in COMPARED_FIELDS
            )
        )
        if not equal:
            mismatches.append(
                {
                    "scenario": key[0],
                    "query_id": key[1],
                    "candidate": row,
                    "reference": expected,
                }
            )
            candidate_truth = bool(row.get("answer"))
            reference_truth = bool(
                expected.get("answer")
                if expected is not None
                else False
            )
            if candidate_truth and not reference_truth:
                false_positive += 1
            elif not candidate_truth and reference_truth:
                false_negative += 1
            else:
                false_positive += 1
                false_negative += 1
    missing = sorted(
        set(reference_by_key)
        - {
            (row["scenario"], row["query_id"])
            for row in candidate_rows
        }
    )
    exact = not mismatches and not missing and (
        len(candidate_rows) == len(reference_rows)
    )
    return {
        "status": "PASS" if exact else "FAIL",
        "query_count": len(candidate_rows),
        "mismatch_count": len(mismatches) + len(missing),
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "missing_query_keys": [
            {"scenario": row[0], "query_id": row[1]}
            for row in missing
        ],
        "mismatches": mismatches,
        "compared_fields": list(COMPARED_FIELDS),
        "schema_version": "order-v2-compare-output-v2",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    try:
        payload = json.loads(
            Path(args.input).read_text(encoding="utf-8")
        )
        result = compare_payload(payload)
        output.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return 0 if result["status"] == "PASS" else 3
    except Exception as exc:
        output.write_text(
            json.dumps(
                {
                    "status": "FAIL",
                    "reason_code": str(exc),
                    "partial_success": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
