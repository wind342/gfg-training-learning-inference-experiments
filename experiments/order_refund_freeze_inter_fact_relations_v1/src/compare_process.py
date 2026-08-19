from __future__ import annotations

import argparse
from typing import Any

from ..common import canonical_sha256, load_json, write_json


def compare(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {
        "candidate_answers",
        "reference_answers",
        "trace_answers",
        "schema_version",
    }:
        raise RuntimeError("COMPARE_INPUT_SCHEMA_MISMATCH")
    candidate = {
        (row["scenario"], row["query_id"]): row
        for row in payload["candidate_answers"]["answers"]
    }
    reference = {
        (row["scenario"], row["query_id"]): row
        for row in payload["reference_answers"]["answers"]
    }
    if set(candidate) != set(reference):
        raise RuntimeError("QUERY_KEY_SET_MISMATCH")
    rows = []
    false_positive_count = 0
    false_negative_count = 0
    for key in sorted(candidate):
        candidate_row = candidate[key]
        reference_row = reference[key]
        equal = candidate_row["answer"] == reference_row["answer"]
        false_positive = int(
            not equal
            and bool(candidate_row["answer"])
            and not bool(reference_row["answer"])
        )
        false_negative = int(
            not equal
            and not bool(candidate_row["answer"])
            and bool(reference_row["answer"])
        )
        if not equal and not (false_positive or false_negative):
            false_positive = 1
            false_negative = 1
        false_positive_count += false_positive
        false_negative_count += false_negative
        rows.append(
            {
                "query_id": key[1],
                "scenario": key[0],
                "exact_target": candidate_row["exact_target"],
                "candidate_answer": candidate_row["answer"],
                "independent_reference_answer": reference_row["answer"],
                "status": "PASS" if equal else "FAIL",
                "evidence_path": candidate_row["evidence_path"],
                "result_ids": candidate_row["result_ids"],
                "relation_ids": candidate_row["relation_ids"],
                "explicit_disposition_ids": candidate_row[
                    "explicit_disposition_ids"
                ],
                "false_positive_count": false_positive,
                "false_negative_count": false_negative,
            }
        )
    mismatch_count = sum(row["status"] == "FAIL" for row in rows)
    material = {
        "status": "PASS" if mismatch_count == 0 else "FAIL",
        "query_count": len(rows),
        "mismatch_count": mismatch_count,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "comparisons": rows,
        "trace_answer_count": len(payload["trace_answers"]["answers"]),
        "schema_version": "query-comparison-v1",
    }
    return {
        **material,
        "comparison_sha256": canonical_sha256(material),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        output = compare(load_json(args.input))
    except Exception as error:
        output = {
            "status": "FAIL",
            "reason_code": str(error),
            "schema_version": "query-comparison-v1",
        }
    write_json(args.output, output)
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
