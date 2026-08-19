from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
        if set(payload) != {"candidate", "reference"}:
            raise ValueError("SIGNAL_COMPARE_INPUT_SCOPE_INVALID")
        candidate = payload["candidate"]
        reference = payload["reference"]
        if (
            candidate["status"] != "PASS"
            or reference["status"] != "PASS"
        ):
            raise ValueError("SIGNAL_COMPARE_PROCESS_INPUT_FAILED")
        exact = candidate["answer"] == reference["answer"]
        result = {
            "status": "PASS" if exact else "FAIL",
            "execution_run_id": candidate["execution_run_id"],
            "candidate_reference_exact": exact,
            "candidate_answer": candidate["answer"],
            "reference_answer": reference["answer"],
            "schema_version": "signal-compare-output-v2",
        }
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
        return 0 if exact else 3
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
